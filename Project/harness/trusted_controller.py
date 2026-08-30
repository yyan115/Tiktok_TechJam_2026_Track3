#!/usr/bin/env python3
"""Post-lock authority controller and sole candidate-evaluation entrypoint.

The controller never imports or executes candidate bytes.  It consumes a
one-use permit first, launches ``candidate_worker.py`` in the restricted
Bubblewrap namespace, independently recomputes all correctness references, and
is the only process allowed to append authoritative measurement state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import statistics
import sys
import tempfile
import time
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = ROOT / "Project"
TOOLS = PROJECT / "tools"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TOOLS))

from authority import (  # noqa: E402
    AuthorityError,
    AuthorityStore,
    atomic_write,
    canonical_json,
    iso_utc,
    parse_utc,
    read_json_exact,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from lock_manifest import verify_lock  # noqa: E402
from sandbox import (  # noqa: E402
    IsolatedMount,
    SandboxFiles,
    SandboxResult,
    run_isolated_command,
    run_sandbox,
)


CONTROLLER_VERSION = "2.0.0"
OFFICIAL = ROOT / "torch_transformer_benchmark.py"
SHAPES = PROJECT / "shapes.json"
WORKER = HERE / "candidate_worker.py"
SHAPE6_EVALUATOR = TOOLS / "shape6_local_eval.py"
SHAPE14_EVALUATOR = TOOLS / "shape14_eval.py"
SUBMISSION = PROJECT / "submission" / "torch_transformer_benchmark_submission.py"
MANIFEST = PROJECT / "manifest.json"
TENSORFLOW_OFFICIAL = ROOT / "tensorflow_transformer_benchmark.py"
REQUESTS = PROJECT / "loop" / "requests"
RECEIPTS = PROJECT / "authority" / "receipts"
NUMERICAL = {
    "padding_ratio": 0.0,
    "input_scale": 1.0,
    "rtol": 0.02,
    "atol": 0.002,
    "matmul_precision": "high",
    "allow_tf32": True,
}
TIMING = {"warmup": 20, "repeats": 100, "rounds": 3}
GATE_REQUEST_KINDS = {
    "scientific_attempt", "diagnostic", "calibration", "side_evaluation"
}
RECEIPT_KEYS = {
    "authority_event_id",
    "authority_event_sha256",
    "action",
    "subject_sha256",
    "capability_nonce",
    "role",
}


class ControllerRefusal(RuntimeError):
    pass


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ControllerRefusal(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControllerRefusal(f"{label} must be a JSON object")
    return value


def _regular_repo_file(path: Path, label: str, allowed: tuple[Path, ...]) -> Path:
    if path.is_symlink():
        raise ControllerRefusal(f"{label} cannot be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ControllerRefusal(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise ControllerRefusal(f"{label} must be a regular file")
    if not any(resolved.is_relative_to(directory.resolve()) for directory in allowed):
        raise ControllerRefusal(f"{label} is outside the approved candidate roots")
    return resolved


def _shape(shape_id: int) -> dict[str, Any]:
    document = _strict_object(SHAPES, "shapes.json")
    items = document.get("shapes")
    if not isinstance(items, list):
        raise ControllerRefusal("shapes.json schema is malformed")
    matches = [item for item in items if isinstance(item, dict) and item.get("id") == shape_id]
    if len(matches) != 1:
        raise ControllerRefusal(f"shape {shape_id} is absent or duplicated")
    return matches[0]


def _gate_request(path: Path, requests_root: Path = REQUESTS) -> tuple[dict[str, Any], str, bytes]:
    if path.is_symlink():
        raise ControllerRefusal("gate request cannot be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(requests_root.resolve()):
        raise ControllerRefusal("gate request must live in the content-addressed request store")
    raw = resolved.read_bytes()
    digest = sha256_bytes(raw)
    if resolved.name != f"{digest}.json":
        raise ControllerRefusal("gate request filename/hash binding mismatch")
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ControllerRefusal("gate request is malformed") from exc
    if not isinstance(request, dict) or request.get("schema_version") != 1:
        raise ControllerRefusal("gate request schema is unsupported")
    if request.get("request_kind") not in GATE_REQUEST_KINDS:
        raise ControllerRefusal("unknown gate request kind")
    return request, digest, raw


def normalize_permit_request(request: Mapping[str, Any], request_sha256: str) -> dict[str, Any]:
    kind = request.get("request_kind")
    mode = request.get("mode")
    campaign = request.get("campaign_id")
    shape_id = request.get("shape")
    if not isinstance(campaign, str) or not campaign:
        raise ControllerRefusal("gate request campaign_id is malformed")
    if isinstance(shape_id, bool) or not isinstance(shape_id, int) or not 1 <= shape_id <= 14:
        raise ControllerRefusal("gate request shape must be 1..14")
    expires_epoch = request.get("expires_epoch")
    if not isinstance(expires_epoch, (int, float)) or not math.isfinite(expires_epoch):
        raise ControllerRefusal("gate request expiry is malformed")
    expires = datetime.fromtimestamp(float(expires_epoch), tz=timezone.utc)
    if expires <= utc_now():
        raise ControllerRefusal("gate request has expired")
    if kind == "side_evaluation":
        expected_mode = {6: "shape6", 14: "shape14"}.get(shape_id)
        if (
            mode != expected_mode
            or request.get("candidate_authorized") is not True
            or request.get("promotion_allowed") is not False
        ):
            raise ControllerRefusal("side-evaluation request privileges are malformed")
        candidate_sha = request.get("impl_sha256")
        family_id = None
    elif kind == "diagnostic":
        if mode != "diagnostic" or request.get("candidate_authorized") is not False:
            raise ControllerRefusal("diagnostic request privileges are malformed")
        candidate_sha = request.get("target_sha256")
        family_id = None
    elif kind == "calibration":
        if mode != "calibration" or request.get("candidate_authorized") is not False:
            raise ControllerRefusal("calibration request privileges are malformed")
        candidate_sha = None
        family_id = None
    else:
        if request.get("candidate_authorized") is not True:
            raise ControllerRefusal("scientific request does not authorize candidate bytes")
        candidate_sha = request.get("impl_sha256")
        family = request.get("family")
        if not isinstance(family, dict):
            raise ControllerRefusal("scientific request family is malformed")
        family_id = family.get("family_id")
    if candidate_sha is not None and (
        not isinstance(candidate_sha, str)
        or len(candidate_sha) != 64
        or any(ch not in "0123456789abcdef" for ch in candidate_sha)
    ):
        raise ControllerRefusal("gate request candidate SHA-256 is malformed")
    if family_id is not None and (not isinstance(family_id, str) or not family_id):
        raise ControllerRefusal("gate request family_id is malformed")
    return {
        "request_sha256": request_sha256,
        "campaign_id": campaign,
        "mode": mode,
        "shape_id": shape_id,
        "candidate_sha256": candidate_sha,
        "family_id": family_id,
        "expires_at": iso_utc(expires),
    }


def _load_official():
    spec = importlib.util.spec_from_file_location("trusted_controller_official", OFFICIAL)
    if spec is None or spec.loader is None:
        raise ControllerRefusal("cannot load official benchmark")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _response_schema(response: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "request_id",
        "challenge_nonce",
        "candidate_sha256",
        "official_sha256",
        "shapes_sha256",
        "shape_id",
        "correctness",
        "challenge_outputs",
        "supporting_timing",
        "effective_numerical_state",
        "environment",
    }
    if set(response) != required or response.get("schema_version") != 1:
        raise ControllerRefusal("worker response failed exact schema validation")


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControllerRefusal(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ControllerRefusal(f"{label} must be finite"
                                + (" and positive" if positive else ""))
    return number


def _hex64(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ControllerRefusal(f"{label} must be a lowercase SHA-256")
    return value


def _exact_file_set(directory: Path, expected: set[str]) -> None:
    actual: set[str] = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ControllerRefusal("side evaluator emitted a linked or non-file artifact")
        actual.add(path.name)
    if actual != expected:
        raise ControllerRefusal(
            f"side evaluator artifact set mismatch: expected {sorted(expected)}, "
            f"got {sorted(actual)}"
        )


def _single_new_packet(directory: Path, before: set[str], prefix: str) -> Path:
    after = {path.name for path in directory.iterdir()}
    created = sorted(after - before)
    if len(created) != 1 or not created[0].startswith(prefix) or not created[0].endswith(".json"):
        raise ControllerRefusal(f"side evaluator did not emit exactly one {prefix} packet")
    path = directory / created[0]
    if path.is_symlink() or not path.is_file():
        raise ControllerRefusal("side evaluator packet is not a regular file")
    return path


def _load_side_packet(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ControllerRefusal(f"{label} packet is malformed") from exc
    if not isinstance(value, dict):
        raise ControllerRefusal(f"{label} packet must be an object")
    return value, raw


def _validate_side_binding(
    packet: Mapping[str, Any], *, evaluator: Path, candidate_sha256: str
) -> None:
    expected = {
        "submission_sha256": candidate_sha256,
        "evaluator_sha256": sha256_file(evaluator),
        "official_sha256": sha256_file(OFFICIAL),
        "official_manifest_sha256": sha256_file(MANIFEST),
    }
    if packet.get("binding") != expected:
        raise ControllerRefusal("side packet byte binding mismatch")


def validate_shape6_packet(packet: dict[str, Any], candidate_sha256: str) -> dict[str, Any]:
    if packet.get("schema_version") != "shape6-submission-v2" \
            or packet.get("type") != "shape6_submission_evaluation":
        raise ControllerRefusal("shape-6 packet type/schema mismatch")
    _validate_side_binding(
        packet, evaluator=SHAPE6_EVALUATOR, candidate_sha256=candidate_sha256
    )
    if packet.get("shape") != _shape(6):
        raise ControllerRefusal("shape-6 packet does not bind official shape 6")
    candidate = packet.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("sha256") != candidate_sha256:
        raise ControllerRefusal("shape-6 candidate binding mismatch")
    correctness = packet.get("correctness")
    if not isinstance(correctness, dict):
        raise ControllerRefusal("shape-6 correctness is malformed")
    seeds = correctness.get("seeds")
    trials = correctness.get("trials")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 5
        or len(set(seeds)) != len(seeds)
        or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        or not isinstance(trials, list)
        or len(trials) != len(seeds)
    ):
        raise ControllerRefusal("shape-6 correctness trials/seeds are malformed")
    violations = 0
    nonfinite = 0
    for seed, trial in zip(seeds, trials):
        if not isinstance(trial, dict) or trial.get("seed") != seed:
            raise ControllerRefusal("shape-6 correctness trial binding mismatch")
        trial_violations = trial.get("violations")
        trial_nonfinite = trial.get("nonfinite_elements")
        if (
            isinstance(trial_violations, bool)
            or not isinstance(trial_violations, int)
            or trial_violations < 0
            or isinstance(trial_nonfinite, bool)
            or not isinstance(trial_nonfinite, int)
            or trial_nonfinite < 0
            or trial.get("passed") is not (trial_violations == 0)
        ):
            raise ControllerRefusal("shape-6 correctness trial arithmetic mismatch")
        violations += trial_violations
        nonfinite += trial_nonfinite
    correctness_passed = violations == 0 and nonfinite == 0
    if (
        correctness.get("violations") != violations
        or correctness.get("nonfinite_elements") != nonfinite
        or correctness.get("passed") is not correctness_passed
    ):
        raise ControllerRefusal("shape-6 correctness aggregate mismatch")

    timing = packet.get("timing")
    if not isinstance(timing, dict):
        raise ControllerRefusal("shape-6 timing is malformed")
    if (
        timing.get("warmups") != 20
        or timing.get("repeats_per_round") != 100
        or timing.get("round_count") != 3
    ):
        raise ControllerRefusal("shape-6 timing protocol mismatch")
    samples_value = timing.get("raw_samples_ms")
    if not isinstance(samples_value, list) or len(samples_value) != 300:
        raise ControllerRefusal("shape-6 timing must retain exactly 300 samples")
    samples = [
        _finite_number(value, "shape6 timing sample", positive=True)
        for value in samples_value
    ]
    median_ms = statistics.median(samples)
    _same_float(timing.get("median_ms"), median_ms, "shape6 median_ms")
    rounds = timing.get("rounds")
    if (
        not isinstance(rounds, list)
        or len(rounds) != 3
        or any(
            not isinstance(item, dict)
            or item.get("round") != index
            or item.get("samples_ms") != samples[index * 100:(index + 1) * 100]
            for index, item in enumerate(rounds)
        )
    ):
        raise ControllerRefusal("shape-6 round/raw-sample binding mismatch")

    memory = packet.get("memory")
    if not isinstance(memory, dict) or memory.get("repeats") != 10:
        raise ControllerRefusal("shape-6 memory protocol mismatch")
    allocated = memory.get("settled_allocated_bytes_per_repeat")
    reserved = memory.get("settled_reserved_bytes_per_repeat")
    if (
        not isinstance(allocated, list)
        or not isinstance(reserved, list)
        or len(allocated) != 10
        or len(reserved) != 10
        or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                   for value in allocated + reserved)
    ):
        raise ControllerRefusal("shape-6 allocator histories are malformed")
    def slope(values: list[int]) -> float:
        center_x = (len(values) - 1) / 2.0
        center_y = statistics.fmean(values)
        return sum(
            (index - center_x) * (value - center_y)
            for index, value in enumerate(values)
        ) / sum((index - center_x) ** 2 for index in range(len(values)))
    allocated_slope = slope(allocated)
    reserved_slope = slope(reserved)
    limits = memory.get("limits")
    required_limits = {
        "allocated_slope_bytes_per_repeat": 16 * 2**20,
        "reserved_slope_bytes_per_repeat": 64 * 2**20,
        "allocated_end_growth_bytes": 64 * 2**20,
        "reserved_end_growth_bytes": 256 * 2**20,
        "allocated_max_growth_bytes": 64 * 2**20,
        "reserved_max_growth_bytes": 256 * 2**20,
    }
    if limits != required_limits:
        raise ControllerRefusal("shape-6 memory limits changed")
    allocated_growth = allocated[-1] - allocated[0]
    reserved_growth = reserved[-1] - reserved[0]
    allocated_max_growth = max(allocated) - allocated[0]
    reserved_max_growth = max(reserved) - reserved[0]
    memory_flat = (
        allocated_slope <= required_limits["allocated_slope_bytes_per_repeat"]
        and reserved_slope <= required_limits["reserved_slope_bytes_per_repeat"]
        and allocated_growth <= required_limits["allocated_end_growth_bytes"]
        and reserved_growth <= required_limits["reserved_end_growth_bytes"]
        and allocated_max_growth <= required_limits["allocated_max_growth_bytes"]
        and reserved_max_growth <= required_limits["reserved_max_growth_bytes"]
    )
    for key, expected in (
        ("allocated_slope_bytes_per_repeat", allocated_slope),
        ("reserved_slope_bytes_per_repeat", reserved_slope),
        ("allocated_end_growth_bytes", allocated_growth),
        ("reserved_end_growth_bytes", reserved_growth),
        ("allocated_max_growth_bytes", allocated_max_growth),
        ("reserved_max_growth_bytes", reserved_max_growth),
    ):
        _same_float(memory.get(key), float(expected), f"shape6 memory.{key}")
    if memory.get("flat") is not memory_flat:
        raise ControllerRefusal("shape-6 memory.flat arithmetic mismatch")
    passed = correctness_passed and memory_flat
    if packet.get("passed") is not passed:
        raise ControllerRefusal("shape-6 packet pass flag mismatch")
    return {
        "passed": passed,
        "correctness_passed": correctness_passed,
        "memory_flat": memory_flat,
        "median_ms": median_ms,
        "sample_count": len(samples),
    }


def validate_shape14_packets(
    validation: dict[str, Any],
    decomposition: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    candidate_sha256: str,
    validation_sha256: str,
    decomposition_sha256: str,
) -> dict[str, Any]:
    expected_types = (
        (validation, "shape14-oracle-validation-v2", "shape14_oracle_validation"),
        (decomposition, "shape14-decomposition-v2", "shape14_batch_decomposition_check"),
        (evaluation, "shape14-streamed-v2", "shape14_side_evaluation"),
    )
    for packet, schema, packet_type in expected_types:
        if packet.get("schema_version") != schema or packet.get("type") != packet_type:
            raise ControllerRefusal("shape-14 prerequisite/evaluation type mismatch")
        _validate_side_binding(
            packet, evaluator=SHAPE14_EVALUATOR, candidate_sha256=candidate_sha256
        )
    validation_results = validation.get("results")
    if not isinstance(validation_results, list) or len(validation_results) < 9:
        raise ControllerRefusal("shape-14 oracle validation is incomplete")
    validation_passed = True
    for result in validation_results:
        if not isinstance(result, dict):
            raise ControllerRefusal("shape-14 oracle validation row is malformed")
        error = _finite_number(result.get("max_abs_err"), "shape14 oracle max error")
        mismatches = result.get("mismatch_at_1e-4")
        if isinstance(mismatches, bool) or not isinstance(mismatches, int) or mismatches < 0:
            raise ControllerRefusal("shape-14 oracle mismatch count is malformed")
        validation_passed &= error < 1e-4 and mismatches == 0
    if validation.get("passed") is not validation_passed:
        raise ControllerRefusal("shape-14 oracle validation pass mismatch")

    official_predicate = decomposition.get("official_predicate")
    if not isinstance(official_predicate, dict):
        raise ControllerRefusal("shape-14 decomposition predicate is malformed")
    difference = _finite_number(
        decomposition.get("max_abs_difference"), "shape14 decomposition difference"
    )
    difference_limit = _finite_number(
        decomposition.get("max_abs_difference_limit"),
        "shape14 decomposition limit",
        positive=True,
    )
    decomp_passed = (
        official_predicate.get("violations") == 0
        and official_predicate.get("nonfinite_elements") == 0
        and difference <= difference_limit
    )
    if decomposition.get("passed") is not decomp_passed:
        raise ControllerRefusal("shape-14 decomposition pass mismatch")

    expected_shape14 = {
        key: value for key, value in _shape(14).items() if key != "notes"
    }
    if evaluation.get("shape") != expected_shape14 \
            or evaluation.get("submission_sha256") != candidate_sha256:
        raise ControllerRefusal("shape-14 final shape/submission binding mismatch")
    required = evaluation.get("required_artifacts")
    if not isinstance(required, dict) or set(required) != {
        "oracle_validation", "batch_decomposition"
    }:
        raise ControllerRefusal("shape-14 prerequisite bindings are malformed")
    if (
        required["oracle_validation"].get("sha256") != validation_sha256
        or required["batch_decomposition"].get("sha256") != decomposition_sha256
    ):
        raise ControllerRefusal("shape-14 prerequisite hash binding mismatch")
    correctness = evaluation.get("correctness")
    seeds = evaluation.get("seeds")
    if not isinstance(correctness, dict) or not isinstance(seeds, list) or len(seeds) < 5:
        raise ControllerRefusal("shape-14 correctness evidence is incomplete")
    trials = correctness.get("trials")
    if not isinstance(trials, list) or len(trials) != len(seeds):
        raise ControllerRefusal("shape-14 correctness trial count mismatch")
    violations = 0
    nonfinite = 0
    elements = 0
    for seed, trial in zip(seeds, trials):
        if not isinstance(trial, dict) or trial.get("base_seed") != seed:
            raise ControllerRefusal("shape-14 correctness seed binding mismatch")
        values = [trial.get(key) for key in ("violations", "nonfinite_elements", "elements")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ControllerRefusal("shape-14 correctness counters are malformed")
        violations += values[0]
        nonfinite += values[1]
        elements += values[2]
    correctness_passed = violations == 0 and nonfinite == 0
    if (
        correctness.get("violations") != violations
        or correctness.get("nonfinite_elements") != nonfinite
        or correctness.get("elements") != elements
        or correctness.get("passed") is not correctness_passed
    ):
        raise ControllerRefusal("shape-14 correctness aggregate mismatch")
    timing = evaluation.get("timing")
    if not isinstance(timing, dict):
        raise ControllerRefusal("shape-14 timing is malformed")
    repeat_sums_value = timing.get("gpu_compute_sum_ms_per_repeat")
    wall_value = timing.get("staging_inclusive_wall_ms_per_repeat")
    repeat_count = timing.get("timing_repeats")
    matrix = timing.get("slice_times_ms")
    if (
        isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
        or repeat_count < 3
        or not isinstance(repeat_sums_value, list)
        or len(repeat_sums_value) != repeat_count
        or not isinstance(wall_value, list)
        or len(wall_value) != repeat_count
        or not isinstance(matrix, dict)
        or matrix.get("orientation") != "batch_index x timing_repeat"
    ):
        raise ControllerRefusal("shape-14 timing repeats are malformed")
    matrix_values = matrix.get("values")
    if (
        not isinstance(matrix_values, list)
        or len(matrix_values) != 32
        or any(not isinstance(row, list) or len(row) != repeat_count for row in matrix_values)
    ):
        raise ControllerRefusal("shape-14 timing matrix must be 32 x repeats")
    computed_sums = [0.0] * repeat_count
    for row in matrix_values:
        for index, value in enumerate(row):
            computed_sums[index] += _finite_number(
                value, "shape14 slice timing", positive=True
            )
    repeat_sums = [
        _finite_number(value, "shape14 repeat sum", positive=True)
        for value in repeat_sums_value
    ]
    # Packet values are deliberately rounded to six decimals; bind within the
    # maximum accumulated rounding error for 32 slices.
    for actual, expected in zip(repeat_sums, computed_sums):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=32e-6):
            raise ControllerRefusal("shape-14 repeat sum disagrees with slice timings")
    compute_median = statistics.median(repeat_sums)
    wall = [
        _finite_number(value, "shape14 staging wall timing", positive=True)
        for value in wall_value
    ]
    wall_median = statistics.median(wall)
    for actual, expected, label in (
        (timing.get("gpu_compute_median_of_sums_ms"), compute_median,
         "shape14 compute median"),
        (timing.get("staging_inclusive_wall_median_ms"), wall_median,
         "shape14 wall median"),
    ):
        value = _finite_number(actual, label, positive=True)
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-5):
            raise ControllerRefusal(f"{label} disagrees with retained samples")
    passed = validation_passed and decomp_passed and correctness_passed
    if evaluation.get("passed") is not correctness_passed:
        raise ControllerRefusal("shape-14 evaluation pass flag mismatch")
    return {
        "passed": passed,
        "oracle_validation_passed": validation_passed,
        "decomposition_passed": decomp_passed,
        "correctness_passed": correctness_passed,
        "gpu_compute_median_of_sums_ms": compute_median,
        "staging_inclusive_wall_median_ms": wall_median,
        "timing_repeats": repeat_count,
    }


def _same_float(actual: Any, expected: float, label: str) -> None:
    value = _finite_number(actual, label)
    if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ControllerRefusal(f"worker timing arithmetic mismatch: {label}")


def validate_supporting_timing(
    supporting: Any, timing_args: Mapping[str, int]
) -> dict[str, Any]:
    """Validate every retained sample and recompute all worker summaries.

    Candidate-process timings remain pre-audit evidence.  This prevents a
    malformed response, dropped repeat, or inconsistent median from entering
    even that evidence layer.
    """
    required = {
        "baseline",
        "candidate",
        "event_speedup",
        "baseline_wall_ms_per_iter",
        "candidate_wall_ms_per_iter",
        "wall_speedup",
        "event_wall_speedup_agreement_ratio",
        "suspicious",
        "authority",
    }
    if not isinstance(supporting, dict) or set(supporting) != required:
        raise ControllerRefusal("worker timing has an unexpected schema")
    expected_count = timing_args["repeats"] * timing_args["rounds"]

    def validate_stats(value: Any, label: str) -> dict[str, Any]:
        keys = {
            "median_ms", "mean_ms", "p90_ms", "min_ms", "n_samples",
            "raw_samples_ms",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise ControllerRefusal(f"{label} timing stats schema mismatch")
        samples_value = value["raw_samples_ms"]
        if not isinstance(samples_value, list) or len(samples_value) != expected_count:
            raise ControllerRefusal(f"{label} did not retain every timing sample")
        samples = [
            _finite_number(sample, f"{label}.raw_samples_ms", positive=True)
            for sample in samples_value
        ]
        if value["n_samples"] != expected_count:
            raise ControllerRefusal(f"{label}.n_samples mismatch")
        ordered = sorted(samples)
        p90_index = max(0, min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1))
        recomputed = {
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.fmean(samples),
            "p90_ms": ordered[p90_index],
            "min_ms": min(samples),
        }
        for key, expected in recomputed.items():
            _same_float(value[key], expected, f"{label}.{key}")
        return {**recomputed, "n_samples": expected_count, "raw_samples_ms": samples}

    baseline = validate_stats(supporting["baseline"], "baseline")
    candidate = validate_stats(supporting["candidate"], "candidate")
    event_speedup = baseline["median_ms"] / candidate["median_ms"]
    _same_float(supporting["event_speedup"], event_speedup, "event_speedup")
    baseline_wall = _finite_number(
        supporting["baseline_wall_ms_per_iter"],
        "baseline_wall_ms_per_iter",
        positive=True,
    )
    candidate_wall = _finite_number(
        supporting["candidate_wall_ms_per_iter"],
        "candidate_wall_ms_per_iter",
        positive=True,
    )
    wall_speedup = baseline_wall / candidate_wall
    _same_float(supporting["wall_speedup"], wall_speedup, "wall_speedup")
    agreement = max(event_speedup, wall_speedup) / max(
        min(event_speedup, wall_speedup), 1e-12
    )
    _same_float(
        supporting["event_wall_speedup_agreement_ratio"],
        agreement,
        "event_wall_speedup_agreement_ratio",
    )
    if supporting["suspicious"] is not (agreement > 1.25):
        raise ControllerRefusal("worker suspicious flag disagrees with timing samples")
    if supporting["authority"] != "supporting-worker-measurement":
        raise ControllerRefusal("worker timing authority label is invalid")
    return {
        **supporting,
        "baseline": baseline,
        "candidate": candidate,
        "event_speedup": event_speedup,
        "baseline_wall_ms_per_iter": baseline_wall,
        "candidate_wall_ms_per_iter": candidate_wall,
        "wall_speedup": wall_speedup,
        "event_wall_speedup_agreement_ratio": agreement,
    }


def validate_challenge_outputs(
    *,
    response: dict[str, Any],
    request: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Recompute references in this candidate-free controller process."""
    _response_schema(response)
    bindings = (
        ("request_id", request["request_id"]),
        ("challenge_nonce", request["challenge_nonce"]),
        ("candidate_sha256", request["candidate_sha256"]),
        ("official_sha256", request["official_sha256"]),
        ("shapes_sha256", request["shapes_sha256"]),
        ("shape_id", request["shape_id"]),
    )
    for field, expected in bindings:
        if response.get(field) != expected:
            raise ControllerRefusal(f"worker response binding mismatch: {field}")
    worker_correctness = response.get("correctness")
    if not isinstance(worker_correctness, dict) or set(worker_correctness) != {
        "passed", "baseline_invariant", "anti_cache_passed", "trials"
    }:
        raise ControllerRefusal("worker correctness schema mismatch")
    if any(
        not isinstance(worker_correctness[field], bool)
        for field in ("passed", "baseline_invariant", "anti_cache_passed")
    ) or not isinstance(worker_correctness["trials"], list):
        raise ControllerRefusal("worker correctness fields are malformed")
    environment = response.get("environment")
    if not isinstance(environment, dict) or set(environment) != {
        "python", "torch", "cuda", "gpu"
    } or not all(isinstance(value, str) and value for value in environment.values()):
        raise ControllerRefusal("worker environment schema mismatch")
    effective = response.get("effective_numerical_state")
    if not isinstance(effective, dict) or set(effective) != {
        "float32_matmul_precision", "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "NVIDIA_TF32_OVERRIDE",
    }:
        raise ControllerRefusal("worker numerical-state schema mismatch")
    if (
        effective["float32_matmul_precision"] != NUMERICAL["matmul_precision"]
        or effective["cuda_matmul_allow_tf32"] is not True
        or effective["cudnn_allow_tf32"] is not True
        or effective["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] is not None
        or effective["NVIDIA_TF32_OVERRIDE"] is not None
    ):
        raise ControllerRefusal("worker effective numerical state is not official")
    outputs = response.get("challenge_outputs")
    if not isinstance(outputs, list) or len(outputs) != len(request["seeds"]):
        raise ControllerRefusal("worker challenge-output count mismatch")
    by_seed: dict[int, dict[str, Any]] = {}
    for item in outputs:
        if not isinstance(item, dict) or set(item) != {
            "seed", "filename", "sha256", "shape", "dtype", "nbytes"
        }:
            raise ControllerRefusal("worker output metadata schema mismatch")
        seed = item["seed"]
        if seed in by_seed or seed not in request["seeds"]:
            raise ControllerRefusal("worker output seed is duplicate or unrequested")
        filename = item["filename"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ControllerRefusal("worker output filename is unsafe")
        by_seed[seed] = item

    import torch

    if not torch.cuda.is_available():
        raise ControllerRefusal("CUDA unavailable in trusted controller")
    official = _load_official()
    shape = request["shape"]
    config = official.TransformerConfig(
        batch_size=shape["batch_size"],
        seq_len=shape["seq_len"],
        d_model=shape["d_model"],
        num_heads=shape["num_heads"],
        ffn_dim=shape["ffn_dim"],
        num_layers=shape["num_layers"],
        causal=shape["causal"],
    )
    config.validate()
    device = torch.device("cuda")
    dtype = official.resolve_dtype(request["dtype"])
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision(NUMERICAL["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = NUMERICAL["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = NUMERICAL["allow_tf32"]
    baseline = official.BaselineTransformer(config).to(device=device, dtype=dtype).eval()
    trials: list[dict[str, Any]] = []
    all_passed = True
    expected_shape = [shape["batch_size"], shape["seq_len"], shape["d_model"]]
    expected_elements = math.prod(expected_shape)
    expected_bytes = expected_elements * 4
    with torch.inference_mode():
        for seed in request["seeds"]:
            item = by_seed[seed]
            if (
                item["shape"] != expected_shape
                or item["dtype"] != "float32"
                or item["nbytes"] != expected_bytes
                or not isinstance(item["sha256"], str)
                or len(item["sha256"]) != 64
            ):
                raise ControllerRefusal("worker output metadata does not match official shape")
            path = output_dir / item["filename"]
            if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes:
                raise ControllerRefusal("worker output file is absent, linked, or wrong-sized")
            if sha256_file(path) != item["sha256"]:
                raise ControllerRefusal("worker output file hash mismatch")
            candidate_flat = torch.from_file(
                str(path), shared=False, size=expected_elements, dtype=torch.float32
            )
            candidate = candidate_flat.reshape(expected_shape)
            x, mask = official.generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed,
                padding_ratio=NUMERICAL["padding_ratio"],
                input_scale=NUMERICAL["input_scale"],
            )
            reference = baseline(x, mask).detach().float().cpu()
            result = official.compare_outputs(
                reference, candidate, rtol=NUMERICAL["rtol"], atol=NUMERICAL["atol"]
            )
            trial = {
                "seed": seed,
                "passed": bool(result.passed),
                "failed_elements": int(result.failed_elements),
                "total_elements": int(result.total_elements),
                "max_abs_error": float(result.max_abs_error),
                "max_relative_error": float(result.max_relative_error),
                "mean_abs_error": float(result.mean_abs_error),
                "candidate_output_sha256": item["sha256"],
            }
            trials.append(trial)
            all_passed &= result.passed
    unexpected = {
        path.name for path in output_dir.iterdir()
        if path.name != "response.json"
    } - {item["filename"] for item in outputs}
    if unexpected:
        raise ControllerRefusal(f"worker emitted unexpected files: {sorted(unexpected)}")
    if len(worker_correctness["trials"]) != len(request["seeds"]):
        raise ControllerRefusal("worker correctness trial count mismatch")
    return {"passed": all_passed, "authority": "trusted-controller", "trials": trials}


def _worker_request(
    *,
    mode: str,
    shape_id: int,
    candidate_sha256: str | None,
) -> dict[str, Any]:
    official_seeds = [1234 + index for index in range(5)]
    secret_seeds: list[int] = []
    while len(secret_seeds) < 2:
        candidate = secrets.randbelow(2**31 - 1)
        if candidate not in official_seeds and candidate not in secret_seeds:
            secret_seeds.append(candidate)
    operation = "calibration" if mode == "calibration" else (
        "diagnostic" if mode == "diagnostic" else "candidate"
    )
    return {
        "schema_version": 1,
        "request_id": secrets.token_hex(16),
        "operation": operation,
        "shape_id": shape_id,
        "shape": _shape(shape_id),
        "dtype": "float32",
        "seeds": official_seeds + secret_seeds,
        "timing_args": TIMING,
        "numerical": NUMERICAL,
        "candidate_sha256": candidate_sha256,
        "official_sha256": sha256_file(OFFICIAL),
        "shapes_sha256": sha256_file(SHAPES),
        "challenge_nonce": secrets.token_hex(32),
    }


def _latest_calibration(
    events: list[dict[str, Any]],
    *,
    campaign_id: str,
    shape_id: int,
    worker_environment: Mapping[str, Any],
) -> dict[str, Any] | None:
    for event in reversed(events):
        payload = event.get("payload", {})
        if (
            event.get("kind") == "measurement_recorded"
            and payload.get("mode") == "calibration"
            and payload.get("campaign_id") == campaign_id
            and payload.get("shape_id") == shape_id
            and payload.get("worker_environment") == worker_environment
            and payload.get("timing_args") == TIMING
            and payload.get("numerical") == NUMERICAL
        ):
            # Return the immutable event, not only its payload: the event ID
            # is the calibration binding.  An event cannot contain its own ID
            # inside the already-hashed payload without circularity.
            return event
    return None


def terminalize_consumed_run(method):
    """Guarantee a consumed permit cannot disappear into an open run.

    The normal path records richer failure data for a worker exit.  This
    outer guard covers every later parser, schema, correctness, timing,
    storage, and unexpected controller exception.  Recovery therefore has a
    durable terminal fact even if the controller crashes after consumption.
    """
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        permit_id = kwargs.get("permit_id")
        if permit_id is None and args:
            permit_id = args[0]
        try:
            return method(self, *args, **kwargs)
        except BaseException as exc:
            if isinstance(permit_id, str) and permit_id:
                events = self.store.read_events()
                starts = [
                    event for event in events
                    if event.get("kind") == "run_started"
                    and event.get("payload", {}).get("permit_id") == permit_id
                ]
                if len(starts) == 1:
                    run_id = starts[0]["payload"].get("run_id")
                    terminal = [
                        event for event in events
                        if event.get("kind") in {"measurement_recorded", "run_failed"}
                        and event.get("payload", {}).get("run_id") == run_id
                    ]
                    if not terminal:
                        self.store.append(
                            kind="run_failed",
                            actor="trusted-controller",
                            payload={
                                "run_id": run_id,
                                "started_event_id": starts[0]["event_id"],
                                "reason": "controller_validation_or_storage_failure",
                                "error_type": type(exc).__name__,
                            },
                        )
            raise
    return wrapped


class TrustedController:
    def __init__(self, root: Path = ROOT):
        self.root = root.resolve()
        self.store = AuthorityStore(self.root)

    def verify_lock_only(self) -> dict[str, Any]:
        lock = verify_lock(self.root)
        activations = [
            event for event in self.store.read_events()
            if event.get("kind") in {"lock_activated", "lock_rotated"}
        ]
        active = False
        if activations:
            latest = activations[-1]
            payload = latest.get("payload", {})
            active = (
                payload.get("lock_id") == lock["lock_id"]
                and payload.get("subject_sha256") == lock["manifest_sha256"]
                and payload.get("capability_consumed") is True
                and payload.get("capability_role") == "owner"
                and payload.get("capability_action") in {"lock.activate", "lock.rotate"}
            )
        return {**lock, "active": active}

    def require_lock(self) -> dict[str, Any]:
        lock = self.verify_lock_only()
        if not lock["active"]:
            raise ControllerRefusal(
                "signed LOCK bytes are valid but the epoch is not owner-activated"
            )
        return lock

    def activate_lock(self, capability_path: Path) -> dict[str, Any]:
        lock = verify_lock(self.root)
        events = self.store.read_events()
        activations = [
            event for event in events
            if event.get("kind") in {"lock_activated", "lock_rotated"}
        ]
        if activations:
            previous = activations[-1]
            if previous.get("payload", {}).get("lock_id") == lock["lock_id"]:
                raise ControllerRefusal("this LOCK epoch is already activated")
            action = "lock.rotate"
            kind = "lock_rotated"
            previous_event_sha = previous["event_sha256"]
        else:
            # No authority-generating event may predate initial activation.
            if events:
                raise ControllerRefusal(
                    "initial LOCK activation requires an empty authority journal"
                )
            action = "lock.activate"
            kind = "lock_activated"
            previous_event_sha = None
        capability = _strict_object(capability_path, "LOCK activation capability")
        event = self.store.append_authorized(
            kind=kind,
            actor="trusted-controller",
            payload={
                "subject_sha256": lock["manifest_sha256"],
                "lock_id": lock["lock_id"],
                "epoch": lock["epoch"],
                "previous_activation_event_sha256": previous_event_sha,
            },
            capability_document=capability,
            action=action,
            target=f"lock:{lock['lock_id']}",
        )
        return {
            **lock,
            "active": True,
            "activation_event_id": event["event_id"],
            "activation_event_sha256": event["event_sha256"],
            "activation_action": action,
        }

    def issue_permit(self, request_path: Path, capability_path: Path) -> dict[str, Any]:
        self.require_lock()
        request, digest, raw = _gate_request(
            request_path, self.root / "Project" / "loop" / "requests"
        )
        normalized = normalize_permit_request(request, digest)
        stored_digest, _ = self.store.store_blob(raw, suffix=".json")
        if stored_digest != digest:
            raise ControllerRefusal("request blob changed during ingestion")
        capability = _strict_object(capability_path, "owner capability")
        permit = self.store.issue_permit(
            request=normalized,
            request_blob_sha256=digest,
            capability_document=capability,
        )
        return permit

    def authorize(
        self,
        *,
        capability_path: Path,
        action: str,
        target: str,
        subject_sha256: str,
        campaign_id: str,
    ) -> dict[str, Any]:
        self.require_lock()
        if len(subject_sha256) != 64:
            raise ControllerRefusal("authorized subject SHA-256 is malformed")
        capability = _strict_object(capability_path, "signed capability")
        event = self.store.append_authorized(
            kind=action.replace(".", "_") + "_authorized",
            actor="trusted-controller",
            payload={"subject_sha256": subject_sha256},
            capability_document=capability,
            action=action,
            target=target,
            campaign_id=campaign_id,
        )
        payload = event["payload"]
        receipt = {
            "authority_event_id": event["event_id"],
            "authority_event_sha256": event["event_sha256"],
            "action": action,
            "subject_sha256": subject_sha256,
            "capability_nonce": payload["capability_nonce"],
            "role": payload["capability_role"],
        }
        receipts = self.root / "Project" / "authority" / "receipts"
        receipts.mkdir(parents=True, exist_ok=True)
        path = receipts / f"{event['event_id']}.json"
        if path.exists():
            raise ControllerRefusal("authority receipt path already exists")
        atomic_write(path, canonical_json(receipt) + b"\n", mode=0o440)
        return {**receipt, "receipt_path": str(path.relative_to(self.root))}

    def verify_receipt(
        self, receipt_path: Path, action: str, subject_sha256: str
    ) -> dict[str, Any]:
        self.require_lock()
        receipt = _strict_object(receipt_path, "authority receipt")
        if set(receipt) != RECEIPT_KEYS:
            raise ControllerRefusal("authority receipt schema mismatch")
        if receipt["action"] != action or receipt["subject_sha256"] != subject_sha256:
            raise ControllerRefusal("authority receipt subject/action mismatch")
        events = self.store.read_events()
        matches = [
            event for event in events
            if event["event_id"] == receipt["authority_event_id"]
        ]
        if len(matches) != 1:
            raise ControllerRefusal("authority receipt event is absent or duplicated")
        event = matches[0]
        payload = event["payload"]
        if (
            event["event_sha256"] != receipt["authority_event_sha256"]
            or payload.get("capability_consumed") is not True
            or payload.get("capability_action") != action
            or payload.get("subject_sha256") != subject_sha256
            or payload.get("capability_nonce") != receipt["capability_nonce"]
            or payload.get("capability_role") != receipt["role"]
        ):
            raise ControllerRefusal("authority receipt does not match durable authority event")
        return {
            "valid": True,
            "authority_event_id": event["event_id"],
            "authority_event_sha256": event["event_sha256"],
            "capability_nonce": receipt["capability_nonce"],
            "role": receipt["role"],
            "action": action,
            "subject_sha256": subject_sha256,
        }

    def _bound_gate_request(self, permit: Mapping[str, Any]) -> dict[str, Any]:
        digest = _hex64(permit.get("request_sha256"), "permit request hash")
        path = self.store.paths.blobs / f"{digest}.json"
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ControllerRefusal("permit's immutable gate request is absent or changed")
        request = _strict_object(path, "immutable gate request")
        normalized = normalize_permit_request(request, digest)
        for key, value in normalized.items():
            if permit.get(key) != value:
                raise ControllerRefusal(f"permit/gate-request binding mismatch: {key}")
        return request

    @staticmethod
    def _side_mounts(evaluator: Path, output_dir: Path) -> list[IsolatedMount]:
        return [
            IsolatedMount(evaluator, f"/sandbox/Project/tools/{evaluator.name}"),
            IsolatedMount(
                SUBMISSION,
                "/sandbox/Project/submission/torch_transformer_benchmark_submission.py",
            ),
            IsolatedMount(MANIFEST, "/sandbox/Project/manifest.json"),
            IsolatedMount(OFFICIAL, "/sandbox/torch_transformer_benchmark.py"),
            IsolatedMount(
                TENSORFLOW_OFFICIAL, "/sandbox/tensorflow_transformer_benchmark.py"
            ),
            IsolatedMount(output_dir, "/sandbox/Project/results_side", writable=True),
        ]

    def _run_side_stage(
        self,
        *,
        run_id: str,
        evaluator: Path,
        output_dir: Path,
        arguments: list[str],
        prefix: str,
        stage: str,
        timeout_seconds: int,
    ) -> tuple[dict[str, Any], bytes, str, SandboxResult]:
        before = {path.name for path in output_dir.iterdir()}
        result = run_isolated_command(
            mounts=self._side_mounts(evaluator, output_dir),
            argv=[
                "/usr/bin/python3",
                f"/sandbox/Project/tools/{evaluator.name}",
                *arguments,
            ],
            cwd="/sandbox",
            timeout_seconds=timeout_seconds,
        )
        stdout_sha, _ = self.store.store_blob(result.stdout, suffix=f".{stage}.stdout")
        stderr_sha, _ = self.store.store_blob(result.stderr, suffix=f".{stage}.stderr")
        after = {path.name for path in output_dir.iterdir()}
        created = after - before
        artifact_sha = None
        artifact_raw = None
        artifact_packet = None
        artifact_name = None
        try:
            artifact_path = _single_new_packet(output_dir, before, prefix)
            artifact_packet, artifact_raw = _load_side_packet(artifact_path, stage)
            artifact_sha, _ = self.store.store_blob(artifact_raw, suffix=".json")
            artifact_name = artifact_path.name
            self.store.append(
                kind="side_stage_ingested",
                actor="trusted-controller",
                payload={
                    "run_id": run_id,
                    "stage": stage,
                    "artifact_sha256": artifact_sha,
                    "artifact_name": artifact_name,
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "stdout_sha256": stdout_sha,
                    "stderr_sha256": stderr_sha,
                },
            )
        except Exception:
            # Preserve every adverse byte we can before failing closed.  An
            # unexpected artifact set remains visible through its names and
            # the captured process streams.
            self.store.append(
                kind="side_stage_ingest_failed",
                actor="trusted-controller",
                payload={
                    "run_id": run_id,
                    "stage": stage,
                    "created_artifact_names": sorted(created),
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "stdout_sha256": stdout_sha,
                    "stderr_sha256": stderr_sha,
                },
            )
            raise
        if result.timed_out or result.returncode != 0:
            failed = self.store.append(
                kind="run_failed",
                actor="trusted-controller",
                payload={
                    "run_id": run_id,
                    "reason": "side_stage_timeout" if result.timed_out else "side_stage_failed",
                    "stage": stage,
                    "returncode": result.returncode,
                    "artifact_sha256": artifact_sha,
                    "stdout_sha256": stdout_sha,
                    "stderr_sha256": stderr_sha,
                },
            )
            raise ControllerRefusal(
                f"{stage} failed; adverse evidence preserved in {failed['event_id']}"
            )
        assert artifact_packet is not None and artifact_raw is not None
        assert artifact_sha is not None and artifact_name is not None
        return artifact_packet, artifact_raw, artifact_sha, result

    @terminalize_consumed_run
    def run_side(self, *, permit_id: str, timeout_seconds: int) -> dict[str, Any]:
        lock = self.require_lock()
        issued = [
            event for event in self.store.read_events()
            if event["kind"] == "permit_issued"
            and event["payload"].get("permit_id") == permit_id
        ]
        if len(issued) != 1:
            raise ControllerRefusal("side permit is absent or duplicated")
        permit = issued[0]["payload"]
        mode = permit.get("mode")
        shape_id = permit.get("shape_id")
        if (mode, shape_id) not in {("shape6", 6), ("shape14", 14)}:
            raise ControllerRefusal("permit is not a dedicated side-shape permit")
        request = self._bound_gate_request(permit)
        if request.get("request_kind") != "side_evaluation":
            raise ControllerRefusal("side permit did not originate from a side request")
        candidate_sha = _hex64(permit.get("candidate_sha256"), "side candidate hash")
        if (
            request.get("impl_path")
            != "Project/submission/torch_transformer_benchmark_submission.py"
            or sha256_file(SUBMISSION) != candidate_sha
        ):
            raise ControllerRefusal("side request does not bind the exact generated submission")
        self.store.store_blob(SUBMISSION.read_bytes(), suffix=".py")
        consumed = self.store.consume_permit(
            permit_id=permit_id,
            mode=mode,
            shape_id=shape_id,
            candidate_sha256=candidate_sha,
        )
        run_id = f"run-{secrets.token_hex(16)}"
        started = self.store.append(
            kind="run_started",
            actor="trusted-controller",
            payload={
                "run_id": run_id,
                "permit_id": permit_id,
                "consumed_event_id": consumed["consumed_event_id"],
                "campaign_id": permit["campaign_id"],
                "mode": mode,
                "lane": mode,
                "shape_id": shape_id,
                "candidate_sha256": candidate_sha,
                "gate_request_sha256": permit["request_sha256"],
                "lock_id": lock["lock_id"],
                "lock_manifest_sha256": lock["manifest_sha256"],
            },
        )
        stage_packets: list[dict[str, Any]] = []
        stage_blobs: list[dict[str, str]] = []
        with tempfile.TemporaryDirectory(prefix=f"trusted-{mode}-") as temp:
            output_dir = Path(temp) / "results_side"
            output_dir.mkdir()
            if mode == "shape6":
                packet, raw, digest, _ = self._run_side_stage(
                    run_id=run_id,
                    evaluator=SHAPE6_EVALUATOR,
                    output_dir=output_dir,
                    arguments=[],
                    prefix="shape6_submission_",
                    stage="shape6-eval",
                    timeout_seconds=timeout_seconds,
                )
                summary = validate_shape6_packet(packet, candidate_sha)
                entry_id = packet.get("entry_id")
                stage_packets.append(packet)
                stage_blobs.append({"stage": "shape6-eval", "sha256": digest})
                side_evidence_sha = digest
            else:
                validation, _, validation_sha, _ = self._run_side_stage(
                    run_id=run_id,
                    evaluator=SHAPE14_EVALUATOR,
                    output_dir=output_dir,
                    arguments=["validate", "--seeds", "3"],
                    prefix="shape14_validation_",
                    stage="shape14-validate",
                    timeout_seconds=timeout_seconds,
                )
                validation_name = next(
                    name for name in {path.name for path in output_dir.iterdir()}
                    if name.startswith("shape14_validation_")
                )
                decomposition, _, decomposition_sha, _ = self._run_side_stage(
                    run_id=run_id,
                    evaluator=SHAPE14_EVALUATOR,
                    output_dir=output_dir,
                    arguments=[
                        "decomp-check", "--batch", "8", "--seq", "2048",
                        "--seed", "1234", "--max-abs-difference", "0.00001",
                    ],
                    prefix="shape14_decomposition_",
                    stage="shape14-decomposition",
                    timeout_seconds=timeout_seconds,
                )
                decomposition_name = next(
                    name for name in {path.name for path in output_dir.iterdir()}
                    if name.startswith("shape14_decomposition_")
                )
                evaluation, _, evaluation_sha, _ = self._run_side_stage(
                    run_id=run_id,
                    evaluator=SHAPE14_EVALUATOR,
                    output_dir=output_dir,
                    arguments=[
                        "eval",
                        "--validation-packet",
                        f"/sandbox/Project/results_side/{validation_name}",
                        "--decomposition-packet",
                        f"/sandbox/Project/results_side/{decomposition_name}",
                        "--correctness-seeds", "5",
                        "--timing-repeats", "3",
                        "--warmup", "3",
                    ],
                    prefix="shape14_streamed_",
                    stage="shape14-eval",
                    timeout_seconds=timeout_seconds,
                )
                # Later stages had write access to the prerequisite directory;
                # prove those immutable prerequisite bytes did not change.
                validation_path = output_dir / validation_name
                decomposition_path = output_dir / decomposition_name
                if (
                    sha256_file(validation_path) != validation_sha
                    or sha256_file(decomposition_path) != decomposition_sha
                ):
                    raise ControllerRefusal("shape-14 prerequisite packet changed between stages")
                summary = validate_shape14_packets(
                    validation,
                    decomposition,
                    evaluation,
                    candidate_sha256=candidate_sha,
                    validation_sha256=validation_sha,
                    decomposition_sha256=decomposition_sha,
                )
                entry_id = evaluation.get("entry_id")
                stage_packets.extend([validation, decomposition, evaluation])
                stage_blobs.extend([
                    {"stage": "shape14-validate", "sha256": validation_sha},
                    {"stage": "shape14-decomposition", "sha256": decomposition_sha},
                    {"stage": "shape14-eval", "sha256": evaluation_sha},
                ])
                side_evidence_sha = evaluation_sha
            _exact_file_set(
                output_dir,
                {
                    next(
                        path.name for path in output_dir.iterdir()
                        if sha256_file(path) == item["sha256"]
                    )
                    for item in stage_blobs
                },
            )
        if not isinstance(entry_id, str) or not re.fullmatch(
            r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", entry_id
        ):
            raise ControllerRefusal("side evaluator entry ID is malformed")
        measurement = self.store.append(
            kind="measurement_recorded",
            actor="trusted-controller",
            payload={
                "entry_id": entry_id,
                "run_id": run_id,
                "started_event_id": started["event_id"],
                "permit_id": permit_id,
                "campaign_id": permit["campaign_id"],
                "mode": mode,
                "lane": mode,
                "shape_id": shape_id,
                "candidate_sha256": candidate_sha,
                "family_id": None,
                "gate_request_sha256": permit["request_sha256"],
                "side_evidence_sha256": side_evidence_sha,
                "side_stage_artifacts": stage_blobs,
                "controller_validation": summary,
                "evidence_eligible_pre_audit": summary["passed"],
                "promotion_eligible": False,
                "promotion_blocker": "side_evidence_not_primary_champion",
                "lock_id": lock["lock_id"],
                "lock_manifest_sha256": lock["manifest_sha256"],
            },
        )
        wrapper = {
            "schema_version": 1,
            "entry_id": entry_id,
            "lane": mode,
            "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "candidate_sha256": candidate_sha,
            "permit_id": permit_id,
            "gate_request_sha256": permit["request_sha256"],
            "campaign_id": permit["campaign_id"],
            "mode": mode,
            "shape_id": shape_id,
            "side_evidence_sha256": side_evidence_sha,
            "side_stage_artifacts": stage_blobs,
            "controller_validation": summary,
            "side_evidence_packets": stage_packets,
            "lock_manifest_sha256": lock["manifest_sha256"],
        }
        packet_sha, _ = self.store.store_blob(canonical_json(wrapper) + b"\n", suffix=".json")
        binding = self.store.append(
            kind="measurement_packet_bound",
            actor="trusted-controller",
            payload={
                "entry_id": entry_id,
                "measurement_event_id": measurement["event_id"],
                "measurement_event_sha256": measurement["event_sha256"],
                "candidate_sha256": candidate_sha,
                "packet_sha256": packet_sha,
                "side_evidence_sha256": side_evidence_sha,
                "lane": mode,
            },
        )
        audit_enqueued = False
        if summary["passed"]:
            from audit_authority import enqueue_audit
            enqueue_audit(
                entry_id=entry_id,
                candidate_sha256=candidate_sha,
                packet_sha256=packet_sha,
                measurement_event_sha256=measurement["event_sha256"],
                lane=mode,
            )
            audit_enqueued = True
        return {
            "entry_id": entry_id,
            "lane": mode,
            "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "packet_binding_event_id": binding["event_id"],
            "packet_sha256": packet_sha,
            "side_evidence_sha256": side_evidence_sha,
            "passed": summary["passed"],
            "audit_enqueued": audit_enqueued,
            "promotion_eligible": False,
        }

    @terminalize_consumed_run
    def run_primary(
        self,
        *,
        permit_id: str,
        shape_id: int,
        impl_path: Path | None,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        lock = self.require_lock()
        events_before = self.store.read_events()
        issued = [
            event for event in events_before
            if event["kind"] == "permit_issued"
            and event["payload"].get("permit_id") == permit_id
        ]
        if len(issued) != 1:
            raise ControllerRefusal("permit is absent or duplicated")
        permit = issued[0]["payload"]
        gate_request = self._bound_gate_request(permit)
        mode = permit["mode"]
        if permit["shape_id"] != shape_id:
            raise ControllerRefusal("CLI shape disagrees with permit")
        if mode in {"shape6", "shape14"} or gate_request.get("request_kind") == "side_evaluation":
            raise ControllerRefusal("dedicated side request must use the side command")
        if shape_id in {6, 14}:
            raise ControllerRefusal("shape requires its dedicated controller lane")
        candidate_sha = permit["candidate_sha256"]
        candidate: Path
        if mode == "calibration":
            if impl_path is not None or candidate_sha is not None:
                raise ControllerRefusal("calibration cannot receive candidate bytes")
            candidate = WORKER  # mounted placeholder; worker never loads it
        else:
            if impl_path is None:
                raise ControllerRefusal("candidate path is required")
            candidate = _regular_repo_file(
                impl_path,
                "candidate",
                (PROJECT / "kernels", PROJECT / "submission"),
            )
            actual_sha = sha256_file(candidate)
            if actual_sha != candidate_sha:
                raise ControllerRefusal("candidate bytes disagree with one-use permit")
            self.store.store_blob(candidate.read_bytes(), suffix=".py")

        worker_request = _worker_request(
            mode=mode,
            shape_id=shape_id,
            candidate_sha256=candidate_sha,
        )
        request_bytes = canonical_json(worker_request) + b"\n"
        request_sha, _ = self.store.store_blob(request_bytes, suffix=".json")
        # The state transition is irrevocable before candidate process launch.
        consumed = self.store.consume_permit(
            permit_id=permit_id,
            mode=mode,
            shape_id=shape_id,
            candidate_sha256=candidate_sha,
        )
        run_id = f"run-{secrets.token_hex(16)}"
        started = self.store.append(
            kind="run_started",
            actor="trusted-controller",
            payload={
                "run_id": run_id,
                "permit_id": permit_id,
                "consumed_event_id": consumed["consumed_event_id"],
                "campaign_id": permit["campaign_id"],
                "mode": mode,
                "shape_id": shape_id,
                "candidate_sha256": candidate_sha,
                "worker_request_sha256": request_sha,
                "lock_id": lock["lock_id"],
                "lock_manifest_sha256": lock["manifest_sha256"],
            },
        )

        with tempfile.TemporaryDirectory(prefix="trusted-controller-") as temp:
            temp_root = Path(temp)
            request_file = temp_root / "request.json"
            request_file.write_bytes(request_bytes)
            output_dir = temp_root / "output"
            output_dir.mkdir()
            before = time.perf_counter_ns()
            sandbox_result = run_sandbox(
                SandboxFiles(WORKER, candidate, OFFICIAL, SHAPES, request_file, output_dir),
                timeout_seconds=timeout_seconds,
            )
            elapsed_ms = (time.perf_counter_ns() - before) / 1e6
            stdout_sha, _ = self.store.store_blob(sandbox_result.stdout, suffix=".stdout")
            stderr_sha, _ = self.store.store_blob(sandbox_result.stderr, suffix=".stderr")
            if sandbox_result.timed_out or sandbox_result.returncode != 0:
                failed = self.store.append(
                    kind="run_failed",
                    actor="trusted-controller",
                    payload={
                        "run_id": run_id,
                        "started_event_id": started["event_id"],
                        "reason": "timeout" if sandbox_result.timed_out else "worker_exit",
                        "returncode": sandbox_result.returncode,
                        "stdout_sha256": stdout_sha,
                        "stderr_sha256": stderr_sha,
                        "elapsed_ms": elapsed_ms,
                    },
                )
                raise ControllerRefusal(
                    f"candidate worker failed; durable event {failed['event_id']}"
                )
            response_path = output_dir / "response.json"
            if response_path.is_symlink() or not response_path.is_file():
                raise ControllerRefusal("worker completed without a regular response")
            response_bytes = response_path.read_bytes()
            response_sha, _ = self.store.store_blob(response_bytes, suffix=".json")
            try:
                response = json.loads(response_bytes)
            except json.JSONDecodeError as exc:
                raise ControllerRefusal("worker response is malformed") from exc
            if not isinstance(response, dict):
                raise ControllerRefusal("worker response is not an object")
            correctness = validate_challenge_outputs(
                response=response, request=worker_request, output_dir=output_dir
            )

        supporting = validate_supporting_timing(
            response["supporting_timing"], worker_request["timing_args"]
        )
        event_speedup = supporting.get("event_speedup")
        if not isinstance(event_speedup, (int, float)) or not math.isfinite(event_speedup):
            raise ControllerRefusal("worker speedup is invalid")
        performance_eligible = False
        threshold = None
        calibration_event_id = None
        noise = None
        if mode == "calibration":
            noise = abs(1.0 - float(event_speedup))
            threshold = 1.0 + max(0.03, 3.0 * noise)
        elif mode in {"optimization", "confirmation"}:
            calibration = _latest_calibration(
                self.store.read_events(),
                campaign_id=permit["campaign_id"],
                shape_id=shape_id,
                worker_environment=response["environment"],
            )
            if calibration is not None:
                calibration_payload = calibration["payload"]
                threshold = calibration_payload["promotion_threshold"]
                calibration_event_id = calibration["event_id"]
                performance_eligible = (
                    correctness["passed"]
                    and response["correctness"].get("passed") is True
                    and supporting.get("suspicious") is False
                    and float(event_speedup) > threshold
                    and mode == "optimization"
                )
        lane = "scratch" if mode in {"screening", "correctness"} else "primary"
        measurement_payload: dict[str, Any] = {
            "run_id": run_id,
            "started_event_id": started["event_id"],
            "permit_id": permit_id,
            "campaign_id": permit["campaign_id"],
            "mode": mode,
            "lane": lane,
            "shape_id": shape_id,
            "candidate_sha256": candidate_sha,
            "family_id": permit["family_id"],
            "worker_request_sha256": request_sha,
            "worker_response_sha256": response_sha,
            "stdout_sha256": stdout_sha,
            "stderr_sha256": stderr_sha,
            "controller_correctness": correctness,
            "supporting_timing": supporting,
            "worker_environment": response["environment"],
            "effective_numerical_state": response["effective_numerical_state"],
            "timing_args": TIMING,
            "numerical": NUMERICAL,
            "controller_process_elapsed_ms": elapsed_ms,
            "promotion_threshold": threshold,
            "calibrated_noise": noise,
            "calibration_event_id": calibration_event_id,
            "performance_eligible": performance_eligible,
            "promotion_eligible": False,
            "promotion_blocker": "audit_required" if performance_eligible else "performance_or_correctness",
            "lock_id": lock["lock_id"],
            "lock_manifest_sha256": lock["manifest_sha256"],
        }
        measurement = self.store.append(
            kind="measurement_recorded",
            actor="trusted-controller",
            payload=measurement_payload,
        )
        # The payload stores a stable self-reference for later calibration
        # binding via the immutable event that actually contains it.
        packet = {
            "schema_version": 1,
            "entry_id": run_id,
            "lane": lane,
            "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "candidate_sha256": candidate_sha,
            "permit_id": permit_id,
            "gate_request_sha256": permit["request_sha256"],
            "campaign_id": permit["campaign_id"],
            "mode": mode,
            "shape_id": shape_id,
            "family_id": permit["family_id"],
            "worker_request_sha256": request_sha,
            "worker_response_sha256": response_sha,
            "controller_correctness": correctness,
            "supporting_timing": supporting,
            "lock_manifest_sha256": lock["manifest_sha256"],
        }
        packet_sha, _ = self.store.store_blob(canonical_json(packet) + b"\n", suffix=".json")
        binding = self.store.append(
            kind="measurement_packet_bound",
            actor="trusted-controller",
            payload={
                "entry_id": run_id,
                "measurement_event_id": measurement["event_id"],
                "measurement_event_sha256": measurement["event_sha256"],
                "candidate_sha256": candidate_sha,
                "packet_sha256": packet_sha,
                "lane": lane,
            },
        )
        if mode in {"optimization", "confirmation"}:
            try:
                from audit_authority import enqueue_audit
                enqueue_audit(
                    entry_id=run_id,
                    candidate_sha256=candidate_sha,
                    packet_sha256=packet_sha,
                    measurement_event_sha256=measurement["event_sha256"],
                    lane="primary",
                )
            except Exception as exc:
                self.store.append(
                    kind="audit_enqueue_failed",
                    actor="trusted-controller",
                    payload={
                        "entry_id": run_id,
                        "measurement_event_sha256": measurement["event_sha256"],
                        "error_type": type(exc).__name__,
                    },
                )
        return {
            "entry_id": run_id,
            "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "packet_binding_event_id": binding["event_id"],
            "packet_sha256": packet_sha,
            "correct": correctness["passed"],
            "event_speedup": event_speedup,
            "performance_eligible": performance_eligible,
            "promotion_eligible": False,
            "promotion_blocker": "audit_required" if performance_eligible else "performance_or_correctness",
        }

    def status(self) -> dict[str, Any]:
        lock = self.require_lock()
        events = self.store.read_events()
        measurements = [event for event in events if event["kind"] == "measurement_recorded"]
        consumed = [event for event in events if event["kind"] == "permit_consumed"]
        issued = [event for event in events if event["kind"] == "permit_issued"]
        return {
            "controller_version": CONTROLLER_VERSION,
            "lock": lock,
            "event_count": len(events),
            "permits_issued": len(issued),
            "permits_consumed": len(consumed),
            "measurements": len(measurements),
            "open_permits": len(issued) - len(consumed),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Trusted post-lock controller")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("verify-lock")
    activate = sub.add_parser("activate-lock")
    activate.add_argument("--capability", required=True)
    issue = sub.add_parser("issue-permit")
    issue.add_argument("--request", required=True)
    issue.add_argument("--capability", required=True)
    receipt = sub.add_parser("verify-receipt")
    receipt.add_argument("--receipt", required=True)
    receipt.add_argument("--action", required=True)
    receipt.add_argument("--subject-sha256", required=True)
    authorize = sub.add_parser("authorize")
    authorize.add_argument("--capability", required=True)
    authorize.add_argument("--action", required=True)
    authorize.add_argument("--target", required=True)
    authorize.add_argument("--subject-sha256", required=True)
    authorize.add_argument("--campaign", required=True)
    run = sub.add_parser("run")
    run.add_argument("--permit", required=True)
    run.add_argument("--shape", type=int, required=True)
    run.add_argument("--impl")
    run.add_argument("--timeout", type=int, default=1800)
    side = sub.add_parser("side")
    side.add_argument("--permit", required=True)
    side.add_argument("--timeout", type=int, default=21600)
    args = parser.parse_args()
    controller = TrustedController(ROOT)
    if args.command == "verify-lock":
        output = controller.verify_lock_only()
    elif args.command == "activate-lock":
        output = controller.activate_lock(Path(args.capability))
    elif args.command == "status":
        output = controller.status()
    elif args.command == "issue-permit":
        output = controller.issue_permit(Path(args.request), Path(args.capability))
    elif args.command == "verify-receipt":
        output = controller.verify_receipt(
            Path(args.receipt), args.action, args.subject_sha256
        )
    elif args.command == "authorize":
        output = controller.authorize(
            capability_path=Path(args.capability),
            action=args.action,
            target=args.target,
            subject_sha256=args.subject_sha256,
            campaign_id=args.campaign,
        )
    elif args.command == "run":
        output = controller.run_primary(
            permit_id=args.permit,
            shape_id=args.shape,
            impl_path=Path(args.impl) if args.impl else None,
            timeout_seconds=args.timeout,
        )
    elif args.command == "side":
        output = controller.run_side(
            permit_id=args.permit,
            timeout_seconds=args.timeout,
        )
    else:
        raise ControllerRefusal("unknown command")
    print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuthorityError, ControllerRefusal) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
