#!/usr/bin/env python3
"""Post-lock authority controller and sole candidate-evaluation entrypoint.

The controller never imports or executes candidate bytes.  It consumes a
one-use permit first, launches ``candidate_worker.py`` in the restricted
Bubblewrap namespace, independently recomputes all correctness references, and
is the only process allowed to append authoritative measurement state.

The diagnostic lane (``run_diagnostic``) has the same shape with a different
worker and a very different product.  It launches ``profile_worker.py`` in the
same namespace to profile bytes that already exist, and what it returns is a
*profile artifact*, never a performance result: a diagnostic authorizes no
candidate bytes, promotes nothing, spends no scientific attempt, and its
timings are instrument-distorted by construction.  What it does produce is the
counter-evidence ``run_gate.py`` demands before any optimization direction may
be opened, so the finalized artifact is hashed and that digest is bound into
both the measurement event and the measurement packet -- exactly the binding
``run_gate.py::_reconcile_authority_diagnostic`` re-checks before it will
record a profile record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import secrets
import shutil
import statistics
import sys
import tempfile
import time
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
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
    SandboxError,
    SandboxFiles,
    SandboxResult,
    run_isolated_command,
    run_sandbox,
)


CONTROLLER_VERSION = "2.0.0"
OFFICIAL = ROOT / "torch_transformer_benchmark.py"
SHAPES = PROJECT / "shapes.json"
WORKER = HERE / "candidate_worker.py"
PROFILE_WORKER = HERE / "profile_worker.py"
SHAPE6_EVALUATOR = TOOLS / "shape6_local_eval.py"
SHAPE14_EVALUATOR = TOOLS / "shape14_eval.py"
SUBMISSION = PROJECT / "submission" / "torch_transformer_benchmark_submission.py"
MANIFEST = PROJECT / "manifest.json"
TENSORFLOW_OFFICIAL = ROOT / "tensorflow_transformer_benchmark.py"
REQUESTS = PROJECT / "loop" / "requests"
RECEIPTS = PROJECT / "authority" / "receipts"
# The gate's trusted counter-evidence namespace.  Every byte of diagnostic
# evidence lands under here and nowhere else; the gate re-hashes all of it.
PROFILE_EVIDENCE = PROJECT / "loop" / "profile_evidence"
MECHANISM_CATALOG = PROJECT / "loop" / "mechanism_catalog.json"
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
# --- diagnostic (profiling) lane -------------------------------------------
# Profiling is not benchmarking.  nsys/ncu/torch.profiler serialize launches,
# replay kernels, and copy counters back to the host, so an instrumented run is
# routinely one to two orders of magnitude slower than the same work untouched,
# and its wall/event times describe the instrument rather than the kernel.  Two
# consequences are wired into the code below.  First, the lane needs a far
# longer default timeout than a benchmark run (3h against the 30min the
# candidate lane uses), because the same shape under a profiler can take hours.
# Second, nothing measured here may ever surface as a performance claim: the
# measurement's supporting_timing carries a neutral 1.0 speedup stamped
# suspicious=True, so any present or future consumer that mistakes it for a
# performance number fails closed instead of publishing an instrumented time.
# Speedups come from the runner/candidate lane only.
DIAGNOSTIC_TIMEOUT_S = 3 * 60 * 60
MAX_PROFILE_RAW_FILES = 256
MAX_PROFILE_RAW_BYTES = 2 * 1024**3
MAX_PROFILE_TREE_DEPTH = 8
# Exactly the field set run_gate.py::_reconcile_authority_diagnostic accepts.
# It compares with set equality, so one extra or one missing key makes the
# artifact permanently unreconcilable -- and an unreconciled request blocks
# every later request.  The controller therefore refuses such an artifact
# before it becomes a measurement, which downgrades a deadlock into an
# ordinary, settleable infrastructure failure.
PROFILE_ARTIFACT_KEYS = frozenset({
    "schema_version", "profile_record_id", "request_id", "campaign_id",
    "shape", "target_sha256", "tool", "tool_version", "created_epoch",
    "machine_state_sha256", "route", "metrics", "supported_bottlenecks",
    "raw_artifacts", "gate_request_sha256",
})
# Profiler toolchains that exist inside the jail but not on its PATH.  /usr is
# bound read-only in the namespace, so a CUDA install under /usr/local is
# already reachable by absolute path; the worker is told where to look rather
# than left to guess or to shell out through a PATH it does not control.
PROFILER_TOOL_DIRS = (
    "/usr/local/cuda/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
)
# Conservative filename component: no separators, no leading dot, no traversal.
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
CONTROLLER_MACHINE_STATE = "controller_machine_state.json"
RECEIPT_KEYS = {
    "authority_event_id",
    "authority_event_sha256",
    "action",
    "subject_sha256",
    "capability_nonce",
    "role",
}
WORKER_ENVIRONMENT_KEYS = frozenset({
    "python", "torch", "cuda", "gpu", "driver", "triton",
})


class ControllerRefusal(RuntimeError):
    pass


def _reject_constant(value: Any) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ControllerRefusal(f"invalid {label}: {exc}") from exc
    return _strict_object_bytes(raw, label)


def _strict_object_bytes(raw: bytes, label: str) -> dict[str, Any]:
    """Parse one JSON object, refusing NaN/Infinity.

    ``json.loads`` accepts those non-standard constants by default, and they
    survive into any structure the controller would otherwise hash and store,
    where ``canonical_json`` (allow_nan=False) then raises far from the cause.
    """
    try:
        value = json.loads(raw, parse_constant=_reject_constant)
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
        # A diagnostic is read-only evidence gathering.  It may not authorize
        # candidate bytes, may not promote, and may not be charged as a
        # scientific attempt; target_sha256 names the bytes being profiled, and
        # binds them, but confers no authority over them.
        if (
            mode != "diagnostic"
            or request.get("candidate_authorized") is not False
            or request.get("promotion_allowed") is not False
            or request.get("scientific_strike_eligible") is not False
        ):
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


def validate_worker_environment(environment: Any) -> dict[str, str]:
    if (
        not isinstance(environment, dict)
        or set(environment) != WORKER_ENVIRONMENT_KEYS
        or any(
            not isinstance(value, str)
            or not value.strip()
            or value.strip().lower() == "unknown"
            for value in environment.values()
        )
    ):
        raise ControllerRefusal("worker environment schema mismatch")
    return {key: environment[key].strip() for key in WORKER_ENVIRONMENT_KEYS}


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
    validate_worker_environment(response.get("environment"))
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


def _catalog_bottlenecks() -> dict[str, dict[str, Any]]:
    """Read the owner-frozen bottleneck taxonomy that the gate also reads.

    The controller checks the same tool/metric coverage the gate checks at
    reconcile time.  That duplication is deliberate: a measurement the gate
    later refuses leaves its request unreconciled, and an unreconciled request
    blocks every later request, so a failure has to happen here -- before the
    measurement exists -- where it becomes a durable, settleable run_failed.
    """
    document = _strict_object(MECHANISM_CATALOG, "mechanism catalog")
    bottlenecks = document.get("bottlenecks")
    if not isinstance(bottlenecks, dict) or not bottlenecks:
        raise ControllerRefusal("mechanism catalog has no bottleneck taxonomy")
    for name, entry in bottlenecks.items():
        tools = entry.get("evidence_tools") if isinstance(entry, dict) else None
        metrics = entry.get("required_metrics") if isinstance(entry, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(tools, list)
            or not tools
            or any(not isinstance(tool, str) or not tool for tool in tools)
            or not isinstance(metrics, list)
            or not metrics
            or any(not isinstance(metric, str) or not metric for metric in metrics)
        ):
            raise ControllerRefusal(f"malformed bottleneck catalog entry: {name!r}")
    return bottlenecks


def _machine_state_document() -> dict[str, Any]:
    """Capture the host state the profile artifact is bound to.

    The worker cannot be trusted for this and, inside the namespace, cannot
    even see most of it.  The controller captures it, stores it as a raw
    artifact under the trusted evidence namespace, and binds its digest into
    the artifact, so ``machine_state_sha256`` names a file the gate re-hashes
    instead of an unbacked hex string.  Load average is included because a
    diagnostic taken on a busy box is weaker evidence and should say so.
    """
    driver = None
    try:
        version_file = Path("/proc/driver/nvidia/version")
        if version_file.is_file():
            driver = version_file.read_text(encoding="utf-8", errors="replace").strip()[:512]
    except OSError:
        driver = None
    try:
        load_average = [float(value) for value in os.getloadavg()]
    except (OSError, AttributeError):
        load_average = None
    uname = platform.uname()
    return {
        "schema_version": 1,
        "captured_by": "trusted-controller",
        "controller_version": CONTROLLER_VERSION,
        "captured_utc": iso_utc(utc_now()),
        "system": uname.system,
        "kernel_release": uname.release,
        "machine": uname.machine,
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "load_average": load_average,
        "nvidia_driver_version": driver,
        "official_sha256": sha256_file(OFFICIAL),
        "shapes_sha256": sha256_file(SHAPES),
    }


def _diagnostic_request_fields(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one immutable diagnostic gate request and extract its bindings."""
    if request.get("request_kind") != "diagnostic" or request.get("mode") != "diagnostic":
        raise ControllerRefusal("permit did not originate from a diagnostic request")
    if (
        request.get("candidate_authorized") is not False
        or request.get("promotion_allowed") is not False
        or request.get("scientific_strike_eligible") is not False
        or request.get("impl_path") is not None
        or request.get("impl_sha256") is not None
        or request.get("ledger") is not None
    ):
        raise ControllerRefusal("diagnostic request privileges are malformed")
    record_id = request.get("profile_record_id")
    if not isinstance(record_id, str) or not SAFE_NAME.fullmatch(record_id):
        raise ControllerRefusal("diagnostic profile_record_id is malformed")
    declared = request.get("profile_output")
    if not isinstance(declared, str) or not declared:
        raise ControllerRefusal("diagnostic profile_output is missing")
    relative = PurePosixPath(declared)
    if relative.is_absolute() or ".." in relative.parts:
        raise ControllerRefusal("diagnostic profile_output must be a repo-relative path")
    artifact_path = (ROOT / Path(declared)).resolve()
    evidence_root = PROFILE_EVIDENCE.resolve()
    # Directly inside the trusted namespace, plainly named, and a JSON file:
    # the gate resolves this exact path and refuses anything that escapes.
    if (
        artifact_path.parent != evidence_root
        or not artifact_path.name.endswith(".json")
        or not SAFE_NAME.fullmatch(artifact_path.name)
    ):
        raise ControllerRefusal("diagnostic profile_output escapes its trusted namespace")
    tool = request.get("tool")
    route = request.get("route")
    question = request.get("question")
    bottlenecks = request.get("supported_bottlenecks")
    if (
        not isinstance(tool, str)
        or not tool
        or not isinstance(route, str)
        or not route.strip()
        or not isinstance(question, str)
        or not question.strip()
        or not isinstance(bottlenecks, list)
        or not bottlenecks
        or any(not isinstance(item, str) or not item for item in bottlenecks)
        or len(set(bottlenecks)) != len(bottlenecks)
    ):
        raise ControllerRefusal("diagnostic tool/route/question/bottlenecks are malformed")
    catalog = _catalog_bottlenecks()
    required_metrics: dict[str, list[str]] = {}
    for bottleneck in bottlenecks:
        entry = catalog.get(bottleneck)
        if entry is None:
            raise ControllerRefusal(f"diagnostic declares unknown bottleneck {bottleneck!r}")
        if tool not in entry["evidence_tools"]:
            raise ControllerRefusal(
                f"{tool!r} is not admissible evidence for {bottleneck!r}"
            )
        required_metrics[bottleneck] = list(entry["required_metrics"])
    return {
        "record_id": record_id,
        "artifact_path": artifact_path,
        "artifact_name": artifact_path.name,
        "raw_dir": evidence_root / f"{record_id}.raw",
        "tool": tool,
        "route": route.strip(),
        "question": question.strip(),
        "supported_bottlenecks": list(bottlenecks),
        "required_metrics": required_metrics,
    }


def _resolve_diagnostic_target(target_sha256: str, explicit: Path | None) -> Path:
    """Locate the already-existing bytes a diagnostic profiles.

    The permit binds a SHA-256, never a path, and confers no authority over
    those bytes: they are mounted read-only and are never promoted, recorded as
    a candidate measurement, or charged to a mechanism family.
    """
    roots = (PROJECT / "kernels", PROJECT / "submission")
    if explicit is not None:
        target = _regular_repo_file(explicit, "diagnostic target", roots)
        if sha256_file(target) != target_sha256:
            raise ControllerRefusal("diagnostic target bytes disagree with the permit")
        return target
    matches: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.is_symlink() or not path.is_file():
                continue
            if sha256_file(path) == target_sha256:
                matches.add(path.resolve())
    if len(matches) != 1:
        raise ControllerRefusal(
            "diagnostic target bytes are absent from or ambiguous within the "
            "approved roots; pass --target explicitly"
        )
    return matches.pop()


def _scan_output_tree(directory: Path, prefix: str, depth: int) -> list[tuple[str, Path]]:
    if depth > MAX_PROFILE_TREE_DEPTH:
        raise ControllerRefusal("profile worker output is nested too deeply")
    found: list[tuple[str, Path]] = []
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.is_symlink():
            raise ControllerRefusal("profile worker emitted a symlink")
        relative = f"{prefix}{entry.name}"
        if entry.is_dir():
            found.extend(_scan_output_tree(entry, f"{relative}/", depth + 1))
        elif entry.is_file():
            found.append((relative, entry))
        else:
            raise ControllerRefusal("profile worker emitted a non-regular file")
    return found


def _collect_worker_outputs(
    output_dir: Path, artifact_name: str
) -> tuple[Path, list[tuple[str, Path]]]:
    """Split the worker's output directory into artifact plus raw evidence.

    Interface assumption (``profile_worker.py`` is authored in parallel): the
    worker writes its profile artifact JSON into ``/output`` under the filename
    the request names, and every other file it leaves there is raw evidence.
    The fallback -- one single top-level ``.json`` -- keeps a worker that names
    the artifact differently working, and anything more ambiguous is refused
    rather than guessed.
    """
    entries = _scan_output_tree(output_dir, "", 0)
    if not entries:
        raise ControllerRefusal("profile worker produced no output")
    if len(entries) > MAX_PROFILE_RAW_FILES:
        raise ControllerRefusal(
            f"profile worker emitted more than {MAX_PROFILE_RAW_FILES} files"
        )
    total_bytes = sum(path.stat().st_size for _, path in entries)
    if total_bytes > MAX_PROFILE_RAW_BYTES:
        raise ControllerRefusal("profile worker output exceeds the raw evidence size cap")
    by_relative = dict(entries)
    if artifact_name in by_relative:
        chosen = artifact_name
    else:
        top_level_json = [
            name for name in by_relative
            if "/" not in name and name.endswith(".json")
        ]
        if len(top_level_json) != 1:
            raise ControllerRefusal(
                f"profile worker did not emit {artifact_name!r} and its output is "
                "ambiguous about which file is the profile artifact"
            )
        chosen = top_level_json[0]
    if CONTROLLER_MACHINE_STATE in by_relative:
        raise ControllerRefusal(
            f"profile worker may not emit the reserved name {CONTROLLER_MACHINE_STATE!r}"
        )
    raw = [(name, path) for name, path in entries if name != chosen]
    return by_relative[chosen], raw


def _ingest_raw_artifacts(
    raw_dir: Path,
    raw_files: list[tuple[str, Path]],
    machine_state_bytes: bytes,
) -> list[dict[str, str]]:
    """Move the worker's raw evidence into the gate's trusted namespace.

    The sandbox output directory is a temporary mount that disappears with the
    run, so anything the artifact cites has to be copied out before it can be
    cited.  Every copied file is re-hashed at its destination and written
    read-only, because the gate re-hashes each one at reconcile time and a
    mismatch there is unrecoverable.
    """
    raw_dir.mkdir(parents=True)
    artifacts: list[dict[str, str]] = []
    for relative, source in raw_files:
        destination = raw_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination, follow_symlinks=False)
        os.chmod(destination, 0o444)
        artifacts.append({
            "path": str(destination.relative_to(ROOT)),
            "sha256": sha256_file(destination),
        })
    state_path = raw_dir / CONTROLLER_MACHINE_STATE
    atomic_write(state_path, machine_state_bytes, mode=0o444)
    artifacts.append({
        "path": str(state_path.relative_to(ROOT)),
        "sha256": sha256_file(state_path),
    })
    return sorted(artifacts, key=lambda item: item["path"])


def _finalize_profile_artifact(
    emitted: Mapping[str, Any],
    *,
    bound: Mapping[str, Any],
    raw_artifacts: list[dict[str, str]],
    started_epoch: float,
) -> dict[str, Any]:
    """Turn the worker's emitted artifact into the artifact the gate accepts.

    Three classes of field, three rules:

    * Authority-owned (record id, request/campaign/gate bindings, shape,
      target, tool, route, bottlenecks, machine state).  The controller owns
      these because they come from the immutable gate request or from the host.
      A worker that leaves one out gets it filled in; a worker that contradicts
      one is refused -- authority is never negotiated with the worker.
    * Worker-owned (tool_version, metrics, created_epoch).  These are the
      actual evidence and the controller cannot verify them; it checks only
      that they are well-formed and plausible.  Whether the counters support
      the diagnosis is the auditor's job, not the controller's.
    * Filesystem-owned (raw_artifacts).  Derived from what is actually on disk
      after ingestion, never from the worker's claim, since the gate will
      re-hash exactly what is on disk.
    """
    if not isinstance(emitted, Mapping):
        raise ControllerRefusal("profile artifact must be a JSON object")
    unknown = sorted(set(emitted) - set(PROFILE_ARTIFACT_KEYS))
    if unknown:
        raise ControllerRefusal(f"profile artifact has unknown field(s): {unknown}")
    final: dict[str, Any] = {"schema_version": 1}
    if emitted.get("schema_version", 1) != 1:
        raise ControllerRefusal("profile artifact schema_version is unsupported")
    for key, value in bound.items():
        claimed = emitted.get(key)
        if claimed is not None and claimed != value:
            raise ControllerRefusal(
                f"profile artifact contradicts its authority binding: {key}"
            )
        final[key] = value
    tool_version = emitted.get("tool_version")
    if not isinstance(tool_version, str) or not tool_version.strip():
        raise ControllerRefusal("profile artifact must name its profiler version")
    final["tool_version"] = tool_version
    created = emitted.get("created_epoch")
    created = time.time() if created is None else _finite_number(created, "created_epoch")
    if not started_epoch - 3600.0 <= created <= time.time() + 3600.0:
        raise ControllerRefusal("profile created_epoch is outside this run's window")
    final["created_epoch"] = created
    metrics = emitted.get("metrics")
    if (
        not isinstance(metrics, dict)
        or not metrics
        or any(not isinstance(key, str) or not key for key in metrics)
    ):
        raise ControllerRefusal("profile metrics must be a non-empty object keyed by name")
    final["metrics"] = dict(metrics)
    claimed_raw = emitted.get("raw_artifacts")
    if claimed_raw is not None:
        if not isinstance(claimed_raw, list):
            raise ControllerRefusal("profile raw_artifacts must be a list")
        present = {PurePosixPath(item["path"]).name for item in raw_artifacts}
        for item in claimed_raw:
            name = item.get("path") if isinstance(item, dict) else None
            if not isinstance(name, str) or PurePosixPath(name).name not in present:
                raise ControllerRefusal(
                    "profile artifact cites raw evidence the worker did not produce"
                )
    final["raw_artifacts"] = raw_artifacts
    if set(final) != set(PROFILE_ARTIFACT_KEYS):
        raise ControllerRefusal("finalized profile artifact field set is wrong")
    return final


def _verify_profile_artifact(
    artifact: Mapping[str, Any],
    *,
    bound: Mapping[str, Any],
    required_metrics: Mapping[str, list[str]],
) -> None:
    """Re-run the gate's own acceptance test before the measurement exists.

    This mirrors run_gate.py::_reconcile_authority_diagnostic deliberately.  If
    the artifact would be refused there, the controller must refuse it here:
    there it wedges the request registry permanently, here it is one settleable
    infrastructure failure.
    """
    if set(artifact) != set(PROFILE_ARTIFACT_KEYS) or artifact.get("schema_version") != 1:
        raise ControllerRefusal("profile artifact has an unknown/missing field")
    if any(artifact.get(key) != value for key, value in bound.items()):
        raise ControllerRefusal("profile artifact disagrees with its authority bindings")
    created = artifact["created_epoch"]
    if (
        not isinstance(artifact["metrics"], dict)
        or not artifact["metrics"]
        or not isinstance(artifact["raw_artifacts"], list)
        or isinstance(created, bool)
        or not isinstance(created, (int, float))
        or not math.isfinite(float(created))
    ):
        raise ControllerRefusal("profile metrics/raw artifacts/created_epoch are malformed")
    _hex64(artifact["machine_state_sha256"], "profile machine state hash")
    evidence_root = PROFILE_EVIDENCE.resolve()
    for raw in artifact["raw_artifacts"]:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"path", "sha256"}
            or not isinstance(raw.get("path"), str)
        ):
            raise ControllerRefusal("profile raw artifact reference is malformed")
        _hex64(raw["sha256"], "profile raw artifact hash")
        raw_path = (ROOT / raw["path"]).resolve()
        if not raw_path.is_relative_to(evidence_root):
            raise ControllerRefusal("profile raw artifact escaped its namespace")
        if not raw_path.is_file() or sha256_file(raw_path) != raw["sha256"]:
            raise ControllerRefusal("profile raw artifact is missing or changed")
    for bottleneck, metrics in required_metrics.items():
        missing = [
            metric for metric in metrics
            if metric not in artifact["metrics"] or artifact["metrics"][metric] is None
        ]
        if missing:
            raise ControllerRefusal(
                f"profile evidence is insufficient for {bottleneck}: missing {missing}"
            )


def _profile_worker_request(
    *,
    fields: Mapping[str, Any],
    gate_request: Mapping[str, Any],
    gate_request_sha256: str,
    campaign_id: str,
    shape_id: int,
    target_sha256: str,
    target_destination: str,
    target_alias: str,
    machine_state_sha256: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Describe the diagnostic to the profiler worker.

    ``request_id`` is deliberately the *gate* request id: the artifact must
    echo that value, so a worker that copies the field through is correct by
    construction.  Every in-namespace path the worker needs is named here
    rather than assumed, and the artifact's exact field list is included so the
    worker never has to reverse-engineer the gate's schema.
    """
    return {
        "schema_version": 1,
        "operation": "diagnostic",
        "request_id": gate_request["request_id"],
        "gate_request_sha256": gate_request_sha256,
        "profile_record_id": fields["record_id"],
        "campaign_id": campaign_id,
        "shape_id": shape_id,
        "shape": _shape(shape_id),
        "target_sha256": target_sha256,
        "target_path": target_destination,
        "target_path_alias": target_alias,
        "official_path": "/work/official.py",
        "shapes_path": "/work/shapes.json",
        "official_sha256": sha256_file(OFFICIAL),
        "shapes_sha256": sha256_file(SHAPES),
        "output_dir": "/output",
        "artifact_filename": fields["artifact_name"],
        "artifact_fields": sorted(PROFILE_ARTIFACT_KEYS),
        "tool": fields["tool"],
        "route": fields["route"],
        "question": fields["question"],
        "supported_bottlenecks": fields["supported_bottlenecks"],
        "required_metrics": fields["required_metrics"],
        "machine_state_sha256": machine_state_sha256,
        "tool_search_paths": list(PROFILER_TOOL_DIRS),
        "timeout_seconds": timeout_seconds,
        # Loud, machine-readable statement of what this run is not.  Profiler
        # overhead makes every timing taken here an artefact of the instrument.
        "is_performance_measurement": False,
        "notes": (
            "Read-only diagnosis. Emit counters and mechanism evidence only; "
            "timings under instrumentation are not performance numbers."
        ),
    }


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
    # Diagnostics never reach the candidate worker: profiling has its own
    # worker, its own request schema, and its own lane (``run_diagnostic``).
    operation = "calibration" if mode == "calibration" else "candidate"
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
    def run_diagnostic(
        self,
        *,
        permit_id: str,
        target_path: Path | None = None,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Run the profiler worker and bind its artifact to controller authority.

        This lane exists because the competence gate will not open an
        optimization direction without counter-evidence, and counter-evidence
        is a profile record.  A profile record only exists once the gate can
        reconcile a diagnostic request, and it only reconciles one whose
        artifact digest appears in *both* the measurement event and the
        measurement packet.  That binding is what this method produces.

        Invariants, none of which the diagnostic lane may relax:

        * No candidate authority.  The permit carries may_modify_candidate
          False; the target is mounted read-only and is never treated as an
          authorized candidate.
        * No promotion.  promotion_eligible and performance_eligible are False
          and no audit is enqueued -- a profile can never become a champion.
        * No scientific attempt.  The gate charges nothing to the campaign or
          to a mechanism family for a diagnostic.
        * Exactly one permit, consumed transactionally before the worker is
          launched, and a durable terminal event either way: a crash, a
          timeout, or any validation failure after consumption becomes a
          run_failed (via ``terminalize_consumed_run``), which the gate settles
          as an infrastructure failure rather than deadlocking on it.
        * Nothing measured here is a performance number.
        """
        if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 24 * 3600:
            raise ControllerRefusal("diagnostic timeout must be 1s..24h")
        lock = self.require_lock()
        issued = [
            event for event in self.store.read_events()
            if event["kind"] == "permit_issued"
            and event["payload"].get("permit_id") == permit_id
        ]
        if len(issued) != 1:
            raise ControllerRefusal("diagnostic permit is absent or duplicated")
        permit = issued[0]["payload"]
        if permit.get("mode") != "diagnostic":
            raise ControllerRefusal("permit is not a diagnostic permit")
        if (permit.get("may_modify_candidate") is not False
                or permit.get("may_promote") is not False):
            raise ControllerRefusal(
                "diagnostic permit carries candidate or promotion privilege"
            )
        gate_request = self._bound_gate_request(permit)
        fields = _diagnostic_request_fields(gate_request)
        shape_id = permit["shape_id"]
        target_sha = _hex64(permit.get("candidate_sha256"), "diagnostic target hash")
        if gate_request.get("target_sha256") != target_sha:
            raise ControllerRefusal("permit target disagrees with its immutable gate request")
        if PROFILE_WORKER.is_symlink() or not PROFILE_WORKER.is_file():
            raise ControllerRefusal("profile worker is absent; the diagnostic lane cannot run")
        target = _resolve_diagnostic_target(target_sha, target_path)
        PROFILE_EVIDENCE.mkdir(parents=True, exist_ok=True)
        artifact_path = fields["artifact_path"]
        raw_dir = fields["raw_dir"]
        if artifact_path.exists() or raw_dir.exists():
            raise ControllerRefusal(
                "diagnostic evidence paths already exist; refusing to overwrite evidence"
            )
        machine_state_bytes = canonical_json(_machine_state_document()) + b"\n"
        machine_state_sha = sha256_bytes(machine_state_bytes)
        self.store.store_blob(machine_state_bytes, suffix=".json")
        # Keep the profiled bytes themselves: a profile record is only
        # interpretable against the source it was taken from.
        self.store.store_blob(target.read_bytes(), suffix=".py")
        target_destination = "/work/target.py"
        # Also exposed under the candidate-worker convention so a worker built
        # from candidate_worker.py finds the bytes at the path it expects.
        target_alias = "/work/candidate.py"
        worker_request = _profile_worker_request(
            fields=fields,
            gate_request=gate_request,
            gate_request_sha256=permit["request_sha256"],
            campaign_id=permit["campaign_id"],
            shape_id=shape_id,
            target_sha256=target_sha,
            target_destination=target_destination,
            target_alias=target_alias,
            machine_state_sha256=machine_state_sha,
            timeout_seconds=timeout_seconds,
        )
        request_bytes = canonical_json(worker_request) + b"\n"
        request_sha, _ = self.store.store_blob(request_bytes, suffix=".json")
        # Irrevocable before any worker process starts.
        consumed = self.store.consume_permit(
            permit_id=permit_id,
            mode="diagnostic",
            shape_id=shape_id,
            candidate_sha256=target_sha,
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
                "mode": "diagnostic",
                "shape_id": shape_id,
                "candidate_sha256": target_sha,
                "worker_request_sha256": request_sha,
                "profile_record_id": fields["record_id"],
                "tool": fields["tool"],
                "target_path": str(target.relative_to(ROOT)),
                "timeout_seconds": timeout_seconds,
                "lock_id": lock["lock_id"],
                "lock_manifest_sha256": lock["manifest_sha256"],
            },
        )
        started_epoch = time.time()
        with tempfile.TemporaryDirectory(prefix="trusted-diagnostic-") as temp:
            temp_root = Path(temp)
            request_file = temp_root / "request.json"
            request_file.write_bytes(request_bytes)
            output_dir = temp_root / "output"
            output_dir.mkdir()
            before = time.perf_counter_ns()
            sandbox_result = run_isolated_command(
                mounts=[
                    IsolatedMount(PROFILE_WORKER, "/work/profile_worker.py"),
                    IsolatedMount(request_file, "/work/request.json"),
                    IsolatedMount(target, target_destination),
                    IsolatedMount(target, target_alias),
                    IsolatedMount(OFFICIAL, "/work/official.py"),
                    IsolatedMount(SHAPES, "/work/shapes.json"),
                    IsolatedMount(output_dir, "/output", writable=True),
                ],
                argv=[
                    "/usr/bin/python3",
                    "/work/profile_worker.py",
                    "--request",
                    "/work/request.json",
                    "--output",
                    "/output",
                ],
                cwd="/work",
                # Profilers are slow by design; this is not a benchmark budget.
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
                        "reason": "diagnostic_timeout" if sandbox_result.timed_out
                                  else "profile_worker_exit",
                        "returncode": sandbox_result.returncode,
                        "profile_record_id": fields["record_id"],
                        "stdout_sha256": stdout_sha,
                        "stderr_sha256": stderr_sha,
                        "elapsed_ms": elapsed_ms,
                    },
                )
                raise ControllerRefusal(
                    f"profile worker failed; durable event {failed['event_id']}"
                )
            emitted_path, raw_files = _collect_worker_outputs(
                output_dir, fields["artifact_name"]
            )
            emitted_bytes = emitted_path.read_bytes()
            response_sha, _ = self.store.store_blob(emitted_bytes, suffix=".json")
            emitted = _strict_object_bytes(emitted_bytes, "profile artifact")
            # The sandbox output mount dies with this block, so the evidence is
            # copied into the gate's namespace before anything can cite it.
            raw_artifacts = _ingest_raw_artifacts(raw_dir, raw_files, machine_state_bytes)
        bound = {
            "profile_record_id": fields["record_id"],
            "request_id": gate_request["request_id"],
            "campaign_id": permit["campaign_id"],
            "shape": shape_id,
            "target_sha256": target_sha,
            "tool": fields["tool"],
            "route": fields["route"],
            "supported_bottlenecks": fields["supported_bottlenecks"],
            "gate_request_sha256": permit["request_sha256"],
            "machine_state_sha256": machine_state_sha,
        }
        artifact = _finalize_profile_artifact(
            emitted,
            bound=bound,
            raw_artifacts=raw_artifacts,
            started_epoch=started_epoch,
        )
        _verify_profile_artifact(
            artifact, bound=bound, required_metrics=fields["required_metrics"]
        )
        artifact_bytes = canonical_json(artifact) + b"\n"
        atomic_write(artifact_path, artifact_bytes, mode=0o444)
        digest = sha256_file(artifact_path)
        if digest != sha256_bytes(artifact_bytes):
            raise ControllerRefusal("profile artifact changed while being written")
        self.store.store_blob(artifact_bytes, suffix=".json")
        # A diagnostic asserts nothing about numerical correctness: it never
        # runs the official challenge.  "passed" states only that the artifact
        # and its raw evidence validated.
        correctness = {
            "passed": True,
            "authority": "trusted-controller",
            "scope": "diagnostic_artifact_integrity",
            "numerical_correctness_checked": False,
            "trials": [],
        }
        # A profiler distorts every time it observes.  The gate's schema
        # requires a finite speedup here, so this carries the neutral 1.0 and
        # is stamped suspicious, which makes every performance-eligibility test
        # in this file and downstream fail closed on it.
        timing = {
            "event_speedup": 1.0,
            "suspicious": True,
            "authority": "none",
            "measurement_class": "profiler-instrumented",
            "note": (
                "Profiler overhead makes these timings meaningless as "
                "performance; 1.0 is a neutral placeholder, never a result."
            ),
        }
        measurement = self.store.append(
            kind="measurement_recorded",
            actor="trusted-controller",
            payload={
                "run_id": run_id,
                "started_event_id": started["event_id"],
                "permit_id": permit_id,
                "campaign_id": permit["campaign_id"],
                "mode": "diagnostic",
                # The gate derives the lane from the mode; "primary" is the
                # namespace label only, and carries no promotion authority.
                "lane": "primary",
                "shape_id": shape_id,
                "candidate_sha256": target_sha,
                "family_id": permit["family_id"],
                "worker_request_sha256": request_sha,
                "worker_response_sha256": response_sha,
                "stdout_sha256": stdout_sha,
                "stderr_sha256": stderr_sha,
                "controller_correctness": correctness,
                "supporting_timing": timing,
                "diagnostic_profile_sha256": digest,
                "profile_record_id": fields["record_id"],
                "profile_artifact_path": str(artifact_path.relative_to(ROOT)),
                "profile_raw_artifacts": raw_artifacts,
                "machine_state_sha256": machine_state_sha,
                "tool": fields["tool"],
                "tool_version": artifact["tool_version"],
                "route": fields["route"],
                "supported_bottlenecks": fields["supported_bottlenecks"],
                "target_path": str(target.relative_to(ROOT)),
                "controller_process_elapsed_ms": elapsed_ms,
                "promotion_threshold": None,
                "calibrated_noise": None,
                "calibration_event_id": None,
                "performance_eligible": False,
                "promotion_eligible": False,
                "promotion_blocker": "diagnostic_evidence_never_promotes",
                "scientific_attempt": False,
                "lock_id": lock["lock_id"],
                "lock_manifest_sha256": lock["manifest_sha256"],
            },
        )
        # Exactly the gate's packet schema plus the one optional diagnostic
        # field; any extra or missing key is refused at reconcile.
        packet = {
            "schema_version": 1,
            "entry_id": run_id,
            "lane": "primary",
            "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "candidate_sha256": target_sha,
            "permit_id": permit_id,
            "gate_request_sha256": permit["request_sha256"],
            "campaign_id": permit["campaign_id"],
            "mode": "diagnostic",
            "shape_id": shape_id,
            "family_id": permit["family_id"],
            "worker_request_sha256": request_sha,
            "worker_response_sha256": response_sha,
            "controller_correctness": correctness,
            "supporting_timing": timing,
            "diagnostic_profile_sha256": digest,
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
                "candidate_sha256": target_sha,
                "packet_sha256": packet_sha,
                "lane": "primary",
            },
        )
        # No audit is enqueued: an audit decides champion eligibility, and a
        # diagnostic can never produce one.  The gate's reconcile turns this
        # into a profile record, and the auditor judges the diagnosis later
        # against the card that cites it.
        return {
            "entry_id": run_id,
            "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "packet_binding_event_id": binding["event_id"],
            "packet_sha256": packet_sha,
            "profile_record_id": fields["record_id"],
            "profile_artifact_path": str(artifact_path.relative_to(ROOT)),
            "diagnostic_profile_sha256": digest,
            "raw_artifact_count": len(raw_artifacts),
            "tool": fields["tool"],
            "supported_bottlenecks": fields["supported_bottlenecks"],
            "is_performance_measurement": False,
            "performance_eligible": False,
            "promotion_eligible": False,
            "scientific_attempt": False,
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
        if mode == "diagnostic" or gate_request.get("request_kind") == "diagnostic":
            # A diagnostic profiles existing bytes; it must never be executed
            # as a candidate measurement, which is what this lane produces.
            raise ControllerRefusal("diagnostic request must use the diagnostic command")
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
    diagnostic = sub.add_parser("diagnostic")
    diagnostic.add_argument("--permit", required=True)
    diagnostic.add_argument(
        "--target",
        help="path to the bytes being profiled; resolved from the permit's "
             "target SHA-256 when omitted",
    )
    # Profilers run far longer than benchmarks; this default is three hours.
    diagnostic.add_argument("--timeout", type=int, default=DIAGNOSTIC_TIMEOUT_S)
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
    elif args.command == "diagnostic":
        output = controller.run_diagnostic(
            permit_id=args.permit,
            target_path=Path(args.target) if args.target else None,
            timeout_seconds=args.timeout,
        )
    else:
        raise ControllerRefusal("unknown command")
    print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuthorityError, ControllerRefusal, SandboxError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
