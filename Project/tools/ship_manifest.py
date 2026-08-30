#!/usr/bin/env python3
"""Build a fail-closed manifest from an explicit final evidence map.

Nothing is selected by speed, recency, a Markdown leaderboard, or filesystem
discovery. The owner supplies one exact evidence selector for every official
shape. Each selection must bind to the exact generated submission, pinned
official bytes, immutable evidence bytes, the measured box, and an eligible
decision from the shared audit authority.

Field contract, deliberately split so that no field can imply another:

``audit_verdict``     the integrity verdict of a real, bound audit result
                      event, or ``null``. Never prose, never synthesized.
``evidence_status``   whether this row may ship.
``evidence_class``    ``post-lock-bound`` or ``legacy-pre-lock``.
``reference_method``  plain description of what the result was checked against.

Regenerate at freeze, AFTER the final commit. The working tree and every
selected input must already be committed, so the recorded ``git_revision``
always contains the submission bytes it names. The output is
``Project/results_side/SHIP_MANIFEST.json`` unless ``--output`` is supplied.

``--diagnose`` is a read-only report that explains, per official shape, what
binding is missing today. It writes nothing and relaxes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "Project"
TOOLS = PROJECT / "tools"
HARNESS = PROJECT / "harness"
JOURNAL = PROJECT / "results" / "JOURNAL.jsonl"
SIDE = PROJECT / "results_side"
SUBMISSION = PROJECT / "submission" / "torch_transformer_benchmark_submission.py"
OFFICIAL = ROOT / "torch_transformer_benchmark.py"
OFFICIAL_MANIFEST = PROJECT / "manifest.json"
SHAPES_FILE = PROJECT / "shapes.json"
MAP_SCHEMA_FILE = TOOLS / "final_evidence_map.schema.json"
DEFAULT_OUTPUT = SIDE / "SHIP_MANIFEST.json"
MAP_SCHEMA = "final-evidence-map-v1"
SUBMISSION_REL = SUBMISSION.relative_to(ROOT).as_posix()
MIB = 2**20
MANIFEST_SCHEMA = "ship-manifest-v3"
DIAGNOSIS_SCHEMA = "ship-manifest-diagnosis-v1"
AUDIT_EVENTS_REL = "Project/audits/audit_events.jsonl"
LEGACY_VERDICTS_REL = "Project/audits/verdicts.jsonl"
LEGACY_PACKETS_REL = "Project/audits/packets"
# Two, and only two, evidence classes may appear in any output of this module.
POST_LOCK = "post-lock-bound"
LEGACY_PRE_LOCK = "legacy-pre-lock"
UNESTABLISHED = "not-established"
# A shippable measurement names the box it ran on inside the evidence itself.
REQUIRED_ENV_KEYS = ("gpu", "driver", "torch", "cuda", "triton")
# Exactly the selector fields this module consumes.  The map schema is checked
# against these at test time so the two can never drift apart.
CONTROLLER_SELECTOR_FIELDS = frozenset({
    "kind", "entry_id", "measurement_event_sha256", "audit_packet",
    "selection_rationale",
})
JOURNAL_SELECTOR_FIELDS = frozenset({"kind", "entry_id", "selection_rationale"})
SIDE_SELECTOR_FIELDS = frozenset({
    "kind", "entry_id", "measurement_event_sha256", "side_evidence_sha256",
    "audit_packet", "selection_rationale",
})
SELECTOR_FIELDS = {
    "controller": CONTROLLER_SELECTOR_FIELDS,
    "journal": JOURNAL_SELECTOR_FIELDS,
    "side_controller": SIDE_SELECTOR_FIELDS,
}
SHAPE6_MEMORY_LIMITS = {
    "allocated_slope_bytes_per_repeat": 16 * MIB,
    "reserved_slope_bytes_per_repeat": 64 * MIB,
    "allocated_end_growth_bytes": 64 * MIB,
    "reserved_end_growth_bytes": 256 * MIB,
    "allocated_max_growth_bytes": 64 * MIB,
    "reserved_max_growth_bytes": 256 * MIB,
}
# What Project/harness/trusted_controller.py actually stamps into every primary
# worker request and measurement.  These are pinned here so a run taken with
# different tolerances, a different matmul precision, a different timing
# protocol or a different official shape table can never be selected -- the
# side lane has enforced official numerics from the start, the primary lane
# enforced none.
CONTROLLER_TIMING = {"warmup": 20, "repeats": 100, "rounds": 3}
CONTROLLER_NUMERICAL = {
    "padding_ratio": 0.0,
    "input_scale": 1.0,
    "rtol": 0.02,
    "atol": 0.002,
    "matmul_precision": "high",
    "allow_tf32": True,
}
# Reported back by Project/harness/candidate_worker.py from the live process.
CONTROLLER_EFFECTIVE_NUMERICAL_STATE = {
    "float32_matmul_precision": "high",
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None,
    "NVIDIA_TF32_OVERRIDE": None,
}
# The five official generator seeds; the controller appends secret seeds.
CONTROLLER_OFFICIAL_SEEDS = [1234, 1235, 1236, 1237, 1238]
OFFICIAL_NUMERICAL_STATE = {
    "dtype": "float32",
    "matmul_precision": "high",
    "cuda_matmul_allow_tf32": True,
    "cudnn_allow_tf32": True,
    "padding_ratio": 0.0,
    "input_scale": 1.0,
    "atol": 0.002,
    "rtol": 0.02,
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None,
    "NVIDIA_TF32_OVERRIDE": None,
}

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))
from audit_authority import (  # noqa: E402
    AuditAuthorityError,
    audit_decision,
    load_bound_packet,
    require_audit_enqueue,
)
from authority import AuthorityError, AuthorityStore  # noqa: E402


class ManifestError(RuntimeError):
    """The requested final evidence set is missing or internally inconsistent."""


class ManifestRefusal(ManifestError):
    """One refusal carrying an honest, per-shape reason for every shape."""

    def __init__(self, shapes: dict[str, dict[str, Any]]):
        self.shapes = shapes
        lines = [
            f"  shape {row['shape_id']} [{row['evidence_class']}] "
            f"{row['selector_kind']}: {row['reason']}"
            for row in sorted(shapes.values(), key=lambda item: item["shape_id"])
        ]
        super().__init__(
            f"{len(shapes)} of the official shapes have no shippable evidence:\n"
            + "\n".join(lines)
        )


def authority_blobs_dir() -> Path:
    """Derived at call time so tests can redirect the repository root."""
    return ROOT / "Project" / "authority" / "blobs"


def legacy_packets_dir() -> Path:
    return ROOT / LEGACY_PACKETS_REL


def audit_events_path() -> Path:
    return ROOT / AUDIT_EVENTS_REL


def legacy_verdicts_path() -> Path:
    return ROOT / LEGACY_VERDICTS_REL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"JSON root must be an object: {path}")
    return value


def resolve_repo_file(path_value: str, *, required_parent: Path | None = None) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ManifestError("evidence path must be a non-empty repository path")
    unresolved = ROOT / path_value
    if unresolved.is_symlink():
        raise ManifestError(f"evidence must not be a symlink: {path_value}")
    candidate = unresolved.resolve(strict=True)
    root = ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"evidence path escapes repository: {path_value}") from exc
    if required_parent is not None:
        try:
            candidate.relative_to(required_parent.resolve())
        except ValueError as exc:
            raise ManifestError(
                f"evidence path is outside {required_parent}: {path_value}"
            ) from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise ManifestError(f"evidence must be a regular non-symlink file: {path_value}")
    return candidate


def git_output(*arguments: str, binary: bool = False):
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise ManifestError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout if binary else result.stdout.decode().strip()


def require_committed_file(head: str, path: Path) -> str:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    committed = git_output("show", f"{head}:{relative}", binary=True)
    committed_sha = hashlib.sha256(committed).hexdigest()
    disk_sha = sha256_file(path)
    if committed_sha != disk_sha:
        raise ManifestError(
            f"{relative} differs from HEAD {head}: {disk_sha} != {committed_sha}"
        )
    return disk_sha


def freeze_provenance(evidence_map_path: Path) -> tuple[str, str]:
    status = git_output("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        changed = ", ".join(line[3:] for line in status.splitlines()[:8])
        suffix = " ..." if len(status.splitlines()) > 8 else ""
        raise ManifestError(
            "refusing to generate ship manifest: working tree has "
            f"uncommitted changes ({changed}{suffix})"
        )
    head = git_output("rev-parse", "HEAD")
    submission_sha = require_committed_file(head, SUBMISSION)
    require_committed_file(head, evidence_map_path)
    return head, submission_sha


def official_shape_map() -> dict[int, dict[str, Any]]:
    payload = read_json(SHAPES_FILE)
    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        raise ManifestError("Project/shapes.json has no shapes array")
    result = {}
    for shape in shapes:
        if not isinstance(shape, dict) or not isinstance(shape.get("id"), int):
            raise ManifestError("malformed official shape row")
        if shape["id"] in result:
            raise ManifestError(f"duplicate official shape id {shape['id']}")
        result[shape["id"]] = shape
    return result


def read_journal() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for lineno, raw in enumerate(JOURNAL.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise ManifestError(f"{JOURNAL}:{lineno}: blank row")
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{JOURNAL}:{lineno}: malformed JSON") from exc
        if not isinstance(row, dict) or not isinstance(row.get("entry_id"), str):
            raise ManifestError(f"{JOURNAL}:{lineno}: malformed journal row")
        entry_id = row["entry_id"]
        if entry_id in rows:
            raise ManifestError(f"duplicate journal entry_id: {entry_id}")
        rows[entry_id] = row
    return rows


def exact_shape(actual: Any, expected: dict[str, Any], label: str) -> None:
    if not isinstance(actual, dict):
        raise ManifestError(f"{label} has no shape object")
    keys = (
        "id", "batch_size", "seq_len", "d_model", "num_heads",
        "ffn_dim", "num_layers", "causal",
    )
    mismatches = {
        key: (actual.get(key), expected.get(key))
        for key in keys if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise ManifestError(f"{label} does not match official shape: {mismatches}")


def require_positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ManifestError(f"{label} must be finite and positive")
    return number


def bound_audit_decision(entry_id: str, candidate_sha: str | None = None,
                         packet_sha: str | None = None):
    """Ask the one shared authority, always against this repository's ledgers."""
    return audit_decision(
        entry_id,
        candidate_sha256=candidate_sha,
        packet_sha256=packet_sha,
        events_path=audit_events_path(),
        legacy_path=legacy_verdicts_path(),
        artifact_root=ROOT,
    )


def audit_verdict_fields(decision: dict[str, Any]) -> dict[str, Any]:
    """Report a verdict only when a real, bound audit result event exists.

    ``audit_verdict`` is null in every other case.  Nothing in this module may
    place descriptive prose in this field: how a result was referenced belongs
    in ``reference_method`` and whether it may ship belongs in
    ``evidence_status``.  The historical defect this replaces synthesized an
    audit-shaped sentence about oracle validation and pushed it into
    ``audit_verdict`` for shapes 6 and 14, which reads as an independent
    approval that never existed -- and for shape 6 that sentence also named a
    reference computation the evaluator never performed.  The exact fabricated
    strings are pinned as forbidden literals by
    ``NoSynthesizedVerdictTests.test_historical_fabricated_sentences_are_gone``,
    so they must not be reproduced anywhere in this file, prose included.
    """
    event_sha = decision.get("effective_event_sha256")
    if not event_sha:
        return {
            "audit_verdict": None,
            "audit_technical_verdict": None,
            "audit_verdict_event_sha256": None,
            "audit_verdict_source": None,
        }
    return {
        "audit_verdict": decision["integrity_status"],
        "audit_technical_verdict": decision["technical_status"],
        "audit_verdict_event_sha256": event_sha,
        "audit_verdict_source": AUDIT_EVENTS_REL,
    }


def eligible_audit(entry_id: str, candidate_sha: str,
                   packet_sha: str) -> dict[str, Any]:
    """Task-06 filter, fail-closed.

    RULE_VIOLATION, RETEST, NEEDS_CONTEXT, a blocking technical verdict, a
    legacy unbound row and a wholly absent verdict all land in
    ``blocking_reasons`` and all refuse here.  There is no path that ships a
    row without a bound, independently recorded audit result event.
    """
    try:
        decision = bound_audit_decision(entry_id, candidate_sha, packet_sha)
    except AuditAuthorityError as exc:
        raise ManifestError(f"audit authority rejected {entry_id}: {exc}") from exc
    if not decision.promotion_eligible:
        reasons = ", ".join(decision.blocking_reasons) or "not eligible"
        raise ManifestError(f"audit blocks {entry_id}: {reasons}")
    payload = decision.as_dict()
    if not payload.get("effective_event_sha256"):
        raise ManifestError(
            f"audit for {entry_id} claims eligibility with no bound result event"
        )
    return payload


def require_env_binding(env: Any, label: str) -> dict[str, str]:
    """Bind the measured box: device, driver, torch, cuda and triton."""
    if not isinstance(env, dict):
        raise ManifestError(f"{label} carries no environment fingerprint")
    missing = [
        key for key in REQUIRED_ENV_KEYS
        if not isinstance(env.get(key), str)
        or not env[key].strip()
        or env[key].strip().lower() == "unknown"
    ]
    if missing:
        raise ManifestError(
            f"{label} environment fingerprint does not bind {missing}; shippable "
            "evidence must name device, driver, torch, cuda and triton versions"
        )
    return {key: env[key] for key in REQUIRED_ENV_KEYS}


def prospective_audit_packet(reference: Any, entry_id: str,
                             candidate_sha: str):
    if not isinstance(reference, dict):
        raise ManifestError("normalized audit_packet selector is required")
    packet_sha = reference.get("sha256")
    expected_path = f"Project/authority/blobs/{packet_sha}.json"
    if reference.get("path") != expected_path:
        raise ManifestError(f"audit packet path must be {expected_path}")
    try:
        bound = load_bound_packet(
            entry_id,
            packet_sha256=packet_sha,
            authority_blobs=authority_blobs_dir(),
        )
    except AuditAuthorityError as exc:
        raise ManifestError(f"normalized audit packet is invalid: {exc}") from exc
    if bound.path.relative_to(ROOT).as_posix() != expected_path:
        raise ManifestError("audit packet resolved to wrong path")
    if bound.sha256 != packet_sha:
        raise ManifestError("audit packet selector SHA mismatch")
    if bound.candidate_sha256 != candidate_sha:
        raise ManifestError("audit packet contains mixed candidate bytes")
    return bound


def journal_submission_sha(row: dict[str, Any]) -> str | None:
    explicit = row.get("submission_sha256")
    if isinstance(explicit, str):
        return explicit
    impl = row.get("impl")
    if isinstance(impl, dict) and impl.get("path") == SUBMISSION_REL:
        return impl.get("sha256")
    return None


def journal_evidence_reasons(selector: dict[str, Any], shape_id: int,
                             expected_shape: dict[str, Any],
                             submission_sha: str,
                             rows: dict[str, dict[str, Any]]) -> list[str]:
    """Enumerate every reason a frozen-runner journal row cannot be shipped.

    ``Project/results/JOURNAL.jsonl`` is written by the pre-lock runner, which
    executes candidate source inside its own process and consumes no permit.
    Rows there are evidence history, never ship evidence, so this returns a
    reason list and never an entry.  It still walks the whole binding chain so
    the refusal names the exact missing binding rather than a category.
    """
    reasons = [
        "legacy_pre_lock_frozen_runner_journal: measured by Project/harness/"
        "runner.py before the trust boundary existed (candidate source executed "
        "in the runner process, no permit consumed, no measurement event)"
    ]
    if set(selector) - JOURNAL_SELECTOR_FIELDS:
        reasons.append("unknown_journal_selector_fields")
    entry_id = selector.get("entry_id")
    row = rows.get(entry_id)
    if row is None:
        reasons.append(f"journal_entry_absent:{entry_id}")
        return reasons
    if row.get("type") != "candidate" or row.get("shape_id") != shape_id:
        reasons.append("selected_row_is_not_this_shapes_candidate_result")
    else:
        try:
            exact_shape(row.get("shape"), expected_shape, f"journal entry {entry_id}")
        except ManifestError as exc:
            reasons.append(f"shape_mismatch:{exc}")
    correctness = row.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("passed") is not True:
        reasons.append("journal_correctness_did_not_pass")
    measured_submission_sha = journal_submission_sha(row)
    if measured_submission_sha != submission_sha:
        reasons.append(
            "not_bound_to_final_submission: measured "
            f"{measured_submission_sha!r}, manifest names {submission_sha}"
        )
    try:
        require_env_binding(row.get("env"), f"journal entry {entry_id}")
    except ManifestError as exc:
        reasons.append(f"environment_binding:{exc}")
    packet_sha = None
    try:
        bound_packet = load_bound_packet(
            entry_id, packets_dir=legacy_packets_dir(),
            authority_blobs=authority_blobs_dir(),
        )
    except AuditAuthorityError as exc:
        reasons.append(f"audit_packet_invalid:{exc}")
    else:
        packet_sha = bound_packet.sha256
        if bound_packet.candidate_sha256 != submission_sha:
            reasons.append("audit_packet_holds_different_candidate_bytes")
        if bound_packet.measurement_event_sha256 is None:
            reasons.append("audit_packet_binds_no_measurement_event")
    try:
        decision = bound_audit_decision(entry_id, measured_submission_sha, packet_sha)
    except AuditAuthorityError as exc:
        reasons.append(f"audit_authority_rejected:{exc}")
    else:
        if not decision.promotion_eligible:
            reasons.extend(decision.blocking_reasons)
    return reasons


def authority_blob(digest: str, suffix: str) -> Path:
    path = resolve_repo_file(
        f"Project/authority/blobs/{digest}{suffix}",
        required_parent=PROJECT / "authority" / "blobs",
    )
    if sha256_file(path) != digest:
        raise ManifestError(f"authority blob hash mismatch: {path}")
    return path


def validate_timing_samples(timing: Any, expected_count: int,
                            label: str) -> float:
    if not isinstance(timing, dict):
        raise ManifestError(f"{label} timing is not an object")
    samples = timing.get("raw_samples_ms")
    if not isinstance(samples, list) or len(samples) != expected_count:
        raise ManifestError(
            f"{label} timing must retain exactly {expected_count} raw samples"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in samples
    ):
        raise ManifestError(f"{label} timing contains invalid raw samples")
    actual_median = statistics.median(float(value) for value in samples)
    recorded_median = require_positive_number(
        timing.get("median_ms"), f"{label} recorded median"
    )
    if not math.isclose(actual_median, recorded_median, rel_tol=1e-12, abs_tol=1e-12):
        raise ManifestError(f"{label} recorded median does not match raw samples")
    if timing.get("n_samples") != expected_count:
        raise ManifestError(f"{label} n_samples does not match retained samples")
    return actual_median


def build_controller_entry(selector: dict[str, Any], shape_id: int,
                           expected_shape: dict[str, Any],
                           submission_sha: str) -> dict[str, Any]:
    if set(selector) - CONTROLLER_SELECTOR_FIELDS:
        raise ManifestError(f"shape {shape_id}: unknown controller selector fields")
    entry_id = selector.get("entry_id")
    measurement_sha = selector.get("measurement_event_sha256")
    try:
        events = AuthorityStore(ROOT).read_events()
    except AuthorityError as exc:
        raise ManifestError(f"trusted-controller authority is invalid: {exc}") from exc
    entry_measurements = [
        event for event in events
        if event.get("kind") == "measurement_recorded"
        and event.get("payload", {}).get("entry_id") == entry_id
    ]
    measurements = [
        event for event in events
        if event.get("kind") == "measurement_recorded"
        and event.get("event_sha256") == measurement_sha
    ]
    if len(measurements) != 1 or entry_measurements != measurements:
        raise ManifestError(
            f"shape {shape_id}: exact controller measurement is absent or duplicated"
        )
    measurement = measurements[0]
    payload = measurement.get("payload")
    if not isinstance(payload, dict):
        raise ManifestError(f"shape {shape_id}: controller measurement payload is malformed")
    if (
        payload.get("candidate_sha256") != submission_sha
        or payload.get("shape_id") != shape_id
        or payload.get("lane") != "primary"
        or payload.get("mode") not in {"optimization", "confirmation"}
    ):
        raise ManifestError(
            f"shape {shape_id}: controller measurement has mixed candidate/shape/lane/mode"
        )
    correctness = payload.get("controller_correctness")
    if not isinstance(correctness, dict) or correctness.get("passed") is not True:
        raise ManifestError(f"shape {shape_id}: trusted-controller correctness did not pass")
    timing_args = payload.get("timing_args")
    if (
        timing_args != CONTROLLER_TIMING
        or payload.get("numerical") != CONTROLLER_NUMERICAL
        or payload.get("effective_numerical_state")
        != CONTROLLER_EFFECTIVE_NUMERICAL_STATE
    ):
        raise ManifestError(
            f"shape {shape_id}: timing protocol or numerical state is not official"
        )
    expected_count = timing_args["repeats"] * timing_args["rounds"]
    supporting = payload.get("supporting_timing")
    if not isinstance(supporting, dict) or supporting.get("suspicious") is not False:
        raise ManifestError(f"shape {shape_id}: supporting timing is missing or suspicious")
    baseline_median = validate_timing_samples(
        supporting.get("baseline"), expected_count, f"shape {shape_id} baseline"
    )
    candidate_median = validate_timing_samples(
        supporting.get("candidate"), expected_count, f"shape {shape_id} candidate"
    )
    speedup = require_positive_number(
        supporting.get("event_speedup"), f"shape {shape_id} speedup"
    )
    if not math.isclose(
        baseline_median / candidate_median, speedup,
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise ManifestError(f"shape {shape_id}: speedup does not match raw medians")
    request_sha = payload.get("worker_request_sha256")
    request_path = authority_blob(request_sha, ".json")
    request = read_json(request_path)
    exact_shape(request.get("shape"), expected_shape, f"shape {shape_id} worker request")
    if (
        request.get("candidate_sha256") != submission_sha
        or request.get("official_sha256") != sha256_file(OFFICIAL)
        or request.get("shape_id") != shape_id
    ):
        raise ManifestError(f"shape {shape_id}: worker request binding is inconsistent")
    # The request is the instruction the isolated worker actually executed, so
    # the official shape table, numerics and timing protocol are bound here as
    # well as in the measurement.  ``operation`` matters on its own: a
    # calibration run also lands in lane "primary".
    seeds = request.get("seeds")
    official_seed_count = len(CONTROLLER_OFFICIAL_SEEDS)
    if (
        request.get("operation") != "candidate"
        or request.get("dtype") != "float32"
        or request.get("shapes_sha256") != sha256_file(SHAPES_FILE)
        or request.get("timing_args") != CONTROLLER_TIMING
        or request.get("numerical") != CONTROLLER_NUMERICAL
        or not isinstance(seeds, list)
        or seeds[:official_seed_count] != CONTROLLER_OFFICIAL_SEEDS
        or len(seeds) <= official_seed_count
        or len(set(seeds)) != len(seeds)
    ):
        raise ManifestError(
            f"shape {shape_id}: worker request is not an official candidate "
            "measurement over the pinned shape table, seeds and numerics"
        )
    bound_packet = prospective_audit_packet(
        selector.get("audit_packet"), entry_id, submission_sha
    )
    if bound_packet.measurement_event_sha256 != measurement_sha:
        raise ManifestError(f"shape {shape_id}: audit packet binds another measurement")
    if bound_packet.lane != "primary":
        raise ManifestError(f"shape {shape_id}: audit packet lane is not primary")
    entry_bindings = [
        event for event in events
        if event.get("kind") == "measurement_packet_bound"
        and event.get("payload", {}).get("entry_id") == entry_id
    ]
    binding_events = [
        event for event in events
        if event.get("kind") == "measurement_packet_bound"
        and event.get("payload", {}).get("entry_id") == entry_id
        and event.get("payload", {}).get("measurement_event_id") == measurement["event_id"]
        and event.get("payload", {}).get("measurement_event_sha256") == measurement_sha
        and event.get("payload", {}).get("packet_sha256") == bound_packet.sha256
        and event.get("payload", {}).get("candidate_sha256") == submission_sha
        and event.get("payload", {}).get("lane") == "primary"
    ]
    if len(binding_events) != 1 or entry_bindings != binding_events:
        raise ManifestError(f"shape {shape_id}: controller packet binding event is absent")
    try:
        queue = require_audit_enqueue(
            entry_id=entry_id,
            candidate_sha256=submission_sha,
            packet_sha256=bound_packet.sha256,
            measurement_event_sha256=measurement_sha,
            lane="primary",
            events_path=audit_events_path(),
        )
    except AuditAuthorityError as exc:
        raise ManifestError(f"shape {shape_id}: audit enqueue is invalid: {exc}") from exc
    environment = require_env_binding(
        payload.get("worker_environment"), f"shape {shape_id} controller measurement"
    )
    audit = eligible_audit(entry_id, submission_sha, bound_packet.sha256)
    return {
        "evidence_kind": "trusted-controller-authority-event",
        "evidence_class": POST_LOCK,
        "evidence_status": "SHIPPABLE",
        "environment": environment,
        "authority_journal": "Project/authority/events.jsonl",
        "evidence_path": "Project/authority/events.jsonl",
        "measurement_event_id": measurement["event_id"],
        "measurement_event_sha256": measurement_sha,
        "entry_id": entry_id,
        "audit_packet_path": str(bound_packet.path.relative_to(ROOT)),
        "audit_packet_sha256": bound_packet.sha256,
        "audit_enqueue_event_sha256": queue["event_sha256"],
        "candidate": {"path": SUBMISSION_REL, "sha256": submission_sha},
        "submission_sha256": submission_sha,
        "correctness": correctness,
        "timing": supporting,
        "median_ms": candidate_median,
        "speedup_vs_baseline": speedup,
        "reference_method": (
            "trusted-controller independent output validation; paired pinned-official timing in isolated worker"
        ),
        **audit_verdict_fields(audit),
        "audit_authority": audit,
        "selection_rationale": selector["selection_rationale"],
    }


def require_packet_binding(packet: dict[str, Any], submission_sha: str,
                           evaluator_path: Path) -> None:
    binding = packet.get("binding")
    expected = {
        "submission_sha256": submission_sha,
        "evaluator_sha256": sha256_file(evaluator_path),
        "official_sha256": sha256_file(OFFICIAL),
        "official_manifest_sha256": sha256_file(OFFICIAL_MANIFEST),
    }
    if binding != expected:
        raise ManifestError(f"side packet binding mismatch: {binding!r} != {expected!r}")


def require_shape14_dependencies(
    packet: dict[str, Any],
    submission_sha: str,
    artifacts: dict[str, tuple[dict[str, Any], str]],
) -> None:
    required = packet.get("required_artifacts")
    if not isinstance(required, dict) or set(required) != {
        "oracle_validation", "batch_decomposition"
    }:
        raise ManifestError("shape 14 packet lacks required validation artifacts")
    expected = {
        "oracle_validation": (
            "shape14_oracle_validation", "shape14-oracle-validation-v2"
        ),
        "batch_decomposition": (
            "shape14_batch_decomposition_check", "shape14-decomposition-v2"
        ),
    }
    for key, (artifact_type, schema) in expected.items():
        reference = required.get(key)
        if not isinstance(reference, dict):
            raise ManifestError(f"shape 14 packet lacks {key} reference")
        artifact, artifact_sha = artifacts[key]
        if reference.get("sha256") != artifact_sha:
            raise ManifestError(f"shape 14 {key} artifact hash mismatch")
        if reference.get("schema_version") != schema:
            raise ManifestError(f"shape 14 {key} referenced schema mismatch")
        path_value = reference.get("path")
        if (
            not isinstance(path_value, str)
            or not path_value.startswith("Project/results_side/")
            or Path(path_value).name != path_value.removeprefix("Project/results_side/")
        ):
            raise ManifestError(f"shape 14 {key} artifact path is not an isolated-stage path")
        if artifact.get("type") != artifact_type or artifact.get("schema_version") != schema:
            raise ManifestError(f"shape 14 {key} artifact type/schema mismatch")
        if artifact.get("passed") is not True:
            raise ManifestError(f"shape 14 {key} artifact did not pass")
        require_packet_binding(
            artifact, submission_sha, PROJECT / "tools" / "shape14_eval.py"
        )


def numeric_series(value: Any, label: str, *, count: int,
                   positive: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise ManifestError(f"{label} must contain exactly {count} samples")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ManifestError(f"{label} contains a non-numeric sample")
        number = float(item)
        if not math.isfinite(number) or number < 0 or (positive and number <= 0):
            raise ManifestError(f"{label} contains an invalid sample")
        result.append(number)
    return result


def same_number(actual: Any, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ManifestError(f"{label} must be numeric")
    number = float(actual)
    if not math.isfinite(number) or not math.isclose(
        number, expected, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ManifestError(f"{label} is inconsistent with retained samples")


def series_slope(values: list[float]) -> float:
    center_x = (len(values) - 1) / 2.0
    center_y = statistics.fmean(values)
    denominator = sum((index - center_x) ** 2 for index in range(len(values)))
    return sum(
        (index - center_x) * (value - center_y)
        for index, value in enumerate(values)
    ) / denominator


def require_official_numerics(packet: dict[str, Any], label: str) -> None:
    if packet.get("numerical_state") != OFFICIAL_NUMERICAL_STATE:
        raise ManifestError(f"{label} did not use the official numerical state")


def require_shape6_protocol(packet: dict[str, Any]) -> None:
    require_official_numerics(packet, "shape 6")
    correctness = packet.get("correctness")
    if not isinstance(correctness, dict):
        raise ManifestError("shape 6 correctness is malformed")
    seeds = correctness.get("seeds")
    trials = correctness.get("trials")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or not isinstance(trials, list)
        or len(trials) != len(seeds)
    ):
        raise ManifestError("shape 6 must retain at least five unique correctness trials")
    for seed, trial in zip(seeds, trials):
        if (
            not isinstance(trial, dict)
            or trial.get("seed") != seed
            or trial.get("passed") is not True
            or trial.get("violations") != 0
            or trial.get("nonfinite_elements") != 0
        ):
            raise ManifestError("shape 6 retained a failed or malformed correctness trial")
    if (
        correctness.get("passed") is not True
        or correctness.get("violations") != 0
        or correctness.get("nonfinite_elements") != 0
    ):
        raise ManifestError("shape 6 aggregate correctness is inconsistent")

    memory = packet.get("memory")
    if not isinstance(memory, dict) or memory.get("limits") != SHAPE6_MEMORY_LIMITS:
        raise ManifestError("shape 6 memory protocol or limits are inconsistent")
    if memory.get("warmups") != 3 or memory.get("repeats") != 10:
        raise ManifestError("shape 6 memory trend did not use the required protocol")
    for key in ("peak_allocated", "peak_reserved", "settled_allocated", "settled_reserved"):
        numeric_series(
            memory.get(f"{key}_bytes_per_repeat"),
            f"shape 6 {key}",
            count=10,
        )
    allocated = numeric_series(
        memory.get("settled_allocated_bytes_per_repeat"),
        "shape 6 settled allocated",
        count=10,
    )
    reserved = numeric_series(
        memory.get("settled_reserved_bytes_per_repeat"),
        "shape 6 settled reserved",
        count=10,
    )
    recomputed = {
        "allocated_slope_bytes_per_repeat": series_slope(allocated),
        "reserved_slope_bytes_per_repeat": series_slope(reserved),
        "allocated_end_growth_bytes": allocated[-1] - allocated[0],
        "reserved_end_growth_bytes": reserved[-1] - reserved[0],
        "allocated_max_growth_bytes": max(allocated) - allocated[0],
        "reserved_max_growth_bytes": max(reserved) - reserved[0],
    }
    for key, value in recomputed.items():
        same_number(memory.get(key), value, f"shape 6 {key}")
    memory_flat = all(
        recomputed[key] <= limit for key, limit in SHAPE6_MEMORY_LIMITS.items()
    )
    if memory.get("flat") is not memory_flat or not memory_flat:
        raise ManifestError("shape 6 memory.flat is false or inconsistent")

    timing = packet.get("timing")
    if not isinstance(timing, dict) or (
        timing.get("warmups") != 20
        or timing.get("repeats_per_round") != 100
        or timing.get("round_count") != 3
        or timing.get("speedup_vs_baseline") is not None
    ):
        raise ManifestError("shape 6 timing protocol is inconsistent")
    raw = numeric_series(
        timing.get("raw_samples_ms"), "shape 6 timing", count=300, positive=True
    )
    rounds = timing.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 3:
        raise ManifestError("shape 6 timing rounds are malformed")
    retained = []
    for index, round_row in enumerate(rounds):
        if not isinstance(round_row, dict) or round_row.get("round") != index:
            raise ManifestError("shape 6 timing round index is malformed")
        retained.extend(numeric_series(
            round_row.get("samples_ms"),
            f"shape 6 timing round {index}",
            count=100,
            positive=True,
        ))
    if retained != raw:
        raise ManifestError("shape 6 raw timing samples disagree with retained rounds")
    same_number(
        timing.get("median_ms"), statistics.median(raw), "shape 6 true median"
    )


def require_shape14_protocol(packet: dict[str, Any],
                             expected_shape: dict[str, Any]) -> None:
    """Validate the streamed shape-14 packet against the official batch size.

    Shape 14 is measured as ``B`` serial B=1 slices summed per repeat, where
    ``B`` is the official batch size and nothing else.  A historical defect
    selected a B=1 packet and compared it against a B=2 packet; latencies at
    different batch sizes are not comparable, so the slice count is anchored to
    the official shape here and no baseline speedup may appear at all.
    """
    require_official_numerics(packet, "shape 14")
    slices = expected_shape.get("batch_size")
    if isinstance(slices, bool) or not isinstance(slices, int) or slices < 1:
        raise ManifestError("shape 14 official batch size is malformed")
    packet_shape = packet.get("shape")
    if not isinstance(packet_shape, dict) or packet_shape.get("batch_size") != slices:
        raise ManifestError(
            f"shape 14 packet batch size is not the official {slices}: "
            "batch sizes are not comparable across packets"
        )
    correctness = packet.get("correctness")
    seeds = packet.get("seeds")
    if (
        not isinstance(correctness, dict)
        or correctness.get("passed") is not True
        or correctness.get("violations") != 0
        or correctness.get("nonfinite_elements") != 0
        or not isinstance(seeds, list)
        or len(seeds) < 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or not isinstance(correctness.get("trials"), list)
        or len(correctness["trials"]) != len(seeds)
    ):
        raise ManifestError("shape 14 correctness protocol is incomplete or failed")
    for seed, trial in zip(seeds, correctness["trials"]):
        if (
            not isinstance(trial, dict)
            or trial.get("base_seed") != seed
            or trial.get("violations") != 0
            or trial.get("nonfinite_elements") != 0
        ):
            raise ManifestError("shape 14 retained a failed correctness trial")

    timing = packet.get("timing")
    if not isinstance(timing, dict):
        raise ManifestError("shape 14 timing is malformed")
    comparison_keys = sorted(
        key for key in
        ("speedup", "speedup_vs_baseline", "baseline", "baseline_median_ms",
         "baseline_timing", "baseline_samples_ms")
        if key in timing or key in packet
    )
    if comparison_keys:
        raise ManifestError(
            f"shape 14 evidence must carry no baseline comparison {comparison_keys}: "
            "the official dense baseline is infeasible at this shape and serial "
            "B=1 slices are not comparable to any other batch size"
        )
    protocol = timing.get("protocol")
    if not isinstance(protocol, str) or "B=1" not in protocol:
        raise ManifestError(
            "shape 14 timing must declare the serial B=1 decomposition it measured"
        )
    repeats = timing.get("timing_repeats")
    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 3
        or not isinstance(timing.get("warmup_slices"), int)
        or timing["warmup_slices"] < 3
    ):
        raise ManifestError("shape 14 timing needs at least three warmups and repeats")
    matrix = timing.get("slice_times_ms")
    if (
        not isinstance(matrix, dict)
        or matrix.get("orientation") != "batch_index x timing_repeat"
        or not isinstance(matrix.get("values"), list)
        or len(matrix["values"]) != slices
    ):
        raise ManifestError(
            f"shape 14 per-slice timing matrix must hold exactly {slices} B=1 slices"
        )
    by_batch = [
        numeric_series(row, f"shape 14 batch timing {index}",
                       count=repeats, positive=True)
        for index, row in enumerate(matrix["values"])
    ]
    compute = numeric_series(
        timing.get("gpu_compute_sum_ms_per_repeat"),
        "shape 14 compute sums",
        count=repeats,
        positive=True,
    )
    wall = numeric_series(
        timing.get("staging_inclusive_wall_ms_per_repeat"),
        "shape 14 staging-inclusive wall",
        count=repeats,
        positive=True,
    )
    for repeat in range(repeats):
        expected_sum = sum(by_batch[batch][repeat] for batch in range(slices))
        same_number(compute[repeat], expected_sum, "shape 14 compute sum")
        if wall[repeat] + 1e-9 < compute[repeat]:
            raise ManifestError("shape 14 staging-inclusive wall is below GPU compute")
    same_number(
        timing.get("gpu_compute_median_of_sums_ms"),
        statistics.median(compute),
        "shape 14 compute median",
    )
    same_number(
        timing.get("staging_inclusive_wall_median_ms"),
        statistics.median(wall),
        "shape 14 wall median",
    )


def build_side_controller_entry(
    selector: dict[str, Any],
    shape_id: int,
    expected_shape: dict[str, Any],
    submission_sha: str,
) -> dict[str, Any]:
    if set(selector) - SIDE_SELECTOR_FIELDS or shape_id not in {6, 14}:
        raise ManifestError(f"shape {shape_id}: invalid side-controller selector")
    entry_id = selector.get("entry_id")
    measurement_sha = selector.get("measurement_event_sha256")
    side_evidence_sha = selector.get("side_evidence_sha256")
    lane = f"shape{shape_id}"
    bound_packet = prospective_audit_packet(
        selector.get("audit_packet"), entry_id, submission_sha
    )
    wrapper = bound_packet.payload
    if (
        bound_packet.measurement_event_sha256 != measurement_sha
        or bound_packet.lane != lane
        or wrapper.get("shape_id") != shape_id
        or wrapper.get("mode") != lane
        or wrapper.get("side_evidence_sha256") != side_evidence_sha
    ):
        raise ManifestError(f"shape {shape_id}: normalized side wrapper binding mismatch")

    try:
        events = AuthorityStore(ROOT).read_events()
    except AuthorityError as exc:
        raise ManifestError(f"side-controller authority is invalid: {exc}") from exc
    entry_measurements = [
        event for event in events
        if event.get("kind") == "measurement_recorded"
        and event.get("payload", {}).get("entry_id") == entry_id
    ]
    measurements = [
        event for event in events
        if event.get("kind") == "measurement_recorded"
        and event.get("event_sha256") == measurement_sha
    ]
    if len(measurements) != 1 or entry_measurements != measurements:
        raise ManifestError(f"shape {shape_id}: side measurement is absent or duplicated")
    measurement = measurements[0]
    payload = measurement.get("payload")
    if not isinstance(payload, dict):
        raise ManifestError(f"shape {shape_id}: side measurement payload is malformed")
    bindings = {
        "entry_id": entry_id,
        "mode": lane,
        "lane": lane,
        "shape_id": shape_id,
        "candidate_sha256": submission_sha,
        "side_evidence_sha256": side_evidence_sha,
        "side_stage_artifacts": wrapper.get("side_stage_artifacts"),
        "controller_validation": wrapper.get("controller_validation"),
        "gate_request_sha256": wrapper.get("gate_request_sha256"),
    }
    if any(payload.get(key) != value for key, value in bindings.items()) or (
        payload.get("evidence_eligible_pre_audit") is not True
        or payload.get("promotion_eligible") is not False
        or not isinstance(payload.get("controller_validation"), dict)
        or payload["controller_validation"].get("passed") is not True
        or wrapper.get("measurement_event_id") != measurement.get("event_id")
    ):
        raise ManifestError(f"shape {shape_id}: side measurement/wrapper chain is mixed")

    entry_bindings = [
        event for event in events
        if event.get("kind") == "measurement_packet_bound"
        and event.get("payload", {}).get("entry_id") == entry_id
    ]
    binding_events = [
        event for event in events
        if event.get("kind") == "measurement_packet_bound"
        and event.get("payload", {}).get("entry_id") == entry_id
        and event.get("payload", {}).get("measurement_event_id") == measurement["event_id"]
        and event.get("payload", {}).get("measurement_event_sha256") == measurement_sha
        and event.get("payload", {}).get("candidate_sha256") == submission_sha
        and event.get("payload", {}).get("packet_sha256") == bound_packet.sha256
        and event.get("payload", {}).get("side_evidence_sha256") == side_evidence_sha
        and event.get("payload", {}).get("lane") == lane
    ]
    if len(binding_events) != 1 or entry_bindings != binding_events:
        raise ManifestError(f"shape {shape_id}: side packet binding event is absent")

    permit_id = payload.get("permit_id")
    run_id = payload.get("run_id")
    issued = [
        event for event in events
        if event.get("kind") == "permit_issued"
        and event.get("payload", {}).get("permit_id") == permit_id
    ]
    consumed = [
        event for event in events
        if event.get("kind") == "permit_consumed"
        and event.get("payload", {}).get("permit_id") == permit_id
    ]
    started = [
        event for event in events
        if event.get("kind") == "run_started"
        and event.get("payload", {}).get("run_id") == run_id
    ]
    if len(issued) != 1 or len(consumed) != 1 or len(started) != 1:
        raise ManifestError(f"shape {shape_id}: one-use side permit chain is incomplete")
    permit_expected = {
        "mode": lane,
        "shape_id": shape_id,
        "candidate_sha256": submission_sha,
        "request_sha256": payload.get("gate_request_sha256"),
    }
    if any(issued[0]["payload"].get(key) != value
           for key, value in permit_expected.items()) or any(
        consumed[0]["payload"].get(key) != value
        for key, value in permit_expected.items() if key != "request_sha256"
    ) or (
        consumed[0]["payload"].get("issued_event_id") != issued[0]["event_id"]
        or started[0]["event_id"] != payload.get("started_event_id")
        or started[0]["payload"].get("consumed_event_id") != consumed[0]["event_id"]
        or started[0]["payload"].get("permit_id") != permit_id
        or started[0]["payload"].get("mode") != lane
        or started[0]["payload"].get("lane") != lane
        or started[0]["payload"].get("shape_id") != shape_id
        or started[0]["payload"].get("candidate_sha256") != submission_sha
        or started[0]["payload"].get("gate_request_sha256")
        != payload.get("gate_request_sha256")
    ):
        raise ManifestError(f"shape {shape_id}: side permit/run bindings are inconsistent")
    event_positions = {event["event_id"]: index for index, event in enumerate(events)}
    if not (
        event_positions[issued[0]["event_id"]]
        < event_positions[consumed[0]["event_id"]]
        < event_positions[started[0]["event_id"]]
        < event_positions[measurement["event_id"]]
        < event_positions[binding_events[0]["event_id"]]
    ):
        raise ManifestError(f"shape {shape_id}: side authority transitions are out of order")

    request = read_json(authority_blob(payload.get("gate_request_sha256"), ".json"))
    if (
        request.get("request_kind") != "side_evaluation"
        or request.get("mode") != lane
        or request.get("shape") != shape_id
        or request.get("impl_sha256") != submission_sha
        or request.get("impl_path") != SUBMISSION_REL
        or request.get("candidate_authorized") is not True
        or request.get("promotion_allowed") is not False
    ):
        raise ManifestError(f"shape {shape_id}: gate request is not an exact side permit")

    expected_stages = ["shape6-eval"] if shape_id == 6 else [
        "shape14-validate", "shape14-decomposition", "shape14-eval"
    ]
    stage_refs = wrapper.get("side_stage_artifacts")
    embedded = wrapper.get("side_evidence_packets")
    if (
        not isinstance(stage_refs, list)
        or not all(isinstance(item, dict) for item in stage_refs)
        or [item.get("stage") for item in stage_refs] != expected_stages
        or not isinstance(embedded, list)
        or len(embedded) != len(expected_stages)
    ):
        raise ManifestError(f"shape {shape_id}: side stage list is incomplete")
    stage_packets: dict[str, tuple[dict[str, Any], str]] = {}
    all_stage_events = [
        event for event in events
        if event.get("kind") == "side_stage_ingested"
        and event.get("payload", {}).get("run_id") == run_id
    ]
    if len(all_stage_events) != len(expected_stages) or any(
        event.get("kind") in {"side_stage_ingest_failed", "run_failed"}
        and event.get("payload", {}).get("run_id") == run_id
        for event in events
    ):
        raise ManifestError(f"shape {shape_id}: side run has missing or adverse stages")
    for index, (stage, reference) in enumerate(zip(expected_stages, stage_refs)):
        digest = reference.get("sha256")
        stage_packet = read_json(authority_blob(digest, ".json"))
        if stage_packet != embedded[index]:
            raise ManifestError(f"shape {shape_id}: embedded {stage} packet changed")
        ingested = [
            event for event in events
            if event.get("kind") == "side_stage_ingested"
            and event.get("payload", {}).get("run_id") == run_id
            and event.get("payload", {}).get("stage") == stage
            and event.get("payload", {}).get("artifact_sha256") == digest
            and event.get("payload", {}).get("returncode") == 0
            and event.get("payload", {}).get("timed_out") is False
        ]
        if len(ingested) != 1 or not (
            event_positions[started[0]["event_id"]]
            < event_positions[ingested[0]["event_id"]]
            < event_positions[measurement["event_id"]]
        ):
            raise ManifestError(f"shape {shape_id}: {stage} ingestion is not authoritative")
        stage_packets[stage] = (stage_packet, digest)
    final_packet, final_sha = stage_packets[expected_stages[-1]]
    if final_sha != side_evidence_sha or final_packet.get("entry_id") != entry_id:
        raise ManifestError(f"shape {shape_id}: final side evidence binding mismatch")
    exact_shape(final_packet.get("shape"), expected_shape, f"shape {shape_id} side packet")
    candidate = final_packet.get("candidate")
    if (
        final_packet.get("passed") is not True
        or not isinstance(candidate, dict)
        or candidate.get("path") != SUBMISSION_REL
        or candidate.get("sha256") != submission_sha
    ):
        raise ManifestError(f"shape {shape_id}: side packet did not pass exact submission")
    evaluator = PROJECT / "tools" / (
        "shape6_local_eval.py" if shape_id == 6 else "shape14_eval.py"
    )
    require_packet_binding(final_packet, submission_sha, evaluator)
    if shape_id == 6:
        if (
            final_packet.get("schema_version") != "shape6-submission-v2"
            or final_packet.get("type") != "shape6_submission_evaluation"
        ):
            raise ManifestError("shape 6 side packet type/schema is invalid")
        require_shape6_protocol(final_packet)
        median_ms = final_packet.get("timing", {}).get("median_ms")
        # The reference actually computed by shape6_local_eval.py is the
        # pinned official baseline evaluated in batch chunks -- the full
        # B=10000 batch does not fit on the project GPU.  The computation is
        # batch-independent, so chunking is exact, but the manifest must never
        # describe this as a full-batch official run: that was the historical
        # factual error in the fabricated shape-6 sentence.
        reference_method = (
            "batch-chunked pinned official baseline "
            "(batch-independent exact computation); "
            "no full-batch official reference was run"
        )
    else:
        if (
            final_packet.get("schema_version") != "shape14-streamed-v2"
            or final_packet.get("type") != "shape14_side_evaluation"
        ):
            raise ManifestError("shape 14 side packet type/schema is invalid")
        validation_packet, validation_sha = stage_packets["shape14-validate"]
        decomposition_packet, decomposition_sha = stage_packets["shape14-decomposition"]
        require_shape14_dependencies(
            final_packet,
            submission_sha,
            {
                "oracle_validation": (validation_packet, validation_sha),
                "batch_decomposition": (decomposition_packet, decomposition_sha),
            },
        )
        require_shape14_protocol(final_packet, expected_shape)
        median_ms = final_packet.get("timing", {}).get(
            "gpu_compute_median_of_sums_ms"
        )
        # Never hard-code the slice count: it is the official batch size that
        # require_shape14_protocol just anchored the packet to, so the prose
        # and the validated protocol cannot drift apart.
        official_batch = expected_shape["batch_size"]
        reference_method = (
            "validated streamed fp32 oracle; "
            f"{official_batch} serial B=1 slices, "
            f"not literal B={official_batch}"
        )
    median_ms = require_positive_number(median_ms, f"shape {shape_id} median")
    environment = require_env_binding(
        final_packet.get("env"), f"shape {shape_id} side packet"
    )
    try:
        queue = require_audit_enqueue(
            entry_id=entry_id,
            candidate_sha256=submission_sha,
            packet_sha256=bound_packet.sha256,
            measurement_event_sha256=measurement_sha,
            lane=lane,
            events_path=audit_events_path(),
        )
    except AuditAuthorityError as exc:
        raise ManifestError(f"shape {shape_id}: audit enqueue is invalid: {exc}") from exc
    audit = eligible_audit(entry_id, submission_sha, bound_packet.sha256)
    return {
        "evidence_kind": "trusted-controller-side-authority-event",
        "evidence_class": POST_LOCK,
        "evidence_status": "SHIPPABLE",
        "environment": environment,
        "authority_journal": "Project/authority/events.jsonl",
        "measurement_event_id": measurement["event_id"],
        "measurement_event_sha256": measurement_sha,
        "evidence_path": f"Project/authority/blobs/{side_evidence_sha}.json",
        "evidence_sha256": side_evidence_sha,
        "entry_id": entry_id,
        "audit_packet_path": str(bound_packet.path.relative_to(ROOT)),
        "audit_packet_sha256": bound_packet.sha256,
        "audit_enqueue_event_sha256": queue["event_sha256"],
        "candidate": candidate,
        "submission_sha256": submission_sha,
        "correctness": final_packet["correctness"],
        "memory": final_packet.get("memory"),
        "timing": final_packet["timing"],
        "median_ms": median_ms,
        "reference_method": reference_method,
        **audit_verdict_fields(audit),
        "audit_authority": audit,
        "selection_rationale": selector["selection_rationale"],
    }


def validate_evidence_map(payload: dict[str, Any], submission_sha: str,
                          official_sha: str,
                          expected_shapes: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    try:
        from jsonschema import Draft202012Validator

        schema = read_json(MAP_SCHEMA_FILE)
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(payload),
            key=lambda error: tuple(map(str, error.absolute_path)),
        )
    except Exception as exc:
        raise ManifestError(f"final evidence map schema is unavailable: {exc}") from exc
    if errors:
        rendered = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise ManifestError(f"final evidence map is invalid: {rendered}")
    if payload.get("schema_version") != MAP_SCHEMA:
        raise ManifestError(f"evidence map schema must be {MAP_SCHEMA}")
    if payload.get("submission_sha256") != submission_sha:
        raise ManifestError("evidence map is not bound to the final submission")
    if payload.get("official_sha256") != official_sha:
        raise ManifestError("evidence map is not bound to pinned official bytes")
    selectors = payload.get("shapes")
    if not isinstance(selectors, dict):
        raise ManifestError("evidence map shapes must be an object")
    expected_keys = {str(shape_id) for shape_id in expected_shapes}
    actual_keys = set(selectors)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=int)
        extra = sorted(actual_keys - expected_keys)
        raise ManifestError(f"evidence map must select all shapes; missing={missing}, extra={extra}")
    for shape_id, selector in selectors.items():
        if not isinstance(selector, dict) or selector.get("kind") not in SELECTOR_FIELDS:
            raise ManifestError(f"shape {shape_id}: invalid evidence selector")
        unknown = sorted(set(selector) - SELECTOR_FIELDS[selector["kind"]])
        if unknown:
            raise ManifestError(f"shape {shape_id}: unknown selector fields {unknown}")
        if not isinstance(selector.get("selection_rationale"), str) \
                or not selector["selection_rationale"].strip():
            raise ManifestError(
                f"shape {shape_id}: every selection must state its rationale"
            )
        numeric_shape = int(shape_id)
        if numeric_shape in {6, 14} and selector.get("kind") != "side_controller":
            raise ManifestError(
                f"shape {shape_id}: dedicated side-controller evidence is required"
            )
        if numeric_shape not in {6, 14} and selector.get("kind") == "side_controller":
            raise ManifestError(
                f"shape {shape_id}: side-controller evidence is not valid for this shape"
            )
    return selectors


def build_manifest(evidence_map_path: Path, head: str,
                   submission_sha: str) -> dict[str, Any]:
    """Build the manifest, or refuse with one honest reason per failing shape.

    Every shape is attempted.  A single unusable shape does not hide the state
    of the other thirteen: the refusal reports all of them at once, naming the
    exact missing binding, so the reader can see the whole evidence chain.
    """
    official_sha = sha256_file(OFFICIAL)
    expected_shapes = official_shape_map()
    map_payload = read_json(evidence_map_path)
    selectors = validate_evidence_map(
        map_payload, submission_sha, official_sha, expected_shapes
    )
    rows: dict[str, dict[str, Any]] | None = None
    selections: dict[str, Any] = {}
    refusals: dict[str, Any] = {}
    for shape_id in sorted(expected_shapes):
        selector = selectors[str(shape_id)]
        kind = selector["kind"]
        evidence_class = LEGACY_PRE_LOCK if kind == "journal" else UNESTABLISHED
        try:
            if kind == "controller":
                selection = build_controller_entry(
                    selector, shape_id, expected_shapes[shape_id], submission_sha
                )
            elif kind == "journal":
                if rows is None:
                    rows = read_journal()
                raise ManifestError("; ".join(journal_evidence_reasons(
                    selector, shape_id, expected_shapes[shape_id],
                    submission_sha, rows,
                )))
            else:
                selection = build_side_controller_entry(
                    selector, shape_id, expected_shapes[shape_id], submission_sha
                )
        except (ManifestError, AuditAuthorityError, AuthorityError, OSError) as exc:
            row = {
                "shape_id": shape_id,
                "selector_kind": kind,
                "entry_id": selector.get("entry_id"),
                "evidence_class": evidence_class,
                "evidence_status": "REFUSED",
                "audit_verdict": None,
                "reason": str(exc),
            }
            if kind == "side_controller":
                # A side selection fails on its first missing binding, which is
                # usually just "no such blob".  Say what is actually on disk for
                # this shape instead, so the refusal names the specific missing
                # binding rather than stopping at the symptom.
                row["side_evidence_diagnosis"] = side_stage_diagnosis(shape_id)
            refusals[str(shape_id)] = row
        else:
            selections[str(shape_id)] = selection
    if refusals:
        raise ManifestRefusal(refusals)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_revision": head,
        "git_revision_contains_submission": True,
        "evidence_classes": {
            POST_LOCK: sorted(
                (int(key) for key, row in selections.items()
                 if row["evidence_class"] == POST_LOCK),
            ),
            LEGACY_PRE_LOCK: [],
        },
        "legacy_evidence_policy": (
            "Project/results/JOURNAL.jsonl and Project/results_side/*.json are "
            "pre-lock evidence history and are never shippable; every shape here "
            "is a post-lock trusted-controller measurement with a bound, "
            "independently recorded audit verdict"
        ),
        "submission_file": SUBMISSION_REL,
        "submission_sha256": submission_sha,
        "official_script_sha256": official_sha,
        "official_manifest_sha256": sha256_file(OFFICIAL_MANIFEST),
        "evidence_map": {
            "path": str(evidence_map_path.relative_to(ROOT)),
            "sha256": sha256_file(evidence_map_path),
            "schema_version": MAP_SCHEMA,
            "schema_path": str(MAP_SCHEMA_FILE.relative_to(ROOT)),
            "schema_sha256": sha256_file(MAP_SCHEMA_FILE),
        },
        "shapes": selections,
        "selection_policy": "explicit owner final evidence map; PASS-eligible audit authority required",
    }


def side_stage_diagnosis(shape_id: int) -> list[str]:
    """Why the loose shape 6 / shape 14 side files cannot be selected."""
    reasons = []
    pattern = f"shape{shape_id}_*.json"
    loose = sorted(SIDE.glob(pattern)) if SIDE.is_dir() else []
    if not loose:
        reasons.append(f"no_side_evidence_present:{pattern}")
    else:
        names = ", ".join(path.name for path in loose)
        reasons.append(
            f"{len(loose)} loose side file(s) in Project/results_side ({names}) "
            "are outside the authority store: no permit, no run, no "
            "measurement_recorded event and no content-addressed blob"
        )
        for path in loose:
            try:
                packet = read_json(path)
            except ManifestError as exc:
                reasons.append(f"{path.name}:unreadable:{exc}")
                continue
            if "binding" not in packet:
                reasons.append(
                    f"{path.name}: no binding block (submission/evaluator/"
                    "official/manifest SHAs are not bound together)"
                )
            try:
                require_env_binding(packet.get("env"), path.name)
            except ManifestError as exc:
                reasons.append(str(exc))
            measured_batch = (packet.get("shape") or {}).get("batch_size")
            reasons.append(f"{path.name}: measured batch_size={measured_batch!r}")
    return reasons


def diagnose(submission_sha: str, head: str | None,
             tree_clean: bool) -> dict[str, Any]:
    """Explain, per official shape, exactly what is missing today.

    This is a read-only report.  It never writes a manifest and never softens a
    rule; it exists so that a refusal names the missing binding for each shape
    instead of stopping at the first problem.
    """
    expected_shapes = official_shape_map()
    try:
        rows = read_journal()
    except (ManifestError, OSError) as exc:
        rows = {}
        journal_error = str(exc)
    else:
        journal_error = None
    per_shape: dict[str, Any] = {}
    for shape_id in sorted(expected_shapes):
        candidates = [
            row for row in rows.values()
            if row.get("type") == "candidate" and row.get("shape_id") == shape_id
        ]
        promoted = [row for row in candidates if row.get("promoted") is True]
        examined = []
        statuses: dict[str, int] = {}
        for row in promoted:
            impl = row.get("impl") if isinstance(row.get("impl"), dict) else {}
            try:
                decision = bound_audit_decision(
                    row["entry_id"], impl.get("sha256"), None
                )
            except AuditAuthorityError as exc:
                examined.append({
                    "entry_id": row["entry_id"],
                    "error": str(exc),
                })
                statuses["AUTHORITY_ERROR"] = statuses.get("AUTHORITY_ERROR", 0) + 1
                continue
            statuses[decision.integrity_status] = (
                statuses.get(decision.integrity_status, 0) + 1
            )
            examined.append({
                "entry_id": row["entry_id"],
                "measured_path": impl.get("path"),
                "integrity_status": decision.integrity_status,
                "technical_status": decision.technical_status,
                "promotion_eligible": decision.promotion_eligible,
                "blocking_reasons": list(decision.blocking_reasons),
            })
        eligible = [item for item in examined if item.get("promotion_eligible")]
        reasons: list[str] = []
        if journal_error:
            reasons.append(f"journal_unreadable:{journal_error}")
        if shape_id in {6, 14}:
            reasons.extend(side_stage_diagnosis(shape_id))
        if not promoted and shape_id not in {6, 14}:
            reasons.append("no promoted candidate row exists for this shape")
        if promoted:
            reasons.append(
                f"{len(promoted)} promoted pre-lock journal row(s); "
                f"integrity status counts {dict(sorted(statuses.items()))}; "
                "0 promotion-eligible"
            )
            distinct = sorted({
                reason
                for item in examined
                for reason in item.get("blocking_reasons", [])
            })
            reasons.extend(f"blocking:{reason}" for reason in distinct)
            measured = sorted({
                item.get("measured_path") for item in examined
                if item.get("measured_path")
            })
            if measured and SUBMISSION_REL not in measured:
                reasons.append(
                    "measured bytes are kernel files "
                    f"({', '.join(measured)}), not {SUBMISSION_REL}"
                )
        per_shape[str(shape_id)] = {
            "shape_id": shape_id,
            "evidence_class": LEGACY_PRE_LOCK,
            "evidence_status": "NOT_SHIPPABLE",
            "audit_verdict": None,
            "shippable_evidence_count": len(eligible),
            "reasons": reasons,
            "examined": examined,
        }
    unshippable = [
        key for key, row in per_shape.items()
        if row["shippable_evidence_count"] == 0
    ]
    return {
        "schema_version": DIAGNOSIS_SCHEMA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_revision": head,
        "working_tree_clean": tree_clean,
        "submission_file": SUBMISSION_REL,
        "submission_sha256": submission_sha,
        "authority_events_present": (ROOT / "Project" / "authority" / "events.jsonl").exists(),
        "audit_events_present": audit_events_path().exists(),
        "shippable": not unshippable,
        "unshippable_shapes": sorted(unshippable, key=int),
        "conclusion": (
            "No official shape has post-lock bound evidence. Everything on disk "
            "is legacy pre-lock evidence history. A full re-measurement campaign "
            "under Project/harness/trusted_controller.py must run before a ship "
            "manifest can exist."
        ) if unshippable else "every shape has post-lock bound evidence",
        "shapes": per_shape,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build explicit fail-closed ship manifest")
    parser.add_argument("--evidence-map",
                        help="committed final-evidence-map-v1 JSON file "
                             "(required unless --diagnose)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="manifest output path")
    parser.add_argument("--report", action="store_true",
                        help="print the exact chosen evidence and audit state")
    parser.add_argument("--diagnose", action="store_true",
                        help="read-only: explain per shape what evidence is "
                             "missing; writes nothing and never softens a rule")
    return parser.parse_args()


def atomic_write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    try:
        if args.diagnose:
            status = git_output(
                "status", "--porcelain=v1", "--untracked-files=all")
            head = git_output("rev-parse", "HEAD")
            report = diagnose(sha256_file(SUBMISSION), head, not status)
            print(json.dumps(report, indent=2, sort_keys=True))
            if not report["shippable"]:
                print(
                    "SHIP MANIFEST REFUSED: " + report["conclusion"],
                    file=sys.stderr,
                )
                return 2
            return 0
        if not args.evidence_map:
            raise ManifestError("--evidence-map is required unless --diagnose")
        evidence_map_path = resolve_repo_file(args.evidence_map)
        output = Path(args.output).resolve()
        try:
            output.relative_to(SIDE.resolve())
        except ValueError as exc:
            raise ManifestError("output must remain inside Project/results_side") from exc
        if output == evidence_map_path:
            raise ManifestError("output must not overwrite the evidence map")
        head, submission_sha = freeze_provenance(evidence_map_path)
        manifest = build_manifest(evidence_map_path, head, submission_sha)
        atomic_write_manifest(output, manifest)
        if args.report:
            report = {
                shape_id: {
                    "entry_id": entry["entry_id"],
                    "evidence_path": entry["evidence_path"],
                    "median_ms": entry["median_ms"],
                    "integrity_status": entry["audit_authority"]["integrity_status"],
                    "technical_status": entry["audit_authority"]["technical_status"],
                }
                for shape_id, entry in manifest["shapes"].items()
            }
            print(json.dumps(report, indent=2, sort_keys=True))
        print(
            f"{output} — {len(manifest['shapes'])} explicitly selected, "
            "audit-eligible shapes"
        )
        return 0
    except ManifestRefusal as exc:
        print(f"SHIP MANIFEST REFUSED: {exc}", file=sys.stderr)
        print(json.dumps(exc.shapes, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    except (ManifestError, AuditAuthorityError, AuthorityError, OSError) as exc:
        print(f"SHIP MANIFEST REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
