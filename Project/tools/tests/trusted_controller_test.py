#!/usr/bin/env python3
"""CPU-only invariants for the post-lock trusted controller."""

from __future__ import annotations

import copy
import math
import statistics
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Project" / "harness"))
sys.path.insert(0, str(REPO / "Project" / "tools"))

from authority import AuthorityStore  # noqa: E402
from candidate_worker import (  # noqa: E402
    WORKER_ENVIRONMENT_KEYS as PRODUCER_ENVIRONMENT_KEYS,
)
from ship_manifest import REQUIRED_ENV_KEYS  # noqa: E402
from trusted_controller import (  # noqa: E402
    ControllerRefusal,
    MANIFEST,
    OFFICIAL,
    SHAPE6_EVALUATOR,
    SHAPES,
    SUBMISSION,
    WORKER_ENVIRONMENT_KEYS,
    sha256_file,
    terminalize_consumed_run,
    validate_shape6_packet,
    validate_supporting_timing,
    validate_worker_environment,
)


def stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    p90 = ordered[max(0, min(len(ordered) - 1, math.ceil(.9 * len(ordered)) - 1))]
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": p90,
        "min_ms": min(samples),
        "n_samples": len(samples),
        "raw_samples_ms": samples,
    }


def timing_fixture() -> dict:
    baseline_samples = [2.0, 2.1, 1.9, 2.2, 2.0, 1.8]
    candidate_samples = [1.0, 1.1, .9, 1.2, 1.0, .8]
    event_speedup = statistics.median(baseline_samples) / statistics.median(
        candidate_samples
    )
    baseline_wall = 2.04
    candidate_wall = 1.02
    wall_speedup = baseline_wall / candidate_wall
    agreement = max(event_speedup, wall_speedup) / min(event_speedup, wall_speedup)
    return {
        "baseline": stats(baseline_samples),
        "candidate": stats(candidate_samples),
        "event_speedup": event_speedup,
        "baseline_wall_ms_per_iter": baseline_wall,
        "candidate_wall_ms_per_iter": candidate_wall,
        "wall_speedup": wall_speedup,
        "event_wall_speedup_agreement_ratio": agreement,
        "suspicious": agreement > 1.25,
        "authority": "supporting-worker-measurement",
    }


def shape6_fixture() -> dict:
    shape = next(item for item in __import__("json").loads(
        SHAPES.read_text()
    )["shapes"] if item["id"] == 6)
    candidate_sha = sha256_file(SUBMISSION)
    samples = [1.0 + index / 1000 for index in range(300)]
    zeros = [0] * 10
    limits = {
        "allocated_slope_bytes_per_repeat": 16 * 2**20,
        "reserved_slope_bytes_per_repeat": 64 * 2**20,
        "allocated_end_growth_bytes": 64 * 2**20,
        "reserved_end_growth_bytes": 256 * 2**20,
        "allocated_max_growth_bytes": 64 * 2**20,
        "reserved_max_growth_bytes": 256 * 2**20,
    }
    return {
        "schema_version": "shape6-submission-v2",
        "type": "shape6_submission_evaluation",
        "shape": shape,
        "binding": {
            "submission_sha256": candidate_sha,
            "evaluator_sha256": sha256_file(SHAPE6_EVALUATOR),
            "official_sha256": sha256_file(OFFICIAL),
            "official_manifest_sha256": sha256_file(MANIFEST),
        },
        "candidate": {"sha256": candidate_sha},
        "correctness": {
            "seeds": [1234, 1235, 1236, 1237, 1238],
            "trials": [
                {"seed": seed, "violations": 0, "nonfinite_elements": 0,
                 "passed": True}
                for seed in [1234, 1235, 1236, 1237, 1238]
            ],
            "violations": 0,
            "nonfinite_elements": 0,
            "passed": True,
        },
        "timing": {
            "warmups": 20,
            "repeats_per_round": 100,
            "round_count": 3,
            "raw_samples_ms": samples,
            "median_ms": statistics.median(samples),
            "rounds": [
                {"round": index, "samples_ms": samples[index * 100:(index + 1) * 100]}
                for index in range(3)
            ],
        },
        "memory": {
            "repeats": 10,
            "settled_allocated_bytes_per_repeat": zeros,
            "settled_reserved_bytes_per_repeat": zeros,
            "allocated_slope_bytes_per_repeat": 0.0,
            "reserved_slope_bytes_per_repeat": 0.0,
            "allocated_end_growth_bytes": 0,
            "reserved_end_growth_bytes": 0,
            "allocated_max_growth_bytes": 0,
            "reserved_max_growth_bytes": 0,
            "limits": limits,
            "flat": True,
        },
        "passed": True,
    }


class FailingController:
    def __init__(self, root: Path):
        self.store = AuthorityStore(root)

    @terminalize_consumed_run
    def run(self, *, permit_id: str):
        self.store.append(
            kind="run_started",
            actor="trusted-controller",
            payload={"run_id": "run-test", "permit_id": permit_id},
        )
        raise ControllerRefusal("deliberate validation failure")


def main() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        print(("PASS " if condition else "FAIL ") + name)
        if not condition:
            failures.append(name)

    valid = timing_fixture()
    normalized = validate_supporting_timing(valid, {"warmup": 1, "repeats": 3, "rounds": 2})
    check("all raw timing samples and arithmetic validate", normalized["event_speedup"] == 2.0)

    bad_median = copy.deepcopy(valid)
    bad_median["candidate"]["median_ms"] += 0.01
    try:
        validate_supporting_timing(bad_median, {"warmup": 1, "repeats": 3, "rounds": 2})
    except ControllerRefusal:
        median_denied = True
    else:
        median_denied = False
    check("forged timing median is denied", median_denied)

    missing_sample = copy.deepcopy(valid)
    missing_sample["baseline"]["raw_samples_ms"].pop()
    try:
        validate_supporting_timing(missing_sample, {"warmup": 1, "repeats": 3, "rounds": 2})
    except ControllerRefusal:
        sample_denied = True
    else:
        sample_denied = False
    check("dropped raw timing sample is denied", sample_denied)

    check(
        "worker and controller environment key sets match",
        PRODUCER_ENVIRONMENT_KEYS == WORKER_ENVIRONMENT_KEYS,
    )
    check(
        "controller environment covers every shipping requirement",
        set(REQUIRED_ENV_KEYS) <= WORKER_ENVIRONMENT_KEYS,
    )
    valid_environment = {
        "python": "3.14.7",
        "torch": "2.12.0+cu130",
        "cuda": "13.0",
        "gpu": "NVIDIA GeForce RTX 3060 Ti",
        "driver": "610.57.04",
        "triton": "3.7.0",
    }
    check(
        "complete worker environment validates",
        validate_worker_environment(valid_environment) == valid_environment,
    )
    for bad_field in ("driver", "triton"):
        malformed = dict(valid_environment)
        malformed[bad_field] = "unknown"
        try:
            validate_worker_environment(malformed)
        except ControllerRefusal:
            refused = True
        else:
            refused = False
        check(f"unknown {bad_field} is denied before evidence is recorded", refused)

    shape6 = shape6_fixture()
    shape6_summary = validate_shape6_packet(shape6, sha256_file(SUBMISSION))
    check("shape-6 side packet arithmetic validates", shape6_summary["passed"] is True)
    forged_shape6 = copy.deepcopy(shape6)
    forged_shape6["memory"]["flat"] = False
    try:
        validate_shape6_packet(forged_shape6, sha256_file(SUBMISSION))
    except ControllerRefusal:
        memory_denied = True
    else:
        memory_denied = False
    check("forged shape-6 memory.flat is denied", memory_denied)

    with tempfile.TemporaryDirectory(prefix="controller-terminal-") as temp:
        controller = FailingController(Path(temp))
        try:
            controller.run(permit_id="permit-test")
        except ControllerRefusal:
            pass
        events = controller.store.read_events()
        terminal = [event for event in events if event["kind"] == "run_failed"]
        check("post-start exception becomes durable run_failed", len(terminal) == 1)
        check(
            "durable failure binds exact run",
            bool(terminal and terminal[0]["payload"].get("run_id") == "run-test"),
        )

    print(f"\n{len(failures)} failure(s)" if failures else "\nALL GREEN")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
