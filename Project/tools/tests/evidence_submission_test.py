#!/usr/bin/env python3
"""CPU/static regression tests for evidence paths and submission safety.

The manifest tests below deliberately drive the PRODUCTION functions in
``ship_manifest.py`` and the real ``audit_authority``/``authority`` modules.
Fixtures only supply an isolated temporary repository root; no assertion is
made against a re-implementation of the logic under test, and no test reads or
writes the real ledgers under ``Project/results``, ``Project/results_side``,
``Project/audits`` or ``Project/authority`` other than read-only.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "Project" / "tools"
HARNESS = ROOT / "Project" / "harness"
for _path in (TOOLS, HARNESS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import audit_authority
import build_submission
import run_gate
import shape14_eval
import shape6_local_eval
import ship_manifest
from audit_authority import CodexIdentity
from authority import AuthorityStore

SUBMISSION_REL = ship_manifest.SUBMISSION_REL
CODEX = CodexIdentity(
    invoked_path="/usr/local/bin/codex",
    resolved_path="/usr/local/bin/codex",
    sha256="ab" * 32,
)


def load_submission():
    path = ROOT / "Project/submission/torch_transformer_benchmark_submission.py"
    spec = importlib.util.spec_from_file_location("evidence_test_submission", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Isolated fixture repository
# --------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_blob(payload) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


class FixtureRepo:
    """A throwaway repository root holding a complete post-lock evidence chain."""

    def __init__(self, directory: Path):
        self.root = directory
        self.project = directory / "Project"
        self.blobs = self.project / "authority" / "blobs"
        self.events_path = self.project / "audits" / "audit_events.jsonl"
        self.lock_path = self.project / "audits" / ".audit_authority.lock"
        for path in (
            self.blobs,
            self.project / "audits" / "auto",
            self.project / "audits" / "packets",
            self.project / "results",
            self.project / "results_side",
            self.project / "submission",
            self.project / "tools",
        ):
            path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "Project" / "shapes.json", self.project / "shapes.json")
        self.shapes = {
            shape["id"]: shape
            for shape in json.loads((self.project / "shapes.json").read_text())["shapes"]
        }
        self.submission_path = self.root / SUBMISSION_REL
        self.submission_bytes = b"# isolated fixture submission\n"
        self.submission_path.write_bytes(self.submission_bytes)
        self.submission_sha = sha256_bytes(self.submission_bytes)
        self.official_path = self.root / "torch_transformer_benchmark.py"
        self.official_path.write_bytes(b"# isolated fixture official\n")
        self.official_sha = sha256_bytes(self.official_path.read_bytes())
        self.official_manifest_path = self.project / "manifest.json"
        self.official_manifest_path.write_text(json.dumps({"official_commit": "fixture"}))
        self.manifest_sha = sha256_bytes(self.official_manifest_path.read_bytes())
        for name in ("shape6_local_eval.py", "shape14_eval.py"):
            (self.project / "tools" / name).write_bytes(f"# {name}\n".encode())
        shutil.copyfile(
            TOOLS / "final_evidence_map.schema.json",
            self.project / "tools" / "final_evidence_map.schema.json",
        )
        (self.blobs / f"{self.submission_sha}.py").write_bytes(self.submission_bytes)
        (self.project / "results" / "JOURNAL.jsonl").write_text("")
        self.store = AuthorityStore(self.root)
        self.store.ensure_layout()

    # -- primitives ------------------------------------------------------
    def evaluator_sha(self, name: str) -> str:
        return sha256_bytes((self.project / "tools" / name).read_bytes())

    def binding(self, evaluator: str) -> dict:
        return {
            "submission_sha256": self.submission_sha,
            "evaluator_sha256": self.evaluator_sha(evaluator),
            "official_sha256": self.official_sha,
            "official_manifest_sha256": self.manifest_sha,
        }

    def env(self) -> dict:
        return {
            "gpu": "NVIDIA GeForce RTX 3060 Ti",
            "driver": "610.57.04",
            "torch": "2.12.0+cu130",
            "cuda": "13.0",
            "triton": "3.7.0",
            "python": "3.14.7",
            "hostname": "fixture",
        }

    def put_blob(self, payload, suffix=".json") -> str:
        data = canonical_blob(payload)
        digest = sha256_bytes(data)
        (self.blobs / f"{digest}{suffix}").write_bytes(data)
        return digest

    def append(self, kind: str, payload: dict) -> dict:
        return self.store.append(kind=kind, actor="fixture-controller", payload=payload)

    # -- audit chain -----------------------------------------------------
    def bind_audit(self, *, entry_id, packet_sha, measurement_sha, lane,
                   integrity="PASS", technical="PASS",
                   candidate_sha=None) -> dict:
        candidate_sha = candidate_sha or self.submission_sha
        queue = audit_authority.enqueue_audit(
            entry_id=entry_id,
            candidate_sha256=candidate_sha,
            packet_sha256=packet_sha,
            measurement_event_sha256=measurement_sha,
            lane=lane,
            path=self.events_path,
            lock_path=self.lock_path,
        )
        nonce = sha256_bytes(entry_id.encode())
        attempt_id = f"attempt-{entry_id}"
        audit_authority.record_attempt_started(
            entry_id=entry_id,
            attempt_id=attempt_id,
            attempt_nonce=nonce,
            packet_sha256=packet_sha,
            candidate_sha256=candidate_sha,
            codex=CODEX,
            path=self.events_path,
            lock_path=self.lock_path,
            measurement_event_sha256=measurement_sha,
            lane=lane,
            queue_event_sha256=queue["event_sha256"],
        )
        verdict = {
            "schema_version": 2,
            "attempt_nonce": nonce,
            "entry_id": entry_id,
            "packet_sha256": packet_sha,
            "candidate_sha256": candidate_sha,
            "integrity": {
                "verdict": integrity,
                "findings": [],
                "retest_request": "re-measure on a quiet box" if integrity == "RETEST" else "",
                "summary": f"integrity {integrity}",
            },
            "technical_review": {
                "verdict": technical,
                "findings": [],
                "summary": f"technical {technical}",
            },
            "summary": "fixture verdict",
        }
        artifact = {
            "artifact_type": "audit_response",
            "attempt_id": attempt_id,
            "entry_id": entry_id,
            "packet_sha256": packet_sha,
            "candidate_sha256": candidate_sha,
            "measurement_event_sha256": measurement_sha,
            "lane": lane,
            "verdict_schema_sha256": audit_authority.sha256_file(
                audit_authority.SCHEMA_PATH
            ),
            "codex": CODEX.as_dict(),
            "validated_result": verdict,
            "stdout": json.dumps(verdict, sort_keys=True),
            "parser_error": "",
            "returncode": 0,
        }
        name = f"{attempt_id}.response.json"
        artifact_path = self.project / "audits" / "auto" / name
        artifact_sha = audit_authority.exclusive_write_json(artifact_path, artifact)
        return audit_authority.record_audit_result(
            attempt_id=attempt_id,
            result=verdict,
            artifact_path=f"Project/audits/auto/{name}",
            artifact_sha256=artifact_sha,
            path=self.events_path,
            lock_path=self.lock_path,
            artifact_root=self.root,
        )

    # -- primary controller measurement ----------------------------------
    def controller_measurement(self, *, entry_id, shape_id, candidate_ms=1.0,
                               baseline_ms=2.0, environment=None,
                               integrity="PASS", technical="PASS",
                               operation="candidate", request_overrides=None,
                               payload_overrides=None) -> dict:
        shape = self.shapes[shape_id]
        # Mirrors Project/harness/trusted_controller.py::_worker_request field
        # for field.  A fixture request that invents its own shape would let
        # this suite pass while the production path is broken.
        request = {
            "schema_version": 1,
            "request_id": "0" * 32,
            "operation": operation,
            "shape_id": shape_id,
            "shape": {key: shape[key] for key in (
                "id", "batch_size", "seq_len", "d_model", "num_heads",
                "ffn_dim", "num_layers", "causal")},
            "dtype": "float32",
            "seeds": list(ship_manifest.CONTROLLER_OFFICIAL_SEEDS) + [909091, 909092],
            "timing_args": dict(ship_manifest.CONTROLLER_TIMING),
            "numerical": dict(ship_manifest.CONTROLLER_NUMERICAL),
            "candidate_sha256": self.submission_sha,
            "official_sha256": self.official_sha,
            "shapes_sha256": sha256_bytes(
                (self.project / "shapes.json").read_bytes()
            ),
            "challenge_nonce": "1" * 64,
        }
        request.update(request_overrides or {})
        request_sha = self.put_blob(request)
        baseline = [baseline_ms] * 300
        candidate = [candidate_ms] * 300
        payload = {
            "entry_id": entry_id,
            "shape_id": shape_id,
            "lane": "primary",
            "mode": "optimization",
            "candidate_sha256": self.submission_sha,
            "worker_request_sha256": request_sha,
            "worker_environment": environment or self.env(),
            "controller_correctness": {"passed": True, "violations": 0},
            "timing_args": dict(ship_manifest.CONTROLLER_TIMING),
            "numerical": dict(ship_manifest.CONTROLLER_NUMERICAL),
            "effective_numerical_state": dict(
                ship_manifest.CONTROLLER_EFFECTIVE_NUMERICAL_STATE
            ),
            "supporting_timing": {
                "suspicious": False,
                "baseline": {
                    "raw_samples_ms": baseline,
                    "median_ms": statistics.median(baseline),
                    "n_samples": 300,
                },
                "candidate": {
                    "raw_samples_ms": candidate,
                    "median_ms": statistics.median(candidate),
                    "n_samples": 300,
                },
                "event_speedup": baseline_ms / candidate_ms,
            },
        }
        payload.update(payload_overrides or {})
        measurement = self.append("measurement_recorded", payload)
        packet = {
            "schema_version": 1,
            "entry_id": entry_id,
            "candidate_sha256": self.submission_sha,
            "measurement_event_sha256": measurement["event_sha256"],
            "measurement_event_id": measurement["event_id"],
            "lane": "primary",
            "shape_id": shape_id,
        }
        packet_sha = self.put_blob(packet)
        self.append("measurement_packet_bound", {
            "entry_id": entry_id,
            "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "packet_sha256": packet_sha,
            "candidate_sha256": self.submission_sha,
            "lane": "primary",
        })
        self.bind_audit(
            entry_id=entry_id,
            packet_sha=packet_sha,
            measurement_sha=measurement["event_sha256"],
            lane="primary",
            integrity=integrity,
            technical=technical,
        )
        return {
            "kind": "controller",
            "entry_id": entry_id,
            "measurement_event_sha256": measurement["event_sha256"],
            "audit_packet": {
                "path": f"Project/authority/blobs/{packet_sha}.json",
                "sha256": packet_sha,
            },
            "selection_rationale": "fixture selection",
        }

    # -- side-lane measurement -------------------------------------------
    def side_measurement(self, *, entry_id, shape_id, stage_packets,
                         integrity="PASS", technical="PASS") -> dict:
        lane = f"shape{shape_id}"
        stages = list(stage_packets)
        request = {
            "request_kind": "side_evaluation",
            "mode": lane,
            "shape": shape_id,
            "impl_sha256": self.submission_sha,
            "impl_path": SUBMISSION_REL,
            "candidate_authorized": True,
            "promotion_allowed": False,
        }
        request_sha = self.put_blob(request)
        permit_id = f"permit-{entry_id}"
        run_id = f"run-{entry_id}"
        issued = self.append("permit_issued", {
            "permit_id": permit_id, "mode": lane, "shape_id": shape_id,
            "candidate_sha256": self.submission_sha, "request_sha256": request_sha,
        })
        consumed = self.append("permit_consumed", {
            "permit_id": permit_id, "mode": lane, "shape_id": shape_id,
            "candidate_sha256": self.submission_sha,
            "issued_event_id": issued["event_id"],
        })
        started = self.append("run_started", {
            "run_id": run_id, "permit_id": permit_id,
            "consumed_event_id": consumed["event_id"], "mode": lane, "lane": lane,
            "shape_id": shape_id, "candidate_sha256": self.submission_sha,
            "gate_request_sha256": request_sha,
        })
        stage_refs = []
        embedded = []
        for stage_name, packet in stages:
            digest = self.put_blob(packet)
            self.append("side_stage_ingested", {
                "run_id": run_id, "stage": stage_name,
                "artifact_sha256": digest, "returncode": 0, "timed_out": False,
            })
            stage_refs.append({
                "stage": stage_name,
                "sha256": digest,
                "path": f"Project/authority/blobs/{digest}.json",
            })
            embedded.append(packet)
        side_evidence_sha = stage_refs[-1]["sha256"]
        validation = {"passed": True, "method": "controller output comparison"}
        payload = {
            "entry_id": entry_id, "mode": lane, "lane": lane, "shape_id": shape_id,
            "candidate_sha256": self.submission_sha,
            "side_evidence_sha256": side_evidence_sha,
            "side_stage_artifacts": stage_refs,
            "controller_validation": validation,
            "gate_request_sha256": request_sha,
            "evidence_eligible_pre_audit": True,
            "promotion_eligible": False,
            "permit_id": permit_id,
            "run_id": run_id,
            "started_event_id": started["event_id"],
        }
        measurement = self.append("measurement_recorded", payload)
        wrapper = {
            "schema_version": 1,
            "entry_id": entry_id,
            "candidate_sha256": self.submission_sha,
            "measurement_event_sha256": measurement["event_sha256"],
            "measurement_event_id": measurement["event_id"],
            "lane": lane,
            "mode": lane,
            "shape_id": shape_id,
            "side_evidence_sha256": side_evidence_sha,
            "side_stage_artifacts": stage_refs,
            "side_evidence_packets": embedded,
            "controller_validation": validation,
            "gate_request_sha256": request_sha,
        }
        packet_sha = self.put_blob(wrapper)
        self.append("measurement_packet_bound", {
            "entry_id": entry_id,
            "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "packet_sha256": packet_sha,
            "candidate_sha256": self.submission_sha,
            "side_evidence_sha256": side_evidence_sha,
            "lane": lane,
        })
        self.bind_audit(
            entry_id=entry_id, packet_sha=packet_sha,
            measurement_sha=measurement["event_sha256"], lane=lane,
            integrity=integrity, technical=technical,
        )
        return {
            "kind": "side_controller",
            "entry_id": entry_id,
            "measurement_event_sha256": measurement["event_sha256"],
            "side_evidence_sha256": side_evidence_sha,
            "audit_packet": {
                "path": f"Project/authority/blobs/{packet_sha}.json",
                "sha256": packet_sha,
            },
            "selection_rationale": "fixture selection",
        }

    # -- packet bodies ----------------------------------------------------
    def shape6_packet(self, entry_id, *, allocated=None, declared_flat=True,
                      candidate_sha=None, env=None) -> dict:
        # Every protocol number below is read out of ship_manifest, which
        # reads it out of the controller.  A fixture that restated 20/100/3
        # /300/10 would be yet another writer of the protocol, and the
        # packet tests would then only prove that two hardcodes happen to
        # match -- which is precisely the failure this consolidation exists
        # to remove.  Derived here, a controller protocol change moves the
        # fixture and the code under test together, and the drift tests
        # below are what catch a controller that has actually parted ways.
        shape = self.shapes[6]
        seeds = [1, 2, 3, 4, 5]
        memory_repeats = ship_manifest.SHAPE6_MEMORY_REPEATS
        allocated = (allocated if allocated is not None
                     else [100.0] * memory_repeats)
        reserved = [200.0] * memory_repeats
        samples = [1.0] * ship_manifest.SHAPE6_RAW_SAMPLE_COUNT
        recomputed = {
            "allocated_slope_bytes_per_repeat": ship_manifest.series_slope(allocated),
            "reserved_slope_bytes_per_repeat": ship_manifest.series_slope(reserved),
            "allocated_end_growth_bytes": allocated[-1] - allocated[0],
            "reserved_end_growth_bytes": reserved[-1] - reserved[0],
            "allocated_max_growth_bytes": max(allocated) - allocated[0],
            "reserved_max_growth_bytes": max(reserved) - reserved[0],
        }
        return {
            "type": "shape6_submission_evaluation",
            "schema_version": "shape6-submission-v2",
            "entry_id": entry_id,
            "passed": True,
            "shape": {key: shape[key] for key in (
                "id", "batch_size", "seq_len", "d_model", "num_heads",
                "ffn_dim", "num_layers", "causal")},
            "binding": self.binding("shape6_local_eval.py"),
            "candidate": {
                "path": SUBMISSION_REL,
                "sha256": candidate_sha or self.submission_sha,
            },
            "env": env if env is not None else self.env(),
            "numerical_state": dict(ship_manifest.OFFICIAL_NUMERICAL_STATE),
            "correctness": {
                "passed": True, "violations": 0, "nonfinite_elements": 0,
                "seeds": seeds,
                "trials": [
                    {"seed": seed, "passed": True, "violations": 0,
                     "nonfinite_elements": 0}
                    for seed in seeds
                ],
            },
            "memory": {
                "flat": declared_flat,
                "warmups": ship_manifest.SHAPE6_MEMORY_WARMUPS,
                "repeats": memory_repeats,
                "limits": dict(ship_manifest.SHAPE6_MEMORY_LIMITS),
                "peak_allocated_bytes_per_repeat": allocated,
                "peak_reserved_bytes_per_repeat": reserved,
                "settled_allocated_bytes_per_repeat": allocated,
                "settled_reserved_bytes_per_repeat": reserved,
                **recomputed,
            },
            "timing": {
                "warmups": ship_manifest.CONTROLLER_TIMING["warmup"],
                "repeats_per_round": ship_manifest.CONTROLLER_TIMING["repeats"],
                "round_count": ship_manifest.CONTROLLER_TIMING["rounds"],
                "speedup_vs_baseline": None,
                "raw_samples_ms": samples,
                "rounds": [
                    {"round": index,
                     "samples_ms": samples[
                         index * ship_manifest.CONTROLLER_TIMING["repeats"]:
                         (index + 1) * ship_manifest.CONTROLLER_TIMING["repeats"]
                     ]}
                    for index in range(ship_manifest.CONTROLLER_TIMING["rounds"])
                ],
                "median_ms": 1.0,
            },
        }

    def shape14_stage_packets(self, entry_id, *, slices=None, repeats=None,
                              extra_timing=None) -> list:
        shape = self.shapes[14]
        if repeats is None:
            repeats = ship_manifest.SHAPE14_MIN_TIMING_REPEATS
        slices = slices if slices is not None else shape["batch_size"]
        binding = self.binding("shape14_eval.py")
        validation = {
            "type": "shape14_oracle_validation",
            "schema_version": "shape14-oracle-validation-v2",
            "passed": True,
            "binding": dict(binding),
        }
        decomposition = {
            "type": "shape14_batch_decomposition_check",
            "schema_version": "shape14-decomposition-v2",
            "passed": True,
            "binding": dict(binding),
        }
        validation_sha = sha256_bytes(canonical_blob(validation))
        decomposition_sha = sha256_bytes(canonical_blob(decomposition))
        seeds = [11, 12, 13, 14, 15]
        matrix = [[1.0] * repeats for _ in range(slices)]
        compute = [float(slices)] * repeats
        wall = [float(slices) + 8.0] * repeats
        timing = {
            "protocol": "32 serial B=1 submission calls; CUDA events exclude staging",
            "warmup_slices": ship_manifest.SHAPE14_MIN_WARMUP_SLICES,
            "timing_repeats": repeats,
            "slice_times_ms": {
                "orientation": "batch_index x timing_repeat",
                "values": matrix,
            },
            "gpu_compute_sum_ms_per_repeat": compute,
            "gpu_compute_median_of_sums_ms": statistics.median(compute),
            "staging_inclusive_wall_ms_per_repeat": wall,
            "staging_inclusive_wall_median_ms": statistics.median(wall),
        }
        timing.update(extra_timing or {})
        evaluation = {
            "type": "shape14_side_evaluation",
            "schema_version": "shape14-streamed-v2",
            "entry_id": entry_id,
            "passed": True,
            "shape": {key: shape[key] for key in (
                "id", "batch_size", "seq_len", "d_model", "num_heads",
                "ffn_dim", "num_layers", "causal")},
            "binding": dict(binding),
            "candidate": {"path": SUBMISSION_REL, "sha256": self.submission_sha},
            "env": self.env(),
            "numerical_state": dict(ship_manifest.OFFICIAL_NUMERICAL_STATE),
            "seeds": seeds,
            "required_artifacts": {
                "oracle_validation": {
                    "path": "Project/results_side/shape14_validation.json",
                    "sha256": validation_sha,
                    "schema_version": "shape14-oracle-validation-v2",
                },
                "batch_decomposition": {
                    "path": "Project/results_side/shape14_decomposition.json",
                    "sha256": decomposition_sha,
                    "schema_version": "shape14-decomposition-v2",
                },
            },
            "correctness": {
                "passed": True, "violations": 0, "nonfinite_elements": 0,
                "trials": [
                    {"base_seed": seed, "violations": 0, "nonfinite_elements": 0}
                    for seed in seeds
                ],
            },
            "timing": timing,
        }
        return [
            ("shape14-validate", validation),
            ("shape14-decomposition", decomposition),
            ("shape14-eval", evaluation),
        ]


@contextlib.contextmanager
def fixture_repo():
    with tempfile.TemporaryDirectory() as directory:
        repo = FixtureRepo(Path(directory).resolve())
        project = repo.project
        targets = {
            "ROOT": repo.root,
            "PROJECT": project,
            "TOOLS": project / "tools",
            "HARNESS": project / "harness",
            "JOURNAL": project / "results" / "JOURNAL.jsonl",
            "SIDE": project / "results_side",
            "SUBMISSION": repo.submission_path,
            "OFFICIAL": repo.official_path,
            "OFFICIAL_MANIFEST": repo.official_manifest_path,
            "SHAPES_FILE": project / "shapes.json",
            "MAP_SCHEMA_FILE": project / "tools" / "final_evidence_map.schema.json",
            "DEFAULT_OUTPUT": project / "results_side" / "SHIP_MANIFEST.json",
        }
        with contextlib.ExitStack() as stack:
            for name, value in targets.items():
                stack.enter_context(mock.patch.object(ship_manifest, name, value))
            yield repo


# --------------------------------------------------------------------------
# Existing evaluator / submission coverage
# --------------------------------------------------------------------------

class Shape6Tests(unittest.TestCase):
    def test_requires_five_unique_seeds(self):
        with self.assertRaises(Exception):
            shape6_local_eval.parse_seeds("1,2,3,4")
        with self.assertRaises(Exception):
            shape6_local_eval.parse_seeds("1,2,3,4,4")
        self.assertEqual(
            shape6_local_eval.parse_seeds("1,2,3,4,5"), [1, 2, 3, 4, 5]
        )

    def test_true_linear_trend_and_both_allocator_views_gate(self):
        flat = shape6_local_eval.memory_assessment(
            [100, 101, 100, 101, 100],
            [200, 200, 201, 201, 200],
        )
        self.assertTrue(flat["flat"])
        leaking = shape6_local_eval.memory_assessment(
            [index * 32 * 2**20 for index in range(5)],
            [index * 128 * 2**20 for index in range(5)],
        )
        self.assertFalse(leaking["flat"])
        transient_growth = shape6_local_eval.memory_assessment(
            [0, 256 * 2**20, 0, 0, 0],
            [0, 512 * 2**20, 0, 0, 0],
        )
        self.assertFalse(transient_growth["flat"])

    def test_even_count_uses_real_statistics_median(self):
        self.assertEqual(shape6_local_eval.statistics.median([1, 2, 9, 10]), 5.5)


class Shape14BindingTests(unittest.TestCase):
    def test_artifact_must_pass_and_match_every_binding_field(self):
        binding = {"submission_sha256": "a" * 64}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            side = Path(directory)
            artifact = side / "artifact.json"
            artifact.write_text(json.dumps({
                "type": "wanted",
                "schema_version": "v1",
                "passed": True,
                "binding": binding,
            }))
            with mock.patch.object(shape14_eval, "SIDE_RESULTS", side):
                path, payload = shape14_eval.require_bound_artifact(
                    str(artifact), "wanted", "v1", binding
                )
                self.assertEqual(path, artifact.resolve())
                self.assertTrue(payload["passed"])
                stale = dict(binding, evaluator_sha256="b" * 64)
                with self.assertRaises(SystemExit):
                    shape14_eval.require_bound_artifact(
                        str(artifact), "wanted", "v1", stale
                    )
                payload["passed"] = False
                artifact.write_text(json.dumps(payload))
                with self.assertRaises(SystemExit):
                    shape14_eval.require_bound_artifact(
                        str(artifact), "wanted", "v1", binding
                    )


class SubmissionBuildTests(unittest.TestCase):
    def test_current_submission_exactly_matches_splice_and_proves_ranges(self):
        output = build_submission.OUT.read_bytes()
        self.assertEqual(output, build_submission.expected_submission())
        proof = build_submission.verify_output(output)
        self.assertTrue(proof["verified"])
        self.assertEqual(proof["submission_sha256"], build_submission.sha256(output))

    def test_prefix_suffix_and_region_tampering_each_fail(self):
        output = build_submission.expected_submission()
        prefix, _region, suffix = build_submission.split_designated_region(
            build_submission.OFFICIAL.read_bytes()
        )
        cases = []
        prefix_bad = bytearray(output)
        prefix_bad[0] ^= 1
        cases.append(bytes(prefix_bad))
        suffix_bad = bytearray(output)
        suffix_bad[-1] ^= 1
        cases.append(bytes(suffix_bad))
        region_bad = bytearray(output)
        region_bad[len(prefix) + 10] ^= 1
        cases.append(bytes(region_bad))
        self.assertGreater(len(suffix), 0)
        for case in cases:
            with self.subTest(case=build_submission.sha256(case)):
                with self.assertRaises(build_submission.ProvenanceError):
                    build_submission.verify_output(case)

    def test_builder_refuses_official_bytes_not_pinned_by_manifest(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            official = Path(directory) / "torch_transformer_benchmark.py"
            manifest = Path(directory) / "manifest.json"
            official.write_bytes(build_submission.OFFICIAL.read_bytes() + b"# drift\n")
            manifest.write_text(json.dumps({
                "official_commit": "pinned",
                "files": {official.name: "0" * 64},
            }))
            with mock.patch.object(build_submission, "OFFICIAL", official), mock.patch.object(
                build_submission, "OFFICIAL_MANIFEST", manifest
            ):
                with self.assertRaises(build_submission.ProvenanceError):
                    build_submission.pinned_official_proof()


class SubmissionRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.submission = load_submission()

    def test_cpu_route_remains_exact_baseline(self):
        import torch

        torch.manual_seed(7)
        cfg = self.submission.TransformerConfig(
            batch_size=2, seq_len=8, d_model=16, num_heads=4,
            ffn_dim=16, num_layers=1, causal=True,
        )
        baseline = self.submission.BaselineTransformer(cfg).eval()
        candidate = self.submission.UserOptimizedTransformer(cfg).eval()
        candidate.load_state_dict(baseline.state_dict(), strict=True)
        x = torch.randn(2, 8, 16)
        mask = torch.ones(2, 8, dtype=torch.bool)
        with torch.inference_mode():
            expected = baseline(x, mask)
            actual = candidate(x, mask)
        self.assertTrue(torch.equal(expected, actual))

    def test_weight_recopy_removes_every_derived_cache(self):
        cfg = self.submission.TransformerConfig(
            batch_size=1, seq_len=8, d_model=16, num_heads=4,
            ffn_dim=16, num_layers=1, causal=True,
        )
        candidate = self.submission.UserOptimizedTransformer(cfg)
        candidate.__dict__["_sub_bufs"] = object()
        candidate.__dict__["_sub_graph_state"] = object()
        candidate.__dict__["_sub_route_status"] = object()
        layer = candidate.layers[0]
        layer.__dict__["_sub_fused_cache"] = object()
        layer.__dict__["_sub_fp16_cache"] = object()
        candidate.load_state_dict(candidate.state_dict(), strict=True)
        for name in ("_sub_bufs", "_sub_graph_state", "_sub_route_status"):
            self.assertNotIn(name, candidate.__dict__)
        self.assertNotIn("_sub_fused_cache", layer.__dict__)
        self.assertNotIn("_sub_fp16_cache", layer.__dict__)

    def test_only_triton_resource_failures_are_fallback_eligible(self):
        resource_type = type(
            "OutOfResources", (Exception,), {"__module__": "triton.runtime.errors"}
        )
        self.assertTrue(
            self.submission._sub_allowed_preflight_fallback(
                resource_type("out of resources: shared memory")
            )
        )
        self.assertFalse(
            self.submission._sub_allowed_preflight_fallback(
                RuntimeError("CUDA illegal memory access")
            )
        )
        self.assertEqual(self.submission._GRAPH_TRIGGER_CALLS, 1)
        self.assertGreaterEqual(self.submission._GRAPH_SIDE_STREAM_WARMUPS, 3)

    def test_resource_fallback_batch_chunks_exact_baseline(self):
        import torch

        cfg = self.submission.TransformerConfig(
            batch_size=5, seq_len=4096, d_model=4, num_heads=4,
            ffn_dim=4, num_layers=1, causal=True,
        )
        candidate = self.submission.UserOptimizedTransformer(cfg)
        x = torch.randn(5, 4096, 4)
        batches = []

        def identity_baseline(_self, value, _mask):
            batches.append(value.shape[0])
            return value

        with mock.patch.object(
            self.submission.BaselineTransformer,
            "forward",
            autospec=True,
            side_effect=identity_baseline,
        ):
            output = candidate._sub_controlled_baseline(x)
        self.assertTrue(torch.equal(output, x))
        self.assertEqual(batches, [2, 2, 1])


# --------------------------------------------------------------------------
# Defect 1: no synthesized audit text may reach the audit_verdict field
# --------------------------------------------------------------------------

class NoSynthesizedVerdictTests(unittest.TestCase):
    SOURCE = (TOOLS / "ship_manifest.py").read_text()

    def test_historical_fabricated_sentences_are_gone(self):
        for phrase in (
            "side-evidence (oracle validated vs official dense)",
            "side-evidence (validated vs batch-chunked official baseline)",
            "side-evidence (streamed oracle validated vs pinned official dense)",
            "unaudited",
        ):
            self.assertNotIn(phrase, self.SOURCE, phrase)

    def test_only_audit_verdict_fields_may_produce_a_non_null_verdict(self):
        """No dict literal outside audit_verdict_fields sets audit_verdict."""
        tree = ast.parse(self.SOURCE)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "audit_verdict_fields":
                continue
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "audit_verdict":
                    if not (isinstance(value, ast.Constant) and value.value is None):
                        offenders.append(ast.dump(value))
        allowed = {
            ast.dump(value)
            for function in ast.walk(tree)
            if isinstance(function, ast.FunctionDef)
            and function.name == "audit_verdict_fields"
            for node in ast.walk(function)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "audit_verdict"
        }
        self.assertEqual([item for item in offenders if item not in allowed], [])

    def test_verdict_is_null_without_a_bound_audit_result_event(self):
        fields = ship_manifest.audit_verdict_fields({
            "integrity_status": "LEGACY_UNBOUND",
            "technical_status": "LEGACY_UNBOUND",
            "effective_event_sha256": None,
        })
        self.assertIsNone(fields["audit_verdict"])
        self.assertIsNone(fields["audit_technical_verdict"])
        self.assertIsNone(fields["audit_verdict_event_sha256"])
        self.assertIsNone(fields["audit_verdict_source"])

    def test_verdict_mirrors_the_recorded_event_and_never_prose(self):
        with fixture_repo() as repo:
            entry_id = "run-" + "1" * 32
            selector = repo.controller_measurement(entry_id=entry_id, shape_id=1)
            entry = ship_manifest.build_controller_entry(
                selector, 1, repo.shapes[1], repo.submission_sha
            )
            recorded = [
                json.loads(line)
                for line in repo.events_path.read_text().splitlines()
                if line.strip()
            ]
            result = [row for row in recorded if row["event_type"] == "audit_result"][-1]
            self.assertEqual(entry["audit_verdict"], "PASS")
            self.assertEqual(
                entry["audit_verdict"],
                result["result"]["integrity"]["verdict"],
            )
            self.assertEqual(
                entry["audit_verdict_event_sha256"], result["event_sha256"]
            )
            self.assertEqual(entry["evidence_class"], ship_manifest.POST_LOCK)
            self.assertEqual(entry["evidence_status"], "SHIPPABLE")
            self.assertIn("reference_method", entry)
            self.assertNotIn("side-evidence", entry["reference_method"])

    def test_shape6_reference_method_names_the_chunked_baseline_it_used(self):
        with fixture_repo() as repo:
            entry_id = "20260830-120006-aaaaaa"
            stages = [("shape6-eval", repo.shape6_packet(entry_id))]
            selector = repo.side_measurement(
                entry_id=entry_id, shape_id=6, stage_packets=stages
            )
            entry = ship_manifest.build_side_controller_entry(
                selector, 6, repo.shapes[6], repo.submission_sha
            )
            # The old text claimed a dense full-batch official reference; the
            # real reference is the batch-chunked official computation.
            self.assertIn("batch-chunked", entry["reference_method"])
            self.assertNotIn("dense", entry["reference_method"])
            self.assertEqual(entry["audit_verdict"], "PASS")


# --------------------------------------------------------------------------
# Defect 2: selection comes from the explicit map, never from "fastest ever"
# --------------------------------------------------------------------------

class ExplicitSelectionTests(unittest.TestCase):
    def test_named_slower_entry_wins_over_a_faster_unnamed_one(self):
        with fixture_repo() as repo:
            slow = repo.controller_measurement(
                entry_id="run-" + "a" * 32, shape_id=1, candidate_ms=4.0
            )
            fast = repo.controller_measurement(
                entry_id="run-" + "b" * 32, shape_id=1, candidate_ms=0.25
            )
            entry = ship_manifest.build_controller_entry(
                slow, 1, repo.shapes[1], repo.submission_sha
            )
            self.assertEqual(entry["entry_id"], slow["entry_id"])
            self.assertEqual(entry["median_ms"], 4.0)
            faster = ship_manifest.build_controller_entry(
                fast, 1, repo.shapes[1], repo.submission_sha
            )
            self.assertLess(faster["median_ms"], entry["median_ms"])

    def test_wrong_measurement_hash_refuses(self):
        with fixture_repo() as repo:
            selector = repo.controller_measurement(
                entry_id="run-" + "c" * 32, shape_id=3
            )
            tampered = dict(selector, measurement_event_sha256="d" * 64)
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_controller_entry(
                    tampered, 3, repo.shapes[3], repo.submission_sha
                )
            self.assertIn("absent or duplicated", str(caught.exception))

    def test_wrong_packet_hash_refuses(self):
        with fixture_repo() as repo:
            selector = repo.controller_measurement(
                entry_id="run-" + "e" * 32, shape_id=4
            )
            bogus = "f" * 64
            tampered = dict(selector, audit_packet={
                "path": f"Project/authority/blobs/{bogus}.json",
                "sha256": bogus,
            })
            with self.assertRaises(ship_manifest.ManifestError):
                ship_manifest.build_controller_entry(
                    tampered, 4, repo.shapes[4], repo.submission_sha
                )

    def test_packet_path_must_match_its_own_hash(self):
        with fixture_repo() as repo:
            selector = repo.controller_measurement(
                entry_id="run-" + "0" * 32, shape_id=5
            )
            packet_sha = selector["audit_packet"]["sha256"]
            tampered = dict(selector, audit_packet={
                "path": "Project/authority/blobs/elsewhere.json",
                "sha256": packet_sha,
            })
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_controller_entry(
                    tampered, 5, repo.shapes[5], repo.submission_sha
                )
            self.assertIn("audit packet path must be", str(caught.exception))


# --------------------------------------------------------------------------
# Defect 3: the long-deferred task-06 hard-verdict filter, fail-closed
# --------------------------------------------------------------------------

class HardVerdictFilterTests(unittest.TestCase):
    def _decision_case(self, repo, entry_id, shape_id, **kwargs):
        selector = repo.controller_measurement(
            entry_id=entry_id, shape_id=shape_id, **kwargs
        )
        return selector

    def test_rule_violation_retest_and_needs_context_are_all_ineligible(self):
        for index, verdict in enumerate(
            ("RULE_VIOLATION", "RETEST", "NEEDS_CONTEXT"), start=1
        ):
            with self.subTest(verdict=verdict), fixture_repo() as repo:
                selector = self._decision_case(
                    repo, "run-" + f"{index}" * 32, 2, integrity=verdict
                )
                with self.assertRaises(ship_manifest.ManifestError) as caught:
                    ship_manifest.build_controller_entry(
                        selector, 2, repo.shapes[2], repo.submission_sha
                    )
                self.assertIn(verdict, str(caught.exception))

    def test_blocking_technical_verdicts_are_ineligible(self):
        for index, verdict in enumerate(
            ("WEAK_DIAGNOSIS", "MISSING_EVIDENCE"), start=4
        ):
            with self.subTest(verdict=verdict), fixture_repo() as repo:
                selector = self._decision_case(
                    repo, "run-" + f"{index}" * 32, 2, technical=verdict
                )
                with self.assertRaises(ship_manifest.ManifestError) as caught:
                    ship_manifest.build_controller_entry(
                        selector, 2, repo.shapes[2], repo.submission_sha
                    )
                self.assertIn(verdict, str(caught.exception))

    def test_a_row_with_no_verdict_at_all_is_ineligible(self):
        with fixture_repo() as repo:
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.eligible_audit(
                    "20260830-120000-abcdef", repo.submission_sha, "a" * 64
                )
            self.assertIn("missing_audit_verdict", str(caught.exception))

    def test_legacy_pass_without_nonce_packet_candidate_binding_is_ineligible(self):
        with fixture_repo() as repo:
            legacy = repo.project / "audits" / "verdicts.jsonl"
            legacy.write_text(json.dumps({
                "entry_id": "20260828-023313-6a45e2",
                "verdict": "PASS",
            }) + "\n")
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.eligible_audit(
                    "20260828-023313-6a45e2", repo.submission_sha, "a" * 64
                )
            self.assertIn(
                "legacy_verdict_lacks_nonce_packet_and_candidate_binding",
                str(caught.exception),
            )

    def test_a_hard_verdict_on_the_same_bytes_cannot_be_escaped_by_a_new_entry(self):
        with fixture_repo() as repo:
            repo.controller_measurement(
                entry_id="run-" + "7" * 32, shape_id=2, integrity="RULE_VIOLATION"
            )
            clean = repo.controller_measurement(entry_id="run-" + "8" * 32, shape_id=2)
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_controller_entry(
                    clean, 2, repo.shapes[2], repo.submission_sha
                )
            self.assertIn("unresolved_first_hard_verdict", str(caught.exception))

    def test_eligible_audit_refuses_a_decision_without_a_result_event(self):
        blocked = SimpleNamespace(
            promotion_eligible=False,
            blocking_reasons=("missing_audit_verdict",),
        )
        with mock.patch.object(
            ship_manifest, "bound_audit_decision", return_value=blocked
        ):
            with self.assertRaises(ship_manifest.ManifestError):
                ship_manifest.eligible_audit(
                    "20260830-120000-abcdef", "a" * 64, "b" * 64
                )
        hollow = SimpleNamespace(
            promotion_eligible=True,
            blocking_reasons=(),
            as_dict=lambda: {"effective_event_sha256": None},
        )
        with mock.patch.object(
            ship_manifest, "bound_audit_decision", return_value=hollow
        ):
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.eligible_audit(
                    "20260830-120000-abcdef", "a" * 64, "b" * 64
                )
            self.assertIn("no bound result event", str(caught.exception))


# --------------------------------------------------------------------------
# Defect 4: shape-6 memory flatness is a hard condition
# --------------------------------------------------------------------------

class Shape6MemoryConditionTests(unittest.TestCase):
    def test_declared_flat_false_is_refused(self):
        with fixture_repo() as repo:
            packet = repo.shape6_packet("20260830-120006-aaaaaa", declared_flat=False)
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.require_shape6_protocol(packet)
            self.assertIn("memory.flat", str(caught.exception))
            packet["memory"]["flat"] = True
            ship_manifest.require_shape6_protocol(packet)

    def test_a_lying_flat_true_over_a_rising_series_is_refused(self):
        with fixture_repo() as repo:
            rising = [
                float(index) * 32 * 2**20
                for index in range(ship_manifest.SHAPE6_MEMORY_REPEATS)
            ]
            packet = repo.shape6_packet(
                "20260830-120006-aaaaaa", allocated=rising, declared_flat=True
            )
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.require_shape6_protocol(packet)
            self.assertIn("memory.flat", str(caught.exception))

    def test_recomputed_statistics_must_match_the_retained_series(self):
        with fixture_repo() as repo:
            packet = repo.shape6_packet("20260830-120006-aaaaaa")
            packet["memory"]["allocated_end_growth_bytes"] = 1.0
            with self.assertRaises(ship_manifest.ManifestError):
                ship_manifest.require_shape6_protocol(packet)

    def test_memory_condition_blocks_the_whole_shape6_selection(self):
        with fixture_repo() as repo:
            entry_id = "20260830-120006-aaaaaa"
            rising = [
                float(index) * 32 * 2**20
                for index in range(ship_manifest.SHAPE6_MEMORY_REPEATS)
            ]
            packet = repo.shape6_packet(entry_id, allocated=rising, declared_flat=True)
            selector = repo.side_measurement(
                entry_id=entry_id, shape_id=6,
                stage_packets=[("shape6-eval", packet)],
            )
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_side_controller_entry(
                    selector, 6, repo.shapes[6], repo.submission_sha
                )
            self.assertIn("memory.flat", str(caught.exception))


# --------------------------------------------------------------------------
# Defect 5: one submission SHA, bound into every packet
# --------------------------------------------------------------------------

class SubmissionBindingTests(unittest.TestCase):
    def test_side_packet_measured_from_other_bytes_is_refused(self):
        with fixture_repo() as repo:
            entry_id = "20260830-120006-aaaaaa"
            packet = repo.shape6_packet(entry_id, candidate_sha="9" * 64)
            selector = repo.side_measurement(
                entry_id=entry_id, shape_id=6,
                stage_packets=[("shape6-eval", packet)],
            )
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_side_controller_entry(
                    selector, 6, repo.shapes[6], repo.submission_sha
                )
            self.assertIn("did not pass exact submission", str(caught.exception))

    def test_every_binding_field_is_required(self):
        with fixture_repo() as repo:
            evaluator = repo.project / "tools" / "shape6_local_eval.py"
            packet = {"binding": repo.binding("shape6_local_eval.py")}
            ship_manifest.require_packet_binding(
                packet, repo.submission_sha, evaluator
            )
            for field in (
                "submission_sha256", "evaluator_sha256",
                "official_sha256", "official_manifest_sha256",
            ):
                with self.subTest(field=field):
                    broken = {"binding": dict(packet["binding"], **{field: "0" * 64})}
                    with self.assertRaises(ship_manifest.ManifestError):
                        ship_manifest.require_packet_binding(
                            broken, repo.submission_sha, evaluator
                        )

    def test_environment_must_bind_device_driver_torch_cuda_and_triton(self):
        complete = {
            "gpu": "NVIDIA GeForce RTX 3060 Ti", "driver": "610.57.04",
            "torch": "2.12.0+cu130", "cuda": "13.0", "triton": "3.7.0",
        }
        self.assertEqual(
            ship_manifest.require_env_binding(complete, "packet"), complete
        )
        for key in ship_manifest.REQUIRED_ENV_KEYS:
            with self.subTest(missing=key):
                partial = {name: value for name, value in complete.items() if name != key}
                with self.assertRaises(ship_manifest.ManifestError) as caught:
                    ship_manifest.require_env_binding(partial, "packet")
                self.assertIn(key, str(caught.exception))
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.require_env_binding(dict(complete, driver="unknown"), "p")
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.require_env_binding(None, "p")

    def test_controller_entry_refuses_an_unbound_environment(self):
        with fixture_repo() as repo:
            selector = repo.controller_measurement(
                entry_id="run-" + "6" * 32, shape_id=7,
                environment={"python": "3.14.7", "torch": "2.12.0+cu130",
                             "cuda": "13.0", "gpu": "RTX 3060 Ti"},
            )
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_controller_entry(
                    selector, 7, repo.shapes[7], repo.submission_sha
                )
            self.assertIn("driver", str(caught.exception))
            self.assertIn("triton", str(caught.exception))

    def test_side_entry_refuses_an_unbound_environment(self):
        with fixture_repo() as repo:
            entry_id = "20260830-120006-aaaaaa"
            env = repo.env()
            env.pop("triton")
            packet = repo.shape6_packet(entry_id, env=env)
            selector = repo.side_measurement(
                entry_id=entry_id, shape_id=6,
                stage_packets=[("shape6-eval", packet)],
            )
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_side_controller_entry(
                    selector, 6, repo.shapes[6], repo.submission_sha
                )
            self.assertIn("triton", str(caught.exception))


# --------------------------------------------------------------------------
# The primary lane must bind the official protocol, not just the hashes
# --------------------------------------------------------------------------

class PrimaryProtocolBindingTests(unittest.TestCase):
    """The side lane always enforced official numerics; the primary lane must too."""

    def test_the_fixture_request_matches_the_real_controller_request(self):
        """Guard against the fixture drifting away from the production path.

        ``build_controller_entry`` reads a worker request that only
        ``Project/harness/trusted_controller.py`` ever writes.  If this suite
        invented its own request shape, every controller test below would pass
        against evidence the controller cannot produce.
        """
        sys.path.insert(0, str(HARNESS))
        import trusted_controller

        real = trusted_controller._worker_request(
            mode="optimization", shape_id=1, candidate_sha256="a" * 64
        )
        with fixture_repo() as repo:
            repo.controller_measurement(entry_id="run-" + "7" * 32, shape_id=1)
            payload = [
                event["payload"] for event in repo.store.read_events()
                if event.get("kind") == "measurement_recorded"
            ][-1]
            request = json.loads(
                (repo.blobs / f"{payload['worker_request_sha256']}.json").read_text()
            )
        self.assertEqual(set(request), set(real))
        self.assertEqual(
            request["timing_args"], trusted_controller.TIMING
        )
        self.assertEqual(request["numerical"], trusted_controller.NUMERICAL)
        self.assertEqual(
            ship_manifest.CONTROLLER_TIMING, trusted_controller.TIMING
        )
        self.assertEqual(
            ship_manifest.CONTROLLER_NUMERICAL, trusted_controller.NUMERICAL
        )
        self.assertEqual(
            ship_manifest.CONTROLLER_OFFICIAL_SEEDS, real["seeds"][:5]
        )

    def test_a_calibration_request_can_never_be_selected(self):
        """A calibration run reaches lane "primary" too; only candidates ship."""
        with fixture_repo() as repo:
            selector = repo.controller_measurement(
                entry_id="run-" + "8" * 32, shape_id=1, operation="calibration"
            )
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_controller_entry(
                    selector, 1, repo.shapes[1], repo.submission_sha
                )
            self.assertIn("official candidate measurement", str(caught.exception))

    def test_a_request_pinning_another_shape_table_is_refused(self):
        with fixture_repo() as repo:
            selector = repo.controller_measurement(
                entry_id="run-" + "9" * 32, shape_id=1,
                request_overrides={"shapes_sha256": "b" * 64},
            )
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_controller_entry(
                    selector, 1, repo.shapes[1], repo.submission_sha
                )
            self.assertIn("pinned shape table", str(caught.exception))

    def test_only_the_five_official_seeds_without_secret_seeds_is_refused(self):
        with fixture_repo() as repo:
            selector = repo.controller_measurement(
                entry_id="run-" + "6" * 32, shape_id=1,
                request_overrides={
                    "seeds": list(ship_manifest.CONTROLLER_OFFICIAL_SEEDS)
                },
            )
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_controller_entry(
                    selector, 1, repo.shapes[1], repo.submission_sha
                )
            self.assertIn("seeds", str(caught.exception))

    def test_altered_tolerances_in_the_request_are_refused(self):
        with fixture_repo() as repo:
            loosened = dict(ship_manifest.CONTROLLER_NUMERICAL, rtol=0.5)
            selector = repo.controller_measurement(
                entry_id="run-" + "4" * 32, shape_id=1,
                request_overrides={"numerical": loosened},
            )
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_controller_entry(
                    selector, 1, repo.shapes[1], repo.submission_sha
                )
            self.assertIn("official candidate measurement", str(caught.exception))

    def test_a_measurement_taken_with_tf32_disabled_is_refused(self):
        with fixture_repo() as repo:
            state = dict(
                ship_manifest.CONTROLLER_EFFECTIVE_NUMERICAL_STATE,
                cuda_matmul_allow_tf32=False,
            )
            selector = repo.controller_measurement(
                entry_id="run-" + "3" * 32, shape_id=1,
                payload_overrides={"effective_numerical_state": state},
            )
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_controller_entry(
                    selector, 1, repo.shapes[1], repo.submission_sha
                )
            self.assertIn("numerical state is not official", str(caught.exception))

    def test_a_measurement_that_reports_no_numerical_state_is_refused(self):
        with fixture_repo() as repo:
            selector = repo.controller_measurement(
                entry_id="run-" + "2" * 32, shape_id=1,
                payload_overrides={"effective_numerical_state": None},
            )
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.build_controller_entry(
                    selector, 1, repo.shapes[1], repo.submission_sha
                )
            self.assertIn("numerical state is not official", str(caught.exception))


# --------------------------------------------------------------------------
# One timing protocol, one writer
#
# The benchmark timing protocol (warmup / repeats / rounds) was written down
# independently in five places.  Two of them disagreeing is what wedged the
# gate: a campaign bound to one protocol while the controller stamped another
# into every measurement, so no calibration could reconcile and no experiment
# could be planned.  The consolidation makes trusted_controller.TIMING the
# only writer, and every other site derive from it.
#
# These tests exist to fail when that consolidation regresses.  They check
# three separate things, because "the numbers happen to match today" is
# exactly the assurance that failed last time:
#   1. every consumer agrees with the controller RIGHT NOW;
#   2. no consumer restates a protocol number as a bare literal, so agreement
#      cannot be an accident of two hardcodes lining up;
#   3. a controller whose own shape-6/shape-14 copies part ways with its
#      published TIMING is REFUSED, loudly, on the shipping path.
# --------------------------------------------------------------------------

CONTROLLER_SOURCE_TEXT = ship_manifest.CONTROLLER_SOURCE.read_text(encoding="utf-8")


@contextlib.contextmanager
def drifted_controller(old: str, new: str):
    """Run the body against an in-memory controller with one literal changed.

    The file on disk is never touched -- it is write-denied, and the question
    under test is what ship_manifest does when it READS a controller that
    disagrees with itself.  ``old`` is built from the live protocol by every
    caller, so a legitimate protocol change moves these mutations with it
    instead of quietly making them no-ops.
    """
    assert CONTROLLER_SOURCE_TEXT.count(old) == 1, \
        f"mutation anchor is not unique in the controller: {old!r}"
    tree = ast.parse(CONTROLLER_SOURCE_TEXT.replace(old, new))
    with mock.patch.object(ship_manifest, "_CONTROLLER_AST", tree):
        yield


def function_int_literals(path: Path, names: set[str]) -> dict[str, list[int]]:
    """Bare integer literals inside the named module-level functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, list[int]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name in names:
            found[node.name] = sorted({
                child.value for child in ast.walk(node)
                if isinstance(child, ast.Constant) and type(child.value) is int
            })
    assert set(found) == names, f"missing functions: {sorted(names - set(found))}"
    return found


def attribute_floors(path: Path, function: str, root: str) -> dict[str, int]:
    """``root.attr < N`` floors enforced inside one module-level function."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    floors: dict[str, int] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function):
            continue
        for child in ast.walk(node):
            if (isinstance(child, ast.Compare) and len(child.ops) == 1
                    and isinstance(child.ops[0], ast.Lt)
                    and isinstance(child.left, ast.Attribute)
                    and isinstance(child.left.value, ast.Name)
                    and child.left.value.id == root
                    and isinstance(child.comparators[0], ast.Constant)
                    and type(child.comparators[0].value) is int):
                floors[child.left.attr] = child.comparators[0].value
    return floors


def argparse_defaults(path: Path, flags: set[str]) -> dict[str, int]:
    """``add_argument("--flag", ..., default=N)`` values for the named flags."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defaults: dict[str, int] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value in flags):
            continue
        for keyword in node.keywords:
            if (keyword.arg == "default" and isinstance(keyword.value, ast.Constant)
                    and type(keyword.value.value) is int):
                defaults[first.value] = keyword.value.value
    return defaults


class TimingProtocolConsolidationTests(unittest.TestCase):
    """The timing protocol has one writer; everything else derives from it."""

    # -- 1. every writer agrees today ---------------------------------------

    def test_every_writer_of_the_timing_protocol_agrees(self):
        import trusted_controller

        protocol = trusted_controller.TIMING
        with self.subTest(writer="ship_manifest.CONTROLLER_TIMING"):
            self.assertEqual(ship_manifest.CONTROLLER_TIMING, protocol)
        with self.subTest(writer="run_gate.controller_timing_protocol"):
            self.assertEqual(run_gate.controller_timing_protocol()[0], protocol)
        with self.subTest(writer="shape6_local_eval OFFICIAL_*"):
            self.assertEqual(
                (shape6_local_eval.OFFICIAL_WARMUPS,
                 shape6_local_eval.OFFICIAL_REPEATS,
                 shape6_local_eval.OFFICIAL_ROUNDS),
                (protocol["warmup"], protocol["repeats"], protocol["rounds"]),
            )
        with self.subTest(writer="trusted_controller shape-6 literals"):
            self.assertEqual(
                ship_manifest._controller_shape6_literals()["get"],
                {
                    "timing.warmups": protocol["warmup"],
                    "timing.repeats_per_round": protocol["repeats"],
                    "timing.round_count": protocol["rounds"],
                    "memory.repeats": ship_manifest.SHAPE6_MEMORY_REPEATS,
                },
            )
        with self.subTest(writer="derived sample count"):
            # 300 is not a number, it is repeats x rounds.
            self.assertEqual(
                ship_manifest.SHAPE6_RAW_SAMPLE_COUNT,
                protocol["repeats"] * protocol["rounds"],
            )
        ship_manifest.require_controller_protocol_agreement()

    def test_the_memory_protocol_is_pinned_to_the_producer_that_authors_it(self):
        """3/10 is NOT derivable from the timing protocol -- so pin its source.

        The shape-6 memory trend runs a separate, coarser protocol.  Its only
        author is shape6_local_eval.py; the controller re-pins the repeat count
        and nothing at all pins the warmup count.  ship_manifest names both as
        constants so they are searchable, and this test is the only thing that
        holds them to their origin.
        """
        self.assertEqual(
            (ship_manifest.SHAPE6_MEMORY_WARMUPS,
             ship_manifest.SHAPE6_MEMORY_REPEATS),
            (shape6_local_eval.MEMORY_WARMUPS, shape6_local_eval.MEMORY_REPEATS),
        )
        self.assertEqual(
            ship_manifest._controller_shape6_literals()["get"]["memory.repeats"],
            ship_manifest.SHAPE6_MEMORY_REPEATS,
        )
        # The byte thresholds ride along with the same protocol and are the
        # one remaining copy in this area, so pin them to their author too.
        self.assertEqual(
            ship_manifest.SHAPE6_MEMORY_LIMITS, shape6_local_eval.MEMORY_LIMITS
        )

    def test_the_shape14_floors_are_pinned_to_every_writer(self):
        """Shape 14 is not measured under the CUDA-event protocol at all."""
        floors = attribute_floors(TOOLS / "shape14_eval.py", "cmd_eval", "args")
        self.assertEqual(
            floors.get("timing_repeats"), ship_manifest.SHAPE14_MIN_TIMING_REPEATS
        )
        self.assertEqual(
            floors.get("warmup"), ship_manifest.SHAPE14_MIN_WARMUP_SLICES
        )
        defaults = argparse_defaults(
            TOOLS / "shape14_eval.py", {"--timing-repeats", "--warmup"}
        )
        self.assertGreaterEqual(
            defaults["--timing-repeats"], ship_manifest.SHAPE14_MIN_TIMING_REPEATS
        )
        self.assertGreaterEqual(
            defaults["--warmup"], ship_manifest.SHAPE14_MIN_WARMUP_SLICES
        )
        controller = ship_manifest._controller_shape14_literals()
        self.assertEqual(
            controller["floors"].get("repeat_count"),
            ship_manifest.SHAPE14_MIN_TIMING_REPEATS,
        )
        self.assertGreaterEqual(
            int(controller["stage_args"]["--timing-repeats"]),
            ship_manifest.SHAPE14_MIN_TIMING_REPEATS,
        )
        self.assertGreaterEqual(
            int(controller["stage_args"]["--warmup"]),
            ship_manifest.SHAPE14_MIN_WARMUP_SLICES,
        )
        ship_manifest.require_controller_shape14_agreement()

    # -- 2. no consumer restates a protocol number --------------------------

    def test_no_protocol_number_is_written_down_again_in_the_shipping_path(self):
        """Agreement must be structural, not a coincidence of two hardcodes."""
        protocol = ship_manifest.CONTROLLER_TIMING
        forbidden = {
            protocol["warmup"],
            protocol["repeats"],
            protocol["rounds"],
            ship_manifest.SHAPE6_RAW_SAMPLE_COUNT,
            ship_manifest.SHAPE6_MEMORY_WARMUPS,
            ship_manifest.SHAPE6_MEMORY_REPEATS,
            ship_manifest.SHAPE14_MIN_TIMING_REPEATS,
            ship_manifest.SHAPE14_MIN_WARMUP_SLICES,
        }
        consumers = {
            "require_shape6_protocol", "require_shape14_protocol",
            "build_controller_entry", "build_side_controller_entry",
            "validate_timing_samples",
        }
        literals = function_int_literals(TOOLS / "ship_manifest.py", consumers)
        for name, values in sorted(literals.items()):
            with self.subTest(consumer=name):
                self.assertEqual(
                    sorted(forbidden.intersection(values)), [],
                    f"{name} restates a protocol number as a bare literal; "
                    "derive it from CONTROLLER_TIMING instead",
                )
        # The residue is deliberately pinned: 0 (violation/nonfinite counts),
        # 1 (indexing and single-match counts) and 5 (the minimum correctness
        # seed count, a correctness rule and not a timing one).  A new bare
        # integer in either protocol validator has to be justified here.
        self.assertEqual(literals["require_shape6_protocol"], [0, 1, 5])
        self.assertEqual(literals["require_shape14_protocol"], [0, 1, 5])

    def test_the_derived_constants_are_derived_and_not_transcribed(self):
        """A literal that happens to equal the protocol is still a second writer.

        Value equality cannot tell "read from the controller" apart from
        "typed in again and correct today", and the second one is what wedged
        the gate.  So this asserts the SHAPE of the assignments: the protocol
        is a call, and the sample count is arithmetic on it.
        """
        tree = ast.parse((TOOLS / "ship_manifest.py").read_text(encoding="utf-8"))
        assigned = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned[target.id] = node.value
        self.assertIsInstance(
            assigned["CONTROLLER_TIMING"], ast.Call,
            "CONTROLLER_TIMING must be read from the controller, not written here",
        )
        product = assigned["SHAPE6_RAW_SAMPLE_COUNT"]
        self.assertIsInstance(
            product, ast.BinOp,
            "the raw sample count must be repeats x rounds, not the number 300",
        )
        self.assertIsInstance(product.op, ast.Mult)
        subscripted = {
            node.value.id
            for node in ast.walk(product)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        }
        self.assertEqual(subscripted, {"CONTROLLER_TIMING"})
        # The three that are NOT derivable are declared as plain constants on
        # purpose -- searchable, and documented where they come from -- and the
        # tests above pin each of them to its actual author.
        for name in ("SHAPE6_MEMORY_WARMUPS", "SHAPE6_MEMORY_REPEATS",
                     "SHAPE14_MIN_TIMING_REPEATS", "SHAPE14_MIN_WARMUP_SLICES"):
            with self.subTest(constant=name):
                self.assertIsInstance(assigned[name], ast.Constant)

    # -- 3. a self-contradicting controller is refused, loudly --------------

    def controller_drifts(self):
        protocol = ship_manifest.CONTROLLER_TIMING
        warmup = protocol["warmup"]
        repeats = protocol["repeats"]
        rounds = protocol["rounds"]
        raw = ship_manifest.SHAPE6_RAW_SAMPLE_COUNT
        memory = ship_manifest.SHAPE6_MEMORY_REPEATS
        floor = ship_manifest.SHAPE14_MIN_TIMING_REPEATS
        warmup_slices = ship_manifest.SHAPE14_MIN_WARMUP_SLICES
        return (
            ("shape-6 warmups",
             f'timing.get("warmups") != {warmup}',
             f'timing.get("warmups") != {warmup + 7}',
             f"timing.warmups == {warmup + 7}"),
            ("shape-6 repeats per round",
             f'timing.get("repeats_per_round") != {repeats}',
             f'timing.get("repeats_per_round") != {repeats + 7}',
             f"timing.repeats_per_round == {repeats + 7}"),
            ("shape-6 round count",
             f'timing.get("round_count") != {rounds}',
             f'timing.get("round_count") != {rounds + 7}',
             f"timing.round_count == {rounds + 7}"),
            ("shape-6 retained raw samples",
             f"len(samples_value) != {raw}",
             f"len(samples_value) != {raw + 7}",
             "retained-series lengths"),
            ("shape-6 memory repeats",
             f'memory.get("repeats") != {memory}',
             f'memory.get("repeats") != {memory + 7}',
             f"memory.repeats == {memory + 7}"),
            ("shape-6 round slicing",
             f"samples[index * {repeats}:(index + 1) * {repeats}]",
             f"samples[index * {repeats + 7}:(index + 1) * {repeats + 7}]",
             "cuts the shape-6 raw samples into rounds"),
            ("shape-14 repeat floor",
             f"or repeat_count < {floor}\n",
             f"or repeat_count < {floor + 7}\n",
             f"accepts shape-14 evidence at {floor + 7} timing repeats"),
            ("shape-14 commissioned warmup",
             f'"--warmup", "{warmup_slices}",',
             '"--warmup", "1",',
             "commissions the shape-14 evaluator with --warmup 1"),
        )

    def test_a_controller_that_drifts_from_its_own_protocol_is_refused(self):
        for label, old, new, fragment in self.controller_drifts():
            with self.subTest(drift=label), drifted_controller(old, new):
                with self.assertRaises(ship_manifest.ManifestError) as caught:
                    ship_manifest.require_controller_protocol_agreement()
                message = str(caught.exception)
                self.assertIn("split-brain", message)
                self.assertIn(fragment, message)
                self.assertIn("do not silence this".lower(), message.lower())

    def test_a_controller_that_stops_pinning_the_protocol_fails_closed(self):
        """Absence is not agreement: an unreadable pin refuses, never assumes."""
        protocol = ship_manifest.CONTROLLER_TIMING
        cases = (
            ("shape-6 warmups check deleted",
             f'timing.get("warmups") != {protocol["warmup"]}\n        or ',
             "",
             "no longer pins timing.warmups in validate_shape6_packet()"),
            ("shape-6 validator renamed",
             "def validate_shape6_packet(",
             "def _retired_validate_shape6_packet(",
             "no longer pins a module-level validate_shape6_packet()"),
            ("shape-14 validator renamed",
             "def validate_shape14_packets(",
             "def _retired_validate_shape14_packets(",
             "no longer pins a module-level validate_shape14_packets()"),
            ("shape-14 stage arguments gone",
             '"--timing-repeats", "'
             f'{ship_manifest.SHAPE14_MIN_TIMING_REPEATS}",\n',
             "",
             "exactly one shape-14 evaluator argument vector (0 found)"),
        )
        for label, old, new, fragment in cases:
            with self.subTest(missing=label), drifted_controller(old, new):
                with self.assertRaises(ship_manifest.ManifestError) as caught:
                    ship_manifest.require_controller_protocol_agreement()
                self.assertIn("unprovable", str(caught.exception))
                self.assertIn(fragment, str(caught.exception))

    def test_a_controller_that_cannot_be_read_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "trusted_controller.py"
            broken.write_text("def (:\n", encoding="utf-8")
            missing = Path(directory) / "absent_controller.py"
            for label, path in (("unparseable", broken), ("absent", missing)):
                with self.subTest(controller=label), \
                        mock.patch.object(ship_manifest, "CONTROLLER_SOURCE", path), \
                        mock.patch.object(ship_manifest, "_CONTROLLER_AST", None):
                    with self.assertRaises(ship_manifest.ManifestError) as caught:
                        ship_manifest.require_controller_protocol_agreement()
                    self.assertIn(
                        "cannot read the controller timing protocol",
                        str(caught.exception),
                    )

    # -- the refusal reaches the shipping path, not just the helper ---------

    def test_controller_drift_refuses_shape6_evidence_that_is_otherwise_perfect(self):
        protocol = ship_manifest.CONTROLLER_TIMING
        with fixture_repo() as repo:
            packet = repo.shape6_packet("20260830-120006-aaaaaa")
            ship_manifest.require_shape6_protocol(packet)
            with drifted_controller(
                f'timing.get("repeats_per_round") != {protocol["repeats"]}',
                f'timing.get("repeats_per_round") != {protocol["repeats"] + 7}',
            ):
                with self.assertRaises(ship_manifest.ManifestError) as caught:
                    ship_manifest.require_shape6_protocol(packet)
                self.assertIn("split-brain timing protocol", str(caught.exception))

    def test_controller_drift_refuses_a_whole_shape6_selection(self):
        protocol = ship_manifest.CONTROLLER_TIMING
        with fixture_repo() as repo:
            entry_id = "20260830-120006-aaaaaa"
            selector = repo.side_measurement(
                entry_id=entry_id, shape_id=6,
                stage_packets=[("shape6-eval", repo.shape6_packet(entry_id))],
            )
            ship_manifest.build_side_controller_entry(
                selector, 6, repo.shapes[6], repo.submission_sha
            )
            with drifted_controller(
                f'timing.get("warmups") != {protocol["warmup"]}',
                f'timing.get("warmups") != {protocol["warmup"] + 7}',
            ):
                with self.assertRaises(ship_manifest.ManifestError) as caught:
                    ship_manifest.build_side_controller_entry(
                        selector, 6, repo.shapes[6], repo.submission_sha
                    )
                self.assertIn("split-brain timing protocol", str(caught.exception))

    def test_controller_drift_refuses_shape14_evidence(self):
        floor = ship_manifest.SHAPE14_MIN_TIMING_REPEATS
        with fixture_repo() as repo:
            entry_id = "20260830-120014-bbbbbb"
            packet = repo.shape14_stage_packets(entry_id)[-1][1]
            ship_manifest.require_shape14_protocol(packet, repo.shapes[14])
            with drifted_controller(
                f"or repeat_count < {floor}\n", f"or repeat_count < {floor + 7}\n"
            ):
                with self.assertRaises(ship_manifest.ManifestError) as caught:
                    ship_manifest.require_shape14_protocol(packet, repo.shapes[14])
                self.assertIn("split-brain shape-14 floor", str(caught.exception))

    # -- evidence that misses the protocol is refused ------------------------

    def test_shape6_evidence_must_match_the_derived_protocol_exactly(self):
        protocol = ship_manifest.CONTROLLER_TIMING
        raw = ship_manifest.SHAPE6_RAW_SAMPLE_COUNT
        memory_repeats = ship_manifest.SHAPE6_MEMORY_REPEATS

        def off(field, value):
            def mutate(packet):
                packet["timing"][field] = value
            return mutate

        def memory_off(field, value):
            def mutate(packet):
                packet["memory"][field] = value
            return mutate

        mutations = (
            ("timing warmups", off("warmups", protocol["warmup"] + 1),
             "timing protocol is inconsistent"),
            ("timing repeats per round",
             off("repeats_per_round", protocol["repeats"] + 1),
             "timing protocol is inconsistent"),
            ("timing round count", off("round_count", protocol["rounds"] + 1),
             "timing protocol is inconsistent"),
            ("one raw sample short",
             lambda packet: packet["timing"].__setitem__(
                 "raw_samples_ms", packet["timing"]["raw_samples_ms"][:-1]),
             f"must contain exactly {raw} samples"),
            ("one round short",
             lambda packet: packet["timing"].__setitem__(
                 "rounds", packet["timing"]["rounds"][:-1]),
             "timing rounds are malformed"),
            ("a short round",
             lambda packet: packet["timing"]["rounds"][0].__setitem__(
                 "samples_ms", packet["timing"]["rounds"][0]["samples_ms"][:-1]),
             f"must contain exactly {protocol['repeats']} samples"),
            ("memory warmups",
             memory_off("warmups", ship_manifest.SHAPE6_MEMORY_WARMUPS + 1),
             f"{ship_manifest.SHAPE6_MEMORY_WARMUPS}-warmup"),
            ("memory repeats", memory_off("repeats", memory_repeats + 1),
             f"{memory_repeats}-repeat protocol"),
            ("a short memory series",
             lambda packet: packet["memory"].__setitem__(
                 "peak_allocated_bytes_per_repeat",
                 packet["memory"]["peak_allocated_bytes_per_repeat"][:-1]),
             f"must contain exactly {memory_repeats} samples"),
        )
        with fixture_repo() as repo:
            for label, mutate, fragment in mutations:
                with self.subTest(evidence=label):
                    packet = repo.shape6_packet("20260830-120006-aaaaaa")
                    mutate(packet)
                    with self.assertRaises(ship_manifest.ManifestError) as caught:
                        ship_manifest.require_shape6_protocol(packet)
                    self.assertIn(fragment, str(caught.exception))

    def test_shape14_evidence_below_the_floor_is_refused(self):
        floor = ship_manifest.SHAPE14_MIN_TIMING_REPEATS
        warmups = ship_manifest.SHAPE14_MIN_WARMUP_SLICES
        with fixture_repo() as repo:
            entry_id = "20260830-120014-bbbbbb"
            at_floor = repo.shape14_stage_packets(entry_id, repeats=floor)[-1][1]
            ship_manifest.require_shape14_protocol(at_floor, repo.shapes[14])
            below = repo.shape14_stage_packets(entry_id, repeats=floor - 1)[-1][1]
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.require_shape14_protocol(below, repo.shapes[14])
            self.assertIn(f"{floor} timing repeats", str(caught.exception))
            short_warmup = repo.shape14_stage_packets(
                entry_id, extra_timing={"warmup_slices": warmups - 1}
            )[-1][1]
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.require_shape14_protocol(short_warmup, repo.shapes[14])
            self.assertIn(f"{warmups} warmup", str(caught.exception))


# --------------------------------------------------------------------------
# Shape 14: never compare incomparable batch sizes
# --------------------------------------------------------------------------

class Shape14ComparabilityTests(unittest.TestCase):
    def test_full_streamed_selection_binds_the_official_batch(self):
        with fixture_repo() as repo:
            entry_id = "20260830-120014-bbbbbb"
            selector = repo.side_measurement(
                entry_id=entry_id, shape_id=14,
                stage_packets=repo.shape14_stage_packets(entry_id),
            )
            entry = ship_manifest.build_side_controller_entry(
                selector, 14, repo.shapes[14], repo.submission_sha
            )
            self.assertEqual(entry["evidence_class"], ship_manifest.POST_LOCK)
            self.assertEqual(entry["audit_verdict"], "PASS")
            self.assertIn("32 serial B=1 slices", entry["reference_method"])
            self.assertEqual(entry["median_ms"], float(repo.shapes[14]["batch_size"]))

    def test_a_b1_or_b2_slice_matrix_is_refused(self):
        with fixture_repo() as repo:
            entry_id = "20260830-120014-bbbbbb"
            for slices in (1, 2, 31, 33):
                with self.subTest(slices=slices):
                    packet = repo.shape14_stage_packets(entry_id, slices=slices)[-1][1]
                    with self.assertRaises(ship_manifest.ManifestError) as caught:
                        ship_manifest.require_shape14_protocol(
                            packet, repo.shapes[14]
                        )
                    self.assertIn("32", str(caught.exception))

    def test_a_packet_measured_at_another_batch_size_is_refused(self):
        with fixture_repo() as repo:
            entry_id = "20260830-120014-bbbbbb"
            packet = repo.shape14_stage_packets(entry_id)[-1][1]
            packet["shape"]["batch_size"] = 1
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.require_shape14_protocol(packet, repo.shapes[14])
            self.assertIn("not comparable", str(caught.exception))

    def test_any_baseline_speedup_on_shape14_is_refused(self):
        with fixture_repo() as repo:
            entry_id = "20260830-120014-bbbbbb"
            for field in ("speedup", "speedup_vs_baseline", "baseline_median_ms"):
                with self.subTest(field=field):
                    packet = repo.shape14_stage_packets(
                        entry_id, extra_timing={field: 2.0}
                    )[-1][1]
                    with self.assertRaises(ship_manifest.ManifestError) as caught:
                        ship_manifest.require_shape14_protocol(
                            packet, repo.shapes[14]
                        )
                    self.assertIn("no baseline comparison", str(caught.exception))

    def test_the_decomposition_must_be_declared_in_the_protocol_string(self):
        with fixture_repo() as repo:
            entry_id = "20260830-120014-bbbbbb"
            packet = repo.shape14_stage_packets(entry_id)[-1][1]
            packet["timing"]["protocol"] = "one call at B=32"
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.require_shape14_protocol(packet, repo.shapes[14])
            self.assertIn("serial B=1 decomposition", str(caught.exception))

    # stage index in shape14_stage_packets -> required_artifacts key
    DEPENDENCY_STAGES = ((0, "oracle_validation"), (1, "batch_decomposition"))

    def test_validation_and_decomposition_artifacts_must_pass(self):
        """A failed prerequisite artifact must refuse *because it failed*.

        Each case needs its own fixture repository: the audit authority
        correctly refuses to re-queue one entry id under different bindings,
        so reusing one ledger across cases would test the authority rather
        than the manifest.  The reference hash is recomputed after the flip,
        otherwise this would only re-prove the hash-mismatch rule below.
        """
        for stage_index, key in self.DEPENDENCY_STAGES:
            with self.subTest(stage=key), fixture_repo() as repo:
                entry_id = "20260830-120014-bbbbbb"
                stages = repo.shape14_stage_packets(entry_id)
                stages[stage_index][1]["passed"] = False
                evaluation = stages[-1][1]
                evaluation["required_artifacts"][key]["sha256"] = sha256_bytes(
                    canonical_blob(stages[stage_index][1])
                )
                selector = repo.side_measurement(
                    entry_id=entry_id, shape_id=14, stage_packets=stages
                )
                with self.assertRaises(ship_manifest.ManifestError) as caught:
                    ship_manifest.build_side_controller_entry(
                        selector, 14, repo.shapes[14], repo.submission_sha
                    )
                self.assertIn(f"shape 14 {key} artifact did not pass",
                              str(caught.exception))

    def test_a_dependency_artifact_that_changed_after_hashing_is_refused(self):
        """The evaluation packet's pinned dependency hashes are load-bearing."""
        for stage_index, key in self.DEPENDENCY_STAGES:
            with self.subTest(stage=key), fixture_repo() as repo:
                entry_id = "20260830-120014-bbbbbb"
                stages = repo.shape14_stage_packets(entry_id)
                # Body changes, pinned required_artifacts hash does not.
                stages[stage_index][1]["passed"] = False
                selector = repo.side_measurement(
                    entry_id=entry_id, shape_id=14, stage_packets=stages
                )
                with self.assertRaises(ship_manifest.ManifestError) as caught:
                    ship_manifest.build_side_controller_entry(
                        selector, 14, repo.shapes[14], repo.submission_sha
                    )
                self.assertIn(f"shape 14 {key} artifact hash mismatch",
                              str(caught.exception))


# --------------------------------------------------------------------------
# The evidence map schema must express exactly what the manifest consumes
# --------------------------------------------------------------------------

class EvidenceMapSchemaTests(unittest.TestCase):
    SCHEMA = json.loads((TOOLS / "final_evidence_map.schema.json").read_text())

    def branches(self):
        return {
            branch["properties"]["kind"]["const"]: branch
            for branch in self.SCHEMA["$defs"]["selector"]["oneOf"]
        }

    def test_schema_expresses_exactly_the_consumed_selector_fields(self):
        branches = self.branches()
        self.assertEqual(set(branches), set(ship_manifest.SELECTOR_FIELDS))
        for kind, branch in branches.items():
            with self.subTest(kind=kind):
                self.assertEqual(
                    set(branch["properties"]), set(ship_manifest.SELECTOR_FIELDS[kind])
                )
                self.assertFalse(branch["additionalProperties"])
                self.assertTrue(set(branch["required"]) <= set(branch["properties"]))

    def test_every_selection_must_state_a_rationale(self):
        for kind, branch in self.branches().items():
            with self.subTest(kind=kind):
                self.assertIn("selection_rationale", branch["required"])

    def test_all_fourteen_official_shapes_are_required(self):
        official = {
            str(shape["id"])
            for shape in json.loads((ROOT / "Project/shapes.json").read_text())["shapes"]
        }
        self.assertEqual(set(self.SCHEMA["properties"]["shapes"]["required"]), official)
        self.assertEqual(
            set(self.SCHEMA["properties"]["shapes"]["properties"]), official
        )

    def test_map_must_bind_submission_and_official_hashes(self):
        expected = {shape_id: {"id": shape_id} for shape_id in range(1, 15)}
        payload = self.valid_payload()
        ship_manifest.validate_evidence_map(payload, "a" * 64, "b" * 64, expected)
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.validate_evidence_map(payload, "c" * 64, "b" * 64, expected)
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.validate_evidence_map(payload, "a" * 64, "c" * 64, expected)

    def valid_payload(self):
        audit_sha = "c" * 64
        controller = {
            "kind": "controller",
            "entry_id": "run-" + "d" * 32,
            "measurement_event_sha256": "e" * 64,
            "audit_packet": {
                "path": f"Project/authority/blobs/{audit_sha}.json",
                "sha256": audit_sha,
            },
            "selection_rationale": "named by the owner",
        }
        payload = {
            "schema_version": ship_manifest.MAP_SCHEMA,
            "submission_sha256": "a" * 64,
            "official_sha256": "b" * 64,
            "shapes": {str(shape_id): dict(controller) for shape_id in range(1, 15)},
        }
        for shape_id in (6, 14):
            payload["shapes"][str(shape_id)] = {
                "kind": "side_controller",
                "entry_id": f"20260830-1200{shape_id:02d}-abcdef",
                "measurement_event_sha256": "f" * 64,
                "side_evidence_sha256": "1" * 64,
                "audit_packet": {
                    "path": f"Project/authority/blobs/{audit_sha}.json",
                    "sha256": audit_sha,
                },
                "selection_rationale": "named by the owner",
            }
        return payload

    def test_missing_rationale_is_refused(self):
        expected = {shape_id: {"id": shape_id} for shape_id in range(1, 15)}
        payload = self.valid_payload()
        payload["shapes"]["1"].pop("selection_rationale")
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.validate_evidence_map(payload, "a" * 64, "b" * 64, expected)

    def test_map_must_select_every_official_shape(self):
        expected = {1: {"id": 1}, 2: {"id": 2}}
        payload = {
            "schema_version": ship_manifest.MAP_SCHEMA,
            "submission_sha256": "a" * 64,
            "official_sha256": "b" * 64,
            "shapes": {"1": {"kind": "journal", "entry_id": "x"}},
        }
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.validate_evidence_map(payload, "a" * 64, "b" * 64, expected)

    def test_shapes_6_and_14_demand_dedicated_side_evidence(self):
        expected = {shape_id: {"id": shape_id} for shape_id in range(1, 15)}
        payload = self.valid_payload()
        payload["shapes"]["6"] = dict(payload["shapes"]["1"])
        with self.assertRaises(ship_manifest.ManifestError) as caught:
            ship_manifest.validate_evidence_map(payload, "a" * 64, "b" * 64, expected)
        self.assertIn("dedicated side-controller", str(caught.exception))

    def test_loose_side_packet_selector_is_rejected(self):
        expected = {shape_id: {"id": shape_id} for shape_id in range(1, 15)}
        payload = self.valid_payload()
        payload["shapes"]["6"] = {
            "kind": "side_packet",
            "path": "Project/results_side/loose.json",
        }
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.validate_evidence_map(payload, "a" * 64, "b" * 64, expected)

    def test_raw_median_validation(self):
        samples = [float(index + 1) for index in range(300)]
        timing = {"raw_samples_ms": samples, "median_ms": 150.5, "n_samples": 300}
        self.assertEqual(
            ship_manifest.validate_timing_samples(timing, 300, "candidate"), 150.5
        )
        for broken in (
            {**timing, "median_ms": 150.0},
            {**timing, "raw_samples_ms": samples[:-1]},
            {**timing, "n_samples": 299},
            {**timing, "raw_samples_ms": [0.0] + samples[1:]},
        ):
            with self.subTest(broken=sorted(broken)):
                with self.assertRaises(ship_manifest.ManifestError):
                    ship_manifest.validate_timing_samples(broken, 300, "candidate")


# --------------------------------------------------------------------------
# Legacy evidence is refused, per shape, with an honest reason
# --------------------------------------------------------------------------

class LegacyRefusalTests(unittest.TestCase):
    def journal_row(self, repo, entry_id, shape_id, *, impl_path):
        shape = repo.shapes[shape_id]
        return {
            "type": "candidate",
            "entry_id": entry_id,
            "shape_id": shape_id,
            "promoted": True,
            "shape": {key: shape[key] for key in (
                "id", "batch_size", "seq_len", "d_model", "num_heads",
                "ffn_dim", "num_layers", "causal")},
            "impl": {"name": "k004", "path": impl_path, "sha256": "3" * 64},
            "correctness": {"passed": True},
            "timing": {"candidate": {"median_ms": 0.5}, "speedup": 10.9},
            "env": repo.env(),
        }

    def test_journal_selection_is_structurally_not_shippable(self):
        with fixture_repo() as repo:
            entry_id = "20260828-023313-6a45e2"
            rows = {entry_id: self.journal_row(
                repo, entry_id, 1, impl_path="Project/kernels/k004_graphed_triton.py"
            )}
            reasons = ship_manifest.journal_evidence_reasons(
                {"kind": "journal", "entry_id": entry_id,
                 "selection_rationale": "historic best"},
                1, repo.shapes[1], repo.submission_sha, rows,
            )
            joined = " ".join(reasons)
            self.assertIn("legacy_pre_lock_frozen_runner_journal", joined)
            self.assertIn("not_bound_to_final_submission", joined)
            self.assertIn("missing_audit_verdict", joined)

    def test_even_a_perfect_journal_row_still_refuses(self):
        """A row measured from the exact submission bytes is still pre-lock."""
        with fixture_repo() as repo:
            entry_id = "20260828-023313-6a45e2"
            row = self.journal_row(repo, entry_id, 1, impl_path=SUBMISSION_REL)
            row["impl"]["sha256"] = repo.submission_sha
            reasons = ship_manifest.journal_evidence_reasons(
                {"kind": "journal", "entry_id": entry_id,
                 "selection_rationale": "historic best"},
                1, repo.shapes[1], repo.submission_sha, {entry_id: row},
            )
            self.assertTrue(reasons)
            self.assertIn("legacy_pre_lock_frozen_runner_journal", reasons[0])

    def test_build_manifest_reports_every_failing_shape_at_once(self):
        with fixture_repo() as repo:
            entry_id = "20260828-023313-6a45e2"
            row = self.journal_row(
                repo, entry_id, 1, impl_path="Project/kernels/k004_graphed_triton.py"
            )
            (repo.project / "results" / "JOURNAL.jsonl").write_text(
                json.dumps(row) + "\n"
            )
            good = repo.controller_measurement(entry_id="run-" + "5" * 32, shape_id=2)
            audit_sha = "c" * 64
            selectors = {}
            for shape_id in range(1, 15):
                if shape_id == 1:
                    selectors["1"] = {
                        "kind": "journal", "entry_id": entry_id,
                        "selection_rationale": "historic best",
                    }
                elif shape_id == 2:
                    selectors["2"] = good
                elif shape_id in (6, 14):
                    selectors[str(shape_id)] = {
                        "kind": "side_controller",
                        "entry_id": f"20260830-1200{shape_id:02d}-abcdef",
                        "measurement_event_sha256": "f" * 64,
                        "side_evidence_sha256": "1" * 64,
                        "audit_packet": {
                            "path": f"Project/authority/blobs/{audit_sha}.json",
                            "sha256": audit_sha,
                        },
                        "selection_rationale": "not measured yet",
                    }
                else:
                    selectors[str(shape_id)] = {
                        "kind": "controller",
                        "entry_id": "run-" + f"{shape_id:032x}",
                        "measurement_event_sha256": "e" * 64,
                        "audit_packet": {
                            "path": f"Project/authority/blobs/{audit_sha}.json",
                            "sha256": audit_sha,
                        },
                        "selection_rationale": "not measured yet",
                    }
            map_path = repo.project / "final_evidence_map.json"
            map_path.write_text(json.dumps({
                "schema_version": ship_manifest.MAP_SCHEMA,
                "submission_sha256": repo.submission_sha,
                "official_sha256": repo.official_sha,
                "shapes": selectors,
            }, indent=2))
            with self.assertRaises(ship_manifest.ManifestRefusal) as caught:
                ship_manifest.build_manifest(map_path, "deadbeef", repo.submission_sha)
            refusal = caught.exception
            self.assertEqual(len(refusal.shapes), 13)
            self.assertNotIn("2", refusal.shapes)
            self.assertEqual(
                refusal.shapes["1"]["evidence_class"], ship_manifest.LEGACY_PRE_LOCK
            )
            self.assertIn(
                "legacy_pre_lock_frozen_runner_journal", refusal.shapes["1"]["reason"]
            )
            for row in refusal.shapes.values():
                self.assertIsNone(row["audit_verdict"])
                self.assertEqual(row["evidence_status"], "REFUSED")
            # A side shape must name what is actually on disk for it, not stop
            # at "the blob you named does not exist".
            for shape_id in ("6", "14"):
                with self.subTest(shape=shape_id):
                    diagnosis = refusal.shapes[shape_id]["side_evidence_diagnosis"]
                    self.assertTrue(diagnosis)
                    self.assertIn(
                        f"no_side_evidence_present:shape{shape_id}_*.json",
                        diagnosis,
                    )
            self.assertNotIn("side_evidence_diagnosis", refusal.shapes["1"])
            self.assertIn("shape 1", str(refusal))
            self.assertIn("shape 14", str(refusal))

    def test_full_manifest_emits_only_when_every_shape_is_bound(self):
        with fixture_repo() as repo:
            selectors = {}
            for shape_id in range(1, 15):
                if shape_id == 6:
                    entry_id = "20260830-120006-aaaaaa"
                    selectors["6"] = repo.side_measurement(
                        entry_id=entry_id, shape_id=6,
                        stage_packets=[("shape6-eval", repo.shape6_packet(entry_id))],
                    )
                elif shape_id == 14:
                    entry_id = "20260830-120014-bbbbbb"
                    selectors["14"] = repo.side_measurement(
                        entry_id=entry_id, shape_id=14,
                        stage_packets=repo.shape14_stage_packets(entry_id),
                    )
                else:
                    selectors[str(shape_id)] = repo.controller_measurement(
                        entry_id="run-" + f"{shape_id:032x}", shape_id=shape_id,
                    )
            map_path = repo.project / "final_evidence_map.json"
            map_path.write_text(json.dumps({
                "schema_version": ship_manifest.MAP_SCHEMA,
                "submission_sha256": repo.submission_sha,
                "official_sha256": repo.official_sha,
                "shapes": selectors,
            }, indent=2))
            manifest = ship_manifest.build_manifest(
                map_path, "cafebabe", repo.submission_sha
            )
            self.assertEqual(manifest["schema_version"], ship_manifest.MANIFEST_SCHEMA)
            self.assertEqual(len(manifest["shapes"]), 14)
            self.assertEqual(manifest["git_revision"], "cafebabe")
            self.assertEqual(
                manifest["evidence_classes"][ship_manifest.LEGACY_PRE_LOCK], []
            )
            self.assertEqual(
                manifest["evidence_classes"][ship_manifest.POST_LOCK],
                list(range(1, 15)),
            )
            for shape_id, entry in manifest["shapes"].items():
                with self.subTest(shape=shape_id):
                    self.assertEqual(entry["submission_sha256"], repo.submission_sha)
                    self.assertEqual(entry["evidence_class"], ship_manifest.POST_LOCK)
                    self.assertEqual(entry["evidence_status"], "SHIPPABLE")
                    self.assertEqual(entry["audit_verdict"], "PASS")
                    self.assertEqual(
                        set(entry["environment"]),
                        set(ship_manifest.REQUIRED_ENV_KEYS),
                    )
                    self.assertTrue(entry["selection_rationale"])

    def test_one_bad_shape_prevents_the_whole_manifest(self):
        with fixture_repo() as repo:
            selectors = {}
            for shape_id in range(1, 15):
                if shape_id == 6:
                    entry_id = "20260830-120006-aaaaaa"
                    selectors["6"] = repo.side_measurement(
                        entry_id=entry_id, shape_id=6,
                        stage_packets=[("shape6-eval", repo.shape6_packet(entry_id))],
                    )
                elif shape_id == 14:
                    entry_id = "20260830-120014-bbbbbb"
                    selectors["14"] = repo.side_measurement(
                        entry_id=entry_id, shape_id=14,
                        stage_packets=repo.shape14_stage_packets(entry_id),
                    )
                else:
                    selectors[str(shape_id)] = repo.controller_measurement(
                        entry_id="run-" + f"{shape_id:032x}", shape_id=shape_id,
                        integrity="RULE_VIOLATION" if shape_id == 9 else "PASS",
                    )
            map_path = repo.project / "final_evidence_map.json"
            map_path.write_text(json.dumps({
                "schema_version": ship_manifest.MAP_SCHEMA,
                "submission_sha256": repo.submission_sha,
                "official_sha256": repo.official_sha,
                "shapes": selectors,
            }, indent=2))
            with self.assertRaises(ship_manifest.ManifestRefusal) as caught:
                ship_manifest.build_manifest(
                    map_path, "cafebabe", repo.submission_sha
                )
            self.assertIn("9", caught.exception.shapes)
            self.assertIn("RULE_VIOLATION", caught.exception.shapes["9"]["reason"])
            self.assertFalse(
                (repo.project / "results_side" / "SHIP_MANIFEST.json").exists()
            )


# --------------------------------------------------------------------------
# Freeze provenance: the recorded revision always contains the submission
# --------------------------------------------------------------------------

class FreezeProvenanceTests(unittest.TestCase):
    def git(self, root, *arguments):
        return subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, check=True, text=True,
        ).stdout.strip()

    @contextlib.contextmanager
    def git_repo(self):
        with fixture_repo() as repo:
            self.git(repo.root, "init", "--quiet")
            self.git(repo.root, "config", "user.email", "fixture@example.invalid")
            self.git(repo.root, "config", "user.name", "fixture")
            yield repo

    def test_recorded_revision_contains_the_named_submission_bytes(self):
        with self.git_repo() as repo:
            map_path = repo.project / "final_evidence_map.json"
            map_path.write_text("{}\n")
            self.git(repo.root, "add", "-A")
            self.git(repo.root, "commit", "--quiet", "-m", "freeze")
            head, submission_sha = ship_manifest.freeze_provenance(map_path)
            self.assertEqual(submission_sha, repo.submission_sha)
            self.assertEqual(head, self.git(repo.root, "rev-parse", "HEAD"))
            committed = subprocess.run(
                ["git", "show", f"{head}:{SUBMISSION_REL}"],
                cwd=repo.root, capture_output=True, check=True,
            ).stdout
            self.assertEqual(hashlib.sha256(committed).hexdigest(), submission_sha)

    def test_a_revision_that_predates_the_submission_is_refused(self):
        with self.git_repo() as repo:
            map_path = repo.project / "final_evidence_map.json"
            map_path.write_text("{}\n")
            self.git(repo.root, "add", "-A")
            self.git(repo.root, "commit", "--quiet", "-m", "first")
            repo.submission_path.write_bytes(b"# newer submission bytes\n")
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.freeze_provenance(map_path)
            self.assertIn("uncommitted changes", str(caught.exception))

    def test_an_uncommitted_evidence_map_is_refused(self):
        with self.git_repo() as repo:
            map_path = repo.project / "final_evidence_map.json"
            map_path.write_text("{}\n")
            self.git(repo.root, "add", "-A")
            self.git(repo.root, "commit", "--quiet", "-m", "first")
            map_path.write_text('{"drift": true}\n')
            with self.assertRaises(ship_manifest.ManifestError):
                ship_manifest.freeze_provenance(map_path)

    def test_untracked_evidence_blocks_the_freeze(self):
        with self.git_repo() as repo:
            map_path = repo.project / "final_evidence_map.json"
            map_path.write_text("{}\n")
            self.git(repo.root, "add", "-A")
            self.git(repo.root, "commit", "--quiet", "-m", "first")
            (repo.project / "results_side" / "stray.json").write_text("{}\n")
            with self.assertRaises(ship_manifest.ManifestError) as caught:
                ship_manifest.freeze_provenance(map_path)
            self.assertIn("uncommitted changes", str(caught.exception))


# --------------------------------------------------------------------------
# The real repository, today: everything refuses
# --------------------------------------------------------------------------

class LiveEvidenceDiagnosisTests(unittest.TestCase):
    """Read-only assertions against the real ledgers."""

    @classmethod
    def setUpClass(cls):
        cls.report = ship_manifest.diagnose(
            ship_manifest.sha256_file(ship_manifest.SUBMISSION), "HEAD", False
        )

    def test_no_official_shape_has_shippable_evidence_today(self):
        self.assertFalse(self.report["shippable"])
        self.assertEqual(
            self.report["unshippable_shapes"],
            [str(index) for index in range(1, 15)],
        )

    def test_every_shape_states_a_specific_reason_and_no_verdict(self):
        for shape_id, row in self.report["shapes"].items():
            with self.subTest(shape=shape_id):
                self.assertIsNone(row["audit_verdict"])
                self.assertEqual(row["evidence_status"], "NOT_SHIPPABLE")
                self.assertEqual(row["evidence_class"], ship_manifest.LEGACY_PRE_LOCK)
                self.assertEqual(row["shippable_evidence_count"], 0)
                self.assertTrue(row["reasons"])

    def test_every_examined_legacy_row_is_unbound_or_hard_blocked(self):
        examined = [
            item
            for row in self.report["shapes"].values()
            for item in row["examined"]
        ]
        self.assertTrue(examined)
        self.assertEqual([item for item in examined if item["promotion_eligible"]], [])
        for item in examined:
            with self.subTest(entry=item["entry_id"]):
                self.assertTrue(item["blocking_reasons"])

    def test_side_shapes_name_their_missing_authority_binding(self):
        for shape_id in ("6", "14"):
            joined = " ".join(self.report["shapes"][shape_id]["reasons"])
            with self.subTest(shape=shape_id):
                self.assertIn("outside the authority store", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
