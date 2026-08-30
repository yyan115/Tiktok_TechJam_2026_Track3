#!/usr/bin/env python3
"""CPU/static regression tests for evidence paths and submission safety."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "Project" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_submission
import shape14_eval
import shape6_local_eval
import ship_manifest


def load_submission():
    path = ROOT / "Project/submission/torch_transformer_benchmark_submission.py"
    spec = importlib.util.spec_from_file_location("evidence_test_submission", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


class ManifestTests(unittest.TestCase):
    def eligible_decision(self, entry_id, candidate_sha256=None,
                          packet_sha256=None):
        return SimpleNamespace(
            promotion_eligible=True,
            blocking_reasons=(),
            as_dict=lambda: {
                "entry_id": entry_id,
                "promotion_eligible": True,
                "integrity_status": "PASS",
                "technical_status": "PASS",
                "blocking_reasons": [],
                "candidate_sha256": candidate_sha256,
                "packet_sha256": packet_sha256,
            },
        )

    def test_evidence_map_requires_exactly_all_official_shapes(self):
        expected = {1: {"id": 1}, 2: {"id": 2}}
        payload = {
            "schema_version": ship_manifest.MAP_SCHEMA,
            "submission_sha256": "a" * 64,
            "official_sha256": "b" * 64,
            "shapes": {"1": {"kind": "journal", "entry_id": "x"}},
        }
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.validate_evidence_map(
                payload, "a" * 64, "b" * 64, expected
            )

    def test_controller_selector_schema_and_raw_median_validation(self):
        audit_sha = "c" * 64
        selector = {
            "kind": "controller",
            "entry_id": "run-" + "d" * 32,
            "measurement_event_sha256": "e" * 64,
            "audit_packet": {
                "path": f"Project/authority/blobs/{audit_sha}.json",
                "sha256": audit_sha,
            },
        }
        expected = {shape_id: {"id": shape_id} for shape_id in range(1, 15)}
        payload = {
            "schema_version": ship_manifest.MAP_SCHEMA,
            "submission_sha256": "a" * 64,
            "official_sha256": "b" * 64,
            "shapes": {str(shape_id): selector for shape_id in expected},
        }
        for shape_id in (6, 14):
            payload["shapes"][str(shape_id)] = {
                "kind": "side_controller",
                "entry_id": f"20260830-12000{shape_id % 10}-abcdef",
                "measurement_event_sha256": "f" * 64,
                "side_evidence_sha256": "1" * 64,
                "audit_packet": {
                    "path": f"Project/authority/blobs/{audit_sha}.json",
                    "sha256": audit_sha,
                },
            }
        validated = ship_manifest.validate_evidence_map(
            payload, "a" * 64, "b" * 64, expected
        )
        self.assertEqual(len(validated), 14)
        samples = [float(index + 1) for index in range(300)]
        timing = {
            "raw_samples_ms": samples,
            "median_ms": 150.5,
            "n_samples": 300,
        }
        self.assertEqual(
            ship_manifest.validate_timing_samples(timing, 300, "candidate"),
            150.5,
        )
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.validate_timing_samples(
                {**timing, "median_ms": 150.0}, 300, "candidate"
            )
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.validate_timing_samples(
                {**timing, "raw_samples_ms": samples[:-1]}, 300, "candidate"
            )

    def test_shape6_memory_false_is_ineligible_even_with_pass_audit(self):
        samples = [1.0] * 300
        allocator = [100] * 10
        packet = {
            "numerical_state": dict(ship_manifest.OFFICIAL_NUMERICAL_STATE),
            "correctness": {
                "passed": True,
                "violations": 0,
                "nonfinite_elements": 0,
                "seeds": [1, 2, 3, 4, 5],
                "trials": [
                    {
                        "seed": seed,
                        "passed": True,
                        "violations": 0,
                        "nonfinite_elements": 0,
                    }
                    for seed in [1, 2, 3, 4, 5]
                ],
            },
            "memory": {
                "flat": False,
                "warmups": 3,
                "repeats": 10,
                "limits": dict(ship_manifest.SHAPE6_MEMORY_LIMITS),
                "peak_allocated_bytes_per_repeat": allocator,
                "peak_reserved_bytes_per_repeat": allocator,
                "settled_allocated_bytes_per_repeat": allocator,
                "settled_reserved_bytes_per_repeat": allocator,
                "allocated_slope_bytes_per_repeat": 0.0,
                "reserved_slope_bytes_per_repeat": 0.0,
                "allocated_end_growth_bytes": 0,
                "reserved_end_growth_bytes": 0,
                "allocated_max_growth_bytes": 0,
                "reserved_max_growth_bytes": 0,
            },
            "timing": {
                "warmups": 20,
                "repeats_per_round": 100,
                "round_count": 3,
                "speedup_vs_baseline": None,
                "raw_samples_ms": samples,
                "rounds": [
                    {"round": index, "samples_ms": samples[index * 100:(index + 1) * 100]}
                    for index in range(3)
                ],
                "median_ms": 1.0,
            },
        }
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.require_shape6_protocol(packet)
        packet["memory"]["flat"] = True
        ship_manifest.require_shape6_protocol(packet)

    def test_loose_side_packet_selector_is_rejected_by_schema(self):
        expected = {shape_id: {"id": shape_id} for shape_id in range(1, 15)}
        audit_sha = "c" * 64
        side = {
            "kind": "side_packet",
            "path": "Project/results_side/loose.json",
            "audit_packet": {
                "path": f"Project/authority/blobs/{audit_sha}.json",
                "sha256": audit_sha,
            },
        }
        controller = {
            "kind": "controller",
            "entry_id": "run-" + "d" * 32,
            "measurement_event_sha256": "e" * 64,
            "audit_packet": {
                "path": f"Project/authority/blobs/{audit_sha}.json",
                "sha256": audit_sha,
            },
        }
        payload = {
            "schema_version": ship_manifest.MAP_SCHEMA,
            "submission_sha256": "a" * 64,
            "official_sha256": "b" * 64,
            "shapes": {str(shape_id): controller for shape_id in expected},
        }
        payload["shapes"]["14"] = {
            "kind": "side_controller",
            "entry_id": "20260830-120014-abcdef",
            "measurement_event_sha256": "f" * 64,
            "side_evidence_sha256": "1" * 64,
            "audit_packet": {
                "path": f"Project/authority/blobs/{audit_sha}.json",
                "sha256": audit_sha,
            },
        }
        payload["shapes"]["6"] = side
        with self.assertRaises(ship_manifest.ManifestError):
            ship_manifest.validate_evidence_map(
                payload, "a" * 64, "b" * 64, expected
            )
        payload["shapes"]["6"] = {
            "kind": "side_controller",
            "entry_id": "20260830-120000-abcdef",
            "measurement_event_sha256": "f" * 64,
            "side_evidence_sha256": "1" * 64,
            "audit_packet": {
                "path": f"Project/authority/blobs/{audit_sha}.json",
                "sha256": audit_sha,
            },
        }
        validated = ship_manifest.validate_evidence_map(
            payload, "a" * 64, "b" * 64, expected
        )
        self.assertEqual(validated["6"]["kind"], "side_controller")

    def test_missing_or_blocked_audit_fails_closed(self):
        blocked = SimpleNamespace(
            promotion_eligible=False,
            blocking_reasons=("missing_audit_verdict",),
        )
        with mock.patch.object(ship_manifest, "audit_decision", return_value=blocked):
            with self.assertRaises(ship_manifest.ManifestError):
                ship_manifest.eligible_audit(
                    "20260830-120000-abcdef", "a" * 64, "b" * 64
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
