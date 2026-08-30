#!/usr/bin/env python3
"""Focused no-GPU tests for the prospective audit authority and watcher."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_authority as aa
import champion_watch as cw

ENTRY = "20260830-120000-abcdef"
SECOND_ENTRY = "20260830-120001-fedcba"
RUN_ENTRY = "run-0123456789abcdef0123456789abcdef"
CANDIDATE = "b" * 64
PACKET = "c" * 64
NONCE = "d" * 64
ARTIFACT = "e" * 64
CODEX = aa.CodexIdentity("/usr/local/bin/codex", "/opt/codex", "f" * 64)


def verdict(
    integrity: str = "PASS",
    technical: str = "PASS",
    *,
    entry_id: str = ENTRY,
    packet_sha256: str = PACKET,
    candidate_sha256: str = CANDIDATE,
    nonce: str = NONCE,
) -> dict:
    return {
        "schema_version": 2,
        "attempt_nonce": nonce,
        "entry_id": entry_id,
        "packet_sha256": packet_sha256,
        "candidate_sha256": candidate_sha256,
        "integrity": {
            "verdict": integrity,
            "findings": [],
            "retest_request": "repeat exact bytes" if integrity == "RETEST" else "",
            "summary": "integrity reviewed",
        },
        "technical_review": {
            "verdict": technical,
            "findings": [],
            "summary": "diagnosis reviewed",
        },
        "summary": "complete independent review",
    }


class AuditAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.events = self.root / "audit_events.jsonl"
        self.lock = self.root / ".lock"
        self.legacy = self.root / "verdicts.jsonl"
        self.legacy.write_text("")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def start(
        self,
        attempt: str,
        *,
        entry_id: str = ENTRY,
        packet_sha256: str = PACKET,
        candidate_sha256: str = CANDIDATE,
        nonce: str = NONCE,
        measurement_event_sha256: str = "0" * 64,
        lane: str = "legacy-primary",
        queue_event_sha256: str = "",
    ) -> dict:
        return aa.record_attempt_started(
            entry_id=entry_id, attempt_id=attempt, attempt_nonce=nonce,
            packet_sha256=packet_sha256,
            candidate_sha256=candidate_sha256,
            codex=CODEX,
            measurement_event_sha256=measurement_event_sha256,
            lane=lane,
            queue_event_sha256=queue_event_sha256,
            path=self.events, lock_path=self.lock)

    def result(
        self,
        attempt: str,
        document: dict,
        *,
        entry_id: str = ENTRY,
        packet_sha256: str = PACKET,
        candidate_sha256: str = CANDIDATE,
        measurement_event_sha256: str = "0" * 64,
        lane: str = "legacy-primary",
    ) -> dict:
        artifact = self.root / "Project" / "audits" / "auto" / f"{attempt}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact_payload = {
            "artifact_version": 2,
            "artifact_type": "audit_response",
            "attempt_id": attempt,
            "entry_id": entry_id,
            "packet_sha256": packet_sha256,
            "candidate_sha256": candidate_sha256,
            "measurement_event_sha256": measurement_event_sha256,
            "lane": lane,
            "verdict_schema_sha256": aa.sha256_file(aa.SCHEMA_PATH),
            "codex": CODEX.as_dict(),
            "returncode": 0,
            "stdout": json.dumps(document),
            "parser_error": "",
            "validated_result": document,
        }
        artifact_sha = aa.exclusive_write_json(artifact, artifact_payload)
        return aa.record_audit_result(
            attempt_id=attempt, result=document,
            artifact_path=f"Project/audits/auto/{attempt}.json",
            artifact_sha256=artifact_sha, path=self.events, lock_path=self.lock,
            artifact_root=self.root)

    def decision(self) -> aa.AuditDecision:
        return aa.audit_decision(
            ENTRY, CANDIDATE, events_path=self.events, legacy_path=self.legacy,
            artifact_root=self.root)

    def test_exact_full_schema_and_bindings(self) -> None:
        raw = json.dumps(verdict())
        parsed = aa.validate_verdict_document(
            raw, attempt_nonce=NONCE, entry_id=ENTRY,
            packet_sha256=PACKET, candidate_sha256=CANDIDATE)
        self.assertEqual(parsed["integrity"]["verdict"], "PASS")
        with self.assertRaises(aa.AuditAuthorityError):
            aa.validate_verdict_document(
                raw + "\ntrailer", attempt_nonce=NONCE, entry_id=ENTRY,
                packet_sha256=PACKET, candidate_sha256=CANDIDATE)
        with self.assertRaises(aa.AuditAuthorityError):
            aa.validate_verdict_document(
                '{"verdict":"PASS"}', attempt_nonce=NONCE, entry_id=ENTRY,
                packet_sha256=PACKET, candidate_sha256=CANDIDATE)
        wrong = verdict()
        wrong["attempt_nonce"] = "0" * 64
        with self.assertRaises(aa.AuditAuthorityError):
            aa.validate_verdict_document(
                json.dumps(wrong), attempt_nonce=NONCE, entry_id=ENTRY,
                packet_sha256=PACKET, candidate_sha256=CANDIDATE)
        extra = verdict()
        extra["verdict"] = "PASS"
        with self.assertRaises(aa.AuditAuthorityError):
            aa.validate_verdict_document(
                json.dumps(extra), attempt_nonce=NONCE, entry_id=ENTRY,
                packet_sha256=PACKET, candidate_sha256=CANDIDATE)
        duplicate = raw[:-1] + ',"entry_id":"' + ENTRY + '"}'
        with self.assertRaises(aa.AuditAuthorityError):
            aa.validate_verdict_document(
                duplicate, attempt_nonce=NONCE, entry_id=ENTRY,
                packet_sha256=PACKET, candidate_sha256=CANDIDATE)

    def test_retest_request_conditional_is_enforced(self) -> None:
        bad = verdict("RETEST")
        bad["integrity"]["retest_request"] = ""
        with self.assertRaises(aa.AuditAuthorityError):
            aa.validate_verdict_document(
                json.dumps(bad), attempt_nonce=NONCE, entry_id=ENTRY,
                packet_sha256=PACKET, candidate_sha256=CANDIDATE)

    def test_first_hard_verdict_latches_until_resolution(self) -> None:
        self.start("one")
        hard = self.result("one", verdict("RULE_VIOLATION"))
        second_packet = "a" * 64
        second_nonce = "1" * 64
        self.start(
            "two",
            entry_id=SECOND_ENTRY,
            packet_sha256=second_packet,
            nonce=second_nonce,
        )
        passed = self.result(
            "two",
            verdict(
                entry_id=SECOND_ENTRY,
                packet_sha256=second_packet,
                nonce=second_nonce,
            ),
            entry_id=SECOND_ENTRY,
            packet_sha256=second_packet,
        )
        decision = aa.audit_decision(
            SECOND_ENTRY,
            CANDIDATE,
            packet_sha256=second_packet,
            events_path=self.events,
            legacy_path=self.legacy,
            artifact_root=self.root,
        )
        self.assertFalse(decision.promotion_eligible)
        self.assertEqual(decision.active_hard_event_sha256, hard["event_sha256"])
        harness = Path(__file__).resolve().parents[2] / "harness"
        if str(harness) not in sys.path:
            sys.path.insert(0, str(harness))
        from authority import AuthorityStore

        capability_nonce = "external-owner-nonce"
        authority_event = AuthorityStore(self.root).append(
            kind="audit_resolve_authorized",
            actor="trusted-controller",
            payload={
                "capability_consumed": True,
                "capability_action": "audit.resolve",
                "capability_target": f"audit:{ENTRY}",
                "capability_role": "owner",
                "subject_sha256": hard["event_sha256"],
                "capability_nonce": capability_nonce,
                "owner_key_sha256": "9" * 64,
            },
        )
        aa.record_resolution(
            entry_id=ENTRY,
            target_event_sha256=hard["event_sha256"],
            resolution_kind="FINDING_OVERTURNED",
            rationale="owner-authorized independent finding correction",
            authority_event_id=authority_event["event_id"],
            capability_nonce=capability_nonce,
            superseding_event_sha256=passed["event_sha256"],
            path=self.events, lock_path=self.lock, legacy_path=self.legacy,
            authority_root=self.root)
        resolved = aa.audit_decision(
            SECOND_ENTRY,
            CANDIDATE,
            packet_sha256=second_packet,
            events_path=self.events,
            legacy_path=self.legacy,
            artifact_root=self.root,
        )
        self.assertTrue(resolved.promotion_eligible)

    def test_retest_hard_verdict_follows_candidate_across_entry_ids(self) -> None:
        self.start("needs-retest")
        hard = self.result("needs-retest", verdict("RETEST"))

        second_packet = "a" * 64
        second_nonce = "1" * 64
        self.start(
            "retest-pass",
            entry_id=SECOND_ENTRY,
            packet_sha256=second_packet,
            nonce=second_nonce,
        )
        passed = self.result(
            "retest-pass",
            verdict(
                entry_id=SECOND_ENTRY,
                packet_sha256=second_packet,
                nonce=second_nonce,
            ),
            entry_id=SECOND_ENTRY,
            packet_sha256=second_packet,
        )
        decision = aa.audit_decision(
            SECOND_ENTRY,
            CANDIDATE,
            packet_sha256=second_packet,
            events_path=self.events,
            legacy_path=self.legacy,
            artifact_root=self.root,
        )
        self.assertFalse(decision.promotion_eligible)
        self.assertEqual(decision.active_hard_event_sha256, hard["event_sha256"])

        harness = Path(__file__).resolve().parents[2] / "harness"
        if str(harness) not in sys.path:
            sys.path.insert(0, str(harness))
        from authority import AuthorityStore

        capability_nonce = "external-owner-retest-nonce"
        authority_event = AuthorityStore(self.root).append(
            kind="audit_resolve_authorized",
            actor="trusted-controller",
            payload={
                "capability_consumed": True,
                "capability_action": "audit.resolve",
                "capability_target": f"audit:{ENTRY}",
                "capability_role": "owner",
                "subject_sha256": hard["event_sha256"],
                "capability_nonce": capability_nonce,
                "owner_key_sha256": "9" * 64,
            },
        )
        aa.record_resolution(
            entry_id=ENTRY,
            target_event_sha256=hard["event_sha256"],
            resolution_kind="RETEST_SATISFIED",
            rationale="independent retest of identical candidate bytes passed",
            authority_event_id=authority_event["event_id"],
            capability_nonce=capability_nonce,
            superseding_event_sha256=passed["event_sha256"],
            path=self.events,
            lock_path=self.lock,
            legacy_path=self.legacy,
            authority_root=self.root,
        )
        resolved = aa.audit_decision(
            SECOND_ENTRY,
            CANDIDATE,
            packet_sha256=second_packet,
            events_path=self.events,
            legacy_path=self.legacy,
            artifact_root=self.root,
        )
        self.assertTrue(resolved.promotion_eligible)

    def test_weak_diagnosis_pauses_then_second_review_can_pass(self) -> None:
        self.start("weak")
        self.result("weak", verdict("PASS", "WEAK_DIAGNOSIS"))
        decision = self.decision()
        self.assertFalse(decision.promotion_eligible)
        self.assertIn("technical:WEAK_DIAGNOSIS", decision.blocking_reasons)
        second_packet = "a" * 64
        second_nonce = "1" * 64
        self.start(
            "second-review",
            entry_id=SECOND_ENTRY,
            packet_sha256=second_packet,
            nonce=second_nonce,
        )
        self.result(
            "second-review",
            verdict(
                entry_id=SECOND_ENTRY,
                packet_sha256=second_packet,
                nonce=second_nonce,
            ),
            entry_id=SECOND_ENTRY,
            packet_sha256=second_packet,
        )
        second_decision = aa.audit_decision(
            SECOND_ENTRY,
            CANDIDATE,
            packet_sha256=second_packet,
            events_path=self.events,
            legacy_path=self.legacy,
            artifact_root=self.root,
        )
        self.assertTrue(second_decision.promotion_eligible)

    def test_technical_disagreement_is_advisory(self) -> None:
        self.start("advisory")
        self.result("advisory", verdict("PASS", "TECHNICAL_DISAGREEMENT"))
        self.assertTrue(self.decision().promotion_eligible)
        with self.assertRaises(aa.AuditAuthorityError):
            self.start("same-entry-rewrite")

    def test_missing_or_changed_response_artifact_revokes_eligibility(self) -> None:
        self.start("artifact-bound")
        self.result("artifact-bound", verdict())
        self.assertTrue(self.decision().promotion_eligible)
        artifact = (
            self.root / "Project" / "audits" / "auto" / "artifact-bound.json"
        )
        artifact.write_text("{}\n")
        decision = self.decision()
        self.assertFalse(decision.promotion_eligible)
        self.assertTrue(any(
            reason.startswith("response_artifact_invalid:")
            for reason in decision.blocking_reasons
        ))

    def test_missing_and_legacy_unbound_pass_block_promotion(self) -> None:
        self.assertFalse(self.decision().promotion_eligible)
        self.legacy.write_text(json.dumps({
            "entry_id": ENTRY, "verdict": "PASS", "recorded": "old"
        }) + "\n")
        decision = self.decision()
        self.assertFalse(decision.promotion_eligible)
        self.assertEqual(decision.technical_status, "LEGACY_UNBOUND")

    def test_hash_chain_tampering_fails_closed(self) -> None:
        self.start("chain")
        text = self.events.read_text()
        self.events.write_text(text.replace('"attempt_id":"chain"',
                                            '"attempt_id":"other"'))
        with self.assertRaises(aa.AuditAuthorityError):
            aa.read_events(self.events)

    def test_only_one_active_attempt(self) -> None:
        self.start("active")
        with self.assertRaises(aa.AuditAuthorityError):
            self.start("parallel")

    def test_retry_cap_is_derived_from_durable_events(self) -> None:
        for index in range(aa.MAX_FAILED_ATTEMPTS):
            attempt = f"fail-{index}"
            self.start(attempt)
            aa.record_attempt_failure(
                attempt_id=attempt, reason="test failure",
                path=self.events, lock_path=self.lock)
        summary = aa.attempt_summary(ENTRY, events_path=self.events)
        self.assertTrue(summary["retry_exhausted"])

    def test_packet_source_binding(self) -> None:
        packets = self.root / "packets"
        packets.mkdir()
        source = "def custom_kernel(data):\n    return data\n"
        source_sha = hashlib.sha256(source.encode()).hexdigest()
        payload = {
            "entry": {"entry_id": ENTRY, "impl": {"sha256": source_sha}},
            "candidate_source": source,
            "candidate_source_sha256_now": source_sha,
            "candidate_source_matches_journal": True,
        }
        (packets / f"{ENTRY}.json").write_text(json.dumps(payload))
        bound = aa.load_bound_packet(ENTRY, packets_dir=packets)
        self.assertEqual(bound.candidate_sha256, source_sha)
        payload["candidate_source"] += "# changed\n"
        (packets / f"{ENTRY}.json").write_text(json.dumps(payload))
        with self.assertRaises(aa.AuditAuthorityError):
            aa.load_bound_packet(ENTRY, packets_dir=packets)

    def test_controller_content_addressed_packet_and_run_id(self) -> None:
        blobs = self.root / "Project" / "authority" / "blobs"
        blobs.mkdir(parents=True)
        source = b"def custom_kernel(data):\n    return data\n"
        candidate_sha = hashlib.sha256(source).hexdigest()
        (blobs / f"{candidate_sha}.py").write_bytes(source)
        measurement_sha = "6" * 64
        packet = {
            "schema_version": 1,
            "entry_id": RUN_ENTRY,
            "lane": "shape14",
            "measurement_event_sha256": measurement_sha,
            "candidate_sha256": candidate_sha,
            "worker_request_sha256": "4" * 64,
            "worker_response_sha256": "5" * 64,
        }
        raw = aa._canonical(packet)
        packet_sha = hashlib.sha256(raw).hexdigest()
        (blobs / f"{packet_sha}.json").write_bytes(raw)
        bound = aa.load_bound_packet(
            RUN_ENTRY,
            packet_sha256=packet_sha,
            authority_blobs=blobs,
        )
        self.assertEqual(bound.candidate_sha256, candidate_sha)
        self.assertEqual(bound.measurement_event_sha256, measurement_sha)
        self.assertEqual(bound.lane, "shape14")
        self.assertEqual(bound.candidate_source_path.name, f"{candidate_sha}.py")
        (blobs / f"{candidate_sha}.py").write_bytes(source + b"# changed\n")
        with self.assertRaises(aa.AuditAuthorityError):
            aa.load_bound_packet(
                RUN_ENTRY,
                packet_sha256=packet_sha,
                authority_blobs=blobs,
            )

    def test_absolute_pinned_codex_and_home_refusal(self) -> None:
        executable = self.root / "bin" / "codex"
        executable.parent.mkdir()
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        digest = aa.sha256_file(executable)
        identity = aa.resolve_codex_identity(
            executable, digest, home_path=self.root / "unrelated-home")
        self.assertEqual(identity.sha256, digest)
        with self.assertRaises(aa.AuditAuthorityError):
            aa.resolve_codex_identity(executable, "0" * 64,
                                      home_path=self.root / "unrelated-home")
        with self.assertRaises(aa.AuditAuthorityError):
            aa.resolve_codex_identity(executable, digest, home_path=self.root)

    def test_o_excl_artifacts_never_overwrite(self) -> None:
        artifact = self.root / "one.json"
        first = aa.exclusive_write_json(artifact, {"codex": CODEX.as_dict()})
        with self.assertRaises(FileExistsError):
            aa.exclusive_write_json(artifact, {"replacement": True})
        self.assertEqual(first, aa.sha256_file(artifact))

    def test_watcher_reconstructs_from_zero_without_leaderboard_or_cache(self) -> None:
        journal = self.root / "JOURNAL.jsonl"
        journal.write_text(json.dumps({
            "entry_id": ENTRY,
            "type": "candidate",
            "promotion_candidate": True,
            "shape_id": 8,
            "impl": {"sha256": CANDIDATE},
        }) + "\n")
        snapshot = cw.queue_snapshot(
            journal_path=journal, events_path=self.events,
            legacy_path=self.legacy)
        self.assertEqual(
            [row["entry_id"] for row in snapshot["pending_rows"]], [ENTRY])
        self.assertEqual(snapshot["active_rows"], [])
        self.assertEqual(snapshot["owner_attention_rows"], [])

        self.legacy.write_text(json.dumps({
            "entry_id": ENTRY,
            "verdict": "PASS",
            "recorded": "legacy-final",
        }) + "\n")
        settled = cw.queue_snapshot(
            journal_path=journal,
            events_path=self.events,
            legacy_path=self.legacy,
        )
        self.assertEqual(settled["pending_rows"], [])

        queued = aa.enqueue_audit(
            entry_id=ENTRY,
            candidate_sha256=CANDIDATE,
            packet_sha256=PACKET,
            measurement_event_sha256="8" * 64,
            lane="primary",
            path=self.events,
            lock_path=self.lock,
        )
        prospective = cw.queue_snapshot(
            journal_path=journal,
            events_path=self.events,
            legacy_path=self.legacy,
        )
        self.assertEqual(len(prospective["pending_rows"]), 1)
        self.assertEqual(
            prospective["pending_rows"][0]["queue_event_sha256"],
            queued["event_sha256"],
        )

    def test_controller_queue_is_idempotent_and_includes_side_lanes(self) -> None:
        journal = self.root / "JOURNAL.jsonl"
        journal.write_text("")
        measurement = "8" * 64
        first = aa.enqueue_audit(
            entry_id=ENTRY,
            candidate_sha256=CANDIDATE,
            packet_sha256=PACKET,
            measurement_event_sha256=measurement,
            lane="shape14",
            path=self.events,
            lock_path=self.lock,
        )
        repeated = aa.enqueue_audit(
            entry_id=ENTRY,
            candidate_sha256=CANDIDATE,
            packet_sha256=PACKET,
            measurement_event_sha256=measurement,
            lane="shape14",
            path=self.events,
            lock_path=self.lock,
        )
        self.assertEqual(first["event_sha256"], repeated["event_sha256"])
        with self.assertRaises(aa.AuditAuthorityError):
            aa.enqueue_audit(
                entry_id=ENTRY,
                candidate_sha256=CANDIDATE,
                packet_sha256="7" * 64,
                measurement_event_sha256=measurement,
                lane="shape14",
                path=self.events,
                lock_path=self.lock,
            )
        snapshot = cw.queue_snapshot(
            journal_path=journal,
            events_path=self.events,
            legacy_path=self.legacy,
        )
        self.assertEqual(len(snapshot["pending_rows"]), 1)
        self.assertEqual(snapshot["pending_rows"][0]["lane"], "shape14")
        self.assertEqual(
            snapshot["pending_rows"][0]["measurement_event_sha256"],
            measurement,
        )

    def test_controller_attempt_and_decision_require_exact_queue_binding(self) -> None:
        measurement = "8" * 64
        blobs = self.root / "Project" / "authority" / "blobs"
        blobs.mkdir(parents=True)
        source = b"def custom_kernel(data):\n    return data\n"
        candidate_sha = hashlib.sha256(source).hexdigest()
        (blobs / f"{candidate_sha}.py").write_bytes(source)
        packet = {
            "schema_version": 1,
            "entry_id": RUN_ENTRY,
            "candidate_sha256": candidate_sha,
            "measurement_event_sha256": measurement,
            "lane": "shape6",
        }
        packet_raw = aa._canonical(packet)
        packet_sha = hashlib.sha256(packet_raw).hexdigest()
        (blobs / f"{packet_sha}.json").write_bytes(packet_raw)
        queue = aa.enqueue_audit(
            entry_id=RUN_ENTRY,
            candidate_sha256=candidate_sha,
            packet_sha256=packet_sha,
            measurement_event_sha256=measurement,
            lane="shape6",
            path=self.events,
            lock_path=self.lock,
        )
        with self.assertRaises(aa.AuditAuthorityError):
            self.start(
                "unbound-run",
                entry_id=RUN_ENTRY,
                candidate_sha256=candidate_sha,
                packet_sha256=packet_sha,
            )
        exact = aa.require_audit_enqueue(
            entry_id=RUN_ENTRY,
            candidate_sha256=candidate_sha,
            packet_sha256=packet_sha,
            measurement_event_sha256=measurement,
            lane="shape6",
            events_path=self.events,
        )
        self.assertEqual(exact["event_sha256"], queue["event_sha256"])
        self.start(
            "bound-run",
            entry_id=RUN_ENTRY,
            candidate_sha256=candidate_sha,
            packet_sha256=packet_sha,
            measurement_event_sha256=measurement,
            lane="shape6",
            queue_event_sha256=queue["event_sha256"],
        )
        self.result(
            "bound-run",
            verdict(
                entry_id=RUN_ENTRY,
                candidate_sha256=candidate_sha,
                packet_sha256=packet_sha,
            ),
            entry_id=RUN_ENTRY,
            candidate_sha256=candidate_sha,
            packet_sha256=packet_sha,
            measurement_event_sha256=measurement,
            lane="shape6",
        )
        decision = aa.audit_decision(
            RUN_ENTRY,
            candidate_sha,
            packet_sha256=packet_sha,
            events_path=self.events,
            legacy_path=self.legacy,
            artifact_root=self.root,
        )
        self.assertTrue(decision.promotion_eligible)


if __name__ == "__main__":
    unittest.main(verbosity=2)
