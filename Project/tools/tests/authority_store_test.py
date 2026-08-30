#!/usr/bin/env python3
"""Cold, filesystem-level tests for the post-lock authority store."""

from __future__ import annotations

import base64
import json
import multiprocessing
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Project" / "harness"))

from authority import (  # noqa: E402
    AuthorityError,
    AuthorityStore,
    canonical_json,
    iso_utc,
    sha256_bytes,
    utc_now,
)


def append_child(root: str, index: int) -> None:
    AuthorityStore(Path(root)).append(
        kind="concurrent_test", actor="test-child", payload={"index": index}
    )


def sign_capability(
    private: Ed25519PrivateKey,
    *,
    campaign: str = "campaign-test",
    capability_id: str = "cap-test-1",
    max_uses: int = 2,
    expires_delta: timedelta = timedelta(hours=1),
) -> dict:
    now = utc_now()
    document = {
        "schema_version": 1,
        "capability_id": capability_id,
        "role": "owner",
        "campaign_id": campaign,
        "actions": ["permit.issue", "verdict.resolve"],
        "targets": ["shape:*", "verdict:*"],
        "issued_at": iso_utc(now - timedelta(seconds=1)),
        "expires_at": iso_utc(now + expires_delta),
        "max_uses": max_uses,
        "nonce": "owner-nonce-test",
    }
    document["signature"] = base64.b64encode(
        private.sign(canonical_json(document))
    ).decode("ascii")
    return document


def permit_request(
    mode: str = "optimization",
    candidate: str | None = "a" * 64,
    request_sha256: str = "e" * 64,
) -> dict:
    body = {
        "campaign_id": "campaign-test",
        "mode": mode,
        "shape_id": 3,
        "candidate_sha256": candidate,
        "family_id": None if mode == "diagnostic" else "family:test",
        "expires_at": iso_utc(utc_now() + timedelta(minutes=30)),
    }
    return {"request_sha256": request_sha256, **body}


def main() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        print(("PASS " if condition else "FAIL ") + name)
        if not condition:
            failures.append(name)

    with tempfile.TemporaryDirectory(prefix="authority-test-") as temp:
        root = Path(temp)
        store = AuthorityStore(root)

        private = Ed25519PrivateKey.generate()
        public_pem = private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        store.ensure_layout()
        store.paths.public_key.write_bytes(public_pem)
        store.paths.critic_public_key.write_bytes(public_pem)

        first = store.append(kind="epoch_started", actor="test", payload={"epoch": 1})
        events = store.read_events()
        check("single event validates", len(events) == 1 and events[0] == first)

        digest, blob = store.store_blob(b"immutable evidence", suffix=".json")
        digest2, blob2 = store.store_blob(b"immutable evidence", suffix=".json")
        check("content-addressed blob is idempotent", digest == digest2 and blob == blob2)

        # Concurrent processes must serialize complete hash-chain transitions.
        ctx = multiprocessing.get_context("spawn")
        children = [ctx.Process(target=append_child, args=(str(root), i)) for i in range(12)]
        for child in children:
            child.start()
        for child in children:
            child.join(20)
        check("concurrent append processes exit cleanly", all(c.exitcode == 0 for c in children))
        check("concurrent event chain validates", len(store.read_events()) == 13)

        capability = sign_capability(private)
        request_bytes = canonical_json({"gate_request": "primary"})
        request_blob_sha, _ = store.store_blob(request_bytes, suffix=".json")
        request = permit_request(request_sha256=request_blob_sha)
        permit = store.issue_permit(
            request=request,
            request_blob_sha256=request_blob_sha,
            capability_document=capability,
        )
        check("signed owner capability issues bound permit", permit["candidate_sha256"] == "a" * 64)

        consumed = store.consume_permit(
            permit_id=permit["permit_id"],
            mode="optimization",
            shape_id=3,
            candidate_sha256="a" * 64,
        )
        check("one-use permit consumes", consumed["permit_id"] == permit["permit_id"])
        try:
            store.consume_permit(
                permit_id=permit["permit_id"],
                mode="optimization",
                shape_id=3,
                candidate_sha256="a" * 64,
            )
        except AuthorityError:
            duplicate_denied = True
        else:
            duplicate_denied = False
        check("duplicate permit consumption denied", duplicate_denied)

        bad = dict(capability)
        bad["nonce"] = "forged"
        try:
            store.verify_capability(
                bad, action="permit.issue", target="shape:3", campaign_id="campaign-test"
            )
        except AuthorityError:
            bad_denied = True
        else:
            bad_denied = False
        check("tampered owner capability denied", bad_denied)

        # Second authorized use succeeds; the third exhausts max_uses.
        event = store.append_authorized(
            kind="verdict_resolution",
            actor="owner",
            payload={"subject_sha256": "b" * 64},
            capability_document=capability,
            action="verdict.resolve",
            target="verdict:entry-1",
            campaign_id="campaign-test",
        )
        receipt = {
            "authority_event_id": event["event_id"],
            "action": "verdict.resolve",
            "subject_sha256": "b" * 64,
            "capability_nonce": event["payload"]["capability_nonce"],
        }
        check("journal-backed authority receipt verifies", store.verify_receipt(**receipt))
        try:
            store.append_authorized(
                kind="verdict_resolution",
                actor="owner",
                payload={"subject_sha256": "c" * 64},
                capability_document=capability,
                action="verdict.resolve",
                target="verdict:entry-2",
                campaign_id="campaign-test",
            )
        except AuthorityError:
            exhausted = True
        else:
            exhausted = False
        check("capability use exhaustion is fail-closed", exhausted)

        diagnostic_bytes = canonical_json({"gate_request": "diagnostic"})
        diag_sha, _ = store.store_blob(diagnostic_bytes, suffix=".json")
        diagnostic = permit_request(
            "diagnostic", "d" * 64, request_sha256=diag_sha
        )
        diag_cap = sign_capability(private, capability_id="cap-test-diag", max_uses=1)
        diag_permit = store.issue_permit(
            request=diagnostic,
            request_blob_sha256=diag_sha,
            capability_document=diag_cap,
        )
        check(
            "diagnostic permit cannot modify or promote",
            diag_permit["may_modify_candidate"] is False
            and diag_permit["may_promote"] is False,
        )

        # A torn/corrupt tail never gets silently skipped.
        with store.paths.events.open("ab") as stream:
            stream.write(b'{"partial":')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            store.read_events()
        except AuthorityError:
            corrupt_denied = True
        else:
            corrupt_denied = False
        check("corrupt journal tail fails closed", corrupt_denied)

    print(f"\n{len(failures)} failure(s)" if failures else "\nALL GREEN")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
