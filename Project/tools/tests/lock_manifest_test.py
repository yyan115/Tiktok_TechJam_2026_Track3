#!/usr/bin/env python3
"""Detached-signature and byte-integrity tests for owner LOCK."""

from __future__ import annotations

import base64
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Project" / "harness"))

from authority import AuthorityError, AuthorityPaths, atomic_write, canonical_json  # noqa: E402
from lock_manifest import create_lock_document, safe_relative_path, verify_lock  # noqa: E402
from trusted_controller import ControllerRefusal, TrustedController  # noqa: E402


def signed_capability(private_key, *, action: str, target: str, nonce: str) -> dict:
    now = datetime.now(timezone.utc)
    value = {
        "schema_version": 1,
        "capability_id": f"capability-{nonce}",
        "role": "owner",
        "campaign_id": "lock-test",
        "actions": [action],
        "targets": [target],
        "issued_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        "max_uses": 1,
        "nonce": nonce,
    }
    value["signature"] = base64.b64encode(
        private_key.sign(canonical_json(value))
    ).decode("ascii")
    return value


def main() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        print(("PASS " if condition else "FAIL ") + name)
        if not condition:
            failures.append(name)

    with tempfile.TemporaryDirectory(prefix="lock-manifest-") as temp:
        root = Path(temp)
        paths = AuthorityPaths(root)
        paths.directory.mkdir(parents=True)
        protected = root / "Project" / "harness" / "controller.py"
        protected.parent.mkdir(parents=True)
        protected.write_text("trusted bytes\n")

        owner = Ed25519PrivateKey.generate()
        critic = Ed25519PrivateKey.generate()
        owner_pem = owner.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        critic_pem = critic.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        paths.public_key.write_bytes(owner_pem)
        paths.critic_public_key.write_bytes(critic_pem)
        document = create_lock_document(
            root=root,
            protected_files=["Project/harness/controller.py"],
            owner_public_key=paths.public_key,
            critic_public_key=paths.critic_public_key,
            rules_snapshot_sha256="a" * 64,
            epoch="post-lock-test",
            lock_id="lock-test-0123456789abcdef",
        )
        atomic_write(paths.lock_manifest, canonical_json(document) + b"\n")
        signature = owner.sign(canonical_json(document))
        atomic_write(
            paths.lock_signature,
            base64.b64encode(signature) + b"\n",
            mode=0o444,
        )
        result = verify_lock(root)
        check("valid detached owner signature and bytes pass", result["valid"] is True)

        controller = TrustedController(root)
        check("valid signed LOCK starts inactive", controller.verify_lock_only()["active"] is False)
        activation_cap = root / "activation.json"
        activation_cap.write_text(json.dumps(signed_capability(
            owner,
            action="lock.activate",
            target=f"lock:{document['lock_id']}",
            nonce="initial-activation-nonce",
        )))
        activated = controller.activate_lock(activation_cap)
        check("owner capability activates exact LOCK", activated["active"] is True)
        check("active LOCK passes controller requirement", controller.require_lock()["active"] is True)

        protected.write_text("mutated\n")
        try:
            verify_lock(root)
        except AuthorityError:
            mutation_denied = True
        else:
            mutation_denied = False
        check("protected-byte mutation fails closed", mutation_denied)
        protected.write_text("trusted bytes\n")

        forged = Ed25519PrivateKey.generate()
        forged_signature = forged.sign(canonical_json(document))
        atomic_write(paths.lock_signature, base64.b64encode(forged_signature) + b"\n")
        try:
            verify_lock(root)
        except AuthorityError:
            forged_denied = True
        else:
            forged_denied = False
        check("forged LOCK signature fails closed", forged_denied)

        # Restore the first signature, rotate to a second signed epoch, then
        # prove restoring the old signed bytes does not roll authority back.
        atomic_write(paths.lock_signature, base64.b64encode(signature) + b"\n")
        old_document_bytes = canonical_json(document) + b"\n"
        old_signature_bytes = base64.b64encode(signature) + b"\n"
        second = create_lock_document(
            root=root,
            protected_files=["Project/harness/controller.py"],
            owner_public_key=paths.public_key,
            critic_public_key=paths.critic_public_key,
            rules_snapshot_sha256="b" * 64,
            epoch="post-lock-test-2",
            lock_id="lock-test-fedcba9876543210",
        )
        second_signature = owner.sign(canonical_json(second))
        atomic_write(paths.lock_manifest, canonical_json(second) + b"\n")
        atomic_write(paths.lock_signature, base64.b64encode(second_signature) + b"\n")
        rotation_cap = root / "rotation.json"
        rotation_cap.write_text(json.dumps(signed_capability(
            owner,
            action="lock.rotate",
            target=f"lock:{second['lock_id']}",
            nonce="rotation-activation-nonce",
        )))
        check("new signed epoch rotates under owner authority", controller.activate_lock(rotation_cap)["active"] is True)
        atomic_write(paths.lock_manifest, old_document_bytes)
        atomic_write(paths.lock_signature, old_signature_bytes)
        check("old signed LOCK rollback is inactive", controller.verify_lock_only()["active"] is False)
        try:
            controller.require_lock()
        except ControllerRefusal:
            rollback_refused = True
        else:
            rollback_refused = False
        check("controller refuses old signed LOCK rollback", rollback_refused)

        traversal_denied = False
        try:
            safe_relative_path("../outside")
        except AuthorityError:
            traversal_denied = True
        check("LOCK path traversal denied", traversal_denied)

    print(f"\n{len(failures)} failure(s)" if failures else "\nALL GREEN")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
