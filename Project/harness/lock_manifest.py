#!/usr/bin/env python3
"""Owner-signed post-FIX lock manifest.

The lock document is data, not authority by itself.  It becomes authoritative
only when its detached Ed25519 signature validates under the separately
installed owner public key and every protected byte matches its recorded hash.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from authority import (
    AUTHORITY_SCHEMA_VERSION,
    AuthorityError,
    AuthorityPaths,
    canonical_json,
    iso_utc,
    parse_utc,
    read_json_exact,
    sha256_bytes,
    sha256_file,
)


LOCK_KEYS = {
    "schema_version",
    "lock_id",
    "created_at",
    "epoch",
    "owner_key_sha256",
    "critic_key_sha256",
    "rules_snapshot_sha256",
    "protected_files",
}


def _load_ed25519(path: Path, label: str) -> tuple[Ed25519PublicKey, str]:
    try:
        value = serialization.load_pem_public_key(path.read_bytes())
    except Exception as exc:
        raise AuthorityError(f"cannot load {label} public key: {exc}") from exc
    if not isinstance(value, Ed25519PublicKey):
        raise AuthorityError(f"{label} key must be Ed25519")
    raw = value.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return value, sha256_bytes(raw)


def safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AuthorityError("protected file path must be non-empty text")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise AuthorityError(f"unsafe protected file path: {value!r}")
    normalized = str(pure)
    if normalized != value or value.endswith("/"):
        raise AuthorityError(f"non-canonical protected file path: {value!r}")
    return value


def create_lock_document(
    *,
    root: Path,
    protected_files: Iterable[str],
    owner_public_key: Path,
    critic_public_key: Path,
    rules_snapshot_sha256: str,
    epoch: str,
    lock_id: str,
) -> dict[str, Any]:
    resolved_root = root.resolve()
    _, owner_fingerprint = _load_ed25519(owner_public_key, "owner")
    _, critic_fingerprint = _load_ed25519(critic_public_key, "critic")
    if not isinstance(rules_snapshot_sha256, str) or len(rules_snapshot_sha256) != 64:
        raise AuthorityError("rules snapshot SHA-256 is malformed")
    if not isinstance(epoch, str) or not epoch:
        raise AuthorityError("epoch must be non-empty")
    if not isinstance(lock_id, str) or len(lock_id) < 16:
        raise AuthorityError("lock_id must be unpredictable")
    hashes: dict[str, str] = {}
    for raw in sorted(set(protected_files)):
        relative = safe_relative_path(raw)
        path = resolved_root / relative
        if path.is_symlink() or not path.is_file():
            raise AuthorityError(f"protected path is not a regular file: {relative}")
        hashes[relative] = sha256_file(path)
    if not hashes:
        raise AuthorityError("lock manifest cannot protect an empty file set")
    return {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "lock_id": lock_id,
        "created_at": iso_utc(),
        "epoch": epoch,
        "owner_key_sha256": owner_fingerprint,
        "critic_key_sha256": critic_fingerprint,
        "rules_snapshot_sha256": rules_snapshot_sha256,
        "protected_files": hashes,
    }


def verify_lock(root: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    paths = AuthorityPaths(resolved_root)
    document = read_json_exact(paths.lock_manifest, name="LOCK manifest")
    if set(document) != LOCK_KEYS:
        raise AuthorityError("LOCK manifest has unexpected or missing fields")
    if document["schema_version"] != AUTHORITY_SCHEMA_VERSION:
        raise AuthorityError("unsupported LOCK schema")
    for field in ("lock_id", "epoch"):
        if not isinstance(document[field], str) or not document[field]:
            raise AuthorityError(f"LOCK {field} must be non-empty")
    parse_utc(document["created_at"], "LOCK.created_at")
    owner_key, owner_fingerprint = _load_ed25519(paths.public_key, "owner")
    _, critic_fingerprint = _load_ed25519(paths.critic_public_key, "critic")
    if document["owner_key_sha256"] != owner_fingerprint:
        raise AuthorityError("LOCK owner-key fingerprint mismatch")
    if document["critic_key_sha256"] != critic_fingerprint:
        raise AuthorityError("LOCK critic-key fingerprint mismatch")
    try:
        signature = base64.b64decode(
            paths.lock_signature.read_text(encoding="ascii").strip(), validate=True
        )
    except Exception as exc:
        raise AuthorityError("LOCK signature is absent or malformed") from exc
    try:
        owner_key.verify(signature, canonical_json(document))
    except InvalidSignature as exc:
        raise AuthorityError("LOCK signature is invalid") from exc
    protected = document["protected_files"]
    if not isinstance(protected, dict) or not protected:
        raise AuthorityError("LOCK protected_files must be a non-empty object")
    mismatches: list[dict[str, str]] = []
    for raw, expected in protected.items():
        relative = safe_relative_path(raw)
        if not isinstance(expected, str) or len(expected) != 64:
            raise AuthorityError(f"invalid protected hash for {relative}")
        path = resolved_root / relative
        if path.is_symlink() or not path.is_file():
            actual = "missing-or-not-regular"
        else:
            actual = sha256_file(path)
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    if mismatches:
        raise AuthorityError(f"LOCK integrity mismatch: {mismatches}")
    return {
        "valid": True,
        "lock_id": document["lock_id"],
        "epoch": document["epoch"],
        "manifest_sha256": hashlib.sha256(canonical_json(document)).hexdigest(),
        "protected_file_count": len(protected),
        "owner_key_sha256": owner_fingerprint,
        "critic_key_sha256": critic_fingerprint,
        "rules_snapshot_sha256": document["rules_snapshot_sha256"],
    }

