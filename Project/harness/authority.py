#!/usr/bin/env python3
"""Durable authority primitives for the post-lock controller.

This module deliberately contains no benchmark or candidate code.  It owns the
small pieces that must remain trustworthy when a candidate process is hostile:

* a strictly validated, hash-chained, append-only event journal;
* flock + fsync transitions (including directory fsync);
* content-addressed immutable evidence blobs;
* owner capabilities signed by an Ed25519 key kept outside the workspace; and
* one-use permits that are consumed before a worker is launched.

The public key and lock manifest are installed by the owner during LOCK.  The
private key is never accepted from a path below the repository or the current
user's home directory.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import secrets
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


AUTHORITY_SCHEMA_VERSION = 1
EVENT_REQUIRED = {
    "schema_version",
    "event_id",
    "recorded_at",
    "kind",
    "actor",
    "payload",
    "prev_event_sha256",
    "event_sha256",
}
CAPABILITY_REQUIRED = {
    "schema_version",
    "capability_id",
    "role",
    "campaign_id",
    "actions",
    "targets",
    "issued_at",
    "expires_at",
    "max_uses",
    "nonce",
    "signature",
}
PERMIT_MODES = {
    "diagnostic",
    "calibration",
    "screening",
    "correctness",
    "optimization",
    "confirmation",
    "shape6",
    "shape14",
}


class AuthorityError(RuntimeError):
    """Fail-closed authority or durability error."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime | None = None) -> str:
    current = value or utc_now()
    return current.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise AuthorityError(f"{field} must be a non-empty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuthorityError(f"{field} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise AuthorityError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _assert_plain_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuthorityError(f"{name} must be a JSON object")
    return value


def read_json_exact(path: Path, *, name: str = "JSON document") -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AuthorityError(f"{name} must be a regular non-symlink file: {path}")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as stream:
                raw = stream.read()
        finally:
            os.close(fd)
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"cannot read valid {name} at {path}: {exc}") from exc
    return _assert_plain_object(value, name)


def atomic_write(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    """Durably replace ``path`` without following a caller-controlled temp path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        os.fchmod(fd, mode)
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temp, path)
        fsync_directory(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class AuthorityPaths:
    root: Path

    @property
    def directory(self) -> Path:
        return self.root / "Project" / "authority"

    @property
    def events(self) -> Path:
        return self.directory / "events.jsonl"

    @property
    def lock_file(self) -> Path:
        return self.directory / ".authority.lock"

    @property
    def blobs(self) -> Path:
        return self.directory / "blobs"

    @property
    def public_key(self) -> Path:
        return self.directory / "owner_public_key.pem"

    @property
    def critic_public_key(self) -> Path:
        return self.directory / "critic_public_key.pem"

    @property
    def lock_manifest(self) -> Path:
        return self.directory / "LOCK.json"

    @property
    def lock_signature(self) -> Path:
        return self.directory / "LOCK.sig"


class AuthorityStore:
    def __init__(self, root: Path):
        self.paths = AuthorityPaths(root.resolve())

    def ensure_layout(self) -> None:
        self.paths.directory.mkdir(parents=True, exist_ok=True)
        self.paths.blobs.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.ensure_layout()
        fd = os.open(
            self.paths.lock_file,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _read_events_unlocked(self) -> list[dict[str, Any]]:
        path = self.paths.events
        if not path.exists():
            return []
        if path.is_symlink() or not path.is_file():
            raise AuthorityError("authority journal must be a regular non-symlink file")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as stream:
                    lines = stream.read().splitlines()
            finally:
                os.close(fd)
        except (OSError, UnicodeError) as exc:
            raise AuthorityError(f"cannot read authority journal: {exc}") from exc
        events: list[dict[str, Any]] = []
        previous: str | None = None
        for number, line in enumerate(lines, start=1):
            if not line:
                raise AuthorityError(f"blank authority-journal row at line {number}")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuthorityError(
                    f"malformed authority-journal row at line {number}"
                ) from exc
            event = _assert_plain_object(event, f"event line {number}")
            if set(event) != EVENT_REQUIRED:
                raise AuthorityError(
                    f"event line {number} has unexpected schema keys"
                )
            if event["schema_version"] != AUTHORITY_SCHEMA_VERSION:
                raise AuthorityError(f"unsupported event schema at line {number}")
            if event["prev_event_sha256"] != previous:
                raise AuthorityError(f"broken event chain at line {number}")
            supplied = event["event_sha256"]
            if not isinstance(supplied, str) or len(supplied) != 64:
                raise AuthorityError(f"invalid event hash at line {number}")
            unsigned = {k: v for k, v in event.items() if k != "event_sha256"}
            actual = sha256_bytes(canonical_json(unsigned))
            if actual != supplied:
                raise AuthorityError(f"event hash mismatch at line {number}")
            parse_utc(event["recorded_at"], f"event[{number}].recorded_at")
            if not isinstance(event["payload"], dict):
                raise AuthorityError(f"event payload is not an object at line {number}")
            events.append(event)
            previous = supplied
        return events

    def read_events(self) -> list[dict[str, Any]]:
        with self.locked():
            return self._read_events_unlocked()

    def _append_unlocked(
        self,
        *,
        kind: str,
        actor: str,
        payload: Mapping[str, Any],
        events: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(kind, str) or not kind:
            raise AuthorityError("event kind must be non-empty")
        if not isinstance(actor, str) or not actor:
            raise AuthorityError("event actor must be non-empty")
        prior = list(events) if events is not None else self._read_events_unlocked()
        event: dict[str, Any] = {
            "schema_version": AUTHORITY_SCHEMA_VERSION,
            "event_id": f"evt-{utc_now().strftime('%Y%m%dT%H%M%S.%fZ')}-{secrets.token_hex(6)}",
            "recorded_at": iso_utc(),
            "kind": kind,
            "actor": actor,
            "payload": dict(payload),
            "prev_event_sha256": prior[-1]["event_sha256"] if prior else None,
        }
        event["event_sha256"] = sha256_bytes(canonical_json(event))
        line = canonical_json(event) + b"\n"
        self.ensure_layout()
        fd = os.open(
            self.paths.events,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o640,
        )
        try:
            offset = 0
            while offset < len(line):
                written = os.write(fd, line[offset:])
                if written <= 0:
                    raise AuthorityError("short write to authority journal")
                offset += written
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(self.paths.directory)
        return event

    def append(self, *, kind: str, actor: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.locked():
            events = self._read_events_unlocked()
            return self._append_unlocked(kind=kind, actor=actor, payload=payload, events=events)

    def store_blob(self, data: bytes, *, suffix: str = "") -> tuple[str, Path]:
        digest = sha256_bytes(data)
        if suffix and (not suffix.startswith(".") or "/" in suffix or "\\" in suffix):
            raise AuthorityError("blob suffix must be a simple extension")
        self.ensure_layout()
        path = self.paths.blobs / f"{digest}{suffix}"
        try:
            fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o440,
            )
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise AuthorityError("content-addressed blob path is not a regular file")
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    existing = stream.read()
            finally:
                os.close(fd)
            if existing != data:
                raise AuthorityError("content-addressed blob collision")
            return digest, path
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(fd, data[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(path.parent)
        return digest, path

    def _load_public_key(self, role: str) -> tuple[Ed25519PublicKey, str]:
        if role == "owner":
            path = self.paths.public_key
        elif role == "critic":
            path = self.paths.critic_public_key
        else:
            raise AuthorityError("capability role must be owner or critic")
        try:
            data = path.read_bytes()
            key = serialization.load_pem_public_key(data)
        except Exception as exc:
            raise AuthorityError(f"cannot load owner public key: {exc}") from exc
        if not isinstance(key, Ed25519PublicKey):
            raise AuthorityError("owner public key must be Ed25519")
        raw = key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return key, sha256_bytes(raw)

    def verify_capability(
        self,
        document: Mapping[str, Any],
        *,
        action: str,
        target: str,
        campaign_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        capability = dict(document)
        if set(capability) != CAPABILITY_REQUIRED:
            raise AuthorityError("owner capability has unexpected or missing fields")
        if capability["schema_version"] != AUTHORITY_SCHEMA_VERSION:
            raise AuthorityError("unsupported capability schema")
        for field in ("capability_id", "campaign_id", "nonce"):
            if not isinstance(capability[field], str) or not capability[field]:
                raise AuthorityError(f"capability {field} must be non-empty")
        role = capability["role"]
        if role not in {"owner", "critic"}:
            raise AuthorityError("capability role must be owner or critic")
        if role == "critic" and action != "technical.review":
            raise AuthorityError("critic capability cannot exercise owner authority")
        if role == "owner" and action == "technical.review":
            raise AuthorityError("owner and independent critic roles must remain separate")
        if campaign_id is not None and capability["campaign_id"] != campaign_id:
            raise AuthorityError("capability campaign mismatch")
        actions = capability["actions"]
        targets = capability["targets"]
        if not isinstance(actions, list) or not actions or not all(
            isinstance(v, str) and v for v in actions
        ):
            raise AuthorityError("capability actions must be a non-empty string list")
        if not isinstance(targets, list) or not targets or not all(
            isinstance(v, str) and v for v in targets
        ):
            raise AuthorityError("capability targets must be a non-empty string list")
        if action not in actions:
            raise AuthorityError(f"capability does not authorize {action}")
        target_prefix = target.split(":", 1)[0] + ":*" if ":" in target else "*"
        if target not in targets and target_prefix not in targets and "*" not in targets:
            raise AuthorityError(f"capability does not authorize target {target}")
        max_uses = capability["max_uses"]
        if isinstance(max_uses, bool) or not isinstance(max_uses, int) or max_uses < 1:
            raise AuthorityError("capability max_uses must be a positive integer")
        current = (now or utc_now()).astimezone(timezone.utc)
        issued = parse_utc(capability["issued_at"], "capability.issued_at")
        expires = parse_utc(capability["expires_at"], "capability.expires_at")
        if expires <= issued:
            raise AuthorityError("capability expires_at must follow issued_at")
        if issued > current + timedelta(minutes=5):
            raise AuthorityError("capability is not yet valid")
        if current >= expires:
            raise AuthorityError("capability has expired")
        signature_text = capability["signature"]
        if not isinstance(signature_text, str) or not signature_text:
            raise AuthorityError("capability signature must be base64 text")
        try:
            signature = base64.b64decode(signature_text, validate=True)
        except Exception as exc:
            raise AuthorityError("capability signature is not valid base64") from exc
        unsigned = {k: v for k, v in capability.items() if k != "signature"}
        key, fingerprint = self._load_public_key(role)
        try:
            key.verify(signature, canonical_json(unsigned))
        except InvalidSignature as exc:
            raise AuthorityError("owner capability signature is invalid") from exc
        capability["owner_key_sha256"] = fingerprint
        return capability

    @staticmethod
    def _capability_use_count(events: Sequence[Mapping[str, Any]], capability_id: str) -> int:
        return sum(
            1
            for event in events
            if event.get("payload", {}).get("capability_id") == capability_id
            and event.get("payload", {}).get("capability_consumed") is True
        )

    def append_authorized(
        self,
        *,
        kind: str,
        actor: str,
        payload: Mapping[str, Any],
        capability_document: Mapping[str, Any],
        action: str,
        target: str,
        campaign_id: str | None = None,
    ) -> dict[str, Any]:
        with self.locked():
            events = self._read_events_unlocked()
            capability = self.verify_capability(
                capability_document,
                action=action,
                target=target,
                campaign_id=campaign_id,
            )
            uses = self._capability_use_count(events, capability["capability_id"])
            if uses >= capability["max_uses"]:
                raise AuthorityError("owner capability use limit is exhausted")
            authorized_payload = dict(payload)
            use_number = uses + 1
            receipt_nonce = sha256_bytes(
                canonical_json(
                    {
                        "capability_id": capability["capability_id"],
                        "base_nonce": capability["nonce"],
                        "use_number": use_number,
                        "action": action,
                        "target": target,
                    }
                )
            )
            authorized_payload.update(
                {
                    "capability_consumed": True,
                    "capability_id": capability["capability_id"],
                    # Every authorized use gets a unique receipt nonce even
                    # when a scoped capability deliberately permits multiple
                    # operations.  The signed base nonce remains recorded for
                    # provenance, while consumers can safely enforce one-use
                    # receipts.
                    "capability_nonce": receipt_nonce,
                    "capability_base_nonce": capability["nonce"],
                    "capability_action": action,
                    "capability_target": target,
                    "capability_role": capability["role"],
                    "owner_key_sha256": capability["owner_key_sha256"],
                    "capability_use_number": use_number,
                }
            )
            return self._append_unlocked(
                kind=kind,
                actor=actor,
                payload=authorized_payload,
                events=events,
            )

    def issue_permit(
        self,
        *,
        request: Mapping[str, Any],
        request_blob_sha256: str,
        capability_document: Mapping[str, Any],
    ) -> dict[str, Any]:
        required = {
            "request_sha256",
            "campaign_id",
            "mode",
            "shape_id",
            "candidate_sha256",
            "family_id",
            "expires_at",
        }
        if set(request) != required:
            raise AuthorityError("permit request has unexpected or missing fields")
        if request["request_sha256"] != request_blob_sha256:
            raise AuthorityError("permit request artifact binding mismatch")
        if not isinstance(request_blob_sha256, str) or len(request_blob_sha256) != 64:
            raise AuthorityError("permit request blob SHA-256 is malformed")
        mode = request["mode"]
        if mode not in PERMIT_MODES:
            raise AuthorityError(f"unsupported permit mode {mode!r}")
        shape_id = request["shape_id"]
        if isinstance(shape_id, bool) or not isinstance(shape_id, int) or not 1 <= shape_id <= 14:
            raise AuthorityError("permit shape_id must be 1..14")
        candidate_sha = request["candidate_sha256"]
        if mode == "calibration":
            if candidate_sha is not None:
                raise AuthorityError("calibration permit cannot authorize candidate bytes")
        elif (
            not isinstance(candidate_sha, str)
            or len(candidate_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in candidate_sha)
        ):
            raise AuthorityError(f"{mode} permit requires a candidate SHA-256")
        if mode == "diagnostic" and request["family_id"] is not None:
            raise AuthorityError("diagnostic permit cannot spend a mechanism family")
        expires = parse_utc(request["expires_at"], "permit_request.expires_at")
        if expires <= utc_now() or expires > utc_now() + timedelta(hours=12):
            raise AuthorityError("permit expiry must be within the next 12 hours")
        permit_id = f"permit-{secrets.token_hex(16)}"
        payload = {
            "permit_id": permit_id,
            **dict(request),
            "request_blob_sha256": request_blob_sha256,
            "may_modify_candidate": mode not in {"diagnostic", "calibration"},
            # Screening/correctness and confirmation can establish evidence,
            # but never open promotion authority by themselves.  Promotion is
            # subsequently derived from an eligible optimization/side-lane
            # measurement plus its bound independent audit.
            "may_promote": mode == "optimization",
        }
        target = f"shape:{shape_id}"
        event = self.append_authorized(
            kind="permit_issued",
            actor="trusted-controller",
            payload=payload,
            capability_document=capability_document,
            action="permit.issue",
            target=target,
            campaign_id=str(request["campaign_id"]),
        )
        return {**payload, "authority_event_id": event["event_id"]}

    def consume_permit(
        self,
        *,
        permit_id: str,
        mode: str,
        shape_id: int,
        candidate_sha256: str | None,
    ) -> dict[str, Any]:
        with self.locked():
            events = self._read_events_unlocked()
            issued = [
                event for event in events
                if event["kind"] == "permit_issued"
                and event["payload"].get("permit_id") == permit_id
            ]
            if len(issued) != 1:
                raise AuthorityError("permit is absent or was issued more than once")
            if any(
                event["kind"] == "permit_consumed"
                and event["payload"].get("permit_id") == permit_id
                for event in events
            ):
                raise AuthorityError("permit has already been consumed")
            permit = issued[0]["payload"]
            if permit.get("mode") != mode or permit.get("shape_id") != shape_id:
                raise AuthorityError("permit mode/shape binding mismatch")
            if permit.get("candidate_sha256") != candidate_sha256:
                raise AuthorityError("permit candidate binding mismatch")
            if parse_utc(permit.get("expires_at"), "permit.expires_at") <= utc_now():
                raise AuthorityError("permit has expired")
            consumed = self._append_unlocked(
                kind="permit_consumed",
                actor="trusted-controller",
                payload={
                    "permit_id": permit_id,
                    "issued_event_id": issued[0]["event_id"],
                    "mode": mode,
                    "shape_id": shape_id,
                    "candidate_sha256": candidate_sha256,
                },
                events=events,
            )
            return {**permit, "consumed_event_id": consumed["event_id"]}

    def verify_receipt(
        self,
        *,
        authority_event_id: str,
        action: str,
        subject_sha256: str,
        capability_nonce: str,
    ) -> bool:
        events = self.read_events()
        matches = [event for event in events if event["event_id"] == authority_event_id]
        if len(matches) != 1:
            return False
        payload = matches[0]["payload"]
        return (
            payload.get("capability_consumed") is True
            and payload.get("capability_action") == action
            and payload.get("subject_sha256") == subject_sha256
            and payload.get("capability_nonce") == capability_nonce
        )


def private_key_path_is_external(path: Path, root: Path) -> bool:
    """Owner tooling guard: private material must not live in repo or $HOME."""
    resolved = path.expanduser().resolve()
    workspace = root.resolve()
    home = Path.home().resolve()
    return not resolved.is_relative_to(workspace) and not resolved.is_relative_to(home)
