#!/usr/bin/env python3
"""Fail-closed authority for benchmark audits.

This module is intentionally independent of leaderboard Markdown.  The trusted
controller, manifest builder, and watcher consume the same API and derive audit
state from the primary result journal, the preserved legacy verdict ledger, and
the prospective hash-chained audit event journal.

The legacy ``verdicts.jsonl`` file is read-only input.  New attempts, bound
results, failures, and explicit owner-authorized resolutions are appended to
``audit_events.jsonl`` under one lock with fsync.  An integrity
RULE_VIOLATION/RETEST is first-write-wins until an explicit resolution targets
that exact event hash.  Technical review remains a separate channel:
WEAK_DIAGNOSIS and MISSING_EVIDENCE pause promotion, while
TECHNICAL_DISAGREEMENT is advisory.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import pwd
import re
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
AUDITS_DIR = ROOT / "Project" / "audits"
PRIMARY_JOURNAL = ROOT / "Project" / "results" / "JOURNAL.jsonl"
LEGACY_VERDICTS = AUDITS_DIR / "verdicts.jsonl"
AUDIT_EVENTS = AUDITS_DIR / "audit_events.jsonl"
AUDIT_LOCK = AUDITS_DIR / ".audit_authority.lock"
PACKETS_DIR = AUDITS_DIR / "packets"
AUTHORITY_BLOBS = ROOT / "Project" / "authority" / "blobs"
SCHEMA_PATH = AUDITS_DIR / "verdict_schema.json"

ENTRY_RE = re.compile(
    r"^(?:[0-9]{8}-[0-9]{6}(?:-[0-9a-f]{6})?|run-[0-9a-f]{32})$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
NONCE_RE = SHA_RE
LEGACY_FINAL = frozenset({"PASS", "RETEST", "NEEDS_CONTEXT", "RULE_VIOLATION"})
HARD_INTEGRITY = frozenset({"RETEST", "RULE_VIOLATION"})
BLOCKING_INTEGRITY = frozenset({"RETEST", "NEEDS_CONTEXT", "RULE_VIOLATION"})
BLOCKING_TECHNICAL = frozenset({"WEAK_DIAGNOSIS", "MISSING_EVIDENCE"})
ADVISORY_TECHNICAL = frozenset({"TECHNICAL_DISAGREEMENT"})
MAX_FAILED_ATTEMPTS = 3
AUDIT_LANES = frozenset({"primary", "shape6", "shape14"})


class AuditAuthorityError(RuntimeError):
    """The audit record is absent, malformed, inconsistent, or tampered."""


@dataclass(frozen=True)
class CodexIdentity:
    invoked_path: str
    resolved_path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "invoked_path": self.invoked_path,
            "resolved_path": self.resolved_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class BoundPacket:
    entry_id: str
    path: Path
    sha256: str
    candidate_sha256: str
    payload: dict[str, Any]
    candidate_source_path: Path
    measurement_event_sha256: str | None
    lane: str | None


@dataclass(frozen=True)
class AuditDecision:
    entry_id: str
    promotion_eligible: bool
    integrity_status: str
    technical_status: str
    blocking_reasons: tuple[str, ...]
    effective_event_sha256: str | None
    active_hard_event_sha256: str | None
    candidate_sha256: str | None
    packet_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "promotion_eligible": self.promotion_eligible,
            "integrity_status": self.integrity_status,
            "technical_status": self.technical_status,
            "blocking_reasons": list(self.blocking_reasons),
            "effective_event_sha256": self.effective_event_sha256,
            "active_hard_event_sha256": self.active_hard_event_sha256,
            "candidate_sha256": self.candidate_sha256,
            "packet_sha256": self.packet_sha256,
        }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def _strict_json_object(text: str, label: str) -> dict[str, Any]:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AuditAuthorityError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def no_nonfinite(value: str):
        raise AuditAuthorityError(
            f"{label} contains non-standard JSON constant {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=no_nonfinite,
        )
    except AuditAuthorityError:
        raise
    except json.JSONDecodeError as exc:
        raise AuditAuthorityError(f"{label} is malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditAuthorityError(f"{label} must be one JSON object")
    return value


def _event_digest(event: Mapping[str, Any]) -> str:
    unsigned = dict(event)
    unsigned.pop("event_sha256", None)
    return sha256_bytes(_canonical(unsigned))


def _require_entry_id(entry_id: str) -> None:
    if not isinstance(entry_id, str) or not ENTRY_RE.fullmatch(entry_id):
        raise AuditAuthorityError(f"invalid audit entry id: {entry_id!r}")


def _require_sha(value: str, label: str) -> None:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise AuditAuthorityError(f"{label} must be a lowercase SHA-256")


@contextlib.contextmanager
def authority_lock(lock_path: Path = AUDIT_LOCK) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_events(path: Path = AUDIT_EVENTS) -> list[dict[str, Any]]:
    """Read and verify the complete prospective audit hash chain.

    A malformed line, missing sequence, broken predecessor, or changed byte
    fails closed.  Callers must never silently skip damaged authority rows.
    """
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    previous = "0" * 64
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise AuditAuthorityError(
                f"{path}: blank line {lineno} inside audit authority journal")
        event = _strict_json_object(raw, f"{path}:{lineno} event")
        if event.get("event_version") != 1 or event.get("seq") != lineno:
            raise AuditAuthorityError(f"{path}:{lineno}: invalid version/sequence")
        if event.get("previous_event_sha256") != previous:
            raise AuditAuthorityError(f"{path}:{lineno}: broken predecessor hash")
        digest = _event_digest(event)
        if event.get("event_sha256") != digest:
            raise AuditAuthorityError(f"{path}:{lineno}: event hash mismatch")
        if event.get("event_type") not in {
            "audit_enqueued", "attempt_started", "attempt_failed",
            "audit_result", "resolution"
        }:
            raise AuditAuthorityError(f"{path}:{lineno}: unknown event type")
        events.append(event)
        previous = digest
    return events


def _append_event_locked(payload: Mapping[str, Any],
                         path: Path = AUDIT_EVENTS) -> dict[str, Any]:
    events = read_events(path)
    event = dict(payload)
    forbidden = {"event_version", "seq", "previous_event_sha256", "event_sha256"}
    overlap = forbidden.intersection(event)
    if overlap:
        raise AuditAuthorityError(f"caller supplied reserved event fields: {sorted(overlap)}")
    event["event_version"] = 1
    event["seq"] = len(events) + 1
    event["recorded"] = event.get("recorded") or _now()
    event["previous_event_sha256"] = (
        events[-1]["event_sha256"] if events else "0" * 64
    )
    event["event_sha256"] = _event_digest(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        data = _canonical(event)
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)
    return event


def resolve_codex_identity(invoked_path: Path, pinned_sha256: str,
                           *, home_path: Path | None = None) -> CodexIdentity:
    """Verify one absolute, non-HOME Codex launcher against its pinned bytes."""
    if not invoked_path.is_absolute():
        raise AuditAuthorityError("Codex executable path must be absolute")
    _require_sha(pinned_sha256, "pinned Codex hash")
    try:
        resolved = invoked_path.resolve(strict=True)
    except OSError as exc:
        raise AuditAuthorityError(f"pinned Codex executable is unavailable: {exc}") from exc
    homes: list[Path] = []
    if home_path is not None:
        homes.append(home_path.resolve())
    else:
        homes.append(Path(pwd.getpwuid(os.getuid()).pw_dir).resolve())
    environment_home = os.environ.get("HOME")
    if environment_home and Path(environment_home).is_absolute():
        homes.append(Path(environment_home).resolve())
    for candidate in (invoked_path.resolve(strict=False), resolved):
        for home in homes:
            try:
                candidate.relative_to(home)
            except ValueError:
                continue
            raise AuditAuthorityError("Codex executable under HOME is refused")
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise AuditAuthorityError("pinned Codex path is not an executable file")
    actual = sha256_file(resolved)
    if actual != pinned_sha256:
        raise AuditAuthorityError(
            f"pinned Codex hash mismatch: expected {pinned_sha256}, got {actual}")
    return CodexIdentity(str(invoked_path), str(resolved), actual)


def load_bound_packet(
    entry_id: str,
    *,
    packet_sha256: str | None = None,
    packets_dir: Path = PACKETS_DIR,
    authority_blobs: Path = AUTHORITY_BLOBS,
) -> BoundPacket:
    """Load a prospective content-addressed packet or a preserved legacy one.

    Prospective controller packets live at
    ``Project/authority/blobs/<packet_sha256>.json``.  Their candidate bytes
    are independently content-addressed at ``<candidate_sha256>.py``.  Omitting
    ``packet_sha256`` selects only the legacy packet store for historical
    compatibility; promotion code should always pass the exact queue binding.
    """
    _require_entry_id(entry_id)
    if packet_sha256 is not None:
        _require_sha(packet_sha256, "packet hash")
        try:
            base = authority_blobs.resolve(strict=True)
        except OSError as exc:
            raise AuditAuthorityError("authority blob store is absent") from exc
        path = authority_blobs / f"{packet_sha256}.json"
        if path.is_symlink():
            raise AuditAuthorityError("authority packet blob must not be a symlink")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(base)
        except (OSError, ValueError) as exc:
            raise AuditAuthorityError("authority packet blob is absent or escapes store") from exc
        raw = resolved.read_bytes()
        if sha256_bytes(raw) != packet_sha256:
            raise AuditAuthorityError("authority packet filename/hash binding is broken")
        try:
            packet = _strict_json_object(raw.decode("utf-8"), "authority packet")
        except UnicodeDecodeError as exc:
            raise AuditAuthorityError("authority packet is not UTF-8") from exc
        if packet.get("schema_version") != 1:
            raise AuditAuthorityError("authority packet schema_version must be 1")
        if packet.get("entry_id") != entry_id:
            raise AuditAuthorityError("authority packet entry_id binding is invalid")
        candidate_sha = packet.get("candidate_sha256")
        measurement_sha = packet.get("measurement_event_sha256")
        lane = packet.get("lane")
        _require_sha(candidate_sha, "authority packet candidate hash")
        _require_sha(measurement_sha, "authority packet measurement event hash")
        if measurement_sha == "0" * 64:
            raise AuditAuthorityError(
                "authority packet measurement event hash must not be zero")
        if lane not in AUDIT_LANES:
            raise AuditAuthorityError("authority packet lane is invalid")
        source_path = authority_blobs / f"{candidate_sha}.py"
        if source_path.is_symlink():
            raise AuditAuthorityError("candidate source blob must not be a symlink")
        try:
            source_resolved = source_path.resolve(strict=True)
            source_resolved.relative_to(base)
        except (OSError, ValueError) as exc:
            raise AuditAuthorityError("candidate source blob is absent or escapes store") from exc
        if not source_resolved.is_file() or sha256_file(source_resolved) != candidate_sha:
            raise AuditAuthorityError("candidate source blob hash binding is broken")
        return BoundPacket(
            entry_id=entry_id,
            path=resolved,
            sha256=packet_sha256,
            candidate_sha256=candidate_sha,
            payload=packet,
            candidate_source_path=source_resolved,
            measurement_event_sha256=measurement_sha,
            lane=lane,
        )

    # Preserved legacy packet format: measured source is embedded in the JSON.
    base = packets_dir.resolve(strict=True)
    path = packets_dir / f"{entry_id}.json"
    if path.is_symlink():
        raise AuditAuthorityError("packet must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError) as exc:
        raise AuditAuthorityError(f"packet is absent or escapes packet store: {path}") from exc
    if not resolved.is_file():
        raise AuditAuthorityError("packet must be a regular file")
    raw = resolved.read_bytes()
    try:
        packet = _strict_json_object(raw.decode("utf-8"), "legacy packet")
    except UnicodeDecodeError as exc:
        raise AuditAuthorityError("legacy packet is not UTF-8") from exc
    if not isinstance(packet.get("entry"), dict):
        raise AuditAuthorityError("packet must contain one entry object")
    entry = packet["entry"]
    if entry.get("entry_id") != entry_id:
        raise AuditAuthorityError("packet entry_id does not match requested entry")
    candidate_sha = entry.get("impl", {}).get("sha256")
    _require_sha(candidate_sha, "journaled candidate hash")
    source = packet.get("candidate_source")
    if not isinstance(source, str):
        raise AuditAuthorityError("packet lacks the measured candidate source")
    source_sha = sha256_bytes(source.encode("utf-8"))
    if source_sha != candidate_sha:
        raise AuditAuthorityError(
            "packet candidate source bytes do not match the journaled candidate hash")
    advertised_values = {
        value for value in (
            packet.get("candidate_source_sha256"),
            packet.get("candidate_source_sha256_now"),
        ) if value is not None
    }
    if advertised_values and advertised_values != {source_sha}:
        raise AuditAuthorityError("packet's advertised candidate hash is inconsistent")
    if packet.get("candidate_source_matches_journal") is False:
        raise AuditAuthorityError("packet explicitly reports source/journal drift")
    return BoundPacket(
        entry_id=entry_id,
        path=resolved,
        sha256=sha256_bytes(raw),
        candidate_sha256=source_sha,
        payload=packet,
        candidate_source_path=resolved,
        measurement_event_sha256=None,
        lane=None,
    )


def load_verdict_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    try:
        schema = _strict_json_object(
            path.read_text(encoding="utf-8"), "verdict schema")
    except Exception as exc:
        raise AuditAuthorityError(f"verdict schema unavailable or malformed: {exc}") from exc
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise AuditAuthorityError(f"verdict schema is invalid: {exc}") from exc
    return schema


def validate_verdict_document(
    stdout: str,
    *,
    attempt_nonce: str,
    entry_id: str,
    packet_sha256: str,
    candidate_sha256: str,
    returncode: int = 0,
    schema_path: Path = SCHEMA_PATH,
) -> dict[str, Any]:
    """Accept exactly one complete JSON object with exact attempt bindings."""
    if isinstance(returncode, bool) or returncode != 0:
        raise AuditAuthorityError(f"auditor exited nonzero ({returncode})")
    if not NONCE_RE.fullmatch(attempt_nonce):
        raise AuditAuthorityError("invalid expected attempt nonce")
    try:
        document = _strict_json_object(stdout, "auditor stdout")
    except AuditAuthorityError as exc:
        raise AuditAuthorityError(
            "auditor stdout must be exactly one duplicate-free JSON object "
            "with no banners") from exc
    schema = load_verdict_schema(schema_path)
    try:
        from jsonschema import Draft202012Validator

        errors = sorted(Draft202012Validator(schema).iter_errors(document),
                        key=lambda err: list(err.absolute_path))
    except Exception as exc:
        raise AuditAuthorityError(f"full verdict validation failed: {exc}") from exc
    if errors:
        rendered = "; ".join(
            f"{'/'.join(map(str, err.absolute_path)) or '<root>'}: {err.message}"
            for err in errors[:8]
        )
        raise AuditAuthorityError(f"verdict does not match full schema: {rendered}")
    expected = {
        "attempt_nonce": attempt_nonce,
        "entry_id": entry_id,
        "packet_sha256": packet_sha256,
        "candidate_sha256": candidate_sha256,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise AuditAuthorityError(f"verdict {key} does not match this attempt")
    return document


def exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Create one immutable-named artifact; an existing name is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)
    return sha256_bytes(data)


def store_content_addressed_json(
    payload: Mapping[str, Any], *, directory: Path = AUTHORITY_BLOBS,
    suffix: str = ".audit.json",
) -> tuple[Path, str]:
    """Durably store JSON under its byte hash without overwrite semantics."""
    if not re.fullmatch(r"\.[a-z0-9.-]+", suffix):
        raise AuditAuthorityError("content-addressed artifact suffix is invalid")
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    digest = sha256_bytes(data)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}{suffix}"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o440)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise AuditAuthorityError("content-addressed artifact collision")
        return path, digest
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(directory)
    return path, digest


def _verified_response_artifact(
    artifact_path: str, expected_sha256: str, *, artifact_root: Path,
) -> tuple[Path, dict[str, Any]]:
    _require_sha(expected_sha256, "response artifact hash")
    if not isinstance(artifact_path, str) or not artifact_path \
            or Path(artifact_path).is_absolute():
        raise AuditAuthorityError("response artifact must be a relative repository path")
    allowed_parents = (
        (artifact_root / "Project" / "audits" / "auto").resolve(),
        (artifact_root / "Project" / "authority" / "blobs").resolve(),
    )
    unresolved = artifact_root / artifact_path
    if unresolved.is_symlink():
        raise AuditAuthorityError("response artifact must not be a symlink")
    try:
        resolved = unresolved.resolve(strict=True)
        if not any(resolved.is_relative_to(parent) for parent in allowed_parents):
            raise ValueError("outside artifact stores")
    except (OSError, ValueError) as exc:
        raise AuditAuthorityError(
            "response artifact is absent or outside protected artifact stores") from exc
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise AuditAuthorityError("response artifact bytes do not match recorded hash")
    try:
        payload = _strict_json_object(
            resolved.read_text(encoding="utf-8"), "response artifact")
    except Exception as exc:
        raise AuditAuthorityError(f"response artifact is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("artifact_type") != "audit_response":
        raise AuditAuthorityError("response artifact has the wrong type")
    return resolved, payload


def enqueue_audit(
    *,
    entry_id: str,
    candidate_sha256: str,
    packet_sha256: str,
    measurement_event_sha256: str,
    lane: str,
    path: Path = AUDIT_EVENTS,
    lock_path: Path = AUDIT_LOCK,
) -> dict[str, Any]:
    """Durably enqueue one exact controller measurement for independent audit.

    Repeating the identical call returns the original queue event.  Reusing an
    entry id with different candidate, packet, measurement, or lane bindings is
    refused rather than silently replacing authority state.
    """
    _require_entry_id(entry_id)
    _require_sha(candidate_sha256, "candidate hash")
    _require_sha(packet_sha256, "packet hash")
    _require_sha(measurement_event_sha256, "measurement event hash")
    if measurement_event_sha256 == "0" * 64:
        raise AuditAuthorityError("measurement event hash must not be the zero sentinel")
    if lane not in AUDIT_LANES:
        raise AuditAuthorityError(
            f"audit lane must be one of {sorted(AUDIT_LANES)}")
    requested = {
        "entry_id": entry_id,
        "candidate_sha256": candidate_sha256,
        "packet_sha256": packet_sha256,
        "measurement_event_sha256": measurement_event_sha256,
        "lane": lane,
    }
    with authority_lock(lock_path):
        events = read_events(path)
        prior = [e for e in events if e.get("event_type") == "audit_enqueued"
                 and e.get("entry_id") == entry_id]
        if prior:
            if len(prior) != 1 or any(
                prior[0].get(key) != value for key, value in requested.items()
            ):
                raise AuditAuthorityError(
                    "entry id was already queued with different audit bindings")
            return prior[0]
        return _append_event_locked({"event_type": "audit_enqueued", **requested}, path)


def require_audit_enqueue(
    *,
    entry_id: str,
    candidate_sha256: str,
    packet_sha256: str,
    measurement_event_sha256: str,
    lane: str,
    events_path: Path = AUDIT_EVENTS,
) -> dict[str, Any]:
    """Return the one exact durable queue binding or fail closed.

    This is the controller/manifest-facing verifier for primary and side-lane
    evidence.  It deliberately compares every binding field; the presence of
    an audit result alone is not proof that the audited packet belongs to the
    selected measurement.
    """
    _require_entry_id(entry_id)
    _require_sha(candidate_sha256, "candidate hash")
    _require_sha(packet_sha256, "packet hash")
    _require_sha(measurement_event_sha256, "measurement event hash")
    if lane not in AUDIT_LANES:
        raise AuditAuthorityError(
            f"audit lane must be one of {sorted(AUDIT_LANES)}")
    matches = [
        event for event in read_events(events_path)
        if event.get("event_type") == "audit_enqueued"
        and event.get("entry_id") == entry_id
    ]
    if len(matches) != 1:
        raise AuditAuthorityError(
            "exactly one durable audit enqueue is required for this entry")
    event = matches[0]
    expected = {
        "candidate_sha256": candidate_sha256,
        "packet_sha256": packet_sha256,
        "measurement_event_sha256": measurement_event_sha256,
        "lane": lane,
    }
    for key, value in expected.items():
        if event.get(key) != value:
            raise AuditAuthorityError(
                f"audit enqueue {key} does not match selected evidence")
    return event


def record_attempt_started(
    *, entry_id: str, attempt_id: str, attempt_nonce: str,
    packet_sha256: str, candidate_sha256: str, codex: CodexIdentity,
    path: Path = AUDIT_EVENTS, lock_path: Path = AUDIT_LOCK,
    measurement_event_sha256: str = "0" * 64,
    lane: str = "legacy-primary", queue_event_sha256: str = "",
) -> dict[str, Any]:
    _require_entry_id(entry_id)
    _require_sha(attempt_nonce, "attempt nonce")
    _require_sha(packet_sha256, "packet hash")
    _require_sha(candidate_sha256, "candidate hash")
    _require_sha(measurement_event_sha256, "measurement event hash")
    if lane not in AUDIT_LANES | {"legacy-primary"}:
        raise AuditAuthorityError(f"unknown audit lane: {lane}")
    if queue_event_sha256:
        _require_sha(queue_event_sha256, "queue event hash")
    _require_sha(codex.sha256, "Codex executable hash")
    if not Path(codex.invoked_path).is_absolute() or not Path(codex.resolved_path).is_absolute():
        raise AuditAuthorityError("recorded Codex paths must be absolute")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise AuditAuthorityError("attempt_id is required")
    with authority_lock(lock_path):
        events = read_events(path)
        if entry_id.startswith("run-") and lane not in AUDIT_LANES:
            raise AuditAuthorityError(
                "controller run ids require a prospective audit lane")
        if lane in AUDIT_LANES:
            if not queue_event_sha256:
                raise AuditAuthorityError(
                    "prospective audit attempts require the exact queue event hash")
            queue_matches = [
                event for event in events
                if event.get("event_type") == "audit_enqueued"
                and event.get("event_sha256") == queue_event_sha256
            ]
            if len(queue_matches) != 1:
                raise AuditAuthorityError(
                    "prospective audit queue event is absent or duplicated")
            queue = queue_matches[0]
            queue_expected = {
                "entry_id": entry_id,
                "packet_sha256": packet_sha256,
                "candidate_sha256": candidate_sha256,
                "measurement_event_sha256": measurement_event_sha256,
                "lane": lane,
            }
            if any(queue.get(key) != value
                   for key, value in queue_expected.items()):
                raise AuditAuthorityError(
                    "prospective attempt does not match its durable queue binding")
        elif queue_event_sha256:
            raise AuditAuthorityError(
                "legacy audit attempts cannot claim a prospective queue event")
        if any(e.get("attempt_id") == attempt_id for e in events):
            raise AuditAuthorityError(f"duplicate attempt id: {attempt_id}")
        terminal = {
            e.get("attempt_id") for e in events
            if e.get("event_type") in {"attempt_failed", "audit_result"}
        }
        if any(e.get("event_type") == "attempt_started"
               and e.get("attempt_id") not in terminal for e in events):
            raise AuditAuthorityError(
                "another audit attempt is already active; one auditor at a time")
        if any(
            e.get("event_type") == "audit_result"
            and e.get("entry_id") == entry_id
            and e.get("candidate_sha256") == candidate_sha256 for e in events
        ):
            raise AuditAuthorityError(
                "this entry/candidate already has a bound result; create a new "
                "controller measurement entry for any independent re-review")
        return _append_event_locked({
            "event_type": "attempt_started",
            "entry_id": entry_id,
            "attempt_id": attempt_id,
            "attempt_nonce_sha256": sha256_bytes(attempt_nonce.encode("ascii")),
            "packet_sha256": packet_sha256,
            "candidate_sha256": candidate_sha256,
            "measurement_event_sha256": measurement_event_sha256,
            "lane": lane,
            "queue_event_sha256": queue_event_sha256,
            "verdict_schema_sha256": sha256_file(SCHEMA_PATH),
            "codex": codex.as_dict(),
        }, path)


def _attempt_start(events: Sequence[Mapping[str, Any]],
                   attempt_id: str) -> Mapping[str, Any]:
    starts = [e for e in events if e.get("event_type") == "attempt_started"
              and e.get("attempt_id") == attempt_id]
    if len(starts) != 1:
        raise AuditAuthorityError(f"attempt {attempt_id!r} has {len(starts)} starts")
    if any(e.get("attempt_id") == attempt_id and e.get("event_type") in {
        "attempt_failed", "audit_result"
    } for e in events):
        raise AuditAuthorityError(f"attempt {attempt_id!r} already has a terminal event")
    return starts[0]


def record_attempt_failure(
    *, attempt_id: str, reason: str, artifact_path: str = "",
    artifact_sha256: str = "", path: Path = AUDIT_EVENTS,
    lock_path: Path = AUDIT_LOCK,
) -> dict[str, Any]:
    with authority_lock(lock_path):
        events = read_events(path)
        start = _attempt_start(events, attempt_id)
        if artifact_sha256:
            _require_sha(artifact_sha256, "failure artifact hash")
        return _append_event_locked({
            "event_type": "attempt_failed",
            "entry_id": start["entry_id"],
            "attempt_id": attempt_id,
            "reason": str(reason)[:4000],
            "artifact_path": artifact_path,
            "artifact_sha256": artifact_sha256,
            "codex": start["codex"],
        }, path)


def record_audit_result(
    *, attempt_id: str, result: Mapping[str, Any], artifact_path: str,
    artifact_sha256: str, path: Path = AUDIT_EVENTS,
    lock_path: Path = AUDIT_LOCK, artifact_root: Path = ROOT,
) -> dict[str, Any]:
    _, artifact = _verified_response_artifact(
        artifact_path, artifact_sha256, artifact_root=artifact_root)
    with authority_lock(lock_path):
        events = read_events(path)
        start = _attempt_start(events, attempt_id)
        for key in ("entry_id", "packet_sha256", "candidate_sha256"):
            if result.get(key) != start.get(key):
                raise AuditAuthorityError(f"result {key} is not bound to attempt")
        nonce = result.get("attempt_nonce")
        if not isinstance(nonce, str) or sha256_bytes(nonce.encode("utf-8")) != start.get(
                "attempt_nonce_sha256"):
            raise AuditAuthorityError("result nonce is not bound to attempt")
        schema = load_verdict_schema()
        try:
            from jsonschema import Draft202012Validator

            errors = list(Draft202012Validator(schema).iter_errors(dict(result)))
        except Exception as exc:
            raise AuditAuthorityError(f"full result schema validation failed: {exc}") from exc
        if errors:
            raise AuditAuthorityError(
                "result failed full schema validation: " + errors[0].message)
        artifact_expectations = {
            "attempt_id": attempt_id,
            "entry_id": start["entry_id"],
            "packet_sha256": start["packet_sha256"],
            "candidate_sha256": start["candidate_sha256"],
            "measurement_event_sha256": start.get("measurement_event_sha256"),
            "lane": start.get("lane"),
            "verdict_schema_sha256": start.get("verdict_schema_sha256"),
            "codex": start["codex"],
            "validated_result": dict(result),
        }
        for key, expected in artifact_expectations.items():
            if artifact.get(key) != expected:
                raise AuditAuthorityError(
                    f"response artifact {key} is not bound to this attempt")
        raw_stdout = artifact.get("stdout")
        if not isinstance(raw_stdout, str) or artifact.get("parser_error") != "":
            raise AuditAuthorityError(
                "response artifact does not contain a clean raw auditor response")
        reparsed = validate_verdict_document(
            raw_stdout,
            attempt_nonce=nonce,
            entry_id=start["entry_id"],
            packet_sha256=start["packet_sha256"],
            candidate_sha256=start["candidate_sha256"],
            returncode=artifact.get("returncode"),
        )
        if reparsed != dict(result):
            raise AuditAuthorityError(
                "validated result differs from the exact raw auditor response")
        return _append_event_locked({
            "event_type": "audit_result",
            "entry_id": start["entry_id"],
            "attempt_id": attempt_id,
            "packet_sha256": start["packet_sha256"],
            "candidate_sha256": start["candidate_sha256"],
            "measurement_event_sha256": start.get("measurement_event_sha256"),
            "lane": start.get("lane"),
            "queue_event_sha256": start.get("queue_event_sha256"),
            "verdict_schema_sha256": start.get("verdict_schema_sha256"),
            "response_artifact": artifact_path,
            "response_sha256": artifact_sha256,
            "codex": start["codex"],
            "result": dict(result),
        }, path)


def record_resolution(
    *, entry_id: str, target_event_sha256: str, resolution_kind: str,
    rationale: str, authority_event_id: str, capability_nonce: str,
    superseding_event_sha256: str = "", path: Path = AUDIT_EVENTS,
    lock_path: Path = AUDIT_LOCK, legacy_path: Path = LEGACY_VERDICTS,
    authority_root: Path = ROOT, journal_path: Path = PRIMARY_JOURNAL,
) -> dict[str, Any]:
    """Append an explicit resolution after external owner auth was verified.

    The referenced authority event must be an already-consumed, signed owner
    capability for action ``audit.resolve`` whose subject is the exact target
    verdict hash.  Pasted owner text and workspace files are never accepted as
    authentication.
    """
    _require_entry_id(entry_id)
    _require_sha(target_event_sha256, "target event hash")
    if not isinstance(authority_event_id, str) or not authority_event_id:
        raise AuditAuthorityError("authority event id is required")
    if not isinstance(capability_nonce, str) or not capability_nonce:
        raise AuditAuthorityError("capability nonce is required")
    if superseding_event_sha256:
        _require_sha(superseding_event_sha256, "superseding event hash")
    allowed = {"FINDING_OVERTURNED", "RETEST_SATISFIED", "TECHNICAL_SUPERSEDED"}
    if resolution_kind not in allowed:
        raise AuditAuthorityError(f"unknown resolution kind: {resolution_kind}")
    if not rationale.strip():
        raise AuditAuthorityError("resolution rationale must not be empty")
    with authority_lock(lock_path):
        events = read_events(path)
        target = next((e for e in events
                       if e.get("event_sha256") == target_event_sha256), None)
        if target is None:
            target = next((e for e in _legacy_records(legacy_path)
                           if e.get("event_sha256") == target_event_sha256), None)
        if target is None or target.get("entry_id") != entry_id:
            raise AuditAuthorityError("resolution target is absent or belongs to another entry")
        if target.get("event_type") not in {"audit_result", "legacy_verdict"}:
            raise AuditAuthorityError("only an audit verdict can be resolved")
        if target.get("event_type") == "audit_result":
            target_integrity = target.get("result", {}).get("integrity", {}).get("verdict")
            target_technical = target.get("result", {}).get(
                "technical_review", {}).get("verdict")
        else:
            target_integrity = target.get("integrity_verdict")
            target_technical = None
        compatible = {
            "FINDING_OVERTURNED": target_integrity == "RULE_VIOLATION",
            "RETEST_SATISFIED": target_integrity == "RETEST",
            "TECHNICAL_SUPERSEDED": target_technical in BLOCKING_TECHNICAL,
        }
        if not compatible[resolution_kind]:
            raise AuditAuthorityError(
                f"{resolution_kind} cannot resolve this verdict type")
        if resolution_kind in {"RETEST_SATISFIED", "TECHNICAL_SUPERSEDED"} \
                and not superseding_event_sha256:
            raise AuditAuthorityError(
                f"{resolution_kind} requires a bound superseding audit result")
        if any(e.get("event_type") == "resolution"
               and e.get("target_event_sha256") == target_event_sha256
               for e in events):
            raise AuditAuthorityError("audit result already has a resolution")
        if superseding_event_sha256:
            superseding = next((
                e for e in events
                if e.get("event_sha256") == superseding_event_sha256
                and e.get("event_type") == "audit_result"
            ), None)
            if superseding is None:
                raise AuditAuthorityError("superseding audit result is absent")
            if target.get("event_type") == "audit_result":
                target_candidate_sha = target.get("candidate_sha256")
            else:
                target_row = next((
                    row for row in _read_result_journal(journal_path)
                    if row.get("entry_id") == entry_id
                ), None)
                target_candidate_sha = (target_row or {}).get("impl", {}).get("sha256")
            if not target_candidate_sha:
                raise AuditAuthorityError(
                    "cannot establish target candidate bytes for superseding review")
            if superseding.get("candidate_sha256") != target_candidate_sha:
                raise AuditAuthorityError("superseding result covers different candidate bytes")
            superseding_integrity = superseding.get("result", {}).get(
                "integrity", {}).get("verdict")
            superseding_technical = superseding.get("result", {}).get(
                "technical_review", {}).get("verdict")
            if resolution_kind == "RETEST_SATISFIED" \
                    and superseding_integrity != "PASS":
                raise AuditAuthorityError(
                    "RETEST_SATISFIED requires a superseding integrity PASS")
            if resolution_kind == "TECHNICAL_SUPERSEDED" \
                    and superseding_technical not in {"PASS", "TECHNICAL_DISAGREEMENT"}:
                raise AuditAuthorityError(
                    "TECHNICAL_SUPERSEDED requires a nonblocking technical review")
        harness = authority_root.resolve() / "Project" / "harness"
        if str(harness) not in sys.path:
            sys.path.insert(0, str(harness))
        try:
            from authority import AuthorityStore

            store = AuthorityStore(authority_root)
            if not store.verify_receipt(
                authority_event_id=authority_event_id,
                action="audit.resolve",
                subject_sha256=target_event_sha256,
                capability_nonce=capability_nonce,
            ):
                raise AuditAuthorityError(
                    "owner authority receipt is absent, invalid, or not bound "
                    "to this verdict")
            authority_matches = [
                event for event in store.read_events()
                if event.get("event_id") == authority_event_id
            ]
            if len(authority_matches) != 1:
                raise AuditAuthorityError("owner authority event is not unique")
            authority_event = authority_matches[0]
            authority_payload = authority_event.get("payload", {})
            if authority_event.get("kind") != "audit_resolve_authorized" \
                    or authority_event.get("actor") != "trusted-controller" \
                    or authority_payload.get("capability_role") != "owner" \
                    or authority_payload.get("capability_target") != f"audit:{entry_id}" \
                    or not SHA_RE.fullmatch(str(
                        authority_payload.get("owner_key_sha256", ""))):
                raise AuditAuthorityError(
                    "resolution receipt was not created by the signed owner "
                    "authorization path")
            authority_event_sha256 = authority_event["event_sha256"]
        except AuditAuthorityError:
            raise
        except Exception as exc:
            raise AuditAuthorityError(
                f"cannot verify owner resolution authority: {exc}") from exc
        return _append_event_locked({
            "event_type": "resolution",
            "entry_id": entry_id,
            "target_event_sha256": target_event_sha256,
            "resolution_kind": resolution_kind,
            "rationale": rationale[:4000],
            "authority_event_id": authority_event_id,
            "authority_event_sha256": authority_event_sha256,
            "capability_nonce_sha256": sha256_bytes(
                capability_nonce.encode("utf-8")),
            "superseding_event_sha256": superseding_event_sha256,
            "superseding_entry_id": (
                superseding.get("entry_id") if superseding_event_sha256 else ""),
        }, path)


def _legacy_records(path: Path = LEGACY_VERDICTS) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise AuditAuthorityError(f"{path}:{lineno}: blank legacy verdict line")
        row = _strict_json_object(raw, f"{path}:{lineno} legacy verdict")
        _require_entry_id(row.get("entry_id"))
        verdict = row.get("verdict")
        if verdict not in LEGACY_FINAL | {"JUDGE_ERROR", "TIMEOUT"}:
            raise AuditAuthorityError(f"{path}:{lineno}: unknown legacy verdict")
        records.append({
            "event_type": "legacy_verdict",
            "entry_id": row["entry_id"],
            "integrity_verdict": verdict,
            "event_sha256": sha256_bytes(raw.encode("utf-8")),
            "legacy": True,
            "row": row,
        })
    return records


def audit_decision(
    entry_id: str,
    candidate_sha256: str | None = None,
    packet_sha256: str | None = None,
    *,
    events_path: Path = AUDIT_EVENTS,
    legacy_path: Path = LEGACY_VERDICTS,
    artifact_root: Path = ROOT,
) -> AuditDecision:
    """Return the fail-closed promotion decision for one exact candidate.

    Callers that can promote or ship must supply ``candidate_sha256``.  A
    legacy unbound PASS remains evidence history but is never prospective
    promotion authority.
    """
    _require_entry_id(entry_id)
    if candidate_sha256 is not None:
        _require_sha(candidate_sha256, "candidate hash")
    if packet_sha256 is not None:
        _require_sha(packet_sha256, "packet hash")
    events = read_events(events_path)
    legacy = [r for r in _legacy_records(legacy_path) if r["entry_id"] == entry_id]
    all_results = [e for e in events if e.get("event_type") == "audit_result"]
    results = [e for e in all_results if e.get("entry_id") == entry_id]
    resolutions = {
        e.get("target_event_sha256") for e in events
        if e.get("event_type") == "resolution"
    }
    reasons: list[str] = []
    if candidate_sha256 is None:
        reasons.append("candidate_sha256_not_supplied")
    bound_results = []
    for result in results:
        if candidate_sha256 is not None and result.get("candidate_sha256") != candidate_sha256:
            continue
        if packet_sha256 is not None and result.get("packet_sha256") != packet_sha256:
            continue
        bound_results.append(result)

    # A prospective hard finding binds the candidate bytes, not merely the
    # controller-generated entry id.  Re-enqueuing identical bytes under a new
    # run id therefore cannot evade a RULE_VIOLATION or RETEST.  Legacy rows do
    # not carry trustworthy candidate binding and remain entry-scoped.
    hard_result_scope = (
        [result for result in all_results
         if result.get("candidate_sha256") == candidate_sha256]
        if candidate_sha256 is not None else results
    )
    hard_records: list[tuple[str, str]] = []
    for record in legacy:
        if record["integrity_verdict"] in HARD_INTEGRITY:
            hard_records.append((record["event_sha256"], record["integrity_verdict"]))
    for result in hard_result_scope:
        verdict = result.get("result", {}).get("integrity", {}).get("verdict")
        if verdict in HARD_INTEGRITY:
            hard_records.append((result["event_sha256"], verdict))
    active_hard = next(((event_hash, verdict) for event_hash, verdict in hard_records
                        if event_hash not in resolutions), None)
    if active_hard is not None:
        reasons.append(f"unresolved_first_hard_verdict:{active_hard[1]}")

    effective = bound_results[-1] if bound_results else None
    if effective is not None:
        integrity = effective["result"]["integrity"]["verdict"]
        technical = effective["result"]["technical_review"]["verdict"]
        if integrity in BLOCKING_INTEGRITY:
            reasons.append(f"integrity:{integrity}")
        if technical in BLOCKING_TECHNICAL:
            reasons.append(f"technical:{technical}")
        effective_lane = effective.get("lane")
        if entry_id.startswith("run-") and effective_lane not in AUDIT_LANES:
            reasons.append("controller_entry_lacks_prospective_lane_binding")
        if effective_lane in AUDIT_LANES:
            try:
                bound_packet = load_bound_packet(
                    entry_id,
                    packet_sha256=effective["packet_sha256"],
                    authority_blobs=(
                        artifact_root / "Project" / "authority" / "blobs"),
                )
                packet_expected = {
                    "candidate_sha256": effective["candidate_sha256"],
                    "measurement_event_sha256": effective[
                        "measurement_event_sha256"],
                    "lane": effective_lane,
                }
                if any(getattr(bound_packet, key) != value
                       for key, value in packet_expected.items()):
                    raise AuditAuthorityError(
                        "bound packet fields differ from the audit result")
                queue = require_audit_enqueue(
                    entry_id=entry_id,
                    candidate_sha256=effective["candidate_sha256"],
                    packet_sha256=effective["packet_sha256"],
                    measurement_event_sha256=effective["measurement_event_sha256"],
                    lane=effective_lane,
                    events_path=events_path,
                )
                if effective.get("queue_event_sha256") != queue["event_sha256"]:
                    raise AuditAuthorityError(
                        "audit result does not name its exact enqueue event")
            except AuditAuthorityError as exc:
                reasons.append(f"prospective_binding_invalid:{exc}")
        try:
            _verified_response_artifact(
                effective.get("response_artifact", ""),
                effective.get("response_sha256", ""),
                artifact_root=artifact_root,
            )
        except AuditAuthorityError as exc:
            reasons.append(f"response_artifact_invalid:{exc}")
        if effective.get("verdict_schema_sha256") != sha256_file(SCHEMA_PATH):
            reasons.append("verdict_schema_changed_since_audit")
    elif legacy:
        integrity = legacy[-1]["integrity_verdict"]
        technical = "LEGACY_UNBOUND"
        reasons.append("legacy_verdict_lacks_nonce_packet_and_candidate_binding")
        if integrity in BLOCKING_INTEGRITY:
            reasons.append(f"integrity:{integrity}")
    else:
        integrity = "MISSING"
        technical = "MISSING"
        reasons.append("missing_audit_verdict")

    eligible = (
        not reasons
        and effective is not None
        and integrity == "PASS"
        and technical in ({"PASS"} | ADVISORY_TECHNICAL)
    )
    return AuditDecision(
        entry_id=entry_id,
        promotion_eligible=eligible,
        integrity_status=integrity,
        technical_status=technical,
        blocking_reasons=tuple(dict.fromkeys(reasons)),
        effective_event_sha256=(effective or {}).get("event_sha256"),
        active_hard_event_sha256=active_hard[0] if active_hard else None,
        candidate_sha256=candidate_sha256,
        packet_sha256=(effective or {}).get("packet_sha256"),
    )


def _read_result_journal(path: Path = PRIMARY_JOURNAL) -> list[dict[str, Any]]:
    if not path.exists():
        raise AuditAuthorityError(f"primary result journal is missing: {path}")
    rows = []
    ids: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise AuditAuthorityError(f"{path}:{lineno}: blank journal line")
        row = _strict_json_object(raw, f"{path}:{lineno} result row")
        entry_id = row.get("entry_id")
        _require_entry_id(entry_id)
        digest = sha256_bytes(raw.encode("utf-8"))
        if entry_id in ids and ids[entry_id] != digest:
            raise AuditAuthorityError(f"conflicting duplicate journal id: {entry_id}")
        ids[entry_id] = digest
        rows.append(row)
    return rows


def audit_candidates(journal_path: Path = PRIMARY_JOURNAL) -> list[dict[str, str]]:
    """Derive the durable audit queue from result rows, never Markdown.

    Every legacy ``promoted`` candidate is included conservatively.  The new
    controller may set ``promotion_candidate`` or ``audit_required`` before a
    result becomes champion, avoiding the audit/eligibility circularity.
    """
    found: dict[str, dict[str, str]] = {}
    for row in _read_result_journal(journal_path):
        if row.get("type") != "candidate":
            continue
        if not (row.get("promoted") or row.get("promotion_candidate")
                or row.get("audit_required")):
            continue
        entry_id = row["entry_id"]
        candidate_sha = row.get("impl", {}).get("sha256")
        _require_sha(candidate_sha, f"candidate hash for {entry_id}")
        found[entry_id] = {
            "entry_id": entry_id,
            "candidate_sha256": candidate_sha,
            "shape_id": str(row.get("shape_id", "")),
            "packet_sha256": "",
            "measurement_event_sha256": "",
            "lane": "legacy-primary",
            "queue_event_sha256": "",
        }
    return list(found.values())


def required_audit_candidates(
    *, journal_path: Path = PRIMARY_JOURNAL,
    events_path: Path = AUDIT_EVENTS,
) -> list[dict[str, str]]:
    """Union legacy journal candidates with controller-enqueued all-lane work."""
    legacy = {row["entry_id"]: row for row in audit_candidates(journal_path)}
    found: dict[str, dict[str, str]] = {}
    for event in read_events(events_path):
        if event.get("event_type") != "audit_enqueued":
            continue
        entry_id = event["entry_id"]
        if entry_id in found:
            raise AuditAuthorityError(
                f"duplicate prospective audit enqueue for {entry_id}")
        queued = {
            "entry_id": entry_id,
            "candidate_sha256": event["candidate_sha256"],
            "shape_id": event["lane"].removeprefix("shape"),
            "packet_sha256": event["packet_sha256"],
            "measurement_event_sha256": event["measurement_event_sha256"],
            "lane": event["lane"],
            "queue_event_sha256": event["event_sha256"],
        }
        prior = legacy.get(entry_id)
        if prior is not None and prior["candidate_sha256"] != queued["candidate_sha256"]:
            raise AuditAuthorityError(
                f"journal and queue disagree on candidate bytes for {entry_id}")
        found[entry_id] = {**(prior or {}), **queued}
    # Prospective controller queue is ordered first so a historical backlog can
    # never starve the current grind's audit.  Legacy rows remain durable work.
    for entry_id, row in legacy.items():
        found.setdefault(entry_id, row)
    return list(found.values())


def attempt_summary(entry_id: str, *, events_path: Path = AUDIT_EVENTS) -> dict[str, Any]:
    _require_entry_id(entry_id)
    events = [e for e in read_events(events_path) if e.get("entry_id") == entry_id]
    starts = [e for e in events if e.get("event_type") == "attempt_started"]
    terminal = {e.get("attempt_id"): e for e in events
                if e.get("event_type") in {"attempt_failed", "audit_result"}}
    active = [e for e in starts if e.get("attempt_id") not in terminal]
    failures = [e for e in terminal.values() if e.get("event_type") == "attempt_failed"]
    results = [e for e in terminal.values() if e.get("event_type") == "audit_result"]
    return {
        "attempts": len(starts),
        "failed_attempts": len(failures),
        "has_result": bool(results),
        "active_attempt_ids": [e["attempt_id"] for e in active],
        "retry_exhausted": not results and len(failures) >= MAX_FAILED_ATTEMPTS,
    }


def pending_audit_entries(
    *, journal_path: Path = PRIMARY_JOURNAL,
    events_path: Path = AUDIT_EVENTS,
    legacy_path: Path = LEGACY_VERDICTS,
) -> list[dict[str, Any]]:
    """Return retryable candidates, reconstructing correctly from zero state."""
    out = []
    legacy_final_ids = {
        record["entry_id"] for record in _legacy_records(legacy_path)
        if record["integrity_verdict"] in LEGACY_FINAL
    }
    for candidate in required_audit_candidates(
            journal_path=journal_path, events_path=events_path):
        # Preserved legacy final rows settle only the historical backlog.  They
        # never authorize prospective promotion, and never settle a new
        # controller enqueue even if an entry id were somehow reused.
        if not candidate.get("packet_sha256") \
                and candidate["entry_id"] in legacy_final_ids:
            continue
        summary = attempt_summary(candidate["entry_id"], events_path=events_path)
        if summary["has_result"] or summary["active_attempt_ids"]:
            continue
        if summary["retry_exhausted"]:
            continue
        out.append({**candidate, **summary})
    return out


def owner_attention_entries(
    *, journal_path: Path = PRIMARY_JOURNAL,
    events_path: Path = AUDIT_EVENTS,
    legacy_path: Path = LEGACY_VERDICTS,
) -> list[dict[str, Any]]:
    out = []
    legacy_final_ids = {
        record["entry_id"] for record in _legacy_records(legacy_path)
        if record["integrity_verdict"] in LEGACY_FINAL
    }
    for candidate in required_audit_candidates(
            journal_path=journal_path, events_path=events_path):
        if not candidate.get("packet_sha256") \
                and candidate["entry_id"] in legacy_final_ids:
            continue
        summary = attempt_summary(candidate["entry_id"], events_path=events_path)
        if summary["retry_exhausted"]:
            out.append({**candidate, **summary})
    return out


def active_attempts(*, events_path: Path = AUDIT_EVENTS) -> list[dict[str, Any]]:
    events = read_events(events_path)
    terminal = {e.get("attempt_id") for e in events
                if e.get("event_type") in {"attempt_failed", "audit_result"}}
    return [e for e in events if e.get("event_type") == "attempt_started"
            and e.get("attempt_id") not in terminal]


def audit_status_map(
    entries: Iterable[Mapping[str, Any]], *, events_path: Path = AUDIT_EVENTS,
    legacy_path: Path = LEGACY_VERDICTS, artifact_root: Path = ROOT,
) -> dict[str, AuditDecision]:
    """Convenience API for a controller/manifest evaluating several rows."""
    out = {}
    for entry in entries:
        entry_id = str(entry.get("entry_id", ""))
        candidate_sha = (
            entry.get("candidate_sha256")
            or entry.get("impl", {}).get("sha256")
        )
        packet_sha = (
            entry.get("packet_sha256")
            or entry.get("audit_packet_sha256")
        )
        out[entry_id] = audit_decision(
            entry_id, candidate_sha, packet_sha,
            events_path=events_path, legacy_path=legacy_path,
            artifact_root=artifact_root,
        )
    return out
