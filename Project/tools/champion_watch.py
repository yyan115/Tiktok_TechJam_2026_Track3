#!/usr/bin/env python3
"""Durable audit watcher derived from the result journal, never Markdown.

From a completely empty prospective audit-event journal this watcher
reconstructs every audit-required candidate from JOURNAL.jsonl.  Attempts and
retry failures are durable events, so deleting a cache file, dethroning a
candidate, or restarting a session cannot suppress an audit.  At most one
auditor runs at once to avoid inflating launch-bound benchmark baselines.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Sequence

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_authority import (
    AUDIT_EVENTS,
    LEGACY_VERDICTS,
    PRIMARY_JOURNAL,
    ROOT,
    AuditAuthorityError,
    active_attempts,
    exclusive_write_json,
    load_bound_packet,
    owner_attention_entries,
    pending_audit_entries,
    read_events,
    record_attempt_failure,
    record_attempt_started,
    resolve_codex_identity,
    store_content_addressed_json,
)
from audit_champion import (
    AUDIT_LOG_DIR,
    CODEX_EXECUTABLE,
    PINNED_CODEX_SHA256,
    marker_path,
)

AUDITOR = Path(__file__).with_name("audit_champion.py")
RUN_GATE = Path(__file__).with_name("run_gate.py")
STALE_ATTEMPT_SECONDS = 300
MAX_CONCURRENT_AUDITS = 1


def queue_snapshot(*, journal_path: Path = PRIMARY_JOURNAL,
                   events_path: Path = AUDIT_EVENTS,
                   legacy_path: Path = LEGACY_VERDICTS) -> dict:
    """Testable cold-start reconstruction from durable journals only."""
    pending = pending_audit_entries(
        journal_path=journal_path, events_path=events_path,
        legacy_path=legacy_path)
    owner_attention = owner_attention_entries(
        journal_path=journal_path, events_path=events_path,
        legacy_path=legacy_path)
    active = active_attempts(events_path=events_path)
    return {
        "pending_rows": pending,
        "owner_attention_rows": owner_attention,
        "active_rows": active,
    }


def run_gate_post() -> None:
    """Reconcile the controller after shell activity; failure remains closed."""
    try:
        subprocess.run(
            [sys.executable, str(RUN_GATE), "reconcile"],
            timeout=20,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# Entrypoint -> the subcommands of it that occupy the box.  `side` (6h default
# timeout) and `diagnostic` (3h) are the longest runs in the project and the
# side lane is the ONLY evidence path for shapes 6 and 14; omitting them let an
# auditor be launched into the middle of a shape-14 timing run.  Contention
# inflates graphed-candidate speedups by up to 3x (LESSONS #19, #22), which is
# the leading explanation for why the published headline does not reproduce.
BUSY_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "runner.py": frozenset({"run", "calibrate"}),
    "trusted_controller.py": frozenset({"run", "side", "diagnostic", "calibrate"}),
}


def _process_argv(pid: int) -> list[str]:
    """The argv of one pid, or [] when it is gone, empty, or unreadable."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return []
    return [part for part in raw.decode("utf-8", "replace").split("\0") if part]


def _self_and_ancestors() -> set[int]:
    """This process and every parent of it.

    The watcher fires from a hook, so its own wrapper shell is alive while it
    looks.  A shell that merely NAMES a command is not a run of it.
    """
    exempt = {os.getpid()}
    pid = os.getpid()
    while pid > 1:
        parent = 0
        try:
            for line in Path(f"/proc/{pid}/status").read_text(
                    encoding="utf-8", errors="replace").splitlines():
                if line.startswith("PPid:"):
                    parent = int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            break
        if parent <= 0 or parent in exempt:
            break
        exempt.add(parent)
        pid = parent
    return exempt


def _argv_is_busy(argv: Sequence[str]) -> bool:
    """True only when this argv IS a benchmark/controller invocation.

    Matching is on whole argv elements, never on the joined command line.
    `pgrep -f` matched any process whose command TEXT contained the pattern —
    including this watcher's own `bash -c` wrapper — so any shell that merely
    mentioned `runner.py run` made the box look busy and silently skipped the
    audit launch.  A `bash -c` string is a single argv element, so an exact
    element match cannot be fooled by it.
    """
    if len(argv) < 2:
        return False
    head = PurePosixPath(argv[0]).name
    if head in BUSY_SUBCOMMANDS:  # ./runner.py run
        return argv[1] in BUSY_SUBCOMMANDS[head]
    if not head.startswith("python"):
        return False
    for index in range(1, len(argv) - 1):  # python3 [-u] <script> <subcommand>
        script = PurePosixPath(argv[index]).name
        if script in BUSY_SUBCOMMANDS:
            return argv[index + 1] in BUSY_SUBCOMMANDS[script]
    return False


def runner_busy() -> bool:
    """Never add auditor CPU load while a benchmark/controller run is active."""
    try:
        exempt = _self_and_ancestors()
        candidates = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except Exception:
        return True  # inability to establish idleness is not permission
    for pid in candidates:
        if pid in exempt:
            continue
        if _argv_is_busy(_process_argv(pid)):
            return True
    return False


def _parse_time(value: str) -> float:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except Exception:
        return 0.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminal_attempt_ids(events: list[dict]) -> set[str]:
    return {
        str(event.get("attempt_id"))
        for event in events
        if event.get("event_type") in {"attempt_failed", "audit_result"}
    }


def clean_terminal_markers() -> None:
    terminals = _terminal_attempt_ids(read_events())
    for marker in AUDIT_LOG_DIR.glob("audit_*.running.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("attempt_id") in terminals:
                marker.unlink()
        except Exception:
            # Unknown marker bytes are not authority and are left for owner
            # inspection; they never count as a durable completed attempt.
            continue


def reap_abandoned_attempts(now_epoch: float | None = None) -> None:
    """Turn an old start-without-terminal into a durable failed attempt."""
    now_epoch = time.time() if now_epoch is None else now_epoch
    for start in active_attempts():
        attempt_id = start["attempt_id"]
        marker = marker_path(start["entry_id"], attempt_id)
        pid = 0
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            if marker_data.get("attempt_id") != attempt_id:
                raise ValueError("marker attempt mismatch")
            pid = int(marker_data.get("pid", 0))
        except Exception:
            pid = 0
        if pid > 0 and _pid_alive(pid):
            continue
        age = now_epoch - _parse_time(str(start.get("recorded", "")))
        if age < STALE_ATTEMPT_SECONDS:
            continue
        try:
            record_attempt_failure(
                attempt_id=attempt_id,
                reason="AUDITOR_PROCESS_ABANDONED_WITHOUT_TERMINAL_EVENT",
            )
        except AuditAuthorityError:
            continue
        try:
            marker.unlink()
        except FileNotFoundError:
            pass


def _new_attempt_id() -> str:
    return f"{int(time.time())}-{os.getpid()}-{secrets.token_hex(8)}"


def _write_prelaunch_failure(entry: dict, identity: dict[str, str],
                             attempt_id: str, reason: str) -> None:
    payload = {
        "artifact_version": 2,
        "artifact_type": "audit_prelaunch_failure",
        "entry_id": entry["entry_id"],
        "attempt_id": attempt_id,
        "candidate_sha256": entry["candidate_sha256"],
        "packet_sha256": entry.get("packet_sha256", ""),
        "measurement_event_sha256": entry.get("measurement_event_sha256", ""),
        "lane": entry.get("lane", ""),
        "codex": identity,
        "reason": reason,
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    artifact, artifact_sha = store_content_addressed_json(
        payload, suffix=".audit-prelaunch-failure.json")
    record_attempt_failure(
        attempt_id=attempt_id,
        reason=reason,
        artifact_path=str(artifact.relative_to(ROOT)),
        artifact_sha256=artifact_sha,
    )


def launch_entry(entry: dict, identity) -> int:
    """Claim and launch one entry through atomic authority operations."""
    attempt_id = _new_attempt_id()
    nonce = secrets.token_hex(32)
    try:
        packet = load_bound_packet(
            entry["entry_id"],
            packet_sha256=entry.get("packet_sha256") or None,
        )
        if packet.candidate_sha256 != entry["candidate_sha256"]:
            raise AuditAuthorityError("packet candidate differs from result journal")
        if entry.get("packet_sha256") and packet.sha256 != entry["packet_sha256"]:
            raise AuditAuthorityError("packet bytes differ from controller queue binding")
        if entry.get("measurement_event_sha256") \
                and packet.measurement_event_sha256 != entry["measurement_event_sha256"]:
            raise AuditAuthorityError(
                "packet measurement differs from controller queue binding")
        if packet.lane is not None and packet.lane != entry.get("lane"):
            raise AuditAuthorityError("packet lane differs from controller queue binding")
    except AuditAuthorityError as exc:
        # A packet preflight failure cannot be converted into a well-bound
        # attempt.  It remains pending and therefore ineligible until the
        # trusted controller regenerates the measured-source packet.
        if entry.get("packet_sha256"):
            record_attempt_started(
                entry_id=entry["entry_id"],
                attempt_id=attempt_id,
                attempt_nonce=nonce,
                packet_sha256=entry["packet_sha256"],
                candidate_sha256=entry["candidate_sha256"],
                codex=identity,
                measurement_event_sha256=entry["measurement_event_sha256"],
                lane=entry["lane"],
                queue_event_sha256=entry["queue_event_sha256"],
            )
            _write_prelaunch_failure(
                entry, identity.as_dict(), attempt_id,
                f"PACKET_PREFLIGHT_FAILED: {exc}")
        print(f"[champion-watch] {entry['entry_id']} pending: {exc}", file=sys.stderr)
        return -1 if entry.get("packet_sha256") else 0

    record_attempt_started(
        entry_id=entry["entry_id"],
        attempt_id=attempt_id,
        attempt_nonce=nonce,
        packet_sha256=packet.sha256,
        candidate_sha256=packet.candidate_sha256,
        codex=identity,
        measurement_event_sha256=(
            entry.get("measurement_event_sha256") or "0" * 64),
        lane=entry.get("lane") or "legacy-primary",
        queue_event_sha256=entry.get("queue_event_sha256") or "",
    )

    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = AUDIT_LOG_DIR / f"audit_{entry['entry_id']}.{attempt_id}.log"
    log_fd = os.open(str(log), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    header = json.dumps({
        "artifact_version": 2,
        "artifact_type": "audit_process_log",
        "entry_id": entry["entry_id"],
        "attempt_id": attempt_id,
        "packet_sha256": packet.sha256,
        "candidate_sha256": packet.candidate_sha256,
        "codex": identity.as_dict(),
    }, sort_keys=True).encode("utf-8") + b"\n"
    os.write(log_fd, header)
    os.fsync(log_fd)
    command = [
        sys.executable, str(AUDITOR), entry["entry_id"],
        "--attempt-id", attempt_id,
        "--nonce", nonce,
        "--packet-sha256", packet.sha256,
        "--candidate-sha256", packet.candidate_sha256,
        "--measurement-event-sha256", (
            entry.get("measurement_event_sha256") or "0" * 64),
        "--lane", entry.get("lane") or "legacy-primary",
    ]
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(ROOT),
            close_fds=True,
        )
    except Exception as exc:
        os.close(log_fd)
        _write_prelaunch_failure(
            entry, identity.as_dict(), attempt_id,
            f"AUDITOR_POPEN_FAILED: {type(exc).__name__}: {exc}")
        return -1
    os.close(log_fd)
    marker_payload = {
        "artifact_version": 2,
        "artifact_type": "audit_running_marker",
        "entry_id": entry["entry_id"],
        "attempt_id": attempt_id,
        "pid": proc.pid,
        "codex": identity.as_dict(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        exclusive_write_json(marker_path(entry["entry_id"], attempt_id), marker_payload)
    except Exception as exc:
        # The durable start remains active and fail closed.  Do not kill a
        # successfully launched auditor merely because its liveness hint failed.
        print(f"[champion-watch] marker creation failed: {exc}", file=sys.stderr)
    print(
        f"[champion-watch] audit launched for {entry['entry_id']} "
        f"(attempt {attempt_id}; log {log.relative_to(ROOT)})")
    return 1


def watch_once(*, max_launches: int = 1, dry_run: bool = False,
               reconcile_gate: bool = True) -> dict:
    if reconcile_gate and not dry_run:
        run_gate_post()
    identity = resolve_codex_identity(CODEX_EXECUTABLE, PINNED_CODEX_SHA256)
    if not dry_run:
        clean_terminal_markers()
        reap_abandoned_attempts()
    queue = queue_snapshot()
    pending = queue["pending_rows"]
    owner_attention = queue["owner_attention_rows"]
    active = queue["active_rows"]
    summary = {
        "pending": [entry["entry_id"] for entry in pending],
        "active": [entry["entry_id"] for entry in active],
        "owner_attention": [entry["entry_id"] for entry in owner_attention],
        "launched": 0,
        "attempted": 0,
    }
    if dry_run or runner_busy() or len(active) >= MAX_CONCURRENT_AUDITS:
        return summary
    capacity = min(
        max(0, max_launches),
        MAX_CONCURRENT_AUDITS - len(active),
    )
    for entry in pending:
        if summary["attempted"] >= capacity:
            break
        outcome = launch_entry(entry, identity)
        if outcome:
            summary["attempted"] += 1
        if outcome > 0:
            summary["launched"] += 1
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch pending durable audits")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-launches", type=int, default=1)
    parser.add_argument("--no-reconcile", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = watch_once(
            max_launches=args.max_launches,
            dry_run=args.dry_run,
            reconcile_gate=not args.no_reconcile,
        )
    except AuditAuthorityError as exc:
        print(f"[champion-watch] REFUSED: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif summary["owner_attention"]:
        print(
            "[champion-watch] audit retries exhausted; owner attention: "
            + ", ".join(summary["owner_attention"]),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
