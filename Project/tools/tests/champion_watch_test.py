#!/usr/bin/env python3
"""Cold, no-GPU tests for the durable audit watcher (champion_watch.py).

Every case runs against an isolated temporary repository.  The real
``Project/audits`` and ``Project/results`` trees are never read for authority
and never written; the last case proves it byte-for-byte.  No GPU work, no
network, and the real ``codex`` binary is never resolved, hashed, or executed:
identity resolution is stubbed and the auditor process is replaced by a local
stub script.

The property under test is the one the historical watcher lacked: absence of a
verdict must COST something.  A killed auditor may never look "handled", and
response artifacts (created at launch) may never stand in for a durable
verdict.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "Project" / "tools"
HARNESS = REPO / "Project" / "harness"
for _path in (str(TOOLS), str(HARNESS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import audit_authority as aa
import audit_champion as ac

# =====================================================================
# Mutation self-test.  A suite that cannot detect a deliberately broken
# watcher is not a suite, so the breakages are kept here and replayed on
# demand with `python3 champion_watch_test.py --mutants`.  Each entry
# removes exactly one safety property from champion_watch.py IN MEMORY --
# the file on disk is never modified -- and the suite must go red for it.
# A mutant flagged ``equivalent`` provably cannot change behaviour (see the
# note in case_packet_preflight_binding) and is expected to survive.
# =====================================================================
MUTANTS: list[tuple[str, str, str, bool]] = [
    (
        'runner_busy fails OPEN on pgrep error',
        '            return True  # inability to establish idleness is not permission',
        '            return False',
        False,
    ),
    (
        'runner_busy ignores the controller pattern',
        '        "trusted_controller.py (run|calibrate)",\n',
        '',
        False,
    ),
    (
        '_pid_alive treats a permission-denied pid as dead',
        '    except PermissionError:\n        return True',
        '    except PermissionError:\n        return False',
        False,
    ),
    (
        '_pid_alive treats a missing pid as alive (never reaps)',
        '    except ProcessLookupError:\n        return False',
        '    except ProcessLookupError:\n        return True',
        False,
    ),
    (
        'terminal ids include live starts (cleans live markers)',
        'if event.get("event_type") in {"attempt_failed", "audit_result"}',
        'if event.get("event_type") in {"attempt_failed", "audit_result", "attempt_started"}',
        False,
    ),
    (
        'clean_terminal_markers deletes unreadable markers',
        '        except Exception:\n            # Unknown marker bytes are not authority and are left for owner\n            # inspection; they never count as a durable completed attempt.\n            continue',
        '        except Exception:\n            marker.unlink()',
        False,
    ),
    (
        'reap ignores the stale grace window',
        '        age = now_epoch - _parse_time(str(start.get("recorded", "")))\n        if age < STALE_ATTEMPT_SECONDS:\n            continue',
        '        age = now_epoch - _parse_time(str(start.get("recorded", "")))',
        False,
    ),
    (
        'reap ignores a live auditor pid',
        '        if pid > 0 and _pid_alive(pid):\n            continue',
        '        if False:\n            continue',
        False,
    ),
    (
        'reap trusts a marker naming a different attempt',
        '            if marker_data.get("attempt_id") != attempt_id:\n                raise ValueError("marker attempt mismatch")\n',
        '',
        False,
    ),
    (
        'STALE_ATTEMPT_SECONDS collapses to zero',
        'STALE_ATTEMPT_SECONDS = 300',
        'STALE_ATTEMPT_SECONDS = 0',
        False,
    ),
    (
        'MAX_CONCURRENT_AUDITS raised to 5',
        'MAX_CONCURRENT_AUDITS = 1',
        'MAX_CONCURRENT_AUDITS = 5',
        False,
    ),
    (
        'attempt ids are not unique',
        '    return f"{int(time.time())}-{os.getpid()}-{secrets.token_hex(8)}"',
        '    return "fixed-attempt-id"',
        False,
    ),
    (
        'attempt nonce is a fixed constant, not fresh per attempt',
        '    nonce = secrets.token_hex(32)',
        '    nonce = "a1" * 32',
        False,
    ),
    (
        'the durable claim is recorded only after the auditor is running',
        '    record_attempt_started(\n        entry_id=entry["entry_id"],\n        attempt_id=attempt_id,\n        attempt_nonce=nonce,\n        packet_sha256=packet.sha256,',
        '    _DEFERRED_START = lambda: record_attempt_started(\n        entry_id=entry["entry_id"],\n        attempt_id=attempt_id,\n        attempt_nonce=nonce,\n        packet_sha256=packet.sha256,',
        False,
    ),
    (
        'running marker is written non-exclusively (clobbers a rival claim)',
        '        exclusive_write_json(marker_path(entry["entry_id"], attempt_id), marker_payload)',
        '        marker_path(entry["entry_id"], attempt_id).write_text(json.dumps(marker_payload))',
        False,
    ),
    (
        'audit log is opened with O_TRUNC instead of O_EXCL',
        '    log_fd = os.open(str(log), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)',
        '    log_fd = os.open(str(log), os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)',
        False,
    ),
    (
        'preflight skips the candidate-bytes check',
        '        if packet.candidate_sha256 != entry["candidate_sha256"]:\n            raise AuditAuthorityError("packet candidate differs from result journal")',
        '        pass',
        False,
    ),
    (
        'preflight skips the packet-hash check',
        '        if entry.get("packet_sha256") and packet.sha256 != entry["packet_sha256"]:\n            raise AuditAuthorityError("packet bytes differ from controller queue binding")',
        '        pass',
        True,
    ),
    (
        'preflight skips the measurement check',
        '        if entry.get("measurement_event_sha256") \\\n                and packet.measurement_event_sha256 != entry["measurement_event_sha256"]:\n            raise AuditAuthorityError(\n                "packet measurement differs from controller queue binding")',
        '        pass',
        False,
    ),
    (
        'preflight skips the lane check',
        '        if packet.lane is not None and packet.lane != entry.get("lane"):\n            raise AuditAuthorityError("packet lane differs from controller queue binding")',
        '        pass',
        False,
    ),
    (
        'a failed preflight records no durable failure',
        '        if entry.get("packet_sha256"):\n            record_attempt_started(',
        '        if False:\n            record_attempt_started(',
        False,
    ),
    (
        'a failed preflight is not counted as an attempt',
        '        return -1 if entry.get("packet_sha256") else 0',
        '        return 0',
        False,
    ),
    (
        'a preflight failure is reported as a successful launch',
        '        return -1 if entry.get("packet_sha256") else 0',
        '        return 1',
        False,
    ),
    (
        'launched counter counts refusals too',
        '        if outcome > 0:\n            summary["launched"] += 1',
        '        if outcome != 0:\n            summary["launched"] += 1',
        False,
    ),
    (
        'the auditor is launched without its nonce',
        '        "--nonce", nonce,\n',
        '',
        False,
    ),
    (
        'the auditor is handed the nonce hash instead of the nonce',
        '        "--nonce", nonce,',
        '        "--nonce", __import__("hashlib").sha256(nonce.encode()).hexdigest(),',
        False,
    ),
    (
        'the auditor is launched with the queue packet, not the verified one',
        '        "--packet-sha256", packet.sha256,',
        '        "--packet-sha256", "0" * 64,',
        False,
    ),
    (
        'the auditor is always told lane legacy-primary',
        '        "--lane", entry.get("lane") or "legacy-primary",',
        '        "--lane", "legacy-primary",',
        False,
    ),
    (
        'dry-run still launches auditors',
        '    if dry_run or runner_busy() or len(active) >= MAX_CONCURRENT_AUDITS:',
        '    if runner_busy() or len(active) >= MAX_CONCURRENT_AUDITS:',
        False,
    ),
    (
        'dry-run ignores runner_busy',
        '    if dry_run or runner_busy() or len(active) >= MAX_CONCURRENT_AUDITS:',
        '    if dry_run or len(active) >= MAX_CONCURRENT_AUDITS:',
        False,
    ),
    (
        'dry-run still mutates markers and reaps attempts',
        '    if not dry_run:\n        clean_terminal_markers()\n        reap_abandoned_attempts()',
        '    if True:\n        clean_terminal_markers()\n        reap_abandoned_attempts()',
        False,
    ),
    (
        'dry-run still reconciles the gate',
        '    if reconcile_gate and not dry_run:',
        '    if reconcile_gate:',
        False,
    ),
    (
        '--no-reconcile is ignored',
        '    if reconcile_gate and not dry_run:',
        '    if not dry_run:',
        False,
    ),
    (
        'capacity ignores the active-attempt count',
        '    capacity = min(\n        max(0, max_launches),\n        MAX_CONCURRENT_AUDITS - len(active),\n    )',
        '    capacity = max(0, max_launches)',
        False,
    ),
    (
        'a negative launch budget is taken as its absolute value',
        '        max(0, max_launches),',
        '        abs(max_launches),',
        False,
    ),
    (
        'owner-attention rows are dropped from the snapshot',
        '        "owner_attention_rows": owner_attention,',
        '        "owner_attention_rows": [],',
        False,
    ),
    (
        'pending rows are reported as already handled',
        '        "pending_rows": pending,',
        '        "pending_rows": [],',
        False,
    ),
    (
        'the snapshot reports owner-attention rows as pending work',
        '    pending = queue["pending_rows"]',
        '    pending = queue["pending_rows"] + queue["owner_attention_rows"]',
        False,
    ),
    (
        'an unreadable authority journal exits 0',
        '        print(f"[champion-watch] REFUSED: {exc}", file=sys.stderr)\n        return 1',
        '        print(f"[champion-watch] REFUSED: {exc}", file=sys.stderr)\n        return 0',
        False,
    ),
    (
        'a corrupt journal is swallowed and reported as an empty backlog',
        '    except AuditAuthorityError as exc:\n        print(f"[champion-watch] REFUSED: {exc}", file=sys.stderr)\n        return 1',
        "    except AuditAuthorityError as exc:\n        summary = {'pending': [], 'active': [], 'owner_attention': [], 'launched': 0, 'attempted': 0}",
        False,
    ),
    (
        '--dry-run flag is inverted',
        '            dry_run=args.dry_run,',
        '            dry_run=not args.dry_run,',
        False,
    ),
    (
        'pending queue ignores settled audits and relaunches everything',
        '    pending = pending_audit_entries(\n        journal_path=journal_path, events_path=events_path,\n        legacy_path=legacy_path)',
        "    pending = __import__('audit_authority').required_audit_candidates(\n        journal_path=journal_path, events_path=events_path)",
        False,
    ),
    (
        'the auditor is told a zero measurement event',
        '        "--measurement-event-sha256", (\n            entry.get("measurement_event_sha256") or "0" * 64),',
        '        "--measurement-event-sha256", "0" * 64,',
        False,
    ),
    (
        'clean_terminal_markers deletes every marker it finds',
        '            if payload.get("attempt_id") in terminals:\n                marker.unlink()',
        '            marker.unlink()',
        False,
    ),
    (
        'reap deletes the claim marker without recording the failure',
        '        try:\n            record_attempt_failure(\n                attempt_id=attempt_id,\n                reason="AUDITOR_PROCESS_ABANDONED_WITHOUT_TERMINAL_EVENT",\n            )\n        except AuditAuthorityError:\n            continue',
        '        pass',
        False,
    ),
]


def _install_mutant(index: int) -> None:
    """Compile a mutated champion_watch into sys.modules before it is used."""
    import types

    name, old, new, _equivalent = MUTANTS[index]
    watcher = TOOLS / "champion_watch.py"
    source = watcher.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise SystemExit(
            f"mutation anchor is stale for {name!r}: champion_watch.py changed")
    module = types.ModuleType("champion_watch")
    module.__file__ = str(watcher)
    sys.modules["champion_watch"] = module
    exec(compile(source.replace(old, new), str(watcher), "exec"),
         module.__dict__)


if os.environ.get("CW_TEST_MUTANT"):
    _install_mutant(int(os.environ["CW_TEST_MUTANT"]))

import champion_watch as cw  # noqa: E402 - may be a mutant installed above

ZERO = "0" * 64
RUN_ENTRY = "run-0123456789abcdef0123456789abcdef"
RUN_ENTRY_2 = "run-fedcba9876543210fedcba9876543210"
LEGACY_ENTRY = "20260830-120000-abcdef"
LEGACY_ENTRY_2 = "20260830-120001-fedcba"
LEGACY_ENTRY_3 = "20260830-120002-0a0a0a"
LEGACY_ENTRY_4 = "20260830-120003-0b0b0b"
CANDIDATE = "b" * 64
PACKET = "c" * 64
NONCE = "d" * 64
NONCE_2 = "1" * 64
NONCE_3 = "2" * 64
NONCE_4 = "3" * 64

STUB_IDENTITY = aa.CodexIdentity(
    "/usr/local/lib/stub/codex", "/usr/local/lib/stub/codex", "f" * 64)

# Captured before any fixture replaces it, so the real implementation can be
# exercised against a fake subprocess module.
REAL_RUNNER_BUSY = cw.runner_busy

REAL_SCHEMA_BYTES = (REPO / "Project" / "audits" / "verdict_schema.json").read_bytes()

SAMPLE_LEADERBOARD = f"""# LEADERBOARD (auto-generated by the trusted runner)

## Shape 8 | NVIDIA GeForce RTX 3060 Ti | float32

| impl | speedup | correct | promoted | audit | entry |
|---|---|---|---|---|---|
| k009_fused ★ | 11.150x | PASS | yes | PASS | {RUN_ENTRY} |
"""

# Test double for audit_champion.py.  It never invokes codex.
STUB_AUDITOR = '''#!/usr/bin/env python3
"""Auditor stub: records its argv, then dies the way the mode asks."""
import json
import os
import signal
import sys
import time

log = os.environ.get("CW_STUB_LOG", "")
if log:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "pid": os.getpid(), "cwd": os.getcwd(), "argv": sys.argv[1:],
        }) + "\\n")
        handle.flush()
        os.fsync(handle.fileno())

mode = os.environ.get("CW_STUB_MODE", "kill")
if mode == "kill":            # auditor killed before any terminal event
    os.kill(os.getpid(), signal.SIGKILL)
elif mode == "crash":         # auditor crashes on its own
    sys.exit(9)
elif mode == "sleep":         # auditor still running
    time.sleep(600)
sys.exit(0)
'''

results: list[tuple[str, bool]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    results.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name
          + (f"  [{detail}]" if detail and not ok else ""))


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def bound(func, **forced):
    """Pin fixture paths onto an authority call; caller defaults never win."""
    def call(*args, **kwargs):
        merged = dict(kwargs)
        merged.update(forced)
        return func(*args, **merged)
    call.__name__ = getattr(func, "__name__", "bound")
    return call


def proc_state(pid: int) -> str | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        return data.rsplit(")", 1)[1].split()[0]
    except IndexError:
        return None


def wait_until_dead(pid: int, timeout: float = 10.0) -> str | None:
    """Wait until the auditor child is a zombie or gone (never reaping it)."""
    deadline = time.time() + timeout
    state = proc_state(pid)
    while time.time() < deadline and state not in ("Z", None):
        time.sleep(0.005)
        state = proc_state(pid)
    return state


def reap(pid: int) -> None:
    for _ in range(500):
        try:
            done, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if done:
            return
        time.sleep(0.005)


def is_stub_auditor(pid: int) -> bool:
    """Only ever signal a process that really is our auditor stub."""
    if pid <= 1 or pid == os.getpid():
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"stub_auditor.py" in cmdline


def kill_and_reap(pid: int) -> None:
    if is_stub_auditor(pid):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, 9)
    reap(pid)


def tree_snapshot(root: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_dir():
            out[rel] = "dir"
        elif path.is_file():
            out[rel] = (path.stat().st_size,
                        hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            out[rel] = "other"
    return out


class Fixture:
    """An isolated repository the watcher's authority calls are pinned to."""

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="champion-watch-test-")
        self.root = Path(self.tmp.name)
        self.audits = self.root / "Project" / "audits"
        self.auto = self.audits / "auto"
        self.packets = self.audits / "packets"
        self.blobs = self.root / "Project" / "authority" / "blobs"
        self.results_dir = self.root / "Project" / "results"
        for directory in (self.auto, self.packets, self.blobs, self.results_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.events = self.audits / "audit_events.jsonl"
        self.lock = self.audits / ".audit_authority.lock"
        self.legacy = self.audits / "verdicts.jsonl"
        self.legacy.write_text("", encoding="utf-8")
        self.journal = self.results_dir / "JOURNAL.jsonl"
        self.journal.write_text("", encoding="utf-8")
        self.leaderboard = self.results_dir / "LEADERBOARD.md"
        self.leaderboard.write_text(SAMPLE_LEADERBOARD, encoding="utf-8")
        self.schema = self.root / "verdict_schema.json"
        self.schema.write_bytes(REAL_SCHEMA_BYTES)
        self.stub = self.root / "stub_auditor.py"
        self.stub.write_text(STUB_AUDITOR, encoding="utf-8")
        self.stub_log = self.root / "stub_calls.jsonl"
        self.reconcile_calls: list[float] = []
        self._saved: list[tuple[object, str, object]] = []
        self._saved_env: dict[str, str | None] = {}
        self._apply()

    # -- patching -----------------------------------------------------
    def _set(self, module, name: str, value) -> None:
        self._saved.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def _setenv(self, name: str, value: str) -> None:
        if name not in self._saved_env:
            self._saved_env[name] = os.environ.get(name)
        os.environ[name] = value

    def _reconcile_spy(self) -> None:
        self.reconcile_calls.append(time.time())

    def _apply(self) -> None:
        # Content-addressed schema copy: mutating it models "schema changed
        # since the audit" without touching the repository's own schema.
        self._set(aa, "SCHEMA_PATH", self.schema)
        self._set(ac, "AUDIT_LOG_DIR", self.auto)
        self._set(cw, "AUDIT_LOG_DIR", self.auto)
        self._set(cw, "ROOT", self.root)
        self._set(cw, "AUDITOR", self.stub)
        self._set(cw, "PRIMARY_JOURNAL", self.journal)
        self._set(cw, "AUDIT_EVENTS", self.events)
        self._set(cw, "LEGACY_VERDICTS", self.legacy)
        self._set(cw, "read_events", bound(aa.read_events, path=self.events))
        self._set(cw, "active_attempts",
                  bound(aa.active_attempts, events_path=self.events))
        self._set(cw, "pending_audit_entries",
                  bound(aa.pending_audit_entries, journal_path=self.journal,
                        events_path=self.events, legacy_path=self.legacy))
        self._set(cw, "owner_attention_entries",
                  bound(aa.owner_attention_entries, journal_path=self.journal,
                        events_path=self.events, legacy_path=self.legacy))
        self._set(cw, "record_attempt_started",
                  bound(aa.record_attempt_started, path=self.events,
                        lock_path=self.lock))
        self._set(cw, "record_attempt_failure",
                  bound(aa.record_attempt_failure, path=self.events,
                        lock_path=self.lock))
        self._set(cw, "store_content_addressed_json",
                  bound(aa.store_content_addressed_json, directory=self.blobs))
        self._set(cw, "load_bound_packet",
                  bound(aa.load_bound_packet, packets_dir=self.packets,
                        authority_blobs=self.blobs))
        self._set(cw, "resolve_codex_identity",
                  lambda *args, **kwargs: STUB_IDENTITY)
        self._set(cw, "runner_busy", lambda: False)
        self._set(cw, "run_gate_post", self._reconcile_spy)
        self._setenv("CW_STUB_LOG", str(self.stub_log))
        self._setenv("CW_STUB_MODE", "kill")

    def stub_mode(self, mode: str) -> None:
        self._setenv("CW_STUB_MODE", mode)

    def close(self) -> None:
        for marker in self.auto.glob("audit_*.running.json"):
            with contextlib.suppress(Exception):
                kill_and_reap(int(json.loads(marker.read_text())["pid"]))
        for module, name, value in reversed(self._saved):
            setattr(module, name, value)
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.tmp.cleanup()

    def __enter__(self) -> Fixture:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- authority helpers --------------------------------------------
    def snapshot(self) -> dict:
        return cw.queue_snapshot(journal_path=self.journal,
                                 events_path=self.events,
                                 legacy_path=self.legacy)

    def pending_ids(self) -> list[str]:
        return [row["entry_id"] for row in self.snapshot()["pending_rows"]]

    def owner_ids(self) -> list[str]:
        return [row["entry_id"] for row in self.snapshot()["owner_attention_rows"]]

    def required_ids(self) -> list[str]:
        return [row["entry_id"] for row in aa.required_audit_candidates(
            journal_path=self.journal, events_path=self.events)]

    def events_of(self, event_type: str) -> list[dict]:
        return [event for event in aa.read_events(self.events)
                if event.get("event_type") == event_type]

    def decision(self, entry_id: str, candidate_sha: str | None,
                 packet_sha: str | None = None) -> aa.AuditDecision:
        return aa.audit_decision(entry_id, candidate_sha, packet_sha,
                                 events_path=self.events,
                                 legacy_path=self.legacy,
                                 artifact_root=self.root)

    def journal_rows(self, rows: list[dict]) -> None:
        self.journal.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8")

    def start(self, attempt: str, *, entry_id: str = LEGACY_ENTRY,
              candidate_sha256: str = CANDIDATE, packet_sha256: str = PACKET,
              nonce: str = NONCE, measurement: str = ZERO,
              lane: str = "legacy-primary", queue: str = "") -> dict:
        return aa.record_attempt_started(
            entry_id=entry_id, attempt_id=attempt, attempt_nonce=nonce,
            packet_sha256=packet_sha256, candidate_sha256=candidate_sha256,
            codex=STUB_IDENTITY, measurement_event_sha256=measurement,
            lane=lane, queue_event_sha256=queue,
            path=self.events, lock_path=self.lock)

    def result(self, attempt: str, document: dict, *,
               entry_id: str = LEGACY_ENTRY, candidate_sha256: str = CANDIDATE,
               packet_sha256: str = PACKET, measurement: str = ZERO,
               lane: str = "legacy-primary", queue: str = "",
               artifact_dir: Path | None = None,
               artifact_name: str | None = None,
               artifact_overrides: dict | None = None,
               stdout: str | None = None) -> dict:
        artifact = (artifact_dir or self.auto) / f"{artifact_name or attempt}.json"
        payload = {
            "artifact_version": 2,
            "artifact_type": "audit_response",
            "attempt_id": attempt,
            "entry_id": entry_id,
            "packet_sha256": packet_sha256,
            "candidate_sha256": candidate_sha256,
            "measurement_event_sha256": measurement,
            "lane": lane,
            "verdict_schema_sha256": aa.sha256_file(aa.SCHEMA_PATH),
            "codex": STUB_IDENTITY.as_dict(),
            "returncode": 0,
            "stdout": json.dumps(document) if stdout is None else stdout,
            "parser_error": "",
            "validated_result": document,
        }
        payload.update(artifact_overrides or {})
        artifact_sha = aa.exclusive_write_json(artifact, payload)
        return aa.record_audit_result(
            attempt_id=attempt, result=document,
            artifact_path=str(artifact.relative_to(self.root)),
            artifact_sha256=artifact_sha, path=self.events,
            lock_path=self.lock, artifact_root=self.root)


def verdict(integrity: str = "PASS", technical: str = "PASS", *,
            entry_id: str = LEGACY_ENTRY, packet_sha256: str = PACKET,
            candidate_sha256: str = CANDIDATE, nonce: str = NONCE) -> dict:
    return {
        "schema_version": 2,
        "attempt_nonce": nonce,
        "entry_id": entry_id,
        "packet_sha256": packet_sha256,
        "candidate_sha256": candidate_sha256,
        "integrity": {
            "verdict": integrity,
            "findings": [],
            "retest_request": "rerun identical bytes" if integrity == "RETEST" else "",
            "summary": "integrity reviewed",
        },
        "technical_review": {
            "verdict": technical,
            "findings": [],
            "summary": "diagnosis reviewed",
        },
        "summary": "complete independent review",
    }


def enqueue_prospective(fx: Fixture, *, entry_id: str = RUN_ENTRY,
                        source: bytes = b"def custom_kernel(x):\n    return x\n",
                        lane: str = "primary", measurement: str = "8" * 64,
                        write_source: bool = True,
                        write_packet: bool = True) -> dict:
    candidate_sha = hashlib.sha256(source).hexdigest()
    if write_source:
        (fx.blobs / f"{candidate_sha}.py").write_bytes(source)
    packet = {
        "schema_version": 1,
        "entry_id": entry_id,
        "candidate_sha256": candidate_sha,
        "measurement_event_sha256": measurement,
        "lane": lane,
    }
    raw = aa._canonical(packet)
    packet_sha = hashlib.sha256(raw).hexdigest()
    if write_packet:
        (fx.blobs / f"{packet_sha}.json").write_bytes(raw)
    queue = aa.enqueue_audit(
        entry_id=entry_id, candidate_sha256=candidate_sha,
        packet_sha256=packet_sha, measurement_event_sha256=measurement,
        lane=lane, path=fx.events, lock_path=fx.lock)
    return {
        "entry_id": entry_id,
        "candidate_sha256": candidate_sha,
        "packet_sha256": packet_sha,
        "measurement_event_sha256": measurement,
        "lane": lane,
        "queue_event_sha256": queue["event_sha256"],
    }


def wait_for_stub_calls(fx: Fixture, count: int,
                        timeout: float = 15.0) -> list[dict]:
    """Wait for the launched auditor stubs to record themselves."""
    deadline = time.time() + timeout
    calls: list[dict] = []
    while time.time() < deadline:
        if fx.stub_log.exists():
            calls = [json.loads(line) for line
                     in fx.stub_log.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
            if len(calls) >= count:
                return calls
        time.sleep(0.01)
    return calls


def run_main(argv: list[str]) -> tuple[int, str]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = cw.main(argv)
    return code, stream.getvalue()


def safe_watch(**kwargs) -> tuple[dict | None, BaseException | None]:
    """Run one watcher pass, reporting a refusal instead of aborting."""
    try:
        return cw.watch_once(**kwargs), None
    except BaseException as exc:  # noqa: BLE001 - the test reports it
        return None, exc


def refuses(callable_, *args, **kwargs) -> bool:
    try:
        callable_(*args, **kwargs)
    except aa.AuditAuthorityError:
        return True
    except Exception:
        return False
    return False


@contextlib.contextmanager
def frozen_clock(stamp: str = "2020-01-01T00:00:00+0000"):
    """Append durable events stamped in the past, without ever sleeping."""
    original = aa._now
    aa._now = lambda: stamp
    try:
        yield
    finally:
        aa._now = original


class OsShim:
    """Delegates to the real os module but forces one os.kill outcome."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def kill(self, pid: int, signal_number: int) -> None:
        raise self._error

    def __getattr__(self, name):
        return getattr(os, name)


def fixed_attempt_id(fx: "Fixture", attempt_id: str) -> None:
    """Make the next claim land on a predictable, pre-creatable path."""
    fx._set(cw, "_new_attempt_id", lambda: attempt_id)


def reap_stub_calls(calls: list[dict]) -> None:
    for call in calls:
        kill_and_reap(int(call["pid"]))


# =====================================================================
# 1 + 3.  Backlog is journal-derived; Markdown is never authority.
# =====================================================================
def case_backlog_from_durable_journals() -> None:
    section("backlog derives from the durable journals, never Markdown")
    with Fixture() as fx:
        fx.journal_rows([
            {"entry_id": LEGACY_ENTRY, "type": "candidate", "promoted": True,
             "shape_id": 8, "impl": {"sha256": CANDIDATE}},
            {"entry_id": LEGACY_ENTRY_2, "type": "candidate",
             "promotion_candidate": True, "shape_id": 1,
             "impl": {"sha256": "a" * 64}},
            {"entry_id": LEGACY_ENTRY_3, "type": "candidate", "shape_id": 2,
             "impl": {"sha256": "9" * 64}},
        ])
        entry = enqueue_prospective(fx)
        baseline = fx.snapshot()
        pending = [row["entry_id"] for row in baseline["pending_rows"]]
        check("promoted + promotion_candidate journal rows enter the backlog",
              LEGACY_ENTRY in pending and LEGACY_ENTRY_2 in pending, str(pending))
        check("un-flagged journal rows do not enter the backlog",
              LEGACY_ENTRY_3 not in pending, str(pending))
        check("controller enqueue enters the backlog first",
              pending[0] == RUN_ENTRY, str(pending))
        check("queue binding is carried on the pending row",
              baseline["pending_rows"][0]["queue_event_sha256"]
              == entry["queue_event_sha256"])

        fx.leaderboard.write_text("\x00\x01 not markdown at all \x00", encoding="utf-8")
        corrupted = [row["entry_id"] for row in fx.snapshot()["pending_rows"]]
        check("corrupted LEADERBOARD.md leaves the backlog unchanged",
              corrupted == pending, f"{corrupted} != {pending}")

        fx.leaderboard.unlink()
        deleted = [row["entry_id"] for row in fx.snapshot()["pending_rows"]]
        check("deleted LEADERBOARD.md leaves the backlog unchanged",
              deleted == pending, f"{deleted} != {pending}")

        fx.leaderboard.write_text(
            SAMPLE_LEADERBOARD.replace("PASS", "PASS")
            + f"| forged ★ | 99x | PASS | yes | PASS | {LEGACY_ENTRY} |\n",
            encoding="utf-8")
        forged = [row["entry_id"] for row in fx.snapshot()["pending_rows"]]
        check("a forged Markdown 'PASS' row cannot settle an audit",
              forged == pending, f"{forged} != {pending}")

        watcher_source = (TOOLS / "champion_watch.py").read_text(encoding="utf-8")
        authority_source = (TOOLS / "audit_authority.py").read_text(encoding="utf-8")
        check("champion_watch.py never mentions the leaderboard or Markdown",
              "LEADERBOARD" not in watcher_source
              and ".md" not in watcher_source.lower())
        check("audit_authority.py never mentions the leaderboard file",
              "LEADERBOARD.md" not in authority_source)

        # A preserved legacy row settles only historical work, never a
        # prospective controller enqueue for the same id.
        fx.legacy.write_text(json.dumps({
            "entry_id": LEGACY_ENTRY, "verdict": "PASS", "recorded": "legacy",
        }) + "\n", encoding="utf-8")
        settled = [row["entry_id"] for row in fx.snapshot()["pending_rows"]]
        check("legacy final row retires only the legacy backlog entry",
              LEGACY_ENTRY not in settled and LEGACY_ENTRY_2 in settled,
              str(settled))
        fx.legacy.write_text(json.dumps({
            "entry_id": RUN_ENTRY, "verdict": "PASS", "recorded": "legacy",
        }) + "\n", encoding="utf-8")
        check("legacy PASS cannot settle a prospective controller enqueue",
              RUN_ENTRY in [row["entry_id"] for row in fx.snapshot()["pending_rows"]])


# =====================================================================
# 1.  A killed auditor is never "handled".
# =====================================================================
def _three_strikes(mode: str) -> None:
    """Three consecutive dead auditors must never settle an entry."""
    with Fixture() as fx:
        entry = enqueue_prospective(fx)
        fx.stub_mode(mode)
        for attempt in range(1, 4):
            before = fx.pending_ids()
            check(f"{mode} {attempt}: entry is queued before launch",
                  before == [RUN_ENTRY], str(before))
            summary = cw.watch_once()
            check(f"{mode} {attempt}: watcher launched exactly one auditor",
                  summary["launched"] == 1 and summary["attempted"] == 1,
                  str(summary))
            marker = sorted(fx.auto.glob("audit_*.running.json"))
            pid = json.loads(marker[-1].read_text())["pid"]
            state = wait_until_dead(pid)
            check(f"{mode} {attempt}: auditor died without a terminal event",
                  state in ("Z", None)
                  and not fx.events_of("audit_result")
                  and len(fx.events_of("attempt_failed")) == attempt - 1,
                  str(state))
            if attempt == 1:
                # An unreaped child still answers signal 0: a watcher that
                # stays resident would treat the corpse as a live auditor.
                check("unreaped auditor corpse still reports as alive "
                      "(liveness hazard; entry stays ineligible either way)",
                      state != "Z" or cw._pid_alive(pid) is True)
            reap(pid)
            check(f"{mode} {attempt}: entry is not relaunched while the attempt "
                  "is open", cw.watch_once()["launched"] == 0)
            decision = fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                                   entry["packet_sha256"])
            check(f"{mode} {attempt}: promotion stays ineligible",
                  decision.promotion_eligible is False
                  and "missing_audit_verdict" in decision.blocking_reasons,
                  str(decision.blocking_reasons))
            # Only after the stale window does the dead attempt become a
            # durable failure and the entry return to the queue.
            cw.reap_abandoned_attempts(
                now_epoch=time.time() + cw.STALE_ATTEMPT_SECONDS + 1)
            check(f"{mode} {attempt}: abandoned attempt became a durable failure",
                  len(fx.events_of("attempt_failed")) == attempt,
                  str(len(fx.events_of("attempt_failed"))))
            check(f"{mode} {attempt}: stale running marker was removed",
                  not list(fx.auto.glob("audit_*.running.json")))
            if attempt < aa.MAX_FAILED_ATTEMPTS:
                check(f"{mode} {attempt}: entry is queued again",
                      fx.pending_ids() == [RUN_ENTRY])

        check(f"after three {mode}s the entry is still required work",
              fx.required_ids() == [RUN_ENTRY], str(fx.required_ids()))
        check(f"after three {mode}s the entry escalates to owner attention",
              fx.owner_ids() == [RUN_ENTRY], str(fx.owner_ids()))
        final = fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                            entry["packet_sha256"])
        check(f"after three {mode}s promotion is still ineligible",
              final.promotion_eligible is False
              and final.integrity_status == "MISSING",
              str(final.blocking_reasons))
        check(f"three {mode}s never produced an audit result event",
              not fx.events_of("audit_result"))
        code, _out = run_main([])
        check("watcher exits 0 while an entry needs owner attention "
              "(exit status is not a starvation signal)", code == 0)



def case_killed_auditor_stays_ineligible() -> None:
    section("killed or crashed auditor: three dead auditors never "
            "produce 'handled'")
    for mode in ("kill", "crash"):
        _three_strikes(mode)


# =====================================================================
# 1 + 6.  Response artifacts are not evidence.
# =====================================================================
def case_response_artifacts_are_not_evidence() -> None:
    section("response artifacts never stand in for a durable verdict")
    with Fixture() as fx:
        entry = enqueue_prospective(fx)
        # The historical exploit: artifacts existed the moment the auditor was
        # launched, so three kills parked the entry silently and for free.
        for index in range(5):
            (fx.auto / f"audit_{RUN_ENTRY}.fake-{index}.json").write_text(
                json.dumps({"artifact_type": "audit_response",
                            "validated_result": verdict(entry_id=RUN_ENTRY)}),
                encoding="utf-8")
            (fx.auto / f"audit_{RUN_ENTRY}.fake-{index}.log").write_text(
                "audit finished: PASS\n", encoding="utf-8")
        check("five forged response artifacts leave the entry queued",
              fx.pending_ids() == [RUN_ENTRY], str(fx.pending_ids()))
        summary = aa.attempt_summary(RUN_ENTRY, events_path=fx.events)
        check("attempt bookkeeping counts durable rows, not artifacts",
              summary["attempts"] == 0 and summary["failed_attempts"] == 0
              and summary["has_result"] is False, str(summary))
        decision = fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                               entry["packet_sha256"])
        check("forged artifacts do not make the entry promotable",
              decision.promotion_eligible is False)

        # A forged running marker cannot mint a durable attempt either.
        forged_marker = ac.marker_path(RUN_ENTRY, "forged-attempt")
        aa.exclusive_write_json(forged_marker, {
            "artifact_type": "audit_running_marker", "entry_id": RUN_ENTRY,
            "attempt_id": "forged-attempt", "pid": os.getpid()})
        check("forged running marker does not remove the entry from the queue",
              fx.pending_ids() == [RUN_ENTRY])
        check("forged running marker is not a durable active attempt",
              cw.active_attempts() == [])
        cw.clean_terminal_markers()
        check("marker for an unknown attempt is preserved for the owner",
              forged_marker.exists())


# =====================================================================
# 2.  Missing verdict => promotion_eligible False.
# =====================================================================
def case_missing_verdict_blocks_promotion() -> None:
    section("missing verdict is ineligible")
    with Fixture() as fx:
        entry = enqueue_prospective(fx)
        decision = fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                               entry["packet_sha256"])
        check("enqueued-but-unaudited candidate is ineligible",
              decision.promotion_eligible is False
              and decision.integrity_status == "MISSING"
              and "missing_audit_verdict" in decision.blocking_reasons,
              str(decision.blocking_reasons))
        check("no candidate hash supplied is also ineligible",
              fx.decision(RUN_ENTRY, None).promotion_eligible is False)
        # A verdict bound to different bytes is not this candidate's verdict.
        other = enqueue_prospective(
            fx, entry_id=RUN_ENTRY_2, source=b"VALUE = 2\n", lane="shape6")
        fx.start("other-attempt", entry_id=RUN_ENTRY_2,
                 candidate_sha256=other["candidate_sha256"],
                 packet_sha256=other["packet_sha256"], nonce=NONCE_2,
                 measurement=other["measurement_event_sha256"],
                 lane="shape6", queue=other["queue_event_sha256"])
        fx.result("other-attempt",
                  verdict(entry_id=RUN_ENTRY_2,
                          candidate_sha256=other["candidate_sha256"],
                          packet_sha256=other["packet_sha256"], nonce=NONCE_2),
                  entry_id=RUN_ENTRY_2,
                  candidate_sha256=other["candidate_sha256"],
                  packet_sha256=other["packet_sha256"],
                  measurement=other["measurement_event_sha256"], lane="shape6")
        check("another entry's PASS does not cover this candidate",
              fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                          entry["packet_sha256"]).promotion_eligible is False)
        check("the audited entry itself is eligible",
              fx.decision(RUN_ENTRY_2, other["candidate_sha256"],
                          other["packet_sha256"]).promotion_eligible is True)
        check("an entry with a bound audit result leaves the pending queue",
              fx.pending_ids() == [RUN_ENTRY], str(fx.pending_ids()))
        check("the settled entry is still derivable as required work "
              "(history is never erased)",
              sorted(fx.required_ids()) == sorted([RUN_ENTRY, RUN_ENTRY_2]),
              str(fx.required_ids()))
        check("the settled entry is never relaunched",
              cw.watch_once(dry_run=True)["pending"] == [RUN_ENTRY])
        check("legacy unbound PASS is history, not promotion authority",
              _legacy_pass_is_not_authority(fx))


def _legacy_pass_is_not_authority(fx: Fixture) -> bool:
    fx.legacy.write_text(json.dumps({
        "entry_id": LEGACY_ENTRY, "verdict": "PASS", "recorded": "legacy",
    }) + "\n", encoding="utf-8")
    decision = fx.decision(LEGACY_ENTRY, CANDIDATE)
    return (decision.promotion_eligible is False
            and decision.technical_status == "LEGACY_UNBOUND")


# =====================================================================
# 4.  Hard integrity verdicts are first-write-wins.
# =====================================================================
def case_hard_verdict_first_write_wins() -> None:
    section("hard integrity verdicts latch until an authenticated resolution")
    for hard, kind in (("RULE_VIOLATION", "FINDING_OVERTURNED"),
                       ("RETEST", "RETEST_SATISFIED")):
        with Fixture() as fx:
            fx.start("first", nonce=NONCE)
            hard_event = fx.result("first", verdict(hard))
            check(f"{hard}: the audited entry is ineligible",
                  fx.decision(LEGACY_ENTRY, CANDIDATE).promotion_eligible is False)
            check(f"{hard}: a second verdict on the same entry/candidate is "
                  "refused", refuses(fx.start, "rewrite", nonce=NONCE_2))

            # Same candidate bytes re-queued under a fresh entry id.
            second_packet = "a" * 64
            fx.start("second", entry_id=LEGACY_ENTRY_2,
                     packet_sha256=second_packet, nonce=NONCE_2)
            passed = fx.result("second",
                               verdict(entry_id=LEGACY_ENTRY_2,
                                       packet_sha256=second_packet,
                                       nonce=NONCE_2),
                               entry_id=LEGACY_ENTRY_2,
                               packet_sha256=second_packet)
            later = fx.decision(LEGACY_ENTRY_2, CANDIDATE, second_packet)
            check(f"{hard}: a later PASS on the same bytes does not overwrite it",
                  later.promotion_eligible is False
                  and later.active_hard_event_sha256 == hard_event["event_sha256"]
                  and f"unresolved_first_hard_verdict:{hard}"
                  in later.blocking_reasons, str(later.blocking_reasons))

            check(f"{hard}: unauthenticated resolution is refused",
                  refuses(aa.record_resolution, entry_id=LEGACY_ENTRY,
                          target_event_sha256=hard_event["event_sha256"],
                          resolution_kind=kind,
                          rationale="the owner said so in chat",
                          authority_event_id="made-up-event",
                          capability_nonce="made-up-nonce",
                          superseding_event_sha256=passed["event_sha256"],
                          path=fx.events, lock_path=fx.lock,
                          legacy_path=fx.legacy, authority_root=fx.root))

            from authority import AuthorityStore

            nonce = f"owner-nonce-{hard}"
            store = AuthorityStore(fx.root)
            authority_event = store.append(
                kind="audit_resolve_authorized", actor="trusted-controller",
                payload={
                    "capability_consumed": True,
                    "capability_action": "audit.resolve",
                    "capability_target": f"audit:{LEGACY_ENTRY}",
                    "capability_role": "owner",
                    "subject_sha256": hard_event["event_sha256"],
                    "capability_nonce": nonce,
                    "owner_key_sha256": "9" * 64,
                })
            wrong_target = store.append(
                kind="audit_resolve_authorized", actor="trusted-controller",
                payload={
                    "capability_consumed": True,
                    "capability_action": "audit.resolve",
                    "capability_target": f"audit:{LEGACY_ENTRY}",
                    "capability_role": "owner",
                    "subject_sha256": passed["event_sha256"],
                    "capability_nonce": nonce,
                    "owner_key_sha256": "9" * 64,
                })
            check(f"{hard}: resolution authorized for another event hash is "
                  "refused",
                  refuses(aa.record_resolution, entry_id=LEGACY_ENTRY,
                          target_event_sha256=hard_event["event_sha256"],
                          resolution_kind=kind, rationale="mis-bound receipt",
                          authority_event_id=wrong_target["event_id"],
                          capability_nonce=nonce,
                          superseding_event_sha256=passed["event_sha256"],
                          path=fx.events, lock_path=fx.lock,
                          legacy_path=fx.legacy, authority_root=fx.root))
            check(f"{hard}: resolution with the wrong capability nonce is refused",
                  refuses(aa.record_resolution, entry_id=LEGACY_ENTRY,
                          target_event_sha256=hard_event["event_sha256"],
                          resolution_kind=kind, rationale="replayed nonce",
                          authority_event_id=authority_event["event_id"],
                          capability_nonce="not-the-nonce",
                          superseding_event_sha256=passed["event_sha256"],
                          path=fx.events, lock_path=fx.lock,
                          legacy_path=fx.legacy, authority_root=fx.root))
            aa.record_resolution(
                entry_id=LEGACY_ENTRY,
                target_event_sha256=hard_event["event_sha256"],
                resolution_kind=kind,
                rationale="owner-authorized independent correction",
                authority_event_id=authority_event["event_id"],
                capability_nonce=nonce,
                superseding_event_sha256=passed["event_sha256"],
                path=fx.events, lock_path=fx.lock, legacy_path=fx.legacy,
                authority_root=fx.root)
            resolved = fx.decision(LEGACY_ENTRY_2, CANDIDATE, second_packet)
            check(f"{hard}: only the exact authenticated resolution clears it",
                  resolved.promotion_eligible is True,
                  str(resolved.blocking_reasons))
            check(f"{hard}: the same verdict cannot be resolved twice",
                  refuses(aa.record_resolution, entry_id=LEGACY_ENTRY,
                          target_event_sha256=hard_event["event_sha256"],
                          resolution_kind=kind, rationale="second bite",
                          authority_event_id=authority_event["event_id"],
                          capability_nonce=nonce,
                          superseding_event_sha256=passed["event_sha256"],
                          path=fx.events, lock_path=fx.lock,
                          legacy_path=fx.legacy, authority_root=fx.root))


def case_hard_verdict_blocks_the_watcher_queue() -> None:
    section("a legacy hard verdict still blocks, and NEEDS_CONTEXT blocks too")
    with Fixture() as fx:
        fx.legacy.write_text(json.dumps({
            "entry_id": LEGACY_ENTRY, "verdict": "RULE_VIOLATION",
            "recorded": "legacy",
        }) + "\n", encoding="utf-8")
        decision = fx.decision(LEGACY_ENTRY, CANDIDATE)
        check("legacy RULE_VIOLATION row latches as an unresolved hard verdict",
              decision.promotion_eligible is False
              and decision.active_hard_event_sha256 is not None,
              str(decision.blocking_reasons))
    with Fixture() as fx:
        fx.start("ctx")
        fx.result("ctx", verdict("NEEDS_CONTEXT"))
        decision = fx.decision(LEGACY_ENTRY, CANDIDATE)
        check("NEEDS_CONTEXT blocks promotion without latching as hard",
              decision.promotion_eligible is False
              and decision.active_hard_event_sha256 is None
              and "integrity:NEEDS_CONTEXT" in decision.blocking_reasons,
              str(decision.blocking_reasons))


# =====================================================================
# 5.  Technical review is a separate channel.
# =====================================================================
def case_technical_channel_is_separate() -> None:
    section("technical review blocks (weak/missing) or advises (disagreement)")
    expectations = {
        "WEAK_DIAGNOSIS": False,
        "MISSING_EVIDENCE": False,
        "TECHNICAL_DISAGREEMENT": True,
        "PASS": True,
    }
    for technical, eligible in expectations.items():
        with Fixture() as fx:
            fx.start("tech")
            fx.result("tech", verdict("PASS", technical))
            decision = fx.decision(LEGACY_ENTRY, CANDIDATE)
            check(f"integrity PASS + technical {technical} => eligible="
                  f"{eligible}",
                  decision.promotion_eligible is eligible,
                  str(decision.blocking_reasons))
            check(f"{technical}: integrity channel is untouched",
                  decision.integrity_status == "PASS"
                  and decision.technical_status == technical)
            if not eligible:
                check(f"{technical}: blocks on the technical channel only",
                      f"technical:{technical}" in decision.blocking_reasons
                      and not any(reason.startswith("integrity:")
                                  for reason in decision.blocking_reasons),
                      str(decision.blocking_reasons))
                check(f"{technical}: is not treated as a hard integrity latch",
                      decision.active_hard_event_sha256 is None)
    with Fixture() as fx:
        fx.start("mixed")
        fx.result("mixed", verdict("RULE_VIOLATION", "TECHNICAL_DISAGREEMENT"))
        decision = fx.decision(LEGACY_ENTRY, CANDIDATE)
        check("advisory technical verdict never rescues a hard integrity one",
              decision.promotion_eligible is False,
              str(decision.blocking_reasons))


# =====================================================================
# 6.  Retry caps come from durable rows and terminate.
# =====================================================================
def case_retry_cap_from_durable_rows() -> None:
    section("retry cap counts durable rows and stops at MAX_FAILED_ATTEMPTS")
    with Fixture() as fx:
        # A prospective entry whose candidate blob is missing: every launch
        # fails preflight and must leave a durable, counted failure.
        entry = enqueue_prospective(fx, write_source=False)
        launches = 0
        for index in range(aa.MAX_FAILED_ATTEMPTS + 3):
            summary = cw.watch_once()
            launches += summary["launched"]
            if index < aa.MAX_FAILED_ATTEMPTS:
                check(f"preflight failure {index + 1} recorded durably",
                      len(fx.events_of("attempt_failed")) == index + 1,
                      str(len(fx.events_of("attempt_failed"))))
        check("no auditor process was ever launched for a bad packet",
              launches == 0 and not fx.stub_log.exists())
        check("failures stop at MAX_FAILED_ATTEMPTS and do not loop forever",
              len(fx.events_of("attempt_failed")) == aa.MAX_FAILED_ATTEMPTS,
              str(len(fx.events_of("attempt_failed"))))
        check("attempt_started rows are capped too",
              len(fx.events_of("attempt_started")) == aa.MAX_FAILED_ATTEMPTS)
        check("exhausted entry leaves the pending queue for owner attention",
              fx.pending_ids() == [] and fx.owner_ids() == [RUN_ENTRY])
        check("exhausted entry remains required work, never 'handled'",
              fx.required_ids() == [RUN_ENTRY])
        check("exhausted entry is still ineligible",
              fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                          entry["packet_sha256"]).promotion_eligible is False)
        check("every durable failure names a stored prelaunch artifact",
              all(row["artifact_sha256"] and row["artifact_path"]
                  for row in fx.events_of("attempt_failed")))
        summary = aa.attempt_summary(RUN_ENTRY, events_path=fx.events)
        check("retry_exhausted is derived from failed durable rows",
              summary["failed_attempts"] == aa.MAX_FAILED_ATTEMPTS
              and summary["retry_exhausted"] is True, str(summary))
        stderr_summary = cw.watch_once()
        check("an exhausted entry is never launched again",
              stderr_summary["launched"] == 0
              and stderr_summary["owner_attention"] == [RUN_ENTRY])

    with Fixture() as fx:
        # Legacy-lane entry with no packet at all: fail-closed on eligibility,
        # but no durable row is created, so the cap never engages.
        fx.journal_rows([{"entry_id": LEGACY_ENTRY, "type": "candidate",
                          "promoted": True, "shape_id": 8,
                          "impl": {"sha256": CANDIDATE}}])
        for _ in range(aa.MAX_FAILED_ATTEMPTS + 2):
            cw.watch_once()
        check("legacy entry with an absent packet never becomes eligible",
              fx.decision(LEGACY_ENTRY, CANDIDATE).promotion_eligible is False)
        check("legacy preflight failure writes no durable attempt row "
              "(retry cap and owner escalation never engage)",
              fx.events_of("attempt_started") == []
              and fx.events_of("attempt_failed") == []
              and fx.owner_ids() == []
              and fx.pending_ids() == [LEGACY_ENTRY])


# =====================================================================
# 7.  Two watchers cannot both claim one entry.
# =====================================================================
def case_concurrent_watchers_single_claim() -> None:
    section("exclusive claim: two watchers, one auditor")
    with Fixture() as fx:
        entry = enqueue_prospective(fx)
        fx.stub_mode("crash")
        pending = fx.snapshot()["pending_rows"][0]
        barrier = fx.root / "race-go"
        pids = []
        for _ in range(2):
            pid = os.fork()
            if pid == 0:  # child watcher
                code = 3
                try:
                    while not barrier.exists():
                        time.sleep(0.002)
                    code = 0 if cw.launch_entry(pending, STUB_IDENTITY) > 0 else 1
                except aa.AuditAuthorityError:
                    code = 2
                except BaseException:
                    code = 4
                finally:
                    os._exit(code)
            pids.append(pid)
        barrier.write_text("go", encoding="utf-8")
        codes = sorted(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
                       for pid in pids)
        check("exactly one racing watcher claimed the entry, the other refused",
              codes == [0, 2], str(codes))
        starts = fx.events_of("attempt_started")
        check("exactly one durable attempt_started row exists",
              len(starts) == 1, str(len(starts)))
        markers = list(fx.auto.glob("audit_*.running.json"))
        check("exactly one running marker exists", len(markers) == 1,
              str(markers))
        check("the marker belongs to the winning attempt",
              json.loads(markers[0].read_text())["attempt_id"]
              == starts[0]["attempt_id"])
        launched = wait_for_stub_calls(fx, 1)
        # Wait for a hypothetical SECOND auditor too: settling for the first
        # sighting would let a double-launch pass unnoticed.
        check("exactly one auditor process was started",
              len(wait_for_stub_calls(fx, 2, timeout=2.0)) == 1,
              str(len(launched)))
        check("the launched auditor is bound to the exact packet/candidate",
              "--packet-sha256" in launched[0]["argv"]
              and entry["packet_sha256"] in launched[0]["argv"]
              and entry["candidate_sha256"] in launched[0]["argv"])
        check("the launched auditor carries a fresh 64-hex nonce",
              _nonce_of(launched[0]["argv"]) is not None)
        check("the durable journal stores only the nonce hash, never the nonce",
              _nonce_of(launched[0]["argv"])
              not in fx.events.read_text(encoding="utf-8"))
        check("a same-process second claim is refused as well",
              refuses(cw.launch_entry, pending, STUB_IDENTITY))
        check("the O_EXCL marker name is never reused",
              _marker_is_exclusive(fx, starts[0]))
        check("MAX_CONCURRENT_AUDITS caps the watcher at one live auditor",
              cw.watch_once()["launched"] == 0 and cw.MAX_CONCURRENT_AUDITS == 1)


# =====================================================================
# 7 + 8.  Claim hygiene: a fresh nonce per attempt, and no file the
# watcher did not create is ever overwritten, truncated or deleted.
# =====================================================================
def case_claim_hygiene() -> None:
    section("claim hygiene: fresh nonces, no clobbering, no silent deletion")

    with Fixture() as fx:
        enqueue_prospective(fx)
        fx.stub_mode("crash")
        for _round in range(2):
            row = fx.snapshot()["pending_rows"][0]
            cw.launch_entry(row, STUB_IDENTITY)
            aa.record_attempt_failure(
                attempt_id=fx.events_of("attempt_started")[-1]["attempt_id"],
                reason="TEST_CLOSED_THE_ATTEMPT", path=fx.events,
                lock_path=fx.lock)
        calls = wait_for_stub_calls(fx, 2)
        nonces = [_nonce_of(call["argv"]) for call in calls]
        starts = fx.events_of("attempt_started")
        check("a retried entry is launched again after its failure is durable",
              len(calls) == 2 and len(starts) == 2, str(len(calls)))
        check("every attempt carries its own fresh 64-hex nonce",
              all(nonces) and len(set(nonces)) == 2, str(nonces))
        check("every attempt carries its own durable attempt id",
              len({start["attempt_id"] for start in starts}) == 2)
        check("the durable rows record two different nonce hashes",
              len({start["attempt_nonce_sha256"] for start in starts}) == 2)
        check("no nonce handed to an auditor ever appears in the journal",
              not any(nonce in fx.events.read_text(encoding="utf-8")
                      for nonce in nonces if nonce))
        reap_stub_calls(calls)

    with Fixture() as fx:
        enqueue_prospective(fx)
        fx.stub_mode("crash")
        fixed_attempt_id(fx, "collision-attempt")
        marker = ac.marker_path(RUN_ENTRY, "collision-attempt")
        rival = b'{"artifact_type": "audit_running_marker", "attempt_id": "rival"}\n'
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(rival)
        outcome = cw.launch_entry(fx.snapshot()["pending_rows"][0], STUB_IDENTITY)
        check("an existing claim marker is never overwritten",
              marker.read_bytes() == rival,
              marker.read_bytes()[:80].decode("utf-8", "replace"))
        check("the durable attempt row, not the marker, is what claims the "
              "entry, so a marker collision does not lose the audit",
              outcome == 1 and len(fx.events_of("attempt_started")) == 1)
        reap_stub_calls(wait_for_stub_calls(fx, 1))

    with Fixture() as fx:
        entry = enqueue_prospective(fx)
        fx.stub_mode("crash")
        fixed_attempt_id(fx, "log-collision")
        log = fx.auto / f"audit_{RUN_ENTRY}.log-collision.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_bytes(b"PRIOR AUDIT EVIDENCE\n")
        raised: BaseException | None = None
        try:
            cw.launch_entry(fx.snapshot()["pending_rows"][0], STUB_IDENTITY)
        except BaseException as exc:  # noqa: BLE001 - the test reports it
            raised = exc
        check("an existing audit log is never truncated or reused",
              log.read_bytes() == b"PRIOR AUDIT EVIDENCE\n")
        check("no auditor is started when its log name is already taken",
              not fx.stub_log.exists() and isinstance(raised, FileExistsError),
              repr(raised))
        check("the entry is not promotable after the aborted launch",
              fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                          entry["packet_sha256"]).promotion_eligible is False)
        check("DEFECT, pinned: an aborted launch leaves a durable open attempt "
              "that only the stale reaper can clear",
              len(fx.events_of("attempt_started")) == 1
              and fx.events_of("attempt_failed") == []
              and fx.pending_ids() == [])
        cw.reap_abandoned_attempts(
            now_epoch=time.time() + cw.STALE_ATTEMPT_SECONDS + 1)
        check("the stale reaper does recover the aborted launch",
              len(fx.events_of("attempt_failed")) == 1
              and fx.pending_ids() == [RUN_ENTRY])

    with Fixture() as fx:
        entry = enqueue_prospective(fx)
        fx.start("live-attempt", entry_id=RUN_ENTRY,
                 candidate_sha256=entry["candidate_sha256"],
                 packet_sha256=entry["packet_sha256"], nonce=NONCE,
                 measurement=entry["measurement_event_sha256"],
                 lane="primary", queue=entry["queue_event_sha256"])
        live_marker = ac.marker_path(RUN_ENTRY, "live-attempt")
        aa.exclusive_write_json(live_marker, {"entry_id": RUN_ENTRY,
                                              "attempt_id": "live-attempt",
                                              "pid": os.getpid()})
        broken = fx.auto / f"audit_{RUN_ENTRY}.truncated.running.json"
        broken.write_bytes(b'{"entry_id": "run-", "attempt_id": "trunc')
        cw.clean_terminal_markers()
        check("a live attempt's claim marker survives marker cleanup",
              live_marker.exists())
        check("an unreadable claim marker is preserved for the owner, "
              "never deleted", broken.exists())
        aa.record_attempt_failure(attempt_id="live-attempt",
                                  reason="TEST_CLOSED_THE_ATTEMPT",
                                  path=fx.events, lock_path=fx.lock)
        cw.clean_terminal_markers()
        check("once the attempt is terminal its marker is cleaned up",
              not live_marker.exists() and broken.exists())

        original_os = cw.os
        try:
            cw.os = OsShim(PermissionError(1, "operation not permitted"))
            check("a pid owned by another user counts as ALIVE, so a foreign "
                  "auditor is never reclaimed", cw._pid_alive(4242) is True)
            cw.os = OsShim(ProcessLookupError(3, "no such process"))
            check("a vanished pid counts as dead", cw._pid_alive(4242) is False)
        finally:
            cw.os = original_os

    for lane in ("shape6", "shape14"):
        with Fixture() as fx:
            entry = enqueue_prospective(
                fx, source=f"VALUE = '{lane}'\n".encode(), lane=lane)
            fx.stub_mode("crash")
            cw.watch_once()
            calls = wait_for_stub_calls(fx, 1)
            argv = calls[0]["argv"] if calls else []
            check(f"{lane}: the auditor is told the entry's real lane",
                  "--lane" in argv
                  and argv[argv.index("--lane") + 1] == lane, str(argv))
            check(f"{lane}: the auditor is told the exact measurement event",
                  "--measurement-event-sha256" in argv
                  and argv[argv.index("--measurement-event-sha256") + 1]
                  == entry["measurement_event_sha256"], str(argv))
            check(f"{lane}: the auditor is told the entry and its attempt id",
                  argv[:1] == [RUN_ENTRY] and "--attempt-id" in argv, str(argv))
            reap_stub_calls(calls)


def enqueue_mismatched(fx: Fixture, *, packet_candidate: bytes | None = None,
                       packet_lane: str | None = None,
                       packet_measurement: str | None = None) -> dict:
    """Queue an entry whose packet disagrees with its durable queue binding."""
    source = b"def custom_kernel(x):\n    return x\n"
    candidate_sha = hashlib.sha256(source).hexdigest()
    (fx.blobs / f"{candidate_sha}.py").write_bytes(source)
    measurement = "8" * 64
    packet_source = packet_candidate if packet_candidate is not None else source
    packet_candidate_sha = hashlib.sha256(packet_source).hexdigest()
    (fx.blobs / f"{packet_candidate_sha}.py").write_bytes(packet_source)
    packet = {
        "schema_version": 1,
        "entry_id": RUN_ENTRY,
        "candidate_sha256": packet_candidate_sha,
        "measurement_event_sha256": packet_measurement or measurement,
        "lane": packet_lane or "primary",
    }
    raw = aa._canonical(packet)
    packet_sha = hashlib.sha256(raw).hexdigest()
    (fx.blobs / f"{packet_sha}.json").write_bytes(raw)
    aa.enqueue_audit(entry_id=RUN_ENTRY, candidate_sha256=candidate_sha,
                     packet_sha256=packet_sha,
                     measurement_event_sha256=measurement, lane="primary",
                     path=fx.events, lock_path=fx.lock)
    return {"entry_id": RUN_ENTRY, "candidate_sha256": candidate_sha,
            "packet_sha256": packet_sha}


def case_packet_preflight_binding() -> None:
    section("a packet that disagrees with its queue binding is never audited")
    scenarios = {
        "packet names different candidate bytes":
            {"packet_candidate": b"BACKDOOR = 1\n"},
        "packet names a different lane": {"packet_lane": "shape14"},
        "packet names a different measurement": {"packet_measurement": "7" * 64},
    }
    for name, kwargs in scenarios.items():
        with Fixture() as fx:
            entry = enqueue_mismatched(fx, **kwargs)
            summary, exc = safe_watch()
            check(f"{name}: the watcher does not launch an auditor",
                  summary is not None and summary["launched"] == 0
                  and not fx.stub_log.exists(),
                  f"exc={exc!r} summary={summary}")
            failures = fx.events_of("attempt_failed")
            check(f"{name}: the refusal is recorded durably",
                  len(failures) == 1
                  and failures[0]["reason"].startswith("PACKET_PREFLIGHT_FAILED"),
                  str(failures))
            check(f"{name}: the attempt is counted against the retry cap",
                  aa.attempt_summary(RUN_ENTRY, events_path=fx.events)[
                      "failed_attempts"] == 1)
            check(f"{name}: the entry stays ineligible",
                  fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                              entry["packet_sha256"]).promotion_eligible is False)

    with Fixture() as fx:
        enqueue_mismatched(fx, packet_candidate=b"BACKDOOR = 1\n")
        enqueue_prospective(fx, entry_id=RUN_ENTRY_2, source=b"VALUE = 2\n",
                            lane="shape6")
        fx.stub_mode("crash")
        summary, exc = safe_watch()
        check("a failed preflight consumes the pass's single-audit budget",
              summary is not None and summary["attempted"] == 1
              and summary["launched"] == 0, f"exc={exc!r} summary={summary}")
        starts = fx.events_of("attempt_started")
        check("a healthy second entry is not swept up in the same pass",
              len(starts) == 1 and starts[0]["entry_id"] == RUN_ENTRY
              and not fx.stub_log.exists(), str(starts))
        check("the healthy second entry stays queued for the next pass",
              RUN_ENTRY_2 in fx.pending_ids(), str(fx.pending_ids()))

    # champion_watch.py:216-217 re-compares packet.sha256 to the queue
    # binding.  load_bound_packet returns the exact hash it was asked for
    # (after verifying the blob bytes against that name), so that branch is
    # unreachable -- deleting it is an EQUIVALENT mutation that no test can
    # kill.  These checks pin the invariant that makes it dead code.
    with Fixture() as fx:
        entry = enqueue_prospective(fx)
        loaded = aa.load_bound_packet(RUN_ENTRY,
                                      packet_sha256=entry["packet_sha256"],
                                      packets_dir=fx.packets,
                                      authority_blobs=fx.blobs)
        check("load_bound_packet returns exactly the packet hash requested",
              loaded.sha256 == entry["packet_sha256"])
        (fx.blobs / f"{entry['packet_sha256']}.json").write_bytes(b"{}\n")
        check("a blob whose bytes stop matching its name is refused by the "
              "loader itself, before any watcher-side comparison",
              refuses(aa.load_bound_packet, RUN_ENTRY,
                      packet_sha256=entry["packet_sha256"],
                      packets_dir=fx.packets, authority_blobs=fx.blobs))
        summary, exc = safe_watch()
        check("a corrupted packet blob blocks the launch and is recorded "
              "durably",
              summary is not None and summary["launched"] == 0
              and not fx.stub_log.exists()
              and len(fx.events_of("attempt_failed")) == 1,
              f"exc={exc!r} summary={summary}")
        check("the corrupted-packet entry stays ineligible",
              fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                          entry["packet_sha256"]).promotion_eligible is False)


def case_launch_budget() -> None:
    section("launch budget: one auditor per pass, never a burst")
    with Fixture() as fx:
        enqueue_prospective(fx)
        enqueue_prospective(fx, entry_id=RUN_ENTRY_2, source=b"VALUE = 2\n",
                            lane="shape6")
        fx.stub_mode("crash")
        check("two entries are queued", len(fx.pending_ids()) == 2)
        code, _out = run_main(["--max-launches", "5"])
        check("--max-launches cannot exceed MAX_CONCURRENT_AUDITS",
              code == 0 and len(fx.events_of("attempt_started")) == 1,
              str(len(fx.events_of("attempt_started"))))
        wait_for_stub_calls(fx, 1)
        check("exactly one auditor process was started for the whole pass",
              len(wait_for_stub_calls(fx, 2, timeout=1.0)) == 1)
        for marker in fx.auto.glob("audit_*.running.json"):
            kill_and_reap(int(json.loads(marker.read_text())["pid"]))
    with Fixture() as fx:
        enqueue_prospective(fx)
        fx.stub_mode("crash")
        check("--max-launches 0 launches nothing",
              cw.watch_once(max_launches=0)["launched"] == 0
              and fx.events_of("attempt_started") == [])
        check("a negative launch budget launches nothing",
              cw.watch_once(max_launches=-5)["launched"] == 0
              and fx.events_of("attempt_started") == [])
        check("the entry is still queued afterwards",
              fx.pending_ids() == [RUN_ENTRY])
    with Fixture() as fx:
        first = enqueue_prospective(fx)
        enqueue_prospective(fx, entry_id=RUN_ENTRY_2, source=b"VALUE = 2\n",
                            lane="shape6")
        fx.start("already-running", entry_id=RUN_ENTRY,
                 candidate_sha256=first["candidate_sha256"],
                 packet_sha256=first["packet_sha256"], nonce=NONCE,
                 measurement=first["measurement_event_sha256"], lane="primary",
                 queue=first["queue_event_sha256"])
        fx.stub_mode("crash")
        summary, exc = safe_watch(max_launches=4)
        check("a second entry is not audited while one auditor is live",
              summary is not None and summary["launched"] == 0
              and summary["attempted"] == 0
              and len(fx.events_of("attempt_started")) == 1
              and not fx.stub_log.exists(),
              f"exc={exc!r} summary={summary}")
        check("the waiting entry is reported as pending, not lost",
              summary is not None and summary["pending"] == [RUN_ENTRY_2]
              and summary["active"] == [RUN_ENTRY], str(summary))


def _nonce_of(argv: list[str]) -> str | None:
    if "--nonce" not in argv:
        return None
    value = argv[argv.index("--nonce") + 1]
    return value if aa.NONCE_RE.fullmatch(value) else None


def _marker_is_exclusive(fx: Fixture, start: dict) -> bool:
    path = ac.marker_path(start["entry_id"], start["attempt_id"])
    before = path.read_bytes()
    try:
        aa.exclusive_write_json(path, {"attempt_id": "stolen"})
    except FileExistsError:
        return path.read_bytes() == before
    return False


# =====================================================================
# 8.  Stale markers are reclaimed; live ones are respected.
# =====================================================================
def case_stale_and_live_markers() -> None:
    section("claim markers: stale reclaimed, live respected")
    with Fixture() as fx:
        entry = enqueue_prospective(fx)
        start = fx.start("attempt-a", entry_id=RUN_ENTRY,
                         candidate_sha256=entry["candidate_sha256"],
                         packet_sha256=entry["packet_sha256"], nonce=NONCE,
                         measurement=entry["measurement_event_sha256"],
                         lane="primary", queue=entry["queue_event_sha256"])
        marker = ac.marker_path(RUN_ENTRY, "attempt-a")

        # (a) live pid: never reclaimed, no matter how old.
        aa.exclusive_write_json(marker, {"entry_id": RUN_ENTRY,
                                         "attempt_id": "attempt-a",
                                         "pid": os.getpid()})
        cw.reap_abandoned_attempts(now_epoch=time.time() + 10 ** 6)
        check("a live auditor pid is never reclaimed",
              fx.events_of("attempt_failed") == [] and marker.exists())
        check("a live attempt keeps the entry out of the pending queue",
              fx.pending_ids() == [])
        check("a live-but-stalled attempt is still not promotable",
              fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                          entry["packet_sha256"]).promotion_eligible is False)

        # (b) dead pid, inside the grace window: not yet reclaimed.
        dead_pid = _dead_pid()
        marker.unlink()
        aa.exclusive_write_json(marker, {"entry_id": RUN_ENTRY,
                                         "attempt_id": "attempt-a",
                                         "pid": dead_pid})
        cw.reap_abandoned_attempts(now_epoch=time.time())
        check("a dead auditor inside the grace window is not yet reclaimed",
              fx.events_of("attempt_failed") == [])

        # (c) dead pid, past the stale window: reclaimed durably.
        cw.reap_abandoned_attempts(
            now_epoch=time.time() + cw.STALE_ATTEMPT_SECONDS + 1)
        failures = fx.events_of("attempt_failed")
        check("a dead auditor past the stale window becomes a durable failure",
              len(failures) == 1
              and failures[0]["reason"].startswith("AUDITOR_PROCESS_ABANDONED"),
              str(failures))
        check("the reclaimed marker is deleted", not marker.exists())
        check("the reclaimed entry returns to the pending queue",
              fx.pending_ids() == [RUN_ENTRY])

        # (d) a start with no marker at all is reclaimed after the window.
        fx.start("attempt-b", entry_id=RUN_ENTRY,
                 candidate_sha256=entry["candidate_sha256"],
                 packet_sha256=entry["packet_sha256"], nonce=NONCE_2,
                 measurement=entry["measurement_event_sha256"],
                 lane="primary", queue=entry["queue_event_sha256"])
        cw.reap_abandoned_attempts(
            now_epoch=time.time() + cw.STALE_ATTEMPT_SECONDS + 1)
        check("an attempt with no marker at all is reclaimed after the window",
              len(fx.events_of("attempt_failed")) == 2)

        # (e) a marker whose attempt id does not match is not trusted.
        fx.start("attempt-c", entry_id=RUN_ENTRY,
                 candidate_sha256=entry["candidate_sha256"],
                 packet_sha256=entry["packet_sha256"], nonce=NONCE_3,
                 measurement=entry["measurement_event_sha256"],
                 lane="primary", queue=entry["queue_event_sha256"])
        mismatched = ac.marker_path(RUN_ENTRY, "attempt-c")
        aa.exclusive_write_json(mismatched, {"entry_id": RUN_ENTRY,
                                             "attempt_id": "someone-else",
                                             "pid": os.getpid()})
        cw.reap_abandoned_attempts(
            now_epoch=time.time() + cw.STALE_ATTEMPT_SECONDS + 1)
        check("a marker naming a different attempt cannot shield the attempt",
              len(fx.events_of("attempt_failed")) == 3)
        check("terminal markers are cleaned up after the attempt closes",
              _terminal_marker_cleanup(fx, start))


def _dead_pid() -> int:
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def _terminal_marker_cleanup(fx: Fixture, start: dict) -> bool:
    marker = ac.marker_path(start["entry_id"], start["attempt_id"])
    if not marker.exists():
        aa.exclusive_write_json(marker, {"entry_id": start["entry_id"],
                                         "attempt_id": start["attempt_id"],
                                         "pid": os.getpid()})
    cw.clean_terminal_markers()
    return not marker.exists()


# =====================================================================
# 9.  Malformed or truncated authority lines fail closed.
# =====================================================================
def case_corrupt_journals_fail_closed() -> None:
    section("malformed or truncated authority lines refuse, never skip")
    corruptions = {
        "truncated final event line":
            lambda fx: fx.events.write_text(
                fx.events.read_text(encoding="utf-8")[:-25], encoding="utf-8"),
        "blank line inside the event journal":
            lambda fx: fx.events.write_text(
                fx.events.read_text(encoding="utf-8") + "\n", encoding="utf-8"),
        "flipped byte breaks the hash chain":
            lambda fx: fx.events.write_text(
                fx.events.read_text(encoding="utf-8").replace(
                    '"lane":"primary"', '"lane":"shape6"'), encoding="utf-8"),
        "duplicate JSON key in an event":
            lambda fx: fx.events.write_text(
                fx.events.read_text(encoding="utf-8").replace(
                    '{"candidate_sha256"',
                    '{"seq":1,"candidate_sha256"'), encoding="utf-8"),
        "unknown event type":
            lambda fx: fx.events.write_text(
                fx.events.read_text(encoding="utf-8").replace(
                    '"event_type":"audit_enqueued"',
                    '"event_type":"audit_waived"'), encoding="utf-8"),
        "malformed result journal row":
            lambda fx: fx.journal.write_text(
                '{"entry_id": "20260830-120000-abcdef", "type": "candi',
                encoding="utf-8"),
        "blank line in the result journal":
            lambda fx: fx.journal.write_text(
                fx.journal.read_text(encoding="utf-8") + "\n\n",
                encoding="utf-8"),
        "deleted result journal":
            lambda fx: fx.journal.unlink(),
        "conflicting duplicate journal entry id":
            lambda fx: fx.journal.write_text(
                json.dumps({"entry_id": LEGACY_ENTRY, "type": "candidate",
                            "promoted": True, "impl": {"sha256": CANDIDATE}})
                + "\n"
                + json.dumps({"entry_id": LEGACY_ENTRY, "type": "candidate",
                              "promoted": True, "impl": {"sha256": "a" * 64}})
                + "\n", encoding="utf-8"),
        "unknown legacy verdict word":
            lambda fx: fx.legacy.write_text(
                json.dumps({"entry_id": LEGACY_ENTRY, "verdict": "APPROVED"})
                + "\n", encoding="utf-8"),
        "legacy verdict with an invalid entry id":
            lambda fx: fx.legacy.write_text(
                json.dumps({"entry_id": "../../etc/passwd", "verdict": "PASS"})
                + "\n", encoding="utf-8"),
    }
    for name, corrupt in corruptions.items():
        with Fixture() as fx:
            fx.journal_rows([{"entry_id": LEGACY_ENTRY, "type": "candidate",
                              "promoted": True, "shape_id": 8,
                              "impl": {"sha256": CANDIDATE}}])
            enqueue_prospective(fx)
            corrupt(fx)
            refused = refuses(fx.snapshot)
            code, out = run_main([])
            check(f"{name}: refuses instead of silently skipping",
                  refused and code == 1, f"refused={refused} rc={code}")
            check(f"{name}: nothing is launched while authority is unreadable",
                  not fx.stub_log.exists() and out == "")


# =====================================================================
# 10.  A misbound verdict is rejected.
# =====================================================================
def case_verdict_binding_is_enforced() -> None:
    section("verdicts must be bound to packet, candidate, nonce and schema")
    with Fixture() as fx:
        raw = json.dumps(verdict())
        check("a correctly bound verdict parses",
              aa.validate_verdict_document(
                  raw, attempt_nonce=NONCE, entry_id=LEGACY_ENTRY,
                  packet_sha256=PACKET, candidate_sha256=CANDIDATE
              )["integrity"]["verdict"] == "PASS")
        check("wrong packet hash is rejected",
              refuses(aa.validate_verdict_document, raw, attempt_nonce=NONCE,
                      entry_id=LEGACY_ENTRY, packet_sha256="a" * 64,
                      candidate_sha256=CANDIDATE))
        check("wrong candidate hash is rejected",
              refuses(aa.validate_verdict_document, raw, attempt_nonce=NONCE,
                      entry_id=LEGACY_ENTRY, packet_sha256=PACKET,
                      candidate_sha256="a" * 64))
        check("wrong nonce is rejected",
              refuses(aa.validate_verdict_document, raw, attempt_nonce=NONCE_2,
                      entry_id=LEGACY_ENTRY, packet_sha256=PACKET,
                      candidate_sha256=CANDIDATE))
        check("wrong entry id is rejected",
              refuses(aa.validate_verdict_document, raw, attempt_nonce=NONCE,
                      entry_id=LEGACY_ENTRY_2, packet_sha256=PACKET,
                      candidate_sha256=CANDIDATE))
        check("a bare decoy PASS is rejected",
              refuses(aa.validate_verdict_document, '{"verdict": "PASS"}',
                      attempt_nonce=NONCE, entry_id=LEGACY_ENTRY,
                      packet_sha256=PACKET, candidate_sha256=CANDIDATE))
        check("a banner plus a JSON object is rejected",
              refuses(aa.validate_verdict_document, "thinking...\n" + raw,
                      attempt_nonce=NONCE, entry_id=LEGACY_ENTRY,
                      packet_sha256=PACKET, candidate_sha256=CANDIDATE))
        check("a nonzero auditor exit code is rejected",
              refuses(aa.validate_verdict_document, raw, attempt_nonce=NONCE,
                      entry_id=LEGACY_ENTRY, packet_sha256=PACKET,
                      candidate_sha256=CANDIDATE, returncode=1))

    with Fixture() as fx:
        fx.start("bound")
        check("a result whose nonce is not the attempt's nonce is refused",
              refuses(fx.result, "bound", verdict(nonce=NONCE_2),
                      artifact_name="wrong-nonce"))
        check("a result bound to another packet is refused",
              refuses(fx.result, "bound", verdict(packet_sha256="a" * 64),
                      artifact_name="wrong-packet"))
        check("a result bound to other candidate bytes is refused",
              refuses(fx.result, "bound", verdict(candidate_sha256="a" * 64),
                      artifact_name="wrong-candidate"))
        check("a response artifact outside the protected stores is refused",
              refuses(fx.result, "bound", verdict(), artifact_dir=fx.root,
                      artifact_name="outside-store"))
        check("a response artifact whose stdout differs from the result is "
              "refused",
              refuses(fx.result, "bound", verdict(),
                      artifact_name="stdout-drift",
                      stdout=json.dumps(verdict("RULE_VIOLATION"))))
        check("a response artifact claiming another attempt is refused",
              refuses(fx.result, "bound", verdict(),
                      artifact_name="wrong-attempt",
                      artifact_overrides={"attempt_id": "somebody-else"}))
        check("a response artifact carrying a parser error is refused",
              refuses(fx.result, "bound", verdict(),
                      artifact_name="parser-error",
                      artifact_overrides={"parser_error": "banner stripped"}))
        fx.result("bound", verdict(), artifact_name="clean")
        check("the correctly bound result is eligible",
              fx.decision(LEGACY_ENTRY, CANDIDATE).promotion_eligible is True)

        fx.schema.write_bytes(REAL_SCHEMA_BYTES.replace(b'"maxItems": 100',
                                                        b'"maxItems": 101'))
        changed = fx.decision(LEGACY_ENTRY, CANDIDATE)
        check("a verdict schema change since the audit revokes eligibility",
              changed.promotion_eligible is False
              and "verdict_schema_changed_since_audit" in changed.blocking_reasons,
              str(changed.blocking_reasons))
        fx.schema.write_bytes(REAL_SCHEMA_BYTES)

        artifact = fx.auto / "clean.json"
        artifact.write_text("{}\n", encoding="utf-8")
        tampered = fx.decision(LEGACY_ENTRY, CANDIDATE)
        check("editing the stored response artifact revokes eligibility",
              tampered.promotion_eligible is False
              and any(reason.startswith("response_artifact_invalid:")
                      for reason in tampered.blocking_reasons),
              str(tampered.blocking_reasons))

    with Fixture() as fx:
        entry = enqueue_prospective(fx)
        fx.start("mismatched-queue", entry_id=RUN_ENTRY,
                 candidate_sha256=entry["candidate_sha256"],
                 packet_sha256=entry["packet_sha256"], nonce=NONCE,
                 measurement=entry["measurement_event_sha256"], lane="primary",
                 queue=entry["queue_event_sha256"])
        fx.result("mismatched-queue",
                  verdict(entry_id=RUN_ENTRY,
                          candidate_sha256=entry["candidate_sha256"],
                          packet_sha256=entry["packet_sha256"]),
                  entry_id=RUN_ENTRY,
                  candidate_sha256=entry["candidate_sha256"],
                  packet_sha256=entry["packet_sha256"],
                  measurement=entry["measurement_event_sha256"], lane="primary")
        check("a fully bound prospective audit is eligible",
              fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                          entry["packet_sha256"]).promotion_eligible is True)
        (fx.blobs / f"{entry['candidate_sha256']}.py").write_bytes(b"tampered\n")
        drifted = fx.decision(RUN_ENTRY, entry["candidate_sha256"],
                              entry["packet_sha256"])
        check("candidate bytes changing after the audit revokes eligibility",
              drifted.promotion_eligible is False
              and any(reason.startswith("prospective_binding_invalid:")
                      for reason in drifted.blocking_reasons),
              str(drifted.blocking_reasons))
        check("a run- entry audited without a lane binding is refused",
              refuses(fx.start, "no-lane", entry_id=RUN_ENTRY_2,
                      candidate_sha256=CANDIDATE, packet_sha256=PACKET,
                      nonce=NONCE_4))


# =====================================================================
# 11.  --dry-run and --no-reconcile mutate nothing.
# =====================================================================
def case_dry_run_and_no_reconcile() -> None:
    section("--dry-run and --no-reconcile")
    with Fixture() as fx:
        enqueue_prospective(fx)
        fx.journal_rows([{"entry_id": LEGACY_ENTRY, "type": "candidate",
                          "promoted": True, "shape_id": 8,
                          "impl": {"sha256": CANDIDATE}}])
        before = tree_snapshot(fx.root)
        code, out = run_main(["--dry-run"])
        after = tree_snapshot(fx.root)
        check("--dry-run exits 0", code == 0)
        check("--dry-run mutates no file in the repository", before == after,
              str(sorted(set(after) ^ set(before))))
        check("--dry-run launches no auditor process", not fx.stub_log.exists())
        check("--dry-run runs no gate reconcile", fx.reconcile_calls == [])
        check("--dry-run records no durable audit event",
              fx.events_of("attempt_started") == [])
        parsed = json.loads(out)
        check("--dry-run prints the real backlog it would work on",
              parsed["pending"] == [RUN_ENTRY, LEGACY_ENTRY]
              and parsed["launched"] == 0 and parsed["attempted"] == 0,
              out.strip())

        code, out = run_main(["--dry-run", "--no-reconcile"])
        check("--dry-run --no-reconcile still mutates nothing",
              code == 0 and tree_snapshot(fx.root) == before
              and fx.reconcile_calls == [])

    with Fixture() as fx:
        before = tree_snapshot(fx.root)
        code, _out = run_main(["--no-reconcile"])
        check("--no-reconcile on an empty backlog mutates nothing",
              code == 0 and tree_snapshot(fx.root) == before)
        check("--no-reconcile runs no gate reconcile subprocess",
              fx.reconcile_calls == [])
        code, _out = run_main([])
        check("without --no-reconcile the gate reconcile does run",
              code == 0 and len(fx.reconcile_calls) == 1)
        check("the reconcile target is run_gate.py",
              cw.RUN_GATE.name == "run_gate.py")

    with Fixture() as fx:
        enqueue_prospective(fx)
        fx.stub_mode("crash")
        code, _out = run_main(["--no-reconcile"])
        check("--no-reconcile still performs real audit work",
              code == 0 and len(fx.events_of("attempt_started")) == 1
              and fx.reconcile_calls == [])
        for marker in fx.auto.glob("audit_*.running.json"):
            kill_and_reap(int(json.loads(marker.read_text())["pid"]))

    # A dry run must not touch state that a real pass WOULD change: the
    # marker sweeper and the stale-attempt reaper are both mutating.
    with Fixture() as fx:
        entry = enqueue_prospective(fx)
        fx.stub_mode("crash")
        fx.start("closed-attempt", entry_id=RUN_ENTRY,
                 candidate_sha256=entry["candidate_sha256"],
                 packet_sha256=entry["packet_sha256"], nonce=NONCE,
                 measurement=entry["measurement_event_sha256"],
                 lane="primary", queue=entry["queue_event_sha256"])
        closed_marker = ac.marker_path(RUN_ENTRY, "closed-attempt")
        aa.exclusive_write_json(closed_marker, {"entry_id": RUN_ENTRY,
                                                "attempt_id": "closed-attempt",
                                                "pid": os.getpid()})
        aa.record_attempt_failure(attempt_id="closed-attempt",
                                  reason="TEST_CLOSED_THE_ATTEMPT",
                                  path=fx.events, lock_path=fx.lock)
        with frozen_clock():
            fx.start("abandoned-attempt", entry_id=RUN_ENTRY,
                     candidate_sha256=entry["candidate_sha256"],
                     packet_sha256=entry["packet_sha256"], nonce=NONCE_2,
                     measurement=entry["measurement_event_sha256"],
                     lane="primary", queue=entry["queue_event_sha256"])
        aa.exclusive_write_json(
            ac.marker_path(RUN_ENTRY, "abandoned-attempt"),
            {"entry_id": RUN_ENTRY, "attempt_id": "abandoned-attempt",
             "pid": _dead_pid()})
        before = tree_snapshot(fx.root)
        events_before = fx.events.read_bytes()
        code, out = run_main(["--dry-run"])
        check("--dry-run does not sweep terminal markers and does not reap "
              "an abandoned attempt",
              code == 0 and tree_snapshot(fx.root) == before
              and fx.events.read_bytes() == events_before,
              str(sorted(set(tree_snapshot(fx.root).items())
                         ^ set(before.items()))[:3]))
        check("--dry-run reports the abandoned attempt as active, not pending",
              json.loads(out)["active"] == [RUN_ENTRY]
              and json.loads(out)["pending"] == [], out.strip())
        check("--dry-run left the abandoned attempt open, not failed",
              fx.events_of("attempt_failed")[-1]["attempt_id"]
              == "closed-attempt")

        # Proof the assertion above is not vacuous: a real pass DOES mutate
        # exactly this state.
        code, _out = run_main(["--no-reconcile"])
        reaped = [event["attempt_id"] for event in fx.events_of("attempt_failed")]
        check("a real pass sweeps the terminal marker and reaps the "
              "abandoned attempt (so the dry-run check is meaningful)",
              code == 0 and not closed_marker.exists()
              and "abandoned-attempt" in reaped, str(reaped))
        reap_stub_calls(wait_for_stub_calls(fx, 1))


# =====================================================================
# Extra: idleness is fail-closed and concurrency is capped.
# =====================================================================
def case_runner_busy_is_fail_closed() -> None:
    section("auditor never competes with a live benchmark")

    class FakeCompleted:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    class FakeSubprocess:
        DEVNULL = -3
        STDOUT = -2

        def __init__(self, behaviour) -> None:
            self.behaviour = behaviour
            self.calls: list[list[str]] = []

        def run(self, command, **kwargs):
            self.calls.append(command)
            return self.behaviour(command)

    with Fixture() as fx:
        original = cw.subprocess
        try:
            probe = FakeSubprocess(lambda command: FakeCompleted(""))
            cw.subprocess = probe
            check("an idle box is not busy", REAL_RUNNER_BUSY() is False)
            check("idleness is probed for runner and controller runs",
                  [command[2] for command in probe.calls]
                  == ["runner.py (run|calibrate)",
                      "trusted_controller.py (run|calibrate)"],
                  str(probe.calls))
            cw.subprocess = FakeSubprocess(lambda command: FakeCompleted("4321\n"))
            check("a live runner marks the box busy", REAL_RUNNER_BUSY() is True)

            def explode(command):
                raise OSError("pgrep unavailable")

            cw.subprocess = FakeSubprocess(explode)
            check("inability to establish idleness is not permission",
                  REAL_RUNNER_BUSY() is True)
        finally:
            cw.subprocess = original

        enqueue_prospective(fx)
        cw.runner_busy = lambda: True
        summary = cw.watch_once()
        check("a busy box launches nothing but still reports the backlog",
              summary["launched"] == 0 and summary["pending"] == [RUN_ENTRY])
        check("a busy box records no durable attempt",
              fx.events_of("attempt_started") == [])
        cw.runner_busy = lambda: False


# =====================================================================
# Safety: the real repository is untouched.
# =====================================================================
def real_state_snapshot() -> dict[str, object]:
    out: dict[str, object] = {}
    for base in (REPO / "Project" / "audits", REPO / "Project" / "results",
                 REPO / "Project" / "authority"):
        out[str(base)] = base.exists()
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            stat = path.stat()
            out[str(path)] = ("dir" if path.is_dir()
                              else (stat.st_size, stat.st_mtime_ns))
    return out


def case_real_repository_untouched(before: dict[str, object]) -> None:
    section("isolation")
    after = real_state_snapshot()
    changed = sorted(key for key in set(before) | set(after)
                     if before.get(key) != after.get(key))
    check("no file under Project/audits, results or authority changed",
          changed == [], str(changed[:6]))
    check("the real audit event journal was never created by these tests",
          before.get(str(aa.AUDIT_EVENTS)) == after.get(str(aa.AUDIT_EVENTS)))
    check("module paths were restored after every fixture",
          aa.SCHEMA_PATH == REPO / "Project" / "audits" / "verdict_schema.json"
          and cw.ROOT == REPO
          and cw.AUDIT_LOG_DIR == REPO / "Project" / "audits" / "auto")


def main() -> int:
    before = real_state_snapshot()
    case_backlog_from_durable_journals()
    case_killed_auditor_stays_ineligible()
    case_response_artifacts_are_not_evidence()
    case_missing_verdict_blocks_promotion()
    case_hard_verdict_first_write_wins()
    case_hard_verdict_blocks_the_watcher_queue()
    case_technical_channel_is_separate()
    case_retry_cap_from_durable_rows()
    case_concurrent_watchers_single_claim()
    case_claim_hygiene()
    case_launch_budget()
    case_packet_preflight_binding()
    case_stale_and_live_markers()
    case_corrupt_journals_fail_closed()
    case_verdict_binding_is_enforced()
    case_dry_run_and_no_reconcile()
    case_runner_busy_is_fail_closed()
    case_real_repository_untouched(before)

    failed = [name for name, ok in results if not ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} passed"
          + (" — ALL GREEN" if not failed else f" — FAILURES: {failed}"))
    return 1 if failed else 0


def run_mutation_sweep() -> int:
    """Replay every mutation; report which ones this suite fails to detect."""
    import subprocess

    survivors: list[str] = []
    for index, (name, _old, _new, equivalent) in enumerate(MUTANTS):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            env={**os.environ, "CW_TEST_MUTANT": str(index),
                 "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=900, check=False)
        detected = proc.returncode != 0
        if detected:
            note = " (documented as equivalent, yet detected)" if equivalent else ""
            print(f"KILLED    [{index:2d}] {name}{note}")
        elif equivalent:
            print(f"survived  [{index:2d}] {name} (equivalent mutation, expected)")
        else:
            print(f"SURVIVED  [{index:2d}] {name}")
            survivors.append(name)
    print()
    print(f"{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutations detected"
          + (" — no undetected breakage" if not survivors
             else f" — UNDETECTED: {survivors}"))
    return 1 if survivors else 0


if __name__ == "__main__":
    if "--mutants" in sys.argv[1:]:
        raise SystemExit(run_mutation_sweep())
    raise SystemExit(main())
