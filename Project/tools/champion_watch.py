#!/usr/bin/env python3
"""Champion watcher — the mechanical trigger for per-champion audits.

Runs from a PostToolUse hook after every shell command (async, ~ms when idle):
reads the runner-generated leaderboard, compares the current champion entry ids
against a cache, and for every NEWLY crowned champion launches a DETACHED
background audit (Project/tools/audit_champion.py) that runs codex read-only
and records the verdict via the frozen runner's `record-verdict` command.

Design notes:
- Lives OUTSIDE the frozen harness (tools/); reads results, never writes them —
  verdict recording goes through the pinned runner.
- Fires mechanically, not by agent choice (owner's requirement, 28 Aug).
- The cache marks a champion as handled the moment its audit LAUNCHES, so
  repeated hook invocations never double-fire.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD = ROOT / "Project" / "results" / "LEADERBOARD.md"
CACHE = Path(__file__).parent / ".champion_cache.json"
AUDIT_LOG_DIR = ROOT / "Project" / "audits" / "auto"


def current_champions() -> list:
    if not LEADERBOARD.exists():
        return []
    champions = []
    for line in LEADERBOARD.read_text().splitlines():
        if "★" in line and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells:
                entry_id = cells[-1]
                if re.fullmatch(r"[0-9]{8}-[0-9]{6}(-[0-9a-f]{6})?", entry_id):
                    champions.append(entry_id)
    return champions


def run_gate_post() -> None:
    """v3: reconcile any IN_FLIGHT attempt after every shell command —
    idempotent (run_gate.py reconcile no-ops when nothing is in flight).
    Never blocks audits."""
    try:
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "run_gate.py"),
             "reconcile"],
            timeout=20, check=False, stdin=subprocess.DEVNULL)
    except Exception:
        pass


def verdict_ids() -> set:
    """Entry ids with a DURABLE verdict row — the only real 'handled'."""
    ids = set()
    try:
        for line in (ROOT / "Project" / "audits" /
                     "verdicts.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line).get("entry_id"))
            except Exception:
                pass
    except Exception:
        pass
    return ids


def running_ids() -> set:
    """Entry ids with a LIVE audit process marker. Stale markers (dead pid,
    or unparseable and old) are cleaned so crashed audits refire instead of
    being suppressed forever (reviewer round 3)."""
    import os
    import time as _t
    out = set()
    for m in AUDIT_LOG_DIR.glob("audit_*.running"):
        eid = m.name[len("audit_"):-len(".running")]
        try:
            pid = int(m.read_text().strip())
            os.kill(pid, 0)  # raises if the process is gone
            out.add(eid)
        except (ValueError, ProcessLookupError, PermissionError):
            if _t.time() - m.stat().st_mtime > 300:
                try:
                    m.unlink()
                except Exception:
                    pass
            else:
                out.add(eid)  # too fresh to declare dead
    return out


def main() -> int:
    run_gate_post()
    # Reviewer round 3: the launch-time cache marked audits handled before
    # any verdict existed (24 entries, 6 current champions, were silently
    # suppressed). 'Handled' is now DERIVED: a durable verdict row, or a
    # provably live audit process. Everything else refires. The legacy
    # cache file is ignored (kept only as history).
    champions = current_champions()
    done = verdict_ids() | running_ids()
    new = [c for c in champions if c not in done]
    if not new:
        return 0
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    import os
    for entry_id in new:
        marker = AUDIT_LOG_DIR / f"audit_{entry_id}.running"
        try:  # exclusive claim: rapid hook passes cannot double-launch
            fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        log = AUDIT_LOG_DIR / f"audit_{entry_id}.log"
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).parent / "audit_champion.py"), entry_id],
            stdin=subprocess.DEVNULL,
            stdout=open(log, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detached: survives the hook and the session
            cwd=str(ROOT),
        )
        with os.fdopen(fd, "w") as fh:
            fh.write(str(proc.pid))
        print(f"[champion-watch] new champion {entry_id} — background audit launched "
              f"(log: {log.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
