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
    """Mechanical gate-close (owner mandate 29 Aug): after ANY referee run,
    slam the two-step gate shut and count the try. Reads the PostToolUse
    payload from stdin (this hook receives it); never blocks audits."""
    try:
        payload = json.load(sys.stdin)
        command = payload.get("tool_input", {}).get("command", "") or ""
        if "runner.py" in command and "--impl" in command:
            subprocess.run(
                [sys.executable, str(Path(__file__).parent / "run_gate.py"),
                 "post", "--command", command],
                timeout=15, check=False)
    except Exception:
        pass


def main() -> int:
    run_gate_post()
    champions = current_champions()
    try:
        cache = set(json.loads(CACHE.read_text()))
    except Exception:
        cache = set()
    new = [c for c in champions if c not in cache]
    if not new:
        return 0
    # Mark handled BEFORE launching (no double-fire on rapid hook calls).
    CACHE.write_text(json.dumps(sorted(cache | set(champions))))
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    for entry_id in new:
        log = AUDIT_LOG_DIR / f"audit_{entry_id}.log"
        subprocess.Popen(
            [sys.executable, str(Path(__file__).parent / "audit_champion.py"), entry_id],
            stdin=subprocess.DEVNULL,
            stdout=open(log, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detached: survives the hook and the session
            cwd=str(ROOT),
        )
        print(f"[champion-watch] new champion {entry_id} — background audit launched "
              f"(log: {log.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
