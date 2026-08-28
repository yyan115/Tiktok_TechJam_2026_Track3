#!/usr/bin/env python3
"""Detached champion auditor: evidence packet -> codex (read-only, blind) ->
verdict recorded through the frozen runner. Launched by champion_watch.py;
runs to completion on its own, no agent involvement.

Verdict handling: whatever codex returns (PASS/RETEST/NEEDS_CONTEXT/
RULE_VIOLATION) is recorded verbatim; unparseable/failed runs record
JUDGE_ERROR. Nothing here ever blocks the grind — the audit column and the
journal trail carry the outcome.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "Project" / "harness" / "runner.py"
SCHEMA = ROOT / "Project" / "audits" / "verdict_schema.json"
AUDIT_LOG_DIR = ROOT / "Project" / "audits" / "auto"

PROMPT = """You are the independent auditor for this repository's GPU-benchmark results (an automatic per-champion audit; the agent did not write this prompt for this specific result). A new champion was just crowned: journal entry {entry_id}. Its neutral evidence packet is at {packet} (JSON: the full journal entry with raw timing samples, tripwire outcomes, environment fingerprint, calibration reference, and the candidate's full source code). Audit it blind: (1) is the speedup plausible for this shape and technique, or implausibly large? (2) does the candidate source contain anything that could game the measurement (caching, timer tricks, harness tampering, input mutation) that the tripwires might have missed? (3) are the tripwire results, calibration threshold, and promotion internally consistent? (4) any rule violations against Project/PLAN.md and Project/RUNBOOK.md? Where it helps, consult outside sources. Review only, change nothing. Respond with JSON matching the provided schema."""


def wait_for_idle_runner() -> None:
    """Respect the one-runner-process rule (auditor finding): wait until no
    benchmark process is active before touching shared records."""
    for _ in range(60):
        check = subprocess.run(["pgrep", "-f", "runner.py (run|calibrate)"],
                               capture_output=True, text=True)
        if not check.stdout.strip():
            return
        time.sleep(10)


def record(entry_id: str, verdict: str, source_log: Path) -> None:
    subprocess.run(
        [sys.executable, str(RUNNER), "record-verdict", "--id", entry_id,
         "--verdict", verdict, "--source", str(source_log)],
        cwd=str(ROOT), timeout=120,
    )


def main() -> int:
    entry_id = sys.argv[1]
    log = AUDIT_LOG_DIR / f"audit_{entry_id}.log"
    print(f"[auto-audit] {time.strftime('%F %T')} starting for {entry_id}")

    packet = subprocess.run(
        [sys.executable, str(RUNNER), "packet", "--id", entry_id],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    packet_path = packet.stdout.strip().splitlines()[-1] if packet.returncode == 0 else ""
    if not packet_path:
        print(f"[auto-audit] packet generation failed:\n{packet.stdout}\n{packet.stderr}")
        record(entry_id, "JUDGE_ERROR", log)
        return 1

    try:
        result = subprocess.run(
            ["codex", "exec", "-s", "read-only",
             "--output-schema", str(SCHEMA),
             PROMPT.format(entry_id=entry_id, packet=packet_path)],
            cwd=str(ROOT), stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=2400,
        )
        output = result.stdout + result.stderr
        print(output[-4000:])
        matches = re.findall(r'\{"verdict":\s*"(PASS|RETEST|NEEDS_CONTEXT|RULE_VIOLATION)"', output)
        verdict = matches[-1] if matches else "JUDGE_ERROR"
    except subprocess.TimeoutExpired:
        verdict = "TIMEOUT"
    except Exception as exc:  # noqa: BLE001
        print(f"[auto-audit] launcher error: {exc}")
        verdict = "JUDGE_ERROR"

    sys.stdout.flush()  # the log must be on disk before its hash is recorded
    wait_for_idle_runner()
    record(entry_id, verdict, log)
    print(f"[auto-audit] {time.strftime('%F %T')} recorded {verdict} for {entry_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
