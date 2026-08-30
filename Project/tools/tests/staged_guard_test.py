#!/usr/bin/env python3
"""Adversarial tests for the exact guard bytes staged for owner LOCK."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GUARD = ROOT / "Project" / "lock_staging" / "guard_bash.py"


def decision(payload) -> str:
    result = subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if not result.stdout.strip():
        return "allow"
    value = json.loads(result.stdout)
    return value["hookSpecificOutput"]["permissionDecision"]


def bash(command: object):
    return {"tool_input": {"command": command}}


def main() -> int:
    cases = [
        ("controller status", bash("python3 Project/harness/trusted_controller.py status"), "allow"),
        ("gate status", bash("python3 Project/tools/run_gate.py status"), "allow"),
        ("build submission", bash("python3 Project/tools/build_submission.py"), "allow"),
        ("locked test", bash("python3 Project/tools/tests/trusted_controller_test.py"), "allow"),
        ("git status", bash("git status --short --branch"), "allow"),
        ("git mutable add", bash("git add -- Project/kernels/k100.py"), "allow"),
        ("git commit", bash("git commit -m measured-candidate"), "allow"),
        ("read sed", bash("sed -n 1,40p -- Project/GRIND_ENTRYPOINT.md"), "allow"),
        ("read rg", bash("rg -n permit -- Project/harness"), "allow"),
        ("gpu identity", bash("nvidia-smi -L"), "allow"),
        ("empty", bash(""), "deny"),
        ("non-string", bash(7), "deny"),
        ("non-object", [], "deny"),
        ("missing tool input", {}, "deny"),
        ("malformed json surrogate", {"tool_input": None}, "deny"),
        ("python c", bash("python3 -c print(1)"), "deny"),
        ("python module", bash("python3 -m pathlib"), "deny"),
        ("relative cd", bash("cd Project"), "deny"),
        ("chained command", bash("git status --short && python3 evil.py"), "deny"),
        ("pipe", bash("git status --short | tee out"), "deny"),
        ("redirect", bash("git status --short > Project/drafts/out"), "deny"),
        ("command substitution", bash("sha256sum -- $(pwd)"), "deny"),
        ("backticks", bash("sha256sum -- `pwd`"), "deny"),
        ("brace expansion", bash("ls -- Project/{harness,tools}"), "deny"),
        ("glob expansion", bash("ls -- Project/*"), "deny"),
        ("stash", bash("git stash"), "deny"),
        ("hard reset", bash("git reset --hard"), "deny"),
        ("checkout", bash("git checkout -- Project/harness/runner.py"), "deny"),
        ("clean", bash("git clean -fd"), "deny"),
        ("recursive delete", bash("rm -rf Project"), "deny"),
        ("find delete", bash("find Project -delete"), "deny"),
        ("copy referee", bash("cp candidate.py Project/harness/runner.py"), "deny"),
        ("install", bash("install candidate.py Project/harness/runner.py"), "deny"),
        ("dd", bash("dd if=x of=Project/manifest.json"), "deny"),
        ("rsync", bash("rsync x Project/harness/"), "deny"),
        ("shred", bash("shred Project/results/JOURNAL.jsonl"), "deny"),
        ("protected git add", bash("git add -- Project/harness/runner.py"), "deny"),
        ("direct side evaluator", bash("python3 Project/tools/shape6_local_eval.py"), "deny"),
        ("direct auditor", bash("python3 Project/tools/audit_champion.py"), "deny"),
        ("direct codex", bash("codex exec review"), "deny"),
        ("curl", bash("curl https://example.com"), "deny"),
        ("over length", bash("x" * 5000), "deny"),
    ]
    failures: list[str] = []
    for name, payload, expected in cases:
        actual = decision(payload)
        passed = actual == expected
        print(("PASS " if passed else "FAIL ") + name)
        if not passed:
            failures.append(f"{name}: expected {expected}, got {actual}")
    print(f"\n{len(failures)} failure(s)" if failures else "\nALL GREEN")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
