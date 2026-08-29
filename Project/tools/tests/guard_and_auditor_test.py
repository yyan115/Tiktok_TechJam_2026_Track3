#!/usr/bin/env python3
"""Committed regression tests for (a) the owner-patch state-write guard —
extracted verbatim from OWNER_PATCH_card_gate.md so the tested bytes are
the paste bytes — and (b) the champion auditor's verdict parser."""
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PATCH = REPO / "Project" / "loop" / "OWNER_PATCH_card_gate.md"

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def load_state_write_reason():
    text = PATCH.read_text()
    m = re.search(r"(def state_write_reason\(command\):.*?\n    return None\n)",
                  text, re.DOTALL)
    assert m, "state_write_reason block not found in the owner patch"
    ns = {"re": re}
    exec(m.group(1), ns)  # the paste bytes themselves, not a copy
    return ns["state_write_reason"]


def guard_cases(fn):
    cases = [
        ("echo x >> Project/loop/gate_state.json", True),
        ("python3 Project/tools/run_gate.py status", False),
        ("python3 Project/tools/run_gate.py reconcile", False),
        ('codex exec "review this" > Project/audits/strategy/x_raw.log', False),
        ("codex exec review > Project/audits/auto/x.log", True),
        ("codex exec review > Project/tools/.champion_cache.json", True),
        ("echo forged > Project/audits/strategy/fake.log", True),
        ("sed -i s/x/y/ Project/tools/run_gate.py", True),
        ("sed -n 5,10p Project/tools/run_gate.py", True),   # sed is OUT entirely
        ("sed -n s/x/y/w\\ out.txt Project/loop/gate_state.json", True),
        ("less -O /tmp/o Project/tools/run_gate.py", True),  # less is OUT
        ("cat Project/loop/gate_log.jsonl", False),
        ("grep -n foo Project/tools/audit_champion.py", False),
        ("git add Project/loop/gate_state.json Project/loop/gate_log.jsonl", False),
        ("git log --output=Project/loop/gate_log.jsonl", True),
        ("python3 -c \"open('Project/loop/gate_state.json','w')\"", True),
        ("tail -5 Project/audits/verdicts.jsonl", False),
        ("python3 Project/tools/champion_watch.py", False),
        ("cp evil.py Project/tools/run_gate.py", True),
        ("rm Project/loop/.gate.lock", True),
        ("mv Project/loop/permits_used/x.json /tmp/", True),
        ("tee Project/audits/verdicts.jsonl < fake", True),
        ("ls Project/tools", False),
    ]
    for cmd, want_deny in cases:
        got = fn(cmd)
        check(f"guard: {cmd[:58]}", (got is not None) == want_deny, str(got))


def auditor_cases():
    spec = importlib.util.spec_from_file_location(
        "ac", REPO / "Project" / "tools" / "audit_champion.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    pv = ac.parse_verdict
    ok = '{"verdict": "PASS", "reasoning": "all checks out"}'
    check("parser: clean single doc -> PASS", pv(ok, 0) == "PASS")
    check("parser: nonzero exit -> JUDGE_ERROR", pv(ok, 1) == "JUDGE_ERROR")
    prose = 'thinking about the "verdict": "PASS" token in prose only'
    check("parser: prose token without a JSON doc -> JUDGE_ERROR",
          pv(prose, 0) == "JUDGE_ERROR")
    decoy = ('{"verdict": "PASS"}\n{\n  "verdict": "RULE_VIOLATION",\n'
             '  "reasoning": "tampering found"\n}')
    check("parser: minified decoy vs pretty real -> JUDGE_ERROR",
          pv(decoy, 0) == "JUDGE_ERROR")
    dup = '{"verdict": "RETEST"}\nnoise\n{"verdict": "RETEST"}'
    check("parser: repeated same-verdict docs -> that verdict",
          pv(dup, 0) == "RETEST")
    bad = '{"verdict": "MAYBE"}'
    check("parser: unknown verdict value -> JUDGE_ERROR",
          pv(bad, 0) == "JUDGE_ERROR")
    check("parser: empty stdout -> JUDGE_ERROR", pv("", 0) == "JUDGE_ERROR")


def main() -> int:
    guard_cases(load_state_write_reason())
    auditor_cases()
    print()
    fails = [n for n, ok in results if not ok]
    print(f"{len(results) - len(fails)}/{len(results)} passed"
          + (f" — FAILURES: {fails}" if fails else " — ALL GREEN"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
