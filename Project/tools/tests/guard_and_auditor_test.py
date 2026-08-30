#!/usr/bin/env python3
"""Regression tests for the staged guard and bound auditor parser."""
import importlib.util
import re
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
        ("mv Project/audits/packets /tmp/stash", True),      # dir node itself
        ("mv Project/audits/strategy /tmp/s2", True),
        ("rm -r Project/audits/auto", True),
        ("echo x > Project/audits/audit_backlog.txt", True),
        ("cat Project/audits/audit_backlog.txt", False),
        ("codex exec r > Project/audits/audit_backlog.txt", True),
    ]
    for cmd, want_deny in cases:
        got = fn(cmd)
        check(f"guard: {cmd[:58]}", (got is not None) == want_deny, str(got))


def load_protected_node_reason():
    """Block A3, extracted from the owner patch — the paste bytes themselves."""
    text = PATCH.read_text()
    m = re.search(r"(PROTECTED_NODES = .*?\n    return None\n)", text, re.DOTALL)
    assert m, "protected_node_reason block not found in the owner patch"
    ns = {"re": re}
    exec(m.group(1), ns)
    return ns["protected_node_reason"]


def node_cases(fn):
    P = "Project"  # kept out of literals so the repo's own bash guard
    C = ".claude"  # doesn't bounce edits to this file
    cases = [
        # tree roots as whole operands -> DENY (the round-6 hole)
        (f"mv {C} /tmp/x", True),
        (f"mv {P}/harness /tmp/x", True),
        (f"mv {P}/results /tmp/x", True),
        (f"mv {P}/loop /tmp/x", True),
        (f"mv {P}/tools /tmp/x", True),
        (f"mv {P}/audits /tmp/x", True),
        (f"mv {P} /tmp/x", True),
        (f"mv {P}/harness/ /tmp/x", True),
        (f"cp -r /tmp/fake {P}/harness", True),
        (f"rsync -a /tmp/fake/ {P}/tools", True),
        (f"tar -xf evil.tar -C {P}/harness", True),
        (f"rm -rf {P}/audits/packets", True),
        (f"ln -s /tmp/fake {P}/results", True),
        (f'mv "{P}/harness" /tmp/x', True),           # quotes stripped first
        (f"echo ok; mv {P}/audits /tmp/x", True),     # later segment
        # ordinary work inside the trees -> ALLOW (A2/WRITE_PATTERNS own these)
        (f"mv {P}/tools/smokes/x.py /tmp/", False),
        (f"cp /tmp/a.py {P}/scratchpad/b.py", False),
        (f"mv {P}/drafts/a.md {P}/drafts/b.md", False),
        (f"ls {P}/tools", False),                     # not a write op
        (f"cat {P}/harness/runner.py", False),
        (f"grep -rn foo {P}/audits", False),
        ("rm /tmp/claude-1000/scratch/x", False),
    ]
    for cmd, want_deny in cases:
        got = fn(cmd)
        check(f"node: {cmd[:58]}", (got is not None) == want_deny, str(got))


def auditor_cases():
    spec = importlib.util.spec_from_file_location(
        "ac", REPO / "Project" / "tools" / "audit_champion.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    nonce, packet, candidate = "d" * 64, "c" * 64, "b" * 64
    entry = "20260830-120000-abcdef"
    document = {
        "schema_version": 2,
        "attempt_nonce": nonce,
        "entry_id": entry,
        "packet_sha256": packet,
        "candidate_sha256": candidate,
        "integrity": {
            "verdict": "PASS", "findings": [], "retest_request": "",
            "summary": "integrity reviewed",
        },
        "technical_review": {
            "verdict": "PASS", "findings": [],
            "summary": "diagnosis reviewed",
        },
        "summary": "complete review",
    }

    def accepted(raw, returncode=0):
        try:
            ac.validate_verdict_document(
                raw, attempt_nonce=nonce, entry_id=entry,
                packet_sha256=packet, candidate_sha256=candidate,
                returncode=returncode)
            return True
        except Exception:
            return False

    ok = __import__("json").dumps(document)
    check("parser: exact full bound object accepted", accepted(ok))
    check("parser: nonzero exit refused", not accepted(ok, 1))
    check("parser: bare PASS object refused",
          not accepted('{"verdict":"PASS"}'))
    check("parser: prose token refused",
          not accepted('thinking about the "verdict": "PASS" token'))
    check("parser: decoy plus real object refused",
          not accepted('{"verdict":"PASS"}\n' + ok))
    check("parser: duplicate objects refused", not accepted(ok + "\n" + ok))
    check("parser: banners around object refused",
          not accepted("banner\n" + ok + "\ntrailer"))
    wrong_nonce = dict(document)
    wrong_nonce["attempt_nonce"] = "0" * 64
    check("parser: wrong attempt nonce refused",
          not accepted(__import__("json").dumps(wrong_nonce)))
    check("parser: empty stdout refused", not accepted(""))


def main() -> int:
    guard_cases(load_state_write_reason())
    node_cases(load_protected_node_reason())
    auditor_cases()
    print()
    fails = [n for n, ok in results if not ok]
    print(f"{len(results) - len(fails)}/{len(results)} passed"
          + (f" — FAILURES: {fails}" if fails else " — ALL GREEN"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
