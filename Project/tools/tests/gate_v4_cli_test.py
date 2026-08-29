#!/usr/bin/env python3
"""Authority v4 CLI test suite (committed per the 30 Aug external review).

Exercises the DEPLOYED run_gate.py command line via subprocess in a
throwaway sandbox tree — no monkeypatching, no internal calls — so parser
regressions and CLI wiring breakages fail loudly (the class of bug the
earlier internals-only suite missed).

Scope note (honest): the reconcile inner leg (real ledger-row semantics) is
deliberately NOT simulated with synthetic rows here; it is covered by the
live post-paste proof-tests against a real runner row, which is stronger
evidence than fabricated journal JSON.
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GATE_SRC = REPO / "Project" / "tools" / "run_gate.py"

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def build_sandbox() -> Path:
    sb = Path(tempfile.mkdtemp(prefix="gate_v4_cli_"))
    (sb / "Project" / "tools").mkdir(parents=True)
    (sb / "Project" / "loop").mkdir(parents=True)
    (sb / "Project" / "research").mkdir(parents=True)
    (sb / "Project" / "results").mkdir(parents=True)
    (sb / "Project" / "audits").mkdir(parents=True)
    shutil.copyfile(GATE_SRC, sb / "Project" / "tools" / "run_gate.py")
    (sb / "Project" / "research" / "INDEX.md").write_text("test index\n")
    (sb / "Project" / "research" / "note_a.md").write_text("note a\n")
    (sb / "Project" / "research" / "note_b.md").write_text("note b\n")
    (sb / "Project" / "results" / "JOURNAL.jsonl").write_text("")
    (sb / "Project" / "audits" / "verdicts.jsonl").write_text("")
    (sb / "Project" / "loop" / "cards.jsonl").write_text(
        json.dumps({"direction_family_id": "TESTDIR", "status": "open"}) + "\n")
    (sb / "cand.py").write_text("x = 1\n")
    (sb / "cite.md").write_text("line one\nline two\nline three\n")
    return sb


def main() -> int:
    sb = build_sandbox()
    gate = sb / "Project" / "tools" / "run_gate.py"
    journal = sb / "Project" / "results" / "JOURNAL.jsonl"
    verdicts = sb / "Project" / "audits" / "verdicts.jsonl"
    loop = sb / "Project" / "loop"
    cand_sha = hashlib.sha256((sb / "cand.py").read_bytes()).hexdigest()

    def run(*args):
        r = subprocess.run([sys.executable, str(gate), *args],
                           capture_output=True, text=True, cwd=str(sb))
        return r.returncode, r.stdout + r.stderr

    def research():
        rc, out = run("research", "--index-hash", "WRONG",
                      "--notes", "note_a.md,note_b.md", "--summary", "s" * 210)
        m = re.search(r"current: ([0-9a-f]{16})", out)
        assert m, f"could not parse index hash from: {out}"
        return run("research", "--index-hash", m.group(1),
                   "--notes", "note_a.md,note_b.md", "--summary", "s" * 210)

    PLAN_STD = ["--direction", "TESTDIR", "--hypothesis", "h" * 60,
                "--kill", "k" * 30, "--sources", "cite.md:1-2",
                "--reasoning", "r" * 120, "--prediction", "1.2x"]

    rc, out = run("init")
    check("init succeeds", rc == 0, out)

    rc, out = run("plan", "--mode", "optimization", "--shape", "3",
                  "--impl", "cand.py", *PLAN_STD)
    check("plan before research refused", rc == 1 and "research step required" in out, out)

    rc, out = research()
    check("research accepted", rc == 0, out)

    rc, out = run("plan", "--mode", "screening", "--shape", "3",
                  "--impl", "cand.py", *PLAN_STD)
    check("screening plan without bounds refused (CLI reaches issue_permit)",
          rc == 1 and "predict-min" in out, out)

    rc, out = run("plan", "--mode", "screening", "--shape", "3",
                  "--impl", "cand.py", "--predict-min", "1.1",
                  "--predict-max", "inf", *PLAN_STD)
    check("infinite prediction bound refused", rc == 1 and "FINITE" in out, out)

    rc, out = run("plan", "--mode", "optimization", "--shape", "3",
                  "--impl", "cand.py", *PLAN_STD)
    check("optimization plan issues a permit via the CLI (Finding-1 catcher)",
          rc == 0 and (loop / "permit.json").exists(), out)

    rc, out = run("research", "--index-hash", "x", "--notes", "a,b",
                  "--summary", "s" * 210)
    check("research refused while a permit is armed",
          rc == 1 and "armed or in flight" in out, out)

    # simulate the guard's consume step: permit -> in_flight
    permit = json.loads((loop / "permit.json").read_text())
    permit["ledger_pre_lines"] = 0
    permit["consumed"] = "2026-08-30T06:00:00+0800"
    (loop / "in_flight.json").write_text(json.dumps(permit))
    (loop / "permit.json").unlink()
    rc, out = run("research", "--index-hash", "x", "--notes", "a,b",
                  "--summary", "s" * 210)
    check("research refused while an attempt is in flight",
          rc == 1 and "armed or in flight" in out, out)
    (loop / "in_flight.json").unlink()

    # ---- verdict brake via the CLI ----
    RVKEY = ("champ-entry-1", "2099-01-01T00:00:00+0800")
    verdicts.write_text(json.dumps({"entry_id": RVKEY[0], "recorded": RVKEY[1],
                                    "verdict": "RULE_VIOLATION"}) + "\n")
    rc, out = research()
    check("research still allowed under a violation", rc == 0, out)
    rc, out = run("plan", "--mode", "optimization", "--shape", "4",
                  "--impl", "cand.py", *PLAN_STD)
    check("violation freezes permits via the CLI",
          rc == 1 and "OWNER-ONLY" in out, out)

    rc, out = run("verdict-clear", "--kind", "violation", "--entry-id", RVKEY[0],
                  "--recorded", RVKEY[1], "--owner-quote", "short")
    check("short owner quote refused via CLI", rc == 1 and "owner" in out.lower(), out)

    QUOTE = "cleared: I read the audit, finding is stale vs lesson 17, proceed"
    rc, out = run("verdict-clear", "--kind", "violation", "--entry-id", RVKEY[0],
                  "--recorded", RVKEY[1], "--owner-quote", QUOTE)
    check("owner-quote clear succeeds via CLI", rc == 0, out)

    verdicts.write_text(verdicts.read_text() + json.dumps(
        {"entry_id": "champ-entry-2", "recorded": "2099-01-02T00:00:00+0800",
         "verdict": "RULE_VIOLATION"}) + "\n")
    rc, out = run("verdict-clear", "--kind", "violation", "--entry-id",
                  "champ-entry-2", "--recorded", "2099-01-02T00:00:00+0800",
                  "--owner-quote", QUOTE)
    check("owner quote is one-use", rc == 1 and "one-use" in out, out)
    QUOTE2 = "cleared: second ruling — checked entry two myself, it is fine, go"
    rc, out = run("verdict-clear", "--kind", "violation", "--entry-id",
                  "champ-entry-2", "--recorded", "2099-01-02T00:00:00+0800",
                  "--owner-quote", QUOTE2)
    check("fresh owner quote clears the second violation", rc == 0, out)

    # ---- RETEST binding via the CLI ----
    RT_ENTRY, RT_REC = "20260830-010000-orig", "2099-01-03T00:00:00+0800"
    verdicts.write_text(verdicts.read_text() + json.dumps(
        {"entry_id": RT_ENTRY, "recorded": RT_REC, "verdict": "RETEST"}) + "\n")

    rc, out = research()
    check("research allowed under a retest", rc == 0, out)
    rc, out = run("plan", "--mode", "optimization", "--shape", "3",
                  "--impl", "cand.py", *PLAN_STD)
    check("retest blocks optimization via CLI", rc == 1 and "RETEST" in out, out)
    rc, out = run("plan", "--mode", "calibration", "--shape", "3", *PLAN_STD)
    check("retest blocks calibration too", rc == 1 and "RETEST" in out, out)
    rc, out = run("plan", "--mode", "confirmation", "--shape", "3",
                  "--impl", "cand.py", *PLAN_STD)
    check("confirmation refused while retested entry absent from journal",
          rc == 1 and "absent from the primary journal" in out, out)

    def jrow(eid, sha, passed):
        return {"entry_id": eid, "impl": {"sha256": sha}, "shape_id": 3,
                "shape": {"batch_size": 4, "seq_len": 128},
                "correctness": {"passed": passed}}
    OTHER_SHA = "f" * 64
    journal.write_text(json.dumps(jrow(RT_ENTRY, OTHER_SHA, True)) + "\n")
    rc, out = run("plan", "--mode", "confirmation", "--shape", "3",
                  "--impl", "cand.py", *PLAN_STD)
    check("confirmation with WRONG bytes refused under retest",
          rc == 1 and "cross-shape or cross-candidate" in out, out)

    journal.write_text(json.dumps(jrow(RT_ENTRY, cand_sha, True)) + "\n")
    rc, out = run("plan", "--mode", "confirmation", "--shape", "4",
                  "--impl", "cand.py", *PLAN_STD)
    check("confirmation on the WRONG shape refused under retest",
          rc == 1 and "cross-shape or cross-candidate" in out, out)

    rc, out = run("plan", "--mode", "confirmation", "--shape", "3",
                  "--impl", "cand.py", *PLAN_STD)
    check("confirmation with the retested (bytes, shape) is allowed",
          rc == 0 and (loop / "permit.json").exists(), out)
    permit_snapshot = json.loads((loop / "permit.json").read_text())
    check("permit snapshots the verdict-record line count",
          isinstance(permit_snapshot.get("verdict_lines"), int)
          and permit_snapshot["verdict_lines"] >= 3, str(permit_snapshot))
    (loop / "permit.json").unlink()

    rc, out = research()
    check("research reopens after the armed permit was cleared", rc == 0, out)
    rc, out = run("plan", "--mode", "confirmation", "--shape", "3",
                  "--impl", "cand.py", "--ledger", str(sb / "scratch.jsonl"),
                  *PLAN_STD)
    check("confirmation on a scratch ledger refused",
          rc == 1 and "PRIMARY" in out, out)

    # mechanical retest clear: too-early confirmation row refused
    early = jrow("20970101-000000-conf", cand_sha, True)
    journal.write_text(json.dumps(jrow(RT_ENTRY, cand_sha, True)) + "\n"
                       + json.dumps(early) + "\n")
    rc, out = run("verdict-clear", "--kind", "retest", "--entry-id", RT_ENTRY,
                  "--recorded", RT_REC, "--confirm-entry", "20970101-000000-conf")
    check("confirmation row predating the verdict refused",
          rc == 1 and "after_verdict=False" in out, out)

    good = jrow("20990104-000000-conf", cand_sha, True)
    journal.write_text(json.dumps(jrow(RT_ENTRY, cand_sha, True)) + "\n"
                       + json.dumps(good) + "\n")
    rc, out = run("verdict-clear", "--kind", "retest", "--entry-id", RT_ENTRY,
                  "--recorded", RT_REC, "--confirm-entry", "20990104-000000-conf")
    check("matching row WITHOUT a reconciled confirmation permit refused",
          rc == 1 and "reconciled confirmation" in out, out)

    # witness the confirmation in the gate's own transition log (same seq —
    # a higher seq would put state behind the log and fail closed)
    seq_now = json.loads((loop / "gate_state.json").read_text())["seq"]
    with (loop / "gate_log.jsonl").open("a") as fh:
        fh.write(json.dumps({"ts": "t", "step": "reconcile",
                             "mode": "confirmation",
                             "entry_id": "20990104-000000-conf",
                             "state_seq": seq_now}) + "\n")
    rc, out = run("verdict-clear", "--kind", "retest", "--entry-id", RT_ENTRY,
                  "--recorded", RT_REC, "--confirm-entry", "20990104-000000-conf")
    check("mechanical retest clear succeeds via CLI", rc == 0, out)

    rc, out = run("verdict-clear", "--kind", "retest", "--entry-id", RT_ENTRY,
                  "--recorded", RT_REC, "--confirm-entry", "20990104-000000-conf")
    check("cleared retest cannot be cleared twice", rc == 1, out)

    # ---- verdict-record tampering fails closed ----
    saved = verdicts.read_text()
    verdicts.unlink()
    rc, out = research()
    rc, out = run("plan", "--mode", "optimization", "--shape", "5",
                  "--impl", "cand.py", *PLAN_STD)
    check("missing verdict record freezes permits",
          rc == 1 and "OWNER-ONLY" in out, out)
    verdicts.write_text(saved + json.dumps(
        {"entry_id": "dup-e", "recorded": "2099-02-01T00:00:00+0800",
         "verdict": "PASS"}) + "\n" + json.dumps(
        {"entry_id": "dup-e", "recorded": "2099-02-01T00:00:00+0800",
         "verdict": "RULE_VIOLATION"}) + "\n")
    rc, out = run("plan", "--mode", "optimization", "--shape", "5",
                  "--impl", "cand.py", *PLAN_STD)
    check("conflicting duplicate verdict rows freeze permits",
          rc == 1 and "OWNER-ONLY" in out, out)
    verdicts.write_text(saved)

    # ---- misc CLI guards ----
    rc, out = run("screen-judge", "--direction", "TESTDIR", "--shape", "3",
                  "--observed", "1.2")
    check("screen-judge without pending judgment refused", rc == 1, out)
    rc, out = run("reopen", "--group", "TESTDIR|3",
                  "--critic-log", "Project/audits/strategy/nope.log")
    check("reopen of a non-closed group refused", rc == 1, out)

    print()
    fails = [n for n, ok in results if not ok]
    print(f"{len(results) - len(fails)}/{len(results)} passed"
          + (f" — FAILURES: {fails}" if fails else " — ALL GREEN"))
    shutil.rmtree(sb, ignore_errors=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
