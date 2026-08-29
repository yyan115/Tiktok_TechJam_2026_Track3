#!/usr/bin/env python3
"""Two-step run gate (owner-mandated 29 Aug): RESEARCH -> PLAN -> one run.

The blocking side lives in .claude/hooks/guard_bash.py (owner-applied patch;
locked against the agent). This tool is the only way to open the gate, and it
validates artifacts, not promises:

  research  requires the CURRENT research INDEX hash (proves it was read
            this cycle, not from memory), >=2 existing research-base files
            cited, and a >=200-char summary of what was learned. Appends an
            auditable entry to Project/loop/gate_log.jsonl.
  plan      only accepted AFTER research in the same cycle; requires a
            hypothesis, a prediction CONTAINING A NUMBER, and kill criteria.
  post      called mechanically from the PostToolUse watcher after every
            referee run: closes the gate again (research+plan reset), counts
            the try against the impl's direction, reads the run's OWN journal
            entry for the speedup, resets the count on improvement, and
            STOPS the direction after 3 tries without improvement.
  status    prints the gate state.
  unlock    OWNER-ONLY by trust-model convention: reopens a stopped
            direction or resets the cycle.

State: Project/loop/gate_state.json · Log: Project/loop/gate_log.jsonl
Trust model unchanged (PLAN.md): guards against drift and momentum, not a
deliberately malicious agent editing state files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOOP = ROOT / "Project" / "loop"
STATE = LOOP / "gate_state.json"
LOG = LOOP / "gate_log.jsonl"
INDEX = ROOT / "Project" / "research" / "INDEX.md"
MAX_TRIES = 3


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"research_done": False, "plan_done": False,
                "cycle": 0, "families": {}}


def save_state(st: dict) -> None:
    LOOP.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, sort_keys=True))


def log(entry: dict) -> None:
    LOOP.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def index_hash() -> str:
    return hashlib.sha256(INDEX.read_bytes()).hexdigest()[:16]


def cmd_research(args) -> int:
    st = load_state()
    if args.index_hash != index_hash():
        print(f"REFUSED: --index-hash does not match the CURRENT research index "
              f"(expected {index_hash()}). Read Project/research/INDEX.md first "
              f"— this cycle, not from memory.")
        return 1
    notes = [n.strip() for n in args.notes.split(",") if n.strip()]
    missing = [n for n in notes if not (ROOT / "Project" / "research" / n).exists()]
    if len(notes) < 2 or missing:
        print(f"REFUSED: cite >=2 EXISTING research-base files (missing: {missing}).")
        return 1
    if len(args.summary.strip()) < 200:
        print("REFUSED: summary under 200 chars — write down what was actually "
              "learned and how it bears on the next run.")
        return 1
    st["research_done"] = True
    st["plan_done"] = False
    st["cycle"] = st.get("cycle", 0) + 1
    save_state(st)
    log({"ts": now(), "step": "research", "cycle": st["cycle"],
         "index_hash": args.index_hash, "notes": notes, "summary": args.summary})
    print(f"RESEARCH step accepted (cycle {st['cycle']}). Next: run_gate.py plan ...")
    return 0


def cmd_plan(args) -> int:
    st = load_state()
    if not st.get("research_done"):
        print("REFUSED: the RESEARCH step has not been completed this cycle. "
              "Two steps, in order.")
        return 1
    if st.get("plan_done"):
        print("NOTE: plan already recorded this cycle.")
        return 0
    if not re.search(r"\d", args.prediction):
        print("REFUSED: the prediction must contain a NUMBER (a preregistered "
              "quantitative range).")
        return 1
    if len(args.hypothesis.strip()) < 50 or len(args.kill.strip()) < 20:
        print("REFUSED: hypothesis (>=50 chars) and kill criteria (>=20 chars) required.")
        return 1
    st["plan_done"] = True
    save_state(st)
    log({"ts": now(), "step": "plan", "cycle": st.get("cycle"),
         "hypothesis": args.hypothesis, "prediction": args.prediction,
         "kill": args.kill})
    print("PLAN step accepted. The gate is OPEN for exactly one referee run.")
    return 0


def _last_journal_speedup(journal_path: Path):
    try:
        lines = [l for l in journal_path.read_text().splitlines() if l.strip()]
        e = json.loads(lines[-1])
        t = e.get("timing") or {}
        return t.get("speedup"), e.get("impl", {}).get("name")
    except Exception:
        return None, None


def cmd_post(args) -> int:
    """Called from the PostToolUse watcher with the observed Bash command."""
    cmd = args.command or ""
    m = re.search(r"runner\.py\s+(?:\S+\s+)*run\b[^|;&]*?--impl\s+(\S+)", cmd)
    if not m:
        return 0
    st = load_state()
    # The try consumed the open gate — slam it shut for the next run.
    st["research_done"] = False
    st["plan_done"] = False
    base = m.group(1).strip("'\"").split("/")[-1]
    fam = st.setdefault("families", {}).setdefault(
        base, {"tries_without_improvement": 0, "best_speedup": None, "stopped": False})
    lm = re.search(r"--ledger\s+(\S+)", cmd)
    journal = Path(lm.group(1).strip("'\"")) if lm else ROOT / "Project/results/JOURNAL.jsonl"
    speedup, _name = _last_journal_speedup(journal)
    improved = (speedup is not None and
                (fam["best_speedup"] is None or speedup > fam["best_speedup"]))
    if improved:
        fam["best_speedup"] = speedup
        fam["tries_without_improvement"] = 0
    else:
        fam["tries_without_improvement"] += 1
        if fam["tries_without_improvement"] >= MAX_TRIES:
            fam["stopped"] = True
    save_state(st)
    log({"ts": now(), "step": "post_run", "impl": base, "speedup": speedup,
         "improved": improved,
         "tries_without_improvement": fam["tries_without_improvement"],
         "stopped": fam["stopped"]})
    if fam["stopped"]:
        print(f"[run-gate] DIRECTION STOPPED: {base} — {MAX_TRIES} tries without "
              "improvement. Owner unlock required.")
    return 0


def cmd_status(_args) -> int:
    st = load_state()
    st["_current_index_hash"] = index_hash()
    print(json.dumps(st, indent=1, sort_keys=True))
    return 0


def cmd_unlock(args) -> int:
    st = load_state()
    if args.family:
        fam = st.get("families", {}).get(args.family)
        if fam:
            fam["stopped"] = False
            fam["tries_without_improvement"] = 0
    else:
        st["research_done"] = False
        st["plan_done"] = False
    save_state(st)
    log({"ts": now(), "step": "owner_unlock", "family": args.family})
    print("Unlocked. (Owner-only action by trust-model convention.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-step run gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("research")
    r.add_argument("--index-hash", required=True,
                   help="first 16 hex chars of sha256(Project/research/INDEX.md)")
    r.add_argument("--notes", required=True,
                   help="comma-separated research-base filenames actually read")
    r.add_argument("--summary", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--prediction", required=True)
    p.add_argument("--kill", required=True)
    po = sub.add_parser("post")
    po.add_argument("--command", required=True)
    sub.add_parser("status")
    u = sub.add_parser("unlock")
    u.add_argument("--family", default=None)
    args = ap.parse_args()
    return {"research": cmd_research, "plan": cmd_plan, "post": cmd_post,
            "status": cmd_status, "unlock": cmd_unlock}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
