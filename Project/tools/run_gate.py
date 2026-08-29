#!/usr/bin/env python3
"""Run gate v3 — permit architecture (rebuilt after external REJECT, 29 Aug).

Every referee invocation consumes a ONE-USE PERMIT that was issued by a
validated thinking step and is bound to the exact attempt (direction card,
mode, shape, impl file + sha, canonical ledger, expiry). The owner-side
guard consumes the permit atomically BEFORE execution (ARMED -> IN_FLIGHT);
the post watcher reconciles the outcome from the bound ledger's new row
(exactly one, sha-matched) or records an execution failure. Everything
fails CLOSED.

Thinking tiers (anti-filler, per review):
  research + plan   full two steps — required to OPEN a new direction card
                    (research: current INDEX hash, >=2 existing notes,
                    summary; plan: hypothesis, numeric prediction, kill
                    criteria, file:line citations quoted into the log).
  delta             concise structured step for the NEXT attempt within an
                    open direction: what changed since the last attempt +
                    a numeric prediction. No filler packets for tuning
                    iterations; budget still enforced.
Both tiers issue exactly one permit.

Strikes (per direction card, per shape+profile comparison group):
  optimization mode: improvement = clean, correct, promoted-comparable row
    whose speedup exceeds the group's best by >=3% (the promotion margin
    floor — epsilon noise never resets strikes).
  screening mode: the attempt's declared prediction range decides
    hit/miss; a miss is a strike.
  confirmation / correctness / calibration modes: never strike, never
    "improve" — they exist so those runs are gated without being punished.
  execution failure (no new row / crash / sha mismatch): separate counter;
    3 consecutive execution failures also close the direction.
  3 strikes without improvement => direction CLOSED, postmortem debt
  registered. Closed directions reopen ONLY via a one-use critic receipt
  bound to the closure nonce. There is no unlock command.

State: Project/loop/gate_state.json · Permits: Project/loop/permit.json
(consumed -> Project/loop/permits_used/) · Log: Project/loop/gate_log.jsonl
Trust model (PLAN.md): mistakes-not-malice; the guard side lives in the
owner-locked hook, and the citation/receipt trails are auditor-verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOOP = ROOT / "Project" / "loop"
STATE = LOOP / "gate_state.json"
PERMIT = LOOP / "permit.json"
USED = LOOP / "permits_used"
INFLIGHT = LOOP / "in_flight.json"
LOG = LOOP / "gate_log.jsonl"
INDEX = ROOT / "Project" / "research" / "INDEX.md"
CARDS = LOOP / "cards.jsonl"
DEFAULT_JOURNAL = ROOT / "Project" / "results" / "JOURNAL.jsonl"
MAX_STRIKES = 3
MAX_EXEC_FAILURES = 3
PERMIT_TTL_S = 45 * 60
IMPROVE_MARGIN = 1.03
MODES = ("optimization", "screening", "confirmation", "correctness", "calibration")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def save_state(st: dict) -> None:
    LOOP.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
    tmp.replace(STATE)


def log(entry: dict) -> None:
    LOOP.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def index_hash() -> str:
    return hashlib.sha256(INDEX.read_bytes()).hexdigest()[:16]


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def open_cards() -> dict:
    """direction_family_id -> card (latest row per family), non-closed only."""
    fams = {}
    for line in (CARDS.read_text().splitlines() if CARDS.exists() else []):
        if not line.strip():
            continue
        try:
            c = json.loads(line)
        except Exception:
            continue
        fams[c.get("direction_family_id")] = c
    return {k: v for k, v in fams.items()
            if "killed" not in str(v.get("status", "")).lower()
            and "closed" not in str(v.get("status", "")).lower()}


def parse_citations(spec: str):
    out = []
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        m = re.fullmatch(r"(.+?):(\d+)(?:-(\d+))?", item)
        if not m:
            return None, f"bad citation format: '{item}'"
        rel, a, b = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        path = ROOT / rel
        if not path.exists():
            path = ROOT / "Project" / rel
        if not path.exists():
            return None, f"cited file does not exist: '{rel}'"
        lines = path.read_text().splitlines()
        if not (1 <= a <= b <= len(lines)):
            return None, f"cited lines {a}-{b} out of range for {rel}"
        out.append({"file": rel, "lines": f"{a}-{b}",
                    "quoted": "\n".join(lines[a - 1:b])[:2000]})
    return out, None


def issue_permit(st, direction, mode, shape, impl, ledger, prediction, plan_ref):
    if PERMIT.exists():
        return None, "a permit is already ARMED — one attempt at a time"
    if INFLIGHT.exists():
        return None, "an attempt is IN_FLIGHT and unreconciled — run `reconcile` first"
    impl_p = (ROOT / impl).resolve() if impl else None
    if mode != "calibration":
        if impl_p is None or not impl_p.exists():
            return None, f"impl file not found: {impl}"
    ledger_p = Path(ledger).resolve() if ledger else DEFAULT_JOURNAL
    pre_lines = len(ledger_p.read_text().splitlines()) if ledger_p.exists() else 0
    permit = {
        "permit_id": secrets.token_hex(8),
        "direction_id": direction,
        "mode": mode,
        "shape": int(shape),
        "impl_path": str(impl_p.relative_to(ROOT)) if impl_p else None,
        "impl_sha256": sha_file(impl_p) if impl_p else None,
        "ledger": str(ledger_p),
        "ledger_pre_lines": pre_lines,
        "prediction": prediction,
        "plan_ref": plan_ref,
        "issued": now(),
        "expires_epoch": time.time() + PERMIT_TTL_S,
    }
    LOOP.mkdir(parents=True, exist_ok=True)
    PERMIT.write_text(json.dumps(permit, indent=1, sort_keys=True))
    return permit, None


def cmd_research(args) -> int:
    st = load_json(STATE, {})
    pending = st.get("pending_postmortem", [])
    if pending and len((args.postmortem or "").strip()) < 200:
        print(f"REFUSED: direction(s) {pending} were CLOSED. A >=200-char "
              "--postmortem (predicted vs happened, why it failed, what it "
              "rules out) is mandatory before any new research cycle.")
        return 1
    if args.index_hash != index_hash():
        print(f"REFUSED: --index-hash mismatch (current: {index_hash()}). "
              "Read Project/research/INDEX.md THIS cycle.")
        return 1
    notes = [n.strip() for n in args.notes.split(",") if n.strip()]
    missing = [n for n in notes if not (ROOT / "Project" / "research" / n).exists()]
    if len(notes) < 2 or missing:
        print(f"REFUSED: >=2 existing research-base files (missing: {missing}).")
        return 1
    if len(args.summary.strip()) < 200:
        print("REFUSED: summary under 200 chars.")
        return 1
    st["research_cycle"] = st.get("research_cycle", 0) + 1
    st["research_open"] = True
    if pending:
        st["pending_postmortem"] = []
    save_state(st)
    entry = {"ts": now(), "step": "research", "cycle": st["research_cycle"],
             "index_hash": args.index_hash, "notes": notes, "summary": args.summary}
    if pending:
        entry["postmortem_for"] = pending
        entry["postmortem"] = args.postmortem
    log(entry)
    print(f"RESEARCH accepted (cycle {st['research_cycle']}). Next: plan (new "
          "direction card required).")
    return 0


def cmd_plan(args) -> int:
    st = load_json(STATE, {})
    if not st.get("research_open"):
        print("REFUSED: research step required first (this cycle). Two steps, in order.")
        return 1
    cards = open_cards()
    card = cards.get(args.direction)
    if card is None:
        print(f"REFUSED: --direction must name an OPEN card family in "
              f"{CARDS} (found: {sorted(cards)}). Open the card first — the "
              "card IS the direction's identity.")
        return 1
    if args.mode not in MODES:
        print(f"REFUSED: --mode must be one of {MODES}.")
        return 1
    if not re.search(r"\d", args.prediction):
        print("REFUSED: numeric prediction required.")
        return 1
    if len(args.hypothesis.strip()) < 50 or len(args.kill.strip()) < 20:
        print("REFUSED: hypothesis >=50 chars and kill criteria >=20 chars.")
        return 1
    citations, err = parse_citations(args.sources or "")
    if err or not citations:
        print(f"REFUSED: valid --sources citations required ({err}).")
        return 1
    if len(args.reasoning.strip()) < 100:
        print("REFUSED: --reasoning >=100 chars.")
        return 1
    plan_id = secrets.token_hex(6)
    permit, perr = issue_permit(st, args.direction, args.mode, args.shape,
                                args.impl, args.ledger, args.prediction, plan_id)
    if perr:
        print(f"REFUSED: {perr}")
        return 1
    st["research_open"] = False
    save_state(st)
    log({"ts": now(), "step": "plan", "plan_id": plan_id,
         "direction": args.direction, "mode": args.mode, "shape": args.shape,
         "hypothesis": args.hypothesis, "prediction": args.prediction,
         "kill": args.kill, "citations": citations,
         "reasoning": args.reasoning, "permit_id": permit["permit_id"]})
    print(f"PLAN accepted. Permit {permit['permit_id']} ARMED for ONE run: "
          f"direction={args.direction} mode={args.mode} shape={args.shape}.")
    return 0


def cmd_delta(args) -> int:
    """Concise continuation within an open direction — no research packet,
    but still: what changed + numeric prediction + one permit."""
    cards = open_cards()
    card = cards.get(args.direction)
    if card is None:
        print(f"REFUSED: no open card for '{args.direction}'.")
        return 1
    st = load_json(STATE, {})
    grp = st.get("groups", {}).get(f"{args.direction}|{args.shape}", {})
    if grp.get("closed"):
        print("REFUSED: this direction+shape group is CLOSED.")
        return 1
    if len(args.changed.strip()) < 40 or not re.search(r"\d", args.prediction):
        print("REFUSED: --changed (>=40 chars, the exact delta from the last "
              "attempt) and a numeric --prediction are required.")
        return 1
    plan_id = secrets.token_hex(6)
    permit, perr = issue_permit(st, args.direction, args.mode, args.shape,
                                args.impl, args.ledger, args.prediction, plan_id)
    if perr:
        print(f"REFUSED: {perr}")
        return 1
    save_state(st)
    log({"ts": now(), "step": "delta", "plan_id": plan_id,
         "direction": args.direction, "mode": args.mode, "shape": args.shape,
         "changed": args.changed, "prediction": args.prediction,
         "permit_id": permit["permit_id"]})
    print(f"DELTA accepted. Permit {permit['permit_id']} ARMED for ONE run.")
    return 0


def cmd_reconcile(args) -> int:
    """Called by the PostToolUse watcher (and idempotently by anyone):
    resolves an IN_FLIGHT attempt from its bound ledger."""
    fl = load_json(INFLIGHT, None)
    if fl is None:
        return 0
    st = load_json(STATE, {})
    ledger = Path(fl["ledger"])
    rows = [l for l in ledger.read_text().splitlines() if l.strip()] if ledger.exists() else []
    new_rows = rows[fl["ledger_pre_lines"]:]
    gkey = f"{fl['direction_id']}|{fl['shape']}"
    grp = st.setdefault("groups", {}).setdefault(
        gkey, {"best_speedup": None, "strikes": 0, "exec_failures": 0,
               "closed": False, "closure_nonce": None})
    outcome = {"ts": now(), "step": "reconcile", "permit_id": fl["permit_id"],
               "direction": fl["direction_id"], "shape": fl["shape"],
               "mode": fl["mode"]}
    entry = None
    if len(new_rows) == 1:
        try:
            e = json.loads(new_rows[0])
            if fl["impl_sha256"] is None or \
               e.get("impl", {}).get("sha256") == fl["impl_sha256"]:
                entry = e
        except Exception:
            entry = None
    if entry is None:
        grp["exec_failures"] += 1
        outcome["result"] = "execution_failure"
        outcome["new_rows"] = len(new_rows)
        if grp["exec_failures"] >= MAX_EXEC_FAILURES:
            grp["closed"] = True
            grp["closure_nonce"] = secrets.token_hex(8)
            st.setdefault("pending_postmortem", []).append(gkey)
            outcome["closed"] = True
            outcome["closure_nonce"] = grp["closure_nonce"]
    else:
        grp["exec_failures"] = 0
        t = entry.get("timing") or {}
        speed = t.get("speedup")
        correct = bool((entry.get("correctness") or {}).get("passed"))
        clean = not ((t.get("wall_check") or {}).get("suspicious") or
                     (t.get("anti_cache_check") or {}).get("suspicious"))
        outcome.update({"speedup": speed, "correct": correct, "clean": clean,
                        "entry_id": entry.get("entry_id")})
        if fl["mode"] == "optimization":
            improved = (correct and clean and speed is not None and
                        (grp["best_speedup"] is None or
                         speed > grp["best_speedup"] * IMPROVE_MARGIN))
            if improved:
                grp["best_speedup"] = speed
                grp["strikes"] = 0
            else:
                grp["strikes"] += 1
            outcome["improved"] = improved
        elif fl["mode"] == "screening":
            outcome["declared_prediction"] = fl.get("prediction")
            outcome["needs_range_judgment"] = True
            grp["strikes"] += 0 if correct else 1
        # confirmation/correctness/calibration: recorded, never strike.
        if grp["strikes"] >= MAX_STRIKES:
            grp["closed"] = True
            grp["closure_nonce"] = secrets.token_hex(8)
            st.setdefault("pending_postmortem", []).append(gkey)
            outcome["closed"] = True
            outcome["closure_nonce"] = grp["closure_nonce"]
    save_state(st)
    log(outcome)
    USED.mkdir(exist_ok=True)
    Path(INFLIGHT).replace(USED / f"{fl['permit_id']}.reconciled.json")
    if outcome.get("closed"):
        print(f"[run-gate] GROUP CLOSED: {gkey} (nonce {grp['closure_nonce']}). "
              "Postmortem now mandatory; reopening needs a critic receipt "
              "bound to this nonce.")
    return 0


def cmd_screen_judge(args) -> int:
    """Record the screening range hit/miss explicitly (auditable)."""
    st = load_json(STATE, {})
    gkey = f"{args.direction}|{args.shape}"
    grp = st.setdefault("groups", {}).setdefault(
        gkey, {"best_speedup": None, "strikes": 0, "exec_failures": 0,
               "closed": False, "closure_nonce": None})
    if args.result == "miss":
        grp["strikes"] += 1
        if grp["strikes"] >= MAX_STRIKES:
            grp["closed"] = True
            grp["closure_nonce"] = secrets.token_hex(8)
            st.setdefault("pending_postmortem", []).append(gkey)
    save_state(st)
    log({"ts": now(), "step": "screen_judge", "group": gkey,
         "result": args.result, "observed": args.observed,
         "strikes": grp["strikes"], "closed": grp["closed"]})
    print(f"screening {args.result} recorded for {gkey} "
          f"(strikes {grp['strikes']}/{MAX_STRIKES}).")
    return 0


def cmd_reopen(args) -> int:
    st = load_json(STATE, {})
    grp = st.get("groups", {}).get(args.group)
    if not grp or not grp.get("closed"):
        print("Nothing closed under that group key.")
        return 1
    nonce = grp.get("closure_nonce")
    critic = Path(args.critic_log)
    if not (nonce and critic.exists()):
        print("REFUSED: closure nonce or critic log missing.")
        return 1
    text = critic.read_text(errors="ignore")
    if nonce not in text or not re.search(r"CRITIC:\s*(continue|narrow)", text):
        print("REFUSED: the critic log must contain the EXACT closure nonce "
              f"({nonce}) and end CRITIC: continue|narrow. Receipts are "
              "one-use and closure-bound.")
        return 1
    grp["closed"] = False
    grp["strikes"] = 0
    grp["exec_failures"] = 0
    grp["closure_nonce"] = None
    save_state(st)
    log({"ts": now(), "step": "reopen", "group": args.group,
         "critic_log": str(critic), "consumed_nonce": nonce})
    print(f"Reopened {args.group} on critic authority (nonce consumed).")
    return 0


def cmd_status(_args) -> int:
    st = load_json(STATE, {})
    st["_index_hash_now"] = index_hash()
    st["_permit_armed"] = PERMIT.exists()
    st["_in_flight"] = INFLIGHT.exists()
    print(json.dumps(st, indent=1, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run gate v3 (permit architecture)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("research")
    r.add_argument("--index-hash", required=True)
    r.add_argument("--notes", required=True)
    r.add_argument("--summary", required=True)
    r.add_argument("--postmortem", default=None)
    p = sub.add_parser("plan")
    for a, req in (("--direction", True), ("--mode", True), ("--shape", True),
                   ("--impl", False), ("--ledger", False),
                   ("--hypothesis", True), ("--prediction", True),
                   ("--kill", True), ("--sources", True), ("--reasoning", True)):
        p.add_argument(a, required=req, default=None)
    d = sub.add_parser("delta")
    for a, req in (("--direction", True), ("--mode", True), ("--shape", True),
                   ("--impl", False), ("--ledger", False),
                   ("--changed", True), ("--prediction", True)):
        d.add_argument(a, required=req, default=None)
    sub.add_parser("reconcile")
    sj = sub.add_parser("screen-judge")
    sj.add_argument("--direction", required=True)
    sj.add_argument("--shape", required=True)
    sj.add_argument("--result", required=True, choices=("hit", "miss"))
    sj.add_argument("--observed", required=True)
    ro = sub.add_parser("reopen")
    ro.add_argument("--group", required=True)
    ro.add_argument("--critic-log", required=True)
    sub.add_parser("status")
    args = ap.parse_args()
    return {"research": cmd_research, "plan": cmd_plan, "delta": cmd_delta,
            "reconcile": cmd_reconcile, "screen-judge": cmd_screen_judge,
            "reopen": cmd_reopen, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
