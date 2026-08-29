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


_LOCK_REF = []


def load_state_strict():
    """Fail CLOSED: issuance and reconciliation refuse to run on missing or
    corrupt state — a wiped state file must not erase closures/debts. Also
    acquires the shared gate lock for the life of this process, serializing
    every state transition against the watcher; and checks the state's
    sequence number against the log so a stale git-restored file (missing
    later transitions) is refused."""
    if not _LOCK_REF:
        _LOCK_REF.append(gate_lock())
    if not STATE.exists():
        raise SystemExit("REFUSED: gate state missing. Run `run_gate.py init` "
                         "once (records the event) if this is genuinely new.")
    try:
        st = json.loads(STATE.read_text())
    except Exception:
        raise SystemExit("REFUSED: gate state unreadable/corrupt — fail "
                         "closed. Bring Project/loop/gate_state.json back "
                         "from version control before any further attempts.")
    logged_seq = 0
    if LOG.exists():
        for i, line in enumerate(LOG.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                s = json.loads(line).get("state_seq")
            except Exception:
                raise SystemExit(f"REFUSED: gate_log.jsonl line {i} is "
                                 "malformed — fail closed. Repair or remove "
                                 "that line (with a note) before proceeding.")
            if isinstance(s, int):
                logged_seq = max(logged_seq, s)
    if st.get("seq", 0) < logged_seq:
        raise SystemExit(f"REFUSED: state seq {st.get('seq', 0)} is BEHIND the "
                         f"log's {logged_seq} — a stale restore is missing "
                         "transitions. Reconstruct state to match the log "
                         "before proceeding.")
    return st


def save_state(st: dict) -> None:
    LOOP.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
    tmp.replace(STATE)


def commit(st: dict, entry: dict) -> None:
    """Crash-safe transition: the log row carrying the NEW seq is durably
    appended BEFORE state is saved. A crash in between leaves state BEHIND
    the log, which the strict loader refuses — fail closed, never laundered."""
    seq = st.get("seq", 0) + 1
    entry["state_seq"] = seq
    log(entry)
    st["seq"] = seq
    save_state(st)


def log(entry: dict) -> None:
    import os
    LOOP.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
    if USED.exists() and any(f.name.startswith("claim.") for f in USED.iterdir()):
        return None, ("a stranded reconciliation claim exists in permits_used/ "
                      "— investigate and restore it before new permits")
    if st.get("pending_screen_judgment"):
        return None, ("the previous screening attempt has no recorded hit/miss "
                      "judgment — run `screen-judge` first")
    if mode not in MODES:
        return None, f"mode must be one of {MODES}"
    shape = int(shape)  # canonical — '014' == 14 everywhere
    if st.get("pending_postmortem"):
        return None, (f"postmortem debt outstanding for "
                      f"{st['pending_postmortem']} — a research step with "
                      "--postmortem must come first")
    gkey = f"{direction}|{shape}"
    grp = st.get("groups", {}).get(gkey, {})
    if grp.get("closed"):
        return None, f"group {gkey} is CLOSED (reopen needs a critic receipt)"
    if mode == "calibration":
        if impl:
            return None, "calibration permits must not carry --impl (that's a candidate run)"
    if mode in ("confirmation", "correctness"):
        if grp.get("nonstrike_budget", 2) <= 0:
            return None, (f"non-strike budget exhausted for {gkey} — further "
                          "attempts must be optimization or screening (which "
                          "can strike)")
    impl_p = (ROOT / impl).resolve() if impl else None
    if mode != "calibration":
        if impl_p is None or not impl_p.exists():
            return None, f"impl file not found: {impl}"
        if mode == "optimization":
            if sha_file(impl_p) in grp.get("attempted_shas", []):
                return None, ("optimization permit refused: these exact "
                              "candidate bytes were already attempted in this "
                              "group — an identical re-run is 'confirmation', "
                              "label it honestly")
    ledger_p = Path(ledger).resolve() if ledger else DEFAULT_JOURNAL
    pre_lines = (len([l for l in ledger_p.read_text().splitlines() if l.strip()])
                 if ledger_p.exists() else 0)
    permit = {
        "permit_id": secrets.token_hex(8),
        "direction_id": direction,
        "mode": mode,
        "shape": shape,
        "impl_path": str(impl_p.relative_to(ROOT)) if impl_p else None,
        "impl_sha256": sha_file(impl_p) if impl_p else None,
        "ledger": str(ledger_p),
        "ledger_pre_lines": pre_lines,
        "prediction": prediction,
        "plan_ref": plan_ref,
        "issued": now(),
        "expires_epoch": time.time() + PERMIT_TTL_S,
    }
    return permit, None


def arm_permit(permit) -> None:
    """Write the permit file LAST — after the thinking transition committed.
    A crash in between leaves a committed plan and no permit (safe: redo the
    delta), never a usable permit without its committed thinking step."""
    LOOP.mkdir(parents=True, exist_ok=True)
    PERMIT.write_text(json.dumps(permit, indent=1, sort_keys=True))


def cmd_research(args) -> int:
    st = load_state_strict()
    if PERMIT.exists() or INFLIGHT.exists():
        print("REFUSED: an attempt is armed or in flight — finish and "
              "reconcile it before starting a new research cycle.")
        return 1
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
    entry = {"ts": now(), "step": "research", "cycle": st["research_cycle"],
             "index_hash": args.index_hash, "notes": notes, "summary": args.summary}
    if pending:
        entry["postmortem_for"] = pending
        entry["postmortem"] = args.postmortem
    commit(st, entry)
    print(f"RESEARCH accepted (cycle {st['research_cycle']}). Next: plan (new "
          "direction card required).")
    return 0


def cmd_plan(args) -> int:
    st = load_state_strict()
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
    commit(st, {"ts": now(), "step": "plan", "plan_id": plan_id,
         "direction": args.direction, "mode": args.mode, "shape": args.shape,
         "hypothesis": args.hypothesis, "prediction": args.prediction,
         "kill": args.kill, "citations": citations,
         "reasoning": args.reasoning, "permit_id": permit["permit_id"]})
    arm_permit(permit)
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
    st = load_state_strict()
    had_plan = False
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("step") == "plan" and e.get("direction") == args.direction:
                had_plan = True
                break
    if not had_plan:
        print("REFUSED: this direction has never had a FULL plan step — "
              "deltas only continue an already-planned direction.")
        return 1
    gkey = f"{args.direction}|{int(args.shape)}"
    grp = st.get("groups", {}).get(gkey, {})
    attempts = grp.get("attempts", 0)
    budget_attempts = int(card.get("budget_attempts", 6))
    if attempts >= budget_attempts:
        print(f"REFUSED: attempt budget exhausted for {gkey} "
              f"({attempts}/{budget_attempts}). Requires a full research+plan "
              "cycle (which forces the rethink) or the card's closure.")
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
    commit(st, {"ts": now(), "step": "delta", "plan_id": plan_id,
         "direction": args.direction, "mode": args.mode, "shape": args.shape,
         "changed": args.changed, "prediction": args.prediction,
         "permit_id": permit["permit_id"]})
    arm_permit(permit)
    print(f"DELTA accepted. Permit {permit['permit_id']} ARMED for ONE run.")
    return 0


def gate_lock():
    """One shared advisory lock serializing EVERY state transition (CLI and
    watcher). IDEMPOTENT within a process (flock on a second fd of the same
    file would self-deadlock). Blocks up to 30s, then fails CLOSED."""
    import fcntl
    if _LOCK_REF:
        return _LOCK_REF[0]
    LOOP.mkdir(parents=True, exist_ok=True)
    fh = open(LOOP / ".gate.lock", "w")
    deadline = time.time() + 30
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _LOCK_REF.append(fh)
            return fh
        except OSError:
            if time.time() > deadline:
                raise SystemExit("REFUSED: gate lock busy >30s — fail closed.")
            time.sleep(0.2)


def cmd_reconcile(args) -> int:
    """Called by the PostToolUse watcher (and idempotently by anyone):
    resolves an IN_FLIGHT attempt from its bound ledger. CLAIM-FIRST: the
    in-flight file is atomically claimed BEFORE any derived read, under the
    shared gate lock; a still-running attempt is restored intact."""
    import subprocess
    if not INFLIGHT.exists():
        return 0
    gate_lock()  # idempotent process-wide acquisition
    try:
        if not INFLIGHT.exists():
            return 0  # re-check under the lock: another reconciler won
        USED.mkdir(exist_ok=True)
        # Crash-leftover quarantine FIRST — while in_flight.json still blocks
        # the guard, so no window exists in which a stale ARMED permit is
        # consumable. Any armed permit coexisting with an in-flight attempt
        # is by definition stale (one attempt at a time).
        if PERMIT.exists():
            try:
                PERMIT.rename(USED / f"stale-permit.{secrets.token_hex(4)}.json")
            except Exception:
                raise SystemExit("REFUSED: could not quarantine the stale "
                                 "permit — reconciliation aborted (in_flight "
                                 "keeps blocking).")
        claim = USED / f"claim.{secrets.token_hex(4)}.json"
        try:
            INFLIGHT.rename(claim)  # atomic claim precedes every read
        except FileNotFoundError:
            return 0
        fl = load_json(claim, None)
        def _schema_ok(d):
            try:
                return (isinstance(d, dict)
                        and d.get("mode") in MODES
                        and isinstance(d.get("permit_id"), str) and d["permit_id"]
                        and isinstance(d.get("direction_id"), str) and d["direction_id"]
                        and int(d.get("shape")) >= 0
                        and isinstance(d.get("ledger"), str)
                        and int(d.get("ledger_pre_lines")) >= 0
                        and (d.get("impl_sha256") is None
                             or isinstance(d.get("impl_sha256"), str)))
            except Exception:
                return False
        if not _schema_ok(fl):
            # Invalid payload (bad JSON OR bad schema): the stranded claim
            # blocks all future permits at issuance AND at the guard —
            # fail closed, human/agent repairs against the log.
            raise SystemExit("REFUSED: in-flight payload malformed/invalid — "
                             "claim stranded to keep the gate closed; repair "
                             "against gate_log.jsonl.")
        st = load_state_strict()  # fail closed before any effects
        # Orphan protection: referee still running (script or module form) or
        # young rowless attempt -> RESTORE the claim and wait for a later pass.
        running = subprocess.run(
            ["pgrep", "-f", r"(runner\.py|harness[./]runner)\s+(run|calibrate)"],
            capture_output=True, text=True).stdout.strip()
        ledger = Path(fl["ledger"])
        rows = ([l for l in ledger.read_text().splitlines() if l.strip()]
                if ledger.exists() else [])
        new_rows = rows[fl["ledger_pre_lines"]:]
        import time as _t
        age = _t.time() - float(fl.get("consumed_epoch",
                                fl.get("expires_epoch", _t.time()) - PERMIT_TTL_S))
        if running or (not new_rows and age < 120):
            claim.rename(INFLIGHT)  # restore untouched; reconcile later
            return 0
        return _reconcile_locked(st, fl, claim, new_rows)
    finally:
        pass  # the process-wide lock is held until exit by design


def _reconcile_locked(st, fl, claim, new_rows) -> int:
    gkey = f"{fl['direction_id']}|{int(fl['shape'])}"
    grp = st.setdefault("groups", {}).setdefault(
        gkey, {"best_speedup": None, "strikes": 0, "exec_failures": 0,
               "closed": False, "closure_nonce": None, "attempts": 0,
               "nonstrike_budget": 2, "last_attempt_sha": None})
    grp.setdefault("attempts", 0)
    grp.setdefault("nonstrike_budget", 2)
    grp["attempts"] += 1
    if fl.get("impl_sha256"):
        shas = grp.setdefault("attempted_shas", [])
        if fl["impl_sha256"] not in shas:
            shas.append(fl["impl_sha256"])
    if fl["mode"] in ("confirmation", "correctness"):
        grp["nonstrike_budget"] = grp.get("nonstrike_budget", 2) - 1
    outcome = {"ts": now(), "step": "reconcile", "permit_id": fl["permit_id"],
               "direction": fl["direction_id"], "shape": fl["shape"],
               "mode": fl["mode"]}
    entry = None
    if len(new_rows) == 1:
        try:
            e = json.loads(new_rows[0])
            row_ok = (
                (fl["impl_sha256"] is None or
                 e.get("impl", {}).get("sha256") == fl["impl_sha256"])
                and int(e.get("shape_id", -1)) == int(fl["shape"])
            )
            entry = e if row_ok else None
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
        comparable = (entry.get("type") == "candidate" and
                      entry.get("profile") == "primary")
        outcome["comparable_primary"] = comparable
        if fl["mode"] == "optimization":
            prev_best = grp["best_speedup"]
            qualifying = (comparable and correct and clean and speed is not None)
            if qualifying and (prev_best is None or speed > prev_best):
                grp["best_speedup"] = speed  # true best, ANY margin
            improved = (qualifying and
                        (prev_best is None or speed > prev_best * IMPROVE_MARGIN))
            if improved:
                grp["strikes"] = 0
            else:
                grp["strikes"] += 1
            outcome["improved"] = improved
            outcome["prev_best"] = prev_best
        elif fl["mode"] == "screening":
            outcome["declared_prediction"] = fl.get("prediction")
            st["pending_screen_judgment"] = {
                "permit_id": fl["permit_id"], "group": gkey,
                "observed_speedup": speed, "row_correct": correct}
        # confirmation/correctness/calibration: recorded, never strike.
        if grp["strikes"] >= MAX_STRIKES:
            grp["closed"] = True
            grp["closure_nonce"] = secrets.token_hex(8)
            st.setdefault("pending_postmortem", []).append(gkey)
            outcome["closed"] = True
            outcome["closure_nonce"] = grp["closure_nonce"]
    commit(st, outcome)
    USED.mkdir(exist_ok=True)
    claim.replace(USED / f"{fl['permit_id']}.reconciled.json")
    if outcome.get("closed"):
        print(f"[run-gate] GROUP CLOSED: {gkey} (nonce {grp['closure_nonce']}). "
              "Postmortem now mandatory; reopening needs a critic receipt "
              "bound to this nonce.")
    return 0


def cmd_screen_judge(args) -> int:
    """Record the screening range hit/miss — bound to the pending screening
    attempt (one-use); the observed value must match the reconciled row."""
    st = load_state_strict()  # locked + seq-checked from the first read
    pend = st.get("pending_screen_judgment")
    gkey = f"{args.direction}|{int(args.shape)}"
    if not pend or pend.get("group") != gkey:
        print("REFUSED: no pending screening judgment for this group.")
        return 1
    obs = pend.get("observed_speedup")
    try:
        stated = float(args.observed)
    except ValueError:
        print("REFUSED: --observed must be the numeric speedup from the row.")
        return 1
    if obs is not None and abs(stated - obs) > max(0.01 * abs(obs), 1e-6):
        print(f"REFUSED: --observed {stated} does not match the reconciled "
              f"row's speedup {obs}.")
        return 1
    forced_miss = not pend.get("row_correct", True)
    st["pending_screen_judgment"] = None
    grp = st.setdefault("groups", {}).setdefault(
        gkey, {"best_speedup": None, "strikes": 0, "exec_failures": 0,
               "closed": False, "closure_nonce": None})
    if forced_miss and args.result == "hit":
        print("NOTE: the row failed correctness — recorded as a MISS "
              "regardless of the stated result.")
    if args.result == "miss" or forced_miss:
        grp["strikes"] += 1
        if grp["strikes"] >= MAX_STRIKES:
            grp["closed"] = True
            grp["closure_nonce"] = secrets.token_hex(8)
            st.setdefault("pending_postmortem", []).append(gkey)
    commit(st, {"ts": now(), "step": "screen_judge", "group": gkey,
         "result": args.result, "observed": args.observed,
         "strikes": grp["strikes"], "closed": grp["closed"]})
    print(f"screening {args.result} recorded for {gkey} "
          f"(strikes {grp['strikes']}/{MAX_STRIKES}).")
    return 0


def cmd_reopen(args) -> int:
    st = load_state_strict()
    grp = st.get("groups", {}).get(args.group)
    if not grp or not grp.get("closed"):
        print("Nothing closed under that group key.")
        return 1
    nonce = grp.get("closure_nonce")
    critic = Path(args.critic_log)
    if not (nonce and critic.exists()):
        print("REFUSED: closure nonce or critic log missing.")
        return 1
    strategy_dir = (ROOT / "Project" / "audits" / "strategy").resolve()
    if strategy_dir not in critic.resolve().parents:
        print("REFUSED: critic receipts must live in Project/audits/strategy/ "
              "(real critic output, not an arbitrary file).")
        return 1
    text = critic.read_text(errors="ignore")
    nonblank = [l.strip() for l in text.strip().splitlines() if l.strip()]
    verdict_ok = bool(nonblank) and re.fullmatch(
        r"CRITIC:\s*(continue|narrow)", nonblank[-1]) is not None
    if nonce not in text or not verdict_ok:
        print("REFUSED: the critic log must contain the EXACT closure nonce "
              f"({nonce}) and CONCLUDE (final lines) with CRITIC: continue|"
              "narrow. Receipts are one-use and closure-bound.")
        return 1
    if nonce in st.get("consumed_nonces", []):
        print("REFUSED: this closure nonce was already consumed.")
        return 1
    st.setdefault("consumed_nonces", []).append(nonce)
    grp["closed"] = False
    grp["strikes"] = 0
    grp["exec_failures"] = 0
    grp["closure_nonce"] = None
    commit(st, {"ts": now(), "step": "reopen", "group": args.group,
         "critic_log": str(critic), "consumed_nonce": nonce})
    print(f"Reopened {args.group} on critic authority (nonce consumed).")
    return 0


def cmd_init(_args) -> int:
    if STATE.exists():
        print("REFUSED: state already exists — init is first-time only.")
        return 1
    if LOG.exists() and LOG.stat().st_size > 0:
        print("REFUSED: gate history exists (gate_log.jsonl) — a missing "
              "state file must be brought back from version control, "
              "never re-initialized.")
        return 1
    if USED.exists() and any(USED.iterdir()):
        print("REFUSED: consumed permits exist — bring state back from "
              "version control.")
        return 1
    st = {"research_cycle": 0, "research_open": False, "groups": {},
          "pending_postmortem": [], "consumed_nonces": [],
          "pending_screen_judgment": None, "seq": 0}
    commit(st, {"ts": now(), "step": "init"})
    print("Gate state initialized CLOSED (event logged).")
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
    sub.add_parser("init")
    args = ap.parse_args()
    return {"research": cmd_research, "plan": cmd_plan, "delta": cmd_delta,
            "reconcile": cmd_reconcile, "screen-judge": cmd_screen_judge,
            "reopen": cmd_reopen, "status": cmd_status,
            "init": cmd_init}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
