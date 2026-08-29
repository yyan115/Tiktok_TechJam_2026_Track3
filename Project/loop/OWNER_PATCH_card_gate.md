# OWNER PATCH — FINAL v4 (round-13 approval + 30 Aug authority hardening; supersedes every earlier version): permit-consuming run gate + state-write protection

Rebuilt after the external reviewer REJECTED v2 (stale-journal attribution,
cross-shape strike nonsense, regex evasions, self-service unlock). v3 is a
one-use-permit architecture: a validated thinking step ARMS exactly one
permit bound to the exact attempt; this guard CONSUMES it atomically before
the run executes; the watcher reconciles the outcome from the bound ledger.
Fail-closed everywhere.

HONEST SCOPE (30 Aug reviewer calibration): both blocks are COOPERATIVE
PATTERN-BASED guards over literal command text — accident prevention plus
bar-raising, per the repo's stated trust model. They match spellings before
shell expansion and do not observe canonical executable paths, and the
frozen runner performs no internal permit verification. So the true claim
is "every straightforward invocation is gated and every listed record
resists direct writes"; it is NOT absolute isolation. Deliberate evasion
remains possible and remains naked, logged misconduct (GATE_DESIGN HONESTY
LEDGER items 3).

## Paste into .claude/hooks/guard_bash.py

Block A — insert AFTER the `WRITE_PATTERNS = [pat.replace(...)]` line:

```python
# --- Permit gate v3 (owner mandate 29 Aug). A referee run/calibrate is
# allowed ONLY by atomically consuming the single ARMED permit that a
# validated think-step issued (Project/tools/run_gate.py). Anything
# unparseable, mismatched, duplicated, or stale => DENY (fail closed).
def permit_gate_reason(command):
    # Normalize EXACTLY like bash where it matters: backslash-newline is
    # DELETED (so run\<newline>ner.py reforms), quotes are removed (so
    # fragmented paths reform), then split on separators incl. newlines.
    norm = command.replace("\\\n", "").replace('"', "").replace("'", "")
    # A referee-adjacent command with substitution or loops is denied BEFORE
    # any segment matching — `a=run; runner.py "$a"` must not slip through
    # because the literal word run is missing.
    ANCHOR = r"(runner\.py|harness[./]runner)"  # single source of truth
    touches_referee = re.search(ANCHOR, norm)
    if touches_referee:
        if re.search(r"\b(for|while|until)\b", norm):
            return ("Blocked: shell loops around a referee invocation violate "
                    "one-permit-one-run. Write the single literal command.")
        if "$" in norm or "`" in norm or "<(" in norm or ">(" in norm:
            return ("Blocked: variable/command/process substitution near a "
                    "referee command makes permit bindings unverifiable. "
                    "Write single literal commands.")
        if len(re.findall(ANCHOR, norm)) > 1:
            return "Blocked: multiple referee invocations in one command (one permit = one run)."
    REF = ANCHOR + r"\s+(?:\S+\s+)*(run|calibrate)\b"
    segs = [s for s in re.split(r"[|;&\n\r]+", norm) if re.search(REF, s)]
    if not segs:
        return None
    if len(segs) > 1:
        return "Blocked: multiple referee invocations in one command (one permit = one run)."
    seg = segs[0]
    # No other segment of this command may touch the permitted impl file
    # (write-then-run in one command would benchmark unpermitted bytes).
    all_segs = [s for s in re.split(r"[|;&\n\r]+", norm) if s.strip()]
    other = [s for s in all_segs if not re.search(REF, s)]
    # Abbreviated long options are DENIED outright — the guard refuses to
    # guess argparse's expansion, so bindings can't be split across
    # duplicate abbreviated flags.
    known = ("--shape", "--impl", "--ledger", "--dtype", "--warmup",
             "--repeats", "--rounds")
    for tok in set(re.findall(r"--[a-z-]+", seg)):
        if tok not in known and any(k.startswith(tok) for k in known):
            return (f"Blocked: abbreviated option {tok} — spell referee "
                    "options in full so permit bindings are unambiguous.")
    import pathlib, time as _t
    root = pathlib.Path(__file__).resolve().parents[2]
    loop = root / "Project/loop"
    permit_p = loop / "permit.json"
    # LOCK FIRST: every artifact read, validation, cleanup, and the consume
    # below happen under the same shared flock the gate machinery uses — a
    # delayed guard can never act on a stale snapshot or delete a fresh
    # permit. Contention = deny. The fd stays open through the consume.
    import fcntl as _f
    loop.mkdir(parents=True, exist_ok=True)
    _lock_fh = open(loop / ".gate.lock", "w")
    try:
        _f.flock(_lock_fh, _f.LOCK_EX | _f.LOCK_NB)
    except OSError:
        return ("Blocked: the gate lock is held (reconciliation or a "
                "think-step in progress) — retry in a moment.")
    if (loop / "in_flight.json").exists():
        if permit_p.exists():
            # A crash between consume steps left BOTH artifacts: in_flight is
            # authoritative; the permit must never be reusable. Repair here.
            try: permit_p.unlink()
            except Exception: pass
        return ("Blocked: a previous attempt is unreconciled (in_flight.json). "
                "Run: python3 Project/tools/run_gate.py reconcile")
    used_dir2 = loop / "permits_used"
    if used_dir2.exists() and any(f.name.startswith("claim.")
                                  for f in used_dir2.iterdir()):
        return ("Blocked: a stranded reconciliation claim exists — the gate "
                "stays closed until it is repaired against gate_log.jsonl.")
    try:
        permit = json.loads(permit_p.read_text())
    except Exception:
        return ("Blocked by the permit gate: no ARMED permit. Complete the "
                "think-step first (run_gate.py research+plan for a new "
                "direction, or run_gate.py delta within an open one).")
    if _t.time() > float(permit.get("expires_epoch", 0)):
        try: permit_p.unlink()
        except Exception: pass
        return "Blocked: the ARMED permit expired. Issue a fresh think-step."
    # Brake recheck at consumption (reviewer round 1): if any audit verdict
    # landed after this permit was issued, the verdict brake must be
    # re-evaluated — the stale permit is void.
    _vf = root / "Project/audits/verdicts.jsonl"
    if not _vf.exists():
        return ("Blocked: the audit verdict record is MISSING — that is "
                "never a green light (fail closed).")
    try:
        _vc = len([l for l in _vf.read_text().splitlines() if l.strip()])
    except Exception:
        return "Blocked: cannot read the verdict record (fail closed)."
    if _vc != int(permit.get("verdict_lines", -1)):
        try: permit_p.unlink()
        except Exception: pass
        return ("Blocked: audit verdicts changed since this permit was "
                "issued — re-run the think-step so the brake re-evaluates.")
    used_dir = loop / "permits_used"
    pid = str(permit.get("permit_id", ""))
    if pid and used_dir.exists() and any(f.name.startswith(pid)
                                         for f in used_dir.iterdir()):
        try: permit_p.unlink()
        except Exception: pass
        return ("Blocked: this permit id was already consumed (stale leftover "
                "quarantined). Issue a fresh think-step.")
    def _opt(name):
        ms = re.findall(name + r"(?:=|\s+)([^\s]+)", seg)
        return ms[-1] if ms else None  # argparse takes the LAST duplicate
    shape = _opt("--shape")
    impl = _opt("--impl")
    ledger = _opt("--ledger")
    if shape is None or int(shape) != int(permit.get("shape", -1)):
        return f"Blocked: permit is bound to shape {permit.get('shape')}, command has {shape}."
    pi = permit.get("impl_path")
    if (pi is None) != (impl is None):
        return "Blocked: permit/command impl presence mismatch (calibration vs run)."
    if impl is not None:
        cand = pathlib.Path(impl)
        cand = cand if cand.is_absolute() else (root / impl)
        try:
            import hashlib as _h
            actual = _h.sha256(cand.read_bytes()).hexdigest()
        except Exception:
            return "Blocked: cannot read the impl file to verify the permit binding."
        if str(cand.resolve().relative_to(root)) != pi or actual != permit.get("impl_sha256"):
            return ("Blocked: permit is bound to different candidate bytes "
                    f"({pi} @ {str(permit.get('impl_sha256'))[:12]}…).")
        base = cand.name
        if any(base in s for s in other):
            return ("Blocked: another segment of this command references the "
                    "permitted candidate file — modify-then-run in one "
                    "command would benchmark unpermitted bytes.")
    lp = pathlib.Path(ledger).resolve() if ledger else (root / "Project/results/JOURNAL.jsonl")
    if str(lp) != permit.get("ledger"):
        return f"Blocked: permit is bound to ledger {permit.get('ledger')}."
    # No other segment may touch the bound ledger — run-then-rewrite in one
    # command would launder the evidence the reconcile step trusts.
    lbase = pathlib.Path(permit.get("ledger", "")).name
    if lbase and any(lbase in s for s in other):
        return ("Blocked: another segment of this command references the "
                "bound ledger file — evidence laundering pattern.")
    if permit.get("mode") in ("optimization", "confirmation"):
        # Deny any long option that PREFIX-matches a profile override —
        # argparse accepts abbreviations (--dtyp, --warm, --rep, --rou).
        # Confirmation runs satisfy RETESTs, so they get the same binding.
        for tok in re.findall(r"--[a-z-]+", seg):
            if any(f.startswith(tok) for f in
                   ("--dtype", "--warmup", "--repeats", "--rounds")):
                return ("Blocked: optimization/confirmation permits are bound "
                        f"to the primary profile — override option {tok} present.")
    # CONSUME: ARMED -> IN_FLIGHT (still under the lock acquired above),
    # O_EXCL for the destination, pre-run ledger offset captured NOW,
    # before execution.
    try:
        import os as _os
        lp2 = pathlib.Path(permit["ledger"])
        permit["ledger_pre_lines"] = (
            len([l for l in lp2.read_text().splitlines() if l.strip()])
            if lp2.exists() else 0)
        permit["consumed"] = _t.strftime("%Y-%m-%dT%H:%M:%S%z")
        permit["consumed_epoch"] = _t.time()
        fd = _os.open(str(loop / "in_flight.json"),
                      _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY, 0o644)
        with _os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(permit, indent=1, sort_keys=True))
        permit_p.unlink()
    except FileExistsError:
        return "Blocked: another attempt is already IN_FLIGHT."
    except Exception:
        return "Blocked: permit consumption failed (fail closed)."
    return None
```

Block A2 — insert immediately AFTER Block A (same anchor region). Protects
the gate's memory, the verdict ledger, Sol's receipt store, and the three
enforcer tools from direct agent writes (Track-2 lesson: the actor must not
be able to edit its own referee):

```python
# --- State-write protection (owner mandate 30 Aug, Track-2 postmortem).
# Gate state, the verdict ledger, critic receipts, and the enforcer tools
# change ONLY through run_gate.py's validated commands, the audit pipeline,
# or a real codex call. Deny-biased: unusual access patterns bounce.
def state_write_reason(command):
    norm = command.replace("\\\n", "").replace('"', "").replace("'", "")
    GS = (r"Project/(loop/(gate_state\.json|gate_log\.jsonl|permit\.json|"
          r"in_flight\.json|permits_used|\.gate\.lock)|"
          r"audits/(strategy/|verdicts\.jsonl|auto/|verdict_schema\.json)|"
          r"tools/(run_gate\.py|audit_champion\.py|champion_watch\.py|"
          r"\.champion_cache\.json))")
    segs = [s for s in re.split(r"[|;&\n\r]+", norm) if re.search(GS, s)]
    for s in segs:
        t = s.strip()
        if (re.search(r"\bcodex\s+exec\b", t)
                and not re.search(r"Project/(loop/|audits/(verdicts|auto/|"
                                  r"verdict_schema)|tools/)", t)):
            continue  # critic consultations may write ONLY strategy receipts
        if (re.match(r"(python3?\s+)?\S*run_gate\.py\s+(research|plan|delta|"
                     r"reconcile|screen-judge|verdict-clear|reopen|status|"
                     r"init)\b", t) and ">" not in t):
            continue  # the gate's own validated commands, no redirection
        if (re.match(r"(python3?\s+)?\S*(audit_champion|champion_watch)\.py\b", t)
                and ">" not in t):
            continue  # launching the auditor/watcher, no redirection
        if (re.match(r"(cat|head|tail|less|grep|wc|ls|stat|file|sha256sum|"
                     r"diff|sed\s+-n)\b", t) and ">" not in t
                and not re.search(r"\s-i\b|--in-place", t)):
            continue  # plain reads (sed only without any in-place flag)
        if (re.match(r"git\s+(add|commit|log|show|diff|status)\b", t)
                and ">" not in t
                and not re.search(r"\s--?o(ut(put)?)?\b|--output", t)):
            continue  # version control on state; no redirection/output opts
        return ("Blocked: gate state, the verdict ledger, critic receipts, "
                "and the enforcer tools change only through run_gate.py, "
                "the audit pipeline, or a real codex call — direct writes "
                "are the Track-2 disease (self-serviced authority).")
    return None
```

Block B — insert into `main()` at the VERY END, AFTER the WRITE_PATTERNS
loop (i.e., after the last existing deny check, just before the function
returns). Permit consumption must be the FINAL authorization action: a
command that any other check would deny must be denied BEFORE its permit is
consumed, or doomed commands burn permits and record false failures. The
state-write check runs FIRST inside the same fail-closed wrapper (it never
consumes anything).

```python
    # FAIL CLOSED: any exception inside the gates is itself a DENY — a hook
    # crash exits nonzero-but-not-2, which Claude Code treats as ALLOW.
    try:
        gate = state_write_reason(command) or permit_gate_reason(command)
    except Exception as exc:
        gate = f"Blocked: permit gate error ({type(exc).__name__}) — fail closed."
    if gate:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
              "permissionDecision": "deny",
              "permissionDecisionReason": gate}}))
        return
```

## ALSO: add these lines to the "deny" array in .claude/settings.json
(blocks the Edit/Write tools on the same files; the guard above covers
bash). Paste inside the existing `"deny": [ ... ]` list:

```json
      "Edit(Project/loop/gate_state.json)", "Write(Project/loop/gate_state.json)",
      "Edit(Project/loop/gate_log.jsonl)", "Write(Project/loop/gate_log.jsonl)",
      "Edit(Project/loop/permit.json)", "Write(Project/loop/permit.json)",
      "Edit(Project/loop/in_flight.json)", "Write(Project/loop/in_flight.json)",
      "Edit(Project/loop/permits_used/**)", "Write(Project/loop/permits_used/**)",
      "Edit(Project/audits/strategy/**)", "Write(Project/audits/strategy/**)",
      "Edit(Project/audits/verdicts.jsonl)", "Write(Project/audits/verdicts.jsonl)",
      "Edit(Project/tools/run_gate.py)", "Write(Project/tools/run_gate.py)",
      "Edit(Project/tools/audit_champion.py)", "Write(Project/tools/audit_champion.py)",
      "Edit(Project/tools/champion_watch.py)", "Write(Project/tools/champion_watch.py)",
      "Edit(Project/audits/verdict_schema.json)", "Write(Project/audits/verdict_schema.json)",
      "Edit(Project/audits/auto/**)", "Write(Project/audits/auto/**)",
      "Edit(Project/loop/.gate.lock)", "Write(Project/loop/.gate.lock)",
      "Edit(Project/tools/.champion_cache.json)", "Write(Project/tools/.champion_cache.json)",
```

Note the tradeoff you are accepting: after this paste, Fable cannot edit
the gate or auditor at all — future gate fixes need you to lift the
specific rule, let Fable patch, then re-add it.

## Deliberate decisions from review round 1 (owner-ratified design, not bugs)
- CRITIC-AUTHORIZED REOPENING STAYS. The reviewer wants every reopen to be
  an owner decision. The 3-strike closure is an INTERNAL research
  heuristic (policy taxonomy: internal heuristics may be reopened
  prospectively), not a competition rule — the critic-appeal path is the
  owner-approved design and keeps the owner out of routine research flow.
  Receipts are length-checked, nonce-bound, one-use, sha-recorded, and
  (post-paste) writable only via a real codex call. Residual: HONESTY
  LEDGER item 3.
- cards.jsonl STAYS AGENT-WRITABLE. Cards are research documents; closure
  authority lives in gate_state groups, which a card edit cannot reopen
  (plan checks the group, not the card status).
- TIMESTAMP ORDERING is retained for the verdict cutoff: every writer on
  this box stamps +0800; the brake additionally fails closed on missing,
  malformed, or conflicting-duplicate verdict rows, and permits are void
  if the verdict count changes before consumption.

## Test after applying (agent runs, expected results stated)
1. Any `runner.py run ...` with no permit → BOUNCES with the think-step message.
2. Full think-step → permit ARMED → the exact bound command passes ONCE →
   in_flight.json exists → the same command again → BOUNCES (unreconciled).
3. `run_gate.py reconcile` → outcome recorded from the bound ledger row →
   next run BOUNCES again (no permit) until a new think-step.
4. A command with mismatched --shape / edited impl bytes / different
   --ledger (incl. `--ledger=path` form) → BOUNCES on binding mismatch.
5. `echo x >> Project/loop/gate_state.json` → BOUNCES (state-write).
6. `echo forged > Project/audits/strategy/fake.log` → BOUNCES; a real
   `codex exec ... > Project/audits/strategy/x_raw.log` → ALLOWED.
7. Edit tool on Project/tools/run_gate.py → BOUNCES (settings deny rule).
8. `python3 Project/tools/run_gate.py status` → still works (carve-out).
```
