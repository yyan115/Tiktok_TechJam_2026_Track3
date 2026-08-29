# OWNER PATCH v3.2 (supersedes v1/v2/v3): permit-consuming run gate

Rebuilt after the external reviewer REJECTED v2 (stale-journal attribution,
cross-shape strike nonsense, regex evasions, self-service unlock). v3 is a
one-use-permit architecture: a validated thinking step ARMS exactly one
permit bound to the exact attempt; this guard CONSUMES it atomically before
the run executes; the watcher reconciles the outcome from the bound ledger.
Fail-closed everywhere.

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
    if permit.get("mode") == "optimization":
        # Deny any long option that PREFIX-matches a profile override —
        # argparse accepts abbreviations (--dtyp, --warm, --rep, --rou).
        for tok in re.findall(r"--[a-z-]+", seg):
            if any(f.startswith(tok) for f in
                   ("--dtype", "--warmup", "--repeats", "--rounds")):
                return ("Blocked: optimization permits are bound to the "
                        f"primary profile — override option {tok} present.")
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

Block B — insert into `main()`, after the `recursive_rm_outside_tmp` check:

```python
    # FAIL CLOSED: any exception inside the gate is itself a DENY — a hook
    # crash exits nonzero-but-not-2, which Claude Code treats as ALLOW.
    try:
        gate = permit_gate_reason(command)
    except Exception as exc:
        gate = f"Blocked: permit gate error ({type(exc).__name__}) — fail closed."
    if gate:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
              "permissionDecision": "deny",
              "permissionDecisionReason": gate}}))
        return
```

## Test after applying (agent runs, expected results stated)
1. Any `runner.py run ...` with no permit → BOUNCES with the think-step message.
2. Full think-step → permit ARMED → the exact bound command passes ONCE →
   in_flight.json exists → the same command again → BOUNCES (unreconciled).
3. `run_gate.py reconcile` → outcome recorded from the bound ledger row →
   next run BOUNCES again (no permit) until a new think-step.
4. A command with mismatched --shape / edited impl bytes / different
   --ledger (incl. `--ledger=path` form) → BOUNCES on binding mismatch.
```
