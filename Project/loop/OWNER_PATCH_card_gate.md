# OWNER PATCH v3 (supersedes v1/v2): permit-consuming run gate

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
    hits = re.findall(r"runner\.py\s+(?:\S+\s+)*(run|calibrate)\b[^|;&]*", command)
    if not hits:
        return None
    if len(hits) > 1:
        return "Blocked: multiple referee invocations in one command (one permit = one run)."
    seg = re.search(r"runner\.py\s+(?:\S+\s+)*(?:run|calibrate)\b[^|;&]*", command).group(0)
    import pathlib, time as _t
    root = pathlib.Path(__file__).resolve().parents[2]
    loop = root / "Project/loop"
    permit_p = loop / "permit.json"
    if (loop / "in_flight.json").exists():
        return ("Blocked: a previous attempt is unreconciled (in_flight.json). "
                "Run: python3 Project/tools/run_gate.py reconcile")
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
    def _opt(name):
        m = re.search(name + r"(?:=|\s+)['\"]?([^\s'\"]+)", seg)
        return m.group(1) if m else None
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
    lp = pathlib.Path(ledger).resolve() if ledger else (root / "Project/results/JOURNAL.jsonl")
    if str(lp) != permit.get("ledger"):
        return f"Blocked: permit is bound to ledger {permit.get('ledger')}."
    if permit.get("mode") == "optimization":
        for flag in ("--dtype", "--warmup", "--repeats", "--rounds"):
            if flag in seg:
                return ("Blocked: optimization permits are bound to the "
                        "primary profile — no timing/dtype overrides "
                        f"({flag} present).")
    # CONSUME: ARMED -> IN_FLIGHT, atomically (O_EXCL prevents double
    # consumption), pre-run ledger offset captured NOW, before execution.
    try:
        import os as _os
        lp2 = pathlib.Path(permit["ledger"])
        permit["ledger_pre_lines"] = (
            len(lp2.read_text().splitlines()) if lp2.exists() else 0)
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
    gate = permit_gate_reason(command)
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
