# OWNER PATCH v2 (supersedes v1): the mechanical two-step run gate

What you asked for, verbatim implemented: after EVERY referee run the gate
closes mechanically; the next run is DENIED until Claude completes TWO
explicit ordered steps — (1) RESEARCH, validated against the saved research
base (current INDEX hash + existing note files + written summary), then
(2) PLAN (numeric prediction + kill criteria). Max 3 runs per direction
without improvement (improvement judged from the referee's own journal, not
Claude's claims) → the direction is STOPPED until YOU unlock it.

Already built and self-tested by the agent (it cannot build the blocking
side — .claude/ is locked against it, by your design):
- Project/tools/run_gate.py         (the gate CLI: research/plan/post/status/unlock)
- Project/tools/champion_watch.py   (extended: mechanically closes the gate
                                     after every run via the existing PostToolUse hook)
- Project/loop/gate_state.json      (state, initialized CLOSED)
- Project/loop/gate_log.jsonl       (auditable trail of every step)
Verified behaviors: bad index-hash refused · plan-before-research refused ·
gate auto-closes after each run · 3 non-improving tries → STOPPED.

## THE ONE THING YOU DO: paste this into .claude/hooks/guard_bash.py

Insert AFTER the `WRITE_PATTERNS = [pat.replace(...)]` line (~line 49):

```python
# --- Two-step run gate (owner mandate 29 Aug): a referee run is allowed
# only when Project/loop/gate_state.json shows research_done AND plan_done
# (both are set only by validated run_gate.py steps and are cleared
# mechanically after every run by the PostToolUse watcher). A direction
# stopped after 3 non-improving tries stays blocked until owner unlock.
def run_gate_reason(command):
    m = re.search(r"runner\.py\s+(?:\S+\s+)*run\b[^|;&]*?--impl\s+(\S+)", command)
    if m is None:
        return None
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    try:
        st = json.loads((root / "Project/loop/gate_state.json").read_text())
    except Exception:
        return ("Blocked: run-gate state unreadable. Complete the two-step "
                "gate (Project/tools/run_gate.py research, then plan).")
    base = m.group(1).strip("'\"").split("/")[-1]
    fam = st.get("families", {}).get(base, {})
    if fam.get("stopped"):
        return (f"Blocked: direction '{base}' is STOPPED (3 tries without "
                "improvement per the referee journal). OWNER unlock required: "
                "python3 Project/tools/run_gate.py unlock --family " + base)
    if not (st.get("research_done") and st.get("plan_done")):
        return ("Blocked by the two-step gate: (1) python3 Project/tools/"
                "run_gate.py research --index-hash <sha16 of research/INDEX.md> "
                "--notes <files read> --summary <what was learned>; THEN "
                "(2) run_gate.py plan --hypothesis ... --prediction <numeric> "
                "--kill ... Then exactly one referee run is allowed.")
    return None
```

Insert into `main()`, directly after the `recursive_rm_outside_tmp` check:

```python
    gate = run_gate_reason(command)
    if gate:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
              "permissionDecision": "deny",
              "permissionDecisionReason": gate}}))
        return
```

(Optional, from patch v1 — still recommended: the SessionStart change that
also injects Project/research/INDEX.md, in .claude/settings.json.)

## Test after applying (agent runs; expected results stated)
1. `python3 Project/harness/runner.py run --shape 3 --impl Project/kernels/k009_fused_tuned.py --ledger /tmp/x.jsonl`
   → MUST BOUNCE with the two-step gate message (gate is currently closed).
2. Agent performs research+plan steps, run allowed once, then the very next
   run attempt MUST BOUNCE again without new steps.

## Honesty note (trust model, unchanged from PLAN.md)
State lives in Project/loop/ which the agent can write; the guard in
.claude/ it cannot. This stops momentum and self-deception mechanically —
the same mistakes-not-malice standard as every other lock in this project.
