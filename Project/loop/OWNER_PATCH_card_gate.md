# OWNER PATCH: mechanical card gate + research-index injection

Purpose (your ask, 29 Aug afternoon): a hook that PHYSICALLY blocks
benchmarking any kernel that has no open experiment card. Card existence is
a binary invariant a hook can enforce; card QUALITY stays with the critic
and the preregistered-prediction rules. You apply this because
.claude/** is (correctly) locked against the agent.

## 1. Edit .claude/hooks/guard_bash.py

Insert this block AFTER the `WRITE_PATTERNS = [pat.replace(...)]` line
(~line 49) and BEFORE `def recursive_rm_outside_tmp`:

```python
# --- Card gate (research-first loop, 29 Aug): running the referee on a
# kernel file requires an OPEN experiment card naming it. Binary check
# only — card quality is enforced by the critic rules, not here.
# Grandfathered: pre-loop champions that re-run during board re-passes.
GRANDFATHERED_IMPLS = {
    "k001_sdpa.py", "k002_fused_qkv.py", "k003_triton_attention.py",
    "k004_graphed_triton.py", "k005_fp16_graphed.py", "k006_fp16_hd128.py",
    "k007_fused_block.py", "k009_fused_tuned.py", "k010_fused_ln.py",
    "k012_split_heads.py",
}


def card_gate_reason(command: str):
    m = re.search(r"runner\.py\s+(?:\S+\s+)*run\b[^|;&]*?--impl\s+(\S+)", command)
    if not m:
        return None
    base = m.group(1).strip("'\"").split("/")[-1]
    if base in GRANDFATHERED_IMPLS:
        return None
    import pathlib
    cards = pathlib.Path(__file__).resolve().parents[2] / "Project/loop/cards.jsonl"
    try:
        for line in cards.read_text().splitlines():
            if not line.strip():
                continue
            card = json.loads(line)
            if "killed" in str(card.get("status", "")).lower():
                continue
            if base in json.dumps(card):
                return None
    except Exception:
        pass  # unreadable cards file falls through to deny (deny-biased)
    return (f"Blocked by the card gate: '{base}' has no open experiment card "
            "in Project/loop/cards.jsonl. Preregister the card (hypothesis, "
            "prediction ranges, kill criteria) before the referee touches it "
            "- research-first loop, Project/drafts/harness_v2_proposal.md.")
```

Then insert into `main()`, directly after the `recursive_rm_outside_tmp`
check and before the WRITE_PATTERNS loop:

```python
    reason = card_gate_reason(command)
    if reason:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
              "permissionDecision": "deny",
              "permissionDecisionReason": reason}}))
        return
```

## 2. Edit .claude/settings.json — SessionStart also injects the research index

Change the SessionStart command value to:

```
echo '=== PROJECT STATE (auto-injected by SessionStart hook; read Project/PLAN.md for the full plan) ==='; cat "$CLAUDE_PROJECT_DIR/Project/memory/STATE.md" 2>/dev/null || echo 'STATE.md not found'; echo '=== RESEARCH INDEX (check before any research or build) ==='; cat "$CLAUDE_PROJECT_DIR/Project/research/INDEX.md" 2>/dev/null || true
```

## 3. Test after applying (agent runs these; both must behave as stated)

- `python3 Project/harness/runner.py run --shape 3 --impl Project/kernels/zzz_nocard.py`
  → must BOUNCE with the card-gate message (file existence irrelevant —
  the gate fires first).
- `python3 Project/harness/runner.py run --shape 11 --impl Project/kernels/k012_split_heads.py --ledger /tmp/x.jsonl`
  → must pass the gate (grandfathered).
- Restart the session once so the new SessionStart injection is live.

## 4. Optional knob you asked about ("rethink after 1-2 failed optimizations")

The converged rule is TWO preregistered-prediction misses per direction
family → forced critic review (plus kill-criteria hits and budget
exhaustion, each forcing it alone). It fired correctly last night (C3
killed, C4 closed). If you want ONE miss to force the rethink, say so and
the agent flips the threshold in harness_v2_proposal.md §1 and cards
tooling — no hook change needed.
