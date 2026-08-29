# The Run Gate — full design (v2, for external review)

Goal (owner's spec): after every benchmark attempt the system must FORCE a
stop-and-rethink, no matter what the agent believes. Two explicit ordered
steps reopen it: (1) RESEARCH grounded in the saved research base, then
(2) PLAN with exact cited sources. Three attempts without improvement close
a direction. Fully autonomous: closures force a written postmortem and a
DIFFERENT direction; no human unlock in the loop.

## Enforcement split (who can touch what)
- BLOCKING side: ~20 lines in .claude/hooks/guard_bash.py (owner-applied;
  the agent cannot edit .claude/**). Denies any `runner.py run --impl` when
  the gate is closed or the direction is CLOSED.
- MACHINERY side (agent-built, self-tested): Project/tools/run_gate.py +
  auto-close wired into the existing PostToolUse watcher; state in
  Project/loop/gate_state.json; auditable trail in Project/loop/gate_log.jsonl.
- Trust model unchanged (PLAN.md): mistakes-not-malice. State is agent-
  writable; the deny logic is not. Faked citations are caught downstream.

## The cycle, mechanically
1. Gate starts CLOSED. Any referee run bounces off the guard.
2. `run_gate.py research` — validated, not promised:
   - must carry the CURRENT sha16 of research/INDEX.md (proves this-cycle read),
   - must cite >=2 EXISTING research-base files,
   - must include a >=200-char written summary,
   - if any direction was just CLOSED: a >=200-char POSTMORTEM is mandatory
     (predicted vs happened, why it failed, what it rules out) before the
     step is accepted — the unbiased-restart forcing function.
3. `run_gate.py plan` — refused unless research happened this cycle:
   - hypothesis (>=50 chars), NUMERIC prediction, kill criteria,
   - CITATIONS as file:line-line — validated to exist and resolve; the gate
     copies the EXACT QUOTED TEXT into the log,
   - reasoning (>=100 chars) why those sources justify the plan.
4. One referee run is now allowed. The moment it executes, the PostToolUse
   watcher calls `run_gate.py post`: gate CLOSES again, the try is counted
   against the impl's direction, and improvement is judged from the
   REFEREE'S OWN JOURNAL entry (never the agent's claim).
5. Improvement resets the direction's counter. 3 tries without improvement
   => direction CLOSED + postmortem debt registered. Agent continues
   autonomously on a different direction (same two steps required).
6. Reopening a CLOSED direction: only via `reopen --critic-log <file>` where
   the file is an existing external-critic verdict naming the direction and
   ending CRITIC: continue|narrow. Self-blessing impossible; owner not needed.
7. VERIFICATION LOOP: the per-champion blind auditor's prompt now includes
   citation verification — open the cited files at the cited lines, compare
   with the stored quotes, and treat fabricated/mismatched/irrelevant
   citations as RULE_VIOLATION findings.

## Verified in self-test (29 Aug)
bad index hash refused · plan-before-research refused · empty/fake/out-of-
range citations refused · quoted text stored · gate auto-closes after every
run · 3 non-improving tries => CLOSED · postmortem then forced on the next
research step · reopen refused without a critic verdict file.

## Known limitations (stated, not hidden)
- Screening runs and production runs both count as tries (intentional:
  a try is a try).
- The agent could in principle edit gate_state.json directly — outside the
  trust model, and the gate_log + auditor citations make it visible.
- Improvement = higher speedup in the journal the run wrote; calibration
  runs and non-run commands don't touch the gate.
