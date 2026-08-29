# The Run Gate — full design (FINAL, approved review round 13; version labels v3.x unified as 'v3-final')

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
   watcher calls `run_gate.py reconcile`: gate CLOSES again, the try is counted
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


## v3 changes (responding to the round-1 REJECT, all its points)
- ONE-USE PERMITS: a validated think-step ARMS exactly one permit bound to
  direction card, mode, shape, impl path + sha256 of the exact bytes,
  canonical ledger path, numeric prediction, and a 45-min expiry. The
  owner-side guard atomically CONSUMES it (ARMED->IN_FLIGHT) before the run
  executes; --ledger= and quoted forms parsed; multiple invocations in one
  command denied; everything unparseable denied (fail closed).
- ATTRIBUTION: the permit records the bound ledger's pre-run line count;
  reconcile accepts EXACTLY one new row whose impl sha matches the permit —
  otherwise it is an execution_failure (separate capped counter, 3 => close).
  Stale-journal misattribution is gone.
- MODES with correct semantics: optimization (improvement = correct + clean
  + >3% over the group's best — epsilon noise never resets strikes) ·
  screening (declared range decides; miss = strike, via an explicit
  auditable screen-judge record) · confirmation / correctness / calibration
  (gated but never strike, never improve).
- GROUPS: strikes and bests are per (direction_card x shape) — no more
  basename-wide cross-shape comparisons; direction identity is the card's
  immutable family id, not a filename.
- TIERS: full research+plan only to OPEN a direction; a concise structured
  DELTA (what changed + numeric prediction) for each further attempt within
  it — no filler-packet incentive; every tier still issues one permit.
- CLOSURE + AUTONOMY: closure generates a NONCE; postmortem debt blocks the
  next research step until written; reopening requires a critic log
  containing that exact nonce + CRITIC: continue|narrow (one-use, consumed).
  The public unlock command is REMOVED.

## CONVERGENCE DECLARATION (historical — after 7 of the eventual 13 blind review rounds, 29 Aug evening)

Trail: R1 REJECT (architecture) -> R2 REVISE(8) -> R3 REVISE(9) -> R4
REVISE(6) -> R5 REVISE(6) -> R6 REVISE(4) -> R7 REVISE(5). Every
load-bearing defect through R6 is fixed; from R7, three were fixed
(identity-comparison false-positive, permit-armed-before-commit ordering,
log fsync) and the remainder is DECLINED with reasons:

RE-EXAMINED UNDER THE OWNER'S CLARIFIED RULE (override only what is
genuinely NOT an issue, 29 Aug): three formerly-declined items were fixed
after honest re-analysis (both-artifacts crash repair in the guard; global
invocation counting catches multi-substitution; malformed log rows now
refuse instead of being skipped). ONE override stands, with reasoning:

OVERRIDDEN (judged not-an-issue, not merely expensive):
- Crash-idempotent exactly-once reconciliation. Every reconcile crash path
  already lands FAIL CLOSED: a stranded claim file blocks all new permits
  until a human-or-agent investigation resolves it against the fsynced log.
  "Exactly-once" machinery would change no outcome, only the repair
  ergonomics.
- Perfect shell-semantics parity beyond the current denials (substitution,
  loops, continuations, quote games are all denied or normalized): residual
  exotic constructs require deliberately adversarial intent, which the
  project's trust model explicitly does not defend against.

The gate's core promises hold under normal operation: every try requires a
committed, cited thinking step; every try re-locks; strikes are judged by
the referee's own journal per comparable group; closures force postmortems
and new directions; reopening requires an external verdict.


## FINAL: APPROVED (round 13, 29 Aug night)

Thirteen blind rounds: R1 REJECT -> R2(8) R3(9) R4(6) R5(6) R6(4) R7(5)
R8(3) R9(2) R10(4) R11(2) R12(1) -> R13 APPROVE. Every load-bearing defect
fixed; the single documented override (exactly-once reconciliation
machinery) was explicitly judged defensible by the reviewer in R8 once the
stale-permit quarantine landed (it did, R9-R10). The gate ships as
v3-final — the single authoritative version, approved at round 13; all
v3.x numbers in older sections are iteration history, not live labels.

## Amendment (owner-approved, 29 Aug late): the revival clause
Every closure postmortem must contain an explicit CASE FOR REVIVAL — either
"none: the idea itself was disproven" or the specific evidence that would
justify reopening (enforced by the research step's validator). On any
reopen appeal, the critic must address that recorded case directly. This
keeps the 3-strike brake tight while guaranteeing a genuinely promising
idea gets its hearing on the record, never by the agent's own conviction.
