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

## Amendment v4 (owner-approved GO, 30 Aug): authority hardening after the Track-2 violation
Triggered by the Track 2 postmortem (the sister project's agent consciously
overrode the organizers' convergence rule 11 times through a harness-provided
flag). Full compilation + plan: AUTHORITY_V4_PLAN.md. Changes live in
run_gate.py (22/22 sandbox tests):
- Screening hit/miss is COMPUTED from --predict-min/--predict-max bound at
  permit time; the agent's self-declared --result is gone.
- Hard-verdict brake: a post-gate RULE_VIOLATION freezes ALL permits until
  `verdict-clear --kind violation --owner-quote` (owner's literal chat
  ruling; transcript is the proof). A RETEST allows only confirmation-mode
  permits until `verdict-clear --kind retest --confirm-entry` passes the
  MECHANICAL check (same bytes + same shape + newer + passed, read from the
  journal). No AI-owned free-text unlock exists anywhere.
- Owner paste v4 adds Block A2 + settings deny rules: gate state files, the
  verdict ledger, critic receipts, and the three enforcer tools become
  unwritable by the agent outside validated commands / real codex calls.

## Amendment v4.1 (30 Aug, folding the blind reviewer's REVISE — distilled by the owner)
All six findings addressed: (1) plan/delta parsers actually carry
--predict-min/--predict-max now (the reviewer caught a real CLI-breaking
regression my internals-only tests missed); (2) owner quotes are one-use
(hash-tracked) and receipts must be full-length critiques with sha256
journaled — plus the honest admission that text provenance on one machine
is forge-obvious, not forge-proof; (3) RETEST now blocks everything except
confirmations bound to the retested bytes on the primary journal, and the
mechanical clear requires the confirmation to postdate the VERDICT, not
just the original row; (4) optimization/confirmation permits refuse scratch
ledgers; screening with no recorded measurement is an execution failure,
never judged from a typed number; the champion auditor accepts verdicts
only from cleanly-exited, unambiguous, stdout-only reviews (else
JUDGE_ERROR); (5) the owner patch now states its honest scope (cooperative
pattern guard) and its allow-branches all reject redirection, with codex
receipts writable only into strategy/; (6) a committed CLI-level suite
(Project/tools/tests/gate_v4_cli_test.py, 26/26) drives the DEPLOYED
command line via subprocess — no monkeypatching. Residual (accepted, in
HONESTY LEDGER): frozen runner has no internal permit check; the reviewer's
retest_request field is not preserved by the frozen recorder, so mechanical
clears enforce bytes/shape/recency/correctness, not bespoke procedures.

## Amendment v4.2 (30 Aug, convergence-loop round 1 REVISE folded)
Real catches fixed: screen-judge's no-measurement path now closes the group
at MAX_EXEC_FAILURES and reconcile no longer resets the counter on
speed-less rows; prediction bounds must be FINITE; RETEST confirmations
bind to (bytes, shape_id) pairs — no cross-shape/cross-candidate satisfying
— and a mechanical clear additionally requires the gate log to witness a
reconciled confirmation-mode permit producing that row; permits snapshot
the verdict-record line count and the guard voids them at consumption if a
new verdict landed; a missing/unreadable verdict record or conflicting
duplicate rows brake everything; guard protections extend to the lock file,
audit schema, and auto-audit store. DELIBERATE non-adoptions (owner-ratified,
recorded in OWNER_PATCH): critic reopening stays (internal-heuristic class),
cards stay agent-writable (closure authority lives in gate state), +0800
timestamp ordering stays (single-writer box, with tamper brakes). CLI suite
now 32/32.

## Amendment v4.3 (30 Aug, convergence-loop round 2 folded — reviewer confirmed round-1 fixes present, 32/32 suite run by the reviewer itself; "implementation failures, not design objections")
(1) One verdict-record snapshot feeds both brake evaluation and the count
bound into permits; hollow rows ({}) brake; guard denies consume when the
record is MISSING, not just changed; the auditor serializes recording
behind the runner-idle wait. (2) Guard: no other command segment may touch
the bound run record file (evidence-laundering pattern). (3) Prediction
ranges must be falsifiable: min >= 0.5, max <= 2x min. (4) Bounded retest
sampling: max 3 confirmation permits per retest then owner-only; once
satisfying evidence exists further re-rolls are refused — clear it or lose
it. (5) A2: sed in-place and git output flags excluded from read allowances,
codex writes only into strategy/, champion cache + lock file protected.
(6) Auditor: formatting-agnostic verdict parse (decoy-resistant), full
stdout retained as an immutable response artifact hashed as recorded,
record success verified, failed records re-queue the champion. NOTE (also
reviewer-found): the 57 historical audit receipts hash-mismatch their logs
by exactly the two post-record status lines — systematic, explainable,
verifiable by stripping those lines; future receipts use the immutable
response artifact. Suite: 37/37.

## Amendment v4.4 (30 Aug, convergence-loop round 3 folded — 5 items, all
implementation-layer; reviewer confirmed round-2 fixes on nominal paths)
(1) Verdict recording is now serialized: the auditor records only while
holding the shared gate lock with no attempt in flight — no post-verdict
run can slip between a permit's brake check and execution. (2) The watcher's
launch-time cache is GONE: audit-handled is derived from durable verdict
rows plus live process markers with stale-marker cleanup; crashed audits
refire instead of being suppressed (this also surfaces a 24-entry backlog —
NARROWINGS 10b). Response artifacts are per-attempt unique and claimed with
exclusive-create (no double-launch, no overwritten receipts). (3) The
verdict is parsed from ONE complete JSON document with a token cross-check;
prose, decoys, and partial objects are JUDGE_ERROR. (4) sed and less are
removed from the guard's read allowance entirely (native write options);
other segments may not carry the bound run-record name OR a --ledger token.
(5) Guard + parser regression tests are COMMITTED, with the guard tested
from the owner patch's own bytes (tests/guard_and_auditor_test.py, 30/30;
CLI suite 37/37).

## HONESTY LEDGER (binding for the report)
1. NOT LIVE UNTIL PASTED: this design has zero force until the owner paste
   lands in .claude/hooks/guard_bash.py. The gate's history to date is
   self-tests only.
2. PRE-GATE LABELING: every measurement recorded before the paste timestamp
   was produced WITHOUT a blocking gate. The report must never imply the
   kernel campaign ran under this gate. Its integrity rests on the frozen
   runner, tripwires, committed bytes, and Sol audits — which is true and
   sufficient, and is what we say.
3. RESIDUAL LIMITS (single machine, single user): critic receipts and owner
   quotes are forge-OBVIOUS, not forge-PROOF — evasion requires deliberate
   multi-step misconduct that the journals and transcripts expose; guard
   coverage is pattern-based (bash) + tool deny rules, not OS enforcement;
   the frozen runner does not internally verify permits (deferred re-freeze,
   post-competition wishlist). Claims above this line are never made.

## Amendment (owner-approved, 29 Aug late): the revival clause
Every closure postmortem must contain an explicit CASE FOR REVIVAL — either
"none: the idea itself was disproven" or the specific evidence that would
justify reopening (enforced by the research step's validator). On any
reopen appeal, the critic must address that recorded case directly. This
keeps the 3-strike brake tight while guaranteeing a genuinely promising
idea gets its hearing on the record, never by the agent's own conviction.
