# Outer-loop proposal, DRAFT 3 — post review round 2

29 Aug ~06:00. Round 2 returned REVISE(6); every item is applied below.
Residual-disagreement resolutions accepted as reworded by the reviewer.
No implementation until convergence + owner approval.

## 0. Corrections of record (round-2 item 1) — DONE, not promised
- Research notes rewritten IN PLACE as source-of-truth (stale 6.7x framing,
  200 KB estimate, dead shape-2 play, "shape-14 = pure compute" all removed
  from live text; INDEX refreshed; KernelAgent indexed).
- Rubric corrected: Technical 35 / Innovation 20 / Impact 20 / Feasibility
  15 / Presentation 10 (final event only). 65% of the score is not raw
  kernel numbers.
- Operational: branch pushed (was 46 commits unbacked); untested k013
  leftover deleted; strategy prompts + verdict extracts committed, raw
  reviewer logs gitignored per LESSONS 14.

## 1. Experiment-card outer loop (binding rules, round-2 item 2)
Cards in Project/loop/cards.jsonl, schema as draft 2, PLUS binding rules:
- Budget counts CUMULATIVE DIRECTION-FAMILY spend (a family = card + its
  descendants/renames); >= 60 min of family spend requires a critic pass —
  splitting or renaming cards cannot evade it.
- Prediction = a PREREGISTERED quantitative range on a named metric in the
  card; a miss is decided by the recorded range, not judgment after the
  fact. TWO independent quantitative misses in a family (regardless of
  interleaving) => review_required.
- Hitting kill-criteria, requesting a budget extension, or reaching the
  organizer-answer cutoff each TRIGGER review_required on their own.
- review_required BLOCKS descendant cards until every critic recommendation
  is disposed: accept | reject-with-evidence | defer-until-condition (the
  condition recorded). Critic verdicts are continue | narrow | kill; killed
  directions reopen only on their recorded changed-premise or explicit
  owner override. All dispositions logged in the card (decision closure).
- Card creation is 10-15 minutes by hand; NO beam/process tooling beyond
  cards.jsonl + lineage.jsonl + the sensitivity-board generator script.
Lineage (Project/loop/lineage.jsonl) and a small top-K live set per
direction as in draft 2.

## 2. Score model (round-2 item 3)
- Organizer questions (owner sends TODAY): weights; MFU precision/peak
  convention + cap; exact bandwidth term; evidence accepted for shapes 6/14
  (candidate-only vs alternative reference); fp16-internal confirmation in
  writing; language/built-ins constraints; HARDWARE MIXING (may different
  shapes be reported from different machines, e.g. local 3060 Ti + rented
  card, and how are cross-device numbers normalized?).
- ANSWER CUTOFF: if no answer by 30 Aug 12:00 SGT (rental booking time),
  the loop defaults to: candidate-only MFU for 6/14, fp16-peak MFU +
  roofline-efficiency both reported, and the sensitivity board ships in the
  README so any convention is pre-computed. Cutoff reached = a critic
  trigger (score facts changed from "pending" to "assumed").
- SCORE SCENARIOS maintained in the sensitivity board: S1 equal-weight
  roofline-relative; S2 equal-weight absolute-MFU (fp16 peak); S3
  FLOP-weighted absolute-MFU. Allocation is re-checked against all three;
  the current order (below) is robust across them because shape 14 is the
  largest term in S2/S3 and a zero in none.

## 3. Card 1 = the complete shippable shape-14 path (round-2 item 4)
One card covers candidate + evaluator + acceptance, end to end:
- CANDIDATE: authored FA2-style long-sequence attention (online softmax,
  causal tiling, query-block parallelism per head) inside the existing
  big-d stack; batch-microchunked execution into ONE preallocated output;
  NO CUDA graph at this size; size-aware warmup/repeats.
- EVALUATOR: an INDEPENDENTLY PINNED special-shape evaluator tool
  (Project/tools/shape14_eval.py, own sha in manifest listing style), NOT a
  frozen-runner edit: streamed batch/head oracle, streamed comparison,
  never materializing more than a few GiB; reduced-size FULL-MODEL
  multi-seed checks first, then ONE full-scale run on the rental card.
- The frozen runner is not touched now. IF organizer answers force
  runner-integrated evidence, everything is prototyped first and exactly
  ONE consolidated re-freeze happens late (a runner edit retires every
  champion per RUNBOOK — that cost is paid at most once, deliberately).
- Sensitivity-board generator is an EXTERNAL script reading the journal —
  no runner edit needed for MFU reporting either.
- Dispatcher acceptance: the submission file's big-d route gains the FA2
  path behind the same exact-fallback guards; all-dials regression re-run.

## 4. Allocation and deadline sequence (round-2 items 5-6)
NOW (before any kernel work): organizer questions sent; rental RESERVED for
30 Aug (shape 14 primary; shape-6 dense-baseline comparison only appended
on the SAME instance if organizers require it — shape 6 never justifies a
standalone rental, larger card, delay, or extra tuning session); Devpost
registration verified; branch backup (done).
1. Card 1: shape-14 path (above) — the critical path.
2. Card 2: shape-6 candidate-only local MFU + correctness; FIRST prove the
   repeated NO-GRAPH timing route (graph capture at B=10000 is unproven for
   memory; graph savings are negligible at that size).
3. Card 3: shape-8 chunked fp16-acc — STARTS with the plain-GEMM falsifier
   (is authored Triton GEMM within ~10% of cuBLAS at M=8192, K=N=1024?);
   family cap 2-3 hours; sqrt(K)*u error model is a heuristic, so the
   referee's full-model tolerance check is the decider, not the model.
4. Card 4: at most ONE of shape 11 (head-dim-8 specialization) or shape 13
   (attention retune), selected by the then-current score scenario.
5. Sequence-persistent 3/4/12 and shape 2: CUT unless coverage, rental
   evidence, submission, README, and video are all green.
- Torch-profiler table: retained as a BOUNDED DIAGNOSTIC on promoted
  survivors (not a gate, no universal claims); NCU selected-metrics only on
  the current direction's hotspot.
- PRACTICAL CODE FREEZE: 31 Aug 12:00 SGT. FINAL BUFFER from 01 Sep 02:00
  SGT: clean-checkout reproduction, submission rebuild + all-dials
  regression, final dispatcher audit, video/upload/link verification,
  Devpost dry run. Owner override inside the buffer only for
  submission-blocking defects, never performance work.

## 5. Standing integrity mechanics (unchanged from draft 2)
Unique-source-SHA audits + final dispatcher audit; strategy critic at ultra,
cross-family, blind, neutral inputs; benchmark lock semantics (one runner,
quiet box, %CPU-verified).
