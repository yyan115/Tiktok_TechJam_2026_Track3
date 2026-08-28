# Outer-loop proposal, DRAFT 2 — post external review round 1

29 Aug ~05:15. Draft 1 received VERDICT: REVISE with 8 required revisions
(Project/audits/strategy/strategy_review1_raw.log). Draft 2 responds to every
one; deltas and residual disagreements are marked. Still NO implementation
until this converges + owner approves.

## Response to the review's re-diagnosis (accepted)

The reviewer's root causes replace my F1-F4 framing: (1) no trustworthy
score model; (2) false coverage confidence on shape 14; (3) no outer-loop
allocation by expected points/hour; (4) greedy single-incumbent search;
(5) reviews without decision closure; (6) too much process infrastructure
for the deadline. I verified its two decisive checks myself: the judging
rubric in README.md (Technical Execution 35%, Innovation 20%, Impact 20%,
Feasibility 20%) and the shape-2 single-SM floor arithmetic (~137us floor
vs 144.4us current champion — my single-CTA play was impossible; killed,
recorded in research/megakernels-persistent.md).

## Revision 1 — lightweight experiment-card outer loop (replaces "harness v2 hooks")

ACCEPTED. No per-candidate research-header hook, no reflection-recency hook,
no 2-negatives trigger. Instead, one EXPERIMENT CARD per macro-direction
(JSONL in Project/loop/cards.jsonl), created BEFORE work starts, schema per
KernelAgent's reflexion records (research/kernelagent.md):
  {id, direction, hypothesis, cheapest_decisive_test, prediction(+arithmetic),
   budget_minutes, kill_criteria, status, expected_vs_observed, lessons,
   avoid_patterns, try_patterns, followups}
Config/tile variants inherit the card id. Card close REQUIRES
expected-vs-observed — reflection folded in, no separate artifact. Staged
falsification is the core discipline: hypothesis → cheapest decisive test
(often arithmetic or a 5-minute microbench) → implementation only if it
survives. (This discipline, applied earlier, kills k008 and k011 on paper.)

## Revision 2 — repair and prove the shape-14 path (hard blocker, top priority)

ACCEPTED IN FULL, with the reviewer's own repair list adopted:
(a) oracle path must BYPASS the dense Evaluation constructor (its invariance
probe would build ~600 GiB of scores); (b) batch/head-STREAMED oracle and
streamed comparison (no 64 GiB score chunks); (c) no CUDA graph at this
size; (d) batch-microchunked candidate execution into one preallocated
output; (e) size-aware warmup/repeats; (f) reduced-size FULL-MODEL
multi-seed checks first, then ONE full-scale run. Technique focus: ~94% of
shape-14 useful FLOPs are attention ⇒ authored FlashAttention-2-style
kernel (online softmax, causal tiling, query-block parallelism within a
head) is the primary lever, not GEMM work. The amendment code doc gets
rewritten to this spec before any re-freeze conversation.

## Revision 3 — score model: obtain or hedge

ACCEPTED. Two actions: (1) DRAFTED organizer question set (below) for the
owner to send at morning: exact weights; MFU precision/peak convention and
cap (our fp32-denominator numbers exceed 100% — shape 8 at 129% — so the
convention is not ordinary utilization); the exact bandwidth term; hardware
mixing rules; the shape-14 reference procedure. (2) Until answered, the
leaderboard gains a SENSITIVITY BOARD: per shape — raw latency, useful
TF/s, fp32-equivalent throughput, fp16-peak MFU, and roofline efficiency
(max of compute/memory SOL, KernelAgent's headline metric) — so any
convention the organizers pick is already computed.

## Revision 4 — allocation reordered (accepted, one hedge)

ADOPTED ORDER: 1) shape-14 survival+correctness+authored long-seq attention;
2) shape-6 candidate-only local MFU + correctness (it is d=128/~86% linear,
fits locally in 3.4 GiB — NOT big-d, my taxonomy was wrong); 3) shape-8
authored GEMM/epilogue (chunked fp16-acc per the error model in
research/gemm-epilogue-fusion.md; naive variant pre-killed); 4) shape-11
head-dim-8 specialization (same FLOPs as 9/10 but 0.96ms vs 0.61ms — the
clearest anomaly on the board); 5) shape-13 attention retune (~57%
attention); 6) ONE timeboxed sequence-persistent family for 3/4/12
(independent CTA per batch sequence, NO cross-CTA sync — reviewer's
suggestion, Triton-expressible); 7) shape 2 last, only under a changed
premise (fp16-acc or multi-CTA), explicitly high-risk.
HEDGE (disagreement, stated): rental is NOT cancelled for shape 6 — it is
CONDITIONAL on the organizers' answer about baseline-comparison requirements
at shapes 6/14. If candidate-only numbers suffice, no rental for 6.

## Revision 5 — selective multi-fidelity profiling

ACCEPTED with one retained cheap layer: ladder = correctness screen →
timing screen → torch-profiler kernel table on PROMOTED survivors only
(near-free, catches k011-class regressions) → NCU SELECTED metrics
(CudaForge's finding: full metric dumps degrade judge quality) only on the
actual hotspot of the current direction. Profiling always separate from
official timing; benchmark lock semantics stay (one runner, quiet box).

## Revision 6 — structured lineage instead of TRIED.md

ACCEPTED. Project/loop/lineage.jsonl: {strategy_id, parent_id, source_sha,
device, shape, exact_change, prediction, outcome(dist over rounds),
counters_summary, failure_scope, revisit_condition} — successes AND
neutrals recorded, not just negatives. Live diversity: maintain a small
top-K beam per direction (KernelAgent) rather than a single incumbent; the
strategy critic receives the beam, not one champion.

## Revision 7 — prior-art corrections + cuBLASLt demotion

DONE. All six corrections applied to Project/research/ (cuPilot inference,
CudaForge selected-metrics, TritonForge success-rate context, AMK precision
asymmetry, MPK scope/1.7x, AlphaEvolve structure) + KernelAgent note added
as the primary template (organizer-shown). "Deeper cuBLASLt epilogues"
REMOVED from the portfolio as a primary play (authored-kernel rule);
F.linear/cuBLAS stays only as the already-shipped baseline plumbing around
authored kernels, flagged for the organizers' innovation-policy question.

## Revision 8 — hard packaging buffer

ACCEPTED. Rubric reality: 35% technical / 65% innovation+impact+feasibility
+presentation. Hard-reserve: final 10 hours before the 1 Sep noon deadline
are packaging-only (video, README, reproduction paths, Devpost), and one
earlier 2-hour block for the video's tamper-demo rehearsal. No kernel work
inside the buffer, no exceptions without owner override.

## Strategy critic (redesigned per review)

Triggers: (a) opening a new expensive direction (= new experiment card with
budget > 60 min); (b) changed score facts (organizer answers, rubric
discoveries); (c) direction-level stall = card budget exhausted or two
consecutive PREDICTION ERRORS (not mere negatives — being wrong about WHY
matters, failed predictions are the signal). Review must return: explicit
accept/reject of the direction, alternatives considered, expected
points/hour, cheapest falsifier — and the loop RECORDS THE DECISION AND
RESPONSE (decision closure, the review's core process complaint).

## Integrity audits (reduced per review)

Audit each unique source SHA once + the final dispatcher; a champion
winning additional shapes with identical bytes does not re-fire. Effort
stays high; strategy reviews stay ultra.

## Organizer question draft (owner sends)

1. Per-shape weights in the score? 2. MFU denominator: which precision's
peak, and is it capped at 100%? 3. What exactly does "bandwidth considered"
mean — roofline-relative credit? 4. For shapes where the reference script
cannot run (6 OOM on 8 GB, 14 infeasible anywhere), what evidence is
accepted — candidate-only timing vs a validated alternative reference?
5. Is fp16-internal computation acceptable under the stated tolerance
(webinar said yes — confirming in writing)? 6. Any constraint on
implementation language or on torch built-ins as fallbacks?

## What implementation happens the moment this converges

1. Project/loop/ (cards.jsonl, lineage.jsonl, sensitivity board generator)
   — ~1 hour, the ONLY process code built.
2. Card 1: shape-14 evaluator repair + streamed oracle (rev 2 spec).
3. Card 2: shape-6 candidate-only local MFU path.
4. Card 3: shape-8 chunked fp16-acc GEMM (error model already written).
5. Cards 4+: allocation order above. CUDA C++ used where the card's
   cheapest-test says it wins (toolchain verified live).
