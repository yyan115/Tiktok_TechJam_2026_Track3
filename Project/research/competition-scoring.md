# Competition scoring — source of truth (REWRITTEN 29 Aug ~14:05 from the full webinar transcript, MEETING-NOTES2.md)

## Confirmed by the organizer (Haoda, Track-3 owner) — quotes paraphrased tight
- Score (technical execution) = WEIGHTED SUM OF MFUs per test case; every
  shape must pass precision (rel<=0.02 OR abs<=0.002) or scores ZERO.
- Weights: NOT FIXED YET ("about the weights, I'm considering"); bandwidth
  will be taken into account; hardware limitations DISCLOSED IN THE TECH
  REPORT are taken into account. => the sensitivity board hedge is the
  right posture; asking for exact weights is unnecessary.
- JUDGES DO NOT RERUN ANYTHING ("judges will not rerun your script...we
  will not re-run the benchmark"). Scoring is from the submission:
  source code + README (how to run) + SKILLS used to guide the agent +
  TECH REPORT (device, memory/bandwidth, AI tools + models used, how best
  performance was reached) + INTERACTION HISTORY samples.
  => our loop artifacts (cards, lineage, audit ledger, evidence packets,
  research base) are literally the scored deliverable class.
- SINGLE GPU TYPE for the submitted result ("just for a single type...if
  you have multiple devices, choose one"). Own machine strongly preferred;
  industrial chips discouraged (H200: "too much open source...we need you
  to implement and optimize yourself"; supercomputers: "we will see if you
  have some innovations").
  => RENTAL IS CONTRA-INDICATED. The result device = RTX 3060 Ti.
- Shape 14: participants are EXPECTED to "divide the computation into
  several blocks and run it one by one" on their OWN device; "finish the
  computation on your own device and submit the tech report". For the
  baseline, ORGANIZERS WILL PROVIDE INPUT/OUTPUT PAIRS (released "maybe at
  the final") to compare against.
  => our microchunked local path is the intended solution shape; the
  validated streamed oracle referees until their pairs arrive, then their
  pairs are the final check.
- One framework suffices (torch OR tf — same math). Any language
  (Python/Triton/CUDA/lower). Any flags, only precision matters.
  Quantization internally is fine — only input/output precision counted.
  Compilation time and first run EXCLUDED; warmup ~20 typical.
- The updated PyTorch script's default thresholds equal the problem
  statement (rel 0.02 / abs 0.002) — matches our pinned copy's defaults.

## Still genuinely unknown (hedged, not asked — user's call 29 Aug)
- Exact per-shape weights (organizer himself undecided) → sensitivity board.
- MFU denominator convention → report multiple conventions + state device
  peaks and bandwidth explicitly in the tech report (organizer: disclosed
  limitations are considered).

## Plan consequences (critic-reviewed before action — changed-score-facts trigger)
1. KILL THE RENTAL: single-GPU-type rule + own-machine spirit + expected
   block-decomposition => shape-14 full B=32 evidence on the 3060 Ti via
   host-streamed batch microchunks (input staged/generated per-slice;
   candidate-only timing per slice, aggregated with method stated).
2. Tech report + skills + interaction history are FIRST-CLASS deliverables
   (not an afterthought): the trust-harness/loop story is directly scored.
3. organizer_questions.md: SUPERSEDED for items 1-5,7 (answered above);
   user chose not to ask anything non-essential.
