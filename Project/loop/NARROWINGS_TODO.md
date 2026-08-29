# Active work queue: score-facts critic narrowings (ITEM1/2/3 all "narrow", 29 Aug PM)
Apply IN ORDER before optimization restarts; each is harness/evidence work.
1. Rental kill = STRATEGIC CHOICE (not organizer prohibition); keep rental as
   contingency requiring a whole-board rerun on that GPU type. Purge rental
   claims from tech_report/README drafts; runbook gets contingency framing.
2. REWRITE shape14_eval.py eval path: never allocate full x/out (24.4 GiB at
   B=32) — generate deterministic B=1 slices from recorded (trial,batch-idx)
   seeds, run, compare per slice vs streamed oracle, reduce stats, discard.
   Validate batch-decomposition equivalence at reduced S first.
3. Replace torch.isclose with the OFFICIAL predicate (abs_ok OR rel_ok,
   torch_transformer_benchmark.py:312 semantics) everywhere in side tools.
4. Timing = "serial batch-decomposed GPU-compute time": per repeat run all 32
   slices, record the SUM; report median-of-sums + raw slice matrix + a
   staging-inclusive wall time. NEVER multiply a B=1 median (B=2 = 2x+9%).
5. All report prose/tables/skills/curated interactions ready BY CODE FREEZE
   (31 Aug noon); packaging window is assembly+video only.
6. Deliverables per transcript; loop artifacts CURATED as supporting
   evidence — do not claim they are "literally scored".
7. organizer_questions.md -> RETIRED/NO-SEND. Unresolved list (hedge in
   report): weights; MFU denominator+cap; bandwidth adjustment; zero-one-
   shape vs zero-everything; compute-only slice timing acceptance;
   built-ins policy; whether/when organizer I/O pairs appear; RUBRIC
   DUALITY (external audit — CORRECTED 30 Aug): the audit claimed Devpost
   rules = 4 equal criteria but never reconciled README §3.6, which is the
   OFFICIAL TRACK STATEMENT (per shapes.json provenance) and gives an
   explicit 5-way 35/20/20/15/10 table (Presentation 10% = final event
   only). The specific statement controls over generic site boilerplate:
   README rubric LEADS, Devpost framing mentioned as secondary.
8. Sensitivity board: recompute + SOURCE every denominator. 3060 Ti GA104:
   fp16-tensor-with-FP32-acc ~= 2x fp32 shader rate (~32.4 TF, what OUR
   kernels use); fp16-with-fp16-acc dense ~= 4x (~65 TF, absolute fp16
   roof). Label both, cite the GA10x whitepaper + spec page; report MFU
   against BOTH. (Shape-14 "80% of roof" becomes ~40% of the 65 TF roof —
   restate honestly everywhere.) ALSO (external audit): present scores
   under several weightings — equal-by-shape, FLOP-weighted, weighted-MFU
   variants, bandwidth-aware — so no optimization chases a number that
   contributes nothing under the real formula.
9. NEW (research adoption): cards gain required fields per KernelAgent/
   Baidu: diagnosis_evidence (the 6-field compressed profile record) +
   prescription_source (research-note/pattern citation). ncu confirmed
   installed; selected-metrics only, on the current direction's hotspot.
   PLUS (external audit P8, was MISSING here): score_scenarios — which of
   the item-8 weighting scenarios this experiment would actually improve;
   an experiment that helps under NO scenario doesn't get a card.

10. POST-RESTART (Sol dispositions, 29 Aug night; sharpened by the external
    audit): capture FRESH evidence packets for shapes 14 and 6 that EXECUTE
    THE GENERATED SUBMISSION FILE ITSELF (import UserOptimizedTransformer
    from torch_transformer_benchmark_submission.py — NOT the standalone
    k014/k015 kernel files; audit 7.3.1) — the existing packets cite the
    pre-integration submission. Packets MUST use >=5 deterministic seeds
    (the official script's --accuracy-trials default; shape-6 integrated
    smoke hit max err 0.00184 vs 0.002 atol — one seed is too thin a
    margin), the exact official predicate (item 3), and record the
    submission sha INSIDE the packet, and record peak RESERVED memory
    alongside peak allocated (audit P4 — reserved exposes fragmentation
    headroom on the 8GB card). Also:
    report prose + skills/interaction-history curation DONE BY CODE FREEZE,
    not left to packaging; dashboard is LOCAL-ONLY (never expose/record the
    raw Sol log expanders unredacted); public release = fresh/squashed only
    (the untracked transcript survives in git history at 49bfd47).

10b. AUDIT BACKLOG (reviewer round 3 discovery): 24 champions were marked
    audit-handled with NO verdict row (watcher cached before launch) — six
    are CURRENT champions. The rebuilt watcher now derives handled-ness
    from verdict rows + live process markers, so these REFIRE automatically
    on its next pass; run them on an idle box (contention rule) and check
    the tally reaches parity before freeze.

11a. Ship manifest verdict filter = codex task 06 (after task 03): hard
    verdicts exclude entries from selection unless verdict-clear'd; no
    silent shape drops; --report transparency flag.
11b. REPORT: label every pre-paste measurement PRE-GATE per the GATE_DESIGN
    HONESTY LEDGER; lead the report's process story with the Track-2
    violation -> diagnosis -> authority-v4 arc (judge-facing gold).
11c. AT FREEZE (CDC adoption): independent adversarial final review — 3-5
    read-only reviewers, ONE named defect class each (rules/eligibility;
    evidence-vs-artifact binding; report claims vs data; shape-6/14
    evidence; reproduction), independent reports merged BEFORE any change.

11. AT FREEZE (Sol dispositions item 3): fix ship_manifest.py THEN regenerate
    SHIP_MANIFEST.json from the final clean commit. Three generator bugs:
    (a) recorded git revision predates the submission sha it cites — record
    the revision AFTER the final commit; (b) shape-14 entry selection
    compares B=1 vs B=2 latencies (incomparable) — select by recency/route
    match, never cross-batch latency; (c) shape-6 labeled "oracle validated
    vs official dense" — it is validated vs the batch-chunked official
    baseline; label it exactly that. Loopback binding for the dashboard is
    DONE (both .streamlit configs, 29 Aug night).
    (d) MISSING UNTIL 29 Aug (external audit P9): rerun the COMPLETE
    official-script board — all 12 ordinary shapes at their official dials —
    against the FINAL frozen submission sha, plus the item-10 extreme
    packets at that same sha. One sha, one board, every number cites it.
    The historical 12-shape sweep predates the extreme-route integration
    and must be labeled historical in the report, never presented as a
    measurement of the final artifact. VARIANCE (found re-auditing the
    audit): shape 3 official-dials swung 11.961x -> 15.173x between two
    runs of the SAME route — tiny launch-bound shapes are noisy on the
    official script. Final board = best-of-N or median-of-N sweeps per
    small shape, N>=3, all N reported; the report states the noise.
