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
   built-ins policy; whether/when organizer I/O pairs appear.
8. Sensitivity board: recompute + SOURCE every denominator. 3060 Ti GA104:
   fp16-tensor-with-FP32-acc ~= 2x fp32 shader rate (~32.4 TF, what OUR
   kernels use); fp16-with-fp16-acc dense ~= 4x (~65 TF, absolute fp16
   roof). Label both, cite the GA10x whitepaper + spec page; report MFU
   against BOTH. (Shape-14 "80% of roof" becomes ~40% of the 65 TF roof —
   restate honestly everywhere.)
9. NEW (research adoption): cards gain required fields per KernelAgent/
   Baidu: diagnosis_evidence (the 6-field compressed profile record) +
   prescription_source (research-note/pattern citation). ncu confirmed
   installed; selected-metrics only, on the current direction's hotspot.
