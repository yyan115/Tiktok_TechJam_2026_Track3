# STATE — read this first in every session

Updated: 2026-08-28 20:02 (Day 1 CONTINUES — user's correction: ~4 days total, 28 Aug = Day 1, days never "close"; work 24/7 until exhausted or told stop)

## TIMELINE (user-corrected, binding)
Day 1 = 28 Aug (today, ongoing) · Days 2-3 = grind + rental (48-80 GB card for shapes 6+14, re-tune, MFU) · Final day = packaging; submission closes 1 Sep 12:00 GMT+8. There is no "closed" day — continuous work, only the user's stop ends a day.

## FIRST ACTIONS FOR A FRESH SESSION (in order)
1. Locks test (Edit on torch_transformer_benchmark.py AND Project/harness/runner.py MUST bounce — verified again 28 Aug 19:47). Guard etiquette: never put 'clean'/'reset'/'restore' after 'git' in one command segment.
2. Check the audit ledger tally (Project/audits/verdicts.jsonl) — the 28 Aug ~20:00 champion wave (24 new champions) audits asynchronously; zero-byte logs in audits/auto/ = audit died with its session, re-fire by re-running that champion.
3. NEVER benchmark while codex audits run (`pgrep -f "codex exec"` must be empty) — contention INFLATES graphed-candidate ratios (LESSONS #19).
4. Resume the lever queue below on branch `grind-day1`.

## SCOREBOARD (29 Aug ~01:40 — k009 era; FP32 primary, RTX 3060 Ti; mostly under audit load, Stage-5 idle re-pass owed)
k009 (@ a5d525f) champions: 1: 9.91x · 2: 17.05x · 3: 15.77x · 4: 11.65x · 5: 10.65x · 7: **26.46x** · 9: 5.63x · 10: 7.89x · 11: 15.33x · 12: 15.03x · 13: **29.12x**; k006 (@ d0341e5) shape 8: 1.79x — geomean ≈ **11.4x**. All 12 correct+promoted, tripwires clean, committed bytes.
**SUBMISSION ARTIFACT BUILT + END-TO-END GREEN (@ 1bb6e63)**: Project/submission/torch_transformer_benchmark_submission.py = official script with ONLY the UserOptimizedTransformer region replaced (build_submission.py proves outside-region bytes identical); the UNTOUCHED official code paths report PASS + 12.88x (shape-3 dials), PASS + 1.84x (shape-8 dials), PASS + 13.07x (shape-11 dials). Regenerate after any dispatcher change.
Two auditor RETESTs (k007-era shapes 2/12: idle-box, beefed recipe) are SUPERSEDED in target but not in spirit — apply their recipes to the current k009 champions during the idle re-pass.
**SHAPE-14 CORE PROVEN**: k006 kernel at seq=100k causal vs chunked fp32 oracle — 0 violations, max err 6.99e-4, 305 MiB (Project/tools/smokes/shape14_core_smoke.py). **SHAPE-6 CORE PROVEN**: k007 full B=10000 vs batch-chunked official baseline — 0 violations, 3.4 GiB (shape6_core_smoke.py). Full-scale timing for both = rental day.

## AUDIT LEDGER interpretation
Historic: 15 PASS · 10 RULE_VIOLATION on ORIGINAL k004 = provenance only (superseded by self-contained re-sweeps). Transition-window pair decoded 28 Aug evening: 193139 NEEDS_CONTEXT = packet carried post-edit source (answered: measured bytes now committed pre-run, always). 193243 RULE_VIOLATION = real minor findings, BOTH FIXED @ 90f1c8c: exact official forward signature (extra training=False removed from k004+k005) + padded-mask key masking in k005 (verified 24/24 vs baseline, scratchpad/padded_mask_smoke.py — smoke EVERY branch before runner time, LESSONS #17). 193545 = PASS (k004 self-contained).

## LEVER QUEUE (user's standing order = keep going)
1. PROMOTIONS RUNNING (user's "keep going", ~21:00): k007 on 1-5,7,9-13 + k006 on 8, production ledger, UNDER audit load — justified because PLAN Stage 5 mandates a final clean measurement pass for the ship set anyway; contention-era promotions are intermediate. FINAL-PASS OBLIGATION: re-measure the entire ship set on a provably idle box before consolidation, and treat any shape whose margin is thin (shape 2: k007 13.1 vs k005 11.9 screening) as undecided until then.
2. k007 design: whole block = 2 authored Triton kernels (norm+QKV | flash-attn-all-heads with out-proj folded into the head loop + norm2 + erf-GELU FFN), fp32 residuals, fp16 dots, CUDA-graphed. Screening ran under 5 codex audits — deltas far exceed contention noise but ship numbers = clean re-runs only.
3. Shape 8 (d=1024) is the one k007 can't cover (register budget) — k006's 1.82x stands; possible later lever: norm/GELU epilogue fusion around cuBLAS fp16 GEMMs (~10-20% upside, research 28 Aug).
4. Amendment bundle: DRAFTED @ Project/amendments/amendment_v1.1_bundle.md (MFU + official subcommand + shape-14 oracle) — needs user review + formal re-freeze (TIMEBOXED 1-2 rounds).
5. Rental day (48-80 GB) for shapes 6+14 (shape 6 baseline OOM on 8GB CONFIRMED empirically); re-tune there; MFU numbers.

## Standing rules (unchanged)
Never touch frozen/protected files (locks enforce). All benchmarks via the pinned runner + shape id. ONE runner process; NO other GPU work during sweeps; sequential calls; idle box (no codex). Champions auto-audit. Commit candidate bytes BEFORE first runner contact. Reflection after each block. Plain language; the user's stop overrides everything.

## Packaging (final day)
Merge to main before submitting · judge-facing READMEs (user applies) · TEMP files out of judge path · T3 video opens with TAMPER demo · T2 packages first · final ~8h protected · Devpost registration AND submission close 1 Sep 12:00 noon.
