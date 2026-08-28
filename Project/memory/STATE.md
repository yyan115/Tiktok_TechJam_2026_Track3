# STATE — read this first in every session

Updated: 2026-08-28 20:02 (Day 1 CONTINUES — user's correction: ~4 days total, 28 Aug = Day 1, days never "close"; work 24/7 until exhausted or told stop)

## TIMELINE (user-corrected, binding)
Day 1 = 28 Aug (today, ongoing) · Days 2-3 = grind + rental (48-80 GB card for shapes 6+14, re-tune, MFU) · Final day = packaging; submission closes 1 Sep 12:00 GMT+8. There is no "closed" day — continuous work, only the user's stop ends a day.

## FIRST ACTIONS FOR A FRESH SESSION (in order)
1. Locks test (Edit on torch_transformer_benchmark.py AND Project/harness/runner.py MUST bounce — verified again 28 Aug 19:47). Guard etiquette: never put 'clean'/'reset'/'restore' after 'git' in one command segment.
2. Check the audit ledger tally (Project/audits/verdicts.jsonl) — the 28 Aug ~20:00 champion wave (24 new champions) audits asynchronously; zero-byte logs in audits/auto/ = audit died with its session, re-fire by re-running that champion.
3. NEVER benchmark while codex audits run (`pgrep -f "codex exec"` must be empty) — contention INFLATES graphed-candidate ratios (LESSONS #19).
4. Resume the lever queue below on branch `grind-day1`.

## SCOREBOARD (clean idle-box champions, authored+self-contained, signature-exact, padded-path-fixed, committed bytes @ 90f1c8c; FP32 primary, RTX 3060 Ti, 28 Aug ~19:55 sweep)
Best per shape: 1: 3.34x (k005) · 2: **11.92x** (k005) · 3: 9.70x (k004) · 4: 4.21x (k005) · 5: 2.88x (k005) · 7: 5.06x (k005) · 8: 1.58x (k005) · 9: 1.42x (k005) · 10: 2.44x (k005) · 11: 7.36x (k005) · 12: 4.00x (k005) · 13: **11.49x** (k005) — geomean ≈ **4.3x**. All 24 runs correct+promoted.
Shape 9/10 question CLOSED: idle-box k004 = 1.14x/1.56x — the old 3.14x/4.09x were contention-inflated, not real. Ship numbers = this board.
**SHAPE-14 CORE PROVEN**: authored kernel at seq=100k causal vs chunked fp32 oracle — 0 violations, max err 5.3e-4, 337 MiB (scratchpad/shape14_core_smoke.py). 100k perf tuning = rental-day work.

## AUDIT LEDGER interpretation
Historic: 15 PASS · 10 RULE_VIOLATION on ORIGINAL k004 = provenance only (superseded by self-contained re-sweeps). Transition-window pair decoded 28 Aug evening: 193139 NEEDS_CONTEXT = packet carried post-edit source (answered: measured bytes now committed pre-run, always). 193243 RULE_VIOLATION = real minor findings, BOTH FIXED @ 90f1c8c: exact official forward signature (extra training=False removed from k004+k005) + padded-mask key masking in k005 (verified 24/24 vs baseline, scratchpad/padded_mask_smoke.py — smoke EVERY branch before runner time, LESSONS #17). 193545 = PASS (k004 self-contained).

## LEVER QUEUE (user's standing order = keep going)
1. **PROMOTIONS PENDING (idle box required)**: k007 fused-block megakernel (@ 8270909, smoke 42/42) SCREENED 2-4x ABOVE EVERY CHAMPION on the 11 d<=128 shapes — screening geomean ~11.7x (shape 7: 21.9x, shape 13: 28.3x, shape 5: 9.2x). Also k006 (@ d0341e5) screened ahead on shape 8 (1.82x vs 1.58x). Once `pgrep -f "codex exec"` is empty: production runs of k007 on 1-5,7,9-13 and k006 on 8.
2. k007 design: whole block = 2 authored Triton kernels (norm+QKV | flash-attn-all-heads with out-proj folded into the head loop + norm2 + erf-GELU FFN), fp32 residuals, fp16 dots, CUDA-graphed. Screening ran under 5 codex audits — deltas far exceed contention noise but ship numbers = clean re-runs only.
3. Shape 8 (d=1024) is the one k007 can't cover (register budget) — k006's 1.82x stands; possible later lever: norm/GELU epilogue fusion around cuBLAS fp16 GEMMs (~10-20% upside, research 28 Aug).
4. Amendment bundle: DRAFTED @ Project/amendments/amendment_v1.1_bundle.md (MFU + official subcommand + shape-14 oracle) — needs user review + formal re-freeze (TIMEBOXED 1-2 rounds).
5. Rental day (48-80 GB) for shapes 6+14 (shape 6 baseline OOM on 8GB CONFIRMED empirically); re-tune there; MFU numbers.

## Standing rules (unchanged)
Never touch frozen/protected files (locks enforce). All benchmarks via the pinned runner + shape id. ONE runner process; NO other GPU work during sweeps; sequential calls; idle box (no codex). Champions auto-audit. Commit candidate bytes BEFORE first runner contact. Reflection after each block. Plain language; the user's stop overrides everything.

## Packaging (final day)
Merge to main before submitting · judge-facing READMEs (user applies) · TEMP files out of judge path · T3 video opens with TAMPER demo · T2 packages first · final ~8h protected · Devpost registration AND submission close 1 Sep 12:00 noon.
