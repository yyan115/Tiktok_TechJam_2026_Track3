# STATE — read this first in every session

Updated: 2026-08-28 20:02 (Day 1 CONTINUES — user's correction: ~4 days total, 28 Aug = Day 1, days never "close"; work 24/7 until exhausted or told stop)

## TIMELINE (user-corrected, binding)
Day 1 = 28 Aug (today, ongoing) · Days 2-3 = grind + rental (48-80 GB card for shapes 6+14, re-tune, MFU) · Final day = packaging; submission closes 1 Sep 12:00 GMT+8. There is no "closed" day — continuous work, only the user's stop ends a day.

## FIRST ACTIONS FOR A FRESH SESSION (in order)
1. Locks test (Edit on torch_transformer_benchmark.py AND Project/harness/runner.py MUST bounce — verified again 28 Aug 19:47). Guard etiquette: never put 'clean'/'reset'/'restore' after 'git' in one command segment.
2. Check the audit ledger tally (Project/audits/verdicts.jsonl) — the 28 Aug ~20:00 champion wave (24 new champions) audits asynchronously; zero-byte logs in audits/auto/ = audit died with its session, re-fire by re-running that champion.
3. NEVER benchmark while codex audits run (`pgrep -f "codex exec"` must be empty) — contention INFLATES graphed-candidate ratios (LESSONS #19).
4. Resume the lever queue below on branch `grind-day1`.

## SCOREBOARD (28 Aug ~21:05 promotion sweep — k007 megakernel era; FP32 primary, RTX 3060 Ti; measured UNDER audit load, Stage-5 idle re-pass owed)
k007 (@ 8270909) champions: 1: 8.39x · 2: 11.92x* · 3: 12.18x · 4: 9.20x · 5: 9.24x · 7: **22.41x** · 9: 4.41x · 10: 6.63x · 11: 12.95x · 12: 13.65x · 13: **28.31x**; k006 (@ d0341e5) shape 8: 1.79x — geomean ≈ **9.6x** (was 4.3x). All 12 correct+promoted, tripwires clean, committed bytes.
(*) Shape 2 statistical tie with k005 (11.9242 vs 11.9176) — idle re-pass decides.
Idle-box k005/k004 fallback board (bytes @ 90f1c8c, ~19:55 sweep) remains valid as the conservative floor: geomean ≈ 4.3x.
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
