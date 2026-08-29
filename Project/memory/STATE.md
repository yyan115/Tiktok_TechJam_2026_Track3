# STATE — read this first in every session

Updated: 2026-08-29 13:55 — DAY 2 (user's numbering). Usage-limit outage ~07:00-13:40 swallowed the morning. MODE: harness-hardening + research ONLY until the user restarts optimization ("get proper harness up before we start again").

## TIMELINE (user-corrected, binding)
Day 2 = 29 Aug afternoon onward. CLOCKS NOW TIGHT: organizer-question cutoff 30 Aug 12:00 SGT (~22h); rental 30 Aug; code freeze 31 Aug 12:00; packaging 31 Aug noon → 1 Sep 02:00; final 10h contingency; submission+registration close 1 Sep 12:00 GMT+8.
USER-GATED ITEMS OUTSTANDING (do these first, five minutes total): send Project/drafts/organizer_questions.md · reserve 48GB rental for 30 Aug · verify Devpost registration · apply Project/loop/OWNER_PATCH_card_gate.md (the mechanical card gate you asked for).

## FIRST ACTIONS FOR A FRESH SESSION (in order)
1. Locks test (Edit on torch_transformer_benchmark.py AND Project/harness/runner.py MUST bounce — verified again 28 Aug 19:47). Guard etiquette: never put 'clean'/'reset'/'restore' after 'git' in one command segment.
2. Check the audit ledger tally (Project/audits/verdicts.jsonl) — the 28 Aug ~20:00 champion wave (24 new champions) audits asynchronously; zero-byte logs in audits/auto/ = audit died with its session, re-fire by re-running that champion.
3. NEVER benchmark while codex audits run (`pgrep -f "codex exec"` must be empty) — contention INFLATES graphed-candidate ratios (LESSONS #19).
4. Resume the lever queue below on branch `grind-day1`.

## SCOREBOARD — SHIP-GRADE (29 Aug 02:30 QUIET-BOX pass, load 1.6 evidenced; FP32 primary, RTX 3060 Ti; Stage-5 measurement obligation DISCHARGED)
k009 (@ a5d525f): 1: 11.15x · 2: 14.98x · 3: 14.67x · 4: 7.93x · 5: 10.79x · 7: **21.50x** · 9: 7.24x · 10: 9.60x · 11: 14.64x · 12: 10.38x · 13: **29.34x**; k010 (@ 8157558) shape 8: 2.13x — **geomean ≈ 11.0x**. All 12 correct+promoted, tripwires clean, committed bytes, quiet box.
BOTH AUDITOR RETESTS SATISFIED: shape 2 strict recipe (w20/r1000/x5) = 13.24x stable; shape 12 baseline wall/event agreement 4.8% (<10% required), no suspicious flags.
Contention lesson refined: load cuts BOTH ways per shape (9/10 were better idle, 4/12 worse) — only quiet-box numbers are comparable.
**SUBMISSION ARTIFACT BUILT + END-TO-END GREEN (@ 1bb6e63)**: Project/submission/torch_transformer_benchmark_submission.py = official script with ONLY the UserOptimizedTransformer region replaced (build_submission.py proves outside-region bytes identical); the UNTOUCHED official code paths report PASS + 12.88x (shape-3 dials), PASS + 1.84x (shape-8 dials), PASS + 13.07x (shape-11 dials). Regenerate after any dispatcher change.
Two auditor RETESTs (k007-era shapes 2/12: idle-box, beefed recipe) are SUPERSEDED in target but not in spirit — apply their recipes to the current k009 champions during the idle re-pass.
**SHAPE-14 CORE PROVEN**: k006 kernel at seq=100k causal vs chunked fp32 oracle — 0 violations, max err 6.99e-4, 305 MiB (Project/tools/smokes/shape14_core_smoke.py). **SHAPE-6 CORE PROVEN**: k007 full B=10000 vs batch-chunked official baseline — 0 violations, 3.4 GiB (shape6_core_smoke.py). Full-scale timing for both = rental day.

## AUDIT LEDGER interpretation
Historic: 15 PASS · 10 RULE_VIOLATION on ORIGINAL k004 = provenance only (superseded by self-contained re-sweeps). Transition-window pair decoded 28 Aug evening: 193139 NEEDS_CONTEXT = packet carried post-edit source (answered: measured bytes now committed pre-run, always). 193243 RULE_VIOLATION = real minor findings, BOTH FIXED @ 90f1c8c: exact official forward signature (extra training=False removed from k004+k005) + padded-mask key masking in k005 (verified 24/24 vs baseline, scratchpad/padded_mask_smoke.py — smoke EVERY branch before runner time, LESSONS #17). 193545 = PASS (k004 self-contained).

## STATUS ~06:50 — RESEARCH-FIRST LOOP ERA (user-mandated ~03:30 after catching build-measure-retry cadence)
- User post-mortem verdicts stand: the harness audited honesty, never strategy; research corpus went unused; norms decayed. BINDING NEW PROCESS: research → proposal → blind Sol-ULTRA critique → revise → converge BEFORE implementation (memory: research-first-loop-directive). Research base now lives in Project/research/ (8 source-of-truth notes; check INDEX before any research/build).
- Blind strategy review (user-voiced, no Claude identity): R1 REVISE(8) — killed my shape-2 single-CTA play by arithmetic, exposed the shape-14 evaluator as non-runnable (~600 GiB), reclassified shape 6 (piggyback-only rental), found the REAL rubric in README (35/20/20/15/10 — 65% non-technical!). R2 REVISE(6) — notes rewritten in place, critic rules hardened. R3 "narrow REVISE, strategy CONVERGED"(2) — family-wide review pause + Card-1 evidence contract, both applied in DRAFT 4. R4 convergence check IN FLIGHT.
- SUPERSEDED: amendment_v1.1_bundle/_code docs (runner is NOT edited; shape-14 gets an independently pinned side evaluator + Project/results_side/ packets; at most one consolidated re-freeze late, only if organizers force integration). day2_plan.md superseded by harness_v2_proposal.md's allocation.
- CUDA C++ toolchain LIVE (user installed gcc15; probe verified). Sol audits at effort HIGH (champion audits) / ULTRA (strategy).
- Night experiments ledger: k010 ADOPTED (shape 8 → 2.13x); k008 int8 NEGATIVE; k011 NEGATIVE; k012 TIE (not adopted); k013 deleted untested.

## ON CONVERGENCE (R4 APPROVE or minor): implement in this order (harness_v2_proposal.md §4)
0. Owner sends Project/drafts/organizer_questions.md (cutoff 30 Aug 12:00 SGT); rental reserved for 30 Aug; Devpost registration verified.
1. Card 1: shape-14 shippable path (FA2-style authored attention + pinned side evaluator + evidence packet contract).
2. Card 2: shape-6 no-graph local candidate-only MFU. 3. Card 3: shape-8 chunked fp16-acc (plain-GEMM falsifier first, 2-3h cap). 4. At most one of 11/13 (tie-break: 11). Cut the rest unless everything green.
CODE FREEZE 31 Aug 12:00 SGT · packaging 31 Aug noon → 1 Sep 02:00 · final 10h = reproduction/contingency only.

## Standing rules (unchanged)
Never touch frozen/protected files (locks enforce). All benchmarks via the pinned runner + shape id. ONE runner process; NO other GPU work during sweeps; sequential calls; idle box (no codex). Champions auto-audit. Commit candidate bytes BEFORE first runner contact. Reflection after each block. Plain language; the user's stop overrides everything.

## Packaging (final day)
Merge to main before submitting · judge-facing READMEs (user applies) · TEMP files out of judge path · T3 video opens with TAMPER demo · T2 packages first · final ~8h protected · Devpost registration AND submission close 1 Sep 12:00 noon.
