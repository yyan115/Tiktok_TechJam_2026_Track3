# STATE — read this first in every session

## !! READ Project/HANDOVER.md FIRST — it is the single source of truth !!
30 Aug ~12:00. One file now holds current state, every open defect, and the
FIX -> LOCK -> GRIND plan. It supersedes TEMP-PROGRESS-LOG (deleted),
ROUND7_FINDINGS (merged), NARROWINGS_TODO, AUTHORITY_V4_PLAN and
harness_v2_proposal as the working todo.
The SCOREBOARD below is NOT safe to ship as written: all 12 backing rows
carry baselines 6-63% slower than their own calibration on identical
official code, so "quiet box" fails and the honest geomean is ~9-11x, not
11.0x. See HANDOVER.md 3.1.

Updated: 2026-08-30 ~05:30 — AUTHORITY v4 BUILT + OWNER-APPROVED (after the Track 2 convergence-override violation; full story: Project/loop/AUTHORITY_V4_PLAN.md, lesson 23). Gate hardened in run_gate.py (computed screening, hard-verdict brake, verdict-clear owner/mechanical paths; 22/22 sandbox tests). Awaiting ONE owner action: paste Project/loop/OWNER_PATCH_card_gate.md v4 into .claude/hooks/guard_bash.py (Block A + Block A2 after WRITE_PATTERNS defs, Block B at the END of main) AND add its listed deny lines to .claude/settings.json. Then the agent runs the 8 proof-tests and the caged grind restarts autonomously (shape-14 streamed evidence first). Until then: NARROWINGS/tooling work only, no optimization runs. NOTE: all measurements to date are PRE-GATE (see GATE_DESIGN HONESTY LEDGER); codex tasks 01-06 briefs in Project/audits/codex_tasks/.

## TIMELINE (binding)
CODE FREEZE 31 Aug 20:00 SGT (owner moved it from 12:00 on 30 Aug) -> packaging/report/video 31 Aug 20:00 -> 1 Sep 02:00 (SIX hours) -> final 10h reproduction/contingency -> submission+registration close 1 Sep 12:00 GMT+8.
CONSEQUENCE of the later freeze: packaging shrank 14h -> 6h, so report/README/video PROSE must be finished BEFORE the freeze; only numbers and assembly wait for it.
SUPERSEDED by the webinar transcript (research/competition-scoring.md): NO rental (single-GPU rule, own-machine; shape-14 = block decomposition locally). NO organizer questions (retired/no-send). Amendment v1.1 docs superseded (frozen runner untouched; side evaluator + results_side packets instead).
OWNER ITEMS: (1) the gate paste above; (2) verify Devpost registration. Nothing else.

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
0. RETIRED (webinar answered them): organizer questions NO-SEND; rental CANCELLED. Only surviving owner item here: verify Devpost registration.
1. Card 1: shape-14 shippable path (FA2-style authored attention + pinned side evaluator + evidence packet contract).
2. Card 2: shape-6 no-graph local candidate-only MFU. 3. Card 3: shape-8 chunked fp16-acc (plain-GEMM falsifier first, 2-3h cap). 4. At most one of 11/13 (tie-break: 11). Cut the rest unless everything green.
CODE FREEZE 31 Aug 20:00 SGT · packaging 31 Aug 20:00 → 1 Sep 02:00 · final 10h = reproduction/contingency only.

## Standing rules (unchanged)
Never touch frozen/protected files (locks enforce). All benchmarks via the pinned runner + shape id. ONE runner process; NO other GPU work during sweeps; sequential calls; idle box (no codex). Champions auto-audit. Commit candidate bytes BEFORE first runner contact. Reflection after each block. Plain language; the user's stop overrides everything.

## Packaging (final day)
Merge to main before submitting · judge-facing READMEs (user applies) · TEMP files out of judge path · T3 video opens with TAMPER demo · T2 packages first · final ~8h protected · Devpost registration AND submission close 1 Sep 12:00 noon.
