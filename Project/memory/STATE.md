# STATE — read this first in every session

Updated: 2026-08-29 03:15 (Day 1 grind, autonomous window until NOON 29 Aug — user returns then to review + plan Day 2 from Project/drafts/day2_plan.md)

## TIMELINE (user-corrected, binding)
Day 1 officially ends 12:00 noon 29 Aug (user's word, ~02:00). Days 2-3 = amendment re-freeze + rental (shapes 6+14, MFU) + polish · Final day = packaging; submission AND registration close 1 Sep 12:00 GMT+8. No "closed" days — continuous work, only the user's stop ends a day.

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

## STATUS AT ~03:15 (all measurement obligations DONE; steward mode until noon)
- Quiet-box ship board banked (above) + official grader passes ALL 12 dial sets (evidence: Project/drafts/official_grader_all_dials_20260829.txt, judge-side geomean ~10.3x).
- Audits now run Sol at effort HIGH (user-directed, fc6f015; global default was ultra). Quiet-box champion audits in flight — triage each wave; RULE_VIOLATION = loud, fix-first.
- Experiments closed tonight: k010 ADOPTED (shape 8 → 2.13x, in submission); k008 int8 NEGATIVE (tolerance); k011 NEGATIVE (traffic > occupancy); k012 TIE on shape 4 (not adopted).
- Etiquette refinements: pgrep counts dead audit wrappers — check per-process %CPU; user's browser adds ~30% of one core even "idle".

## USER GATES AT NOON (see Project/drafts/day2_plan.md)
1. Amendment v1.1 approval (amendment_v1.1_bundle.md + amendment_v1.1_code.md — insertion-ready, line-anchored) → unlocks shape-14 scoring + MFU.
2. Rental booking (drafts/rental_day_runbook.md turnkey; 48 GB class, ~2-3h, AFTER amendment).
3. MFU denominator convention (fp32 peak recommended).
Drafts awaiting review: README (with results + torch.compile tables), video script (with per-scene commands), day2_plan.

## Standing rules (unchanged)
Never touch frozen/protected files (locks enforce). All benchmarks via the pinned runner + shape id. ONE runner process; NO other GPU work during sweeps; sequential calls; idle box (no codex). Champions auto-audit. Commit candidate bytes BEFORE first runner contact. Reflection after each block. Plain language; the user's stop overrides everything.

## Packaging (final day)
Merge to main before submitting · judge-facing READMEs (user applies) · TEMP files out of judge path · T3 video opens with TAMPER demo · T2 packages first · final ~8h protected · Devpost registration AND submission close 1 Sep 12:00 noon.
