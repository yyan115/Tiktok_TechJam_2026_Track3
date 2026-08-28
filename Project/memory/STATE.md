# STATE — read this first in every session

Updated: 2026-08-28 ~09:15 (setup finished, committed; user at work)

## Where things stand
- **Referee (trusted runner): v0.9.3-unfrozen, ready for the user's freeze approval.** Full review history: Sol rejected v0.9.0 (RULE_VIOLATION, real flaws) → hardened v0.9.1 → Sol PASS "sound to freeze for shapes 1–13" → Sol's 2 minors applied (v0.9.2) → codex deep handoff review found more real issues (input-tampering hole, provenance gaps) → fixed in v0.9.3. Cold-start handoff simulation PASSED (fresh agent reconstructed everything from files alone).
- **Current champion: k001_sdpa, 1.614x on shape 1** (FP32 primary profile, all tripwires clean). Red-team suite (rt01 tamper / rt02 cache) caught under v0.9.3, runs on scratch ledgers.
- Both repos committed and pushed on branch `initial-architecture` (this repo + ../Tiktok_TechJam_2026_Track2, whose setup is also complete: baselines reproduced, iteration harness working, iteration 1 journaled at valid primary 0.6015).
- Read RUNBOOK.md for commands. Raw audit logs are gitignored (private).

## User's next steps (also in TEMP-PROGRESS-LOG.md at repo root)
1. Follow Project/audits/freeze_checklist.md IN ORDER (paste 2 lines → restart → verify → "freeze approved").
2. Say "grind" → Track 3 optimization on shapes 1–13 begins.
3. Say "go track 2" → its harness gets a Sol review, then the autonomous run.
4. This weekend: RunPod account for shape 14 + final numbers; check Devpost registration (window opens 29 Aug 12:00 GMT+8, closes 1 Sep 12:00).

## Standing rules (never violate)
1. Never edit: official scripts, README.md, shapes.json, manifest.json, Project/results/** (runner-written only), .claude/**. After freeze: Project/harness/.
2. All benchmarks via the runner with a shape id. No raw dials. Calibrate before comparing.
3. Promotion: correctness + threshold + all cross-checks clean + primary profile + current runner sha. Sol audits at checkpoints only; JUDGE_ERROR never blocks.
4. Never modify the repo during an active external review; bind reviews to a committed sha.
5. Plain language to the user; explicit "go" before repo actions.

## Work queue (after user's "grind")
- Per shape 1–13, worst-first: calibrate → k001 sweep → CUDA graphs whole-stack candidate → internal fp16/bf16 (vs FP32 reference) → Triton fused kernels → torch.compile comparison. Fresh web research per technique.
- Watch: shape 6 (batch 10000) may OOM in fp32 on 8 GB — record it. Shapes 7/11 (head size 8) → custom-kernel edge.
- Stage 4 (shape 14): chunked oracle as user-approved amendment + re-audit; rented GPU (user).
- Stage 5: sanctioned-copy official acceptance runs (runner `official` subcommand still unbuilt — do before final sweep).
- Packaging (day 3): tech report from DECISIONS/JOURNAL, README swap (user applies), 3-min video script, Devpost.
