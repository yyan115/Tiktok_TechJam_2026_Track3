# STATE — read this first in every session

Updated: 2026-08-28 ~13:00 (v1.0.2 — review loop CLOSED: round-6 verdict YES)

## Where things stand
- **Referee: v1.0.2, sha-PINNED in manifest.json (freeze candidate).** EVERY subcommand (measuring AND reporting) verifies the pin before producing output; under the cooperative trust model drift is self-defeating, and the absolute guarantee is external (git + manifest re-verification). Review history: Sol rejected v0.9.0 → Sol PASS (v0.9.2) → codex 14-finding review → v0.9.3 → codex confirmation (3 blockers) → v1.0.0 → codex round 3 (4 defects: freeze wording, guard holes, calibration-key gaps, stale state) → v1.0.1 (manifest pin) → codex round 4 (3 blockers: reporting subcommands bypassed the pin; /tmp-exemption + abbreviated-option guard holes; write-surface wording) → **v1.0.2** (all subcommands gated, tokenizing rm guard, precise write-surface documentation).
- Current champion (re-validated under v1.0.2): k001_sdpa on shape 1, FP32 primary profile, 1.612x — see Project/results/LEADERBOARD.md; both red-team attacks re-verified caught under v1.0.2 with durable committed evidence in Project/audits/redteam_v1.0.2/.
- Both repos on branch `initial-architecture`, pushed. Track 2: lab bench v0.2.0 rebuilt after its own codex round 1 (8 findings), re-review pending.
- Freeze-candidate commits: 7ad64de → 81e077b → 69d8e3f → d46d911 → this closing doc-polish commit. The AUTHORITATIVE frozen-commit pointer is the bottom line of Project/audits/freeze_checklist.md. Codex round-6 verdict: YES (none load-bearing remaining); preserved in Project/audits/track3_handoff_verdict_round6.md.

## User's next steps → TEMP-PROGRESS-LOG.md (repo root), then Project/audits/freeze_checklist.md
Short version: paste 2 deny lines → restart → verify locks → "freeze approved" → "grind" → "go track 2".

## Standing rules (never violate)
1. Never edit: official scripts, README.md, shapes.json, manifest.json, Project/results/** (runner-written only), .claude/**, and Project/harness/** (freeze candidate — treat as locked now).
2. All benchmarks via the runner with a shape id; calibrate before comparing; ONE runner process at a time.
3. Champions: promoted + pinned-runner sha + latest-calibration environment key + above the LATEST calibration's threshold. Sol/codex at checkpoints; JUDGE_ERROR never blocks.
4. Never modify the repo during an active external review; reviews bind to a committed sha.
5. Plain language to the user; explicit "go" before repo actions.
6. Memory files: split any that pass ~200 lines (Aug-2026 practice; see memory-system research note).

## Work queue (after user's "grind")
- Shapes 1–13 worst-first: calibrate → k001 sweep → CUDA-graphs whole-stack candidate → internal fp16/bf16 vs FP32 reference → Triton fused kernels → torch.compile comparison. Fresh web research per technique.
- Watch: shape 6 (batch 10000) may OOM in fp32 on 8 GB — record it. Shapes 7/11 (head dim 8) → custom-kernel edge.
- Stage 4+5 amendments BUNDLED (cold-start drill agent's suggestion, adopted 28 Aug): the shape-14 chunked oracle AND the `official` acceptance subcommand go into ONE user-approved re-freeze cycle (single pin update + re-validation + re-audit) instead of two. Rented GPU needed for shape 14 (user).
- Packaging: tech report from DECISIONS/JOURNAL, README swap (user applies), 3-min video script, Devpost. Submission window 29 Aug 12:00 → 1 Sep 12:00 GMT+8.
