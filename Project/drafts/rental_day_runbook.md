# Rental-day runbook (shapes 6 + 14) — DRAFT, review before spending money

Goal: full-scale numbers for the two shapes the 8 GB card cannot judge —
shape 6 (baseline OOM) and shape 14 (baseline mathematically infeasible;
oracle path needs amendment v1.1 approved). Everything else ships from the
local board. Minimize rented hours: every step below is scripted and the
kernels are already correctness-proven locally (shape6/shape14 core smokes).

## Card choice

- Shape 6 needs the BASELINE to fit: ~2.6 GB score tensors x transient copies
  x batch 10000 — 24 GB is probably enough (A5000/3090/4090-24G), 48 GB
  (A6000/L40S) is safe. Shape 14 candidate-side fits anywhere; its chunked
  ORACLE in fp32 wants headroom — 48 GB covers both comfortably. 80 GB (A100)
  only if cheap that day.
- sm86/sm89 preferred (same tuning family as the 3060 Ti; LESSONS #2: retune
  everything regardless). Avoid Hopper-only images.

## Environment pinning (do FIRST, before any measurement)

1. git clone the repo at the submission branch; `git log -1` recorded.
2. Python 3.12+ venv; `pip install torch==2.12.* triton==3.7.*` (match local
   majors; record exact wheels). `python3 Project/harness/runner.py env` and
   `check` — hashes MUST pass before anything runs.
3. `nvidia-smi -q -d CLOCK,POWER` snapshot into the journal directory.
4. Add the rented GPU's fp32 peak to device_peaks.json (amendment v1.1) for
   MFU — from the spec sheet, cited.

## Measurement order (one runner process, idle box, sequential)

1. `calibrate` on shapes 6, 14-oracle, and 8 (noise floor on THIS card).
2. Shape 6: baseline feasibility check, then the current d<=128 champion
   (k009 as of 29 Aug — check LEADERBOARD) — batch 10000 = 1.28M tokens is
   deep-grid territory; if speedup disappoints, retune BLOCK_T/BLOCK_M, one
   committed tuning round max.
3. Shape 14 (SUPERSEDES the old amendment-v1.1 requirement, 29 Aug review):
   evidence comes from the INDEPENDENTLY PINNED side evaluator
   (Project/tools/shape14_eval.py per harness_v2_proposal Card 1) — the
   frozen runner is NOT edited. Sequence: streamed-oracle validation vs the
   official dense path at feasible lengths (done locally), then ONE
   full-scale B=32/S=100000 run producing the immutable evidence packet
   (shas, env, seeds, error stats, raw samples, peak memory) into
   Project/results_side/. Candidate = the FA2-style authored attention
   inside the big-d stack (Card 1), not plain k010.
4. Re-run the FULL 12-shape ship set once on the rented card as a secondary
   profile (cheap, ~10 min) — cross-device evidence for the narrative, never
   mixed with the primary RTX 3060 Ti board.
5. Copy journal + packets off the box BEFORE terminating the instance.

## Budget guard

Everything above fits in ~2-3 rented hours if the amendment landed first.
Hard rule: no exploratory tuning marathons on the meter — one retune round
per shape, then take the number.

## Preconditions checklist (do at home, free)

- [ ] Card-1 side evaluator built + oracle validated locally (replaces the old amendment-v1.1 precondition).
- [ ] device_peaks.json entries for candidate rental cards.
- [ ] Shape-6/14 smokes green at HEAD (they are, 28 Aug).
- [ ] Idle re-pass of the local board done (Stage-5 obligation + 2 RETESTs).
