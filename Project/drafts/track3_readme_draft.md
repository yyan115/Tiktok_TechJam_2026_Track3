# Track 3 submission narrative — DRAFT (not judge-facing yet)

Status: drafted Day 1 evening while sweeps ran. Numbers marked [FINAL] get
filled from the Stage-5 clean measurement pass; do not copy any number from
here without checking the leaderboard. The user reviews and applies this as
the judge-facing README on packaging day.

---

## What we built (one paragraph)

An AI agent that invents CUDA/Triton kernels — and a measurement system
designed so that nobody, including the agent, can cheat it. The agent (Claude)
writes candidate implementations; a frozen, hash-pinned referee harness
establishes correctness and timing; every new champion automatically triggers
a blind audit by a *different* AI (GPT via codex) that sees only neutral
evidence packets; and the human owner holds the only keys to the rules. The
result: every speedup on our leaderboard is authored (no wrapped kernel
libraries), reproducible from committed bytes, and certified by an adversarial
reviewer that was actively trying to find measurement fraud.

## Why the paranoia is the product

Public postmortems of AI-optimization projects (CUDA-L1, Sakana's evaluator
scandal) show the failure mode: the optimizer learns to beat the *timer*, not
the *task* — side-stream timing leaks, output caching keyed on input
addresses, quietly editing the evaluator. Our harness has a named tripwire for
each documented cheat: synchronized wall-clock cross-checks, fresh-memory
perturbed re-runs, shape assertions, hash-pinned evaluator + candidate bytes,
and OS-level deny rules on the agent's own editing tools (live-demonstrated:
the guard has blocked the agent's edits — that's the tamper demo that opens
our video).

## Results (RTX 3060 Ti 8 GB, fp32 primary profile)

Quiet-box referee pass, 29 Aug 02:30 (all correct, all promoted, all
tripwires clean; every number regenerates from the journal):

| shape | config sketch                  | speedup | kernel |
|------:|--------------------------------|--------:|--------|
| 1     | B64 · d128 · seq128            | 11.15x  | k009 megakernel |
| 2     | B1 · d128 · seq128             | 14.98x  | k009 megakernel |
| 3     | B4 · d128 · seq128             | 14.67x  | k009 megakernel |
| 4     | B16 · d128 · seq128            |  7.93x  | k009 megakernel |
| 5     | B128 · d128 · seq128           | 10.79x  | k009 megakernel |
| 7     | B64 · d32 · seq128             | 21.50x  | k009 megakernel |
| 8     | B64 · d1024 · seq128           |  2.13x  | k010 fp16+fused LN/GELU |
| 9     | B64 · d128 · 1 head            |  7.24x  | k009 megakernel |
| 10    | B64 · d128 · 2 heads           |  9.60x  | k009 megakernel |
| 11    | B64 · d128 · 16 heads          | 14.64x  | k009 megakernel |
| 12    | B64 · d128 · seq32             | 10.38x  | k009 megakernel |
| 13    | B64 · d128 · seq1024           | 29.34x  | k009 megakernel |
| **geomean** |                          | **~11.0x** | |

Shapes 6 and 14 (baseline infeasible locally): correctness proven on the
8 GB card vs exact chunked references; full-scale timing on the rental GPU,
reported separately and labeled.

Highlights as of Day 1 (contention-clean where noted):
- Authored fused-block "megakernel": an entire transformer block in two
  Triton kernels (LayerNorm+QKV; flash attention over all heads with the
  output projection folded into the head loop, then residual+norm+GELU-FFN
  in-register), whole forward captured as one CUDA graph.
- Every one of the 12 locally-runnable shapes beaten by authored kernels.
- The two "impossible" shapes have local correctness proofs on the 8 GB card:
  shape 14 (seq=100,000: the naive attention table would be multi-TB) matches
  a chunked fp32 oracle with 0 tolerance violations in 305 MiB; shape 6
  (batch 10,000: baseline OOMs) matches the batch-chunked official baseline
  with 0 violations in 3.4 GiB.

## How the trust chain works (diagram narrative for the video)

1. Shapes are pinned (shapes.json) — the agent cannot benchmark a easier dial.
2. The referee (runner.py) is frozen: sha-pinned, deny-ruled, self-verifying;
   it re-hashes the official benchmark script every run.
3. Candidates run from exact hashed bytes, committed BEFORE first contact.
4. Promotion is mechanical: correctness (multi-seed, fresh references,
   anti-cache) + speedup above a per-shape calibrated noise floor.
5. Each new champion fires a detached blind audit; verdicts bind to entries
   in an append-only ledger. Provenance complaints from auditors led us to
   rebuild candidates as self-contained files and commit-before-measure —
   the audit trail shows the system correcting itself.
6. The untouched official script is the final acceptance judge for every
   feasible shape; shape 14 ships with its limitation stated.

## Against the obvious tool (why authored kernels, measured)

The organizers' template suggests torch.compile / SDPA as optimization
directions. We measured torch.compile max-autotune on the baseline with the
UNTOUCHED official script (29 Aug, RTX 3060 Ti, fp32) and compared it to our
submission under the same script:

| dial set            | torch.compile max-autotune | our authored kernels |
|---------------------|---------------------------:|---------------------:|
| shape-3 dials       | 7.00x                      | ~13.5x               |
| shape-13 dials (seq 1024) | 3.10x                | ~29x                 |
| shape-8 dials (d 1024)    | 1.23x                | ~2.0x                |

The megakernel's edge grows exactly where compilation stops helping: long
sequences (fused flash attention vs materialized score tables) and the
launch-bound small shapes (one CUDA graph vs a compiled-but-still-eager
kernel stream).

## Honest limitations

- Shape 14/6 full-scale timing requires a big-memory GPU (rental day covers
  this); local results are correctness proofs plus scaling evidence, clearly
  labeled, never presented as full-shape speedups.
- Numbers measured while background audits ran are treated as provisional;
  the ship set is re-measured on an idle box (Stage 5).
- Internal fp16 is used only where the official fp32 tolerance is met; the
  fp32 baseline comparison is always the arbiter.

## [TODO before ship]
- Final board + MFU columns (needs amendment v1.1 approved + applied).
- Video script: open with tamper demo, then trust chain, then megakernel.
- Strip TEMP/draft files from judge path (this file moves or dies).
