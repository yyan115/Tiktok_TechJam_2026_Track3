# Plan of Record — Hardened Cross-Reviewed Loop (approved 28 Aug 2026, incl. final lighter-loop amendments)

## The system in one line

Fable (Claude) invents faster implementations → a deterministic trusted runner establishes correctness and timing → checkpoint audits by Sol (GPT-5.6 via fresh `codex exec`, subscription only) → trusted champions ship → the user retains final authority.

## Ground rules

- **Trust model:** guards against *mistakes*, not malice. Git history + file hashes + hooks (block accidental edits) + runner tripwires (fresh-memory rerun, shape assertions, sync-timing cross-check) + Sol checkpoint review. No OS lockdowns.
- **The untouched official script is the final judge** (commit 31c1a27, re-hash-checked every run). The custom runner supplements it for development, never replaces it.
- **Promotion rule:** a correctness-passing improvement that beats the measured noise floor becomes the **working champion immediately**; its audit status is recorded separately. Every comparison-affecting setting (GPU, driver, CUDA, torch, dtype, TF32/matmul flags, code + benchmark hashes, harness version) lives in the recorded profile; only like-for-like profiles are compared.
- **Sol audits at checkpoints only** (never per promotion):
  1. Stage-1: runner + shapes.json + calibration results, before the runner freezes (user approves after this cross-review).
  2. Suspicious or implausibly large results.
  3. The final selected champion set, before consolidation — only implementations that ship in the final dispatcher require a clean final audit.
  4. Stall adviser, only when genuinely useful (separate role from auditor, never mixed).
- **Sol verdicts:** one JSON schema — PASS · RETEST (one round, fixed test menu executed by the trusted runner) · NEEDS_CONTEXT (missing *factual* evidence only) · RULE_VIOLATION; infrastructure failures are JUDGE_ERROR / TIMEOUT and **never block continued optimization**. First-pass audits are blind (neutral runner-generated evidence packet, no Fable commentary).
- **Dtype policy:** the primary official leaderboard uses the default FP32 baseline (script defaults, TF32 as shipped). Candidates may use reduced precision internally if they pass against that FP32 reference. Whole-benchmark FP16/BF16 runs are secondary profiles, never compared against FP32 runs.
- **Timebox:** Stage 0 + Stage 1 ≤ half a day combined; infrastructure is minimal. The deadline that matters is the first real optimized candidate measured on the GPU.

## Stages

- **Stage 0 — Rails.** shapes.json (14 exact configs; script defaults match none; causal always passed), manifest.json hashes, hooks + deny rules. Acceptance: a forbidden edit bounces.
- **Stage 1 — Trusted runner.** One command: shape id → hash check → multi-seed correctness + tripwires → timing distribution + noise floor → append `results/JOURNAL.jsonl` → leaderboard + evidence packets derived. Freeze after Sol checkpoint review + user approval.
- **Stage 2 — Loop proof.** Easy candidates (SDPA, torch.compile, CUDA graphs, internal precision) through the full pipeline; real 3060 Ti profiles set per-shape targets.
- **Stage 3 — Grind.** Per shape, worst-first: profile → fused Triton kernels + whole-stack CUDA graphs (small shapes) / tensor-core work (big shapes) → runner → champion. Stalls → Sol adviser.
- **Stage 4 — Shape 14.** Chunked reference; *measure* agreement vs naive baseline at every feasible length; full-scale correctness on big-memory GPU vs the validated chunked reference. Smaller-sequence speedups are scaling evidence, never the full-shape-14 figure.
- **Stage 5 — Consolidation.** Dispatcher routes to champions with clean final audits only. Final acceptance: untouched official script for every feasible shape; shape 14 accepted via the validated chunked reference with the official-baseline limitation stated clearly. Closing audit: hashes intact, git clean, leaderboard regenerates from journal.

## Authority

User holds final authority: Stage-1 freeze approval, promotion vetoes at will, sign-off on everything that ships. Fable works autonomously between those gates and leaves clear checkpoints when blocked.
