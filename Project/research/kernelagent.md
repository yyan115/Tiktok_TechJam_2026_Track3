# KernelAgent (Meta/PyTorch) — the loop template (researched 29 Aug; organizer-shown per DECISIONS)

Source: https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/ (Mar 2026; repo github.com/meta-pytorch/KernelAgent)
Regime: Triton forward-pass kernels, KernelBench L1, H100. Confidence: primary source, read in full.

- Loop: Profile (NCU) → Diagnose (roofline SOL + root causes each CITING a
  specific counter) → Prescribe (architecture-aware fixes from a curated
  pattern DB) → Orchestrate (history + reflexion → search strategy) →
  parallel Explore workers → Measure (gated correctness, benchmark lock
  against GPU contention).
- Search: beam over TOP-K kernels, never a single incumbent; parallel
  orthogonal hypotheses per round; architectural shifts (not param tweaks)
  repeatedly needed to escape local minima (matvec case: 9.5ms → 1.95ms,
  the winning move was a one-row-per-program redesign).
- Reflexion record per attempt: was_diagnosis_correct, was_fix_effective,
  expected vs actual outcome, lessons, avoid_patterns, try_patterns —
  broadcast to all agents next round. THIS is the experiment-card schema.
- Stop criterion: successive rounds without roofline-efficiency or runtime
  delta ⇒ stop. Headline metric = roofline efficiency (max of compute SOL,
  memory SOL) — 89% on H100; 1.56x vs default torch.compile, 65/100 tasks.
- Their stated lessons: hard verifiable gates keep agents honest; shared
  structured memory prevents duplicate exploration; diversity + clear
  termination beats sequential single-path refinement.

Implication: our outer loop should copy this shape nearly verbatim, scaled
to one GPU (hypothesis-parallel in design, serial in measurement).
