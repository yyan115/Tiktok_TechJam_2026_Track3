# Agent-loop design for kernel optimization (researched 29 Aug 2026)

## cuPilot — strategy-coordinated multi-agent kernel evolution
Source: https://arxiv.org/abs/2512.16465 (+ html)
- Core: "strategy" = explicit high-level optimization concept (e.g. "tensor
  core utilization", "thread block swizzling") maintained as an intermediate
  semantic layer SEPARATE from code. Evolution (selection/crossover) happens
  at the STRATEGY level, then strategies are translated to code.
- Loop: generate strategies → apply to kernel → ALIGN strategy to what the
  code actually did → tournament-select on performance AND hardware
  utilization counters → crossover strategies (not code).
- Roofline-guided prompting: classify each kernel compute- vs memory-bound
  (kernel description + GPU specs + precision); the classification steers
  which strategies are generated and which counters are watched. Ablation:
  -44.2% latency over two epochs on four representative kernels.
- Historical database + RAG strategy seeding: store (initial kernel,
  optimized kernel, metrics); seed new tasks from similar past records.
  Ablation: -54.1% latency after ONE epoch — the LARGEST single component.
  == the user's "research memory with a mandatory check" instinct, validated.
- Headline: 3.09x avg over PyTorch on a 100-kernel benchmark.

## CudaForge — Coder+Judge with hardware feedback
Source: https://openreview.net/forum?id=f4GtuI2blh
- Two agents (Coder, Judge), training-free, Nsight Compute counters fed into
  the loop. 97.6% correctness, 1.68x avg, cheap, generalizes across GPUs.
- Implication: our loop lacked the Judge-with-counters; wall-time-only
  feedback hides WHERE time goes (k011 built blind on a latency guess).

## AutoMegaKernel — agent proposes configs, architecture guarantees safety
Source: https://arxiv.org/html/2606.09682
- Agent emits structured ScheduleConfig (tiling, fusion grouping, SM
  assignment, pipelining), a FROZEN deterministic VM lowers it: "the
  architecture, not the agent, guarantees correctness" — same philosophy as
  our frozen referee. Static validator (DAG acyclicity, wait-satisfiability,
  happens-before) had zero false-accepts on 7,160 adversarial schedules.
- Perf calibration: see megakernels-persistent.md.

## Sakana AI CUDA Engineer + evolutionary-loop family
- Stages: convert → translate → optimize → compose; the famous reward-hacking
  postmortem is why our tripwires exist. AlphaEvolve/OpenEvolve class:
  generate-evaluate-select + EVOLUTIONARY MEMORY of attempts, so failed
  directions are not blindly retried.
- Survey/collection: https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation

## Implications for our harness v2 (feeds harness_v2_proposal.md)
1. Maintain an explicit per-shape STRATEGY table, evolved/critiqued
   separately from code (cuPilot).
2. Research base with mandatory pre-work check (this directory) = cuPilot's
   RAG database, their biggest ablation win.
3. Hardware counters (ncu / torch profiler kernel table) in EVERY candidate
   evaluation, tournament-style (CudaForge).
4. TRIED.md ledger = evolutionary memory (AlphaEvolve class).
5. Frozen-referee philosophy already matches AutoMegaKernel — keep.
