# Megakernels / persistent kernels (source-of-truth rewrite 29 Aug, post 2 review rounds)

## Honest performance picture
- Mirage Persistent Kernel (MPK): all-SM compiler/runtime with an SM-level
  task graph, evaluated on LLM SERVING; current paper reports up to ~1.7x.
  It is NOT evidence for one-CTA whole-model kernels.
  https://arxiv.org/html/2512.22219v1
- AutoMegaKernel (https://arxiv.org/html/2606.09682): agent proposes
  structured ScheduleConfigs, a frozen VM lowers them (architecture
  guarantees correctness — same philosophy as our referee). Its consumer-GPU
  "win" (1.19-1.23x on RTX 5090) was W8A16 vs BF16 graphed cuBLAS at
  batch-1 decode — precision-asymmetric; at EQUAL precision it trailed
  graphed cuBLAS by ~13%, and it loses on A100/H100. Its safety requires
  one-block-per-SM cooperative residency + static DAG/wait-satisfiability/
  happens-before validation — not a quick atomics transplant.
- Net: against a CUDA-GRAPHED, block-fused baseline (ours), megakernel
  upside on consumer silicon is tens of percent at best, concentrated where
  inter-kernel gaps are a large fraction of runtime.

## Triton expressiveness boundary
- Triton has no grid-level sync between programs — multi-CTA persistent
  kernels with cross-block dependencies need CUDA C++ (cooperative groups
  or counter protocols). Single-program kernels need no grid sync and ARE
  Triton-expressible.
- Toolchain: CUDA C++ is LIVE locally (gcc15 installed 29 Aug; CUDA 13
  supports GCC 15 hosts; probe kernel verified via load_inline -ccbin g++-15).

## Application map for OUR shapes (post-review)
- Shape 2 (B=1): DEAD as a single-CTA play. Arithmetic: 117.44 MFLOP at one
  SM's share of fp16 peak (32.5 TF / 38 SMs) floors at ~137 us; the current
  champion is 144.4 us. (Weights are 768 KiB fp16 — 6 d x d matrices x 4
  layers — not the 200 KB once claimed.) Only a changed premise (fp16-acc
  or multi-CTA) reopens it; it sits LAST in the allocation.
- Shapes 3/4/12 (small-token): the cheap correct idea is SEQUENCE-PERSISTENT
  INDEPENDENT CTAs — one program runs all 4 layers for ONE batch sequence;
  sequences are independent under causal attention, so no cross-CTA sync
  exists at all; Triton-expressible. Timeboxed, AFTER 14/6/8/11/13, and only
  if coverage/packaging are green.
- Shape 14: attention-dominated (~94% of useful FLOPs) — the lever is an
  authored FlashAttention-2-style kernel (online softmax, causal tiling,
  query-block parallelism within each head), NOT megakernel or GEMM work.
- Big-d shapes generally: megakernels are the wrong tool (AMK loses there).
