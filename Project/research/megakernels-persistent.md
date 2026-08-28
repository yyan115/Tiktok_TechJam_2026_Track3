# Megakernels / persistent kernels for low-latency inference (researched 29 Aug 2026)

## Landscape
- Mirage Persistent Kernel (MPK): compiler+runtime mega-kernelizing whole
  models; claims 1.2-6.7x LATENCY reduction for LLM inference (vs unfused
  baselines). https://arxiv.org/html/2512.22219v1 and Zhihao Jia's writeup
  https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17
- Launch overhead cited at ~14.6% of end-to-end inference in one analysis;
  5-10us per launch compounding across layers.
- Ada-MK (industrial deployment, DAG-search): https://arxiv.org/html/2605.11581v1
- Fleet (multi-die megakernel abstraction): https://arxiv.org/html/2604.15379

## The honest calibration (CRITICAL — sets our expectations)
AutoMegaKernel (https://arxiv.org/html/2606.09682) measured against
CUDA-GRAPHED cuBLAS baselines (i.e., against a baseline like OUR champions,
not against naive eager):
- Consumer RTX 5090: megakernel wins 1.19-1.23x — and only with int8
  weight-only precision asymmetry; at equal precision it TRAILS by ~13%.
- A100/H100: LOSES to graphed cuBLAS (cross-SM sync overhead not amortized).
- Sync model: monotonic uint32 producer/consumer counters, no locks.
Takeaway: once a model is CUDA-graphed and block-fused (which ours is), the
residual megakernel win on consumer silicon is TENS OF PERCENT, not multiples.
The 6.7x-class claims compare against weaker baselines.

## Triton expressiveness boundary
- Documented: Triton has NO grid-level synchronization between programs —
  multi-CTA persistent kernels with cross-block dependencies are
  inexpressible. (Multiple sources incl. FlashRNN paper, MegaKernel topic
  overviews; also the reason MPK/AutoMegaKernel emit CUDA.)
- BUT a SINGLE-PROGRAM kernel needs no grid sync: for B=1 d=128 seq=128,
  the entire 4-layer forward fits one program (activations 128x128 fp16 in
  SRAM, ~200KB weights streamed from L2). Triton CAN express that.
- TritonForge (profiling-guided Triton optimization):
  https://arxiv.org/html/2512.09196v1

## Application map for OUR shapes (predictions, to be memo'd before build)
- Shape 2 (B=1): single-program Triton whole-model kernel. Bounded by
  removing residual inter-kernel gaps (~10-20us of ~50us forward):
  predicted 1.2-1.5x on that shape. NO toolchain blocker.
- Shapes 12/4 (small-token, multi-CTA): CUDA counter-sync persistent kernel;
  BLOCKED locally (nvcc13+gcc16 segfault; no g++-14), viable on rental or
  after owner installs gcc14. Predicted 1.1-1.3x class.
- Big shapes: megakernel NOT the lever (AutoMegaKernel loses there);
  GEMM-side techniques instead.

## Toolchain unlock (verified 29 Aug ~04:05)
CUDA 13.0 officially supports GCC 15 hosts
(https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)
and Fedora 44 ships gcc15-c++ in updates. LIVE as of 29 Aug ~04:30: owner installed gcc15-c++; probe kernel compiled
and ran via load_inline with -ccbin g++-15. Multi-CTA persistent kernels are
now locally buildable.
