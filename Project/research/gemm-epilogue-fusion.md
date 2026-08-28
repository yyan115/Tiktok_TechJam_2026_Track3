# GEMM epilogue / elementwise fusion (researched 28-29 Aug 2026)

- cuBLAS 12.x fuses common epilogues (bias, GELU) with matmuls:
  https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/
- Average ~1.45x for GEMMs with epilogue fusion where the elementwise tail
  was bandwidth-bound (survey via
  https://www.emergentmind.com/topics/fused-triton-kernels and tile-overlap
  paper https://arxiv.org/html/2607.02521v1).
- Fused GEMM+activation kernels reach ~79.5% peak BF16 utilization in
  roofline studies (https://github.com/bassrehab/triton-kernels).
- VALIDATED LOCALLY: k010 (fused LayerNorm->fp16 + in-place erf-GELU around
  fp16 cuBLAS GEMMs) took shape 8 from 1.79x to 2.13x (+14%, quiet-box) —
  the one research-driven build of Day 1 and its cleanest win.
- Remaining headroom on shape 8: fp16-ACCUMULATE tensor-core GEMMs are 2x
  the fp16/fp32-acc rate on sm86 — needs an error model FIRST (see
  quantization-tolerance.md pattern: sqrt(K)-scaled rounding noise vs the
  2e-3/2% criterion; K=1024 makes this borderline). Also torch has no
  fp16-acc path — would need authored Triton GEMM competitive with cuBLAS,
  historically ~85-95% of cuBLAS on these sizes: net gain uncertain. Memo
  required before any build.
