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

## Triton fp16-accumulate expressibility (verified 29 Aug)
tl.dot accepts accumulators of {float16, float32, int32} and an out_dtype
parameter (https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
— fp16-acc HMMA is Triton-expressible on sm86; no CUDA C++ needed for the
shape-8 fp16-acc experiment. Error model still mandatory before build.

## fp16-accumulate ERROR MODEL (pre-build memo, 29 Aug ~04:35 — k008 lesson applied)
fp16 unit roundoff u = 2^-11 ~ 4.9e-4. Stochastic accumulation error over a
K-length dot ~ sqrt(K)*u relative.
- Naive fp16-acc, K=1024 (shape 8): 32*u ~ 1.6% per GEMM; compounding across
  4 layers' residual stream (~sqrt(4)x): ~3% > the 2% rel criterion.
  PREDICTED FAIL — do not build the naive variant.
- Chunked fp16-acc (fp16 MMA within K-chunks of 128, fp32 across chunks):
  sqrt(128)*u ~ 0.55% per GEMM, ~1.1% compounded — borderline PASS with
  margin ~2x; keeps most of the 2x fp16-acc rate (fp32 add every 128 K-steps
  is amortized). BUILD ONLY THIS VARIANT, kill-criteria: referee tolerance
  fail on shape 8, or <10% end-to-end gain (Triton GEMM must also be within
  ~10% of cuBLAS at M=8192 K=N=1024 for the 2x MMA rate to net positive —
  measure the plain GEMM first before fusing anything).
