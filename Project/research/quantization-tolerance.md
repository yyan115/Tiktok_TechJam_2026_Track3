# Quantization vs this benchmark's tolerance (researched 29 Aug 2026, post-hoc of k008)

- The W8A8 literature (LLM.int8, SmoothQuant lineage) targets workloads
  tolerating ~0.5-1%+ output error; per-token dynamic + per-channel weights
  is the standard recipe and its error floor is set by int8 rounding noise
  compounding across layers.
- Our criterion is abs 2e-3 OR rel 2% per element, fp32 reference, 4 layers.
  k008 measured ~3.5e-2 max err / ~12% violations at d=1024 L=4 — an order
  of magnitude out, exactly what the literature predicts. fp16-internal
  (with fp32 accumulate/norms/residuals) sits at ~1e-3 — inside.
- RULE OF THUMB extracted: estimate error BEFORE building any reduced-
  precision candidate: per-GEMM relative noise ~ (quant step / signal) *
  sqrt-ish accumulation across K and layers; compare against 2e-3 abs on
  O(1) outputs. int8: predictably out. fp16-acc GEMM at K=1024: borderline
  (~1.5% class) — modelable, must be memo'd first.
- PROCESS LESSON: this note existing on 28 Aug would have saved the k008
  build entirely. That is the research-gate argument in one line.
