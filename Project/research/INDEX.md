# RESEARCH BASE — check BEFORE researching or proposing anything

Rule (user-mandated 29 Aug): every research finding worth using gets a note
here; every new research pass STARTS by reading this index; every candidate
memo must cite the notes it relies on. Notes store structured findings +
stable URLs + key quotes (not PDFs — arxiv/URLs are re-fetchable; notes are
what we act on). One topic per file.

| note | one-line takeaway |
|---|---|
| [agent-loop-design.md](agent-loop-design.md) | Winning kernel-agents separate STRATEGY from code, keep tried-direction memory, and put hardware counters in the loop |
| [megakernels-persistent.md](megakernels-persistent.md) | Megakernels are the low-latency endgame but honest wins vs CUDA-graphed baselines are ~1.2x on consumer GPUs; Triton can't grid-sync (multi-CTA), single-CTA IS Triton-expressible |
| [reviewer-bias.md](reviewer-bias.md) | LLM reviewers over-weight technical validity, under-weight strategy; same-family panels are self-critique; blind + cross-family + explicit strategy prompts required |
| [gemm-epilogue-fusion.md](gemm-epilogue-fusion.md) | Epilogue/elementwise fusion ~1.45x on bandwidth-bound GEMM paths (source of k010's +14%) |
| [quantization-tolerance.md](quantization-tolerance.md) | W8A8 int8 error (~1%+) is designed for tolerant workloads; predictably fails abs-2e-3 criteria (k008 confirmed empirically) |
| [competition-scoring.md](competition-scoring.md) | Webinar: weighted per-shape MFU, "bandwidth considered", fail=zero; public rubric NOT online — confirm weights via official info doc (user gate) |
| [roofline-table.md](roofline-table.md) | Per-shape FLOPs/bytes/AI + achieved TF: d=128 shapes are LATENCY-bound (megakernel regime); 8/13 near compute roof; 8/14 = fp16-acc territory |
