TASK (ANALYSIS — writes exactly ONE new file, changes nothing else):
Explain a memory anomaly from source-code reading alone.

OBSERVATION: Project/results_side/shape14_20260829-042452_B1_S100000.json
records peak allocated ~4.894 GiB for batch size 1, while
shape14_20260829-042810_B2_S100000.json records ~4.738 GiB for batch size
2. The SMALLER workload peaked HIGHER. Both runs: seq len 100000, d_model
1024, 16 heads, 2 layers, seed 1234, same machine (RTX 3060 Ti 8GB).

CONSTRAINTS (hard):
- Modify NOTHING except creating the single output file named below.
- NEVER touch: torch_transformer_benchmark.py, tensorflow_transformer_benchmark.py,
  README.md, Project/shapes.json, Project/manifest.json, Project/harness/**,
  Project/results/**, .claude/**, Project/loop/**, Project/tools/**,
  Project/submission/**, Project/kernels/** (read-only reference expected).
- NO GPU execution. This is a source-reading task.
- Commit only the one new file. Never place the words 'clean', 'reset' or
  'restore' after 'git' in any command or commit message.

READ: Project/kernels/k014_shape14.py, the long-sequence route inside
Project/submission/dispatcher_region.py (batch micro-chunking, staging
frees, the seq-chunked tail), Project/tools/shape14_eval.py (how each run
allocated inputs/outputs and measured peaks), and both packets.

DELIVER: Project/audits/external/memory_inversion_b1_gt_b2.md containing:
1. The most likely mechanism (e.g. allocator caching/fragmentation,
   different code path taken at B=1 vs B=2, evaluator-side staging,
   measurement-reset timing), argued from specific cited lines.
2. Alternative explanations ranked, each with the evidence that would
   confirm or kill it.
3. A concrete prediction for what peak the FULL streamed 32-slice protocol
   should show if your primary explanation is right — so the upcoming real
   run doubles as the test of this analysis.
Plain language where possible; cite file:line for every claim.
