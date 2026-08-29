TASK: Fix the fatal allocation bug in Project/tools/shape14_eval.py by
rewriting its eval path to a fully streamed per-slice protocol. This is a
TOOL bugfix — do not touch any kernel or the submission implementation.

CONSTRAINTS (hard):
- NEVER modify: torch_transformer_benchmark.py, tensorflow_transformer_benchmark.py,
  README.md, Project/shapes.json, Project/manifest.json, Project/harness/**,
  Project/results/**, .claude/**, Project/loop/**, Project/tools/run_gate.py,
  Project/submission/**, Project/kernels/**. Read-only reference to all of
  these is fine and expected.
- NO GPU execution, NO benchmark runs. Verify with `python3 -m py_compile`
  only. The maintainer runs GPU validation later.
- Commit when done with a message stating exactly what changed. Never place
  the words 'clean', 'reset' or 'restore' after 'git' in any command or
  commit message.

THE BUG: the current eval path builds the full shape-14 input via fresh_x
(B=32 x S=100000 x D=1024 fp32 = 12.2 GiB) and receives a full output
(another 12.2 GiB) before any slicing. Input+output alone exceed the 8 GB
RTX 3060 Ti. The full B=32 workload has therefore never been measured.

REQUIRED DESIGN (agreed by three independent reviews — implement exactly):

1. KEEP the existing streamed fp32 oracle functions byte-for-byte — their
   math is validated against the official dense baseline (packet
   Project/results_side/validation_20260829-041941.json). Do not alter them.

2. CANDIDATE SOURCE: the generated submission file itself. Import
   UserOptimizedTransformer from
   Project/submission/torch_transformer_benchmark_submission.py (importlib
   from path; the file has a __main__ guard). Keep the evaluator's existing
   weight-initialization/sync semantics identical to what it does today —
   only where the candidate class comes from changes.

3. STREAMED EVAL (new eval path). Never allocate a [32, S, D] tensor:
   For each timing repeat r (default 3, flag --timing-repeats):
     for batch index i in 0..31:
       - build ONE slice [1, S, D] on GPU from a deterministic generator
         seeded with (base_seed * 1000 + i); record the formula in the packet
       - run the candidate on the slice; time the compute with CUDA events
       - free slice + output before the next index
     - full-workload sample = SUM of the 32 slice times
     - also record a staging-inclusive wall time for the whole repeat
       (time.perf_counter around everything incl. generation and frees)
   Report median of the full-workload sums + the raw 32 x repeats matrix.
   NEVER multiply one slice's median by 32 anywhere.

4. CORRECTNESS pass (separate from timing, flag --correctness-seeds,
   default 5 seeds): per seed, per slice: candidate output vs the streamed
   oracle using the OFFICIAL predicate copied from
   torch_transformer_benchmark.py (read it around line 305-318):
   finite-mask AND (abs_err <= atol OR abs_err <= rtol * ref.abs()).
   torch.isclose is FORBIDDEN (it is additive and more permissive — the
   official file's own comment says so). Accumulate total violations,
   global max abs error, mean abs error, worst slice index.

5. MEMORY: reset torch.cuda peak stats per repeat; record BOTH
   max_memory_allocated and max_memory_reserved maxima.

6. PACKET: write to Project/results_side/ (NEVER Project/results/) named
   shape14_streamed_<UTCstamp>.json containing: config, seed list + slice
   seed formula, timing matrix, sums, median_of_sums_ms, wall_ms per
   repeat, violations, max/mean abs error, peak alloc+reserved,
   predicate: "official-abs-OR-rel", atol/rtol used, torch/cuda/driver
   versions, device name, sha256 of the SUBMISSION FILE, sha256 of the
   evaluator file itself, and the packet-schema version string "streamed-v1".

7. NEW SUBCOMMAND decomp-check: at a reduced size that fits (default B=8,
   S=2048), generate ONE full input, compare candidate(full batch) against
   torch.cat of candidate on each B=1 slice of that same tensor, report max
   abs difference (expect exact or ~1e-6; it must NOT approach 2e-3).
   The maintainer runs this on GPU before trusting the streamed protocol.

8. Keep the existing validate subcommand working unchanged.

If any part of this spec conflicts with what you find in the code, do the
non-conflicting parts and describe the conflict in a NOTES section of the
commit message rather than improvising a different design.
