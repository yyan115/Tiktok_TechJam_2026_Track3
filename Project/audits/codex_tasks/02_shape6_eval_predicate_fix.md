TASK: Fix the wrong pass/fail formula in Project/tools/shape6_local_eval.py.

CONSTRAINTS (hard):
- NEVER modify: torch_transformer_benchmark.py, tensorflow_transformer_benchmark.py,
  README.md, Project/shapes.json, Project/manifest.json, Project/harness/**,
  Project/results/**, .claude/**, Project/loop/**, Project/tools/run_gate.py,
  Project/submission/**, Project/kernels/**. Read-only reference is fine.
- NO GPU execution, NO benchmark runs. `python3 -m py_compile` only.
- Commit when done. Never place the words 'clean', 'reset' or 'restore'
  after 'git' in any command or commit message.

THE BUG: the evaluator judges correctness with torch.isclose, whose
condition is additive (abs_err <= atol + rtol*|ref|). The official
benchmark (torch_transformer_benchmark.py, read around line 305-318) uses
an OR condition — finite-mask AND (abs_err <= atol OR abs_err <= rtol*|ref|)
— and its own comment explicitly says isclose is more permissive and is
not used.

FIX:
1. Replace every torch.isclose-based check with the official semantics,
   copied faithfully (float() casts, finite mask, abs_ok OR rel_ok, count
   of failed elements). Keep the evaluator's existing ATOL/RTOL constants
   — verify they equal the official script's CLI defaults and state both
   values in the packet.
2. Also record in the result packet: peak reserved memory alongside peak
   allocated (torch.cuda.max_memory_reserved), the predicate name
   "official-abs-OR-rel", and the sha256 of the candidate file evaluated.
3. Add an optional --seeds flag (comma-separated, default the current
   single seed) looping the correctness check over multiple seeds and
   aggregating violations/max-error across all of them; timing stays
   single-protocol as it is today.
4. Do the same predicate replacement in any OTHER file under
   Project/tools/ that uses torch.isclose for pass/fail judgment EXCEPT
   Project/tools/shape14_eval.py (a separate task rewrites that one) —
   list every file you changed in the commit message.
