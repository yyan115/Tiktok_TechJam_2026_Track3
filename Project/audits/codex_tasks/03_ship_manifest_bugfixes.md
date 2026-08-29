TASK: Fix three recorded bugs in Project/tools/ship_manifest.py (the
generator that maps all 14 shapes to their evidence). CODE FIX ONLY — do
NOT regenerate Project/results_side/SHIP_MANIFEST.json now; regeneration
happens at code freeze from the final commit.

CONSTRAINTS (hard):
- NEVER modify: torch_transformer_benchmark.py, tensorflow_transformer_benchmark.py,
  README.md, Project/shapes.json, Project/manifest.json, Project/harness/**,
  Project/results/**, .claude/**, Project/loop/**, Project/tools/run_gate.py,
  Project/submission/**, Project/kernels/**, and do not overwrite existing
  files in Project/results_side/. Read-only reference is fine.
- NO GPU execution, NO benchmark runs. `python3 -m py_compile` only.
- Commit when done. Never place the words 'clean', 'reset' or 'restore'
  after 'git' in any command or commit message.

BUG (a): the manifest records the current git revision at generation time,
which can predate the submission sha it cites. Fix: refuse to run when the
working tree has uncommitted changes; record the HEAD revision AND assert
that `git show <HEAD>:Project/submission/torch_transformer_benchmark_submission.py`
hashes to the same sha256 as the on-disk submission file — abort with a
clear message if not. Document in the module docstring: "regenerate at
freeze, AFTER the final commit".

BUG (b): the shape-14 entry is selected by comparing latencies across
packets with different batch sizes (a B=1 median vs a B=2 median is
meaningless). Fix: never compare latency across differing batch_size.
Prefer, in order: a packet with schema version "streamed-v1" (full
32-slice protocol); otherwise the most recent packet; and list ALL other
shape-14 packets under an "alternates" key so nothing is hidden.

BUG (c): the shape-6 entry is labeled "oracle validated vs official
dense". It was actually validated against the BATCH-CHUNKED official
baseline (chunks of 500 batch rows; exact because batch entries never
interact). Fix the label to exactly that wording and make the audit_verdict
strings per-shape truthful rather than one shared constant, if applicable.
