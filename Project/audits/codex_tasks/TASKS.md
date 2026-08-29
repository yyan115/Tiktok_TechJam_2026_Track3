# Codex task briefs (30 Aug) — bug-fix / tooling work only, NO optimization

Run each from the repo root, one task per codex run:

    codex exec "$(cat Project/audits/codex_tasks/01_shape14_eval_streaming_fix.md)"

(Task 05 is analysis-only — it writes exactly one findings file.)

Ranked order: 01 (critical path) > 02 > 03 > 04 > 05.

## Hard constraints for EVERY task (repeated in each brief)
- NEVER modify: torch_transformer_benchmark.py, tensorflow_transformer_benchmark.py,
  README.md, Project/shapes.json, Project/manifest.json, Project/harness/**,
  Project/results/** (runner-owned), .claude/**. Read-only reference is fine.
- NO GPU work, NO benchmark runs, NO Project/harness/runner.py invocations.
  Syntax-check with `python3 -m py_compile <file>` only. GPU validation is a
  separate later step by the maintainer.
- Do not touch Project/loop/** (gate machinery) or Project/tools/run_gate.py.
- Commit when done, message stating exactly what changed. Never place the
  words 'clean', 'reset', or 'restore' after 'git' in any command or message.
- Stay strictly on the brief. If something in the brief looks wrong, write
  the concern into a NOTES section of the commit message instead of improvising.
