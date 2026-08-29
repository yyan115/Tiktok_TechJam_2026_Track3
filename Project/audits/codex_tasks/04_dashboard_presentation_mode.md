TASK: Add a "Presentation mode" to Project/tools/dashboard.py (Streamlit).
You fixed bugs in this file recently, so you know its layout. Purpose: the
dashboard currently renders raw private log excerpts (external reviewer
prompts/responses, raw file tails, absolute filesystem paths). It must be
screen-recordable for a demo video without leaking any of that.

CONSTRAINTS (hard):
- NEVER modify: torch_transformer_benchmark.py, tensorflow_transformer_benchmark.py,
  README.md, Project/shapes.json, Project/manifest.json, Project/harness/**,
  Project/results/**, .claude/**, Project/loop/**, Project/tools/run_gate.py,
  Project/submission/**, Project/kernels/**. Read-only reference is fine.
- NO GPU execution, NO benchmark runs. `python3 -m py_compile` only (it is
  fine NOT to launch streamlit; the maintainer will eyeball it).
- Commit when done. Never place the words 'clean', 'reset' or 'restore'
  after 'git' in any command or commit message.

SPEC:
1. Sidebar toggle "Presentation mode (hide private logs)", default OFF.
2. When ON:
   - Raw log excerpt expanders / code blocks show a placeholder like
     "(hidden in presentation mode)" instead of file contents — event
     titles, timestamps, friendly names and verdict labels stay visible.
   - No absolute filesystem path is rendered anywhere (strip to basename
     or hide).
   - Reviewer prompt/response text is never rendered.
3. When OFF: behavior identical to today.
4. Keep the forced dark theme, auto-refresh fragment, metric cards,
   scoreboard and Right-now strip exactly as they are.
5. State the toggle's session-only nature in a small caption (no config
   file writes).
