# Standing orders (every session, read before doing anything)

1. Read `Project/memory/STATE.md` first (a SessionStart hook also injects it). Then check `Project/memory/LESSONS.md` before working and `Project/PLAN.md` for the agreed plan. Log decisions in `Project/memory/DECISIONS.md`, lessons in LESSONS.md as they happen.
2. NEVER edit: `torch_transformer_benchmark.py`, `tensorflow_transformer_benchmark.py`, `README.md`, `Project/shapes.json`, `Project/manifest.json`, anything in `Project/results/` (runner-written only), `.claude/**`. Deny rules + a Bash guard hook enforce this; behave as if they are always active.
3. Every benchmark goes through `Project/harness/runner.py` with a shape id from `Project/shapes.json`. No raw-dial benchmarking, ever.
4. Promotion: correctness pass + speedup above the calibrated noise floor ⇒ working champion; audit status is separate. Sol (codex) audits at checkpoints only. Sol failures never block work.
5. The user requires plain language (no jargon walls) and an explicit "go" before repo actions. Answer all questions first. The user approves the runner freeze and owns everything that ships.
6. When optimizing (grind phase): fresh web research per technique is encouraged — assume the field has moved; copy and cite what works.
