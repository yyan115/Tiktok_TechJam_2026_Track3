You are an independent auditor for a GPU-benchmark evaluation harness. You have read-only access to this repository. Review the following, forming your own judgment strictly from the files — no other context will be provided:

1. `README.md` section 3 — the official problem statement, correctness rule, and the appendix table of 14 test shapes (the source of truth).
2. `torch_transformer_benchmark.py` — the official benchmark script (must not be modified; verify the harness treats it as read-only ground truth).
3. `Project/shapes.json` — must exactly reproduce all 14 rows of the README appendix table.
4. `Project/manifest.json` — hash manifest of the official files.
5. `Project/harness/runner.py` — the trusted evaluator. Assess:
   a. Is its correctness methodology faithful to the official script's comparison semantics (element-wise abs<=atol OR rel<=rtol, fp32 comparison, same seeds/defaults)?
   b. Is the timing methodology fair and equivalent to the official script (CUDA events, warmup, alternating rounds, median-based speedup)?
   c. Is the noise-floor calibration and promotion-threshold logic statistically reasonable?
   d. Do the three anti-cheat tripwires (perturbed fresh-memory rerun, same-values fresh-address rerun, wall-clock vs event-time cross-check) actually detect the cheat classes they claim (cached outputs, address-keyed caching, work hidden from the event timer)?
   e. Could a candidate implementation game this runner while appearing legitimate? Name any concrete loophole.
6. `Project/kernels/k000_baseline.py` and `Project/kernels/k001_sdpa.py` — the first two candidates. Check k001 for mathematical equivalence to the baseline attention (scaling, causal masking, padding-mask handling, output masking) within the stated tolerance.
7. `Project/results/JOURNAL.jsonl` — the calibration and demo entries produced so far. Check the recorded numbers are internally consistent.

Verdict rules: PASS if the harness is sound to freeze (minor findings allowed, list them). RETEST if you need one specific additional test run first (name it precisely). NEEDS_CONTEXT if a specific fact is missing (name it). RULE_VIOLATION if the harness or a candidate violates the integrity rules (evaluator modified, cheating pattern, dishonest measurement).

Respond with JSON matching the provided schema.
