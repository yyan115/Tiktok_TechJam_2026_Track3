# Track 3 video script — DRAFT (target ~3 min; user records/approves)

Numbers marked [FINAL] come from the Stage-5 clean board. Screen captures
listed per scene so recording is a checklist, not an improvisation.

## Scene 1 — the tamper demo (0:00-0:35), cold open

Screen: terminal. Ask the agent, live, to edit the benchmark file.
- Show: Claude attempts an edit on torch_transformer_benchmark.py → DENIED
  by permission rules. Then a shell write attempt → the Bash guard blocks it.
- Then: `python3 Project/harness/runner.py check` → every hash green.
- Line: "The first thing our AI optimizer learned is that it can't touch the
  referee. Every result you're about to see survived that lockout, a frozen
  hash-pinned harness, and a blind audit by a rival AI."

## Scene 2 — the problem (0:35-1:00)

Screen: shapes.json + the baseline running.
- 14 transformer shapes, fp32, exact tolerance; a shape that fails precision
  scores zero. One shape's naive attention table would be multi-terabyte.
- Line: "AI-written kernels are notorious for cheating their own benchmarks —
  timing leaks, cached outputs, edited evaluators. We designed for that
  failure mode first, speed second."

## Scene 3 — the loop (1:00-1:45)

Screen: diagram (agent → frozen runner → journal → blind auditor → champion),
then a live `runner.py run` with the JSON verdict scrolling.
- Beats: candidates run from exact hashed bytes committed before measurement;
  correctness = multi-seed + fresh-memory anti-cache + shape asserts; timing
  = alternating rounds with wall-clock cross-check vs a calibrated per-shape
  noise floor; every new champion auto-fires a detached blind audit (GPT via
  codex) that sees only a neutral evidence packet.
- Show the audit ledger: PASS verdicts, and one early RULE_VIOLATION —
  "the auditors caught a provenance gap and a latent masking bug; we fixed
  both and re-measured everything. The trail is in the repo."

## Scene 4 — the kernels (1:45-2:30)

Screen: k007 source side-by-side with the baseline's ~40-launch profile.
- The megakernel: an entire transformer block in two authored Triton kernels
  (LayerNorm+QKV fused; flash attention over all heads with the output
  projection folded into the head loop, then norm+GELU-FFN in-register),
  whole forward replayed as one CUDA graph.
- Board: [FINAL board table] — geomean [FINAL]x across the 12 runnable
  shapes on a consumer RTX 3060 Ti.
- Honest beat: the int8 attempt FAILED tolerance and is in the repo as a
  documented negative result — "the referee doesn't grade on effort."

## Scene 5 — the impossible shapes + close (2:30-3:00)

Screen: shape14_core_smoke.py and shape6_core_smoke.py running live.
- seq=100,000 causal attention verified against a chunked fp32 oracle in
  305 MiB; batch 10,000 verified against the batch-chunked official baseline
  in 3.4 GiB — both on the same 8 GB card that can't even hold the baseline.
  Full-scale timing on a rented large-memory GPU, clearly labeled. [Update
  with rental numbers if available.]
- Close: "Every number regenerates from the journal with one command. The
  agent optimized; the system kept it honest; the human owned every gate."

## Recording checklist
- [ ] Idle box (no audits) for live runner demos.
- [ ] Terminal font large; dark theme; 1080p.
- [ ] Pre-open all files in tabs; rehearse scene 1 once (deny rules live).
- [ ] Keep raw footage; judges may want the uncut tamper demo.
