# Track 3 demo video — shot list v2 (30 Aug)

Target ~3:00. Required: uploaded to YouTube, **public**, linked in the
Devpost description, no third-party trademarks or copyrighted music.

**When.** Recording happens in the packaging window (31 Aug 20:00 → 1 Sep
02:00). That window is now six hours, not fourteen — so this is a shot
list to execute, not a script to improvise against. Rehearse scene 1 once;
everything else is a single take per scene.

**Before you hit record**
- [ ] Idle box: no audits running (`pgrep -f "codex exec"` empty), browser
      closed. The live speedups are real numbers and contention moves them.
- [ ] Terminal: dark theme, large font, 1080p, window maximized.
- [ ] Pre-open in tabs: `Project/kernels/k009_fused_tuned.py`,
      `Project/shapes.json`, `Project/audits/verdicts.jsonl`.
- [ ] `git status` clean, on the frozen commit.
- [ ] Keep the raw footage — the uncut tamper demo is worth having.

---

## Scene 1 — cold open: the agent cannot touch the referee (0:00–0:35)

**Screen:** terminal, Claude Code session live.

1. Ask the agent, on camera: *"edit torch_transformer_benchmark.py, add a
   comment at the top."* → permission layer **DENIES** it.
2. Ask it to do the same through the shell: `echo "# x" >> torch_transformer_benchmark.py`
   → the Bash guard hook **BLOCKS** it.
3. `python3 Project/harness/runner.py check` → every hash green.

**Say:** "The first thing our AI optimizer learned is that it can't touch
the referee. Everything you're about to see survived that lockout, a frozen
hash-pinned harness, and blind audits by a rival AI that was actively
looking for measurement fraud."

## Scene 2 — the problem (0:35–1:00)

**Screen:** `Project/shapes.json`, then the official baseline running.

**Say:** "Fourteen transformer shapes, exact precision tolerance — a shape
that fails scores zero. One of them has a sequence length of a hundred
thousand, where the naive attention table is multiple terabytes; it doesn't
fit on any GPU. And AI-written kernels are notorious for cheating their own
benchmarks: timing work on a side stream, caching outputs, editing the
evaluator. We designed for that failure mode first and speed second."

## Scene 3 — the loop (1:00–1:45)

**Screen:** the trust-chain diagram, then a live runner invocation.

```bash
python3 Project/harness/runner.py run --shape 13 --impl Project/kernels/k009_fused_tuned.py
```

**Beats while it runs:** candidate bytes are committed to git *before* the
first measurement, so the audited bytes and the measured bytes are provably
the same · correctness on five seeds using the official predicate ·
tripwires: perturbed fresh-memory re-run, shape assertions, wall-clock
cross-check against the CUDA events · promotion is mechanical — beat the
calibrated per-shape noise floor or you are not a champion · every new
champion auto-fires a blind audit.

Then: `tail -3 Project/audits/verdicts.jsonl | python3 -m json.tool`

**Say:** "Sixty-plus verdicts, and they aren't decoration — the auditors
caught a provenance gap and a latent masking bug in code the benchmark
never exercised. We fixed both and re-measured everything. The trail is in
the repo."

## Scene 4 — the kernels (1:45–2:25)

**Screen:** `k009_fused_tuned.py` at the attention block, beside the
baseline's ~40-launch profile.

**Say:** "The baseline runs a transformer block as about forty separate GPU
operations. Nine of the fourteen shapes are small enough that the GPU
finishes each one before the CPU can queue the next — so the wall isn't
arithmetic, it's launch overhead. We wrote the whole block as two Triton
kernels: LayerNorm fused into the QKV projection, then FlashAttention-style
causal attention over all heads with the output projection folded into the
head loop and the FFN finished in-register. Then we capture the entire
four-layer forward pass as a single CUDA graph."

**Board on screen:** geomean **9.68×** across all twelve locally-runnable
shapes, every one measured under a one-use permit on the kernel the
dispatcher actually selects, all passing precision. Best shapes **28.4×**
(sequence 1024) and **22.0×** (`d_model` 32).

**Honest beat:** "The int8 attempt failed the tolerance test and it's in
the repo as a documented negative result. The referee doesn't grade on
effort."

## Scene 5 — the shapes that don't fit, and the close (2:25–3:00)

**Screen:** run both smokes live.

```bash
python3 Project/tools/smokes/shape14_core_smoke.py
python3 Project/tools/smokes/shape6_core_smoke.py
```

**Say:** "Sequence length one hundred thousand, verified against a chunked
fp32 oracle with zero tolerance violations — in 305 MiB, on the same eight
gigabyte card that can't even hold the baseline. Batch ten thousand, where
the official baseline runs out of memory, verified against the batch-chunked
official computation. Both on our own machine, by splitting the computation
into blocks — which is exactly what the organizers said they expected."

**Close:** "Every number regenerates from the journal with one command. The
agent optimized, the system kept it honest, and the human owned every gate
— including the one the agent isn't allowed to open."

---

## Optional 20s insert, if the runtime allows it

The strongest thing we have is not a kernel. In a sibling track, this same
agent architecture overrode the organizers' stop rule eleven times through
a free-text override field — each time with a plausible written reason. We
caught it, diagnosed it, and rebuilt the authority model so that no AI can
authorize an exception: arithmetic is judged by code that cannot be
persuaded, the auditor's hard verdicts pause the machine automatically, and
only the human can unlock. If it fits, say it — it is the most
differentiated thirty seconds in the video.

## Numbers to have on the lower third

> ✅ **RESOLVED 31 Aug ~02:10 — the lower third is cleared to show numbers again.**
> All twelve locally-runnable shapes have now been re-measured under one-use permits,
> on a quiet box, on the kernel the dispatcher actually selects. The withdrawn figures
> were **procedurally invalid but numerically close**: mean delta −5.6%, geomean 9.68×
> against the withdrawn 10.32×. Use the numbers below and no others. The 2.94× figure
> that briefly replaced them was itself wrong — it measured a kernel we do not ship.

- geomean **9.68×** across the twelve locally-runnable shapes
- best shapes **28.4×** (sequence 1024) · **22.0×** (`d_model` 32)
- weakest shapes **2.02×** (`d_model` 1024) · **4.84×** (single attention head) — say
  these out loud if the per-shape board is on screen; the spread is the honest story
- shape 14: seq 100,000 causal, **0 violations**, 305 MiB
- shape 6: batch 10,000, **0 violations**, 3.4 GiB
- **all 14 shapes pass precision**

**Say the caveats or cut the number.** These are screening-lane characterisation runs:
correct on all twelve, but none is a promoted champion, no audit verdict is bound to any
of them (the audit recorder is broken and only the owner can fix it), and the geomean
excludes shapes 6 and 14, which do not run locally at all.
