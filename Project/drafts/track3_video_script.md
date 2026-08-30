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
3. `python3 Project/harness/trusted_controller.py verify-lock` → prints
   `"valid": true`, `"protected_file_count": 29`, and the lock id.

> ⚠ **Not `runner.py check`** — that was the pre-LOCK command and it now dies
> with `invalid choice: 'check'`. Verified 31 Aug. Use `verify-lock` above,
> which is also the better shot: it names the lock and counts the files it
> pins.

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

> ⚠ **The command in the previous draft does not work, and that is now the
> scene.** `python3 Project/harness/runner.py run --shape 13 --impl …`
> fails instantly:
> `trusted_controller.py run: error: the following arguments are required: --permit`.
> Since the LOCK, `runner.py` is a 27-line shim that forwards everything to
> the trusted controller, and the controller will not time anything without
> a one-use permit. **Verified 31 Aug ~04:20 by running it.** Do not put the
> old command on camera expecting output.

Show the refusal first — it is the best three seconds in the film:

```bash
python3 Project/harness/runner.py run --shape 13 --impl Project/kernels/k009_fused_tuned.py
# error: the following arguments are required: --permit
```

**Say:** "Even I can't start a measurement. The agent asks the gate for
permission, the gate issues one permit bound to one file hash for one run,
and the permit is consumed the moment it's used."

Then the real thing — a permitted run, three commands:

```bash
python3 Project/tools/run_gate.py delta   --campaign CAMP-POSTLOCK ...   # emits a request
python3 Project/harness/trusted_controller.py issue-permit --request … --capability …
python3 Project/harness/trusted_controller.py run --permit permit-… --shape 13 \
        --impl Project/submission/torch_transformer_benchmark_submission.py
```

> **Check before recording:** the owner capabilities minted for the overnight
> grind expire roughly **21:00 on 31 Aug**. If you record after that, minting
> a fresh permit needs the signing key and the ceremony again. Mint a
> short-lived capability *before* you start filming.

**Beats while it runs:** the candidate is snapshotted into a
content-addressed blob at permit issue and that hash is bound into the
permit, the request and the result packet, so the audited bytes and the
measured bytes are the same object · correctness on **seven trials** — five
fixed seeds plus two drawn at random, so a kernel can't be tuned to the seed
list · tripwires: perturbed fresh-memory re-run, shape assertions,
wall-clock cross-check against the CUDA events · promotion is mechanical —
beat the calibrated per-shape noise floor or you are not a champion · a new
champion auto-fires a blind audit.

> **Corrected 31 Aug — these three had drifted from the other two documents.**
> The earlier version said "committed to git before the first measurement"
> (that was the *pre-LOCK* rule, and the auditor found it insufficient — a
> packet could still cite the current source hash rather than the measured
> one), "five seeds" (it is seven), and "every new champion auto-fires a
> blind audit" in the present tense (the audit recorder is currently broken;
> no post-LOCK row carries a verdict). Say the corrected version or say
> nothing — the tech report §4 and §6 carry the full story.

Then: `tail -3 Project/audits/verdicts.jsonl | python3 -m json.tool`

**Say:** "Eighty-one verdicts, and a third of them are rule violations
against us. Most are procedural — runs that predate the gate and can't prove
what plan they came from. But two changed the code: a provenance gap where a
packet cited the wrong source hash, and a masking bug in a path the
benchmark never exercises, so no measurement was ever affected and it was
still wrong. A rival model found both. We fixed them and re-measured."

> **Verified 31 Aug** by counting the ledger and reading the responses:
> 81 verdicts — 42 PASS, 28 RULE_VIOLATION, 8 NEEDS_CONTEXT, 3 RETEST. Both
> named findings are quoted verbatim in the tech report §4. Say the 28 out
> loud; a judge who opens `verdicts.jsonl` will see it, and owning it is far
> stronger than "sixty-plus verdicts".

## Scene 4 — the kernels (1:45–2:25)

**Screen:** `k009_fused_tuned.py` at the attention block, beside the
baseline's ~40-launch profile.

**Say:** "The baseline runs a transformer block as about forty separate GPU
operations. On the smallest shapes the GPU sits idle 86% of the time waiting
for the CPU to queue the next one. So we wrote the whole block as two Triton
kernels: LayerNorm fused into the QKV projection, then FlashAttention-style
causal attention over all heads with the output projection folded into the
head loop and the FFN finished in-register. Then we capture the entire
four-layer forward pass as a single CUDA graph."

**Then the beat that actually matters:** "But the graph isn't where the win
comes from. One of these shapes has a baseline that's idle only *one* percent
of the time — there are no launch gaps left to remove — and we're still nine
times faster there. Keeping the whole block in registers doesn't recover
waiting; it deletes memory traffic. We only know that because we measured it,
and it contradicted what we'd written in our own report."

**Board on screen:** geomean **9.45×** across all twelve locally-runnable
shapes — every one measured on the submission file itself, under a one-use
permit, all passing precision. Best shapes **28.3×** (sequence 1024) and
**21.0×** (`d_model` 32).

**Honest beat:** "The int8 attempt failed the tolerance test and it's in
the repo as a documented negative result. The referee doesn't grade on
effort."

## Scene 5 — the shapes that don't fit, and the close (2:25–3:00)

**Screen:** run both smokes live.

```bash
python3 Project/tools/smokes/shape14_core_smoke.py
python3 Project/tools/smokes/shape6_core_smoke.py
```

> ⚠ **Dry-run these two before you record.** The *full* evidence evaluators
> (`shape14_eval.py`, `shape6_local_eval.py`) are currently broken by a
> one-line device-comparison bug and abort instantly — see the tech report
> §2.4. These smokes are different files and were **not** testable from the
> agent's command allowlist, so their status tonight is unknown. Find out
> off-camera, not on it. If they fail the same way, the fix is the same
> one-liner and the scene still works with the packets on screen instead.

**Say:** "Sequence length one hundred thousand, verified against a chunked
fp32 oracle with zero tolerance violations — in 305 MiB, on the same eight
gigabyte card that can't even hold the baseline. Batch ten thousand, where
the official baseline runs out of memory, verified against the batch-chunked
official computation. Both on our own machine, by splitting the computation
into blocks — which is exactly what the organizers said they expected."

**Close:** "Every number is content-addressed and bound to the permit that
produced it. The agent optimized, the system kept it honest, and the human
owned every gate — including the one the agent isn't allowed to open."

> Changed from "every number regenerates from the journal with one command",
> which stopped being true at the LOCK: `runner.py leaderboard` no longer
> exists and the post-LOCK board lives in the authority log and its packets.
> Verified 31 Aug.

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

- geomean **9.45×** across the twelve locally-runnable shapes, measured on the
  submission file itself
- best shapes **28.3×** (sequence 1024) · **21.0×** (`d_model` 32)
- weakest shapes **2.02×** (`d_model` 1024) · **4.35×** (single attention head) — say
  these out loud if the per-shape board is on screen; the spread is the honest story
- correctness: **167,559,168 element comparisons, 0 failures** across the
  twelve runnable shapes (23.9M output elements × 7 trials each)
- shape 14: seq 100,000 causal, **0 violations**, 305 MiB
- shape 6: batch 10,000, **0 violations**, 3.4 GiB
- **all 14 shapes pass precision** — but say it precisely if pressed, because
  the evidence is two different grades: **12 shapes** are verified post-LOCK
  under the official predicate, 7 trials each (5 fixed seeds + 2 random),
  `correct: true`, on the shipped file; **shapes 6 and 14** have no runnable
  official baseline, so they are verified against validated oracles on
  **one seed each, against a pre-integration file**, and those packets
  cannot currently be regenerated (owner-only tooling fix). Both grades are
  real; they are not the same grade.

**Say the caveats or cut the number.** These are screening-lane characterisation runs:
correct on all twelve, but none is a promoted champion, no audit verdict is bound to any
of them (the audit recorder is broken and only the owner can fix it), and the geomean
excludes shapes 6 and 14, which do not run locally at all.
