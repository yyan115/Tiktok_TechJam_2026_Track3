# Track 3 demo video — shot list v3 (1 Sep, final board)

> **Numbers in this script are the single-artifact board**, measured on
> `c2028c4823ff756b062940e4eff35d5a6a341e9538b755256509cf3432e7794b`, which is
> the file that ships. `Project/BOARD.md` is the source. The tech report, the
> README and this script now carry the same numbers, which was not true of any
> earlier version.
>
> **If you change one number, change it in all four.** A lower-third that
> disagrees with the tech report is the single most damaging thing that could go
> on screen.

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
python3 Project/tools/run_gate.py delta   --campaign CAMP-FINAL ...      # emits a request
python3 Project/harness/trusted_controller.py issue-permit --request … --capability …
python3 Project/harness/trusted_controller.py run --permit permit-… --shape 13 \
        --impl Project/submission/torch_transformer_benchmark_submission.py
```

> **Check before recording:** owner capabilities are short-lived by design. The
> one used for the final measurement pass was a 6-hour, 200-use capability.
> Issuing a permit on camera needs a live one, so mint a fresh short-lived
> capability immediately before you start filming, and check its expiry.

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
for the CPU to queue the next one. So we wrote the whole block as three Triton
kernels: LayerNorm fused into the QKV projection, then FlashAttention-style
causal attention with each head writing into its own slice of a shared buffer,
then one full-width output projection with the FFN finished in-register. Then we
capture the entire forward pass as a single CUDA graph."

**Then the beat that matters:** "But the graph isn't where the win comes from.
One of these shapes has a baseline that's idle only *one* percent of the time.
There are no launch gaps left to remove, and we're still nearly ten times faster
there. Keeping the whole block in registers doesn't recover waiting, it deletes
memory traffic. We only know that because we measured it, and it contradicted
what we'd written in our own report."

**Second honest beat, if there is room:** "The third kernel is named after
splitting the head loop across the GPU, and that part did almost nothing. The
gain came from the side effect: once the heads share one buffer, the output
projection runs at full width instead of being padded up to the tensor core's
16-wide minimum. We measured the mechanism we were proud of at roughly zero and
the side effect at plus twenty-six percent."

**Board on screen:** geomean **11.87×** across the twelve shapes with a runnable
baseline, every one measured on the submission file itself, under a single-use
permit, all passing precision. Best shapes **31.5×** (sequence 1024) and
**29.1×** (`d_model` 32). Mean utilisation across all fourteen: **42.7%**.

**Honest beat:** "The int8 attempt failed the tolerance test and it's in
the repo as a documented negative result. The referee doesn't grade on
effort."

## Scene 5 — the shapes that don't fit, and the close (2:25–3:00)

**Screen:** run both smokes live.

```bash
python3 Project/tools/smokes/shape14_core_smoke.py
python3 Project/tools/smokes/shape6_core_smoke.py
```

> ⚠ **Dry-run these two before you record.** The full evidence evaluators
> (`shape14_eval.py`, `shape6_local_eval.py`) now work: the one-line
> device-comparison bug that used to abort them has been fixed, and both
> produced the shape 6 and shape 14 rows on the shipping artifact. These smokes
> are different files and were never testable from the agent's command
> allowlist, so their status is still unknown. Find out off-camera, not on it.
> If they fail, the scene still works with the packets on screen instead.

**Say:** "Sequence length one hundred thousand, causal, verified against a
chunked fp32 reference with zero tolerance violations across sixteen billion
elements, on the same eight gigabyte card that cannot even hold the baseline. It
runs at eighty-eight point seven percent of what the card can physically do, and
it is ninety-nine point nine percent of all the arithmetic in the benchmark.
Batch ten thousand, where the official baseline runs out of memory, verified
against the batch-chunked official computation. Both on our own machine, by
splitting the computation into blocks, which is what the organizers said they
expected."

**Say the limit out loud, in the same breath:** "Shape fourteen is timed as
thirty-two sequential batch-one calls, not one batch-thirty-two call. The full
hundred-thousand-token sequence is real. The batch dimension is decomposed, and
we label it that way everywhere."

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

> ✅ **All fourteen shapes are measured on one artifact**,
> `c2028c48…`, which is the file that ships. This is the first board where that
> is true. Earlier boards drew rows from up to four different builds and were
> withdrawn for that reason. Use these numbers and no others.

- geomean **11.87×** across the twelve shapes with a runnable baseline, measured
  on the submission file itself
- utilisation is a **range, not a number**, and say why: the organiser has not
  decided the weighting. Equal weight across 14 gives **42.7%**; weighted by
  arithmetic it is **88.6%**; by memory traffic **87.1%**. Same measurements.
  One cause: shape 14 is 99.87% of all the arithmetic and we run it at 88.7%.
  **We optimise against the worst of those and publish all of them.**
- **7.42× against PyTorch's own fused flash attention** — say the caveat in the
  same breath or do not say the number: it was measured on a different build, and
  PyTorch runs at fp32 there while we run fp16, so part of that margin is
  precision rather than kernel engineering
- best shapes **31.5×** (sequence 1024) · **29.1×** (`d_model` 32) · **26.7×**
  (batch 1)
- weakest shapes **2.37×** (`d_model` 1024) · **4.93×** (single attention head).
  Say these out loud if the per-shape board is on screen. The spread is the
  honest story, and the weakest row is weak because its baseline was already at
  30% utilisation.
- correctness: **17,370,759,168 element comparisons, 0 failures**, across 94
  trials
- shape 14: seq 100,000 causal, **48.271 s**, **88.7% of the physical maximum**,
  0 violations, 2.80 GiB peak
- shape 6: batch 10,000, **60.39 ms**, **59.8% of the physical maximum**,
  0 violations, 3.67 GiB peak
- **all 14 shapes pass precision.** Say it precisely if pressed, because the
  evidence is two grades. **Twelve shapes** are verified under the official
  predicate, 7 trials each, five fixed seeds plus two random, `correct: true`,
  on the shipped file. **Shapes 6 and 14** have no runnable official baseline,
  so they are verified against validated references on **5 seeds each**, on that
  same shipped file. Both grades are real. They are not the same grade.

**Say the caveats or cut the number.** These are screening-lane characterisation
runs. Every row is correct and bound to a permit and an artifact hash, but none
is a promoted champion and **no audit verdict is bound to any of them**, because
the audit recorder is broken and only the owner can fix it. The geomean covers
the twelve shapes with a baseline and excludes 6 and 14, which have none.

**One more, if a judge asks about the PyTorch comparison.** Our margin over
PyTorch's own flash attention is real but it is not a clean measurement: it was
taken on a different day on a different build, and PyTorch runs at fp32 there
while our kernels compute at fp16. Both are legal under the precision rules.
Part of that margin is precision rather than kernel engineering, and we say so
in the methodology rather than putting it in the headline.
