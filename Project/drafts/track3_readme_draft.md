# Judge-facing README — v3 (1 Sep, final single-artifact board)

> **Not applied yet.** The owner reviews this and applies it as the public
> repo's README at packaging. It is written to cover every element the
> deliverables list requires: project overview, setup, reproduction steps,
> a reflection on limitations, and contributions. All numbers are the
> single-artifact board in `Project/BOARD.md`.

---

# Transformer Kernel Optimization on a Consumer GPU — TikTok TechJam 2026, Track 3

**An AI agent that writes GPU kernels, and a referee it is not allowed to
touch.**

On an NVIDIA RTX 3060 Ti (consumer, 8 GB), authored Triton/CUDA kernels run
the track's 14 test shapes with a **geometric-mean 11.87× speedup** across
the twelve that have a runnable official baseline, ranging from 2.37× on the
one already near its arithmetic roofline to 31.51× on the longest sequence.

**Every row is measured on one artifact**,
`c2028c4823ff756b062940e4eff35d5a6a341e9538b755256509cf3432e7794b`, which is the
submission file itself, each under a single-use permit bound to that hash. That
matters more than it sounds: earlier versions of this board drew its rows from
four different builds, and a geometric mean over such a board is not a statement
about any one program.

All 14 shapes pass the precision test (`abs_err ≤ 0.002 OR rel_err ≤ 0.02`): the
twelve with a runnable official baseline on 7 trials each under the official
predicate, and shapes 6 and 14 against validated oracles on 5 seeds each, since
no official baseline for those exists on this hardware. That is **94 trials and
17,370,759,168 element comparisons with zero violations**.

The two shapes that cannot run in their official form on this hardware run on
the same card and are verified against exact references. The larger of them,
100,000 tokens at batch 32, reaches **88.7% of the card's physical maximum**.

Across all fourteen shapes weighted equally, mean model FLOPs utilisation is
**42.7%**.

## Project overview

Modern AI-written kernels have a documented failure mode: the optimizer
learns to beat the *timer* rather than the *task*. CUDA-L1's postmortem
found 33% of its reinforcement-learned "solutions" timed work on a side
stream the clock never saw; Sakana's optimizer edited its own evaluator.
We therefore built the measurement system first and gave the kernel-writing
agent no authority over it:

- **Pinned shapes.** The official script's defaults match none of the 14
  test shapes (defaults are batch 8, `d_model` 512, causal *off*). All 14
  live in `Project/shapes.json`; the runner takes a shape id, never loose
  dials, so the agent cannot benchmark an easier problem.
- **A frozen referee.** `Project/harness/runner.py` is hash-pinned and
  protected by tool deny-rules plus a shell guard hook. It re-hashes the
  official benchmark on every run.
- **A tripwire per documented cheat.** Wall-clock cross-check against CUDA
  events (side-stream timing), perturbed fresh-memory re-runs (output
  caching keyed on input address), shape assertions (silently shrinking the
  problem), and for provenance the candidate is snapshotted into a
  **content-addressed blob at permit issue**, with that hash bound into the
  permit, the request and the result packet — so the audited bytes and the
  measured bytes are the same object, not two objects asserted to match.
  (The earlier rule was "commit to git before measuring"; our own auditor
  found that insufficient, because a packet could still cite the current
  source hash rather than the measured one.)
- **A rival AI as auditor.** A new champion fires a blind audit by a
  different model family (GPT-5.6 via Codex CLI) that sees only a neutral
  evidence packet — no commentary from the optimizer — and returns a typed
  verdict. Hard verdicts pause the machine; only the human owner can clear
  them. **81 verdicts are in the ledger and 28 of them are rule violations
  against us** (see the tech report §4, which quotes two that changed the
  code).

  Two honest qualifications on that, both stated rather than buried.

  The auditor backend is a *choice*. When the Codex quota ran out we wired
  Claude as a fallback, and same-family review is a weaker independence claim
  than cross-family review. Every artifact records which backend ran and whether
  it was an independent vendor.

  And **no audit verdict is bound to any row of the results below.** Earlier
  versions of this file said the audit *recorder* was broken. That diagnosis was
  wrong and is corrected here. The recorder is fine and **the auditor never
  starts**: our verdict schema (`Project/audits/verdict_schema.json`, line 70)
  uses `allOf`, which OpenAI's structured-output mode does not permit, so the
  request fails with HTTP 400 before the model reads the packet, and Codex exits
  1 with empty stdout.

  This reproduced live on 1 September. The shape-6 measurement enqueued an audit,
  the watcher launched the auditor three times inside a minute, all three
  returned the same schema error, each failure was written into the hash-chained
  audit journal, and the entry escalated to `owner_attention`. Every part of the
  machinery worked except the schema. The fix is one edit to a file the
  optimizing agent is denied write access to, correctly, since an agent that can
  repair its own auditor does not have one.

The kernels are the deliverable; the governance is the idea. Details of
both are in **[`Project/drafts/tech_report_draft.md`](tech_report_draft.md)**
(the tech report), including the incident in a sibling track that caused us
to rebuild the authority model from scratch.

## Results

### The shipped route, measured under permit

Every figure below was measured under a single-use permit bound to the
candidate's file hash, on an otherwise-idle machine, using the official
predicate. The twelve shapes with a runnable baseline are checked on **seven
trials** each: five fixed seeds plus two drawn at random per run, so a candidate
cannot be tuned to the seed list. Shapes 6 and 14 are checked on five seeds each.

Every trial on every shape passes with **zero failing elements**:

| group | trials | elements compared | violations |
|---|--:|--:|--:|
| twelve primary shapes | 84 | 167,559,168 | 0 |
| shape 6 | 5 | 819,200,000 | 0 |
| shape 14 | 5 | 16,384,000,000 | 0 |
| **total** | **94** | **17,370,759,168** | **0** |

All fourteen rows below were measured on one artifact,
`c2028c4823ff756b062940e4eff35d5a6a341e9538b755256509cf3432e7794b`, which is the
submission file a judge would run. Per-row packet hashes and entry ids are in
`Project/BOARD.md` section 2.

| # | shape | TikTok baseline | **OURS** | **speedup** | **our MFU** | hard floor |
|---|---|--:|--:|--:|--:|--:|
| 1 | B64 s128 d128 h4 | 5.0586 ms | **0.5622 ms** | **8.998×** | 41.1% | 0.2313 ms |
| 2 | B1 s128 d128 h4 | 1.8039 ms | **0.0676 ms** | **26.691×** | 5.3% | 0.0036 ms |
| 3 | B4 s128 d128 h4 | 1.7618 ms | **0.0891 ms** | **19.776×** | 16.2% | 0.0145 ms |
| 4 | B16 s128 d128 h4 | 1.7547 ms | **0.1649 ms** | **10.643×** | 35.1% | 0.0578 ms |
| 5 | B128 s128 d128 h4 | 9.8473 ms | **1.0025 ms** | **9.823×** | 46.1% | 0.4625 ms |
| 6 | B10000 s128 d128 h4 | *out of memory* | **60.3873 ms** | *no baseline* | 59.8% | 36.1355 ms |
| 7 | B64 s128 d32 h4 | 3.4028 ms | **0.1167 ms** | **29.149×** | 17.7% | 0.0206 ms |
| 8 | B64 s128 d1024 h4 | 43.1206 ms | **18.2272 ms** | **2.366×** | 71.1% | 12.9511 ms |
| 9 | B64 s128 d128 h1 | 2.9604 ms | **0.6001 ms** | **4.933×** | 38.5% | 0.2313 ms |
| 10 | B64 s128 d128 h2 | 3.9045 ms | **0.5704 ms** | **6.846×** | 40.5% | 0.2313 ms |
| 11 | B64 s128 d128 h16 | 12.0433 ms | **0.6554 ms** | **18.377×** | 35.3% | 0.2313 ms |
| 12 | B64 s32 d128 h4 | 1.7644 ms | **0.1516 ms** | **11.642×** | 34.1% | 0.0517 ms |
| 13 | B64 s1024 d128 h4 | 169.9159 ms | **5.3919 ms** | **31.513×** | 68.6% | 3.7003 ms |
| 14 | B32 s100000 d1024 h16 | *infeasible* | **48.271 s** | *no baseline* | **88.7%** | 42.808 s |

**Geometric mean over the twelve shapes with a baseline: 11.87×.**
**Mean MFU over all fourteen, weighted equally: 42.7%.**

Twelve shapes route to the fused-block megakernel. Shapes 8 and 14, the two with
`d_model` 1024, route to the fp16 tensor-core stack, a different kernel inside
the same file. Each packet records which route ran.

**Read the spread, not just the mean.** The result ranges from 2.37× to 31.51×.
The win is largest where the baseline is launch-bound. On shapes 2 and 3 the
baseline's GPU sits idle 86% and 83% of the time waiting for the CPU to queue
work. It is smallest where the baseline is already doing real arithmetic: shape
8's baseline reaches 0.30 MFU against 0.002 on shape 2, our kernel takes it to
0.71, and 2.37× is what doubling utilisation looks like. It is also small where
attention is a single well-shaped matmul, as on shape 9 with one head.

A geometric mean over that range is a summary and not a description, so the
per-shape table above is the primary result and the mean is the shorthand.

**MFU is the column that is scored.** The organiser has said the technical score
is a weighted sum of per-shape model FLOPs utilisation, and that the weights are
not yet decided. The speedup column answers "did you beat the reference"; the MFU
column answers "how much of the machine are you using", and only the second one
has a ceiling that does not depend on how slow the reference is.

**Numbers this project has withdrawn, and why.** An earlier board measured
10.32× on the organizers' untouched script. Those runs were **procedurally
invalid**: no permit, no bound audit verdict, and baselines that `HANDOVER.md`
§3.1 records as 6 to 63% slower than their own calibration. We withdrew them and
re-measured everything under the gate.

Worth stating because it cuts against us: the withdrawn board was **not**
inflated. Per shape it scattered from −22.4% to +21.6%, in both directions, and
uncorrelated with baseline device idle. The old numbers were roughly right and
improperly obtained. We report the ones that were properly obtained, which are
now higher than the withdrawn figure rather than lower, because the kernels
improved in between. Procedurally invalid and numerically wrong are different
failures, and only the first one happened.

A second withdrawal matters more, because it is about this board's own
construction. A later 10.6858× geomean was quoted as if it described one program.
It did not: its rows came from four different artifacts. **That is what the
single-artifact requirement above exists to prevent**, and it is why every row
here carries the same hash.

> **Caveats that travel with these numbers.** They are *characterisation* runs in
> a screening lane. Every row is correct and every row is bound to a permit and an
> artifact hash, but none was promoted to champion through the promotion gate, and
> **no audit verdict is bound to any of them**. Our audit recorder broke
> mid-campaign and only the human owner is permitted to repair it, since an agent
> that can repair its own auditor does not have one. The geometric mean covers the
> twelve shapes with a runnable official baseline and excludes shapes 6 and 14,
> which have none on this hardware.

### Shapes 6 and 14 (no runnable official baseline)

Both are measured on the same artifact as every other row.

| | shape 6 (batch 10,000) | shape 14 (sequence 100,000) |
|---|---|---|
| Why the baseline can't run | dense baseline runs out of memory on 8 GB | naive attention table is multi-terabyte on *any* GPU |
| What we ran | full batch, single call | full sequence, causal, 32 serial B=1 calls |
| Measured time | **60.3873 ms** | **48.271 s** |
| Achieved rate | 19.45 TFLOP/s | 28.82 TFLOP/s |
| MFU | **59.8%** | **88.7%** |
| Hard floor | 36.1355 ms | 42.808 s |
| Verified against | batch-chunked official computation, identical math | streamed fp32 oracle, itself validated against the untouched official implementation at 1,024 / 2,048 / 4,096 tokens |
| Seeds | 5 | 5 |
| Violations | 0 (worst abs err 1.43e-3) | 0 (worst abs err 9.83e-4) |
| Elements compared | 819,200,000 | 16,384,000,000 |
| Peak memory | 3.67 GiB allocated | 2.80 GiB allocated, 3.19 GiB reserved |
| Repeat spread | flat across 10 repeats, zero growth | 0.019% across 3 repeats |

**Shape 14 is now measured**, replacing the extrapolation that earlier drafts
reported as pending. It is 99.89% of all the arithmetic in the benchmark and it
is our strongest row, at 88.7% of what the card can physically do.

Two labels travel with both rows. Their timing is **candidate-only**, since there
is no baseline to divide by, so neither has a speedup and neither ever can. And
shape 14's timing is **32 serial batch-1 calls, not one literal batch-32 call**,
which is a decomposition the shape requires on this hardware and which the packet
states in its own `limitation` field.

## What the kernels do

The baseline executes a transformer block as roughly forty separate GPU
operations. Nine of the fourteen shapes are `d_model` 128 with sequence
length 128 — small enough that the GPU finishes each operation before the
CPU can queue the next, so launch overhead, not arithmetic, is the wall.

- **The fused-block megakernel** (ships on 12 of the 14 shapes): an entire
  transformer block in **three** authored Triton kernels.
  `_sub_norm_qkv` fuses LayerNorm into the QKV projection. `_sub_attn_heads`
  runs FlashAttention-style causal attention on a `(q_tiles, batch, heads)`
  grid, writing each head into its own column of a shared context buffer.
  `_sub_attn_block_tail` then does one full-width output projection over the
  assembled context and finishes residual, norm and GELU-FFN in-register. The
  whole multi-layer forward pass is captured as **one CUDA graph** and replayed.

  The third kernel is the most recent structural change and it is worth being
  precise about why it paid. Splitting the head loop into its own grid dimension
  was the stated motivation, and that part did close to nothing, because every
  shape except shape 2 already had a grid wider than the card's 38 SMs. The gain
  came from the other half of the same change: with the heads written into one
  buffer, the output projection runs at **full width** instead of once per head
  padded up to Triton's 16-wide `tl.dot` minimum. On the two shapes with head
  width 8 that was worth +26.4% between them, against +4.3% on the other nine.

- **The fp16 tensor-core stack — shapes 8 and 14**, the two with `d_model` 1024:
  here the block is compute-bound rather than launch-bound, so the win is
  epilogue fusion around the GEMM boundaries and not launch elimination.
  Measured on shape 8: the baseline reaches **30.0% MFU**, against 0.2% on shape
  2, and our kernel takes it to **71.1%**. The 2.37× is what more than doubling
  an already-decent utilisation looks like, and it is the smallest speedup on the
  board for that reason. The same route carries shape 14 to **88.7%**, the
  highest utilisation we reach anywhere.
- **The oversized shapes are handled inside the shipped file, not by separate
  kernels.** `dispatcher_region.py` carries an exact batch-chunked path for
  shapes where dense attention is feasible but the whole batch is not, and a
  sequence-chunked tail with row-wise exact math for shape 14. `k014` and `k015`
  were the development kernels where those ideas were worked out; what ships is
  the integrated version. Shape 6 runs as **one call at batch 10,000**. Shape 14
  is called 32 times at batch 1 by its evaluator, with the 100,000-token
  sequence handled inside each call.
- **Dispatch**: one `UserOptimizedTransformer` inspects the incoming shape and
  routes, which is the mechanism the track explicitly permits. The route taken is
  recorded in every evidence packet, so the claim is checkable rather than
  asserted.

No external kernel library is wrapped, with no FlashAttention package and no
xFormers. The kernels are authored. `torch.compile` and SDPA appear only as
correctness fallbacks and as a measured comparison: `max-autotune` reached
7.00× / 3.10× / 1.23× on the shape-3 / 13 / 8 dials, against our current
19.78× / 31.51× / 2.37×.

Two caveats on that comparison, because it is not like-for-like. The
`torch.compile` figures were measured on 29 August on an earlier build, so the
two halves come from different artifacts. And our route computes in fp16 with
fp32 accumulation while `torch.compile` and SDPA run the model in fp32 with
TF32 matmul, so part of the margin is precision rather than kernel engineering.
Both clear the competition's precision requirement, so both are legal, but the
difference has to be named. See `Project/MEASUREMENT_METHODOLOGY.md` §7.3.

## Setup and installation

```bash
git clone <repo-url> && cd Tiktok_TechJam_2026_Track3
python3 -m pip install torch triton     # torch 2.12.0+cu130, triton 3.7.0
nvidia-smi                              # CUDA 13.0, driver 610.57.04 used here
```

Requires an NVIDIA GPU with compute capability ≥ 8.0. Everything here was
developed and measured on an RTX 3060 Ti (sm_86, 8 GB); kernels tuned for
this card's 99 KB shared-memory-per-block limit will need re-tuning on a
datacenter card, and autotuned configs from big GPUs will not run here.

## Steps to reproduce our results

```bash
# 1. The submission itself — the official script with ONLY the sanctioned
#    UserOptimizedTransformer region replaced. Any shape's dials work.
python3 Project/submission/torch_transformer_benchmark_submission.py \
        --batch-size 64 --seq-len 1024 --d-model 128 --heads 4 \
        --ffn-dim 128 --layers 4 --causal

# 2. Prove that everything outside the replaced region is byte-identical
#    to the official script.
python3 Project/tools/build_submission.py --verify

# 3. Any shape through our frozen referee. NOTE: since the LOCK, runner.py is
#    a shim onto the trusted controller and will NOT time anything without a
#    one-use permit -- that refusal is the design, not a bug:
#      error: the following arguments are required: --permit
#    The full permitted sequence (request -> permit -> run) is in
#    Project/RUNBOOK.md; the request step needs an owner-signed capability.
python3 Project/harness/runner.py run --shape 13 \
        --impl Project/kernels/k009_fused_tuned.py   # refuses, by design

# 4. Regenerate the score-sensitivity board. (VERIFIED 31 Aug: this one works.)
python3 Project/tools/sensitivity_board.py        # -> Project/results_side/SENSITIVITY.md

#    NOTE: `runner.py leaderboard` no longer exists -- the LOCK replaced runner.py
#    with a shim onto the trusted controller, which has no such subcommand. The
#    post-LOCK speedup board is NOT in Project/results/JOURNAL.jsonl; see below.

# 5. The two shapes that don't fit in 8 GB.
python3 Project/tools/smokes/shape14_core_smoke.py
python3 Project/tools/smokes/shape6_core_smoke.py
```

**Where the published numbers actually live** (corrected 31 Aug after
checking, rather than asserted):

- The **speedup board** — every row of both tables above — is in the
  controller's append-only authority log, `Project/authority/events.jsonl`,
  as `measurement_recorded` events, each carrying a content-addressed
  measurement packet under `Project/authority/blobs/<packet_sha>.json` with
  the full 300-sample baseline and candidate distributions, the permit id,
  the candidate sha256 and the environment. The scientific record of *why*
  each run happened is `Project/loop/gate_log.jsonl`.
- `Project/results/JOURNAL.jsonl` holds the **pre-LOCK** history. The
  post-LOCK board is deliberately not in it: screening-lane runs write to
  the scratch namespace, not the primary journal, which is what stops a
  characterisation run from being mistaken for a champion.
- Side-evaluator packets for shapes 6 and 14 are in `Project/results_side/`.

Each entry records the GPU, driver, CUDA, torch, Triton, dtype, TF32 flags,
code hash and harness version, so only like-for-like profiles are compared.

## Repository map

| path | what it is |
|---|---|
| `Project/kernels/` | every candidate implementation, including the failures |
| `Project/harness/runner.py` | the frozen referee (hash-pinned, protected) |
| `Project/submission/` | the single-file submission + byte-identity prover |
| `Project/results/` | **pre-LOCK history only.** `JOURNAL.jsonl` is the append-only pre-gate journal. `LEADERBOARD.md` is a frozen pre-LOCK artifact that stars the max-ever row per shape across invocations that are not comparable, so it is **not a result and must not be quoted as one** — see the reflection section. The post-LOCK board is in `Project/authority/`. |
| `Project/results_side/` | evidence packets for shapes 6 and 14, score-sensitivity board |
| `Project/audits/` | verdict ledger, evidence packets, review prompts |
| `Project/research/` | source-of-truth research notes every proposal must cite |
| `Project/loop/` | the experiment-gate design and its honesty ledger |
| `Project/memory/` | decisions, lessons, running state |

## Reflection: limitations, and what we would improve

**Limitations we are explicit about.**

- Shape 14 is timed as **32 serial batch-1 calls, not one literal batch-32
  call**. The full 100,000-token sequence is real and so is the 48.271 s, but
  the batch dimension is decomposed, which the packet states in its own
  `limitation` field. Earlier drafts reported this row as pending an
  extrapolation. It is now measured, and the extrapolation was never used.
- **Shapes 6 and 14 use CPU RNG in their evaluators**, so their inputs are not
  bit-identical to a default judge run. Their packets are bound to the shipping
  artifact and carry 5 seeds each, which resolves an earlier limitation where
  they were one seed and cited a pre-integration file. The one-line device
  comparison bug that blocked that re-capture has been fixed by the owner.
- **Shape 14's correctness reference is one we wrote.** TikTok's baseline cannot
  run at 100,000 tokens, so we compute a chunked reference performing identical
  mathematics in bounded memory, and validate that reference against their real
  baseline at 1,024, 2,048 and 4,096 tokens, where theirs still fits. That
  validation is a separate check with its own limit, and both it and the
  kernel's own correctness check pass. Full detail in
  `Project/MEASUREMENT_METHODOLOGY.md` §7.1 and §7.2.
- Sub-millisecond shapes are noisy on a consumer card, and we have four
  separate measurements of how noisy, which are worth keeping apart. *Within*
  one invocation, baseline-against-itself calibration noise is **0.03 to 0.4%**,
  and that is what sets each shape's promotion threshold. *Across* invocations
  of identical work, GPU clock state alone moves absolute time by about **9%**.
  Measuring the shipped file against its own kernel module in separate
  invocations gave **−10.0% to +2.6%** per shape. And the worst case we have
  observed, **byte-identical code measured 13.2% apart minutes apart on shape
  12**, is the number that actually bounds what a per-shape difference can mean.
  The calibration figure is the misleading one: it measures second-to-second
  steadiness inside one process, not run-to-run reproducibility, and it is
  roughly two orders of magnitude too small for that job. The older "±25%"
  figure came from comparing two uncontrolled pre-gate boards and should not be
  quoted for these. Each published row is the **median of 300 paired samples
  inside a single invocation**. We never average across invocations, because
  absolute latencies are not comparable across processes.
- The published speedup board was measured **after** the enforcement gate
  went live — every row carries a one-use permit bound to the candidate's
  file hash — but in the **screening lane**, which cannot promote. So these
  are characterisation runs, not promoted champions, and no audit verdict is
  bound to them: our audit recorder failed mid-campaign and repairing it
  requires the human owner, who alone can write to those files. We would
  rather ship a correctly-labelled measurement than an over-claimed one.
- An earlier version of this README published a 10.32× geometric mean from runs
  that had no permit and used baselines 6 to 63% off their own calibration. We
  withdrew it and re-measured under the gate. A later version published 10.6858×
  as if it described one program, when its rows came from four different
  artifacts. Both corrections are in the git history rather than quietly
  overwritten, because the process failures are more interesting than the
  numbers. The board in this file is the first that is one artifact throughout,
  and it is higher than either withdrawn figure, because the kernels improved in
  between rather than because the accounting changed.
- **`Project/results/LEADERBOARD.md` is in the repository and is not a result.**
  It is a frozen pre-LOCK artifact, last generated 30 Aug, by a command that no
  longer exists. It stars the max-ever row per shape across invocations that are
  not comparable, which selects for whichever run happened to have the slowest
  baseline. Its own shape-1 rows show the effect: the starred k009 run reads
  11.150× on a 7.2044 ms baseline, while another k009 run reads 9.910× on a
  5.7539 ms baseline, and our measured baseline is 5.0586 ms. We keep the file
  because deleting inconvenient history is worse than labelling it, but nothing
  in it should be read as a claim.
- One GPU, one architecture, one framework (PyTorch). Nothing here is
  validated on the TensorFlow path.
- On a single-user machine, our anti-tamper measures are
  forge-*obvious*, not forge-*proof*. We state the ceiling rather than
  implying we exceeded it.

**What we would do with more time.**

1. **The launch-bound family (shapes 2, 3, 7, 12)** measures 5.3%, 16.2%, 17.7%
   and 34.1% MFU, the four lowest on the board, because the grid cannot fill
   38 SMs and per-call cost does not shrink with problem size. A
   sequence-persistent kernel is the honest next step. Published results for
   that class suggest around 1.2×, which is why our own score-sensitivity board
   ranks it below the extreme shapes.
2. **Profiler-in-the-loop.** Diagnosis is currently a human-readable
   research note. The agent should read hardware counters directly and
   prescribe from them.
3. **Runner-internal permit checks.** Enforcement currently lives in a
   shell guard and tool deny-rules; putting it inside the referee needs a
   re-freeze we deliberately deferred until after the competition.
4. **TensorFlow parity** for the second benchmark path.

## Development tools, APIs, libraries

- **Claude Code** (terminal agent) — Claude Fable 5, later Claude Opus 5:
  authored the kernels, the harness and the process machinery.
- **OpenAI Codex CLI** — GPT-5.6, reasoning effort high/ultra: blind
  champion audits and blind strategy reviews.
- **PyTorch 2.12**, **Triton 3.7**, CUDA 13.0, VS Code, git.
- No external datasets or APIs — the benchmark generates its own tensors.

## Contributions

Solo entry. The human owner set every rule, held sole authority over
freezes, promotions and everything that shipped, and made the calls the
agents were structurally forbidden from making. The AI agents wrote
kernels and audited each other under that authority.
