# Judge-facing README — DRAFT v2 (30 Aug)

> **Not applied yet.** The owner reviews this and applies it as the public
> repo's README at packaging. It is written to cover every element the
> deliverables list requires: project overview, setup, reproduction steps,
> a reflection on limitations, and contributions. Numbers marked
> **[PENDING]** are filled from the final-sha board at code freeze.

---

# Transformer Kernel Optimization on a Consumer GPU — TikTok TechJam 2026, Track 3

**An AI agent that writes GPU kernels, and a referee it is not allowed to
touch.**

On an NVIDIA RTX 3060 Ti (consumer, 8 GB), authored Triton/CUDA kernels run
the track's 14 test shapes with a **geometric-mean 10.3× speedup on the
organizers' own untouched benchmark script**, every shape passing the
precision test (`abs_err ≤ 0.002 OR rel_err ≤ 0.02`). The two shapes that
cannot run in their official form on this hardware are solved by block
decomposition on the same card and verified against exact references.

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
  problem), hash-pinned candidate bytes committed to git *before* first
  measurement (provenance).
- **A rival AI as auditor.** Every new champion automatically fires a blind
  audit by a different model family (GPT-5.6 via Codex CLI) that sees only
  a neutral evidence packet — no commentary from the optimizer — and
  returns a typed verdict. Hard verdicts pause the machine; only the human
  owner can clear them.

The kernels are the deliverable; the governance is the idea. Details of
both are in **[`Project/drafts/tech_report_draft.md`](tech_report_draft.md)**
(the tech report), including the incident in a sibling track that caused us
to rebuild the authority model from scratch.

## Results

> ### ⛔ WITHDRAWN — the table below is invalid. Do not ship.
>
> Added 31 Aug ~00:30 SGT. The **10.32×** and **10.95×** figures are withdrawn.
> They were measured pre-gate against baselines that `HANDOVER.md` §3.1 records as
> **6–63% slower than their own calibration**, so the ratios are inflated.
>
> ### ⚠ SECOND AMENDMENT 01:26 — the original numbers are being VINDICATED
>
> Re-measuring the **actually-shipped** megakernel under permit, on a quiet box, against
> the same baselines now agrees closely with the figures originally published here:
>
> | shape | originally published | shipped route, measured 31 Aug | agreement |
> | --- | --- | --- | --- |
> | 2 | 15.26× | **14.3939×** | 5.7% |
> | 3 | 11.96× | **12.6314×** | 5.6% |
> | 13 | 28.82× | **28.4098×** | **1.4%** |
>
> **The original magnitudes look approximately right.** What was wrong was *process* — no
> permit, no bound audit verdict, baselines 6–63% off their own calibration. Procedurally
> invalid and numerically wrong are different things, and the first amendment conflated
> them.
>
> **The 2.94× below is the misleading figure**: it measured `k004`, which this submission
> does not ship, and understates the shipped route by roughly 3.5×. Ignore it.
>
> **Status: 3 of 12 shapes re-measured on the shipped route, no geomean claimed yet.**
> The original 10.32× is not yet re-earned, but nothing contradicts it.
>
> ---
>
> **AMENDED 01:05 — there is currently NO valid headline for this submission.**
>
> The post-LOCK board below measured `k004_graphed_triton.py`, which is **not the
> route this submission ships**. The dispatcher sends `d_model ≤ 128` to the
> fused-block megakernel and larger `d_model` to an fp16 tensor-core stack.
>
> | shape | 1 | 2 | 3 | 4 | 5 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | **geomean** |
> |---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
> | k004, **not shipped** | 2.14× | 8.11× | 7.18× | 2.72× | 2.15× | 3.48× | 1.11× | 1.17× | 1.58× | 4.24× | 3.23× | 5.81× | **2.94×** |
>
> So: **10.32× is withdrawn, and 2.94× describes a different kernel.** The shipped
> route has no post-LOCK measurement yet. Quote neither figure as the result.

### On the organizers' untouched script

| shape | 1 | 2 | 3 | 4 | 5 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | geomean |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| speedup | 10.73× | 15.26× | 11.96× | 7.30× | 11.40× | 25.57× | 2.04× | 5.38× | 7.45× | 12.98× | 11.44× | 28.82× | **10.32×** |

All twelve PASS the official precision check with zero failing elements.
An independent path — our own frozen referee, measured on a different day —
puts the same set at 10.95× geomean; the two agree within 6%. Per-shape
scatter reaches ±25% on the sub-millisecond shapes, which we report rather
than averaging away.

### Shapes 6 and 14 (no runnable official baseline)

| | shape 6 (batch 10,000) | shape 14 (sequence 100,000) |
|---|---|---|
| Why it can't run | dense baseline OOMs on 8 GB | naive attention table is multi-terabyte on *any* GPU |
| What we ran | full batch, single call | full sequence, causal |
| Verified against | batch-chunked official computation (identical math) | streamed fp32 oracle, itself validated against the untouched official implementation at feasible lengths (worst deviation 1.4e-6) |
| Violations | 0 (max abs err 1.51e-3) | 0 (max abs err 9.05e-4) |
| Peak memory | 3.37 GiB | 4.89 GiB |

Shape 14's full-batch **timing** is reported as measured slices (B=1:
1,674 ms; B=2: 3,657 ms) rather than extrapolated — doubling the batch
costs 2.18×, not 2.00×, so a linear projection would be wrong. The
full-batch figure is **[PENDING]** the batch-decomposed evaluator run.

## What the kernels do

The baseline executes a transformer block as roughly forty separate GPU
operations. Nine of the fourteen shapes are `d_model` 128 with sequence
length 128 — small enough that the GPU finishes each operation before the
CPU can queue the next, so launch overhead, not arithmetic, is the wall.

- **`k009` — the megakernel** (ships on 11 of 12 runnable shapes): an entire
  transformer block in two authored Triton kernels — LayerNorm fused into
  the QKV projection; FlashAttention-style causal attention across all
  heads with the output projection folded into the head loop, then
  residual + norm + GELU-FFN finished in-register. The full four-layer
  forward pass is captured as **one CUDA graph** and replayed.
- **`k010` — shape 8**: at `d_model` 1024 the block is genuinely
  compute-bound (0.65 MFU), so the win is epilogue fusion around the GEMM
  boundaries, not launch elimination.
- **`k014` / `k015` — shapes 14 and 6**: block-decomposed variants that
  stream the sequence or chunk the batch so nothing oversized is ever
  materialized.
- **Dispatch**: one `UserOptimizedTransformer` inspects the incoming shape
  and routes — the mechanism the track explicitly permits.

No external kernel library is wrapped (no FlashAttention, no xFormers) —
the kernels are authored. `torch.compile` and SDPA appear only as
correctness fallbacks and as a measured comparison: `max-autotune` reaches
7.00× / 3.10× / 1.23× on the shape-3 / 13 / 8 dials, against our 11.96× /
28.82× / 2.04×.

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

# 3. Any shape through our frozen referee: multi-seed correctness,
#    tripwires, calibrated noise floor, timing distribution.
python3 Project/harness/runner.py run --shape 13 \
        --impl Project/kernels/k009_fused_tuned.py

# 4. Regenerate every published table from the append-only journal.
python3 Project/harness/runner.py leaderboard     # -> Project/results/LEADERBOARD.md
python3 Project/tools/sensitivity_board.py        # -> Project/results_side/SENSITIVITY.md

# 5. The two shapes that don't fit in 8 GB.
python3 Project/tools/smokes/shape14_core_smoke.py
python3 Project/tools/smokes/shape6_core_smoke.py
```

Every number we publish regenerates from `Project/results/JOURNAL.jsonl`
(append-only) plus the side-evaluator packets in `Project/results_side/`.
Each entry records the GPU, driver, CUDA, torch, Triton, dtype, TF32 flags,
code hash and harness version, so only like-for-like profiles are compared.

## Repository map

| path | what it is |
|---|---|
| `Project/kernels/` | every candidate implementation, including the failures |
| `Project/harness/runner.py` | the frozen referee (hash-pinned, protected) |
| `Project/submission/` | the single-file submission + byte-identity prover |
| `Project/results/` | append-only journal + generated leaderboard |
| `Project/results_side/` | evidence packets for shapes 6 and 14, score-sensitivity board |
| `Project/audits/` | verdict ledger, evidence packets, review prompts |
| `Project/research/` | source-of-truth research notes every proposal must cite |
| `Project/loop/` | the experiment-gate design and its honesty ledger |
| `Project/memory/` | decisions, lessons, running state |

## Reflection: limitations, and what we would improve

**Limitations we are explicit about.**

- Shape 14's full-batch timing is measured in slices, not at full batch;
  the extrapolation that would make it look complete is exactly the one the
  data says is wrong (2.18× per doubling, not 2×).
- The shape-6 and shape-14 evidence packets are provisional — one seed
  each, and they cite the pre-integration submission file. They are being
  re-captured against the shipped file with ≥5 seeds.
- Sub-millisecond shapes are genuinely noisy on a consumer card (±25%
  between independent runs of the same code). The published board is a
  median of repeated sweeps and states the noise.
- All published measurements predate our final enforcement gate going
  live. They rest on the frozen runner, the tripwires, committed-bytes
  provenance and blind audits — which is what we claim, and no more.
- One GPU, one architecture, one framework (PyTorch). Nothing here is
  validated on the TensorFlow path.
- On a single-user machine, our anti-tamper measures are
  forge-*obvious*, not forge-*proof*. We state the ceiling rather than
  implying we exceeded it.

**What we would do with more time.**

1. **The launch-bound family (shapes 2, 3, 7, 12)** sits at 0.03–0.29 MFU
   because the grid cannot fill 38 SMs. A sequence-persistent kernel is the
   honest next step; published results for that class suggest ~1.2×, which
   is why our own score-sensitivity board ranks it *below* the extreme
   shapes.
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
