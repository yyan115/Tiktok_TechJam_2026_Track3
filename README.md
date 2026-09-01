# Cryptographically Signed Agent Loop for GPU Kernels

**TikTok TechJam 2026, Track 3.** An AI agent writes the GPU kernels. A separate
locked program measures them, and the agent has no way to change it.

This project combines modern kernel-agent techniques with cryptographic signing
and loop engineering, so the agent doing the optimising stays on task and cannot
fake its own results. Every number below came out of a locked program the agent
cannot edit, using a single-use permit from a run budget only a human can sign.

**Run on one NVIDIA RTX 3060 Ti — a consumer desktop card with 8 GB.**

- **11.87× geometric-mean speedup** over the twelve of fourteen shapes that have
  a runnable official baseline — from 2.37× on the shape already near its
  arithmetic limit to **31.51×** on the longest sequence.
- **Shape 14 reaches 88.7% of what the card can physically do**: 100,000 tokens
  at batch 32, a shape the official code cannot run at all on this hardware.
- **We beat PyTorch's own `sdpa` on every shape**, by 2.3× to 20.9× — and on the
  two extreme shapes it cannot run at all while ours does.
- **The one shape where we had to write our own reference, we held to twice the
  competition's strictness.** Shape 14 is too large for TikTok's baseline to run
  at all, so we wrote a streamed reference and validated it against their real
  implementation at 1,024, 2,048 and 4,096 tokens at **1e-3 absolute — twice as
  strict as the competition's 2e-3**.
- **94 correctness trials, 17,370,759,168 elements compared, zero violations.**
- **Mean utilisation is 42.7%** with every shape weighted equally. The organiser
  has not published the weighting, and the same fourteen measurements give
  **35.5% to 88.6%** depending on which rule is used — we published all five and
  optimised against the low end.

**All fourteen rows come from running the same single file** — the exact one we
are submitting, `c2028c4823ff756b062940e4eff35d5a6a341e9538b755256509cf3432e7794b`.
Not fourteen scripts, and not a different version of our code for each shape.

So the 11.87× above describes the file you would actually run, rather than being
an average across several versions of it.

On correctness we went past what was asked. The official script checks five
fixed seeds; we check **seven — five fixed plus two drawn at random per run**,
so a kernel cannot be tuned to the seed list. The twelve shapes with a runnable
baseline are checked that way under the official predicate
(`abs_err ≤ 0.002 OR rel_err ≤ 0.02`); shapes 6 and 14 are checked against
validated references on five seeds each, since no official baseline for them
exists on this hardware.

## Project overview

Modern AI-written kernels have a documented failure mode: the optimizer
learns to beat the *timer* rather than the *task*. CUDA-L1's postmortem
found 33% of its reinforcement-learned "solutions" timed work on a side
stream the clock never saw; Sakana's optimizer edited its own evaluator.

We therefore built the measurement system first and gave the kernel-writing
agent no authority over it. **The agent cannot measure its own work.** It can
edit the first box below and nothing else, and every stage after the permit is
hash-pinned under a signature whose private key only the human owner holds.

```
  AI writes  ─▶  PERMIT      ─▶  LOCKED     ─▶  SANDBOX    ─▶  EVIDENCE     ┄▶  AUDIT
  the kernel     one-use,        REFEREE        no network,    300 samples,     GPT-5.6,
  Triton/CUDA    bound to        29 files,      no home,       every trial,     packet only,
                 its sha256      Ed25519        read-only      hash-chained     typed verdict
                     ▲
                     │ the owner signs ONE capability worth N runs; the system
                     │ spends it one permit at a time and stops dead at zero
  └ AI edits ┘   └──────── AI has no key and no write access, ever ────────┘
```

**272 permits issued, 271 consumed.** The audit leg is dotted because a schema
bug stopped it running — see the limitations.

The pieces:

- **Pinned shapes.** All 14 live in `Project/shapes.json`. The runner takes a
  shape id, never loose dials, so the agent cannot benchmark an easier problem.
- **A frozen referee.** The trusted controller re-hashes all 29 protected files,
  the official benchmark among them, before every single run and refuses
  everything on one mismatch.
- **A tripwire per known cheat.** Wall-clock cross-checked against CUDA events
  catches side-stream timing; fresh-memory re-runs catch output caching; and the
  request's shape must match `shapes.json` exactly, with `shapes.json` itself
  hash-checked, so the problem cannot be quietly shrunk.
- **Provenance.** The candidate is copied into a content-addressed blob when its
  permit is issued, and that hash goes into the permit, the request and the
  result packet. The bytes that get audited are the same bytes that got measured.
- **A second model as auditor.** A different model family (GPT-5.6 via Codex
  CLI) gets only the evidence packet and returns a typed verdict. Hard verdicts
  pause the machine; only the owner can clear them. **81 verdicts, 28 of them
  rule violations against us**, in `Project/audits/verdicts.jsonl`. Those are
  from earlier in the campaign; none is bound to the final board, for the reason
  in the limitations below.

### The loop: research, plan, three strikes

The agent could not run the benchmark whenever it wanted. Before each attempt it
had to do four things:

1. **Research.** Must cite at least two existing notes from `Project/research/`
   and carry the current hash of the index, which proves it was read this cycle.
2. **Plan.** Refused unless research happened this cycle. Needs a hypothesis, a
   **numeric** prediction, kill criteria, and citations as `file:line` — the
   gate looks each one up and copies the quoted text into the log, so a made-up
   citation gets caught.
3. **One run.** The gate shuts the moment it executes. Whether it improved is
   read from the referee's own journal, never from the agent's claim.
4. **Three strikes.** Three attempts with no improvement closes off that
   approach. The agent then has to write up what it predicted, what actually
   happened, and what that rules out, before it can start on a different one.

`Project/memory/` keeps the decisions and lessons between sessions, so the same
mistake does not get repeated later.

The kernels are what we are submitting. The system around them is the part we
think is worth showing. We built it after a sibling track went wrong: the agent
there controlled its own measurement, and we had to throw the numbers away.

Link to the full project technical breakdown is here:
**[`Project/TECHNICAL_BREAKDOWN.md`](Project/TECHNICAL_BREAKDOWN.md)** — how the
signing, the permits, the sandbox, the gate and the audit actually work, with the
file and line number for every claim.

## Results

### The shipped route

Every figure below was measured on an idle machine under a permit tied to that
exact file, using the official correctness test. Every trial on all fourteen
shapes passes, with **zero failing elements**:

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

| # | shape | TikTok baseline | PyTorch sdpa † | **OURS** | **speedup** | vs sdpa † | **our MFU** | hard floor |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | B64 s128 d128 h4 | 5.0586 ms | 1.67× | **0.5622 ms** | **8.998×** | 5.4× | 41.1% | 0.2313 ms |
| 2 | B1 s128 d128 h4 | 1.8039 ms | 1.28× | **0.0676 ms** | **26.691×** | 20.9× | 5.3% | 0.0036 ms |
| 3 | B4 s128 d128 h4 | 1.7618 ms | 1.34× | **0.0891 ms** | **19.776×** | 14.8× | 16.2% | 0.0145 ms |
| 4 | B16 s128 d128 h4 | 1.7547 ms | 1.39× | **0.1649 ms** | **10.643×** | 7.7× | 35.1% | 0.0578 ms |
| 5 | B128 s128 d128 h4 | 9.8473 ms | 1.66× | **1.0025 ms** | **9.823×** | 5.9× | 46.1% | 0.4625 ms |
| 6 | B10000 s128 d128 h4 | *out of memory* | *cannot run* | **60.3873 ms** | *no baseline* | **runs** | 59.8% | 36.1355 ms |
| 7 | B64 s128 d32 h4 | 3.4028 ms | 2.18× | **0.1167 ms** | **29.149×** | 13.4× | 17.7% | 0.0206 ms |
| 8 | B64 s128 d1024 h4 | 43.1206 ms | 1.02× | **18.2272 ms** | **2.366×** | **2.3×** | 71.1% | 12.9511 ms |
| 9 | B64 s128 d128 h1 | 2.9604 ms | 1.11× | **0.6001 ms** | **4.933×** | 4.4× | 38.5% | 0.2313 ms |
| 10 | B64 s128 d128 h2 | 3.9045 ms | 1.31× | **0.5704 ms** | **6.846×** | 5.2× | 40.5% | 0.2313 ms |
| 11 | B64 s128 d128 h16 | 12.0433 ms | 2.59× | **0.6554 ms** | **18.377×** | 7.1× | 35.3% | 0.2313 ms |
| 12 | B64 s32 d128 h4 | 1.7644 ms | 1.27× | **0.1516 ms** | **11.642×** | 9.2× | 34.1% | 0.0517 ms |
| 13 | B64 s1024 d128 h4 | 169.9159 ms | 3.97× | **5.3919 ms** | **31.513×** | 7.9× | 68.6% | 3.7003 ms |
| 14 | B32 s100000 d1024 h16 | *infeasible* | *cannot run* | **48.271 s** | *no baseline* | **runs** | **88.7%** | 42.808 s |

**Geometric mean over the twelve shapes with a baseline: 11.87×.**
**Geometric mean against PyTorch's sdpa: 7.42×**, with the † caveats below.

### The score depends on a weighting the organiser has not decided

Technical Execution is a weighted sum of per-shape MFU, and the organiser has
said the weights are still open and that bandwidth will be considered. The same
fourteen measurements, combined five ways:

| how the 14 shapes are combined | our score | what it rewards |
|---|--:|---|
| geometric mean, all 14 | **35.5%** | punishes the worst shape hardest |
| **equal weight, all 14** | **42.7%** | every shape counts the same |
| equal weight, the 12 with a baseline | 37.5% | excludes the two extreme shapes |
| **bandwidth-weighted, all 14** | **87.1%** | shapes moving more bytes count more |
| **FLOP-weighted, all 14** | **88.6%** | shapes doing more arithmetic count more |

**35.5% to 88.6% on identical numbers.** The cause is one shape: shape 14 is
99.87% of all the arithmetic in the benchmark and 94.4% of the minimum bytes
moved, and we reach 88.7% on it. So any work-proportional weighting is close to a
report of shape 14 alone, while any per-shape weighting is dominated by shape 2
at 5.3%, which occupancy caps near 21% for any implementation.

**We optimise against equal weight.** It is not the lowest of the five — the
geometric mean is harsher at 35.5% — but it is the reading that keeps pressure on
the small shapes where the headroom actually is, and it sits far below the
work-proportional weightings that would flatter us most. We publish all five
because the rule is not ours to pick.

**† The two sdpa columns are rougher than the rest of the table.** We measured
sdpa on 28 August on an older build, so those columns are our speedup divided by
theirs rather than a head-to-head run. sdpa also runs in fp32 while ours runs in
fp16, so some of the gap is precision and not kernel work. Both pass the
competition's precision test. The main `speedup` column has neither problem —
same process, same input, same build.

The sdpa columns are there to answer one question: is 11.87× about our kernels,
or about a slow reference? PyTorch's own optimised attention only reaches 1.02×
to 3.97× on these shapes, so the reference is not the explanation. And on shapes
6 and 14 it cannot run at all.

Twelve shapes route to the fused-block megakernel. Shapes 8 and 14, the two with
`d_model` 1024, route to the fp16 tensor-core stack, a different kernel inside
the same file. Each packet records which route ran.

**The spread matters as much as the mean.** The result ranges from 2.37× to 31.51×.
The win is largest where the baseline is launch-bound. On shapes 2 and 3 the
baseline's GPU sits idle 86% and 83% of the time waiting for the CPU to queue
work. It is smallest where the baseline is already doing real arithmetic: shape
8's baseline reaches 0.30 MFU against 0.002 on shape 2, our kernel takes it to
0.71, and 2.37× is what doubling utilisation looks like. It is also small where
attention is a single well-shaped matmul, as on shape 9 with one head.

One average over that range hides a lot, so the table is the real result and the
mean is shorthand.

**MFU is the column that gets scored.** The organiser has said the technical
score is a weighted sum of per-shape MFU. Speedup tells you whether we beat the
reference; MFU tells you how much of the card we are actually using, and that
number does not move just because the reference is slow.

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

Shape 14 is 99.87% of all the arithmetic in the benchmark, and it is our best
row at 88.7% of what the card can physically do.

Two things to be clear about. Neither row has a speedup, because there is no
baseline to divide by. And shape 14 runs as **32 serial batch-1 calls, not one
batch-32 call** — the whole thing does not fit on 8 GB at once. The packet says
so in its own `limitation` field.

## Setup and installation

```bash
git clone <repo-url> && cd Tiktok_TechJam_2026_Track3

# Kernels and benchmark.
python3 -m pip install torch triton

# Required by the harness. The trusted controller verifies an Ed25519-signed
# LOCK over 29 files before it will run anything, so the LOCK check and
# reproduction step 3 below both need this.
python3 -m pip install cryptography

nvidia-smi
```

Every number in this README was measured on **torch 2.12.0+cu130, triton 3.7.0,
CUDA 13.0, driver 610.57.04**. Note that `pip install torch` gives you whichever
CUDA build matches your driver, which is not necessarily cu130; pin from
PyTorch's own wheel index if you need to match our environment exactly.

Requires an NVIDIA GPU with compute capability ≥ 8.0. Everything here was
developed and measured on an RTX 3060 Ti (sm_86, 8 GB); kernels tuned for
this card's 99 KB shared-memory-per-block limit will need re-tuning on a
datacenter card, and autotuned configs from big GPUs will not run here.

## Steps to reproduce our results

Run these from the repository root, in order. All four work on a fresh clone.

```bash
# 1. Run the submission. This is TikTok's own script with only the sanctioned
#    UserOptimizedTransformer region replaced, so THEIR code does the timing
#    and prints the speedup. These are shape 13's dials; every shape's dials
#    are in Project/shapes.json.
python3 Project/submission/torch_transformer_benchmark_submission.py \
        --batch-size 64 --seq-len 1024 --d-model 128 --heads 4 \
        --ffn-dim 128 --layers 4 --causal

# 2. Prove everything outside that region is byte-identical to the official
#    script. Prints "verified": true and the submission's sha256, which is the
#    c2028c48... hash every number in this README is measured on.
python3 Project/tools/build_submission.py --check-only

# 3. Prove the measurement system itself was never edited. 29 files are pinned
#    by hash under an Ed25519 signature. Prints "valid": true, "active": true.
python3 Project/harness/trusted_controller.py verify-lock

# 4. Run the lock's own test suite, including forged-signature and
#    mutated-byte cases. Prints ALL GREEN.
python3 Project/tools/tests/lock_manifest_test.py
```

Step 1 needs a CUDA GPU. Steps 2 to 4 run anywhere Python and `cryptography`
are installed, including a machine with no GPU at all.

**The referee will refuse you, and that is the point.** `Project/harness/runner.py`
is a shim onto the trusted controller and will not time anything without a
single-use permit:

```
$ python3 Project/harness/runner.py run --shape 13 --impl Project/kernels/k009_fused_tuned.py
trusted_controller.py run: error: the following arguments are required: --permit
```

Issuing a permit needs an owner-signed capability, so nobody who clones this
repository can produce one — which is exactly the property the whole design
exists to have. The full sequence is in `Project/RUNBOOK.md`. Shapes 6 and 14
are gated the same way, through `run_gate.py side-evaluate`; their evidence
packets are already in `Project/results_side/`.

**Where the published numbers actually live:**

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

## Kernel Code

All of it is in one file: **[`Project/submission/dispatcher_region.py`](Project/submission/dispatcher_region.py)**,
1,363 lines. That file is the only thing we wrote — it is spliced verbatim into
TikTok's script to produce the submission, and nothing outside it is changed.

The track lets you replace exactly one thing: the `UserOptimizedTransformer`
class. That is the only region we touch, and it is what this file fills.

Inside it, the work is **eight Triton GPU kernels** plus **one class** that picks
which of them to run. The kernels sit behind an `if _TRITON_OK:` guard, so on a
machine without Triton the whole file degrades to the unchanged baseline path:

| line | what it is |
|--:|---|
| 45 | `_sub_gelu` — GELU activation |
| 218 | `_sub_attn_fwd` — standalone attention |
| 344 | `_sub_norm_qkv` — LayerNorm fused into the QKV projection |
| 419 | `_sub_attn_heads` — causal flash attention, one program per (tile, batch, head) |
| 573 | `_sub_attn_block_tail` — output projection, residual, norm and GELU-FFN in one |
| 638 | `_sub_ln_fp16` — fp16 LayerNorm |
| 668 | `_sub_gelu_fp16` — fp16 GELU |
| 707 | `_sub_final_norm` — final norm |
| **804** | **`class UserOptimizedTransformer`** — the dispatcher |
| 931 | `_fused_forward` — the megakernel route, 12 of 14 shapes |
| 1017 | `_fp16_forward` — the tensor-core route, shapes 8 and 14 |

Here is the hot loop of the attention kernel, `_sub_attn_heads`. Key blocks
strictly below the diagonal need no causal mask and no bounds mask, so they get
their own loop with both removed, and the softmax runs in base 2 because NVIDIA
hardware only has a base-2 exponential:

```python
if SPLIT:
    full_end = (pid_m * BLOCK_M) // BLOCK_N * BLOCK_N
    # Stage 1: entirely below the diagonal. No causal mask, no bounds mask.
    for n_start in range(0, full_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        k = tl.load(qkv_base + offs_n[:, None] * (3 * D) + D + h * HD + offs_hd[None, :],
                    mask=hd_mask[None, :], other=0.0)
        v = tl.load(qkv_base + offs_n[:, None] * (3 * D) + 2 * D + h * HD + offs_hd[None, :],
                    mask=hd_mask[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
```

That one change took the attention kernel on shape 13 from **632.7 µs to
582.6 µs**, measured on paired diagnostics whose two untouched kernels agreed to
within 0.2%.

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

  The third kernel is our latest change, and it worked for a different reason
  than we expected. We split the head loop into its own grid dimension for more
  parallelism, and that did almost nothing — every shape except shape 2 already
  filled the card's 38 SMs. The actual win came from the side effect: once the
  heads all write into one buffer, the output projection runs at **full width**
  instead of once per head padded up to Triton's 16-wide `tl.dot` minimum. On the
  two shapes with head width 8 that was +26.4%, against +4.3% on the other nine.

- **The fp16 tensor-core stack — shapes 8 and 14**, the two with `d_model` 1024.
  These are big enough that the GPU is busy doing real arithmetic, so there is no
  launch overhead to delete. The win comes from fusing work into the GEMM
  boundaries instead. On shape 8 the baseline already hits **30.0% MFU** (shape 2
  manages 0.2%), and we take it to **71.1%**. That is why shape 8 has the
  smallest speedup on the board — there was less waste to remove. The same route
  takes shape 14 to **88.7%**, our highest anywhere.
- **The two oversized shapes live in the same file**, not in separate kernels.
  `dispatcher_region.py` has a batch-chunked path for shapes where the maths fits
  but the whole batch does not, and a sequence-chunked tail for shape 14. `k014`
  and `k015` were where we worked those ideas out; the shipped version is the
  integrated one. Shape 6 runs as **one call at batch 10,000**; shape 14 is
  called 32 times at batch 1, with all 100,000 tokens handled inside each call.
- **Dispatch.** `UserOptimizedTransformer` looks at the incoming shape and picks
  a route, which the track explicitly allows. Every evidence packet records which
  route ran, so anyone can check.

No external kernel library is wrapped, with no FlashAttention package and no
xFormers. The kernels are authored. `torch.compile` and SDPA are only there as
fallbacks and as something to measure against: `max-autotune` reached 7.00× / 3.10× / 1.23× on the shape-3 / 13 / 8
dials, against our 19.78× / 31.51× / 2.37×. Same caveat as the sdpa columns —
different build, and fp32 against our fp16.

## Repository map

| path | what it is |
|---|---|
| `Project/kernels/` | every candidate implementation, including the failures |
| `Project/harness/` | the referee: the trusted controller, the sandbox, and the signature checking |
| `Project/submission/` | the single-file submission + byte-identity prover |
| `Project/results/` | **pre-LOCK history only.** `JOURNAL.jsonl` is the append-only pre-gate journal. The post-LOCK board is in `Project/authority/`. |
| `Project/results_side/` | evidence packets for shapes 6 and 14 |
| `Project/audits/` | verdict ledger, evidence packets, review prompts |
| `Project/research/` | the research notes every plan had to cite |
| `Project/loop/` | the gate design, its state, and the log of every attempt |
| `Project/memory/` | decisions, lessons, running state |

## Reflection: limitations, and what we would improve

**Limitations.**

- **Shape 14 is timed as 32 serial batch-1 calls, not one batch-32 call.** The
  full 100,000-token sequence is real and so is the 48.271 s, but the batch
  dimension is split up, because the whole thing does not fit on an 8 GB card at
  once. The packet says so in its own `limitation` field.
- **The research base was built and then under-used.** Every plan had to cite it
  and every plan did, but citing a note is not the same as acting on it. The
  clearest example is **CUDA Agent** (ByteDance Seed and Tsinghua,
  [cuda-agent.github.io](https://cuda-agent.github.io/)) — a paper from the
  sponsor's own research group describing almost exactly the agent design we
  arrived at independently: protected profiling scripts the agent cannot edit,
  forbidden fallback calls, measurement it cannot game. We read it, wrote it up
  in `Project/research/cuda-agent-tiktok.md`, and then took less from it than we
  should have. Their ReAct loop profiles first and implements second; ours
  planned first and profiled only when something looked wrong.
- **We wrote CUDA kernels as well as Triton, and ran out of time to make them
  faster.** `k018`, `k023` and `k024` are hand-written CUDA. None of them beat
  the Triton versions before the deadline, so none of them ships. That is a time
  limit, not a verdict on CUDA.
- **No audit verdict is bound to the published board.** Our verdict schema uses
  `allOf`, which OpenAI's structured-output mode rejects, so the auditor fails
  before it reads the packet. Only the owner is allowed to fix that file.
- One GPU, one architecture, one framework. Nothing is validated on the
  TensorFlow path.

**What we would do with more time.**

Keep optimising. We were still finding wins when the deadline arrived — the last
structural change to the attention kernel landed a day before freeze and took
shape 13 from 632.7 µs to 582.6 µs on its own. The four launch-bound shapes
(2, 3, 7, 12) sit at 5.3%, 16.2%, 17.7% and 34.1% utilisation and are where the
remaining headroom obviously is. We froze this table because we had to, not
because we had run out of ideas.

## Development tools, APIs, libraries

- **Claude Code** (terminal agent) — Claude Fable 5, later Claude Opus 5:
  authored the kernels, the harness and the process machinery.
- **OpenAI Codex CLI** — GPT-5.6, reasoning effort high/ultra: champion audits
  and strategy reviews, each given only the evidence packet.
- **PyTorch 2.12**, **Triton 3.7**, CUDA 13.0, VS Code, git.
- No external datasets or APIs — the benchmark generates its own tensors.

## Contributions

Solo entry. The human owner set every rule, held sole authority over freezes,
promotions and everything that shipped, and made the decisions the agents were
not allowed to make. The AI agents wrote kernels and audited each other under
that authority.
