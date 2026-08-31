# Measurement methodology — for third-party review

**The question being asked of the reviewer:** *are we measuring the right things?*
Not "will this win", and not "are these numbers honestly obtained". The question
is whether the metrics, reference points and ceilings below are the correct ones
for this competition, and whether anything important is missing from them.

Written 1 Sep 2026. TikTok TechJam 2026, Track 3. Hardware: one RTX 3060 Ti.

---

## 1. What the competition measures

From the organiser's own webinar answers, quoted directly:

> *"The final score over the technical execution, I think, will be the weighted
> sum over the MFUs."*

> *"First of all, every shape should pass for the precision test. Or else it
> will... get a zero point."*

> *"About the weights, I'm considering. We will take the bandwidth into account."*

> *"No, judges will not rerun your script."*

Asked directly whether process or speed outcome mattered more:

> *"I think the better result you can outcome would be better, rather than the
> speed, which means you need output higher MFU kernels rather than coding maybe
> one hours first."*

**So four facts govern everything:**

1. The technical score is a **weighted sum of per-shape MFU**
2. Any shape failing precision scores **zero**, regardless of speed
3. The weights are **not decided**, and bandwidth will be considered
4. **Judges never execute our code.** They read what we report, so the evidence
   *is* the deliverable

Precision threshold, from the problem statement: relative error < 0.02 **or**
absolute error < 0.002.

The 14 test shapes are fixed and published in the appendix.

---

## 2. What we measure, and exactly how

**MFU = FLOPs ÷ time ÷ the card's peak rate.** Two of those three terms are
constants, so all measurement uncertainty lives in `time`.

Every timing figure below is produced by the organiser's own unmodified script.
We replace exactly one class inside it; the timing loop, the input generator and
the correctness predicate are all theirs.

**The protocol, which the script fixes and we do not control:**

- Baseline and our version are timed **in the same process**, on the **same input
  tensor**, built once and never regenerated
- Timing is **CUDA events recorded on the stream**, not a wall clock
- 20 warmup iterations, 100 repeats, 3 rounds → 300 samples per model
- The reported figure is the **median** of all 300
- Compilation and first-run cost are excluded, as the organiser permits

**Why the same process matters.** Clock speed, temperature and background load
affect both sides equally within one invocation, so they cancel out of the ratio.
Comparing our time today against a baseline recorded yesterday would let machine
drift masquerade as a speedup. That mistake invalidated this project's first
headline number and is why every row is a paired measurement.

**Machine conditions.** GPU clocks are locked at 1665 MHz for every measurement,
verified before and during each run. The box is checked idle (0–1% utilisation,
no competing GPU process, no audit running) before any timed run.

---

## 3. Three reference points, and why each exists

A single reference cannot answer the question "is this good", so we report three.

### 3.1 TikTok's own baseline — *what we are asked to beat*

`BaselineTransformer` in `torch_transformer_benchmark.py`. It builds the full
attention score matrix, applies the causal mask, softmaxes in fp32 and multiplies
again. It is deliberately naive.

**Its weakness:** beating it is not evidence of much. A speedup against a slow
reference measures how slow the reference is.

### 3.2 PyTorch's own optimised attention — *what a competent competitor achieves*

`torch.nn.functional.scaled_dot_product_attention`, PyTorch's built-in fused
flash attention, wired into the same benchmark. It is NVIDIA-tuned, written by
full-time specialists, and is what any strong entrant reaches for first.

**It is meant to be the honest bar**, and it is the right reference to have. But
our measurement of it has two defects that section 7.3 sets out in full: it is
not same-session with our column, and it runs at fp32 while our kernels compute
in fp16. Read section 7.3 before quoting any margin over PyTorch.

### 3.3 The physical maximum — *what the hardware can do*

Derived in section 4. Unreachable in practice, for reasons that are themselves
worth reporting, but it bounds the argument: nobody can be more than this fast.

---

## 4. How the physical maximum is derived

Three independent limits. Whichever binds first is the floor.

### 4.1 What precision our kernels actually use

This was read from our own source rather than assumed, because an earlier version
of this calculation got it wrong and produced an impossible result — one shape
appeared to exceed 100% of the limit, which is what exposed the error.

`_sub_pack_fused_layer` (`Project/submission/dispatcher_region.py:765-774`) casts
**every** weight with `.half()`. Activations are cast at line 388. Every `tl.dot`
therefore takes **fp16 inputs**, and its accumulator is fp32 — visible at lines
394-396 where the result is added to a `.to(tl.float32)` bias.

**fp16 inputs with fp32 accumulation runs at 32.5 TFLOP/s on an RTX 3060 Ti.**
Not 16.2 (that is fp32 input), and not 65 (that is fp16 accumulation, which we do
not use).

### 4.2 Compute limit

```
compute_ms = GFLOP / 32.5
```

### 4.3 Bandwidth limit

```
memory_ms = minimum_bytes_moved / 448 GB/s
```

448 GB/s is this card's memory bandwidth. All fourteen shapes turn out to be
compute-bound: the arithmetic takes longer than the data movement in every case.

### 4.4 The floor

```
physical_floor = max(compute_ms, memory_ms)
```

### 4.5 Occupancy — why the floor is unreachable on small shapes

A kernel launching N thread blocks can occupy at most N of this card's **38**
streaming multiprocessors. Below 38, part of the machine is idle by construction.

Shape 2 launches **8** blocks — it can use **21%** of the GPU. Shape 3 launches 32
and can use 84%. Every other shape saturates.

### 4.6 The FLOP accounting, so it can be checked

```
linear per layer     = tokens × (d·3d + d·d + d·ffn + ffn·d) × 2
attention per layer  = 2 matmuls × B·H·S·S·head_dim × 2, halved for causal
```

Summing these reproduces the independently published per-shape GFLOP figures on
**all fourteen shapes**. That reconciliation is the check that the accounting is
right, and it is printed by `Project/loop/ceiling.py` on every run.

**Reproduce:** `python3 Project/loop/ceiling.py`

### 4.7 Why 100% is unreachable for a transformer, not merely hard

The numerator counts only matrix multiplication. The model also performs 8
LayerNorms, 4 softmaxes with exponentials, 4 GELUs and 8 residual additions per
forward pass. **None of these contribute a single FLOP to the numerator, and all
of them consume time.** A transformer therefore cannot reach 100% MFU under this
definition regardless of implementation quality.

This is why section 3.2 exists. The physical maximum bounds the argument; the
competent-alternative comparison is what actually answers "is this good".

---

## 5. The table

**Every row below was measured on one artifact,
`c2028c4823ff756b062940e4eff35d5a6a341e9538b755256509cf3432e7794b`**, which is
the submission file that ships. This is the first version of this table where
that is true. See section 10.

| # | shape | GFLOP | TikTok baseline | its MFU | PyTorch sdpa † | **OURS** | **speedup** | vs sdpa † | **our MFU** | physical floor |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | B64 s128 d128 h4 | 7.52 | 5.0586 ms | 4.6% | 1.67× | **0.5622 ms** | **8.998×** | 5.4× | **41.1%** | 0.2313 ms |
| 2 | B1 s128 d128 h4 | 0.117 | 1.8039 ms | 0.2% | 1.28× | **0.0676 ms** | **26.691×** | 20.9× | **5.3%** | 0.0036 ms |
| 3 | B4 s128 d128 h4 | 0.470 | 1.7618 ms | 0.8% | 1.34× | **0.0891 ms** | **19.776×** | 14.8× | **16.2%** | 0.0145 ms |
| 4 | B16 s128 d128 h4 | 1.879 | 1.7547 ms | 3.3% | 1.39× | **0.1649 ms** | **10.643×** | 7.7× | **35.1%** | 0.0578 ms |
| 5 | B128 s128 d128 h4 | 15.03 | 9.8473 ms | 4.7% | 1.66× | **1.0025 ms** | **9.823×** | 5.9× | **46.1%** | 0.4625 ms |
| 6 | B10000 s128 d128 h4 | 1,174.41 | *out of memory* | — | *cannot run* | **60.3873 ms** | *no baseline* | **runs** | **59.8%** | 36.1355 ms |
| 7 | B64 s128 d32 h4 | 0.671 | 3.4028 ms | 0.6% | 2.18× | **0.1167 ms** | **29.149×** | 13.4× | **17.7%** | 0.0206 ms |
| 8 | B64 s128 d1024 h4 | 420.91 | 43.1206 ms | 30.0% | 1.02× | **18.2272 ms** | **2.366×** | **2.3×** | **71.1%** | 12.9511 ms |
| 9 | B64 s128 d128 h1 | 7.52 | 2.9604 ms | 7.8% | 1.11× | **0.6001 ms** | **4.933×** | 4.4× | **38.5%** | 0.2313 ms |
| 10 | B64 s128 d128 h2 | 7.52 | 3.9045 ms | 5.9% | 1.31× | **0.5704 ms** | **6.846×** | 5.2× | **40.5%** | 0.2313 ms |
| 11 | B64 s128 d128 h16 | 7.52 | 12.0433 ms | 1.9% | 2.59× | **0.6554 ms** | **18.377×** | 7.1× | **35.3%** | 0.2313 ms |
| 12 | B64 s32 d128 h4 | 1.678 | 1.7644 ms | 2.9% | 1.27× | **0.1516 ms** | **11.642×** | 9.2× | **34.1%** | 0.0517 ms |
| 13 | B64 s1024 d128 h4 | 120.26 | 169.9159 ms | 2.2% | 3.97× | **5.3919 ms** | **31.513×** | 7.9× | **68.6%** | 3.7003 ms |
| 14 | B32 s100000 d1024 h16 | **1,391,250.64** | *infeasible* | — | *cannot run* | **48.271 s** | *no baseline* | **runs** | **88.7%** | 42.808 s |

**† The two sdpa columns are a weaker grade of evidence than the rest of the
table and section 7.3 sets out why in full. In short: they were measured on
28 August on a different build, so `vs sdpa` is our speedup divided by theirs
rather than a paired measurement; and sdpa runs the model at fp32 with TF32
matmul while our kernels compute at fp16, so part of that margin is precision
rather than kernel engineering. Both are legal under the 2e-3 predicate. The
`speedup` column against TikTok's own baseline has neither problem.**

**Geometric mean speedup over the twelve shapes that have a baseline: 11.87×.**

**Geometric mean against sdpa over the twelve it can run: 7.42×**, with the †
caveats attached.

**Mean MFU across all fourteen shapes, weighted equally: 42.7%.** This is the
figure closest to what the organiser described as the technical score, since the
score is a weighted sum of per-shape MFU and the weights are not yet published.

Shapes 6 and 14 have no speedup and cannot have one. TikTok's baseline runs out
of memory at batch 10,000 on this card, and its dense attention matrix at 100,000
tokens would need roughly 160 TB. There is no baseline time to divide by, so both
rows report achieved MFU against the physical floor instead.

**Shape 14 carries 99.89% of all the arithmetic in the benchmark** and is the
strongest row at 88.7% of the physical maximum.

**Worst row is shape 2 at 5.3%**, and section 9 explains why that is close to the
limit of what the shape allows rather than a failure of implementation.

Per-row packet hashes, entry ids and the full-precision speedup values are in
`Project/BOARD.md` section 2, so any row can be traced to the packet that
produced it.

---

## 6. Correctness methodology

Speed is worthless here: any shape failing precision scores zero.

- The predicate is the organiser's own: finite **and** (absolute ≤ 2e-3 **or**
  relative ≤ 2%), applied element-wise
- The twelve primary shapes are each checked on **7 trials with 7 distinct output
  hashes**, so a cached or replayed answer cannot pass
- Shapes 6 and 14 are each checked on **5 seeds**

On the shipping artifact `c2028c48…`:

| group | trials | elements compared | violations |
|---|--:|--:|--:|
| twelve primary shapes | 84 | 167,559,168 | 0 |
| shape 6 | 5 | 819,200,000 | 0 |
| shape 14 | 5 | 16,384,000,000 | 0 |
| **total** | **94** | **17,370,759,168** | **0** |

Worst observed error against the 2e-3 budget: shape 6 at 0.0014288, shape 13 at
0.0011601, shape 14 at 0.0009825. No trial on any shape produced a non-finite
element.

Across the whole campaign, roughly 110 recorded runs on every shape and every
version of the code, there has been **zero correctness failures**.

---

## 7. Methodology decisions that could reasonably be challenged

These are placed here rather than left to be discovered.

### 7.1 We loosened our own shape-14 validation threshold from 1e-4 to 1e-3

This looks like moving a goalpost. The defence:

The official numerical profile mandates TF32 (`allow_tf32 = True`,
`matmul_precision("high")`). TF32 has 10 mantissa bits, giving roughly **4.9e-4**
relative precision. A 1e-4 agreement threshold is therefore **below the number
format's own noise floor** and unreachable by any implementation, including a
perfectly correct one.

**Two independent lines of evidence say it is a format limit and not
accumulated error.**

**First, the disagreement is flat across sequence length.** The current
validation packet (`Project/authority/blobs/054e8dfe…json`, bound to
`c2028c48…`) runs nine checks of the streamed oracle against the pinned dense
official baseline:

| sequence | seeds | worst absolute disagreement |
|--:|--:|--:|
| 1,024 | 1234, 1235, 1236 | 6.22e-4 |
| 2,048 | 1234, 1235, 1236 | 6.06e-4 |
| 4,096 | 1234, 1235, 1236 | 5.79e-4 |

Accumulated error would grow with length. Over a fourfold increase it does not
grow at all, and if anything it falls slightly.

**Second, and this is the decisive one: we have the same comparison measured
with TF32 off.** An earlier validation run on 29 August
(`Project/results_side/validation_20260829-041941.json`) ran the identical oracle
against the identical dense baseline, but did not configure the official
numerical profile. Its recorded criterion is *"oracle must match pinned dense
within 1e-4 abs (fp32 reassociation only)"*, it carries no `numerical_state`
block at all, and it returned:

| sequence | worst absolute disagreement, TF32 off |
|--:|--:|
| 1,024 | 9.54e-7 |
| 2,048 | 1.43e-6 |
| 4,096 | 1.19e-6 |

The current evaluator differs in that it calls `configure_official_numerics`
(`Project/tools/shape14_eval.py:87-92`), which sets
`torch.set_float32_matmul_precision("high")` and
`allow_tf32 = True` on both cuBLAS and cuDNN, and records that state in the
packet.

**So the same code, compared the same way, disagrees by 1.4e-6 in plain fp32 and
by 6.2e-4 under the mandated TF32 profile.** That is a factor of roughly 440,
and it is entirely attributable to the number format, because nothing else
changed. It also puts a number on the two error sources separately:

- error from our streamed decomposition: about **1.4e-6**
- error added by TF32, which the competition mandates: about **6.2e-4**

The threshold moved from 1e-4 to 1e-3 at the same moment the evaluator was
corrected to use the profile the competition actually requires. **1e-3 remains
twice as strict as the competition's own 2e-3 requirement**, and all nine checks
pass it with roughly 40% margin.

The 1.4e-6 figure should not be quoted as the shape-14 validation result, because
it comes from a run that does not use the mandated numerics. It is reported here
as the measurement that isolates our algorithm's own error, which is the useful
thing it tells us.

### 7.2 Shape 14's correctness reference is one we wrote ourselves

TikTok's baseline cannot run at 100,000 tokens — the dense attention matrix would
need roughly 160 TB. So we compute a chunked reference that performs identical
mathematics in bounded memory.

**The chain of trust:**

1. Our chunked reference is verified against TikTok's *real* baseline at 1,024,
   2,048 and 4,096 tokens, where their version can run
2. Only then is it used as the reference at 100,000 tokens
3. A separate check confirms that evaluating sequences one at a time gives the
   same answer as evaluating them together

The organiser has said he will supply official input/output pairs for shape 14 at
the final. **Our reference is a stand-in until those arrive, not a substitute for
them.**

### 7.3 The PyTorch comparison is not same-session, and it is not same-precision

This is the most challengeable thing in the document and it has two separate
problems. A third-party auditor found the second one.

`torch.nn.functional.scaled_dot_product_attention`, PyTorch's own fused flash
attention, was wired into the same benchmark as `Project/kernels/k001_sdpa.py`
and measured against the same baseline. Those results are:

| # | 1 | 2 | 3 | 4 | 5 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PyTorch sdpa speedup | 1.67× | 1.28× | 1.34× | 1.39× | 1.66× | 2.18× | 1.02× | 1.11× | 1.31× | 2.59× | 1.27× | 3.97× |

**Problem 1: not same-session.** Those figures were measured on 28 August on an
earlier version of the harness and on a different build. Same machine, same
shapes, but not the same invocation as our column. Dividing our speedup by theirs
to get a ratio assumes the two baselines agree, and that assumption is not tested.

**Problem 2: not same-precision, and this is the one that changes the reading.**
`k001_sdpa.py` replaces only the attention inner math. It leaves the model in
fp32 with TF32 matmul enabled, which is what the official baseline does. Our
route does not: `_sub_pack_fused_layer`
(`Project/submission/dispatcher_region.py:760-780`) casts every weight to fp16
with `.half()`, and activations are cast at line 388. Both satisfy the
competition's 2e-3 / 2% predicate, so both are legal. But it means **part of any
margin we show over PyTorch is a precision difference rather than kernel
engineering**, and the comparison must be labelled that way.

Carrying our new single-artifact times against the 28 August sdpa figures gives a
geometric mean of **7.42×**, against 7.49× on the previous mixed-build board. It
is in the section 5 table under a † so the comparison is visible rather than
buried, but **it should not be the headline**. It is an estimate with both
caveats above attached. The defensible comparison is the one against TikTok's own
baseline, which is paired inside one process, on one build, at the same precision
on both sides.

**What the sdpa column is genuinely good for**, and it is worth having: it
answers whether 11.87× is a statement about our kernels or about a slow
reference. A competent off-the-shelf alternative reaches 1.02× to 3.97× on these
shapes. Most of our margin is therefore not explained by the reference being
naive. And on shapes 6 and 14 sdpa cannot run at all while ours does, which is a
capability difference rather than a speed one and carries neither caveat.

**What would fix it:** re-run `k001_sdpa.py` on the current artifact in the same
session as our column, and separately run an fp16 variant of it, so the precision
effect and the kernel effect can be told apart. Neither has been done.

### 7.4 Reproducibility is uneven across shapes

Shape 14 repeats to **0.019%** across its three timing repeats: 48,271.04 ms,
48,276.47 ms, 48,267.13 ms.

Small shapes are far worse. Shape 12 was observed at **11.2516× and 9.7638× on
byte-identical code minutes apart, a 13.2% spread.** The campaign's own
calibrated noise floor for shapes of that class reads about 0.15%, which is wrong
by roughly two orders of magnitude, because it is computed by timing the baseline
against itself inside one process. That measures second-to-second steadiness and
not run-to-run reproducibility.

The consequence, stated rather than hidden: **no single small-shape row should be
read to more than two significant figures, and no per-shape difference smaller
than about 13% is resolvable at all.** The geometric mean over twelve shapes
averages that scatter down and is the figure to quote.

---

## 8. What we do NOT measure, and arguably should

This is the section where the review is most valuable.

**`torch.compile` on the current build.** An earlier version of this document
said `torch.compile` had never been run. **That was wrong**, and the error is
recorded here rather than quietly deleted. `Project/memory/DECISIONS.md`
(29 Aug ~02:35) records it as measured through the official script's own
`--compile max-autotune` flag, giving **7.0× on shape-3 dials, 3.1× on shape-13
dials and 1.2× on shape-8 dials**, against our figures at the time of 13.5×, 29×
and 2.0×.

What is true is narrower: that measurement is **pre-gate, from 29 August, on a
build eleven artifacts old**, and there is no measurement of `torch.compile` on
`c2028c48…`. So it cannot be placed alongside the section 5 table, which is the
whole point of that table. The three figures are the best available estimate and
nothing more.

**A per-shape practical ceiling.** We know NVIDIA's own library achieves roughly
72% of theoretical peak on shape 8 — because 77% of that shape's runtime is
`cutlass` and `ampere_fp16_s1688gemm`, NVIDIA's code, not ours. We have no
equivalent figure for the other thirteen shapes, so "88.7% on shape 14" cannot
currently be compared against what a state-of-the-art implementation would reach
on *that* shape.

**Bandwidth utilisation.** The organiser stated plainly that bandwidth will be
taken into account. We report achieved FLOP rates and nothing about achieved
GB/s.

**Memory footprint.** Measured only on the two shapes that exist because they
exhaust memory, and only because their evaluators check for leaks. Shape 6 settles
at 3.06 GiB allocated and peaks at 3.67 GiB, identical on all 10 repeats, with
zero growth in allocated or reserved bytes. Shape 14 peaks at 2.80 GiB allocated
and 3.19 GiB reserved, identical on all 3 repeats. Nothing is measured on the
other twelve shapes.

**An fp16 PyTorch reference.** Section 7.3 explains why this matters: our
comparison against `scaled_dot_product_attention` runs it at fp32 while we
compute at fp16, so the margin mixes precision with implementation. Running an
fp16 sdpa variant would separate them. It has not been done.

**Energy, or performance per watt.** Not measured. Probably out of scope, but not
considered.

---

## 9. Known weaknesses in the results themselves

Stated because they are the honest interpretation of the low numbers, and because
naming them is what makes the high numbers credible.

**Shape 2 at 5.3% MFU.** It runs 13 kernel launches per forward pass. The smallest
of them takes 2.33 µs to do work that needs 0.29 µs of memory traffic — it spends
eight times longer being launched than working. Total launch cost is roughly 30 µs
against 3.7 µs of actual arithmetic. Combined with the 21% occupancy ceiling, its
realistically reachable MFU is around 12%, and we reach 5.3%.

**Shapes 7 and 11 lose exactly 2× on attention.** Their head width is 8, but the
tensor cores require a minimum of 16, so the hardware pads and performs double the
necessary work. Measured: shape 1's attention runs at 32.3% of peak, shape 7's at
**15.6%**, shape 11's at **16.1%** — almost exactly half, which is the padding,
observed rather than inferred.

**Shape 8's remaining headroom is small.** Its residual additions run at 406 GB/s
against a 448 GB/s ceiling — 91% of the memory bandwidth the card physically has.
Three-quarters of that shape's runtime is already NVIDIA's own library.

---

## 10. Provenance note

**Resolved.** An earlier version of this document disclosed that rows 1 to 13
came from artifact `630a456c…` and row 14 from `c2028c48…`, which differ by two
correctness fixes made on 31 August, and said a single-artifact pass was the next
step.

That pass has now run. **All fourteen rows in section 5 are measured on
`c2028c4823ff756b062940e4eff35d5a6a341e9538b755256509cf3432e7794b`.** Shapes 1,
13 and 14 were measured on it on 1 September between 00:53 and 01:49 SGT, and
shapes 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 and 12 between 02:15 and 02:44 SGT. Each
row is bound to a single-use permit carrying that hash. `Project/BOARD.md`
section 2 lists the packet sha256 and entry id for every row.

The disclosure is kept rather than deleted because the reason for it still
stands: a table whose rows come from different builds is not a table of one
program, and that distinction has already cost this project one withdrawn
headline figure.

Two things this pass did **not** fix, and they travel with the table:

1. **These are screening-lane measurements.** No row is a promoted champion and
   no independent audit verdict is bound to any row, because the audit recording
   path is broken and is owner-only to repair. The measurements are real and each
   is bound to a permit and an artifact hash. The adjudication layer on top of
   them is missing.
2. **Shapes 6 and 14 remain side evidence.** Their evaluators use CPU RNG, so
   their inputs are not bit-identical to a default judge run, and shape 14's
   timing is 32 serial batch-1 calls rather than one literal batch-32 call. Both
   packets say so, and both must be labelled that way wherever they are quoted.
