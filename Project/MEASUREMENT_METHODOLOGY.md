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

**This is the honest bar.** If we beat this, we have beaten the realistic
alternative rather than a strawman.

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

| # | shape | GFLOP | TikTok baseline | **PyTorch's best** | **OURS** | **× vs PyTorch** | our MFU | physical floor |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | B64 s128 d128 h4 | 7.52 | 5.078 ms | 1.67× | **0.541 ms · 9.39×** | **5.61×** | 42.8% | 0.231 ms |
| 2 | B1 s128 d128 h4 | 0.12 | 1.745 ms | 1.28× | **0.068 ms · 25.82×** | **20.25×** | 5.3% | 0.004 ms |
| 3 | B4 s128 d128 h4 | 0.47 | 1.744 ms | 1.34× | **0.084 ms · 20.77×** | **15.48×** | 17.2% | 0.014 ms |
| 4 | B16 s128 d128 h4 | 1.88 | 1.761 ms | 1.39× | **0.163 ms · 10.82×** | **7.79×** | 35.5% | 0.058 ms |
| 5 | B128 s128 d128 h4 | 15.03 | 9.882 ms | 1.66× | **0.959 ms · 10.31×** | **6.21×** | 48.3% | 0.463 ms |
| 6 | B10000 s128 d128 h4 | 1,174.41 | *out of memory* | *cannot run* | **60.26 ms** | **runs** | 60.0% | 36.14 ms |
| 7 | B64 s128 d32 h4 | 0.67 | 3.390 ms | 2.18× | **0.113 ms · 30.11×** | **13.81×** | 18.3% | 0.021 ms |
| 8 | B64 s128 d1024 h4 | 420.91 | 43.143 ms | 1.02× | **18.081 ms · 2.39×** | **2.35×** | 71.6% | 12.951 ms |
| 9 | B64 s128 d128 h1 | 7.52 | 2.973 ms | 1.11× | **0.581 ms · 5.12×** | **4.62×** | 39.8% | 0.231 ms |
| 10 | B64 s128 d128 h2 | 7.52 | 3.915 ms | 1.31× | **0.551 ms · 7.11×** | **5.41×** | 42.0% | 0.231 ms |
| 11 | B64 s128 d128 h16 | 7.52 | 12.051 ms | 2.59× | **0.634 ms · 19.01×** | **7.35×** | 36.5% | 0.231 ms |
| 12 | B64 s32 d128 h4 | 1.68 | — | 1.27× | *not yet measured* | — | — | 0.052 ms |
| 13 | B64 s1024 d128 h4 | 120.26 | 169.935 ms | 3.97× | **5.244 ms · 32.41×** | **8.16×** | 70.6% | 3.700 ms |
| 14 | B32 s100000 d1024 h16 | **1,391,250.64** | *impossible* | *cannot run* | **48.271 s** | **runs** | **88.7%** | 42.808 s |
| | | | | **geomean 1.00×** | | **7.49×** | | |

**Headline reading:** we are **7.49× faster than PyTorch's own flash attention**,
geometric mean across the shapes it can run, with a worst case of 2.35×. On two
shapes PyTorch's implementation cannot run at all and ours does.

**Shape 14 carries 99.89% of all the arithmetic in the benchmark** and is our
strongest result at 88.7% of the physical maximum.

---

## 6. Correctness methodology

Speed is worthless here: any shape failing precision scores zero.

- The predicate is the organiser's own: finite **and** (absolute ≤ 2e-3 **or**
  relative ≤ 2%), applied element-wise
- Checked on **7 trials with 7 distinct output hashes**, so a cached or replayed
  answer cannot pass
- **Zero correctness failures across roughly 110 recorded runs**, on every shape,
  on every version of the code

Shape 6 passes on 5 seeds with a worst-case error of 0.00143 against a 0.002
budget, over 163,840,000 elements per trial.

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

Evidence it is a format limit and not accumulated error: the observed
disagreement is **flat across sequence lengths** — 6.2e-4 at 1024 tokens,
6.1e-4 at 2048. Accumulated error would grow with length. It does not.

**1e-3 remains twice as strict as the competition's own 2e-3 requirement.**

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

### 7.3 The PyTorch comparison column is not same-session

Those figures were measured on 28 August on an earlier version of our measurement
harness. Same machine, same shapes, but not the same invocation as our column.

The margins are 2.35× to 20.25×, far larger than any harness difference could
account for — but it is not a like-for-like paired measurement and should not be
presented as one.

### 7.4 Reproducibility is uneven across shapes

Shape 14 repeats to **0.02%** across three timing repeats. Small shapes are far
worse: byte-identical code has been observed to differ by **13%** between separate
invocations, which is why per-shape deltas below that are not treated as
meaningful and why the geometric mean over many shapes is the figure we quote.

---

## 8. What we do NOT measure, and arguably should

This is the section where the review is most valuable.

**`torch.compile`.** Never run. It is the first thing most PyTorch engineers try,
and we have no number for it. Our "competent alternative" reference is
`scaled_dot_product_attention` alone.

**A per-shape practical ceiling.** We know NVIDIA's own library achieves roughly
72% of theoretical peak on shape 8 — because 77% of that shape's runtime is
`cutlass` and `ampere_fp16_s1688gemm`, NVIDIA's code, not ours. We have no
equivalent figure for the other thirteen shapes, so "88.7% on shape 14" cannot
currently be compared against what a state-of-the-art implementation would reach
on *that* shape.

**Bandwidth utilisation.** The organiser stated plainly that bandwidth will be
taken into account. We report achieved FLOP rates and nothing about achieved
GB/s.

**Memory footprint.** Not measured on any shape, despite two shapes existing
specifically because they exhaust memory.

**Shape 12.** No current measurement.

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

At the time of writing, rows 1–13 in section 5 were measured on artifact
`630a456c…` and row 14 on `c2028c48…`, which differ by two correctness fixes made
on 31 August. A single-artifact measurement pass covering all fourteen shapes is
the immediate next step, and the published table will carry one hash for every
row.

This is disclosed because a table whose rows come from different builds is not a
table of one program, and that distinction has already cost this project one
withdrawn headline figure.
