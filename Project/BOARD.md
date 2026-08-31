# The board — all 14 shapes, one artifact

Every row below was measured on

```
c2028c4823ff756b062940e4eff35d5a6a341e9538b755256509cf3432e7794b
```

which is `Project/submission/torch_transformer_benchmark_submission.py`, the file
that ships. RTX 3060 Ti, clocks locked at 1665 MHz, box verified idle before and
during every run, one single-use permit per row, every row `correct: true`.

This is the first board in this project where all fourteen rows come from the
same build. Earlier boards mixed rows from up to four different artifacts, and
the geometric mean of such a board is not a statement about any one program. See
section 8.

Measured 1 Sep 2026, 02:15 to 02:44 SGT, except shape 14 which was measured at
00:53 SGT on the same artifact and took 80 minutes.

---

## 1. The table

| # | shape | GFLOP | TikTok baseline | its MFU | PyTorch sdpa † | OURS | speedup | vs sdpa † | our MFU | hard floor | correct |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| 1 | B64 s128 d128 h4 | 7.52 | 5.0586 ms | 4.6% | 1.67× | **0.5622 ms** | **8.998×** | 5.4× | **41.1%** | 0.2313 ms | 7 trials |
| 2 | B1 s128 d128 h4 | 0.117 | 1.8039 ms | 0.2% | 1.28× | **0.0676 ms** | **26.691×** | 20.9× | **5.3%** | 0.0036 ms | 7 trials |
| 3 | B4 s128 d128 h4 | 0.470 | 1.7618 ms | 0.8% | 1.34× | **0.0891 ms** | **19.776×** | 14.8× | **16.2%** | 0.0145 ms | 7 trials |
| 4 | B16 s128 d128 h4 | 1.879 | 1.7547 ms | 3.3% | 1.39× | **0.1649 ms** | **10.643×** | 7.7× | **35.1%** | 0.0578 ms | 7 trials |
| 5 | B128 s128 d128 h4 | 15.03 | 9.8473 ms | 4.7% | 1.66× | **1.0025 ms** | **9.823×** | 5.9× | **46.1%** | 0.4625 ms | 7 trials |
| 6 | B10000 s128 d128 h4 | 1,174.41 | out of memory | — | *cannot run* | **60.3873 ms** | none possible | **runs** | **59.8%** | 36.1355 ms | 5 seeds |
| 7 | B64 s128 d32 h4 | 0.671 | 3.4028 ms | 0.6% | 2.18× | **0.1167 ms** | **29.149×** | 13.4× | **17.7%** | 0.0206 ms | 7 trials |
| 8 | B64 s128 d1024 h4 | 420.91 | 43.1206 ms | 30.0% | 1.02× | **18.2272 ms** | **2.366×** | **2.3×** | **71.1%** | 12.9511 ms | 7 trials |
| 9 | B64 s128 d128 h1 | 7.52 | 2.9604 ms | 7.8% | 1.11× | **0.6001 ms** | **4.933×** | 4.4× | **38.5%** | 0.2313 ms | 7 trials |
| 10 | B64 s128 d128 h2 | 7.52 | 3.9045 ms | 5.9% | 1.31× | **0.5704 ms** | **6.846×** | 5.2× | **40.5%** | 0.2313 ms | 7 trials |
| 11 | B64 s128 d128 h16 | 7.52 | 12.0433 ms | 1.9% | 2.59× | **0.6554 ms** | **18.377×** | 7.1× | **35.3%** | 0.2313 ms | 7 trials |
| 12 | B64 s32 d128 h4 | 1.678 | 1.7644 ms | 2.9% | 1.27× | **0.1516 ms** | **11.642×** | 9.2× | **34.1%** | 0.0517 ms | 7 trials |
| 13 | B64 s1024 d128 h4 | 120.26 | 169.9159 ms | 2.2% | 3.97× | **5.3919 ms** | **31.513×** | 7.9× | **68.6%** | 3.7003 ms | 7 trials |
| 14 | B32 s100000 d1024 h16 | **1,391,250.64** | infeasible | — | *cannot run* | **48,271.04 ms** | none possible | **runs** | **88.7%** | 42,807.71 ms | 5 seeds |

**Geometric mean speedup over the twelve shapes that have a baseline: 11.87×.**

**Geometric mean against PyTorch's sdpa, over the twelve it can run: 7.42×** —
subject to the two caveats below, which are not small.

---

## 1a. The score under five weightings, because the weighting is not decided

The organiser has stated that Technical Execution is **a weighted sum of
per-shape MFU**, that **the weights are not yet decided**, and that bandwidth
will be taken into account. A single aggregate number would therefore be a bet on
an unpublished rule. These are the same fourteen measurements added up five ways:

| how the 14 shapes are combined | our score | what it rewards |
|---|--:|---|
| geometric mean, all 14 | **35.5%** | punishes the worst shape hardest |
| **equal weight, all 14** | **42.7%** | every shape counts the same |
| equal weight, the 12 with a runnable baseline | 37.5% | excludes the two extreme shapes |
| **bandwidth-weighted, all 14** | **87.1%** | shapes that move more bytes count more |
| **FLOP-weighted, all 14** | **88.6%** | shapes that do more arithmetic count more |

**The spread is 35.5% to 88.6% on identical measurements.** That is not noise and
it is not a choice we get to make. It is the entire range the undecided weighting
covers, and we report all of it rather than quoting whichever end flatters us.

**Why the range is so wide.** Shape 14 is **99.87% of all the arithmetic in the
benchmark** (1,391,251 of 1,393,016 GFLOP) and **94.4% of the minimum bytes
moved**. Any work-proportional weighting is therefore close to a report of shape
14 alone, where we reach 88.7%. Any per-shape weighting is dominated instead by
shape 2 at 5.3%, which is capped near 21% by occupancy no matter who writes it.

**Which one we optimise against:** equal weight across all 14, by owner
direction. It is not the lowest of the five — the geometric mean is harsher at
35.5%, and equal weight over the twelve with a baseline is 37.5% — but it is the
reading that keeps pressure on the small shapes, which is where our remaining
headroom is, and it sits far below the two work-proportional weightings that
would flatter us most.

**Provenance note:** the FLOP weighting uses per-shape GFLOP derived in §5.6 and
checked by hand. The bandwidth weighting uses the `ideal MB` column of
`Project/research/roofline-table.md`, which is a research note rather than a
measured packet, so that row is a weaker citation than the rest of this board.

---

### † Read this before quoting either sdpa column

The two columns marked † are **not the same grade of evidence as the rest of the
table**, and they are in it only so the comparison is visible rather than hidden
in an appendix. Two problems, both real.

**1. Different build, different day, not paired.** Those sdpa figures were
measured on **28 August** on an earlier harness and an earlier build.
Every other number in this table is from `c2028c48…`. The `vs sdpa` column is
computed as our speedup divided by theirs, which is only valid if the two runs
saw the same baseline — and that has not been tested. Treat those cells as an
estimate, not a measurement.

**2. Different precision, and it favours us.** `Project/kernels/k001_sdpa.py`
replaces only the attention inner math and leaves the model in **fp32 with TF32
matmul**, which is what the official baseline does. Our route casts every weight
and activation to **fp16** with fp32 accumulation. Both clear the competition's
2e-3 predicate, so both are legal, but **part of the `vs sdpa` margin is
precision rather than kernel engineering.** A third-party auditor raised this and
it is the honest reading.

The `speedup` column against TikTok's own baseline has neither problem: same
process, same input tensor, same build, same precision on both sides. **That is
the number to defend.** The sdpa columns answer a different and softer question,
which is whether 11.87× reflects our work or merely a slow reference.

**What would fix it:** re-run `k001_sdpa.py` on `c2028c48…` in the same session,
and separately run an fp16 variant of it. The first kills problem 1. The gap
between fp32-sdpa and fp16-sdpa is then the precision effect, and the gap between
fp16-sdpa and us is the kernel engineering. Neither has been done. It is about
40 minutes of gate work and the family budgets allow it.

Shapes 6 and 14 have no speedup and never can. TikTok's baseline runs out of
memory at batch 10,000 on an 8 GB card, and at 100,000 tokens its dense attention
matrix would need roughly 160 TB. There is no baseline time to divide by. Both
rows are reported as achieved MFU against the physical floor, which is the
quantity the competition scores anyway.

Full speedup values to the precision the harness reported them, so any row can be
traced back to its packet:

| # | speedup |
|---|---|
| 1 | 8.998178492470325 |
| 2 | 26.690577668035758 |
| 3 | 19.77586136674765 |
| 4 | 10.643245254177664 |
| 5 | 9.822778066735113 |
| 7 | 29.14912157812199 |
| 8 | 2.365730383991632 |
| 9 | 4.933447086602175 |
| 10 | 6.845601407935701 |
| 11 | 18.376563557856052 |
| 12 | 11.641891147155942 |
| 13 | 31.513341454732778 |

---

## 2. How to check any row

Each row is bound to its measurement by a content-addressed packet under
`Project/authority/blobs/`. The packet carries the artifact hash, both median
times, all 300 raw samples per side, and every correctness trial.

| # | packet sha256 | entry id |
|---|---|---|
| 1 | `0047a1a336de55e32722d55aa636937f8a13e9248e5504e289ee5f135c251957` | run-3bf9dc6a6f7d2e4149df94040238d8a2 |
| 2 | `6f060685699abd99063ede11ecdda80267b19b0bc58786105f1a70af74605e00` | run-41ffb099990c8b3fd0171e32c6fcfb39 |
| 3 | `816dabfe745d572e3798c543d55ee2968d37b0dd5238b30ec5a2eee35da1950f` | run-61a20c23329aee3f1f24bc98661604f5 |
| 4 | `2a1f1d0cb74c919c224282d00dc98fc0fb57fdcebbad0f319f3d347c29f783ea` | run-5246140960c2578a1890e39a80be7a93 |
| 5 | `5fc4e21dc2765438fc86cb63f11e3d5ef6a4dc879074827189dca36609ea5f06` | run-25f917101b416f9894fc70cf108c7086 |
| 6 | `daa1ccec88c59bf03d11031678074cb45adcd19ef5ffc1a5c903a1da8e817d60` | 20260831-184424-5d887b |
| 7 | `7c89d0bbf409ab1b5a0e71696002887f8e5f85083603f070ecb47d7bba20e46a` | run-1d99153f9c565bb45b7509791b7553e8 |
| 8 | `4ac907b9a29015256c17bc976d8f40dab494ce9e52050be9ad93eb1c1d05436c` | run-d8642d4c447b0d13f646042bed8931dc |
| 9 | `7890607c8184f297aaf2e21d7d58888985fd65d93f8db21298ae95e37ead0abe` | run-ed857b747dbce3d88ec418453b52005a |
| 10 | `147901587fee6b800c908c3b88dab1e55fd9af01730b046bcf19488fcffaea81` | run-af9d38af4aab9fd127737be496aa81e9 |
| 11 | `d97e6b87a07b9e63b10cffa0624d99a082322bcb54493a71a7b9210db6701786` | run-2a58a74234db096723f47919c61ed0fc |
| 12 | `e720d6cba761f14b81c02cff2c7ce3a1f33c72870e6eaa9e7ca7bda699f16347` | run-14cc33373c13717da705adfd972f4f19 |
| 13 | `315ff617031c89ce1e96fcddcc8e0ab12933ef59dd669b32422f6559bcbae156` | run-146621d7a74d7eadac09f298b3f682bf |
| 14 | `7d4f73d44809257c88be45d057b47d802282d0daf62d4dca4f1d4d28dd2a11c3` | 20260831-165326-77499e |

Each packet's `submission_sha256` reads `c2028c48…`. The permits that authorized
these runs are in the hash-chained journal at `Project/authority/events.jsonl`.

---

## 3. What the columns mean

**GFLOP** is how much arithmetic the shape requires. It is fixed by the problem
and not by us. The accounting is in section 6.

**TikTok baseline** is `BaselineTransformer` in `torch_transformer_benchmark.py`,
their own reference implementation. It builds the full attention score matrix,
applies the causal mask with `masked_fill`, softmaxes in fp32, then multiplies
again. It is timed by the official script inside the same process as ours, on the
same input tensor, so clock state and thermal drift affect both sides equally and
cancel out of the ratio.

**MFU** is the achieved arithmetic rate divided by the card's peak rate. The
organiser has stated the technical score is a weighted sum of per-shape MFU, so
this is the column that is actually scored.

**Hard floor** is the fastest the shape can physically run on this card. Nothing
can beat it. Derivation in section 5.

**Correct** is the number of independent trials the official predicate was
checked on. Each trial uses a different seed and produces a different output
hash, so a cached or replayed answer cannot pass.

---

## 4. Correctness

The predicate is the organiser's own, applied element by element: finite, and
either absolute error at or below 2e-3 or relative error at or below 2%.

The twelve primary shapes were each checked on 7 trials with 7 distinct output
hashes. Shapes 6 and 14 were each checked on 5 seeds.

| group | trials | elements compared | violations |
|---|--:|--:|--:|
| twelve primary shapes | 84 | 167,559,168 | 0 |
| shape 6 | 5 | 819,200,000 | 0 |
| shape 14 | 5 | 16,384,000,000 | 0 |
| **total** | **94** | **17,370,759,168** | **0** |

Worst observed error, against a 2e-3 budget:

- shape 6: 0.0014288 across 163,840,000 elements per trial
- shape 14: 0.0009825 across 3,276,800,000 elements per trial
- shape 13: 0.0011601 across 8,388,608 elements per trial

No trial on any shape produced a non-finite element.

---

## 5. How the hard floor is calculated

Three independent limits. The floor is whichever binds first.

### 5.1 What precision the kernels use

This was read from the source rather than assumed, because an earlier version of
this calculation got it wrong and produced a shape apparently faster than the
speed of light.

`_sub_pack_fused_layer` in `Project/submission/dispatcher_region.py:760-780`
casts every weight with `.half()`: `w_qkv`, `b_qkv`, `w_o`, `b_o`, `w_f1`, `b_f1`,
`w_f2`, `b_f2`. Only the LayerNorm scale and bias stay fp32, and those carry
negligible arithmetic. Activations are cast at line 388. Every `tl.dot` therefore
takes fp16 inputs, and its accumulator is fp32, visible at lines 394 to 396 where
the result is added to a `.to(tl.float32)` bias.

fp16 inputs with fp32 accumulation runs at **32.5 TFLOP/s** on an RTX 3060 Ti.
Not 16.2, which is fp32 input. Not 65, which is fp16 accumulation, which we do
not use.

### 5.2 Compute limit

```
compute_ms = GFLOP / 32.5
```

### 5.3 Bandwidth limit

```
memory_ms = minimum_bytes_moved / 448 GB/s
```

All fourteen shapes are compute-bound. The arithmetic takes longer than the data
movement in every case, with shape 8 closest to balanced at 12.95 ms of compute
against 0.26 ms of traffic.

### 5.4 The floor

```
hard_floor = max(compute_ms, memory_ms)
```

### 5.5 Occupancy, and why shapes 2 and 3 cannot reach their floor

A kernel launching N thread blocks can occupy at most N of this card's 38
streaming multiprocessors. Below 38, part of the machine is idle by construction
and no implementation can reach the hard floor.

With a 64-row attention tile, blocks are `ceil(seq/64) × batch × heads`:

| shape | blocks | fraction of machine reachable |
|---|--:|--:|
| 2 | 8 | 21% |
| 3 | 32 | 84% |
| all others | 128 to 800,256 | 100% |

Shape 2 is one sequence of 128 tokens. There is not enough work in it to fill
this GPU. Its 5.3% MFU is 25% of what its occupancy ceiling permits, and the
remaining gap is fixed per-call cost that does not shrink with problem size. See
`Project/MEASUREMENT_METHODOLOGY.md` section 9 for the launch-cost arithmetic.

### 5.6 The FLOP accounting, so it can be checked

```
linear per layer     = tokens × (d·3d + d·d + d·ffn + ffn·d) × 2
attention per layer  = 2 matmuls × B·H·S·S·head_dim × 2, halved for causal
```

Worked once, on shape 1, so the rest can be trusted. Shape 1 is batch 64,
sequence 128, `d_model` 128, `ffn` 128, 4 heads, 4 layers, so tokens is 8,192:

```
linear    = 8192 × (128·384 + 128·128 + 128·128 + 128·128) × 2
          = 8192 × 98,304 × 2 = 1.6106 GFLOP per layer, 6.4425 over 4 layers
attention = 2 × (64·4·128·128·32) × 2 / 2
          = 0.2684 GFLOP per layer, 1.0737 over 4 layers
total     = 7.516 GFLOP
```

which is the 7.52 in the table. The same arithmetic reproduces the published
per-shape GFLOP on all fourteen shapes, including 1,391,250.6 for shape 14.

**Reproduce:** `python3 Project/loop/ceiling.py`

---

## 6. The PyTorch comparison — what it is

`torch.nn.functional.scaled_dot_product_attention` is PyTorch's own fused flash
attention. It is NVIDIA-tuned, written by full-time specialists, and it is what
any strong entrant reaches for first. We wired it into the same benchmark as
`Project/kernels/k001_sdpa.py` and measured it against the same baseline.

It exists in this board to answer one question: **is 11.87× a statement about our
kernels, or merely about how slow TikTok's reference is?** The answer is that a
competent off-the-shelf alternative gets 1.02× to 3.97× on these shapes, so most
of our margin is not explained by a weak reference.

The two caveats on those columns are stated in full under the table in section 1.
They are not footnotes to skip: the figures are cross-build, and sdpa runs at
fp32 while we run fp16. Both matter, and the second one favours us.

**On two shapes it cannot run at all and ours does.** Shape 6 exhausts memory at
batch 10,000, and shape 14's dense attention table would need roughly 160 TB.
That is not a speed comparison, it is a capability difference, and it is the one
part of this section that needs no caveat.

---

## 7. Reproducibility is uneven, and the small shapes are the problem

Shape 14 repeats to 0.019% across its three timing repeats: 48,271.04 ms,
48,276.47 ms, 48,267.13 ms.

Small shapes are far worse. Shape 12 was observed at 11.2516× and 9.7638× on
byte-identical code minutes apart, a 13.2% spread. The campaign's calibrated
noise floor for shapes of that class reads about 0.15%, which is wrong by roughly
two orders of magnitude, because it is computed by timing the baseline against
itself inside one process. That measures second-to-second steadiness, not
run-to-run reproducibility.

The consequence is stated rather than hidden: **no single small-shape row should
be read to more than two significant figures, and no per-shape difference smaller
than about 13% is resolvable.** The geometric mean over twelve shapes averages
that scatter down and is the figure to quote.

Three rows moved noticeably against the previous build, and in both directions:
shape 2 read higher, shape 3 and shape 9 read lower. Those differences are inside
the scatter above and are not evidence of anything.

---

## 8. What this board fixes, and what it does not

**Fixes:** every row is one artifact. The previous headline board drew shapes 4,
5, 9, 10, 12 and 13 from `2778b747…`, shapes 1 and 11 from `418952bf…`, shapes 2
and 7 from `599f5dad…` and shape 3 from `301d7063…`. A geometric mean over rows
from four builds is not the speedup of the file that ships, and it must not be
quoted as one.

**Does not fix:** these are screening-lane measurements. Nothing here is a
promoted champion, and no independent audit verdict is bound to any row, because
the audit recording path is broken and is owner-only to repair. The measurements
are real and each is bound to a permit and an artifact hash. What is missing is
the adjudication layer on top of them, and that limitation travels with the
board.

**Also does not fix:** shapes 6 and 14 remain side evidence. Their evaluators use
CPU RNG, so their inputs are not bit-identical to a default judge run, and shape
14's timing is 32 serial batch-1 calls rather than one literal batch-32 call.
Both are labelled that way in their own packets and must be labelled that way
anywhere they are quoted.

---

## 9. Why the measurement can be trusted

**A signed lock makes it mechanically impossible for the agent to have edited the
benchmark.** 29 files are hash-pinned, including `torch_transformer_benchmark.py`,
`Project/shapes.json` and the measurement harness. The controller re-verifies all
29 before every run and refuses everything on a single mismatch. The signing key
is the owner's. Check it with:

```
python3 Project/harness/trusted_controller.py verify-lock
```

**Every run needed a single-use permit** issued against an owner-signed
capability, bound to the artifact hash, recorded in an append-only hash-chained
journal. 272 permits have been issued and 271 consumed over the life of this
campaign.

**Baseline and candidate are timed in one process** on one input tensor built
once, with CUDA events on the same stream, 20 warmup iterations, 100 repeats,
3 rounds, and the reported figure is the median of all 300 samples. Every repeat
is kept. There is no best-of and no dropped round.

**The box was verified quiet** before the campaign and again partway through:
1665 MHz SM clock, 7001 MHz memory, 0% utilisation, 47 to 49 W, no audit running.
