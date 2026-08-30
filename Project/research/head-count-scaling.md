# Head-count scaling: what actually varies across shapes 1/9/10/11 (measured 31 Aug 2026)

Shapes 1, 9, 10 and 11 are the same problem four times. Batch 64, sequence 128,
`d_model` 128, ffn 128, 4 layers, causal — **identical in every dial except head
count** (4, 1, 2, 16). Their arithmetic is identical too: attention FLOPs are
`B·H·S·S·(d/H)·2`, and the `H` cancels. All four are **7.52 GFLOP**, which the
roofline table independently agrees with.

So any difference in speedup across these four is a difference in *implementation
efficiency*, not in problem size. That makes them a natural controlled experiment,
and nobody had read it as one.

## The measurement

Post-LOCK, quiet box, one-use permit per row, baseline and candidate paired inside
a single invocation, median of 300 samples (warmup 20 / repeats 100 / rounds 3).

| heads | head_dim | shape | baseline ms | candidate ms | speedup |
|---|---|---|---|---|---|
| 1 | 128 | 9 | 2.7085 | 0.5601 | 4.8355x |
| 2 | 64 | 10 | 3.6168 | 0.5509 | 6.5651x |
| 4 | 32 | 1 | 4.7514 | 0.5704 | 8.3303x |
| 16 | 8 | 11 | 11.6337 | 0.9175 | 12.6797x |

## Finding 1 — the speedup spread is mostly a BASELINE property

**The baseline slows down 4.3x as head count goes 1 → 16, for arithmetic that does
not change.** The official implementation reshapes and processes per head, so its
op count and its per-op overhead scale with `H` even though its FLOPs do not.

Our kernel handles all heads inside one fused pass. From 1 to 4 heads its time is
**flat within 3.5%** (0.5601 / 0.5509 / 0.5704 ms) — exactly what identical
arithmetic should produce.

The consequence for how results are reported: **"we are 12.7x on shape 11 and 4.8x
on shape 9" mostly describes the baseline, not us.** On shape 9 the baseline is
already close to well-shaped — one head is one clean large matmul — so there is
little waste to remove and our advantage is smallest. Shape 9 is not our weak
point; it is the baseline's strong point. Any narrative that reads the per-shape
spread as our kernel varying in quality is reading the wrong variable.

## Finding 2 — but our kernel DOES pay at 16 heads, and the documented cause does not explain it

Our candidate jumps from ~0.5605 ms (mean of the 1/2/4-head cases) to **0.9175 ms**
at 16 heads. That is **+63.7%** on identical FLOPs, and it is real: it drops our
achieved throughput from ~13.4 TF/s to **8.20 TF/s**, MFU 0.41 → **0.25**.

The obvious suspect is the padding recorded in
[small-head-dim-padding.md](small-head-dim-padding.md): at 16 heads `head_dim` is
8, and `tl.dot` has a 16-wide minimum, so the attention dots are computed at twice
the necessary width.

**Do the arithmetic before crediting it** (LESSONS 32). Per layer, for these dials:

| term | MFLOP |
|---|---|
| QKV projection (3 matmuls) | 805.3 |
| output projection | 268.4 |
| attention (scores + AV, causal-halved) | 268.4 |
| FFN (2 matmuls) | 536.9 |
| **total** | **1879** |

Attention is **14.3%** of the block. Doubling it — which is the *entire* effect of
padding 8 to 16 — can raise total work by at most **14.3%**.

**Measured penalty is 63.7%. Padding can account for at most about a fifth of it.**

So the documented padding is real but is **not** the dominant cause, and the
research base should stop treating it as the explanation for shape 11. The
remaining ~50% is unexplained.

**Hypothesis, explicitly not a finding:** with `head_dim` 8 the per-head tiles are
far below the tensor-core tile shape, so each of the 16 head iterations runs at
poor efficiency and the fused kernel's inner loop is 16x longer with 16x less work
per step — a tile-shape and loop-overhead effect rather than a FLOP-count effect.
That predicts the penalty tracks head *count* (loop length), not head *dim*
(padding), and shape 7 discriminates: it also has `head_dim` 8 but only 4 heads,
and it is the second-best shape on the whole board at 21.96x.

**The isolating experiment** is a torch-profiler diagnostic on shape 11 and shape 1
with identical bytes, comparing per-kernel device time for the K2 attention kernel
alone. If K2's share on shape 11 is roughly 16/4 of shape 1's, the loop-length
story holds; if it is roughly 2x, padding holds. This costs a diagnostic permit and
no attempt budget.

### RESOLVED — the discriminator was run, and it refuted the hypothesis above

Two torch-profiler diagnostics, identical k009 bytes, 20 profiled iterations each
(`profile-1127da61e6696eed99c7d67f` shape 11, `profile-923ddf58435a05b18e66a4ba`
shape 1). Per-call device time:

| kernel | shape 1 (4 heads) | shape 11 (16 heads) | ratio |
|---|---|---|---|
| `_attn_block_tail` (K2: attention + out-proj + residual + norm2 + FFN) | 74.054 us | 157.653 us | **2.129x** |
| `_norm_qkv` (K1) | 49.046 us | 50.933 us | 1.038x |
| `_final_norm` | 20.278 us | 20.683 us | 1.020x |
| DtoD copy | 19.783 us | 19.989 us | 1.010x |
| **total device time per forward** | **558 us** | **901 us** | **1.614x** |

**The penalty is entirely inside K2. Every other kernel is flat within 4%.**

**K2 is 2.13x, not 4x.** The preregistered discriminator said 16/4 = 4x means
loop-length, ~2x means padding. It came out at 2.13x, so **the loop-length
hypothesis above is refuted and the padding mechanism is confirmed as the locus.**
I wrote that hypothesis and the measurement killed it; it stays on the page rather
than being edited out.

**But padding's FLOP arithmetic still does not account for the size of it**, and
that part of the original analysis survives. Within K2 the work is out-proj 268.4 +
attention 268.4 + FFN 536.9 = 1073 MFLOP per layer. Padding takes attention to
536.9, so K2's work rises to 1341 MFLOP — **+25%**. K2's *time* rises **+113%**.
Achieved throughput inside K2 therefore falls from **14.5 TF/s to 8.5 TF/s**, a 41%
efficiency loss on top of the extra arithmetic.

**Final reading.** The `head_dim`-8 penalty is real, is confined to K2, and is
roughly half extra arithmetic (the `tl.dot` 16-wide minimum) and half efficiency
collapse on the padded path — the padded dots do twice the work *and* run at
lower throughput than the same kernel achieves at `head_dim` 32. It is one
mechanism with two costs, not two mechanisms.

Note also what this says about the whole-block picture: attention is only 14.3% of
the block's FLOPs but dominates K2's *time*, since doubling it roughly doubles a
kernel that also contains the out-projection and the entire FFN. Any future work on
these shapes should treat attention as the expensive part of the fused block
despite its small FLOP share.

## Finding 3 — shape 11 is the clearest remaining optimization target on the board

It is the only shape whose candidate is measurably inefficient *relative to a
directly comparable sibling running the same code on the same arithmetic*. Every
other shape's low MFU is explained by the problem being small (grid cannot fill 38
SMs). Shape 11's is not: shapes 1, 9 and 10 reach MFU ~0.41 on identical work.

Closing the gap entirely would take shape 11 from 12.68x to roughly 20.8x
(0.9175 → 0.5605 ms against an unchanged 11.6337 ms baseline). On the twelve-shape
geometric mean that is worth about **+4.2%** (9.68x → ~10.09x), which is real but
is not a headline — and the estimate assumes the gap closes completely, which
nothing yet supports. **It is recorded as a bounded target, not a plan**, and per
LESSONS 35 no numeric prediction band is attached to it.

## Sources

Measured rows: `Project/authority/blobs/` measurement packets bound to
`run-b2309764be77beec950ca57cc5dfe53f` (shape 1),
`run-dfaef515a4889d3a433cd0fdab78abd9` (9),
`run-46f1df2be123a1bee048a70cc00a5a11` (10),
`run-21cf42fde9f95eea989fb85f9c715e38` (11). FLOP counts cross-checked by hand
against [roofline-table.md](roofline-table.md) line 7.
