# WIN PLAN — how the twelve conditions in WIN_BAR.md go green

Written 31 Aug 2026 after reading the complete repository: both webinar transcripts, the
README, all three external audits, HANDOVER, PLAN, RUNBOOK, GRIND_ENTRYPOINT, STATE,
LESSONS (59), DECISIONS (2,277 lines), HOTSPOT_COVERAGE, all 15 research notes, all four
judge-facing drafts, `dispatcher_region.py` in full, and the design rationale of all 29
kernels.

**The bar is `Project/WIN_BAR.md`. This file is only the route to it.**

---

## The one ordering mistake this plan exists to avoid

Every earlier version of this plan said *measure the board first, then optimise.* That is
backwards and it would waste the board: **any kernel change invalidates every row measured
before it.** This project has already paid that price four times over — the headline has
been 10.32× → 2.94× → 9.68× → 9.45× → 10.14×, and each move was a re-measure forced by
either the wrong artifact or a new one.

So the order is: **fix the instrument → change the kernels → FREEZE → measure once.**

> **Ordering bug found and fixed in the second revision of this plan.** The first version
> put the evaluator fix in Phase 5b — *after* the Phase 4 freeze — while lever **3f (shape
> 14 tuning) depends on that fix.** The single largest lever in the plan was therefore
> scheduled after the point at which no kernel may change. Corrected below: the owner fix
> is **Phase 0, requested at T=0 and running in parallel with everything**, and all shape
> 6/14 work happens **before** the freeze.

---

## PHASE 0 — request the owner fix immediately, then carry on without waiting

Two one-line changes, both in LOCK-protected files that are Write-denied to me (correctly —
they are the code that decides whether our own claims are true):

- `Project/tools/shape6_local_eval.py:146` — device set at line 296
- `Project/tools/shape14_eval.py:274` — device set at line 289

Both compare `mask.device != device` where `device = torch.device("cuda")` (no index) and
the official generator returns the mask on `cuda:0`. PyTorch treats those as unequal, so
both evaluators abort on their first check. Fix: compare `mask.device.type != device.type`,
or normalise with `device = torch.zeros(0, device=device).device`.

**This gates conditions 1, 2 and 3 — the entire no-zero half of the bar — and lever 3f.**
Nothing else in this plan waits on it, so it is requested first and worked around until it
lands.

---

## PHASE 1 — prove the instrument. Nothing else is meaningful first.

**Why first.** On 31 Aug, byte-identical bytes returned **11.2516× and 9.7638× on shape 12
— a 13.2% spread**, while the campaign's calibrated noise floor for that class reads
~0.15%. The floor is wrong by two orders of magnitude because calibration times the
baseline against itself *inside one process*, which measures seconds-of-steadiness, not
run-to-run reproducibility (LESSONS 59). Lengthening the autotuner's benchmarking window
halved it to 5.1% **and raised the mean 9%** — the short window was crowning genuinely
worse kernels. GPU clocks are now locked at 1665/7001 MHz.

**Do:** three byte-identical screening runs of the current artifact on shape 12 (the
noisiest shape on the board), locked clocks, quiet box.

**Gate — per lever, not global.** The first draft said *"spread > 5% aborts the optimisation
phases entirely."* **That is wrong and would have thrown away work that stays perfectly
measurable.** A 5% instrument cannot resolve the 3.3–4.6% input copy, but it resolves shape
8's 24% elementwise tax without difficulty. The rule is a comparison, not a cliff:

> **Run a lever only if its arithmetic ceiling exceeds the measured spread.
> Kill the rest on paper.**

Against the eight levers, at each candidate spread:

| measured spread | dies | survives |
|---|---|---|
| < 2% | nothing | all eight |
| 2–5% | 3a (3.3–4.6%) is marginal — needs replicates to claim | 3b, 3c, 3d, 3f, 3g, 3h, 3e |
| 5–10% | 3a, 3c (≤5%), 3g on most shapes | 3b (24%), 3f, 3d, 3h |
| > 10% | everything except 3b and 3f | 3b (24%), 3f (untuned shape) |

Note that **3f survives at every resolution** — an untuned kernel on the heaviest shape is
not a few-percent question — which is a second reason it is not scheduled last.

The measured spread is recorded once and becomes the floor under every later claim
(condition 5). Nothing smaller than it is ever published.

**Cost:** ~20 minutes, 3 permits. Needs nothing from the owner.

---

## PHASE 2 — buy back the precision margin. Cheap, and it protects a zero.

**Why.** Shape 6's last measured max absolute error was **0.00184 against a 0.002 limit —
92% of the budget, on a single seed.** Under the organizer's rule a shape that fails
precision scores **zero**, and shape 6 is 1,174 GFLOP, the second-heaviest workload on the
board. This is the single largest downside risk in the project.

The shipped LayerNorm computes variance as `E[x²] − mean²` at all three sites
(`dispatcher_region.py:338, 566, 663`). That is the textbook catastrophic-cancellation
form: when the mean is large relative to the standard deviation, both terms are large and
nearly equal, and the subtraction destroys precision. It came from `k019`, and **k019's own
record shows it earns nothing** — shape 1 returned 8.3732× against 8.3830× and the
falsifier fired (`gate_log.jsonl:462`).

**Do:** revert all three sites to the two-pass form (`mean` first, then `E[(x−mean)²]`).
Rebuild, verify byte-identity outside the sanctioned region.

**This is insurance, and the first draft of this plan wrongly called it "free."** k019
measured **+3.3% on the module and nothing integrated — but only on shape 1.**
`_sub_norm_qkv` carries up to **37.1% of device time (shape 10)**, so reverting could cost
up to roughly **1% end-to-end** on the shapes where that kernel is heaviest. That is a
price, not zero.

**Take the price anyway, and here is the arithmetic that says so.** Paying ≤1% across
eleven shapes buys error headroom on a **1,174-GFLOP shape sitting at 92% of its budget**,
where the failure mode is not a slow row but a **zero**. Expected value is not close.

**Gate:** measure it on **shape 10** (where `_sub_norm_qkv` is heaviest), not shape 1
(where k019 was measured and found nothing). If the cost exceeds the Phase-1 resolution,
keep the revert anyway but record the measured price — and if it exceeds 2%, gate the
one-pass form to the shapes that are not near the precision cliff instead of reverting
board-wide.

**Cost:** ~30 minutes. Needs nothing from the owner. **Before the freeze**, so the margin
is in the bytes that get measured.

---

## PHASE 3 — the eight untried levers, cheapest-and-broadest first

> **Twice revised under challenge.** Draft 1 listed **three** levers. Stress-testing it
> found three more (3a graph input copy, 3c mask sync, 3f shape-14 tuning) — two of them
> *cheaper and better evidenced* than two I had picked. A second pass found two more
> (3g `_sub_final_norm`, 3h the shape-11 recheck). **Three became eight**, and the two
> best-value entries in the list — 3a and 3f — were both missing from the first draft.
>
> The pattern in what I missed is worth naming, because it will recur: **I searched the
> components I had already been working on and skipped the ones adjacent to them** — the
> input copy sits beside the output clone I removed, `_sub_final_norm` sits beside the three
> kernels I profiled, and shape 14 was invisible only because it cannot currently be
> measured. **Unmeasurable is not the same as searched.**

**Run order:** 3a → 3g → 3b → 3c → 3d → 3h → 3e, with **3f jumping the queue the moment
Phase 0 lands.** Cheapest and broadest first; anything that fails its arithmetic ceiling
dies on paper without a run.

### 3a. The CUDA graph INPUT copy — cheapest, broadest, and already proven once

**Why this is first.** `HOTSPOT_COVERAGE.md` states it plainly: *"The input copy is
untouched."* It is **3.3%–4.6% of device time on every one of the eleven fused shapes**
(and 0.89% on shape 8), one `Memcpy DtoD` per forward, cost proportional to input bytes.

**It is the same mechanism that produced the best gain-per-effort of the entire campaign.**
Removing the *output* clone was worth **+0.87% to +7.04% on six shapes** (LESSONS 55), and
it became possible only after someone read `torch_transformer_benchmark.py` and established
that neither the timing loop (line 494, result discarded) nor the accuracy loop (line 392,
compared on the next line) retains the returned tensor. **That same reading has never been
done for the input side.**

**READ, AND CONFIRMED — this is no longer a question.** `torch_transformer_benchmark.py`:

- **line 529** — `x, valid_mask = generate_random_case(...)` creates the input **once**;
- **lines 539–540** — both warmups take that same `x`;
- **lines 546–560** — every timed round passes that same `x` to `benchmark_once`;
- **line 483** — `benchmark_once` holds `x` as a parameter and never regenerates it;
- **line 525** — the script prints it out loud: *"timing excludes random-data generation and
  uses a fixed input."*

So across warmup and the whole timed region the input is **one tensor object, at one
address, with unchanging values**. `state["static_x"].copy_(x)` therefore copies identical
bytes from the same source to the same destination **on every timed iteration**. It is
provably redundant for the entire measured region.

**The change.** Capture the graph against the caller's tensor directly instead of a private
`static_x`, so no copy exists at all. Guard it exactly as the existing metadata guard works
(`dispatcher_region.py:1096`): record `x.data_ptr()` at capture and compare on replay. The
accuracy loop passes a fresh tensor per trial, so its pointer differs, the guard fires, and
it takes the safe path — which is the correct outcome, not a hazard. In-place mutation of
the same buffer is likewise correct: a graph reading live memory computes the new values.

**Ceiling:** 3.3–4.6% on eleven fused shapes, 0.89% on shape 8. **Cannot-help shape:**
shape 6 (runs eagerly, no graph) — the built-in null control. **This is the same species of
find as the output clone, which was the best gain-per-effort of the campaign, and it comes
from the same act: reading the caller instead of the kernel (LESSONS 55).**

### 3b. Shape 8's untouched elementwise tax — heavy under any weighting

These are the only large components with no measured search. Everything else on the fused
route has been attacked: `_sub_norm_qkv` took four candidates (k011 occupancy split lost
15%; k022 grid promotion won and shipped; k018/k023/k024 CUDA lost 3× after five fixes;
k019 won nothing), and `_sub_attn_block_tail` took four (split-at-LayerNorm killed on
arithmetic; k025 live-set shrink lost; k026 pipelining lost; k027 erf won and shipped).
Those two are searched. These three are not.

**Each lever follows the same protocol, and it is not optional:**

1. Write the arithmetic ceiling **before** building. If the ceiling sits under the Phase-1
   resolution, close it on paper and move on. (LESSONS 32.)
2. Name the shape where the mechanism **cannot** help, and measure that one too — this is
   how shape 7 (−5.1%) and shape 13 (−6.9%) were shipped as regressions. (LESSONS 53.)
3. Diagnostic → screening → keep or kill. Kill fast; a lost candidate costs one run.

### 3a. Shape 8's untouched elementwise tax — heavy under any weighting

**The evidence.** Shape 8's profile (`profile-79950fcb`, the newest in the repo) splits as
**62.7% two vendor GEMMs, 24.0% three generic `at::native` elementwise kernels across 640
launches, 11.2% authored Triton.** On the eleven fused shapes that entire class of work was
fused away. Shape 8's route never received it, and `HOTSPOT_COVERAGE.md` records **no
dedicated search of this route, ever.**

**What those 640 launches are.** Reading `_fp16_forward` (`dispatcher_region.py:926-974`):
per layer it does `.transpose(1,2).contiguous()` on **q, k and v** and again on `context`
— four full copies of a [B,H,S,D] tensor — plus two residual adds and their `.float()`
casts. At d_model 1024 those are large.

**The fix, and why it is nearly free.** `_sub_attn_fwd` **already takes explicit strides**
for q, k, v and out (`dispatcher_region.py:208-211`). The `.contiguous()` calls exist only
because the launch reshapes first. Passing the transposed strides directly removes three
large copies per layer without touching the kernel's arithmetic. The residual-add and cast
can fold into the existing `_sub_ln_fp16` the same way `k010` folded LayerNorm and GELU.

**Ceiling:** removing the copies addresses up to 24.0% of shape 8's device time → up to
**1.32×** on the second-heaviest shape. Realistically less; the gate is Phase-1 resolution.

**Shape where it cannot help:** every fused shape (they do not take this branch) — so they
are the built-in null control and must measure unchanged.

### 3c. The mask host synchronization — a full sync on every forward

**The evidence.** `bool(valid_token_mask.all())` runs at `dispatcher_region.py:1029` on
every forward. Device-side cost is measured and material on the small shapes: **5.00% on
shape 2**, 4.52% on shape 3, 4.38% on shape 7, 3.43% on shape 12 (reduce kernel plus the
`Memcpy DtoH`). `HOTSPOT_COVERAGE.md` marks this row **"NO — never searched."**

**The honest limit, which is why this is third and not first.** The *host*-side figure is
not an opportunity and must not be treated as one. `cudaStreamSynchronize` reads 75% of self
CPU on shape 9 and **94.05% on shape 8** — but on shape 8 `Self CUDA time total` is 377 ms
against a 379 ms host total, so the host is blocked because the device is *busy*. Reading
that 94% as headroom would be wrong by nearly the whole amount.

**So the experiment must separate queue-drain from a genuine stall before anything is
built.** The device-side 3–5% on the small shapes is real and removable; the host-side
number is not established as anything.

**Two candidate routes, both avoiding the cache hazard the external audit warned about
(C6):** make the kernel consume the mask directly so no host decision is needed, or copy
the mask into graph-static storage and branch inside the kernel. **A mask *cache* keyed on
object identity is explicitly rejected** — in-place mutation, storage reuse and view
aliasing all defeat it, and we do not currently have that bug. Do not introduce it.

**Ceiling:** 3–5% on the four small shapes. **Cannot-help shape:** 13 (0.30% device).

### 3d. Sequence-persistent CTAs — the biggest un-attempted idea on the board

**The evidence.** Shapes 2, 3, 7 and 12 sit at **0.03–0.32 MFU** because the grid cannot
fill 38 SMs. `megakernels-persistent.md` names the fix and states it is **Triton-expressible
with no cross-CTA synchronisation at all**, because sequences are independent under causal
attention. Our own DECISIONS log priced it at **~1.2× on four shapes ≈ +6.3% geomean —
larger than any other remaining option** — and then **declined it for time and approval, not
for evidence.**

**I was wrong to call these shapes "physics, not headroom."** Grid starvation is a property
of the work decomposition, which is a choice. k017 already moved it once: splitting
attention took shape 2 from 8 CTAs to 32 and that shape gained **+18.2%**.

**Do the cheap version first.** Full persistence is a large rewrite. `_sub_attn_block_tail`
still launches on grid `(q_tiles, B)` — **8 CTAs on shape 2**. Promoting the FFN hidden
dimension to a grid axis is the *same mechanism that already won* on `_sub_norm_qkv` (k022,
+24.1% on shape 2) and costs a fraction of the work. Try that first; escalate to full
sequence-persistence only if it pays and there is more to take.

**Ceiling:** grid 8 → 16/32 CTAs on the starved shapes.
**Shape where it cannot help:** shape 13 (65,536 tokens — the grid is already full, and this
is exactly where k022's identical mechanism cost **−6.9%**). Measure it or repeat LESSONS 53.

### 3e. Shrinking `_sub_attn_heads`' live set — ranked last, and may close on paper

`k020` attacked this kernel by making K/V resident — it **enlarged** the live set and lost
(32.15 µs against 24.54 µs). The lesson drawn was "shrink, don't enlarge," and nothing has
tried shrinking. But the accumulator is `[BLOCK_M, HD_PAD]` fp32 and **autotune already
sweeps BLOCK_M**, so much of the shrink space is covered. Write the arithmetic first; if it
does not clear the Phase-1 resolution, close it on paper and say so.

### 3g. `_sub_final_norm` — the one kernel nobody gave a config list to

**Missed in both earlier drafts.** It is **3.0%–10.9%** of device time, and
`HOTSPOT_COVERAGE.md` dismisses it as *"too small to justify a beam on any shape except 2
and 3"* — which concedes it matters on two shapes and then does nothing about them. Look at
the launch (`dispatcher_region.py:920-923`):

```python
_sub_final_norm[(triton.cdiv(tokens, 128),)](
    ..., BLOCK_T=128, num_warps=4)
```

**It is the only kernel in the fused route that is not autotuned.** `BLOCK_T` and
`num_warps` are hardcoded. On shape 2, `tokens` is 128, so the grid is `cdiv(128,128)` =
**one CTA on a 38-SM card** — the identical grid starvation that k022 fixed on
`_sub_norm_qkv` for +24.1%, sitting untouched in the one kernel that never got a config
list.

**Ceiling:** up to 10.9% of device time on the shapes where it is largest, and those are
exactly the starved ones. **Cost:** adding an autotune decorator — the cheapest item in
this plan. **Cannot-help shape:** 13 (3.0%, grid already full).

### 3h. Re-check shape 11 before dismissing it — the target was closed on two routes, not on evidence

Shape 11 sat at **MFU 0.25 against 0.41 for shapes 1/9/10** — the same 7.52 GFLOP problem
at a different head count, and **the only shape on the board measurably inefficient against
a directly comparable control.** `head-count-scaling.md` calls it "the clearest remaining
optimization target."

It was then dismissed twice: head-packing is arithmetically impossible (QK^T contracts over
the feature axis), and `tl.sum`-reduce trades tensor cores for CUDA cores. **Both are true
about those two routes and neither is true about the target** — which is precisely the
LESSONS 49 error, and I have now made it twice in this plan.

Meanwhile k017 moved shape 11 **+40.3%** (12.59 → 17.66), which should put its MFU near
0.35. **Nobody re-measured it.** So the first action is not a kernel: it is one line of
arithmetic on the post-split numbers. If the gap to 0.41 has closed, the target is closed
*with evidence*. If it has not, the residue is quantified and gets its own beam.

**Cost:** a recomputation from Phase 6's board. **Do not build anything here until that
number exists.**

### 3f. Shape 14's kernel has NEVER been performance-tuned — and it is the heaviest shape

**This is not a fallback. It is arguably the largest single lever in the project, and it
was invisible because the shape cannot currently be measured.**

`DECISIONS.md`, 28 Aug: shape 14's core was proven correct on this card — seq 100,000
causal, zero tolerance violations — and then **"100k perf tuning deferred (configs target
short seqs)."** It has never been revisited. `k014` is `k010` plus long-sequence memory
discipline; its autotune configs were written for `seq_len` 128.

Shape 14 is **1,391,250 GFLOP — roughly 11,000× a typical shape and 1,180× shape 13.**
Under any FLOP- or work-weighted score it dominates everything else combined, and our own
research note warns never to pick a target off that axis alone — but equally, never to
leave the top of it untuned.

`Project/research/megakernels-persistent.md` already names the technique: shape 14 is
**~94% attention by useful FLOPs**, so its lever is an authored FlashAttention-2-style
kernel — online softmax, causal tiling, query-block parallelism within each head — **not**
megakernel or GEMM work. Two of those three (base-2 softmax, causal block elimination) were
built for the fused route on 31 Aug as `k028` and are sitting in
`dispatcher_region.py` right now, gated at `seq_len >= 256`. **Shape 14 does not use that
kernel** — `d_model` 1024 routes to `_fp16_forward`, which calls `_sub_attn_fwd`, a
different kernel that still uses `tl.exp` and an ungated causal mask on **every** block.

At `seq_len` 100,000 the causal loop split is worth vastly more than it is at 1,024: the
fraction of key blocks that lie strictly below the diagonal approaches 1, so nearly every
block skips both the mask and the bounds check. On shape 13 that change was worth −7.9% on
the attention kernel; shape 14's inner loop is ~100× longer.

**Gate:** blocked until **Phase 0**'s evaluator fix makes shape 14 measurable at all —
which is why that fix is requested at T=0 rather than scheduled late. The moment it lands,
this jumps the queue: it is the heaviest shape on the board and the least optimised, and it
must be done **before the freeze**, not after.

---

## PHASE 4 — FREEZE

One `build_submission.py` run, one sha, `verified: true`. **No kernel change after this
point.** Every row of the board carries this hash (condition 6).

---

## PHASE 5 — measure once, completely

**5a. The twelve.** Screening run per shape on the frozen sha. Fast path: `delta` needs no
research cycle; each shape needs one diagnostic first, for counter-evidence bound to these
bytes.

**5a-bis. Shape 6's missing CUDA graph — one free check, not a lever.**
Shape 6 is the **only** shape that runs eagerly: `dispatcher_region.py:1051` routes
`B*S >= 262144` away from graph capture, and the comment justifying it says capture memory
at that size is **"unproven"** in its own words. The launch argument is probably sound —
at 1.28M tokens every kernel runs for milliseconds, so launch savings really are
negligible — but "probably" has cost this project a headline four times. It rides along
with the shape-6 measurement at no extra cost. Measure it; do not assume it.

**5b. Shapes 6 and 14 — measured on the frozen bytes.** The evaluator fix landed back in
Phase 0; shape 14's tuning (3f) landed before the freeze. This is the final measurement:
shape 6 at **≥5 seeds** (currently one), shape 14 validate → decomposition check → full
32-slice evaluation.

**5c. MFU.** Compute per shape from the measured candidate medians and the known FLOP
counts, against **both** 32.4 and 64.8 TF/s. No re-running needed — the inputs already
exist (conditions 7).

**5d. Three weightings.** Equal-weight, FLOP-weighted, bandwidth-aware. The organizer has
not decided the weights, and the two extremes disagree violently: under FLOP-weighting
shape 14 alone is ~99.9% of the score and shape 2 rounds to nothing; under equal-weighting
they count the same. **Reporting one number is a bet. Reporting three is the hedge the
organizer's own uncertainty demands** (condition 8).

---

## PHASE 6 — one source of truth

Reconcile every number to the frozen board. Today three documents carry three different
headlines — **10.3×** (a figure we formally withdrew as procedurally invalid), **9.45×**,
and **10.14×** — and none matches the current build. Conditions 11 and 12.

---

## Abort conditions, stated in advance

- **Phase 1 spread > 5%** → skip Phase 3 entirely. Optimisation is unmeasurable; spend
  everything on Phases 2, 4, 5, 6.
- **Any Phase 3 lever fails its arithmetic ceiling** → close it on paper, no run.
- **Any Phase 3 lever regresses its "cannot help" shape beyond resolution** → gate it or
  revert it. Do not ship it board-wide. (This is exactly how shapes 7 and 13 were
  regressed.)
- **Shape 6 measures above 2e-3 on any seed** → stop all optimisation. A zero on a
  1,174-GFLOP shape outweighs every gain available anywhere else.

## If every lever fails — the answer the first draft did not have

Eight levers can all lose. Two of the last three candidates in this project did. So:

**A 0-for-8 result is itself a finding, and it is not a dead end.** It would mean the fused
route is genuinely searched — which is exactly the claim `HOTSPOT_COVERAGE.md` exists to
let us make honestly, and which no competitor can make about their own work without the
same table. In that case the remaining time goes to **3f**, because shape 14 has never been
tuned at all and carries more work than every other shape combined; tuning the heaviest,
least-optimised shape is strictly better than re-litigating eleven that have taken ten
candidates between them.

**And the floor is not low.** Even at 0-for-8 the position is: fourteen shapes measured and
correct on one artifact, MFU reported under three weightings, every hotspot either searched
or closed with the measurement that closed it, and kernels already running 2.02×–33.65×.
The bar in `WIN_BAR.md` is written so that all twelve conditions can go green **without a
single further speedup** — because completeness and defensibility are what a judge can
check, and speed we cannot prove is worth less than speed we can.

## What is deliberately not in this plan

`_sub_attn_block_tail` and `_sub_norm_qkv` beyond what has shipped — eight candidates
between them, and the last three lost. `int8` — measured at 3.5e-2 against a 2e-3 limit.
CUDA C++ rewrites — 149.7 µs against Triton's 49.3 after five rounds of fixes. Head-packing
for `head_dim` 8 — arithmetically impossible, QK^T contracts over the feature axis.
