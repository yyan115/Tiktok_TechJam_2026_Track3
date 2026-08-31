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

**Gate:**
- spread **< 2%** → condition 4 green; the resolution number is recorded and becomes the
  floor under every later claim (condition 5).
- spread **2–5%** → usable, but no delta below the spread may ever be claimed. Proceed.
- spread **> 5%** → **ABORT the optimisation phases entirely.** At that resolution no
  kernel change can be told from noise, and the only honest use of the remaining time is
  Phases 2, 4, 5 and 6 — correctness, coverage and a clean board.

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

**Gate:** shape 1 screening must land within the Phase-1 resolution of its current value.
If it costs more than that, keep the one-pass form and instead attack shape 6's margin
directly once it is measurable.

**Cost:** ~30 minutes. Needs nothing from the owner. **Do this before the freeze** so the
margin is in the bytes that get measured.

---

## PHASE 3 — the three untried levers, hardest-hitting first

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

### 3b. Sequence-persistent CTAs — the biggest un-attempted idea on the board

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

### 3c. Shrinking `_sub_attn_heads`' live set — ranked last, and may close on paper

`k020` attacked this kernel by making K/V resident — it **enlarged** the live set and lost
(32.15 µs against 24.54 µs). The lesson drawn was "shrink, don't enlarge," and nothing has
tried shrinking. But the accumulator is `[BLOCK_M, HD_PAD]` fp32 and **autotune already
sweeps BLOCK_M**, so much of the shrink space is covered. Write the arithmetic first; if it
does not clear the Phase-1 resolution, close it on paper and say so.

---

## PHASE 4 — FREEZE

One `build_submission.py` run, one sha, `verified: true`. **No kernel change after this
point.** Every row of the board carries this hash (condition 6).

---

## PHASE 5 — measure once, completely

**5a. The twelve.** Screening run per shape on the frozen sha. Fast path: `delta` needs no
research cycle; each shape needs one diagnostic first, for counter-evidence bound to these
bytes.

**5b. Shapes 6 and 14 — the zero-risk, and the only owner dependency in this plan.**

Both evaluators abort on their first check. `device = torch.device("cuda")` is compared
against tensors that materialise on `cuda:0`, and PyTorch treats those as unequal:

- `Project/tools/shape6_local_eval.py:146` (device set at line 296)
- `Project/tools/shape14_eval.py:274` (device set at line 289)

Fix either way: compare `mask.device.type != device.type`, or normalise first with
`device = torch.zeros(0, device=device).device`. **Both files are LOCK-protected and
Write-denied to me — correctly, since they are the code that decides whether our own claims
are true.** One line each; it unblocks conditions 1, 2 and 3.

Then: shape 6 at **≥5 seeds** (currently one), shape 14 validate → decomposition check →
full 32-slice evaluation.

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

## What is deliberately not in this plan

`_sub_attn_block_tail` and `_sub_norm_qkv` beyond what has shipped — eight candidates
between them, and the last three lost. `int8` — measured at 3.5e-2 against a 2e-3 limit.
CUDA C++ rewrites — 149.7 µs against Triton's 49.3 after five rounds of fixes. Head-packing
for `head_dim` 8 — arithmetically impossible, QK^T contracts over the feature axis.
