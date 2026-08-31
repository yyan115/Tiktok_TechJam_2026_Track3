# HOTSPOT_COVERAGE — what share of the route each component holds, and whether anyone has looked at it

Written 31 Aug 2026. Internal working document. Nothing here is a claim about speedup;
every number is a share of profiled device time or a per-call kernel time, read off a
named profile.

## The rule this table exists to enforce

**Never declare a route exhausted while a component holding a large share of runtime has
had no dedicated architectural search.**

This is not a general principle, it is a specific correction. `_sub_attn_block_tail` held
about 35% of device time on most fused shapes for three days and was never opened. It was
not skipped for a reason — it was skipped because nothing in the loop asked "which of these
is large and unsearched?" When it was finally opened it gave up a measurable result
(`k027`, below) within one beam round. The cost of not having this table was those three
days.

The table below is the checklist. A component is only allowed to be dropped from the
search when its "architectural beam done?" column says YES **and** the beam actually failed
— not when attention drifted elsewhere.

## How to read the numbers

- **Source.** `Project/loop/profile_evidence/profile-<id>.raw/raw/torch-profiler-key-averages.txt`,
  with `profile-<id>.json` giving the shape and `target_sha256`. Every figure below carries
  its profile id.
- **Per-call time.** Kernels inside the layer loop are called 80 times per profile
  (4 layers x 20 iterations). The per-call figure — one layer, one iteration — is
  `Self CUDA / 80`, which is exactly the profiler's own `CUDA time avg` column, and that is
  what every historical microsecond figure in this project refers to. Checked: shape 2
  `_sub_norm_qkv` reads 3.952 us in `profile-da728b009a598d1f3ed08163`, matching the
  "8.502 -> 3.954 us" claim in `Project/loop/gate_log.jsonl:384`. Divide by 20 instead and
  you get a per-forward figure four times larger; do not mix the two.
- **No cross-build averaging.** The eleven fused shapes were last profiled on **four
  different builds** of `Project/submission/torch_transformer_benchmark_submission.py`.
  Every row states which. A percentage from one build is not comparable in detail with one
  from another, and none of them have been averaged.
- **Shapes 6 and 14 have no profile of any kind** — not one exists in
  `Project/loop/profile_evidence`. They are not locally runnable. Nothing in this document
  covers them.
- **`ncu` has never been run.** Nsight Compute requires root and is deliberately absent
  from this project's command allowlist (`Project/memory/DECISIONS.md:233`, and
  `LESSONS.md:24` records that it was once wrongly listed as a tool in use). So there are
  **no occupancy, register, or spill counters anywhere in this repo.** Wherever a limiting
  mechanism below is stated as register pressure or live-set size, it is an inference from
  wall-clock behaviour, not a measurement.

---

## The table

| component | % of device time | measured on shapes | profiled? | architectural beam done? | what is known to limit it | next untried idea |
| --- | --- | --- | --- | --- | --- | --- |
| `_sub_norm_qkv` | 20.9% – 37.1% | 11 fused shapes (1,2,3,4,5,7,9,10,11,12,13) | yes | **YES** — grid-dim promotion of the chunk loop; gated after two regressions | Grid starvation on low-token shapes; the gate's cost term is a 3x re-read of `x` that grows with token count. Live-set argument is inference, not counters | Chunk count is a binary 1-or-3 step. It has never been swept past 3, and the token sweep has an untested gap between 2048 and 8192 |
| `_sub_attn_heads` | 12.0% – 48.8% | same 11 fused shapes | yes | **NO** — never had a dedicated search | Nothing established. One failed candidate (`k020`) and a roofline-derived efficiency estimate, both below | Never opened. The nearest concrete lead already in the record: it re-reads K/V about 2.5x under causal tiling and rescales the accumulator every inner iteration |
| `_sub_attn_block_tail` | 20.4% – 41.0% | same 11 fused shapes | yes | **YES, just completed** — four candidates, one net gain | The exact-erf GELU is worth 5.1 us/call on shape 1, measured by deletion. Register pressure is **not** established | ~1.9 us of the 5.1 us GELU cost is still unrecovered after `k027` |
| device-to-device copies (CUDA graph input copy) | 3.3% – 4.6% post-clone-removal; 3.3% – 7.3% on builds that still clone | 11 fused shapes + shape 8 | yes | **PARTIAL** — output clone removed, input copy untouched | The input copy is one `Memcpy DtoD` per forward, cost proportional to input bytes | Whether the benchmark's input tensor is stable enough across calls to write into the graph's static input buffer directly. This is a question, not a plan; the output clone only became removable after reading the official script |
| `valid_token_mask.all()` host synchronization | device side 0.30% – 5.00%; **host side not measured** | 11 fused shapes + shape 8 | device side yes; the stall itself no | **NO** | Device cost is small and known. The `cudaStreamSynchronize` figure is host wall time that includes waiting for queued GPU work and is **not** established as recoverable | Nothing can be proposed until an experiment separates queue-drain from a genuine stall. See the shape-8 argument below |
| fp16 route (shape 8) | 62.7% vendor GEMM, 24.0% generic elementwise, 11.2% authored Triton | shape 8 only | yes, once | **NO** — no dedicated search at all | Nothing established | Never opened. 24.0% of device time sits in three generic `at::native` elementwise kernels that no fusion has touched |
| `_sub_final_norm` | 3.0% – 10.9% | 11 fused shapes | yes | no, and correctly so — too small to justify a beam on any shape except 2 and 3 | — | — |

---

## Per-component detail

### `_sub_norm_qkv` — LayerNorm + Q/K/V projection

Share by shape, each from that shape's most recent shipped-file profile (appendix A):
37.1% (10), 36.2% (1), 35.9% (12), 35.8% (9), 34.9% (5), 33.0% (4), 30.7% (11), 28.8% (3),
24.3% (2), 23.6% (7), 20.9% (13).

**Beam: done, and it produced the change that shipped.** The chunk loop used
`tl.static_range(3)`, which unrolls, so up to three 128x128 fp16 weight tiles could be live
at once. Promoting the chunk to a grid dimension moved shape 2 from **8.502 to 3.954 us per
call** — sourced to `Project/loop/gate_log.jsonl:384`, whose `counter_evidence_id` is
`profile-71a67b2cf926f0dba4fe9b56`. The current shape-2 profile
(`profile-da728b009a598d1f3ed08163`) reads 3.952 us, consistent.

**It is gated, and the gate was earned by two regressions**, both recorded inline at
`Project/submission/dispatcher_region.py:738-760`:

- width: at `d_model` 32 the weight tile is 2 KB, so there is no register pressure to
  relieve while the duplicated LayerNorm is paid in full — **shape 7 measured -5.1%**;
- token count: 128 tokens +24.1%, 512 tokens +15.8%, 2048 tokens +5.4% and +0.6%,
  8192 tokens -0.1% and -1.3%, **65536 tokens -6.9% (shape 13)**.

Hence `qkv_chunks = 3 if (D >= 64 and tokens <= 4096) else 1`. The comment states plainly
that 4096 sits in the untested gap between the last measured gain and the first measured
loss.

**What is asserted rather than known:** that three live weight tiles were the mechanism.
The kernel got faster when the tiles stopped being simultaneously live, which is consistent
with register pressure and also consistent with the grid tripling from 1 to 3 programs on a
38-SM card. The dispatcher comment treats these as two independent terms, which is the
honest reading. Separating them needs occupancy and register counters, i.e. `ncu`, which
this project cannot run.

Also on this kernel: single-pass LayerNorm statistics (`k019`) measured +3.3% on the module
and **nothing measurable once integrated** — shape 1 returned 8.3732x against 8.3830x and
the falsifier fired (`Project/loop/gate_log.jsonl:462`). The code is still in the shipped
file (`dispatcher_region.py:286-287`) because it is not a loss and is simpler, but it earns
no claim.

### `_sub_attn_heads` — causal flash attention, grid `(q_tiles, B, H)`

Share by shape: 48.8% (13), 41.3% (7), 30.6% (11), 22.4% (9), 20.2% (5), 19.0% (10),
18.9% (4), 17.9% (1), 15.8% (3), 15.6% (2), 12.0% (12).

**This is the gap.** It is the largest single component on the two shapes where the route
is slowest relative to its own roofline, and **it has never had a dedicated architectural
search**. The only thing ever tried against it was `k020`, sequence-resident K/V, and that
was a beam *member* on a card aimed at `_sub_norm_qkv`, not a search of this kernel.

What is on the record about it:

- **`k020` failed: 32.15 us against 24.54 us for the streaming kernel it replaced**, on
  shape 1 (`Project/loop/gate_log.jsonl:378` and `:381`). The reading taken at the time was
  that holding K, V and the transpose in registers spills, so the card is register-pressure
  limited rather than memory-traffic limited.
- **That reading is an inference and must not harden into a fact.** One candidate that
  *enlarged* the live set got slower. That is consistent with spilling and also with
  several other things. It does not measure spilling, and it does not prove that a
  candidate which *shrinks* the live set will get faster. Proving either direction needs
  Nsight Compute counters, and `ncu` requires root and is not on this project's allowlist.
- **Roofline efficiency ~34%, the worst of the three fused kernels** — the other two sit at
  about 50% and 52% of the fp16-with-fp32-accumulate roof
  (`Project/loop/gate_log.jsonl:373`). This is derived from
  `Project/research/roofline-table.md`, not from hardware counters, and the same log entry
  is where the "50, 52 and 34 percent" figures originate.
- The same entry names a concrete unexploited property: the kernel streams over sequences
  of 128 that already fit, paying a roughly 2.5x redundant K/V read under causal tiling
  plus a full accumulator rescale every inner iteration. **Nothing has been built against
  that.**

### `_sub_attn_block_tail` — out-projection + residual + LayerNorm2 + FFN(GELU) + residual

Share by shape: 41.0% (2), 39.2% (12), 38.2% (3), 36.0% (4), 35.3% (10), 34.8% (5),
34.0% (1), 33.5% (9), 28.6% (11), 23.3% (13), 20.4% (7).

**Beam: complete.** Four candidates, all profiled on shape 1 against
`Project/kernels/*.py` modules. These four profiles use the unprefixed kernel names
(`_block_tail`, `_norm_qkv`, `_attn_heads`, `_final_norm`):

| candidate | target | profile id | `_block_tail` us/call |
| --- | --- | --- | --- |
| split at LayerNorm | — | rejected on arithmetic, never built | — |
| `k025` FFN hidden in halves | `Project/kernels/k025_tail_halfffn.py` | `profile-3238c17e01a2ab1e6b163a6f` | **48.405** |
| `k026` persistent stride loop | `Project/kernels/k026_tail_persistent.py` | `profile-2279b2020c6dfee37f74205e` | **48.428** |
| GELU-deleted probe (deliberately incorrect) | `Project/kernels/probe_gelu_cost.py` | `profile-a54e004df00d0722d9acd155` | **41.378** |
| `k027` Abramowitz-Stegun erf | `Project/kernels/k027_fast_erf.py` | `profile-37b0c3b6d2cdc69b05fb56be` | **43.291** |

Shipped-file reference on the same shape: **46.469 us/call**
(`profile-832fd31066b1c368b5e50f5a`).

- Split-at-LayerNorm was killed on arithmetic before it was built: it would push `h1`
  (4.19 MB) and `h2` (2.10 MB) out to global and back, **12.6 MB of extra traffic per
  layer**. Source: `Project/kernels/k025_tail_halfffn.py:11`.
- `k025` and `k026` are flat against each other (48.405 vs 48.428) and both *above* the
  shipped 46.469. **Caveat that must travel with those two numbers:** they are kernel
  modules and 46.469 is the shipped submission file, so this is a cross-artifact
  comparison. The safe statement is that neither candidate beat the other and neither beat
  the shipped file; "no gain" is right, "no change" would be too strong.
- The probe sizes the GELU: **46.469 - 41.378 = 5.091 us/call**. The probe is deliberately
  incorrect — it deletes the GELU — so it is a cost measurement, never a candidate.
- `k027` recovered **46.469 - 43.291 = 3.178 us** of that 5.091, and measured +2.7%
  end-to-end on shape 1.

**What one failed candidate does not prove.** `k025` reduced live state and did not get
faster. That does **not** establish that register pressure is irrelevant in this kernel.
It establishes that one particular way of reducing live state, at one shape, did not pay.
Settling the register-pressure question requires Nsight Compute occupancy and spill
counters; `ncu` needs root and is deliberately absent from this project's command
allowlist, so **the question is open and will stay open on this hardware setup.**

Remaining: about **1.9 us/call** of the measured 5.1 us GELU cost is still on the table
after `k027`.

### Device-to-device copies (CUDA graph input copy)

Two `Memcpy DtoD` per forward on older builds — the CUDA graph input copy and the graph
output clone. **The output clone was removed** in build `2778b747`; builds carrying the
removal show 20 `aten::copy_` calls instead of 40, and no `aten::clone` row.

| shape | profile | build | DtoD share | calls |
| --- | --- | --- | --- | --- |
| 13 | `profile-1f701c73c75025b4036ba1c5` | `2778b747` | 3.32% | 20 |
| 12 | `profile-9eb91d1128ad453b0ecc5629` | `2778b747` | 3.85% | 20 |
| 9 | `profile-9460a66e29b90e2dc5082d36` | `2778b747` | 3.90% | 20 |
| 4 | `profile-408b588a1a75354429929041` | `2778b747` | 3.75% | 20 |
| 10 | `profile-520c2e30494601939a838b79` | `2778b747` | 4.07% | 20 |
| 5 | `profile-17943ee6244dcb9f3cd246c9` | `2778b747` | 4.55% | 20 |
| 8 | `profile-79950fcb9f605041365c7dc9` | `9d7e67ab` | 0.89% | 20 |
| 1 | `profile-832fd31066b1c368b5e50f5a` | `418952bf` | 7.33% | 40 |
| 11 | `profile-e3013a6912f61e91b4449a78` | `418952bf` | 6.24% | 40 |
| 7 | `profile-d3ab12dfcfd36b88ffe41759` | `599f5dad` | 7.10% | 40 |
| 3 | `profile-d91f02ef2de78d8df295f53d` | `301d7063` | 4.04% | 40 |
| 2 | `profile-da728b009a598d1f3ed08163` | `599f5dad` | 3.25% | 40 |

Shapes 1, 2, 3, 7 and 11 **have not been re-profiled since the clone was removed.** Their
DtoD shares above are pre-removal and are the last measurement that exists for them.

**On the "+0.9% to +7.0% across seven shapes" figure.** The source table is
`Project/loop/geomean_camp_final.py:48-56`. It holds seven entries, and the code comment
directly above it says "measured on six shapes". Computed from that table: shape 12
**+7.04%**, shape 5 +4.85%, shape 9 +3.43%, shape 10 +3.42%, shape 13 +2.51%, shape 8
**+0.87%** — that is the +0.9% to +7.0% band, and it covers **six** shapes. The seventh
entry, shape 4, computes to **-3.75%** using the two-replicate geomean recorded there.
So the honest statement is: *six shapes gained between +0.9% and +7.0%; the seventh
measured a loss.* Do not quote the band as covering seven.

The input copy is untouched. Whether it can go the same way as the output clone is a
question about whether the official benchmark script's input tensor is stable enough across
calls to be written into the graph's static input buffer. The output clone only became
removable after someone read the script and established that neither the timing loop nor
the accuracy loop retains the returned tensor across a call
(`Project/loop/gate_log.jsonl:506`). The same reading has not been done for the input side.

### `valid_token_mask.all()` host synchronization

**Device-side cost is small and measured.** It is the `reduce_kernel` plus the
`Memcpy DtoH (Device -> Pinned)`:

| shape | reduce | DtoH | total device |
| --- | --- | --- | --- |
| 2 | 3.52% | 1.48% | **5.00%** |
| 3 | 3.26% | 1.26% | 4.52% |
| 7 | 3.42% | 0.96% | 4.38% |
| 12 | 2.62% | 0.81% | 3.43% |
| 4 | 2.37% | 0.74% | 3.11% |
| 10 | 0.86% | 0.22% | 1.08% |
| 1 | 0.85% | 0.21% | 1.06% |
| 9 | 0.83% | 0.22% | 1.05% |
| 11 | 0.71% | 0.17% | 0.88% |
| 5 | 0.63% | 0.13% | 0.76% |
| 13 | 0.27% | 0.03% | **0.30%** |

**The host-side figure is not an opportunity and must not be written down as one.**
`cudaStreamSynchronize` is 75.35% of self CPU time in the shape-9 profile
(`profile-9460a66e29b90e2dc5082d36`: 9.617 ms of 12.764 ms) and 73.92% on shape 10. **That
is host wall time that includes waiting for queued GPU work.** It is not established as
recoverable, and no experiment in this repo separates the two.

Shape 8 shows why the distinction is not pedantic. In `profile-79950fcb9f605041365c7dc9`,
`cudaStreamSynchronize` is **94.05% of self CPU** (356.531 ms) — while `Self CUDA time
total` for the same window is **377.062 ms** against a `Self CPU time total` of 379.088 ms.
The host is almost entirely blocked, and the device is almost entirely busy. Essentially
all of that sync is queue-drain on real work. Reading 94% as headroom would be wrong by
almost the whole amount.

**Status: unmeasured, not an opportunity worth N microseconds.** Nothing should be proposed
here until an experiment can attribute the stall.

### fp16 route (shape 8)

`d_model` 1024 takes the other dispatcher branch and shares nothing with the fused route.
**No dedicated search has ever been run against it.** Most recent profile:
`profile-79950fcb9f605041365c7dc9`, build `9d7e67ab`, 06:59:16 on 31 Aug — the newest
profile in the repo, and the only one taken on that build.

| kernel | calls | share | us/call |
| --- | --- | --- | --- |
| `cutlass::Kernel2<cutlass_80_tensorop_f16_s16816...>` | 80 | **31.99%** | 1508 |
| `ampere_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_sta...` | 240 | **30.73%** | 482.9 |
| `at::native::vectorized_elementwise_kernel<4, ...>` | 160 | 10.52% | 247.8 |
| `at::native::elementwise_kernel<128, 4, ...>` | 320 | 7.93% | 93.5 |
| `at::native::unrolled_elementwise_kernel<...>` | 160 | 5.53% | 130.4 |
| `_sub_ln_fp16` | 160 | 5.26% | 124.1 |
| `_sub_attn_fwd` | 80 | 4.13% | 194.6 |
| `_sub_gelu_fp16` | 80 | 1.76% | 83.2 |
| `at::native::vectorized_l...` (layer norm) | 20 | 1.21% | 228.4 |
| `Memcpy DtoD` | 20 | 0.89% | 168.4 |

Two observations that follow directly and have never been acted on:

1. **62.72% of device time is in two vendor GEMM kernels.** Whatever is done here is a
   library-selection or tiling question, not an authored-kernel question.
2. **23.98% is in three generic `at::native` elementwise kernels** — 640 launches per
   profile. On the fused route this class of work was fused away; on this route it was
   never touched.

Shape 8's headline is 2.02x, and `LESSONS.md:41` records the correction that matters here:
the baseline sits at 0.34 MFU and the candidate at 0.68, so 2.02x is what doubling achieved
utilisation looks like — it is **not** a nearly-full roofline. The route is not obviously
exhausted; it is unexamined.

---

## Appendix A — the profile used for each shape

Most recent profile of `Project/submission/torch_transformer_benchmark_submission.py` for
each shape, by position in `Project/authority/events.jsonl` (append-ordered). The events
line is the `run_started` record.

| shape | B / heads / head_dim / seq | profile id | build (`target_sha256` prefix) | recorded | events line |
| --- | --- | --- | --- | --- | --- |
| 1 | 64 / 4 / 32 / 128 | `profile-832fd31066b1c368b5e50f5a` | `418952bf` | 04:53:58 | 864 |
| 2 | 1 / 4 / 32 / 128 | `profile-da728b009a598d1f3ed08163` | `599f5dad` | 03:44:32 | 715 |
| 3 | 4 / 4 / 32 / 128 | `profile-d91f02ef2de78d8df295f53d` | `301d7063` | 03:37:30 | 685 |
| 4 | 16 / 4 / 32 / 128 | `profile-408b588a1a75354429929041` | `2778b747` | 05:54:52 | 914 |
| 5 | 128 / 4 / 32 / 128 | `profile-17943ee6244dcb9f3cd246c9` | `2778b747` | 05:47:48 | 884 |
| 7 | 64 / 4 / 8 / 128 | `profile-d3ab12dfcfd36b88ffe41759` | `599f5dad` | 03:42:28 | 705 |
| 8 | 64 / 4 / 256 / 128 (fp16) | `profile-79950fcb9f605041365c7dc9` | `9d7e67ab` | 06:59:16 | 978 |
| 9 | 64 / 1 / 128 / 128 | `profile-9460a66e29b90e2dc5082d36` | `2778b747` | 05:50:07 | 894 |
| 10 | 64 / 2 / 64 / 128 | `profile-520c2e30494601939a838b79` | `2778b747` | 05:44:24 | 874 |
| 11 | 64 / 16 / 8 / 128 | `profile-e3013a6912f61e91b4449a78` | `418952bf` | 04:43:03 | 819 |
| 12 | 64 / 4 / 32 / 32 | `profile-9eb91d1128ad453b0ecc5629` | `2778b747` | 05:52:32 | 904 |
| 13 | 64 / 4 / 32 / 1024 | `profile-1f701c73c75025b4036ba1c5` | `2778b747` | 05:59:05 | 929 |

Earlier profiles exist for every one of these shapes — 33 profiles carry the three-kernel
`_sub_attn_heads` split and many more predate it. They are earlier builds and are not
averaged in.

**Four builds are represented, and that is a real limitation of this table.** Only shapes
4, 5, 9, 10, 12 and 13 sit on `2778b747`. Shapes 1 and 11 are on `418952bf`, shapes 2 and 7
on `599f5dad`, shape 3 on `301d7063` — all of which still carry the graph output clone.
Shape 8 alone is on `9d7e67ab`. A single-build board would need five more profiling runs.

Separately: `Project/memory/STATE.md` quotes the shipped artifact as sha `54057a33`. That
sha last appears in `events.jsonl` at line 483 (01:54 on 31 Aug) and at least five later
builds followed it. STATE is behind the campaign on this point.

## Appendix B — profiles of the block-tail beam candidates

All four are shape 1, all target `Project/kernels/*.py` modules rather than the submission
file, and all use unprefixed kernel names.

| profile id | target | events line | recorded |
| --- | --- | --- | --- |
| `profile-3238c17e01a2ab1e6b163a6f` | `k025_tail_halfffn.py` | 949 | 06:44:20 |
| `profile-2279b2020c6dfee37f74205e` | `k026_tail_persistent.py` | 958 | 06:49:28 |
| `profile-a54e004df00d0722d9acd155` | `probe_gelu_cost.py` | 963 | 06:52:07 |
| `profile-37b0c3b6d2cdc69b05fb56be` | `k027_fast_erf.py` | 968 | 06:54:46 |

## Appendix C — shapes with no profile

- **Shape 6** (B=10000): no profile record exists in `Project/loop/profile_evidence`.
  Not locally runnable.
- **Shape 14** (seq 100000, 2 layers): no profile record exists. Not locally runnable, and
  its evaluator is blocked on the owner-only device-comparison bug recorded as STATE item 1.

For both, every cell of this table would read **not measured**. They are excluded rather
than estimated.
