# STATE — first ten seconds of a session

This file is deliberately short and holds no plan. Two files do that:

- **`Project/GRIND_ENTRYPOINT.md`** — the operating manual: the commands, the one
  next permitted action, the lanes, the stop conditions. Get it by running
  `python3 Project/tools/session_bootstrap.py`.
- **`Project/HANDOVER.md`** — state, open defects and the FIX → LOCK → GRIND plan.

Then read all of `Project/memory/LESSONS.md`, every session, and
`Project/research/INDEX.md` before relying on any research note.

**If a command and a document disagree, the command is right** — including this one.

Updated: 2026-08-31 ~00:15 SGT. Branch `grind-lastday`.

---

## 000. ⚠ LATEST: the ORIGINAL board is being vindicated. Read this before anything below.

31 Aug ~01:25 SGT. Re-measuring the **actually-shipped** megakernel (k009) now agrees
closely with the pre-gate figures I withdrew three hours ago:

| shape | pre-gate (withdrawn) | **shipped route, measured tonight** | delta | k004 (wrong kernel) |
| --- | --- | --- | --- | --- |
| 13 | 28.82x | **28.4098x** | −1.4% | 5.8096x |
| 7 | 25.57x | **21.9645x** | −14.1% | 3.4781x |
| 2 | 15.26x | **14.3939x** | −5.7% | 8.1115x |
| 11 | 12.98x | **12.6797x** | −2.3% | 4.2433x |
| 3 | 11.96x | **12.6314x** | +5.6% | 7.1845x |
| 12 | 11.44x | **10.8141x** | −5.5% | 3.2334x |
| 4 | 7.30x | **8.8774x** | **+21.6%** | 2.7175x |
| 1 | 10.73x | **8.3303x** | **−22.4%** | 2.1428x |

**Eight of twelve done. Mean delta −3.0%, scatter −22.4% to +21.6%, six below and two
above.** So the old board was neither systematically inflated nor systematically deflated —
it was **approximately right on average with per-shape scatter of roughly ±22%**. That is
the honest characterisation, and it is a more useful claim than either of my earlier
readings because it is both bounded and two-sided.

The k009-over-k004 ratio ranges **1.76x to 6.31x** — strongly shape-dependent, which is why
the "constant 1.77x" claim was wrong.

**Two of my own claims are refuted:**

1. *"The pre-gate board was systematically inflated."* **Wrong.** It is procedurally
   invalid — no permit, no bound verdict, baselines 6–63% off calibration — and
   numerically close. I conflated those.
2. *"The shipped route beats k004 by a consistent ~1.77x."* **Wrong** — generalised from
   two points. Shape 13's ratio is **4.89x**. Shape-dependent, like everything else here.

**The 2.94x figure is the misleading one** and my own WITHDRAWN blocks briefly made it the
headline, understating by ~3.5x — the same error I accused the drafts of, inverted. The
tech report block is corrected; README and video script still need the same second
amendment.

**No geomean is claimed.** 3 of 12 shipped-route shapes measured. The original 10.32x is
not re-earned, but it is no longer contradicted.

## 00. ⛔⛔ I MEASURED THE WRONG KERNEL. The 2.94x board is NOT the submission.

Found 31 Aug ~01:05 SGT while auditing the tech report. **This is my error and it
supersedes the framing of every entry below.**

The whole twelve-shape post-LOCK board measured
**`Project/kernels/k004_graphed_triton.py`** — fp32 authored Triton attention plus a
whole-forward CUDA graph. **That is not what this submission ships.**
`Project/submission/dispatcher_region.py` routes:

- `d_model <= 128`, causal, no padding mask → **fused-block megakernel** (k009-class:
  LayerNorm+QKV fused, flash attention over all heads with the output projection folded
  into the head loop, residual + norm + GELU FFN in-register), CUDA-graphed. **That is
  11 of the 12 primary shapes.**
- larger `d_model` → **fp16 tensor-core stack** with fp32 accumulation. **That is shape 8.**

**How it happened:** runbook step 11's worked example profiles
`k004_graphed_triton.py`. I followed the example, made k004 the campaign's candidate, and
never checked it against the dispatcher. 22 attempts characterise a route that does not
ship.

**What survives, precisely:**
- The **12 calibrated noise floors and promotion thresholds** — candidate-independent.
- **Every baseline profile** (launch counts, idle fractions, kernel breakdowns) —
  measured on `k000_baseline.py`, unaffected.
- **The whole regime model** — head width, quadratic sequence, launch-bound small batch —
  because those are *baseline* weaknesses, not properties of k004.
- The method scorecard and all LESSONS.

**What does not survive:** 2.94x as a claim about this submission. It is a valid
controlled measurement of the wrong kernel.

**Correction propagated** to all three judge-facing drafts, which briefly carried my
2.94x as if it replaced the withdrawn 10.32x. They now say plainly that **the shipped
route has no valid post-LOCK measurement**. The video script's speed lower-third is
marked "leave blank".

**Next action (highest value, 38 attempts remain):** re-screen the *shipped* route —
`k009_fused_tuned.py` for `d_model <= 128` shapes and the fp16 stack for shape 8 —
using the same quiet-box screening protocol. The calibrations and baseline profiles are
already in place, so this is fast.

## 0. ⛔ READ FIRST — three judge-facing drafts carried a 3.5x overstatement

Found 31 Aug ~00:30 SGT, **19 hours before freeze**. `Project/drafts/` was written
30 Aug ~11:25, *before* the LOCK and before any of tonight's re-measurement, and three
deliverables quoted the dead pre-gate board as fact:

| file | claimed | measured tonight |
| --- | --- | --- |
| `tech_report_draft.md` §2.1, §2.2 | 10.32× and 10.95× geomean | **2.94×** |
| `track3_readme_draft.md` | 10.32× geomean | **2.94×** |
| `track3_video_script.md` | 10.3× geomean, 28.8× best shape | **2.94×**, best **8.11×** |

`devpost_description.md` is clean — it carries no numeric claims.

**All three now carry an unmissable WITHDRAWN block with the corrected table**, placed
above the stale text rather than replacing it, so the correction is auditable. The
original prose is retained unedited.

This is LESSONS 24 recurring — *"an unsourced number in an informal note becomes a claim
in the report"* — this time in the report itself. **Owner: the drafts still need a full
read-through before freeze; I corrected the headline figures I could verify, but I have
not audited every sentence of 23 KB of report prose for other pre-gate claims.**

## 1. THE ONE THING BLOCKING EVERYTHING — owner only

**Audit results cannot be recorded. Nothing can promote until this is fixed.**

Three audit attempts ran on entry `run-be8e56a55edd1926a84bf5d1efc0b154`, cost roughly
**$7.50** and about 20 minutes of GPU-idle waiting, and **all three failed to record**:

| attempt | failure |
| --- | --- |
| 1 | `verdict does not match full schema` — every required property reported missing |
| 2 | `auditor stdout must be exactly one duplicate-free JSON object with no banners` |
| 3 | `AUDITOR_PROCESS_ABANDONED_WITHOUT_TERMINAL_EVENT` |

The retry cap is exhausted and the entry is escalated to `owner_attention` permanently.
**The auditor itself worked** — both attempts 1 and 2 produced complete, high-quality
verdict documents which are durable at
`Project/authority/blobs/0b3fa1ce…audit-response.json` and `…/92b7c588….audit-response.json`.
Both returned integrity **RETEST** and technical **WEAK_DIAGNOSIS**, and both were correct
about a real attribution error. The failure is in the *recording* path, not the auditing.
`Project/tools/audit_champion.py` and the audit authority are inside the LOCK and
Write-denied to the agent, so this is owner work.

Consequence: **0 of 60 attempts have produced a promotable result, and none can until this
is fixed.** All measurement below is screening-lane, which cannot promote by design.

---

## 2. THE COMPLETE QUIET-BOX BOARD — all 12 primary shapes measured

k004 (authored Triton flash attention + whole-forward CUDA graph capture) against the
unmodified eager baseline. Every run: verified-quiet box (`champion_watch --dry-run`
showing `active: []`, SM 210 MHz, ~1–3% util, ~14 W), campaign timing protocol
(warmup 20, repeats 100, rounds 3), **`correct: true` on every seed**, screening lane,
**0 strikes**.

| shape | B | heads | head_dim | seq | baseline idle | **k004** |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 4 | 32 | 128 | 86.0% | **8.1115x** |
| 3 | 4 | 4 | 32 | 128 | 82.6% | **7.1845x** |
| 13 | 64 | 4 | 32 | 1024 | — | **5.8096x** |
| 11 | 64 | 16 | 8 | 128 | — | **4.2433x** |
| 7 | 64 | 4 | 8 | 128 | 3.2% | **3.4781x** |
| 12 | 64 | 4 | 32 | 32 | 69.8% | **3.2334x** |
| 4 | 16 | 4 | 32 | 128 | 49.2% | **2.7175x** |
| 5 | 128 | 4 | 32 | 128 | 1.0% | **2.1475x** |
| 1 | 64 | 4 | 32 | 128 | 3.4% | **2.1428x** |
| 10 | 64 | 2 | 64 | 128 | — | **1.5833x** |
| 9 | 64 | 1 | 128 | 128 | — | **1.1723x** |
| 8 | 64 | 4 | 256 | 128 | 0.2% | **1.1060x** |

**Geomean across these twelve: 2.94x.** Minimum 1.1060x — the candidate is **never slower
than the baseline on any measured shape**.

**Caveat on the headline:** the campaign's official scenario is `geomean-shapes-1-13`,
which includes **shape 6** (B=10000). Shape 6 is a dedicated side lane and is NOT in the
table above, so 2.94x is the geomean of the twelve primary shapes and is **not** the
official scenario figure. Do not quote it as such.

---

## 3. What explains the board — four measured baseline weaknesses

None of these is a strength of our kernel. They are all defects of the eager route that
our route simply does not have.

1. **Narrow head dimension is the biggest single lever.** Holding heads at 4 and narrowing
   head_dim from 32 to 8 moves 2.1428x (shape 1) → **3.4781x** (shape 7). Adding twelve
   more heads on top only moves it to 4.2433x (shape 11). *Head width dominates head
   count* — an earlier entry had this backwards and is corrected in DECISIONS.
2. **Quadratic sequence traffic.** At S=1024 the baseline materializes ~1.07 GB of score
   tensor per layer and spends **71.2%** of its device time on `masked_fill`, scale and
   softmax over it. Shape 13 → 5.8096x.
3. **Launch-bound small batch.** Baseline idle fraction orders these cleanly:
   1.0% → 2.1475x, 3.4% → 2.1428x, 49.2% → 2.7175x, 69.8% → 3.2334x, 82.6% → 7.1845x,
   86.0% → 8.1115x. No saturation at the extreme.
4. **Nothing to exploit.** Shape 8 (d=1024, head_dim 256, 99.8% device-busy, ~98% linear)
   has none of the above and lands at 1.1060x.

**Not a factor: problem size.** Shape 5 doubles shape 1's batch and moves the result by
0.2%.

---

## 4. Method finding, and it is the honest headline about process

| kind of claim | record |
| --- | --- |
| numeric prediction bands | **0 for 14** |
| qualitative regime hypotheses with preregistered falsifiers | **6 for 6** |

Every numeric band missed, including one derived from a measured per-kernel breakdown of
the very shape being predicted. Per a commitment preregistered on card C11, numeric bands
were **retired** mid-campaign (LESSONS 35); the gate requires the field, so it is now
filled as an explicitly low-confidence placeholder and no conclusion is drawn from it.

The six qualitative hypotheses all held, including the two hardest cases: one aimed at the
**low** end (shape 8 predicted to be worst — it was) and one **two-sided** bracket (shape 4
required to land inside 2.1428–3.2334 — it landed at 2.7175). Total strike cost of fourteen
consecutive numeric misses: **zero**, because every run was `--prediction-kind
characterization` in the scratch lane.

**I can classify regimes reliably. I cannot forecast magnitudes at all.** That distinction
determines which claims this project's evidence can carry.

---

## 5. Ledger

22 of 60 attempts spent (1 optimization, 21 screening), **0 promoted**, **0 strikes**,
12 shapes calibrated with immutable thresholds, 24 profiles, 13 families registered,
14 research cycles, no permit armed, lock active and valid at 29 files, campaign not
stalled, tree clean.

Capabilities `/tmp/cap_grind.json`, `/tmp/cap_family.json`, `/tmp/cap_stall.json` expire
**31 Aug ~21:25 SGT**.

## 6. Next actions

1. **Owner:** fix the audit recording path. Everything else is ready.
2. Once fixed: re-run the strongest shapes in `--mode optimization` and audit them. The
   screening numbers give well-founded expectations for every shape.
3. Open research target: **head_dim 8 padding waste** in the Triton kernel — shape 11 and
   shape 7 show narrow head_dim is where the baseline is weakest, so a kernel that avoids
   the 2x pad to 16 could go beyond 4.24x.

## 7. Standing rules (unchanged)

Never touch frozen/protected files. Every benchmark goes through a permit and the trusted
controller. One GPU process at a time. Never benchmark while an audit runs. Never compare
absolute times across invocations (LESSONS 11). Trace every number to the artifact that
produced it (LESSONS 24). Guard etiquette: never put `clean`, `reset`, `restore` or
`checkout` after `git` in one command segment. Plain language. The owner's explicit "go"
is required before repo actions, and the owner's stop overrides everything, immediately.

## 8. Clock

CODE FREEZE 31 Aug 20:00 SGT → packaging to 1 Sep 02:00 → submission AND Devpost
registration close 1 Sep 12:00 GMT+8.
