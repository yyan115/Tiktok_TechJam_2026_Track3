# THE WIN BAR — the standing goal, and the twelve conditions that settle it

**The goal (owner, 31 Aug 2026, binding): whatever we ship must be so good that Claude
says plainly WE WIN. Full stop.**

## The rule this file exists to enforce

"Can we win?" was repeatedly answered with *"I have no competitor data, I can't tell
you."* **That answer is banned.** It is not what is being asked, it is useless, and
repeating it reads as evasion.

The owner is not asking for a probability about strangers. They are setting a standard
for the work. So the question is reframed into one that evidence can actually settle:

> **Is there any gap left in our own work that a judge could point at?**

Every condition below is objective — a command, a number, or an artifact settles it, not
an opinion. Report status as **"N of 12 green"** and name the ones that are not.

**When all twelve are green, the answer to "can we win" is YES, said plainly.** At that
point it is a statement about the work having no hole in it, which is a thing that can be
known — not a guess about other teams, which is not.

---

## What the score actually is (so the bar aims at the right thing)

From the organizer's own webinar (`Project/research/competition-scoring.md`, sourced to
`MEETING-NOTES2.md` lines 104, 139, 188, 219):

- **Technical Execution = weighted sum of per-shape MFU.**
- **Every shape must pass precision (abs ≤ 2e-3 OR rel ≤ 2%) or that shape scores ZERO.**
- Weights are **undecided by the organizer himself**; bandwidth is considered; disclosed
  hardware limitations are considered.
- Judges **do not rerun anything.**

MFU = FLOPs ÷ time ÷ peak. For a fixed shape the FLOPs are fixed, so **MFU ∝ 1/time** —
every speedup is proportionally an MFU gain, and the two are the same objective.

---

## The twelve conditions

### A. No shape can score zero  *(highest value: a zero cannot be out-optimized)*

| # | condition | how it is settled |
|---|---|---|
| **1** | All 14 shapes pass the official predicate on the **final** artifact | 12 via controller `correct: true`; 6 and 14 via their evaluators |
| **2** | Shapes 6 and 14 carry **≥5 seeds each** on the final artifact | evaluator packets, not the current one-seed pre-integration packets |
| **3** | Shape 6 passes on **every one of ≥5 seeds**, and **we know what sets its error** — not merely that a number came out under the line | currently 0.00184 = **92% of budget used**, on one seed, cause unexamined |

> **Condition 3 first read "≥25% headroom under 2e-3", and that was a number I made up.**
> Part of shape 6's larger error is not degraded precision at all: it has
> **163,840,000 output elements against shape 1's 1,048,576 — 156× more samples of the same
> error distribution**, and the *maximum* of more samples is larger by construction
> (roughly √(ln N) scaling, ≈17% here). Demanding a fixed headroom could therefore be
> demanding something the sampling makes impossible, and would have sent me optimising
> precision that was never the problem. But 17% does not explain 1.0e-3 → 1.84e-3 either,
> so something else is also there. **The condition is now "pass on 5 seeds and know the
> cause", because a margin whose mechanism is unknown is not a margin you control.**

### B. The instrument can resolve what we claim

| # | condition | how it is settled |
|---|---|---|
| **4** | Byte-identical replicate spread on the noisiest shape is **< 2%** | run the same artifact twice on shape 12 under locked clocks |
| **5** | **No published delta is smaller than that measured spread** | check every claimed gain against condition 4's number |
| **6** | Every board row is measured on **one artifact sha** | all 12 rows carry the same `target_sha256` |

### C. The scored quantity is actually reported

| # | condition | how it is settled |
|---|---|---|
| **7** | **MFU per shape**, against both denominators (32.4 and 64.8 TF/s) | computed from measured candidate medians ÷ known FLOPs |
| **8** | Score reported under **≥3 weightings** — equal, FLOP-weighted, bandwidth-aware | the weights are unknown; a single number is a bet, three is a hedge |

### D. Nothing large is left unsearched

| # | condition | how it is settled |
|---|---|---|
| **9** | **Every component in the `HOTSPOT_COVERAGE.md` table** — not a percentage cut-off — is either searched with a measured outcome, or explicitly closed with the measurement that closed it | no "NO" left in the beam column without a measured reason beside it |
| **10** | **All eight levers in `WIN_PLAN.md` Phase 3** are each measured or killed on paper | a screening row, or a written arithmetic kill, for every one of 3a–3h |

> **Condition 9 originally read ">15% of device time", and condition 10 named three
> levers.** Both were wrong within an hour of being written. The 15% threshold would have
> excluded `_sub_final_norm` (3.0–10.9%) and the mask synchronisation (0.3–5.0%) — and
> those are *exactly* the two things a later pass found unsearched. **A threshold chosen to
> bound effort quietly re-scopes the claim it is attached to.** The table enumerates seven
> components; requiring all seven costs nothing extra and cannot be gamed by picking a
> number. Three levers became eight over two revisions (`WIN_PLAN.md` Phase 3), and this
> file did not follow until challenged — the same correction-drift failure recorded as
> LESSONS 48, committed against my own bar.

### E. Every claim traces to an artifact

| # | condition | how it is settled |
|---|---|---|
| **11** | Every number we state traces to a hash — packet, profile, or event | no figure sourced to a note or a memory |
| **12** | **No two of our documents disagree** on any number | they currently do: 10.3× vs 9.45× vs 10.14× |

---

## Current status: 0 of 12 green

Stated bluntly so the gap is not softened.

| # | status | why |
|---|---|---|
| 1 | ❌ | shapes 6 and 14 cannot be evaluated — one-line device bug in each evaluator |
| 2 | ❌ | both carry one seed, against a pre-integration artifact |
| 3 | ❌ | 92% of budget used, and unmeasurable until #1 |
| 4 | ❌ | last measured spread 5.1% (autotune fixed); clocks now locked, unverified |
| 5 | ❌ | depends on 4 |
| 6 | ❌ | **zero shapes measured on the current artifact `f462e320`** — and shape 12's family has **0 of 12 attempts left**, so its row cannot be measured at all without an owner budget raise |
| 7 | ❌ | the MFU table in the report is built on pre-split kernel-module times |
| 8 | ❌ | `SENSITIVITY.md` is pre-gate by construction and quotes a withdrawn figure |
| 9 | ❌ | four table rows still read "NO": `_sub_attn_heads`, the fp16 route, the mask sync, and the graph input copy (partial) |
| 10 | ❌ | none of the eight attempted |
| 11 | ❌ | several figures trace to notes rather than packets |
| 12 | ❌ | three documents, three different headline numbers |

**Nothing here is unreachable.** Ten of the twelve are work I can do; two need one
owner-side line each. The plan that turns them green is `Project/WIN_PLAN.md`.
