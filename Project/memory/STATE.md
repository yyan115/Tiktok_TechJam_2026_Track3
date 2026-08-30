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
