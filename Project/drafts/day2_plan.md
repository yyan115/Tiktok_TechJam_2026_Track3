# Day 2 plan — for the user's noon review (drafted ~03:00, 29 Aug)

## Where Day 1 ended (headline)

- **Ship board (quiet-box, referee-verified): geomean ~11.0x across all 12
  locally-runnable shapes.** Every shape passes precision; every champion is
  authored, committed-before-measured, tripwire-clean.
- **The submission file already exists and passes the judges' own script**
  (only the sanctioned class region replaced, byte-proof included; robust to
  padding, fp16-dtype, and --compile-user flags).
- Shapes 6 + 14: correctness proven locally on the 8 GB card; full-scale
  timing awaits rental.
- Evidence extras: torch.compile comparison table, two documented negative
  results (int8, split-QKV), 60+ audit verdicts in the ledger.

## Decisions only you can make (in priority order)

1. **Amendment v1.1 approval** — Project/amendments/ has the rationale AND
   complete line-anchored code (MFU reporting, official-acceptance
   subcommand, shape-14 oracle path). Without it, shape 14 cannot be scored
   at all and MFU (the organizers' scoring metric!) is not recorded.
   Estimated application time: ~30-60 min including self-test + Sol review.
2. **Rental booking** — Project/drafts/rental_day_runbook.md is turnkey.
   Recommendation: 48 GB (A6000/L40S class), ~2-3 hours, AFTER the amendment
   lands. Total spend estimate: low tens of dollars.
3. **Open MFU question** (from the amendment doc): fp32 peak as the single
   MFU denominator for all candidates — recommended, conservative — or
   per-dtype peaks?

## Proposed Day-2 sequence

1. Morning (you + agent): amendment review → apply → self-test → Sol diff
   review → manifest pin update (the formal re-freeze, timeboxed).
2. Rental session (~2-3h): runbook step by step — calibrate, shape 6, shape
   14 oracle, full-board secondary profile, MFU everywhere, copy journal off.
3. Afternoon: final board regenerated with MFU columns; README/video drafts
   updated with final numbers; your review pass on both.
4. Evening: T2 packaging begins per the standing plan (T2 first, then T3).

## What the agent can keep doing alone (if you're delayed)

- Keep the audit ledger healthy (Sol now runs at effort "high" per your
  instruction — faster verdicts).
- Screening-level tuning experiments on shapes 4/12/8 (lowest scores),
  promoted only through the referee as always.
- Video-script rehearsal assets: exact terminal commands per scene.

## Honest risks going into Day 2

- Shape 4 and 12 sit lower than their neighbors (7.9x / 10.4x) — real
  numbers, already honest, but there may be another ~1.3x on the table if a
  tuning idea lands; bounded upside, don't over-invest.
- Rental-card architecture differs (LESSONS #2): budget one retune round per
  shape, take the number, leave.
- The deadline math: submission + registration close 1 Sep 12:00 noon GMT+8;
  final ~8h protected for packaging per the standing plan.
