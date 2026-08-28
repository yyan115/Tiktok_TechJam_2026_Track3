# Rental runbook — REWRITTEN 29 Aug (post strategy-review convergence)

Purpose: ONE rented session, shape-14 first and primary. Shape 6 is
piggyback-only and conditional. The frozen runner is NOT used for shape 14
(it refuses it by design); evidence comes from the independently pinned
side evaluator per harness_v2_proposal Card 1.

## Booking
- Reserve for 30 Aug; 48 GB class (A6000/L40S); sm86/sm89 preferred.
- Booking cutoff doubles as the organizer-answer cutoff (30 Aug 12:00 SGT):
  no answer => candidate-only evidence plan below is final.

## Preconditions (all free, at home, BEFORE the meter starts)
- [ ] Card 1 built and green locally: FA2-style candidate + pinned side
      evaluator, streamed oracle VALIDATED vs the pinned official dense
      path (full-model, multi-seed, feasible lengths).
- [ ] Card 2 proven locally: shape-6 repeated NO-GRAPH timing route
      (memory-safe under repetition), candidate-only MFU script.
- [ ] device_peaks.json entry for the booked card (spec-sheet, cited).
- [ ] Evidence-packet writer produces Project/results_side/ packets locally.
- [ ] Branch pushed; runbook (this file) re-read on the box.

## On the rented box (order is binding)
1. Environment pin: clone at pinned commit; record `git log -1`,
   `runner.py env`, `runner.py check` (hash gate must pass), nvidia-smi
   clocks snapshot. Match torch/triton majors; record exact wheels.
2. SHAPE 14 (primary, the reason the meter is running):
   a. Reduced-size full-model multi-seed checks (side evaluator).
   b. ONE full-scale B=32/S=100000 run -> immutable evidence packet
      (evaluator/candidate/submission/official shas, config, env, seeds,
      error stats, raw timing samples, peak alloc/reserved memory).
   c. One retune round maximum if performance disappoints; second full-run
      only if the retune changed the candidate.
3. SHAPE 6 (conditional piggyback, same instance, only after 14 is banked):
   - Default (organizer silence): candidate-only MFU via the proven
     no-graph route + correctness vs batch-chunked official baseline.
   - Only if organizers required a dense-baseline comparison: runner
     calibrate + run for shape 6 (the baseline fits on 48 GB).
4. OPTIONAL, only if time remains cheap: 12-shape secondary-profile board
   for the cross-device narrative (never mixed with the primary board).
5. Copy Project/results_side/ packets + any journal deltas OFF the box;
   verify checksums locally BEFORE terminating the instance.

## Budget guard
Shape 14 path ~1-2 h including one retune. Hard rules: no exploratory
tuning marathons; shape 6 never extends the session, upgrades the card, or
delays booking; the meter stops when 14 (+conditional 6) evidence is
checksummed locally.
