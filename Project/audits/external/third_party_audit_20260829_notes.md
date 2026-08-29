# Independent 3rd-party AI audit — 29 Aug 2026 (owner-commissioned)

Owner sent the full repo + handoff package to a separate AI for a blind
technical audit at snapshot 5b5929d / submission sha 4408f94f... The full
report lives with the owner; this note records the verdict + the deltas we
folded into the queue. (Valuable for the skills/interaction-history
deliverable: THREE independent reviewers — Sol dispositions, this external
audit, and our own NARROWINGS queue — converged on the same list.)

## Verdict (quoted)
"A strong, plausible winning contender with proven ordinary-shape
acceleration, credible full-size Shape 6 execution, credible full-sequence
Shape 14 slice execution, and one major remaining technical evidence task:
a genuine streamed B=32 Shape 14 run."

## Key confirmations
- Independent static anti-cheat inspection: NO manipulation patterns found.
- build_submission.py --check-only independently rerun: outside-region bytes
  identical to the official script.
- Official-script sweep recomputed from our saved transcript: geomean
  10.32x across the 12 ordinary shapes (official dials; our runner quiet-box
  geomean 11.0x uses different recipes — LABEL BOTH, never mix).
- torch.isclose mismatch confirmed REAL but HARMLESS on existing evidence:
  every recorded max-abs error < 0.002 official atol, so no false passes.
- AI-agent workflow concern WITHDRAWN: Track-3 workshop explicitly expects
  agent-written kernels + interaction history.
- "Stop expanding infrastructure" — matches our own restart plan (gate is
  done; grind = kernel/evidence work only).

## Its priority list → our queue mapping
P1 streamed shape-14 evaluator (never allocate full B=32 x/out) = item 2.
P2 exact official predicate in side tools = item 3. P3 batch-decomposition
equivalence check at reduced S = item 2 tail. P4 full 32-slice timing
protocol (median of sums + staging wall) = item 4. P5 shape-6 current-sha
multi-seed packet = item 10 (sharpened today). P6 profile before picking
next shape = item 9 (ncu selected metrics). P7 score-sensitivity scenarios
= item 8 (sharpened today). P9 one full current-artifact board at freeze =
item 11.

## Where the AUDIT itself is off (adversarial re-read, 30 Aug)
1. RUBRIC: it asserts Devpost rules = "four equally weighted criteria" and
   never reconciles README §3.6 — the official track statement's explicit
   5-way 35/20/20/15/10 table (verified in-file; Presentation 10% is
   "final event only"). The specific statement controls; the audit's
   downstream strategy framing leans on the weaker 4-equal reading.
2. VARIANCE BLINDNESS: it printed shape-3 official results of 11.961x
   (historical sweep) and 15.173x (post-integration retest) — a 27% swing
   on the SAME route — without flagging that launch-bound small shapes are
   noisy on the official script. Consequence queued (item 11d: N>=3
   sweeps for small shapes on the final board).
3. MEMORY INVERSION unremarked: shape-14 B=1 packet peaks at 4.894 GiB vs
   B=2 at 4.738 GiB — smaller workload, MORE memory. Needs an allocator/
   staging explanation before these numbers go in the report; the audit
   repeated both blindly.
Verified-ACCURATE side (so the above is calibration, not dismissal): all 7
sha256 hashes, all 12 sweep rows digit-for-digit, the 38-file count, the
line-312 predicate description, CHUNK=500, single-seed packets (1234), and
every arithmetic spot-check (12.207 GiB, 9.20% superlinearity, 14.01 TF/s,
53,582.332 = 1674.448x32). Painful bonus: the official script's OWN comment
at the predicate says isclose "is slightly more permissive and is not used"
— our side tools ignored a warning written in the official file.

## Genuinely new points folded into the queue today
1. Extreme-shape packets need MULTIPLE SEEDS (shape-6 integrated smoke max
   err 0.00184 vs 0.002 atol — margin too thin for one seed) + the official
   predicate + the submission sha recorded inside the packet. → item 10.
2. Sensitivity board should score under several weightings (equal-by-shape,
   FLOP-weighted, weighted-MFU variants, bandwidth-aware). → item 8.
3. Rubric duality: repo README rubric (35/20/20/15/10) vs current Devpost
   rules (4 equally weighted criteria). Report must frame against BOTH;
   official rules control where they conflict. → item 7 hedge list.
