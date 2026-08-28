# Competition scoring — what we actually know (updated 29 Aug 2026)

## Confirmed (organizers' webinar, 28 Aug — our notes)
- Score = weighted sum of per-shape MFUs, "bandwidth considered".
- A shape failing precision scores ZERO (pass-everywhere dominates).
- Consumer cards named as the expected hardware ("your own machine" spirit).
- Innovation policy per our strategy review: no wrapping external kernel
  projects; torch built-ins are eligible low-innovation fallbacks;
  project-authored kernels are the scoring play. No language restriction
  recorded anywhere.

## NOT publicly confirmed (checked 29 Aug ~04:00)
- The Devpost page (https://tiktoktechjam2026.devpost.com/) carries no
  Track-3 rubric; track specifics live in the info doc
  (https://bit.ly/TikTokTechJam2026Info) and organizer channels.
- OPEN QUESTIONS for the user/organizers: (1) per-shape weights (equal?
  FLOP-proportional?), (2) is MFU roofline-relative ("bandwidth considered"
  reading) or absolute-peak-relative, (3) how shapes 6/14 are scored when
  the official baseline cannot run them.

## Strategy sensitivity (why the answers matter)
- If roofline-relative: small launch-bound shapes can score well; our
  megakernel board is directly score-aligned; per-shape roofline table
  doubles as the score model.
- If absolute-MFU: big-d shapes (6, 8, 14) dominate scoring potential;
  rental day and shape-8 GEMM work rise to the top of priorities.
- Either way ~half the judging is narrative/polish (recorded earlier from
  rules): the trust-harness story + reproducibility remains a first-class
  deliverable.
