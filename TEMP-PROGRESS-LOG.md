# READ THIS ONE FILE (updated 29 Aug night)

## PART 1 — EVERYTHING SINCE YOU STOPPED ME (the "you're not doing CUDA" moment)

**1. I owned the failure.** You caught me in guess → run → guess again.
True: I had built two kernels (the int8 one, and a rework) that 15 minutes
of reading would have killed before writing a line. The harness checked
honesty but nothing ever checked whether I was being smart.

**2. We built a research library and I actually used it.**
`Project/research/` — 10 notes from real papers, including TikTok's own
CUDA Agent paper and the multi-agent repo from the organizer's slides.
Fun finding: TikTok's anti-cheating setup is basically the same harness we
built on our own. That's going in our report.

**3. Sol blind-reviewed the whole strategy (4 rounds) and changed it:**
- It KILLED my big CUDA idea for shape 2 — with math. A one-block CUDA
  kernel physically cannot beat what we already have there (chip's
  per-block ceiling = 137µs, we're already at 144µs). Building it would
  have wasted a day.
- It found the judges' scoresheet hiding in our own README: only 35% of
  the score is raw speed/technical. 65% is story, insight, report quality.
- It reordered priorities: fix shape 14's missing proof FIRST, then 6,
  then 8 — polish tiny shapes LAST.

**4. Your webinar transcript changed the plan again:** judges never re-run
our code (they read the report), one GPU type only, own machine preferred.
So the RENTAL IS CANCELLED — money saved — and shape 14 is solved the way
the organizer literally described: split into chunks on your card.

**5. Under the new think-first rules we ran 4 experiments:**
- Shape 14: PROVEN at full 100,000 length on your 8GB card — 0 errors,
  the biggest scoring hole, closed.
- Shape 6: PROVEN at full batch 10,000 — this is why rental died.
- Shape 8 fp16 trick: its own pre-test killed it in 25 minutes, no kernel
  built (the new rules doing exactly their job).
- Shape 11: tried honestly, came out a tie, closed instead of ground on.

**6. The cage (your spec):** built, then Sol attacked it for 13 rounds
(threw out v1 entirely, found ~50 real holes total) until round 13 said
APPROVE. Every hole fixed or — once — skipped with written "genuinely not
an issue" reasoning that Sol itself later agreed with.

## PART 2 — ARE WE DOING CUDA NOW?

**We CAN (you unlocked it — gcc15 installed, test kernel compiled and ran)
and we WILL wherever it wins. But the reviewed math says the remaining
CUDA-only ideas don't win:**
- The shape-2 CUDA kernel: impossible (see above, killed by arithmetic).
- The CUDA persistent-kernel family for shapes 3/4/12: ranked LAST by the
  review ("only if everything else is green"), and the cheap version of
  that idea works in Triton anyway.
- The rules don't care about language — organizer said "Triton, CUDA, or
  even lower level, whatever you like." Judges score the speed numbers
  and the report. Our 11x average is language-blind.

So: CUDA sits loaded in the toolbox; the cage decides. If a new card's
pre-test shows a CUDA route beating a Triton route, CUDA gets written.
Nothing currently queued needs it — what's left is shape retunes, then
the report/video (65% of the score, remember).

## PART 3 — WHAT YOU DO (the only thing, ~2 min)

1. Open `Project/loop/OWNER_PATCH_card_gate.md`
2. Paste its TWO code blocks into `.claude/hooks/guard_bash.py`:
   - Block A → right after the `WRITE_PATTERNS = [pat.replace(...)]` line
   - Block B → at the VERY END of `main()` (after the last existing check)
3. Save. Tell me: **"applied"**

Then I run 4 proof-tests (you'll watch runs bounce off the lock) and the
grind restarts itself — caged, cited, self-stopping. Code freeze 31 Aug
noon → packaging → submit before 1 Sep noon.
