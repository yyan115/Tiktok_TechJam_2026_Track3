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

## PART 2 — ARE WE DOING CUDA NOW? (and why "CUDA = fastest" is a myth here)

**The intuition "CUDA should be the best" is understandable but wrong in
an important way. Here's the real picture:**

- Triton and CUDA both turn into THE SAME machine code on the GPU.
  Triton is a power tool that writes that machine code for you; CUDA C++
  is writing it by hand. Hand-writing gives you more CONTROL, not more
  speed by default. Think manual vs automatic transmission: the manual
  doesn't make the engine bigger — it only wins on the rare corner where
  you need clutch control.
- The engine is the GPU, and our kernels are already near its redline:
  shape 8 runs at ~64% of the chip's theoretical ceiling (and our
  hand-rolled GEMM measured 98% as fast as NVIDIA's own library); shape
  14 and 13 are at ~60-80% of their ceilings. The wall we're hitting now
  is PHYSICS, not language. Rewriting the same kernel in CUDA gets the
  same machine code and the same number.
- CUDA's genuinely exclusive trick is making many GPU blocks talk to each
  other mid-kernel (Triton can't). We checked what that trick is worth on
  OUR test shapes, with math and published measurements: shape 2 has at
  most ~5% left before its hard ceiling (137µs floor vs our 144µs — in
  ANY language), and the published best for that whole trick-class vs
  what we already do is ~1.2x on a couple of small shapes. Real, but the
  smallest prize on the board — that's why the review ranked it last,
  NOT because CUDA is bad.
- The judges score the speed numbers and the report. The organizer
  literally said "Triton, CUDA, or even lower level, whatever you like."

**So yes — CUDA is unlocked (you installed the compiler, I verified it
works) and it WILL be used the moment any experiment's pre-test shows a
CUDA route beating the Triton route.** The cage decides with numbers, not
taste. Right now the biggest remaining points are shape retunes and the
report/video (65% of the score), and those are language-neutral.

## PART 3 — WHAT YOU DO (the only thing, ~2 min)

1. Open `Project/loop/OWNER_PATCH_card_gate.md`
2. Paste its TWO code blocks into `.claude/hooks/guard_bash.py`:
   - Block A → right after the `WRITE_PATTERNS = [pat.replace(...)]` line
   - Block B → at the VERY END of `main()` (after the last existing check)
3. Save. Tell me: **"applied"**

Then I run 4 proof-tests (you'll watch runs bounce off the lock) and the
grind restarts itself — caged, cited, self-stopping. Code freeze 31 Aug
noon → packaging → submit before 1 Sep noon.
