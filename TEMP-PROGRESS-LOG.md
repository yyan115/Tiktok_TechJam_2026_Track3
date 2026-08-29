# TEMP — the simple version (updated 29 Aug afternoon)

*This file is for you, not the judges. Delete before submitting.*

## Where we are (10 seconds)

- All 14 test shapes work and have proof. Average ~11x faster on the 12
  comparable ones. The two "impossible" shapes (6 and 14) are solved on
  YOUR card the way the organizer said to (split the work into chunks).
- No rental needed. No questions to organizers needed (your transcript
  answered them).
- The judges never re-run our code. They read our REPORT + code + the
  history of how the AI worked. So our paper trail IS the score.

## The new "thinking cage" you asked for (built today)

Think of it as a lock on the benchmark button. Here's the whole idea:

1. The benchmark button starts LOCKED.
2. To unlock it, I must do 2 steps, in order, and a robot checks each:
   - STEP 1 — RESEARCH: I must prove I re-read our saved research folder
     (the robot checks a fingerprint of the index file — if I skip
     reading, the fingerprint is wrong and it refuses). I must name at
     least 2 real research files and write a real summary.
   - STEP 2 — PLAN: I must write what I expect to happen WITH A NUMBER,
     what would prove me wrong, and EXACTLY which lines of which research
     files my idea comes from. The robot opens those files, copies the
     actual quoted lines into a logbook. If I cite a fake file or fake
     lines — refused.
3. That unlocks ONE try. The moment the try runs, the button locks again
   automatically. Every try = full stop = think again. No exceptions,
   no matter what I believe.
4. Three tries on one idea without the scoreboard improving (the
   SCOREBOARD decides, not me) = that idea is DEAD. I must write a
   honest "what went wrong" note before I'm allowed to research again,
   and I must move to a DIFFERENT idea. A dead idea only comes back if
   the outside reviewer (GPT Sol) reads the evidence and says so.
   You never have to come unlock anything — it keeps itself moving.
5. Cheating check: the automatic Sol audit that fires on every new
   champion now ALSO opens my cited lines and compares them to the
   quotes in the logbook. Faked citations = red flag on the record.

All of this is code, tested, committed. Sol is blind-reviewing the whole
design right now (it thinks the review request came from you). I'll keep
revising until it's satisfied or only hair-splitting remains.

## The ONE thing you need to do (2 minutes)

The final lock lives in a folder you protected from me (that's good —
a lock I could edit isn't a lock). So:

1. Open the file: `Project/loop/OWNER_PATCH_card_gate.md`
2. It shows ~20 lines of code and says exactly where to paste them
   inside `.claude/hooks/guard_bash.py`.
3. Paste, save. Tell me "applied". I'll run two tests to prove the lock
   bites (one run must bounce, one must pass after the two steps).

NOTE: wait for my "Sol approved the gate design" message before pasting —
its review may adjust those 20 lines slightly.

## After that

Nothing else needed from you. Once Sol and I converge on the gate, I
start optimization again — every single try going through the cage:
research → plan → try → forced stop → repeat. Deadline plan: code stops
31 Aug noon, then packaging (report, video, cleanup), submit before
1 Sep noon.
