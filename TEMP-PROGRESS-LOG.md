# TEMP — read this when you're back (Track 3)

*Written 28 Aug ~09:30 by Claude before you closed the session. Delete this file once done.*

## Where things stand (10-second version)

Everything is built, tested, and committed on branch `initial-architecture` (pushed to GitHub).
The referee works and already has one verified result: **1.61x speedup on test size 1**.
It survived 4 rounds of independent AI review — flaws were found, fixed, and re-verified each time.
Nothing can start until you do the steps below.

## YOUR TO-DO, in order

**1. Restart Claude in this folder** (`claude --continue` in a terminal here, or open a fresh session — both work; the project's memory files bring any session up to speed automatically).

**2. Do the freeze (~3 min).** Open `Project/audits/freeze_checklist.md` and follow it top to bottom. Plain version: you paste 2 lines into `.claude/settings.json` (they lock the referee so nobody, including the AI, can quietly change the measuring stick), restart once more, check the lock actually blocks, then tell Claude **"freeze approved"**.

**3. Say "grind".** That starts the real work: optimizing all 13 runnable test sizes. Fully autonomous, every result logged and verifiable.

**4. Say "go track 2" too** (see the other repo's TEMP file) — both can run.

**5. Sometime today/tomorrow (5 min):** check your Devpost — you must be registered, and confirm you can create TWO submissions (one per track). The submission window is **29 Aug 12:00 noon → 1 Sep 12:00 noon**.

**6. This weekend (not urgent today):** make a RunPod account (~$20–50 budget). Needed only for test size 14 (too big for your GPU) and the final official numbers.

## One loose end that resolves itself

The second AI reviewer (codex) was mid-way through its final confirmation of the committed
code when it hit your ChatGPT plan's usage limit (resets 11:35 AM). An automatic retry is
armed — the verdict will be in the session log / scratchpad by the time you read this. Ask
Claude "what did the confirmation review say" when you're back. Its review so far had found
no new problems.

## What the plan is after your steps

Grind on 13 sizes (your GPU) → rent big GPU for size 14 + official final numbers →
tech report + README + 3-min video (day 3) → submit BEFORE the deadline with hours to spare.
Full plan: `Project/PLAN.md`. Current status always in: `Project/memory/STATE.md`.
