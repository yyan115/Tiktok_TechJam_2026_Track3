# TEMP — read this when you're back (Track 3)

*Updated 28 Aug 16:06. Delete this file once done.*

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

## Review status (updated ~11:30)

CLOSED with a YES. Codex reviewed the referee SIX times, finding real flaws each round
until there were none load-bearing left ("Overall competition-weekend reliance: YES, after
the mandatory freeze-arm steps" — the freeze-arm steps are YOUR checklist steps). The
referee is v1.0.2, fingerprint-pinned in the manifest (a drifted referee refuses to run
any command). Full verdict: Project/audits/track3_handoff_verdict_round6.md. Track 2's
bench is in the same loop (round 4 pending as of this writing) — ask "review status".


## WHAT WE ACTUALLY BUILT (plain words, one line each, with the file to open)

**The machine:**
- **The referee** — one script that tests every speed-up attempt: checks the answers match TikTok's original, times it fairly, writes the result to the logbook. I run it; I can't change it. → `Project/harness/runner.py`
- **The traps inside the referee** — catch the known ways AIs fake speed results: cached answers, hidden work, tampered math, modified inputs. All were proven to fire by attacking them. → `Project/harness/redteam/` (the attack files)
- **The fingerprint pin** — the referee's exact fingerprint is recorded; if even one byte of it changes, it refuses to run anything. Tampering turns itself off. → `Project/manifest.json`
- **The locks** — settings that make my editing tools refuse to touch TikTok's files, the referee, or the results. YOU arm these (your 2-line paste + restart). → `.claude/settings.json`
- **The exact 14 test sizes** — written down because TikTok's script silently tests the WRONG sizes on default settings. → `Project/shapes.json`

**The memory (so no session ever starts blank):**
- **Status board** — where we are, what's next; auto-loaded into every new session. → `Project/memory/STATE.md`
- **Diary** — everything we discussed and decided, in plain language, dated. → `Project/memory/DECISIONS.md`
- **Mistakes list** — things proven not to work, never to be retried. → `Project/memory/LESSONS.md`
- **Logbook** — every test result, machine-written, one line each; I cannot edit it by hand. → `Project/results/JOURNAL.jsonl`
- **Scoreboard** — best verified result per test size, auto-generated from the logbook. → `Project/results/LEADERBOARD.md`

**The oversight:**
- **Auto-audit** — every newly crowned champion automatically triggers a background GPT review of that result (mechanical trigger, not Claude's choice); verdicts fill the scoreboard's audit column. → `Project/tools/champion_watch.py`
- **Second-AI review trail** — codex (GPT) reviewed the referee SIX times, rejecting it for real flaws until none remained. Its final sign-off, verbatim: → `Project/audits/track3_handoff_verdict_round6.md`
- **Operating manual** — every command, what writes what, how to recover. → `Project/RUNBOOK.md`

## HOW TO CHECK IT YOURSELF (10 min, no code reading)

1. Read the reviewer's final verdict (short, plain English): `Project/audits/track3_handoff_verdict_round6.md`
2. Skim the diary for the story: `Project/memory/DECISIONS.md`
3. Watch a cheater get caught LIVE — run these two commands in this folder:
   `python3 Project/harness/runner.py check`   (integrity: should print green/verified)
   `python3 Project/harness/runner.py run --shape 1 --impl Project/harness/redteam/rt01_monkeypatch.py --ledger /tmp/rt.jsonl`   (should print TAMPER DETECTED and abort)
4. After your restart: tell Claude "try to edit the runner" — watch the lock block it.
5. Anytime, forever: any number Claude claims → say "show me the journal entry" — every result traces to one logbook line.

## What the plan is after your steps

Grind on 13 sizes (your GPU) → rent big GPU for size 14 + official final numbers →
tech report + README + 3-min video (day 3) → submit BEFORE the deadline with hours to spare.
Full plan: `Project/PLAN.md`. Current status always in: `Project/memory/STATE.md`.
