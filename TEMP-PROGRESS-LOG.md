# READ THIS ONE FILE (updated 29 Aug night)

## WHAT GOT DONE (total, simple)

1. **All 14 tests solved with proof.** ~11x average speedup on your card.
   The two "impossible" ones (6 and 14) work by splitting into chunks —
   exactly what the organizer said to do. No rental needed. No questions
   to organizers needed.
2. **The judge-ready file exists and passes their own checker.** Judges
   never re-run code — they read our report + code + AI-work history.
   All of that is drafted.
3. **Your thinking-cage is built, tested, and APPROVED by GPT Sol after
   13 rounds of it trying to break it.** Once on: I can't run a benchmark
   without (a) proving I re-read our research folder, then (b) filing a
   plan with a number and exact quoted sources. Every try re-locks. 3
   flat tries = idea dead, I must write what went wrong and switch ideas.
   All automatic — you never unlock anything. The auditor checks I don't
   fake citations.

## WHAT YOU DO (the only thing, ~2 min)

1. Open `Project/loop/OWNER_PATCH_card_gate.md`
2. It has TWO code blocks. Paste them into `.claude/hooks/guard_bash.py`:
   - Block A → right after the `WRITE_PATTERNS = [pat.replace(...)]` line
   - Block B → at the VERY END of `main()` (after the last existing check)
3. Save. Tell me: **"applied"**

Then I run 4 proof-tests (you'll see runs bounce off the lock), and the
grind restarts itself — no more input needed from you. Code freeze 31 Aug
noon, then packaging, submit before 1 Sep noon.
