# Runner freeze — your approval steps (IN THIS ORDER)

## What changed since the last version of this file
The independent reviewer (codex) caught that the old ceremony was impossible: it had
Claude editing the referee AFTER your locks armed — which the locks themselves forbid.
Corrected ceremony: **the referee is already finished (v1.0.0), revalidated, and
committed BEFORE you act.** Your steps only arm the locks and declare approval.
No edit ever happens after arming.

## Your steps
1. Open `.claude/settings.json` and add these two lines inside the `"deny": [...]` list
   (comma after the previous entry):

   ```
   "Edit(Project/harness/**)",
   "Write(Project/harness/**)"
   ```

2. Restart the Claude session (close and reopen, or `claude --continue`). This arms ALL
   locks — last night's plus your two new lines.
3. Verify the locks bite: ask Claude to try editing `torch_transformer_benchmark.py` AND
   `Project/harness/runner.py` — both must be blocked. (If not, stop and say so.)
4. Say **"freeze approved"**. Nothing gets edited — Claude just records your approval in
   DECISIONS.md with the frozen commit hash, and the referee is hands-off from then on.

## What you are approving
- The exact committed referee (v1.0.0) — the same bytes the reviewers examined; the
  commit hash is the artifact's identity.
- Scope: shapes 1–13. Shape 14 is refused by the runner until its special handling
  (chunked reference oracle) is added later as a user-approved, re-audited change.
- Accepted residual: candidate code runs in the same process as the referee; a truly
  malicious candidate could theoretically attack timing channels the tamper probes don't
  watch. Trust model is "guards against mistakes, not malice" — both reviewers judged
  this acceptable for this project.
- Enforcement reality (stated plainly, per reviewer): the Bash guard hook is a seatbelt
  against accidents, not the lock. The lock is the deny rules you arm in step 1, plus the
  committed hashes and git history that make any tampering visible.

## Review trail
Sol rounds 1–3 (`stage1_review*` files): rejected v0.9.0 → PASS "sound to freeze" on
v0.9.2. Codex handoff review: 14 findings → 9 fixed in v0.9.3. Codex confirmation
review: 3 blockers + 1 overrule overturned → all fixed in v1.0.0 (this commit).
