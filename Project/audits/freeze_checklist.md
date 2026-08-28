# Runner freeze — your approval steps (IN THIS ORDER)

## What you are approving, identified exactly
- The referee: `Project/harness/runner.py`, **v1.0.1**, sha256 `4fdfb97e1bf2859532064d0f0e64335c6cfa449fa7d239dfc2da3d9785725d95`.
  This same hash is PINNED in `Project/manifest.json` — from now on the runner refuses to
  run at all if its bytes differ from the pin, so a modified referee can never bless its
  own results. Changing the pin requires editing manifest.json, which your locks forbid
  to Claude's tools.
- The frozen commit hash is recorded at the bottom of this file and in STATE.md.
- Scope: shapes 1–13 (shape 14 refused until its chunked-oracle amendment, re-audited).
- Accepted residual: candidate code shares the referee's process; a truly malicious
  candidate could attack channels the probes don't watch. Trust model: mistakes, not
  malice — both reviewers judged this acceptable here.
- Enforcement, stated precisely: your deny rules lock Claude's file-editing tools; they
  do NOT cover writes by arbitrary subprocesses (per Claude Code's own docs). What covers
  those: the manifest pin (a changed runner refuses to run), committed hashes + git
  history (any tamper is visible and provable), and the Bash guard hook (an accident
  seatbelt only).

## Your steps
1. Open `.claude/settings.json`, add inside `"deny": [...]` (comma after previous entry):

   ```
   "Edit(Project/harness/**)",
   "Write(Project/harness/**)"
   ```

2. Restart the Claude session (`claude --continue` works). This arms ALL locks.
3. Verify: ask Claude to try editing `torch_transformer_benchmark.py` AND
   `Project/harness/runner.py` — both must be blocked. If not, stop and say so.
4. Say **"freeze approved"**. After that, zero edits to the harness or any protected
   file, ever. The single post-approval write is the approval note appended to
   `Project/memory/DECISIONS.md` — the memory diary, which is deliberately OUTSIDE the
   protected set (it's where approvals are supposed to be recorded).

## Review trail
Sol rounds 1–3: rejected v0.9.0 → PASS on v0.9.2. Codex handoff review: 14 findings →
v0.9.3. Codex confirmation: 3 blockers + 1 overrule overturned → v1.0.0. Codex round 3:
4 defects (freeze-wording contradiction, guard holes, calibration-key gaps, stale
injected state) → fixed in v1.0.1 with the manifest pin added. Verdicts + raw-log
hashes: `Project/audits/`.

FROZEN COMMIT: ddd89db (branch initial-architecture)
