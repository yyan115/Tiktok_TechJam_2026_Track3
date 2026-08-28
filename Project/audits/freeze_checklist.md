# Runner freeze — your approval steps (IN THIS ORDER)

## What you are approving, identified exactly
- The referee: `Project/harness/runner.py`, **v1.0.2**, sha256 `203aba8d2a0955d6f8bb0044812ba6f78d3cc147983dc26230c5811b2345c462`.
  This hash is PINNED in `Project/manifest.json`; EVERY runner subcommand (measuring and
  reporting alike) verifies the pin before producing output, so under our cooperative
  trust model, drifted referee bytes are self-defeating rather than self-certifying. The
  absolute guarantee is external: git history + the manifest let anyone re-verify the
  exact bytes. Changing the pin requires editing manifest.json, which your locks forbid
  to Claude's tools.
- The frozen commit hash is recorded at the bottom of this file and in STATE.md.
- Scope: shapes 1–13 (shape 14 refused until its chunked-oracle amendment, re-audited).
- Accepted residual: candidate code shares the referee's process; a truly malicious
  candidate could attack channels the probes don't watch. Trust model: mistakes, not
  malice — both reviewers judged this acceptable here.
- Enforcement, stated precisely: your deny rules lock Claude's file-editing tools; they
  do NOT cover writes by arbitrary subprocesses (per Claude Code's own docs). What covers
  those: the manifest pin (a changed runner refuses to run), committed hashes + git
  history (any tamper is visible and provable), and the Bash guard hook — which is a best-effort
  accident seatbelt, explicitly NOT an invariant (it cannot fully parse shell).

## Your steps
1. Open `.claude/settings.json`, add inside `"deny": [...]` (comma after previous entry):

   ```
   "Edit(Project/harness/**)",
   "Write(Project/harness/**)"
   ```

2. Restart the Claude session (`claude --continue` works). This arms ALL locks.
3. Verify: ask Claude to try editing `torch_transformer_benchmark.py` AND
   `Project/harness/runner.py` — both must be blocked. If not, stop and say so.
4. Say **"freeze approved"**. From then on, FOR THE LIFETIME OF THIS FREEZE (i.e.
   until a formal re-freeze you approve), the write surface is exactly this:
   - Claude's tools: NO edits to the harness or any protected file. The one
     post-approval write is the approval note in `Project/memory/DECISIONS.md` (the
     memory diary — deliberately outside the protected set).
   - The pinned runner (its complete write surface, all by design): appends
     `Project/results/JOURNAL.jsonl`, regenerates `Project/results/LEADERBOARD.md`,
     appends any explicitly-passed `--ledger` scratch file, writes evidence packets
     under `Project/audits/packets/`, and appends `Project/audits/verdicts.jsonl`
     via `record-verdict`.
   - Future harness amendments (shape-14 oracle, official-acceptance subcommand): only
     via the formal re-freeze procedure — you approve a pin update, then full
     re-validation and re-audit before further results count.

## Review trail
Sol rounds 1–3: rejected v0.9.0 → PASS on v0.9.2. Codex handoff review: 14 findings →
v0.9.3. Codex confirmation: 3 blockers + 1 overrule overturned → v1.0.0. Codex round 3:
4 defects → v1.0.1 (manifest pin). Codex round 4: 3 blockers (pin didn't gate reporting
subcommands; /tmp exemption hole + abbreviated-option bypasses in the guard; write-surface
wording) → fixed in v1.0.2. Verdicts + raw-log hashes: `Project/audits/`.

FROZEN COMMIT: 22cfcd6 (branch initial-architecture)
