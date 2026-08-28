# Runner freeze — your approval steps (IN THIS ORDER)

## What happened (short version)
The referee was built, attacked, hardened, and reviewed four times by an independent AI
auditor: Sol rejected v0.9.0 (real flaws), passed v0.9.2 ("sound to freeze for shapes 1–13",
`stage1_review3_verdict.json`), then a separate deep handoff review by codex found more real
issues (input-tampering hole, provenance gaps) which are fixed in v0.9.3. Both documented
cheating attacks are implemented as red-team tests and get caught. A final confirmation
review is bound to the committed code.

## Your steps — order matters (settings only load on a fresh start)
1. Open `.claude/settings.json` and add these two lines inside the `"deny": [...]` list
   (comma after the previous entry):

   ```
   "Edit(Project/harness/**)",
   "Write(Project/harness/**)"
   ```

2. NOW restart the Claude session (close and reopen, or `claude --continue`). This arms
   ALL the locks — the ones from last night plus your two new lines.
3. Verify: ask Claude to try editing `torch_transformer_benchmark.py` — it must be blocked.
   (If it isn't, stop and say so.)
4. Skim Sol's verdict if you want: `Project/audits/stage1_review3_verdict.json`.
5. Say **"freeze approved"** — Claude makes one final authorized edit (version string →
   1.0.0), re-runs the shape-1 validation set under 1.0.0, and the referee is hands-off
   from then on.

## The two decisions folded into "freeze approved"
- Scope: freeze covers shapes 1–13. Shape 14 is refused by the runner until its special
  handling (chunked reference oracle) is added later as a user-approved, re-audited change.
- Accepted residual: candidate code runs in the same process as the referee; a truly
  malicious candidate could theoretically attack timing channels the tamper probes don't
  watch. Our trust model is "guards against mistakes, not malice" — Sol judged this
  acceptable. Saying "freeze approved" accepts it too.
