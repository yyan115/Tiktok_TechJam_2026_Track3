# Packaging checklist + the two decisions only the owner can make (30 Aug)

Timeline this is written against: **code freeze 31 Aug 20:00 SGT** →
packaging/recording 20:00 → 1 Sep 02:00 (**six hours**) → final 10 h
reproduction and contingency → **Devpost registration AND submission both
close 1 Sep 12:00 GMT+8**.

Because packaging is now six hours instead of fourteen, everything below
that is *writing* must be finished before the freeze. What is left for the
window is: assembly, the video, and the final measurement board.

---

## DECISION 1 — the public repo's history (owner call, needed before any push)

**The fact.** `Project/audits/strategy_review_codex.md` — a 791 KB raw
strategy-review transcript — was committed at `49bfd47` on 28 Aug. It was
untracked two commits later (`db3a3f7`) and is correctly gitignored today.
**But it remains in the git history, and that history is already pushed to
all three remote branches** (`origin/main`, `origin/grind-day1`,
`origin/initial-architecture`). If the GitHub repo is public, that file is
publicly readable right now via the history.

Our own standing rule (LESSONS #14) is that raw AI-review logs stay
private: they embed session transcripts, absolute paths and identifiers.
Untracking a file does not retract it.

Good news: no other raw log was ever committed — `handoff_review1_raw.log`,
the three `stage1_review*_raw.log` files and the authority-v4 raw log are
all untracked and always were.

**Options.**

| | approach | keeps history | effort | risk |
|---|---|---|---|---|
| A | **Fresh public repo**: squash the whole project into one clean initial commit (or a short curated series) and push that as the submission repo | no | ~20 min | lowest — nothing private can leak from a history that doesn't exist; but the "AI process trail" story loses its git provenance |
| B | **Rewrite history** (`git filter-repo` / BFG) to purge that one blob, then force-push all branches | yes, minus the blob | ~30 min + care | force-push on a repo you may already have linked; any fork/cache keeps the old objects; GitHub needs a cache-purge request to fully drop them |
| C | **Leave it** and accept it as public | yes | 0 | it is a raw model transcript of our own session — no credentials, but it contains local paths and unfiltered reasoning we said we would not publish |

**Recommendation: A.** The judges never re-run the code and are not
grading git archaeology; a clean single-commit public repo is normal for a
hackathon submission, costs twenty minutes, and removes the whole class of
problem. The process trail the judges *do* see (verdict ledger, evidence
packets, research notes, decision logs) is all committed content and
survives a squash intact.

If you prefer B, do it before anything else on freeze day — every later
step assumes a stable remote.

## DECISION 2 — third-party content in the public path (owner call)

Two items in the repo root belong to the organizers, not to us:

- `Track3_Slide1.png` … `Track3_Slide4.png` — the organizers' own slides.
- `MEETING-NOTES2.md` — a verbatim transcript of the organizers' webinar,
  including the speaker's self-introduction and career details.
  `MEETING-NOTES.md` is the same class of material.

They are tracked and would ship publicly. The video rules explicitly forbid
third-party trademarks or copyrighted content without permission, and the
same caution applies to a public repo. Neither file is needed by a judge:
everything we *use* from them is already distilled into
`Project/research/competition-scoring.md` with the conclusions cited.

**Recommendation:** remove all six from the public path (keep local
copies). If you want to preserve the reasoning trail, keep the research
note — it carries the substance without republishing someone else's
recording or slide deck.

---

## Judge-path inventory (what a judge should see)

**Keep and feature**

- `Project/submission/` — the single-file submission + byte-identity prover
- `Project/kernels/` — all candidates, including the failures
- `Project/harness/runner.py`, `Project/shapes.json`, `Project/manifest.json`
- `Project/results/` (journal + leaderboard), `Project/results_side/`
- `Project/audits/` — verdict ledger, evidence packets, review prompts
- `Project/research/`, `Project/loop/`, `Project/memory/`
- `Project/tools/`
- the new README (applied at root) and the tech report

**Move, rename, or drop before pushing**

| item | why | action |
|---|---|---|
| `TEMP-PROGRESS-LOG.md` | internal note to the owner, written in the second person; contains one performance claim ("98% as fast as cuBLAS") we could not source | delete from the judge path |
| `MEETING-NOTES.md`, `MEETING-NOTES2.md` | third-party transcript (Decision 2) | remove from public path |
| `Track3_Slide*.png` | organizers' slides (Decision 2) | remove from public path |
| `Project/drafts/` | working drafts, including this file | promote the three that ship (README → root, tech report → `docs/`, video script → keep or drop) and drop the rest, or keep the folder clearly labeled as working notes |
| `Project/drafts/rental_day_runbook.md`, `organizer_questions.md`, `day2_plan.md` | superseded; each already carries an ARCHIVED/RETIRED banner | fine to keep as history, or drop — they cost nothing but explain nothing |
| `.streamlit/`, `Project/tools/dashboard.py` | local-only dashboard; must never be exposed with raw-log expanders | keep the code, confirm it is loopback-bound, never record it with raw logs open |
| `__pycache__/`, `.ruff_cache/` | noise | gitignored already; confirm none is tracked |

**Verified clean:** no credentials, no API keys, and no raw AI-review log is
tracked. The only tracked file containing an absolute local path is
`Project/tools/tests/guard_and_auditor_test.py`, which is a test fixture —
harmless, but worth a glance before pushing.

---

## Deliverables (from the track statement §3.5)

1. **Written project description on Devpost** — draft in
   `Project/drafts/devpost_description.md`. Must name: how the solution
   addresses the problem, development tools, APIs, libraries/frameworks,
   datasets/assets.
2. **Public code repository** with a README covering project overview,
   setup and installation, steps to reproduce, a reflection on limitations
   and what you would improve, and team contributions — draft in
   `Project/drafts/track3_readme_draft.md`, ready to apply at root.
3. **Demo video**, uploaded to YouTube, **set to public**, linked in the
   Devpost description — shot list in
   `Project/drafts/track3_video_script.md`.
4. **Tech report** — `Project/drafts/tech_report_draft.md`. The track text
   says a clear tech report naming the environment (CPU, GPU, disk), the
   optimizations, the AI skills/tools used and the final results earns
   bonus points. It is written; only `[PENDING]` numbers remain.

---

## Freeze-day order of operations

**Before the freeze (i.e. now → 31 Aug 20:00) — all writing, no GPU:**

- [x] Tech report rewritten with sourced numbers and honest labels
- [x] README draft restructured to the required sections
- [x] Video shot list updated
- [x] Score-sensitivity board rebuilt (multi-denominator, ship-pass selection)
- [ ] Devpost description drafted
- [ ] Skills / interaction-history samples curated (a scored deliverable class)
- [ ] Owner: Decision 1 and Decision 2 above
- [ ] Owner: **verify Devpost registration** — registration closes at the
      same moment as submission, and it is the one failure mode that makes
      everything else worthless

**At the freeze (GPU work, idle box):**

- [ ] Re-run the complete official-dials board against the final submission
      sha, N ≥ 3 per small shape, all N reported (narrowing 11d)
- [ ] Re-capture the shape-6 and shape-14 packets against the shipped
      submission file, ≥ 5 seeds (narrowing 10)
- [ ] Shape 14 full batch-decomposed timing (narrowings 2 and 4)
- [ ] Fix and regenerate `SHIP_MANIFEST.json` from the final commit
      (narrowing 11a–c)
- [ ] Audit-ledger parity check: every current champion carries a verdict
- [ ] Independent adversarial final review, one defect class per reviewer
      (narrowing 11c)

**Packaging window (31 Aug 20:00 → 1 Sep 02:00):**

- [ ] Fill every `[PENDING]` from the final board
- [ ] Apply the README at root; move the tech report into the judge path
- [ ] Execute Decisions 1 and 2, merge to `main`, push
- [ ] Record and upload the video (public), link it in Devpost
- [ ] Submit on Devpost

**Final 10 hours:** clean-checkout reproduction, submission rebuild, link
verification. No new work.
