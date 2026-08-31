# STATUS — what is measured, on what file. Read this before any other document.

Written 31 Aug ~20:15 SGT. Every line below was checked against generated state
this hour, not from notes. Where a claim came from prose it says so.

---

## 1. The one thing that matters

**The file in `Project/submission/` right now has never been measured.**

| | |
|---|---|
| submission file on disk | `630a456c6a3eeb6f8dc4832e53e6ce9bb3fa25813b0257ff11a674b9cee2f378` |
| measurements on it | **none** |
| correctness checks on it | **none** |

Verified by searching `Project/loop/gate_state.json` for that hash: no match, in
any lane. The build before it (`b4fa07a9…`) appears once, as a read-only
diagnostic, and the controller's own note on that record says *"timings under
instrumentation are not performance numbers."* Diagnostics run no correctness
check at all — `run_route` in `Project/harness/profile_worker.py:1343-1348`
calls the model in a loop and checks nothing.

## 2. The last build that does have a full board

| | |
|---|---|
| artifact | `54057a3389489e6cf7653727b3893bad6e10d88ae94381227b037431cc7086b2` |
| shapes measured | 12 of 14 |
| correctness | `correct: true` on every row |
| geomean | **10.14x** |
| best shape | 30.90x (shape 13) |
| worst shape | 2.02x (shape 8) |
| lane | screening, so not promoted |
| file kept at | `Project/authority/blobs/54057a33….py` (checked, present) |

Per-shape: 1 8.38 · 2 15.53 · 3 12.15 · 4 10.07 · 5 9.31 · 7 23.86 · 8 2.02 ·
9 4.55 · 10 6.30 · 11 17.66 · 12 10.54 · 13 30.90.

**This is the board every judge-facing number in `Project/drafts/` describes.**

## 3. Defect found this hour, and it is in a judge-facing file

`Project/drafts/tech_report_draft.md:232` cites
`Project/loop/geomean_camp_final.py` as the source of the headline board.

**That script computes a different board.** Its rows come from four artifacts
(`2778b747`, `599f5dad`, `301d7063`, `418952bf`), its shape-11 value is 17.42
rather than 17.66, and its geomean is **10.6858x**, not 10.14x.

Both boards are real measurements. The difference is that 10.14x is twelve rows
from **one** file and 10.6858x is twelve rows from **four**. The report quotes
the right number and cites the wrong file. `STATE.md` §0b already recorded that
the 10.6858x board was never single-build; the citation was never updated to
match.

**Consequence:** a judge who follows the citation finds numbers that do not
match the table. This must be fixed before submission. It does not make 10.14x
wrong.

## 4. What is unmeasured between §2 and §1

These landed after `54057a33…` and have no twelve-shape board:

- QKV grid split, then a token-count gate on it
- single-pass LayerNorm statistics
- graph replay returning its buffer instead of cloning
- k028 attention (base-2 softmax, causal loop split)
- fast erf GELU
- autotune stabilization (longer internal benchmarking)
- 31 Aug evening: graph input-copy skip, final-norm autotune, fp16 residual and
  repack removal, mask-predicate caching

Several have their own single-shape measurements. **None of them has a board.**
Their combined effect on the geomean is unknown and could be negative.

## 5. What is certain, and what can never be

**Certain — correctness.** The official predicate is a yes/no decided by the
official code, on seven trials with seven distinct output hashes. All twelve
rows of the §2 board passed. A shape that fails scores zero, so this is the half
that can sink the submission, and on §2's file it is clean.

**Not certain — timings.** A GPU timing is a physical measurement. Repeats land
within about 1% on large shapes and up to about 5% on small ones. One shape
(12, seq 32) was measured 13.2% apart on byte-identical bytes before the
autotune fix. Nobody's GPU benchmark is exact; the standard everywhere is
median plus stated spread.

**Defensible sentence:** *"10.14x geomean over 12 shapes, measured on this exact
file, baseline and candidate timed in the same run, reproducible to about ±5%."*
**Not defensible:** *"10.14x."*

## 6. Blocked, owner-only

1. **Audit recording is broken.** Nothing can be promoted to champion and no
   verdict binds to any row. `STATE.md` §1.
2. **Shapes 6 and 14 evaluators** compare a mask's device against
   `torch.device("cuda")` while the generator returns `cuda:0`. One line each:
   `Project/tools/shape14_eval.py:274`, `Project/tools/shape6_local_eval.py:146`.
3. **File deletion and moves.** The post-LOCK shell blocks `rm`, `mkdir` and
   `git mv` by design, so §7 is a list for you, not something the agent can do.

## 7. Files that no longer play a part — safe to delete or move to an archive

Each is self-labelled dead in its own first lines except where noted.

| file | why |
|---|---|
| `Project/drafts/day2_plan.md` | header says SUPERSEDED 29 Aug |
| `Project/drafts/harness_v2_proposal.md` | header says SUPERSEDED 30 Aug |
| `Project/drafts/rental_day_runbook.md` | header says ARCHIVED, rental cancelled |
| `Project/drafts/organizer_questions.md` | header says RETIRED / NO-SEND |
| `Project/drafts/official_grader_all_dials_20260829.txt` | raw dump, superseded by the shape board |
| `Project/kernels/__pycache__/` | build artifact |
| `Project/kernels/k028_smoke.py` | cannot run post-LOCK, guard blocks smokes |
| `Project/kernels/fp16_route_smoke.py` | same |
| `Project/OWNER_HANDOFF_TONIGHT.md` | time-specific to 30 Aug |
| `Project/WIN_BAR.md`, `Project/WIN_PLAN.md` | working notes from 31 Aug, several claims superseded by this file |

**Keep everything else.** In particular `Project/authority/`, `Project/loop/`
and `Project/audits/` are the evidence trail and are what makes any number here
checkable. `Project/kernels/k0*.py` are the work history and are cited by the
report.

## 8. The decision

**A. Restore `54057a33…` as the submission file.** The repo becomes
self-consistent: the file that ships is the file all twelve numbers describe,
and every one passed correctness. Reversible — the newer work stays in git and
in the blobs.

**B. Measure the current file.** Needs a quiet box and roughly an hour. If it
passes correctness and beats 10.14x, it ships instead. If anything fails, fall
back to A.

**A first, then B with whatever time is left.** That way the floor is 10.14x and
nothing can end worse than it is now.
