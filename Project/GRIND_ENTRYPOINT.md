# GRIND ENTRYPOINT — canonical cold start

Read `CLAUDE.md` and this file, then start. This file tells you how to find out where
you are, what the one next permitted action is, and what you may not do. It does not
restate the plan. `Project/HANDOVER.md` is the single source of truth for current state
and open defects; `Project/PLAN.md` and `Project/RUNBOOK.md` remain authoritative for the
plan of record and for operating instructions. This file is not a sixth plan and must
never grow into one.

## 1. What this project is, and what winning means

TikTok TechJam 2026 Track 3: make the official PyTorch transformer benchmark run faster
on 14 fixed shapes, on one RTX 3060 Ti, without changing what it computes. Winning is a
submission file byte-identical to the official script outside its one designated region,
correct under the official predicate on every shape, with a measured speedup we can
defend line by line — the conditions, the baseline, the repeats, the audit. Judges do not
rerun anything (`Project/research/competition-scoring.md`), so the evidence IS the
product; a number we cannot defend is worth less than a smaller number we can. The
kernels and their measured results are primary; the control harness is supporting story.
Rubric: Technical 35 / Innovation 20 / Impact 20 / Feasibility 15 / Presentation 10.

## 2. First five minutes

```
python3 Project/tools/session_bootstrap.py
python3 Project/tools/run_gate.py reconcile
python3 Project/tools/run_gate.py status
python3 Project/harness/trusted_controller.py status
python3 Project/tools/champion_watch.py --dry-run
```

Reconcile FIRST, then read status: an unsettled run makes every other reading stale.
Then read `Project/HANDOVER.md`, `Project/memory/STATE.md`, all of
`Project/memory/LESSONS.md` (every session, all of it), and `Project/research/INDEX.md`
plus the notes you are about to rely on.

**Generated state beats prose.** If a command and a document disagree, the command is
right and the document is stale — including this one, including STATE.md, including any
scoreboard. Never carry a number from a note into a claim; trace it to the artifact that
produced it (LESSONS #24).

## 3. Discovering state mechanically

| Question | Command | Read |
|---|---|---|
| Is the box locked? | `trusted_controller.py status` / `verify-lock` | a `REFUSED: LOCK ...` line means pre-LOCK; nothing can run |
| Campaign and budgets | `run_gate.py status` | `active_campaign`, `campaigns[cid]`: `max_total_attempts`, `scientific_attempts`, `max_calibrations_per_shape`, `max_total_calibrations`, `stall_window`, `timing_config`, `score_scenarios`, `stalled` |
| Families, strikes, closures | `run_gate.py status` | `family_registry`, `family_admissions`, `groups["<family>\|<shape>"]` → `strikes`, `closed`, `best_speedup`, `budget_snapshot`, `infrastructure_paused` |
| Counter evidence on hand | `run_gate.py status` | `profiles` (each carries `metrics`, `supported_bottlenecks`, `target_sha256`) |
| What is waiting on me | `run_gate.py status` | `_permit_armed`, `pending_screen_judgment`, `pending_audit_decisions`, `pending_postmortem` |
| Audit backlog | `champion_watch.py --dry-run` | `pending`, `active`, `owner_attention` |
| Verdict pressure | `wc -l -- Project/audits/verdicts.jsonl`, `Project/audits/audit_events.jsonl`, gate `cleared_verdicts` | hard verdicts freeze permits |
| Champion bytes | `sha256sum -- Project/submission/torch_transformer_benchmark_submission.py` | this is your `--target-sha256` |
| History | `git log --oneline -20`, `tail -n 40 -- Project/loop/gate_log.jsonl` | what actually happened |

`Project/results/LEADERBOARD.md` is NOT a ship selection: it stars max-ever rows across
incomparable invocations (HANDOVER 3.3, LESSONS #11). Never quote it as a result.

### The single next permitted action — first match wins

1. Controller `status` refuses on LOCK → **pre-LOCK**. No GPU work is possible; the
   controller refuses every run. Do the remaining FIX items in HANDOVER §4 Phase 1, then
   stop and tell the owner LOCK is next. Do not start experiments, do not improvise a
   measurement path.
2. `_permit_armed` true, or a request issued and unsettled → finish that one attempt, or
   `run_gate.py reconcile`. One attempt at a time, always.
3. `pending_screen_judgment` set → `run_gate.py screen-judge` (§4 step 6).
4. `pending_audit_decisions` non-empty → let `champion_watch.py` run the bound audit,
   then `run_gate.py audit-finalize --entry-id <id>`.
5. `pending_postmortem` non-empty → a `research` step carrying `--postmortem` first.
6. Uncleared RULE_VIOLATION or RETEST → §7.
7. Campaign `stalled` → research, an untried family, and an owner stall receipt.
8. Otherwise → the loop below, starting at the diagnostic.

## 4. The mandatory loop

Shapes 6 and 14 never use `run`. They have a dedicated side lane
(`run_gate.py side-evaluate`, then `trusted_controller.py side`), and side evidence is
never a primary champion.

**1. Diagnostic permit.** Profile before you prescribe. No direction opens on a hunch.
```
python3 Project/tools/run_gate.py diagnostic --campaign <CID> --shape <N> \
  --target-sha256 <champion sha> --tool nsys --supports launch-overhead \
  --question "<the concrete question, >=40 chars>" --route "<code route being profiled>"
python3 Project/harness/trusted_controller.py issue-permit \
  --request Project/loop/requests/<request>.json --capability <owner-signed capability>
```
`--tool` must appear in that bottleneck's `evidence_tools` in
`Project/loop/mechanism_catalog.json`. A diagnostic permit cannot authorize candidate
bytes, promotion, a primary-ledger write, or a strike.

**2. Profile.**
```
python3 Project/harness/trusted_controller.py run --permit <permit-id> --shape <N> \
  --impl Project/submission/torch_transformer_benchmark_submission.py
```
The profile artifact must land in `Project/loop/profile_evidence/` and be hash-bound by
the controller. Nothing else may write there. See §8 residual 1 — this lane is not wired
end to end yet.

**3. Counter-backed diagnosis.**
```
python3 Project/tools/run_gate.py reconcile
python3 Project/tools/run_gate.py status
```
The profile record appears under `profiles`. Pick the bottleneck id from the catalog whose
`required_metrics` your record actually contains. "Launch-bound" is a claim about
`kernel_launches`, not a vibe — k011 was built on an occupancy guess and lost 15%
(LESSONS #21).

**4. Research.**
```
python3 Project/tools/run_gate.py research --campaign <CID> --index-hash <_index_hash_now> \
  --notes roofline-table.md,agent-loop-design.md --summary "<>=200 chars>" \
  [--postmortem "<>=200 chars, must state a case for revival>"]
```
Fresh web research per technique is encouraged; assume the field has moved. Anything worth
using becomes a note in `Project/research/` and gets cited.

**5. Trusted family assignment.** Family identity comes from the catalog or a
controller-verified admission — never from prose you write. Same shape plus same mechanism
is the SAME family; a child family needs a >=100-char material novelty basis and an owner
resolution. You prepare the spec (`family_id`, `shape`, `mechanism`, `bottleneck`,
`changed_resource`, `expected_counter_change`, `parent_family_id`, `budget_attempts`,
`budget_minutes`, `admission`, `allow_new_attempts`); the OWNER signs the capability and
runs `trusted_controller.py authorize`; you then run:
```
python3 Project/tools/run_gate.py family-register --campaign <CID> \
  --family-spec <spec.json> --authority-receipt Project/authority/receipts/<event>.json
```

**6. Cheap falsifier.** Kill the idea cheaply before you pay for it. Screening runs in the
scratch lane and cannot promote.
```
python3 Project/tools/run_gate.py plan --campaign <CID> --direction <family-id> \
  --mode screening --shape <N> --impl Project/kernels/kNNN.py \
  --hypothesis "<>=50>" --prediction "<number>" --prediction-kind win \
  --predict-min <x> --predict-max <y> --target-sha256 <sha> --bottleneck <catalog id> \
  --counter-evidence <profile-record-id> --falsifier "<>=60>" --falsifier-kill "<>=20>" \
  --prior-family-verdict <NONE|seq:<n>:<result>> --kill "<>=20>" \
  --sources "research/roofline-table.md:12-18" --reasoning "<>=100>"
```
Then, after the run and `reconcile`:
```
python3 Project/tools/run_gate.py screen-judge --direction <family-id> --shape <N> \
  --observed <speedup>
```
The gate computes hit or miss from the bounds stored at permit time; `--observed` is only
an attention check. You never declare your own result. If you do not know the exact
`--prior-family-verdict` string, run the command once and read it from the refusal.

**7. Implement.** Edit only `Project/kernels/kNNN.py` or
`Project/submission/dispatcher_region.py`. Every branch you add gets a local smoke against
the baseline before it costs GPU time (LESSONS #17). Keep the official
`forward(self, x, valid_token_mask=None)` signature exactly (LESSONS #16).

**8. Preflight.** Commit the candidate bytes BEFORE first controller contact — the permit
binds a sha, and back-editing a measured candidate is evidence corruption.
```
git add -- Project/kernels/kNNN.py
git commit -m "candidate kNNN: <one line>"
sha256sum -- Project/kernels/kNNN.py
python3 Project/tools/tests/competence_gate_test.py
python3 Project/tools/tests/trusted_controller_test.py
```

**9. Correctness.** `--mode correctness` (scratch lane). A correctness failure is a strike.

**10. Sanitize.** `compute-sanitizer racecheck` and `memcheck` on every authored kernel,
looped until clean. It can force code changes, so it precedes the frozen bytes, not the
board. It is not reachable from the post-LOCK agent shell — see §8 residual 2.

**11. Paired clean benchmark.**
```
python3 Project/tools/run_gate.py plan|delta --campaign <CID> --direction <family-id> \
  --mode optimization --shape <N> --impl Project/kernels/kNNN.py ...
python3 Project/harness/trusted_controller.py issue-permit --request ... --capability ...
python3 Project/harness/trusted_controller.py run --permit <id> --shape <N> --impl <path>
python3 Project/tools/run_gate.py reconcile
```
`delta` is the concise continuation of an already-planned direction; `plan` opens one.
Both emit exactly one request for exactly one run.

**12. Profile delta.** A second diagnostic against the new bytes: did the counter you
predicted actually move, in the predicted direction? A speedup with an unmoved counter is
an unexplained result, not a win.

**13. Technical review.** The auditor's second job: does the counter evidence support the
diagnosis, and did the change address it. `WEAK_DIAGNOSIS` and `MISSING_EVIDENCE` pause
promotion; `TECHNICAL_DISAGREEMENT` is advisory. This channel is separate from integrity.

**14. Integrity audit.** `python3 Project/tools/champion_watch.py` launches the bound blind
audit for the pending entry. You cannot record your own verdict; that path is denied to
the agent shell by design.

**15. Promote or reject.**
```
python3 Project/tools/run_gate.py audit-finalize --entry-id <entry-id>
```
Eligibility is derived, never asserted: correct, clean, above the calibrated noise floor,
and carrying a bound independent audit. An improvement must beat the group's best by at
least 3%; anything less is not an improvement and adds a strike.

**16. Durable memory.** Append what happened to `Project/memory/DECISIONS.md`, any new rule
to `Project/memory/LESSONS.md`, any reusable finding to a `Project/research/` note. A
scratch script worth replaying belongs in `Project/tools/smokes/` (LESSONS #20). Rejected
probes are as valuable as accepted ones — record them so nobody re-explores.

## 5. Hard measurement rules

These exist because they were broken and it cost this project its headline number
(HANDOVER 3.1). They are not style preferences.

- **One GPU process at a time.** Ever. No parallel sweeps, no background GPU work.
- **Never benchmark while an audit runs.** Contention inflates graphed-candidate ratios by
  up to 3x (LESSONS #22). Establish an idle box before anything timed.
- **Profiling and timing never interleave.** A profiled run is not a timed run.
- **Profiler output backs mechanism claims only.** It never becomes a published
  performance number, in any document, ever.
- **Every speedup comes from the trusted controller**, through a permit, at a recorded
  sha. No raw-dial benchmarking, no ad-hoc script, no `python3 -c` timing.
- **Never compare absolute times across invocations.** GPU clock state moves about 10%
  between runs; only within-entry paired speedups are comparable (LESSONS #11). Any figure
  derived by mixing absolute times across invocations is invalid and gets deleted, not
  adjusted.
- **Keep every repeat.** No cherry-picking, no dropping a bad round, no best-of.
- **Predeclare the aggregate** (median, geomean, which scenario) before you look. The
  campaign's `timing_config` and `score_scenarios` are fixed at campaign open for exactly
  this reason.
- Record machine state before and after; report event and wall timing side by side, and
  flag divergence.

## 6. Frozen versus yours

Frozen, never edit: `torch_transformer_benchmark.py`,
`tensorflow_transformer_benchmark.py`, `README.md`, `Project/shapes.json`,
`Project/manifest.json`, everything the runner writes under `Project/results/`,
`.claude/**`, and after LOCK the whole authority surface — `Project/authority/**`,
`Project/loop/gate_state.json`, `gate_log.jsonl`, `cards.jsonl`, `permits_used/`,
`profile_evidence/`, the audit ledgers under `Project/audits/`, `Project/tools/run_gate.py`
and `Project/harness/**`.

Yours to edit: `Project/kernels/`, `Project/submission/dispatcher_region.py` and the
generated submission, `Project/research/`, `Project/drafts/`, and the memory files under
`Project/memory/`.

Post-LOCK the Bash surface is a deny-by-default allowlist
(`Project/lock_staging/guard_bash.py`): the locked Python entrypoints, a few read-only
tools (`sed -n`, `head`, `tail`, `wc`, `ls`, `stat`, `sha256sum`, `rg`, all `--`-separated
and repo-relative), and a narrow `git` subset. No pipes, no redirection, no globbing, no
`python3 -c`, no `cat`. If a command bounces, that is the design working: find the
sanctioned route or stop and ask. Never argue with the guard, never work around it, never
edit it.

Guard etiquette that predates LOCK and still applies: never put `clean`, `reset`,
`restore`, or `checkout` after `git` in one command segment.

## 7. Stop conditions

- **Three strikes closes a family.** Three non-improving attempts (a screening miss, a
  correctness failure, a non-qualifying optimization result) close the direction and
  register postmortem debt. A closed family reopens ONLY through an owner-authorized
  `reopen` bound to its closure nonce. There is no unlock command, and no argument you can
  write is an exception (LESSONS #23).
- **Campaign stall brake.** If the last `stall_window` production outcomes contain no
  meaningful improvement, the campaign stalls. Recovery needs all of: a fresh research
  cycle, a profile captured after the stall, an untried mechanism family, and an owner
  stall receipt.
- **A missing audit verdict makes a result ineligible.** Absence is not a pass. An entry
  with no bound verdict never promotes and never appears in a claim.
- **Hard verdicts freeze permits.** An uncleared RULE_VIOLATION freezes every new permit.
  RETEST clears mechanically (`verdict-clear --kind retest --confirm-entry <row>`: same
  bytes, same shape, passed, newer than the verdict, produced under a reconciled
  confirmation permit). RULE_VIOLATION is owner-only: a signed controller capability bound
  to that exact verdict and resolution. Workspace text has no authority.
- **Budgets are ceilings, not targets:** per-family attempts from the immutable budget
  snapshot, per-campaign attempts, per-shape and total calibration caps; three consecutive
  infrastructure failures pause a group.
- **The owner's stop overrides everything**, immediately, mid-anything.

## 8. Known residual risks — do not overclaim

Call this system **AI control with published residuals**. Never call it zero trust; that
label is false and would be caught.

1. **The diagnostic lane is not wired end to end.** The gate emits diagnostic requests and
   validates profile artifacts, but the controller never binds `diagnostic_profile_sha256`
   and the worker does not profile — so a diagnostic cannot reconcile, and until it can,
   `plan --counter-evidence` has nothing valid to cite. Completing this lane is real work,
   not a formality. Do not route around it by prescribing without counters.
2. **No profiler is mounted in the sandbox** and none is on the post-LOCK allowlist, so
   `nsys`, `ncu` and `compute-sanitizer` are not reachable from the agent shell. `ncu`
   additionally needs sudo on this box (never modprobe, never reboot). Sanitizer and
   profiler passes are owner-run or pre-LOCK work.
3. **Every measurement in the repo today is pre-gate**, and all 12 published rows carry
   baselines 6-63% slower than their own calibration — so the recorded figure near 11x is
   not a defensible headline. It can only be replaced by a controlled campaign, never by
   arithmetic on old rows (HANDOVER 3.1).
4. **Shapes 6 and 14 are side evidence.** Their evaluators use CPU RNG, so their inputs are
   not bit-identical to a default judge run, and shape 14 proves decomposed execution, not
   a literal single B=32 call. Label both precisely every time.
5. **Runner-internal residuals are documented, not fixed** — candidate source once executed
   inside the runner process, journal appends without locking, retroactive calibration
   selection. Fixing them means editing the frozen referee, which is an owner-only trade
   (HANDOVER 3.8).
6. **The guard is a cooperative, pattern-based control**, not OS isolation. It stops
   accidents and raises the bar; deliberate evasion remains possible, and would be naked,
   logged misconduct.
7. **Authority ergonomics are rough**: privileged transitions need an owner-signed
   capability bound to a subject hash you cannot conveniently compute post-LOCK. Prepare
   the artifact, hand it to the owner, and wait. Waiting is a valid state.
