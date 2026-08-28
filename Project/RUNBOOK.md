# RUNBOOK — how to operate this project (one page)

## Environment
Fedora Linux, RTX 3060 Ti (8 GB), Python 3.14, torch 2.12.0+cu130, triton 3.7.0.
No venv needed — system python3 has everything. Shape 14 additionally needs a rented
≥24 GB GPU (see PLAN.md Stage 4).

## The referee (trusted runner) — all commands run from repo root
```
python3 Project/harness/runner.py check                     # verify official files untouched
python3 Project/harness/runner.py env                       # print environment fingerprint
python3 Project/harness/runner.py calibrate --shape N       # noise floor (required before comparing)
python3 Project/harness/runner.py run --shape N --impl Project/kernels/kXXX.py
python3 Project/harness/runner.py leaderboard               # regenerate LEADERBOARD.md
python3 Project/harness/runner.py packet --id ENTRYID       # neutral evidence packet for audits
```
Add `--ledger /tmp/somewhere.jsonl` to any run to keep test/red-team results OUT of the
production journal.

**Serialization rule: exactly ONE runner process at a time.** The journal has no file
locking by deliberate decision (single-operator project); running two evaluations
concurrently voids the ledger's integrity assumptions.

Red-team suite (run after any harness change; expected results shown):
```
python3 Project/harness/runner.py run --shape 1 --impl Project/harness/redteam/rt01_monkeypatch.py --ledger /tmp/rt.jsonl   # MUST abort: TAMPER DETECTED
python3 Project/harness/runner.py run --shape 1 --impl Project/harness/redteam/rt02_addrcache.py  --ledger /tmp/rt.jsonl   # MUST print correct:false, promoted:false
```

Recording an audit verdict against a journal entry (binds auditor output to the entry;
the leaderboard's audit column reads this):
```
python3 Project/harness/runner.py record-verdict --id ENTRYID --verdict PASS --source Project/audits/<which_review>.log
```

## Candidate contract (files in Project/kernels/)
`build(otb, config) -> torch.nn.Module` — fresh model, parameter names identical to the
baseline (strict weight copy), same forward signature. Optional `copy_weights(baseline,
candidate)` if names differ, plus `NAME` / `DESCRIPTION` strings. Candidates must never
mutate their inputs (checked), never touch the harness or official files (locked), and
never cache outputs (tripwired).

## What gets written where
The runner appends `Project/results/JOURNAL.jsonl` (append-only, machine-written — never
edit by hand) and regenerates `Project/results/LEADERBOARD.md`. Audit verdicts are recorded
to `Project/audits/verdicts.jsonl` as `{"entry_id":..., "verdict":..., "source_log":...}` —
the recorder adds entry_id; Sol's raw output alone doesn't carry it. Raw review logs
(`Project/audits/*_raw.log`) are PRIVATE — gitignored, never publish (they contain
session transcripts).

## Recovery / gotchas
- OOM on a big shape: record the failed entry as-is; do not shrink the problem to force a pass.
- Absolute latencies are NOT comparable across runner invocations (GPU clock state varies
  ~10%); only within-entry speedups count. Champions require the current runner sha.
- After ANY harness edit (user-approved only, post-freeze): version bump → prior promotions
  auto-retire to "legacy" → re-run champions to re-establish.
- Fresh session lost? `Project/memory/STATE.md` is auto-injected on start; CLAUDE.md points
  everywhere else.
- Enforcement layers, honestly ranked: (1) deny rules in `.claude/settings.json` — the
  actual lock; (2) committed hashes + git history — makes tampering visible and provable;
  (3) the Bash guard hook — a seatbelt that catches accidents, never the load-bearing
  protection.
