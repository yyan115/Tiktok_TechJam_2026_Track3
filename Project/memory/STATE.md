# STATE — first ten seconds of a session

This file is deliberately short and holds no plan. Two files do that:

- **`Project/GRIND_ENTRYPOINT.md`** — the operating manual: the commands, the one
  next permitted action, the lanes, the stop conditions. Get it by running
  `python3 Project/tools/session_bootstrap.py`, which prints it and then the live
  controller and gate status.
- **`Project/HANDOVER.md`** — the single source of truth for state, open defects
  and the FIX → LOCK → GRIND plan.

Then read all of `Project/memory/LESSONS.md`, every session, and
`Project/research/INDEX.md` before relying on any research note.

**If a command and a document disagree, the command is right** — including this one.

Updated: 2026-08-30 ~20:30 SGT. Branch `grind-day1`. Every claim below was checked
against the code and the ledgers on that date, not carried over from a note.

## Where the project actually is

**Pre-LOCK. Nothing can be measured.** `trusted_controller.py status` exits 2 with
`REFUSED: LOCK manifest must be a regular non-symlink file` — that is the expected
reading today, not a fault. No GPU work is possible until the owner runs
`Project/OWNER_LOCK.md`.

**The FIX phase is done; LOCK is now purely an owner ceremony.**
`Project/harness/profile_worker.py` is installed (`ef2a866`) and the controller no
longer refuses on it. `Project/authority/blobs/` holds no keys yet, so the ceremony
is `keygen` → build → activate, per `Project/OWNER_LOCK.md`.

**LOCK does not unblock the first card.** 16 uncleared `RULE_VIOLATION` verdicts freeze
every new permit. They are the auto-audit's 30 Aug sweep of the pre-gate board, and all
16 are the same finding — the run predates the citation gate, so it has no plan or
citation trail. The numbers themselves were called credible; the paperwork did not
exist, because the runs came before the thing that requires it. Clearing them is
owner-only and one row at a time (`verdict-clear --kind violation` with a signed
`resolve_integrity_verdict` capability). Moving the cutoff instead is not available:
`st["created"]` is set only by `init`, and `init` refuses on an existing state and a
non-empty log.

**The documented unlock for those verdicts does not work.** `run_gate.py verdict-clear
--kind violation` spends a signed owner capability, prints "resolved by
controller-verified owner authority", and leaves the brake exactly where it was — the
brake is an audit-authority hard event, and `cleared_verdicts` is display state. It also
cannot be minted through the ceremony without `--allow-unknown-action`, because the
ceremony signs `verdict.resolve` while the gate demands `resolve_integrity_verdict`.
Both were found by rehearsing the whole ceremony on a repo copy, and `verdict-clear` now
says so and exits 1. The path that works is
`python3 Project/tools/clear_pregate_verdicts.py` — read its header; it retires all 16
from ONE `audit.resolve` signature and was rehearsed end to end (16 retired, brake off).

**Ten commits landed on 30 Aug between 15:46 and 17:08** (`ed053f2..eae70a1`) and they
changed what is true:
- Candidate code now runs in a bubblewrap jail, and a real Triton kernel has been proven
  to compile and run inside it on this GPU (`Project/tools/tests/sandbox_boundary_test.py`,
  max abs error 0.0).
- Owner authority is Ed25519-signed: keys, a hash-pinned LOCK over 29 files, one-use
  permits, and a controller that refuses everything until the lock validates.
- The competence gate, audit authority, staged allowlist guard and the diagnostic lane
  all exist in code.
- Thirteen test suites live under `Project/tools/tests/`; all thirteen were green at
  `1ed6e20`. Re-run them before trusting any of the above.

**The board is dead until it is re-measured.** 130 promoted rows exist in
`Project/results/JOURNAL.jsonl`; **zero** are promotion-eligible under the new audit
authority — none was taken under a permit and none carries a bound verdict. 16 uncleared
RULE_VIOLATION rows freeze all new permits. So the whole board gets re-measured after
LOCK, and the old ~11x geomean is not quotable in anything (HANDOVER 3.1). Neither is
`Project/results/LEADERBOARD.md`, which stars max-ever rows across incomparable runs.

**Superseded:** the old owner action "paste `Project/loop/OWNER_PATCH_card_gate.md` v4
into the guard" is gone. The guard, settings and runner shim are now staged bytes in
`Project/lock_staging/`, installed by the owner as step 1 of `Project/OWNER_LOCK.md`.

## The two rules that get broken by accident

1. **Idle box before anything timed.** Do NOT check it with `pgrep -f "codex exec"` —
   `-f` matches the joined command line, so any command that merely names a benchmark or
   an auditor reads as one running (DECISIONS.md:53 recorded that false reading once).
   The sound check is in code: `champion_watch.py` (`_argv_is_busy` / `runner_busy()`)
   matches whole argv elements and exempts the caller's own process tree, and it refuses
   to launch an audit while a runner or controller run is live.
   `python3 Project/tools/champion_watch.py --dry-run` reports audits in flight as
   `active`, read from the ledger rather than a process scan.
2. **Commit candidate bytes BEFORE first controller contact.** The permit binds a sha;
   back-editing measured bytes is evidence corruption.

## Standing rules (unchanged)

Never touch frozen/protected files. Every benchmark goes through a permit and the trusted
controller — no raw dials, no ad-hoc script, no `python3 -c` timing. One GPU process at a
time. Never compare absolute times across invocations (LESSONS #11). Trace every number to
the artifact that produced it (LESSONS #24). Guard etiquette: never put `clean`, `reset`,
`restore` or `checkout` after `git` in one command segment. Plain language, no jargon
walls. The owner's explicit "go" is required before repo actions, and the owner's stop
overrides everything, immediately.

## Clock (owner's, not the working queue)

CODE FREEZE 31 Aug 20:00 SGT → packaging 31 Aug 20:00 to 1 Sep 02:00 → final ~10h
reproduction/contingency → submission AND Devpost registration close 1 Sep 12:00 GMT+8.
Report/README/video prose must be finished BEFORE the freeze; only numbers and assembly
wait for it.
