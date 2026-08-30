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

**LOCKED AND OPEN. The grind may start.** Steps 1–9 of
`Project/OWNER_RUNBOOK_POSTLOCK.md` were completed by the owner (via Codex) on 30 Aug
20:53–20:55 SGT and independently re-verified against the live commands afterwards:

- `trusted_controller.py status` → `controller: OK`, lock **active** and **valid**,
  29 protected files, `lock_id lock-b7848d4736461b971acd`,
  epoch `post-fix-20260830T125311Z`.
- All 16 pre-gate `RULE_VIOLATION` brakes retired via
  `clear_pregate_verdicts.py` (`FINDING_ACCEPTED_ROW_RETIRED`). Zero brakes remain.
- `CAMP-POSTLOCK` active: 0 attempts, 0 calibrations, 60-attempt budget,
  `timing_config` bound to the controller protocol.
- 0 permits issued, 0 consumed, 0 measurements, no permit armed.
- Auditor is Claude (`/usr/local/bin/claude-auditor`, root-owned, pinned hash matches).

**Shape 1 groundwork is done (30 Aug 21:11–21:17 SGT): runbook steps 10 and 11.**
Calibration for shape 1 is immutable at **noise 0.003467, promotion threshold 1.03**
(entry `run-aa8def96f24fc77f96be122f04351c47`, `event_speedup 1.00347`, correct).
Two diagnostics ran against `k004_graphed_triton.py` on shape 1, and together they
**killed the runbook's own example direction before it cost an attempt**:

- nsys (`profile-cb6cd3c903aacee64a468d63`): 2 launch API calls per forward.
  **Launch overhead is no longer the shape-1 bottleneck.**
- torch-profiler (`profile-3d9f7dabfba6348163f24495`, same bytes): 58 kernels per
  forward, GPU busy 87.6%. Time splits **GEMM 31% / elementwise+norm 49% /
  our Triton attention 18%**. Almost half the forward is pointwise traffic.

**Do not cite nsys `gpu_idle_fraction` on a graphed route.** It read 0.981 here and
is an artifact — `collect_nsys` passes no `--cuda-graph-trace=node`, so it saw 1 of
58 kernels. torch-profiler read 0.124 on identical bytes. LESSONS 31. The "98% GPU
idle" line in the 30 Aug rehearsal note is withdrawn.

## STOPPED 30 Aug ~22:50 SGT — two things need the owner

**1. Audits cannot be recorded. This blocks everything.** The first real audit ran, cost
$2.49, took 384 s, and returned a complete verdict — integrity **RETEST**, technical
**WEAK_DIAGNOSIS**. `record_audit_result` then refused it with "verdict does not match
full schema", naming every required property as missing while the stored artifact
`Project/authority/blobs/0b3fa1ce…audit-response.json` visibly contains all of them. The
journal shows only `attempt_failed` / `AUDITOR_PROCESS_ABANDONED_WITHOUT_TERMINAL_EVENT`,
so **no hard verdict is latched and no brake is set** — the absence is a bug, not a pass.
`audit_champion.py` and the audit authority are inside the LOCK and Write-denied to the
agent, so this is owner-only. **Until it is fixed nothing can promote**, and each further
attempt burns budget plus ~$2.49 to produce an unpromotable row.

**2. The shape-1 result is real but was credited to the wrong mechanism — my error.**
Attempt 1 measured **2.0748x, correct**, on shape 1. The auditor showed, and I re-derived,
that the bound baseline profile has the forward **96.6% GPU-busy** (0.199 ms of launch
idle per 5.93 ms forward), so graph replay can buy at most **~1.035x** — barely the 1.03
threshold. The other ~93% of the 2.67 ms saving comes from the second change bundled into
k004: an inlined flash-style Triton attention plus fused packed QKV, which removes
repeated passes over a 16.8 MB per-layer score tensor. So `F-shape1-graph`
(`cuda-graph-replay` / `launch-overhead`) is the wrong family for this win, and the five
further graph-replay families registered for shapes 5, 9, 10, 11, 13 carry the same
misframing. They are inert; nothing was spent on them. See LESSONS 32 and 33.

Also independent of the audit: the run tripped the harness's **own** timing tripwire
(`event_wall_speedup_agreement_ratio` 1.2552 against the 1.25 threshold at
`candidate_worker.py:476`), so the controller had already set
`performance_eligible: false`. This run was never promotable. The auditor found **no
manipulation** and traced a benign cause (unpaired sequential wall blocks plus a
disturbance confined to round 3); it asks for a re-measure on a quiet box.

**Ledger at stop:** 1 of 60 attempts spent, 0 promoted, 0 strikes, 12 shapes calibrated
with immutable thresholds, 4 profiles, 6 families registered, no permit armed, lock valid,
campaign not stalled, tree clean on `grind-lastday`.

**What is genuinely banked and defensible:** the 12 calibrated noise floors; the
115-launches-to-2 counter measured on both routes; the nsys graph-blindness finding with
its clean control; and the fact that the control system caught a real attribution error
that its own operator had missed and written up as a success.

**The 49 "pending" audits are inert — do not panic at them.**
`champion_watch.py --dry-run` lists 49 pre-gate entries. Running the watcher skips
every one of them in about a second with `packet is absent`, *before* any auditor
process starts: they predate the packet store, so there is nothing to audit. No
quota is spent, no attempt is recorded, no retry counter moves, and `owner_attention`
stays empty. They do **not** queue ahead of new work — the loop walks past them and
launches the first entry that has a real packet. Verified by running it, 30 Aug.

**The private keys are inside HOME, and that is a real residual.** The runbook said
`~/techjam-keys`, so `keygen` needed `--allow-home-key`. Files are `0600`, but that is
guard-level, not OS isolation: anything running as `admin` can read them. So do **not**
claim the agent cannot reach the signing key. Claim instead: the key is never in the
repo, never in a prompt, never printed; every privileged transition is a signed,
one-use, scope- and time-bounded capability; every use is a separate hash-chained
journal event. Upgrading this means moving the keys outside HOME, after which every
`mint-capability` needs `sudo` — a five-minute change if the stronger claim is wanted.

The passphrase was scrubbed from `OWNER_RUNBOOK_POSTLOCK.md` (30 Aug) and is **not**
rotated: it is a generated string, unrelated to any login, and it is half of a two-part
secret whose other half (the key file in `~/techjam-keys`) is untracked and never ships.
It remains in git history at `dcb4f25`/`dbcd487`; rewriting history on a locked repo
before freeze is a larger risk than the leak, so it stays. Keep it out of new commits.

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
