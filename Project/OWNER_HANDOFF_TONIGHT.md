# Hand-off: what to paste before you sleep

Written 30 Aug ~22:00 SGT. Everything below was checked against the live system,
not copied from a note.

---

## 1. Open the terminal — NOT just any terminal

The commands need the repo and three variables. A fresh terminal has none of them.
Paste this block first, with your real passphrase where the placeholder is:

```bash
cd /home/admin/Desktop/Repos/Tiktok_TechJam_2026_Track3
export KEYS=~/techjam-keys
export CAMP=CAMP-POSTLOCK
export PASS='<the signing-key passphrase>'
```

If you still have the terminal from the lock ceremony open, it already has these
and you can skip straight to step 2.

The passphrase is deliberately not written here — this repo ships publicly.

---

## 2. Paste these three

Each one prints a **wildcard warning**. That is expected and correct: it is
telling you the capability covers everything of its kind rather than one named
thing. That is the entire point.

```bash
printf '%s\n' "$PASS" | \
  python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action permit.issue --target 'shape:*' --campaign "$CAMP" \
    --reason "overnight grind 30-31 Aug" \
    --max-uses 400 --expires-minutes 1400 \
    --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_grind.json

printf '%s\n' "$PASS" | \
  python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action register_family --target 'family:*' --campaign "$CAMP" \
    --reason "overnight grind: agent may open directions unattended" \
    --max-uses 40 --expires-minutes 1400 \
    --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_family.json

printf '%s\n' "$PASS" | \
  python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action resume_stalled_campaign --target '*' --campaign "$CAMP" \
    --reason "overnight grind: agent may clear its own stall brake" \
    --max-uses 5 --expires-minutes 1400 \
    --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_stall.json
```

All three expire in about **23 hours** — roughly 9pm tomorrow, past your 8pm
code freeze.

---

## 3. One more thing, 10 seconds

Three files are sitting uncommitted. They are the bytes the LOCK actually pins,
installed by you during the ceremony. I am blocked from staging them on purpose
(the guard refuses to let the agent commit the enforcer files). So they need you:

```bash
git add .claude/hooks/guard_bash.py .claude/settings.json Project/harness/runner.py
git commit -m 'LOCK: install the guard, settings and runner shim the lock pins'
```

Without this, the repo's history never shows what was actually being enforced
during the grind. That is an evidence gap in the report, not a safety problem.

---

## What you are handing over

You are pre-approving **directions** instead of reviewing each one. Say it plainly
in the report; it is a real change to the control story.

**Still enforced, without you:**

- 60 attempts total. Hard cap.
- 3 failures closes a direction. Mechanical, no override.
- 3% minimum improvement on shape 1, derived from tonight's measured noise floor.
- Every run needs a one-use permit bound to a file hash.
- Every result gets a blind audit before it can be promoted.

**No longer enforced:** you reading my reasoning before GPU time is spent. You
read it in `Project/memory/DECISIONS.md` in the morning instead. I will write up
every direction with why I opened it.

**What I deliberately did NOT ask for:** `resolve_integrity_verdict`. That is the
signature that clears an audit finding of misconduct. If an audit flags my work at
3am, I stop and it waits for you. An agent that can dismiss its own audit findings
does not have an audit.

---

## Full sweep results, 30 Aug 22:00 SGT

**Lock:** active, valid, 29 protected files, `lock-b7848d4736461b971acd`.

**Campaign:** `CAMP-POSTLOCK` active. 3 permits issued, 3 consumed, 0 open.
0 of 60 attempts spent. 1 of 30 calibrations. No permit armed. No strikes.
No pending audit decisions, no pending judgment, no postmortem debt.

**Submission bytes:** re-verified. The submission is still an exact splice of the
official script with only the designated region replaced. `verified: true`.

**Audit backlog:** 49 pending, 0 active, 0 needing your attention. These are the
known inert pre-gate rows — the watcher walks past them in about a second because
they have no packet. They cost nothing and block nothing.

**Test suites: 11 of 12 green.** 500+ individual checks passed, including the
sandbox boundary (a real Triton kernel compiles and runs in the jail, max error
0.0) and the full guard allowlist.

### The one failure, and why it is not a problem

`integration_authority_test.py` fails at its third check and then aborts.

The cause: it expects `Project/harness/runner.py` to contain the old benchmark
runner's internals. During the lock ceremony you replaced that file with the
27-line shim that just forwards everything to the trusted controller — which is
correct and is what makes the permit boundary unavoidable. The test was written
before that swap and nobody updated it.

The two checks it did complete are the ones that matter, and both passed: the
official benchmark publishes a complete timing protocol, and the controller's
timing protocol is that same protocol. So the timing agreement is confirmed.

**But be honest about the cost:** the suite aborts there, so everything after that
section never ran. I do not know whether the rest of it passes. I cannot fix it —
`Project/tools/` is write-denied to the agent, correctly. It is a stale test, not
a broken system, but "13 suites green" is no longer a true sentence and should not
appear in the report. Say "11 of 12, with one suite stale against the post-LOCK
runner shim" instead.

Three further test files exist that the guard cannot run at all
(`auditor_backend_test.py`, `authority_vocabulary_test.py`,
`incumbent_floor_test.py`) — they were written after the guard was staged, so they
are not on its allowlist. Owner-run only.

---

## What I will do while you sleep

1. One free measurement to get the memory-traffic number shape 1 is missing.
2. Open a direction on shape 1 aimed at the pointwise/normalization kernels —
   that is where 49% of the time goes, per tonight's profile.
3. Grind attempts against it.
4. When it dies, open the next one and keep going.
5. Write up every direction and every result as it happens.

## What will stop me

- An audit finding misconduct. I stop, you decide.
- All 60 attempts spent.
- You saying stop.
