# OWNER LOCK — the ceremony you run yourself

This is the half of the control system that the agent is not allowed to touch.

Everything the harness does after this point is gated on a signature that only
you can produce. That only works if the signing key never comes within reach of
the agent. So you run these commands, at your own terminal, in a normal shell —
**not** inside a Claude session, and not by asking the agent to run them for you.

It takes about five minutes.

---

## Why this has to be you

The agent runs as your operating-system user. It can read files, write files,
and run commands. The only thing it cannot do is produce your signature — as
long as your private key is somewhere it never looks.

If the agent could sign its own authorisation, the entire authority system would
be theatre: it could grant itself permission to run anything, promote anything,
and clear any audit finding. That is the one thing this ceremony prevents.

So: **the private key never goes inside the repository, and you never paste it
into a chat.** The tool refuses to write it into the repo at all, and it never
prints it.

---

## What you will have when you finish

| File | What it is |
| --- | --- |
| `~/…/owner_private_key.pem` (outside the repo) | your signing key — the whole system rests on this |
| `~/…/critic_private_key.pem` (outside the repo) | the independent-reviewer key, kept separate on purpose |
| `Project/authority/owner_public_key.pem` | public half; lets the harness check your signature |
| `Project/authority/critic_public_key.pem` | public half of the critic key |
| `Project/authority/LOCK.json` | the list of files that are now frozen, with their hashes |
| `Project/authority/LOCK.sig` | your signature over that list |
| `Project/authority/rules_snapshot.json` | per-document hashes of the rule documents, so `verify` can name which one drifted (advisory, never enforced) |
| one line in `Project/authority/events.jsonl` | the record that you switched the lock on |

---

## Before you start

Open a terminal and go to the repository:

```bash
cd /home/admin/Desktop/Repos/Tiktok_TechJam_2026_Track3
```

Decide where the private keys live. Best is removable media or a separate
volume, because the agent cannot read a USB stick that is not plugged in:

```bash
export KEYS=/run/media/$USER/<your-stick>/techjam-owner-keys
```

If you have no stick right now, you can use your home directory, but understand
the trade: the agent runs as the same user, so only the guard rules — not the
operating system — stand between it and the key file. The tool makes you say so
explicitly:

```bash
export KEYS=$HOME/.techjam-owner-keys      # then add --allow-home-key in step 2
```

---

## Step 1 — put the reviewed files in place

The lock records file hashes **as they are at the moment you build it**. So the
new guard, the new settings, the controller shim and the profile worker must go
in *first*. If you install them afterwards, the lock breaks instantly and the
harness refuses everything.

Back up what is there now, then copy the three staged files that exist today
into place (the fourth is below):

```bash
cp .claude/settings.json ~/settings.json.before-lock
cp .claude/hooks/guard_bash.py ~/guard_bash.py.before-lock
cp Project/harness/runner.py ~/runner.py.before-lock

cp Project/lock_staging/guard_bash.py .claude/hooks/guard_bash.py
cp Project/lock_staging/settings.json .claude/settings.json
cp Project/lock_staging/runner.py Project/harness/runner.py
```

What each one changes, in one line:

- **guard_bash.py** — swaps the old "block a list of bad commands" guard for a
  deny-by-default allowlist that fails closed on anything it cannot parse.
- **settings.json** — the matching deny rules, plus the hooks that verify the
  lock at session start and reconcile audits after every tool call.
- **runner.py** — the old runner used to load candidate code into its own
  process. The replacement is a four-line shim that hands everything to the
  trusted controller, so a one-use permit and the sandbox become unavoidable.

The third one is a real decision, not a formality: after this, any benchmark run
without a permit is refused. That is the point. The old runner is still in git
history and in your backup copy if you ever need to look at it.

### The fourth file — and today it is the thing blocking LOCK

```bash
cp Project/lock_staging/profile_worker.py Project/harness/profile_worker.py
```

`Project/harness/profile_worker.py` is the in-jail worker for the controller's
diagnostic (profiling) lane. **It does not exist yet**, so as of right now
`build-lock` stops before it prints anything:

```
FAILED: these protected files are missing or are not regular files:
        Project/harness/profile_worker.py   [control-plane]   <-- CONTROL PLANE
        Put the file on disk and run build-lock again.
```

Do not `--exclude` it to get past that — the tool explains why in the same
message: the lock would sign cleanly while that file stayed unprotected. The
agent cannot put it there either (`.claude/settings.json` denies
`Write(Project/harness/**)`), so it has to be staged like the other three and
copied in by you. If `Project/lock_staging/profile_worker.py` is not there when
you start, LOCK is blocked and the agent owes you that file first.

---

## Step 2 — make the keys

```bash
python3 Project/tools/owner_lock_ceremony.py keygen --key-dir "$KEYS"
```

Add `--allow-home-key` if `$KEYS` is inside your home directory.
Add `--no-passphrase` if you would rather not be asked for one.

You will be asked for a passphrase twice (press Enter twice for none). Use one
if you can — it means a copied key file is not immediately usable.

**On screen you get:** where each key went, the two public keys being installed
into the repo, and a boxed warning with two long hex fingerprints.

**Do this now, before you go on:** write the *owner* fingerprint down somewhere
off this machine — a phone note, a piece of paper. It is your only way to prove
later that nobody swapped your identity out. See "The honest warning" below.

Then back the keys up:

```bash
cp "$KEYS"/*_private_key.pem /path/to/somewhere/offline/
```

If you lose these keys you can never re-sign or rotate the lock, and the harness
stays shut. If someone else gets them, they own the whole control plane.

---

## Step 3 — look at what is about to be frozen

```bash
python3 Project/tools/owner_lock_ceremony.py build-lock
```

This writes nothing. It prints every file that will be pinned, one line of
reasoning for each, then the list of things deliberately left out and what
protects those instead. It also checks that step 1 actually happened.

**Expect:** 29 files across six groups — the control-plane code (7), the gate and
audit code and schemas (9), the official benchmark files and shape definitions
(4), the tools that assemble what ships (4), the staged guard files (3) and the
live guard files they were copied to (2). Then `TOTAL: 29 files will be
hash-pinned by this lock.` and `DRY RUN — nothing was written`.

If it stops with *"staged bytes are not installed yet"*, go back and finish
step 1. That refusal is the tool stopping you from building a lock that would
break the moment you fixed it.

Read the list. If you want something added or removed:

```bash
# add a file
python3 Project/tools/owner_lock_ceremony.py build-lock --extra Project/tools/dashboard.py
# drop a file
python3 Project/tools/owner_lock_ceremony.py build-lock --exclude Project/tools/shape14_eval.py
# or take whole groups: control-plane, gate-and-audit, benchmark-truth,
# evidence-tools, staging, enforcement
python3 Project/tools/owner_lock_ceremony.py build-lock --group control-plane --group staging
```

---

## Step 4 — build it for real

Same command, with `--yes` on the end:

```bash
python3 Project/tools/owner_lock_ceremony.py build-lock --yes
```

**Expect:** `wrote Project/authority/LOCK.json`, a random `lock_id`, an `epoch`
label, and `protected files     = 29`.

At this point the lock exists but is unsigned, which means it is worth nothing.
The tool says so.

---

## Step 5 — sign it

```bash
python3 Project/tools/owner_lock_ceremony.py sign-lock --private-key "$KEYS/owner_private_key.pem"
```

**Expect:** it confirms the key, the installed public key and `LOCK.json` all
agree, writes `LOCK.sig`, then immediately re-verifies and prints
`verify_lock() says valid=True` with `29 (all hashes match)`.

If the key does not match the lock, it stops and writes nothing. It will not
produce a signature that would fail later.

---

## Step 6 — check it end to end

```bash
python3 Project/tools/owner_lock_ceremony.py verify --expect-owner-key-sha256 <the fingerprint you wrote down>
```

**Expect:** every protected file listed with `match`, `verify_lock(): PASS`, and
at the bottom `PASS — the lock is signed, intact, and every protected byte
matches.`

It will also say `epoch activated : False`. That is right — you have not turned
it on yet.

---

## Step 7 — turn it on

Two commands. The first makes a single-use, twenty-minute permission slip signed
by you. The second spends it.

```bash
python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action lock.activate \
    --campaign lock-2026-08-30 \
    --reason "initial owner activation after the FIX phase" \
    --private-key "$KEYS/owner_private_key.pem" \
    --key-dir "$KEYS"
```

**Expect:** it works out the target from `LOCK.json` itself, confirms the
authority log is empty (required for a first activation), proves the controller
will accept the slip, and prints the exact next command including the file path.
Run that command:

```bash
python3 Project/harness/trusted_controller.py activate-lock --capability "$KEYS/capabilities/capability-<id>.json"
```

**Expect:** a line of JSON containing `"active":true`. Then delete the spent
slip:

```bash
rm "$KEYS/capabilities/capability-<id>.json"
```

---

## Step 8 — confirm, and expect it to open closed

```bash
python3 Project/tools/owner_lock_ceremony.py verify --expect-owner-key-sha256 <your fingerprint>
python3 Project/harness/trusted_controller.py status
```

**Expect:** `epoch activated : True`, and a status line showing the lock as
valid and active.

**Now the part that will look like a bug and is not.** The gate is expected to
open CLOSED:

- There are **16 uncleared RULE_VIOLATION rows** in the audit ledger that post-
  date the cut-off. Under the new rules an unresolved violation blocks, and a
  *missing* verdict blocks too — absence is no longer free. So work that used to
  flow will stop.
- **Every historical measurement is unbound under the new authority.** None of
  them was taken under a permit, none has an authority receipt, and none is
  bound to a signed capability. The harness cannot retroactively vouch for them
  and does not pretend to.

Nothing is broken. This is the system telling you the truth about what it can
and cannot currently stand behind. What follows is one deliberate reconciliation
pass: work through the blocked rows one at a time and either clear them with a
real re-audit or record them as accepted residuals. Do not fix it by loosening
the gate.

---

## Afterwards: the two commands you will actually use

**Let the agent run one experiment.** The agent produces a request file; you
sign a slip for that shape:

```bash
python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action permit.issue --target shape:7 \
    --campaign <campaign id from the request> \
    --reason "shape 7 optimisation attempt 3" \
    --private-key "$KEYS/owner_private_key.pem" --key-dir "$KEYS"

python3 Project/harness/trusted_controller.py issue-permit \
    --request Project/loop/requests/<request>.json \
    --capability "$KEYS/capabilities/capability-<id>.json"
```

**Clear one audit finding.** Same shape, different action:

```bash
python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action audit.resolve --target audit:<entry id> \
    --campaign <campaign id> \
    --reason "re-audited on new bytes, integrity PASS" \
    --private-key "$KEYS/owner_private_key.pem" --key-dir "$KEYS"
```

Every slip is single-use, expires in twenty minutes by default, and names one
action against one target. The reason you type is saved next to it in a
`.provenance.json` file, because the signed slip itself has a closed format with
no room for prose.

---

## If you need to change a protected file later

You cannot edit a locked file and carry on — the harness will refuse everything
until the hashes match again. That is deliberate. The path is:

1. make the change,
2. `build-lock --yes` (this makes a **new** lock id and deletes the stale
   signature),
3. `sign-lock`,
4. `mint-capability --action lock.rotate …`,
5. `trusted_controller.py activate-lock --capability …`.

The old lock cannot be rolled back: restoring the old signed files leaves the
epoch inactive, and the controller refuses it.

---

## The honest warning

Two things you should know, stated plainly rather than buried.

**One.** The private key protects against the agent *forging* authorisation. It
does not protect against the agent *replacing* your identity. Anything running
as your user can overwrite `Project/authority/owner_public_key.pem` with a key
of its own, re-sign `LOCK.json` with it, and rotate the lock to itself. The
harness would verify all of it happily, because it only checks that the lock is
internally consistent — nothing pins it to *your* key across epochs. I confirmed
this against the real code.

The defences available to you today:

- **Detection, always available.** The fingerprint you wrote down in step 2.
  Run `verify --expect-owner-key-sha256 <fingerprint>` and a substitution is
  caught immediately. Make this a habit, and do it before you trust any result.
- **Prevention, optional and fiddly.** Hand the four owner files to root so your
  user cannot rewrite them:

  ```bash
  # do this AFTER step 7, so events.jsonl already exists
  mkdir -p Project/authority/blobs Project/authority/receipts
  sudo chown root:root Project/authority/owner_public_key.pem \
      Project/authority/critic_public_key.pem \
      Project/authority/LOCK.json Project/authority/LOCK.sig
  sudo chown root:root Project/authority
  ```

  Locking the *directory* is the part that actually helps — without it your user
  can simply delete a root-owned file and write a new one in its place. It only
  works because everything the harness creates from here on lands in the
  `blobs/` and `receipts/` subdirectories, which stay yours, and because
  `events.jsonl` and `.authority.lock` already exist by then. Try one permit
  afterwards. If the controller starts refusing, `sudo chown -R $USER
  Project/authority` and fall back to the fingerprint check.

**Two.** Run this ceremony yourself. If you hand these commands to the agent to
execute, the key passes through the agent's session and everything above stops
being true. There is no partial version of this.

---

## Quick reference

```bash
cd /home/admin/Desktop/Repos/Tiktok_TechJam_2026_Track3
export KEYS=/run/media/$USER/<stick>/techjam-owner-keys

# 1. install the reviewed files (see step 1 for the backup lines first)
cp Project/lock_staging/guard_bash.py     .claude/hooks/guard_bash.py
cp Project/lock_staging/settings.json     .claude/settings.json
cp Project/lock_staging/runner.py         Project/harness/runner.py
cp Project/lock_staging/profile_worker.py Project/harness/profile_worker.py

# 2. keys — write the owner fingerprint down, then back the keys up offline
python3 Project/tools/owner_lock_ceremony.py keygen --key-dir "$KEYS"

# 3. review, 4. build, 5. sign, 6. check
python3 Project/tools/owner_lock_ceremony.py build-lock
python3 Project/tools/owner_lock_ceremony.py build-lock --yes
python3 Project/tools/owner_lock_ceremony.py sign-lock --private-key "$KEYS/owner_private_key.pem"
python3 Project/tools/owner_lock_ceremony.py verify --expect-owner-key-sha256 <fingerprint>

# 7. switch it on
python3 Project/tools/owner_lock_ceremony.py mint-capability --action lock.activate \
    --campaign lock-2026-08-30 --reason "initial owner activation" \
    --private-key "$KEYS/owner_private_key.pem" --key-dir "$KEYS"
python3 Project/harness/trusted_controller.py activate-lock --capability "$KEYS/capabilities/<id>.json"

# 8. confirm — and expect blocked work, not a clean board
python3 Project/tools/owner_lock_ceremony.py verify --expect-owner-key-sha256 <fingerprint>
python3 Project/harness/trusted_controller.py status
```

Exit codes from `owner_lock_ceremony.py`: `0` fine, `2` stop and read the
message, `3` fine but something drifted (it will say what).
