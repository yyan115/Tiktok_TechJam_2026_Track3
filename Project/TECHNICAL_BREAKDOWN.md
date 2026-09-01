# Technical breakdown — how the harness actually works

Everything below was read out of the source, and each section names the file and
line so you can check it. Nothing here is described from memory.

The short version: **the agent that writes the kernels can read every file in the
project, but it cannot change the ones that decide whether it won, and it cannot
run a measurement without a permission slip that only a human can produce.**

---

## 1. What "signing" actually means here

Signing is not encryption and it does not hide anything. It answers one question:
**did the person holding the private key approve these exact bytes?**

There are two keypairs.

| Key | Private half | Public half | Used for |
|---|---|---|---|
| **Owner** | your machine, never in the repo | `Project/authority/owner_public_key.pem` | locking files, granting run budgets |
| **Critic** | separate | `Project/authority/critic_public_key.pem` | reopening a closed direction |

The public halves are committed on purpose. A public key can **only check** a
signature; it can never create one. So anyone who clones this repo — a judge, the
agent, you — can verify every signature in it, and none of them can forge a new
one.

`authority.py:662` even enforces where the private key may live: a helper called
`private_key_path_is_external` refuses to proceed if the private key sits inside
the repo or inside `$HOME`.

**Roles are structurally separated.** `authority.py:412-415` refuses an owner
capability that tries to perform `technical.review`, and refuses a critic
capability that tries to do anything else. The owner cannot act as their own
critic even if they wanted to.

---

## 2. The LOCK — a signed list of files

`Project/authority/LOCK.json` is a plain JSON document with exactly eight fields
(`lock_manifest.py:33`), the important one being `protected_files`: a map of
**29 file paths to their SHA-256 hashes**.

`Project/authority/LOCK.sig` is the detached Ed25519 signature over that
document.

### What verification does, in order

`lock_manifest.py:111-165`, every time:

1. Read `LOCK.json` and refuse if it has any unexpected or missing field.
2. Load both public keys, and check their fingerprints match what `LOCK.json`
   claims — so a swapped key file is caught.
3. Decode the base64 signature and verify it against `canonical_json(document)`.
4. **Re-hash all 29 files on disk** and compare each to its recorded hash.
5. Collect every mismatch and refuse with the full list.

Two details that matter. `canonical_json` (`authority.py:78`) serialises with
sorted keys, no whitespace, and `allow_nan=False`, so the bytes being signed are
never ambiguous. And a protected path that is a **symlink** is treated as
`missing-or-not-regular` (`lock_manifest.py:148`) — you cannot point a protected
path at different content.

### Why it cannot be worked around

| Attempt | Result |
|---|---|
| Edit a protected file | Hash no longer matches `LOCK.json` → refused |
| Also edit `LOCK.json` to match | Signature no longer verifies → refused |
| Re-sign `LOCK.json` | Needs the private key, which is not on the machine |
| Replace the public key with your own | Fingerprint no longer matches the document → refused |
| Symlink a protected path elsewhere | Rejected as not a regular file |

There is no order you can perform these in that succeeds.

### Being locked is not enough

`trusted_controller.py:1538-1563` shows the second half. `verify_lock` proves the
bytes are signed and intact; `verify_lock_only` then walks the event journal for
a `lock_activated` or `lock_rotated` event and checks it names **this** lock id,
**this** manifest hash, and was itself authorised by a consumed owner capability.
A validly signed lock that was never activated returns `active: false`, and
`require_lock` refuses to run against it.

---

## 3. Capabilities — the run budget

A capability is a small signed document (`authority.py:49-60`, eleven required
fields) that says: *this role may perform these actions, on these targets, until
this time, at most `max_uses` times.*

Verification (`authority.py:392-459`):

- exact field set, or refuse
- role must be `owner` or `critic`, with the separation rule above
- the requested action must appear in `actions`
- the requested target must match `targets` (exact, `prefix:*`, or `*`)
- `max_uses` must be a positive integer
- `expires_at` must be after `issued_at`; not valid more than 5 minutes early;
  refused once expired
- the signature is checked over the document **with the `signature` field
  removed**

**How the budget is spent.** There is no counter file to tamper with. The count
is derived by walking the journal (`authority.py:462-468`) and counting events
whose payload carries this `capability_id` with `capability_consumed: true`. If
that count has reached `max_uses`, `append_authorized` refuses
(`authority.py:490`).

Every individual use also gets its own derived receipt nonce — the SHA-256 of the
capability id, the signed base nonce, the use number, the action and the target
(`authority.py:494-504`). So even a capability deliberately granting several uses
produces a distinct, unforgeable receipt per use.

---

## 4. Permits — one per measurement

A capability authorises N runs. A **permit** authorises exactly one, and is
welded to one specific file.

The request has exactly eight fields (`authority.py:537-545`) and is checked hard
(`authority.py:546-572`):

- `request_sha256` must equal the hash of the request blob actually stored —
  the request cannot describe one thing and be recorded as another
- `mode` must be one of eight known lanes: `diagnostic`, `calibration`,
  `screening`, `correctness`, `optimization`, `confirmation`, `shape6`, `shape14`
- `shape_id` must be an integer **1 to 14** — there is no way to request a shape
  that is not one of the official fourteen
- a `calibration` permit **may not carry candidate bytes at all**
- every other mode **requires** a 64-character lowercase-hex candidate SHA-256
- a `diagnostic` permit cannot spend a mechanism family
- expiry must be within the next 12 hours

Two authority flags are set at issue time, not claimed later:

- `may_modify_candidate` — false for `diagnostic` and `calibration`
- `may_promote` — **true only for `optimization`**. Screening, correctness and
  confirmation runs can produce evidence but can never crown a champion.

**Consuming a permit** (`authority.py:597-639`) requires, under a lock:
exactly one matching `permit_issued` event, **no** existing `permit_consumed`
event for it, matching mode and shape, matching candidate hash, and an unexpired
expiry. Replay is impossible because the second attempt sees the first
consumption in the journal.

Consumed permits are kept in `Project/loop/permits_used/`. That is where the
**272 issued / 271 consumed** figure comes from — nothing is deleted.

---

## 5. The journal — an append-only hash chain

`Project/authority/events.jsonl` is one JSON object per line. Each event carries
`prev_event_sha256` and `event_sha256` (`authority.py:41-48`).

Reading it (`authority.py:237-284`) validates **every line, every time**:

- no blank lines
- each line parses, and has exactly the expected key set
- `prev_event_sha256` equals the previous line's `event_sha256`
- `event_sha256` equals the SHA-256 of the canonical JSON of the event with
  `event_sha256` removed

So editing any past line breaks that line's own hash, and breaks the `prev`
pointer of every line after it. There is no quiet edit.

Writes (`authority.py:315-330`) open with `O_APPEND | O_NOFOLLOW | O_CLOEXEC`,
loop until the full line is written, `fsync` the file, then `fsync` the
directory. A crash mid-campaign cannot leave a half-written event.

**Blobs.** `store_blob` (`authority.py:338`) writes evidence under a filename
that *is* its own SHA-256. A blob whose bytes stop matching its name is
self-evidently corrupt, which is why the board can cite a packet hash and mean it.

---

## 6. The sandbox

`sandbox.py:197-199` — bubblewrap is mandatory. The comment is explicit:
*"unsandboxed fallback is forbidden."* If `bwrap` is missing the run fails; it
does not silently degrade.

The jail (`sandbox.py:208-237`):

| Flag | Effect |
|---|---|
| `--unshare-user/pid/ipc/uts` | separate user, process, IPC and hostname namespaces |
| `--unshare-net` *or* a seccomp filter | no network |
| `--cap-drop ALL` | no capabilities |
| `--clearenv` | no inherited environment |
| `--die-with-parent`, `--new-session` | cannot outlive or escape its parent |

**No home directory is mounted** (`sandbox.py:239`). The worker, the candidate,
the official benchmark, the shapes file and the request are all bound
**read-only**. Exactly one directory is writable: `/output`.

So the code being measured cannot reach the network, cannot read your home
directory, cannot modify its own source, and cannot touch the authority store.

---

## 7. What the worker actually measures

`candidate_worker.py:308` onwards.

**Timing.** 20 warm-up iterations per side, then `rounds × repeats` =
3 × 100 = **300 samples per side**, and the reported figure is the **median**.
No best-of, no dropped rounds.

**The rounds alternate order** (`candidate_worker.py:410-417`): even rounds run
baseline then candidate, odd rounds run candidate then baseline. Neither side can
systematically benefit from being measured first.

**Three tripwires run alongside:**

1. **Wall-clock cross-check** (`candidate_worker.py:423-431`). The speedup is
   computed twice — once from CUDA events, once from wall-clock — and an
   `agreement_ratio` compares them. Work hidden on a side stream shows up as
   disagreement between the two clocks.

2. **Same-address, fresh-values** (`candidate_worker.py:433-441`). A kernel that
   caches its output keyed on the input's memory address would return a stale
   answer when the buffer is reused — which is exactly what a timing loop does.
   This check reuses the address with new values and compares. It is
   deliberately **untimed**, so it is a correctness challenge and not a
   performance measurement.

3. **Baseline invariance** (`candidate_worker.py:400-402`). The baseline's own
   output is probed before and after, with `torch.equal`. If the candidate
   monkeypatched the reference to make itself match, this fails.

Those three exist because each corresponds to a published cheat. Two of them have
matching red-team attacks in `Project/harness/redteam/` that were run through the
harness and caught.

---

## 8. The gate — how an attempt is earned

`Project/tools/run_gate.py`. The gate starts closed.

### Research (`run_gate.py:1588-1638`)

Refused unless all of:

- no attempt is currently armed or in flight
- `--index-hash` equals the **current** hash of `Project/research/INDEX.md`,
  which proves the index was read this cycle rather than at some point in the past
- at least **two** cited notes that actually exist on disk
- a summary of at least **200 characters**
- if a direction was just closed: a postmortem of at least 200 characters that
  also contains a stated **case for revival**, even if that case is "none —
  disproven"

### Plan (`run_gate.py:1700-1755`)

Refused unless research happened this cycle. Requires a hypothesis, a **numeric**
prediction, kill criteria, reasoning, and citations.

Citations are the interesting part (`run_gate.py:1194-1211`). They are written as
`file:line` or `file:line-line`, and the gate:

1. resolves the path, refusing if the file does not exist
2. checks the line range is actually within the file
3. **reads those exact lines and copies the quoted text into the log**

A citation to a file that does not exist, or to lines that do not exist, is
refused. A citation that exists but does not say what was claimed is now quoted
verbatim in the log where anyone can read it.

### One run, then the gate shuts

The moment a run executes, the watcher calls `reconcile`: the gate closes, the
attempt is counted, and **whether it improved is read from the referee's own
journal row, never from the agent's claim.**

### Strikes (`run_gate.py:81`, `run_gate.py:23-38`)

`MAX_STRIKES = 3`.

- **Improvement** means beating the group's best speedup by **≥3%** — the
  promotion margin floor. Noise cannot reset the counter.
- In screening mode, the attempt's own **declared prediction range** decides
  hit or miss. A miss is a strike.
- Diagnostic work and infrastructure failures do **not** strike. Calibration and
  confirmation never strike or improve.
- Three strikes closes the direction and registers postmortem debt.

**A closed direction reopens only via a one-use critic receipt bound to the
closure nonce. The source says it plainly: "There is no unlock command."** Not
even the owner can simply switch it back on.

### Families — you cannot rename an old idea

Family ids come only from the trusted catalog or a controller-verified admission
event. **Candidate prose cannot create or rename a family.** Variants inherit the
family they came from unless an externally authorised novelty resolution admits a
child. This is what stops three strikes being escaped by relabelling.

---

## 9. The audit

`Project/tools/audit_authority.py:47-53` defines a fixed vocabulary. A verdict is
a typed value, not prose:

| Class | Values | Effect |
|---|---|---|
| Hard integrity | `RETEST`, `RULE_VIOLATION` | latches; only an authenticated resolution clears it |
| Blocking integrity | the above plus `NEEDS_CONTEXT` | blocks promotion |
| Blocking technical | `WEAK_DIAGNOSIS`, `MISSING_EVIDENCE` | blocks on the technical channel |
| Advisory technical | `TECHNICAL_DISAGREEMENT` | recorded, does not block |

`MAX_FAILED_ATTEMPTS = 3`, and audits run in three lanes: `primary`, `shape6`,
`shape14`.

A verdict only counts if it is bound to the packet hash, the candidate hash, the
attempt nonce and the entry id. A bare "PASS" is rejected; so is a correct-looking
verdict carrying the wrong nonce.

**This is the part that did not run.** The verdict schema uses `allOf`, which
OpenAI's structured-output mode rejects, so the request fails before the model
reads the packet. The machinery around it worked — the failure was written to the
hash-chained audit journal and escalated — but **no verdict is bound to any row of
the published board**, and the README says so.

---

## 10. What it adds up to

| To do this | You would need |
|---|---|
| Change the benchmark, harness or shapes | the owner's private key |
| Run a measurement at all | a permit, from a signed capability |
| Run the same candidate twice for a better number | a second permit; the first is recorded as consumed |
| Benchmark an easier problem | a shape id outside 1–14, which is refused |
| Edit a past result | to recompute every subsequent hash in the chain |
| Claim an improvement that did not happen | to beat the referee's own journal row by ≥3% |
| Cite research that does not support the plan | the quoted lines are copied into the log |
| Escape a closed direction by renaming it | a family id, which prose cannot mint |
| Reopen a closed direction | a one-use critic receipt; there is no unlock command |

None of that requires trusting the agent, and none of it requires trusting us.
Every check above runs from files in this repository, against a public key in this
repository, and can be reproduced by:

```bash
python3 Project/harness/trusted_controller.py verify-lock
python3 Project/tools/build_submission.py --check-only
```
