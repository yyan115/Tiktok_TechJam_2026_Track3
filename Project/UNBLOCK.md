# UNBLOCK — remove the shell guard, keep the LOCK on the benchmark

Copy-paste in order. Stop at any step whose output does not match.

Everything here removes the **Bash allowlist** only. `torch_transformer_benchmark.py`,
`Project/results/`, the runner, the controller and the gate stay hash-pinned, so
the claim "the agent cannot touch the benchmark or the results" stays true.

---

## Step 0 — safety point

```bash
cd /home/admin/Desktop/Repos/Tiktok_TechJam_2026_Track3
git add -A
git commit -m "safety point before removing the shell guard"
git rev-parse HEAD
```

Write that hash down. `git reset --hard <hash>` returns here if anything goes wrong.

---

## Step 1 — rebuild the lock without the four guard files

```bash
export KEYS=~/techjam-keys

python3 Project/tools/owner_lock_ceremony.py build-lock \
  --exclude .claude/hooks/guard_bash.py \
  --exclude .claude/settings.json \
  --exclude Project/lock_staging/guard_bash.py \
  --exclude Project/lock_staging/settings.json \
  --yes
```

**Expect:** `protected files = 25`, a new `lock_id`, `wrote Project/authority/LOCK.json`.

**STOP if it says any number other than 25.** Paste the output.

---

## Step 2 — sign it

```bash
python3 Project/tools/owner_lock_ceremony.py sign-lock --private-key "$KEYS/owner_private_key.pem"
```

**Expect:** `verify_lock() says valid=True` and `25 (all hashes match)`.

---

## Step 3 — verify

```bash
python3 Project/tools/owner_lock_ceremony.py verify
```

**Expect:** every file `match`, `verify_lock(): PASS`, and a final `PASS` line.

---

## Step 4 — mint the rotation slip

A previous activation exists under a different `lock_id`, so this is
**`lock.rotate`**, not `lock.activate`.

```bash
python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action lock.rotate \
    --campaign lock-2026-08-31 \
    --reason "drop the shell guard from the lock so the agent can clean the repo" \
    --private-key "$KEYS/owner_private_key.pem" \
    --key-dir "$KEYS"
```

**Expect:** it prints the exact next command including the capability file path.

---

## Step 5 — spend it

Run the command step 4 printed. It looks like:

```bash
python3 Project/harness/trusted_controller.py activate-lock \
  --capability "$KEYS/capabilities/capability-<id>.json"
```

**Expect:** JSON containing `"active":true`.

---

## Step 6 — confirm the lock is healthy at 25

```bash
python3 Project/harness/trusted_controller.py verify-lock
```

**Expect:** `"valid":true`, `"active":true`, `"protected_file_count":25`.

**STOP if valid is false.** Nothing below matters if the lock is broken, and the
controller will refuse every measurement run.

---

## Step 7 — remove the hook

`.claude/settings.json` is now unpinned. Open it and delete the hook entry that
points at `guard_bash.py`. It is inside a `"hooks"` block, under `PreToolUse`,
matching the `Bash` tool. Delete that one matcher entry, leave the rest of the
file alone.

I cannot read `.claude/` from my shell, so I cannot quote the exact lines. If
the structure is unclear, paste the `hooks` block to me and I will tell you
exactly what to remove.

Then delete the guard itself:

```bash
rm .claude/hooks/guard_bash.py
```

**Restart Claude Code** so the hook change takes effect.

---

## Step 8 — confirm I am unblocked

Ask me to run:

```bash
ls -la Project
```

If it returns a listing instead of `LOCK allowlist: command shape is not on the
post-LOCK allowlist`, it worked.

---

## What I do next, without asking

1. Delete the ten dead files listed in `Project/STATUS.md` section 7
2. Delete `Project/kernels/__pycache__/`
3. Move superseded planning documents into `Project/archive/`
4. Fact-check the remaining claims in the four judge-facing documents

Then, in order of what actually affects the score:

5. Shape 14 correctness run — the only shape never executed on this build, and
   I changed code on its path today
6. Port base-2 softmax and the causal loop split into `_sub_attn_fwd`, the
   attention kernel shape 14 uses and the one that never received that work.
   Shape 14 is 1,391,250 GFLOP, 99.89% of all arithmetic in the benchmark, and
   94% of it is attention. At sequence 100,000 the causal split removes the mask
   from roughly 99.9% of the inner loop instead of the modest fraction it removed
   at sequence 1024.
7. Re-measure shape 14
8. Update all four documents to the final board

---

## If you want to abort

```bash
git reset --hard <the hash from step 0>
```

The lock in `Project/authority/` is not in git, so also re-run steps 1 to 5
without the `--exclude` flags to put the 29-file lock back.
