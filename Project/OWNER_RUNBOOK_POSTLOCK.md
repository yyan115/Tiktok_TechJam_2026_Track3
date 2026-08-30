# RUN THIS — lock the system and get to the first measured card

Every command below was run end to end on a full copy of this repo, on this GPU,
on 30 Aug. Where something refused the first time, the fix is already in.

Run it top to bottom. Two commands need `sudo` and will ask for your login
password. Everything else is copy-paste.

**Passphrase for the signing keys — used in steps 2, 4, 5, 7, 9:**

```
Dkkx7pGiVNvp4CyYXtYKd4wBxzpD
```

It is already piped into every command that needs it, so nothing prompts.

Open one terminal and paste this first. Every later step assumes it:

```bash
cd /home/admin/Desktop/Repos/Tiktok_TechJam_2026_Track3
export KEYS=~/techjam-keys
export CAMP=CAMP-POSTLOCK
export PASS=Dkkx7pGiVNvp4CyYXtYKd4wBxzpD
mkdir -p "$KEYS" && chmod 700 "$KEYS"
```

---

## 1. Install the auditor  *(sudo)*

Codex quota is out, so audits run on Claude. The auditor binary has to sit
somewhere the agent cannot overwrite, or the pinned-hash check is meaningless —
the stock install is under your home directory and is refused on purpose.

```bash
sudo cp -L "$(readlink -f "$(command -v claude)")" /usr/local/bin/claude-auditor
sudo chown root:root /usr/local/bin/claude-auditor
sudo chmod 755 /usr/local/bin/claude-auditor
sha256sum /usr/local/bin/claude-auditor
```

Expected hash:

```
fd5f10ff0eb58daec04900466b143ea98aab50abf208a422bc008eaec13f61f7
```

If it differs, your `claude` was updated since 30 Aug. Put the new hash into
`PINNED_CLAUDE_SHA256` in `Project/tools/audit_champion.py` — **before** step 3,
because that file gets frozen by the lock.

Check it:

```bash
python3 Project/tools/tests/auditor_backend_test.py     # expect 33/33 ALL GREEN
```

---

## 2. Make the signing keys

```bash
printf '%s\n%s\n' "$PASS" "$PASS" | \
  python3 Project/tools/owner_lock_ceremony.py keygen --key-dir "$KEYS"
```

Prints two fingerprints. Save them somewhere off this machine.

---

## 3. Install staged bytes, then build the lock

Order matters. The lock records hashes as they are right now — install these
afterwards and the lock fails instantly and the controller refuses everything.

```bash
mkdir -p .claude/hooks
cp Project/lock_staging/guard_bash.py .claude/hooks/guard_bash.py
cp Project/lock_staging/settings.json .claude/settings.json
cp Project/lock_staging/runner.py     Project/harness/runner.py

python3 Project/tools/owner_lock_ceremony.py build-lock
```

That is a dry run. In its output, every line under "staged bytes: are they
installed yet?" must say installed. Then:

```bash
python3 Project/tools/owner_lock_ceremony.py build-lock --yes
```

Expect `protected files = 29`.

---

## 4. Sign the lock

```bash
printf '%s\n' "$PASS" | \
  python3 Project/tools/owner_lock_ceremony.py sign-lock \
    --private-key "$KEYS/owner_private_key.pem"
```

Expect `verify_lock() says valid=True`.

---

## 5. Activate the lock

```bash
printf '%s\n' "$PASS" | \
  python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action lock.activate --campaign "$CAMP" \
    --reason "activate the post-fix lock" \
    --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_activate.json

python3 Project/harness/trusted_controller.py activate-lock \
  --capability /tmp/cap_activate.json

python3 Project/harness/trusted_controller.py status
shred -u /tmp/cap_activate.json
```

`status` must show `"active": true`. Capabilities expire in 20 minutes — if one
lapses, mint it again.

---

## 6. Unfreeze the 16 old audit flags

Nothing can run until this is done. **Do not use `run_gate.py verdict-clear
--kind violation`** — it prints success and does not unfreeze anything.

```bash
printf '%s\n' "$PASS" | \
  python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action audit.resolve --target 'audit:*' --campaign "$CAMP" \
    --reason "retire pre-citation-gate rows before the post-LOCK re-measurement" \
    --max-uses 25 --expires-minutes 60 \
    --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_audit.json

python3 Project/tools/clear_pregate_verdicts.py \
  --capability /tmp/cap_audit.json --campaign "$CAMP"

python3 Project/tools/clear_pregate_verdicts.py \
  --capability /tmp/cap_audit.json --campaign "$CAMP" --yes

shred -u /tmp/cap_audit.json
```

Expect `16 retired. RULE_VIOLATIONs still braking permits: 0`.

---

## 7. Open the campaign

```bash
cat > /tmp/campaign.json <<'EOF'
{
 "schema_version": 1,
 "campaign_id": "CAMP-POSTLOCK",
 "max_total_attempts": 60,
 "max_calibrations_per_shape": 3,
 "max_total_calibrations": 30,
 "stall_window": 6,
 "timing_config": {"warmup": 20, "repeats": 100, "rounds": 3},
 "score_scenarios": ["geomean-shapes-1-13"]
}
EOF

SUBJ=$(python3 -c "import sys,json; sys.path.insert(0,'Project/tools'); import run_gate as g; print(g.sha_json(json.load(open('/tmp/campaign.json'))))")
echo "subject: $SUBJ"

printf '%s\n' "$PASS" | \
  python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action open_campaign --target "campaign:$CAMP" --campaign "$CAMP" \
    --reason "open the post-LOCK re-measurement campaign" \
    --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_camp.json

RCP=$(python3 Project/harness/trusted_controller.py authorize \
  --capability /tmp/cap_camp.json --action open_campaign \
  --target "campaign:$CAMP" --subject-sha256 "$SUBJ" --campaign "$CAMP" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['receipt_path'])")
echo "receipt: $RCP"

python3 Project/tools/run_gate.py campaign-open \
  --spec /tmp/campaign.json --authority-receipt "$RCP"
shred -u /tmp/cap_camp.json
```

Expect `Campaign CAMP-POSTLOCK opened under controller authority.`

---

## 8. Check the box is idle

Timed work on a busy box is worthless. Do **not** check with `pgrep -f`.

```bash
python3 Project/tools/champion_watch.py --dry-run | head -5
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

`"active": []` means clear. Desktop apps are fine; another benchmark or a
running audit is not — wait for it.

---

## 9. Mint the grind capability

Every run needs one of these, and they are single-use by default. This one
covers a whole session.

```bash
printf '%s\n' "$PASS" | \
  python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action permit.issue --target 'shape:*' --campaign "$CAMP" \
    --reason "grind session 30-31 Aug: shapes 1-13" \
    --max-uses 100 --expires-minutes 480 \
    --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_grind.json
```

Keep `/tmp/cap_grind.json` until the session ends, then `shred -u` it. It is a
broad capability — say so in the report rather than implying every run was
signed separately.

---

## 10. Calibrate shape 1

```bash
python3 - <<'EOF'
import json, subprocess, time
g = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,clocks.sm',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
u, m, c = [x.strip() for x in g.split(',')]
json.dump({'captured_epoch': time.time(), 'gpu_utilization': int(u),
           'gpu_memory_used_mib': int(m), 'sm_clock_mhz': int(c),
           'competing_processes': []}, open('machine_state.json','w'), indent=1)
EOF

python3 Project/tools/run_gate.py calibrate \
  --campaign "$CAMP" --shape 1 --machine-state "$PWD/machine_state.json"

REQ=$(ls -t Project/loop/requests/*.json | head -1)
PID=$(python3 Project/harness/trusted_controller.py issue-permit \
  --request "$REQ" --capability /tmp/cap_grind.json \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['permit_id'])")
echo "permit: $PID"

python3 Project/harness/trusted_controller.py run --permit "$PID" --shape 1
python3 Project/tools/run_gate.py reconcile
```

Expect `correct: true` and an `event_speedup` near 1.0. On the rehearsal it was
1.0037, giving `noise 0.003657`. Repeat this step per shape you intend to work.

---

## 11. Profile the current champion

`plan` refuses without profile evidence. nsys works inside the sandbox; `ncu`
needs root and will not.

```bash
TGT=$(sha256sum Project/kernels/k004_graphed_triton.py | awk '{print $1}')

python3 Project/tools/run_gate.py diagnostic \
  --campaign "$CAMP" --shape 1 --target-sha256 "$TGT" --tool nsys \
  --supports launch-overhead \
  --question "Does host launch overhead dominate this route on shape 1?" \
  --route "k004-graphed-triton"

REQ=$(ls -t Project/loop/requests/*.json | head -1)
PID=$(python3 Project/harness/trusted_controller.py issue-permit \
  --request "$REQ" --capability /tmp/cap_grind.json \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['permit_id'])")

python3 Project/harness/trusted_controller.py diagnostic \
  --permit "$PID" --target Project/kernels/k004_graphed_triton.py --timeout 900

python3 Project/tools/run_gate.py reconcile
python3 -c "import sys,json; sys.path.insert(0,'Project/tools'); import run_gate as g; \
print('profile id:', list(g.load_json(g.STATE,{}).get('profiles',{})))"
```

Note the `profile-…` id. `--supports` must name a bottleneck whose tools include
yours: nsys and torch-profiler cover `launch-overhead` and
`host-synchronization`; everything about memory or compute wants `ncu`.

---

## 12. Register a family for the shape

Only needed for shapes with no family yet. The catalog already has
`F-shape14-attn`, `F-shape6-local`, `F-shape8-fp16acc`, `F-shape11-hd8`.

```bash
cat > /tmp/family.json <<'EOF'
{
 "family_id": "F-shape1-graph",
 "shape": 1,
 "mechanism": "cuda-graph-replay",
 "bottleneck": "launch-overhead",
 "changed_resource": "kernel-launches",
 "expected_counter_change": {"kernel_launch_api_calls": "decrease"},
 "parent_family_id": null,
 "budget_attempts": 8,
 "budget_minutes": 90,
 "admission": "controller-authorized",
 "allow_new_attempts": true
}
EOF

FSUB=$(python3 -c "import sys,json; sys.path.insert(0,'Project/tools'); import run_gate as g; \
print(g.sha_json({'campaign_id':'$CAMP','family':json.load(open('/tmp/family.json'))}))")

printf '%s\n' "$PASS" | \
  python3 Project/tools/owner_lock_ceremony.py mint-capability \
    --action register_family --target 'family:F-shape1-graph' --campaign "$CAMP" \
    --reason "register the shape-1 graph-replay family" \
    --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_fam.json

RCP=$(python3 Project/harness/trusted_controller.py authorize \
  --capability /tmp/cap_fam.json --action register_family \
  --target 'family:F-shape1-graph' --subject-sha256 "$FSUB" --campaign "$CAMP" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['receipt_path'])")

python3 Project/tools/run_gate.py family-register \
  --campaign "$CAMP" --family-spec /tmp/family.json --authority-receipt "$RCP"
shred -u /tmp/cap_fam.json
```

`changed_resource` must match the mechanism (`cuda-graph-replay` →
`kernel-launches`) and `admission` must be exactly `controller-authorized`.

---

## 13. First card

```bash
echo '{"direction_family_id": "F-shape1-graph", "status": "open"}' >> Project/loop/cards.jsonl

python3 Project/tools/run_gate.py research --campaign "$CAMP" \
  --index-hash $(sha256sum Project/research/INDEX.md | cut -c1-16) \
  --notes "note-a.md,note-b.md" \
  --summary "<at least 220 characters on what the profile showed and why this direction follows>"

python3 Project/tools/run_gate.py plan --mode optimization --campaign "$CAMP" \
  --direction F-shape1-graph --shape 1 \
  --impl <your_candidate.py> --target-sha256 "$TGT" \
  --bottleneck launch-overhead --counter-evidence <profile-id> \
  --hypothesis "<long, specific>" \
  --prediction "1.08x expected" --prediction-kind win \
  --predict-min 1.07 --predict-max 1.09 \
  --falsifier "<how you would kill it>" --falsifier-kill "<the threshold>" \
  --prior-family-verdict NONE --kill "<direction kill rule>" \
  --sources "<file:line>" --reasoning "<long, specific>"
```

With no champion yet the bar is 1.0, so `--predict-min` must be above **1.03**.
The band also has to be tight against calibrated noise. The gate prints the
arithmetic when it refuses.

`PLAN accepted` means the loop is live. From here: issue-permit → run →
reconcile → audit, one at a time.

---

## When something refuses

```bash
python3 Project/tools/session_bootstrap.py
python3 Project/tools/run_gate.py status
python3 Project/harness/trusted_controller.py status
```

Refusals name the value that was wrong. `REFUSED` is the system working.

## For the report

Codex quota ran out, so audits moved to Claude. Codex was a different vendor
reviewing this repo's work; Claude is the same family reviewing itself. The
blind packet still holds — the auditor sees a content-addressed packet and the
candidate bytes, never the session that produced them, and its verdict is bound
to hashes it cannot choose. Every audit artifact records which backend ran.
