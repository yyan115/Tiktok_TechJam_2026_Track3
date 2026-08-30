# Post-LOCK runbook — from a frozen board to the first measured card

Every command below was run end to end on a full copy of this repo, on this GPU,
on 30 Aug. It is not a plan; it is a transcript with the values swapped out.
Where something refused the first time, the fix is written in.

**Who runs what.** Four commands touch your private key: `keygen`, `sign-lock`,
and every `mint-capability`. Those are yours, at your own terminal. Everything
else can be driven by an agent. `Project/OWNER_LOCK.md` explains why that line
exists; the short version is that an agent holding the signing key can authorise
its own work, and then the whole authority model is decoration.

Set this once in each shell (the agent's too):

```bash
cd /home/admin/Desktop/Repos/Tiktok_TechJam_2026_Track3
KEYS=~/techjam-keys          # OUTSIDE the repo. Never inside it.
CAMP=CAMP-POSTLOCK           # your campaign id; pick one and keep it
```

---

## Step 0 — check the box is idle  *(agent)*

Timed work on a busy box is worthless. Do not use `pgrep -f`; it lies (LESSONS 22).

```bash
python3 Project/tools/champion_watch.py --dry-run     # "active": [] means clear
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Desktop apps (kwin, VS Code, a browser) are fine. Another benchmark or an audit
is not — wait for it.

---

## Step 1 — make your keys  *(YOU — prompts for a passphrase, needs a real terminal)*

```bash
mkdir -p "$KEYS" && chmod 700 "$KEYS"
python3 Project/tools/owner_lock_ceremony.py keygen --key-dir "$KEYS"
```

It asks for a passphrase and will crash with `EOFError` if anything pipes input
into it, so this cannot be scripted. Write the two fingerprints it prints
somewhere off this machine. Back the private keys up offline. Lose them and this
lock can never be rotated.

---

## Step 2 — install the staged bytes, THEN build the lock  *(agent)*

Order matters. The lock pins hashes as they are at build time, so installing
these afterwards makes the lock fail instantly and the controller refuse
everything.

```bash
cp Project/lock_staging/guard_bash.py .claude/hooks/guard_bash.py
cp Project/lock_staging/settings.json .claude/settings.json
cp Project/lock_staging/runner.py     Project/harness/runner.py

python3 Project/tools/owner_lock_ceremony.py build-lock          # dry run: read it
python3 Project/tools/owner_lock_ceremony.py build-lock --yes    # writes LOCK.json
```

The dry run lists all 29 files and a "staged bytes: are they installed yet?"
section. Every line there must say installed before you pass `--yes`.

---

## Step 3 — sign it  *(YOU)*

```bash
python3 Project/tools/owner_lock_ceremony.py sign-lock \
  --private-key "$KEYS/owner_private_key.pem"
```

Look for `verify_lock() says valid=True`.

---

## Step 4 — activate  *(YOU mint, agent runs)*

```bash
# YOU
python3 Project/tools/owner_lock_ceremony.py mint-capability \
  --action lock.activate --campaign "$CAMP" \
  --reason "activate the post-fix lock" \
  --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_activate.json

# agent
python3 Project/harness/trusted_controller.py activate-lock \
  --capability /tmp/cap_activate.json
python3 Project/harness/trusted_controller.py status     # expect "active": true
shred -u /tmp/cap_activate.json
```

Capabilities expire in 20 minutes by default. If it lapses, mint another.

---

## Step 5 — retire the 16 pre-gate verdicts  *(YOU mint, agent runs)*

Until this is done, **no permit can issue at all** — the gate treats an
unresolved hard verdict as a brake on everything, not just the row it names.

Do **not** use `run_gate.py verdict-clear --kind violation`. It reads like the
unlock, spends a signed capability, prints "resolved by controller-verified owner
authority", and does not move the brake. Rehearsed: 16 cleared, 16 still
blocking. It now warns and exits 1.

```bash
# YOU — one signature covers all 16
python3 Project/tools/owner_lock_ceremony.py mint-capability \
  --action audit.resolve --target 'audit:*' --campaign "$CAMP" \
  --reason "retire pre-citation-gate rows so the post-LOCK board can be measured" \
  --max-uses 25 --expires-minutes 60 \
  --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_audit.json

# agent
python3 Project/tools/clear_pregate_verdicts.py \
  --capability /tmp/cap_audit.json --campaign "$CAMP"          # dry run, lists all 16
python3 Project/tools/clear_pregate_verdicts.py \
  --capability /tmp/cap_audit.json --campaign "$CAMP" --yes
shred -u /tmp/cap_audit.json
```

Expect `16 retired. RULE_VIOLATIONs still braking permits: 0`.

The resolution recorded is `FINDING_ACCEPTED_ROW_RETIRED` — the findings stand,
the rows are withdrawn from contention. Not `FINDING_OVERTURNED`, which would
claim the auditors were wrong. They were not; those rows genuinely predate the
citation gate, which is exactly why the board is being re-measured.

---

## Step 6 — open the campaign  *(agent writes spec, YOU mint, agent runs)*

`timing_config` must equal the controller's own protocol or the campaign can
never reconcile a calibration and wedges permanently. These are today's values;
the gate checks them and names both if they drift.

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

# agent — the subject hash is over the spec exactly as written
python3 -c "import sys,json; sys.path.insert(0,'Project/tools'); import run_gate as g; \
print(g.sha_json(json.load(open('/tmp/campaign.json'))))"

# YOU — paste that hash as SUBJ
python3 Project/tools/owner_lock_ceremony.py mint-capability \
  --action open_campaign --target "campaign:$CAMP" --campaign "$CAMP" \
  --reason "open the post-LOCK re-measurement campaign" \
  --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_camp.json

# agent
python3 Project/harness/trusted_controller.py authorize \
  --capability /tmp/cap_camp.json --action open_campaign \
  --target "campaign:$CAMP" --subject-sha256 <SUBJ> --campaign "$CAMP"
# take receipt_path from that output
python3 Project/tools/run_gate.py campaign-open \
  --spec /tmp/campaign.json --authority-receipt <receipt_path>
```

---

## Step 7 — the grind capability  *(YOU — read the tradeoff)*

Every benchmark run needs a `permit.issue` capability, and they are **single-use
by default**. One per run means you sign all night.

One capability can cover a whole session instead:

```bash
python3 Project/tools/owner_lock_ceremony.py mint-capability \
  --action permit.issue --target 'shape:*' --campaign "$CAMP" \
  --reason "grind session <date>: shapes 1-13" \
  --max-uses 100 --expires-minutes 480 \
  --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_grind.json
```

Be honest with yourself about this one. A wildcard, 100-use, 8-hour capability is
a much weaker control than one signature per run — for that window the agent can
issue any permit it likes. It is still bounded (a count, a clock, a scope, and
every use is a separate journal event naming the shape), and it is still a thing
only you can create. But if you use it, say so in the report rather than letting
a reader assume every run carried its own signature. Shorter expiry and a smaller
`--max-uses` cost you a re-mint and buy back real ground.

---

## Step 8 — first calibration  *(agent)*

Per shape, before anything is compared on that shape.

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
python3 Project/harness/trusted_controller.py issue-permit \
  --request "$REQ" --capability /tmp/cap_grind.json      # note the permit_id
python3 Project/harness/trusted_controller.py run --permit <permit_id> --shape 1
python3 Project/tools/run_gate.py reconcile
```

Rehearsed result: `event_speedup 1.0037`, `correct: true`, calibration bound with
`noise 0.003657`. Noise sets the effect floor a win must clear.

---

## Step 9 — diagnostic, so a card has evidence to cite  *(agent)*

`plan` refuses without a profile record. nsys is proven to work inside the jail
(Nsight Systems 2025.3.2, real counters, no degradations). `ncu` needs root and
will not work this way.

```bash
TGT=$(sha256sum Project/kernels/k004_graphed_triton.py | awk '{print $1}')
python3 Project/tools/run_gate.py diagnostic \
  --campaign "$CAMP" --shape 1 --target-sha256 "$TGT" --tool nsys \
  --supports launch-overhead \
  --question "Does host launch overhead dominate this route on shape 1?" \
  --route "k004-graphed-triton"

REQ=$(ls -t Project/loop/requests/*.json | head -1)
python3 Project/harness/trusted_controller.py issue-permit \
  --request "$REQ" --capability /tmp/cap_grind.json
python3 Project/harness/trusted_controller.py diagnostic \
  --permit <permit_id> --target Project/kernels/k004_graphed_triton.py --timeout 900
python3 Project/tools/run_gate.py reconcile     # note the profile-… id
```

`--supports` must name a bottleneck whose `evidence_tools` include your tool:
nsys and torch-profiler cover `launch-overhead` and `host-synchronization`;
`global-memory-traffic`, `compute-throughput`, `occupancy-resource-pressure` and
`tensor-core-utilization` all want `ncu`.

---

## Step 10 — register a family, then plan  *(YOU mint, agent runs)*

Only needed for a shape with no family in the catalog. The catalog already has
`F-shape14-attn`, `F-shape6-local`, `F-shape8-fp16acc`, `F-shape11-hd8`.

Three things refuse a family spec, all of which bit during the rehearsal:
`admission` must be exactly `"controller-authorized"`; `changed_resource` must
equal the mechanism's own (`cuda-graph-replay` → `kernel-launches`); and the
signed subject is `{"campaign_id": …, "family": …}`, **not** the family alone.

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

# agent — subject hash
python3 -c "import sys,json; sys.path.insert(0,'Project/tools'); import run_gate as g; \
print(g.sha_json({'campaign_id':'$CAMP','family':json.load(open('/tmp/family.json'))}))"

# YOU
python3 Project/tools/owner_lock_ceremony.py mint-capability \
  --action register_family --target 'family:F-shape1-graph' --campaign "$CAMP" \
  --reason "register the shape-1 graph-replay family" \
  --private-key "$KEYS/owner_private_key.pem" --out /tmp/cap_fam.json

# agent
python3 Project/harness/trusted_controller.py authorize \
  --capability /tmp/cap_fam.json --action register_family \
  --target 'family:F-shape1-graph' --subject-sha256 <SUBJ> --campaign "$CAMP"
python3 Project/tools/run_gate.py family-register \
  --campaign "$CAMP" --family-spec /tmp/family.json --authority-receipt <receipt_path>
```

Add the card, then research, then plan. A `win` prediction must clear the
incumbent by the effect floor: with no champion yet the incumbent is 1.0 and the
floor is 3%, so `--predict-min` must exceed 1.03. The band also has to be tight
relative to calibrated noise. The gate prints the arithmetic when it refuses.

```bash
echo '{"direction_family_id": "F-shape1-graph", "status": "open"}' >> Project/loop/cards.jsonl
python3 Project/tools/run_gate.py research --campaign "$CAMP" \
  --index-hash $(sha256sum Project/research/INDEX.md | cut -c1-16) \
  --notes "note-a.md,note-b.md" --summary "<220+ chars>"
python3 Project/tools/run_gate.py plan --mode optimization --campaign "$CAMP" \
  --direction F-shape1-graph --shape 1 --impl <candidate.py> --target-sha256 "$TGT" \
  --bottleneck launch-overhead --counter-evidence <profile-id> \
  --hypothesis "<long>" --prediction "1.08x expected" --prediction-kind win \
  --predict-min 1.07 --predict-max 1.09 \
  --falsifier "<how you would kill it>" --falsifier-kill "<the threshold>" \
  --prior-family-verdict NONE --kill "<direction kill rule>" \
  --sources "<file:line>" --reasoning "<long>"
```

`PLAN accepted` means the loop is live. From here it is issue-permit → run →
reconcile → audit, one attempt at a time.

---

## If something refuses

Read the refusal; they are written to be read and they name the value that was
wrong. Then:

```bash
python3 Project/tools/session_bootstrap.py     # the manual, then live status
python3 Project/tools/run_gate.py status
python3 Project/harness/trusted_controller.py status
```

`REFUSED` is the system working. The one failure mode to actually worry about is
a command that reports success and changes nothing — which is what step 5 exists
to route around.
