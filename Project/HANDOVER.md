# HANDOVER — single source of truth, 30 Aug 2026 (v3)

## !! STATUS 30 Aug ~19:15 SGT — v2 BELOW IS THE PRE-FIX PICTURE !!
Eleven commits landed this afternoon (`ed053f2..544676a`). Sections 3.4, 3.5
and 3.8 below describe defects that are now FIXED; they are kept because the
history explains why the design is shaped this way, but do not read them as
current state. What is true now:

- The trust boundary is REAL. Candidate code runs in a bubblewrap jail with no
  home, no repo, no network and dropped capabilities. A real Triton kernel is
  proven compiling and running inside it on the 3060 Ti (max_abs_error 0.0).
  The in-process `exec` path is replaced at LOCK by a shim that redirects to
  the trusted controller.
- Owner and critic authority are Ed25519-signed. Prose unlocks nothing.
- The audit authority is hash-chained; integrity and technical review are
  separate channels; a missing verdict makes a result ineligible.
- Twelve test suites exist and pass. champion_watch went from zero tests to
  278 checks with 47 mutations. The first suite that drives the REAL
  controller against the REAL gate now exists and was proved red under
  mutation in both directions.
- ZERO of 130 historical promoted rows are promotion-eligible under the new
  audit authority. That is correct fail-closed behaviour and it means the
  entire board must be re-measured after LOCK. The old ~11x, the LEADERBOARD
  and SHIP_MANIFEST are not quotable.
- 16 uncleared RULE_VIOLATION rows will freeze permits the moment the gate
  opens. That is expected; it needs one deliberate owner reconciliation pass.

LOCK IS NO-GO. Two things block it, both verified by execution:
1. `Project/harness/profile_worker.py` is not placed, so the diagnostic lane
   refuses and `build-lock` refuses. It cannot be placed by an agent —
   `.claude/settings.json` denies writes to `Project/harness/**`.
2. The scratchpad profile worker does NOT speak the committed controller's
   protocol. The controller emits a flat request; the worker demands a nested
   `gate_request` object and fails with
   `ProfileWorkerError: worker request needs the immutable gate_request object`.
   Copying it as-is would leave every diagnostic broken and LOCK would freeze
   the broken contract. Being fixed; do not place the file until the real
   worker is proven against a real controller request.

Also still open: the timing protocol is consolidated in the gate and in the
manifest's module constant, but the shape-6 path in ship_manifest.py and the
parallel copy in trusted_controller.py still hardcode it, so Wedge 1 is not
fully closed. `candidate_worker.py` emits no driver/triton, so the 12 primary
shapes would produce unshippable evidence; fixing it requires changing the
controller's schema at the same time, and both files are behind the deny.

Operating manual: `Project/GRIND_ENTRYPOINT.md` (reconciled with the code).
First ten seconds: `Project/memory/STATE.md`. Owner ceremony:
`Project/OWNER_LOCK.md`.

---

# (v2, 30 Aug ~14:00 SGT — retained for history)

You have no memory of the conversation that produced this. You do not need it.
This file is the working plan. It supersedes v1 of this file (which carried
five factual errors, corrected in section 5), TEMP-PROGRESS-LOG (deleted),
ROUND7_FINDINGS (merged), NARROWINGS_TODO, AUTHORITY_V4_PLAN and
harness_v2_proposal.

Still authoritative: CLAUDE.md (standing orders), Project/PLAN.md,
Project/RUNBOOK.md, Project/memory/{STATE,DECISIONS,LESSONS}.md,
Project/research/*.md (13 notes, read INDEX.md first),
Project/loop/GATE_DESIGN.md (design record),
Project/loop/OWNER_PATCH_card_gate.md (the paste artifact).

Verification provenance: v1 was audited line-by-line against the repo by a
fresh Claude session AND an independent external auditor (different vendor,
read all 1,709 files, ~60 findings, C-01..C-11 / H-01..H-32 / M-01..M-20).
Where the two agreed, this file states it as fact. Where v1 was wrong, this
file says so explicitly.

OWNER DIRECTIVE (30 Aug, binding): work only. No packaging, submission,
Devpost, video or report assembly until the owner raises it. The plan is
FIX -> LOCK (owner pastes) -> GRIND until the owner says stop. Scope is
never cut for time; the owner manages the clock.

---

## 1. HARD FACTS

- CODE FREEZE 31 Aug 20:00 SGT. Submission AND Devpost registration close
  1 Sep 12:00 GMT+8. (Logistics are the owner's; not the working queue.)
- Hardware: one RTX 3060 Ti 8GB, driver 610.57.04, torch 2.12.0+cu130,
  triton 3.7.0, python 3.14.7. Single-GPU rule, own machine. No rental.
- Rubric (README §3.6, official track statement): Technical 35 /
  Innovation 20 / Impact 20 / Feasibility 15 / Presentation 10.
- JUDGES DO NOT RERUN (research/competition-scoring.md:10). Correction of a
  prior overreach: this raises the evidence burden; it does NOT make higher
  performance worthless, and the kernels + measured results are the primary
  product. The harness is supporting story and bonus.
- Branch grind-day1; main is 134+ commits behind (owner call later).
- Disk: FIXED — ~72 GB free (v1 said 9.8 GB; owner cleared it).
- codex resolves to /usr/local/bin/codex; no ~/.local/bin shim exists today
  (the audited PATH-shim precondition is currently absent; pinning stays in
  the plan as hygiene).

## 2. VERIFIED GOOD — do not re-litigate

- Submission file byte-identical to the official script outside the
  designated region (independent prefix/suffix + AST compare) and
  self-contained; degrades to bit-identical baseline without CUDA/Triton.
- No cheat mechanism in any shipped kernel path (fresh-session read AND
  external static audit agree): no input-identity/address caching, no
  accuracy-vs-timing branch, no skipped math, no async flattery, no input
  mutation. Exact-erf GELU, LN eps 1e-5 biased variance, head_dim**-0.5,
  causal mask in every fast path.
- Speed = launch collapse into authored Triton kernels + fp16 tensor-core
  GEMMs with fp32 accumulation on an fp32 residual stream + whole-forward
  CUDA-graph replay.
- 12 of 14 shapes hold promoted runner rows; 6 and 14 live on side packets.
- Dispatcher fallbacks exact; graph state per-instance (no cross-shape
  reuse hazard on the scoring path); no credentials tracked.
- Input determinism, verified in both sources: the RUNNER uses the same
  seed+trial device-generator scheme as the official script (runner.py:384
  vs official:387), so the 12 standard shapes' inputs are bit-identical to
  a default judge run. The SIDE evaluators use CPU RNG — NOT bit-identical.
  (v1 claimed bit-identity globally; that was wrong for 6/14.)

## 3. WHAT IS WRONG (verified; ordered by cost)

### 3.1 The headline number is not trustworthy as published
All 12 published rows carry baselines 6-63% slower than the runner's own
calibration on identical official code (noise floor 0.004-0.36%), so
contention likely inflated the 10.95x/11.0x. IMPORTANT CORRECTION: v1's
"8.68x corrected bound" is INVALID — it mixed absolute times across runner
invocations, which LESSONS #11 forbids. No corrected number can be derived
that way. The only fix is a controlled campaign (see GRIND). The
10.32x-official vs 10.95x-runner agreement validates the pipeline, never
the box conditions.

### 3.2 Shape 6 can score zero
75% of the abs budget on the packet seed; 92% on the integrated smoke; ONE
seed; and shape6_local_eval.py still tests standalone k015, not the
submission (line 112), with a single-seed default (line 88), CPU RNG,
no official mask/matmul flags, and an off-by-upper median. Exposure is
concentrated in shapes 6 and 14 (the CPU-RNG side evaluators).

### 3.3 Evidence-path defects (current state, re-verified on disk)
- Live SHIP_MANIFEST.json still carries fabricated audit-shaped text
  ("side-evidence (oracle validated vs official dense)") for BOTH 6 and 14
  — and for 6 the description is also factually wrong. ship_manifest.py's
  label was fixed in code (task 03) but the generator (correctly) refuses
  to run on a dirty tree, so the JSON is stale; the code STILL writes
  synthesized text into the audit_verdict field, and the task-06 verdict
  filter (exclude RULE_VIOLATION rows, fail closed) was NEVER implemented
  (Codex's own completion report confirms). packet_is_correct(6) also
  ignores memory.flat.
- shape14_eval.py was rebuilt well (streamed B=32 as 32 serial B=1 slices,
  submission import, official predicate, 5 seeds, median-of-sums,
  reserved-memory) and has NEVER BEEN RUN. It is NOT "one command away":
  it is a substantial run, its RNG is not official-equivalent, and it
  proves decomposed execution (organizer-sanctioned), not a literal
  one-call B=32. decomp-check always exits zero — must become fail-hard.
- results_side/ and both side evaluators are outside every protection and
  authority layer.
- LEADERBOARD stars max-ever rows: shapes 7 and 11 currently star
  RULE_VIOLATION rows; six stars are NEEDS_CONTEXT. Not a ship selection.

### 3.4 The guard/gate (live state)
.claude/hooks/guard_bash.py is the 28 Aug accident guard: fails OPEN on
malformed payloads, blacklist-based, zero permit logic. The v4.6 gate is
design-only, its latest external review ended WITHOUT a final verdict, and
every measurement ever taken is PRE-GATE. The A3 rewrite exists and is
verified 25/25 but is not spliced. Known guard gaps: cd-relative, brace
expansion, python -c, git stash, find -delete, install/dd/rsync/shred,
codex-exec-as-token (quote-strip anchor bug), and .git/CLAUDE.md/PLAN.md/
RUNBOOK.md unprotected.

### 3.5 Authority holes (as-if-pasted analysis, all confirmed)
No brake for a missing verdict (absence is free — highest-leverage fix);
judge binary unpinned; parser accepts decoy/bare PASS (needs nonce + full
schema); scratch bytes launder into primary confirmations; agent-minted
free-text family ids mean THREE-STRIKES HAS NEVER CLOSED ANYTHING;
calibration permits unbounded and cals[-1] retroactively re-crowns;
prediction bands unfalsifiable in placement; champion_watch has zero tests;
owner-quote and critic-receipt paths are unauthenticated text.

### 3.6 Competence is unenforced
The gate demands citations and predictions, never a hardware counter. That
is exactly how k011 was built on an occupancy guess (truth: memory traffic,
LESSONS #21). Zero ncu/nsys/torch.profiler captures exist; the central
launch-bound claim has never been confirmed by a counter; the sanitizer has
never touched the kernels. nsys/torch.profiler/compute-sanitizer work today
with no setup; ncu needs sudo (RmProfilingAdminOnly=1 — never modprobe/
reboot; nsys is the right tool for graphed work anyway).

### 3.7 Shipped-file defects (cheap, real)
_GRAPH_WARMUP_CALLS=3 → graph path covered in 2 of 5 default trials (v1
said 1 of 5 — wrong), zero if --accuracy-trials <= 3 → lower to 1;
_TRITON_OK tests import, not compilation, and the fast path aborts instead
of degrading; static_x.copy_ broadcasts (no shape/dtype/stride assert);
GELU in-place assumes contiguity; weight-recopy after first forward is a
silent no-op (no invalidation, no weight tripwire); padding > 0 is a
performance cliff to ~1.0x (correct but disclose).

### 3.8 Trust boundary (deep; owner decision required)
Candidate source executes inside the runner's own process
(runner.py:189-200 exec) — the categorical zero-trust blocker; the runner
never consumes permits; record-verdict trusts its caller; read_verdicts is
last-write-wins; champion_eligible ignores verdicts; journal append has no
flock/fsync; cals[-1] retroactivity. ALL of these live in the FROZEN
runner. Fixing them means editing the pinned referee (standing order 2) and
re-freezing — an owner-only trade. Until then: label the system honestly as
AI-CONTROL WITH PUBLISHED RESIDUALS, never zero-trust.

---

## 4. THE PLAN: FIX -> LOCK -> GRIND

### PHASE 1 — FIX (agent, no GPU). Everything below lands BEFORE the paste,
because gate/auditor/guard artifacts and ledgers become owner-only after it.

1A SAFETY+TRUTH — DONE 30 Aug (this session):
- [x] 24 verdict rows committed durably (a592fcc; 81 rows at HEAD).
- [x] 14 back-edited packets DISCARDED, restored to measured-source bytes
      (the edits had replaced measured candidate source with current bytes
      — evidence corruption; auditor H-06).
- [x] Parallel session's work landed in one labeled commit (bf2181e).
- [x] This file corrected and committed.

1B GUARD -> ALLOWLIST (delivered as the paste artifact; agent cannot edit
.claude/**): deny-by-default with ~20 real command shapes; all of main()
inside try, fail CLOSED on malformed/non-dict/non-string/empty, length cap;
A3 rewrite spliced; codex-exec carve-out anchored on ORIGINAL text; protect
.git, CLAUDE.md, PLAN.md, RUNBOOK.md, results_side/, STATE.md, drafts/;
one owner break-glass line; matching settings deny rules.

1C AUDITOR INTEGRITY: pin codex absolute path + sha in artifacts, refuse
$HOME; per-attempt nonce + FULL schema validation; BRAKE ON MISSING
VERDICTS (absence = ineligible); retry caps from durable rows; O_EXCL
markers; first-write-wins hard verdicts + explicit resolution records;
backlog derived from the journal, never leaderboard text.

1D COMPETENCE LAYER (all of it): mechanism_catalog.json with OWNER-APPROVED
extension (identity discipline without banning novelty), doubling as the
bottleneck->prescription menu; family_id = hash(mechanism, shape), free
text banned; plan requires --bottleneck + --counter-evidence which the gate
verifies (exists, newer than card, same shape) — no direction opens on a
hunch, ever; prediction bands tied to calibrated noise and required to be
informative (characterization/null runs stay legitimate); campaign-level
stall brake; plan must cite the prior verdict for its family; budget
snapshot at first plan; calibration permits capped and bound; scratch and
primary namespaces structurally separated.

1E JUDGE UPGRADE: profile artifact into the packet; auditor second job —
does the counter evidence support the diagnosis, did the change address it;
WEAK_DIAGNOSIS verdict + brake, kept SEPARATE from integrity verdicts.
[x] packet-as-untrusted-data already landed.

1F EVIDENCE PATHS: ship_manifest — audit_verdict: null + evidence_status +
reference_method (never synthesized verdicts), task-06 filter fail-closed,
selection from an explicit final evidence map, memory.flat condition;
shape6_local_eval REBUILT to full protocol (submission import, official
device-generator inputs + all-true mask + official matmul/TF32 flags,
>=5 seeds, statistics.median, memory-trend, fail-hard exits);
shape14_eval decomp-check made fail-hard + persisted, labels precise
(decomposed serial; GPU-compute vs staging-inclusive named).

1G SUBMISSION HARDENING (code now, measured in GRIND): warmup 3->1; exact
shape/dtype/stride assert before replay; Triton compile-probe with
controlled fallback; GELU contiguity contract; weight-recopy invalidation;
rebuild submission + independent byte-identity verification.

1H TESTS: end-to-end against the STAGED GUARD BYTES and every item above;
champion_watch.py from zero.

PARKED ON OWNER DECISION (frozen-runner boundary): runner-internal permit
consumption; process-isolated candidate execution; verdict-aware
champion_eligible; first-wins read_verdicts; journal flock/fsync;
calibration binding in regenerate_leaderboard. Documented residuals until
the owner authorizes a runner re-freeze.

### PHASE 2 — LOCK (owner, ~5 min; agent STOPS and waits)
Owner pastes the new guard into .claude/hooks/guard_bash.py and the deny
lines into .claude/settings.json, says "applied". Agent runs the proof
battery (the 8 + A3 roots + malformed payloads + allowlist bounce +
permit-less runner bounce). EXPECT THE GATE TO OPEN CLOSED — 16 uncleared
RULE_VIOLATION rows postdate the cutoff; correct behavior; one deliberate
reconciliation pass follows. Gate construction then ENDS permanently
(no round 8+ of gate self-review — merit-based exclusion: six rounds of
self-review produced less than one wave aimed at the work product).

### PHASE 3 — GRIND (under the gate, until the owner says stop)
Serialize all GPU work; quiet box for anything timed; profiling and timing
never interleave; profiler output backs mechanism claims only; every
speedup number comes from the runner. Pipeline order (dependency-correct;
v1 and the old plan had hardening AFTER measurement — wrong, it changes
the bytes):
1. Ship-file hardening lands (1G) ->
2. compute-sanitizer racecheck/memcheck on every authored kernel; loop
   fixes until clean (it can force code changes, so it precedes the board) ->
3. FREEZE BYTES: one submission sha for everything that follows ->
4. MEASURE at that sha: shape-6 full multi-seed protocol (harden the
   tightest op only if the measured margin demands it); shape-14
   validate -> fail-hard decomp-check -> full streamed eval; the
   controlled 12-shape campaign (machine-state recorded before/after,
   N>=3 sweeps on small launch-bound shapes, all repeats kept,
   predeclared median, event+wall side by side, divergence flagged).
   Publish whatever it says. ->
5. PROFILE (read-only diagnosis): nsys launch-collapse timeline,
   torch.profiler per-op counts, sudo-ncu counters on compute-bound
   shapes; fold into 6-field profile decision records ->
6. OPTIMIZATION LOOP, indefinitely: profile -> diagnose from counters ->
   prescribe from the catalog -> card -> implement -> measure -> audit ->
   repeat. Target selection from measured headroom across multiple
   scoring weightings (candidates: 13 never tried; 6 never tuned, runs
   eagerly; 8 only via a materially new mechanism; raw-FLOP weighting is
   ~99.9% shape 14 — never pick a target off that axis alone). CUDA C++
   is unlocked and fires the moment any pre-test shows it beating the
   Triton route. Continuous claim-trace discipline (LESSONS #24).

EXCLUDED ON MERIT (not time): gate self-review rounds; top-K beam
restructure (stall brake is the sound approximation); modprobe/reboot for
ncu; back-editing historical packets; cross-invocation re-denomination;
rental; treating the 10.32/10.95 agreement as quiet-box proof.

---

## 5. v1 ERRORS, CORRECTED HERE (so nobody re-inherits them)

1. "Re-denominated geomean is 8.68x (a bound)" — INVALID method; deleted.
2. "24 pending verdict rows committed — done" — was FALSE when written
   (both commit attempts were guard-blocked on the word 'clean' in the
   message); actually committed 30 Aug at a592fcc.
3. "Graph path gets 1 of 5 trials" — it gets 2 of 5 (capture call 4
   returns replay output; call 5 replays). Fix (warmup 3->1) unchanged.
4. "A judge running defaults gets bit-identical inputs" — TRUE for the 12
   runner shapes, FALSE for shapes 6/14 (side evaluators use CPU RNG).
5. "Shape 14 is one command away" — the tool is good but the run is
   substantial, non-official-RNG, and proves decomposed execution only.
Also stale in v1: disk 9.8GB (now ~72GB); the PATH shim precondition
(~/.local/bin/codex) no longer exists; ship_manifest's shape-6 label is
already fixed in CODE (JSON still stale; field misuse + missing filter
remain); "the agent is the submission, kernels are evidence" — rejected:
kernels + measured evidence are primary, harness is supporting story.

## 6. STANDING RULES (unchanged)
Never edit torch_transformer_benchmark.py, tensorflow_transformer_benchmark.py,
README.md, Project/shapes.json, Project/manifest.json, anything in
Project/results/ (runner-written only), .claude/**. Every benchmark via
Project/harness/runner.py + shapes.json id; no raw-dial benchmarking.
Commit candidate bytes BEFORE first runner contact. Promotion = correctness
pass + speedup above the calibrated noise floor. Guard etiquette: never put
clean/reset/restore/checkout after 'git' in one command segment. Plain
language. The owner's stop overrides everything.
