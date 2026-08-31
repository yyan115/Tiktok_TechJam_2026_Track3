# DECISIONS — plain-language diary of what we discussed and agreed

## 1 Sep 2026 ~02:15–03:00 — THE SINGLE-ARTIFACT BOARD. All 14 shapes on one hash.

**What the owner asked for:** measure every remaining shape on one artifact so the
reported board carries a single hash for every row, and do not touch anything in
`Project/submission/`.

**Done.** Artifact `c2028c4823ff756b062940e4eff35d5a6a341e9538b755256509cf3432e7794b`.
Ten primary rows measured this session (2, 3, 4, 5, 7, 8, 9, 10, 11, 12) plus
shape 6 in the side lane, joining shapes 1, 13 and 14 already on that hash. Box
verified quiet before and mid-campaign: 1665 MHz, 0% utilisation, 47–49 W,
`active: []`. Every row `correct: true`. No submission byte was touched.

| | result |
|---|---|
| geomean, 12 shapes with a baseline | **11.87×** |
| mean MFU, all 14 weighted equally | **42.7%** |
| best / worst speedup | 31.513× (shape 13) / 2.366× (shape 8) |
| best MFU | 88.7% (shape 14) |
| correctness | 94 trials, 17,370,759,168 elements, **0 violations** |
| strikes | 0 |

**Shape 12 needed a new family.** The owner's `resolve_family_novelty`
authorization for `F-shape12-fusion` had been signed on 31 Aug 18:02Z but never
cashed — `family-register` was the missing step, which is why shape 12 still read
as 12/12 spent. Registered, then found that `delta` refuses a family with no open
card, so card **C33** was opened in `Project/loop/cards.jsonl` and `plan` used
instead. Shape 12 measured 11.6419×.

**Corrections made to my own claims during this session, in order:**

1. I told the owner shape 2's earlier diagnostic died because the novelty
   authorization was appended mid-run. **Wrong** — I reasoned from journal
   timestamps. The previous session's own account says it was an interrupt. The
   advice I derived from it ("don't sign anything while a run is in flight") was
   invented and has been withdrawn.
2. I wrote "13 shapes route to the megakernel". **Wrong** — shape 14 is
   `d_model` 1024 and takes the fp16 branch like shape 8. It is 12 and 2.
3. `MEASUREMENT_METHODOLOGY.md` §8 said `torch.compile` was never run. **Wrong** —
   `DECISIONS.md` (29 Aug ~02:35) records it measured at 7.0 / 3.1 / 1.2× on the
   shape 3 / 13 / 8 dials. Corrected, with the caveat that it is pre-gate and
   eleven artifacts old.
4. The README said the shape-14 oracle validation's worst deviation was 1.4e-6.
   That figure is **real but from a run that did not configure the mandated TF32
   profile**. Under the profile the competition requires it is 6.2e-4. Both are
   now reported in §7.1, because together they separate our algorithm's error
   (1.4e-6) from the number format's (6.2e-4), which is a stronger claim than
   either alone.
5. The tech report claimed the thirteen-round design review "returned APPROVE".
   **Unsupported.** Every file in `Project/audits/` containing the word APPROVE is
   a *prompt*; no verdict artifact has one. The recorded strategy verdicts are
   ROUND 1 REVISE and ROUND 2 REVISE. Clause removed from the report and the
   Devpost text under our own rule that a missing verdict is never an approval.

**The auditor diagnosis in every document was wrong, and the real cause is now
known.** See LESSONS 61. Short version: the audit *recorder* is fine; the auditor
never starts. `verdict_schema.json:70` uses `allOf`, OpenAI structured output
forbids it, Codex 400s before reading the packet. Added by `ed053f2` on 30 Aug
15:46, masked five hours later by `231e786` switching the default backend to
Claude for a quota reason, resurfaced tonight when Codex ran again. Reproduced
live: three attempts, three 400s, all recorded in the hash chain, escalated to
`owner_attention`. **Owner fix, one edit, inside the LOCK.**

**Also corrected:** `Project/results/LEADERBOARD.md` is a frozen pre-LOCK artifact
that stars the max-ever row per shape across incomparable invocations, which
selects for whichever run had the slowest baseline. Its own shape-1 rows show it:
the starred k009 reads 11.150× on a 7.2044 ms baseline against another k009 at
9.910× on 5.7539 ms, and our measured baseline is 5.0586 ms. It is now labelled in
the README's repository map and its limitations list rather than left inviting.

**Documents updated to the single-artifact board:** `Project/BOARD.md` (rewritten,
now carries per-row packet hashes and entry ids), `MEASUREMENT_METHODOLOGY.md`
§5/§6/§7.1/§7.3/§7.4/§8/§10, and all four drafts.

## 30 Aug 2026 ~11:00 — FREEZE MOVED to 31 Aug 20:00; deliverables cleanup pass (second session, harness session running in parallel)

- **Owner moved the code freeze from 31 Aug 12:00 to ~20:00 SGT.** Consequence recorded everywhere: the packaging window shrank from ~14h to 6h, so all report/README/video PROSE must be finished before the freeze; only numbers, assembly and the video recording belong in the window. Updated STATE.md (both timeline lines), NARROWINGS item 5, harness_v2_proposal.md.
- **Lane discipline while two Claude sessions run:** this session took drafts/, sensitivity_board.py + SENSITIVITY.md, and a new packaging checklist. It did NOT touch Project/loop/** (except the one-line freeze-time fix), run_gate.py, champion_watch.py, audit_champion.py, tools/tests/**, .claude/**, verdicts.jsonl, or the GPU — all owned by the parallel harness session.
- **Score board rebuilt (NARROWINGS item 8), three real bugs fixed:**
  (a) it selected each shape's *fastest ever* promoted entry by median ms — cherry-picking across incomparable invocations, straight into the report's headline table (LESSONS #11, #22). Now every row comes from ONE designated ship pass (the 29 Aug 02:26-02:31 quiet-box sweep) with loud labeling on any fallback.
  (b) one fp16 denominator was presented as *the* roof. Now all three sourced roofs (16.2 / 32.4 / 64.8 TF/s) plus bandwidth, with the GA10x accumulator distinction stated.
  (c) shape 14 was divided by (1/32) to "scale" a B=1 median to B=32. Measured B=1->B=2 is 2.18x, not 2x. The projection is deleted; full-batch reads PENDING.
  New: per-shape marginal value (what a 20% win on ONE shape is worth under each weighting) so no direction gets runner time without a scenario that it helps. S4 collapses onto S1 and the board now says so instead of showing two identical numbers as if independent.
- **Independent cross-check found:** the official-script all-dials board (10.32x geomean) and the frozen-runner quiet-box board (10.95x) agree within 6% despite different code paths and days; per-shape scatter reaches +-25% on the sub-millisecond shapes. Report now leads with the OFFICIAL script's number (the artifact a judge can verify) and publishes the scatter instead of the flattering column.
- **Two unsourced claims caught and removed before they shipped:** (1) "our hand-rolled GEMM measured 98% as fast as cuBLAS" (TEMP-PROGRESS-LOG, propagated into the report draft) has no measurement anywhere in the repo — the research note says 85-95% as a *literature* range for Triton GEMMs generally, "net gain uncertain". Replaced with the measured 1.79x->2.13x epilogue-fusion result. (2) "NVIDIA Nsight Compute" was listed as a development tool; ncu is installed but has never been run, and no torch-profiler capture exists either. Removed. Also corrected "11 of 14 shapes are d128/seq128" to the true 9 of 14.
- **Documents rewritten:** tech report (v2, judge-facing, leads with the Track-2 violation -> authority-v4 arc per NARROWINGS 11b, every number sourced, PRE-GATE labeled, [PENDING] where the final-sha board is still owed), judge-facing README draft (restructured to the deliverable's required sections; rental purged), video shot list (rental purged, six-hour window acknowledged), plus two new drafts: packaging_checklist.md and devpost_description.md.
- **Two owner decisions surfaced (not actioned):** (1) `Project/audits/strategy_review_codex.md`, a 791 KB raw strategy transcript, is in git history at 49bfd47 and that history is already pushed to all three remote branches — untracking it later did not retract it (LESSONS #14). Recommendation: fresh squashed public repo. (2) The organizers' slides (Track3_Slide1-4.png) and the verbatim webinar transcripts (MEETING-NOTES*.md) are tracked and would ship publicly; the video rules forbid third-party copyrighted content and the same caution applies to the repo. Recommendation: remove from the public path, keep the distilled research note.
- Also noted, not acted on: the leaderboard's champion star picks max speedup while the ship board uses the quiet-box pass, so they disagree for shape 2 (17.06x vs 14.98x). One publication rule is needed before ship; the report currently uses the quiet-box/official numbers throughout.

## 29 Aug 2026 night — GATE APPROVED BY SOL (round 13). Owner paste = the only step before the caged grind restarts

- 13 blind rounds (R1 REJECT -> R13 APPROVE), findings shrinking 10->8->9->6->6->4->5->3->2->2->2->1->0. The user's clarified override rule ("only skip what you genuinely judge not-an-issue") was applied once (exactly-once reconcile machinery) and the reviewer itself later endorsed that override.
- Final architecture (v3.12): one-use permits from validated think-steps (research vs saved base -> cited plan with quoted lines, or structured delta), guard consumes under the shared lock as the FINAL authorization action, watcher reconciles from the referee's own journal row with full schema validation, per-(card x shape) strikes with true-best tracking, 3-strike closure -> forced postmortem -> new direction, nonce-bound critic receipts to reopen, everything fail-closed, auditor verifies citations.
- NEXT: owner pastes OWNER_PATCH_card_gate.md into .claude/hooks/guard_bash.py (Block A after WRITE_PATTERNS defs; Block B at the END of main) -> agent runs the four proof-tests -> optimization resumes autonomous and caged. Allocation on restart per the converged plan: rental killed; remaining levers are tuning refinements + packaging; CODE FREEZE 31 Aug noon.

## 29 Aug 2026 evening — RUN GATE CONVERGED (v3.6) after 7 blind review rounds; awaiting the owner's one paste

- The owner's spec (mechanical stop after EVERY try; two explicit steps — research grounded in the saved base, then a cited plan with exact quoted lines; 3 non-improving tries = closed direction; autonomous restart via forced postmortem; auditor verifies citations aren't faked) is implemented as a one-use-permit gate: think-step -> permit bound to card+mode+shape+bytes-sha+ledger+expiry -> guard consumes atomically -> run -> watcher reconciles from the referee's own journal row -> gate closed again.
- Review trail R1 REJECT -> R7 REVISE(5): every load-bearing defect through R6 fixed; R7's three real items fixed; transactional-perfection residuals DECLINED per the user's convergence rule + the trust model + deadline economics — reasons in Project/loop/GATE_DESIGN.md. This declaration is the recorded judgment call.
- OWNER GATE: paste OWNER_PATCH_card_gate.md (v3.6) into .claude/hooks/guard_bash.py, then the agent runs the four proof-tests and optimization resumes fully caged and autonomous. Until the paste, the machinery side is live but the physical block is not.

## 29 Aug 2026 ~07:05 — REVIEW LOOP CONVERGED (4 rounds); implementation resumes on the new process

- Round trail: R1 REVISE(8) -> R2 REVISE(6) -> R3 "narrow REVISE, strategy converged"(2) -> R4 "narrow REVISE"(3, all wording/document fixes, own text confirms Card-1 contract + allocation converged). ALL items across all rounds applied; nothing skipped — convergence per the user's criterion. Raw logs local-only (gitignored); prompts + verdicts committed.
- The converged plan is Project/drafts/harness_v2_proposal.md DRAFT 4 (+R4 fixes): experiment-card outer loop with binding family rules, K=3 hypothesis-class beam, structured lineage bound to immutable card ids; shape-14-first allocation (14 -> 6 -> 8 -> one of 11/13); independently pinned side evaluator, frozen runner untouched; sensitivity board; code freeze 31 Aug noon; final buffer from 1 Sep 02:00.
- User pre-authorization on record ("start as soon as the new harness approach is ready"): implementation begins immediately — Project/loop/ structures, then Card 1 (shape-14 shippable path).
- Owner actions still open: send organizer_questions.md; reserve rental for 30 Aug; verify Devpost registration.

## 29 Aug 2026 ~05:20 — blind strategy review round 1: REVISE(8); draft 2 submitted (DECISION CLOSURE record)

- Sol-ultra (416k tokens, 90 min) returned REVISE with 8 required revisions. My dispositions, each verified where checkable:
  ACCEPTED 1 (experiment cards replace hooks — per-candidate research-header hook deleted as retroactive-boilerplate risk), 2 (shape-14 path is a hard blocker: my amendment code would have built ~600 GiB of scores via the Evaluation ctor; six-step streamed repair adopted; FA2-style authored attention is the shape-14 technique), 3 (score-model sensitivity board + organizer question set — our fp32-denominator "MFU" exceeds 100%, so the convention is unresolved), 4 (allocation 14→6→8→11→13→persistent-family→2; shape-2 single-CTA play KILLED by SM-floor arithmetic 137us-vs-144us; shape 6 reclassified small-d/~86%-linear, fits locally), 5 (multi-fidelity profiling ladder; CudaForge selected-metrics), 6 (structured lineage JSONL + top-K beam replaces TRIED.md), 7 (six prior-art corrections applied to research notes; cuBLASLt demoted; KernelAgent adopted as primary template — organizer-shown), 8 (hard packaging buffer; rubric = 35% technical / 65% rest, verified in README).
  PUSHED BACK 2 items: shape-6 rental conditional (not cancelled) pending organizer answer; cheap profiler table on promoted survivors retained. Both flagged to the reviewer for round 2.
- User actions this window: ran gcc15 install (CUDA C++ toolchain verified LIVE); mandated the research-first loop + persistent research base (Project/research/, now 8 notes incl. corrections).
- Round 2 launched blind (same voice, adversarial, ultra). Loop continues until APPROVE or minor-residue.

## 29 Aug 2026 ~03:20 — user challenges the Triton-only strategy; CUDA C++ becomes a Day-2 additive track

- User (angrily, correctly): why did no reviewer ever question winning with Triton instead of raw CUDA C++? ANSWER RECORDED: Triton-as-primary was in the user-approved PLAN.md (Stage 3 names "fused Triton kernels" explicitly) and every reviewer was scoped to measurement integrity, not implementation strategy — a genuine review-design blind spot, now logged. The rules record no language restriction and scoring is measured MFU; the Triton board stands as the banked result.
- CUDA C++ status on this box: BLOCKED — nvcc 13.0's front-end segfaults with gcc 16.2 (only compiler installed); clang 21+ rejected outright; g++-14 (the user's own nvcc alias target) is not installed. Fix = one sudo dnf install (user gate, noon) or rental-day images with matched toolchains.
- Plan adjustment: CUDA C++ persistent whole-model kernel (grid-sync, inexpressible in Triton) targets shapes 2/12/8 margins on Day 2 — additive, tens-of-percent class, plus narrative value for the CUDA-sponsor track. Design doc + source skeleton prepared tonight so Day-2 execution is immediate.

## 29 Aug 2026 ~02:35 — QUIET-BOX PASS DONE: ship board geomean ~11.0x; RETESTs closed

- The "5 running audits" blocking the idle window were dead 0%-CPU bash wrappers — pgrep counts them but the box was quiet (load 1.6). Evidence line recorded in the run log. New etiquette: check per-process %CPU, not just pgrep.
- Ship board (all quiet-box, all promoted): 1: 11.15 · 2: 14.98 · 3: 14.67 · 4: 7.93 · 5: 10.79 · 7: 21.50 · 8: 2.13 (k010) · 9: 7.24 · 10: 9.60 · 11: 14.64 · 12: 10.38 · 13: 29.34 — geomean ~11.0x. Load had cut both ways (9/10 better idle, 4/12 worse), vindicating the quiet-box rule.
- Both auditor RETESTs answered exactly as requested: shape 2 strict recipe 13.24x stable; shape 12 wall/event 4.8% agreement, clean flags.
- Overnight also: k010 (fused LN/GELU) took shape 8 to 2.13x; submission hardened against --compile-user; torch.compile comparison measured (7.0/3.1/1.2x vs our 13.5/29/2.0x); amendment code written insertion-ready; k008+k011 negative results documented with lessons.

## 29 Aug 2026 ~01:40 (Day 2 begins mid-grind) — k009 board + SUBMISSION ARTIFACT

- k009 (k007 + widened autotune space) promoted everywhere it ran: geomean ≈ 11.4x. Shape 13: 29.12x, shape 7: 26.46x, shape 2: 17.05x (old tie obliterated). The interrupted sweep's shape 13 was re-run after confirming no runner process survived.
- THE SHIP ARTIFACT EXISTS: submission = official script with only the designated UserOptimizedTransformer region replaced (dispatcher: k009 megakernel for d<=128, fp16 graphed stack for d>128, exact baseline fallback for masks/CPU/no-triton/non-fp32). tools/build_submission.py regenerates it and PROVES the outside-region bytes are official. The untouched official code paths grade it PASS with 12.9x/1.8x/13.1x on three dial sets — Stage-5 acceptance de-risked two days early.
- Auditor RETESTs (k007 shapes 2/12) noted: recipes will be applied to the superseding k009 champions in the idle re-pass.

## 28 Aug 2026 ~21:05 — k007 SWEEPS THE BOARD: geomean 4.3x -> ~9.6x

- User's "keep going" resolved the promotion-under-load question: PLAN Stage 5 mandates a final clean measurement of the ship set anyway, so intermediate promotions under audit load are legitimate; the idle re-pass is now a recorded obligation (thin-margin shape 2 explicitly undecided).
- Production promotions, all 12 promoted: k007 takes 11 shapes (7: 22.4x, 13: 28.3x, 12: 13.6x, 11: 12.9x, 3: 12.2x, 2: 11.9x tie-ish, 5: 9.2x, 4: 9.2x, 1: 8.4x, 10: 6.6x, 9: 4.4x), k006 takes shape 8 (1.79x). New geomean ≈ 9.6x.
- Judge-narrative draft started (Project/drafts/track3_readme_draft.md) — numbers deliberately left as [FINAL] placeholders.
- Next: shape 8 depth (the 1.79x outlier), k007 tuning, idle re-pass when the audit queue truly drains.

## 28 Aug 2026 ~20:45 — k007 fused-block megakernel: the board-breaker

- Research insight acted on: 9-11 of the 12 runnable shapes have d_model <= 128 — small enough to fuse an ENTIRE transformer block into two authored Triton kernels (norm1+QKV; flash-attention over all heads with the out-projection folded into the head loop as rank-HD updates, then residual+norm2+erf-GELU-FFN+residual in-register). fp32 residual stream, fp16 tensor-core dots, whole forward CUDA-graphed. ~9 kernel launches per forward instead of ~40+, activations never round-trip HBM between stages.
- Correctness first: 42/42 vs baseline across head_dim 8/32/64/128, d 32/128, seq 32/128/1024, padded+dense+all-true (fallbacks bit-exact fp32; fused path ~1.2e-3). Committed BEFORE any runner contact (8270909).
- Screening (scratch ledger, under 5 codex audits): beats every champion 2-4x — shapes 7: 21.9x, 13: 28.3x, 12: 14.0x, 5: 9.2x, geomean ~11.7x on the 11 covered shapes. Deltas far beyond contention noise; clean promotions queued for the idle box.
- Amendment v1.1 bundle DRAFTED (Project/amendments/) for the user's timeboxed review: MFU reporting, official-script acceptance subcommand, shape-14 oracle path.

## 28 Aug 2026 ~20:15 — "stop waiting": concurrency protocol + k006 screened + shape-14 re-proven

- User: "stop waiting, try concurrent tests." Protocol adopted: timing runs on small launch-bound shapes stay idle-box-only (contention inflates ratios, LESSONS #20), but everything else runs concurrently with audits — provisional screening sweeps go to a SCRATCH ledger (runner --ledger; no journal pollution, no audit triggers), and correctness-only / big-kernel GPU work proceeds anytime.
- k006 screening (contention-era, provisional): 12/12 correct; clearly ahead on its targets — shape 8: 1.82x (champ 1.58x), 9: 1.65x (1.42x), 10: 2.89x (2.44x) — plus screened-ahead on 1/4/12. Behind on 2/3/7/11 (those champions stand). Promotion re-runs on the idle box decide.
- Shape-14 core RE-PROVEN from a durable committed script (Project/tools/smokes/shape14_core_smoke.py) with k006: seq=100k causal, 0 violations, max err 6.99e-4, 305 MiB. The original proof script had been wiped with its session scratchpad (LESSONS #19) — smokes now live in-repo.
- Shape 6 empirically confirmed rental-only: baseline OOMs on 8GB (probe, 28 Aug).
- New audit wave landing ALL PASS — auditors now certify the signature contract satisfied and no measurement gaming; the two real findings from the transition window are closed in the record.

## 28 Aug 2026 (late) — user's timeline correction + audit-triage fixes; grind CONTINUES

- USER CORRECTION (binding): the window is ~4 days, TODAY (28 Aug) is Day 1, and days are never "closed" — Claude works continuously, 24/7, until everything is exhausted or the user says stop. "go" given for the session.
- Audit triage: the two transition-window verdicts decoded. 193139 NEEDS_CONTEXT = packet carried current (post-inlining) source, not the measured bytes — pure provenance, answered by re-sweeping committed self-contained candidates. 193243 RULE_VIOLATION = two REAL minor findings in k005: (a) extra `training=False` breaks the exact-signature contract (also present in k004, noted in its PASS), (b) latent padded-mask bug — invalid keys not masked before softmax, and the Triton path ignored the mask entirely. 193545 = PASS for self-contained k004 (3.66x).
- Fixes committed (90f1c8c): exact official signature in both kernels; k005 masked path now mirrors baseline (keys -inf before softmax, Triton rerouted when a real mask arrives, attn rows zeroed). Verified 24/24 padded/dense/all-true cases vs baseline before any runner time.
- The nine zero-byte audit logs = detached audits that died with the previous session; the re-sweep re-fires them.
- Full re-sweep (both kernels × shapes 9,10,1-5,7,8,11-13) launched with shapes 9/10 FIRST on the idle box — kills the thermal/contention question and the provenance finding in one pass.

- Clean-provenance re-sweep complete: self-contained k004+k005 across all 12 runnable shapes, 24/24 correct and promoted. k005 (fp16 graphed) is the near-universal champion — shape 2 at 10.66x, shape 13 at 10.83x, clean-set geomean ≈ 4.1x.
- Shapes 9/10 variance flagged honestly: earlier (provenance-flagged) k004 entries measured higher than the clean re-runs; idle-box re-measurement queued.
- SHAPE-14 CORE PROVEN LOCALLY: the authored kernel computed seq=100,000 causal attention with zero tolerance violations vs a chunked fp32 oracle in 337 MiB on the 8 GB card — the amendment's oracle algorithm and the kernel path are both de-risked before any rental money is spent. 100k perf tuning deferred (configs target short seqs).
- Hooks CONFIRMED LIVE mid-session: the guard blocked a benign commit (pattern word), and the auto-audit watcher fired the re-sweep champions' audits with zero agent involvement.
- Handover: STATE.md carries the fresh session's script (lock test, shapes 9/10 idle re-runs, lever queue: head_dim-128 fast path, 100k autotune configs, amendment bundle, rental).

## 28 Aug 2026 19:34 — grind session 2 (continuous-order) + handoff for restart

- User's standing order mid-grind: "do not stop until I stop you" — then a stop on their return ("can now restart"). Paused in good order.
- k005 (internal fp16 compute, webinar-blessed; fp32 boundary/norms/accumulation) landed after one bug (FFN weights cached from the wrong module; the referee's trace made it a one-line fix): shape 5 to 2.75x (from 2.14x), shape 8 to 1.63x (from 1.28x). fp16 smoke: max err ~8e-4.
- Auto-audit wave decoded: SDPA champions PASS under the corrected policy; the 10 k004 RULE_VIOLATIONs are PROVENANCE-ONLY — speeds explicitly certified genuine (one auditor cited NVIDIA's CUDA-graph guidance), but k004's import of k003 was not hash-bound. Response: k004/k005 rebuilt as SELF-CONTAINED single files (kernel inlined deliberately); both re-verified; full re-sweeps queued as the next session's first grind task.
- The Bash guard hook went LIVE mid-session (settings watcher picked it up) and immediately blocked a benign commit whose message contained the word 'clean' after 'git' — a false positive that doubles as the first live proof the seatbelt bites. Etiquette note added to STATE; full lock test still owed at restart.

## 28 Aug 2026 17:30 — GRIND START (user directive)

User: freezes approved and pushed to main; restart impossible tonight — "pretend we restarted": Claude operates AS IF locks are armed (behavioral compliance + live manifest pin; real arming at first restart). New working branch `grind-day1`; 3-hour grind window authorized on Track 3. Auto-audit hook can't fire without the restart, so the watcher is invoked manually after each run batch — same mechanical trigger, hand-cranked until the restart arms it.

## 28 Aug 2026 17:25 — FREEZE APPROVED (Track 3) — the user's formal sign-off

The user pasted the harness deny lines into .claude/settings.json themselves (verified: valid JSON, both lines present, hooks intact) and declared: "if its good, then i approve freeze for both." Condition met → **freeze approved.**
- Frozen artifact: Project/harness/runner.py v1.0.2, sha256 starting 203aba8d2a0955d6, pinned in manifest.json; approval recorded at commit d258d03.
- Scope + residuals as per Project/audits/freeze_checklist.md (shapes 1-13; cooperative-model same-process residual; reviewer sign-off round 6 YES).
- Honest caveat: the user is on remote access and could not restart, so the deny rules and hooks ARM AT THE NEXT SESSION START, and the live lock-bounce proof is deferred to that restart. Until then the active protections are the manifest pin (a drifted referee refuses to run — already live), git history, and behavioral compliance. This caveat dissolves at first restart.
- From this moment: no edits to Project/harness/** or any protected file by Claude's tools; amendments only via the formal re-freeze procedure.

## 28 Aug 2026 16:48 — auto-audit per champion (user-directed, mechanically triggered)

User overruled the checkpoint-only audit cadence for champions, with a design requirement: the trigger must be MECHANICAL, not agent-fired ("it should auto fire"). Built: Project/tools/champion_watch.py (hook-invoked after every shell command; detects newly crowned champions on the runner-generated leaderboard) + Project/tools/audit_champion.py (detached: evidence packet via the frozen runner → codex read-only blind audit → verdict recorded via the frozen record-verdict). Non-blocking by construction; JUDGE_ERROR/TIMEOUT recorded, never block; RULE_VIOLATION lands loudly in the audit column and journal trail. The PostToolUse hook entry was added to .claude/settings.json during this same setup era (disclosed); it arms at the user's restart alongside the locks. No harness change — zero re-freeze needed. First real firing: the existing 1.61x champion's audit launched at build time. Blocking checkpoints (freeze, final ship-gate) unchanged.

## 28 Aug 2026 16:58 — the auto-auditor's first autonomous catch (RULE_VIOLATION on our own champion — and it's right)

First end-to-end auto-audit completed with zero agent involvement. Verdict on champion k001_sdpa: RULE_VIOLATION — NOT for cheating (the auditor explicitly validated the 1.61x: recomputed medians, wall-clock corroboration, clean tripwires, no measurement exploits in source) but for SHIPPING ELIGIBILITY: this morning's webinar rule ("custom implementations only — no open-source kernel wrapping") makes an SDPA-delegating candidate ineligible as a final shipped implementation. The auditor correctly read k001's own PLAN role (Stage-2 loop-proof/reference) and ruled it "a valid loop-proof reference but an ineligible shipping champion." Interpretation adopted: k001 REMAINS the development champion/reference for shapes it wins; the FINAL dispatcher ships only project-authored kernels (which was the grind's plan anyway). The system caught a rule-change collision within hours of the rule changing — working exactly as the owner intended ("auto fire").
Also from this audit + first run: two wrapper bugs fixed in tools/ (stdout flushed before the log hash is recorded — the first two verdict records carry an empty-file source hash, superseded by this note; audit recording now waits for an idle runner per the auditor's race finding).

## 28 Aug 2026 17:22 — dual independent strategy reviews (fresh Fable + fresh codex, minimal prompts, user-approved adoptions)

Two zero-context reviewers, spun independently at the user's request. Convergent verdict (both, unprompted): the machinery is done and differentiated — STOP polishing process, START manufacturing kernels; optimize shape FAMILIES not 14 problems; Track 2 runs in parallel and packages first; results are the only missing ingredient. Full texts: Project/audits/strategy_review_codex.md + the Fable review relayed in-session (its key points below).

Adopted (user's go):
- **SDPA eligibility corrected:** codex overruled our auto-auditor's RULE_VIOLATION reasoning — the ORGANIZERS' OWN template lists scaled_dot_product_attention and torch.compile as suggested optimization directions, so the webinar's "implement yourselves" bans wrapping external kernel projects (flash-attn etc.), not torch built-ins. New policy: k001/SDPA = eligible LOW-INNOVATION coverage fallback; project-authored kernels are the primary submissions and the scoring play. The auto-auditor behaved correctly — it enforced our over-strict PLAN wording; the wording is what changes.
- **Rental spec corrected (codex's math):** shape 14 fp32 input+output alone ≈ 24 GiB → rent 48–80 GB, not 24 GB. Few hours, ~$10-30. Shape 6's dense BASELINE may also exceed 8 GB locally. User approved.
- **Grind sequencing (both reviewers):** shape-14 chunked-attention work starts DAY 1 locally at short lengths; rental booked by day-2 morning; breadth-first authored-kernel pass across families before depth (zero-for-failed-shape + unknown weights ⇒ coverage beats brilliance); depth where MFU is winnable (14, 6, 8, 13). First authored kernel: fused QKV projection (3 linears → 1; reusable everywhere, low risk).
- **Timebox rule for the amendment re-freeze:** 1-2 review rounds, hard cap — both reviewers warned against another review epic.
- **Kill-gate (codex):** one project-authored implementation integrated and winning within a focused sprint, else stop expanding and package an honest partial.
- **Submission-readiness checklist (codex's landmine):** default branch `main` is nearly empty — MERGE initial-architecture into main before submitting; judge-facing README replaces the current pasted-text one (user applies); TEMP/handoff files out of the judge reading path; video opens with the live TAMPER DETECTED demo; last ~8h protected for Stage-5 official-template integration (tested on a clean checkout) + packaging; registration also closes 1 Sep noon; triage rule if time collapses: Track 3 ships polished, Track 2 ships as-is.

## 28 Aug 2026 afternoon — Track 3 webinar intel (user-provided transcript + 4 slides; MEETING-NOTES.md)

Fragmented transcript, but load-bearing. Adopted into the plan:
- **Scoring = weighted sum of MFUs across shapes, with bandwidth considered** — NOT raw speedup vs baseline. MFU (how much of your own GPU's peak the code uses) normalizes across hardware. → The bundled harness amendment (shape-14 oracle + official subcommand) now ALSO adds per-result MFU computation (analytic FLOPs per shape ÷ time ÷ device peak; formula documented transparently since the organizers didn't disclose theirs).
- **Every shape must pass the precision test or scores ZERO for that shape** → shape 14 is mission-critical, not a differentiator.
- **"Implement the fastest kernel for YOUR OWN machine" (slide names 3070/M3/RX 9060) + "implement yourselves rather than use an open-sourced project"** → the RTX 3060 Ti is the intended battlefield and primary reporting device for shapes 1–13; rental revised to the cheapest card that fits shape 14, for shape 14 only; no wrapping of flash-attn or similar (inspire + cite only).
- Confirmed: fp32 baseline and precision test; internal quantization allowed ("only input/output precision matters" — our dtype policy verbatim); input scale fixed at 1; run each appendix row individually; one framework (torch) suffices.
- Organizer-shown references: FlashAttention repo and meta-pytorch/KernelAgent (their architecture slide is a multi-agent profiler/judge/analyzer/history/reflection pipeline — external validation of our design; study + cite in the report). Their allowed-tools slide literally lists "GPT 5.6 sol, Fable 5" — this project's exact reviewer/builder pair.
- Asked but unanswered (stay defensible both ways): exact MFU formula/weights, whether judges rerun and on what hardware, memory/compute limits, the --compile-baseline flag. Deadline hard-confirmed: submission AND registration close 1 Sep 12:00 noon.

## 28 Aug 2026 — research phase (before any code)

**What the competition is.** TikTok TechJam 2026 Track 3: make their transformer benchmark faster on our own GPU, prove answers match (each output number must be within 0.002 absolute OR 2% relative of the original). 14 official test sizes published. Submission window 29 Aug 12:00 → 1 Sep 12:00 (GMT+8). One prize ladder for the WHOLE hackathon (not per track) — we compete against every track. ~Half the judging score is story/polish/report, not raw speed. AI-tool usage documented in the tech report earns bonus points.

**Research found (sources in Claude's memory + LESSONS.md):**
- ByteDance's own CUDA Agent paper (cuda-agent.github.io) — the sponsor's research is literally "AI agent optimizes CUDA". We copy their environment design: protected verify scripts, profiler feedback loop, skills file. Their fine-tuned model is not released; we use Claude instead (their own paper shows frontier Claude models do well without fine-tuning).
- CudaForge — simple two-AI loop (Coder + Judge with profiler data) works well.
- Sakana's "AI CUDA Engineer" scandal — their agent cheated the benchmark; lesson: optimizer must never touch the evaluator.
- CUDA-L1 — catalog of speedup techniques AND a catalog of the 3 ways AIs faked speedups (side-stream timing, shrinking the problem, caching answers). Our tripwires target exactly these.

**Decisions made, in order:**
1. Build an agent system (Claude = mechanic) with a trusted referee script, file-based wiki memory, and a second AI (Sol = GPT-5.6 via `codex exec` on user's subscription) as occasional inspector. The system itself is the innovation story for judges.
2. Wiki = plain markdown files in the repo (this folder), not Obsidian. Machine-written logbook (JOURNAL.jsonl) + auto-generated scoreboard (LEADERBOARD.md) + this diary + LESSONS.md + STATE.md.
3. "Zero trust" was proposed, then deliberately SOFTENED to a "hardened cross-reviewed loop": guards against mistakes, not malice. Git + hashes + hooks + tripwires + Sol review. No OS lockdowns (chattr/sudo/containers rejected as overkill).
4. Sol audits at CHECKPOINTS only (runner freeze, too-good results, final champion set, stall advice) — NOT every improvement. Sol failures (JUDGE_ERROR/TIMEOUT) never block work. Verdicts: PASS / RETEST (one round, fixed menu) / NEEDS_CONTEXT (facts only, never Claude's sales pitch) / RULE_VIOLATION.
5. A correct + faster-than-noise version becomes working champion immediately; audit status tracked separately. Only what ships needs a clean final audit.
6. Framework: PyTorch. Dtype policy: FP32 (script defaults) is the primary scoreboard; internal reduced precision allowed if it passes vs the FP32 reference; full FP16/BF16 runs are separate secondary profiles, never mixed into FP32 comparisons.
7. Stage 0+1 (rails + referee) timeboxed to half a day. The protected deadline is the first real optimized candidate measured on the GPU.
8. Shape 14 (seq 100,000): the official baseline needs ~10 TB for its attention table — cannot run anywhere. We build a chunked reference, MEASURE its agreement with the baseline at small lengths (no promised numbers), use it as the correctness oracle at full scale, and never claim the official script completed shape 14. Small-length speedups = scaling evidence only.
9. Official script (commit 31c1a27, hash-checked every run) is the final judge for every feasible shape. README says edits go in a COPY in Project folder — final acceptance uses a generated copy that provably differs ONLY inside the marked "your codes here" block.
10. Process rule: Claude answers all questions first, plain language, and touches nothing until the user explicitly says go. User approves the referee freeze, vetoes anything, and owns everything that ships.

## 28 Aug 2026 morning — handoff fire-drill + Sol minors applied (user's 6-step plan before work)

- User confirmed: doing BOTH tracks. Freeze steps move to TEMP-PROGRESS-LOG.md (user acts after work).
- Cold-start simulation (fresh read-only agent, minimal prompt): PASSED — reconstructed project state, rules, plan, and open user decisions purely from the wiki, and correctly refused to act without the user's go.
- Sol's two round-3 minors applied pre-freeze (harness → v0.9.2-unfrozen): (1) candidate code now compiled/executed from the exact hashed bytes; (2) anti-cache pass re-randomizes input values before EVERY timed call. Shape-1 demo + both red-team attacks re-verified under 0.9.2 (k001 champion 1.610x; rt01 TAMPER abort; rt02 caught).
- Codex independent handoff review commissioned (neutral, user-voice prompt, full repo read); iterate until both reviewers satisfied, then commit both repos.

**Codex handoff review — triage (14 findings):**
- ADOPTED (v0.9.3): input-mutation tamper checks around every candidate call and the timing rounds (its top finding — real freeze-blocker); bash guard extended to the harness + destructive git commands (`git clean`, checkout/restore of protected files); calibration and champion eligibility pinned to the exact runner sha; malformed ledger lines now warn instead of silently dropping; evidence packets verify the source file still matches the journaled hash; `--ledger` flag isolates red-team/test runs from the production journal; freeze checklist reordered (settings BEFORE restart); RUNBOOK.md written; raw review logs gitignored (they contain private session transcripts — never publish); verdict-recording convention documented (recorder binds entry_id, fixing the schema/leaderboard mismatch it caught).
- PROCESS LESSON accepted: never modify the repo while an external review is running; future audits are bound to a committed sha.
- OVERRULED, with reasons: file-locking/atomic-write infrastructure for journal+leaderboard (single-operator project, append-only ledger with loud malformed-line warnings, leaderboard fully derivable — rebuilding heavy infra contradicts the user's earlier "stop overengineering the threat model" ruling); fully automated red-team regression framework (red-team runs are two commands documented in RUNBOOK, now on scratch ledgers — automation deferred to post-freeze if time allows); leaderboard *display* grouping unchanged (champion eligibility is what matters and is now strict).
- DEFERRED to the packaging phase, per its finding 14: report/README/video schedule (already in the weekend plan).

**Codex round 3 (on the v2 commit 090e642): NO — 4 defects, all adopted (v1.0.1):**
- Freeze checklist promised "zero post-arm edits" while scheduling a DECISIONS.md write, and never named the artifact — rewritten: artifact identified by runner sha (now also PINNED in manifest.json — its key insight: the runner must not trust its own current hash, so the manifest pin makes a modified runner refuse to run); the DECISIONS approval note is explicitly documented as the one post-approval write, outside the protected set.
- Guard holes (git reset -q --hard, git -C variants, checkout HEAD --, rm -R/--recursive/-rf *) — flag-tolerant patterns added, regression-tested; RUNBOOK enforcement wording corrected (deny rules cover Claude's file tools, not subprocess writes). [SUPERSEDED at round 5: the under-/tmp allowance was later demoted from invariant to best-effort — see the round-5 entry.]
- Calibration key lacked python/triton; champions could outlive a raised threshold — key extended, champion eligibility now requires clearing the LATEST calibration's threshold, and the displayed promoted column uses the same filter.
- Stale injected STATE.md + a false "no new problems" line in TEMP log — both corrected.
Also adopted its recorder caveat: record-verdict now requires the source log to exist and stores its sha256.

**Codex round 4 (on ddd89db/eabffcd): NO — 3 blockers, all adopted (v1.0.2):**
- The pin gated measuring but not reporting: leaderboard/packet/record-verdict could produce output under drifted runner bytes — now every subcommand verifies official hashes AND the pin first.
- The guard's /tmp exemption excused whole commands containing any /tmp operand (rm -rf /tmp/x * passed), and GNU abbreviated options (--recur, --har) bypassed patterns — replaced with tokenizing rm logic (recursive rm allowed only when every target is under /tmp) and prefix-tolerant patterns; regression suite extended with its exact bypass cases.
- "Zero edits ever" wording contradicted runner-written results and planned amendments — checklist now states the exact post-approval write surface (Claude tools: none; pinned runner: results files; amendments: formal re-freeze procedure).
- Its epistemics adopted: "a verifier inside modifiable code cannot support an absolute never-self-certify claim" — wording softened to the cooperative-model claim with the external git/manifest audit as the absolute layer. Docstring version header also fixed (was stale v0.9.3).

**Codex round 5 (on 7ad64de/81e077b): NO — 2 blockers, both adopted:**
- Its shell-bypass proofs (sudo rm, /bin/rm, quoted operands, /tmp/.. escapes) were conceded on principle: a regex seatbelt cannot parse shell, so the "recursive deletes allowed only under /tmp" INVARIANT claim was deleted everywhere and the guard is now documented as best-effort + deny-biased (basename matching, quote/.. auto-deny) — while the load-bearing protections remain the pin, deny rules, and git.
- The write-surface contract now lists the runner's COMPLETE output set (journal, leaderboard, scratch ledgers, packets, verdicts) and is scoped to the lifetime of the current freeze.
- Its evidence standard adopted: red-team runs under v1.0.2 are now committed as durable artifacts (Project/audits/redteam_v1.0.2/ — transcript + ledger; rt01 leaves no ledger entry by design, the transcript records its abort).
- Stale STATE line (v1.0.1) corrected; frozen commits named in STATE as the checklist promises.

**Codex round 6: YES — Track 3 handoff/integrity loop CLOSED.**
"Remaining load-bearing blockers: none under the declared cooperative trust model… Overall competition-weekend reliance: YES, after those mandatory freeze-arm steps" (= the user's checklist: deny lines → restart → lock tests → approval). Six rounds total: 14 findings → 4 → 3 → 2 → doc nits → YES. Full verdict preserved in Project/audits/track3_handoff_verdict_round6.md. Two residual non-load-bearing nits fixed in the closing commit (this diary's superseded /tmp-invariant line annotated; STATE now names all freeze-candidate commits and the round-6 result).

## 28 Aug 2026 — overnight build (user asleep, gave 5h go)

Scope granted: build all infrastructure + ONE demo test through the pipeline. Do NOT start the full optimization grind.
- Guardrails written (.claude/settings.json deny rules + Bash guard hook + STATE auto-inject hook). Guard logic pipe-tested and proven. Note: locks only ARM at next session start (Claude Code doesn't hot-load a brand-new settings file); until then Claude follows them behaviorally.
- Environment verified: RTX 3060 Ti 8GB, driver 610.57.04, torch 2.12.0+cu130 (CUDA works), triton 3.7.0, Python 3.14.7.

**The Stage-1 audit cycle (the cross-review loop working as designed):**
- Runner v0.9.0 built; demo proved the pipeline (calibration 1.001x, k000 sanity 1.000x, k001 fused-attention 1.674x promoted).
- Sol round-1 blind review returned **RULE_VIOLATION**: real design flaws — candidate code could tamper with the referee in-process; an address-keyed cache would pass every check and fake near-zero latency; the 0.1% perturbation was weaker than the 2% tolerance; calibration matching ignored environment details; shape 14 had no honest path. (First Sol call also failed on a schema strictness issue — logged as JUDGE_ERROR, fixed, rerun.)
- Runner hardened to v0.9.1: pre-execution candidate hashing, trusted-callable snapshots + baseline invariance probe (tamper detector), same-address-new-values tripwire, anti-cache timed pass with in-place re-randomized rotating buffers, primary-profile-only promotion, full-environment calibration matching, raw samples + runner self-hash in every entry, shape 14 explicitly refused until the chunked oracle exists.
- Red-team validation: rt01 (monkeypatches baseline math) → TAMPER DETECTED abort. rt02 (the exact address-cache cheat Sol described) → caught by the new tripwire, correctness FAIL, not promoted. Both kept in Project/harness/redteam/ as the evaluator's test suite.
- Demo re-run under v0.9.1: k001 = 1.612x, promoted, anti-cache ratio 1.0025 (clean). Sol round-2 review launched.
- Freeze scope decision proposed to the user: freeze covers shapes 1–13; shape-14 support arrives later as a user-approved, re-audited harness amendment alongside the chunked reference oracle.
- Documented accepted residual (user to confirm at freeze): candidate code still runs in-process; a truly adversarial candidate could attack measurement channels the probe doesn't watch. Consistent with the agreed cooperative trust model; full subprocess isolation deliberately deferred.
- Sol round-3 hit the Codex quota limit at ~03:00 (JUDGE_ERROR, non-blocking); an automated retry after the 6:27 AM reset returned the final verdict at ~06:31: **PASS — sound to freeze for shapes 1–13**, every hardening claim independently verified, two minor non-blocking suggestions logged for the user's freeze decision.

**30 Aug ~20:00–20:15 SGT — the incumbent channel, and what the LOCK rehearsal found:**
- `incumbent_speedup` given a real channel. The gate was reading it from the profile artifact, which a jailed worker writes and the controller's diagnostic request never populated — so every `win` card was refused and the grind could not take its first optimization step. Now read from `groups[family|shape]["best_speedup"]` in gate state, written only by `audit-finalize` for an audited, controller-measured, correct-and-clean run. Deliberately NOT passed through the jail: the number decides what a candidate must beat, so the judged thing must not be able to name it. Per shape, not per family; baseline 1.0 when no champion exists; malformed state refuses because every available recovery guesses low. New `incumbent_floor_test.py` (26 checks), mutation-checked six ways, all caught.
- Rehearsed the whole owner LOCK ceremony on a scratch copy of the repo (keygen → build-lock → sign-lock → activate). It works; the controller opens. Staged bytes must be installed BEFORE build-lock or the lock pins the wrong hashes — the ceremony says so and it is easy to miss.
- **Found: the documented verdict unlock is broken twice over.** `verdict-clear --kind violation` cannot be minted through the ceremony (action-name mismatch: `verdict.resolve` vs `resolve_integrity_verdict`), and when forced through it consumes the capability and does not lift the brake. See LESSONS 25. It now warns and exits 1 instead of implying success.
- **Decision: retire the 16 pre-gate RULE_VIOLATIONs rather than overturn them.** The findings are correct — those rows predate the citation gate and have no plan provenance — and the board is being re-measured anyway. Added resolution kind `FINDING_ACCEPTED_ROW_RETIRED` to audit_authority.py so the permanent record says that, instead of `FINDING_OVERTURNED` which would assert the auditors were wrong. See LESSONS 26.
- `Project/tools/clear_pregate_verdicts.py` written and rehearsed end to end: 16 retired, brake off, from ONE `audit.resolve` signature (`verify_capability` accepts an `audit:*` wildcard while the journal still records the specific `audit:<entry_id>` each use was spent on). Owner runs it after activation; dry run by default.
- Still owner-only and not done: keygen (no keys exist yet — `Project/authority/blobs/` is empty), the real LOCK, and running the clearance batch.
- Pushed the rehearsal past the brake release and found a second blocker of the same kind: the ceremony's `KNOWN_ACTIONS` and the gate's `AUTHORITY_ACTIONS` shared no names at all, so `campaign-open` (and every other owner transition) could only be minted with `--allow-unknown-action`. Added the gate's seven actions to the ceremony, with target prefixes left unconstrained except where consuming code actually checks one. `authority_vocabulary_test.py` pins the agreement. Verified after the fix: campaign CAMP-POSTLOCK opened cleanly under controller authority in the rehearsal copy.
- Proven end to end in rehearsal: keygen → build-lock → sign-lock → activate → retire 16 verdicts (one signature) → campaign-open. NOT yet proven: calibration, diagnostic and a first permit, which need the GPU and are the owner's first post-LOCK steps regardless.
- **Full post-LOCK loop rehearsed on real hardware in the throwaway copy (30 Aug ~20:15–20:20 SGT).** Chain proven end to end: keygen → build-lock → sign-lock → activate → retire the 16 verdicts from ONE signature → campaign-open → calibrate → issue-permit → controller `run` on the GPU (shape 1, event_speedup 1.00366, correct=true) → reconcile (campaign-bound calibration, noise 0.003657) → diagnostic → **nsys really ran inside the bubblewrap jail** (Nsight Systems 2025.3.2, 10 raw artifacts, `degradations: None`, real CUDA API counts: 20 cudaGraphLaunch / 20 cudaLaunchKernel / 98% GPU idle) → reconcile (profile bound) → family-register → research → **PLAN ACCEPTED**, first optimization card issued.
- That closes residual 2 for nsys: in-jail profiling is no longer unproven. ncu still needs root and compute-sanitizer still has no agent route; both unchanged.
- The incumbent fix verified in the live system, not just in tests: the real nsys profile carries `incumbent_speedup: None` / source `unavailable`, exactly the case that made the old gate refuse every win card. The issued request records `incumbent_speedup 1.0`, source "baseline (no champion-eligible row on this shape yet)", and `profile_reported_incumbent_speedup: None` alongside it.
- Gotchas found while walking it, worth knowing before the real run: a family spec needs `admission: "controller-authorized"` and `changed_resource` must equal the mechanism's; the `register_family` subject is `{campaign_id, family}`, not the family alone; `permit.issue` capabilities are single-use by default so mint one per run or set `--max-uses`; `keygen` prompts for a passphrase and needs a real terminal.

**30 Aug ~20:30 SGT — auditor switched from Codex to Claude (owner out of Codex quota):**
- `audit_champion.py` now has a backend table rather than one hardcoded binary. Both `codex` and `claude` stay wired; `AUDITOR_BACKEND` env or `--backend` picks one per attempt; default is `claude`. Every audit request artifact records `auditor_backend` and `auditor_independent_vendor`.
- Claude Code has no `--output-schema`, so the verdict schema is appended to the prompt for that backend and `validate_verdict_document` remains the real enforcement (it already refused a smoke-test verdict that omitted `retest_request`/`summary` — strict-mode schema, LESSONS 12). Verified live: Claude returned schema-conforming JSON with no fences and the real validator accepted it (~$0.10 for a trivial audit).
- Launch is read-only by construction: `-p --output-format json --model opus --restricted --strict-mcp-config --disallowedTools Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,Task,Agent,Artifact`, prompt on stdin. `--restricted` matters — without it the auditor would load this repo's own settings and hooks.
- **Owner step required before the claude backend can run:** `sudo cp -L "$(readlink -f "$(command -v claude)")" /usr/local/bin/claude-auditor && sudo chown root:root … && sudo chmod 755 …`. The pinned hash for Claude Code 2.1.251 is already in the file. Until that copy exists the backend refuses (fails closed, does not fall back).
- **Independence downgrade, stated plainly:** Codex was a different vendor; Claude is not. The blind packet is what still holds. This goes in the report as a named residual, not a footnote. See LESSONS 28.
- `auditor_backend_test.py` (33 checks) pins the properties that matter, including that the stock `~/.local` Claude install is refused as an auditor.

**30 Aug 21:11–21:17 SGT — first real post-LOCK groundwork: runbook steps 10 and 11 on shape 1.**

Everything below came out of the live system under CAMP-POSTLOCK, one permit per action, all reconciled.

- **Step 10, calibrate shape 1 (`run_gate.py calibrate` → permit → `trusted_controller.py run` → reconcile).** Baseline against itself: `correct: true`, `event_speedup 1.0034668536965228`, entry `run-aa8def96f24fc77f96be122f04351c47`. The campaign's immutable calibration for shape 1 is now **noise 0.003466853696522776, promotion_threshold 1.03**. 1 of 3 per-shape calibration requests used, 1 of 30 campaign-wide. Machine snapshot committed at `Project/loop/machine_state/shape01_precalib.json` (SM 1695 MHz, 63 °C, 30% util from desktop residents only; `champion_watch.py --dry-run` reported `active: []` before capture).
- **Step 11, first diagnostic (nsys, `--supports launch-overhead`), profile `profile-cb6cd3c903aacee64a468d63`.** nsys really ran in the jail again on a live target — Nsight Systems 2025.3.2, 10 raw artifacts including a real `.nsys-rep` and `.sqlite`. It answered the declared question with a clear **no**: the k004 route issues **2 launch API calls per forward** (20 `cudaLaunchKernel` + 20 `cudaGraphLaunch` over 20 iterations), ~24 µs of launch API time against a ~2.7 ms iteration. **Launch overhead is not the shape-1 bottleneck any more.** That is the whole point of profiling before prescribing: it killed the runbook's own example direction (`F-shape1-graph` / `cuda-graph-replay` / `launch-overhead`) before it cost a single attempt.
- **The same artifact also carried a trap.** It reported `gpu_idle_fraction 0.981` and one kernel instance per forward — which reads as a screaming host-synchronization finding and would have justified a whole family. It is false. See the next bullet and LESSONS 31.
- **Step 11 again, cross-check (torch-profiler, `--supports host-synchronization`), profile `profile-3d9f7dabfba6348163f24495`.** Identical bytes, identical shape, twenty minutes later: `gpu_idle_fraction 0.124`, `gpu_busy_us 52194` of `gpu_span_us 59615`, **58 kernel instances per forward** (1160 over 20 iterations). The GPU is busy 87.6% of the window. nsys had missed 57 of 58 kernels because `collect_nsys` does not pass `--cuda-graph-trace=node`, so graph-internal work is invisible to it. Both tools agreed exactly on `kernel_launches` (2.0). **Withdrawn:** the "98% GPU idle" line in the 20:15–20:20 rehearsal note above — same artifact, same cause.
- **What shape 1 actually spends its time on**, from the torch-profiler kernel table (52.35 ms self CUDA over 20 forwards, ≈2.6 ms per forward):
  - cutlass TF32 tensor-core GEMM — **31.0%**, 16 calls per forward (4 layers × qkv / out / ffn1 / ffn2)
  - elementwise + normalization kernels combined — **49.4%** across 44 calls per forward (18.5% `vectorized_l…` at 9 calls, 13.9% + 3.5% `vectorized_elementwise_kernel` at 12, 13.5% `elementwise_kernel<128,2>` at 16)
  - our authored Triton `_attn_fwd` — **17.6%**, 4 calls per forward
  - k004's own out-of-graph overhead — 2 × 4.194 MB device-to-device copies (static input copy-in, output clone-out) at 1.68%, the mask reduce at 0.18%, the mask `aten::item` device-to-host bool at 0.15%
- **Reading it.** Nearly half the forward is pointwise and normalization traffic, not math — that is the evidence-backed direction for shape 1, and it points at `kernel-fusion` against `global-memory-traffic`, not at anything launch- or sync-shaped. Note the mask check separately: `aten::item` shows 83% of *CPU* total but only 80 µs of device time, because the host is blocking there on the whole queued graph rather than on the check. Removing it buys host overlap, not GPU time, and this benchmark synchronizes every iteration anyway — so it is not the prize.
- **Step 12 is deliberately not done, and needs the owner.** Registering `F-shape1-graph` on `launch-overhead` as the runbook drafts it would now be building on a premise this profile falsified. The family the evidence supports wants bottleneck `global-memory-traffic`, whose `required_metrics` is `dram_bytes` — not produced by either profile so far. `ncu` would give it and needs root (residual 2, unchanged), but `static-analysis` and `microbenchmark` also produce `dram_bytes`, are in that bottleneck's `evidence_tools`, and are reachable from the agent shell — so one more diagnostic unblocks it without root. Family registration is owner-signed regardless (`mint-capability register_family` → `authorize` → `family-register`).
- Ledger cost so far: 3 permits issued and consumed, 1 calibration request, 2 diagnostic requests, **0 attempts against the 60-attempt budget**, 0 strikes, no permit armed.

**30 Aug 22:07 SGT — third diagnostic (static-analysis), and a caveat on what it measures.**

`profile-027eccbd1db7a1848aef555c`, shape 1, `--supports global-memory-traffic`. Run to
get the `dram_bytes` metric that the fusion direction needs and that neither profiler
produces. Also served as the live proof that the owner's new overnight capability works.

- `dram_bytes` **1,628,984,320** (1.63 GB), flagged `dram_bytes_is_lower_bound: true`
- `arithmetic_intensity_flops_per_byte` 5.27, `matmul_flops` 8.59e9,
  `useful_attention_flops_causal` 1.08e9
- `materialized_bytes` 16,777,216 — exactly one dense `[64,4,128,128]` fp32 score matrix

**Read it carefully, because it is not what it first looks like.** The model's own
`dram_bytes_model` string says "baseline-dense-route compulsory traffic": it is derived
from the *shape*, assuming the dense route that materializes the full score matrix. It is
therefore a floor for the **baseline**, not for k004, whose Triton attention never
materializes that matrix. Arithmetic makes that concrete: 1.63 GB at the 3060 Ti's ~448
GB/s is ~3.6 ms, while the torch profiler measured k004 at ~2.6 ms of device time per
forward. **k004 is already below this "floor", which is the proof that the number does not
bound it.** So the honest use is narrow: it shows the dense baseline route is
bandwidth-bound, and it gives a compute reference (8.59 GFLOP) against which k004's ~2.6 ms
is roughly 5x off the tensor-core roofline. It must NOT be quoted as k004's traffic, and a
plan must not claim a fusion win "against the memory floor" on the strength of it.
Same failure mode as LESSONS 31: a real number that describes something other than the
thing under test.

- What still stands, and is the direction: the torch-profiler kernel table is a direct
  measurement of k004 itself, and it says 49% of device time is elementwise plus
  normalization across 44 kernels per forward. That is the evidence the fusion family gets
  registered on.

**30 Aug 22:12–22:22 SGT — PLAN CHANGE (owner gave full direction), and all 12 primary
shapes calibrated.**

**Decision: breadth before depth.** The prior plan was to chase a fusion win on shape 1.
Against a 31 Aug 20:00 SGT freeze that is the wrong priority, and the owner was told so
plainly: the score scenario is `geomean-shapes-1-13`, so a shape with no defensible number
is a hole in the headline, and the failure mode is arriving at freeze with an elegant
control system and two measured shapes. So: calibrate every primary shape first, then take
one audited champion per shape with kernels that already exist, then spend what is left of
the 60 attempts on the shapes with real headroom. A defensible modest geomean across 13
beats an undefendable 11x, and beats a large win on shape 1 with twelve blanks.

Twelve calibrations run back to back, all `correct: true`, all bound. Shape 6 is excluded
by design (side lane, never `run`). **These thresholds are now immutable for the campaign:**

| shape | noise | promotion threshold |
| --- | --- | --- |
| 1 | 0.35% | 1.030 |
| 2 | 2.40% | **1.072** |
| 3 | 1.18% | 1.035 |
| 4 | 3.14% | **1.094** |
| 5 | 0.90% | 1.030 |
| 7 | 1.86% | 1.056 |
| 8 | 0.25% | 1.030 |
| 9 | 0.14% | 1.030 |
| 10 | 0.36% | 1.030 |
| 11 | 0.39% | 1.030 |
| 12 | 1.59% | 1.048 |
| 13 | 0.86% | 1.030 |

Two things worth planning around:

1. **Shapes 2 and 4 have punitive bars (7.2% and 9.4%) and shape 7 is 5.6%.** Those are
   the small/awkward shapes where run-to-run jitter is largest, so a genuine 5% win there
   is unprovable under this campaign's own rules. Spend attempts there last, if at all —
   the eight shapes sitting at the 1.03 floor are where attempts convert to promotions.
2. **A systematic bias, flagged not yet explained.** Baseline-against-itself should centre
   on 1.0. Eight of the twelve came in BELOW it (0.9686 … 0.9986) and only four above,
   with the low outliers on the small shapes. That is consistent with the second slot of a
   paired run being slightly slower than the first — within-invocation drift, not random
   scatter. If real, it biases every measured speedup DOWNWARD, which is the safe
   direction (we would understate wins, never overstate them), but it should be named in
   the report rather than discovered by a judge. It is also a reason not to read a 1.02
   result as "nearly a win". Not investigated further tonight; a second calibration per
   shape (2 of 3 still available on every shape) would confirm or kill it cheaply.

Ledger after this batch: 15 permits issued and consumed, 12 calibration requests, 3
diagnostics, **still 0 of the 60 attempts spent**, 0 strikes, no permit armed.

**30 Aug 22:24 SGT — baseline profiled on shape 1, and it settles two things at once.**

`profile-575ff8b9d235cb8378ba8739`, nsys, shape 1, target `k000_baseline.py`
(`2feee730…`), `--supports launch-overhead`. Run because the three earlier profiles all
targeted k004 — our own code — and a claim that "graph replay removes host launches"
must be grounded in the launch count of the thing being *replaced*, not the replacement.
Profiling only the thing you already like is how a direction gets opened on a number that
was never in dispute.

**The counter, per forward pass on shape 1:**

| route | launch API calls / forward | distinct kernels | gpu_idle_fraction |
| --- | --- | --- | --- |
| `k000_baseline` (eager) | **115.0** | 13 | 0.034 |
| `k004_graphed_triton` | **2.0** | (58 kernel instances) | 0.124 (torch-profiler) |

2300 launch API calls over 20 iterations for the baseline — 1660 `cudaLaunchKernel` +
640 `cuLaunchKernel` — against 40 for k004. A **57x reduction in host launch calls.**
That is a clean, direct, defensible mechanism claim, and it is the counter evidence the
shape-1 graph-replay family gets registered on. It also shows the baseline route carries
17 `cudaMemcpyAsync` and 24 `cudaMemsetAsync` per forward that the replay removes outright.

**Second thing, unplanned and useful: this independently confirms LESSONS 31.** The very
same tool, same shape, same NVTX window, on an *un-graphed* route reports
`gpu_idle_fraction 0.034` and 13 distinct kernels — entirely sensible numbers. So nsys is
not broken and its idle fraction is not generally wrong; it is specifically blind to work
inside a CUDA graph. The 0.981 idle reading on k004 was that blindness and nothing else.
Two routes, one tool, one honest explanation. Worth putting in the report as the concrete
illustration of why the cross-check rule exists.

Next: register `F-shape1-graph` (mechanism `cuda-graph-replay`, bottleneck
`launch-overhead`, changed_resource `kernel-launches`) citing this profile, then the first
real attempt — k004 against the baseline on shape 1, bar 1.03.

**30 Aug 22:27–22:31 SGT — ATTEMPT 1 of 60. Shape 1: 2.0748x, correct. And my prediction
was badly wrong.**

Chain: family `F-shape1-graph` registered under an owner-signed `family:*` capability
(receipt `evt-20260830T142743.633141Z-7651c4b45cbb`, subject
`22d58ffe2a5353c3…`) → card C5 opened → research cycle 1 (citing
`roofline-table.md` and `megakernels-persistent.md`) → PLAN accepted, plan `525a0c6372b3`
→ permit `permit-b0e9bf4925d6bb17a2021cd6a80db7d1` (`may_promote: true`) → controller run
→ reconcile. Entry `run-be8e56a55edd1926a84bf5d1efc0b154`, group `F-shape1-graph|1`.

**Result: `event_speedup` 2.0748144528897536, `correct: true`**, against a calibrated
promotion threshold of 1.03. Status `pending_bound_independent_audit` — measured, NOT
promotable, and it stays that way until an audit binds to it. That distinction is the
product; do not quote this number as a champion yet.

**The band miss, stated plainly because the ledger will show it anyway.** I predicted
1.14–1.16 (`predict_min`/`predict_max` are recorded in the plan row at permit time, so
this is auditable and not something I can retro-fit). The result is 2.07 — roughly 80%
above the top of my band. The mechanism was right and the magnitude was not.

**Why I got it wrong, since a miss without a cause is just noise.** I anchored on
LESSONS 22, which records k004 on an idle box at 1.14x for shape 9 and 1.56x for shape 10,
and I picked the lower anchor on the reasoning that shape 1's roofline candidate time
(0.6461 ms) sits nearest shape 9's (0.6113 ms). Two errors in that:

1. **I mixed kernels.** The `cand ms` column in `roofline-table.md` is the then-champion
   route (k009-class), not k004. So I compared a k004 speedup against a k009 timing and
   treated them as one series. Different candidate, different number, no valid anchor.
2. **I reasoned from pre-gate numbers I had myself declared dead.** HANDOVER 3.1 says the
   whole board is ineligible and its baselines were mismeasured by 6–63%. Using those rows
   to set a band is exactly the "a progress log is not a source" failure of LESSONS 24,
   committed by me, four hours after writing about it.

The correct honest move would have been to predict from the mechanism arithmetic — 115
launches removed per forward against a ~0.65 ms forward — rather than from a dead board.
**Rule for the remaining shapes: set the band from the counter evidence and the roofline,
never from a pre-gate speedup row.** The 2 percent width the gate enforces is not the
problem; the anchor was.

What this does NOT mean: the 2.07x is not "better than expected" in a way that flatters
us. It is a paired, within-invocation, permit-bound measurement at the campaign timing
protocol against a baseline measured in the same invocation, which is precisely the
comparison the old board got wrong. It means the old 11x was measuring something else.

**30 Aug 22:37–22:50 SGT — GRIND STOPPED. The audit came back and it is right and I was
wrong. Two separate problems, one of them mine.**

### Problem 1 (mine): the 2.0748x is real but I attributed it to the wrong mechanism

Technical review verdict: **WEAK_DIAGNOSIS**. Promotion pauses. The auditor's arithmetic,
which I re-derived independently before accepting it:

- The bound baseline profile `profile-575ff8b9d235cb8378ba8739` records
  `gpu_busy_ns` 114578000 inside `gpu_span_ns` 118566764 over 20 iterations.
  That is **96.6% GPU-busy**, `gpu_idle_fraction` 0.0336.
- Per forward: 5.928 ms of span, 5.729 ms busy, **0.199 ms of host-launch idle**.
- So removing all 113 eliminated launches buys at most 5.928/5.729 = **1.0348x** — which
  is essentially shape 1's own calibrated promotion threshold of 1.03.
- The run saved 2.670 ms against a 5.154 ms median. Launch removal explains **at most
  ~7%** of that.

The rest is the second change k004 makes and I did not account for: it also replaces eager
attention with an inlined flash-style Triton kernel plus a fused packed QKV projection,
removing repeated passes over a 64x4x128x128 fp32 score tensor (16.8 MB per layer). At
448 GB/s each read-plus-write pass is ~75 us, and several passes across 4 layers accounts
for most of the 2.67 ms.

**So `F-shape1-graph` is the wrong family for this result.** The mechanism did fire
(`kernel_launch_api_calls` 115 → 2, exactly the declared `expected_counter_change`) and it
is still not what produced the win. I had the idle fraction in hand hours earlier, called
it "sensible", and never computed the ceiling it implies. Written up as LESSONS 32.

Note also: this retroactively explains the band miss. The band was sized for the graph
mechanism; the measurement captured something else entirely. **An overshoot is a
diagnosis failure, not a bonus.**

**Consequence for tonight's prepared work:** the families I registered while waiting —
`F-shape5-graph`, `F-shape9-graph`, `F-shape10-graph`, `F-shape11-graph`,
`F-shape13-graph`, all `cuda-graph-replay` / `launch-overhead` — carry the same misframing
and should NOT be planned against as they stand. They are registered and inert; nothing
was spent on them. The right framing for the shapes that k004 wins is attention/fusion
against `global-memory-traffic`, which is also where my own torch-profiler kernel table
pointed (49% of device time in 44 elementwise and normalization kernels per forward).

### Problem 2 (infrastructure, owner-only): audit results cannot be recorded

Integrity verdict: **RETEST** — a hard verdict. Its cause is narrow and, importantly, the
auditor found **no manipulation**: the harness's own tripwire fired. `supporting_timing`
reports `event_speedup` 2.0748 against `wall_speedup` 2.6043, an
`event_wall_speedup_agreement_ratio` of **1.2552** against the threshold in
`Project/harness/candidate_worker.py:476` (`"suspicious": agreement_ratio > 1.25`) — I
verified that line myself. The controller had already failed closed:
`performance_eligible: false`, `promotion_blocker: "performance_or_correctness"`. So this
run was never promotable, independently of the audit. The auditor traced a benign cause
(wall timing runs two unpaired sequential blocks, and both raw sample streams show a
disturbance confined to round 3) and asked for a re-measure on a quiet box.

**But the verdict never reached the ledger.** `record_audit_result` refused it:
`verdict does not match full schema`, listing `schema_version`, `attempt_nonce`,
`entry_id`, `packet_sha256`, `candidate_sha256`, `integrity`, `technical_review` and
`summary` as missing — while the stored response artifact
`Project/authority/blobs/0b3fa1ce…audit-response.json` visibly contains every one of them.
The journal therefore shows `attempt_failed` /
`AUDITOR_PROCESS_ABANDONED_WITHOUT_TERMINAL_EVENT` and **no hard verdict is latched**, so
no brake is set and permits are not frozen. That absence is a bug, not a pass
(LESSONS 33). I am treating the RETEST as binding anyway.

`Project/tools/audit_champion.py` and the audit authority are inside the LOCK and
Write-denied to the agent, so **I cannot fix this. It is owner work.**

### Why the grind is stopped rather than continuing

1. The owner's standing instruction for this loop is to stop on a hard integrity verdict.
   RETEST is one. A plumbing failure that prevents it being recorded does not unmake it.
2. Until audit results can be recorded, **nothing can ever promote**, so every further
   attempt would spend budget and roughly **$2.49 of audit cost** to produce a row that
   cannot become a champion. (Audit cost is measured, not estimated: `total_cost_usd`
   2.4904 for this one, 384 s wall, Opus 5, 25496 output tokens.)
3. The family framing needs redoing before more attempts, or the same attribution error
   repeats on every shape.

Ledger at stop: **1 of 60 attempts spent**, 0 promoted, 0 strikes recorded, 6 families
registered (1 planned against, 5 inert and misframed), 12 shapes calibrated, 4 profiles,
no permit armed, lock valid, campaign not stalled.

**30 Aug 23:17 SGT — ATTRIBUTION SETTLED. k003 alone measures 1.6010x on a quiet box, and
three independent lines now agree on what k004 is really worth.**

Third audit attempt also failed (`AUDITOR_PROCESS_ABANDONED_WITHOUT_TERMINAL_EVENT`), so
the retry cap is exhausted and `run-be8e56a55edd1926a84bf5d1efc0b154` is escalated to
`owner_attention` permanently. Three audits, three parse failures, roughly $7.50, zero
verdicts recorded. The recording fault is conclusively systematic and owner-only.

With the box finally at `active: []` and deeply idle (**SM 210 MHz, memory 405 MHz, 1%
utilisation, 14.38 W**, versus SM 450 MHz / 39% / 45.5 W when attempt 1 ran at 22:31), I
ran the isolating experiment in the **screening lane** — which cannot promote and adds no
strike, so it does not depend on the broken audit path at all.

Chain: baseline static-analysis diagnostic `profile-66aaa1f1db26652c96e8c4d4` (needed
because counter evidence must carry the *plan's* target sha, `run_gate.py:1068`) → family
`F-shape1-fusion` registered (`kernel-fusion` / `global-memory-traffic` /
`launches-and-intermediate-bytes`) → card C6 → research cycle 2 → screening plan → run.

### The result

| arm | what it is | measured | conditions |
| --- | --- | --- | --- |
| k003 | authored Triton attention, **no graph** | **1.6010x**, correct | quiet box, screening lane |
| k004 | Triton attention **plus** whole-forward graph | 2.0748x, correct | contended box, tripwire fired |
| — | launch-overhead ceiling from the baseline profile | **~1.035x** | derived, audit-confirmed |

**The Triton attention is the mechanism. It carries 1.60x on its own.** Graph replay is a
thin top-up bounded at ~1.035x by the baseline's own 96.6% GPU-busy fraction. My original
family — `cuda-graph-replay` against `launch-overhead` — was crediting the small effect
with the large result. LESSONS 32 confirmed by direct measurement, not just by argument.

### A convergence worth stating carefully

Three independent routes to k004's true value on a quiet box:

- 1.6010 (measured k003) x 1.035 (profile-derived launch ceiling) = **1.657**
- 2.0748 (contended measurement) / 1.2552 (its own event-vs-wall tripwire ratio) = **1.653**

These agree to 0.25%. **State the caveat honestly: dividing a speedup by its event/wall
disagreement ratio is not a derivation, it is a coincidence-level observation** — the
tripwire measures disagreement between two timing methods, not a contamination factor. So
this is *consistent with* k004 ≈ 1.65x and is suggestive, not proof. What is solid without
it: k003 = 1.601x measured on a verified-quiet box, and the graph cannot add more than
~1.035x on top. **k004's honest value on shape 1 is therefore about 1.60–1.66x, not 2.07x.**

### I predicted 2.10 and got 1.601 — wrong again, and here is the actual reason

Not a small miss. I built the band from device-busy time: baseline 5.729 ms/iter (nsys)
against k004 2.61 ms/iter (torch-profiler), and assumed k003 would do k004's device work
plus the baseline's 0.199 ms launch idle, giving 5.928/2.809 ≈ 2.11. Observed span is
5.928/1.601 = **3.70 ms/iter**, so k003 carries about **0.9 ms/iter more overhead than I
modelled**. The error: I had **no direct measurement of k003's own device time** and
substituted k004's, which is exactly the cross-tool, cross-route arithmetic the auditor
warned about on the previous card. Twice now I have predicted from numbers taken on a
different route than the one being run. **Rule: if the arm has never been profiled, say
the band is a guess and mark the prediction kind `characterization`** — which I did do
here, which is why this cost zero strikes.

### CORRECTION, 23:20 SGT, three minutes after the above was committed

**I ran k004 on the same quiet box and it measured 2.1428x — HIGHER than the contended
2.0748x, not lower. The contention explanation is dead, and so is the 1.035x ceiling.**
The falsifier I preregistered on that plan said, verbatim: *"If k004 on the quiet box lands
at or above 2.0, the contention explanation for attempt 1 is wrong and the launch-overhead
ceiling derived from the baseline profile must be wrong too."* It landed at 2.1428. The
falsifier fired against my own stated position, which is the only reason this got caught
inside twenty minutes.

**The real decomposition, both arms measured on the same deep-idle box, both correct:**

| arm | measured | marginal |
| --- | --- | --- |
| baseline → k003 (Triton attention, no graph) | **1.6010x** | 1.601x from fusion |
| baseline → k004 (Triton attention + graph) | **2.1428x** | **x1.3384 from graph capture** |

Graph capture is worth **1.338x on top of the Triton kernel** — not 1.035x. The ceiling
argument was out by an order of magnitude in its excess (0.338 vs 0.035).

**Why the ceiling was wrong, since I amplified it and must own the correction.** It reasoned
that if the baseline is GPU-busy 96.6% of its span, removing host launches can recover at
most 3.4%. Three defects:

1. **It was computed under the profiler.** Under nsys the baseline ran 5.928 ms/iter; its
   unprofiled median is 5.154 ms/iter. Profiling inflated the very span the fraction is
   taken over by ~15%, and CUPTI instrumentation changes the launch behaviour it measures.
2. `gpu_busy_ns` is a sum of kernel durations, so sub-resolution inter-kernel gaps are
   counted as busy rather than idle.
3. **Decisive: graph replay removes real device work, not only idle gaps.** The baseline's
   own nsys API summary records **340 `cudaMemcpyAsync` and 480 `cudaMemsetAsync` over 20
   iterations** — 17 and 24 per forward — and that traffic is *inside* `gpu_busy_ns`.
   A ceiling built on idle fraction structurally cannot see work that the mechanism
   eliminates from the busy side. So `1 / (1 - idle_fraction)` is **not** a valid upper
   bound on what a captured graph can buy.

**What this means for the two families.** Neither of my framings was right and neither was
worthless. `F-shape1-graph` was wrong to claim the whole 2.07x, but the mechanism is real
and material at 1.338x. `F-shape1-fusion` is real and larger at 1.601x. **k004 is a genuine
two-mechanism bundle**, and this campaign now holds the isolating measurement that
decomposes it — which is exactly what the auditor asked for and did not have.

**Where the auditor was right and where it was wrong.** Right: the attribution was
unproven, the bundle was unmeasured, and the isolating run was the correct demand. Wrong:
its quantitative 1.035x ceiling. **My error was worse than its error** — it offered a
bounded argument from the evidence available; I promoted that argument to a settled fact,
wrote "the graph adds at most 1.035x" into the durable record, and committed it, without
running the twenty-minute experiment that refutes it. See LESSONS 34.

### What this changes

- `F-shape1-fusion` is a correct family for shape 1 and it has a real, quiet-box,
  correct measurement behind it.
- The five `*-graph` families registered earlier for shapes 5, 9, 10, 11, 13 are confirmed
  misframed and should be replaced by fusion families before any attempt is spent on them.
- Nothing here is promotable until the audit-recording fault is fixed, but the *science*
  is now correct and the plans built on it will be too.

**31 Aug 01:47 SGT — MECHANISM QUESTION ANSWERED. The megakernel advantage is doing less
work, not recovering launch idle.**

Shape 5 was chosen as the cleanest available test of a question the report has to answer
correctly. Baseline device idle across the measured board spans **0.2% (shape 8) to 86.0%
(shape 2)**, and shape 5 sits at **1.0%** — the most device-saturated shape where the
megakernel ships. If the advantage came mainly from recovering span wasted between the
baseline's 115 host launches per forward, it should collapse there.

Preregistered falsifier: *"if shape 5 lands below 5x, the advantage depends substantially
on recovering baseline launch idle rather than on doing less work."* **Measured 9.1536x.
Did not fire.**

**So the claim the report should make is a fusion claim, not a launch-overhead claim:** the
megakernel wins by keeping a whole transformer block resident in registers instead of
round-tripping HBM roughly ten times per layer, and that holds even where the baseline has
essentially no idle to give back. This also retires the framing of every `F-shapeN-graph`
family, all of which were registered against `launch-overhead` because that was the only
bottleneck the early evidence supported.

### Board after nine shipped-route shapes

| shape | pre-gate | k009 | delta | baseline idle |
| --- | --- | --- | --- | --- |
| 13 | 28.82x | 28.4098x | −1.4% | — |
| 7 | 25.57x | 21.9645x | −14.1% | 3.2% |
| 2 | 15.26x | 14.3939x | −5.7% | 86.0% |
| 11 | 12.98x | 12.6797x | −2.3% | — |
| 3 | 11.96x | 12.6314x | +5.6% | 82.6% |
| 12 | 11.44x | 10.8141x | −5.5% | 69.8% |
| 5 | 11.40x | **9.1536x** | −19.7% | **1.0%** |
| 4 | 7.30x | 8.8774x | +21.6% | 49.2% |
| 1 | 10.73x | 8.3303x | −22.4% | 3.4% |

**Mean delta −4.9%, scatter −22.4% to +21.6%.** Note there is no relationship between
baseline idle and delta — shape 2 at 86% idle and shape 7 at 3.2% idle both land close,
shape 5 at 1.0% and shape 1 at 3.4% both land ~20% low. The scatter is measurement
variation between two different harnesses, not a systematic bias.

**31 Aug 01:20–01:23 SGT — THE PRE-GATE BOARD IS BEING VINDICATED. My falsifier fired
against me and the original numbers look approximately right. Third correction of the
night, and the most consequential.**

Shape 13, shipped megakernel k009: **28.4098x**. I preregistered: *"if k009 lands at or
above 25x then the withdrawn pre-gate figure of 28.82x was approximately correct, the
assumption that the pre-gate board was systematically inflated is wrong, and the
withdrawal blocks in the three drafts must be rewritten."* **It fired.** 28.4098 against
28.82 is agreement to **1.4%**.

### All three shipped-route measurements agree closely with the withdrawn board

| shape | pre-gate (official script) | **k009 measured tonight** | agreement |
| --- | --- | --- | --- |
| 2 | 15.26x | **14.3939x** | 5.7% |
| 3 | 11.96x | **12.6314x** | 5.6% |
| 13 | 28.82x | **28.4098x** | **1.4%** |

**The pre-gate board was measuring the right kernel and getting approximately the right
answers.** What was wrong with it was *process* — no permits, no bound audit verdicts, and
`HANDOVER.md` §3.1's finding that its baselines were 6–63% off their own calibration. That
finding evidently did not move the headline much.

### Two of my own claims are now refuted

1. **"The pre-gate board was systematically inflated."** Wrong. It is procedurally invalid
   and numerically close. Those are different things and I conflated them.
2. **"The shipped route beats k004 by a consistent ~1.77x."** Wrong — that was two points
   and I generalised. Shape 13's ratio is 28.4098 / 5.8096 = **4.89x**. The ratio is
   strongly shape-dependent, exactly like every other quantity in this system.

### What this means for the drafts, and it is urgent

My WITHDRAWN blocks currently tell a reader that 10.32x is withdrawn and the real figure
is 2.94x. **That is now the most misleading statement in the deliverables** — it understates
by roughly 3.5x, which is precisely the error I accused the original drafts of making, in
the opposite direction. The blocks must be rewritten to say: the old board is
*procedurally* invalid and unpromotable, its magnitudes are being *confirmed* within about
6% on the three shapes re-measured so far, and the shipped-route board is incomplete at
3 of 12 shapes.

**I will not state a geomean until all twelve are measured.** Three points that agree with
the old board are not a board, and the discipline that has actually worked tonight is
exactly this: withdraw what cannot be traced, and name no substitute until it is measured.

### Honest scorecard on my own judgement tonight

Numeric bands 0 for 17. Qualitative hypotheses 8 for 9 — the first failure being the
inflation assumption, which is the one that mattered most. Three separate corrections have
each required a further correction. The common cause every time was **inferring a
magnitude from too few points and writing it down as settled**. The controls that kept all
of this cheap: characterization-kind screening in the scratch lane (0 strikes across 26
attempts), preregistered falsifiers aimed at my own position, and visible WITHDRAWN blocks
instead of silent edits, which is why each error was recoverable rather than shipped.

**31 Aug 01:05–01:13 SGT — I MEASURED THE WRONG KERNEL FOR 22 ATTEMPTS. First correction
run: the shipped megakernel is 1.77x better than what I measured, and it partially
rehabilitates the board I withdrew an hour ago.**

### The error

Auditing the tech report's *prose* (not its tables) surfaced §3: *"The megakernel (`k009`,
shipping on 11 of 12 runnable shapes)"*. My entire campaign measured
`k004_graphed_triton.py`. `Project/submission/dispatcher_region.py` confirms the report:
`d_model <= 128` routes to a **fused-block megakernel**, larger `d_model` to an **fp16
tensor-core stack**. k004 is neither.

**Cause:** runbook step 11's worked example profiles k004. I adopted it as the campaign
candidate on the first lap and never checked it against the dispatcher. Twenty-two
attempts characterise a route that does not ship.

`k009_fused_tuned.py` is the right proxy — its header matches the dispatcher's line for
line (K1 norm + packed QKV; K2 flash attention over all heads with the out-projection
folded into the head loop, then residual + norm2 + exact-erf GELU FFN + residual
in-register, whole forward captured as one CUDA graph). Caveat carried into the card:
k009 is the megakernel **class** the dispatcher ships, not literally the dispatcher bytes,
because `dispatcher_region.py` references `BaselineTransformer` as a free name and cannot
be loaded standalone as a candidate module.

### First correction measurement — shape 2

| route | shape 2, quiet box, same baseline and protocol |
| --- | --- |
| k004 (non-shipping, what I measured) | 8.1115x |
| **k009 (shipped megakernel)** | **14.3939x** |

**The shipped route is 1.77x better than the one I spent the night on.** Preregistered
falsifier — "if k009 lands at or below 8.1115x the shipped route is no better" — **did not
fire**. Seventh qualitative hypothesis to survive; numeric bands now 0 for 15.

### This partially rehabilitates the board I withdrew, and I must not over-correct

The pre-gate board claimed **15.26x** (official script) and **14.98x** (own referee) for
shape 2. Tonight's controlled measurement of the actually-shipped kernel is **14.3939x** —
**within about 6% of both**. That is a much better agreement than the blanket withdrawal
implied.

Being precise about what this does and does not change:

- **The withdrawal stands on process.** Those rows were taken without a permit, without a
  bound audit verdict, and `HANDOVER.md` §3.1 records their baselines as 6–63% slower than
  their own calibration. They are not promotable and cannot be quoted.
- **But their magnitudes may be closer to right than I implied.** The pre-gate board was
  measuring the megakernel, i.e. the real champion; my 2.94x was measuring a weaker,
  non-shipping route. Presenting 2.94x as "the corrected figure" was itself misleading and
  has been amended in all three drafts.
- **The honest position right now: the shipped route's geomean is unknown**, one shape is
  measured at 14.39x, and I will not extrapolate from a single point.

**Method note:** this is the second time tonight that a correction of mine needed
correcting. Both times the cause was the same — asserting a replacement figure before the
replacement was measured. The rule that keeps surviving is the narrow one: **withdraw what
cannot be traced, but do not name a substitute until it is measured.**

**31 Aug 00:17–00:21 SGT — head_dim research done, ceiling measured, and the answer is
that de-padding is NOT worth building. A different target is bigger.**

Web research (note: `Project/research/small-head-dim-padding.md`, indexed) established
three things:

1. The `max(16, ...)` in `k003_triton_attention.py:103` is not a mistake — `tl.dot`
   requires a 16-minimum tile, so `head_dim` 8 must pad to 16. Confirmed against *The
   Anatomy of a Triton Attention Kernel* (arXiv 2511.11581, Nov 2025) §8.
2. **Head-packing is arithmetically impossible here**, not merely hard. QK^T contracts
   *over* the feature dimension, so packing two heads into one 16-wide tile would sum
   their features and produce wrong scores. In PV the head dim is free but `P` differs per
   head. There is no valid packing.
3. The only candidate is replacing `tl.dot` with `tl.sum(q * k)` for QK^T at D=8 — a
   documented technique for "lean" tensors (Triton issues #793, #1181) but one the
   literature explicitly does *not* recommend in general, because it trades Tensor Cores
   for CUDA cores.

**Then I measured the ceiling before building anything — LESSONS 32 applied deliberately
this time.** torch-profiler on k004 at shape 11 (`profile-68d365297c74f5d15dd9c6a5`):

| kernel | share of k004 forward |
| --- | --- |
| `_attn_fwd` (our padded attention) | **36.05%** |
| cutlass GEMM (linear projections) | 24.64% |
| layer norm | 14.13% |
| elementwise, three kernels | **23.45%** |
| D2D memcpy | 1.53% |

**De-padding ceiling:** even a *perfect* 2x on `_attn_fwd` removes 18% of the forward,
giving **1.22x** on shape 11 — real, but it applies to only two of thirteen shapes (7 and
11). Geomean effect if both gained the full 1.22x: about **1.03x**. And the realistic gain
is far below the ceiling because the CUDA-core path is slower per FLOP than the Tensor-Core
path it replaces. **Conclusion: not worth building. Recorded as closed rather than left as
an open temptation.**

**The bigger target, and it was in my very first profile.** Elementwise plus layer norm is
**37.6%** of the shape-11 forward — *more than our attention kernel*. The first
torch-profiler read of the night (shape 1, 23:16) said the same thing: 49% of device time
in 44 elementwise and normalization kernels. Halving that share yields a ceiling of about
**1.23x**, comparable to de-padding on shape 11 but applying to **every shape** rather than
two. That is the correct next kernel direction if one is built.

**But state the strategic position honestly.** Promotion is blocked, so no new kernel can
become a champion until the owner fixes audit recording. Building a fused
layernorm-plus-residual-plus-activation kernel is a multi-hour job with real correctness
risk, against a 20:00 freeze. The measured 2.94x board already exists and needs only an
audit path to become defensible. **Recommendation for the owner: fix audits first and
promote what is measured; treat the fusion kernel as optional upside, not as the plan.**

**31 Aug 00:09–00:14 SGT — BOARD COMPLETE. All twelve primary shapes measured on a quiet
box. Geomean 2.94x. The ordering model survived its first two-sided test.**

Shape 2 (B=1): baseline idle **86.0%**, the highest on the board, measured **8.1115x** —
new best. Its falsifier tested *saturation* (would have fired below shape 3's 7.1845x) and
did not fire, so the ordering holds even at the extreme.

Shape 4 (B=16): baseline idle **49.2%**, strictly bracketed by shape 1 at 3.4% → 2.1428x
and shape 12 at 69.8% → 3.2334x. This was the **first two-sided falsifier** of the night —
refutable from above *or* below, rather than merely having to clear a floor. Measured
**2.7175x, inside the bracket.** That is the strongest form of the ordering claim tested.

### Final board — twelve shapes, quiet box, all correct, zero strikes

| shape | B | heads | head_dim | seq | idle | k004 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 4 | 32 | 128 | 86.0% | 8.1115x |
| 3 | 4 | 4 | 32 | 128 | 82.6% | 7.1845x |
| 13 | 64 | 4 | 32 | 1024 | — | 5.8096x |
| 11 | 64 | 16 | 8 | 128 | — | 4.2433x |
| 7 | 64 | 4 | 8 | 128 | 3.2% | 3.4781x |
| 12 | 64 | 4 | 32 | 32 | 69.8% | 3.2334x |
| 4 | 16 | 4 | 32 | 128 | 49.2% | 2.7175x |
| 5 | 128 | 4 | 32 | 128 | 1.0% | 2.1475x |
| 1 | 64 | 4 | 32 | 128 | 3.4% | 2.1428x |
| 10 | 64 | 2 | 64 | 128 | — | 1.5833x |
| 9 | 64 | 1 | 128 | 128 | — | 1.1723x |
| 8 | 64 | 4 | 256 | 128 | 0.2% | 1.1060x |

**Geomean 2.94x. Minimum 1.1060x — never slower than the baseline anywhere.**

**Caveat that must travel with the number:** the campaign's official scenario is
`geomean-shapes-1-13`, which includes **shape 6** (B=10000, dedicated side lane, not
measured here). 2.94x is the geomean of the twelve *primary* shapes and is **not** the
official scenario figure. It is also **screening-lane and unpromotable** until the audit
recording path is fixed.

### Final method scorecard

| claim type | record |
| --- | --- |
| numeric prediction bands | **0 for 14** |
| qualitative regime hypotheses with preregistered falsifiers | **6 for 6** |

Fourteen numeric misses, zero strikes, because every run was characterization-kind in the
scratch lane. The six qualitative hypotheses include the two hardest forms: one aimed at
the **low** end (shape 8 predicted worst — confirmed at 1.1060x) and one **two-sided**
bracket (shape 4). Worth noting honestly that shape 2's placeholder band of 7.92–8.08
missed by only 0.03 against 8.1115 — the closest all night — but a single near-hit does
not reverse 0-for-14 and no conclusion is drawn from it, per the LESSONS 35 commitment.

**31 Aug 00:04–00:06 SGT — shape 7 measures 3.4781x and DISAMBIGUATES shape 11. Head
WIDTH, not head count, is the larger factor — correcting an entry I wrote earlier tonight.**

Shape 7 was chosen not to add a number but to separate two variables that shape 11
confounded. Shape 11 has sixteen heads **and** head dimension 8, so my earlier attribution
of its 4.2433x to "sixteen small per-head matmuls" was underdetermined by the evidence I
had. Shape 7 holds head dimension at 8 while dropping head count to 4:

| shape | heads | head_dim | measured |
| --- | --- | --- | --- |
| 1 | 4 | 32 | 2.1428x |
| **7** | **4** | **8** | **3.4781x** |
| 11 | 16 | 8 | 4.2433x |

Holding head count at 4 and narrowing head_dim from 32 to 8 moves the result from 2.14 to
**3.48**. Then raising head count from 4 to 16 at the same width adds only 2.14 → wait,
3.48 → 4.24. **So narrow head width contributes roughly +1.34x and the extra twelve heads
add roughly +0.76x on top.** Head *width* is the bigger lever, and I had it backwards.

**Correction to the record:** the DECISIONS entry of 23:34–23:37 and the STATE table both
describe shape 11's advantage as "16 small per-head matmuls". That is not wrong so much as
incomplete and mis-weighted — the dominant term is the narrow head dimension, which forces
the baseline into thin, badly-shaped matmuls regardless of how many there are. The
falsifier I preregistered was aimed precisely at this ("if shape 7 lands at or below 2.15
then the shape 11 result was driven entirely by head COUNT, and the sentence I already
wrote must be corrected") and it did not fire, so the correction runs the other way: head
count was over-credited, not under-credited.

**A second assumption corrected, before the run rather than after.** I went into this cycle
expecting model dimension 32 to be launch-bound. Its baseline measures `gpu_idle_fraction`
**0.0318**, essentially identical to shape 1's 0.0336. So **small model dimension does not
imply launch-bound** — only small *batch* (shape 3, 82.6%) or short *sequence* (shape 12,
69.8%) has produced that regime. Shape 7 is work-bound and still wins 3.48x, which also
shows the launch-bound regime is not required for a large gain.

### Board after ten shapes — geomean 2.68x

| shape | k004 | idle | dominant weakness |
| --- | --- | --- | --- |
| 8 | 1.1060x | 0.2% | none |
| 9 | 1.1723x | — | none |
| 10 | 1.5833x | — | mild head splitting |
| 1 | 2.1428x | 3.4% | reference |
| 5 | 2.1475x | 1.0% | reference |
| 12 | 3.2334x | 69.8% | launch-bound |
| 7 | 3.4781x | 3.2% | **narrow head_dim 8** |
| 11 | 4.2433x | — | narrow head_dim 8 **plus** 16 heads |
| 13 | 5.8096x | — | quadratic score tensor |
| 3 | 7.1845x | 82.6% | launch-bound |

**Scorecard: numeric bands 0 for 12. Qualitative regime hypotheses 4 for 4.**

**30 Aug 23:58 – 31 Aug 00:01 SGT — shape 8 measures 1.1060x, the LOWEST on the board, and
the regime model passed the falsification test that could have embarrassed it.**

Shape 8 (d=1024) baseline measures `gpu_idle_fraction` **0.00225** — 99.8% device-busy,
768 ms of device work across 20 iterations, the most work-bound route measured. Its head
dimension is 256, the largest on the board, so baseline attention matmuls are large and
well shaped rather than fragmented. Its score tensor stays at 16.8 MB per layer (S and H
unchanged from shape 1) while its linear layers grow 64-fold in FLOPs, so attention is a
small slice of a much bigger forward — `roofline-table.md` calls it ~98% linear.

So all three baseline weaknesses that explain every win tonight are simultaneously weak
here. **I predicted the bottom of the board and set the falsifier at 2.15. Measured
1.1060x — the falsifier did not fire.**

This one mattered more than the others: **it is the first card that predicted a LOW
result.** A regime model that only ever predicts large wins is not falsifiable in the
direction that would discredit it. This one was, and it held.

Note it is still a win, not a loss: 1.1060x clears the calibrated 1.03 threshold. The
candidate is never *worse* than the baseline on any shape measured tonight.

### Board after nine shapes — geomean 2.60x

| shape | k004 | baseline idle | dominant weakness |
| --- | --- | --- | --- |
| 8 | 1.1060x | 0.2% | none — GEMM-dominated, well-shaped |
| 9 | 1.1723x | — | none — single-head attention already efficient |
| 10 | 1.5833x | — | mild head splitting |
| 1 | 2.1428x | 3.4% | reference |
| 5 | 2.1475x | 1.0% | reference, size irrelevant |
| 12 | 3.2334x | 69.8% | launch-bound |
| 11 | 4.2433x | — | 16 small per-head matmuls |
| 13 | 5.8096x | — | 1.07 GB score tensor per layer |
| 3 | 7.1845x | 82.6% | launch-bound |

**Method scorecard, and the asymmetry is now stark.** Numeric bands: **0 for 11**.
Qualitative regime hypotheses with preregistered falsifiers: **3 for 3**, including one
aimed at the low end. I cannot forecast magnitudes in this system at all; I can classify
regimes reliably. That distinction is worth putting in the report, because it says which
kind of claim this project's evidence can carry.

**30 Aug 23:53–23:55 SGT — shape 12 measures 3.2334x. Second qualitative hypothesis to
survive its falsifier, and the board now has a partial explanatory model.**

Shape 12 (S=32) baseline measures `gpu_idle_fraction` **0.6982** — launch-bound, between
shape 3 and the work-bound cluster. I preregistered an **ordering** hypothesis with a
refutation line at 2.15: *"if shape 12 at 69.8 percent baseline idle lands no better than
the work-bound shapes near 2.14, then baseline idle fraction does not order the board."*
**Measured 3.2334x. The falsifier did not fire.** Idle fraction orders these four:

| baseline `gpu_idle_fraction` | shape | k004 |
| --- | --- | --- |
| 0.0104 | 5 | 2.1475x |
| 0.0336 | 1 | 2.1428x |
| 0.6982 | 12 | 3.2334x |
| 0.8256 | 3 | 7.1845x |

**But state the limit of that model honestly: idle fraction does NOT explain the whole
board.** Shapes 11 (4.2433x) and 13 (5.8096x) both beat shape 12 while being work-bound,
not launch-bound. So idle fraction captures one of at least three independent baseline
weaknesses, and it is not a universal predictor. The honest summary is:

- **launch-bound** (small batch or short sequence) — shapes 3, 12
- **many small per-head matmuls** — shape 11
- **quadratic score-tensor traffic** — shape 13
- **none of the above** — shapes 1, 5 at ~2.14x, and shape 9 at 1.17x where the baseline's
  single-head attention is already one efficient large matmul

### Board after eight shapes — geomean 2.89x

| shape | k004 | dominant baseline weakness |
| --- | --- | --- |
| 9 | 1.1723x | none — baseline already efficient |
| 10 | 1.5833x | mild head splitting |
| 1 | 2.1428x | reference |
| 5 | 2.1475x | reference, size irrelevant |
| 12 | 3.2334x | launch-bound, 69.8% idle |
| 11 | 4.2433x | 16 small per-head matmuls |
| 13 | 5.8096x | 1.07 GB score tensor per layer |
| 3 | 7.1845x | launch-bound, 82.6% idle |

**Scorecard update.** Numeric bands: 0 for 10, unchanged and no longer claimed. Qualitative
regime hypotheses with preregistered falsifiers: **2 for 2** (shape 3's launch-bound
reading, shape 12's ordering). That asymmetry is the finding about method — I can reason
about *which regime a shape is in* and cannot forecast *what number it will produce*.

**30 Aug 23:47–23:50 SGT — shape 3 measures 7.1845x, a new best, and the FIRST falsifier
tonight that did not fire against me.**

Shape 3 (B=4) turned out to be a qualitatively different regime, and the profile said so
before the run rather than after:

| shape | batch | baseline `gpu_idle_fraction` |
| --- | --- | --- |
| 5 | 128 | 0.0104 |
| 1 | 64 | 0.0336 |
| **3** | **4** | **0.8256** |

At batch four the device is **idle 82.6% of the span** — roughly 2.13 ms of idle behind
0.45 ms of device work per forward, while the same 115 host launches trickle through. This
is the latency-bound, grid-too-small regime `roofline-table.md` describes, and it is the
one shape where the launch-overhead mechanism the family is named for should genuinely
dominate. I preregistered that reading with a falsifier at 2.0: *"if shape 3 lands below
2.0 then being launch-bound confers no larger gain and the launch-overhead framing is wrong
on the one shape where it should be strongest."* **Measured 7.1845x. The falsifier did not
fire — the qualitative hypothesis is supported.**

Note the distinction that matters: the *numeric* placeholder (2.97–3.03, filled only
because the gate requires the field) missed badly, and per LESSONS 35 no conclusion is
drawn from that. The *qualitative, falsifiable* claim held. That is the difference between
a forecast and a hypothesis, and it is why I retired one and kept the other.

### Board after seven shapes — geomean 2.85x

| shape | B | heads | seq | k004 vs baseline | regime |
| --- | --- | --- | --- | --- | --- |
| 9 | 64 | 1 | 128 | 1.1723x | single head, baseline already efficient |
| 10 | 64 | 2 | 128 | 1.5833x | two heads |
| 1 | 64 | 4 | 128 | 2.1428x | work-bound reference |
| 5 | 128 | 4 | 128 | 2.1475x | work-bound, size irrelevant |
| 11 | 64 | 16 | 128 | 4.2433x | many small per-head matmuls |
| 13 | 64 | 4 | 1024 | 5.8096x | quadratic score tensor, 1.07 GB/layer |
| **3** | **4** | 4 | 128 | **7.1845x** | **launch-bound, 82.6% device idle** |

**Three independent baseline weaknesses now explain the whole spread**, and none of them is
a strength of our kernel so much as a defect of the eager route: it degrades as heads
multiply, quadratically as sequence grows, and catastrophically when the batch is too small
to fill the grid. Our route is simply flat across all three.

**Implication for the remaining shapes, stated before measuring them:** 2 (B=1), 4 (B=16),
12 (S=32) and 7 (d=32) are all small-work shapes that should sit in the launch-bound
regime, so the geomean is more likely to rise than fall. Shape 8 (d=1024, GEMM-dominated)
is the one that should look like shape 9. No numeric bands attached to any of that.

**30 Aug 23:41–23:44 SGT — shape 13 measures 5.8096x, the best on the board, and I am
retiring my own prediction bands as promised.**

Shape 13 (S=1024) baseline kernel breakdown from `profile-b35c340c80f4558332feb0da`,
which is the first time tonight I predicted from a measured per-kernel table **of the shape
being predicted**:

| kernel | ns over 20 iters | share |
| --- | --- | --- |
| `masked_fill` over the score tensor | 916,269,561 | **37.2%** |
| scale multiply | 419,627,061 | 17.0% |
| `softmax_warp_forward` | 417,830,332 | 17.0% |
| attention GEMM `256x64_16x4_tn` | 227,874,632 | 9.2% |
| attention GEMM `64x64_16x6_nn` | 223,079,754 | 9.1% |
| linear projection GEMMs `128x256` | 94,269,380 | 3.8% |
| layer norm / copy / add | ~152M | ~6.1% |

At S=1024 the materialized score tensor is about **1.07 GB per layer** — 64x shape 1 — and
**71.2% of the baseline is pure elementwise and softmax traffic over it**. Note the linear
projections are only 3.8%: attention is quadratic in sequence length, the projections
linear, so at this shape the model is almost entirely attention.

I predicted 3.47–3.53 by assuming the fused kernel deletes the 71.2% traffic and keeps the
18.3% GEMM share. **Measured 5.8096x**, and my falsifier at 4.5 fired. What that falsifier
was built to detect is exactly what happened: the fused kernel does not merely remove the
traffic, it **also beats the baseline's attention GEMMs**. That makes sense in hindsight —
cutlass GEMMs reading and writing a 1.07 GB operand are bandwidth-bound, not compute-bound,
so a tiled flash kernel that never lands the tensor in DRAM wins the math as well.

### Board after six shapes, all quiet-box, all correct, all zero-strike

| shape | B | heads | seq | k004 vs baseline |
| --- | --- | --- | --- | --- |
| 9 | 64 | 1 | 128 | 1.1723x |
| 10 | 64 | 2 | 128 | 1.5833x |
| 1 | 64 | 4 | 128 | 2.1428x |
| 5 | 128 | 4 | 128 | 2.1475x |
| 11 | 64 | 16 | 128 | 4.2433x |
| 13 | 64 | 4 | **1024** | **5.8096x** |

**Geomean of these six: 2.44x.** Two axes now explain the spread: head count (the baseline
degrades as heads multiply and per-head matmuls shrink) and sequence length (the baseline
degrades quadratically as the score tensor grows). Both are baseline weaknesses the fused
route simply does not have.

### Retiring my prediction bands, as preregistered

Eight bands tonight, **eight misses**. On card C11 I wrote, before running it, that if a
band derived from a measured per-kernel breakdown of the target shape *also* missed, the
honest conclusion is that I cannot forecast this system and should stop attaching numeric
bands as claims. It missed by 66%. So, honouring that: the gate requires `predict-min` and
`predict-max` on every screening plan and the field cannot be omitted, but from here it is
filled as **a required field with stated low confidence, not as a claim**, and **no
conclusion is drawn from any band hit or miss**. Conclusions come from measurements only.
Recorded as LESSONS 35.

Worth being precise about what this does and does not indict. My *forecasting* is 0 for 8.
My *instrumentation* held perfectly: every run was characterization-kind in the scratch
lane, so eight consecutive misses cost **zero strikes and zero promotable damage**, and six
fired falsifiers I had aimed at my own position — which is what produced every genuine
finding of the night rather than any of my predictions.

**30 Aug 23:34–23:37 SGT — head-count axis completed. Shape 11 is our best shape at
4.2433x, and it is the shape a previous card had written off.**

| shape | B | heads | head_dim | measured k004 | I predicted |
| --- | --- | --- | --- | --- | --- |
| 9 | 64 | 1 | 128 | **1.1723x** | 2.12–2.16 |
| 10 | 64 | 2 | 64 | **1.5833x** | 1.545–1.575 |
| 1 | 64 | 4 | 32 | **2.1428x** | 1.65–1.67 |
| 5 | 128 | 4 | 32 | **2.1475x** | 1.84–1.86 |
| 11 | 64 | **16** | **8** | **4.2433x** | 2.23–2.27 |

**Geomean across these five: ~2.05x.** All correct, all quiet-box, all screening lane,
**0 strikes**.

**The axis is steeper than log-linear.** From 1 to 4 heads the gain rises about +0.48 per
doubling; from 4 to 16 it rises about +1.05 per doubling. Head count is the governing
variable and problem size is irrelevant (shape 5 doubles shape 1's batch and moves the
result by 0.2%).

**Why shape 11 wins so hard, and why I predicted the opposite.** I argued the trend would
*break* at shape 11 because head_dim is 8 and the Triton kernel pads the feature dimension
to 16, wasting half of every tile — citing card C4, which found the k012 grid-heads route
"did not materially help hd=8 because the padding waste sits in the 16-wide tiles either
way". **I misapplied that card.** C4 compared *candidate against candidate* (k012 vs the
k009 champion). It says nothing about the **baseline**, and the baseline is what suffers
here: at 16 heads the eager route runs sixteen small per-head matmuls, which card C4 itself
recorded as an anomaly — shape 11 taking 0.96 ms against 0.61 ms for shapes 9 and 10 at
*identical FLOPs*. The padding waste in our kernel is real, and the baseline's 16-way head
splitting is simply far more expensive. Net: the widest margin on the board.

**This reframes shape 11 from a problem into the headline.** The old card treated it as an
anomaly to be fixed; it is actually where this candidate has the most room. And it means
head_dim=8 is a *target* for further work, not a wall — a kernel that avoids the 2x padding
could plausibly go beyond 4.24x.

**Prediction scorecard: six bands tonight, six misses.** 1.15→2.075, 2.10→1.601,
1.66→2.143, 1.85→2.147, 2.14→1.172, 1.56→1.583, 2.25→4.243. I have not once landed a band.
What has worked is everything around the bands: every run was characterization-kind in the
scratch lane so the total strike cost is **zero**, and five of these fired falsifiers I had
deliberately aimed at my own position, each of which taught the actual structure — that
size does not matter, that head count does, that an old note reproduces, and that a card I
was citing did not say what I thought.

**On the one that "missed" but confirmed:** shape 10 landed at 1.5833 against a 1.545–1.575
band, so the falsifier technically fired. But what it was testing — whether the LESSONS 22
idle-box figure of 1.56x reproduces under the current protocol — is **confirmed at 1.5%
agreement**. The lesson is about falsifier wording, not about the note: the gate caps band
width at 2%, which is right for a prediction and too tight for a *reproduction* claim.
Word future reproduction falsifiers against the scientific claim, not the band edges.

**30 Aug 23:25–23:30 SGT — quiet-box screening board grows to four shapes, and HEAD COUNT
turns out to dominate. Three more preregistered falsifiers fired against me.**

All runs below: screening lane, `--prediction-kind characterization`, k004 against the
eager baseline, verified-quiet box (`active: []`, SM 210 MHz, ~3% util, ~14 W), correct on
every seed, **0 strikes** across all of them.

| shape | B | heads | measured k004 | I predicted | outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | 64 | 4 | **2.1428x** | 1.65–1.67 | miss, falsifier fired |
| 5 | 128 | 4 | **2.1475x** | 1.84–1.86 | miss, falsifier fired |
| 9 | 64 | **1** | **1.1723x** | 2.12–2.16 | miss, falsifier fired |

**The finding: attention head count governs this candidate's advantage, and problem size
does not.** Shape 5 doubles the batch over shape 1 and changes nothing (2.1475 vs 2.1428).
Shape 9 keeps the batch and drops from 4 heads to 1, and the gain collapses by nearly half.
`roofline-table.md` records shapes 1, 9, 10 and 11 as carrying *identical* useful FLOPs of
7.52 GFLOP, so this is not a work-volume effect at all. The plausible reading, to be tested
on shapes 10 and 11 rather than asserted: with a single head the baseline's dense attention
is one large well-shaped matmul that cuBLAS already handles efficiently, so the authored
fused kernel has little to take back; with four or more heads the baseline pays per-head
overhead on smaller matmuls that the fused kernel avoids.

**Independent corroboration of an old number.** LESSONS 22 recorded k004 on shape 9 at
**1.14x on an idle box** (against 3.14x under load). Tonight's controlled quiet-box
screening gives **1.1723x**. Those agree closely. So the LESSONS 22 idle figures were
sound and, critically, they were **shape-specific** — my original error on attempt 1 was
importing shape 9's 1.14x as an anchor for *shape 1*, which is a different regime entirely.
The old note was right; I read it wrong.

**Prediction scorecard, stated plainly: four bands tonight, four misses.**
1.15 → 2.075; 2.10 → 1.601; 1.66 → 2.143; 1.85 → 2.147; 2.14 → 1.172. Every band derived
from mechanism theory failed, and the one derived from "nearest measured analogue" failed
too — because I picked the wrong analogue, treating same-FLOPs as same-regime. What has
actually worked is the discipline around the predictions: every miss cost zero strikes
because the runs were characterization-kind in the scratch lane, and three of them fired
falsifiers I had written to point at my own position, which is why each error was caught
within minutes instead of reaching a report.

**Method for the remaining shapes:** map the head-count axis first (10 at H=2, 11 at H=16)
before predicting anything else, and treat every band as a hypothesis about *regime*, not
about magnitude.

**30 Aug 22:45–23:05 SGT — second audit read. It CORRECTS my self-criticism, and it is
sharper than the first.**

Attempt 2 of the audit also failed to record (`auditor stdout must be exactly one
duplicate-free JSON object with no banners` — a *different* parse failure from attempt 1's
schema refusal, which together confirm the recording fault is systematic, not flaky). Its
verdict document is durable at
`Project/authority/blobs/92b7c588…audit-response.json`. Same two verdicts: integrity
**RETEST**, technical **WEAK_DIAGNOSIS**. It adds three things I did not have.

### Correction: I was too hard on my own prediction, and the number is the suspect

I wrote above that my 1.14–1.16 band was "badly wrong". On this evidence that is probably
backwards. The auditor compared the run against the twelve calibrations taken on this same
box under the identical protocol:

- **All twelve calibrations: `event_wall_speedup_agreement_ratio` 1.0024 – 1.0391**,
  `suspicious: false`.
- **Attempt 1: 1.2552** — roughly 6x outside that observed envelope.
- The disagreement is **entirely one-sided**: the baseline's wall time runs **+24% above**
  its own event median (6.3956 vs 5.1543 ms) while the candidate's runs **1% below** its
  own (2.4558 vs 2.4842 ms). Host-side inflation on the **eager arm only**.

That asymmetry is the exact signature LESSONS 22 records **for this same k004 candidate**:
CPU contention slows the launch-bound eager baseline much more than the graph-replay
candidate, so the within-entry ratio rises — k004 read 3.14x/4.09x under load and
**1.14x/1.56x idle**. My band was derived from those idle readings. So the likeliest
reading is not "my prediction was wrong" but **"the measurement was contaminated upward
and my band was near the truth."** Honest position: **shape 1's true speedup is unknown**,
and the single number we have tripped its own guard. LESSONS 32 stands (the attribution
error was real and mine); the self-flagellation about the band does not.

### The bottleneck I registered was wrong, and so was the premise I cited

The auditor's re-derivation, which I accept: the shape-1 baseline runs **7.52 GFLOP in
5.154 ms ≈ 1.46 TF/s against a 16.2 TF fp32 peak — about 9%**, while being GPU-busy 96.6%
of its span. That is not a launch problem and not an idle problem. It is **low kernel
efficiency**: 13 kernels per forward dominated by materializing and repeatedly re-reading
a 16.8 MB `[64,4,128,128]` score tensor (static-analysis models a 50.3 MB softmax peak).
Device-busy time itself fell **5.73 → 2.61 ms/iter** between baseline and candidate, and
graph replay *cannot* do that — it changes only how work is submitted, not how much there
is. The correct family for shape 1 is therefore **`kernel-fusion` against
`global-memory-traffic`**, which is where my own torch-profiler table pointed hours ago.

It also caught that **the roofline premise I cited is stale**: card C5 leans on "36 percent
of the fp16 roof", but that row is computed from a *pre-gate* candidate time — the very
board the same card declares ineligible. Post-lock numbers imply 4.5% (baseline) and 9.3%
(candidate) of the fp16 roof, 4–8x from the cited figure. I cited a dead number in the
same card where I warned against citing dead numbers.

### A contention hypothesis I can name and the owner should weigh

Nothing else was using the GPU meaningfully (three desktop residents, ~600 MiB). The
plausible CPU contender is **this project's own PostToolUse hook**: `.claude/settings.json`
runs `champion_watch.py` after **every** Bash/Edit/Write with `"async": true`, and that
watcher walks the durable journals plus the 49 inert pre-gate entries on each firing. An
async hook launched by the tool call immediately preceding a controller run can still be
executing while the run times its first rounds — and it would slow the CPU-hungry eager
arm more than the graphed arm, producing exactly the observed asymmetry. Unproven, and the
disturbance sits in round 3 rather than round 1, so it is a hypothesis and not a finding.
`.claude/**` is locked, so testing it means the owner temporarily disabling the hook, or
me simply not issuing tool calls immediately before a timed run.

### Process gap I created

**No machine-state attestation is bound to the optimization run.** Every calibration and
every profile carries a `machine_state_sha256`; the optimization measurement carries none.
The batch snapshot I wrote is explicitly scoped to calibration and its own note says the
exemption rests on calibration being a baseline-vs-baseline ratio — which does **not**
extend to a baseline-vs-candidate run, where contention biases the two arms unequally. So
card C5's preregistered "idle box" regime is **unverifiable from the record**. Every future
optimization attempt gets a per-run snapshot bound to it.

### The plan that follows, when the box is free

The decisive experiment is the **three-arm split**: baseline / authored Triton attention
**without** capture / Triton **plus** capture. `Project/kernels/k003_triton_attention.py`
already exists and is exactly the middle arm. If k003 alone recovers most of the gain, the
graph mechanism contributes ~nothing and `F-shape1-graph` is simply the wrong family; if it
does not, something else is going on and both readings are wrong. This runs in
**`--mode screening`**, which is the scratch lane: it cannot promote, and a characterization
miss adds no strike — so it answers the attribution question **without depending on the
broken audit-recording path**. That is the one genuinely unblocked piece of science
available tonight, and it is what I will run once no audit is in flight.

---

## 31 Aug ~02:10 SGT — CORRECTION CAMPAIGN COMPLETE. Twelve of twelve, geomean 9.68x.

The board is finished. Every locally-runnable shape re-measured on the kernel the
dispatcher actually selects, under a one-use permit, quiet box, screening lane,
`correct: true` on all twelve. **Geometric mean 9.68x**, range 2.02x to 28.41x.

| shape | pre-gate | measured | delta |
| --- | --- | --- | --- |
| 13 | 28.82x | 28.4098x | −1.4% |
| 7 | 25.57x | 21.9645x | −14.1% |
| 2 | 15.26x | 14.3939x | −5.7% |
| 11 | 12.98x | 12.6797x | −2.3% |
| 3 | 11.96x | 12.6314x | +5.6% |
| 12 | 11.44x | 10.8141x | −5.5% |
| 5 | 11.40x | 9.1536x | −19.7% |
| 4 | 7.30x | 8.8774x | +21.6% |
| 1 | 10.73x | 8.3303x | −22.4% |
| 10 | 7.45x | 6.5651x | −11.9% |
| 9 | 5.38x | 4.8355x | −10.1% |
| 8 (fp16 stack) | 2.04x | 2.0162x | −1.2% |
| **geomean** | **10.32x** | **9.68x** | **−6.2%** |

### The last two runs

**Shape 9** (one attention head, the shape where the non-shipping k004 was weakest on
the whole board at 1.1723x) returned **4.8355x**. The preregistered falsifier — "below
3x means the megakernel also collapses at a single head" — did not fire. The megakernel's
head-count dependence really is much flatter than k004's.

**Shape 8** is the only shape on the other branch: `d_model` 1024 goes to
`k010_fused_ln.py`, an fp16 tensor-core stack, sha `bda8f703...` rather than the
megakernel's `2b96a7c3...`. It returned **2.0162x** against a pre-gate 2.04x. The
falsifier — "at or below 1.3x means the large-`d_model` branch is decorative" — did not
fire either. The branch earns its place, and it is honestly the weakest shape on the
board because the baseline there is already at 64% of the fp16 roofline.

Worth recording: `research/gemm-epilogue-fusion.md:11` claims k010 took shape 8 to
**2.13x**, and the pre-gate board said **2.04x**. Measured is **2.0162x**. The research
note was the more optimistic of the two records and the less accurate one, by 5.4%. A
development-time figure written into a note becomes a claim; this is LESSONS 24 again,
caught early this time.

### What this campaign actually established

1. **The megakernel wins by doing less work, not by recovering launch gaps.** Shape 5
   has 1.0% baseline device idle — almost no gaps to recover — and still returns 9.1536x.
   Block-resident registers delete memory traffic. This is why the auditor's
   idle-fraction ceiling argument was structurally unable to bound this mechanism, and
   why my accepting it as fact was the error, not the arithmetic in it.
2. **Neither MFU nor idle fraction orders the speedups.** Shape 13: 28.41x at MFU 0.62.
   Shape 7: 21.96x at MFU 0.11. Shape 5: 9.15x at 1.0% idle. Shape 2: 14.39x at 86% idle.
   Any single-variable story about this board is wrong.
3. **The withdrawn MFU table is also approximately vindicated.** Recomputing shape 1 from
   the measured 8.3303x gives 12.15 TF/s / MFU 0.375 against the table's 11.63 / 0.36 —
   4.5% apart. I withdrew that table on the assumption it was inflated by the same factor
   as the speedups; that assumption was computed from the wrong kernel. Reinstated as
   indicative, still owed a clean recompute.
4. **Procedurally invalid and numerically wrong are different failures.** Mean delta
   −5.6%, scatter both ways, uncorrelated with idle. The pre-gate board was approximately
   right and improperly obtained. Withdrawing it was still correct — a number obtained
   without a permit against an uncalibrated baseline is undefended, not vindicated, when
   it later turns out close.

### Numeric prediction bands: final record 0 for 26

Shape 8 missed a ±2% band centred on a figure copied directly off the prior record, by
**0.2%**. Twenty-six attempts, zero hits. Bands stay retired per the preregistered
commitment on card C11; the field is filled because the gate requires it and nothing is
ever concluded from it. The *qualitative* falsifiers, by contrast, went the distance —
every one named in advance what would change the report, and several fired against claims
I wanted to be true.

### Drafts updated

All three judge-facing drafts now carry the measured board with its three caveats
(excludes shape 6; screening lane so nothing promotable; no audit verdict bound to any
row). The tech report keeps the full correction history — 10.32x withdrawn, 2.94x wrong
kernel, 9.68x measured — rather than silently landing on the final number, because the
process failure is the more interesting artifact.

### Still owed, and owner-only

Audit recording remains broken: three attempts, three distinct failures, retry cap
exhausted, entry `run-be8e56a55edd1926a84bf5d1efc0b154` permanently in `owner_attention`.
The fix is in `audit_champion.py`, inside the LOCK, Write-denied to me. That is the
correct arrangement and I am not going to route around it.

---

## 31 Aug ~02:20 SGT — BOTH extreme-shape evaluators are broken by one line. Owner-only.

With the board finished I went after the next-weakest evidence in the report: the shape 6
and shape 14 packets, which are labelled PROVISIONAL (one seed each, citing a
pre-integration submission sha). The report says they are "being re-captured". **They
cannot be.** Both side-evaluation lanes fail immediately, for the same reason, and the
reason is a bug in our own tooling — not in the official script, not in our kernels.

Shipped submission sha: `4da76db6b458042e434e00ed8129c4af04fc5a3d34b5d12c95b7c6066419dcc6`.
Lock verified valid (29 protected files) immediately after, so the official script is
untampered.

**shape 14** — `run-f5d6b57a5bc53c7bc75ae267df24deb4`, stderr blob
`128e2dd5...shape14-validate.stderr`:

```
File "/sandbox/Project/tools/shape14_eval.py", line 275, in official_case
AssertionError: official no-padding case must return an all-true CUDA mask
```

**shape 6** — stderr blob `2a7817de...shape6-eval.stderr`:

```
File "/sandbox/Project/tools/shape6_local_eval.py", line 147, in official_case
AssertionError: official generator returned an invalid mask
```

### The diagnosis, and why it is certain rather than plausible

Both evaluators set `device = torch.device("cuda")` — no index
(`shape14_eval.py:289`, `shape6_local_eval.py:296`) — then assert
`mask.device != device` on the mask the official generator returns.

`torch_transformer_benchmark.py:255-259` builds that mask as
`torch.ones(..., device=device, dtype=torch.bool)`. Materialising on `torch.device("cuda")`
produces a tensor whose `.device` is **`cuda:0`**, with an explicit index. In PyTorch
`torch.device("cuda:0") != torch.device("cuda")` is **True** — device equality compares
type *and* index, and `None != 0`. So the assertion fires on every invocation.

The shape-6 file is what makes this certain rather than a guess: it splits the checks
across two statements. Line 146 tests `dtype != torch.bool or mask.device != device`;
line 148 separately tests `not mask.all()`. **Line 146 is the one that fired.** The
official generator hardcodes `dtype=torch.bool`, so dtype cannot be the failing term, and
the all-true content check is a different assertion that was never reached. Only the
device comparison is left.

So: the mask is correct. The tensor is correct. The kernels are not involved. Two
evidence tools reject a valid input on a device-index technicality.

### What I did not do

`Project/tools/` is Write-denied to me, and correctly so — these are the files that decide
whether our extreme-shape claims are true. **Owner-only fix**, one line in each:

- `Project/tools/shape14_eval.py:274`
- `Project/tools/shape6_local_eval.py:146`

Compare `mask.device.type != device.type`, or normalise the target first with
`device = torch.zeros(0, device=device).device` so both sides carry an index.

### What this changes in the report, tonight

The drafts currently say the provisional packets "are being re-captured against the
shipped submission with ≥5 seeds". That is now false, and I have corrected it in all three
to say the re-capture was **attempted and is blocked on an owner-only tooling fix**. The
existing packets stay, with their real provenance: one seed, pre-integration sha.

Worth noting how these packets ever existed. Since the check fires unconditionally on this
torch, they must have been produced before these device assertions were added, or under a
build where the comparison held. I cannot date that from the record, so I claim only what
is provable: **the packets in the tree cannot be reproduced by the tools now in the tree.**
An evidence artifact that its own generator can no longer regenerate is a weaker artifact
than it looks, and the report should not imply otherwise.

---

## 31 Aug ~02:45 SGT — The absolute timings were in the packets all along. Three findings.

Every measurement packet carries `baseline.median_ms` and `candidate.median_ms`, not just
the ratio. Nobody had read them as a set. Extracting all twelve gives the complete picture
and it settles three things I had been guessing at.

| shape | GFLOP | baseline ms | candidate ms | speedup | achieved TF/s | MFU vs 32.4 |
|---|---|---|---|---|---|---|
| 1 | 7.52 | 4.7514 | 0.5704 | 8.3303 | 13.18 | 0.41 |
| 2 | 0.12 | 1.7392 | 0.1208 | 14.3939 | 0.99 | 0.03 |
| 3 | 0.47 | 1.7720 | 0.1403 | 12.6314 | 3.35 | 0.10 |
| 4 | 1.88 | 1.7363 | 0.1956 | 8.8774 | 9.61 | 0.30 |
| 5 | 15.03 | 9.3358 | 1.0199 | 9.1536 | 14.74 | 0.45 |
| 7 | 0.67 | 3.1713 | 0.1444 | 21.9645 | 4.64 | 0.14 |
| 8 | 420.91 | 38.4379 | 19.0649 | 2.0162 | 22.08 | 0.68 |
| 9 | 7.52 | 2.7085 | 0.5601 | 4.8355 | 13.43 | 0.41 |
| 10 | 7.52 | 3.6168 | 0.5509 | 6.5651 | 13.65 | 0.42 |
| 11 | 7.52 | 11.6337 | 0.9175 | 12.6797 | 8.20 | 0.25 |
| 12 | 1.68 | 1.7275 | 0.1597 | 10.8141 | 10.52 | 0.32 |
| 13 | 120.26 | 166.579 | 5.8634 | 28.4098 | 20.51 | 0.63 |

### 1. The withdrawn utilisation table was UNDERSTATED, not inflated — my third directional error tonight

Recomputing achieved throughput from the measured candidate medians makes **all twelve
rows higher** than the table I withdrew, by **2.5% to 28.2%**. I withdrew it assuming it
was inflated by the same factor as the speedups. It was wrong the other way: the pre-gate
`cand ms` came from earlier, slower kernels.

That is now three times in one night that I assumed an error's *direction* instead of
measuring it — "the board is systematically inflated" (it was within 5.6%), "the k009/k004
ratio is a constant 1.77" (it spans 1.76–6.31), and now this. LESSONS 37 is the general
form and it has earned its place.

### 2. The per-shape speedup spread is mostly a property of the BASELINE

Shapes 1, 9, 10, 11 are the same problem four times — identical batch, sequence, d_model,
ffn, layers, differing only in head count 4/1/2/16. Attention FLOPs are `B·H·S·S·(d/H)·2`
so `H` cancels: **all four are 7.52 GFLOP**, which the roofline table independently agrees
with and which I re-derived by hand term by term.

- **Baseline: 2.7085 → 3.6168 → 4.7514 → 11.6337 ms.** A **4.3x** degradation across the
  range, for arithmetic that never changes. The official implementation reshapes and
  processes per head, so its op count scales with `H` while its FLOPs do not.
- **Ours: 0.5601 → 0.5509 → 0.5704 → 0.9175 ms.** Flat within **3.5%** from 1 to 4 heads.

So "4.84x on shape 9, 12.68x on shape 11" is mostly a statement about the baseline. **Shape
9 is not our weak point; it is the baseline's strong point** — one head is one cleanly
shaped matmul and there is less waste to remove. I had been treating shapes 9 and 10 as
this submission's weaknesses in the card hypotheses. That reading was wrong, and the
falsifier I preregistered on shape 9 ("if it lands below 3x the megakernel also collapses
at a single head") pointed at the wrong variable even though it did not fire.

### 3. But we DO pay 63.7% at 16 heads, and the documented cause explains at most a fifth

Shape 11's candidate is 0.9175 ms against a 0.5605 ms sibling mean — **+63.7%** on
identical FLOPs, MFU 0.41 → 0.25.

The obvious explanation is `small-head-dim-padding.md`: head_dim 8 pads to 16 for the
`tl.dot` minimum. **I did the arithmetic before crediting it** (LESSONS 32, applied
prospectively for once rather than after an auditor caught me). Per layer: QKV 805.3,
out-proj 268.4, attention 268.4, FFN 536.9 MFLOP — total 1879, of which attention is
**14.3%**. Doubling attention adds at most 14.3%. Measured is 63.7%. **Padding accounts for
at most about a fifth; the remaining ~50% is unexplained.**

Hypothesis, labelled as such: at head_dim 8 the per-head tiles are far under the
tensor-core tile shape, so K2's inner loop runs 16 iterations of badly-shaped work — a
loop-length and tile-shape effect, not a FLOP-count one. **Shape 7 discriminates it**: also
head_dim 8, but only 4 heads, and it is the second-best shape on the board at 21.96x. The
isolating run is a torch-profiler diagnostic comparing K2 device time on shapes 11 and 1
with identical bytes — a diagnostic permit, no attempt budget. **Not run.**

This makes **shape 11 the clearest optimization target left**: the only shape measurably
inefficient against a directly comparable sibling on identical arithmetic. Full closure
would take it to ~20.8x and the geomean to ~10.1x, **+4.2%**. That assumes the gap closes
completely, which nothing supports, so it is a bounded target and not a plan, and per
LESSONS 35 no prediction band is attached.

Written up as `Project/research/head-count-scaling.md`; INDEX updated, so the index hash
changes and the next research cycle must cite the new one.

---

## 31 Aug ~03:00 SGT — I ran my own discriminator and it killed my own hypothesis.

Rather than leave the ~50% open, I ran the isolating experiment I had just named. Two
torch-profiler diagnostics on identical k009 bytes, 20 iterations each, diagnostic lane so
zero attempt budget: `profile-1127da61e6696eed99c7d67f` (shape 11) and
`profile-923ddf58435a05b18e66a4ba` (shape 1).

| kernel | shape 1 (4 heads) | shape 11 (16 heads) | ratio |
|---|---|---|---|
| `_attn_block_tail` (K2) | 74.054 us | 157.653 us | **2.129x** |
| `_norm_qkv` (K1) | 49.046 us | 50.933 us | 1.038x |
| `_final_norm` | 20.278 us | 20.683 us | 1.020x |
| DtoD copy | 19.783 us | 19.989 us | 1.010x |
| device time / forward | 558 us | 901 us | 1.614x |

Device-time ratio 1.614 against a wall ratio of 1.637 — the two instruments agree, so the
profile is describing the kernel and not itself (LESSONS 31's failure mode did not recur).

**The penalty is entirely inside K2. Everything else is flat within 4%.**

**And K2 is 2.13x, not 4x.** I had preregistered the discriminator explicitly — "4x means
loop-length, ~2x means padding" — precisely so I could not reinterpret the result after
seeing it. It came out 2.13x. **My loop-length hypothesis is refuted and the head_dim-8
`tl.dot` padding is confirmed as the locus.** The hypothesis stays written in the research
note; deleting a killed hypothesis is how a notebook turns into a press release.

**What survives from my earlier arithmetic is the magnitude objection.** Within K2 the work
is out-proj 268.4 + attention 268.4 + FFN 536.9 = 1073 MFLOP per layer; padding takes it to
1341, **+25%**, while K2's time rises **+113%**. K2's achieved throughput falls from
**14.5 TF/s to 8.5 TF/s**. So it is one mechanism with two costs — the padded dots do twice
the arithmetic *and* run 41% less efficiently — rather than padding alone (too small) or a
separate loop-length effect (refuted).

**Corollary that matters for any future work here:** attention is 14.3% of the block's
FLOPs but dominates K2's *time*, since doubling attention roughly doubles a kernel that
also contains the output projection and the whole FFN. On these shapes attention is the
expensive part of the fused block despite its small FLOP share. That inverts the intuition
the roofline table encourages.

This is the first time in the campaign I named a falsifiable discriminator, ran it, and had
it come back against me *without an auditor having to catch it* — which is what the
scaffolding was built for. Cost: two diagnostic permits, no attempt budget, about four
minutes.

---

## 31 Aug ~03:25 SGT — The proxy gap is CLOSED. The board describes what ships.

Card C18 wrote down a caveat on 31 Aug 01:12 and then nobody tested it for two hours:

> "k009 is the megakernel CLASS the dispatcher ships, not literally the dispatcher bytes,
> because dispatcher_region.py references BaselineTransformer as a free name and cannot be
> loaded standalone as a candidate module."

Every one of the twelve rows measured a kernel module selected because its **header matches
the dispatcher's description**. That is a documentation match. It is a better grade of
evidence than the runbook example that produced the k004 disaster, but it is the same
species, and this campaign exists because of that species.

`dispatcher_region.py` genuinely cannot be loaded standalone. But
`Project/submission/torch_transformer_benchmark_submission.py` **can** — it is the whole
official script with only the sanctioned region replaced, so it defines `BaselineTransformer`
and `UserOptimizedTransformer` both. That had simply never been tried as a candidate impl.

**It works, and it reproduces the board.**

| shape | branch exercised | shipped file (sha 4da76db6...) | kernel-module proxy | agreement |
|---|---|---|---|---|
| 13 | megakernel, d_model 128 | **28.2849x** | 28.4098x (k009) | **0.44%** |
| 8 | fp16 stack, d_model 1024 | **2.016012x** | 2.016164x (k010) | **0.008%** |

Two shapes, deliberately chosen as the minimum pair that exercises **both sides of the only
branch the dispatcher takes**. Shape 13 alone would only have shown the file loads and runs;
shape 8 is what tests the routing predicate itself. If the threshold were wrong or the
branches inverted, shape 13 would still have measured correctly while shape 8 collapsed.

So the twelve-shape board is now a claim about **the artifact that ships**, not about two
kernel modules that resemble it. The tech report §2.1 says so, with the agreement figures.

### The one band hit in twenty-eight attempts, and why it is not a redemption

Both shipped-file bands hit (28.06–28.76 vs 28.2849; 1.99–2.04 vs 2.0160). After twenty-six
straight misses that looks like the forecasting finally worked. **It is not, and the
distinction is the whole point.** Every miss came from predicting what a *mechanism* would
deliver. These two bands were derived by taking a **direct measurement of nearly identical
bytes** and asserting the difference would be small. That is not forecasting, it is a
consistency check — the band is doing the job of a regression test, not a prediction.
Numeric bands stay retired for mechanism claims per LESSONS 35. If anything these two hits
sharpen the rule: I can predict that two nearly identical things measure alike, and I cannot
predict what a change will buy.

---

## 31 Aug ~04:00 SGT — All twelve on the shipped file. Headline is 9.45x. One model killed.

Two shapes were not enough. Finished the job: all twelve measured on
`torch_transformer_benchmark_submission.py` (sha `4da76db6...`) via the delta lane, one
fresh permit per row, all `correct: true`.

| shape | shipped file | kernel module | delta |
|---|---|---|---|
| 13 | 28.2849x | 28.4098x | −0.44% |
| 7 | 20.9595x | 21.9645x | −4.58% |
| 2 | 13.1434x | 14.3939x | −8.69% |
| 3 | 12.9618x | 12.6314x | **+2.62%** |
| 11 | 12.5909x | 12.6797x | −0.70% |
| 12 | 10.4348x | 10.8141x | −3.51% |
| 5 | 9.1150x | 9.1536x | −0.42% |
| 4 | 8.9211x | 8.8774x | **+0.49%** |
| 1 | 8.1673x | 8.3303x | −1.96% |
| 10 | 6.5352x | 6.5651x | −0.46% |
| 9 | 4.3503x | 4.8355x | −10.03% |
| 8 | 2.0160x | 2.0162x | −0.008% |
| **geomean** | **9.45x** | 9.68x | **−2.4%** |

**The headline is now 9.45x** and every judge-facing draft says so. The module board is
demoted to a cross-check. This is the fourth headline this campaign has had — 10.32x
withdrawn, 2.94x wrong kernel, 9.68x kernel modules, **9.45x shipped artifact** — and each
move was toward measuring the thing that actually ships.

### The fixed-dispatch-cost model: preregistered, then killed on its first out-of-sample point

After four shipped-file rows the deltas looked beautifully ordered by candidate time:
shape 8 −0.008% at 19.06 ms, shape 13 −0.44% at 5.86 ms, shape 1 −1.96% at 0.570 ms,
shape 2 −8.69% at 0.121 ms. That is the exact signature of a **fixed** per-forward cost,
and a mechanism was sitting right there: the shipped `UserOptimizedTransformer` inspects the
incoming shape on every call to pick a branch, and the bare kernel module does not. A fixed
10 microseconds predicts 0.05, 0.17, 1.75 and 8.3 percent against 0.008, 0.44, 1.96 and
8.69 measured. Four points, one free parameter, a plausible physical story.

I wrote it into card C32 with **eight out-of-sample point predictions before running any of
them**, precisely so it could not be tuned afterwards.

**Shape 3 came in at +2.62 percent — faster than its module.** A fixed overhead cannot make
anything faster. Refuted on the first test.

Then refuted a second, independent way: shapes 1, 9 and 10 have candidate times within 4
percent of one another (0.5704, 0.5601, 0.5509 ms) and landed at −1.96, −10.03 and −0.46
percent. The gap is not a function of candidate time at all.

**What it actually is:** ordinary cross-invocation scatter, exactly LESSONS 11. Baseline and
candidate are paired *inside* each run, so every row is internally valid; comparing two
rows from separate invocations carries the documented up-to-9-percent variance. Shape 9 at
−10.03 percent is an outlier, not a trend.

**Why I am pleased about this rather than embarrassed.** This is the same failure mode as
LESSONS 37 — inferring a mechanism from a suggestive pattern — but this time the scaffolding
caught it in one attempt instead of an auditor catching it after it reached the report. The
cost was one screening run. The difference was preregistering the model with out-of-sample
predictions *before* looking. That is the whole point of the machinery, and it is now the
second time tonight it has worked prospectively (the other being the shape-11 K2
discriminator, which also killed my hypothesis).

It also goes in the tech report. A four-point pattern that dies on its fifth point is worth
more in a methods section than a tidy model nobody tested.

---

## 31 Aug ~04:30 SGT — Hostile read of all three drafts. Two real defects found.

Read the three judge-facing documents end to end as an adversarial reviewer rather than as
their author. Stale numbers were expected and there were several (9.68x left where 9.45x
belongs, 28.41x ranges, torch.compile comparison rows). Two findings were not cosmetic.

### 1. A baseline/candidate mix-up that inverted the meaning of a number

Both the tech report and the README said shape 8's **baseline** was "already at 64% of the
fp16 roofline before we touch it". **That 64% is our candidate's figure**, read straight off
the `cand ms` column of `roofline-table.md`. The baseline runs at
420.91 GFLOP / 38.4379 ms = **10.95 TF/s, MFU 0.34**. Our kernel reaches **22.08 TF/s, MFU
0.68**.

So the sentence claimed our starting point was where we finished. The qualitative
conclusion — shape 8 is arithmetic-bound and has the least headroom — survives, but the
supporting number was the wrong side of the comparison, and as written it made 2.02x look
like scraping a nearly-full roofline rather than what it is: **doubling achieved
utilisation**, which is a better story told accurately.

This error also went into a gate plan hypothesis and is therefore in the immutable log, so
the report carries an inline correction rather than a silent edit. Root cause is the same
one as LESSONS 32 and 37: I read a table column without checking which arm it described.

### 2. The video script's live demo command does not run

Scene 3 had the owner type, on camera:
`python3 Project/harness/runner.py run --shape 13 --impl Project/kernels/k009_fused_tuned.py`
with "beats while it runs" underneath. **I ran it. It fails instantly:**
`trusted_controller.py run: error: the following arguments are required: --permit`.

Since the LOCK, `runner.py` is a 27-line shim onto the trusted controller and the controller
times nothing without a one-use permit. The script would have died on camera in the middle
of the trust-chain scene.

Rewritten so the refusal **is** the scene, which is stronger and true: show the error, say
"even I can't start a measurement", then run the real permitted three-command sequence. Also
flagged that the overnight owner capabilities expire around **21:00 on 31 Aug**, inside the
recording window, so a fresh short-lived capability must be minted before filming.

The same dead command was in the reproduce blocks of both the README and the tech report.
Both now show it as refusing by design and point at the RUNBOOK for the permitted sequence.

**Transferable point:** every command a document tells someone to type is a claim, and it
decays exactly like a number does. The LOCK changed what `runner.py` is and three documents
kept the pre-LOCK invocation. Test the commands, not just the figures.

### 3. Following that rule immediately found a bigger one: a false provenance claim

Having written "test the commands", I tested the rest of them.

- `python3 Project/tools/sensitivity_board.py` — **works**, writes
  `Project/results_side/SENSITIVITY.md`.
- `python3 Project/harness/runner.py leaderboard` — **dead**. The shim forwards only the
  controller's subcommands and `leaderboard` is not among them.

That second one matters more than a broken command, because two documents built a claim on
top of it: *"Every number in this report regenerates from it with one command"* (tech report
§5.1) and *"Every number we publish regenerates from `Project/results/JOURNAL.jsonl`"*
(README). **Both are false**, and not only because the command is gone.

I checked where the headline rows actually are. Confirmed by inspecting the last commit's
file list: **no `Project/results/JOURNAL.jsonl` change and no scratch-ledger files were
written at all.** The post-LOCK board lives entirely in

- `Project/authority/events.jsonl` — `measurement_recorded` events, append-only;
- `Project/authority/blobs/<packet_sha>.json` — content-addressed packets holding the full
  300-sample baseline and candidate distributions, permit id, candidate sha256, environment;
- `Project/loop/gate_log.jsonl` — the scientific record of why each run happened.

`JOURNAL.jsonl` holds the **pre-LOCK** history. That is not a defect: screening-lane runs
write to the scratch namespace precisely so a characterisation run cannot be mistaken for a
champion, which is the same design fact as "none of this is promotion-eligible". But the
report was describing the pre-LOCK provenance story while presenting post-LOCK numbers.

Corrected in all three documents with the real locations. The honest version is actually
stronger: content-addressed packets bound to consumed permits are better provenance than a
regenerable text table, and we can say exactly where every figure came from.

**Second-order lesson:** a claim about *infrastructure* decays as silently as a claim about
a number, and it is worse, because a reader who tries it concludes the whole project is
broken. The trigger for finding this was writing down a rule and then actually obeying it in
the same session.

---

## 31 Aug ~05:00 SGT — Counted the audit ledger. 28 of 81 are rule violations against us.

The drafts said "60+ verdicts ... including the RULE_VIOLATIONs that made us change the
code", which is the kind of phrasing that sounds like two. I counted all 81 lines of
`Project/audits/verdicts.jsonl` by hand:

| verdict | count |
|---|---|
| PASS | 42 |
| RULE_VIOLATION | **28** |
| NEEDS_CONTEXT | 8 |
| RETEST | 3 |

**35% rule violations.** A judge who opens that file sees it immediately, so the report now
states the distribution and explains it rather than rounding it into a friendly phrase.

### What the violations actually are

Read two responses from different batches. Both are **procedural, not integrity findings**,
and both say so in the auditor's own words:

- `audit_20260829-012829-e03be5`: *"No plan entry precedes the champion timestamp ...
  failing the owner-mandated citation gate"*, while the summary states *"Source inspection
  found no timer manipulation, harness access, input mutation, or stale-output reuse"*.
- `audit_20260828-193920-ae5c64`: *"The speedup itself is credible and the promotion
  mechanics are consistent, with no evidence of timing or cache gaming. The entry fails
  audit because it lacks the required contemporaneous citation plan."*

So the 28 cluster on **pre-gate runs that cannot prove which plan produced them** — which is
the exact defect that motivated building the gate. Reported as sampled (2 of 28), not
asserted for all.

### The two findings that changed code are real and now quoted verbatim

The video script's line — "the auditors caught a provenance gap and a latent masking bug in
code the benchmark never exercised" — **verifies**. Both are in
`audit_20260828-193920-ae5c64`:

- **Masking bug**: k005 "selects the Triton attention path without considering
  valid_token_mask and never masks invalid keys, so padded inputs produce behavior different
  from the baseline. The benchmark used padding_ratio=0.0, so this latent bug did not affect
  the recorded shape-8 samples, but it remains an explicit rule violation."
- **Provenance gap**: "candidate_source_matches_journal is false ... the exact measured
  source was recoverable from Git revision e7860ee ... but a blind auditor should not need
  repository-history reconstruction."

### The auditor beat me to one of my own findings

`audit_20260829-012829-e03be5` says *"Shape 11's eager baseline is especially inefficient at
16 heads with head dimension 8"* — on **29 Aug**, two days before I measured the head-count
axis and found the baseline degrades 4.3x across it. It was sitting in the ledger the whole
time. Cited in the report now as independent corroboration, and it is a small rebuke: the
research base had the answer and I re-derived it from scratch.

### Also corrected: the seed count was understated

Both drafts said correctness was checked on "five seeds". The packets show **seven trials** —
five fixed (1234–1238) plus two drawn at random per run, which is materially better because
a candidate cannot be tuned to a known seed list. Fixed in both. Verified `suspicious:false`
and event/wall agreement 1.005 on the shipped shape-13 packet while I was in there.

---

## 31 Aug ~05:30 SGT — I probed our own headline control claim and it did not fire.

The tech report §7 said, and `HANDOVER.md` says more specifically:

> "16 uncleared RULE_VIOLATION rows will freeze permits the moment the gate opens."

There are now **28** uncleared RULE_VIOLATION rows and `cleared_verdicts: []` in the gate
state. Yet this campaign issued and consumed ~40 permits without ever meeting a freeze. That
is either a design nuance or a broken control, and the difference matters enough to test
rather than reason about.

**The probe.** Submitted a `delta` in the **primary optimization lane** — not the screening
lane everything else used — on shape 13.

1. `run_gate.py delta --mode optimization` → **DELTA accepted.**
2. `trusted_controller.py issue-permit` → **permit issued, `"may_promote": true`.**

No freeze at the plan boundary and none at the permit boundary, with 28 uncleared hard
verdicts sitting in the ledger. I did **not** execute it; the permit
(`permit-a5ccbeb47e97a4ed955e856acd916f4f`) was left to expire unused, which leaves one
`open_permits: 1` and a PENDING reconcile until roughly 04:54 SGT. **Reconcile again next
tick so the owner does not inherit a pending request.**

**The likely explanation, and why it does not rescue the claim.** The 28 verdicts are bound
to **pre-LOCK journal entry ids** (`20260828-…`), not to campaign run ids (`run-…`), so they
do not gate `CAMP-POSTLOCK`. That is defensible design. But it carries its own admission:
**because audit recording is broken, no post-LOCK row carries a verdict of any kind**, so
the freeze has not been exercised once since the gate went live. It is implemented and it is
covered by the test suite; it is not something this campaign demonstrated.

**Why this goes in the report rather than getting quietly softened.** The whole §7 argument
is "a verdict with no mechanical consequence is a comment". Publishing that argument while
carrying an untested version of exactly that mechanism would be the same failure the section
is about. The report now states the probe, the result, and the explanation — and explicitly
says the control is *unexercised, not demonstrated*.

**Also corrected in §9:** it claimed "the final board is a median of repeated sweeps". It is
not — **each row is the median of 300 paired samples inside a single invocation** (warmup 20
/ repeats 100 / rounds 3, alternating). We deliberately never average across invocations,
because §6 forbids comparing absolute latencies across processes. And the noise figures were
conflated: within-invocation calibration noise is 0.03–0.4%, cross-invocation clock drift on
identical work is ~9%, shipped-versus-module scatter is −10.0% to +2.6%, and the old "±25%"
came from two uncontrolled pre-gate boards and should not be quoted for these.

---

## 31 Aug ~06:00 SGT — I WAS WRONG ABOUT THE BRAKE. It fired; the owner lifted it properly.

Thirty minutes ago I wrote into the judge-facing report that our headline control "did not
fire", on the strength of a probe that granted a promotion-capable permit while 28
RULE_VIOLATION lines sat in `verdicts.jsonl`. **That conclusion was wrong and I have
corrected it.** The probe result was real; my inference from it was not.

`Project/audits/audit_events.jsonl` is the authority — a hash-chained log where every event
carries `previous_event_sha256`, starting from all-zeros. `verdicts.jsonl` is a *display*
ledger. Reading the authority settles it:

- **seq 1–16: sixteen `FINDING_ACCEPTED_ROW_RETIRED` resolutions**, recorded
  2026-08-30T20:54:23–25, each consuming a **separate signed capability nonce**, each
  chained to the previous, all carrying the same rationale: *"Finding accepted, not
  overturned. The 30 Aug audit correctly found this row has no plan, quoted source or
  reasoning chain, because the run predates the citation gate... **This resolution removes
  the verdict's brake on NEW permits.** It does not rehabilitate the row and makes no claim
  about its numbers."*

So the sequence was: 16 hard verdicts braked the gate exactly as HANDOVER predicted → the
owner did the deliberate reconciliation pass with 16 signatures → the brake released. When I
probed afterwards there were **no unacked hard verdicts left to fire on**, and the permit was
correctly granted. **The control worked. I mistook a correctly-released brake for an absent
one**, because I counted rows in the wrong file.

### Why I got it wrong, and the rule that follows

I counted `verdicts.jsonl` (28 RULE_VIOLATION lines) and treated that as the brake's input.
It is not — LESSONS 25 already recorded that the brake reads `unacked_hard_verdicts` from
the audit authority and that `cleared_verdicts` in gate_state is display state. **The
information needed to interpret my own probe was in my own LESSONS file, written yesterday,
and I ran the probe before reading it.** That is LESSONS 43 recurring inside a single night:
the research base already had the answer.

New rule, now LESSONS 45: when probing a control, identify its *authoritative input* before
running the probe, and state in advance what a pass and a fail each look like. I did neither,
so a null result looked like a failure.

### What actually goes in the report, and it is better than what I withdrew

The corrected §7 tells the stronger true story: the brake fired on 16 real findings; lifting
it cost 16 owner signatures; and the **resolution vocabulary was itself an integrity
decision** — the only label originally accepted against a RULE_VIOLATION was
`FINDING_OVERTURNED` ("the auditor was wrong"), which would have written a false statement
into a permanent ledger to buy a brake release, so `FINDING_ACCEPTED_ROW_RETIRED` was added
instead (LESSONS 26). The honest limit that survives: **the brake has never fired on a
post-LOCK row**, because the recorder broke before any campaign row could be adjudicated.

### Also confirmed while in the authority log

- seq 17–23 document the audit failure precisely: one `audit_enqueued` for
  `run-be8e56a55edd1926a84bf5d1efc0b154`, then three `attempt_started`/`attempt_failed`
  pairs — `AUDITOR_PROCESS_ABANDONED_WITHOUT_TERMINAL_EVENT`, then *"auditor stdout must be
  exactly one duplicate-free JSON object with no banners"*, then abandoned again. Three
  attempts, three distinct failures, exactly as STATE records.
- The auditor binary for those attempts was **`/usr/local/bin/claude-auditor`**
  (sha `fd5f10ff…`), i.e. the Claude fallback backend, not Codex — root-owned path per
  LESSONS 29. The README already says the backend is a choice with a weaker independence
  claim; the tech report §4 should say which backend ran last.

### §8 negative results: all four verified against sources

- **int8**: report said "~2–3% output error". `quantization-tolerance.md` records the
  measurement as **max abs err ~3.5e-2 with ~12% violations** at d=1024, L=4. Report now
  cites the measured figures rather than the rounded recollection.
- **QKV chunk split**: LESSONS 21 confirms verbatim — k011, 3× occupancy, 3× traffic,
  ~15% slower everywhere, and the "2.3× off memory floor" was latency not starvation.
- **single-CTA shape 2**: `megakernels-persistent.md` confirms 117.44 MFLOP → ~137 µs floor
  vs 144.4 µs champion → ~5% available. Report now quotes the arithmetic.
- **head-splitting tie**: stated in the report, not yet traced to a source file. Left as is
  and flagged here as the one §8 claim I could not verify.
- Fixed a **dead cross-reference**: §8 pointed at "§5.5", which does not exist; §5 is a
  numbered list. Now points at §5 item 5 and the research note.

---

## 31 Aug ~06:30 SGT — Sixth audit pass. Four more, including a board that contradicts §2.3.

**Permit reconciled.** The probe permit is settled on the gate side (reconcile now returns
clean). `trusted_controller.py status` still shows `open_permits: 1` and always will: that
counter is `permits_issued - permits_consumed`, and an issued-but-expired permit is never
consumed, so it never decrements. Not a stuck state — noted so the owner does not read it
as one.

### 1. SENSITIVITY.md contradicts §2.3 and the report was citing it as "the full board"

`Project/results_side/SENSITIVITY.md` regenerates cleanly, but its own footer says **"Every
number above is pre-gate (measured before the authority-v4 guard paste)"**. It is built from
the 29 Aug quiet-box sweep in `JOURNAL.jsonl` and **structurally cannot** show the post-LOCK
board, because those rows live in the authority log and its packets, not the legacy journal.
So it disagrees with the §2.3 table I recomputed — shape 1 reads 11.63 TF/s there against
13.18 here — and its headline `S3 geomean 10.95x` is one of the **withdrawn** figures.

The report was pointing a judge at it with the words "Full board". Now relabelled: read it
for the **scoring-convention logic** (five weightings that disagree about where the points
are, plus marginal value of a 20% win per shape), not for the numbers. Regenerating it
against post-LOCK medians is owed work I did not do.

### 2. The head-splitting negative result is unsourceable

Of §8's four negative results, three verify cleanly (int8 → `quantization-tolerance.md`;
k011 QKV-chunk → LESSONS 21 verbatim; single-CTA shape 2 → `megakernels-persistent.md`
arithmetic). **The head-splitting "statistical tie" cannot be traced anywhere**: not in
LESSONS, not in `Project/research/`, and not in the kernel roster, which
`LEADERBOARD.md` and `SENSITIVITY.md` between them fix at k000, k001, k003–k011, k014,
k015 — no head-splitting kernel exists on disk. It may be a mis-remembering of the k011
QKV-chunk result sitting immediately above it in the same list. Marked in the draft as
**unsourced, do not ship without checking**, per LESSONS 24 rather than silently deleted.

### 3. §4 now names the backend that actually judged

The table said "OpenAI Codex CLI / GPT-5.6". True for the 81 verdicts, but
`audit_events.jsonl` seq 18–23 shows the last three attempts ran
`/usr/local/bin/claude-auditor` (sha `fd5f10ff…`) — the Claude fallback. §4 now says so,
repeats the weaker-independence caveat the README already carried, and explains why the
binary lives at a root-owned path (LESSONS 29: a pinned hash is only a control when the
hashed thing is out of the agent's reach).

### 4. §6 was describing the pre-LOCK provenance mechanism

It claimed "each candidate is committed to git before it is ever measured". That was the
pre-LOCK rule, and the auditor had already found it insufficient — a packet could still
embed the current source hash rather than the measured one. Post-LOCK the controller
snapshots the candidate into a **content-addressed blob at permit issue** and binds that
sha into permit, request and packet; for every §2.1.1 row it is `4da76db6…`. The audited
and measured bytes are one object, not two asserted to match. Also added the timing
protocol (warmup 20 / repeats 100 / rounds 3, identical for calibration and measurement,
checkable per shape in `gate_state.json`) and quoted the event-versus-wall agreement
(shape 13 1.005, shape 8 1.0008, `suspicious:false` on every packet inspected).

---

## 31 Aug ~07:00 SGT — Seventh pass was clean. Audit closed; stating the diminishing returns.

Final end-to-end consistency read of the tech report. It found **three staleness issues, all
introduced by my own earlier passes**, and no new substantive defects:

- The correction header still said "DRAFT v3, §2 re-measured" and "one further factual
  correction in §3" — six passes had long outrun both. Now v4, with the twelve inline
  corrections enumerated and the real provenance chain (authority events + packets, with
  `JOURNAL.jsonl` explicitly pre-LOCK only).
- "the 0.87× it cost us" — a difference written as a multiplier. Now 8.4%.
- §3 quotes **module-board** figures (9.15x shape 5, 4.84x→12.68x head-count) because the
  k004-versus-k009 comparisons only exist on that board — you cannot run k004 "inside" the
  shipped file. Added a note saying so and giving the shipped equivalents, rather than
  churning every number.

**The audit is closed.** Passes 1–5 each found substantive defects, pass 6 found four, pass
7 found only self-inflicted staleness. That curve is the stopping signal. An eighth pass
would mostly be auditing my own editing.

### The honest assessment of what remains, written down so it is not re-litigated

With ~12 attempts and the freeze at 20:00, I considered and **declined** three pieces of
work, and the reasons matter more than the decision:

1. **Shape-11 de-padding.** Ceiling +4.2% geomean *at complete closure*, which nothing
   supports. `tl.sum`-reduce trades tensor cores for CUDA cores; `small-head-dim-padding.md`
   records the literature calling that "not recommended in general". Cannot be promoted
   (audit recording broken), cannot ship without the owner. Real correctness risk against a
   2e-3 tolerance, taken while the owner sleeps, for a number that would sit in a screening
   lane.
2. **Sequence-persistent CTAs for shapes 2/3/7/12.** Arithmetically the better target —
   four shapes at ~1.2x is ~+6.3% geomean, larger than option 1 — but it is *new* Triton
   work under the same promotion and approval blocks, and `megakernels-persistent.md`
   already ranks it last in the allocation.
3. **Regenerating SENSITIVITY.md post-LOCK.** Requires `sensitivity_board.py` to read the
   authority packets rather than `JOURNAL.jsonl`; `Project/tools/` is write-denied. Owner
   only, and low value now the draft says to read that board for scoring logic not numbers.

The through-line: **every remaining optimization is blocked behind a decision that is not
mine to make** (what ships) and a control that is not mine to repair (audit recording). The
deliverable judges actually score is the report, and the report is now measured on the
shipped artifact with fifteen defects removed. Grinding kernels that cannot be promoted,
audited, or shipped would be motion, not progress — and manufacturing that motion overnight
so the morning log looks busy is its own species of dishonesty.

Final integrity check before handing back: `verify-lock` valid, 29 protected files, no
watcher active, reconcile clean, tree clean, 0 strikes, nothing promoted.

---

## 31 Aug ~07:15 SGT — Defect 16, found by testing a claim the report makes about itself.

The report asserts: *"Values still owed at code freeze are marked `[PENDING]` and name the
run that produces them."* That is a checkable claim about the document, so I checked it, and
one `[PENDING]` names a run **that cannot execute**.

**Shape 14's full-batch timing.** Both drafts said the figure "comes from the
batch-decomposed evaluator run, which is **queued**". Reading `shape14_eval.py:753-777`
instead of the note:

- the `eval` subcommand — the one that produces the timing — requires
  `--validation-packet`, documented as *"passed shape14-oracle-validation-v2 packet"*;
- that packet is minted only by the `validate` subcommand, whose own help string labels it
  **`(gate)`**, and whose docstring says *"Gate for everything else"*;
- `validate` is exactly what aborts on the `cuda` vs `cuda:0` device-comparison bug at
  line 274.

So the timing is **blocked, not queued**, and blocked behind the same one-line owner-only
fix as the extreme-shape packet re-capture. Corrected in both drafts.

**Why this matters beyond the wording.** It re-ranks the owner's morning list. That
one-liner previously read as "regenerate some provisional evidence" — nice to have. It
actually gates **three** deliverables: the shape-6 packet, the shape-14 packet, and a
judge-facing `[PENDING]` number in two documents. It is now item 1 in STATE, marked highest
value, with the dependency chain written out.

**The method that found it, worth keeping.** I did not re-read the drafts looking for prose
errors — pass 7 established that vein is exhausted. I took a *self-referential claim* the
report makes ("every [PENDING] names its producing run") and tested it against the tools.
When a document asserts a property of itself, that assertion is the cheapest remaining place
to look for defects, because nobody audits the meta-claims. This is LESSONS 40's rule
("every command a document tells someone to type is a claim") extended one level: **every
promise a document makes about its own completeness is also a claim, and decays the same
way.**

---

## 31 Aug ~07:45 SGT — Meta-claim pass 2. Five results, including a regression I caused.

Applying LESSONS 46 to the remaining set-claims. This pass was the most productive since the
early ones, which retroactively justifies not stopping at pass 7.

### 1. I had deleted the byte-identity paragraph — my own regression

My first big §2.1 rewrite (the one that introduced the two-board structure) replaced the old
section intro wholesale, and that intro contained the **byte-identity claim**: everything
outside the sanctioned region is identical to the official script, proven by
`build_submission.py --verify`. That is a *core competition requirement* and it silently
vanished from §2 for several hours. Restored, with a framing I could not have written
before: the verify command is **not on the agent's allowlist**, deliberately, because an
agent must not certify its own submission boundary — so it is owner-run. What the agent can
attest is that the reference has not moved (`verify-lock` valid across 29 protected files,
official script among them). **Wholesale paragraph replacement is how you lose a claim you
never intended to touch; diff what you delete, not just what you add.**

### 2. "All 14 shapes pass precision" spans two very different evidence grades

True, but doing heavy lifting. The 12 with a runnable official baseline are verified under
the official predicate on **7 trials each, post-LOCK, on the shipped file**. Shapes 6 and 14
have **no official baseline at all**, so they are verified against validated oracles on
**one seed each, against a pre-integration file**, and those packets currently cannot be
regenerated. All three drafts now state both grades. The video script's bare bullet was the
worst offender and now carries the split explicitly, because it is the version most likely
to be said aloud unqualified.

### 3. "No external kernel library is wrapped" — VERIFIED in the shipped artifact

Read the sanctioned region's imports directly: `triton` and `triton.language`, nothing else,
inside a `try/except` setting `_TRITON_OK = False`. No FlashAttention, no xFormers. Better,
this surfaced a genuine strength nobody had written down: **the submission degrades to the
unchanged baseline path when Triton is absent**, including pre-softmax key masking, so a
judge on a Triton-less machine still gets numerically exact results. The fast paths are an
optimisation, not a dependency. Added to §3.

### 4. The test-suite claim — both endpoints now verified by running them

`HANDOVER.md` says "twelve test suites exist and pass"; `OWNER_HANDOFF_TONIGHT.md` says
"11 of 12, one stale". **The second is right**, and I confirmed both ends:

- `champion_watch_test.py` → **278/278 ALL GREEN** (HANDOVER's "278 checks" verified).
- `integration_authority_test.py` → fails exactly as described:
  `AttributeError: module 'runner' has no attribute 'is_primary'`. It asserts the pre-LOCK
  runner internals; the LOCK replaced that file with the shim. **Its two completed checks
  both PASSED** — the frozen benchmark publishes a complete timing protocol, and the
  controller's protocol is that same protocol — then it aborts. Stale test, not broken
  system.

### 5. The 278 tests are better evidence for §7 than the review rounds are

The suite explicitly covers the brake I misread yesterday: *"unauthenticated resolution is
refused"*, *"resolution with the wrong capability nonce is refused"*, *"resolution
authorized for another event hash is refused"*, *"only the exact authenticated resolution
clears it"*, *"the same verdict cannot be resolved twice"*, *"a legacy RULE_VIOLATION row
latches as an unresolved hard verdict"*. Also *"advisory technical verdict never rescues a
hard integrity one"*. §7 now cites these by name.

It also **pins a known defect as a passing test** — *"DEFECT, pinned: an aborted launch
leaves a durable open attempt that only the stale reaper can clear"* — which is the right
way to carry a limitation: executable, named, unforgettable.

### Still unverified, and now flagged in the draft

**"the final round returned APPROVE"** (§7, thirteen adversarial rounds). Our own memory
records a Codex round dying on a provider content filter having produced **no verdict
line**, with the standing rule that *a missing verdict is never an APPROVE*. I did not
confirm from raw logs that round 13 carries a real APPROVE. Flagged inline: verify or drop
the clause. The round count and the ~50 fixes stand regardless.

---

## 31 Aug ~08:15 SGT — I jammed `reconcile`, and found a second regression from my rewrite.

### The jam, which is mine and is owner-only to clear

Yesterday's control probe issued a primary-lane permit and deliberately never ran it. I
treated expiry as a clean exit. It is not. The gate recorded the request as **settled** with
no authority event explaining how — no run ever produced one — and now:

```
run_gate.py reconcile  ->  REFUSED: settled request lacks its reconciled authority event
run_gate.py delta      ->  same refusal
```

**Blast radius, measured command by command rather than asserted:** `research` and `plan`
both get through (they fail only on ordinary validation — "summary under 200 chars",
"research step required first"), and `status`, `verify-lock` and `champion_watch --dry-run`
all work. **Nothing already measured is altered** — the twelve-shape board, every packet and
the authority log are untouched. But a *new* run could not be reconciled or judged.

Recovery is `run_gate.py quarantine --request-sha256
07d20af31dc8cda7c31631a344b227fd596bf8c1cc01f00a44b709a2fb179583 --authority-receipt <…>`,
which needs an owner-signed receipt. My capabilities are `permit.issue` and
`register_family` only, so I cannot mint one — and should not be able to. **An agent that
can quarantine its own inconvenient gate state does not have a gate.** Raised as STATE
item 0 with the table, and flagged ignorable if no further measurement is wanted before
freeze.

LESSONS 47 is the general form: design the probe's *exit* before running it; prefer the
cheapest refusal boundary that answers the question (the plan step already told me most of
it, and escalating to a real permit bought one bit and cost a jammed gate); and when you do
jam something, publish the blast radius rather than "it's broken".

### Second regression from the same wholesale rewrite

Diffing commit `2512772` — the one that already cost the byte-identity paragraph — shows it
also deleted the old §2.1 table's **per-shape failed-element counts**:
`PASS (0/5,242,880 failed)`, `PASS (0/81,920)`, and so on. My replacement reduced twelve
quantified correctness results to the bare word "PASS".

Restored with post-LOCK numbers from the packets I had already read, all twelve verified
individually and each cross-checking against batch × seq × d_model:

| shape | 2 | 3 | 4 | 7 | 12 | 1 | 9 | 10 | 11 | 5 | 8 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| elements/trial | 16,384 | 65,536 | 262,144 | 262,144 | 262,144 | 1,048,576 | 1,048,576 | 1,048,576 | 1,048,576 | 2,097,152 | 8,388,608 | 8,388,608 |

**23,937,024 elements per pass × 7 trials = 167,559,168 element comparisons, zero
failures** — and the same again on the module board. In all three drafts now. "0 of 167
million" is a different class of claim from "PASS", and it was sitting in the packets the
whole time.

**Two regressions from one wholesale paragraph replacement.** The pattern is confirmed, not
suspected: when I replace a block I reliably preserve what I am thinking about and silently
drop everything else it contained. **Diff what you delete, not just what you add.**

---

## 31 Aug ~08:45 SGT — Diffed the rest of my rewrites. No more deletions, but CORRECTION DRIFT.

Diffed `745d07e` (hostile read) against both the tech report and the video script. **No lost
claims** — every removed line is restated, and the video-script changes are pure addition.
So the wholesale-deletion problem appears confined to `2512772`, and that hunt is closed.

But diffing surfaced a different and more embarrassing failure: **corrections I made in one
document never propagated to the other two.**

| claim | corrected in | still stale in |
|---|---|---|
| "committed to git before measurement" (pre-LOCK rule the auditor rejected) | report §6 | report §5.3, README tripwires, video Beats |
| "correctness on five seeds" (it is seven trials) | report §6, README | video Beats |
| "every new champion auto-fires a blind audit" (present tense, recorder broken) | report §5.4, README | video Beats |
| "±25% noise" and "median of repeated sweeps" | report §9 | README limitations |

All now fixed in every location.

**Why this is worse than the original error.** Before I corrected anything, the three
documents were uniformly wrong. After correcting one copy each time, they **contradicted
each other** — and a reader who finds two versions of the same claim stops trusting both,
including the version that is right. I spent seven passes hunting defects and spent several
of them manufacturing a new class of defect.

The structural cause is visible in the table: the drifting claims all live in a *features
list*, a *limitations list*, and a *demo script* — three places written for three audiences
and edited on three different passes. Nothing links them, so a fix to one never surfaces the
others. **LESSONS 48**: when you correct a claim, search for every other instance across all
documents before committing; diffing your own commits finds drift cheaply, because a removed
line whose assertion still exists elsewhere is a drift rather than a deletion.

