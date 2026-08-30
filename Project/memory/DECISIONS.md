# DECISIONS — plain-language diary of what we discussed and agreed

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
