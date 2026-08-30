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
