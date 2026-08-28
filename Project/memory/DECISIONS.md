# DECISIONS — plain-language diary of what we discussed and agreed

## 28 Aug 2026 16:48 — auto-audit per champion (user-directed, mechanically triggered)

User overruled the checkpoint-only audit cadence for champions, with a design requirement: the trigger must be MECHANICAL, not agent-fired ("it should auto fire"). Built: Project/tools/champion_watch.py (hook-invoked after every shell command; detects newly crowned champions on the runner-generated leaderboard) + Project/tools/audit_champion.py (detached: evidence packet via the frozen runner → codex read-only blind audit → verdict recorded via the frozen record-verdict). Non-blocking by construction; JUDGE_ERROR/TIMEOUT recorded, never block; RULE_VIOLATION lands loudly in the audit column and journal trail. The PostToolUse hook entry was added to .claude/settings.json during this same setup era (disclosed); it arms at the user's restart alongside the locks. No harness change — zero re-freeze needed. First real firing: the existing 1.61x champion's audit launched at build time. Blocking checkpoints (freeze, final ship-gate) unchanged.

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
