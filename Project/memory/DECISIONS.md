# DECISIONS — plain-language diary of what we discussed and agreed

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
- Guard holes (git reset -q --hard, git -C variants, checkout HEAD --, rm -R/--recursive/-rf *) — flag-tolerant patterns added, regression-tested; RUNBOOK enforcement wording corrected (deny rules cover Claude's file tools, not subprocess writes).
- Calibration key lacked python/triton; champions could outlive a raised threshold — key extended, champion eligibility now requires clearing the LATEST calibration's threshold, and the displayed promoted column uses the same filter.
- Stale injected STATE.md + a false "no new problems" line in TEMP log — both corrected.
Also adopted its recorder caveat: record-verdict now requires the source log to exist and stores its sha256.

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
