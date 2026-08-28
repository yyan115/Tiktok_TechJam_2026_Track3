# STATE — read this first in every session

Updated: 2026-08-28 19:42 (grind day 1 CLOSED — handover ready; hooks confirmed live)

## FIRST ACTIONS FOR A FRESH SESSION (in order)
1. Locks: the Bash guard is already proven live (it blocked a benign commit whose message contained the word 'clean' after 'git' — false positive, working seatbelt). Still run the full lock test: attempt an Edit on torch_transformer_benchmark.py AND Project/harness/runner.py — both MUST be denied; report to the user; STOP if not.
2. The auto-audit hook (PostToolUse → tools/champion_watch.py) should also be live — verify by checking it fires after your first shell command.
3. Resume the grind on branch `grind-day1` (user's standing order: continuous optimization until stopped). NEXT TASK: full re-sweeps of the SELF-CONTAINED k004 and k005 across shapes 1-5,7-13 (both verified post-inlining: k004 shape3=7.50x, k005 shape8=1.63x) — refreshes every champion with single-file provenance, answering the auditors' one finding.
4. Then the lever queue below.
5. Guard etiquette: never put the words 'clean', 'reset', 'restore' after 'git' inside one command segment (commit messages included) — the seatbelt pattern-matches them.

## SCOREBOARD (clean-provenance champions, all authored+self-contained, referee-verified, FP32 primary, RTX 3060 Ti)
k005sc (fp16 graphed stack) leads nearly everywhere: 1: 2.92x · 2: **10.66x** · 3: 9.00x · 4: 3.94x · 5: 2.73x · 7: 5.21x · 8: 1.65x · 9: 1.40x* · 10: 2.11x* · 11: 6.42x · 12: 4.20x · 13: **10.83x** — clean-set geomean ≈ 4.1x.
(*) Shapes 9/10: EARLIER flagged-provenance k004 entries measured higher (3.14x/4.09x) than the clean re-runs — thermal/CPU-contention variance suspected (20+ codex audit processes ran during re-sweep). NEXT-SESSION TASK: re-run k004sc+k005sc on shapes 9+10 under an idle box; ship-eligible numbers are the clean-provenance set only.
**SHAPE-14 CORE PROVEN on this card**: authored kernel at seq=100,000 causal vs chunked fp32 oracle — 0 tolerance violations, max err 5.3e-4, 337 MiB (scratchpad/shape14_core_smoke.py, replicate any time). Perf at 100k untuned (autotune configs target short seqs) — tuning is rental-day work; correctness is banked.

## AUDIT LEDGER interpretation
15 PASS (SDPA under corrected policy + k003/k005). 10 RULE_VIOLATION on ORIGINAL k004 entries = PROVENANCE ONLY (speeds explicitly certified genuine; the k003 import was not hash-bound) — superseded by the self-contained re-sweeps. 1 historical (morning policy, superseded), 1 NEEDS_CONTEXT (moot after re-sweep).

## LEVER QUEUE (user's standing order = keep going)
1. Extend the Triton fast path to head_dim 128 (shape 9 currently graphed-eager).
2. Shape-14 core proof LOCALLY: smoke the kernel at seq=100k, B=1,H=1 fp16 slices vs a chunked fp32 reference (~13 MB per tensor — fits) — de-risks rental day.
3. k005 tuning: seq-1024 block configs; fp16 attention on shapes where k004 still leads.
4. Amendment bundle re-freeze when user present (TIMEBOXED 1-2 rounds): shape-14 oracle + `official` subcommand + MFU.
5. Day 2: rental (48-80 GB) for shapes 6+14; re-tune there; MFU numbers.

## Standing rules (unchanged)
Never touch frozen/protected files (locks enforce). All benchmarks via the pinned runner + shape id. ONE runner process; NO other GPU work during sweeps; sequential calls. Champions auto-audit. Reflection after each block. Plain language; the user's stop overrides everything.

## Packaging (day 3)
Merge to main before submitting · judge-facing READMEs (user applies) · TEMP files out of judge path · T3 video opens with TAMPER demo · T2 packages first · final ~8h protected · Devpost registration AND submission close 1 Sep 12:00 noon.
