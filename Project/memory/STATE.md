# STATE — read this first in every session

Updated: 2026-08-28 19:34 (grind paused for session restart; guard hook confirmed LIVE mid-session)

## FIRST ACTIONS FOR A FRESH SESSION (in order)
1. Locks: the Bash guard is already proven live (it blocked a benign commit whose message contained the word 'clean' after 'git' — false positive, working seatbelt). Still run the full lock test: attempt an Edit on torch_transformer_benchmark.py AND Project/harness/runner.py — both MUST be denied; report to the user; STOP if not.
2. The auto-audit hook (PostToolUse → tools/champion_watch.py) should also be live — verify by checking it fires after your first shell command.
3. Resume the grind on branch `grind-day1` (user's standing order: continuous optimization until stopped). NEXT TASK: full re-sweeps of the SELF-CONTAINED k004 and k005 across shapes 1-5,7-13 (both verified post-inlining: k004 shape3=7.50x, k005 shape8=1.63x) — refreshes every champion with single-file provenance, answering the auditors' one finding.
4. Then the lever queue below.
5. Guard etiquette: never put the words 'clean', 'reset', 'restore' after 'git' inside one command segment (commit messages included) — the seatbelt pattern-matches them.

## SCOREBOARD (all authored, referee-verified, FP32 primary, RTX 3060 Ti)
Best per shape: 1: 2.14x · 2: 8.19x · 3: 7.52x · 4: 2.85x · 5: 2.75x (k005 fp16) · 7: 3.47x · 8: 1.63x (k005 fp16) · 9: 3.14x · 10: 4.09x · 11: 6.04x · 12: 2.69x · 13: 5.94x — geomean ≈ 3.7x and climbing. Shapes 6+14: rental-bound (48-80 GB, day 2).
Kernel lineage: k001 SDPA (eligible fallback) → k002 fused QKV → k003 authored Triton flash-style attention (IEEE dots) → k004 = k003 + whole-forward CUDA graphs → k005 = k004 + internal fp16 (fp32 boundary/norms/accum). k004/k005 are SELF-CONTAINED single files.

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
