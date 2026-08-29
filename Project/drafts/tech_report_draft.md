# Track 3 Tech Report — DRAFT (contents mandated by the organizer webinar)

Status: skeleton with known-final content filled; [FINAL] marks values that
update after the last measurement pass. The organizer scores FROM THIS
REPORT (judges do not rerun) — precision, honesty, and completeness here
ARE the technical score's carrier.

## 1. Runtime environment (mandated disclosure)

- Machine: personal desktop (owner's own machine, per track guidance).
- GPU: NVIDIA GeForce RTX 3060 Ti, 8 GB GDDR6, 448 GB/s memory bandwidth,
  38 SMs (Ampere GA104, sm_86). Peaks used in MFU math (spec-sheet):
  fp32 CUDA-core ~16.2 TF/s; fp16 tensor-core with fp32 accumulate
  ~32.5 TF/s. All conventions reported side by side (see §5).
- Driver 610.57.04 · CUDA 13.0 · PyTorch 2.12.0+cu130 · Triton 3.7.0 ·
  Python 3.14 · Linux (Fedora 44). Host RAM 15 GB.
- Known hardware limitations (disclosed for weight consideration, per
  webinar): 8 GB VRAM (shape-14 full batch cannot reside on-device — solved
  by block decomposition, §4); consumer clocks vary with thermals (all
  ship numbers measured on a quiet box, methodology §6).

## 2. AI tools used (mandated disclosure)

- Claude Code (terminal agent), model Claude Fable 5 — the optimizer:
  authored every kernel, the measurement harness, and the process loop.
- OpenAI Codex CLI, model GPT-5.6-sol — the independent adversary:
  blind audits of every champion (reasoning effort high) and strategy
  reviews (effort ultra). Cross-model-family by design: the optimizer is
  never its own judge.
- No external kernel libraries wrapped (flash-attn etc. not used); torch
  built-ins only as correctness fallbacks on paths the fast kernels
  don't claim.

## 3. Skills / method used to guide the agents (mandated deliverable)

The submission includes the actual process artifacts, not a narrative
reconstruction:
- Frozen referee harness (hash-pinned runner; multi-seed correctness,
  anti-cache and wall-clock tripwires, calibrated per-shape noise floors,
  append-only journal) — every number regenerates from it.
- Auto-fired blind audits per champion + verdict ledger (60+ verdicts,
  including RULE_VIOLATIONs we fixed and negative results we kept).
- Research-first experiment loop (Project/loop/): preregistered
  quantitative predictions per direction, recorded misses, kill criteria,
  forced external review on failure signals; research base with
  source-of-truth notes (Project/research/).
- Interaction-history samples: strategy-review prompts + verdicts
  (Project/audits/strategy/), experiment cards with
  expected-vs-observed records.

## 4. Results ([FINAL] board inserted at packaging)

Per-shape table from the sensitivity board (median latency, useful TF/s,
MFU under multiple conventions, speedup where a baseline exists), including:
- Shapes 1-13 (minus 6): frozen-runner journal numbers, quiet box.
- Shape 6 (baseline OOMs on 8 GB): candidate-only evidence vs the
  batch-chunked official computation (exact same math) — [FINAL].
- Shape 14 (baseline infeasible on any hardware): block-decomposed
  execution on our own device per the track guidance — correctness vs a
  streamed fp32 oracle that was itself validated against the untouched
  official dense implementation at feasible lengths (worst deviation
  1.4e-6); full-scale evidence packet with shas, seeds, raw samples,
  peak memory — [FINAL]. Organizer-provided input/output pairs will be
  checked the moment they are released.
- Negative results kept and documented (int8 W8A8 fails the tolerance;
  two structural experiments closed by preregistered criteria) — the
  referee does not grade on effort, and neither did we.

## 5. MFU conventions (weights/denominator not fixed by organizers — all reported)

For every shape: useful model FLOPs (formula stated, causal-halved),
median latency, TF/s, and MFU against fp32 peak AND fp16-tensor peak,
plus achieved-bandwidth context. Whichever convention the judges apply,
the number is on the table.

## 6. Measurement methodology (why these numbers are trustworthy)

Warmup excluded (20+ calls; compilation and first run excluded per
webinar), CUDA-event timing with synchronized wall-clock cross-checks,
fresh-value anti-cache reruns, input-mutation tripwires, per-shape noise
floors, quiet-box discipline (no background load during ship measurements),
committed-bytes-before-measurement provenance, and cross-model blind
audits. The tamper demo (agent's edit attempts bouncing off the frozen
files) opens the video.

## 7. How to run (README summarizes; full steps in repo)

Single-file submission script (official benchmark with only the sanctioned
class replaced — byte-identity of everything else proven by
tools/build_submission.py); per-shape dial commands; side-evaluator
commands for shapes 6/14; one command regenerates the leaderboard from the
journal.
