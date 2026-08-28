# Proposal: fixing the optimization loop (harness v2) + revised win strategy

Draft 1 — 29 Aug ~03:45. Research-backed; no implementation until reviewed
(external critique + owner approval). Sources cited inline.

## The four diagnosed failures (from tonight's post-mortem)

F1. The harness audits honesty, not intelligence — no mechanism ever asks
    "is this the right thing to build?"
F2. The agent reward-hacked cadence: build→measure→promote produces a
    reward tick; research produces none, so research starved.
F3. The assembled research corpus went unused; candidates (int8 W8A8, a
    traffic-doubling restructure) contradicted known literature.
F4. Norms without enforcement decay (research order, reflection); only
    mechanically-enforced rules survived the grind. The stall adviser never
    fired because its trigger was self-reported.

## Fixes (each grounded in current literature)

### Fix F1 — a Strategy Critic as a first-class role
- cuPilot (arXiv 2512.16465) shows strategy-level coordination beats
  code-level iteration: strategy is an explicit intermediate representation,
  evolved and critiqued separately from code; roofline-guided prompting
  selects techniques. Result: 3.09x avg over PyTorch across 100 kernels.
- Reviewer-bias literature (Mind the Blind Spots, arXiv 2502.17086) documents
  that LLM reviewers over-attend technical validity and under-attend
  strategy/direction — precisely our observed failure mode.
- CHANGE: two reviewer tracks with different prompts and cadences:
  (a) integrity audits per champion (unchanged, effort high);
  (b) STRATEGY reviews at objective checkpoints (every 5 promoted champions
  OR any stall trigger): reviewer receives the roofline table, the
  technique portfolio with outcomes, and the current board, and is asked
  "what known technique is being ignored; where is effort misallocated
  relative to scoring?" — at maximum reasoning effort, cross-model-family
  (same-family panels are structured self-critique, not verification:
  agent-review-panel's documented limitation).

### Fix F2 — mechanical research gate + evolutionary memory
- AlphaEvolve/OpenEvolve-class systems pair generate-evaluate-select with
  EVOLUTIONARY MEMORY so failed directions are not retried blind.
- CHANGE 1: a hook (same class as the Bash guard) refuses `runner run
  --impl` for any candidate file lacking a research-memo header: sources
  consulted, predicted gain WITH arithmetic (roofline/traffic estimate),
  and kill-criteria. The memo is cheap (10-30 min); the hook makes it
  unskippable — norms→mechanisms (F4).
- CHANGE 2: TRIED.md — an append-only direction ledger (technique,
  prediction, outcome, lesson-link). The strategy critic receives it, so
  repeats and near-repeats get flagged.

### Fix F3 — hardware-feedback and roofline discipline in the inner loop
- CudaForge (OpenReview f4GtuI2blh): Coder+Judge with Nsight Compute
  counters in the loop → 97.6% correctness, 1.68x, generalizes across GPUs.
  TritonForge (arXiv 2512.09196) does the same profiling-guided loop
  Triton-native.
- CHANGE: every candidate's evaluation includes a per-kernel counter
  snapshot (achieved bandwidth, occupancy, issue stalls) vs a per-shape
  ROOFLINE TABLE (bytes moved, FLOPs, theoretical floors) maintained in the
  repo. Predictions in research memos must reference it. (This arithmetic,
  done first, would have killed the k011 restructure before it was built.)

### Fix F3.5 — persistent RESEARCH BASE with a mandatory check (owner-mandated 29 Aug; literature-validated)
- cuPilot's ablations rank their historical-database + RAG strategy seeding
  as the single largest component: -54.1% latency after ONE epoch — bigger
  than roofline guidance itself. A persistent, indexed research memory that
  every new task MUST consult is not hygiene, it is the top-performing
  mechanism in the closest prior art.
- BUILT (initial version): Project/research/ — INDEX.md + per-topic notes
  (agent-loop design, megakernels, reviewer bias, epilogue fusion,
  quantization tolerance, competition scoring), backfilled from Day-1
  research. Notes carry structured findings + stable URLs + implications;
  PDFs are not vendored (re-fetchable), conclusions are.
- CHANGE (needs hooks, hence in this proposal): (a) research memos must cite
  research-base notes or add new ones — checked by the same gate hook;
  (b) SessionStart injects INDEX.md alongside STATE.md so no session starts
  blind; (c) every research pass STARTS by reading INDEX.md — re-researching
  something already noted is itself a loop failure.

### Fix F4 — norms become mechanisms
- Research gate: hook-enforced (above). Reflection: each block ends by
  appending to REFLECTIONS.md (hook checks recency before new candidates).
  Stall trigger: TWO consecutive non-promoted or negative candidates
  mechanically forces the strategy-review checkpoint before the next build.

## Revised win strategy (language-agnostic, scoring-aligned)

Scoring recorded from the organizers: weighted sum of per-shape MFUs,
"bandwidth considered", failed precision = zero. "Bandwidth considered"
reads as roofline-relative credit — launch/bandwidth-bound shapes are
scored against what the memory system permits, not raw FLOP peak. If so,
the per-shape roofline table IS the score model; effort should follow its
gaps. (If weights are per-shape-equal, the same conclusion holds; if
absolute-MFU-weighted, the big-d shapes 6/8/14 dominate and rental rises
further in priority. UNKNOWN — worth asking organizers or hedging.)

Technique portfolio by shape class (each item = memo before build):
1. B=1 (shape 2): SINGLE-PROGRAM whole-model kernel. The documented Triton
   limitation is grid-level sync (Mirage/MPK, MegaKernel lit); a one-CTA
   kernel needs none — 128 tokens x d=128 activations fit one program's
   SRAM; 4 layers of weights (~200 KB fp16) stream from L2. Expressible in
   Triton (proven toolchain), no compiler blocker. Predicted gain: bounded
   by remaining inter-kernel gaps (~10-20 us of ~50 us forward) → ~1.2-1.5x
   on that shape. AutoMegaKernel (arXiv 2606.09682) calibrates megakernel
   claims honestly: vs CUDA-graphed baselines on consumer GPUs the win was
   1.19-1.23x — expect that class, not 6x.
2. Small-batch multi-CTA shapes (12, 4): persistent kernel with
   counter-based producer/consumer sync (AutoMegaKernel's model) — CUDA
   C++ only (Triton cannot grid-sync). LOCALLY BLOCKED (nvcc 13 segfaults
   with gcc16; no g++-14). Route: rental-day toolchain or owner installs
   gcc14. Expected: same ~1.1-1.3x class on those shapes' remainders.
3. Big-d shapes (8 local; 6, 14 rental): GEMM/bandwidth-bound — the MFU
   frontier if scoring is absolute. Portfolio: fp16-acc tensor-core GEMM
   epilogues (2x fp16 rate on sm86; tolerance-risky — memo must include
   error model BEFORE build, unlike k008), deeper cuBLASLt epilogue use,
   k010-class fusion already banked.
4. Rental day doubles as the CUDA-C++ window (matched toolchains) for
   items 2 and any counter-guided retunes.

## Sequencing (proposal)

Day 2 morning: owner reviews this + amendment v1.1 → harness v2 hooks +
roofline table built → strategy-critic round 1 on the portfolio → then and
only then implementation resumes, memo-gated.

## Verification plan for the fixes themselves

The strategy critic's first assignment: attack THIS document. If it finds
the fixes toothless or the strategy misprioritized, iterate before any code.
