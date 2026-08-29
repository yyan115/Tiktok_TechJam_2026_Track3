# Baidu harness-engineering paper (FlashInfer contest, B200) — researched 29 Aug PM

Source: https://arxiv.org/html/2607.17979v1 · Regime: B200/CUDA13.2/PT2.12,
attention/MoE operators, contest conditions. Confidence: primary, read via summary extract.

## Central claim (= the owner's philosophy, validated externally)
"Reliable LLM-driven kernel generation still depends on harness
engineering" — their Agent-Assisted (human-designed harness, conservative
promotion gates, profile-backed decisions) beat Full-Agent autonomous
search by 1.35-13.25x; two Full-Agent artifacts fell BELOW baseline.

## The loop (Algorithm 1) — matches ours where we're strong
state-from-memory -> candidate generation -> REPRESENTATIVE GATE (cheap
screen vs same-round baseline) -> PROFILING of promising candidates only
(NCU + torch profiler) -> full sweep -> promote only if the full
distribution improves with no correctness regressions -> archive
accepted AND rejected probes + profiler evidence.

## THE ADOPTABLE PIECE WE LACK: compressed profile decision records
They compress NCU/torch-profiler output into a small decision record per
candidate — not raw dumps (matches CudaForge's selected-metrics finding):
  1. dominant kernel + % of total time
  2. occupancy and waves-per-SM
  3. memory throughput %, compute throughput %
  4. register / shared-memory pressure
  5. launch count + tail-wave effects
Worked examples: MoE — two GEMMs = 81.6% of time -> focus there, not
microkernels; sparse attention — low-wave launches -> scheduling is the
frontier, not bandwidth.

## Other lessons mapped to us
- Trajectory memory of REJECTED probes prevents re-exploration (our
  lineage.jsonl does this — keep feeding it).
- Low-trial correctness (n=3) missed rare boundary failures; final
  validation needs high-trial replay (we run 5 trials + 2 tripwires + 40
  anti-cache iters — adequate; shape-14 final should use extra seeds).
- Shape-aware dispatch routes only when profiler + latency evidence shows
  distinct bottlenecks (we did this by instinct in the submission
  dispatcher; now it's evidence-policy).
- Language per operator constraint (Triton/CUDA/CuTe as fits) — matches
  our converged language-agnostic stance.
