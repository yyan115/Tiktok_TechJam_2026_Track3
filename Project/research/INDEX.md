# RESEARCH BASE — check BEFORE researching or proposing anything

Rule (user-mandated 29 Aug): every research finding worth using gets a note
here; every new research pass STARTS by reading this index; every candidate
memo must cite the notes it relies on. Notes store structured findings +
stable URLs + key quotes (not PDFs — arxiv/URLs are re-fetchable; notes are
what we act on). One topic per file.

| note | one-line takeaway |
|---|---|
| [agent-loop-design.md](agent-loop-design.md) | Winning kernel-agents separate STRATEGY from code, keep tried-direction memory, and put hardware counters in the loop |
| [megakernels-persistent.md](megakernels-persistent.md) | vs graphed baselines megakernel upside is tens-of-%; shape-2 single-CTA play DEAD (SM-floor math); sequence-persistent CTAs are the cheap correct idea for 3/4/12 |
| [reviewer-bias.md](reviewer-bias.md) | LLM reviewers over-weight technical validity, under-weight strategy; same-family panels are self-critique; blind + cross-family + explicit strategy prompts required |
| [gemm-epilogue-fusion.md](gemm-epilogue-fusion.md) | Epilogue/elementwise fusion ~1.45x on bandwidth-bound GEMM paths (source of k010's +14%) |
| [quantization-tolerance.md](quantization-tolerance.md) | W8A8 int8 error (~1%+) is designed for tolerant workloads; predictably fails abs-2e-3 criteria (k008 confirmed empirically) |
| [competition-scoring.md](competition-scoring.md) | FULL transcript decoded: judges NEVER rerun (tech report+code+skills+history ARE the deliverable); SINGLE GPU type; own-machine preferred => rental contra-indicated; shape-14 block-decomposition expected; weights undecided (hedge) |
| [roofline-table.md](roofline-table.md) | Per-shape FLOPs/bytes/AI + achieved TF: d=128 shapes LATENCY-bound; 8 = chunked-fp16-acc GEMM lever; 14 = FA2-style attention lever; 6 = local candidate-only |
| [kernelagent.md](kernelagent.md) | Meta/PyTorch KernelAgent (organizer-shown): Profile-Diagnose-Prescribe-Orchestrate, top-K beam, reflexion records — the loop template |
| [harness-engineering-baidu.md](harness-engineering-baidu.md) | Baidu/FlashInfer: harness engineering beats agent autonomy 1.35-13x; the 6-field compressed profile decision record to adopt |
| [cuda-agent-tiktok.md](cuda-agent-tiktok.md) | Sponsor's CUDA Agent: protected-scripts anti-hacking design = our harness independently; SKILL.md structure for the skills deliverable |
| [portfolio-orchestration.md](portfolio-orchestration.md) | OpenAI CDC prompt read verbatim: family registry by IDEA not wording, blocked-until-new-mechanism (= our revival clause independently), checklist-armed adversarial reviewers, diverse-first portfolios; adopt freeze-time multi-reviewer final review |
| [small-head-dim-padding.md](small-head-dim-padding.md) | head_dim 8 pads to 16 for the tl.dot minimum, doubling attention FLOPs on shapes 7 and 11 — the shapes where our route already wins most; head-packing is arithmetically impossible, tl.sum-reduce is the one candidate, MEASURE THE CEILING FIRST |
| [head-count-scaling.md](head-count-scaling.md) | Shapes 1/9/10/11 are the same 7.52 GFLOP problem at 1/2/4/16 heads: the BASELINE degrades 4.3x with head count, so the per-shape speedup spread is mostly a baseline property and shape 9 is the baseline's strong point, not our weak one. Our own +63.7% penalty at 16 heads is confined to K2 (2.13x, everything else flat within 4%) — padding confirmed by a preregistered discriminator, loop-length refuted; K2 throughput falls 14.5 -> 8.5 TF/s |
| [gelu-erf-approximation.md](gelu-erf-approximation.md) | A wrong-on-purpose probe sized GELU at 5.1 us, 11% of the tail kernel that owns 35% of device time; A&S 7.1.26 erf recovers 63% of it (46.5 -> 43.3 us, +2.7% end to end on shape 1) with max abs error UNCHANGED, because 1.5e-7 is four orders below the fp32 pipeline's own error and can never be the binding term. Ablate before optimising; any libdevice transcendental in an inner loop is now a candidate |
| [authority-design.md](authority-design.md) | Post-Track-2-violation research: Redwood "AI control" (untrusted-strong + trusted-weak + scarce human = our exact triangle), NIST/METR eval-tampering incident record, maker-checker + break-glass patterns; 7 design rules — no AI-owned text box may unlock anything |
