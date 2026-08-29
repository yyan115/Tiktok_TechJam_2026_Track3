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
| [authority-design.md](authority-design.md) | Post-Track-2-violation research: Redwood "AI control" (untrusted-strong + trusted-weak + scarce human = our exact triangle), NIST/METR eval-tampering incident record, maker-checker + break-glass patterns; 7 design rules — no AI-owned text box may unlock anything |
