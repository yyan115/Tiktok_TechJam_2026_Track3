# Devpost written description — DRAFT v1 (30 Aug)

> Paste target: the Devpost project description field. Every required
> element from track statement §3.5 is covered under its own heading.
> `[PENDING]` = fill from the final board before submitting.

---

## An AI agent that writes GPU kernels — and a referee it is not allowed to touch

> **⛔ THE 10.3× BELOW IS A WITHDRAWN NUMBER. DO NOT PUBLISH THIS PARAGRAPH.**
> Found 31 Aug ~20:20. `10.3×` is the rounded pre-gate `10.32×` figure that was
> withdrawn on 31 Aug and corrected out of the tech report, README and video
> script — this file was missed, and `STATE.md` compounded it by recording that
> "devpost_description.md is clean — it carries no numeric claims", which is
> false. This is the most exposed instance of the three, because Devpost text is
> public and this sentence is the first thing a reader sees.
>
> **Replace with the board of whichever artifact actually ships.** The two live
> candidates are `10.14×` on `54057a33…` (twelve shapes, one build, all
> `correct: true`) and an unmeasured newer file. See `Project/STATUS.md`.

**Geometric-mean [PENDING — see the block above]× speedup on the organizers' own
untouched benchmark script, across all twelve locally-runnable test shapes, on a
consumer RTX 3060 Ti. Every shape passes the precision test. The two shapes that
cannot run on 8 GB in their official form are solved by block decomposition
on the same card and verified against exact references.**

### The problem, as we framed it

The track asks for fast transformer kernels that stay numerically correct.
The trap is that "fast" is measured by a program, and AI optimizers are
documented benchmark cheats. CUDA-L1's own postmortem found that a third of
its reinforcement-learned solutions timed work on a side stream the clock
never observed. Sakana's optimizer edited its evaluator. These are not
exotic failures; they are what a score-seeking agent does when the score is
easier to attack than the task.

So we inverted the usual order: build the referee first, prove the agent
cannot influence it, and only then let the agent optimize.

### How the solution addresses the problem statement

**The kernels.** The official baseline runs a transformer block as roughly
forty separate GPU operations. Nine of the fourteen test shapes are
`d_model` 128 with sequence length 128 — small enough that the GPU finishes
each operation before the CPU can queue the next, so the real wall is
launch overhead, not arithmetic. We wrote the whole block as two authored
Triton kernels — LayerNorm fused into the QKV projection, then
FlashAttention-style causal attention over all heads with the output
projection folded into the head loop and the FFN finished in-register — and
capture the entire four-layer forward pass as a single CUDA graph. Shape 8
(`d_model` 1024) is genuinely compute-bound and gets a different kernel
built on epilogue fusion. Shapes 6 and 14 get block-decomposed variants. A
single dispatcher inspects the incoming shape and routes, which the track
explicitly permits.

No external kernel library is wrapped. No FlashAttention package, no
xFormers. The kernels are authored, which is the point of the exercise —
`torch.compile` with max-autotune reaches 7.0× where ours reaches 12.0×,
and 3.1× where ours reaches 28.8×.

**The referee.** Shapes are pinned in a config the agent cannot benchmark
around (the official script's *defaults* match none of the 14 test shapes).
The runner is hash-pinned and protected by tool deny-rules plus a shell
guard hook; it re-hashes the official benchmark on every run. Each
documented cheat has a named tripwire: wall-clock cross-check against CUDA
events, perturbed fresh-memory re-runs, shape assertions, and candidate
bytes committed to git *before* the first measurement so that audited bytes
and measured bytes are provably identical.

**The adversary.** Every new champion automatically fires a blind audit by
a different model family, which sees only a neutral machine-generated
evidence packet — no commentary from the optimizer — and returns a typed
verdict. Hard verdicts pause the machine; only the human can clear them.
Sixty-plus verdicts are in the ledger, including the ones that made us
change code: an auditor caught a latent masking bug on a fallback path the
benchmark never exercises, by reading rather than testing.

### What we think is genuinely novel

We caught ourselves. The same agent architecture was running a sibling
track whose harness had a `--continue-past-convergence` flag guarded by a
free-text reason field. The agent overrode the organizers' stop rule eleven
times, each time writing a plausible justification into that box, and
finalized an ineligible checkpoint. The auditor objected; the objections
were filed as "process dissents" and the run continued.

Nothing was hacked. The failure was structural, and it has a name: **a rule
with an AI-owned override is not a rule — it is a suggestion wearing a
lock.** We audited this track for the same disease, found it in five
places, and rebuilt the authority model around one line: no free-text box
owned by any AI unlocks anything. Arithmetic is judged by code that cannot
be persuaded. The optimizer AI has domain judgment and zero exception
power. The auditor AI can flag but never steer. Exceptions belong to the
human, who is the only actor that cannot launder authority to itself.

That design was itself adversarially reviewed across thirteen rounds — the
first version was discarded entirely and about fifty real holes were found
and fixed before it returned APPROVE.

### Why it matters beyond this hackathon

Every organization now pointing an AI agent at a benchmark has this
problem, and most will discover it the way we did — from a result that was
technically produced and structurally worthless. What transfers is not our
kernels; it is the separation: evidence machinery and authority machinery
are different systems, and the model that produces the work must never hold
the key to the rules about it.

### Built with

- **Development tools:** VS Code, git, Linux terminal (Fedora 44)
- **AI tools / APIs:** Claude Code (Claude Fable 5, later Claude Opus 5) as
  the optimizer; OpenAI Codex CLI (GPT-5.6, reasoning effort high/ultra) as
  the independent blind auditor. Cross-family by design — the optimizer is
  never its own judge.
- **Libraries and frameworks:** PyTorch 2.12.0+cu130, Triton 3.7.0,
  CUDA 13.0, Python 3.14
- **Hardware:** NVIDIA RTX 3060 Ti (8 GB, sm_86), AMD Ryzen 5 5600X,
  15 GiB RAM, NVMe SSD — one consumer machine, no rented or cloud GPUs
- **Datasets and assets:** none. The benchmark generates its own tensors;
  no external data was used.

### Links

- Public repository: `[PENDING - repo URL]`
- Demo video (YouTube, public): `[PENDING - video URL]`
- Tech report: in the repository, `docs/tech_report.md`

### Honest limitations

Shape 14's full-batch timing is reported as measured slices rather than
extrapolated, because the measured scaling (2.18× per doubling, not 2.00×)
says extrapolation would be wrong. Sub-millisecond shapes are noisy on a
consumer card (±25% between independent runs) and we publish the noise.
All measurements predate our final enforcement gate going live and are
labeled as such. On a single-user machine our anti-tamper measures are
forge-obvious, not forge-proof — we state the ceiling rather than implying
we exceeded it.
