# Devpost written description — v2 (1 Sep, final board)

> Paste target: the Devpost project description field. Every required
> element from track statement §3.5 is covered under its own heading.
> Numbers below are the single-artifact board in `Project/BOARD.md`, measured
> on `c2028c4823ff756b062940e4eff35d5a6a341e9538b755256509cf3432e7794b`.

---

## An AI agent that writes GPU kernels — and a referee it is not allowed to touch

**Geometric-mean 11.87× speedup on the organizers' own untouched benchmark
script, on a consumer RTX 3060 Ti. All fourteen test shapes are measured on one
build, and every one passes the precision test: 94 trials, 17.4 billion element
comparisons, zero violations. The two shapes that cannot run on 8 GB in their
official form run on the same card and are verified against exact references. The
largest of them, 100,000 tokens at batch 32, reaches 88.7% of the card's physical
maximum.**

The 11.87× is over the twelve shapes that have a baseline to compare against.
Shapes 6 and 14 have none, because the official baseline runs out of memory on
one and would need roughly 160 TB of attention matrix on the other.

**On utilisation, we report a range rather than a number, on purpose.** The
organiser has said the technical score is a weighted sum of per-shape MFU and
that the weights are not yet decided. The same fourteen measurements give
**42.7%** if every shape counts equally, **88.6%** if shapes are weighted by the
arithmetic they perform, and **87.1%** if weighted by memory traffic. The spread
is real and has one cause: shape 14 is 99.87% of all the arithmetic in the
benchmark, and we run it at 88.7% of the card's physical maximum. We publish all
of them because the rule is the organiser's to set, and we optimise against the
least favourable one.

Against **PyTorch's own fused flash attention** (`scaled_dot_product_attention`),
rather than TikTok's naive reference, the margin is about **7.42×**. That figure
carries two caveats we state rather than bury: it was measured on a different
build on 28 August, and PyTorch runs the model at fp32 there while our kernels
compute at fp16, so part of it is precision rather than kernel engineering. Both
satisfy the competition's precision requirement. The 11.87× against TikTok's own
baseline has neither caveat, and is the number we stand behind.

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
xFormers. The kernels are authored, which is the point of the exercise. For
scale: `torch.compile(mode="max-autotune")` was measured at 7.0× on shape 3
and 3.1× on shape 13, where our shipping build reaches 19.8× and 31.5×. That
`torch.compile` measurement was taken on 29 August on an earlier build, so it
spans two artifacts and is an estimate rather than a paired measurement.

**The referee.** Shapes are pinned in a config the agent cannot benchmark
around (the official script's *defaults* match none of the 14 test shapes).
The runner is hash-pinned and protected by tool deny-rules plus a shell
guard hook; it re-hashes the official benchmark on every run. Each
documented cheat has a named tripwire: wall-clock cross-check against CUDA
events, perturbed fresh-memory re-runs, shape assertions, and candidate
bytes committed to git *before* the first measurement so that audited bytes
and measured bytes are provably identical.

**The adversary.** A new champion fires a blind audit by a different model
family, which sees only a neutral machine-generated evidence packet, with no
commentary from the optimizer, and returns a typed verdict. Hard verdicts pause
the machine and only the human can clear them. Sixty-plus verdicts are in the
ledger, including the ones that made us change code: an auditor caught a latent
masking bug on a fallback path the benchmark never exercises, by reading rather
than testing. The brake has fired for real, on sixteen findings at once, and cost
sixteen separate human signatures to lift.

Stated plainly because it is the honest limit: **none of the final board's rows
has an audit verdict.** Those rows are screening-lane measurements, each bound to
a permit and an artifact hash, with no verdict attached.

The cause is narrow and we found it by reading the failure rather than assuming
it. Our own verdict schema uses a JSON Schema construct (`allOf`) that OpenAI's
structured-output mode rejects, so the auditor is refused with an HTTP 400 before
it ever sees the evidence. It reproduced three times in one minute on 1
September, each failure recorded in the hash-chained audit journal. The
machinery around it worked correctly: enqueued, launched, retried, recorded,
escalated to the human. The schema is inside the signed lock, so only the human
can fix it, which is the arrangement working rather than failing.

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

That design was itself adversarially reviewed across thirteen rounds by a
different model family. The first version was discarded entirely, and about
fifty real holes were found and fixed. We do not claim the review ended in an
approval, because we went looking for that verdict in our own archive and it is
not there.

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

Shape 14 is timed as 32 serial batch-1 calls, not one literal batch-32 call,
and its correctness reference is one we wrote and validated against the official
baseline at 1,024, 2,048 and 4,096 tokens, where the official version can still
run. Shapes 6 and 14 use CPU RNG in their evaluators, so their inputs are not
bit-identical to a default judge run. Both are labelled that way everywhere.

Small shapes are noisy on a consumer card. The same code, byte for byte, has
been observed 13.2% apart between two runs minutes apart, so no single
small-shape row should be read closely and the geometric mean over twelve shapes
is the figure to quote.

Our comparison against PyTorch's own `scaled_dot_product_attention` runs it at
fp32 while our kernels compute in fp16 with fp32 accumulation. Both clear the
competition's precision requirement, so both are legal, but part of that margin
is precision rather than kernel engineering, and we label it rather than quote it
as a headline.

The final board's rows carry no audit verdict, for the reason given above.

On a single-user machine our anti-tamper measures are forge-obvious, not
forge-proof. We state that ceiling rather than implying we exceeded it.
