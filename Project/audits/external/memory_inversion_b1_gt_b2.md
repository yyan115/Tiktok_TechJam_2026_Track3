# Shape 14 memory inversion: why B1 reported more than B2

**Method:** source and packet reading only; no GPU run was performed.  GiB means
bytes divided by `2**30`.

## Bottom line

This is not a like-for-like batch-scaling result.  The B1 packet bound the
executed candidate to SHA-256 `ee85ee29...d6943f8`, while the B2 packet bound it
to `68d62045...b72a50`; the evaluator SHA is the same in both packets.  The
reported allocated peaks are 4.893825531 GiB and 4.738025665 GiB respectively,
but the allocator-reserved peaks go in the ordinary direction: 5.306640625 GiB
for B1 and 5.431640625 GiB for B2.
(`Project/results_side/shape14_20260829-042452_B1_S100000.json:2-5,18-30`;
`Project/results_side/shape14_20260829-042810_B2_S100000.json:2-5,18-30`)

Hashing the historical files identifies `ee85ee29...` as the original k014 at
commit `8db6826` and `68d62045...` as the memory-disciplined v2 at commit
`851845c` (also the current standalone k014).  This is reproducible with
`git show <commit>:Project/kernels/k014_shape14.py | sha256sum`.  The old
evaluator read, hashed, compiled, and executed the `--impl` bytes directly, then
wrote that candidate hash into the packet; the separate, identical
`submission_sha256` field did **not** identify the code executed by this tool.
(`a39760e:Project/tools/shape14_eval.py:207-217,229-240,283-297`;
`Project/results_side/shape14_20260829-042452_B1_S100000.json:2-5,49-50`;
`Project/results_side/shape14_20260829-042810_B2_S100000.json:2-5,49-50`)

The most likely mechanism is therefore a **candidate-version change**, not a
property by which B1 intrinsically consumes more memory than B2.  In v1, the
full-length fp16 LayerNorm output and packed QKV staging remain referenced after
three contiguous Q/K/V tensors are created, and the entire projection/residual/
FFN tail is full-sequence.  In v2, `qkv` and `h16` are explicitly released as
soon as Q/K/V exist; Q/K/V are then released after attention; and the tail is
performed in 16,384-token slices.
(`8db6826:Project/kernels/k014_shape14.py:220-265`;
`851845c:Project/kernels/k014_shape14.py:220-277`)

That source change is large enough to dominate the extra B2 envelope.  For one
B1 `[100000,1024]` tensor, one fp16 tensor is 0.190734863 GiB.  The early v2
release alone removes four such full tensors (`h16` plus a three-wide `qkv`), or
0.762939453 GiB, before attention continues; v2 additionally releases the three
contiguous Q/K/V tensors and replaces full-length tail temporaries with chunks.
The B2 wrapper does retain a full B2 input and preallocate a full B2 output, but
both versions execute the expensive inner route one B1 micro-batch at a time.
(`851845c:Project/kernels/k014_shape14.py:226-239,259-277,306-330`;
`8db6826:Project/kernels/k014_shape14.py:281-305`;
`Project/results_side/shape14_20260829-042810_B2_S100000.json:39-47`)

The packet timestamp does not override that provenance.  The historical tool
loaded and hashed the candidate before correctness and timing, but created its
timestamp only after both phases.  A long B1 job could therefore finish after a
new source commit while still truthfully recording the earlier bytes it had
already loaded.  The packet's candidate SHA, not the wall-clock order of its
filename, is the binding evidence.
(`a39760e:Project/tools/shape14_eval.py:229-240,243-285,290-292,311-314`)

## Why the old values are not candidate-only peaks

The old evaluator called `reset_peak_memory_stats()` only once, before all
correctness, oracle, warmup, and timing work.  During correctness it held the
full requested-batch input and candidate output while constructing and running
a streamed fp32 oracle one batch slice at a time.  It then moved into timing
without another peak reset and finally read one maximum for the whole process
window.
(`a39760e:Project/tools/shape14_eval.py:240-264,266-279,303-307`)

Consequently, 4.8938 and 4.7380 GiB are maxima of mixed candidate/oracle/staging
phases.  They are valid records of those historical evaluator processes, but
they cannot isolate the candidate forward or support a monotonic B1-versus-B2
claim.  The fact that reserved memory is higher for B2 while allocated memory is
higher for B1 is also consistent with allocator history being a confound rather
than the main source-level explanation.
(`Project/results_side/shape14_20260829-042452_B1_S100000.json:28-30`;
`Project/results_side/shape14_20260829-042810_B2_S100000.json:28-30`)

## Ranked alternatives and falsification tests

The version change above is proven by the hashes; attributing the size of the
inversion specifically to its early releases and chunked tail remains a causal
inference from source.  A decisive confirmation would run v1 and v2 at the same
batch, in fresh processes, with a peak reset after warmup and a candidate-only
measurement window.  If v2 did not materially reduce the peak under that
controlled comparison, the proposed mechanism would be killed even though the
cross-version nature of the old packets would remain proven.
(`8db6826:Project/kernels/k014_shape14.py:226-262`;
`851845c:Project/kernels/k014_shape14.py:226-277`)

1. **Historical evaluator staging and phase mixing — strong contributor.**
   The single reset and retained full-batch input/output let an oracle or
   transition peak win instead of the candidate peak.  Confirm it by recording
   separate post-warmup candidate, oracle, and comparison peaks for the same
   candidate SHA; kill it as an explanation of the difference if isolated
   phases on the two historical SHAs reproduce the same gap without any mixed
   window. (`a39760e:Project/tools/shape14_eval.py:240-279,303-307`)

2. **Caching-allocator fragmentation — plausible secondary noise.**  The B2
   process reserved 0.125 GiB more even though its active-allocation peak was
   0.156 GiB lower.  Confirm a meaningful allocator contribution with fresh-
   process repetitions plus allocated/reserved traces or memory snapshots; kill
   it as the main explanation if candidate-only allocated peaks are stable while
   reserved peaks vary.  The packet pair by itself cannot separate this effect.
   (`Project/results_side/shape14_20260829-042452_B1_S100000.json:28-30`;
   `Project/results_side/shape14_20260829-042810_B2_S100000.json:28-30`)

3. **A genuinely different inner route at B1 versus B2 — unlikely.**  Both
   historical versions select the long-sequence route from sequence length and
   iterate the outer batch with `MICRO_BATCH = 1`; the expensive block therefore
   sees B1 in either case.  Confirm with a profiler/kernel trace or branch
   counters on the exact historical SHAs; kill it with identical per-slice
   long-route traces for those same historical bytes.
   (`8db6826:Project/kernels/k014_shape14.py:281-305`;
   `851845c:Project/kernels/k014_shape14.py:306-330`)

4. **Peak-reset timing, lazy fp16 caches, or first-use library work — low but
   testable.**  The old reset occurred before the first candidate call, so lazy
   half-parameter creation and first-use activity were inside the same maximum
   as correctness and timing.  Confirm by comparing the first post-build call
   with peaks reset after several warmups; kill it if later fresh-process
   repeats settle at the same candidate-only peak.  The current dispatcher does
   create its fp16 parameter cache lazily.
   (`a39760e:Project/tools/shape14_eval.py:234-246`;
   `Project/submission/dispatcher_region.py:382-399`)

5. **Seed or machine differences — poor fit to the evidence.**  Both packets
   record seed 1234 and the same GPU, driver, PyTorch, CUDA, Python, hostname,
   and evaluator SHA.  Only an unrecorded nondeterministic library choice
   remains possible.  Confirm or kill it with repeated fresh processes pinned
   to one candidate SHA; stable peaks would kill this explanation.
   (`Project/results_side/shape14_20260829-042452_B1_S100000.json:18-25,36-47`;
   `Project/results_side/shape14_20260829-042810_B2_S100000.json:18-25,36-47`)

## Prediction for the full streamed 32-slice protocol

This prediction is conditional on the generated submission retaining the
current long-sequence structure.  The current evaluator deletes the oracle and
empties the cache before warmup, warms the candidate, then resets peak stats at
the start of each timing repeat.  Each repeat creates, runs, and deletes 32 B1
slices serially, and records allocated and reserved peaks per repeat.  Thus the
peak is the maximum live footprint of one slice plus persistent state; it is not
32 times a slice and it is not a literal B32 call.
(`Project/tools/shape14_eval.py:374-413,468-490`)

The source-level budget for one current B1 call is:

- Let `F` be one fp16 `[1,100000,1024]` tensor: 204,800,000 bytes, or
  0.190734863 GiB.  The input and residual/output buffers are fp32, while QKV,
  attention context, and tail GEMM buffers are fp16.
  (`Project/tools/shape14_eval.py:240-244`;
  `Project/submission/dispatcher_region.py:382-399,456-495,530-539`)
- At the source-visible high point in layer two, live full-size storage is about
  `14.10176 F`: the evaluator input (`2F`), the long-route preallocated output
  (`2F`), the layer-one residual (`2F`), new `h16 + qkv + q + k + v` (`7F`),
  the prior fp16 context (`1F`), and the prior final tail slice's named
  temporaries (`6 * 1696/100000 F = 0.10176F`).  The 1,696-token final slice is
  the remainder after 16,384-token chunks.  This count follows the two-layer
  loop and its `continue`; it is a source-liveness estimate, not a promise about
  library workspace.
  (`Project/submission/dispatcher_region.py:456-497,530-539`;
  `Project/results_side/shape14_20260829-042810_B2_S100000.json:39-47`)
- `14.10176 F` is 2.689697266 GiB.  The two-layer fp32 model parameters plus the
  lazily retained fp16 linear caches add about 0.070419312 GiB, giving
  **2.760116577 GiB before GEMM/Triton workspace and allocator granularity**.
  The parameter count follows four attention linears, two FFN linears and two
  LayerNorms per layer, two layers, and the final LayerNorm; the fp16 cache holds
  packed QKV, output projection, and both FFN linears per layer.
  (`torch_transformer_benchmark.py:62-75,125-160`;
  `Project/submission/dispatcher_region.py:382-399`)

**Concrete prediction:** in a fresh process, the full 32-slice run should report
candidate-timing `peak_allocated` values centered near **2.85 GiB**, normally
about **2.8-3.0 GiB per repeat**, with **3.2 GiB** as a conservative source-based
upper bound.  The three repeat values should be close to one another rather than
growing with slice count.  Reserved memory may be higher and is not the primary
pass/fail signal because the packet deliberately records it separately.
(`Project/tools/shape14_eval.py:386-413,480-486,549-552`)

Results above 3.2 GiB would falsify this numerical liveness forecast and require
checking the recorded submission SHA, unexpected persistent tensors, and
library workspaces.  A value near the old 4.7-4.9 GiB band should trigger checks
for a historical source or a mixed measurement window, but memory alone cannot
identify either cause; the hashes and phase-separated measurements must do
that.  A rising allocated peak across repeats would instead point to persistent
first-use state or a leak.  None of those outcomes would make the two old
packets a controlled B1-versus-B2 comparison.
(`Project/tools/shape14_eval.py:307-310,386-413,438-486`;
`a39760e:Project/tools/shape14_eval.py:240-279,290-307`)
