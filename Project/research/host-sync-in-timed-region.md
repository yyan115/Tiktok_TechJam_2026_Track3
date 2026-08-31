# A host sync inside a CUDA-event window is charged to you in full

**Status:** measured on this box, 31 Aug 2026. Applies to every shape that
reaches CUDA-graph replay.

## The instrument

The official script does not time with a wall clock. It times with CUDA events
recorded on the stream (`torch_transformer_benchmark.py:488-495`):

```
starts[index].record()
model(x, valid_mask)
ends[index].record()
```

and later reports `start.elapsed_time(end)`. Both events are timestamps taken by
the **device**, in stream order. The reported sample is therefore the device's
own view of the interval — including any period in which the device had nothing
to run because the host had not yet enqueued it.

That last clause is the whole finding. CUDA-event timing is often described as
"measuring GPU time, not CPU time", which invites the conclusion that host work
is free. It is not. Host work is free only while the host stays *ahead* of the
device. The moment the host blocks, the device drains, and every microsecond
between the drain and the next enqueue lands inside the measured window.

## What we were doing

`UserOptimizedTransformer.forward` guarded the fast path with

```python
if valid_token_mask is not None and not bool(valid_token_mask.all()):
    return BaselineTransformer.forward(self, x, valid_token_mask)
```

`bool(...)` on a device tensor is `.item()`, which is a full stream
synchronisation. Per iteration, in order:

1. `starts[i].record()` is enqueued,
2. the `all` reduction is enqueued,
3. `.item()` blocks the host until the device has drained **everything**,
4. the device is now idle while the host unwinds the sync, returns up through
   the PyTorch and Python layers, runs the remaining guards, and issues
   `cudaGraphLaunch`,
5. `ends[i].record()` is enqueued.

Step 4 is pure measured dead time, it recurs every iteration, and it does not
shrink as the shape shrinks — so its *relative* cost is worst exactly where the
device work is smallest.

## Measured, shape 2 (B=1, S=128, d=128 — 128 tokens)

Paired diagnostics, same box, same build except the change.

| | before (`profile-31765b3624bc287cac16c754`) | after (`profile-f54e739a8d1cceba220681a0`) |
|---|---|---|
| `aten::all` CPU total | 2.299 ms / 20 calls | absent |
| `cudaStreamSynchronize` | 889.783 us / 20 calls | absent |
| `aten::_local_scalar_dense` | 1.075 ms / 20 calls | absent |
| `Memcpy DtoH` | 20.479 us / 20 calls | absent |
| `reduce_kernel` | 50.785 us / 20 calls | absent |
| host work per iteration | ~91 us | ~12.6 us |
| device work per iteration | 69.0 us | 64.7 us |

Host work per iteration is computed as self-CPU total minus the profiler's own
`Activity Buffer Request` and minus the two end-of-run `cudaDeviceSynchronize`
calls, divided by 20 iterations.

The four Triton kernels moved by at most 2.5% and the fourth not at all, which
is the control: nothing about the compute changed.

**The loop was host-bound.** 91 us of host work against 69 us of device work
means the device was starved on every iteration regardless of how fast the
kernels were. After the change the forward path contains one `cudaGraphLaunch`
and nothing else, and 12.6 us of host work sits against 64.7 us of device work.

## The fix, and the part of it that matters

Cache the predicate per mask object:

```python
if mask is not cache["obj"]:
    cache["all_true"] = bool(mask.all())
    cache["obj"] = mask
return cache["all_true"]
```

Keying on **object identity** rather than on a flag is deliberate, and it is the
same choice made for the graph input copy:

- it is exact and free, where `data_ptr()` is not (the caching allocator hands a
  freed tensor's address to the next same-sized allocation, so a pointer match
  does not prove the object is the same);
- it preserves correctness-test coverage. The accuracy loop passes a fresh mask
  per trial, so a fresh object always recomputes the real reduction. A flag set
  once at warmup would have made every correctness trial skip the code the timed
  loop skips, which is exactly the shape of the bug in LESSONS 17.

The contract this creates is that a caller mutating the mask in place, without
changing its identity, is served the previous answer. The official script cannot
do that: `valid_mask` is built once at `torch_transformer_benchmark.py:529` and
only ever read. Note also that the official numerical profile uses
`padding_ratio` 0.0 (`Project/harness/profile_worker.py:271`), so the mask is
all-true on every scored shape and the predicate's answer is constant regardless.

## What to take from this generally

1. **Grep the timed region for `.item()`, `bool()`, `int()`, `float()`, `.cpu()`,
   `.tolist()`, `print` of a tensor, and any Python `if` on a device value.**
   Each one is a sync. One per iteration is enough to make a host-bound loop out
   of a device-bound model.
2. **A profiler's "Self CUDA total" will not show you this.** The device time was
   only 6% lower after the fix; the win is in a gap that no kernel row reports.
   Look at host totals and at `cudaStreamSynchronize` instead.
3. **The smaller the shape, the bigger the effect.** Fixed host cost against
   shrinking device work. On this benchmark that means the small-batch and
   short-sequence shapes benefit most, and those are the ones where kernel
   micro-optimisation has the least left to give.
4. **The end-to-end size cannot be read from a profile.** The diagnostic lane's
   `run_route` (`Project/harness/profile_worker.py:1343-1348`) loops the model
   with no events, so the idle gap is never measured there. Only a runner
   measurement settles what this is worth. Treat the host/device accounting above
   as the reason to expect a large effect, not as the effect.
