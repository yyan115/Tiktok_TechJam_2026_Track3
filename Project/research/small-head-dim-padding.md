# Small head dimension and `tl.dot` padding waste (research 31 Aug ~00:20 SGT)

## Why this note exists

The complete post-LOCK screening board (STATE.md, 31 Aug) shows **narrow head dimension is
the single biggest lever** on this benchmark. Holding head count at 4 and narrowing
`head_dim` from 32 to 8 moved our route from **2.1428x** (shape 1) to **3.4781x**
(shape 7). Shape 11 (16 heads, `head_dim` 8) reaches **4.2433x**. So the shapes where the
eager baseline is weakest are exactly the shapes where our own kernel is *also* paying a
padding penalty — which makes `head_dim` 8 a target rather than a wall.

## The exact mechanism in our code

`Project/kernels/k003_triton_attention.py:103`:

```python
d_pad = max(16, triton.next_power_of_2(D))
```

For `D = 8` this yields **16**, not 8. The `max(16, ...)` is not arbitrary — it exists
because `tl.dot` requires a minimum tile of 16 in each dimension. Both attention matmuls
therefore run at half useful width on `head_dim` 8:

| matmul | shape | what the pad wastes |
| --- | --- | --- |
| QK^T | `[BLOCK_M, D_PAD] x [D_PAD, BLOCK_N]` | the **contraction** dim, 8 useful of 16 |
| PV | `[BLOCK_M, BLOCK_N] x [BLOCK_N, D_PAD]` | the **output width**, 8 useful of 16 |

So attention FLOPs are roughly doubled on shapes 7 and 11 versus what the math requires.

## What the literature says (searched 31 Aug 2026)

- **The 16-minimum is real and standard.** `tl.dot` maps to MMA units whose tile sizes are
  multiples of 8/16/32; sub-16 contraction dimensions must be zero-padded. Confirmed in
  *The Anatomy of a Triton Attention Kernel* (arXiv 2511.11581, Nov 2025), §8.
- **Padding is the recommended trade, not a bug.** That same paper states that despite the
  padding constraint `tl.dot` "is generally preferred, as it almost always results in
  better performance", because the compiler maps it directly to Tensor Cores. Elementwise
  multiply plus reduction is named as the alternative and is *not* recommended in general.
- **But the manual path is a documented technique for "lean" tensors.** Triton issue #793
  and discussion #1181 both describe replacing `tl.dot` with `tl.sum(a * b, axis=...)`
  when a matmul is too lean to use tensor cores efficiently. The tradeoff is explicit:
  CUDA-core FMA throughput instead of Tensor-Core throughput.
- **`tl.dot` also has known codegen pathologies on transposed operands** (issue #6569):
  element-by-element transposition through shared memory, sometimes slower than
  re-loading from global with transposed strides. Relevant because our QK^T uses a
  transposed K tile.

The searches found **no** published technique for packing multiple heads into one tile to
fill a sub-16 feature dimension, and no head-dimension-specialised autotune recipe.

## Why head-packing does not work here (arithmetic, not opinion)

The obvious idea — put two `head_dim` 8 heads side by side to make a full 16 — fails on
QK^T: `scores[m,n] = sum_d Q[m,d] K[n,d]` contracts *over* the feature dimension, so
packing two heads would sum their features together and produce a wrong score. It is only
admissible where the head dimension is a **free** index. In PV the head dimension *is*
free, but the probability matrix `P` differs per head, so the two heads cannot share one
`tl.dot` either. **Head packing is arithmetically unavailable for this kernel.**

## The one candidate technique, and its honest cost

Replace `tl.dot` with `tl.sum(q[:, None, :] * k[None, :, :], axis=2)` for the QK^T stage
only, at `D = 8`, keeping `tl.dot` for PV.

Cost to weigh before building it:
- It moves the QK^T from Tensor Cores to CUDA cores. On this card TF32 tensor peak is
  ~16.3 TF against ~16.2 TF fp32 CUDA-core peak, so the *rate* is roughly comparable in
  fp32 — which is unusual and is what makes this worth testing here specifically.
- It materialises a `[BLOCK_M, BLOCK_N, D]` intermediate. At `BLOCK_M=64, BLOCK_N=64,
  D=8` that is 32768 floats, far too large to keep in registers, so it forces small
  blocks and may lose more to occupancy than it gains from de-padding.

## MANDATORY next step before any kernel is written

**Measure the ceiling first.** LESSONS 32: compute what the mechanism can possibly deliver
before crediting it. Our route already achieves 4.2433x on shape 11 *with* the padding, so
the question is not "is the pad wasteful" (it is) but "what share of the remaining time is
the padded attention kernel". A torch-profiler diagnostic on k004 at shape 11 gives that
share directly, and it costs a diagnostic permit rather than an attempt. If `_attn_fwd` is
a small share of the forward, de-padding cannot pay for itself and this note ends here.

## Sources

- [The Anatomy of a Triton Attention Kernel (arXiv 2511.11581)](https://arxiv.org/html/2511.11581v1)
- [triton-lang/triton issue #793 — lean tensor multiplication](https://github.com/openai/triton/issues/793)
- [triton-lang/triton discussion #1181 — element-wise matrix multiplication performance](https://github.com/triton-lang/triton/discussions/1181)
- [triton-lang/triton issue #6569 — tl.dot on transposed matrix](https://github.com/triton-lang/triton/issues/6569)
- [Dao-AILab/flash-attention Triton reference](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py)
