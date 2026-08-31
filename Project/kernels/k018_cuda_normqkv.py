"""k018: hand-written CUDA C++ for the LayerNorm+QKV kernel, Triton elsewhere.

WHY THIS KERNEL AND NOT ANOTHER. The shape-9 diagnostic for the shipped file
(profile-c438a054f9d7866e1b1ccc94, torch-profiler, 20 iterations) splits device
time per forward as:

    _sub_norm_qkv          33.9%   3.917 ms / 80 calls = 48.97 us per layer
    _sub_attn_block_tail   33.1%   3.821 ms / 80 calls = 47.76 us per layer
    _sub_attn_heads        21.4%   2.475 ms / 80 calls = 30.93 us per layer
    Memcpy DtoD             7.0%   graph input copy + output clone
    _sub_final_norm         3.6%

norm_qkv is the single largest item. For shape 9 it does 805 MFLOP per layer
(tokens 8192 x D 128 x 3D 384 x 2) in 48.97 us = 16.4 TF/s, against a 32.4 TF/s
fp16-tensor-core-with-fp32-accumulate roof. Its memory floor is also ~25 us
(4 MB of fp32 x in, 6.3 MB of fp16 qkv out, at 448 GB/s), so compute and memory
floors nearly coincide at ~25 us and the kernel is running at about half of the
achievable rate. That is the largest identified gap on the board.

fp16 ACCUMULATION IS NOT AVAILABLE and this is worth stating because it looks
like free headroom: the 64.8 TF/s figure for this card is fp16-with-fp16-accum,
and the measured max absolute error on shape 13 is already 1.16e-3 against the
official 2e-3 criterion. There is no error budget to spend, so 32.4 TF/s is the
real roof.

WHAT THIS FILE CHANGES. Exactly one kernel. `_norm_qkv` becomes a CUDA C++
kernel compiled through torch.utils.cpp_extension.load_inline; `_attn_heads`,
`_block_tail` and `_final_norm` are byte-identical to k017, which is what the
submission currently ships. So any measured difference is attributable to the
one kernel, which is the whole point of running it this way rather than
rewriting the block.

THE DESIGN, and where it differs from what Triton generates:

  grid  (ceil(tokens/BLOCK_T), 3)   one block per (token tile, Q|K|V chunk)
  block 128 threads = 4 warps

  Each block stages its 128x128 fp16 weight chunk into shared memory once and
  reuses it across the whole token tile, and stages the LayerNormed tile as
  fp16 in shared. The GEMM is wmma 16x16x16 with fp32 accumulators. The weight
  matrix is [3D, D] row-major, so viewing a chunk as [K, N] column-major with
  ldm = D reads it correctly with no transpose and no staging transpose.

  The LayerNorm statistics are computed once per token tile per chunk, so they
  are computed three times overall. That is deliberate for a first version: x
  is 4 MB and stays L2-resident across the three chunk blocks, so the repeat is
  L2 traffic rather than HBM traffic. If this kernel wins, hoisting the norm is
  the obvious follow-up.

FALLBACK, and why this is safe to ship if it wins. Compilation happens at
import and is wrapped: if nvcc, ninja, or a writable build directory is
missing, `_CUDA_OK` stays False and the Triton kernel from k017 runs instead.
That is the same degradation ladder the shipped dispatcher already uses for
Triton itself, so a judge on a box without a CUDA toolchain gets exactly what
ships today rather than an error.

CONSTRAINTS. D <= 128 and D % 16 == 0, FFN <= 128, head_dim <= 128,
seq_len % 32 == 0, causal, no (or all-true) valid_token_mask -- the same fused
route the dispatcher already selects.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NAME = "k018_cuda_normqkv"
DESCRIPTION = ("k017 with the LayerNorm+QKV kernel rewritten in CUDA C++ "
               "(wmma, shared-resident weight chunk); every other kernel "
               "identical to k017.")

WARMUP_CALLS = 3
LN_EPS = tl.constexpr(1e-5)
LN_EPS_F = 1e-5

# --------------------------------------------------------------------------
# The CUDA kernel.
# --------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WARP 32
#define BLOCK_T 64      // tokens per block
#define NCOLS 128       // output columns per block (one of Q, K, V)
#define WARPS 4

// out[t, chunk*D + n] = sum_k ln(x)[t, k] * W[chunk*D + n, k] + B[chunk*D + n]
//
// Shared: Wsh  [NCOLS][D_MAX] fp16   weight chunk, staged once per block
//         Ysh  [BLOCK_T][D_MAX] fp16 LayerNormed tile
// with D_MAX = 128 this is 32 KB + 16 KB = 48 KB, so two blocks fit per SM.
template <int D_MAX>
__global__ __launch_bounds__(WARPS * WARP) void norm_qkv_kernel(
    const float* __restrict__ X,
    const at::Half* __restrict__ W,
    const at::Half* __restrict__ Bias,
    const float* __restrict__ LnW,
    const float* __restrict__ LnB,
    at::Half* __restrict__ Out,
    const int tokens,
    const int D) {
  __shared__ __half Wsh[NCOLS * D_MAX];
  __shared__ __half Ysh[BLOCK_T * D_MAX];
  __shared__ float red[WARPS * WARP];

  const int tile = blockIdx.x;
  const int chunk = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % WARP;
  const int warp = tid / WARP;
  const int t0 = tile * BLOCK_T;

  // ---- stage the weight chunk: W[chunk*D + n, k] for n in [0, D), k in [0, D)
  // Kept as [n][k] so the wmma B fragment can read it column-major with ldm=D.
  const at::Half* Wc = W + (size_t)chunk * D * D;
  for (int i = tid; i < D * D; i += WARPS * WARP) {
    Wsh[i] = __half(Wc[i]);
  }

  // ---- LayerNorm the token tile into shared, as fp16
  // One warp per token row; D <= 128 means at most 4 elements per lane.
  for (int r = warp; r < BLOCK_T; r += WARPS) {
    const int t = t0 + r;
    float sum = 0.f, sqsum = 0.f;
    if (t < tokens) {
      for (int k = lane; k < D; k += WARP) {
        const float v = X[(size_t)t * D + k];
        sum += v;
        sqsum += v * v;
      }
    }
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) {
      sum += __shfl_xor_sync(0xffffffffu, sum, off);
      sqsum += __shfl_xor_sync(0xffffffffu, sqsum, off);
    }
    const float mean = sum / D;
    const float var = sqsum / D - mean * mean;
    const float inv = rsqrtf(var + %(LN_EPS)s f);
    for (int k = lane; k < D; k += WARP) {
      float y = 0.f;
      if (t < tokens) {
        const float v = X[(size_t)t * D + k];
        y = (v - mean) * inv * LnW[k] + LnB[k];
      }
      Ysh[r * D_MAX + k] = __float2half(y);
    }
  }
  __syncthreads();

  // ---- GEMM: [BLOCK_T, D] x [D, NCOLS] with wmma 16x16x16, fp32 accumulate.
  // Warp w owns output rows [w*16, w*16+16) and all NCOLS columns, walking
  // columns in steps of 16. BLOCK_T / 16 == WARPS, so every warp owns one
  // 16-row band.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

  const int row0 = warp * 16;
  for (int n0 = 0; n0 < NCOLS; n0 += 16) {
    if (n0 >= D) break;
    wmma::fill_fragment(acc, 0.0f);
    for (int k0 = 0; k0 < D; k0 += 16) {
      wmma::load_matrix_sync(a_frag, Ysh + row0 * D_MAX + k0, D_MAX);
      // Wsh is [n][k]; as a [K, N] matrix that is column-major with ldm = D.
      wmma::load_matrix_sync(b_frag, Wsh + n0 * D + k0, D);
      wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
    // Epilogue: add bias, convert to fp16, store to the [tokens, 3D] output.
    #pragma unroll
    for (int e = 0; e < acc.num_elements; ++e) {
      red[tid] = acc.x[e];
    }
    __syncwarp();
    // wmma's accumulator layout is opaque, so store through shared and let
    // each lane write the elements it can address by (row, col).
    __shared__ float tile_out[WARPS][16][16];
    wmma::store_matrix_sync(&tile_out[warp][0][0], acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < 16 * 16; idx += WARP) {
      const int r = idx / 16;
      const int c = idx % 16;
      const int t = t0 + row0 + r;
      const int n = n0 + c;
      if (t < tokens && n < D) {
        const float b = __half2float(__half(Bias[chunk * D + n]));
        Out[(size_t)t * (3 * D) + chunk * D + n] =
            at::Half(__float2half(tile_out[warp][r][c] + b));
      }
    }
    __syncwarp();
  }
}

void norm_qkv_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor Bias,
                   torch::Tensor LnW, torch::Tensor LnB, torch::Tensor Out,
                   int64_t tokens, int64_t D) {
  TORCH_CHECK(D <= 128 && D %% 16 == 0, "k018 requires D <= 128 and D %% 16 == 0");
  const dim3 grid((tokens + BLOCK_T - 1) / BLOCK_T, 3);
  const dim3 block(WARPS * WARP);
  auto stream = at::cuda::getCurrentCUDAStream();
  norm_qkv_kernel<128><<<grid, block, 0, stream>>>(
      X.data_ptr<float>(),
      W.data_ptr<at::Half>(),
      Bias.data_ptr<at::Half>(),
      LnW.data_ptr<float>(),
      LnB.data_ptr<float>(),
      Out.data_ptr<at::Half>(),
      (int)tokens, (int)D);
}
""" % {"LN_EPS": "1e-5"}

_CPP_SRC = r"""
#include <torch/extension.h>
void norm_qkv_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor Bias,
                   torch::Tensor LnW, torch::Tensor LnB, torch::Tensor Out,
                   int64_t tokens, int64_t D);
"""

_CUDA_MOD = None
_CUDA_OK = False
_CUDA_ERROR = None


def _load_cuda():
    """Compile the extension once. Any failure leaves the Triton path in place."""
    global _CUDA_MOD, _CUDA_OK, _CUDA_ERROR
    if _CUDA_MOD is not None or _CUDA_ERROR is not None:
        return _CUDA_OK
    try:
        from torch.utils.cpp_extension import load_inline
        build_dir = os.environ.get(
            "K018_BUILD_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_k018_build"))
        os.makedirs(build_dir, exist_ok=True)
        _CUDA_MOD = load_inline(
            name="k018_norm_qkv",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["norm_qkv_cuda"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_86"],
            build_directory=build_dir,
            verbose=False,
        )
        _CUDA_OK = True
    except Exception as exc:  # nvcc missing, no ninja, read-only fs, ptxas error
        _CUDA_ERROR = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
        _CUDA_OK = False
    return _CUDA_OK


# --------------------------------------------------------------------------
# Triton kernels below are unchanged from k017 (the shipped design).
# --------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 16}, num_warps=2),
        triton.Config({"BLOCK_T": 32}, num_warps=2),
        triton.Config({"BLOCK_T": 32}, num_warps=4),
        triton.Config({"BLOCK_T": 64}, num_warps=4),
        triton.Config({"BLOCK_T": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 128}, num_warps=8),
    ],
    key=["D_PAD", "TOKENS"],
)
@triton.jit
def _norm_qkv(
    X, W, Bias, LnW, LnB, Out,
    TOKENS, D: tl.constexpr, D_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D_PAD)
    t_mask = offs_t < TOKENS
    d_mask = offs_d < D

    x = tl.load(X + offs_t[:, None] * D + offs_d[None, :],
                mask=t_mask[:, None] & d_mask[None, :], other=0.0)
    mean = tl.sum(x, axis=1) / D
    diff = tl.where(d_mask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / D
    inv = 1.0 / tl.sqrt(var + LN_EPS)
    lnw = tl.load(LnW + offs_d, mask=d_mask, other=0.0)
    lnb = tl.load(LnB + offs_d, mask=d_mask, other=0.0)
    y = (diff * inv[:, None]) * lnw[None, :] + lnb[None, :]
    y16 = y.to(tl.float16)

    for chunk in tl.static_range(3):
        w = tl.load(W + (chunk * D + offs_d[:, None]) * D + offs_d[None, :],
                    mask=d_mask[:, None] & d_mask[None, :], other=0.0)
        acc = tl.dot(y16, tl.trans(w))
        b = tl.load(Bias + chunk * D + offs_d, mask=d_mask, other=0.0)
        acc = acc + b[None, :].to(tl.float32)
        tl.store(Out + offs_t[:, None] * (3 * D) + chunk * D + offs_d[None, :],
                 acc.to(tl.float16),
                 mask=t_mask[:, None] & d_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8),
    ],
    key=["SEQ", "HD_PAD"],
)
@triton.jit
def _attn_heads(
    QKV, Ctx,
    scale,
    SEQ,
    D: tl.constexpr,
    HD: tl.constexpr, HD_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_hd = tl.arange(0, HD_PAD)
    m_mask = offs_m < SEQ
    hd_mask = offs_hd < HD

    qkv_base = QKV + pid_b * SEQ * (3 * D)
    q = tl.load(qkv_base + offs_m[:, None] * (3 * D) + h * HD + offs_hd[None, :],
                mask=m_mask[:, None] & hd_mask[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HD_PAD], dtype=tl.float32)

    n_end = (pid_m + 1) * BLOCK_M
    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < SEQ
        k = tl.load(qkv_base + offs_n[:, None] * (3 * D) + D + h * HD + offs_hd[None, :],
                    mask=n_mask[:, None] & hd_mask[None, :], other=0.0)
        v = tl.load(qkv_base + offs_n[:, None] * (3 * D) + 2 * D + h * HD + offs_hd[None, :],
                    mask=n_mask[:, None] & hd_mask[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        qk = tl.where(n_mask[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    head_out = (acc / l_i[:, None]).to(tl.float16)
    tl.store(Ctx + pid_b * SEQ * D + offs_m[:, None] * D + h * HD + offs_hd[None, :],
             head_out, mask=m_mask[:, None] & hd_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16}, num_warps=2),
        triton.Config({"BLOCK_M": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 32}, num_warps=2),
        triton.Config({"BLOCK_M": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128}, num_warps=8),
    ],
    key=["SEQ", "D_PAD", "FFN_PAD"],
)
@triton.jit
def _block_tail(
    Ctx, X, Wo, Bo, Ln2W, Ln2B, Wf1, Bf1, Wf2, Bf2, XOut,
    SEQ,
    D: tl.constexpr, D_PAD: tl.constexpr,
    FFN: tl.constexpr, FFN_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_PAD)
    m_mask = offs_m < SEQ
    d_mask = offs_d < D

    ctx = tl.load(Ctx + pid_b * SEQ * D + offs_m[:, None] * D + offs_d[None, :],
                  mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    wo = tl.load(Wo + offs_d[:, None] * D + offs_d[None, :],
                 mask=d_mask[:, None] & d_mask[None, :], other=0.0)
    attn = tl.dot(ctx, tl.trans(wo))
    bo = tl.load(Bo + offs_d, mask=d_mask, other=0.0)
    attn = attn + bo[None, :].to(tl.float32)

    x = tl.load(X + pid_b * SEQ * D + offs_m[:, None] * D + offs_d[None, :],
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    x2 = x + attn

    mean = tl.sum(tl.where(d_mask[None, :], x2, 0.0), axis=1) / D
    diff = tl.where(d_mask[None, :], x2 - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / D
    inv = 1.0 / tl.sqrt(var + LN_EPS)
    ln2w = tl.load(Ln2W + offs_d, mask=d_mask, other=0.0)
    ln2b = tl.load(Ln2B + offs_d, mask=d_mask, other=0.0)
    h2 = ((diff * inv[:, None]) * ln2w[None, :] + ln2b[None, :]).to(tl.float16)

    offs_f = tl.arange(0, FFN_PAD)
    f_mask = offs_f < FFN
    wf1 = tl.load(Wf1 + offs_f[:, None] * D + offs_d[None, :],
                  mask=f_mask[:, None] & d_mask[None, :], other=0.0)
    hid = tl.dot(h2, tl.trans(wf1))
    bf1 = tl.load(Bf1 + offs_f, mask=f_mask, other=0.0)
    hid = hid + bf1[None, :].to(tl.float32)
    hid = 0.5 * hid * (1.0 + tl.math.erf(hid * 0.7071067811865476))
    hid16 = hid.to(tl.float16)
    wf2 = tl.load(Wf2 + offs_d[:, None] * FFN + offs_f[None, :],
                  mask=d_mask[:, None] & f_mask[None, :], other=0.0)
    ffn = tl.dot(hid16, tl.trans(wf2))
    bf2 = tl.load(Bf2 + offs_d, mask=d_mask, other=0.0)
    ffn = ffn + bf2[None, :].to(tl.float32)

    out = x2 + ffn
    tl.store(XOut + pid_b * SEQ * D + offs_m[:, None] * D + offs_d[None, :],
             out, mask=m_mask[:, None] & d_mask[None, :])


@triton.jit
def _final_norm(
    X, LnW, LnB, Out,
    TOKENS, D: tl.constexpr, D_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D_PAD)
    t_mask = offs_t < TOKENS
    d_mask = offs_d < D
    x = tl.load(X + offs_t[:, None] * D + offs_d[None, :],
                mask=t_mask[:, None] & d_mask[None, :], other=0.0)
    mean = tl.sum(x, axis=1) / D
    diff = tl.where(d_mask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) / D
    inv = 1.0 / tl.sqrt(var + LN_EPS)
    lnw = tl.load(LnW + offs_d, mask=d_mask, other=0.0)
    lnb = tl.load(LnB + offs_d, mask=d_mask, other=0.0)
    y = (diff * inv[:, None]) * lnw[None, :] + lnb[None, :]
    tl.store(Out + offs_t[:, None] * D + offs_d[None, :], y,
             mask=t_mask[:, None] & d_mask[None, :])


def _pack_layer(layer):
    cache = getattr(layer, "_k018_cache", None)
    if cache is None:
        attn = layer.attention
        cache = {
            "w_qkv": torch.cat([attn.q_proj.weight, attn.k_proj.weight,
                                attn.v_proj.weight], dim=0).half().contiguous(),
            "b_qkv": torch.cat([attn.q_proj.bias, attn.k_proj.bias,
                                attn.v_proj.bias], dim=0).half().contiguous(),
            "w_o": attn.out_proj.weight.half().contiguous(),
            "b_o": attn.out_proj.bias.half().contiguous(),
            "w_f1": layer.ffn_in.weight.half().contiguous(),
            "b_f1": layer.ffn_in.bias.half().contiguous(),
            "w_f2": layer.ffn_out.weight.half().contiguous(),
            "b_f2": layer.ffn_out.bias.half().contiguous(),
            "ln1_w": layer.norm1.weight.float().contiguous(),
            "ln1_b": layer.norm1.bias.float().contiguous(),
            "ln2_w": layer.norm2.weight.float().contiguous(),
            "ln2_b": layer.norm2.bias.float().contiguous(),
        }
        object.__setattr__(layer, "_k018_cache", cache)
    return cache


def _fused_forward(model, x):
    cfg = model.config
    B, S, D = x.shape
    H = cfg.num_heads
    HD = D // H
    FFN = cfg.ffn_dim
    tokens = B * S
    d_pad = max(16, triton.next_power_of_2(D))
    hd_pad = max(16, triton.next_power_of_2(HD))
    ffn_pad = max(16, triton.next_power_of_2(FFN))
    scale = HD ** -0.5
    use_cuda = _CUDA_OK and D <= 128 and D % 16 == 0

    bufs = getattr(model, "_k018_buf", None)
    if bufs is None or bufs["a"].shape != x.shape:
        bufs = {
            "qkv": torch.empty(tokens, 3 * D, dtype=torch.float16, device=x.device),
            "ctx": torch.empty(tokens, D, dtype=torch.float16, device=x.device),
            "a": torch.empty_like(x),
            "b": torch.empty_like(x),
            "fn_w": model.final_norm.weight.float().contiguous(),
            "fn_b": model.final_norm.bias.float().contiguous(),
        }
        object.__setattr__(model, "_k018_buf", bufs)
    qkv = bufs["qkv"]
    ctx = bufs["ctx"]

    src = x.contiguous()
    for i, layer in enumerate(model.layers):
        dst = bufs["a"] if i % 2 == 0 else bufs["b"]
        c = _pack_layer(layer)
        if use_cuda:
            _CUDA_MOD.norm_qkv_cuda(
                src.view(tokens, D), c["w_qkv"], c["b_qkv"],
                c["ln1_w"], c["ln1_b"], qkv, tokens, D)
        else:
            grid1 = lambda meta: (triton.cdiv(tokens, meta["BLOCK_T"]),)  # noqa: E731
            _norm_qkv[grid1](
                src.view(tokens, D), c["w_qkv"], c["b_qkv"], c["ln1_w"], c["ln1_b"],
                qkv, tokens, D=D, D_PAD=d_pad,
            )
        grid2 = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B, H)  # noqa: E731
        _attn_heads[grid2](
            qkv, ctx, scale, S,
            D=D, HD=HD, HD_PAD=hd_pad,
        )
        grid3 = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B)  # noqa: E731
        _block_tail[grid3](
            ctx, src, c["w_o"], c["b_o"], c["ln2_w"], c["ln2_b"],
            c["w_f1"], c["b_f1"], c["w_f2"], c["b_f2"], dst,
            S, D=D, D_PAD=d_pad, FFN=FFN, FFN_PAD=ffn_pad,
        )
        src = dst
    out = torch.empty_like(src)
    _final_norm[(triton.cdiv(tokens, 128),)](
        src.view(tokens, D), bufs["fn_w"], bufs["fn_b"], out.view(tokens, D),
        tokens, D=D, D_PAD=d_pad, BLOCK_T=128, num_warps=4,
    )
    return out


def build(otb, config):
    _load_cuda()
    model = otb.UserOptimizedTransformer(config)
    fused_ok = (
        config.d_model <= 128 and config.ffn_dim <= 128
        and config.d_model % config.num_heads == 0
        and (config.d_model // config.num_heads) <= 128
        and config.seq_len % 32 == 0 and config.causal
    )

    class CudaNormQkvTransformer(otb.UserOptimizedTransformer):
        cuda_available = _CUDA_OK
        cuda_error = _CUDA_ERROR

        def _eager(self, x, valid_token_mask):
            return otb.BaselineTransformer.forward(self, x, valid_token_mask)

        def forward(self, x, valid_token_mask=None):
            if valid_token_mask is not None and not bool(valid_token_mask.all()):
                return self._eager(x, valid_token_mask)
            if not fused_ok:
                return self._eager(x, None)
            state = getattr(self, "_graph_state", None)
            if state is None:
                state = {"calls": 0, "graph": None, "static_x": None, "static_out": None}
                object.__setattr__(self, "_graph_state", state)
            if state["graph"] is None:
                state["calls"] += 1
                if state["calls"] <= WARMUP_CALLS:
                    return _fused_forward(self, x)
                static_x = x.clone()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        _fused_forward(self, static_x)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    static_out = _fused_forward(self, static_x)
                state.update(graph=graph, static_x=static_x, static_out=static_out)
                graph.replay()
                return static_out.clone()
            state["static_x"].copy_(x)
            state["graph"].replay()
            return state["static_out"].clone()

    model.__class__ = CudaNormQkvTransformer
    return model
