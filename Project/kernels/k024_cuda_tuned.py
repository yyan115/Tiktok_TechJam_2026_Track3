"""k024: CUDA LayerNorm+QKV done properly. Supersedes k023.

k023 was a naive first draft and its 4-5x loss said nothing about CUDA. It said
my kernel was bad, in five specific and well-understood ways. This file fixes
all five. The comparison against Triton is only meaningful once they are fixed.

MEASURED BASELINE on shape 1, per layer: Triton 49.3 us, k023 v1 203.3 us,
k023 v2 248.6 us.

THE FIVE DEFECTS AND THE FIX FOR EACH.

1. WEIGHT REUSE. k023 v1 called wmma::load_matrix_sync on the B fragment from
   GLOBAL inside the innermost loop -- 64 fragment loads per warp per tile with
   no reuse whatsoever. v2 staged the whole 32 KB chunk in shared, which fixed
   the reuse and destroyed occupancy (64 threads/block, ~8%).
   FIX: give each warp ONE 16-wide output column band and let it hold its eight
   B fragments in REGISTERS for the lifetime of the block. Eight fragments x 8
   halves = 64 halves = 32 registers per thread. The weights are read from
   global exactly once per block, never re-read, and cost no shared memory at
   all. This is the pivot the previous two versions both missed: the reuse
   problem and the occupancy problem have a common solution, and it is neither
   "leave it in global" nor "put it in shared".

2. THE INPUT WAS READ TWICE. k023 looped over X to accumulate the moments, then
   looped over X AGAIN to normalise -- 2x the global traffic on the largest
   input, in a kernel whose memory floor is ~24 us of a measured 49.
   FIX: each lane loads its slice once into registers, reduces, then normalises
   the values it is already holding.

3. SCALAR LOADS. `for (k = lane; k < D; k += WARP)` reads one float at a time.
   FIX: one warp per row, lane l takes X[row][4l .. 4l+3] as a single float4.
   32 lanes x 4 floats = 128 = exactly one row at D=128. Four times fewer
   memory instructions for the same bytes, and the store to shared is one
   8-byte half4 per lane.

4. BANK CONFLICTS. The shared tile was stored with stride D=128 halves = 256
   bytes, so every wmma row access hits the same bank.
   FIX: pad the row stride to D+8 halves. 136 is a multiple of 8, which wmma
   requires for a half ldm, and breaks the power-of-two aliasing.

5. NO AUTOTUNING AT ALL. Triton declares seven configs and MEASURES which wins
   per shape. k023 had BLOCK_T and WARPS hardcoded to numbers I picked once by
   hand and never tested. This is probably the single largest term and it is
   not a language difference, it is an effort difference.
   FIX: the module sweeps the configs below at warm-up, times each with CUDA
   events on the real shape, and keeps the winner. Same thing Triton's
   autotuner does, done explicitly.

SHARED BUDGET at D=128, BLOCK_T=64, 8 warps:
    Ysh  64 x 136 halves  = 17.0 KB
    Osh  8 x 16 x 16 fp32 =  8.0 KB
                            25.0 KB  -> two blocks/SM, no opt-in needed

WHAT WOULD STILL BE MISSING if this loses. Multi-stage cp.async pipelining of
the global->shared path, and warp specialisation. Those are the CUTLASS-grade
items. If this version is still far behind Triton then that is the honest
boundary of what hand-written CUDA buys here without a CUTLASS-scale build,
and the answer to "is CUDA the way" is no at this effort level -- which is a
statement about effort, not about the language ceiling.
"""

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NAME = "k024_cuda_tuned"
DESCRIPTION = ("CUDA LayerNorm+QKV with register-resident weight fragments, "
               "single vectorized input read, padded shared, and a warm-up "
               "config sweep; Triton for every other kernel.")

WARMUP_CALLS = 3
LN_EPS = tl.constexpr(1e-5)

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WARP 32
#define D_MAX 128
#define SPAD 8                       // shared row padding, halves
#define LN_EPS 1e-5f

// One block owns one QKV chunk and strides over token tiles.
// Warp w owns output columns [w*16, w*16+16) and holds its B fragments in
// registers for the whole block -- the weights are read from global once.
template <int BLOCK_T, int WARPS>
__global__ __launch_bounds__(WARPS * WARP) void nq_kernel(
    const float* __restrict__ X,
    const __half* __restrict__ W,
    const __half* __restrict__ Bias,
    const float* __restrict__ LnW,
    const float* __restrict__ LnB,
    __half* __restrict__ Out,
    const int tokens,
    const int D) {
  constexpr int STRIDE = D_MAX + SPAD;
  __shared__ __half Ysh[BLOCK_T * STRIDE];
  __shared__ float Osh[WARPS][16][16];

  const int chunk = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % WARP;
  const int warp = tid / WARP;
  // Warp w owns output columns [w*16, w*16+16). WARPS * 16 MUST equal D or the
  // tail columns are silently never written -- an earlier revision allowed
  // WARPS=4 at D=128, which covered only columns 0-63, ran at half the work,
  // and was duly selected by the config sweep as the "fastest". The host now
  // derives WARPS = D/16 so the config space cannot contain a wrong kernel.
  const int n0 = warp * 16;
  const bool active = (n0 < D);

  const __half* Wc = W + (size_t)chunk * D * D;

  // ---- (1) weight fragments into registers, once per block, never re-read.
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[D_MAX / 16];
  const int ksteps = D / 16;
  if (active) {
    for (int k = 0; k < ksteps; ++k) {
      // Wc is [n][k] row-major; as a [K,N] matrix that is column-major, ldm=D.
      wmma::load_matrix_sync(bf[k], Wc + n0 * D + k * 16, D);
    }
  }

  const int n_tiles = (tokens + BLOCK_T - 1) / BLOCK_T;
  const int vec = D / 4;                        // float4 lanes per row

  for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
    const int t0 = tile * BLOCK_T;

    // ---- (2)(3) one warp per row; each lane loads ONE float4, reduces, and
    // normalises the values it already holds. X is read exactly once.
    for (int r = warp; r < BLOCK_T; r += WARPS) {
      const int t = t0 + r;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      const bool live = (t < tokens) && (lane < vec);
      if (live) {
        v = reinterpret_cast<const float4*>(X + (size_t)t * D)[lane];
      }
      float s = v.x + v.y + v.z + v.w;
      float s2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
      #pragma unroll
      for (int off = WARP / 2; off > 0; off >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, off);
        s2 += __shfl_xor_sync(0xffffffffu, s2, off);
      }
      const float mean = s / D;
      const float inv = rsqrtf(s2 / D - mean * mean + LN_EPS);
      if (lane < vec) {
        const float4 gw = reinterpret_cast<const float4*>(LnW)[lane];
        const float4 gb = reinterpret_cast<const float4*>(LnB)[lane];
        __half h[4];
        h[0] = __float2half(live ? (v.x - mean) * inv * gw.x + gb.x : 0.f);
        h[1] = __float2half(live ? (v.y - mean) * inv * gw.y + gb.y : 0.f);
        h[2] = __float2half(live ? (v.z - mean) * inv * gw.z + gb.z : 0.f);
        h[3] = __float2half(live ? (v.w - mean) * inv * gw.w + gb.w : 0.f);
        // (4) padded stride: STRIDE halves, not D, so wmma row reads do not
        // all land in the same bank.
        *reinterpret_cast<short4*>(&Ysh[r * STRIDE + lane * 4]) =
            *reinterpret_cast<short4*>(h);
      }
    }
    __syncthreads();

    // ---- GEMM: this warp's 16 columns for every row band of the tile.
    if (active) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      for (int row0 = 0; row0 < BLOCK_T; row0 += 16) {
        wmma::fill_fragment(acc, 0.0f);
        for (int k = 0; k < ksteps; ++k) {
          wmma::load_matrix_sync(af, Ysh + row0 * STRIDE + k * 16, STRIDE);
          wmma::mma_sync(acc, af, bf[k], acc);
        }
        wmma::store_matrix_sync(&Osh[warp][0][0], acc, 16, wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 256; idx += WARP) {
          const int rr = idx >> 4, cc = idx & 15;
          const int t = t0 + row0 + rr, n = n0 + cc;
          if (t < tokens && n < D) {
            Out[(size_t)t * (3 * D) + chunk * D + n] =
                __float2half(Osh[warp][rr][cc] +
                             __half2float(Bias[chunk * D + n]));
          }
        }
        __syncwarp();
      }
    }
    __syncthreads();
  }
}

// (5) the config sweep. Each entry is a (BLOCK_T, WARPS) pair; the host picks
// the winner by timing them on the real shape at warm-up.
void nq_launch(int cfg, torch::Tensor X, torch::Tensor W, torch::Tensor Bias,
               torch::Tensor LnW, torch::Tensor LnB, torch::Tensor Out,
               int64_t tokens, int64_t D) {
  const int props = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto go = [&](int bt, int warps, auto kern) {
    const int n_tiles = (int)((tokens + bt - 1) / bt);
    int gx = n_tiles < 4 * props ? n_tiles : 4 * props;
    if (gx < 1) gx = 1;
    kern<<<dim3(gx, 3), dim3(warps * WARP), 0,
           at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(),
        reinterpret_cast<const __half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(Bias.data_ptr<at::Half>()),
        LnW.data_ptr<float>(), LnB.data_ptr<float>(),
        reinterpret_cast<__half*>(Out.data_ptr<at::Half>()),
        (int)tokens, (int)D);
  };
  // WARPS is DERIVED from D, never swept: warps must exactly cover the output
  // width. Only the token-tile height is a free parameter. BLOCK_T 256 is
  // absent because at D=128 its shared tile alone is 69 KB.
  const int warps = (int)(D / 16);
  switch (cfg * 3 + (warps == 2 ? 0 : warps == 4 ? 1 : 2)) {
    case 0:  go(32,  2, nq_kernel<32, 2>);   break;
    case 1:  go(32,  4, nq_kernel<32, 4>);   break;
    case 2:  go(32,  8, nq_kernel<32, 8>);   break;
    case 3:  go(64,  2, nq_kernel<64, 2>);   break;
    case 4:  go(64,  4, nq_kernel<64, 4>);   break;
    case 5:  go(64,  8, nq_kernel<64, 8>);   break;
    case 6:  go(128, 2, nq_kernel<128, 2>);  break;
    case 7:  go(128, 4, nq_kernel<128, 4>);  break;
    case 8:  go(128, 8, nq_kernel<128, 8>);  break;
    default: go(64,  8, nq_kernel<64, 8>);   break;
  }
}

int64_t nq_num_configs() { return 3; }   // BLOCK_T in {32, 64, 128}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
void nq_launch(int cfg, torch::Tensor X, torch::Tensor W, torch::Tensor Bias,
               torch::Tensor LnW, torch::Tensor LnB, torch::Tensor Out,
               int64_t tokens, int64_t D);
int64_t nq_num_configs();
"""

_CUDA_MOD = None
_CUDA_OK = False
_CUDA_ERROR = None
_BEST_CFG = None


def _load_cuda():
    global _CUDA_MOD, _CUDA_OK, _CUDA_ERROR
    if _CUDA_MOD is not None or _CUDA_ERROR is not None:
        return _CUDA_OK
    try:
        from torch.utils.cpp_extension import load_inline
        build_dir = os.environ.get("K024_BUILD_DIR", "/tmp/k024_build")
        os.makedirs(build_dir, exist_ok=True)
        # The sandbox host compiler is gcc 16 and CUDA 13's nvcc hard-refuses
        # anything above 15 (crt/host_config.h). g++-15 exists on the host but
        # not in the profile sandbox, so the override is what actually works
        # here -- established by reading the failure, not by guessing.
        base = ["-O3", "--use_fast_math", "-arch=sm_86"]
        for tag, extra in (("ccbin15", ["-ccbin=g++-15"]),
                           ("allow", ["-allow-unsupported-compiler"])):
            try:
                _CUDA_MOD = load_inline(
                    name=f"k024_nq_{tag}",
                    cpp_sources=_CPP_SRC,
                    cuda_sources=_CUDA_SRC,
                    functions=["nq_launch", "nq_num_configs"],
                    extra_cuda_cflags=base + extra,
                    build_directory=build_dir,
                    verbose=False,
                )
                _CUDA_OK = True
                _CUDA_ERROR = None
                return True
            except Exception as exc:
                _CUDA_ERROR = f"[{tag}] {type(exc).__name__}: {str(exc)[:1500]}"
                continue
        _CUDA_OK = False
    except Exception as exc:
        _CUDA_ERROR = f"{type(exc).__name__}: {str(exc)[:1500]}"
        _CUDA_OK = False
    return _CUDA_OK


def _pick_config(x_view, c, qkv, tokens, D):
    """Time every config on the real shape and keep the winner.

    This is what Triton's autotuner does and what k023 conspicuously lacked.
    Run once, outside graph capture, during warm-up.
    """
    global _BEST_CFG
    if _BEST_CFG is not None:
        return _BEST_CFG
    n = int(_CUDA_MOD.nq_num_configs())
    best, best_ms = 0, float("inf")
    for cfg in range(n):
        try:
            for _ in range(3):  # warm the config
                _CUDA_MOD.nq_launch(cfg, x_view, c["w_qkv"], c["b_qkv"],
                                    c["ln1_w"], c["ln1_b"], qkv, tokens, D)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                _CUDA_MOD.nq_launch(cfg, x_view, c["w_qkv"], c["b_qkv"],
                                    c["ln1_w"], c["ln1_b"], qkv, tokens, D)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / 20.0
            if ms < best_ms:
                best_ms, best = ms, cfg
        except Exception:
            continue
    _BEST_CFG = best
    return best


# --------------------------------------------------------------------------
# Triton path. _norm_qkv is the shipped split-grid kernel and is the fallback
# when CUDA is unavailable; the other three are identical to what ships, so a
# measured delta is attributable to the CUDA kernel alone.
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
    TOKENS, D: tl.constexpr, D_PAD: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = tl.program_id(1)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D_PAD)
    t_mask = offs_t < TOKENS
    d_mask = offs_d < D
    x = tl.load(X + offs_t[:, None] * D + offs_d[None, :],
                mask=t_mask[:, None] & d_mask[None, :], other=0.0)
    xm = tl.where(d_mask[None, :], x, 0.0)
    mean = tl.sum(xm, axis=1) / D
    var = tl.sum(xm * xm, axis=1) / D - mean * mean
    inv = 1.0 / tl.sqrt(var + LN_EPS)
    diff = tl.where(d_mask[None, :], x - mean[:, None], 0.0)
    lnw = tl.load(LnW + offs_d, mask=d_mask, other=0.0)
    lnb = tl.load(LnB + offs_d, mask=d_mask, other=0.0)
    y16 = ((diff * inv[:, None]) * lnw[None, :] + lnb[None, :]).to(tl.float16)
    w = tl.load(W + (chunk * D + offs_d[:, None]) * D + offs_d[None, :],
                mask=d_mask[:, None] & d_mask[None, :], other=0.0)
    acc = tl.dot(y16, tl.trans(w))
    b = tl.load(Bias + chunk * D + offs_d, mask=d_mask, other=0.0)
    acc = acc + b[None, :].to(tl.float32)
    tl.store(Out + offs_t[:, None] * (3 * D) + chunk * D + offs_d[None, :],
             acc.to(tl.float16), mask=t_mask[:, None] & d_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=2),
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
    QKV, Ctx, scale, SEQ,
    D: tl.constexpr, HD: tl.constexpr, HD_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    h = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_hd = tl.arange(0, HD_PAD)
    m_mask = offs_m < SEQ
    hd_mask = offs_hd < HD
    base = QKV + pid_b * SEQ * (3 * D)
    q = tl.load(base + offs_m[:, None] * (3 * D) + h * HD + offs_hd[None, :],
                mask=m_mask[:, None] & hd_mask[None, :], other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HD_PAD], dtype=tl.float32)
    for n_start in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < SEQ
        k = tl.load(base + offs_n[:, None] * (3 * D) + D + h * HD + offs_hd[None, :],
                    mask=n_mask[:, None] & hd_mask[None, :], other=0.0)
        v = tl.load(base + offs_n[:, None] * (3 * D) + 2 * D + h * HD + offs_hd[None, :],
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
    tl.store(Ctx + pid_b * SEQ * D + offs_m[:, None] * D + h * HD + offs_hd[None, :],
             (acc / l_i[:, None]).to(tl.float16),
             mask=m_mask[:, None] & hd_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16}, num_warps=2),
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
    Ctx, X, Wo, Bo, Ln2W, Ln2B, Wf1, Bf1, Wf2, Bf2, XOut, SEQ,
    D: tl.constexpr, D_PAD: tl.constexpr,
    FFN: tl.constexpr, FFN_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
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
    x = tl.load(X + pid_b * SEQ * D + offs_m[:, None] * D + offs_d[None, :],
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    x2 = x + attn + bo[None, :].to(tl.float32)
    x2m = tl.where(d_mask[None, :], x2, 0.0)
    mean = tl.sum(x2m, axis=1) / D
    var = tl.sum(x2m * x2m, axis=1) / D - mean * mean
    inv = 1.0 / tl.sqrt(var + LN_EPS)
    diff = tl.where(d_mask[None, :], x2 - mean[:, None], 0.0)
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
    wf2 = tl.load(Wf2 + offs_d[:, None] * FFN + offs_f[None, :],
                  mask=d_mask[:, None] & f_mask[None, :], other=0.0)
    ffn = tl.dot(hid.to(tl.float16), tl.trans(wf2))
    bf2 = tl.load(Bf2 + offs_d, mask=d_mask, other=0.0)
    out = x2 + ffn + bf2[None, :].to(tl.float32)
    tl.store(XOut + pid_b * SEQ * D + offs_m[:, None] * D + offs_d[None, :],
             out, mask=m_mask[:, None] & d_mask[None, :])


@triton.jit
def _final_norm(
    X, LnW, LnB, Out, TOKENS,
    D: tl.constexpr, D_PAD: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D_PAD)
    t_mask = offs_t < TOKENS
    d_mask = offs_d < D
    x = tl.load(X + offs_t[:, None] * D + offs_d[None, :],
                mask=t_mask[:, None] & d_mask[None, :], other=0.0)
    xm = tl.where(d_mask[None, :], x, 0.0)
    mean = tl.sum(xm, axis=1) / D
    var = tl.sum(xm * xm, axis=1) / D - mean * mean
    inv = 1.0 / tl.sqrt(var + LN_EPS)
    diff = tl.where(d_mask[None, :], x - mean[:, None], 0.0)
    lnw = tl.load(LnW + offs_d, mask=d_mask, other=0.0)
    lnb = tl.load(LnB + offs_d, mask=d_mask, other=0.0)
    tl.store(Out + offs_t[:, None] * D + offs_d[None, :],
             (diff * inv[:, None]) * lnw[None, :] + lnb[None, :],
             mask=t_mask[:, None] & d_mask[None, :])


def _pack_layer(layer):
    cache = getattr(layer, "_k024_cache", None)
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
        object.__setattr__(layer, "_k024_cache", cache)
    return cache


def _fused_forward(model, x, tune=False):
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
    # D % 16 == 0 and D <= 128 is the CUDA kernel's contract; D/16 warps must
    # also cover the output width, which holds for every fused-route shape.
    use_cuda = _CUDA_OK and D <= 128 and D % 16 == 0

    bufs = getattr(model, "_k024_buf", None)
    if bufs is None or bufs["a"].shape != x.shape:
        bufs = {
            "qkv": torch.empty(tokens, 3 * D, dtype=torch.float16, device=x.device),
            "ctx": torch.empty(tokens, D, dtype=torch.float16, device=x.device),
            "a": torch.empty_like(x),
            "b": torch.empty_like(x),
            "fn_w": model.final_norm.weight.float().contiguous(),
            "fn_b": model.final_norm.bias.float().contiguous(),
        }
        object.__setattr__(model, "_k024_buf", bufs)
    qkv = bufs["qkv"]
    ctx = bufs["ctx"]

    src = x.contiguous()
    for i, layer in enumerate(model.layers):
        dst = bufs["a"] if i % 2 == 0 else bufs["b"]
        c = _pack_layer(layer)
        if use_cuda:
            if tune:
                _pick_config(src.view(tokens, D), c, qkv, tokens, D)
            _CUDA_MOD.nq_launch(
                _BEST_CFG if _BEST_CFG is not None else 1,
                src.view(tokens, D), c["w_qkv"], c["b_qkv"],
                c["ln1_w"], c["ln1_b"], qkv, tokens, D)
        else:
            g1 = lambda meta: (triton.cdiv(tokens, meta["BLOCK_T"]), 3)  # noqa: E731
            _norm_qkv[g1](src.view(tokens, D), c["w_qkv"], c["b_qkv"],
                          c["ln1_w"], c["ln1_b"], qkv, tokens, D=D, D_PAD=d_pad)
        g2 = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B, H)  # noqa: E731
        _attn_heads[g2](qkv, ctx, scale, S, D=D, HD=HD, HD_PAD=hd_pad)
        g3 = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B)  # noqa: E731
        _block_tail[g3](ctx, src, c["w_o"], c["b_o"], c["ln2_w"], c["ln2_b"],
                        c["w_f1"], c["b_f1"], c["w_f2"], c["b_f2"], dst,
                        S, D=D, D_PAD=d_pad, FFN=FFN, FFN_PAD=ffn_pad)
        src = dst
    out = torch.empty_like(src)
    _final_norm[(triton.cdiv(tokens, 128),)](
        src.view(tokens, D), bufs["fn_w"], bufs["fn_b"], out.view(tokens, D),
        tokens, D=D, D_PAD=d_pad, BLOCK_T=128, num_warps=4)
    return out


def build(otb, config):
    _load_cuda()
    # Probe mode: the diagnostic sandbox has its own filesystem, so a swallowed
    # compile error is invisible from outside and reads as a slow CUDA kernel
    # rather than an absent one. k023 cost two runs to that. Raising is the only
    # channel that carries the reason out.
    if not _CUDA_OK and os.environ.get("K024_QUIET") != "1":
        raise RuntimeError(f"k024 CUDA compile failed: {_CUDA_ERROR}")
    model = otb.UserOptimizedTransformer(config)
    fused_ok = (
        config.d_model <= 128 and config.ffn_dim <= 128
        and config.d_model % config.num_heads == 0
        and (config.d_model // config.num_heads) <= 128
        and config.seq_len % 32 == 0 and config.causal
    )

    class CudaTunedTransformer(otb.UserOptimizedTransformer):
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
                state = {"calls": 0, "graph": None, "static_x": None,
                         "static_out": None}
                object.__setattr__(self, "_graph_state", state)
            if state["graph"] is None:
                state["calls"] += 1
                if state["calls"] <= WARMUP_CALLS:
                    # The config sweep runs on the FIRST eager call only, well
                    # outside graph capture -- it synchronises and would be
                    # illegal inside it.
                    return _fused_forward(self, x, tune=(state["calls"] == 1))
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
                state.update(graph=graph, static_x=static_x,
                             static_out=static_out)
                graph.replay()
                return static_out.clone()
            state["static_x"].copy_(x)
            state["graph"].replay()
            return state["static_out"].clone()

    model.__class__ = CudaTunedTransformer
    return model
