"""k016: k009's fused-block megakernel with an autotune space widened for the
narrow-head shapes (HD=8: shapes 7 and 11).

IDENTICAL MATH to k009 — not one arithmetic line differs. The only change is
the K2 (`_attn_block_tail`) autotune config list.

Why. Measured 31 Aug on the shipped route, shapes 1/9/10/11 are the same 7.52
GFLOP problem at 1/2/4/16 heads. Our candidate is flat within 3.5% from 1 to 4
heads (0.5601 / 0.5509 / 0.5704 ms) and then jumps to 0.9175 ms at 16 heads —
+63.7% on identical arithmetic. A torch-profiler diagnostic on identical bytes
localised the whole penalty to K2 (2.13x; K1, final-norm and the DtoD copies
are all flat within 4%), and K2's achieved throughput falls from 14.5 TF/s at
HD=32 to 8.5 TF/s at HD=8.

That penalty has two components:
  1. ARITHMETIC, and irreducible here: `hd_pad = max(16, next_power_of_2(HD))`
     is 16 for HD=8 because tl.dot needs a 16-minimum tile, so both attention
     matmuls run at half useful width. Padding raises K2's work by ~25%.
  2. EFFICIENCY, and the target of this file: K2's time rises +113%, far more
     than the +25% of extra work. The residue is tiling and occupancy — at
     H=16 the kernel runs 16 sequential head iterations, each holding an
     attn[BLOCK_M, D_PAD] fp32 accumulator plus an FFN hidden tile, so large
     BLOCK_M starves occupancy exactly where the head loop is longest.

k009's config list was written for the d_model-128 four-head shapes and gives
the tuner nothing below 2 warps, nothing above BLOCK_N=64, and no num_stages
choice at all. This file adds those. Autotune keys on (SEQ, D_PAD, H, HD_PAD),
so every other shape re-tunes over a superset and can only match or beat its
k009 configuration; the risk is longer warmup, not a worse pick.

Original k009/k007 header follows.

k009: k007's fused-block megakernel with a widened autotune space.
k007: fused-block megakernel for small d_model (<=128) + CUDA graph.

  K1  norm1 + packed QKV projection      (one pass over the tokens)
  K2  flash attention (all heads) + out-projection + residual + norm2 +
      FFN (GELU exact-erf) + residual    (one pass over query tiles)

plus one tiny final-LayerNorm kernel. Activations stay in registers between
the fused stages instead of round-tripping through HBM ~10 times per layer.
Precision recipe proven in k005: fp32 residual stream and LayerNorm
statistics, fp16 tensor-core dots with fp32 accumulation, exact-erf GELU in
fp32. Whole forward is CUDA-graph captured. Padded (real-mask) inputs and
oversized dims fall back to the eager baseline path, key-masked like the
official reference.

Constraints for the fused path: d_model <= 128, ffn_dim <= 128,
head_dim <= 128, seq_len % 32 == 0, no (or all-true) valid_token_mask.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NAME = "k016_hd8_tuned"
DESCRIPTION = ("k009 fused-block megakernel; K2 autotune space widened for the "
               "narrow-head (HD=8) shapes. Identical math to k009.")

WARMUP_CALLS = 3
LN_EPS = tl.constexpr(1e-5)


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
    """LayerNorm(x) @ W_qkv^T + b -> Out fp16 [tokens, 3*D]."""
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
        # w rows are output features, cols input features: Out = y @ w^T
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
        # --- k009's original eight, kept verbatim so no shape can regress ---
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
        # --- added: small tiles, for the long (H=16) head loop -------------
        # attn[BLOCK_M, 128] fp32 + the FFN hidden tile dominate registers;
        # at BLOCK_M=16 that accumulator is 4x smaller than at 64, which is
        # what should let more CTAs stay resident while the loop is longest.
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=8),
        # --- added: wider N, to amortise the K/V loop over fewer trips -----
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8),
        # --- added: explicit pipelining depth, absent from k009 entirely ---
        # 16 head iterations x an inner K/V loop is a deep nest; the default
        # stage count was never compared against anything here.
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=2),
    ],
    key=["SEQ", "D_PAD", "H", "HD_PAD"],
)
@triton.jit
def _attn_block_tail(
    QKV, X, Wo, Bo, Ln2W, Ln2B, Wf1, Bf1, Wf2, Bf2, XOut,
    scale,
    SEQ, B,
    D: tl.constexpr, D_PAD: tl.constexpr,
    H: tl.constexpr, HD: tl.constexpr, HD_PAD: tl.constexpr,
    FFN: tl.constexpr, FFN_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Per (batch, q-tile): causal flash attention over all H heads, then
    out-proj + residual + norm2 + FFN(erf-GELU) + residual, all in-register."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_PAD)
    offs_hd = tl.arange(0, HD_PAD)
    m_mask = offs_m < SEQ
    d_mask = offs_d < D
    hd_mask = offs_hd < HD

    qkv_base = QKV + pid_b * SEQ * (3 * D)
    # Fold the out-projection into the head loop:
    #   attn = sum_h head_out_h @ Wo[:, h*HD:(h+1)*HD]^T
    # so per-head results never need to be scattered into a concatenated
    # context — each head contributes a rank-HD update to the output tile.
    attn = tl.zeros([BLOCK_M, D_PAD], dtype=tl.float32)

    for h in tl.static_range(H):
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
        # Wo is [D_out, D_in]; this head's input slice is columns h*HD..h*HD+HD
        wo_h = tl.load(Wo + offs_d[:, None] * D + h * HD + offs_hd[None, :],
                       mask=d_mask[:, None] & hd_mask[None, :], other=0.0)
        attn += tl.dot(head_out, tl.trans(wo_h))

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
    wf1 = tl.load(Wf1 + offs_f[:, None] * D + tl.arange(0, D_PAD)[None, :],
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
    """fp16 weight cache per block: packed QKV, out, ffn (+ fp32 norm params)."""
    cache = getattr(layer, "_k007_cache", None)
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
        object.__setattr__(layer, "_k007_cache", cache)
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

    bufs = getattr(model, "_k007_buf", None)
    if bufs is None or bufs["a"].shape != x.shape:
        bufs = {
            "qkv": torch.empty(tokens, 3 * D, dtype=torch.float16, device=x.device),
            "a": torch.empty_like(x),
            "b": torch.empty_like(x),
            "fn_w": model.final_norm.weight.float().contiguous(),
            "fn_b": model.final_norm.bias.float().contiguous(),
        }
        object.__setattr__(model, "_k007_buf", bufs)
    qkv = bufs["qkv"]

    src = x.contiguous()
    for i, layer in enumerate(model.layers):
        dst = bufs["a"] if i % 2 == 0 else bufs["b"]
        c = _pack_layer(layer)
        grid1 = lambda meta: (triton.cdiv(tokens, meta["BLOCK_T"]),)  # noqa: E731
        _norm_qkv[grid1](
            src.view(tokens, D), c["w_qkv"], c["b_qkv"], c["ln1_w"], c["ln1_b"],
            qkv, tokens, D=D, D_PAD=d_pad,
        )
        grid2 = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B)  # noqa: E731
        _attn_block_tail[grid2](
            qkv, src, c["w_o"], c["b_o"], c["ln2_w"], c["ln2_b"],
            c["w_f1"], c["b_f1"], c["w_f2"], c["b_f2"], dst,
            scale, S, B,
            D=D, D_PAD=d_pad, H=H, HD=HD, HD_PAD=hd_pad,
            FFN=FFN, FFN_PAD=ffn_pad,
        )
        src = dst
    out = torch.empty_like(src)
    _final_norm[(triton.cdiv(tokens, 128),)](
        src.view(tokens, D), bufs["fn_w"], bufs["fn_b"], out.view(tokens, D),
        tokens, D=D, D_PAD=d_pad, BLOCK_T=128, num_warps=4,
    )
    return out


def build(otb, config):
    model = otb.UserOptimizedTransformer(config)
    fused_ok = (
        config.d_model <= 128 and config.ffn_dim <= 128
        and config.d_model % config.num_heads == 0
        and (config.d_model // config.num_heads) <= 128
        and config.seq_len % 32 == 0 and config.causal
    )

    class FusedBlockTransformer(otb.UserOptimizedTransformer):
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

    model.__class__ = FusedBlockTransformer
    return model
