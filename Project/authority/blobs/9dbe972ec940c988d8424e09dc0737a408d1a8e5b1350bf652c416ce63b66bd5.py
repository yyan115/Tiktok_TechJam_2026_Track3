"""k020: stop streaming attention over sequences that already fit. Beam member 2 of 2.

THE DEFECT. k017 runs FlashAttention-style STREAMING attention on every shape.
Streaming exists to handle sequences whose score matrix does not fit on chip.
Eleven of the fourteen official shapes have seq_len 128 or 32. Their score rows
fit. We are paying the streaming tax for nothing, twice over:

1. REDUNDANT K/V TRAFFIC. Each q-tile re-reads every K/V block it needs from
   global memory. Under causal masking with S=128 and BLOCK_M=32, the four
   q-tiles read 1 + 2 + 3 + 4 = 10 K/V blocks against a floor of 4.
   That is 2.5x more K/V traffic than the problem requires.

2. THE ONLINE-SOFTMAX RESCALE. Every inner iteration recomputes a running max,
   rescales the whole fp32 accumulator by `acc * alpha[:, None]`, and rescales
   the running denominator. For a [BLOCK_M, HD_PAD] accumulator that is a full
   tile multiply per iteration, purely to maintain numerical state that a
   single-pass softmax over a resident row does not need at all.

The shape-1 profile puts _sub_attn_heads at 17.8% of device time and roughly
34% roofline efficiency -- the least efficient of the three kernels.

THE CHANGE. One CTA per (batch, head). K and V for the whole sequence are
loaded ONCE into the program and stay resident; the kernel then loops its own
q-tiles internally. Because every q-tile sees the complete key range at once,
the softmax is an ordinary max/exp/sum over the row -- no running statistics,
no accumulator rescale, no `l_i`/`m_i` bookkeeping.

    grid (B, H)                      instead of (q_tiles, B, H)
    K, V loaded once per (b, h)      instead of once per (q_tile, b, h)
    single-pass softmax              instead of online rescaling

WHEN IT APPLIES, and why the gate is what it is. The resident path is selected
only when all three hold:

  * S <= 128          -- the resident K/V tile must be small enough to hold.
  * HD <= 64          -- with S_PAD 128 and HD_PAD 64, K and V are 16 KB each.
  * B * H >= 38       -- THIS IS THE IMPORTANT ONE. Collapsing the q-tile grid
                         dimension DIVIDES the CTA count by the number of
                         q-tiles. Shape 2 (B=1, H=4) would fall from 32 CTAs to
                         4 on a 38-SM card, which is strictly worse. The gate
                         requires enough (batch x head) pairs to fill the
                         machine before the trade is taken.

Shapes taking the resident path: 1, 4, 5, 7, 10, 11, 12.
Shapes keeping the streaming path: 2 and 3 (too few CTAs), 9 (HD 128),
13 (S 1024) -- and for shape 13 streaming is the correct algorithm, which is
the whole point: this variant does not replace flash attention, it stops
applying it where it was never needed.

Both kernels are present and the selection is a host-side branch on shape, so
the streaming path is byte-identical to k017 and shapes that keep it are
expected to measure unchanged. That is the built-in control.

Math is unchanged: same causal mask, same fp32 softmax accumulation, same fp16
dots with fp32 accumulate, same fp32 residual stream and LayerNorm statistics.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NAME = "k020_resident_attn"
DESCRIPTION = ("k017 with a sequence-resident attention kernel (one CTA per "
               "batch-head, K/V loaded once, single-pass softmax) selected "
               "when the sequence fits and the grid still fills the SMs.")

WARMUP_CALLS = 3
LN_EPS = tl.constexpr(1e-5)
SM_COUNT = 38  # RTX 3060 Ti


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
        triton.Config({"BLOCK_M": 16}, num_warps=2),
        triton.Config({"BLOCK_M": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 32}, num_warps=2),
        triton.Config({"BLOCK_M": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64}, num_warps=8),
    ],
    key=["S_PAD", "HD_PAD"],
)
@triton.jit
def _attn_resident(
    QKV, Ctx,
    scale,
    SEQ,
    D: tl.constexpr,
    HD: tl.constexpr, HD_PAD: tl.constexpr,
    S_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """One program per (batch, head). K and V resident; q-tiles looped inside.

    The whole key range is visible to every q-tile, so the softmax is a plain
    row-wise max/exp/sum. There is no running maximum, no running denominator
    and no accumulator rescale -- the three things the streaming kernel pays
    on every one of its inner iterations.
    """
    pid_b = tl.program_id(0)
    h = tl.program_id(1)

    offs_s = tl.arange(0, S_PAD)
    offs_hd = tl.arange(0, HD_PAD)
    s_mask = offs_s < SEQ
    hd_mask = offs_hd < HD

    qkv_base = QKV + pid_b * SEQ * (3 * D)

    # Load K and V for the entire sequence exactly once.
    k = tl.load(qkv_base + offs_s[:, None] * (3 * D) + D + h * HD + offs_hd[None, :],
                mask=s_mask[:, None] & hd_mask[None, :], other=0.0)
    v = tl.load(qkv_base + offs_s[:, None] * (3 * D) + 2 * D + h * HD + offs_hd[None, :],
                mask=s_mask[:, None] & hd_mask[None, :], other=0.0)
    kt = tl.trans(k)

    for m_start in range(0, SEQ, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < SEQ
        q = tl.load(qkv_base + offs_m[:, None] * (3 * D) + h * HD + offs_hd[None, :],
                    mask=m_mask[:, None] & hd_mask[None, :], other=0.0)

        qk = tl.dot(q, kt) * scale
        qk = tl.where(offs_m[:, None] >= offs_s[None, :], qk, float("-inf"))
        qk = tl.where(s_mask[None, :], qk, float("-inf"))

        # Single-pass softmax over the complete row.
        m_row = tl.max(qk, axis=1)
        p = tl.exp(qk - m_row[:, None])
        denom = tl.sum(p, axis=1)
        out = tl.dot(p.to(tl.float16), v) / denom[:, None]

        tl.store(Ctx + pid_b * SEQ * D + offs_m[:, None] * D + h * HD + offs_hd[None, :],
                 out.to(tl.float16),
                 mask=m_mask[:, None] & hd_mask[None, :])


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
    """Streaming path, byte-identical to k017. The control arm of this variant."""
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
    cache = getattr(layer, "_k020_cache", None)
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
        object.__setattr__(layer, "_k020_cache", cache)
    return cache


def _resident_ok(B, H, S, HD):
    """Take the resident path only where it is arithmetically the better trade.

    Collapsing the q-tile grid dimension divides the CTA count by the number of
    q-tiles, so it is only worth doing when (batch x head) alone already fills
    the machine.
    """
    return S <= 128 and HD <= 64 and (B * H) >= SM_COUNT


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
    s_pad = max(16, triton.next_power_of_2(S))
    scale = HD ** -0.5
    resident = _resident_ok(B, H, S, HD)

    bufs = getattr(model, "_k020_buf", None)
    if bufs is None or bufs["a"].shape != x.shape:
        bufs = {
            "qkv": torch.empty(tokens, 3 * D, dtype=torch.float16, device=x.device),
            "ctx": torch.empty(tokens, D, dtype=torch.float16, device=x.device),
            "a": torch.empty_like(x),
            "b": torch.empty_like(x),
            "fn_w": model.final_norm.weight.float().contiguous(),
            "fn_b": model.final_norm.bias.float().contiguous(),
        }
        object.__setattr__(model, "_k020_buf", bufs)
    qkv = bufs["qkv"]
    ctx = bufs["ctx"]

    src = x.contiguous()
    for i, layer in enumerate(model.layers):
        dst = bufs["a"] if i % 2 == 0 else bufs["b"]
        c = _pack_layer(layer)
        grid1 = lambda meta: (triton.cdiv(tokens, meta["BLOCK_T"]),)  # noqa: E731
        _norm_qkv[grid1](
            src.view(tokens, D), c["w_qkv"], c["b_qkv"], c["ln1_w"], c["ln1_b"],
            qkv, tokens, D=D, D_PAD=d_pad,
        )
        if resident:
            _attn_resident[(B, H)](
                qkv, ctx, scale, S,
                D=D, HD=HD, HD_PAD=hd_pad, S_PAD=s_pad,
            )
        else:
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
    model = otb.UserOptimizedTransformer(config)
    fused_ok = (
        config.d_model <= 128 and config.ffn_dim <= 128
        and config.d_model % config.num_heads == 0
        and (config.d_model // config.num_heads) <= 128
        and config.seq_len % 32 == 0 and config.causal
    )

    class ResidentAttnTransformer(otb.UserOptimizedTransformer):
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

    model.__class__ = ResidentAttnTransformer
    return model
