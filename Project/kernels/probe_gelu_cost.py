"""PROBE, NOT A CANDIDATE. Sizes the exact-erf GELU inside the block tail.

*** THIS FILE COMPUTES WRONG NUMBERS ON PURPOSE. ***

It replaces the erf-GELU with the identity. The output is numerically incorrect
and this file must NEVER be run in the screening lane, never measured for
speedup, and never integrated. Its only legitimate use is a diagnostic profile,
where correctness is not checked and only per-kernel device time is read.

That "never measured for speedup" line is not boilerplate. Earlier today a
config sweep inside k024 selected a CUDA kernel that computed only half its
output because that was fastest, and it read as a 2x win until the template
parameters were read by hand (LESSONS 56). A file that is deliberately wrong is
exactly the thing that must not leak into a lane that rewards speed.

WHY IT EXISTS. `_sub_attn_block_tail` sits at ~52% of roof and two
architecturally different candidates failed to move it:

    k025  FFN hidden walked in halves (cuts register liveness)   48.4 us
    k026  persistent stride loop (enables Triton's pipeliner)    48.4 us
    shipped                                                      46.5-47.2 us

So neither of the defects that fixed `_sub_norm_qkv` applies here, and guessing
a third is how attempts get wasted. The one component of this kernel whose cost
has never been quantified is the GELU: FLOP accounting does not count erf at
all, but the kernel evaluates it 1.05M times per layer on shape 1 (8192 tokens
x 128 hidden), on the special-function units rather than the tensor cores.

WHAT THE ANSWER MEANS. Let T be the block-tail time this probe reports against
the ~46.5 us baseline:

  - large drop (say below 40 us): erf is a real share of the kernel, and there
    is a SAFE way to attack it. `hid` is cast to fp16 immediately after the
    GELU for the following dot, so the GELU result only ever needs to be
    fp16-accurate (~1e-3 relative). Any erf approximation better than that is
    exact in effect, because the extra precision is discarded on the very next
    instruction. That is a legitimate optimisation, not a tolerance trade.
  - small drop: the GELU is not the story, `_sub_attn_block_tail` is close to
    what this structure can do, and the direction closes with three
    architecturally distinct candidates measured rather than assumed.

Either answer is worth one free diagnostic. Neither is worth an attempt.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NAME = "probe_gelu_cost"
DESCRIPTION = ("PROBE ONLY, numerically incorrect: erf-GELU replaced by the "
               "identity to size its cost in the block tail. Never promote, "
               "never screen, diagnostic lane only.")

WARMUP_CALLS = 3
LN_EPS = tl.constexpr(1e-5)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 32}, num_warps=4),
        triton.Config({"BLOCK_T": 64}, num_warps=4),
        triton.Config({"BLOCK_T": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 128}, num_warps=8),
    ],
    key=["D_PAD", "TOKENS", "CHUNKS"],
)
@triton.jit
def _norm_qkv(
    X, W, Bias, LnW, LnB, Out,
    TOKENS, D: tl.constexpr, D_PAD: tl.constexpr,
    CHUNKS: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk0 = tl.program_id(1)
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
    for step in tl.static_range(3 // CHUNKS):
        chunk = chunk0 + step * CHUNKS
        w = tl.load(W + (chunk * D + offs_d[:, None]) * D + offs_d[None, :],
                    mask=d_mask[:, None] & d_mask[None, :], other=0.0)
        acc = tl.dot(y16, tl.trans(w))
        b = tl.load(Bias + chunk * D + offs_d, mask=d_mask, other=0.0)
        acc = acc + b[None, :].to(tl.float32)
        tl.store(Out + offs_t[:, None] * (3 * D) + chunk * D + offs_d[None, :],
                 acc.to(tl.float16), mask=t_mask[:, None] & d_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=2),
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
    """Shipped block tail with ONE line changed: the GELU is the identity.

    Wrong on purpose. Diagnostic lane only.
    """
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
    # ---- THE PROBE. Shipped line is:
    #   hid = 0.5 * hid * (1.0 + tl.math.erf(hid * 0.7071067811865476))
    # Replaced by the identity so the erf cost shows as a device-time delta.
    # WRONG OUTPUT BY CONSTRUCTION.
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
    cache = getattr(layer, "_probe_cache", None)
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
        object.__setattr__(layer, "_probe_cache", cache)
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

    bufs = getattr(model, "_probe_buf", None)
    if bufs is None or bufs["a"].shape != x.shape:
        bufs = {
            "qkv": torch.empty(tokens, 3 * D, dtype=torch.float16, device=x.device),
            "ctx": torch.empty(tokens, D, dtype=torch.float16, device=x.device),
            "a": torch.empty_like(x),
            "b": torch.empty_like(x),
            "fn_w": model.final_norm.weight.float().contiguous(),
            "fn_b": model.final_norm.bias.float().contiguous(),
        }
        object.__setattr__(model, "_probe_buf", bufs)
    qkv = bufs["qkv"]
    ctx = bufs["ctx"]

    src = x.contiguous()
    for i, layer in enumerate(model.layers):
        dst = bufs["a"] if i % 2 == 0 else bufs["b"]
        c = _pack_layer(layer)
        qkv_chunks = 3 if (D >= 64 and tokens <= 4096) else 1
        g1 = lambda meta: (triton.cdiv(tokens, meta["BLOCK_T"]), qkv_chunks)  # noqa: E731
        _norm_qkv[g1](src.view(tokens, D), c["w_qkv"], c["b_qkv"],
                      c["ln1_w"], c["ln1_b"], qkv, tokens,
                      D=D, D_PAD=d_pad, CHUNKS=qkv_chunks)
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
    model = otb.UserOptimizedTransformer(config)
    fused_ok = (
        config.d_model <= 128 and config.ffn_dim <= 128
        and config.d_model % config.num_heads == 0
        and (config.d_model // config.num_heads) <= 128
        and config.seq_len % 32 == 0 and config.causal
    )

    class GeluProbeTransformer(otb.UserOptimizedTransformer):
        # Flag so anything reading this object can see it is not a candidate.
        numerically_incorrect_probe = True

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
                state.update(graph=graph, static_x=static_x,
                             static_out=static_out)
                graph.replay()
                return static_out.clone()
            state["static_x"].copy_(x)
            state["graph"].replay()
            return state["static_out"]

    model.__class__ = GeluProbeTransformer
    return model
