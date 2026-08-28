"""k008: W8A8 int8 tensor-core GEMMs for the big-d_model shapes (shape 8).

Shape 8 (d=1024, ffn=1024, 8192 tokens) is GEMM-bound and already near the
fp16-vs-TF32 ceiling (~1.8x). The next rung on sm86 is int8 tensor cores
(~2x the fp16-with-fp32-acc rate). The webinar explicitly blessed internal
quantization ("we only care about the input and output precision") and the
tolerance is abs 2e-3 OR rel 2% — the frozen referee is the arbiter.

Recipe:
- Weights: per-output-channel symmetric int8 (packed once, lazily).
- Activations: per-token dynamic symmetric int8, quantized from the fp32
  LayerNorm outputs (well-conditioned, zero-mean) and from fp32 GELU/context.
- GEMMs: torch._int_mm (cuBLASLt int8, int32 accumulate), dequantized in
  fp32 with (token_scale x channel_scale), bias added in fp32.
- Attention: the authored Triton flash kernel (inlined below, k003 lineage,
  head_dim<=256 via per-D_PAD config pruning) on fp16 q/k/v with fp32
  softmax statistics; residual stream and LayerNorms stay fp32.
- Whole forward CUDA-graph captured (k005 pattern).
- Real (non-all-true) padding masks skip the graph and run the same int8
  blocks eagerly, with baseline-exact key masking before softmax.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _prune_configs(configs, named_args, **kwargs):
    d_pad = named_args.get("D_PAD", kwargs.get("D_PAD", 64))
    fp16 = named_args.get("FP16", kwargs.get("FP16", True))
    ebytes = 2 if fp16 else 4
    keep = [
        c for c in configs
        if (c.kwargs["BLOCK_M"] + 2 * c.kwargs["BLOCK_N"]) * d_pad * ebytes <= 64 * 1024
        and c.kwargs["BLOCK_M"] * d_pad <= 16384
    ]
    return keep or configs[:1]


# ---- Inlined authored kernel (k003 lineage; duplicated deliberately so this
# ---- candidate is SELF-CONTAINED — see k004's provenance note).
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4),
    ],
    key=["SEQ", "D_PAD", "CAUSAL", "FP16"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _attn_fwd(
    Q, K, V, Out,
    stride_qh, stride_qm, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_om, stride_od,
    scale,
    SEQ: tl.constexpr, D: tl.constexpr, D_PAD: tl.constexpr,
    CAUSAL: tl.constexpr, FP16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_PAD)
    d_mask = offs_d < D

    q_ptrs = Q + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < SEQ) & d_mask[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_PAD], dtype=tl.float32)

    if CAUSAL:
        n_end = (pid_m + 1) * BLOCK_M
    else:
        n_end = SEQ

    for n_start in range(0, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        k_ptrs = K + pid_bh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < SEQ) & d_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < SEQ) & d_mask[None, :], other=0.0)

        if FP16:
            qk = tl.dot(q, tl.trans(k)) * scale
        else:
            qk = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        qk = tl.where(offs_n[None, :] < SEQ, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if FP16:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        else:
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    out = acc / l_i[:, None]
    o_ptrs = Out + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out, mask=(offs_m[:, None] < SEQ) & d_mask[None, :])


def triton_attention(q, k, v, scale, causal):
    """q,k,v: [B, H, S, D] contiguous. Returns [B, H, S, D] same dtype."""
    B, H, S, D = q.shape
    d_pad = max(16, triton.next_power_of_2(D))
    out = torch.empty_like(q)
    q4 = q.reshape(B * H, S, D)
    k4 = k.reshape(B * H, S, D)
    v4 = v.reshape(B * H, S, D)
    o4 = out.reshape(B * H, S, D)
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B * H)  # noqa: E731
    _attn_fwd[grid](
        q4, k4, v4, o4,
        q4.stride(0), q4.stride(1), q4.stride(2),
        k4.stride(0), k4.stride(1), k4.stride(2),
        v4.stride(0), v4.stride(1), v4.stride(2),
        o4.stride(0), o4.stride(1), o4.stride(2),
        scale, SEQ=S, D=D, D_PAD=d_pad, CAUSAL=causal,
        FP16=(q.dtype == torch.float16),
    )
    return out


NAME = "k008_int8_gemm"
DESCRIPTION = ("W8A8 int8 tensor-core projections/FFN (per-token dynamic + "
               "per-channel weights, fp32 dequant) + authored fp16 Triton "
               "attention, CUDA-graphed. Targets big-d_model shapes.")

WARMUP_CALLS = 3


def _quant_rows(t):
    """Per-row symmetric int8: t [N, K] fp32 -> (int8, fp32 scale [N, 1])."""
    s = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
    return torch.round(t / s).to(torch.int8), s


def _int8_pack(linear):
    w = linear.weight.detach().float()
    s = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    w_i8 = torch.round(w / s[:, None]).to(torch.int8)
    return w_i8.t().contiguous(), s.contiguous(), linear.bias.detach().float().contiguous()


def _int8_cache(block):
    cache = getattr(block, "_int8_cache", None)
    if cache is None:
        attn = block.attention
        qw, qs, qb = _int8_pack(attn.q_proj)
        kw, ks, kb = _int8_pack(attn.k_proj)
        vw, vs, vb = _int8_pack(attn.v_proj)
        cache = {
            "qkv_w": torch.cat([qw, kw, vw], dim=1).contiguous(),
            "qkv_s": torch.cat([qs, ks, vs]).contiguous(),
            "qkv_b": torch.cat([qb, kb, vb]).contiguous(),
        }
        for name, lin in (("out", attn.out_proj), ("f1", block.ffn_in),
                          ("f2", block.ffn_out)):
            w, s, b = _int8_pack(lin)
            cache[f"{name}_w"], cache[f"{name}_s"], cache[f"{name}_b"] = w, s, b
        object.__setattr__(block, "_int8_cache", cache)
    return cache


def _int8_linear(h32, w, s_w, b):
    """h32 [N, K] fp32 -> fp32 [N, out] via int8 GEMM with fp32 dequant."""
    a8, s_a = _quant_rows(h32)
    y = torch._int_mm(a8, w).to(torch.float32)
    return y * (s_a * s_w[None, :]) + b[None, :]


def _make_block_forward(otb):
    def block_forward(self, x, valid_token_mask, causal):
        attn = self.attention
        c = _int8_cache(self)
        B, S, d = x.shape
        tokens = B * S

        h = self.norm1(x)                                   # fp32
        qkv = _int8_linear(h.view(tokens, d), c["qkv_w"], c["qkv_s"], c["qkv_b"])
        qkv16 = qkv.half().view(B, S, 3 * d)
        q, k, v = qkv16.split(d, dim=-1)
        q = q.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()

        # A real (padded) mask only arrives via the eager fallback; it must
        # mask invalid keys before softmax like the baseline. The Triton
        # kernel has no key-mask support, so any real mask takes matmul.
        if valid_token_mask is None and attn.head_dim <= 256 and S % 32 == 0:
            context = triton_attention(q, k, v, attn.scale, causal)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
            if causal:
                cm = torch.ones((S, S), device=x.device,
                                dtype=torch.bool).triu(diagonal=1)
                scores = scores.masked_fill(cm, float("-inf"))
            if valid_token_mask is not None:
                scores = scores.masked_fill(
                    ~valid_token_mask[:, None, None, :], float("-inf"))
            context = torch.matmul(
                torch.softmax(scores.float(), dim=-1).half(), v)
        context = context.transpose(1, 2).contiguous().view(tokens, d)

        attn_out = _int8_linear(context.float(), c["out_w"], c["out_s"], c["out_b"])
        attn_out = attn_out.view(B, S, d)
        if valid_token_mask is not None:
            attn_out = attn_out.masked_fill(~valid_token_mask[..., None], 0)
        x = x + attn_out                                    # residual fp32

        h2 = self.norm2(x)                                  # fp32
        hid = _int8_linear(h2.view(tokens, d), c["f1_w"], c["f1_s"], c["f1_b"])
        hid = F.gelu(hid, approximate="none")
        out = _int8_linear(hid, c["f2_w"], c["f2_s"], c["f2_b"]).view(B, S, d)
        x = x + out                                         # residual fp32
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
    return block_forward


def build(otb, config):
    model = otb.UserOptimizedTransformer(config)
    block_forward = _make_block_forward(otb)

    class Int8Block(otb.BaselineTransformerBlock):
        forward = block_forward

    for layer in model.layers:
        layer.__class__ = Int8Block

    class GraphedInt8Transformer(otb.UserOptimizedTransformer):
        def _eager(self, x, valid_token_mask):
            return otb.BaselineTransformer.forward(self, x, valid_token_mask)

        def forward(self, x, valid_token_mask=None):
            if valid_token_mask is not None and not bool(valid_token_mask.all()):
                return self._eager(x, valid_token_mask)
            state = getattr(self, "_graph_state", None)
            if state is None:
                state = {"calls": 0, "graph": None, "static_x": None, "static_out": None}
                object.__setattr__(self, "_graph_state", state)
            if state["graph"] is None:
                state["calls"] += 1
                if state["calls"] <= WARMUP_CALLS:
                    return self._eager(x, None)
                static_x = x.clone()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        self._eager(static_x, None)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    static_out = self._eager(static_x, None)
                state.update(graph=graph, static_x=static_x, static_out=static_out)
                graph.replay()
                return static_out.clone()
            state["static_x"].copy_(x)
            state["graph"].replay()
            return state["static_out"].clone()

    model.__class__ = GraphedInt8Transformer
    return model
