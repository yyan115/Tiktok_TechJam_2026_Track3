"""k006: k005's graphed fp16 stack with the Triton attention extended to
head_dim 128 and 256 (shapes 9 and 8, where k005 falls back to graphed eager).

Same design as k005 — fp32 boundary/norms/residuals, fp16 tensor-core GEMMs,
fp32 accumulation, whole-forward CUDA graph. The only change is the attention
kernel's reach: block configs are pruned per D_PAD so shared memory
((BM + 2*BN) * D_PAD * 2 bytes) and the fp32 accumulator (BM * D_PAD) always
fit the RTX 3060 Ti (sm86, ~99KB smem/block) — oversized configs are dropped
before compilation instead of crashing (LESSONS.md #2).
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
            # fp16 inputs: tensor cores with native fp32 accumulation.
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
    """q,k,v: [B, H, S, D] contiguous fp32. Returns [B, H, S, D] fp32."""
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


NAME = "k006_fp16_hd128"
DESCRIPTION = ("k005's graphed fp16 stack with the authored Triton attention "
               "extended to head_dim 128/256 (per-D_PAD config pruning).")

WARMUP_CALLS = 3


def _half_params(module):
    cache = getattr(module, "_fp16_cache", None)
    if cache is None:
        cache = {}
        for name in ("q_proj", "k_proj", "v_proj", "out_proj", "ffn_in", "ffn_out"):
            sub = getattr(module, name, None)
            if sub is not None:
                cache[name] = (sub.weight.half().contiguous(),
                               sub.bias.half().contiguous())
        # packed QKV in fp16
        if "q_proj" in cache:
            cache["qkv"] = (
                torch.cat([cache["q_proj"][0], cache["k_proj"][0],
                           cache["v_proj"][0]], dim=0).contiguous(),
                torch.cat([cache["q_proj"][1], cache["k_proj"][1],
                           cache["v_proj"][1]], dim=0).contiguous(),
            )
        object.__setattr__(module, "_fp16_cache", cache)
    return cache


def _make_block_forward(otb):
    def block_forward(self, x, valid_token_mask, causal):
        # x arrives fp32 (residual stream). Norms fp32; heavy math fp16.
        attn = self.attention
        cache = _half_params(attn)          # q/k/v/out_proj (+ packed qkv)
        ffn_cache = _half_params(self)      # ffn_in / ffn_out live on the block
        h = self.norm1(x)                                  # fp32
        h16 = h.half()
        qkv = F.linear(h16, cache["qkv"][0], cache["qkv"][1])
        d = attn.d_model
        q, k, v = qkv.split(d, dim=-1)
        batch, seq_len, _ = h.shape
        q = q.view(batch, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
        k = k.view(batch, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()

        # The graphed path always passes valid_token_mask=None; a real (padded)
        # mask only arrives via the eager fallback and must mask invalid keys
        # before softmax, exactly like the baseline. The Triton kernel has no
        # key-mask support, so any real mask takes the matmul path.
        if valid_token_mask is None and attn.head_dim <= 256 and seq_len % 32 == 0:
            context = triton_attention(q, k, v, attn.scale, causal)  # fp16 in/out, fp32 accum
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
            if causal:
                cm = torch.ones((seq_len, seq_len), device=x.device,
                                dtype=torch.bool).triu(diagonal=1)
                scores = scores.masked_fill(cm, float("-inf"))
            if valid_token_mask is not None:
                scores = scores.masked_fill(
                    ~valid_token_mask[:, None, None, :], float("-inf"))
            context = torch.matmul(
                torch.softmax(scores.float(), dim=-1).half(), v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d)
        attn_out = F.linear(context, cache["out_proj"][0], cache["out_proj"][1])
        if valid_token_mask is not None:
            attn_out = attn_out.masked_fill(~valid_token_mask[..., None], 0)
        x = x + attn_out.float()                           # residual fp32

        h = self.norm2(x)                                  # fp32
        h16 = h.half()
        hidden = F.linear(h16, ffn_cache["ffn_in"][0], ffn_cache["ffn_in"][1])
        hidden = F.gelu(hidden.float(), approximate="none").half()
        ffn_out = F.linear(hidden, ffn_cache["ffn_out"][0], ffn_cache["ffn_out"][1])
        x = x + ffn_out.float()                            # residual fp32
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
    return block_forward


def build(otb, config):
    model = otb.UserOptimizedTransformer(config)
    block_forward = _make_block_forward(otb)

    class FP16Block(otb.BaselineTransformerBlock):
        forward = block_forward

    for layer in model.layers:
        # attention keeps its fp32 params for weight-copy compatibility; the
        # block forward routes through the fp16 caches.
        layer.__class__ = FP16Block

    class GraphedFP16Transformer(otb.UserOptimizedTransformer):
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

    model.__class__ = GraphedFP16Transformer
    return model
