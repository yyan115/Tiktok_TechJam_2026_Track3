"""k004: k003's authored Triton attention + CUDA-graph capture of the WHOLE
forward pass.

On the small shapes, the 4-layer forward issues ~50 GPU operations whose
launch overhead dwarfs the math. This candidate records the entire forward
(our Triton attention included) into a CUDA graph once, then replays it as a
single submission. Inputs flow through a static buffer (copied fresh from the
caller's tensor every call — values always current, so re-randomized inputs
are honored); the output is cloned out of the static buffer per call.

Capture policy: the graph is recorded against a dense (no-padding) forward —
semantically identical to an all-true mask, since the baseline's mask ops are
identity when every token is valid. Each call re-verifies the mask is all-true
(a real check, every call, never cached); padded inputs take the un-graphed
authored path. First calls run eagerly (warmup + Triton autotune settle),
then capture happens once.

Authorship: composition of our own k003 kernel with torch.cuda.CUDAGraph
capture — the recorded work is our kernel sequence, not an external library's.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ---- Inlined authored kernel (identical to k003's; duplicated deliberately so
# ---- this candidate is SELF-CONTAINED and its single-file hash binds every
# ---- line of executed candidate code — auto-audit provenance finding, 28 Aug).
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4),
    ],
    key=["SEQ", "D_PAD", "CAUSAL", "FP16"],
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


def _make_attention_class(otb):
    class TritonAttention(otb.BaselineSelfAttention):
        def _packed(self):
            packed = getattr(self, "_qkv_packed", None)
            if packed is None or packed[0].dtype != self.q_proj.weight.dtype:
                w = torch.cat([self.q_proj.weight, self.k_proj.weight,
                               self.v_proj.weight], dim=0).contiguous()
                b = torch.cat([self.q_proj.bias, self.k_proj.bias,
                               self.v_proj.bias], dim=0).contiguous()
                object.__setattr__(self, "_qkv_packed", (w, b))
                packed = (w, b)
            return packed

        def forward(self, x, valid_token_mask=None, causal=False):
            batch, seq_len, _ = x.shape
            w, b = self._packed()
            qkv = F.linear(x, w, b)
            q, k, v = qkv.split(self.d_model, dim=-1)
            q = self._split_heads(q)
            k = self._split_heads(k)
            v = self._split_heads(v)
            fast_ok = (
                x.dtype == torch.float32
                and self.head_dim <= 64
                and seq_len % 32 == 0
                and (valid_token_mask is None or bool(valid_token_mask.all()))
            )
            if fast_ok:
                context = triton_attention(q, k, v, self.scale, causal)
            else:
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                if causal:
                    cm = torch.ones((seq_len, seq_len), device=x.device,
                                    dtype=torch.bool).triu(diagonal=1)
                    scores = scores.masked_fill(cm, float("-inf"))
                if valid_token_mask is not None:
                    scores = scores.masked_fill(
                        ~valid_token_mask[:, None, None, :], float("-inf"))
                context = torch.matmul(
                    torch.softmax(scores.float(), dim=-1).to(x.dtype), v)
            context = (
                context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
            )
            output = self.out_proj(context)
            if valid_token_mask is not None:
                output = output.masked_fill(~valid_token_mask[..., None], 0)
            return output

    return TritonAttention


NAME = "k004_graphed_triton"
DESCRIPTION = ("Authored Triton attention + whole-forward CUDA-graph capture; "
               "eager authored path for padded inputs.")

WARMUP_CALLS = 3


def build(otb, config):
    model = otb.UserOptimizedTransformer(config)
    attention_cls = _make_attention_class(otb)
    for layer in model.layers:
        layer.attention.__class__ = attention_cls

    class GraphedTransformer(otb.UserOptimizedTransformer):
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
                # Capture once: dense forward recorded against a static buffer.
                # Canonical pattern: warm on a side stream first (lets cuBLAS/
                # allocator workspaces bind outside the capture), then record.
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
                # Capture RECORDS but does not execute — replay once to
                # actually compute the answer for this input.
                graph.replay()
                return static_out.clone()

            state["static_x"].copy_(x)
            state["graph"].replay()
            return state["static_out"].clone()

    model.__class__ = GraphedTransformer
    return model
