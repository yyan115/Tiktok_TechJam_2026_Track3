"""k003: PROJECT-AUTHORED Triton attention kernel + the k002 fused QKV.

The baseline materializes the full [B, H, S, S] score matrix through five
separate ops (matmul, scale+mask fills, fp32 softmax, cast, matmul). This
kernel computes attention in ONE authored Triton kernel using the online-
softmax (flash-style) algorithm: scores are produced tile-by-tile in on-chip
memory and never written to DRAM, softmax runs in fp32 accumulators (matching
the baseline's explicit fp32-softmax semantics), and causal masking happens
in-register. Written from scratch for this benchmark's shape families:
head_dim in {8, 16, 32, 64} (padded to tl.dot's minimum where needed) and
seq_len divisible by 32 (covers shapes 1-7, 10-13). Larger head dims fall
back to the authored k002 path.

Authorship: the Triton kernel below is written for this project (algorithmic
lineage: Milakov & Gimelshein's online softmax and Dao et al.'s FlashAttention,
cited as literature, no code reused). QKV fusion carried over from k002.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

NAME = "k003_triton_attention"
DESCRIPTION = ("Authored Triton flash-style causal attention (online softmax, fp32 "
               "accum) + fused QKV; eager fallback for unsupported configs.")


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
                # Authored eager fallback — identical math to the baseline.
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


def build(otb, config):
    model = otb.UserOptimizedTransformer(config)
    cls = _make_attention_class(otb)
    for layer in model.layers:
        layer.attention.__class__ = cls
    return model
