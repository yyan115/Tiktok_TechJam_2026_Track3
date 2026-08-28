#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every output element:
    abs(user - ref) <= atol
    OR
    abs(user - ref) <= rtol * abs(ref)

The default thresholds are atol=0.001 and rtol=0.01 (1%).
"""

from __future__ import annotations

import argparse
import copy
import math
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class BaselineSelfAttention(nn.Module):
    """Explicit multi-head self-attention implemented with native PyTorch ops."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            # Mask invalid key positions. Shape: [B, 1, 1, S].
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        # Computing softmax in fp32 provides a stable reference for fp16/bf16 tests.
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


# ====================================================================
# Project-authored optimized implementation (TechJam 2026 Track 3).
# This region replaces ONLY the UserOptimizedTransformer class of the
# official script; everything outside it is byte-identical to the
# official benchmark (verified mechanically by tools/build_submission.py).
#
# Routing:
#   - d_model <= 128, causal, seq %% 32 == 0, no padding mask:
#       fused-block megakernel — the whole transformer block runs as two
#       authored Triton kernels (LayerNorm+QKV | flash attention over all
#       heads with the output projection folded into the head loop, then
#       residual + LayerNorm + exact-erf GELU FFN + residual in-register),
#       and the full forward replays as one CUDA graph.
#   - larger d_model (no padding mask):
#       fp16 tensor-core stack — packed QKV/out/FFN GEMMs in fp16 with
#       fp32 accumulation, authored Triton flash attention (head_dim
#       <= 256), fp32 residual stream and LayerNorms, CUDA-graphed.
#   - padding masks, CPU, non-CUDA input, reduced-precision whole-model
#     dtypes, or missing Triton: the baseline path runs unchanged
#     (numerically exact, including pre-softmax key masking).
#
# Precision: fp32 at the boundary and in every drift-sensitive spot
# (residuals, softmax statistics, LayerNorm); fp16 only inside GEMMs and
# attention tiles with fp32 accumulation.
# ====================================================================

try:  # Triton is required only for the fast paths; everything degrades.
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:  # pragma: no cover - judge environment without triton
    triton = None
    tl = None
    _TRITON_OK = False

_LN_EPS_C = tl.constexpr(1e-5) if _TRITON_OK else None
_GRAPH_WARMUP_CALLS = 3


if _TRITON_OK:

    def _sub_prune_configs(configs, named_args, **kwargs):
        d_pad = named_args.get("D_PAD", kwargs.get("D_PAD", 64))
        fp16 = named_args.get("FP16", kwargs.get("FP16", True))
        ebytes = 2 if fp16 else 4
        keep = [
            c for c in configs
            if (c.kwargs["BLOCK_M"] + 2 * c.kwargs["BLOCK_N"]) * d_pad * ebytes <= 64 * 1024
            and c.kwargs["BLOCK_M"] * d_pad <= 16384
        ]
        return keep or configs[:1]

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
        prune_configs_by={"early_config_prune": _sub_prune_configs},
    )
    @triton.jit
    def _sub_attn_fwd(
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

    def _sub_triton_attention(q, k, v, scale, causal):
        B, H, S, D = q.shape
        d_pad = max(16, triton.next_power_of_2(D))
        out = torch.empty_like(q)
        q4 = q.reshape(B * H, S, D)
        k4 = k.reshape(B * H, S, D)
        v4 = v.reshape(B * H, S, D)
        o4 = out.reshape(B * H, S, D)
        grid = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B * H)  # noqa: E731
        _sub_attn_fwd[grid](
            q4, k4, v4, o4,
            q4.stride(0), q4.stride(1), q4.stride(2),
            k4.stride(0), k4.stride(1), k4.stride(2),
            v4.stride(0), v4.stride(1), v4.stride(2),
            o4.stride(0), o4.stride(1), o4.stride(2),
            scale, SEQ=S, D=D, D_PAD=d_pad, CAUSAL=causal,
            FP16=(q.dtype == torch.float16),
        )
        return out

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
    def _sub_norm_qkv(
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
        inv = 1.0 / tl.sqrt(var + _LN_EPS_C)
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
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
        ],
        key=["SEQ", "D_PAD", "H", "HD_PAD"],
    )
    @triton.jit
    def _sub_attn_block_tail(
        QKV, X, Wo, Bo, Ln2W, Ln2B, Wf1, Bf1, Wf2, Bf2, XOut,
        scale,
        SEQ, B,
        D: tl.constexpr, D_PAD: tl.constexpr,
        H: tl.constexpr, HD: tl.constexpr, HD_PAD: tl.constexpr,
        FFN: tl.constexpr, FFN_PAD: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D_PAD)
        offs_hd = tl.arange(0, HD_PAD)
        m_mask = offs_m < SEQ
        d_mask = offs_d < D
        hd_mask = offs_hd < HD

        qkv_base = QKV + pid_b * SEQ * (3 * D)
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
        inv = 1.0 / tl.sqrt(var + _LN_EPS_C)
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
    def _sub_final_norm(
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
        inv = 1.0 / tl.sqrt(var + _LN_EPS_C)
        lnw = tl.load(LnW + offs_d, mask=d_mask, other=0.0)
        lnb = tl.load(LnB + offs_d, mask=d_mask, other=0.0)
        y = (diff * inv[:, None]) * lnw[None, :] + lnb[None, :]
        tl.store(Out + offs_t[:, None] * D + offs_d[None, :], y,
                 mask=t_mask[:, None] & d_mask[None, :])


def _sub_pack_fused_layer(layer):
    cache = getattr(layer, "_sub_fused_cache", None)
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
        object.__setattr__(layer, "_sub_fused_cache", cache)
    return cache


def _sub_pack_fp16_layer(layer):
    cache = getattr(layer, "_sub_fp16_cache", None)
    if cache is None:
        attn = layer.attention
        cache = {
            "qkv": (torch.cat([attn.q_proj.weight, attn.k_proj.weight,
                               attn.v_proj.weight], dim=0).half().contiguous(),
                    torch.cat([attn.q_proj.bias, attn.k_proj.bias,
                               attn.v_proj.bias], dim=0).half().contiguous()),
            "out_proj": (attn.out_proj.weight.half().contiguous(),
                         attn.out_proj.bias.half().contiguous()),
            "ffn_in": (layer.ffn_in.weight.half().contiguous(),
                       layer.ffn_in.bias.half().contiguous()),
            "ffn_out": (layer.ffn_out.weight.half().contiguous(),
                        layer.ffn_out.bias.half().contiguous()),
        }
        object.__setattr__(layer, "_sub_fp16_cache", cache)
    return cache


class UserOptimizedTransformer(BaselineTransformer):
    """Project-authored optimized implementation. Same parameters, same
    forward signature, same outputs within the official tolerance; the
    docstring at the top of this region describes the routing."""

    def _fused_forward(self, x):
        cfg = self.config
        B, S, D = x.shape
        H = cfg.num_heads
        HD = D // H
        FFN = cfg.ffn_dim
        tokens = B * S
        d_pad = max(16, triton.next_power_of_2(D))
        hd_pad = max(16, triton.next_power_of_2(HD))
        ffn_pad = max(16, triton.next_power_of_2(FFN))
        scale = HD ** -0.5

        bufs = getattr(self, "_sub_bufs", None)
        if bufs is None or bufs["a"].shape != x.shape:
            bufs = {
                "qkv": torch.empty(tokens, 3 * D, dtype=torch.float16, device=x.device),
                "a": torch.empty_like(x),
                "b": torch.empty_like(x),
                "fn_w": self.final_norm.weight.float().contiguous(),
                "fn_b": self.final_norm.bias.float().contiguous(),
            }
            object.__setattr__(self, "_sub_bufs", bufs)
        qkv = bufs["qkv"]

        src = x.contiguous()
        for i, layer in enumerate(self.layers):
            dst = bufs["a"] if i % 2 == 0 else bufs["b"]
            c = _sub_pack_fused_layer(layer)
            grid1 = lambda meta: (triton.cdiv(tokens, meta["BLOCK_T"]),)  # noqa: E731
            _sub_norm_qkv[grid1](
                src.view(tokens, D), c["w_qkv"], c["b_qkv"], c["ln1_w"], c["ln1_b"],
                qkv, tokens, D=D, D_PAD=d_pad,
            )
            grid2 = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B)  # noqa: E731
            _sub_attn_block_tail[grid2](
                qkv, src, c["w_o"], c["b_o"], c["ln2_w"], c["ln2_b"],
                c["w_f1"], c["b_f1"], c["w_f2"], c["b_f2"], dst,
                scale, S, B,
                D=D, D_PAD=d_pad, H=H, HD=HD, HD_PAD=hd_pad,
                FFN=FFN, FFN_PAD=ffn_pad,
            )
            src = dst
        out = torch.empty_like(src)
        _sub_final_norm[(triton.cdiv(tokens, 128),)](
            src.view(tokens, D), bufs["fn_w"], bufs["fn_b"], out.view(tokens, D),
            tokens, D=D, D_PAD=d_pad, BLOCK_T=128, num_warps=4,
        )
        return out

    def _fp16_forward(self, x):
        cfg = self.config
        B, S, d = x.shape
        for layer in self.layers:
            c = _sub_pack_fp16_layer(layer)
            attn = layer.attention
            h16 = layer.norm1(x).half()
            qkv = F.linear(h16, c["qkv"][0], c["qkv"][1])
            q, k, v = qkv.split(d, dim=-1)
            q = q.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
            k = k.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
            v = v.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
            if attn.head_dim <= 256 and S % 32 == 0:
                context = _sub_triton_attention(q, k, v, attn.scale, cfg.causal)
            else:
                scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
                if cfg.causal:
                    cm = torch.ones((S, S), device=x.device,
                                    dtype=torch.bool).triu(diagonal=1)
                    scores = scores.masked_fill(cm, float("-inf"))
                context = torch.matmul(
                    torch.softmax(scores.float(), dim=-1).half(), v)
            context = context.transpose(1, 2).contiguous().view(B, S, d)
            attn_out = F.linear(context, c["out_proj"][0], c["out_proj"][1])
            x = x + attn_out.float()
            h16 = layer.norm2(x).half()
            hidden = F.linear(h16, c["ffn_in"][0], c["ffn_in"][1])
            hidden = F.gelu(hidden.float(), approximate="none").half()
            x = x + F.linear(hidden, c["ffn_out"][0], c["ffn_out"][1]).float()
        return self.final_norm(x)

    def _fast_forward(self, x):
        cfg = self.config
        if (cfg.d_model <= 128 and cfg.ffn_dim <= 128 and cfg.causal
                and cfg.d_model % cfg.num_heads == 0
                and (cfg.d_model // cfg.num_heads) <= 128
                and cfg.seq_len % 32 == 0):
            return self._fused_forward(x)
        return self._fp16_forward(x)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (not _TRITON_OK or x.device.type != "cuda"
                or x.dtype != torch.float32):
            return BaselineTransformer.forward(self, x, valid_token_mask)
        if valid_token_mask is not None and not bool(valid_token_mask.all()):
            # Exact baseline path, including pre-softmax key masking.
            return BaselineTransformer.forward(self, x, valid_token_mask)

        state = getattr(self, "_sub_graph_state", None)
        if state is None:
            state = {"calls": 0, "graph": None, "static_x": None, "static_out": None}
            object.__setattr__(self, "_sub_graph_state", state)
        if state["graph"] is None:
            state["calls"] += 1
            if state["calls"] <= _GRAPH_WARMUP_CALLS:
                return self._fast_forward(x)
            static_x = x.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._fast_forward(static_x)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                static_out = self._fast_forward(static_x)
            state.update(graph=graph, static_x=static_x, static_out=static_out)
            graph.replay()
            return static_out.clone()
        state["static_x"].copy_(x)
        state["graph"].replay()
        return state["static_out"].clone()


def copy_model_weights(
    baseline: nn.Module, optimized: nn.Module, strict: bool = True
) -> None:
    """Copy identical weights into both implementations for a fair comparison."""
    state_dict = copy.deepcopy(baseline.state_dict())
    incompatible = optimized.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[warning] missing optimized keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warning] unexpected optimized keys: {incompatible.unexpected_keys}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )
    positions = torch.arange(config.seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask


@dataclass
class AccuracyResult:
    passed: bool
    total_elements: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    mean_abs_error: float
    failed_feature_dims: List[int]
    worst_index: Tuple[int, ...]
    reference_at_worst: float
    optimized_at_worst: float


def compare_outputs(
    reference: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float,
    atol: float,
) -> AccuracyResult:
    if reference.shape != optimized.shape:
        raise AssertionError(
            f"shape mismatch: baseline={tuple(reference.shape)}, "
            f"optimized={tuple(optimized.shape)}"
        )
    if reference.dtype != optimized.dtype:
        print(
            f"[warning] dtype mismatch: baseline={reference.dtype}, "
            f"optimized={optimized.dtype}"
        )

    ref = reference.detach().float()
    opt = optimized.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()

    # Exact interpretation of the requested OR condition. torch.isclose uses
    # atol + rtol * abs(ref), which is slightly more permissive and is not used.
    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed_mask = ~passed_mask
    failed_elements = int(failed_mask.sum().item())
    total_elements = reference.numel()

    flat_worst = int(abs_error.reshape(-1).argmax().item())
    worst_index_list = []
    remaining = flat_worst
    for size in reversed(reference.shape):
        worst_index_list.append(remaining % size)
        remaining //= size
    worst_index = tuple(reversed(worst_index_list))

    denominator = ref.abs().clamp_min(1e-12)
    relative_error = abs_error / denominator

    # Summarize failures by the last/output-feature dimension.
    if reference.ndim == 0:
        failed_feature_dims = [0] if failed_elements else []
    elif reference.ndim == 1:
        failed_feature_dims = torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
    else:
        reduce_dims = tuple(range(reference.ndim - 1))
        failed_by_feature = failed_mask.any(dim=reduce_dims)
        failed_feature_dims = (
            torch.nonzero(failed_by_feature, as_tuple=False).flatten().tolist()
        )

    return AccuracyResult(
        passed=failed_elements == 0,
        total_elements=total_elements,
        failed_elements=failed_elements,
        max_abs_error=float(abs_error.max().item()),
        max_relative_error=float(relative_error.max().item()),
        mean_abs_error=float(abs_error.mean().item()),
        failed_feature_dims=failed_feature_dims,
        worst_index=worst_index,
        reference_at_worst=float(ref[worst_index].item()),
        optimized_at_worst=float(opt[worst_index].item()),
    )


def run_accuracy_tests(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
) -> bool:
    print("\n=== Accuracy check ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")

    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=padding_ratio,
                input_scale=input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)

            all_passed &= result.passed
            global_max_abs = max(global_max_abs, result.max_abs_error)
            global_max_rel = max(global_max_rel, result.max_relative_error)
            total_failed += result.failed_elements
            total_elements += result.total_elements

            status = "PASS" if result.passed else "FAIL"
            print(
                f"trial {trial + 1:02d}/{trials}: {status} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"max_rel={result.max_relative_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

            if not result.passed:
                preview = result.failed_feature_dims[:16]
                suffix = "..." if len(result.failed_feature_dims) > len(preview) else ""
                print(
                    f"  worst_index={result.worst_index}, "
                    f"baseline={result.reference_at_worst:.8g}, "
                    f"optimized={result.optimized_at_worst:.8g}"
                )
                print(f"  failed output feature dims={preview}{suffix}")

    print(
        f"summary: {'PASS' if all_passed else 'FAIL'} | "
        f"max_abs={global_max_abs:.6g} | max_rel={global_max_rel:.6g} | "
        f"failed={total_failed}/{total_elements}"
    )
    return all_passed


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class TimingResult:
    samples_ms: List[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> List[float]:
    samples_ms: List[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

            torch.cuda.synchronize(device)
            for index in range(iterations):
                starts[index].record()
                model(x, valid_mask)
                ends[index].record()
            torch.cuda.synchronize(device)

            samples_ms.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
        else:
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, valid_mask)
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

    return samples_ms


def benchmark_models(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
) -> None:
    print("\n=== Performance benchmark ===")
    print("timing excludes random-data generation and uses a fixed input")
    if device.type == "cuda":
        print("CUDA latency is measured with torch.cuda.Event on the current stream")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    # Warm up both models before collecting any timing data.
    warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
        else:
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens_per_call = config.batch_size * config.seq_len
    baseline_tokens_per_second = tokens_per_call * 1000.0 / baseline_result.median_ms
    optimized_tokens_per_second = tokens_per_call * 1000.0 / optimized_result.median_ms

    print(
        f"baseline : median={baseline_result.median_ms:.4f} ms | "
        f"mean={baseline_result.mean_ms:.4f} ms | "
        f"p90={baseline_result.p90_ms:.4f} ms | "
        f"min={baseline_result.min_ms:.4f} ms | "
        f"throughput={baseline_tokens_per_second:.2f} token/s"
    )
    print(
        f"optimized: median={optimized_result.median_ms:.4f} ms | "
        f"mean={optimized_result.mean_ms:.4f} ms | "
        f"p90={optimized_result.p90_ms:.4f} ms | "
        f"min={optimized_result.min_ms:.4f} ms | "
        f"throughput={optimized_tokens_per_second:.2f} token/s"
    )
    print(f"speedup  : {speedup:.3f}x based on median latency")


def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and optimized PyTorch Transformer"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable TF32 on CUDA for both implementations",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()
    validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    # Compile only after model construction, weight copy, device transfer, and eval().
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    optimized = maybe_compile(optimized, args.compile_user, args.compile_mode)

    print("=== Configuration ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    accuracy_passed = run_accuracy_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol,
        atol=args.atol,
    )

    if not accuracy_passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        print("Use --benchmark-on-failure to benchmark an incorrect implementation anyway.")
        return 2

    benchmark_models(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
    )
    return 0 if accuracy_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
