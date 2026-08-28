"""k002: PROJECT-AUTHORED fusion — one packed QKV projection instead of three.

The baseline launches three separate GEMMs (q_proj, k_proj, v_proj) per layer
per forward. This candidate packs their weights into a single [3d, d] matrix
once (a weight-layout transformation we author, done lazily on first use) and
performs ONE GEMM, cutting two kernel launches per layer — 8 launches saved
per forward on the 4-layer shapes. All remaining math is bit-for-bit the
baseline's (same explicit attention, same fp32 softmax), so correctness risk
is confined to the GEMM fusion itself.

Authorship note (innovation policy): the fusion and weight packing are ours;
no external kernel project is wrapped. torch.nn.functional.linear is the same
primitive the baseline itself uses.
"""

import torch
import torch.nn.functional as F

NAME = "k002_fused_qkv"
DESCRIPTION = "Authored fused single-GEMM QKV projection; rest identical to baseline."


def _make_attention_class(otb):
    class FusedQKVAttention(otb.BaselineSelfAttention):
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

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if causal:
                causal_mask = torch.ones(
                    (seq_len, seq_len), device=x.device, dtype=torch.bool
                ).triu(diagonal=1)
                scores = scores.masked_fill(causal_mask, float("-inf"))
            if valid_token_mask is not None:
                invalid_keys = ~valid_token_mask[:, None, None, :]
                scores = scores.masked_fill(invalid_keys, float("-inf"))
            probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
            context = torch.matmul(probs, v)
            context = (
                context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
            )
            output = self.out_proj(context)
            if valid_token_mask is not None:
                output = output.masked_fill(~valid_token_mask[..., None], 0)
            return output

    return FusedQKVAttention


def build(otb, config):
    model = otb.UserOptimizedTransformer(config)
    cls = _make_attention_class(otb)
    for layer in model.layers:
        layer.attention.__class__ = cls
    return model
