"""k001: swap explicit attention math for PyTorch's fused attention.

The baseline computes attention in ~7 separate steps (matmul, scale, causal
mask, padding mask, fp32 softmax, matmul, reshape), materializing the full
[batch, heads, seq, seq] score table. torch.nn.functional
.scaled_dot_product_attention does the same math in one fused kernel and never
materializes the table.

Structure, parameter names, and every other op are identical to the baseline,
so the strict weight copy works and only the attention inner math changes.
When the padding mask is all-true (the benchmark default) we pass is_causal=True
and no mask, which lets PyTorch pick its fastest backend; otherwise we build the
combined boolean mask explicitly (True = may attend).
"""

import torch
import torch.nn.functional as F

NAME = "k001_sdpa"
DESCRIPTION = "Fused scaled_dot_product_attention replacing explicit attention math."


def _make_attention_class(otb):
    class SDPAAttention(otb.BaselineSelfAttention):
        def forward(self, x, valid_token_mask=None, causal=False):
            batch, seq_len, _ = x.shape
            q = self._split_heads(self.q_proj(x))
            k = self._split_heads(self.k_proj(x))
            v = self._split_heads(self.v_proj(x))

            if valid_token_mask is None or bool(valid_token_mask.all()):
                context = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            else:
                keep = valid_token_mask[:, None, None, :].expand(
                    batch, 1, seq_len, seq_len
                )
                if causal:
                    causal_keep = torch.ones(
                        seq_len, seq_len, dtype=torch.bool, device=x.device
                    ).tril()
                    keep = keep & causal_keep[None, None, :, :]
                context = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)

            context = (
                context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
            )
            output = self.out_proj(context)
            if valid_token_mask is not None:
                output = output.masked_fill(~valid_token_mask[..., None], 0)
            return output

    return SDPAAttention


def build(otb, config):
    model = otb.UserOptimizedTransformer(config)
    sdpa_cls = _make_attention_class(otb)
    for layer in model.layers:
        # Same attribute layout, so swapping the class only changes forward().
        layer.attention.__class__ = sdpa_cls
    return model
