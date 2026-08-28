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
