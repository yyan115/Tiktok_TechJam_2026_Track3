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
# The first eager call is also the exact-specialization compile/launch probe.
# Capture itself still receives the three side-stream warmups recommended for
# CUDA graphs.  Keeping these as two distinct counters lets the five official
# correctness trials exercise eager, capture, and replay paths.
_GRAPH_TRIGGER_CALLS = 1
_GRAPH_SIDE_STREAM_WARMUPS = 3


def _sub_tensor_metadata(tensor):
    """Metadata which must stay invariant for a captured graph replay."""
    return (
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.layout,
    )


def _sub_config_metadata(config):
    return (
        config.batch_size,
        config.seq_len,
        config.d_model,
        config.num_heads,
        config.ffn_dim,
        config.num_layers,
        config.causal,
    )


def _sub_allowed_preflight_fallback(exc):
    """Return whether a launch probe failed for an allowlisted resource limit.

    Correctness, illegal-memory-access, assertion, and arbitrary RuntimeError
    failures deliberately do not match this predicate and therefore propagate.
    """
    exc_type = type(exc)
    qualified = f"{exc_type.__module__}.{exc_type.__name__}".lower()
    message = str(exc).lower()
    resource_markers = (
        "out of resources",
        "outofresources",
        "too many resources requested for launch",
        "shared memory",
        "register spill",
        "ptxas fatal   : entry function",
    )
    triton_origin = "triton" in qualified
    resource_exception = exc_type.__name__.lower() == "outofresources"
    return triton_origin and (
        resource_exception or any(marker in message for marker in resource_markers)
    )


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
    def _sub_ln_fp16(
        X, W, B, Out,
        D: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(X + row * D + offs, mask=mask, other=0.0)
        mean = tl.sum(x, axis=0) / D
        diff = tl.where(mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / D
        inv = 1.0 / tl.sqrt(var + _LN_EPS_C)
        w = tl.load(W + offs, mask=mask, other=0.0)
        b = tl.load(B + offs, mask=mask, other=0.0)
        y = (diff * inv) * w + b
        tl.store(Out + row * D + offs, y.to(tl.float16), mask=mask)

    def _sub_ln_to_fp16(norm, x):
        shape = x.shape
        D = shape[-1]
        rows = x.numel() // D
        out = torch.empty(shape, dtype=torch.float16, device=x.device)
        _sub_ln_fp16[(rows,)](
            x.reshape(rows, D), norm.weight, norm.bias, out.reshape(rows, D),
            D=D, BLOCK_D=triton.next_power_of_2(D),
            num_warps=8 if D >= 1024 else 4,
        )
        return out

    @triton.jit
    def _sub_gelu_fp16(X, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        tl.store(X + offs, y.to(tl.float16), mask=mask)

    def _sub_gelu_fp16_(h):
        # The Triton kernel mutates its input storage.  ``reshape`` is allowed
        # to allocate a copy for non-contiguous tensors, which would silently
        # leave ``h`` unchanged.  Make the storage contract explicit first.
        h = h.contiguous()
        if not h.is_contiguous():  # defensive: never launch on an alias/copy
            raise RuntimeError("GELU fast path requires contiguous storage")
        n = h.numel()
        _sub_gelu_fp16[(triton.cdiv(n, 1024),)](h.view(-1), n, BLOCK=1024,
                                               num_warps=4)
        return h

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

    def _sub_invalidate_runtime(self):
        """Drop every object derived from weights, device, or input metadata."""
        epoch = getattr(self, "_sub_weight_epoch", 0) + 1
        self.__dict__.pop("_sub_bufs", None)
        self.__dict__.pop("_sub_graph_state", None)
        self.__dict__.pop("_sub_route_status", None)
        for layer in getattr(self, "layers", ()):
            layer.__dict__.pop("_sub_fused_cache", None)
            layer.__dict__.pop("_sub_fp16_cache", None)
        object.__setattr__(self, "_sub_weight_epoch", epoch)

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self._sub_invalidate_runtime()
        return result

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._sub_invalidate_runtime()
        return result

    def _sub_refresh_weight_state(self):
        # Official weight copies use load_state_dict(), and device/dtype moves
        # use _apply(); both increment this epoch and invalidate every derived
        # cache.  This O(1) check stays on the latency-critical replay path.
        if not hasattr(self, "_sub_weight_epoch"):
            object.__setattr__(self, "_sub_weight_epoch", 0)
        return self._sub_weight_epoch

    def _sub_route_key(self, x):
        cfg = self.config
        route = "fused" if (
            cfg.d_model <= 128 and cfg.ffn_dim <= 128 and cfg.causal
            and cfg.d_model % cfg.num_heads == 0
            and (cfg.d_model // cfg.num_heads) <= 128
            and cfg.seq_len % 32 == 0
        ) else "fp16"
        capability = torch.cuda.get_device_capability(x.device)
        tensor_metadata = _sub_tensor_metadata(x)
        # Storage offset is a replay invariant, but not a Triton specialization
        # input. Excluding it avoids recompiling each B=1 view of shape 14.
        specialization_metadata = tensor_metadata[:5] + tensor_metadata[6:]
        return (
            route,
            _sub_config_metadata(cfg),
            specialization_metadata,
            tuple(capability),
        )

    def _sub_preflight_fast(self, x):
        """Compile and launch the exact specialization before graph capture.

        Only an explicitly recognized Triton resource-limit failure selects
        the baseline.  Every unexpected failure propagates and fails the run.
        The decision is retained so evidence can report the actual route.
        """
        key = self._sub_route_key(x)
        statuses = getattr(self, "_sub_route_status", None)
        if statuses is None:
            statuses = {}
            object.__setattr__(self, "_sub_route_status", statuses)
        status = statuses.get(key)
        if status is not None and status["route"] == "baseline":
            return self._sub_controlled_baseline(x)
        if status is not None:
            return self._fast_forward(x)
        try:
            output = self._fast_forward(x)
            torch.cuda.synchronize(x.device)
        except Exception as exc:
            if not _sub_allowed_preflight_fallback(exc):
                raise
            if x.shape[1] >= 16384:
                raise RuntimeError(
                    "Triton resource fallback is unavailable for extreme "
                    "sequences because the dense baseline is infeasible"
                ) from exc
            statuses[key] = {
                "route": "baseline",
                "reason": "triton-resource-limit",
                "fallback": "batch-chunked-official-baseline",
                "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
                "message": str(exc),
            }
            return self._sub_controlled_baseline(x)
        statuses[key] = {"route": "fast", "reason": None}
        return output

    def _sub_fast_route_available(self, x):
        status = getattr(self, "_sub_route_status", {}).get(self._sub_route_key(x))
        return status is None or status["route"] == "fast"

    def _sub_runtime_route_report(self):
        """JSON-safe record of every exact specialization decision."""
        statuses = getattr(self, "_sub_route_status", {})
        return [
            {"specialization": repr(key), **value}
            for key, value in sorted(statuses.items(), key=lambda item: repr(item[0]))
        ]

    def _sub_controlled_baseline(self, x):
        """Exact batch-chunked fallback where dense attention is feasible.

        Batch rows never interact.  Capping each dense score table prevents a
        resource-limited fast specialization from turning shape 6 into an OOM.
        Sequence-dominated shape 14 is rejected earlier because batch chunks
        cannot make its S x S table feasible.
        """
        batch, seq_len, _d_model = x.shape
        bytes_per_batch_row = self.config.num_heads * seq_len * seq_len * 4
        score_budget = 512 * 2**20
        chunk = max(1, min(batch, score_budget // max(1, bytes_per_batch_row)))
        if chunk >= batch:
            return BaselineTransformer.forward(self, x, None)
        output = torch.empty_like(x)
        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            output[start:end] = BaselineTransformer.forward(
                self, x[start:end], None
            )
        return output

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
        long_seq = S >= 16384  # extreme-sequence memory discipline (shape-14 class)
        for layer in self.layers:
            c = _sub_pack_fp16_layer(layer)
            attn = layer.attention
            h16 = _sub_ln_to_fp16(layer.norm1, x)
            qkv = F.linear(h16, c["qkv"][0], c["qkv"][1])
            q, k, v = qkv.split(d, dim=-1)
            q = q.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
            k = k.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
            v = v.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
            if long_seq:
                del qkv, h16  # ~0.8 GB of dead staging per micro-batch at S=100k
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
            if long_seq:
                # Sequence-chunked tail: exact row-wise math, O(chunk) transients.
                del q, k, v
                x_new = torch.empty_like(x)
                CH = 16384
                for s0 in range(0, S, CH):
                    s1 = min(s0 + CH, S)
                    a = F.linear(context[:, s0:s1], c["out_proj"][0], c["out_proj"][1])
                    x2 = x[:, s0:s1] + a.float()
                    h16c = _sub_ln_to_fp16(layer.norm2, x2)
                    hid = F.linear(h16c, c["ffn_in"][0], c["ffn_in"][1])
                    hid = _sub_gelu_fp16_(hid)
                    fo = F.linear(hid, c["ffn_out"][0], c["ffn_out"][1])
                    x_new[:, s0:s1] = x2 + fo.float()
                x = x_new
                continue
            attn_out = F.linear(context, c["out_proj"][0], c["out_proj"][1])
            x = x + attn_out.float()
            h16 = _sub_ln_to_fp16(layer.norm2, x)
            hidden = F.linear(h16, c["ffn_in"][0], c["ffn_in"][1])
            hidden = _sub_gelu_fp16_(hidden)
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
        if not _TRITON_OK:
            if x.device.type == "cuda" and x.ndim == 3 and x.shape[1] >= 16384:
                raise RuntimeError(
                    "Triton is unavailable and the dense extreme-sequence "
                    "baseline is infeasible"
                )
            if (x.device.type == "cuda" and x.dtype == torch.float32
                    and x.ndim == 3 and valid_token_mask is not None
                    and valid_token_mask.shape == x.shape[:2]
                    and valid_token_mask.dtype == torch.bool
                    and valid_token_mask.device == x.device
                    and bool(valid_token_mask.all())):
                object.__setattr__(self, "_sub_route_status", {
                    "triton-unavailable": {
                        "route": "baseline",
                        "reason": "triton-unavailable",
                        "fallback": "batch-chunked-official-baseline",
                    }
                })
                return self._sub_controlled_baseline(x)
            return BaselineTransformer.forward(self, x, valid_token_mask)
        if (x.device.type != "cuda" or x.dtype != torch.float32
                or torch.compiler.is_compiling()):
            # torch.compile tracing: hand dynamo the plain baseline graph
            # (compiling correct math) instead of our capture machinery.
            return BaselineTransformer.forward(self, x, valid_token_mask)
        if (x.ndim != 3 or x.shape[1] != self.config.seq_len
                or x.shape[2] != self.config.d_model):
            raise ValueError(
                "input must have shape [batch, config.seq_len, config.d_model]"
            )
        if valid_token_mask is not None and (
            valid_token_mask.shape != x.shape[:2]
            or valid_token_mask.dtype != torch.bool
            or valid_token_mask.device != x.device
        ):
            raise ValueError(
                "valid_token_mask must be bool, colocated, and shaped [batch, seq_len]"
            )
        if valid_token_mask is not None and not bool(valid_token_mask.all()):
            # Exact baseline path, including pre-softmax key masking.
            return BaselineTransformer.forward(self, x, valid_token_mask)

        # Fast kernels and graph replay assume a strided, contiguous tensor.
        # Route uncommon layouts through the exact baseline instead of letting
        # a view/reshape silently change their storage contract.
        if x.layout != torch.strided or not x.is_contiguous():
            return BaselineTransformer.forward(self, x, valid_token_mask)

        self._sub_refresh_weight_state()

        B, S = x.shape[0], x.shape[1]
        if S >= 16384:
            # Extreme sequences (shape-14 class): the naive attention table is
            # infeasible on any hardware — divide the computation into blocks
            # (per the track guidance): batch micro-chunks, no CUDA graphs,
            # streamed into one preallocated output.
            out = torch.empty_like(x)
            for bs in range(0, B, 1):
                out[bs:bs + 1] = self._sub_preflight_fast(x[bs:bs + 1])
            return out
        if B * S >= 262144:
            # Huge token counts (shape-6 class): graph capture memory is
            # unproven at this size and launch savings are negligible when
            # every kernel runs for milliseconds — run eagerly.
            return self._sub_preflight_fast(x)

        state = getattr(self, "_sub_graph_state", None)
        if state is None:
            state = {
                "calls": 0,
                "graph": None,
                "static_x": None,
                "static_out": None,
                "input_metadata": None,
                "config_metadata": None,
                "weight_epoch": None,
            }
            object.__setattr__(self, "_sub_graph_state", state)
        if state["graph"] is None:
            state["calls"] += 1
            if state["calls"] <= _GRAPH_TRIGGER_CALLS:
                return self._sub_preflight_fast(x)
            if not self._sub_fast_route_available(x):
                return self._sub_controlled_baseline(x)
            static_x = x.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(_GRAPH_SIDE_STREAM_WARMUPS):
                    self._fast_forward(static_x)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                static_out = self._fast_forward(static_x)
            state.update(
                graph=graph,
                static_x=static_x,
                static_out=static_out,
                input_metadata=_sub_tensor_metadata(x),
                config_metadata=_sub_config_metadata(self.config),
                weight_epoch=getattr(self, "_sub_weight_epoch"),
            )
            graph.replay()
            return static_out.clone()
        if state["input_metadata"] != _sub_tensor_metadata(x):
            raise RuntimeError(
                "CUDA graph replay input metadata changed: exact shape, dtype, "
                "device, stride, storage offset, and layout must match capture"
            )
        if state["config_metadata"] != _sub_config_metadata(self.config):
            raise RuntimeError("CUDA graph replay configuration changed after capture")
        if state["weight_epoch"] != getattr(self, "_sub_weight_epoch"):
            raise RuntimeError("CUDA graph replay weights changed after capture")
        state["static_x"].copy_(x)
        state["graph"].replay()
        return state["static_out"].clone()
