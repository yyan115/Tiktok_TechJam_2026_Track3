"""Shape-14 core proof: the authored Triton attention at seq=100,000 causal,
verified against a chunked fp32 oracle that never materializes the S x S table.

Runs a B=1, H=1, head_dim=64 slice of shape 14 (d=1024, 16 heads -> hd 64):
fp16 q/k/v are ~12.8 MB each, so this fits the 8 GB card and de-risks rental
day. The oracle streams softmax over (row-chunk x key-chunk) blocks in fp32.

Correctness only — NOT a benchmark (the runner stays the referee for speed).
Rebuilt 28 Aug after the original session-scratchpad copy was wiped
(LESSONS: durable smokes live in Project/tools/smokes/).

Usage: python3 shape14_core_smoke.py [--seq 100000] [--kernel k006]
"""
import argparse
import importlib.util
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ATOL, RTOL = 2e-3, 2e-2


def load_kernel_module(name):
    path = {
        "k005": f"{REPO}/Project/kernels/k005_fp16_graphed.py",
        "k006": f"{REPO}/Project/kernels/k006_fp16_hd128.py",
    }[name]
    spec = importlib.util.spec_from_file_location("cand", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chunked_oracle(q, k, v, scale, q_chunk=2048, k_chunk=8192):
    """Causal attention in fp32 with streaming softmax; no S x S table."""
    S, D = q.shape
    out = torch.empty(S, D, dtype=torch.float32, device=q.device)
    for qs in range(0, S, q_chunk):
        qe = min(qs + q_chunk, S)
        qb = q[qs:qe].float()
        m = torch.full((qe - qs,), float("-inf"), device=q.device)
        l = torch.zeros(qe - qs, device=q.device)
        acc = torch.zeros(qe - qs, D, device=q.device)
        for ks in range(0, qe, k_chunk):
            ke = min(ks + k_chunk, qe)
            scores = qb @ k[ks:ke].float().T * scale
            # causal: query row (qs+i) attends keys <= qs+i
            qidx = torch.arange(qs, qe, device=q.device)[:, None]
            kidx = torch.arange(ks, ke, device=q.device)[None, :]
            scores = scores.masked_fill(kidx > qidx, float("-inf"))
            m_new = torch.maximum(m, scores.max(dim=1).values)
            alpha = torch.exp(m - m_new)
            p = torch.exp(scores - m_new[:, None])
            l = l * alpha + p.sum(dim=1)
            acc = acc * alpha[:, None] + p @ v[ks:ke].float()
            m = m_new
        out[qs:qe] = acc / l[:, None]
    return out


def official_error_stats(reference, candidate):
    """Mirror the official finite AND (absolute OR relative) predicate."""
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"shape mismatch: reference={tuple(reference.shape)}, "
            f"candidate={tuple(candidate.shape)}"
        )
    ref = reference.detach().float()
    opt = candidate.detach().float()
    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()
    abs_ok = abs_error <= ATOL
    rel_ok = abs_error <= RTOL * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)
    failed_elements = int((~passed_mask).sum().item())
    max_abs_error = (float("inf") if not bool(finite_mask.all())
                     else abs_error.max().item())
    return failed_elements, max_abs_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=100_000)
    ap.add_argument("--kernel", choices=["k005", "k006"], default="k006")
    args = ap.parse_args()

    assert args.seq % 32 == 0, "kernel fast path needs seq % 32 == 0"
    torch.manual_seed(0)
    dev = torch.device("cuda")
    S, D = args.seq, 64
    scale = D ** -0.5

    q = torch.randn(S, D, device=dev, dtype=torch.float16)
    k = torch.randn(S, D, device=dev, dtype=torch.float16)
    v = torch.randn(S, D, device=dev, dtype=torch.float16)

    mod = load_kernel_module(args.kernel)
    torch.cuda.reset_peak_memory_stats()
    out = mod.triton_attention(
        q.view(1, 1, S, D), k.view(1, 1, S, D), v.view(1, 1, S, D),
        scale, causal=True,
    ).view(S, D)
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / 2**20

    ref = chunked_oracle(q, k, v, scale)
    bad, maxerr = official_error_stats(ref, out)
    print(f"kernel={args.kernel} seq={S} D={D} causal "
          f"violations={bad} max_abs_err={maxerr:.3e} peak_mem={peak_mib:.0f} MiB")
    print("RESULT:", "PASS" if bad == 0 else "FAIL")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
