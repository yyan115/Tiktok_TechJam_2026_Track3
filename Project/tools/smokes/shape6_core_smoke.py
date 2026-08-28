"""Shape-6 core proof: k007 at full batch 10,000 vs a batch-chunked baseline.

The official baseline OOMs on 8 GB at shape 6 (B=10000, d=128, seq=128: the
per-layer score tensors are ~2.6 GB each). But the model is batch-independent,
so the fp32 eager baseline run in batch chunks IS the official computation,
exactly. k007's fused forward at the full batch fits comfortably (~3 GB).
A pass here banks shape-6 correctness before rental day, like shape 14's.

Correctness only — NOT a benchmark; the baseline cannot be timed locally.
"""
import importlib.util
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
import torch_transformer_benchmark as otb  # noqa: E402

ATOL, RTOL = 2e-3, 2e-2
CHUNK = 500


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    cfg = otb.TransformerConfig(batch_size=10000, seq_len=128, d_model=128,
                                num_heads=4, ffn_dim=128, num_layers=4,
                                causal=True)
    cfg.validate()

    base = otb.BaselineTransformer(cfg).to(dev).eval()
    spec = importlib.util.spec_from_file_location(
        "cand", f"{REPO}/Project/kernels/k007_fused_block.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cand = mod.build(otb, cfg).to(dev).eval()
    otb.copy_model_weights(base, cand)

    x = torch.randn(10000, 128, 128, device=dev)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = cand(x, None)
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / 2**20

    bad, maxerr = 0, 0.0
    with torch.no_grad():
        for s in range(0, 10000, CHUNK):
            ref = base(x[s:s + CHUNK], None).float()
            o = out[s:s + CHUNK].float()
            bad += (~torch.isclose(o, ref, atol=ATOL, rtol=RTOL)).sum().item()
            maxerr = max(maxerr, (o - ref).abs().max().item())
            del ref

    print(f"shape6 B=10000 k007-vs-chunked-baseline: violations={bad} "
          f"max_abs_err={maxerr:.3e} candidate_peak_mem={peak_mib:.0f} MiB")
    print("RESULT:", "PASS" if bad == 0 else "FAIL")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
