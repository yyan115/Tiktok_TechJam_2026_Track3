"""Padded-mask correctness smoke for the fixed k004/k005.

Verifies the auditor's finding is closed: with a real (non-all-true)
valid_token_mask, the candidates must match the fp32 baseline within the
official tolerance (abs 2e-3 OR rel 2%). Also re-checks the dense graphed
path (mask=None) and the all-true-mask path still agree after the edits.

NOT a benchmark. No timing. Runner remains the only referee for speed.
"""
import importlib.util
import sys

import torch

REPO = "/home/admin/Desktop/Repos/Tiktok_TechJam_2026_Track3"
sys.path.insert(0, REPO)
import torch_transformer_benchmark as otb  # noqa: E402

ATOL, RTOL = 2e-3, 2e-2
torch.manual_seed(0)
dev = torch.device("cuda")


def load_kernel(path):
    spec = importlib.util.spec_from_file_location("cand", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(name, out, ref):
    bad = (~torch.isclose(out.float(), ref, atol=ATOL, rtol=RTOL)).sum().item()
    maxerr = (out.float() - ref).abs().max().item()
    print(f"  {name}: violations={bad} max_abs_err={maxerr:.3e}")
    return bad == 0


def run_case(kernel_path, cfg_kwargs, causal, padding_ratio):
    cfg = otb.TransformerConfig(causal=causal, **cfg_kwargs)
    cfg.validate()
    base = otb.BaselineTransformer(cfg).to(dev).eval()
    mod = load_kernel(kernel_path)
    cand = mod.build(otb, cfg).to(dev).eval()
    otb.copy_model_weights(base, cand)

    B, S, D = cfg.batch_size, cfg.seq_len, cfg.d_model
    x = torch.randn(B, S, D, device=dev)

    # padded mask: per-row valid lengths, at least one False row somewhere
    lengths = torch.randint(
        max(1, int(S * (1 - padding_ratio))), S + 1, (B,), device=dev
    )
    lengths[0] = max(1, int(S * (1 - padding_ratio)))  # guarantee padding
    mask = torch.arange(S, device=dev)[None, :] < lengths[:, None]
    all_true = torch.ones(B, S, dtype=torch.bool, device=dev)

    ok = True
    with torch.no_grad():
        ref_pad = base(x, mask).float()
        ref_dense = base(x, None).float()
        ok &= check("padded  ", cand(x, mask), ref_pad)
        # drive past warmup + capture for the graphed dense path
        for _ in range(6):
            out_dense = cand(x, None)
        ok &= check("dense   ", out_dense, ref_dense)
        ok &= check("all-true", cand(x, all_true), ref_dense)
    return ok


CASES = [
    # head_dim=32 -> Triton-eligible attention; masked path must reroute
    dict(batch_size=4, seq_len=128, d_model=256, num_heads=8, ffn_dim=1024, num_layers=4),
    # head_dim=128 -> matmul fallback everywhere
    dict(batch_size=2, seq_len=96, d_model=512, num_heads=4, ffn_dim=2048, num_layers=2),
]

if __name__ == "__main__":
    overall = True
    for kp in [f"{REPO}/Project/kernels/k004_graphed_triton.py",
               f"{REPO}/Project/kernels/k005_fp16_graphed.py"]:
        for cfg_kwargs in CASES:
            for causal in (False, True):
                print(f"{kp.rsplit('/',1)[1]} cfg={cfg_kwargs['d_model']}d/"
                      f"{cfg_kwargs['num_heads']}h seq={cfg_kwargs['seq_len']} causal={causal}")
                overall &= run_case(kp, cfg_kwargs, causal, padding_ratio=0.35)

    print("RESULT:", "ALL PASS" if overall else "FAILURES PRESENT")
    sys.exit(0 if overall else 1)
