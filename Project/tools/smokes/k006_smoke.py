"""Correctness-only smoke for k006 (head_dim 128/256 Triton attention).

Shape-9-like (heads=1, head_dim=128), shape-8-like (d=1024, head_dim=256),
and a head_dim<=64 regression case, each causal and padded. NOT a benchmark.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from padded_mask_smoke import run_case, REPO  # noqa: E402

CASES = [
    # shape-9-like: single head, head_dim 128
    dict(batch_size=8, seq_len=128, d_model=128, num_heads=1, ffn_dim=128, num_layers=4),
    # shape-8-like: head_dim 256
    dict(batch_size=4, seq_len=128, d_model=1024, num_heads=4, ffn_dim=1024, num_layers=2),
    # regression: head_dim 32 (existing fast path with new config list)
    dict(batch_size=4, seq_len=128, d_model=128, num_heads=4, ffn_dim=128, num_layers=4),
]

kp = f"{REPO}/Project/kernels/k006_fp16_hd128.py"
overall = True
for cfg_kwargs in CASES:
    for causal in (False, True):
        print(f"k006 d={cfg_kwargs['d_model']} h={cfg_kwargs['num_heads']} "
              f"(head_dim {cfg_kwargs['d_model']//cfg_kwargs['num_heads']}) causal={causal}")
        overall &= run_case(kp, cfg_kwargs, causal, padding_ratio=0.35)

print("RESULT:", "ALL PASS" if overall else "FAILURES PRESENT")
sys.exit(0 if overall else 1)
