"""Correctness-only smoke for k008 (W8A8 int8 GEMM stack) — the tolerance
question is the whole game here, so check the exact official criterion
(abs 2e-3 OR rel 2%) on shape-8-like and d=512 configs, causal and padded.
NOT a benchmark."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from padded_mask_smoke import run_case, REPO  # noqa: E402

CASES = [
    # shape-8-like (head_dim 256)
    dict(batch_size=8, seq_len=128, d_model=1024, num_heads=4, ffn_dim=1024, num_layers=4),
    # mid-size sanity (head_dim 128)
    dict(batch_size=4, seq_len=128, d_model=512, num_heads=4, ffn_dim=2048, num_layers=2),
]

kp = f"{REPO}/Project/kernels/k008_int8_gemm.py"
overall = True
for cfg_kwargs in CASES:
    for causal in (True, False):
        print(f"k008 d={cfg_kwargs['d_model']} h={cfg_kwargs['num_heads']} "
              f"ffn={cfg_kwargs['ffn_dim']} causal={causal}")
        overall &= run_case(kp, cfg_kwargs, causal, padding_ratio=0.35)

print("RESULT:", "ALL PASS" if overall else "FAILURES PRESENT")
sys.exit(0 if overall else 1)
