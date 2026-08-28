"""Correctness-only smoke for k007 (fused-block megakernel), covering every
runnable-shape dim class: d=128 with heads 1/2/4/16 (head_dim 128/64/32/8),
d=32 (shape 7), seq=32 (shape 12), seq=1024 (shape 13). Causal always (the
fused path requires it; non-causal configs exercise the eager fallback).
NOT a benchmark."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from padded_mask_smoke import run_case, REPO  # noqa: E402

CASES = [
    dict(batch_size=8, seq_len=128, d_model=128, num_heads=4, ffn_dim=128, num_layers=4),
    dict(batch_size=8, seq_len=128, d_model=128, num_heads=1, ffn_dim=128, num_layers=4),
    dict(batch_size=8, seq_len=128, d_model=128, num_heads=2, ffn_dim=128, num_layers=4),
    dict(batch_size=8, seq_len=128, d_model=128, num_heads=16, ffn_dim=128, num_layers=4),
    dict(batch_size=8, seq_len=128, d_model=32, num_heads=4, ffn_dim=32, num_layers=4),
    dict(batch_size=8, seq_len=32, d_model=128, num_heads=4, ffn_dim=128, num_layers=4),
    dict(batch_size=2, seq_len=1024, d_model=128, num_heads=4, ffn_dim=128, num_layers=4),
]

kp = f"{REPO}/Project/kernels/k007_fused_block.py"
overall = True
for cfg_kwargs in CASES:
    for causal in (True, False):
        print(f"k007 d={cfg_kwargs['d_model']} h={cfg_kwargs['num_heads']} "
              f"seq={cfg_kwargs['seq_len']} causal={causal}")
        overall &= run_case(kp, cfg_kwargs, causal, padding_ratio=0.35)

print("RESULT:", "ALL PASS" if overall else "FAILURES PRESENT")
sys.exit(0 if overall else 1)
