"""Correctness smoke for k028 (attention loop split + base-2 softmax).

Lives beside the kernel because Project/tools/ is write-denied post-LOCK; it
reuses the existing harness in Project/tools/smokes/padded_mask_smoke.py
rather than reimplementing the official criterion.

The risk in k028 is concentrated in ONE line:

    full_end = (pid_m * BLOCK_M) // BLOCK_N * BLOCK_N

If that boundary is wrong by one block, the kernel either skips key blocks or
applies no causal mask to a block that straddles the diagonal. Both are
silently wrong output, not a crash. So the cases below are chosen to break
the alignment on purpose rather than to look like the competition shapes:

  - seq_len 96 and 160 are NOT powers of two, so the q-tile count is odd and
    full_end lands mid-block for several BLOCK_M/BLOCK_N pairs.
  - seq_len 32 forces pid_m == 0 for every tile at BLOCK_M >= 32, the only
    case where stage 1 does not execute at all.
  - head_dim 8 is the tl.dot 16-wide padding case (shapes 7 and 11), where
    HD_PAD != HD and hd_mask interacts with the stage-1 loads that no longer
    carry an n_mask.
  - head_dim 64 with one head is the opposite extreme, one CTA per q-tile.

Autotune picks BLOCK_M/BLOCK_N per (SEQ, HD_PAD), so varying SEQ and head_dim
is how a caller reaches different tile pairs. causal=False and the padded
mask both exercise the eager fallback, which must be untouched.

NOT a benchmark. No timing. The runner remains the only referee for speed.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "Project", "tools", "smokes"))
from padded_mask_smoke import run_case  # noqa: E402

CASES = [
    # non-power-of-two sequence, head_dim 32: full_end lands mid-block
    dict(batch_size=4, seq_len=96, d_model=128, num_heads=4, ffn_dim=128, num_layers=4),
    # head_dim 8 -> HD_PAD 16, the shapes 7/11 padding case
    dict(batch_size=4, seq_len=128, d_model=128, num_heads=16, ffn_dim=128, num_layers=4),
    # single head, head_dim 64, odd tile count
    dict(batch_size=2, seq_len=160, d_model=64, num_heads=1, ffn_dim=64, num_layers=2),
    # one q-tile only: pid_m == 0, stage 1 never runs
    dict(batch_size=4, seq_len=32, d_model=128, num_heads=2, ffn_dim=128, num_layers=4),
    # enough tiles that stage 1 dominates
    dict(batch_size=2, seq_len=256, d_model=128, num_heads=4, ffn_dim=128, num_layers=4),
]

kp = os.path.join(REPO, "Project", "kernels", "k028_attn_fa2.py")
overall = True
for cfg_kwargs in CASES:
    for causal in (True, False):
        print(f"k028 seq={cfg_kwargs['seq_len']} d={cfg_kwargs['d_model']} "
              f"h={cfg_kwargs['num_heads']} "
              f"hd={cfg_kwargs['d_model'] // cfg_kwargs['num_heads']} "
              f"causal={causal}")
        overall &= run_case(kp, cfg_kwargs, causal, padding_ratio=0.35)

print("RESULT:", "ALL PASS" if overall else "FAILURES PRESENT")
sys.exit(0 if overall else 1)
