# Score-sensitivity board (auto-generated; do not edit)

Generated 2026-08-29 04:35:45 · peaks: fp32 16.2 TF, fp16(fp32acc) 32.5 TF (RTX 3060 Ti)

| shape | impl | median ms | useful TF/s | fp32-equiv (x peak) | fp16 MFU | speedup | source |
|---|---|---|---|---|---|---|---|
| 1 | k007_fused_block | 0.579 | 12.99 | 0.80 | 0.400 | 8.39x | frozen runner journal |
| 2 | k009_fused_tuned | 0.128 | 0.92 | 0.06 | 0.028 | 17.05x | frozen runner journal |
| 3 | k009_fused_tuned | 0.133 | 3.53 | 0.22 | 0.109 | 15.77x | frozen runner journal |
| 4 | k009_fused_tuned | 0.195 | 9.63 | 0.59 | 0.296 | 11.65x | frozen runner journal |
| 5 | k007_fused_block | 1.037 | 14.49 | 0.89 | 0.446 | 9.24x | frozen runner journal |
| 6 | k015_shape6 | 83.823 | 14.01 | 0.86 | 0.431 | n/a (no runnable baseline) | side evaluator (shape6_20260829-042901.json) |
| 7 | k007_fused_block | 0.145 | 4.62 | 0.28 | 0.142 | 22.41x | frozen runner journal |
| 8 | k010_fused_ln | 19.382 | 21.72 | 1.34 | 0.668 | 2.02x | frozen runner journal |
| 9 | k009_fused_tuned | 0.573 | 13.11 | 0.81 | 0.403 | 5.63x | frozen runner journal |
| 10 | k007_fused_block | 0.557 | 13.49 | 0.83 | 0.415 | 6.63x | frozen runner journal |
| 11 | k007_fused_block | 0.915 | 8.21 | 0.51 | 0.253 | 12.95x | frozen runner journal |
| 12 | k009_fused_tuned | 0.163 | 10.30 | 0.64 | 0.317 | 15.03x | frozen runner journal |
| 13 | k009_fused_tuned | 5.972 | 20.14 | 1.24 | 0.620 | 29.12x | frozen runner journal |
| 14 | k014_shape14 | 53582.332 | 25.96 | 1.60 | 0.799 | n/a (no runnable baseline) | side evaluator (shape14_20260829-042452_B1_S100000.json) — batch-scaled projection |

S1/S2 equal-weight mean fp16-MFU over 14 evidenced shapes: 0.380
S3 FLOP-weighted fp16-MFU: 0.799

Notes: shape-14 row is a batch-scaled projection from the local
B-slice packet until the rental full-scale packet lands; shape-6
and shape-14 rows have no baseline speedup because the official
dense baseline cannot run those shapes (limitation stated in the
packets). Roofline-relative scoring (if confirmed) needs the
organizers' bandwidth term — fp16-MFU stands as proxy.
