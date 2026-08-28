# Per-shape roofline + achieved (score model draft, 29 Aug ~04:15)

RTX 3060 Ti: fp32 16.2 TF · TF32 ~16.3 TF · fp16(fp32acc) ~32.5 TF · fp16(fp16acc) ~65 TF · 448 GB/s; compute/bandwidth balance ~72 FLOP/byte (fp16acc). AI = FLOPs / ideal-fused bytes.

| shape | GFLOP | ideal MB | AI | cand ms | achieved TF | vs 32.5 TF | note |
|---|---|---|---|---|---|---|---|
| 1 | 7.52 | 9.2 | 819 | 0.6461 | 11.63 | 36% | healthy |
| 2 | 0.12 | 0.9 | 128 | 0.1444 | 0.81 | 3% | latency-bound (grid too small for peak) |
| 3 | 0.47 | 1.3 | 358 | 0.1439 | 3.27 | 10% | latency-bound (grid too small for peak) |
| 4 | 1.88 | 2.9 | 652 | 0.2580 | 7.28 | 22% | latency-bound (grid too small for peak) |
| 5 | 15.03 | 17.6 | 856 | 1.0875 | 13.82 | 43% | healthy |
| 6 | 1174.41 | 1311.5 | 895 | - | - | - | not locally runnable |
| 7 | 0.67 | 2.1 | 313 | 0.1853 | 3.62 | 11% | latency-bound (grid too small for peak) |
| 8 | 420.91 | 117.4 | 3584 | 20.1057 | 20.93 | 64% | healthy |
| 9 | 7.52 | 9.2 | 819 | 0.6113 | 12.29 | 38% | healthy |
| 10 | 7.52 | 9.2 | 819 | 0.6032 | 12.46 | 38% | healthy |
| 11 | 7.52 | 9.2 | 819 | 0.9615 | 7.82 | 24% | latency-bound (grid too small for peak) |
| 12 | 1.68 | 2.9 | 582 | 0.2038 | 8.23 | 25% | latency-bound (grid too small for peak) |
| 13 | 120.26 | 67.9 | 1771 | 6.0303 | 19.94 | 61% | healthy |
| 14 | 1391250.64 | 26239.6 | 53021 | - | - | - | not locally runnable |

Readings: every d=128 shape is COMPUTE-bound at ideal fusion (AI >> 72) — their low achieved-TF is a LATENCY/GRID problem (small kernels, few CTAs), the regime megakernels/persistent kernels address. Shapes 8/14: pure compute — fp16-acc GEMM (2x rate) is the only big remaining lever locally; shape 6 = shape-5 physics at 78x the batch (healthy grid, expect good MFU on rental).
