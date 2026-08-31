"""The physical speed limit for each of the 14 shapes, derived not asserted.

Run:  python3 Project/loop/ceiling.py

=============================================================================
WHAT PRECISION OUR KERNELS ACTUALLY USE  (read from source, not assumed)
=============================================================================
An earlier version of this script guessed that the linear layers ran in fp32
(16.2 TF/s) and only attention used tensor cores. That produced a ceiling that
shape 13 APPEARED TO BEAT -- 101% of the limit -- which is impossible and is
what proved the guess wrong.

The source settles it. `_sub_pack_fused_layer`
(Project/submission/dispatcher_region.py:765-774) casts EVERY weight with
`.half()`:  w_qkv, b_qkv, w_o, b_o, w_f1, b_f1, w_f2, b_f2. Only the LayerNorm
scale and bias stay fp32, and those carry negligible FLOPs. Activations are
cast at dispatcher_region.py:388 (`y16 = (...).to(tl.float16)`).

Every `tl.dot` therefore takes fp16 inputs. Its accumulator is fp32 -- visible
at dispatcher_region.py:394-396, where `acc` is added to a `.to(tl.float32)`
bias. fp16 in, fp32 accumulate, is 32.5 TF/s on this card, not 65.

So the ceiling is a single uniform 32.5 TF/s for every shape. This agrees with
the "vs 32.5 TF" column that Project/research/roofline-table.md has used since
29 Aug, which is the independent check that this reading is right.

=============================================================================
THE THREE LIMITS
=============================================================================
1. COMPUTE   = GFLOP / 32.5          how long the arithmetic takes at peak
2. BANDWIDTH = ideal_MB / 448        how long the data movement takes at peak
3. OCCUPANCY: a kernel launching N thread blocks can occupy at most N of the
   38 SMs. Where N < 38 the machine is partly idle by construction and no code
   can reach limits 1 or 2.

HARD FLOOR   = max(1, 2)             nothing can beat this, ever
REACHABLE    = HARD FLOOR / occupancy_fraction
                                     the best a perfect kernel could do on a
                                     shape too small to fill the chip

=============================================================================
FLOP ACCOUNTING  (stated so it can be checked)
=============================================================================
  linear per layer    = tokens * (d*3d + d*d + d*ffn + ffn*d) * 2
  attention per layer = 2 matmuls * B*H*S*S*head_dim * 2, halved for causal

Summing these must reproduce the GFLOP column of roofline-table.md. It does,
on all fourteen shapes -- that is the check that the accounting is correct.

Peaks: 32.5 TF/s and 448 GB/s from roofline-table.md's header; 38 SMs is the
RTX 3060 Ti's SM count. Both trace to NVIDIA's published specification.
"""

PEAK_TF = 32.5      # fp16 inputs, fp32 accumulate -- what every tl.dot here does
BW_GBS = 448.0      # RTX 3060 Ti memory bandwidth
SMS = 38            # RTX 3060 Ti streaming multiprocessors
BLOCK_M = 64        # representative attention tile; see the occupancy note below

# shape: (batch, seq, d_model, heads, layers, ffn, ideal_MB)
SHAPES = {
    1:  (64, 128, 128, 4, 4, 128, 9.2),
    2:  (1, 128, 128, 4, 4, 128, 0.9),
    3:  (4, 128, 128, 4, 4, 128, 1.3),
    4:  (16, 128, 128, 4, 4, 128, 2.9),
    5:  (128, 128, 128, 4, 4, 128, 17.6),
    6:  (10000, 128, 128, 4, 4, 128, 1311.5),
    7:  (64, 128, 32, 4, 4, 32, 2.1),
    8:  (64, 128, 1024, 4, 4, 1024, 117.4),
    9:  (64, 128, 128, 1, 4, 128, 9.2),
    10: (64, 128, 128, 2, 4, 128, 9.2),
    11: (64, 128, 128, 16, 4, 128, 9.2),
    12: (64, 32, 128, 4, 4, 128, 2.9),
    13: (64, 1024, 128, 4, 4, 128, 67.9),
    14: (32, 100000, 1024, 16, 2, 1024, 26239.6),
}

# Published GFLOP from roofline-table.md, for the reconciliation check.
PUBLISHED = {1: 7.52, 2: 0.12, 3: 0.47, 4: 1.88, 5: 15.03, 6: 1174.41,
             7: 0.67, 8: 420.91, 9: 7.52, 10: 7.52, 11: 7.52, 12: 1.68,
             13: 120.26, 14: 1391250.64}

# Measured medians on artifact 630a456c..., 31 Aug, quiet box, one permit per row.
# Shape 6 is candidate-only: the official baseline runs out of memory at B=10000.
OURS_MS = {1: 0.5407, 2: 0.0676, 3: 0.0840, 4: 0.1628, 5: 0.9585, 6: 70.6618,
           7: 0.1126, 8: 18.0813, 9: 0.5806, 10: 0.5509, 11: 0.6339,
           13: 5.2439}

# TikTok's own BaselineTransformer, same process, same input, same run.
BASE_MS = {1: 5.078, 2: 1.745, 3: 1.744, 4: 1.761, 5: 9.882, 7: 3.390,
           8: 43.143, 9: 2.973, 10: 3.915, 11: 12.051, 13: 169.935}


def gflop(b, s, d, h, layers, ffn):
    tokens = b * s
    lin = tokens * (d * 3 * d + d * d + d * ffn + ffn * d) * 2 * layers
    attn = 2 * b * h * s * s * (d // h) * 2 / 2 * layers      # /2 causal
    return (lin + attn) / 1e9


def main():
    print("RECONCILIATION -- computed GFLOP must equal roofline-table.md")
    bad = 0
    for s in sorted(SHAPES):
        g = gflop(*SHAPES[s][:6])
        ok = abs(g - PUBLISHED[s]) / PUBLISHED[s] < 0.005
        bad += 0 if ok else 1
        print(f"  shape {s:>2}: computed {g:>13,.2f}  published {PUBLISHED[s]:>13,.2f}"
              f"   {'match' if ok else 'MISMATCH'}")
    print(f"  -> {14 - bad}/14 reconcile\n")

    print(f"{'sh':>3} {'GFLOP':>13} {'compute':>10} {'memory':>9} {'HARD':>10} "
          f"{'occ':>5} {'REACHABLE':>10} {'ours':>10} {'vs hard':>8} {'vs reach':>9}")
    for s in sorted(SHAPES):
        b, sq, d, h, L, ffn, mb = SHAPES[s]
        g = gflop(b, sq, d, h, L, ffn)
        t_c, t_m = g / PEAK_TF, mb / BW_GBS
        hard = max(t_c, t_m)
        blocks = -(-sq // BLOCK_M) * b * h
        occ = min(1.0, blocks / SMS)
        reach = hard / occ
        o = OURS_MS.get(s)
        got = f"{o:10.4f}" if o else "   not run"
        vh = f"{hard / o:7.0%}" if o else "      --"
        vr = f"{reach / o:8.0%}" if o else "       --"
        print(f"{s:>3} {g:>13,.2f} {t_c:>10.4f} {t_m:>9.4f} {hard:>10.4f} "
              f"{occ:>5.0%} {reach:>10.4f} {got} {vh} {vr}")

    print(f"\n{'sh':>3} {'baseline ms':>12} {'base MFU':>9} {'ours ms':>10} "
          f"{'our MFU':>8} {'speedup':>8}")
    for s in sorted(SHAPES):
        o, bm = OURS_MS.get(s), BASE_MS.get(s)
        if not o:
            continue
        g = gflop(*SHAPES[s][:6])
        om = g / o / PEAK_TF
        if bm:
            print(f"{s:>3} {bm:>12.3f} {g / bm / PEAK_TF:>8.1%} {o:>10.4f} "
                  f"{om:>7.1%} {bm / o:>7.2f}x")
        else:
            print(f"{s:>3} {'OOM':>12} {'--':>8} {o:>10.4f} {om:>7.1%} {'--':>8}")


if __name__ == "__main__":
    raise SystemExit(main())
