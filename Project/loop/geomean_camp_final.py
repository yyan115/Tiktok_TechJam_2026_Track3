"""Board arithmetic for CAMP-FINAL.

Three columns, all measured on the SHIPPED submission file rather than on a
kernel module, each row from one screening-lane permit on a verified-quiet box
with `correct: true` on all seven trials.

  orig   sha 4da76db6...  the 9.45x board
  split  sha 54057a33...  split-head attention (k017 integrated)
  final  sha 418952bf...  QKV chunk promoted to a grid dimension, gated on
                          d_model >= 64 AND tokens <= 4096, plus single-pass
                          LayerNorm statistics

PROVENANCE CAVEAT, and it is why `final_measured_on` exists. Two shapes
exhausted their family attempt budget (8/8) before the final gate was written,
so their numbers were measured on an earlier build:

  shape 2  measured on 599f5dad. The gate routes shape 2 identically on both
           builds (d_model 128 >= 64, 128 tokens <= 4096 -> split), so the
           number transfers by argument. Argument is weaker than measurement.
  shape 11 measured on 599f5dad on the SPLIT path. The final gate routes it
           UNSPLIT (8192 tokens > 4096), so the number does NOT transfer. It is
           expected to recover toward its 17.6588 pre-split value, and the
           board deliberately carries the pessimistic measured figure instead.

Every other row is measured on 418952bf.
"""
import math

# shape: (orig, split, final, final_measured_on)
BOARD = {
    13: (28.2849, 30.89893097938634, 31.130462515181932, "418952bf"),
    7:  (20.9595, 23.860284435392,   23.871212023497815, "599f5dad"),
    2:  (13.1434, 15.53152620870717, 19.26923001580322,  "599f5dad"),
    11: (12.5909, 17.658779395091752, 17.42299255849099, "599f5dad"),
    3:  (12.9618, 12.149999890860638, 14.073210921212526, "301d7063"),
    12: (10.4348, 10.542683163806476, 10.603658529010753, "599f5dad"),
    4:  (8.9211,  10.068965205785407, 10.608835906164902, "599f5dad"),
    5:  (9.1150,  9.30502533934481,   9.329980166061654,  "418952bf"),
    1:  (8.1673,  8.382979171358198,  8.52165997313622,   "418952bf"),
    10: (6.5352,  6.29746065899287,   6.3756664674027785, "418952bf"),
    9:  (4.3503,  4.5459185688676325, 4.5883362633518185, "418952bf"),
    8:  (2.0160,  2.0215504169401375, 2.0192775459548913, "418952bf"),
}

# Tokens per forward, the variable the QKV split gain orders by.
TOKENS = {2: 128, 3: 512, 4: 2048, 12: 2048, 1: 8192, 7: 8192, 9: 8192,
          10: 8192, 11: 8192, 5: 16384, 13: 65536, 8: 8192}
UNTOUCHED = 8   # fp16 branch: no line of this campaign touches it


def geo(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main():
    print(f"{'shape':>6} {'orig':>9} {'split':>9} {'final':>9} "
          f"{'vs orig':>9} {'tokens':>7}  measured on")
    for s in sorted(BOARD, key=lambda k: -BOARD[k][2]):
        o, sp, f, on = BOARD[s]
        flag = "" if on == "418952bf" else "  <- not the final build"
        print(f"{s:>6} {o:>9.4f} {sp:>9.4f} {f:>9.4f} "
              f"{100 * (f / o - 1):>+8.1f}% {TOKENS[s]:>7}  {on}{flag}")

    every = list(BOARD)
    for label, idx in (("orig  ", 0), ("split ", 1), ("final ", 2)):
        print(f"\ngeomean {label} {geo([BOARD[s][idx] for s in every]):.4f}", end="")
    go = geo([BOARD[s][0] for s in every])
    gf = geo([BOARD[s][2] for s in every])
    gs = geo([BOARD[s][1] for s in every])
    print(f"\n\nfinal vs orig   {100 * (gf / go - 1):+.2f}%")
    print(f"final vs split  {100 * (gf / gs - 1):+.2f}%")

    print("\nQKV split gain against token count (the gate's ordering variable):")
    for s in sorted(BOARD, key=lambda k: TOKENS[k]):
        if s == UNTOUCHED:
            continue
        sp, f = BOARD[s][1], BOARD[s][2]
        print(f"  {TOKENS[s]:>6} tokens  shape {s:<3} {100 * (f / sp - 1):>+6.1f}%")

    o8, f8 = BOARD[UNTOUCHED][1], BOARD[UNTOUCHED][2]
    print(f"\nnull control (shape 8, fp16 branch, untouched): "
          f"{100 * (f8 / o8 - 1):+.2f}%")


if __name__ == "__main__":
    raise SystemExit(main())
