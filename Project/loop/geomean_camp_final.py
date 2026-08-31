"""Board arithmetic for CAMP-FINAL: the split-head submission against the
previous submission file, both measured on the shipped artifact itself.

Old column: the twelve rows of the 9.45x board (submission sha 4da76db6...).
New column: the same twelve shapes on the rebuilt file (sha 54057a33...),
each from one screening-lane permit on a verified-quiet box, correct: true.

Every number here is a paired event speedup: candidate against baseline inside
one invocation. Comparing the two columns crosses a session boundary, which is
why the per-shape deltas are weaker evidence than the geomean.
"""
import math

# shape: (old shipped-file speedup, new shipped-file speedup)
BOARD = {
    13: (28.2849, 30.89893097938634),
    7:  (20.9595, 23.860284435392),
    11: (12.5909, 17.658779395091752),
    2:  (13.1434, 15.53152620870717),
    3:  (12.9618, 12.149999890860638),
    12: (10.4348, 10.542683163806476),
    4:  (8.9211,  10.068965205785407),
    5:  (9.1150,  9.30502533934481),
    1:  (8.1673,  8.382979171358198),
    10: (6.5352,  6.29746065899287),
    9:  (4.3503,  4.5459185688676325),
    8:  (2.0160,  2.0215504169401375),
}

# Shapes whose head_dim is 8, where Triton pads every tl.dot to 16 wide.
PADDED = {7, 11}


def geo(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main():
    print(f"{'shape':>6} {'old':>10} {'new':>10} {'delta %':>9}  note")
    for s in sorted(BOARD, key=lambda k: -BOARD[k][1]):
        o, n = BOARD[s]
        note = "head_dim 8" if s in PADDED else ("fp16 branch, untouched"
                                                if s == 8 else "")
        print(f"{s:>6} {o:>10.4f} {n:>10.4f} {100 * (n / o - 1):>+9.1f}  {note}")

    old = [o for o, _ in BOARD.values()]
    new = [n for _, n in BOARD.values()]
    go, gn = geo(old), geo(new)
    print()
    print(f"geomean, 12 shapes   old {go:.4f}   new {gn:.4f}   "
          f"{100 * (gn / go - 1):+.2f}%")

    fused = [s for s in BOARD if s != 8]
    fo = geo([BOARD[s][0] for s in fused])
    fn = geo([BOARD[s][1] for s in fused])
    print(f"geomean, 11 fused    old {fo:.4f}   new {fn:.4f}   "
          f"{100 * (fn / fo - 1):+.2f}%")

    pad = geo([BOARD[s][1] for s in PADDED]) / geo([BOARD[s][0] for s in PADDED])
    rest = [s for s in fused if s not in PADDED]
    other = geo([BOARD[s][1] for s in rest]) / geo([BOARD[s][0] for s in rest])
    print()
    print(f"head_dim 8 shapes (7, 11)      {100 * (pad - 1):+.1f}%")
    print(f"other nine fused shapes        {100 * (other - 1):+.1f}%")

    d = sorted(100 * (n / o - 1) for o, n in BOARD.values())
    print()
    print(f"per-shape deltas: min {d[0]:+.1f}  max {d[-1]:+.1f}  "
          f"median {d[len(d) // 2]:+.1f}  positive {sum(x > 0 for x in d)}/12")


if __name__ == "__main__":
    raise SystemExit(main())
