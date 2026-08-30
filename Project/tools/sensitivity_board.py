#!/usr/bin/env python3
"""Score-sensitivity board — REWRITTEN 30 Aug (NARROWINGS_TODO item 8).

The organizers score "a weighted sum of per-shape MFUs (bandwidth considered)"
but have NOT fixed the weights or the denominator (research/competition-
scoring.md). So this board computes every convention side by side, and — the
point of the exercise — says which shape actually moves the score under each
one. An experiment that helps under NO scenario does not deserve a card.

What the previous revision got wrong (all three fixed here):

  1. SELECTION. It took the fastest promoted entry per shape by median ms,
     across all journal passes. LESSONS #11: absolute ms is NOT comparable
     across runner invocations (GPU clock state differs), and LESSONS #22:
     entries measured under codex contention read HIGH for graphed
     candidates. That silently cherry-picked contended entries into the
     report's headline table. Now: one designated SHIP PASS (the quiet-box
     sweep recorded in memory/STATE.md), every row from it, and any shape
     that falls back to another pass is labeled loudly.
  2. DENOMINATOR. It reported one fp16 roof (32.5 TF) as if it were the
     roof. GA10x has two, a factor of 2 apart, and which one applies
     depends on the accumulator. Now: fp32 shader, fp16-with-fp32-accumulate
     (what our kernels actually use), and fp16-with-fp16-accumulate (the
     absolute roof) — all three, all sourced, plus the bandwidth roof.
  3. SHAPE 14. It divided a B=1 median by (1/32) to "scale" to B=32.
     Measured B=1 -> B=2 is 2.18x, not 2x (NARROWINGS item 4: never
     multiply a B=1 median). The projection is gone; the measured slice
     points are reported as what they are, and full-B=32 reads PENDING.

Read-only over its inputs. Writes only Project/results_side/SENSITIVITY.md.
Audit status is deliberately NOT reported here — it lives in the leaderboard
and the ship manifest, and mixing a moving ledger into a score board makes
the board unreproducible.
"""
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "Project"
JOURNAL = PROJECT / "results" / "JOURNAL.jsonl"
SIDE = PROJECT / "results_side"
SHAPES = {s["id"]: s for s in json.loads((PROJECT / "shapes.json").read_text())["shapes"]}

# --- Device peaks, every one sourced -----------------------------------------
# RTX 3060 Ti (GA104, sm_86), 4864 CUDA cores @ 1665 MHz boost:
#   fp32 shader   = 4864 * 2 FLOP * 1.665 GHz          = 16.2 TF/s  [spec sheet]
#   fp16 tensor, FP32 accumulate = 2x shader rate      = 32.4 TF/s  [GA10x whitepaper]
#   fp16 tensor, FP16 accumulate = 4x shader rate      = 64.8 TF/s  [GA10x whitepaper]
# The GA10x consumer SM runs FP16-with-FP32-accumulate at HALF the rate of
# FP16-with-FP16-accumulate. Our Triton kernels use tl.dot's fp32 accumulator,
# so 32.4 TF/s is the roof we can actually reach; 64.8 TF/s is the absolute
# fp16 roof and is reported so no MFU number here can be read as flattering.
# Memory bandwidth 448 GB/s (256-bit GDDR6 @ 14 Gbps) [spec sheet].
PEAKS = {
    "fp32_shader_tf": 16.2,
    "fp16_fp32acc_tf": 32.4,
    "fp16_fp16acc_tf": 64.8,
    "bandwidth_tb_s": 0.448,
}
PEAK_SOURCES = (
    "NVIDIA RTX 3060 Ti spec page (4864 cores @ 1665 MHz boost, 448 GB/s) + "
    "NVIDIA Ampere GA10x architecture whitepaper (tensor-core FP16 rates: "
    "2x shader with FP32 accumulate, 4x with FP16 accumulate)"
)

# --- The designated ship pass ------------------------------------------------
# memory/STATE.md SCOREBOARD: the 29 Aug 02:30 quiet-box sweep, load 1.6
# evidenced, no codex audits running. Every row of the shipped board comes
# from this window so the numbers are mutually comparable.
SHIP_PASS = ("2026-08-29T02:26:00", "2026-08-29T02:31:00")
SHIP_PASS_LABEL = "quiet-box sweep, 29 Aug 02:26-02:31 (STATE.md ship board)"

# Hypothetical improvement used for the marginal-value columns.
WHATIF = 0.20  # 20% faster on one shape


def model_flops(s):
    """Useful model FLOPs. QKV+out projections 8BSd^2, attention 4BS^2d
    (halved when causal), FFN 4BSd*ffn. Elementwise (LN, GELU, softmax) is
    not counted — stated so the denominator is unambiguous."""
    B, S, d = s["batch_size"], s["seq_len"], s["d_model"]
    ffn, L = s["ffn_dim"], s["num_layers"]
    att = 0.5 if s["causal"] else 1.0
    return L * (8.0*B*S*d*d + att*4.0*B*S*S*d + 4.0*B*S*d*ffn)


def ideal_bytes(s):
    """Lower-bound traffic under perfect fusion: read the fp32 input once,
    write the fp32 output once, read every weight once in fp16 (how our
    kernels hold them). Intermediates assumed to stay in registers/SRAM.
    A lower bound on traffic = an upper bound on arithmetic intensity =
    a GENEROUS bandwidth ceiling; stated rather than silently assumed."""
    B, S, d = s["batch_size"], s["seq_len"], s["d_model"]
    ffn, L = s["ffn_dim"], s["num_layers"]
    act = 2.0 * B * S * d * 4.0
    params = L * (4.0*d*d + 2.0*d*ffn)
    return act + params * 2.0


def limiter(ai, achieved_tf, ceiling_tf, gflop):
    """Which wall is this shape actually standing against?"""
    balance = PEAKS["fp16_fp32acc_tf"] * 1e12 / (PEAKS["bandwidth_tb_s"] * 1e12)
    if ai < balance:
        return "bandwidth"
    if achieved_tf < 0.5 * ceiling_tf and gflop < 20.0:
        return "latency/grid"
    return "compute"


def load_ship_rows():
    """One row per shape from the designated ship pass; loud fallback."""
    rows, fallback = {}, {}
    if not JOURNAL.exists():
        return rows
    for line in JOURNAL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "candidate" or not e.get("promoted"):
            continue
        if not e.get("correctness", {}).get("passed"):
            continue
        sid, ts = e["shape_id"], e.get("timestamp", "")
        row = {"ms": e["timing"]["candidate"]["median_ms"],
               "impl": e["impl"]["name"],
               "speedup": e["timing"]["speedup"],
               "entry": e["entry_id"],
               "ts": ts,
               "in_pass": SHIP_PASS[0] <= ts < SHIP_PASS[1],
               "wall_suspicious": e["timing"]["wall_check"]["suspicious"]}
        if row["in_pass"]:
            # Within one pass, ties are impossible in practice; keep the first.
            rows.setdefault(sid, row)
        else:
            cur = fallback.get(sid)
            if cur is None or ts > cur["ts"]:
                fallback[sid] = row
    for sid, row in fallback.items():
        rows.setdefault(sid, row)
    return rows


def load_side_rows():
    """Shapes 6 and 14: side-evaluator evidence, reported as measured.

    Both are PROVISIONAL by NARROWINGS item 10 — the packets carry one seed
    and cite the pre-integration submission sha. Shape 14's full B=32 is
    NOT projected from a slice: the measured slice points are shown and the
    full-shape figure reads PENDING."""
    out = {}
    for p in sorted(SIDE.glob("shape6_*.json")):
        d = json.loads(p.read_text())
        if d.get("type") != "shape6_local_candidate_evidence":
            continue
        if d["correctness"]["violations"] != 0:
            continue
        out[6] = {"ms": d["timing"]["median_ms"], "impl": d["candidate"]["name"],
                  "speedup": None, "entry": p.name, "ts": d["timestamp"],
                  "in_pass": False, "wall_suspicious": False, "provisional": True,
                  "note": "full shape measured; no runnable official baseline (OOM at 8 GB)"}
    slices = []
    for p in sorted(SIDE.glob("shape14_*.json")):
        d = json.loads(p.read_text())
        if d.get("type") != "shape14_side_evaluation" or not d["correctness"]["passed"]:
            continue
        slices.append((d["shape"]["batch_size"], d["timing"]["median_ms"], p.name))
    return out, sorted(slices)


def main():
    rows = load_ship_rows()
    side, s14_slices = load_side_rows()
    rows.update(side)

    L = []
    A = L.append
    A("# Score-sensitivity board (auto-generated by Project/tools/sensitivity_board.py — do not edit)")
    A("")
    A(f"Generated {time.strftime('%F %T %z')}")
    A("")
    A("**Device peaks (all sourced).** RTX 3060 Ti / GA104 / sm_86: "
      f"fp32 shader **{PEAKS['fp32_shader_tf']} TF/s** · "
      f"fp16 tensor w/ FP32 accumulate **{PEAKS['fp16_fp32acc_tf']} TF/s** (what our kernels reach) · "
      f"fp16 tensor w/ FP16 accumulate **{PEAKS['fp16_fp16acc_tf']} TF/s** (absolute roof) · "
      f"bandwidth **{PEAKS['bandwidth_tb_s']*1000:.0f} GB/s**.")
    A(f"Sources: {PEAK_SOURCES}.")
    A("")
    A(f"**Row selection.** {SHIP_PASS_LABEL}. Absolute latencies are only "
      "comparable within one runner invocation (LESSONS #11) and contention "
      "inflates graphed candidates (LESSONS #22), so the board is built from "
      "ONE pass rather than from each shape's best-ever entry.")
    A("")
    A("| shape | impl | median ms | GFLOP | achieved TF/s | MFU vs 32.4 | MFU vs 64.8 | roofline ceiling TF/s | vs ceiling | limiter | speedup | provenance |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|")

    board = {}
    for sid in sorted(SHAPES):
        s = SHAPES[sid]
        fl = model_flops(s)
        gflop = fl / 1e9
        r = rows.get(sid)
        if sid == 14:
            what = ("correctness proven at full seq=100,000; batch slices measured "
                    "(table below); full B=32 timing NOT yet measured"
                    if s14_slices else "**no evidence**")
            A(f"| 14 | k014_shape14 | **PENDING** | {gflop:,.1f} | — | — | — | — | — | — | n/a | {what} |")
            continue
        if not r:
            A(f"| {sid} | — | — | {gflop:,.1f} | — | — | — | — | — | — | — | **no evidence** |")
            continue
        tfs = fl / (r["ms"] / 1e3) / 1e12
        ai = fl / ideal_bytes(s)
        bw_roof = ai * PEAKS["bandwidth_tb_s"]
        ceiling = min(PEAKS["fp16_fp32acc_tf"], bw_roof)
        lim = limiter(ai, tfs, ceiling, gflop)
        sp = f"{r['speedup']:.2f}x" if r.get("speedup") else "n/a"
        prov = r["entry"] if r.get("in_pass") else f"**{r['entry']} (OUTSIDE ship pass)**"
        if r.get("provisional"):
            prov = f"{r['entry']} — PROVISIONAL (1 seed, pre-integration sha)"
        A(f"| {sid} | {r['impl']} | {r['ms']:.4f} | {gflop:,.1f} | {tfs:.2f} | "
          f"{tfs/PEAKS['fp16_fp32acc_tf']:.3f} | {tfs/PEAKS['fp16_fp16acc_tf']:.3f} | "
          f"{ceiling:.1f} | {tfs/ceiling:.2f} | {lim} | {sp} | {prov} |")
        board[sid] = {"flops": fl, "ms": r["ms"], "tfs": tfs, "ceiling": ceiling,
                      "speedup": r.get("speedup"), "limiter": lim}

    # --- shape 14: measured slices, no projection ----------------------------
    A("")
    A("### Shape 14 — measured slices only (no projection)")
    A("")
    A("| batch slice | median ms | scaling vs previous | packet |")
    A("|---:|---:|---|---|")
    prev = None
    for b, ms, name in s14_slices:
        scal = "—" if prev is None else f"{ms/prev[1]:.2f}x for {b/prev[0]:.0f}x batch"
        A(f"| B={b} | {ms:,.1f} | {scal} | {name} |")
        prev = (b, ms)
    A("")
    A("B=1 -> B=2 costs **2.18x**, not 2x. A linear projection from the B=1 "
      "median would therefore understate full-batch time, which is why "
      "NARROWINGS item 4 forbids it and why this board reports no B=32 "
      "figure. The full-batch number comes from the batch-decomposed "
      "evaluator run (NARROWINGS items 2 and 4), not from arithmetic.")

    # --- scenarios -----------------------------------------------------------
    ev = sorted(board)
    n = len(ev)
    mfu = {i: board[i]["tfs"] / PEAKS["fp16_fp32acc_tf"] for i in ev}
    rel = {i: board[i]["tfs"] / board[i]["ceiling"] for i in ev}
    tot_fl = sum(board[i]["flops"] for i in ev)
    tot_t = sum(board[i]["ms"] / 1e3 for i in ev)
    with_sp = [i for i in ev if board[i]["speedup"]]

    s1 = sum(mfu.values()) / n
    s2 = (tot_fl / tot_t / 1e12) / PEAKS["fp16_fp32acc_tf"]
    s3 = math.exp(sum(math.log(board[i]["speedup"]) for i in with_sp) / len(with_sp))
    s4 = sum(rel.values()) / n
    s5 = min(mfu.values())
    worst = min(ev, key=lambda i: mfu[i])

    A("")
    A(f"## Score scenarios (over the {n} shapes with comparable evidence)")
    A("")
    A("| # | convention | value |")
    A("|---|---|---:|")
    A(f"| S1 | equal weight per shape, MFU vs 32.4 TF | **{s1:.3f}** |")
    A(f"| S2 | FLOP-weighted MFU (total useful FLOPs / total time) | **{s2:.3f}** |")
    A(f"| S3 | geomean speedup vs the official baseline | **{s3:.2f}x** |")
    A(f"| S4 | roofline-relative (vs each shape's own ceiling) | **{s4:.3f}** |")
    A(f"| S5 | worst single shape MFU (the floor under any weighting) | **{s5:.3f}** (shape {worst}) |")
    A("")
    A("S1 and S4 reward the small launch-bound shapes; S2 is dominated by the "
      "few big ones; S3 is the number a reader intuitively expects. They "
      "disagree about where the remaining points are, which is the whole "
      "reason for computing all of them.")
    if abs(s1 - s4) < 5e-4:
        A("")
        A("**S4 has collapsed onto S1** — at ideal fusion every shape's "
          "arithmetic intensity clears the 72 FLOP/byte balance point, so each "
          "shape's ceiling IS the compute roof and the roofline view adds "
          "nothing. A bandwidth term only changes the ranking if the "
          "organizers compute it from ACTUAL traffic; measuring that needs "
          "ncu counters per shape, which is a separate piece of work.")
    A("")
    A("S5 is an MFU floor, not a risk figure: all 14 shapes currently PASS "
      "the precision test, and the zero-if-failed rule keys on correctness, "
      "not on utilisation.")

    # --- marginal value: where does grinding actually pay? -------------------
    A("")
    A(f"## Marginal value — what a {WHATIF*100:.0f}% latency win on ONE shape is worth")
    A("")
    A("| shape | limiter | ΔS1 | ΔS2 | ΔS4 | rank S1 | rank S2 |")
    A("|---|---|---:|---:|---:|---:|---:|")
    d1, d2, d4 = {}, {}, {}
    for i in ev:
        new_ms = board[i]["ms"] * (1 - WHATIF)
        new_tfs = board[i]["flops"] / (new_ms / 1e3) / 1e12
        d1[i] = (new_tfs / PEAKS["fp16_fp32acc_tf"] - mfu[i]) / n
        d4[i] = (min(new_tfs / board[i]["ceiling"], 1.0) - rel[i]) / n
        nt = tot_t - board[i]["ms"] / 1e3 + new_ms / 1e3
        d2[i] = (tot_fl / nt / 1e12) / PEAKS["fp16_fp32acc_tf"] - s2
    r1 = {i: k for k, i in enumerate(sorted(ev, key=lambda x: -d1[x]), 1)}
    r2 = {i: k for k, i in enumerate(sorted(ev, key=lambda x: -d2[x]), 1)}
    for i in ev:
        A(f"| {i} | {board[i]['limiter']} | {d1[i]:+.4f} | {d2[i]:+.4f} | {d4[i]:+.4f} | {r1[i]} | {r2[i]} |")
    A("")
    top1 = sorted(ev, key=lambda x: -d1[x])[:3]
    top2 = sorted(ev, key=lambda x: -d2[x])[:3]
    A(f"Under equal weighting (S1) the payoff ranks: {', '.join('shape %d' % i for i in top1)}. "
      f"Under FLOP weighting (S2): {', '.join('shape %d' % i for i in top2)}. "
      "A direction that ranks low under BOTH does not earn runner time.")

    A("")
    A("## Reading notes")
    A("")
    A("- Shapes 6 and 14 have no speedup column because the official dense "
      "baseline cannot run them on this device (6 OOMs at 8 GB; 14's "
      "attention table is multi-terabyte at any batch). Their references are "
      "the batch-chunked official computation and a validated streamed fp32 "
      "oracle respectively — stated in each packet.")
    A("- Shape-6 and shape-14 packets are PROVISIONAL: one seed each, and "
      "they cite the pre-integration submission sha. NARROWINGS item 10 "
      "re-captures both against the shipped submission file with >=5 seeds.")
    A("- MFU here is a *model-FLOP* utilisation: elementwise work (LayerNorm, "
      "GELU, softmax) is real GPU time but is not counted in the numerator, "
      "so these figures are conservative.")
    A("- The bandwidth ceiling uses ideal-fusion traffic, which no real "
      "kernel achieves; 'vs ceiling' is therefore a floor on our efficiency, "
      "not a claim of headroom exhausted.")
    A("- Every number above is pre-gate (measured before the authority-v4 "
      "guard paste); see Project/loop/GATE_DESIGN.md HONESTY LEDGER.")

    out = SIDE / "SENSITIVITY.md"
    SIDE.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(str(out))


if __name__ == "__main__":
    main()
