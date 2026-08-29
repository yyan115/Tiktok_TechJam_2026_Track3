#!/usr/bin/env python3
"""Score-sensitivity board (converged outer-loop plan, revision 3).

Reads BOTH evidence sources — the frozen runner's journal (authoritative for
shapes it can run) and the pinned side evaluator's packets (shapes 6/14) —
and reports every scoring convention side by side, so whichever convention
the organizers confirm is already computed:

  raw latency · useful TF/s · fp32-equivalent throughput ratio ·
  fp16-peak MFU · speedup (where a baseline exists)

Score scenarios: S1 equal-weight roofline-relative (proxy: fp16-peak MFU),
S2 equal-weight absolute fp16-MFU, S3 FLOP-weighted absolute fp16-MFU.
Output: Project/results_side/SENSITIVITY.md. Read-only over its inputs;
never writes runner-owned files.
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "Project"
JOURNAL = PROJECT / "results" / "JOURNAL.jsonl"
SIDE = PROJECT / "results_side"
SHAPES = {s["id"]: s for s in json.loads((PROJECT / "shapes.json").read_text())["shapes"]}

FP32_PEAK_TF = 16.2
FP16_PEAK_TF = 32.5


def model_flops(s):
    B, S, d = s["batch_size"], s["seq_len"], s["d_model"]
    ffn, L = s["ffn_dim"], s["num_layers"]
    att = 0.5 if s["causal"] else 1.0
    return L * (8.0*B*S*d*d + att*4.0*B*S*S*d + 4.0*B*S*d*ffn)


def main():
    rows = {}
    if JOURNAL.exists():
        entries = [json.loads(l) for l in JOURNAL.read_text().splitlines() if l.strip()]
        for e in entries:
            if e.get("type") != "candidate" or not e.get("promoted"):
                continue
            sid = e["shape_id"]
            ms = e["timing"]["candidate"]["median_ms"]
            cur = rows.get(sid)
            if cur is None or ms < cur["ms"]:
                rows[sid] = {"ms": ms, "impl": e["impl"]["name"],
                             "speedup": e["timing"]["speedup"],
                             "source": "frozen runner journal", "entry": e["entry_id"]}
    for p in sorted(SIDE.glob("shape6_*.json")) + sorted(SIDE.glob("shape14_*.json")):
        d = json.loads(p.read_text())
        if d.get("type") == "shape6_local_candidate_evidence" and d["correctness"]["violations"] == 0:
            sid, ms = 6, d["timing"]["median_ms"]
        elif d.get("type") == "shape14_side_evaluation" and d["correctness"]["passed"]:
            sid = 14
            frac = d["shape"]["batch_size"] / 32
            ms = d["timing"]["median_ms"] / frac  # scaled to full B=32 (labeled)
        else:
            continue
        cur = rows.get(sid)
        if cur is None or ms < cur["ms"]:
            rows[sid] = {"ms": ms, "impl": d["candidate"]["name"], "speedup": None,
                         "source": f"side evaluator ({p.name})"
                                   + (" — batch-scaled projection" if sid == 14 else ""),
                         "entry": p.name}

    lines = ["# Score-sensitivity board (auto-generated; do not edit)",
             "",
             f"Generated {time.strftime('%F %T')} · peaks: fp32 {FP32_PEAK_TF} TF, fp16(fp32acc) {FP16_PEAK_TF} TF (RTX 3060 Ti)",
             "",
             "| shape | impl | median ms | useful TF/s | fp32-equiv (x peak) | fp16 MFU | speedup | source |",
             "|---|---|---|---|---|---|---|---|"]
    s1 = s2 = s3 = 0.0
    wsum = 0.0
    n = 0
    for sid in sorted(SHAPES):
        s = SHAPES[sid]
        fl = model_flops(s)
        r = rows.get(sid)
        if not r:
            lines.append(f"| {sid} | — | — | — | — | — | — | no evidence |")
            continue
        tfs = fl / (r["ms"] / 1e3) / 1e12
        mfu16 = tfs / FP16_PEAK_TF
        eq32 = tfs / FP32_PEAK_TF
        sp = f"{r['speedup']:.2f}x" if r["speedup"] else "n/a (no runnable baseline)"
        lines.append(f"| {sid} | {r['impl']} | {r['ms']:.3f} | {tfs:.2f} | {eq32:.2f} | {mfu16:.3f} | {sp} | {r['source']} |")
        s1 += mfu16
        s2 += mfu16
        s3 += mfu16 * fl
        wsum += fl
        n += 1
    lines += ["",
              f"S1/S2 equal-weight mean fp16-MFU over {n} evidenced shapes: {s1/max(n,1):.3f}",
              f"S3 FLOP-weighted fp16-MFU: {s3/max(wsum,1):.3f}",
              "",
              "Notes: shape-14 row is a batch-scaled projection from the local",
              "B-slice packet until the rental full-scale packet lands; shape-6",
              "and shape-14 rows have no baseline speedup because the official",
              "dense baseline cannot run those shapes (limitation stated in the",
              "packets). Roofline-relative scoring (if confirmed) needs the",
              "organizers' bandwidth term — fp16-MFU stands as proxy."]
    out = SIDE / "SENSITIVITY.md"
    SIDE.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(str(out))


if __name__ == "__main__":
    main()
