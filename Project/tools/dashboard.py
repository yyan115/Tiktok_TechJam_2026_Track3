#!/usr/bin/env python3
"""Live project dashboard (read-only). Run:  streamlit run Project/tools/dashboard.py
Renders existing files only — journal, verdicts, gate log, cards, side
packets. Zero instrumentation, zero writes."""
import json
import re
import subprocess
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "Project"
JOURNAL = P / "results/JOURNAL.jsonl"
VERDICTS = P / "audits/verdicts.jsonl"
GATE_LOG = P / "loop/gate_log.jsonl"
GATE_STATE = P / "loop/gate_state.json"
CARDS = P / "loop/cards.jsonl"
SIDE = P / "results_side"
AUTO_DIR = P / "audits/auto"
STRAT_DIR = P / "audits/strategy"

FRIENDLY = {
    "k001_sdpa": "torch fast-attention baseline", "k002_fused_qkv": "merged-QKV kernel",
    "k003_triton_attention": "authored attention kernel", "k004_graphed_triton": "graphed attention kernel",
    "k005_fp16_graphed": "half-precision graphed kernel", "k006_fp16_hd128": "wide-head kernel",
    "k007_fused_block": "all-in-one fused kernel", "k009_fused_tuned": "all-in-one fused kernel (tuned)",
    "k010_fused_ln": "big-model kernel", "k014_shape14": "extreme-length chunked kernel",
    "k015_shape6": "huge-batch kernel", "k012_split_heads": "split-heads kernel",
}
DIR_FRIENDLY = {
    "F-shape14-attn": "long-text attention (test 14)", "F-shape6-local": "huge-batch proof (test 6)",
    "F-shape8-fp16acc": "fp16 math trick (test 8)", "F-shape11-hd8": "tiny-heads idea (test 11)",
}


def friendly(name):
    return FRIENDLY.get(name, DIR_FRIENDLY.get(name, name))


def jload(path):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


st.set_page_config(page_title="Track 3 Live", layout="wide")
st.title("🏁 Track 3 — Live Board")


@st.fragment(run_every=10)
def render():
    entries = jload(JOURNAL)
    verdicts = {v["entry_id"]: v["verdict"] for v in jload(VERDICTS)}

    # ---------- Scoreboard ----------
    best = {}
    for e in entries:
        if e.get("type") != "candidate" or not e.get("promoted"):
            continue
        sid = e["shape_id"]
        ms = e["timing"]["candidate"]["median_ms"]
        if sid not in best or ms < best[sid]["ms"]:
            best[sid] = {"ms": ms, "speed": e["timing"]["speedup"],
                         "impl": e["impl"]["name"], "when": e["timestamp"],
                         "verdict": verdicts.get(e["entry_id"], "awaiting audit"),
                         "remark": e.get("note", "")}
    side_rows = {}
    for pk in sorted(SIDE.glob("shape*_*.json")):
        d = json.loads(pk.read_text())
        if d.get("type") == "shape6_local_candidate_evidence" and d["correctness"]["violations"] == 0:
            side_rows[6] = {"speed": None, "impl": d["candidate"]["name"], "when": d["timestamp"],
                            "remark": "special proof — official baseline can't run this size"}
        if d.get("type") == "shape14_side_evaluation" and d["correctness"]["passed"]:
            side_rows[14] = {"speed": None, "impl": d["candidate"]["name"], "when": d["timestamp"],
                            "remark": "special proof — solved by splitting into chunks"}

    speeds = [b["speed"] for b in best.values()]
    geo = 1.0
    for s in speeds:
        geo *= s
    geo = geo ** (1 / len(speeds)) if speeds else 0

    st.subheader("Speed scoreboard")
    st.metric("Average speedup (12 comparable tests)", f"{geo:.1f}x")
    rows = []
    for sid in sorted(set(best) | set(side_rows)):
        r = best.get(sid) or side_rows.get(sid)
        rows.append({
            "Test": f"#{sid}",
            "Speed": f"{r['speed']:.1f}x" if r.get("speed") else "proven ✔",
            "Method": friendly(r["impl"]),
            "When": r["when"][:16].replace("T", " "),
            "Audit": ("✅ " + r["verdict"]) if r.get("verdict") in ("PASS",) else r.get("verdict", "special test"),
            "Remark": r.get("remark", "") or "",
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    chart = {f"#{sid}": best[sid]["speed"] for sid in sorted(best)}
    st.bar_chart(chart)

    # ---------- Right now ----------
    st.subheader("Right now")
    gs = json.loads(GATE_STATE.read_text()) if GATE_STATE.exists() else {}
    permit = (P / "loop/permit.json").exists()
    inflight = (P / "loop/in_flight.json").exists()
    cage = ("🟢 OPEN — one attempt armed" if permit else
            "🟡 attempt IN FLIGHT" if inflight else
            "🔒 CLOSED — needs research + plan before the next attempt")
    run_live = subprocess.run(["pgrep", "-f", r"harness[./]runner"],
                              capture_output=True, text=True).stdout.strip()
    sol_live = subprocess.run(["pgrep", "-f", "codex exec"],
                              capture_output=True, text=True).stdout.strip()
    c1, c2, c3 = st.columns(3)
    c1.metric("Thinking cage", cage)
    c2.metric("Benchmark", "🏃 running" if run_live else "idle")
    c3.metric("Sol auditor", "🧠 reviewing" if sol_live else "idle")
    ideas = []
    for card in jload(CARDS):
        status = card.get("status", "")
        icon = "❌ dead" if ("KILLED" in status or "CLOSED" in status.upper() and "local" not in status) \
            else "✅ done" if "complete" in status.lower() else "🔄 " + status[:40]
        ideas.append(f"**{friendly(card['direction_family_id'])}** — {icon}")
    if ideas:
        st.markdown("Ideas: " + " · ".join(ideas))

    # ---------- Event log ----------
    st.subheader("Event log (newest first)")
    show = st.multiselect("Show", ["runs", "cage", "sol"],
                          default=["runs", "cage", "sol"], key="filters")
    events = []
    if "runs" in show:
        for e in entries[-150:]:
            if e.get("type") == "candidate":
                t = e["timing"]["speedup"] if e.get("timing") else None
                events.append((e["timestamp"],
                               f"🏃 [RUN] test #{e['shape_id']} — {friendly(e['impl']['name'])}: "
                               f"{f'{t:.2f}x' if t else 'no timing'}, "
                               f"{'✅ promoted' if e.get('promoted') else 'not promoted'}", None))
    if "cage" in show:
        for g in jload(GATE_LOG)[-150:]:
            step = g.get("step", "?")
            msg = {"research": "📚 [CAGE] research step accepted — sources read & logged",
                   "plan": "🧭 [CAGE] plan approved — prediction: " + str(g.get("prediction", ""))[:60],
                   "delta": "🔧 [CAGE] follow-up attempt approved — " + str(g.get("changed", ""))[:60],
                   "reconcile": f"⚖️ [CAGE] attempt judged — {g.get('result', 'result recorded')}"
                                + (" · idea CLOSED ❌" if g.get("closed") else ""),
                   "screen_judge": f"🎯 [CAGE] screening {g.get('result', '')} recorded",
                   "reopen": "🔓 [CAGE] idea reopened on reviewer authority",
                   "init": "🚪 [CAGE] gate initialized",
                   "owner_unlock": "🔑 [CAGE] owner unlock"}.get(step, f"[CAGE] {step}")
            events.append((g.get("ts", ""), msg, None))
    if "sol" in show:
        for v in jload(VERDICTS)[-100:]:
            events.append((v.get("recorded", ""),
                           f"🧠 [SOL] audit verdict for run {v['entry_id'][-6:]}: {v['verdict']}",
                           v.get("source_log")))
        for lg in sorted(STRAT_DIR.glob("*_raw.log")):
            m = re.search(r"(VERDICT|PREFLIGHT|CRITIC):\s*([A-Za-z-]+)", lg.read_text()[-3000:] if lg.stat().st_size else "")
            if m:
                events.append((time.strftime("%Y-%m-%dT%H:%M", time.localtime(lg.stat().st_mtime)),
                               f"🧠 [SOL] {lg.stem.replace('_raw', '').replace('_', ' ')}: {m.group(2)}",
                               str(lg)))
    events.sort(key=lambda x: x[0], reverse=True)
    for ts, msg, detail in events[:100]:
        line = f"`{ts[:16].replace('T', ' ')}`  {msg}"
        if detail and Path(detail).exists():
            with st.expander(line):
                txt = Path(detail).read_text(errors="ignore")
                st.code(txt[-4000:], language=None)
        else:
            st.markdown(line)


render()
