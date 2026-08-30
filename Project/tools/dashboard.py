#!/usr/bin/env python3
"""Live project dashboard (read-only). Run:  streamlit run Project/tools/dashboard.py
Renders existing files only — journal, verdicts, gate log, cards, side
packets. Zero instrumentation, zero writes."""
import json
import math
import os
import re
import subprocess
import sys
import time
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "Project"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from champion_watch import (  # noqa: E402 - path set above
    _argv_is_busy,
    _process_argv,
    _self_and_ancestors,
)


def _is_benchmark_argv(argv: list[str]) -> bool:
    """A real runner/controller benchmark invocation, not a mention of one."""
    return _argv_is_busy(argv)


def _is_auditor_argv(argv: list[str]) -> bool:
    """A real `codex exec` invocation, not a command that names one."""
    return len(argv) >= 2 and Path(argv[0]).name == "codex" and argv[1] == "exec"


def _busy_pids(predicate) -> str:
    """Pids whose argv satisfies `predicate`, excluding this process tree."""
    try:
        exempt = _self_and_ancestors()
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except Exception:
        return ""  # a read-only dashboard reports unknown as idle, never blocks
    hits = [str(pid) for pid in pids
            if pid not in exempt and predicate(_process_argv(pid))]
    return "\n".join(hits)
JOURNAL = P / "results/JOURNAL.jsonl"
VERDICTS = P / "audits/verdicts.jsonl"
GATE_LOG = P / "loop/gate_log.jsonl"
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
ABSOLUTE_PATHS = (
    re.compile(r"file:///(?:[^\s`'\"<>|]+)"),
    re.compile(r"(?<!\w)(?:[A-Za-z]:[\\/]|\\\\)[^\s`'\"<>|]+"),
    re.compile(r"(?<![\w/])/(?!/)[^\s`'\"<>|]+"),
)


def friendly(name):
    return FRIENDLY.get(name, DIR_FRIENDLY.get(name, name))


def presentation_text(value, enabled):
    """Hide absolute filesystem paths in presentation-facing dynamic text."""
    text = str(value)
    if not enabled:
        return text
    for pattern in ABSOLUTE_PATHS:
        text = pattern.sub("(path hidden)", text)
    return text


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


def metric_grid_html(cards):
    """Render score metrics without Streamlit's fixed, clipping columns."""
    items = []
    for label, value, detail in cards:
        items.append(
            '<div class="score-metric" role="listitem">'
            f'<span class="score-metric__label">{escape(str(label))}</span>'
            f'<span class="score-metric__value">{escape(str(value))}</span>'
            f'<span class="score-metric__detail">{escape(str(detail))}</span>'
            '</div>'
        )
    return '<div class="score-metrics" role="list">' + "".join(items) + '</div>'


def status_grid_html(cards):
    """Render live statuses as wrapping cards at every viewport width."""
    items = []
    for label, value in cards:
        items.append(
            '<div class="status-card" role="listitem">'
            f'<span class="status-card__label">{escape(str(label))}</span>'
            f'<span class="status-card__value">{escape(str(value))}</span>'
            '</div>'
        )
    return '<div class="status-grid" role="list">' + "".join(items) + '</div>'


def scoreboard_html(rows, max_speed):
    """Render every leaderboard field as wrapping, responsive HTML."""
    try:
        safe_max = float(max_speed)
    except (TypeError, ValueError):
        safe_max = 1.0
    if not math.isfinite(safe_max) or safe_max <= 0:
        safe_max = 1.0

    headers = ("Test", "Speed vs baseline", "Method", "When", "Audit", "Remark")
    header = "".join(
        f'<div class="scoreboard__heading" role="columnheader">{escape(label)}</div>'
        for label in headers
    )
    body = []
    for row in rows:
        try:
            speed = float(row["Speedup"]) if row.get("Speedup") is not None else None
        except (TypeError, ValueError):
            speed = None
        if speed is not None and not math.isfinite(speed):
            speed = None

        if speed is None:
            speed_markup = '<span class="score-proof">✓ Proven</span>'
        else:
            width = max(0.0, min(100.0, speed / safe_max * 100))
            speed_label = f"{speed:.1f}x"
            speed_markup = (
                f'<span class="score-speed__value">{speed_label}</span>'
                '<span class="score-speed__track" aria-hidden="true">'
                f'<span style="width:{width:.1f}%"></span></span>'
            )

        audit_tone = row.get("AuditTone")
        if audit_tone not in {"pass", "pending", "special", "issue"}:
            audit_tone = "pending"
        audit_markup = (
            f'<span class="score-audit score-audit--{audit_tone}">'
            f'{escape(str(row["Audit"]))}</span>'
        )
        values = (
            ("Test", escape(str(row["Test"])), "scoreboard__cell--test"),
            ("Speed vs baseline", speed_markup, "scoreboard__cell--speed"),
            ("Method", escape(str(row["Method"])), "scoreboard__cell--method"),
            ("When", escape(str(row["When"])), "scoreboard__cell--when"),
            ("Audit", audit_markup, "scoreboard__cell--audit"),
            ("Remark", escape(str(row["Remark"])), "scoreboard__cell--remark"),
        )
        cells = []
        for label, value, class_name in values:
            cells.append(
                f'<div class="scoreboard__cell {class_name}" role="cell">'
                f'<span class="scoreboard__label">{escape(label)}</span>{value}</div>'
            )
        body.append('<div class="scoreboard__row" role="row">' + "".join(cells) + '</div>')

    if not body:
        return (
            '<div class="scoreboard scoreboard--empty" role="status">'
            '<div class="scoreboard__empty">No passing tests yet.</div></div>'
        )
    return (
        '<div class="scoreboard" role="table" aria-label="Speed leaderboard">'
        f'<div class="scoreboard__header" role="row">{header}</div>'
        '<div class="scoreboard__body" role="rowgroup">'
        + "".join(body)
        + '</div></div>'
    )


st.set_page_config(page_title="Track 3 Live", page_icon="🏁", layout="wide")
st.markdown("""
<style>
  /* Hard-forced dark mode — independent of theme config or launch dir */
  [data-testid="stAppViewContainer"], [data-testid="stHeader"],
  .stApp, body {background: #0b1020 !important; color: #e7ecf5 !important;}
  [data-testid="stHeader"] {background: #0b1020 !important;}
  h1, h2, h3, p, span, label, li {color: #e7ecf5 !important;}
  .block-container {padding-top: 1.2rem; max-width: 1250px;}
  h1 {font-size: 1.9rem; margin-bottom: 0;}
  h3 {margin-top: 1.4rem; border-bottom: 1px solid #26314f; padding-bottom: 4px;}
  code {color: #7dd3fc;}

  /* Score summary: real responsive cards, so labels never get ellipsized. */
  .score-metrics {
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0.8rem; margin: 0.15rem 0 1rem;
  }
  .score-metric {
      min-width: 0; display: flex; flex-direction: column; gap: 0.2rem;
      background: #151b2e; border: 1px solid #26314f; border-radius: 12px;
      padding: 0.9rem 1rem;
  }
  .score-metric__label {color: #aab6cc !important; font-size: 0.78rem; font-weight: 600;}
  .score-metric__value {
      color: #f8fafc !important; font-size: clamp(1.25rem, 2.1vw, 1.65rem);
      font-weight: 700; line-height: 1.2; overflow-wrap: anywhere;
  }
  .score-metric__detail {
      color: #8291ad !important; font-size: 0.75rem; line-height: 1.35;
      white-space: normal; overflow-wrap: anywhere;
  }

  .status-grid {
      display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.8rem; margin: 0.15rem 0 0.85rem;
  }
  .status-card {
      min-width: 0; display: flex; flex-direction: column; gap: 0.3rem;
      background: #151b2e; border: 1px solid #26314f; border-radius: 12px;
      padding: 0.9rem 1rem;
  }
  .status-card__label {color: #aab6cc !important; font-size: 0.78rem; font-weight: 600;}
  .status-card__value {
      color: #f8fafc !important; font-size: 1.12rem; font-weight: 650;
      line-height: 1.35; white-space: normal; overflow-wrap: anywhere;
  }

  /* Native HTML instead of the canvas dataframe: cells can grow and wrap. */
  .scoreboard {
      --score-columns: minmax(3.25rem, 0.55fr) minmax(8.25rem, 1.25fr)
          minmax(9.375rem, 1.65fr) minmax(7.875rem, 1.1fr)
          minmax(7.375rem, 1.05fr) minmax(11.875rem, 2fr);
      width: 100%; border: 1px solid #26314f; border-radius: 12px;
      overflow: hidden; background: #101628;
  }
  .scoreboard__header, .scoreboard__row {
      display: grid; grid-template-columns: var(--score-columns); min-width: 0;
  }
  .scoreboard__header {background: #1a2134; border-bottom: 1px solid #33415f;}
  .scoreboard__heading {
      min-width: 0; padding: 0.68rem 0.65rem; color: #9eacc4;
      font-size: 0.7rem; font-weight: 700; letter-spacing: 0.045em;
      line-height: 1.3; text-transform: uppercase; white-space: normal;
  }
  .scoreboard__body {min-width: 0;}
  .scoreboard__row {background: #101628; transition: background 120ms ease;}
  .scoreboard__row:nth-child(even) {background: #12192b;}
  .scoreboard__row:hover {background: #182238;}
  .scoreboard__row + .scoreboard__row {border-top: 1px solid #26314f;}
  .scoreboard__cell {
      min-width: 0; padding: 0.78rem 0.65rem; color: #e7ecf5;
      font-size: 0.79rem; line-height: 1.42; white-space: normal;
      overflow: visible; overflow-wrap: anywhere; word-break: normal;
      align-self: stretch; display: flex; flex-direction: column;
      justify-content: center;
  }
  .scoreboard__cell + .scoreboard__cell {border-left: 1px solid #202b44;}
  .scoreboard__label {display: none; color: #8291ad !important;}
  .scoreboard__cell--test {color: #f8fafc; font-weight: 750;}
  .scoreboard__cell--when {font-variant-numeric: tabular-nums;}
  .score-speed__value {
      color: #f8fafc !important; font-weight: 750;
      font-variant-numeric: tabular-nums;
  }
  .score-speed__track {
      display: block; width: 100%; height: 0.36rem; margin-top: 0.38rem;
      overflow: hidden; background: #22304a; border-radius: 999px;
  }
  .score-speed__track > span {
      display: block; height: 100%; min-width: 0.22rem; border-radius: inherit;
      background: linear-gradient(90deg, #21d4e8, #48b9ff);
  }
  .score-proof {color: #73e2c3 !important; font-weight: 750;}
  .score-audit {
      display: inline-flex; width: fit-content; max-width: 100%;
      align-items: center; border-radius: 999px; padding: 0.18rem 0.48rem;
      font-size: 0.73rem; font-weight: 700; line-height: 1.35;
      white-space: normal; overflow-wrap: anywhere;
  }
  .score-audit--pass {color: #8ef0b3 !important; background: #123b2d;}
  .score-audit--pending {color: #ffd479 !important; background: #46381a;}
  .score-audit--special {color: #81e6f0 !important; background: #153b46;}
  .score-audit--issue {color: #ff9ca8 !important; background: #4a1f2b;}
  .scoreboard__empty {padding: 1.2rem; color: #aab6cc; text-align: center;}

  @media (max-width: 960px) {
      .score-metrics {grid-template-columns: repeat(2, minmax(0, 1fr));}
      .scoreboard {border: 0; border-radius: 0; overflow: visible; background: transparent;}
      .scoreboard__header {display: none;}
      .scoreboard__body {
          display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.75rem;
      }
      .scoreboard__row, .scoreboard__row:nth-child(even) {
          display: grid; grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr);
          align-content: start; overflow: hidden; background: #12192b;
          border: 1px solid #2a3654; border-radius: 12px;
      }
      .scoreboard__row + .scoreboard__row {border-top: 1px solid #2a3654;}
      .scoreboard__cell {padding: 0.65rem 0.75rem; justify-content: flex-start;}
      .scoreboard__cell + .scoreboard__cell {border-left: 0;}
      .scoreboard__label {
          display: block; margin-bottom: 0.2rem; font-size: 0.64rem;
          font-weight: 700; letter-spacing: 0.045em; text-transform: uppercase;
      }
      .scoreboard__cell--method, .scoreboard__cell--remark {grid-column: 1 / -1;}
      .scoreboard__cell--method, .scoreboard__cell--when,
      .scoreboard__cell--audit, .scoreboard__cell--remark {border-top: 1px solid #26314f;}
  }
  @media (max-width: 680px) {
      .scoreboard__body {grid-template-columns: minmax(0, 1fr);}
      .status-grid {grid-template-columns: minmax(0, 1fr);}
  }
  @media (max-width: 480px) {
      .score-metrics {grid-template-columns: minmax(0, 1fr);}
  }
</style>""", unsafe_allow_html=True)
st.title("🏁 Track 3 — Live Board")
presentation_mode = st.sidebar.toggle(
    "Presentation mode (hide private logs)", value=False, key="presentation_mode"
)
st.sidebar.caption("Session only — this setting is not written to a config file.")


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
    if best:
        top_sid = max(best, key=lambda s: best[s]["speed"])
        best_value = f"{best[top_sid]['speed']:.1f}x"
        best_detail = f"Test #{top_sid}"
    else:
        best_value, best_detail = "—", "No comparable runs yet"
    audit_passes = sum(1 for b in best.values() if b["verdict"] == "PASS")
    st.html(metric_grid_html([
        ("Average speedup", f"{geo:.1f}x", f"{len(speeds)} comparable tests"),
        ("Best test", best_value, best_detail),
        ("Tests passing", f"{len(best) + len(side_rows)} / 14", "Correctness validated"),
        ("Audits passed", f"{audit_passes} / {len(best)}", "Comparable tests"),
    ]))

    rows = []
    for sid in sorted(set(best) | set(side_rows)):
        r = best.get(sid) or side_rows.get(sid)
        verdict = r.get("verdict")
        if verdict == "PASS":
            audit, audit_tone = "✓ Pass", "pass"
        elif verdict == "RULE_VIOLATION":
            audit, audit_tone = "! Rule violation", "issue"
        elif verdict:
            audit, audit_tone = "◷ " + verdict.replace("_", " ").title(), "pending"
        else:
            audit, audit_tone = "◇ Special test", "special"
        rows.append({
            "Test": f"#{sid}",
            "Speedup": r.get("speed"),
            "Method": presentation_text(friendly(r["impl"]), presentation_mode),
            "When": presentation_text(r["when"][:16].replace("T", " "),
                                      presentation_mode),
            "Audit": presentation_text(audit, presentation_mode),
            "AuditTone": audit_tone,
            "Remark": presentation_text(r.get("remark", "") or "—",
                                        presentation_mode),
        })
    max_speed = max((b["speed"] for b in best.values()), default=1)
    st.html(scoreboard_html(rows, max_speed))

    # ---------- Right now ----------
    st.subheader("Right now")
    permit = (P / "loop/permit.json").exists()
    inflight = (P / "loop/in_flight.json").exists()
    cage = ("🟢 OPEN — one attempt armed" if permit else
            "🟡 attempt IN FLIGHT" if inflight else
            "🔒 CLOSED — needs research + plan before the next attempt")
    # `pgrep -f` matches the JOINED command line, so any shell that merely
    # NAMES a benchmark or an auditor reads as one running.  DECISIONS.md:53
    # already recorded that false reading once, and STATE.md/LESSONS #22 tell
    # every session to trust `pgrep -f "codex exec"` before measuring.  Match
    # whole argv elements instead, and never count this process or its
    # ancestors.
    run_live = _busy_pids(_is_benchmark_argv)
    sol_live = _busy_pids(_is_auditor_argv)
    st.html(status_grid_html([
        ("Thinking cage", cage),
        ("Benchmark", "🏃 running" if run_live else "idle"),
        ("Sol auditor", "🧠 reviewing" if sol_live else "idle"),
    ]))
    ideas = []
    for card in jload(CARDS):
        status = card.get("status", "")
        icon = "❌ dead" if ("KILLED" in status or "CLOSED" in status.upper() and "local" not in status) \
            else "✅ done" if "complete" in status.lower() else "🔄 " + status[:40]
        idea = f"**{friendly(card['direction_family_id'])}** — {icon}"
        ideas.append(presentation_text(idea, presentation_mode))
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
            plan_detail = ("" if presentation_mode else
                           " — prediction: " + str(g.get("prediction", ""))[:60])
            delta_detail = ("" if presentation_mode else
                            " — " + str(g.get("changed", ""))[:60])
            msg = {"research": "📚 [CAGE] research step accepted — sources read & logged",
                   "plan": "🧭 [CAGE] plan approved" + plan_detail,
                   "delta": "🔧 [CAGE] follow-up attempt approved" + delta_detail,
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
    with st.container(height=460, border=True):
        for ts, msg, detail in events[:100]:
            safe_ts = presentation_text(ts[:16].replace("T", " "),
                                        presentation_mode)
            safe_msg = presentation_text(msg, presentation_mode)
            line = f"`{safe_ts}` &nbsp; {safe_msg}"
            if detail and Path(detail).exists():
                with st.expander(line):
                    if presentation_mode:
                        st.caption("(hidden in presentation mode)")
                    else:
                        txt = Path(detail).read_text(errors="ignore")
                        st.code(txt[-4000:], language=None)
            else:
                st.markdown(line)


render()
