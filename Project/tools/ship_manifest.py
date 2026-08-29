#!/usr/bin/env python3
"""Ship manifest generator (pre-flight audit item): ONE machine-readable
file mapping every shape to its exact evidence — journal entry or side
packet, candidate source sha, method, device, audit verdict — plus the
submission file's sha and the git revision. Judges (or anyone) can verify
every claimed number from this single file.

Output: Project/results_side/SHIP_MANIFEST.json. Read-only over inputs.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "Project"
JOURNAL = PROJECT / "results" / "JOURNAL.jsonl"
VERDICTS = PROJECT / "audits" / "verdicts.jsonl"
SIDE = PROJECT / "results_side"
SUBMISSION = PROJECT / "submission" / "torch_transformer_benchmark_submission.py"

FRIENDLY = {
    "k007_fused_block": "all-in-one fused block kernel",
    "k009_fused_tuned": "all-in-one fused block kernel (tuned)",
    "k010_fused_ln": "big-model kernel (fused norm/GELU)",
    "k014_shape14": "extreme-sequence kernel (block-decomposed)",
    "k015_shape6": "huge-batch kernel (graph-free)",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    entries = [json.loads(l) for l in JOURNAL.read_text().splitlines() if l.strip()]
    verdicts = {}
    for l in VERDICTS.read_text().splitlines():
        if l.strip():
            v = json.loads(l)
            verdicts[v["entry_id"]] = v["verdict"]

    shapes = {}
    for e in entries:
        if e.get("type") != "candidate" or not e.get("promoted"):
            continue
        sid = e["shape_id"]
        ms = e["timing"]["candidate"]["median_ms"]
        cur = shapes.get(sid)
        if cur is None or ms < cur["median_ms"]:
            shapes[sid] = {
                "evidence": "frozen-runner journal",
                "entry_id": e["entry_id"],
                "timestamp": e["timestamp"],
                "impl": e["impl"]["name"],
                "impl_friendly": FRIENDLY.get(e["impl"]["name"], e["impl"]["name"]),
                "impl_sha256": e["impl"]["sha256"],
                "median_ms": ms,
                "speedup_vs_baseline": e["timing"]["speedup"],
                "correct": e["correctness"]["passed"],
                "audit_verdict": verdicts.get(e["entry_id"], "unaudited"),
                "device": e["env"]["gpu"],
            }

    for p in sorted(SIDE.glob("shape6_*.json")) + sorted(SIDE.glob("shape14_*.json")):
        d = json.loads(p.read_text())
        if d.get("type") == "shape6_local_candidate_evidence" and d["correctness"]["violations"] == 0:
            sid, ms = 6, d["timing"]["median_ms"]
        elif d.get("type") == "shape14_side_evaluation" and d["correctness"]["passed"]:
            sid, ms = 14, d["timing"]["median_ms"]
        else:
            continue
        cur = shapes.get(sid)
        if cur is None or (cur["evidence"] != "frozen-runner journal" and ms < cur["median_ms"]):
            shapes[sid] = {
                "evidence": f"pinned side evaluator packet ({p.name})",
                "packet_sha256": sha(p),
                "timestamp": d["timestamp"],
                "impl": d["candidate"]["name"],
                "impl_friendly": FRIENDLY.get(d["candidate"]["name"], d["candidate"]["name"]),
                "impl_sha256": d["candidate"]["sha256"],
                "median_ms": ms,
                "batch_measured": d["shape"].get("batch_size"),
                "speedup_vs_baseline": None,
                "baseline_note": d.get("limitation"),
                "correct": True,
                "audit_verdict": "side-evidence (oracle validated vs official dense)",
                "device": d["env"]["gpu"],
            }

    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_revision": git_rev,
        "submission_file": str(SUBMISSION.relative_to(ROOT)),
        "submission_sha256": sha(SUBMISSION),
        "official_script_sha256": sha(ROOT / "torch_transformer_benchmark.py"),
        "device_primary": "NVIDIA GeForce RTX 3060 Ti (8 GB)",
        "shapes": {str(k): shapes[k] for k in sorted(shapes)},
        "verification": "each journal entry regenerates via Project/harness/"
                        "runner.py leaderboard; side packets carry their own "
                        "shas, seeds, and raw samples",
    }
    out = SIDE / "SHIP_MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"{out} — {len(shapes)} shapes mapped")


if __name__ == "__main__":
    main()
