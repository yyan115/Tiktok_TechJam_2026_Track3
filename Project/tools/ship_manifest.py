#!/usr/bin/env python3
"""Ship manifest generator (pre-flight audit item): ONE machine-readable
file mapping every shape to its exact evidence — journal entry or side
packet, candidate source sha, method, device, audit verdict — plus the
submission file's sha and the git revision. Judges (or anyone) can verify
every claimed number from this single file.

Freeze rule: regenerate at freeze, AFTER the final commit.
Output: Project/results_side/SHIP_MANIFEST.json. Read-only over inputs.
"""
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
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


def freeze_provenance() -> tuple[str, str]:
    """Require committed inputs and bind the on-disk submission to HEAD."""
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    if status.strip():
        changed = ", ".join(line[3:] for line in status.splitlines()[:8])
        suffix = " ..." if len(status.splitlines()) > 8 else ""
        raise SystemExit(
            "refusing to generate ship manifest: working tree has "
            f"uncommitted changes ({changed}{suffix})"
        )

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    submission_rel = SUBMISSION.relative_to(ROOT).as_posix()
    committed = subprocess.run(
        ["git", "show", f"{head}:{submission_rel}"], cwd=ROOT,
        capture_output=True, check=False,
    )
    if committed.returncode != 0:
        detail = committed.stderr.decode(errors="replace").strip()
        raise SystemExit(
            f"cannot read {submission_rel} from HEAD {head}: {detail}"
        )
    on_disk_sha = sha(SUBMISSION)
    committed_sha = hashlib.sha256(committed.stdout).hexdigest()
    if committed_sha != on_disk_sha:
        raise SystemExit(
            "refusing to generate ship manifest: on-disk submission sha256 "
            f"{on_disk_sha} does not match HEAD {head} blob sha256 {committed_sha}"
        )
    return head, on_disk_sha


def packet_median_ms(packet: dict):
    timing = packet.get("timing", {})
    return timing.get("median_of_sums_ms", timing.get("median_ms"))


def packet_recency(path: Path, packet: dict) -> tuple[float, str]:
    timestamp = packet.get("timestamp")
    if timestamp:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp(), path.name
        except ValueError:
            pass
    return path.stat().st_mtime, path.name


def packet_is_correct(shape_id: int, packet: dict) -> bool:
    correctness = packet.get("correctness", {})
    if shape_id == 6:
        return correctness.get("violations") == 0
    return correctness.get("passed") is True


def side_audit_verdict(shape_id: int) -> str:
    if shape_id == 6:
        return "side-evidence (validated vs batch-chunked official baseline)"
    return "side-evidence (streamed oracle validated vs pinned official dense)"


def side_entry(path: Path, packet: dict, shape_id: int) -> dict:
    candidate = packet.get("candidate", {})
    name = candidate.get("name", "unknown")
    return {
        "evidence": f"pinned side evaluator packet ({path.name})",
        "packet_sha256": sha(path),
        "packet_schema_version": packet.get("schema_version", "legacy"),
        "timestamp": packet.get("timestamp"),
        "impl": name,
        "impl_friendly": FRIENDLY.get(name, name),
        "impl_sha256": candidate.get("sha256"),
        "median_ms": packet_median_ms(packet),
        "timing_protocol": packet.get("timing", {}).get("protocol"),
        "batch_measured": packet.get("shape", {}).get("batch_size"),
        "speedup_vs_baseline": None,
        "baseline_note": packet.get("limitation"),
        "correct": packet_is_correct(shape_id, packet),
        "audit_verdict": side_audit_verdict(shape_id),
        "device": packet.get("env", {}).get("gpu"),
    }


def shape14_alternate(path: Path, packet: dict) -> dict:
    candidate = packet.get("candidate", {})
    return {
        "evidence": f"pinned side evaluator packet ({path.name})",
        "packet_sha256": sha(path),
        "packet_schema_version": packet.get("schema_version", "legacy"),
        "timestamp": packet.get("timestamp"),
        "candidate": candidate.get("name"),
        "candidate_sha256": candidate.get("sha256"),
        "batch_measured": packet.get("shape", {}).get("batch_size"),
        "median_ms": packet_median_ms(packet),
        "timing_protocol": packet.get("timing", {}).get("protocol"),
        "correct": packet_is_correct(14, packet),
    }


def main():
    git_rev, submission_sha = freeze_provenance()
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

    shape6_packets = []
    for p in sorted(SIDE.glob("shape6_*.json")):
        d = json.loads(p.read_text())
        if (d.get("type") == "shape6_local_candidate_evidence" and
                packet_is_correct(6, d)):
            shape6_packets.append((p, d))
    if shape6_packets and shapes.get(6, {}).get("evidence") != "frozen-runner journal":
        shape6_path, shape6_packet = min(
            shape6_packets,
            key=lambda item: packet_median_ms(item[1]),
        )
        shapes[6] = side_entry(shape6_path, shape6_packet, 6)

    all_shape14_packets = []
    for p in sorted(SIDE.glob("shape14_*.json")):
        d = json.loads(p.read_text())
        all_shape14_packets.append((p, d))
    valid_shape14 = [
        item for item in all_shape14_packets
        if (item[1].get("type") == "shape14_side_evaluation" and
            packet_is_correct(14, item[1]))
    ]
    full_streamed = [
        item for item in valid_shape14
        if (item[1].get("schema_version") == "streamed-v1" and
            item[1].get("shape", {}).get("batch_size") == 32)
    ]
    selection_pool = full_streamed or valid_shape14
    if selection_pool and shapes.get(14, {}).get("evidence") != "frozen-runner journal":
        shape14_path, shape14_packet = max(
            selection_pool,
            key=lambda item: packet_recency(item[0], item[1]),
        )
        selected = side_entry(shape14_path, shape14_packet, 14)
        selected["alternates"] = [
            shape14_alternate(path, packet)
            for path, packet in all_shape14_packets
            if path != shape14_path
        ]
        shapes[14] = selected

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_revision": git_rev,
        "submission_file": str(SUBMISSION.relative_to(ROOT)),
        "submission_sha256": submission_sha,
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
