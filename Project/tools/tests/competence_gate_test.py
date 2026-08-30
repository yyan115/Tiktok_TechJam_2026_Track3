#!/usr/bin/env python3
"""Cold-start CLI tests for the v5 competence gate."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GATE = REPO / "Project" / "tools" / "run_gate.py"
CATALOG = REPO / "Project" / "loop" / "mechanism_catalog.json"
CATALOG_SCHEMA = REPO / "Project" / "loop" / "mechanism_catalog.schema.json"
AUDIT_AUTHORITY = REPO / "Project" / "tools" / "audit_authority.py"
AUTHORITY = REPO / "Project" / "harness" / "authority.py"
checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    checks.append((name, bool(condition), detail))
    print(("PASS " if condition else "FAIL ") + name
          + (f"  [{detail}]" if detail and not condition else ""))


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def build() -> Path:
    sb = Path(tempfile.mkdtemp(prefix="competence_gate_"))
    for rel in ("Project/tools", "Project/harness", "Project/loop",
                "Project/research", "Project/results", "Project/audits",
                "Project/authority/blobs", "Project/submission"):
        (sb / rel).mkdir(parents=True)
    shutil.copyfile(GATE, sb / "Project/tools/run_gate.py")
    shutil.copyfile(CATALOG, sb / "Project/loop/mechanism_catalog.json")
    shutil.copyfile(CATALOG_SCHEMA,
                    sb / "Project/loop/mechanism_catalog.schema.json")
    shutil.copyfile(AUDIT_AUTHORITY, sb / "Project/tools/audit_authority.py")
    shutil.copyfile(AUTHORITY, sb / "Project/harness/authority.py")
    (sb / "Project/research/INDEX.md").write_text("test index\n")
    (sb / "Project/research/note-a.md").write_text("note a\n")
    (sb / "Project/research/note-b.md").write_text("note b\n")
    (sb / "Project/results/JOURNAL.jsonl").write_text("")
    (sb / "Project/audits/verdicts.jsonl").write_text("")
    (sb / "Project/loop/cards.jsonl").write_text(
        json.dumps({"direction_family_id": "F-TEST", "status": "open"}) + "\n")
    (sb / "cand.py").write_text("candidate_version = 1\n")
    (sb / "cite.md").write_text("counter evidence supports launch overhead\n"
                                 "graph replay changes launch count\n")
    (sb / "machine.json").write_text(json.dumps({
        "captured_epoch": time.time(), "gpu_utilization": 0,
        "competing_processes": []}))
    (sb / "Project/harness/trusted_controller.py").write_text(r'''#!/usr/bin/env python3
import argparse, json
p=argparse.ArgumentParser(); p.add_argument("cmd")
p.add_argument("--receipt"); p.add_argument("--action")
p.add_argument("--subject-sha256"); a=p.parse_args()
if a.cmd != "verify-receipt": raise SystemExit(2)
try: r=json.load(open(a.receipt))
except Exception: raise SystemExit(3)
if r.get("action") != a.action or r.get("subject_sha256") != a.subject_sha256:
    raise SystemExit(4)
print(json.dumps({"valid": True, "authority_event_id": r["authority_event_id"],
 "authority_event_sha256": r["authority_event_sha256"],
 "capability_nonce": r["capability_nonce"], "role": r["role"],
 "action": r["action"], "subject_sha256": r["subject_sha256"]}))
''')
    return sb


def main() -> int:
    sb = build()
    gate = sb / "Project/tools/run_gate.py"
    loop = sb / "Project/loop"
    journal = sb / "Project/results/JOURNAL.jsonl"

    def run(*args):
        p = subprocess.run([sys.executable, str(gate), *args], cwd=sb,
                           text=True, capture_output=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        return p.returncode, p.stdout + p.stderr

    receipt_no = 0

    def receipt(action, subject):
        nonlocal receipt_no
        receipt_no += 1
        path = sb / f"receipt-{receipt_no}.json"
        path.write_text(json.dumps({
            "action": action, "subject_sha256": digest(subject),
            "authority_event_id": f"evt-{receipt_no}",
            "authority_event_sha256": hashlib.sha256(
                f"event-{receipt_no}".encode()).hexdigest(),
            "capability_nonce": f"nonce-{receipt_no}", "role": "owner",
        }))
        return path

    def state():
        return json.loads((loop / "gate_state.json").read_text())

    event_no = 0
    previous_event_sha = None
    events_path = sb / "Project/authority/events.jsonl"
    blobs = sb / "Project/authority/blobs"

    def append_authority(kind, payload):
        nonlocal event_no, previous_event_sha
        event_no += 1
        event = {
            "schema_version": 1,
            "event_id": f"evt-20260830T140000.000000Z-{event_no:012x}",
            "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "kind": kind, "actor": "trusted-controller", "payload": payload,
            "prev_event_sha256": previous_event_sha,
        }
        unsigned = json.dumps(event, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode()
        event["event_sha256"] = hashlib.sha256(unsigned).hexdigest()
        with events_path.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        previous_event_sha = event["event_sha256"]
        return event

    def consume_projection(*, speedup=1.0, correct=True, suspicious=False,
                           profile_sha=None, failed=False, complete=True):
        projection = json.loads((loop / "permit.json").read_text())
        request_sha = projection["request_sha256"]
        request_raw = (sb / projection["request_path"]).read_bytes()
        (blobs / f"{request_sha}.json").write_bytes(request_raw)
        kind, mode = projection["request_kind"], projection["mode"]
        if kind == "diagnostic":
            candidate_sha, family_id = projection["target_sha256"], None
        elif kind == "calibration":
            candidate_sha, family_id = None, None
        elif kind == "side_evaluation":
            candidate_sha, family_id = projection["impl_sha256"], None
        else:
            candidate_sha = projection["impl_sha256"]
            family_id = projection["family"]["family_id"]
        if candidate_sha and projection.get("impl_path"):
            (blobs / f"{candidate_sha}.py").write_bytes(
                (sb / projection["impl_path"]).read_bytes())
        permit_id = "permit-" + projection["request_id"]
        expires = datetime.fromtimestamp(
            projection["expires_epoch"], tz=timezone.utc).isoformat().replace("+00:00", "Z")
        issue = append_authority("permit_issued", {
            "permit_id": permit_id, "request_sha256": request_sha,
            "campaign_id": projection["campaign_id"], "mode": mode,
            "shape_id": projection["shape"], "candidate_sha256": candidate_sha,
            "family_id": family_id, "expires_at": expires,
            "request_blob_sha256": request_sha,
            "may_modify_candidate": projection["candidate_authorized"],
            "may_promote": mode == "optimization",
            "capability_consumed": True, "capability_id": "cap-test",
            "capability_nonce": "nonce-permit-" + projection["request_id"],
            "capability_action": "permit.issue",
            "capability_target": f"shape:{projection['shape']}",
            "capability_role": "owner", "owner_key_sha256": "1" * 64,
            "capability_use_number": event_no + 1,
        })
        consumed = append_authority("permit_consumed", {
            "permit_id": permit_id, "issued_event_id": issue["event_id"],
            "mode": mode, "shape_id": projection["shape"],
            "candidate_sha256": candidate_sha,
        })
        run_id = "run-" + hashlib.sha256(
            projection["request_id"].encode()).hexdigest()[:32]
        worker_request_sha = hashlib.sha256((run_id + "request").encode()).hexdigest()
        started = append_authority("run_started", {
            "run_id": run_id, "permit_id": permit_id,
            "consumed_event_id": consumed["event_id"],
            "campaign_id": projection["campaign_id"], "mode": mode,
            "shape_id": projection["shape"], "candidate_sha256": candidate_sha,
            "worker_request_sha256": worker_request_sha,
            "lock_id": "lock-test", "lock_manifest_sha256": "2" * 64,
        })
        (loop / "permit.json").unlink()
        projection.update({"_permit_id": permit_id, "_entry_id": run_id})
        if not complete:
            return projection
        if failed:
            terminal = append_authority("run_failed", {
                "run_id": run_id, "started_event_id": started["event_id"],
                "reason": "worker_exit", "returncode": 1,
                "stdout_sha256": "3" * 64, "stderr_sha256": "4" * 64,
                "elapsed_ms": 1.0,
            })
            projection["_terminal_sha256"] = terminal["event_sha256"]
            return projection
        lane = ("scratch" if mode in {"screening", "correctness"}
                else mode if mode in {"shape6", "shape14"} else "primary")
        if mode in {"shape6", "shape14"}:
            entry_id = f"20260830-143000-{event_no:06x}"[-22:]
            stage_packet = {"schema_version": 1, "entry_id": entry_id,
                            "candidate_sha256": candidate_sha,
                            "passed": bool(correct), "speedup": float(speedup)}
            stage_raw = canonical(stage_packet)
            stage_sha = hashlib.sha256(stage_raw).hexdigest()
            (blobs / f"{stage_sha}.json").write_bytes(stage_raw)
            stages = [{"stage": f"{mode}-eval", "sha256": stage_sha}]
            validation = {"passed": bool(correct), "protocol": "fixture"}
            measurement = append_authority("measurement_recorded", {
                "entry_id": entry_id, "run_id": run_id,
                "started_event_id": started["event_id"], "permit_id": permit_id,
                "campaign_id": projection["campaign_id"], "mode": mode,
                "lane": mode, "shape_id": projection["shape"],
                "candidate_sha256": candidate_sha, "family_id": None,
                "gate_request_sha256": request_sha,
                "side_evidence_sha256": stage_sha,
                "side_stage_artifacts": stages,
                "controller_validation": validation,
                "evidence_eligible_pre_audit": bool(correct),
                "promotion_eligible": False,
                "promotion_blocker": "side_evidence_not_primary_champion",
                "lock_id": "lock-test", "lock_manifest_sha256": "2" * 64,
            })
            wrapper = {
                "schema_version": 1, "entry_id": entry_id, "lane": mode,
                "measurement_event_id": measurement["event_id"],
                "measurement_event_sha256": measurement["event_sha256"],
                "candidate_sha256": candidate_sha, "permit_id": permit_id,
                "gate_request_sha256": request_sha,
                "campaign_id": projection["campaign_id"], "mode": mode,
                "shape_id": projection["shape"],
                "side_evidence_sha256": stage_sha,
                "side_stage_artifacts": stages,
                "controller_validation": validation,
                "side_evidence_packets": [stage_packet],
                "lock_manifest_sha256": "2" * 64,
            }
            packet_raw = canonical(wrapper)
            packet_sha = hashlib.sha256(packet_raw).hexdigest()
            (blobs / f"{packet_sha}.json").write_bytes(packet_raw)
            binding = append_authority("measurement_packet_bound", {
                "entry_id": entry_id,
                "measurement_event_id": measurement["event_id"],
                "measurement_event_sha256": measurement["event_sha256"],
                "candidate_sha256": candidate_sha, "packet_sha256": packet_sha,
                "side_evidence_sha256": stage_sha, "lane": mode,
            })
            projection.update({"_entry_id": entry_id,
                               "_measurement_event_sha256": measurement["event_sha256"],
                               "_packet_sha256": packet_sha,
                               "_binding_event_sha256": binding["event_sha256"]})
            return projection
        response_sha = hashlib.sha256((run_id + "response").encode()).hexdigest()
        timing_args = campaign["timing_config"]
        measurement_payload = {
            "run_id": run_id, "started_event_id": started["event_id"],
            "permit_id": permit_id, "campaign_id": projection["campaign_id"],
            "mode": mode, "lane": lane, "shape_id": projection["shape"],
            "candidate_sha256": candidate_sha, "family_id": family_id,
            "worker_request_sha256": worker_request_sha,
            "worker_response_sha256": response_sha,
            "stdout_sha256": "3" * 64, "stderr_sha256": "4" * 64,
            "controller_correctness": {"passed": bool(correct), "trials": []},
            "supporting_timing": {"event_speedup": float(speedup),
                                  "suspicious": bool(suspicious)},
            "worker_environment": {"gpu": "fixture"},
            "effective_numerical_state": {"tf32": True},
            "timing_args": timing_args, "numerical": {"rtol": 0.02},
            "controller_process_elapsed_ms": 1.0,
            "promotion_threshold": 1.03 if mode in {"calibration", "optimization"} else None,
            "calibrated_noise": 0.01 if mode == "calibration" else None,
            "calibration_event_id": "evt-calibration" if mode == "optimization" else None,
            "performance_eligible": bool(mode == "optimization" and correct
                                         and not suspicious and speedup > 1.03),
            "promotion_eligible": False,
            "promotion_blocker": "audit_required" if mode == "optimization"
                                 else "performance_or_correctness",
            "lock_id": "lock-test", "lock_manifest_sha256": "2" * 64,
        }
        if mode == "diagnostic":
            if profile_sha is None:
                raise AssertionError("diagnostic fixture needs profile_sha")
            measurement_payload["diagnostic_profile_sha256"] = profile_sha
        measurement = append_authority("measurement_recorded", measurement_payload)
        packet = {
            "schema_version": 1, "entry_id": run_id, "lane": lane,
            "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "candidate_sha256": candidate_sha, "permit_id": permit_id,
            "gate_request_sha256": request_sha,
            "campaign_id": projection["campaign_id"], "mode": mode,
            "shape_id": projection["shape"], "family_id": family_id,
            "worker_request_sha256": worker_request_sha,
            "worker_response_sha256": response_sha,
            "controller_correctness": measurement_payload["controller_correctness"],
            "supporting_timing": measurement_payload["supporting_timing"],
            "lock_manifest_sha256": "2" * 64,
        }
        if mode == "diagnostic":
            packet["diagnostic_profile_sha256"] = profile_sha
        packet_raw = canonical(packet)
        packet_sha = hashlib.sha256(packet_raw).hexdigest()
        (blobs / f"{packet_sha}.json").write_bytes(packet_raw)
        binding = append_authority("measurement_packet_bound", {
            "entry_id": run_id, "measurement_event_id": measurement["event_id"],
            "measurement_event_sha256": measurement["event_sha256"],
            "candidate_sha256": candidate_sha, "packet_sha256": packet_sha,
            "lane": lane,
        })
        projection.update({"_measurement_event_sha256": measurement["event_sha256"],
                           "_packet_sha256": packet_sha,
                           "_binding_event_sha256": binding["event_sha256"]})
        return projection

    rc, out = run("init")
    check("closed init", rc == 0, out)
    campaign = {
        "schema_version": 1, "campaign_id": "CAMP-1",
        "max_total_attempts": 20, "max_calibrations_per_shape": 2,
        "max_total_calibrations": 10, "stall_window": 4,
        "timing_config": {"warmup": 20, "repeats": 100, "rounds": 3},
        "score_scenarios": ["equal-shape", "flop-weighted"],
    }
    campaign_path = sb / "campaign.json"
    campaign_path.write_text(json.dumps(campaign))
    rc, out = run("campaign-open", "--spec", str(campaign_path),
                  "--authority-receipt", str(sb / "missing-receipt"))
    check("campaign refuses workspace prose/missing authority",
          rc == 1 and "receipt" in out.lower(), out)
    rc, out = run("campaign-open", "--spec", str(campaign_path),
                  "--authority-receipt", str(receipt("open_campaign", campaign)))
    check("controller-authorized bounded campaign opens", rc == 0, out)

    family = {
        "family_id": "F-TEST", "shape": 3,
        "mechanism": "cuda-graph-replay", "bottleneck": "launch-overhead",
        "changed_resource": "kernel-launches",
        "expected_counter_change": {"kernel_launches": "decrease"},
        "parent_family_id": None, "budget_attempts": 8,
        "budget_minutes": 120, "admission": "controller-authorized",
        "allow_new_attempts": True,
    }
    family_path = sb / "family.json"
    family_path.write_text(json.dumps(family))
    subject = {"campaign_id": "CAMP-1", "family": family}
    rc, out = run("family-register", "--campaign", "CAMP-1",
                  "--family-spec", str(family_path), "--authority-receipt",
                  str(receipt("register_family", subject)))
    check("trusted controller assigns family identity", rc == 0, out)

    renamed = dict(family, family_id="F-RENAMED")
    renamed_path = sb / "renamed.json"; renamed_path.write_text(json.dumps(renamed))
    rc, out = run("family-register", "--campaign", "CAMP-1",
                  "--family-spec", str(renamed_path), "--authority-receipt",
                  str(receipt("register_family",
                              {"campaign_id": "CAMP-1", "family": renamed})))
    check("same mechanism rename must inherit existing family",
          rc == 1 and "inherit" in out, out)

    child = dict(family, family_id="F-TEST-FUSION", mechanism="kernel-fusion",
                 changed_resource="launches-and-intermediate-bytes",
                 parent_family_id="F-TEST")
    child_path = sb / "child.json"; child_path.write_text(json.dumps(child))
    novelty = ("Independent reviewer confirms this child changes both the "
               "operation boundary and intermediate lifetime; it is not a "
               "tile/configuration variant and has a distinct falsifier.")
    child_subject = {"campaign_id": "CAMP-1", "family": child,
                     "parent_family_id": "F-TEST", "novelty_basis": novelty}
    rc, out = run("family-register", "--campaign", "CAMP-1",
                  "--family-spec", str(child_path), "--novelty-basis", novelty,
                  "--authority-receipt",
                  str(receipt("resolve_family_novelty", child_subject)))
    check("material child needs explicit novelty resolution", rc == 0, out)

    rc, out = run("calibrate", "--campaign", "CAMP-1", "--shape", "3",
                  "--machine-state", "machine.json")
    check("bounded calibration request emits", rc == 0, out)
    cal = consume_projection()
    check("calibration cannot authorize candidate/promotion",
          cal["candidate_authorized"] is False
          and cal["promotion_allowed"] is False
          and cal["scientific_strike_eligible"] is False, str(cal))
    rc, out = run("reconcile")
    check("calibration binds immutable campaign noise", rc == 0, out)
    rc, out = run("calibrate", "--campaign", "CAMP-1", "--shape", "3",
                  "--machine-state", "machine.json")
    check("designated calibration cannot reroll", rc == 1 and "already" in out, out)

    target_sha = "a" * 64
    rc, out = run("diagnostic", "--campaign", "CAMP-1", "--shape", "3",
                  "--target-sha256", target_sha, "--tool", "nsys",
                  "--supports", "launch-overhead", "--question",
                  "Does host launch overhead dominate the current champion route?",
                  "--route", "dispatcher-shape-3")
    check("diagnostic request emits independently", rc == 0, out)
    diag = json.loads((loop / "permit.json").read_text())
    check("diagnostic is structurally non-mutating/non-promoting",
          diag["ledger"] is None and diag["candidate_authorized"] is False
          and diag["promotion_allowed"] is False
          and diag["scientific_strike_eligible"] is False, str(diag))
    profile_output = sb / diag["profile_output"]
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    raw_profile = profile_output.parent / "raw-timeline.nsys-rep"
    raw_profile.write_bytes(b"trusted profiler output")
    profile_output.write_text(json.dumps({
        "schema_version": 1, "profile_record_id": diag["profile_record_id"],
        "request_id": diag["request_id"], "campaign_id": "CAMP-1",
        "shape": 3, "target_sha256": target_sha, "tool": "nsys",
        "tool_version": "2026.1", "created_epoch": time.time(),
        "machine_state_sha256": hashlib.sha256(b"machine-state").hexdigest(),
        "route": "dispatcher-shape-3",
        "metrics": {"kernel_launches": 41, "incumbent_speedup": 1.0},
        "supported_bottlenecks": ["launch-overhead"],
        "gate_request_sha256": diag["request_sha256"],
        "raw_artifacts": [{"path": str(raw_profile.relative_to(sb)),
                           "sha256": hashlib.sha256(raw_profile.read_bytes()).hexdigest()}],
    }))
    consume_projection(profile_sha=hashlib.sha256(profile_output.read_bytes()).hexdigest())
    rc, out = run("reconcile")
    check("trusted diagnostic artifact reconciles", rc == 0, out)
    check("diagnostic consumes no scientific attempts",
          state()["campaigns"]["CAMP-1"]["scientific_attempts"] == 0
          and not state()["groups"], str(state()["groups"]))
    profile_id = diag["profile_record_id"]

    index_hash = hashlib.sha256(
        (sb / "Project/research/INDEX.md").read_bytes()).hexdigest()[:16]

    def research():
        return run("research", "--campaign", "CAMP-1", "--index-hash", index_hash,
                   "--notes", "note-a.md,note-b.md", "--summary", "R" * 220)

    common = [
        "--campaign", "CAMP-1", "--direction", "F-TEST", "--shape", "3",
        "--impl", "cand.py", "--target-sha256", target_sha,
        "--bottleneck", "launch-overhead", "--counter-evidence", profile_id,
        "--hypothesis", "Graph replay should remove measured host launch gaps " * 2,
        "--prediction", "1.09x expected", "--prediction-kind", "win",
        "--falsifier", "Count launches and run a graph replay microtest before changing the dispatcher.",
        "--falsifier-kill", "Kill if launch count or latency does not improve.",
        "--prior-family-verdict", "NONE", "--kill",
        "Kill the direction if measured performance stays below 1.05x.",
        "--sources", "cite.md:1-2", "--reasoning", "E" * 130,
    ]
    rc, out = research(); check("research follows diagnostic", rc == 0, out)
    rc, out = run("plan", "--mode", "optimization", "--predict-min", "0.9",
                  "--predict-max", "1.8", *common)
    check("broad prediction rejected by calibrated noise",
          rc == 1 and "uninformative" in out, out)
    rc, out = run("plan", "--mode", "optimization", "--predict-min", "1.08",
                  "--predict-max", "1.10", *common)
    check("profile-backed plan emits immutable request", rc == 0, out)
    projection = json.loads((loop / "permit.json").read_text())
    request_path = sb / projection["request_path"]
    check("permit projection is explicitly non-authoritative",
          projection["authority"] == "transport-only-not-a-permit"
          and request_path.exists()
          and hashlib.sha256(request_path.read_bytes()).hexdigest()
          == projection["request_sha256"], str(projection))
    budget = state()["groups"]["F-TEST|3"]["budget_snapshot"]
    check("first plan snapshots immutable family budget",
          budget["budget_attempts"] == 8 and len(budget["catalog_sha256"]) == 64,
          str(budget))
    consume_projection(failed=True)
    rc, out = run("reconcile")
    check("authority run failure excludes scientific strike",
          rc == 0 and state()["groups"]["F-TEST|3"]["strikes"] == 0, out)

    # Characterization may include the incumbent, but its band stays narrow.
    rc, out = research()
    screen = common.copy()
    screen[screen.index("--prediction-kind") + 1] = "characterization"
    screen[screen.index("--prediction") + 1] = "1.00x characterization"
    rc, out = run("plan", "--mode", "screening", "--predict-min", "0.99",
                  "--predict-max", "1.01", *screen)
    check("bounded characterization remains legal", rc == 0, out)
    screening = consume_projection()
    check("scratch namespace is computed",
          screening["ledger_namespace"] == "scratch"
          and "scratch_ledgers/CAMP-1/F-TEST/shape-3.jsonl" in screening["ledger"],
          screening["ledger"])
    scratch = Path(screening["ledger"]); scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text(json.dumps({
        "entry_id": "20260830-141000-b1c2d3", "type": "candidate",
        "profile": "scratch", "shape_id": 3, "promoted": False,
        "impl": {"sha256": hashlib.sha256((sb / "cand.py").read_bytes()).hexdigest()},
        "correctness": {"passed": True},
        "timing": {"speedup": 1.0, "wall_check": {"suspicious": False},
                   "anti_cache_check": {"suspicious": False}},
    }) + "\n")
    rc, out = run("reconcile"); check("scratch result reconciles", rc == 0, out)
    rc, out = run("screen-judge", "--direction", "F-TEST", "--shape", "3",
                  "--observed", "1.0")
    check("characterization miss adds no strike",
          rc == 0 and state()["groups"]["F-TEST|3"]["strikes"] == 0, out)

    prior = None
    for line in (loop / "gate_log.jsonl").read_text().splitlines():
        row = json.loads(line)
        family_match = (row.get("direction") == "F-TEST"
                        and row.get("shape") in (3, "3"))
        if ((family_match or row.get("group") == "F-TEST|3")
                and row.get("step") in ("reconcile", "screen_judge", "audit_finalize")):
            result = row.get("result")
            if not result and row.get("correct") is not None:
                result = "pass" if row["correct"] else "failed-correctness"
            prior = f"seq:{row['state_seq']}:{result or 'recorded'}"
    assert prior
    rc, out = research()
    confirmation = common.copy()
    confirmation[confirmation.index("--prior-family-verdict") + 1] = prior
    rc, out = run("plan", "--mode", "confirmation", *confirmation)
    check("scratch bytes cannot launder into primary confirmation",
          rc == 1 and "prior PRIMARY" in out, out)

    # Scratch bytes may enter primary only through a real optimization request;
    # the resulting row remains promotion-ineligible until a bound audit lands.
    rc, out = run("plan", "--mode", "optimization", "--predict-min", "1.08",
                  "--predict-max", "1.10", *confirmation)
    check("scratch bytes require a primary optimization attempt", rc == 0, out)
    production = consume_projection(speedup=1.09)
    production_entry = production["_entry_id"]
    rc, out = run("reconcile")
    check("primary result becomes pending independent audit", rc == 0
          and production_entry in state()["pending_audit_decisions"], out)
    strikes_before = state()["groups"]["F-TEST|3"]["strikes"]
    rc, out = run("audit-finalize", "--entry-id", production_entry)
    check("missing bound audit cannot promote or judge improvement",
          rc == 1 and "pending" in out
          and state()["groups"]["F-TEST|3"]["strikes"] == strikes_before
          and production_entry in state()["pending_audit_decisions"], out)

    rc, out = research()
    check("research can continue while promotion waits for audit", rc == 0, out)

    generated = sb / "Project/submission/torch_transformer_benchmark_submission.py"
    generated.write_text("generated_submission = True\n")
    generated_sha = hashlib.sha256(generated.read_bytes()).hexdigest()
    rc, out = run("side-evaluate", "--campaign", "CAMP-1", "--shape", "6",
                  "--submission", str(generated),
                  "--submission-sha256", generated_sha)
    check("shape-6 side request is immutable and non-promoting", rc == 0, out)
    side_projection = json.loads((loop / "permit.json").read_text())
    check("side request has no primary/scratch ledger credential",
          side_projection["request_kind"] == "side_evaluation"
          and side_projection["mode"] == "shape6"
          and side_projection["ledger"] is None
          and side_projection["promotion_allowed"] is False
          and side_projection["champion_candidate"] is False,
          str(side_projection))
    side_result = consume_projection(speedup=1.2)
    rc, out = run("reconcile")
    side_pending = state()["pending_audit_decisions"].get(side_result["_entry_id"], {})
    check("side measurement remains final-evidence-only pending audit",
          rc == 0 and side_pending.get("eligibility_scope") == "final_evidence_only",
          out)
    rc, out = run("audit-finalize", "--entry-id", side_result["_entry_id"])
    check("missing bound side audit cannot create evidence eligibility",
          rc == 1 and "pending" in out
          and side_result["_entry_id"] in state()["pending_audit_decisions"], out)

    (loop / "cards.jsonl").write_text((loop / "cards.jsonl").read_text()
        + json.dumps({"direction_family_id": "F-FREE-TEXT", "status": "open"}) + "\n")
    unknown = confirmation.copy(); unknown[unknown.index("--direction") + 1] = "F-FREE-TEXT"
    rc, out = run("plan", "--mode", "optimization", "--predict-min", "1.08",
                  "--predict-max", "1.10", *unknown)
    check("free text cannot mint family", rc == 1 and "cannot create identity" in out, out)

    (loop / "cards.jsonl").write_text((loop / "cards.jsonl").read_text() + "{}\n")
    rc, out = run("plan", "--mode", "optimization", "--predict-min", "1.08",
                  "--predict-max", "1.10", *confirmation)
    check("malformed card row fails closed", rc == 1 and "invalid schema" in out, out)

    failed = [(n, d) for n, ok, d in checks if not ok]
    print(f"{len(checks) - len(failed)}/{len(checks)} passed"
          + (" — ALL GREEN" if not failed else f" — FAILURES: {failed}"))
    shutil.rmtree(sb, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
