#!/usr/bin/env python3
"""Integration suite: the REAL controller driven against the REAL gate.

Every other suite in this repo tests one module with its neighbours replaced by
fixtures.  competence_gate_test.py overwrites trusted_controller.py with a stub
that approves any receipt, hand-builds the authority hash chain instead of
calling issue_permit/consume_permit, and sets the measurement's ``timing_args``
from the same campaign fixture the gate is about to check it against.  A
reviewer mutated ONLY the controller's ``TIMING`` constant -- a change that
wedges the gate on every campaign -- and all three existing suites stayed green.
This suite exists to be the one that goes red.

WHAT IS REAL HERE
  * Project/harness/trusted_controller.py -- imported and executed, not stubbed.
  * Project/harness/authority.py -- real Ed25519 capabilities, real hash-chained
    event log, real issue_permit / consume_permit / store_blob.
  * Project/harness/lock_manifest.py -- a real lock document over real bytes,
    signed with a real owner key and activated through the real controller.
  * Project/tools/run_gate.py -- invoked as a subprocess, exactly as an operator
    would.  Its authority-receipt check re-spawns the real controller.
  * Project/harness/sandbox.py -- the worker really runs inside bubblewrap.

THE SEAMS, AND WHERE THEY ARE
  There is no GPU in this test, so the *measurement* is replaced in places.
  The *authority* never is: no permit, no receipt, no capability, no event, no
  packet and no gate decision is faked anywhere in this file.
    1. Project/harness/candidate_worker.py is replaced, inside the temporary
       tree only, by a stdlib stub that returns plausible timing numbers in the
       exact schema the real controller demands.  The controller still builds
       the worker request, still launches it in the real jail, and still
       validates every retained sample, every summary statistic and every
       binding in the response.
    2. trusted_controller.validate_challenge_outputs is patched, because it
       builds the official model on CUDA to recompute reference outputs.  The
       patch still runs the controller's own _response_schema and re-checks the
       same request/response bindings; only the CUDA recomputation is skipped.
    3. Project/harness/profile_worker.py is replaced by STUB_PROFILE_WORKER in
       section 4 (the "live diagnostic" lane) -- and section 6 exists because
       that stub is exactly what hid a dead diagnostic lane.  See below.

WHY SECTION 6 EXISTS
  The controller's profiler request and the worker that consumes it were
  written in parallel by two agents who never ran them against each other.
  The controller emits a FLAT request; the worker read a nested
  ``gate_request`` object that no controller ever sent, and refused every real
  request with "worker request needs the immutable gate_request object".  Both
  halves reported success against their own assumptions, and THIS SUITE STAYED
  GREEN, because section 4 drives STUB_PROFILE_WORKER -- a stub written from
  the controller's schema, which therefore agrees with the controller by
  construction and can never disagree with it.
  Section 6 drives the REAL profile_worker.py through the REAL controller.
  It profiles with ``static-analysis``, the one catalog tool whose evidence is
  an explicit arithmetic model of the official dense route: it needs no GPU,
  no nsys, no ncu and no root, so nothing in the worker's request handling or
  artifact construction has to be faked to make it run.
  WHAT REMAINS FAKED IN SECTION 6: nothing in the request path, the artifact
  path, the authority path or the gate path.  What is NOT COVERED is the six
  collectors that need CUDA (nsys, ncu, torch-profiler, memory-profile,
  microbenchmark, correctness-evaluator).  They are unreachable here because
  the controller's mount plan is fixed in trusted_controller.run_diagnostic --
  it mounts no profiler and the flat request carries no tool-path override --
  so a stubbed nsys binary cannot be introduced without editing the frozen
  controller.  Those collectors are covered by profile_worker.py --selftest,
  which drives every parser against recorded real tool output, and their
  first live exercise is an owner-run diagnostic on the real box.

THE MUTATION GUARD
  The benchmark timing protocol (warmup/repeats/rounds) is written down in FIVE
  independent places in this repo.  This suite reads all five at runtime and
  asserts they are one object.  No literal 20/100/3 appears in this file: a
  test that copies the value cannot detect the value changing.
    1. torch_transformer_benchmark.py argparse defaults  (frozen ground truth)
    2. Project/harness/runner.py OFFICIAL_DEFAULTS       (frozen referee)
    3. Project/harness/trusted_controller.py TIMING      (the producer)
    4. trusted_controller.validate_shape6_packet literals (side-lane consumer)
    5. Project/tools/shape6_local_eval.py OFFICIAL_*     (side-lane producer)
  Plus the live behavioural halves: the gate accepts a campaign at exactly the
  controller's protocol and refuses every neighbouring one, and a controller
  whose protocol drifts after campaign open cannot reconcile a calibration.

Run: python3 Project/tools/tests/integration_authority_test.py   (no GPU, no
network, writes only under a temporary directory).
"""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OFFICIAL_SCRIPT = REPO / "torch_transformer_benchmark.py"
RUNNER_SRC = REPO / "Project" / "harness" / "runner.py"
CONTROLLER_SRC = REPO / "Project" / "harness" / "trusted_controller.py"
GATE_SRC = REPO / "Project" / "tools" / "run_gate.py"
SHAPE6_EVAL_SRC = REPO / "Project" / "tools" / "shape6_local_eval.py"

# Everything the controller and the gate touch.  Copied verbatim; the temporary
# tree is a real repo as far as both of them are concerned.
COPY_REQUIRED = (
    "torch_transformer_benchmark.py",
    "Project/shapes.json",
    "Project/tools/run_gate.py",
    "Project/tools/audit_authority.py",
    "Project/harness/authority.py",
    "Project/harness/lock_manifest.py",
    "Project/harness/sandbox.py",
    "Project/harness/trusted_controller.py",
    "Project/harness/runner.py",
    "Project/loop/mechanism_catalog.json",
    "Project/loop/mechanism_catalog.schema.json",
)
COPY_OPTIONAL = (
    "tensorflow_transformer_benchmark.py",
    "Project/manifest.json",
    "Project/tools/shape6_local_eval.py",
    "Project/tools/shape14_eval.py",
    "Project/submission/torch_transformer_benchmark_submission.py",
)
# Reloaded per temporary tree so two trees never share a module instance.
HARNESS_MODULES = ("trusted_controller", "authority", "lock_manifest",
                   "sandbox", "runner")

CHECKS: list[tuple[str, bool, str]] = []
SKIPS: list[tuple[str, str]] = []
SANDBOXES: list[Path] = []
_HARNESS_PATH: str | None = None


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    print(("PASS " if condition else "FAIL ") + name
          + (f"  [{detail}]" if detail and not condition else ""))


def skip(name: str, reason: str) -> None:
    """Loudly not-run.  A skip is never a pass and is repeated in the summary."""
    SKIPS.append((name, reason))
    print(f"SKIP {name}  [{reason}]")


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def neighbours(protocol: dict) -> list[tuple[str, dict]]:
    """One-step-off protocols.  Derived, never typed."""
    return [(key, {**protocol, key: protocol[key] + 1}) for key in sorted(protocol)]


# --------------------------------------------------------------------------
# The stub candidate worker: the measurement seam, and nothing else.
# --------------------------------------------------------------------------
STUB_WORKER = '''#!/usr/bin/env python3
"""Stand-in for candidate_worker.py used by integration_authority_test.py.

The real worker builds the official model on the GPU and times it.  There is no
GPU here, so this returns deterministic numbers in the exact response schema the
real controller demands.  It reads its timing protocol from the request the real
controller wrote, so it always answers the controller's own protocol -- it never
carries a protocol of its own.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
for flag in ("--request", "--candidate", "--official", "--shapes", "--output"):
    parser.add_argument(flag, required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
out = Path(args.output)

timing = request["timing_args"]
count = timing["repeats"] * timing["rounds"]


def series(base):
    # Deterministic and strictly positive, with real spread: a calibration whose
    # measured noise is exactly zero is refused by the gate.
    return [base + ((index % 7) - 3) * 0.01 for index in range(count)]


def stats(samples):
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": ordered[index],
        "min_ms": min(samples),
        "n_samples": len(samples),
        "raw_samples_ms": samples,
    }


baseline = stats(series(4.0))
candidate = stats(series(3.96))
event_speedup = baseline["median_ms"] / candidate["median_ms"]
baseline_wall = 4.10
candidate_wall = 4.06
wall_speedup = baseline_wall / candidate_wall
agreement = (max(event_speedup, wall_speedup)
             / max(min(event_speedup, wall_speedup), 1e-12))

shape = request["shape"]
outputs = []
for seed in request["seeds"]:
    name = "challenge-%d.f32" % seed
    payload = hashlib.sha256(
        ("%s-%d" % (request["challenge_nonce"], seed)).encode()).digest()
    (out / name).write_bytes(payload)
    outputs.append({
        "seed": seed,
        "filename": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "shape": [shape["batch_size"], shape["seq_len"], shape["d_model"]],
        "dtype": "float32",
        "nbytes": len(payload),
    })

response = {
    "schema_version": 1,
    "request_id": request["request_id"],
    "challenge_nonce": request["challenge_nonce"],
    "candidate_sha256": request["candidate_sha256"],
    "official_sha256": request["official_sha256"],
    "shapes_sha256": request["shapes_sha256"],
    "shape_id": request["shape_id"],
    "correctness": {
        "passed": True,
        "baseline_invariant": True,
        "anti_cache_passed": True,
        "trials": [{"seed": seed, "passed": True} for seed in request["seeds"]],
    },
    "challenge_outputs": outputs,
    "supporting_timing": {
        "baseline": baseline,
        "candidate": candidate,
        "event_speedup": event_speedup,
        "baseline_wall_ms_per_iter": baseline_wall,
        "candidate_wall_ms_per_iter": candidate_wall,
        "wall_speedup": wall_speedup,
        "event_wall_speedup_agreement_ratio": agreement,
        "suspicious": agreement > 1.25,
        "authority": "supporting-worker-measurement",
    },
    "effective_numerical_state": {
        "float32_matmul_precision": request["numerical"]["matmul_precision"],
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": True,
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None,
        "NVIDIA_TF32_OVERRIDE": None,
    },
    "environment": {"python": "stub", "torch": "stub", "cuda": "stub",
                    "gpu": "integration-test-stub-device", "driver": "stub",
                    "triton": "stub"},
}
(out / "response.json").write_text(json.dumps(response))
print("stub candidate worker: %d retained samples per side" % count)
'''

STUB_PROFILE_WORKER = '''#!/usr/bin/env python3
"""Stand-in for profile_worker.py used by integration_authority_test.py.

The real worker runs nsys/ncu/torch.profiler.  Neither is reachable without a
GPU, so this emits an artifact in the shape the controller and the gate both
demand, with the metric names taken from the request rather than invented here.
"""
import argparse
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
out = Path(args.output)

for key in ("target_path", "target_path_alias", "official_path", "shapes_path"):
    assert Path(request[key]).is_file(), "missing mount %s" % key

(out / "raw_trace.csv").write_text("kernel,launches\\nfused_forward,7\\n")
metrics = {}
for names in request["required_metrics"].values():
    for name in names:
        metrics[name] = 7.0
artifact = {
    "schema_version": 1,
    "profile_record_id": request["profile_record_id"],
    "request_id": request["request_id"],
    "campaign_id": request["campaign_id"],
    "shape": request["shape_id"],
    "target_sha256": request["target_sha256"],
    "tool": request["tool"],
    "tool_version": "integration-test-stub 0.1",
    "created_epoch": time.time(),
    "machine_state_sha256": request["machine_state_sha256"],
    "route": request["route"],
    "metrics": metrics,
    "supported_bottlenecks": request["supported_bottlenecks"],
    "raw_artifacts": [{"path": "raw_trace.csv", "sha256": ""}],
    "gate_request_sha256": request["gate_request_sha256"],
}
(out / request["artifact_filename"]).write_text(json.dumps(artifact))
print("stub profile worker: %d metric(s)" % len(metrics))
'''


# --------------------------------------------------------------------------
# Reading the timing protocol out of each place that writes it down.
# --------------------------------------------------------------------------
def _literal(node: ast.AST):
    return ast.literal_eval(node)


def official_script_protocol(path: Path) -> dict:
    """The frozen benchmark's own argparse defaults: the ground truth.

    The judges run this script at its defaults.  Any protocol that is not this
    one is measuring something the competition does not score.
    """
    flags = {"--warmup": "warmup", "--repeats": "repeats",
             "--benchmark-rounds": "rounds"}
    found: dict[str, int] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in flags):
            for keyword in node.keywords:
                if keyword.arg == "default":
                    found[flags[node.args[0].value]] = _literal(keyword.value)
    return found


def runner_protocol(path: Path) -> dict:
    """Project/harness/runner.py OFFICIAL_DEFAULTS, the frozen referee's copy."""
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "OFFICIAL_DEFAULTS"
                        for t in node.targets)
                and isinstance(node.value, ast.Call)):
            values = {kw.arg: _literal(kw.value) for kw in node.value.keywords
                      if kw.arg is not None}
            return {"warmup": values["warmup"], "repeats": values["repeats"],
                    "rounds": values["benchmark_rounds"]}
    return {}


def module_constants(path: Path, names: dict[str, str]) -> dict:
    """Module-level integer constants, renamed onto the protocol's key names."""
    found: dict[str, int] = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                found[names[target.id]] = _literal(node.value)
    return found


def shape6_side_gate_protocol(path: Path) -> dict:
    """The literals validate_shape6_packet demands of a side-evaluation packet.

    These are written as bare numbers inside the function rather than read from
    TIMING, so they are an independent copy of the protocol and can drift.
    """
    fields = {"warmups": "warmup", "repeats_per_round": "repeats",
              "round_count": "rounds"}
    found: dict[str, int] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.FunctionDef)
                and node.name == "validate_shape6_packet"):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Compare):
                continue
            left = inner.left
            if (isinstance(left, ast.Call)
                    and isinstance(left.func, ast.Attribute)
                    and left.func.attr == "get"
                    and left.args
                    and isinstance(left.args[0], ast.Constant)
                    and left.args[0].value in fields
                    and len(inner.comparators) == 1
                    and isinstance(inner.comparators[0], ast.Constant)):
                found[fields[left.args[0].value]] = inner.comparators[0].value
    return found


def controller_profile_keys(path: Path) -> set:
    """trusted_controller.PROFILE_ARTIFACT_KEYS -- the producer's key set."""
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name)
                        and t.id == "PROFILE_ARTIFACT_KEYS"
                        for t in node.targets)):
            value = node.value
            # Written as frozenset({...}); unwrap the one call literal_eval
            # will not touch, then evaluate the set it wraps.
            if (isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id in {"frozenset", "set"}
                    and len(value.args) == 1):
                value = value.args[0]
            return set(_literal(value))
    return set()


def gate_profile_keys(path: Path) -> set:
    """run_gate._reconcile_authority_diagnostic's inline `required` key set."""
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.FunctionDef)
                and node.name == "_reconcile_authority_diagnostic"):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "required"
                            for t in inner.targets)
                    and isinstance(inner.value, ast.Set)):
                return set(_literal(inner.value))
    return set()


# --------------------------------------------------------------------------
# A temporary tree that is a real repo to the controller and the gate.
# --------------------------------------------------------------------------
def build_tree(profile_worker_source: str | None = None) -> Path:
    """A temporary repo.  ``profile_worker_source`` defaults to the stub.

    Section 6 passes the REAL worker's source here; every other section gets
    STUB_PROFILE_WORKER, which is why section 6 has to exist.
    """
    root = Path(tempfile.mkdtemp(prefix="integration_authority_"))
    SANDBOXES.append(root)
    for relative in COPY_REQUIRED:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    for relative in COPY_OPTIONAL:
        source = REPO / relative
        if source.is_file():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    for relative in ("Project/research", "Project/results", "Project/audits",
                     "Project/authority/blobs", "Project/submission",
                     "Project/kernels", "Project/loop"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    # The seam, written into the tree so the real controller mounts it.
    (root / "Project/harness/candidate_worker.py").write_text(STUB_WORKER)
    (root / "Project/harness/profile_worker.py").write_text(
        profile_worker_source if profile_worker_source is not None
        else STUB_PROFILE_WORKER)
    (root / "Project/research/INDEX.md").write_text("integration index\n")
    (root / "Project/research/note-a.md").write_text("note a\n")
    (root / "Project/research/note-b.md").write_text("note b\n")
    (root / "Project/results/JOURNAL.jsonl").write_text("")
    (root / "Project/audits/verdicts.jsonl").write_text("")
    (root / "Project/loop/cards.jsonl").write_text(
        json.dumps({"direction_family_id": "F-TEST", "status": "open"}) + "\n")
    (root / "machine.json").write_text(json.dumps({
        "captured_epoch": time.time(), "gpu_utilization": 0,
        "competing_processes": []}))
    target = root / "Project/submission/torch_transformer_benchmark_submission.py"
    if not target.is_file():
        target.write_text("# integration-test profile target\n")
    return root


def load_harness(root: Path):
    """Import this tree's controller, never the repo's and never a cached one."""
    global _HARNESS_PATH
    if _HARNESS_PATH is not None and _HARNESS_PATH in sys.path:
        sys.path.remove(_HARNESS_PATH)
    for name in HARNESS_MODULES:
        sys.modules.pop(name, None)
    _HARNESS_PATH = str(root / "Project" / "harness")
    sys.path.insert(0, _HARNESS_PATH)
    importlib.invalidate_caches()
    import trusted_controller  # noqa: PLC0415

    return trusted_controller


def install_lock(root: Path, controller_module):
    """Real keys, a real lock document over the real bytes, really activated."""
    from authority import AuthorityPaths, canonical_json  # noqa: PLC0415
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: PLC0415
        Ed25519PrivateKey,
    )
    from lock_manifest import create_lock_document  # noqa: PLC0415

    paths = AuthorityPaths(root)
    paths.directory.mkdir(parents=True, exist_ok=True)
    owner = Ed25519PrivateKey.generate()
    critic = Ed25519PrivateKey.generate()
    paths.public_key.write_bytes(owner.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo))
    paths.critic_public_key.write_bytes(critic.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo))
    protected = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.is_file() and "Project/authority" not in str(path.relative_to(root))
    )
    document = create_lock_document(
        root=root, protected_files=protected,
        owner_public_key=paths.public_key,
        critic_public_key=paths.critic_public_key,
        rules_snapshot_sha256="a" * 64, epoch="integration",
        lock_id="lock-integration-0123456789abcdef")
    paths.lock_manifest.write_bytes(canonical_json(document) + b"\n")
    paths.lock_signature.write_text(
        base64.b64encode(owner.sign(canonical_json(document))).decode("ascii"))
    controller = controller_module.TrustedController(root)
    activation = controller.activate_lock(
        capability(root, owner, ["lock.activate"], "bootstrap", "lock"))
    return owner, controller, activation


def capability(root: Path, owner, actions, campaign_id: str, nonce: str) -> Path:
    from authority import canonical_json  # noqa: PLC0415

    now = datetime.now(timezone.utc)
    value = {
        "schema_version": 1, "capability_id": f"cap-{nonce}", "role": "owner",
        "campaign_id": campaign_id, "actions": list(actions), "targets": ["*"],
        "issued_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "max_uses": 1, "nonce": nonce,
    }
    value["signature"] = base64.b64encode(
        owner.sign(canonical_json(value))).decode("ascii")
    path = root / f"capability-{nonce}.json"
    path.write_text(json.dumps(value))
    return path


def gate_runner(root: Path):
    def run(*args: str) -> tuple[int, str]:
        process = subprocess.run(
            [sys.executable, str(root / "Project/tools/run_gate.py"), *args],
            cwd=str(root), text=True, capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        return process.returncode, (process.stdout + process.stderr).strip()

    return run


def patch_gpu_seam(controller_module) -> None:
    """Replace only the CUDA reference recomputation, keeping its schema work."""
    module = controller_module

    def without_cuda(*, response, request, output_dir):
        module._response_schema(response)
        module.validate_worker_environment(response.get("environment"))
        for field in ("request_id", "challenge_nonce", "candidate_sha256",
                      "official_sha256", "shapes_sha256", "shape_id"):
            if response.get(field) != request[field]:
                raise module.ControllerRefusal(
                    f"worker response binding mismatch: {field}")
        worker_correctness = response.get("correctness")
        if not isinstance(worker_correctness, dict) or set(worker_correctness) != {
                "passed", "baseline_invariant", "anti_cache_passed", "trials"}:
            raise module.ControllerRefusal("worker correctness schema mismatch")
        if len(worker_correctness["trials"]) != len(request["seeds"]):
            raise module.ControllerRefusal("worker correctness trial count mismatch")
        return {"passed": True, "authority": "trusted-controller",
                "trials": [{"seed": seed, "passed": True}
                           for seed in request["seeds"]]}

    module.validate_challenge_outputs = without_cuda


def open_campaign(root: Path, owner, controller, run, campaign_id: str,
                  timing_config: dict) -> tuple[int, str]:
    spec = {
        "schema_version": 1, "campaign_id": campaign_id,
        "max_total_attempts": 10, "max_calibrations_per_shape": 2,
        "max_total_calibrations": 4, "stall_window": 4,
        "timing_config": timing_config, "score_scenarios": ["median-speedup"],
    }
    spec_path = root / f"campaign-{campaign_id}.json"
    spec_path.write_text(json.dumps(spec))
    receipt = controller.authorize(
        capability_path=capability(root, owner, ["open_campaign"], campaign_id,
                                   f"campaign-{campaign_id}"),
        action="open_campaign", target=f"campaign:{campaign_id}",
        subject_sha256=digest(spec), campaign_id=campaign_id)
    return run("campaign-open", "--spec", str(spec_path),
               "--authority-receipt", str(root / receipt["receipt_path"]))


def latest_request(root: Path) -> tuple[str, dict]:
    state = json.loads((root / "Project/loop/gate_state.json").read_text())
    request_sha = state["request_shas"][-1]
    path = root / "Project/loop/requests" / f"{request_sha}.json"
    return request_sha, json.loads(path.read_text())


def authority_events(root: Path) -> list[dict]:
    path = root / "Project/authority/events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def measurements(root: Path) -> list[dict]:
    return [event["payload"] for event in authority_events(root)
            if event["kind"] == "measurement_recorded"]


# --------------------------------------------------------------------------
# Section 1 -- one protocol, five writers.  No literal appears in this file.
# --------------------------------------------------------------------------
def section_protocol_identity(protocol: dict) -> None:
    official = official_script_protocol(OFFICIAL_SCRIPT)
    referee = runner_protocol(RUNNER_SRC)
    side_gate = shape6_side_gate_protocol(CONTROLLER_SRC)
    side_producer = module_constants(SHAPE6_EVAL_SRC, {
        "OFFICIAL_WARMUPS": "warmup", "OFFICIAL_REPEATS": "repeats",
        "OFFICIAL_ROUNDS": "rounds"}) if SHAPE6_EVAL_SRC.is_file() else None

    check("the frozen benchmark publishes a complete timing protocol",
          set(official) == {"warmup", "repeats", "rounds"}, repr(official))
    check("controller TIMING is the frozen benchmark's own protocol",
          protocol == official,
          f"controller={protocol} official_script={official}")
    check("the frozen runner's OFFICIAL_DEFAULTS is the same protocol",
          referee == official, f"runner={referee} official_script={official}")
    primary_ok, primary_detail = _is_primary_agrees(protocol)
    check("runner.is_primary accepts exactly the controller's protocol",
          primary_ok, primary_detail)
    check("the shape-6 side lane demands the same protocol as TIMING",
          side_gate == protocol,
          f"validate_shape6_packet={side_gate} TIMING={protocol}")
    if side_producer is not None:
        check("shape6_local_eval produces the protocol the controller demands",
              side_producer == side_gate,
              f"evaluator={side_producer} controller={side_gate}")
    else:
        check("shape6_local_eval is present to be checked", False,
              "Project/tools/shape6_local_eval.py is missing")


def _is_primary_agrees(protocol: dict) -> tuple[bool, str]:
    """The frozen referee's own verdict on the controller's protocol.

    runner.is_primary decides whether a measurement counts as the official
    primary profile.  If the controller's protocol is not primary, every number
    the controller ever records is off-profile, whatever the gate thinks.
    """
    import runner  # noqa: PLC0415

    class Args:
        pass

    def verdict(values: dict) -> bool:
        args = Args()
        args.dtype = "float32"
        args.warmup = values["warmup"]
        args.repeats = values["repeats"]
        args.rounds = values["rounds"]
        return bool(runner.is_primary(args))

    if not verdict(protocol):
        return False, (f"runner.is_primary calls the controller's protocol "
                       f"{protocol} NOT primary; it would record off-profile "
                       f"measurements")
    accepted = [values for _, values in neighbours(protocol) if verdict(values)]
    if accepted:
        return False, f"runner.is_primary also accepts {accepted}"
    return True, ""


# --------------------------------------------------------------------------
# Section 2 -- the producer/consumer key set for the diagnostic lane.
# --------------------------------------------------------------------------
def section_profile_key_sets() -> None:
    producer = controller_profile_keys(CONTROLLER_SRC)
    consumer = gate_profile_keys(GATE_SRC)
    check("the controller publishes a profile artifact key set",
          bool(producer), repr(sorted(producer)))
    check("the gate demands a profile artifact key set", bool(consumer),
          repr(sorted(consumer)))
    check("profile artifact key set is one object across controller and gate",
          producer == consumer,
          f"controller_only={sorted(producer - consumer)} "
          f"gate_only={sorted(consumer - producer)}")


# --------------------------------------------------------------------------
# Section 3 -- the live lane: real controller, real gate, real permit.
# --------------------------------------------------------------------------
def prepare_tree(profile_worker_source: str | None = None):
    """A fresh tree with a real, activated lock and a working gate CLI.

    Each tree is finished before the next is built: load_harness swaps the
    process's harness modules, so two trees are never live at once.
    """
    root = build_tree(profile_worker_source)
    controller_module = load_harness(root)
    owner, controller, activation = install_lock(root, controller_module)
    patch_gpu_seam(controller_module)
    run = gate_runner(root)
    run("init")
    return root, controller_module, owner, controller, run, activation


def section_campaign_open_protocol(protocol: dict) -> None:
    """The gate must accept exactly the controller's protocol, and no other.

    Every neighbour gets its own tree: a gate that wrongly ACCEPTS one would
    otherwise occupy the active-campaign slot and make the next probe refuse
    for the wrong reason, turning a real failure into a green one.
    """
    refusals: list[str] = []
    for key, candidate in neighbours(protocol):
        root, module, owner, controller, run, _ = prepare_tree()
        code, output = open_campaign(root, owner, controller, run,
                                     f"neighbour-{key}", candidate)
        state = json.loads((root / "Project/loop/gate_state.json").read_text())
        opened = f"neighbour-{key}" in state.get("campaigns", {})
        check(f"a campaign one step off on {key!r} is refused",
              code != 0 and not opened,
              f"rc={code} opened={opened} {output[:300]}")
        if code != 0 and not opened:
            refusals.append(output)
    check("every refusal names the controller as the protocol's authority",
          len(refusals) == len(protocol)
          and all("controller" in text for text in refusals),
          (refusals[0][:400] if refusals else "no protocol refusal captured"))


def section_live_lane(protocol: dict) -> None:
    root, controller_module, owner, controller, run, activation = prepare_tree()
    live_protocol = dict(controller_module.TIMING)
    check("the tree under test carries the controller's own protocol",
          live_protocol == protocol, f"{live_protocol} vs {protocol}")
    check("the real lock activates through the real controller",
          activation.get("lock_id") == "lock-integration-0123456789abcdef",
          json.dumps(activation)[:200])

    # -- campaign-open accepts exactly the controller's protocol ------------
    code, output = open_campaign(root, owner, controller, run, "integration",
                                 live_protocol)
    check("the controller's own protocol opens a campaign", code == 0,
          output[:400])
    if code != 0:
        return
    state = json.loads((root / "Project/loop/gate_state.json").read_text())
    opened = state.get("campaigns", {}).get("integration", {})
    check("the opened campaign records the controller's protocol",
          opened.get("timing_config") == live_protocol,
          json.dumps(opened.get("timing_config")))

    # -- calibrate -> permit -> real controller run -------------------------
    code, output = run("calibrate", "--campaign", "integration", "--shape", "3",
                       "--machine-state", str(root / "machine.json"))
    check("gate emits a calibration request", code == 0, output[:400])
    if code != 0:
        return
    request_sha, request = latest_request(root)
    check("the calibration request carries the campaign's protocol",
          request["timing_config"] == live_protocol,
          json.dumps(request.get("timing_config")))

    permit = controller.issue_permit(
        root / "Project/loop/requests" / f"{request_sha}.json",
        capability(root, owner, ["permit.issue"], "integration", "permit-1"))
    check("the real controller issues a one-use permit for that request",
          permit.get("request_sha256") == request_sha
          and permit.get("mode") == "calibration",
          json.dumps(permit)[:300])

    try:
        result = controller.run_primary(
            permit_id=permit["permit_id"], shape_id=3, impl_path=None,
            timeout_seconds=120)
        detail = json.dumps(result)[:400]
    except Exception as exc:
        result = {}
        detail = f"{type(exc).__name__}: {exc}"
    check("the real controller completed a measurement",
          isinstance(result, dict)
          and bool(result.get("entry_id"))
          and bool(result.get("measurement_event_sha256"))
          and bool(result.get("packet_sha256")),
          detail)

    recorded = measurements(root)
    check("exactly one measurement reached the authority log",
          len(recorded) == 1, f"{len(recorded)} measurement events")
    stamped = recorded[0].get("timing_args") if recorded else None
    check("the recorded measurement is stamped with the controller's protocol",
          stamped == live_protocol, f"stamped={stamped} controller={live_protocol}")

    # -- the permit is spent ------------------------------------------------
    reused = ""
    try:
        controller.run_primary(permit_id=permit["permit_id"], shape_id=3,
                               impl_path=None, timeout_seconds=120)
        spent = False
    except Exception as exc:  # ControllerRefusal / AuthorityError
        spent = True
        reused = f"{type(exc).__name__}: {exc}"
    check("the permit cannot be spent twice", spent, reused)

    # -- reconcile ----------------------------------------------------------
    code, output = run("reconcile")
    check("the gate reconciles the real controller's measurement",
          code == 0, output[:600])
    state = json.loads((root / "Project/loop/gate_state.json").read_text())
    calibrations = state["campaigns"]["integration"]["calibrations"]
    check("a calibration is bound in gate state for the shape",
          "3" in calibrations, json.dumps(calibrations)[:300])
    bound = calibrations.get("3", {})
    check("the bound calibration carries the controller's protocol",
          bound.get("timing_config") == live_protocol,
          json.dumps(bound.get("timing_config")))
    check("the bound calibration has a positive measured noise floor",
          isinstance(bound.get("noise"), float) and bound["noise"] > 0,
          json.dumps(bound.get("noise")))
    check("the bound calibration points at the real measurement event",
          bound.get("measurement_event_sha256") in {
              event["event_sha256"] for event in authority_events(root)
              if event["kind"] == "measurement_recorded"},
          json.dumps(bound.get("measurement_event_sha256")))
    check("the request is settled after reconcile",
          request_sha in state.get("settled_request_shas", []),
          json.dumps(state.get("settled_request_shas"))[:300])

    code, output = run("reconcile")
    again = json.loads((root / "Project/loop/gate_state.json").read_text())
    check("reconcile is idempotent",
          code == 0
          and again["campaigns"]["integration"]["calibrations"] == calibrations,
          output[:400])

    section_live_diagnostic(root, owner, controller, run)


# --------------------------------------------------------------------------
# Section 4 -- the diagnostic lane, end to end, through the same authority.
# --------------------------------------------------------------------------
def section_live_diagnostic(root: Path, owner, controller, run) -> None:
    target = root / "Project/submission/torch_transformer_benchmark_submission.py"
    target_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    code, output = run(
        "diagnostic", "--campaign", "integration", "--shape", "3",
        "--target-sha256", target_sha, "--tool", "nsys",
        "--supports", "launch-overhead",
        "--question", "Is the graphed forward still collapsing launches on "
                      "shape 3, or has the eager fallback returned?",
        "--route", "graph-replay")
    check("gate emits a diagnostic request", code == 0, output[:400])
    if code != 0:
        return
    request_sha, request = latest_request(root)
    permit = controller.issue_permit(
        root / "Project/loop/requests" / f"{request_sha}.json",
        capability(root, owner, ["permit.issue"], "integration", "permit-2"))
    check("the controller issues a diagnostic permit",
          permit.get("mode") == "diagnostic", json.dumps(permit)[:300])

    refusal = ""
    try:
        result = controller.run_diagnostic(
            permit_id=permit["permit_id"], target_path=target,
            timeout_seconds=180)
        ran = isinstance(result, dict) and bool(result.get("profile_record_id"))
    except Exception as exc:
        ran = False
        refusal = f"{type(exc).__name__}: {exc}"
    check("the real controller produced and bound a profile artifact", ran,
          refusal)
    if not ran:
        return

    code, output = run("reconcile")
    check("the gate reconciles the diagnostic the controller produced",
          code == 0, output[:600])
    state = json.loads((root / "Project/loop/gate_state.json").read_text())
    record = state.get("profiles", {}).get(request["profile_record_id"])
    check("a profile record exists in gate state", record is not None,
          json.dumps(list(state.get("profiles", {})))[:300])
    if record is None:
        return
    catalog = json.loads(
        (root / "Project/loop/mechanism_catalog.json").read_text())
    required = catalog["bottlenecks"]["launch-overhead"]["required_metrics"]
    check("the profile record carries the metrics the catalog demands",
          all(metric in record["metrics"] for metric in required),
          f"required={required} recorded={sorted(record['metrics'])}")


# --------------------------------------------------------------------------
# Section 5 -- controller-side protocol drift after campaign open.
# --------------------------------------------------------------------------
def section_protocol_drift() -> None:
    """The other half of the guard: the gate must not bind a mismatch.

    Campaign-open now reads the controller's protocol, so a mismatch can only
    arise by the controller changing AFTER a campaign is open.  That is exactly
    the reviewer's mutation, arriving one commit late.  Reconcile must refuse
    it, and must bind nothing.
    """
    root, controller_module, owner, controller, run, _ = prepare_tree()
    original = dict(controller_module.TIMING)
    code, output = open_campaign(root, owner, controller, run, "drift", original)
    check("the drift lane opens a campaign at the controller's protocol",
          code == 0, output[:400])
    if code != 0:
        return
    code, output = run("calibrate", "--campaign", "drift", "--shape", "3",
                       "--machine-state", str(root / "machine.json"))
    check("the drift lane emits a calibration request", code == 0, output[:400])
    if code != 0:
        return
    request_sha, _ = latest_request(root)
    permit = controller.issue_permit(
        root / "Project/loop/requests" / f"{request_sha}.json",
        capability(root, owner, ["permit.issue"], "drift", "permit-drift"))

    drifted = {**original, "repeats": original["repeats"] + 1}
    controller_module.TIMING = drifted
    try:
        controller.run_primary(permit_id=permit["permit_id"], shape_id=3,
                               impl_path=None, timeout_seconds=120)
        recorded = measurements(root)
        stamped = recorded[-1].get("timing_args") if recorded else None
        detail = f"stamped={stamped} drifted={drifted}"
    except Exception as exc:
        stamped = None
        detail = f"{type(exc).__name__}: {exc}"
    finally:
        controller_module.TIMING = original
    check("the drifted controller stamped its new protocol",
          stamped == drifted, detail)
    if stamped != drifted:
        return

    code, output = run("reconcile")
    state = json.loads((root / "Project/loop/gate_state.json").read_text())
    calibrations = state["campaigns"]["drift"]["calibrations"]
    check("the gate refuses a measurement whose protocol drifted from the "
          "campaign", code != 0 and not calibrations,
          f"rc={code} calibrations={json.dumps(calibrations)} {output[:300]}")


# --------------------------------------------------------------------------
# Section 6 -- the REAL profile worker against the REAL controller.
#
# Sections 1-5 all pass with a profile worker that cannot parse a single real
# request, because section 4 drives STUB_PROFILE_WORKER.  This section is the
# one that goes red for that.  See "WHY SECTION 6 EXISTS" in the module
# docstring for what is and is not faked here.
# --------------------------------------------------------------------------
def locate_real_profile_worker() -> Path | None:
    """The real profile_worker.py, or None if it is not on this box yet.

    Order: ``$PROFILE_WORKER_SRC``, then the committed location.  There is no
    third guess and no fallback to the stub: a suite that quietly profiles a
    stub while claiming to test the real worker is the exact defect this
    section exists to prevent.
    """
    candidates: list[Path] = []
    override = os.environ.get("PROFILE_WORKER_SRC", "").strip()
    if override:
        candidates.append(Path(override))
    candidates.append(REPO / "Project" / "harness" / "profile_worker.py")
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve()
    return None


def import_profile_worker(path: Path):
    """Import the worker as a module so its parser can be called in process."""
    # An explicit source loader, so a staged copy under any filename (the
    # owner has not placed this file in Project/harness yet) still imports.
    spec = importlib.util.spec_from_loader(
        "real_profile_worker",
        importlib.machinery.SourceFileLoader("real_profile_worker", str(path)))
    module = importlib.util.module_from_spec(spec)
    sys.modules["real_profile_worker"] = module
    spec.loader.exec_module(module)
    return module


def section_real_profile_worker(protocol: dict) -> None:
    worker_path = locate_real_profile_worker()
    if worker_path is None:
        skip("the real profile worker runs the diagnostic lane end to end",
             "profile_worker.py is not at Project/harness/profile_worker.py "
             "and $PROFILE_WORKER_SRC is unset; nothing was tested")
        return
    print(f"real profile worker under test: {worker_path}")
    source = worker_path.read_text()

    try:
        worker = import_profile_worker(worker_path)
    except Exception as exc:
        check("the real profile worker imports", False, f"{type(exc).__name__}: {exc}")
        return
    check("the real profile worker imports", True)

    root, controller_module, owner, controller, run, _ = prepare_tree(source)
    code, output = open_campaign(root, owner, controller, run, "realworker",
                                 protocol)
    check("the real-worker lane opens a campaign", code == 0, output[:400])
    if code != 0:
        return

    # ---- the contract itself: one producer, one consumer, one key set -----
    # getattr throughout: a worker that has drifted far enough to be missing
    # one of these must produce a clean red line, not a traceback that hides
    # every check after it.
    producer_artifact_keys = set(controller_module.PROFILE_ARTIFACT_KEYS)
    consumer_artifact_keys = set(getattr(worker, "ARTIFACT_REQUIRED_KEYS", ()) or ())
    check("the worker's artifact key set is the controller's",
          producer_artifact_keys == consumer_artifact_keys,
          f"controller_only={sorted(producer_artifact_keys - consumer_artifact_keys)} "
          f"worker_only={sorted(consumer_artifact_keys - producer_artifact_keys)}")

    target = root / "Project/submission/torch_transformer_benchmark_submission.py"
    target_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    # static-analysis: the one catalog tool that needs no GPU, no profiler
    # binary and no root, so the real worker runs with nothing stubbed.
    code, output = run(
        "diagnostic", "--campaign", "realworker", "--shape", "3",
        "--target-sha256", target_sha, "--tool", "static-analysis",
        "--supports", "quadratic-materialization",
        "--question", "Does the dense route materialise an O(S^2) score "
                      "matrix large enough to bound shape 3?",
        "--route", "official-dense-attention")
    check("the gate emits a static-analysis diagnostic request", code == 0,
          output[:400])
    if code != 0:
        return
    request_sha, gate_request = latest_request(root)

    # The REAL producer function, on the REAL gate request.  This is the exact
    # document run_diagnostic hands the worker, built by the same code.
    fields = controller_module._diagnostic_request_fields(gate_request)
    worker_request = controller_module._profile_worker_request(
        fields=fields, gate_request=gate_request,
        gate_request_sha256=request_sha, campaign_id="realworker",
        shape_id=gate_request["shape"], target_sha256=target_sha,
        target_destination="/work/target.py", target_alias="/work/candidate.py",
        machine_state_sha256="a" * 64, timeout_seconds=600)
    declared = set(getattr(worker, "CONTROLLER_REQUEST_KEYS", ()) or ())
    check("the controller's request key set is the worker's "
          "CONTROLLER_REQUEST_KEYS",
          bool(declared) and set(worker_request) == declared,
          f"controller_only={sorted(set(worker_request) - declared)} "
          f"worker_only={sorted(declared - set(worker_request))}"
          if declared else "the worker declares no CONTROLLER_REQUEST_KEYS")
    parsed = None
    detail = ""
    try:
        parsed = worker.parse_worker_request(worker_request)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
    check("the real worker parses a real controller-built request",
          parsed is not None, detail)
    if parsed is None:
        return
    try:
        basename = worker.artifact_basename(parsed)
    except Exception as exc:
        basename = f"{type(exc).__name__}: {exc}"
    check("the worker reads the artifact filename the controller named",
          basename == fields["artifact_name"],
          f"{basename} != {fields['artifact_name']}")
    check("the worker takes the artifact's shape from shape_id, not the "
          "shape record", parsed.get("shape_id") == gate_request["shape"],
          json.dumps(parsed.get("shape_id"))[:120])
    check("the worker takes the machine-state hash from the controller",
          parsed.get("machine_state_sha256") == "a" * 64,
          json.dumps(parsed.get("machine_state_sha256"))[:120])
    check("the worker takes its bottleneck terms from the controller's "
          "pinned required_metrics",
          parsed.get("required_metrics") == fields["required_metrics"],
          json.dumps(parsed.get("required_metrics"))[:200])

    # ---- and now the whole lane, for real ---------------------------------
    permit = controller.issue_permit(
        root / "Project/loop/requests" / f"{request_sha}.json",
        capability(root, owner, ["permit.issue"], "realworker", "permit-real"))
    check("the controller issues a diagnostic permit for the real worker",
          permit.get("mode") == "diagnostic", json.dumps(permit)[:300])
    refusal = ""
    result = None
    try:
        result = controller.run_diagnostic(
            permit_id=permit["permit_id"], target_path=target,
            timeout_seconds=900)
    except Exception as exc:
        refusal = f"{type(exc).__name__}: {exc}"
    check("the real worker produced an artifact the real controller accepted",
          isinstance(result, dict) and bool(result.get("profile_record_id")),
          refusal)
    if not isinstance(result, dict):
        return

    code, output = run("reconcile")
    check("the gate reconciles the REAL worker's diagnostic", code == 0,
          output[:600])
    state = json.loads((root / "Project/loop/gate_state.json").read_text())
    record = state.get("profiles", {}).get(gate_request["profile_record_id"])
    check("a profile record from the real worker exists in gate state",
          record is not None,
          json.dumps(list(state.get("profiles", {})))[:300])
    if record is None:
        return
    catalog = json.loads(
        (root / "Project/loop/mechanism_catalog.json").read_text())
    required = catalog["bottlenecks"]["quadratic-materialization"]["required_metrics"]
    check("the real worker's record carries the metrics the catalog demands",
          all(metric in record["metrics"] for metric in required),
          f"required={required} recorded={sorted(record['metrics'])}")

    artifact = json.loads(
        (root / result["profile_artifact_path"]).read_text())
    check("the artifact's shape is the integer shape id",
          artifact["shape"] == gate_request["shape"],
          json.dumps(artifact["shape"]))

    # The machine-state conflict, settled the controller's way.
    raw_by_name = {Path(item["path"]).name: item
                   for item in artifact["raw_artifacts"]}
    controller_state = raw_by_name.get(
        controller_module.CONTROLLER_MACHINE_STATE)
    check("the controller's own machine-state document is in the evidence",
          controller_state is not None, json.dumps(sorted(raw_by_name))[:300])
    check("machine_state_sha256 is the hash of THAT file, not the worker's",
          controller_state is not None
          and artifact["machine_state_sha256"] == controller_state["sha256"]
          and hashlib.sha256(
              (root / controller_state["path"]).read_bytes()).hexdigest()
          == artifact["machine_state_sha256"],
          json.dumps(artifact["machine_state_sha256"]))
    worker_state = raw_by_name.get(
        getattr(worker, "WORKER_MACHINE_STATE_NAME", "worker_machine_state.json"))
    check("the worker's richer capture survives as a separate raw artifact",
          worker_state is not None, json.dumps(sorted(raw_by_name))[:300])
    check("and it is not the authoritative hash",
          worker_state is not None
          and worker_state["sha256"] != artifact["machine_state_sha256"])
    check("every raw artifact the record cites is on disk at its stated hash",
          all((root / item["path"]).is_file()
              and hashlib.sha256((root / item["path"]).read_bytes()).hexdigest()
              == item["sha256"]
              for item in artifact["raw_artifacts"]),
          json.dumps([item["path"] for item in artifact["raw_artifacts"]])[:300])


def main() -> int:
    sys.dont_write_bytecode = True
    print("integration_authority_test — real controller, real gate, "
          "real authority\n")
    root_probe = build_tree()
    controller_module = load_harness(root_probe)
    protocol = dict(controller_module.TIMING)
    print(f"controller timing protocol under test: "
          f"{json.dumps(protocol, sort_keys=True)}\n")

    print("-- one protocol, five writers ------------------------------------")
    section_protocol_identity(protocol)
    print("\n-- diagnostic artifact contract ---------------------------------")
    section_profile_key_sets()
    print("\n-- campaign-open binds the controller's protocol -----------------")
    section_campaign_open_protocol(protocol)
    print("\n-- live lane: gate -> permit -> controller -> gate ---------------")
    section_live_lane(protocol)
    print("\n-- controller protocol drift after campaign open -----------------")
    section_protocol_drift()
    print("\n-- the REAL profile worker, end to end --------------------------")
    section_real_profile_worker(protocol)

    for path in SANDBOXES:
        shutil.rmtree(path, ignore_errors=True)
    failed = [(name, detail) for name, ok, detail in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} passed"
          + (f", {len(SKIPS)} SKIPPED" if SKIPS else "")
          + (" — ALL GREEN" if not failed and not SKIPS else ""))
    for name, detail in failed:
        print(f"  FAILED: {name}" + (f"\n          {detail}" if detail else ""))
    for name, reason in SKIPS:
        print(f"  SKIPPED: {name}\n           {reason}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
