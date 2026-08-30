#!/usr/bin/env python3
"""Diagnostic profiler worker: the missing producer of gate profile evidence.

The gate (``Project/tools/run_gate.py``) refuses to open a direction card
without ``--counter-evidence <profile_record_id>``, and a profile record can
only enter gate state through ``_reconcile_authority_diagnostic``.  Until this
file existed nothing in the repository ever produced that artifact, so after
the owner lock no experiment could ever be authorised.  This worker runs a
real profiling tool against a real route and emits exactly the artifact the
gate demands - or refuses and explains itself.  It never invents a number.

WHY IT LOOKS LIKE THIS
----------------------
LESSONS #21: a previous agent asserted an occupancy bottleneck it had never
measured and spent a day building the wrong kernel.  Every metric below is
parsed out of the bytes a tool actually wrote, and every one of those byte
streams is kept as a hashed raw artifact so a reviewer can recompute the
number.  Where a tool is unavailable (``ncu`` behind ``RmProfilingAdminOnly=1``
is the live example) the worker records the refusal and exits non-zero; it
does not substitute a plausible-looking value, and it does not write a
half-populated artifact that the gate would have to catch.

HOW THE CONTROLLER INVOKES IT
-----------------------------
``Project/harness/trusted_controller.py::run_diagnostic`` runs this file
through ``sandbox.run_isolated_command``.  The controller owns the mount plan
and this worker reads every path out of its request rather than assuming one::

    IsolatedMount(HERE / "profile_worker.py", "/work/profile_worker.py")
    IsolatedMount(request_file,               "/work/request.json")
    IsolatedMount(target,                     "/work/target.py")
    IsolatedMount(target,                     "/work/candidate.py")   # alias
    IsolatedMount(OFFICIAL,                   "/work/official.py")
    IsolatedMount(SHAPES,                     "/work/shapes.json")
    IsolatedMount(output_dir,                 "/output", writable=True)

    run_isolated_command(
        mounts=...,
        argv=["/usr/bin/python3", "/work/profile_worker.py",
              "--request", "/work/request.json", "--output", "/output"],
        cwd="/work",
        timeout_seconds=...,
    )

Note what is NOT mounted: mechanism_catalog.json.  The controller resolves the
catalog itself and sends the resulting terms as ``required_metrics``, so this
worker never needs to read - or guess at - the catalog file.

WORKER REQUEST SCHEMA - THE CONTROLLER IS THE AUTHORITY
-------------------------------------------------------
The request is FLAT.  There is no nested ``gate_request`` object; every field
below is a top-level key of the JSON document
``trusted_controller._profile_worker_request`` emits, and this worker requires
all of them (:data:`CONTROLLER_REQUEST_KEYS`).  A missing key is refused as
controller/worker drift rather than defaulted around: the first version of
this file read a ``gate_request`` object the controller never sent, and the
diagnostic lane was dead on arrival because nothing ever tested the two halves
against each other.

  schema_version        int, must be 1
  operation             "diagnostic"
  request_id            the GATE request id; the artifact echoes it verbatim
  gate_request_sha256   64 hex - sha256 of the immutable gate request bytes
                        (``permit["request_sha256"]``); echoed into the artifact
  profile_record_id     the gate's record id; also an artifact field
  campaign_id           echoed into the artifact
  shape_id              int 1..14 - THIS is the artifact's ``shape`` field
  shape                 the shapes.json OBJECT for that id (batch_size, seq_len,
                        d_model, num_heads, ffn_dim, num_layers, causal).  Note
                        the deliberate asymmetry: the request's ``shape`` is the
                        record, the artifact's ``shape`` is the integer id.
  target_sha256         64 hex - the bytes being profiled
  target_path           absolute in-sandbox path of those bytes
  target_path_alias     a second mount of the same bytes (the candidate_worker
                        convention); checked to hold identical bytes
  official_path         absolute in-sandbox path of the official benchmark
  shapes_path           absolute in-sandbox path of shapes.json
  official_sha256       64 hex - checked against official_path
  shapes_sha256         64 hex - checked against shapes_path
  output_dir            the writable output mount ("/output")
  artifact_filename     the exact filename the artifact must be written under
  artifact_fields       the artifact's exact field list, sent so the worker
                        never has to reverse-engineer the gate's schema; the
                        worker refuses if it disagrees with its own
  tool                  which collector to run
  route                 the code route under diagnosis; echoed into the artifact
  question              the concrete question this diagnostic was opened to
                        answer; recorded in the worker's raw evidence
  supported_bottlenecks the bottleneck ids this evidence must support
  required_metrics      {bottleneck: [metric, ...]} - the CATALOG TERMS the
                        controller pinned from the real mechanism catalog and
                        that the gate will re-check.  This is the authority on
                        what the artifact must contain; the embedded table in
                        this file is advisory only.
  machine_state_sha256  64 hex - the CONTROLLER's own machine-state capture.
                        The controller writes that document into the evidence
                        directory as "controller_machine_state.json" and
                        refuses any artifact claiming a different value, so the
                        worker echoes it and never mints its own.
  tool_search_paths     absolute directories to look for nsys/ncu/... in
  timeout_seconds       the controller's whole-run budget; per-tool subprocess
                        timeouts are derived from it
  is_performance_measurement  always false; refused if it is not
  notes                 controller prose, kept in the worker's raw evidence

Everything else this worker can do (dtype, iteration counts, sanitizer passes,
tool path overrides, an ncu sudo prefix) is a worker-side DEFAULT, not a
request field: the controller does not send those, and the worker does not
invent request keys to get them.

WHAT IT WRITES INTO --output
----------------------------
  <artifact_filename>           the gate artifact, exactly the required key set
  raw/...                       every hashed evidence file
  profile_worker_manifest.json  a placement + degradation log for reviewers

The controller does NOT read the manifest.  It treats everything in /output
except the artifact as raw evidence: it copies each file to
``Project/loop/profile_evidence/<profile_record_id>.raw/<path under /output>``,
re-hashes it there, appends its own ``controller_machine_state.json``, and
rewrites the artifact's ``raw_artifacts`` from what is actually on disk.  The
worker predicts exactly those destinations (see ``Evidence.repo_path``) so the
two lists agree, but the controller's is the one that counts.  It then writes
the finalized artifact to the path the gate request named and binds its sha256
into the measurement event and the packet as ``diagnostic_profile_sha256`` -
the two places ``_reconcile_authority_diagnostic`` checks.

FIELD OWNERSHIP (``_finalize_profile_artifact``)
------------------------------------------------
  authority-owned   profile_record_id, request_id, campaign_id, shape,
                    target_sha256, tool, route, supported_bottlenecks,
                    gate_request_sha256, machine_state_sha256.  Omitting one is
                    fine (the controller fills it in); CONTRADICTING one is
                    refused.  This worker echoes them, so both readings agree.
  worker-owned      tool_version, metrics, created_epoch.  The real evidence.
  filesystem-owned  raw_artifacts.  Derived from disk after ingestion.

TOOL -> METRIC MAP (mechanism_catalog.json vocabulary)
------------------------------------------------------
  nsys                  kernel_launches, gpu_idle_fraction
                        (nsys profile --trace=cuda,nvtx around an NVTX-bounded
                         measured region; nsys stats -r cuda_api_sum /
                         cuda_gpu_kern_sum / cuda_gpu_trace, CSV parsed)
  torch-profiler        kernel_launches, gpu_idle_fraction
                        (chrome trace: cat=="cuda_runtime" launch APIs counted;
                         cat in {kernel,gpu_memcpy,gpu_memset} intervals merged)
  ncu                   dram_bytes, compute_utilization, achieved_occupancy,
                        tensor_core_utilization
                        (ncu --csv --metrics ... --graph-profiling node, so a
                         CUDA-graph route profiles as individual kernel nodes)
  memory-profile        peak_reserved_bytes, materialized_bytes
                        (torch.cuda.max_memory_reserved and the largest single
                         allocation seen in the allocator history)
  static-analysis       dram_bytes, materialized_bytes, peak_reserved_bytes
                        (explicit per-term compulsory-traffic accounting for
                         the official dense route; every term is in the raw
                         artifact, and every value is labelled a model)
  microbenchmark        dram_bytes, compute_utilization,
                        tensor_core_utilization, peak_reserved_bytes,
                        materialized_bytes, failed_elements
                        (route timed against measured on-device roofline
                         probes - no vendor peak table is trusted)
  correctness-evaluator failed_elements
                        (official compare_outputs against the official
                         baseline over the requested seeds)

``characterization_metric`` (for unknown-characterization) is always emitted,
set to that tool's headline value.

SELF-TEST
---------
``python3 profile_worker.py --selftest`` runs every parser against recorded
tool-output fixtures, exercises the interval merge, builds a real
static-analysis artifact end to end in a temporary directory, and re-checks
that artifact with a from-scratch reimplementation of the gate's acceptance
rules.  It needs no GPU and no CUDA tooling.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

WORKER_VERSION = "1.0.0"
ARTIFACT_SCHEMA_VERSION = 1
WORKER_REQUEST_SCHEMA_VERSION = 1

DEFAULT_EVIDENCE_ROOT = "Project/loop/profile_evidence"
MANIFEST_NAME = "profile_worker_manifest.json"
RAW_DIRNAME = "raw"
NVTX_RANGE = "profile-measured"

# The controller captures the AUTHORITATIVE machine state itself, hashes it,
# sends the digest as machine_state_sha256, and writes the document into the
# evidence directory under this reserved name.  _collect_worker_outputs refuses
# any worker output using it, and _finalize_profile_artifact refuses an
# artifact that claims a different digest.  The worker's own capture is richer
# (GPU utilisation, rival compute processes, clocks, throttle reasons, load
# average, memory pressure) and still ships -- as one more hashed raw artifact,
# under a name that cannot be mistaken for the authority's, and never as the
# artifact's hash.
CONTROLLER_MACHINE_STATE_NAME = "controller_machine_state.json"
WORKER_MACHINE_STATE_NAME = "worker_machine_state.json"

# Exactly the top-level keys trusted_controller._profile_worker_request emits.
# The request is flat; there is no nested gate_request object.  Every key is
# required, because a key that vanished means the controller moved and this
# file did not.
CONTROLLER_REQUEST_KEYS = frozenset({
    "schema_version", "operation", "request_id", "gate_request_sha256",
    "profile_record_id", "campaign_id", "shape_id", "shape", "target_sha256",
    "target_path", "target_path_alias", "official_path", "shapes_path",
    "official_sha256", "shapes_sha256", "output_dir", "artifact_filename",
    "artifact_fields", "tool", "route", "question", "supported_bottlenecks",
    "required_metrics", "machine_state_sha256", "tool_search_paths",
    "timeout_seconds", "is_performance_measurement", "notes",
})
# Mirrors trusted_controller.SAFE_NAME: one plain filename component.
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

# Exactly the key set _reconcile_authority_diagnostic compares with
# ``set(artifact) != required``.  Extra or missing keys are fatal there, so
# this constant is the contract and the worker validates against it.
ARTIFACT_REQUIRED_KEYS = frozenset({
    "schema_version", "profile_record_id", "request_id", "campaign_id",
    "shape", "target_sha256", "tool", "tool_version", "created_epoch",
    "machine_state_sha256", "route", "metrics", "supported_bottlenecks",
    "raw_artifacts", "gate_request_sha256",
})
# Every authority-owned artifact field, exactly the ``bound`` mapping in
# trusted_controller.run_diagnostic.  Omitting one of these is fine (the
# controller fills it in from the immutable gate request); contradicting one
# is refused.  Note "shape": the artifact carries the integer shape id, which
# the flat worker request calls ``shape_id`` -- its ``shape`` key is the
# shapes.json record.
GATE_BOUND_KEYS = (
    "profile_record_id", "request_id", "campaign_id", "shape",
    "target_sha256", "tool", "route", "supported_bottlenecks",
    "gate_request_sha256", "machine_state_sha256",
)

HEX64 = re.compile(r"[0-9a-f]{64}")
CUDA_API_NAME = re.compile(r"^(?P<base>cu[A-Za-z0-9_]*?)(?:_v\d+)?$")

LAUNCH_API_NAMES = frozenset({
    "cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchKernelEx",
    "cudaLaunchCooperativeKernel", "cuLaunchKernel", "cuLaunchKernelEx",
    "cuLaunchCooperativeKernel", "cuLaunchGrid", "cudaLaunchHostFunc",
})
GRAPH_LAUNCH_NAMES = frozenset({"cudaGraphLaunch", "cuGraphLaunch"})
DEVICE_TRACE_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})

# Numerical profile of the official benchmark, mirrored from
# trusted_controller.NUMERICAL so a profiled route is the scored route.
NUMERICAL = {
    "padding_ratio": 0.0,
    "input_scale": 1.0,
    "rtol": 0.02,
    "atol": 0.002,
    "matmul_precision": "high",
    "allow_tf32": True,
}
DTYPE_ITEMSIZE = {"float32": 4, "float16": 2, "bfloat16": 2}

# Advisory copy of the bottleneck table in Project/loop/mechanism_catalog.json.
# The controller does NOT mount the catalog: it resolves the terms itself and
# sends them as required_metrics, pinned into the immutable gate request, and
# the gate re-checks those same pinned terms.  So this table never decides
# anything -- a disagreement with it is reported as a degradation and the
# controller's terms win.  Refusing on a stale local copy could only reject
# evidence the trusted side already authorized.
CATALOG_FALLBACK: dict[str, dict[str, list[str]]] = {
    "launch-overhead": {
        "evidence_tools": ["nsys", "torch-profiler"],
        "required_metrics": ["kernel_launches"]},
    "host-synchronization": {
        "evidence_tools": ["nsys", "torch-profiler"],
        "required_metrics": ["gpu_idle_fraction"]},
    "global-memory-traffic": {
        "evidence_tools": ["ncu", "microbenchmark", "static-analysis"],
        "required_metrics": ["dram_bytes"]},
    "compute-throughput": {
        "evidence_tools": ["ncu", "microbenchmark"],
        "required_metrics": ["compute_utilization"]},
    "occupancy-resource-pressure": {
        "evidence_tools": ["ncu"],
        "required_metrics": ["achieved_occupancy"]},
    "tensor-core-utilization": {
        "evidence_tools": ["ncu", "microbenchmark"],
        "required_metrics": ["tensor_core_utilization"]},
    "memory-capacity": {
        "evidence_tools": ["memory-profile", "static-analysis", "microbenchmark"],
        "required_metrics": ["peak_reserved_bytes"]},
    "quadratic-materialization": {
        "evidence_tools": ["memory-profile", "static-analysis", "microbenchmark"],
        "required_metrics": ["materialized_bytes"]},
    "numerical-precision": {
        "evidence_tools": ["microbenchmark", "correctness-evaluator"],
        "required_metrics": ["failed_elements"]},
    "unknown-characterization": {
        "evidence_tools": ["nsys", "ncu", "torch-profiler", "microbenchmark",
                           "memory-profile", "static-analysis"],
        "required_metrics": ["characterization_metric"]},
}

# Which catalog metric names each collector in this file is ABLE to produce.
# The check is capability, not guarantee: a collector may legitimately omit a
# conditional metric, and the post-collection gate-equivalent check catches
# that.  What this table buys is an early, honest refusal when the controller
# authorized a tool that could never answer the question -- before any GPU
# time is spent.  It is a fact about this worker's own code, so unlike the
# catalog fallback above it cannot go stale relative to the gate.
COLLECTOR_METRICS: dict[str, frozenset[str]] = {
    "nsys": frozenset({
        "kernel_launches", "gpu_idle_fraction", "characterization_metric"}),
    "torch-profiler": frozenset({
        "kernel_launches", "gpu_idle_fraction", "characterization_metric"}),
    "ncu": frozenset({
        "dram_bytes", "compute_utilization", "achieved_occupancy",
        "tensor_core_utilization", "characterization_metric"}),
    "memory-profile": frozenset({
        "peak_reserved_bytes", "materialized_bytes",
        "characterization_metric"}),
    "static-analysis": frozenset({
        "dram_bytes", "materialized_bytes", "peak_reserved_bytes",
        "characterization_metric"}),
    "microbenchmark": frozenset({
        "dram_bytes", "materialized_bytes", "peak_reserved_bytes",
        "compute_utilization", "tensor_core_utilization", "failed_elements",
        "characterization_metric"}),
    "correctness-evaluator": frozenset({
        "failed_elements", "characterization_metric"}),
}

# ncu counters, chosen so one pass covers every ncu-eligible bottleneck.
NCU_METRICS = (
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.sum",
    "gpu__time_duration.sum",
)

MAX_RAW_FILE_BYTES = 48 * 1024 * 1024
MAX_TOTAL_RAW_BYTES = 96 * 1024 * 1024
DEFAULT_TOOL_TIMEOUT_S = 1800


class ProfileWorkerError(RuntimeError):
    """Refusal.  Nothing is fabricated to get past one of these."""


class ToolUnavailable(ProfileWorkerError):
    """The requested tool exists in the catalog but not on this machine."""


class ToolOutputError(ProfileWorkerError):
    """The tool ran but its output could not be parsed into real metrics."""


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - the message is the point
        raise ProfileWorkerError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProfileWorkerError(f"{label} must be a JSON object")
    return value


def write_exclusive(path: Path, data: bytes) -> None:
    """Create-and-write, never clobber.  Mirrors candidate_worker.py."""
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileWorkerError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ProfileWorkerError(f"{label} is not finite")
    return number


def bounded_int(value: Any, label: str, low: int, high: int, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileWorkerError(f"{label} must be an integer")
    if not low <= value <= high:
        raise ProfileWorkerError(f"{label} must be within [{low}, {high}]")
    return value


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> tuple[float, float, int]:
    """Return (busy, span, count) over half-open [start, end) intervals.

    ``busy`` is the measure of the union, so concurrent kernels on different
    streams are counted once - anything else would let a multi-stream route
    report a nonsense idle fraction below zero.
    """
    clean = [(float(a), float(b)) for a, b in intervals if float(b) > float(a)]
    if not clean:
        return 0.0, 0.0, 0
    clean.sort()
    busy = 0.0
    span_start = clean[0][0]
    span_end = clean[0][1]
    current_start, current_end = clean[0]
    for start, end in clean[1:]:
        span_end = max(span_end, end)
        if start > current_end:
            busy += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    busy += current_end - current_start
    return busy, span_end - span_start, len(clean)


def idle_fraction(busy: float, span: float) -> float:
    if span <= 0:
        raise ToolOutputError("profiled device timeline has zero span")
    return max(0.0, min(1.0, 1.0 - busy / span))


# --------------------------------------------------------------------------
# tool discovery
# --------------------------------------------------------------------------

TOOL_SEARCH_GLOBS = (
    "/usr/local/cuda/bin/{name}",
    "/usr/local/cuda-*/bin/{name}",
    "/opt/nvidia/nsight-systems/*/target-linux-x64/{name}",
    "/opt/nvidia/nsight-systems-cli/*/target-linux-x64/{name}",
    "/opt/nvidia/nsight-compute/*/{name}",
    "/usr/bin/{name}",
    "/usr/local/bin/{name}",
)


def discover_tool(name: str, overrides: dict[str, str] | None = None,
                  search_paths: Sequence[str] | None = None) -> str:
    """Absolute path of an external tool, or ToolUnavailable.

    The sandbox PATH is /usr/bin:/bin(:/usr/sbin:/sbin) while nsys, ncu and
    compute-sanitizer live under /usr/local/cuda/bin, so a bare ``which`` is
    not enough inside the jail.  ``search_paths`` is the controller's own
    ``tool_search_paths`` (PROFILER_TOOL_DIRS): it is told to the worker
    rather than guessed, and it is tried before this file's glob table.
    """
    if overrides and name in overrides:
        candidate = overrides[name]
        if not isinstance(candidate, str) or not candidate.startswith("/"):
            raise ProfileWorkerError(f"tool_paths[{name!r}] must be an absolute path")
        if os.access(candidate, os.X_OK):
            return candidate
        raise ToolUnavailable(f"{name} override {candidate!r} is not executable")
    found = shutil.which(name)
    if found:
        return found
    for directory in list(search_paths or ()):
        candidate = os.path.join(directory, name)
        if os.access(candidate, os.X_OK):
            return candidate
    for pattern in TOOL_SEARCH_GLOBS:
        for match in sorted(glob.glob(pattern.format(name=name)), reverse=True):
            if os.access(match, os.X_OK):
                return match
    raise ToolUnavailable(f"{name} was not found on this machine")


def run_tool(
    argv: Sequence[str],
    *,
    timeout_seconds: int,
    label: str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> tuple[int, str, str]:
    environment = dict(os.environ)
    environment.setdefault("TMPDIR", "/tmp")
    if env:
        environment.update(env)
    try:
        completed = subprocess.run(  # noqa: S603 - argv is fully constructed here
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            env=environment,
            cwd=cwd,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolOutputError(f"{label} timed out after {timeout_seconds}s") from exc
    except OSError as exc:
        raise ToolUnavailable(f"{label} could not be executed: {exc}") from exc
    return (
        completed.returncode,
        completed.stdout.decode("utf-8", "replace"),
        completed.stderr.decode("utf-8", "replace"),
    )


def tool_version_string(binary: str, timeout_seconds: int = 60) -> str:
    code, out, err = run_tool([binary, "--version"], timeout_seconds=timeout_seconds,
                              label=f"{Path(binary).name} --version")
    text = (out or err).strip()
    if code != 0 and not text:
        raise ToolOutputError(f"{binary} --version failed with rc={code}")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r"\d+\.\d+", line):
            return line[:120]
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[0][:120] if lines else "unknown"


# --------------------------------------------------------------------------
# CSV / trace parsers.  These are pure functions - the self-test drives them
# from recorded tool output with no GPU present.
# --------------------------------------------------------------------------


def normalize_column(name: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", str(name))
    text = text.replace("/", " ").replace("%", " pct ")
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_csv_table(text: str, required_columns: Iterable[str]) -> list[dict[str, str]]:
    """Extract the first CSV table containing every required column.

    nsys and ncu both wrap their CSV in progress chatter and section banners,
    and both have renamed columns between releases.  Matching on normalised
    column names and refusing outright when one is missing keeps a version
    change loud instead of silently producing a zero.
    """
    required = {normalize_column(column) for column in required_columns}
    rows = list(csv.reader(io.StringIO(text)))
    header_index = None
    header: list[str] = []
    for index, row in enumerate(rows):
        candidate = [normalize_column(cell) for cell in row]
        if required.issubset(set(candidate)):
            header_index = index
            header = candidate
            break
    if header_index is None:
        raise ToolOutputError(
            "tool CSV lacks required column(s): " + ", ".join(sorted(required)))
    table: list[dict[str, str]] = []
    for row in rows[header_index + 1:]:
        if not row or all(not cell.strip() for cell in row):
            continue
        if row[0].strip().startswith("**"):
            break
        if len(row) != len(header):
            continue
        table.append(dict(zip(header, row)))
    return table


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip('"').replace(",", "")
    if not text or text.lower() in {"n/a", "na", "nan", "-", "inf", "-inf"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def api_base_name(name: str) -> str:
    text = str(name).strip().strip('"')
    match = CUDA_API_NAME.match(text)
    return match.group("base") if match else text


def parse_nsys_api_sum(text: str) -> dict[str, Any]:
    """Launch counts from ``nsys stats -r cuda_api_sum -f csv``."""
    rows = parse_csv_table(text, ("Num Calls", "Name"))
    if not rows:
        raise ToolOutputError("cuda_api_sum produced no rows")
    launches = 0
    graph_launches = 0
    by_name: dict[str, int] = {}
    for row in rows:
        name = api_base_name(row.get("name", ""))
        calls = parse_number(row.get("num calls"))
        if calls is None:
            continue
        count = int(round(calls))
        by_name[name] = by_name.get(name, 0) + count
        if name in LAUNCH_API_NAMES:
            launches += count
        elif name in GRAPH_LAUNCH_NAMES:
            graph_launches += count
    if launches == 0 and graph_launches == 0:
        raise ToolOutputError(
            "cuda_api_sum contained no kernel-launch or graph-launch API rows")
    return {
        "kernel_launch_api_calls": launches,
        "graph_launch_api_calls": graph_launches,
        "cuda_api_calls_by_name": by_name,
    }


def parse_nsys_kern_sum(text: str) -> dict[str, Any]:
    """Kernel instance counts from ``nsys stats -r cuda_gpu_kern_sum``."""
    rows = parse_csv_table(text, ("Instances", "Name"))
    instances = 0
    distinct = 0
    for row in rows:
        count = parse_number(row.get("instances"))
        if count is None:
            continue
        instances += int(round(count))
        distinct += 1
    return {"kernel_instances": instances, "distinct_kernels": distinct}


def parse_nsys_gpu_trace(text: str) -> dict[str, Any]:
    """GPU busy/idle from ``nsys stats -r cuda_gpu_trace`` (nanoseconds)."""
    rows = parse_csv_table(text, ("Start", "Duration", "Name"))
    intervals: list[tuple[float, float]] = []
    for row in rows:
        start = parse_number(row.get("start"))
        duration = parse_number(row.get("duration"))
        if start is None or duration is None:
            continue
        intervals.append((start, start + duration))
    if not intervals:
        raise ToolOutputError("cuda_gpu_trace contained no timed device activity")
    busy, span, count = merge_intervals(intervals)
    return {
        "gpu_busy_ns": busy,
        "gpu_span_ns": span,
        "gpu_idle_fraction": idle_fraction(busy, span),
        "device_activity_records": count,
    }


def parse_torch_chrome_trace(document: Any) -> dict[str, Any]:
    """Launch counts and GPU idle fraction from a torch.profiler chrome trace.

    Trace timestamps are microseconds.  Launch counts come from the CUPTI
    runtime rows (``cat == "cuda_runtime"``), device occupancy from the merged
    union of kernel / memcpy / memset rows.
    """
    events = document.get("traceEvents") if isinstance(document, dict) else None
    if not isinstance(events, list):
        raise ToolOutputError("chrome trace has no traceEvents array")
    launches = 0
    graph_launches = 0
    by_name: dict[str, int] = {}
    intervals: list[tuple[float, float]] = []
    kernel_records = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        category = str(event.get("cat", ""))
        name = api_base_name(event.get("name", ""))
        if category in {"cuda_runtime", "cuda_driver", "Runtime"}:
            if name in LAUNCH_API_NAMES:
                launches += 1
                by_name[name] = by_name.get(name, 0) + 1
            elif name in GRAPH_LAUNCH_NAMES:
                graph_launches += 1
                by_name[name] = by_name.get(name, 0) + 1
        if category in DEVICE_TRACE_CATEGORIES and event.get("ph") == "X":
            start = parse_number(event.get("ts"))
            duration = parse_number(event.get("dur"))
            if start is None or duration is None:
                continue
            intervals.append((start, start + duration))
            if category == "kernel":
                kernel_records += 1
    if launches == 0 and graph_launches == 0:
        raise ToolOutputError("chrome trace contained no CUDA launch API records")
    if not intervals:
        raise ToolOutputError("chrome trace contained no device activity records")
    busy, span, count = merge_intervals(intervals)
    return {
        "kernel_launch_api_calls": launches,
        "graph_launch_api_calls": graph_launches,
        "cuda_api_calls_by_name": by_name,
        "kernel_instances": kernel_records,
        "gpu_busy_us": busy,
        "gpu_span_us": span,
        "gpu_idle_fraction": idle_fraction(busy, span),
        "device_activity_records": count,
    }


NCU_PERMISSION_MARKERS = (
    "ERR_NVGPUCTRPERM",
    "does not have permission to access NVIDIA GPU Performance Counters",
    "The user does not have permission",
)


def ncu_permission_denied(text: str) -> bool:
    return any(marker in text for marker in NCU_PERMISSION_MARKERS)


def parse_ncu_csv(text: str) -> dict[str, list[float]]:
    """Per-kernel counter values from ``ncu --csv``.

    Handles both emitted shapes: the long form (one row per metric, with
    "Metric Name"/"Metric Value" columns) and the wide raw-page form (one row
    per kernel, one column per metric).
    """
    try:
        rows = parse_csv_table(text, ("Metric Name", "Metric Value"))
    except ToolOutputError:
        rows = []
    if rows:
        collected: dict[str, list[float]] = {}
        for row in rows:
            metric = str(row.get("metric name", "")).strip().strip('"')
            value = parse_number(row.get("metric value"))
            if not metric or value is None:
                continue
            collected.setdefault(metric, []).append(value)
        if collected:
            return collected
    # Wide form: locate metric columns by exact counter name.
    all_rows = list(csv.reader(io.StringIO(text)))
    header_index = None
    for index, row in enumerate(all_rows):
        if any(cell.strip().strip('"') in NCU_METRICS for cell in row):
            header_index = index
            break
    if header_index is None:
        raise ToolOutputError("ncu CSV contained no recognisable metric columns")
    header = [cell.strip().strip('"') for cell in all_rows[header_index]]
    collected = {}
    for row in all_rows[header_index + 1:]:
        if len(row) != len(header):
            continue
        for column, cell in zip(header, row):
            if column not in NCU_METRICS:
                continue
            value = parse_number(cell)
            if value is not None:
                collected.setdefault(column, []).append(value)
    if not collected:
        raise ToolOutputError("ncu CSV metric columns held no numeric values")
    return collected


def weighted_mean(values: Sequence[float], weights: Sequence[float] | None) -> float:
    if not values:
        raise ToolOutputError("cannot average an empty counter series")
    if weights and len(weights) == len(values) and sum(weights) > 0:
        total = sum(weights)
        return sum(v * w for v, w in zip(values, weights)) / total
    return sum(values) / len(values)


SANITIZER_ERROR_SUMMARY = re.compile(
    r"ERROR SUMMARY:\s*(\d+)\s+error", re.IGNORECASE)
SANITIZER_RACE_SUMMARY = re.compile(
    r"RACECHECK SUMMARY:\s*(\d+)\s+hazard[s]?\s+displayed\s*"
    r"\((\d+)\s+error[s]?,\s*(\d+)\s+warning[s]?\)", re.IGNORECASE)


def parse_compute_sanitizer(text: str) -> dict[str, Any]:
    """memcheck / racecheck findings from compute-sanitizer console output."""
    errors = SANITIZER_ERROR_SUMMARY.search(text)
    race = SANITIZER_RACE_SUMMARY.search(text)
    if errors is None and race is None:
        raise ToolOutputError("compute-sanitizer emitted no summary line")
    result: dict[str, Any] = {
        "sanitizer_errors": int(errors.group(1)) if errors else None,
    }
    if race is not None:
        result["sanitizer_hazards_displayed"] = int(race.group(1))
        result["sanitizer_race_errors"] = int(race.group(2))
        result["sanitizer_race_warnings"] = int(race.group(3))
        if result["sanitizer_errors"] is None:
            result["sanitizer_errors"] = int(race.group(2))
    if result["sanitizer_errors"] is None:
        raise ToolOutputError("compute-sanitizer summary had no error count")
    result["sanitizer_clean"] = result["sanitizer_errors"] == 0
    return result


# --------------------------------------------------------------------------
# machine state
# --------------------------------------------------------------------------

NVIDIA_SMI_QUERIES = (
    ("index,uuid,name,driver_version,pstate,persistence_mode,compute_mode,"
     "utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,"
     "clocks.current.sm,clocks.current.memory,clocks.max.sm,clocks.max.memory,"
     "temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active"),
    ("index,uuid,name,driver_version,utilization.gpu,utilization.memory,"
     "memory.total,memory.used,memory.free,clocks.current.sm,"
     "clocks.current.memory,temperature.gpu"),
    "index,name,utilization.gpu,memory.total,memory.used,memory.free",
)


def _nvidia_smi_snapshot(binary: str | None, timeout_seconds: int) -> dict[str, Any]:
    if binary is None:
        return {"available": False, "error": "nvidia-smi not found"}
    last = ""
    for query in NVIDIA_SMI_QUERIES:
        code, out, err = run_tool(
            [binary, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            timeout_seconds=timeout_seconds, label="nvidia-smi --query-gpu")
        if code == 0 and out.strip():
            fields = [field.strip() for field in query.split(",")]
            gpus = []
            for line in out.strip().splitlines():
                cells = [cell.strip() for cell in line.split(",")]
                if len(cells) != len(fields):
                    continue
                gpus.append(dict(zip(fields, cells)))
            return {"available": True, "query": query, "gpus": gpus,
                    "raw": out.strip()}
        last = (err or out).strip()[:500]
    return {"available": False, "error": last}


def _compute_apps_snapshot(binary: str | None, timeout_seconds: int) -> dict[str, Any]:
    if binary is None:
        return {"available": False, "error": "nvidia-smi not found"}
    code, out, err = run_tool(
        [binary, "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        timeout_seconds=timeout_seconds, label="nvidia-smi --query-compute-apps")
    if code != 0:
        return {"available": False, "error": (err or out).strip()[:500]}
    processes = []
    for line in out.strip().splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) == 3:
            processes.append({"pid": cells[0], "process_name": cells[1],
                              "used_memory_mib": cells[2]})
    return {"available": True, "count": len(processes), "processes": processes,
            "raw": out.strip()}


def _host_snapshot() -> dict[str, Any]:
    host: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "uname": " ".join(platform.uname()),
    }
    try:
        host["loadavg"] = list(os.getloadavg())
    except OSError:
        host["loadavg"] = None
    try:
        meminfo = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable", "MemFree"}:
                meminfo[key] = rest.strip()
        host["meminfo"] = meminfo
    except OSError:
        host["meminfo"] = None
    try:
        host["cpu_pressure"] = Path("/proc/pressure/cpu").read_text().strip()
    except OSError:
        host["cpu_pressure"] = None
    return host


def capture_machine_state(phase: str, *, overrides: dict[str, str] | None,
                          search_paths: Sequence[str] | None = None,
                          timeout_seconds: int = 60) -> dict[str, Any]:
    """One real observation of the box: GPU load, memory, clocks, rivals.

    This is SUPPORTING evidence, not the artifact's machine state.  The
    controller captures its own document, hashes it, sends the digest in the
    request and writes the document into the evidence directory as
    ``controller_machine_state.json``; contradicting that digest gets the
    artifact refused.  What is captured here -- GPU utilisation, rival
    compute processes, current and maximum clocks, throttle reasons, load
    average, memory pressure -- the controller cannot see from outside the
    jail, and a diagnostic taken on a busy box is weaker evidence, so it
    still ships: as one more hashed raw artifact under
    ``worker_machine_state.json``, referenced from ``metrics`` and never
    from ``machine_state_sha256``.
    """
    try:
        binary: str | None = discover_tool("nvidia-smi", overrides, search_paths)
    except ToolUnavailable:
        binary = None
    return {
        "phase": phase,
        "captured_epoch": time.time(),
        "captured_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nvidia_smi": _nvidia_smi_snapshot(binary, timeout_seconds),
        "compute_apps": _compute_apps_snapshot(binary, timeout_seconds),
        "host": _host_snapshot(),
    }


# --------------------------------------------------------------------------
# request handling
# --------------------------------------------------------------------------


def _hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ProfileWorkerError(f"{label} must be 64 lowercase hex characters")
    return value


def parse_worker_request(document: dict[str, Any]) -> dict[str, Any]:
    """Validate the controller's FLAT worker request and fill worker defaults.

    The controller is the trusted side and it owns this schema
    (``trusted_controller._profile_worker_request``).  Every key it emits is
    required here and nothing is nested.  Values the controller does not send
    are worker-side knobs with conservative defaults; they are not invented
    request fields, and this worker never asks the controller to grow one.

    Two asymmetries are worth naming because both are easy to get backwards:

    * the request's ``shape`` is the shapes.json RECORD while the artifact's
      ``shape`` is the integer id, which the request calls ``shape_id``;
    * ``machine_state_sha256`` is supplied BY the controller, not computed
      here -- see :data:`CONTROLLER_MACHINE_STATE_NAME`.
    """
    if not isinstance(document, dict):
        raise ProfileWorkerError("worker request must be a JSON object")
    if document.get("schema_version") != WORKER_REQUEST_SCHEMA_VERSION:
        raise ProfileWorkerError("unsupported worker request schema_version")
    if document.get("operation") != "diagnostic":
        raise ProfileWorkerError(
            "worker request operation must be 'diagnostic'; this worker "
            "profiles bytes that already exist and evaluates no candidate")
    missing = sorted(CONTROLLER_REQUEST_KEYS - set(document))
    if missing:
        raise ProfileWorkerError(
            "controller worker request is missing " + ", ".join(missing)
            + " -- the controller and this worker have drifted apart; the "
            "controller defines the contract, so this worker is the side "
            "that must be corrected")
    unexpected = sorted(set(document) - CONTROLLER_REQUEST_KEYS)

    shape_id = document["shape_id"]
    if (isinstance(shape_id, bool) or not isinstance(shape_id, int)
            or not 1 <= shape_id <= 14):
        raise ProfileWorkerError("shape_id must be an int 1..14")
    shape = document["shape"]
    if not isinstance(shape, dict) or shape.get("id") != shape_id:
        raise ProfileWorkerError(
            "shape must be the shapes.json record whose id is shape_id "
            "(the request carries the record; the artifact carries the id)")
    route = document["route"]
    if not isinstance(route, str) or not route.strip():
        raise ProfileWorkerError("route must be a non-empty string")
    question = document["question"]
    if not isinstance(question, str) or not question.strip():
        raise ProfileWorkerError("question must be a non-empty string")
    tool = document["tool"]
    if not isinstance(tool, str) or not tool:
        raise ProfileWorkerError("tool must be a non-empty string")
    bottlenecks = document["supported_bottlenecks"]
    if (not isinstance(bottlenecks, list) or not bottlenecks
            or any(not isinstance(item, str) or not item for item in bottlenecks)
            or len(set(bottlenecks)) != len(bottlenecks)):
        raise ProfileWorkerError(
            "supported_bottlenecks must be a non-empty list of distinct ids")
    # The catalog terms, pinned by the controller from the real
    # mechanism_catalog.json.  These -- not the fallback table in this file --
    # are what the artifact has to satisfy, because they are what the gate
    # re-checks at reconcile.
    required_metrics = document["required_metrics"]
    if (not isinstance(required_metrics, dict)
            or set(required_metrics) != set(bottlenecks)):
        raise ProfileWorkerError(
            "required_metrics must name exactly the declared bottlenecks")
    normalized_metrics: dict[str, list[str]] = {}
    for name, metrics in required_metrics.items():
        if (not isinstance(metrics, list) or not metrics
                or any(not isinstance(metric, str) or not metric
                       for metric in metrics)):
            raise ProfileWorkerError(f"required_metrics[{name!r}] is malformed")
        normalized_metrics[name] = list(metrics)
    # The controller sends the artifact's field list so the worker never has
    # to reverse-engineer the gate's schema.  Disagreement is exactly the
    # class of defect this contract exists to make loud.
    artifact_fields = document["artifact_fields"]
    if (not isinstance(artifact_fields, list)
            or any(not isinstance(field, str) or not field
                   for field in artifact_fields)):
        raise ProfileWorkerError("artifact_fields must be a list of field names")
    if set(artifact_fields) != set(ARTIFACT_REQUIRED_KEYS):
        raise ProfileWorkerError(
            "controller artifact_fields disagree with this worker's artifact "
            f"schema (controller_only="
            f"{sorted(set(artifact_fields) - set(ARTIFACT_REQUIRED_KEYS))}, "
            f"worker_only="
            f"{sorted(set(ARTIFACT_REQUIRED_KEYS) - set(artifact_fields))})")
    artifact_filename = document["artifact_filename"]
    if (not isinstance(artifact_filename, str)
            or not SAFE_NAME.fullmatch(artifact_filename)
            or not artifact_filename.endswith(".json")):
        raise ProfileWorkerError(
            "artifact_filename must be one plain *.json filename component")
    tool_search_paths = document["tool_search_paths"]
    if (not isinstance(tool_search_paths, list)
            or any(not isinstance(item, str) or not item.startswith("/")
                   for item in tool_search_paths)):
        raise ProfileWorkerError(
            "tool_search_paths must be a list of absolute directories")
    timeout_seconds = document["timeout_seconds"]
    if (isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 24 * 3600):
        raise ProfileWorkerError("timeout_seconds must be an int 1..86400")
    if document["is_performance_measurement"] is not False:
        raise ProfileWorkerError(
            "a diagnostic is never a performance measurement; refusing a "
            "request that claims to be one")
    controller_notes = document["notes"]
    if not isinstance(controller_notes, str):
        raise ProfileWorkerError("notes must be a string")

    parsed: dict[str, Any] = {
        # --- controller-owned; echoed into the artifact verbatim -----------
        "operation": "diagnostic",
        "profile_record_id": document["profile_record_id"],
        "request_id": document["request_id"],
        "campaign_id": document["campaign_id"],
        "gate_request_sha256": _hex64(document["gate_request_sha256"],
                                      "gate_request_sha256"),
        "shape_id": shape_id,
        "shape": shape,
        "target_sha256": _hex64(document["target_sha256"], "target_sha256"),
        "tool": tool,
        "route": route.strip(),
        "question": question.strip(),
        "supported_bottlenecks": list(bottlenecks),
        "required_metrics": normalized_metrics,
        "machine_state_sha256": _hex64(document["machine_state_sha256"],
                                       "machine_state_sha256"),
        "artifact_fields": sorted(artifact_fields),
        "artifact_filename": artifact_filename,
        # --- controller-owned paths and budgets ----------------------------
        "target_path": document["target_path"],
        "target_path_alias": document["target_path_alias"],
        "official_path": document["official_path"],
        "shapes_path": document["shapes_path"],
        "official_sha256": _hex64(document["official_sha256"], "official_sha256"),
        "shapes_sha256": _hex64(document["shapes_sha256"], "shapes_sha256"),
        "output_dir": document["output_dir"],
        "tool_search_paths": list(tool_search_paths),
        "timeout_seconds": timeout_seconds,
        "controller_notes": controller_notes.strip(),
        # --- worker-side defaults: the flat request carries none of these --
        # The gate resolves the artifact by the path in its own request, so
        # the evidence root is a constant here rather than a request field.
        "evidence_root": DEFAULT_EVIDENCE_ROOT,
        "catalog_path": None,
        "route_kind": "candidate",
        "dtype": "float32",
        "iterations": 20,
        "warmup": 10,
        "seed": 991_827_331,
        "incumbent_speedup": None,
        "measure_incumbent_speedup": False,
        "sanitizer": None,
        "tool_paths": {},
        "ncu_command_prefix": [],
        # Leave the controller room to kill the process and still get a
        # refusal manifest: a per-tool budget at the whole-run budget would
        # let one tool consume it entirely.
        "tool_timeout_seconds": bounded_int(
            max(30, min(7200, int(timeout_seconds * 0.8))),
            "tool_timeout_seconds", 30, 7200, DEFAULT_TOOL_TIMEOUT_S),
        "request_notes": [],
    }
    for key in ("profile_record_id", "request_id", "campaign_id"):
        if not isinstance(parsed[key], str) or not parsed[key]:
            raise ProfileWorkerError(f"{key} must be a non-empty string")
    if not SAFE_NAME.fullmatch(parsed["profile_record_id"]):
        raise ProfileWorkerError(
            "profile_record_id must be one plain filename component")
    for key in ("target_path", "target_path_alias", "official_path",
                "shapes_path", "output_dir"):
        value = parsed[key]
        if not isinstance(value, str) or not value.startswith("/"):
            raise ProfileWorkerError(f"{key} must be an absolute in-sandbox path")
    if artifact_filename != f"{parsed['profile_record_id']}.json":
        parsed["request_notes"].append(
            "artifact_filename is not <profile_record_id>.json; writing the "
            "name the controller asked for, which is what it looks for")
    if unexpected:
        # Not fatal: the controller is the authority and may legitimately add
        # a field.  Loud, because silent tolerance is how the two halves of
        # this interface drifted apart in the first place.
        parsed["request_notes"].append(
            "controller sent key(s) this worker does not consume: "
            + ", ".join(unexpected))
    return parsed


def load_catalog(parsed: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """The advisory bottleneck table.  Never the authority -- see above.

    ``catalog_path`` is a worker-side default of None because the controller's
    mount plan does not include mechanism_catalog.json; the parameter stays so
    an out-of-jail rehearsal can point at the real file.
    """
    path = parsed.get("catalog_path")
    if path:
        catalog_path = Path(path)
        if catalog_path.is_file():
            document = read_object(catalog_path, "mechanism catalog")
            bottlenecks = document.get("bottlenecks")
            if isinstance(bottlenecks, dict) and bottlenecks:
                return bottlenecks, "mounted"
            raise ProfileWorkerError("mechanism catalog has no bottlenecks table")
        raise ProfileWorkerError(f"catalog_path {path!r} does not exist")
    return CATALOG_FALLBACK, "embedded-fallback"


def load_shape(shapes_path: Path, shape_id: int) -> dict[str, Any]:
    document = read_object(shapes_path, "shapes.json")
    items = document.get("shapes")
    if not isinstance(items, list):
        raise ProfileWorkerError("shapes.json schema is malformed")
    matches = [item for item in items
               if isinstance(item, dict) and item.get("id") == shape_id]
    if len(matches) != 1:
        raise ProfileWorkerError(f"shape {shape_id} is absent or duplicated in shapes.json")
    shape = matches[0]
    for key in ("batch_size", "seq_len", "d_model", "num_heads", "ffn_dim",
                "num_layers", "causal"):
        if key not in shape:
            raise ProfileWorkerError(f"shape {shape_id} is missing {key}")
    return shape


def verify_target(parsed: dict[str, Any]) -> Path:
    """The artifact claims to describe specific bytes; prove it holds them.

    The flat request carries the sha of all three inputs, so all three are
    checked.  A profile taken against the wrong official script or the wrong
    shapes.json is uninterpretable evidence even when the target is right,
    and the alias mount is checked too because the controller binds the same
    file twice and a divergence there would mean the jail is not what it says.
    """
    target = Path(parsed["target_path"])
    if target.is_symlink() or not target.is_file():
        raise ProfileWorkerError("target_path must be a regular non-symlink file")
    digest = sha256_file(target)
    expected = parsed["target_sha256"]
    if digest != expected:
        raise ProfileWorkerError(
            f"target bytes do not match target_sha256 ({digest} != {expected})")
    alias = Path(parsed["target_path_alias"])
    if alias.is_file() and sha256_file(alias) != expected:
        raise ProfileWorkerError(
            "target_path_alias holds different bytes from target_path")
    for label, path_key, sha_key in (
            ("official benchmark", "official_path", "official_sha256"),
            ("shapes.json", "shapes_path", "shapes_sha256")):
        path = Path(parsed[path_key])
        if path.is_symlink() or not path.is_file():
            raise ProfileWorkerError(
                f"{label} is missing at {parsed[path_key]}")
        found = sha256_file(path)
        if found != parsed[sha_key]:
            raise ProfileWorkerError(
                f"{label} bytes disagree with the controller's {sha_key} "
                f"({found} != {parsed[sha_key]})")
    return target


def route_shape(parsed: dict[str, Any]) -> dict[str, Any]:
    """The shape record, read from the mounted shapes.json and cross-checked.

    The controller sends the record inline as ``shape`` and mounts shapes.json
    as well.  If the two disagree the profile would describe a shape the gate
    never authorized, so this refuses instead of picking a winner.
    """
    shape = load_shape(Path(parsed["shapes_path"]), parsed["shape_id"])
    if shape != parsed["shape"]:
        raise ProfileWorkerError(
            "the mounted shapes.json disagrees with the shape record in the "
            "controller's request")
    return shape


# --------------------------------------------------------------------------
# route construction and execution (needs a GPU)
# --------------------------------------------------------------------------


class RouteContext:
    def __init__(self, torch: Any, official: Any, config: Any, device: Any,
                 dtype: Any, model: Any, baseline: Any, inputs: Any, mask: Any,
                 shape: dict[str, Any], description: str) -> None:
        self.torch = torch
        self.official = official
        self.config = config
        self.device = device
        self.dtype = dtype
        self.model = model
        self.baseline = baseline
        self.inputs = inputs
        self.mask = mask
        self.shape = shape
        self.description = description


def _load_module_exact(path: Path, name: str) -> Any:
    source = Path(path).read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def build_route(parsed: dict[str, Any]) -> RouteContext:
    """Instantiate the exact route the diagnostic claims to describe."""
    try:
        import torch  # noqa: PLC0415 - lazy so CPU-only paths never need it
    except ImportError as exc:
        raise ToolUnavailable(f"torch is unavailable in this namespace: {exc}") from exc
    if not torch.cuda.is_available():
        raise ToolUnavailable("CUDA is unavailable in this namespace")

    official = _load_module_exact(Path(parsed["official_path"]), "profile_official")
    shape = route_shape(parsed)
    config = official.TransformerConfig(
        batch_size=shape["batch_size"], seq_len=shape["seq_len"],
        d_model=shape["d_model"], num_heads=shape["num_heads"],
        ffn_dim=shape["ffn_dim"], num_layers=shape["num_layers"],
        causal=shape["causal"],
    )
    config.validate()
    device = torch.device("cuda")
    dtype = official.resolve_dtype(parsed["dtype"])
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision(NUMERICAL["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = NUMERICAL["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = NUMERICAL["allow_tf32"]

    baseline = official.BaselineTransformer(config).to(device=device, dtype=dtype).eval()
    baseline_state = {key: value.detach().cpu().clone()
                      for key, value in baseline.state_dict().items()}
    if parsed["route_kind"] == "baseline":
        model = baseline
        description = "official-baseline"
    else:
        target_module = _load_module_exact(Path(parsed["target_path"]), "profile_target")
        if hasattr(target_module, "build"):
            model = target_module.build(official, config)
        elif hasattr(target_module, "UserOptimizedTransformer"):
            model = target_module.UserOptimizedTransformer(config)
        else:
            raise ProfileWorkerError(
                "target must define build() or UserOptimizedTransformer")
        if hasattr(target_module, "copy_weights"):
            target_module.copy_weights(baseline, model)
        else:
            model.load_state_dict(baseline_state, strict=True)
        model = model.to(device=device, dtype=dtype).eval()
        description = "target-UserOptimizedTransformer"

    inputs, mask = official.generate_random_case(
        config, device, dtype, parsed["seed"],
        NUMERICAL["padding_ratio"], NUMERICAL["input_scale"],
    )
    return RouteContext(torch, official, config, device, dtype, model, baseline,
                        inputs, mask, shape, description)


def run_route(ctx: RouteContext, iterations: int) -> None:
    torch = ctx.torch
    with torch.inference_mode():
        for _ in range(iterations):
            ctx.model(ctx.inputs, ctx.mask)
    torch.cuda.synchronize()


def execute_route_child(parsed: dict[str, Any], summary_out: Path) -> dict[str, Any]:
    """``--execute-route`` mode: the process nsys / ncu / the sanitizer wrap.

    The measured window is bounded twice - by an NVTX range (so
    ``nsys stats --filter-nvtx`` can select it) and by the CUDA profiler API
    (so ``ncu --profile-from-start off`` collects only these launches).  Both
    are needed: warmup launches folded into the counts would quietly deflate
    the per-iteration numbers.
    """
    ctx = build_route(parsed)
    torch = ctx.torch
    run_route(ctx, parsed["warmup"])
    try:
        cudart = torch.cuda.cudart()
    except Exception:  # noqa: BLE001
        cudart = None
    started = time.perf_counter()
    torch.cuda.nvtx.range_push(NVTX_RANGE)
    if cudart is not None:
        try:
            cudart.cudaProfilerStart()
        except Exception:  # noqa: BLE001 - a missing profiler API is not fatal
            pass
    run_route(ctx, parsed["iterations"])
    if cudart is not None:
        try:
            cudart.cudaProfilerStop()
        except Exception:  # noqa: BLE001
            pass
    torch.cuda.nvtx.range_pop()
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "route": ctx.description,
        "shape_id": parsed["shape_id"],
        "iterations": parsed["iterations"],
        "warmup": parsed["warmup"],
        "dtype": parsed["dtype"],
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "wall_seconds_under_tool": elapsed,
        "wall_seconds_authority": "profiler-perturbed; never a speedup number",
        "nvtx_range": NVTX_RANGE,
    }
    write_exclusive(summary_out, canonical_json(summary) + b"\n")
    return summary


# --------------------------------------------------------------------------
# evidence staging
# --------------------------------------------------------------------------


class Evidence:
    """Collects raw files and their intended repository destinations."""

    def __init__(self, output_dir: Path, evidence_root: str, record_id: str) -> None:
        self.output_dir = output_dir
        self.raw_dir = output_dir / RAW_DIRNAME
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_root = evidence_root
        self.record_id = record_id
        self.placements: list[dict[str, str]] = []
        self.total_bytes = 0

    def repo_path(self, name: str) -> str:
        """Exactly where ``_ingest_raw_artifacts`` will put this file.

        The controller copies ``<output>/raw/<name>`` to
        ``<evidence>/<record_id>.raw/raw/<name>`` and re-hashes it there,
        then rewrites the artifact's ``raw_artifacts`` from disk.  Naming
        the same destination here makes the worker's claim and the
        controller's computation the same list instead of two lists that
        happen to share basenames.
        """
        return f"{self.evidence_root}/{self.record_id}.raw/{RAW_DIRNAME}/{name}"

    def add_text(self, name: str, text: str) -> Path:
        return self.add_bytes(name, text.encode("utf-8"))

    def add_bytes(self, name: str, data: bytes) -> Path:
        path = self.raw_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_exclusive(path, data)
        return self.adopt(path)

    def adopt(self, path: Path) -> Path:
        """Register an already-written file (e.g. one a tool produced)."""
        path = Path(path).resolve()
        if not path.is_file():
            raise ProfileWorkerError(f"raw artifact {path} is missing")
        try:
            relative = path.relative_to(self.raw_dir.resolve())
        except ValueError as exc:
            raise ProfileWorkerError(
                f"raw artifact {path} is outside the worker output directory") from exc
        size = path.stat().st_size
        if size > MAX_RAW_FILE_BYTES:
            raise ProfileWorkerError(
                f"raw artifact {relative} is {size} bytes; over the "
                f"{MAX_RAW_FILE_BYTES} byte per-file cap")
        name = str(relative)
        if Path(name).name == CONTROLLER_MACHINE_STATE_NAME:
            raise ProfileWorkerError(
                f"{CONTROLLER_MACHINE_STATE_NAME} is the controller's "
                "reserved name; _collect_worker_outputs refuses any worker "
                "output that uses it")
        if any(item["output_relpath"] == f"{RAW_DIRNAME}/{name}" for item in self.placements):
            return path
        self.total_bytes += size
        if self.total_bytes > MAX_TOTAL_RAW_BYTES:
            raise ProfileWorkerError("raw evidence exceeded the total size cap")
        self.placements.append({
            "output_relpath": f"{RAW_DIRNAME}/{name}",
            "repo_relpath": self.repo_path(name),
            "sha256": sha256_file(path),
            "bytes": str(size),
        })
        return path

    def try_adopt(self, path: Path, notes: list[str]) -> None:
        """Adopt a nice-to-have file, skipping it if it would blow the caps."""
        try:
            self.adopt(path)
        except ProfileWorkerError as exc:
            notes.append(f"optional raw artifact {Path(path).name} not kept: {exc}")

    def raw_artifacts(self) -> list[dict[str, str]]:
        # Exactly {"path", "sha256"} - the gate rejects any other key set.
        return [{"path": item["repo_relpath"], "sha256": item["sha256"]}
                for item in self.placements]

    def local_lookup(self) -> dict[str, Path]:
        return {item["repo_relpath"]: self.output_dir / item["output_relpath"]
                for item in self.placements}


def truncate_for_evidence(text: str, limit: int = 512 * 1024) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[TRUNCATED at {limit} characters by profile_worker]\n"


# --------------------------------------------------------------------------
# collectors
# --------------------------------------------------------------------------


def _child_argv(parsed: dict[str, Any], request_path: Path, output_dir: Path,
                summary_out: Path) -> list[str]:
    return [
        sys.executable or "/usr/bin/python3",
        str(Path(__file__).resolve()),
        "--execute-route",
        "--request", str(request_path),
        "--output", str(output_dir),
        "--summary-out", str(summary_out),
    ]


def collect_nsys(parsed: dict[str, Any], evidence: Evidence, request_path: Path,
                 notes: list[str]) -> tuple[dict[str, Any], str]:
    binary = discover_tool("nsys", parsed["tool_paths"],
                           parsed["tool_search_paths"])
    version = tool_version_string(binary)
    timeout = parsed["tool_timeout_seconds"]
    report_base = evidence.raw_dir / "nsys-timeline"
    summary_out = evidence.raw_dir / "nsys-route-summary.json"
    argv = [
        binary, "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "--output", str(report_base),
        *_child_argv(parsed, request_path, evidence.output_dir, summary_out),
    ]
    code, out, err = run_tool(argv, timeout_seconds=timeout, label="nsys profile")
    evidence.add_text("nsys-profile.log", truncate_for_evidence(
        f"$ {' '.join(argv)}\n--- rc={code}\n--- stdout\n{out}\n--- stderr\n{err}\n"))
    report = None
    for candidate in (Path(str(report_base) + ".nsys-rep"), report_base):
        if candidate.is_file():
            report = candidate
            break
    if report is None:
        raise ToolOutputError(
            f"nsys produced no report file (rc={code}); stderr: {err.strip()[:400]}")
    if code != 0:
        raise ToolOutputError(f"nsys profile failed rc={code}: {err.strip()[:400]}")
    evidence.adopt(report)
    if summary_out.is_file():
        evidence.adopt(summary_out)

    def stats(report_name: str, filtered: bool) -> str:
        stats_argv = [binary, "stats", "-r", report_name, "-f", "csv", "-o", "-",
                      "--force-export=true"]
        if filtered:
            stats_argv.extend(["--filter-nvtx", NVTX_RANGE])
        stats_argv.append(str(report))
        rc, stdout, stderr = run_tool(stats_argv, timeout_seconds=timeout,
                                      label=f"nsys stats {report_name}")
        if rc != 0:
            raise ToolOutputError(
                f"nsys stats {report_name} failed rc={rc}: {stderr.strip()[:400]}")
        return stdout

    window = "nvtx:" + NVTX_RANGE
    try:
        api_text = stats("cuda_api_sum", True)
        api = parse_nsys_api_sum(api_text)
    except ToolOutputError:
        # An NVTX-less capture is still real evidence; it just measures the
        # whole process, warmup included.  Say so instead of pretending.
        window = "whole-process"
        notes.append("NVTX-filtered nsys stats produced nothing; fell back to "
                     "whole-process counts including warmup launches")
        api_text = stats("cuda_api_sum", False)
        api = parse_nsys_api_sum(api_text)
    evidence.add_text("nsys-cuda_api_sum.csv", truncate_for_evidence(api_text))

    filtered = window.startswith("nvtx:")
    kern_text = stats("cuda_gpu_kern_sum", filtered)
    evidence.add_text("nsys-cuda_gpu_kern_sum.csv", truncate_for_evidence(kern_text))
    kern = parse_nsys_kern_sum(kern_text)

    trace_text = stats("cuda_gpu_trace", filtered)
    evidence.add_text("nsys-cuda_gpu_trace.csv", truncate_for_evidence(trace_text))
    trace = parse_nsys_gpu_trace(trace_text)

    # nsys stats writes a .sqlite beside the report; it is the queryable form
    # of the same bytes, so keep it when it fits inside the evidence caps.
    sqlite_path = Path(str(report_base) + ".sqlite")
    if sqlite_path.is_file():
        evidence.try_adopt(sqlite_path, notes)

    divisor = parsed["iterations"] if filtered else (parsed["iterations"] + parsed["warmup"])
    divisor = max(1, divisor)
    total_launches = api["kernel_launch_api_calls"] + api["graph_launch_api_calls"]
    metrics = {
        "kernel_launches": total_launches / divisor,
        "kernel_launches_total": total_launches,
        "kernel_launch_api_calls": api["kernel_launch_api_calls"],
        "graph_launch_api_calls": api["graph_launch_api_calls"],
        "cuda_api_calls_by_name": api["cuda_api_calls_by_name"],
        "kernel_instances_total": kern["kernel_instances"],
        "kernel_instances": kern["kernel_instances"] / divisor,
        "distinct_kernels": kern["distinct_kernels"],
        "gpu_idle_fraction": trace["gpu_idle_fraction"],
        "gpu_busy_ns": trace["gpu_busy_ns"],
        "gpu_span_ns": trace["gpu_span_ns"],
        "device_activity_records": trace["device_activity_records"],
        "profiled_iterations": divisor,
        "measurement_window": window,
        "kernel_launches_definition":
            "CUDA launch + graph-launch API calls per profiled forward pass",
        "characterization_metric": total_launches / divisor,
    }
    return metrics, f"nsys {version}"


def collect_torch_profiler(parsed: dict[str, Any], evidence: Evidence,
                           notes: list[str]) -> tuple[dict[str, Any], str]:
    ctx = build_route(parsed)
    torch = ctx.torch
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    run_route(ctx, parsed["warmup"])
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        run_route(ctx, parsed["iterations"])
    trace_path = evidence.raw_dir / "torch-profiler-trace.json"
    prof.export_chrome_trace(str(trace_path))
    evidence.adopt(trace_path)
    try:
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60)
    except Exception:  # noqa: BLE001 - sort key names drift across torch versions
        table = prof.key_averages().table(row_limit=60)
    evidence.add_text("torch-profiler-key-averages.txt", truncate_for_evidence(table))

    document = json.loads(trace_path.read_text(encoding="utf-8"))
    parsed_trace = parse_torch_chrome_trace(document)
    divisor = max(1, parsed["iterations"])
    total_launches = (parsed_trace["kernel_launch_api_calls"]
                      + parsed_trace["graph_launch_api_calls"])
    metrics = {
        "kernel_launches": total_launches / divisor,
        "kernel_launches_total": total_launches,
        "kernel_launch_api_calls": parsed_trace["kernel_launch_api_calls"],
        "graph_launch_api_calls": parsed_trace["graph_launch_api_calls"],
        "cuda_api_calls_by_name": parsed_trace["cuda_api_calls_by_name"],
        "kernel_instances_total": parsed_trace["kernel_instances"],
        "kernel_instances": parsed_trace["kernel_instances"] / divisor,
        "gpu_idle_fraction": parsed_trace["gpu_idle_fraction"],
        "gpu_busy_us": parsed_trace["gpu_busy_us"],
        "gpu_span_us": parsed_trace["gpu_span_us"],
        "device_activity_records": parsed_trace["device_activity_records"],
        "profiled_iterations": divisor,
        "measurement_window": "profiler-context (warmup excluded)",
        "kernel_launches_definition":
            "CUDA launch + graph-launch API calls per profiled forward pass",
        "route": ctx.description,
        "characterization_metric": total_launches / divisor,
    }
    return metrics, f"torch.profiler/{torch.__version__}"


def collect_ncu(parsed: dict[str, Any], evidence: Evidence, request_path: Path,
                notes: list[str]) -> tuple[dict[str, Any], str]:
    binary = discover_tool("ncu", parsed["tool_paths"],
                           parsed["tool_search_paths"])
    version = tool_version_string(binary)
    timeout = parsed["tool_timeout_seconds"]
    summary_out = evidence.raw_dir / "ncu-route-summary.json"
    prefix = list(parsed["ncu_command_prefix"])
    if prefix:
        notes.append(f"ncu invoked through prefix {prefix!r}; under bubblewrap "
                     "with --cap-drop ALL this cannot gain privilege")
    argv = [
        *prefix, binary,
        "--csv",
        "--target-processes", "all",
        "--profile-from-start", "off",
        "--graph-profiling", "node",
        "--replay-mode", "kernel",
        "--metrics", ",".join(NCU_METRICS),
        *_child_argv(parsed, request_path, evidence.output_dir, summary_out),
    ]
    code, out, err = run_tool(argv, timeout_seconds=timeout, label="ncu")
    evidence.add_text("ncu-stdout.csv", truncate_for_evidence(out))
    evidence.add_text("ncu-stderr.log", truncate_for_evidence(
        f"$ {' '.join(argv)}\n--- rc={code}\n{err}\n"))
    if ncu_permission_denied(out) or ncu_permission_denied(err):
        raise ToolUnavailable(
            "ncu is blocked by the driver parameter RmProfilingAdminOnly=1 "
            "(ERR_NVGPUCTRPERM). It needs `sudo ncu`, which the sandbox cannot "
            "grant. Re-run the diagnostic with tool=nsys, or have the owner run "
            "this worker outside the sandbox with an ncu_command_prefix. Never "
            "modprobe or reboot to work around this.")
    if code != 0:
        raise ToolOutputError(f"ncu failed rc={code}: {err.strip()[:400]}")
    if summary_out.is_file():
        evidence.adopt(summary_out)

    counters = parse_ncu_csv(out)
    durations = counters.get("gpu__time_duration.sum", [])

    def total(name: str) -> float | None:
        values = counters.get(name)
        return float(sum(values)) if values else None

    def fraction(name: str) -> float | None:
        values = counters.get(name)
        if not values:
            return None
        return weighted_mean(values, durations) / 100.0

    dram = total("dram__bytes.sum")
    if dram is None:
        read = total("dram__bytes_read.sum")
        write = total("dram__bytes_write.sum")
        if read is None and write is None:
            raise ToolOutputError("ncu returned no DRAM byte counters")
        dram = (read or 0.0) + (write or 0.0)
    compute = fraction("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    occupancy = fraction("sm__warps_active.avg.pct_of_peak_sustained_active")
    tensor = fraction("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")
    missing = [name for name, value in (
        ("compute_utilization", compute),
        ("achieved_occupancy", occupancy),
        ("tensor_core_utilization", tensor)) if value is None]
    if missing:
        raise ToolOutputError(
            f"ncu returned no values for {missing}; the counter names may have "
            "changed in this Nsight Compute release")
    kernels = len(counters.get("dram__bytes.sum",
                               counters.get("gpu__time_duration.sum", [])))
    divisor = max(1, parsed["iterations"])
    metrics = {
        "dram_bytes": dram / divisor,
        "dram_bytes_total": dram,
        "compute_utilization": compute,
        "compute_utilization_pct": compute * 100.0,
        "achieved_occupancy": occupancy,
        "achieved_occupancy_pct": occupancy * 100.0,
        "tensor_core_utilization": tensor,
        "tensor_core_utilization_pct": tensor * 100.0,
        "tensor_pipe_instructions": total("sm__inst_executed_pipe_tensor.sum"),
        "gpu_time_duration_ns_total": total("gpu__time_duration.sum"),
        "ncu_kernels_profiled": kernels,
        "profiled_iterations": divisor,
        "aggregation": "duration-weighted mean for rate counters, sum for byte counters",
        "graph_profiling": "node (CUDA graphs profiled as individual kernel nodes)",
        "characterization_metric": compute,
    }
    return ({key: value for key, value in metrics.items() if value is not None},
            f"ncu {version}")


def collect_memory_profile(parsed: dict[str, Any], evidence: Evidence,
                           notes: list[str]) -> tuple[dict[str, Any], str]:
    ctx = build_route(parsed)
    torch = ctx.torch
    run_route(ctx, parsed["warmup"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    history = False
    try:
        torch.cuda.memory._record_memory_history(max_entries=200_000)
        history = True
    except Exception as exc:  # noqa: BLE001 - private API, guarded on purpose
        notes.append(f"allocator history unavailable: {exc}")
    run_route(ctx, parsed["iterations"])
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    peak_reserved = float(torch.cuda.max_memory_reserved())
    peak_allocated = float(torch.cuda.max_memory_allocated())
    largest_alloc = None
    alloc_events = 0
    if history:
        try:
            snapshot = torch.cuda.memory._snapshot()
            for trace in snapshot.get("device_traces", []) or []:
                for entry in trace:
                    if entry.get("action") in {"alloc", "segment_alloc"}:
                        alloc_events += 1
                        size = float(entry.get("size", 0))
                        largest_alloc = size if largest_alloc is None else max(
                            largest_alloc, size)
            evidence.add_text("torch-memory-history.json", truncate_for_evidence(
                json.dumps({"device_traces_summary": {
                    "alloc_events": alloc_events,
                    "largest_single_allocation_bytes": largest_alloc,
                }, "segments": len(snapshot.get("segments", []) or [])}, indent=1)))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"allocator snapshot could not be read: {exc}")
        finally:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:  # noqa: BLE001
                pass
    if largest_alloc is None or largest_alloc <= 0:
        largest_alloc = peak_allocated
        materialized_source = "peak-allocated-fallback"
        notes.append("materialized_bytes fell back to peak allocated bytes; the "
                     "allocator history was not readable")
    else:
        materialized_source = "largest-single-allocation-in-allocator-history"
    evidence.add_text("torch-memory-stats.json", truncate_for_evidence(
        json.dumps({key: value for key, value in stats.items()
                    if isinstance(value, (int, float))}, indent=1, sort_keys=True)))
    metrics = {
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_bytes": peak_allocated,
        "materialized_bytes": float(largest_alloc),
        "materialized_bytes_source": materialized_source,
        "allocation_events": alloc_events,
        "profiled_iterations": parsed["iterations"],
        "route": ctx.description,
        "characterization_metric": peak_reserved,
    }
    return metrics, f"torch.cuda.memory/{torch.__version__}"


def static_route_model(shape: dict[str, Any], dtype_name: str) -> dict[str, Any]:
    """Explicit FLOP/byte accounting for the official dense route.

    Every term below is named and kept in the raw artifact so a reviewer can
    check the arithmetic line by line.  The traffic model credits no cache
    reuse: each intermediate is written once and read once, parameters are
    read once per forward.  That makes ``dram_bytes`` a compulsory-traffic
    LOWER BOUND, and the artifact says so in its own metrics.
    """
    B = int(shape["batch_size"])
    S = int(shape["seq_len"])
    D = int(shape["d_model"])
    H = int(shape["num_heads"])
    F = int(shape["ffn_dim"])
    L = int(shape["num_layers"])
    causal = bool(shape["causal"])
    if D % H:
        raise ProfileWorkerError("d_model is not divisible by num_heads")
    head_dim = D // H
    e = DTYPE_ITEMSIZE[dtype_name]

    act = B * S * D * e             # one [B, S, D] activation
    ffn_act = B * S * F * e         # one [B, S, F] activation
    scores_e = B * H * S * S * e    # one dense score matrix in route dtype
    scores_f32 = B * H * S * S * 4  # the fp32 softmax copy the baseline makes

    params_per_layer = 4 * (D * D + D) + (D * F + F) + (F * D + D) + 2 * (2 * D)
    params = L * params_per_layer + 2 * D
    param_bytes = params * e

    terms: list[dict[str, Any]] = [
        {"term": "parameters_read_once", "bytes": params_per_layer * e},
        {"term": "norm1_rw", "bytes": 2 * act},
        {"term": "qkv_projection_rw", "bytes": 3 * (act + act)},
        {"term": "split_heads_contiguous_rw", "bytes": 3 * (act + act)},
        {"term": "scores_matmul", "bytes": 2 * act + scores_e},
        {"term": "scale_multiply_rw", "bytes": 2 * scores_e},
    ]
    if causal:
        terms.append({"term": "causal_masked_fill_rw", "bytes": 2 * scores_e})
    terms.extend([
        {"term": "key_mask_masked_fill_rw", "bytes": 2 * scores_e},
        {"term": "scores_to_float32", "bytes": scores_e + scores_f32},
        {"term": "softmax_fp32_rw", "bytes": 2 * scores_f32},
        {"term": "probs_to_dtype", "bytes": scores_f32 + scores_e},
        {"term": "probs_times_v", "bytes": scores_e + act + act},
        {"term": "context_transpose_contiguous_rw", "bytes": 2 * act},
        {"term": "out_projection_rw", "bytes": 2 * act},
        {"term": "attention_output_masked_fill_rw", "bytes": 2 * act},
        {"term": "residual_add_1", "bytes": 3 * act},
        {"term": "norm2_rw", "bytes": 2 * act},
        {"term": "ffn_in", "bytes": act + ffn_act},
        {"term": "gelu_rw", "bytes": 2 * ffn_act},
        {"term": "ffn_out", "bytes": ffn_act + act},
        {"term": "residual_add_2", "bytes": 3 * act},
        {"term": "block_masked_fill_rw", "bytes": 2 * act},
    ])
    per_layer_bytes = sum(int(term["bytes"]) for term in terms)
    tail_bytes = 2 * act + 2 * act + 2 * D * e  # final_norm rw, masked_fill rw, params
    dram_bytes = L * per_layer_bytes + tail_bytes

    projection_flops = 4 * 2 * B * S * D * D
    score_flops = 2 * B * H * S * S * head_dim
    context_flops = 2 * B * H * S * S * head_dim
    ffn_flops = 2 * B * S * D * F + 2 * B * S * F * D
    matmul_flops_per_layer = projection_flops + score_flops + context_flops + ffn_flops
    matmul_flops = L * matmul_flops_per_layer
    causal_useful_ratio = ((S + 1) / (2 * S)) if causal else 1.0
    useful_attention_flops = L * (score_flops + context_flops) * causal_useful_ratio

    softmax_peak = scores_e + 2 * scores_f32
    working_activations = 8 * act + 2 * ffn_act
    peak_reserved_model = param_bytes + softmax_peak + working_activations

    return {
        "dims": {"batch_size": B, "seq_len": S, "d_model": D, "num_heads": H,
                 "head_dim": head_dim, "ffn_dim": F, "num_layers": L,
                 "causal": causal, "dtype": dtype_name, "itemsize": e},
        "traffic_terms_per_layer": terms,
        "traffic_tail_bytes": tail_bytes,
        "dram_bytes": dram_bytes,
        "dram_bytes_model": "baseline-dense-route compulsory traffic, one write "
                            "+ one read per intermediate, no cache reuse credited",
        "parameter_bytes": param_bytes,
        "parameters": params,
        "matmul_flops": matmul_flops,
        "matmul_flops_per_layer": matmul_flops_per_layer,
        "projection_flops_per_layer": projection_flops,
        "attention_flops_per_layer": score_flops + context_flops,
        "ffn_flops_per_layer": ffn_flops,
        "useful_attention_flops_causal": useful_attention_flops,
        "materialized_bytes": scores_e,
        "materialized_bytes_softmax_peak": softmax_peak,
        "materialized_bytes_all_layers_if_retained": L * scores_e,
        "peak_reserved_bytes": peak_reserved_model,
        "activation_bytes": act,
        "ffn_activation_bytes": ffn_act,
        "score_matrix_bytes": scores_e,
        "score_matrix_fp32_bytes": scores_f32,
    }


def collect_static_analysis(parsed: dict[str, Any], evidence: Evidence,
                            notes: list[str]) -> tuple[dict[str, Any], str]:
    shape = route_shape(parsed)
    model = static_route_model(shape, parsed["dtype"])
    evidence.add_text("static-analysis-model.json",
                      json.dumps(model, indent=1, sort_keys=True))
    notes.append("static-analysis values are analytic models, not counters; "
                 "dram_bytes is a compulsory-traffic lower bound")
    metrics = {
        "dram_bytes": float(model["dram_bytes"]),
        "dram_bytes_model": model["dram_bytes_model"],
        "dram_bytes_is_lower_bound": True,
        "materialized_bytes": float(model["materialized_bytes"]),
        "materialized_bytes_model": "one dense [B,H,S,S] score matrix in route dtype",
        "materialized_bytes_softmax_peak": float(model["materialized_bytes_softmax_peak"]),
        "peak_reserved_bytes": float(model["peak_reserved_bytes"]),
        "peak_reserved_bytes_model": "parameters + softmax-time score peak + "
                                     "working activation set; analytic lower bound",
        "matmul_flops": float(model["matmul_flops"]),
        "useful_attention_flops_causal": float(model["useful_attention_flops_causal"]),
        "parameter_bytes": float(model["parameter_bytes"]),
        "arithmetic_intensity_flops_per_byte":
            float(model["matmul_flops"]) / float(model["dram_bytes"]),
        "evidence_kind": "analytic-model (no hardware counter)",
        "characterization_metric": float(model["dram_bytes"]),
    }
    version = (f"profile_worker.static_analysis/{WORKER_VERSION} "
               f"python{platform.python_version()}")
    return metrics, version


def _event_time_ms(ctx: RouteContext, model: Any, iterations: int) -> float:
    torch = ctx.torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        torch.cuda.synchronize()
        start.record()
        for _ in range(iterations):
            model(ctx.inputs, ctx.mask)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _roofline_probes(ctx: RouteContext, notes: list[str]) -> dict[str, float]:
    """Measure this device's achievable peaks instead of trusting a table.

    LESSONS #24: a vendor peak quoted from memory becomes a claim nobody can
    trace.  These probes are a few lines of cuBLAS/copy work whose result is
    recorded next to the number it normalises.
    """
    torch = ctx.torch
    device = ctx.device
    free_bytes, _total = torch.cuda.mem_get_info()
    probes: dict[str, float] = {}

    def gemm_flops(dtype: Any, size: int) -> float:
        a = torch.randn((size, size), device=device, dtype=dtype)
        b = torch.randn((size, size), device=device, dtype=dtype)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            a @ b
        end.record()
        torch.cuda.synchronize()
        seconds = start.elapsed_time(end) / 1000.0 / 10
        del a, b
        torch.cuda.empty_cache()
        return 2.0 * size * size * size / seconds

    size = 4096
    while size >= 512 and 3 * size * size * 4 > free_bytes * 0.25:
        size //= 2
    probes["gemm_probe_size"] = float(size)
    probes["measured_peak_flops_fp32"] = gemm_flops(torch.float32, size)
    try:
        probes["measured_peak_flops_fp16"] = gemm_flops(torch.float16, size)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"fp16 roofline probe failed: {exc}")

    copy_bytes = int(min(256 * 1024 * 1024, free_bytes * 0.15))
    copy_bytes = max(16 * 1024 * 1024, copy_bytes - copy_bytes % 4096)
    src = torch.empty(copy_bytes, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    dst.copy_(src)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1000.0 / 10
    probes["measured_peak_bandwidth_bytes_per_s"] = 2.0 * copy_bytes / seconds
    probes["bandwidth_probe_bytes"] = float(copy_bytes)
    del src, dst
    torch.cuda.empty_cache()
    return probes


def collect_microbenchmark(parsed: dict[str, Any], evidence: Evidence,
                           notes: list[str]) -> tuple[dict[str, Any], str]:
    ctx = build_route(parsed)
    torch = ctx.torch
    model = static_route_model(ctx.shape, parsed["dtype"])
    probes = _roofline_probes(ctx, notes)

    run_route(ctx, parsed["warmup"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ms_per_iter = _event_time_ms(ctx, ctx.model, parsed["iterations"])
    seconds = ms_per_iter / 1000.0
    peak_reserved = float(torch.cuda.max_memory_reserved())
    peak_allocated = float(torch.cuda.max_memory_allocated())

    achieved_flops = float(model["matmul_flops"]) / seconds
    fp32_peak = probes.get("measured_peak_flops_fp32")
    fp16_peak = probes.get("measured_peak_flops_fp16")
    dtype_peak = (fp16_peak if parsed["dtype"] in {"float16", "bfloat16"} and fp16_peak
                  else fp32_peak)
    compute_utilization = achieved_flops / dtype_peak if dtype_peak else None
    tensor_utilization = achieved_flops / fp16_peak if fp16_peak else None
    achieved_bandwidth = float(model["dram_bytes"]) / seconds
    bandwidth_peak = probes.get("measured_peak_bandwidth_bytes_per_s")

    # Correctness against the official baseline, so numerical-precision is
    # answerable from the same run.
    with torch.inference_mode():
        reference = ctx.baseline(ctx.inputs, ctx.mask)
        candidate = ctx.model(ctx.inputs, ctx.mask)
        result = ctx.official.compare_outputs(
            reference, candidate, rtol=NUMERICAL["rtol"], atol=NUMERICAL["atol"])

    evidence.add_text("microbenchmark-model.json",
                      json.dumps({"static_model": model, "roofline_probes": probes,
                                  "ms_per_iteration": ms_per_iter},
                                 indent=1, sort_keys=True))
    metrics: dict[str, Any] = {
        "dram_bytes": float(model["dram_bytes"]),
        "dram_bytes_model": model["dram_bytes_model"],
        "dram_bytes_is_lower_bound": True,
        "materialized_bytes": float(model["materialized_bytes"]),
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_bytes": peak_allocated,
        "failed_elements": int(result.failed_elements),
        "total_elements": int(result.total_elements),
        "max_relative_error": float(result.max_relative_error),
        "max_abs_error": float(result.max_abs_error),
        "route_ms_per_iteration": ms_per_iter,
        "route_ms_authority": "supporting diagnostic timing; never a promotion number",
        "achieved_flops_per_s": achieved_flops,
        "achieved_dram_bytes_per_s": achieved_bandwidth,
        "profiled_iterations": parsed["iterations"],
        "route": ctx.description,
        "peak_source": "measured on-device roofline probe in this same process",
        **{key: float(value) for key, value in probes.items()},
    }
    if compute_utilization is not None:
        metrics["compute_utilization"] = compute_utilization
        metrics["compute_utilization_method"] = (
            "analytic matmul FLOPs / measured route time / measured GEMM peak")
        metrics["characterization_metric"] = compute_utilization
    else:
        metrics["characterization_metric"] = achieved_bandwidth
    if tensor_utilization is not None:
        metrics["tensor_core_utilization"] = tensor_utilization
        metrics["tensor_core_utilization_method"] = (
            "route matmul FLOP rate / measured fp16 GEMM peak on this device")
    if bandwidth_peak:
        metrics["dram_bandwidth_fraction_of_measured_peak"] = (
            achieved_bandwidth / bandwidth_peak)
    return metrics, f"microbenchmark(torch {torch.__version__})/{WORKER_VERSION}"


def collect_correctness_evaluator(parsed: dict[str, Any], evidence: Evidence,
                                  notes: list[str]) -> tuple[dict[str, Any], str]:
    ctx = build_route(parsed)
    torch = ctx.torch
    official = ctx.official
    seeds = [parsed["seed"] + offset for offset in (0, 1, 2, 3, 4)]
    trials = []
    worst_failed = 0
    total_elements = 0
    worst_rel = 0.0
    worst_abs = 0.0
    passed = True
    with torch.inference_mode():
        for seed in seeds:
            x, mask = official.generate_random_case(
                ctx.config, ctx.device, ctx.dtype, seed,
                NUMERICAL["padding_ratio"], NUMERICAL["input_scale"])
            reference = ctx.baseline(x, mask)
            candidate = ctx.model(x, mask)
            result = official.compare_outputs(
                reference, candidate, rtol=NUMERICAL["rtol"], atol=NUMERICAL["atol"])
            trials.append({
                "seed": seed, "passed": bool(result.passed),
                "failed_elements": int(result.failed_elements),
                "total_elements": int(result.total_elements),
                "max_abs_error": float(result.max_abs_error),
                "max_relative_error": float(result.max_relative_error),
            })
            worst_failed = max(worst_failed, int(result.failed_elements))
            total_elements = int(result.total_elements)
            worst_rel = max(worst_rel, float(result.max_relative_error))
            worst_abs = max(worst_abs, float(result.max_abs_error))
            passed &= bool(result.passed)
    evidence.add_text("correctness-trials.json",
                      json.dumps({"trials": trials, "rtol": NUMERICAL["rtol"],
                                  "atol": NUMERICAL["atol"]}, indent=1))
    metrics = {
        "failed_elements": worst_failed,
        "total_elements": total_elements,
        "failed_fraction": (worst_failed / total_elements) if total_elements else 0.0,
        "max_relative_error": worst_rel,
        "max_abs_error": worst_abs,
        "all_seeds_passed": passed,
        "seeds": seeds,
        "route": ctx.description,
        "characterization_metric": float(worst_failed),
    }
    return metrics, f"official.compare_outputs(torch {torch.__version__})/{WORKER_VERSION}"


def collect_sanitizer(parsed: dict[str, Any], evidence: Evidence,
                      request_path: Path, notes: list[str]) -> dict[str, Any]:
    binary = discover_tool("compute-sanitizer", parsed["tool_paths"],
                           parsed["tool_search_paths"])
    version = tool_version_string(binary)
    summary_out = evidence.raw_dir / "sanitizer-route-summary.json"
    argv = [binary, "--tool", parsed["sanitizer"], "--print-limit", "50",
            *_child_argv(parsed, request_path, evidence.output_dir, summary_out)]
    code, out, err = run_tool(argv, timeout_seconds=parsed["tool_timeout_seconds"],
                              label="compute-sanitizer")
    evidence.add_text(f"compute-sanitizer-{parsed['sanitizer']}.log",
                      truncate_for_evidence(
                          f"$ {' '.join(argv)}\n--- rc={code}\n"
                          f"--- stdout\n{out}\n--- stderr\n{err}\n"))
    if summary_out.is_file():
        evidence.adopt(summary_out)
    findings = parse_compute_sanitizer(out + "\n" + err)
    findings["sanitizer_tool"] = parsed["sanitizer"]
    findings["sanitizer_version"] = version
    if not findings.get("sanitizer_clean"):
        notes.append(f"compute-sanitizer {parsed['sanitizer']} reported "
                     f"{findings['sanitizer_errors']} error(s)")
    return findings


COLLECTORS: dict[str, Callable[..., tuple[dict[str, Any], str]]] = {
    "nsys": collect_nsys,
    "torch-profiler": collect_torch_profiler,
    "ncu": collect_ncu,
    "memory-profile": collect_memory_profile,
    "static-analysis": collect_static_analysis,
    "microbenchmark": collect_microbenchmark,
    "correctness-evaluator": collect_correctness_evaluator,
}
CHILD_PROCESS_TOOLS = {"nsys", "ncu"}


# --------------------------------------------------------------------------
# artifact assembly and gate-equivalent validation
# --------------------------------------------------------------------------


def gate_equivalent_check(
    artifact: dict[str, Any],
    parsed: dict[str, Any],
    evidence_root: str,
    file_lookup: dict[str, Path],
) -> None:
    """Re-run the acceptance rules of BOTH consumers before anything leaves.

    Two consumers read this artifact and both refuse on set equality:
    ``trusted_controller._finalize_profile_artifact`` /
    ``_verify_profile_artifact`` first, then
    ``run_gate._reconcile_authority_diagnostic``.  A refusal at the gate
    wedges the request registry permanently; a refusal in the controller is a
    settleable infrastructure failure; a refusal here costs nothing at all.
    So this is written from those rules rather than sharing code with them: if
    this worker drifts, the mismatch should surface here.

    The bindings come from the flat request, which is the controller's own
    projection of the immutable gate request -- including
    ``machine_state_sha256``, which is authority-owned and merely echoed.
    """
    expected_fields = set(parsed["artifact_fields"]) or set(ARTIFACT_REQUIRED_KEYS)
    if set(artifact) != expected_fields:
        extra = sorted(set(artifact) - expected_fields)
        missing = sorted(expected_fields - set(artifact))
        raise ProfileWorkerError(
            f"artifact key set mismatch (extra={extra}, missing={missing})")
    if artifact["schema_version"] != 1:
        raise ProfileWorkerError("artifact schema_version must be 1")
    # Every authority-owned field, exactly as _finalize_profile_artifact binds
    # them.  Contradicting one is refused there; echoing them is what makes
    # this worker's artifact and the controller's finalized artifact identical.
    bindings = {
        "profile_record_id": parsed["profile_record_id"],
        "request_id": parsed["request_id"],
        "campaign_id": parsed["campaign_id"],
        "shape": parsed["shape_id"],
        "target_sha256": parsed["target_sha256"],
        "tool": parsed["tool"],
        "route": parsed["route"],
        "supported_bottlenecks": parsed["supported_bottlenecks"],
        "gate_request_sha256": parsed["gate_request_sha256"],
        "machine_state_sha256": parsed["machine_state_sha256"],
    }
    for key, value in bindings.items():
        if artifact[key] != value:
            raise ProfileWorkerError(
                f"artifact {key} does not echo the controller's binding")
    if not isinstance(artifact["metrics"], dict) or not artifact["metrics"]:
        raise ProfileWorkerError("artifact metrics must be a non-empty object")
    if not isinstance(artifact["raw_artifacts"], list) or not artifact["raw_artifacts"]:
        raise ProfileWorkerError("artifact must carry at least one raw artifact")
    created = artifact["created_epoch"]
    if (isinstance(created, bool) or not isinstance(created, (int, float))
            or not math.isfinite(float(created))):
        raise ProfileWorkerError("created_epoch must be a finite number")
    now = time.time()
    # _finalize_profile_artifact accepts started_epoch-3600 .. now+3600; this
    # is the tighter half of that window, so the worker fails first.
    if float(created) > now + 60 or now - float(created) > 3600:
        raise ProfileWorkerError(
            "created_epoch is outside the controller's acceptance window")
    if not HEX64.fullmatch(str(artifact["machine_state_sha256"])):
        raise ProfileWorkerError("machine_state_sha256 must be 64 lowercase hex")
    if not isinstance(artifact["tool_version"], str) or not artifact["tool_version"]:
        raise ProfileWorkerError("tool_version must be a non-empty string")
    for raw in artifact["raw_artifacts"]:
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
            raise ProfileWorkerError("raw artifact entries must hold exactly path+sha256")
        path = str(raw["path"])
        if not HEX64.fullmatch(str(raw["sha256"])):
            raise ProfileWorkerError(f"raw artifact {path} has a malformed sha256")
        normalized = os.path.normpath(path)
        if normalized.startswith("/") or ".." in normalized.split("/"):
            raise ProfileWorkerError(f"raw artifact path {path} is not repo-relative")
        if not (normalized == evidence_root
                or normalized.startswith(evidence_root.rstrip("/") + "/")):
            raise ProfileWorkerError(f"raw artifact {path} escapes {evidence_root}")
        if Path(normalized).name == CONTROLLER_MACHINE_STATE_NAME:
            raise ProfileWorkerError(
                "the worker may not claim the controller's reserved "
                f"{CONTROLLER_MACHINE_STATE_NAME}")
        local = file_lookup.get(path)
        if local is None or not Path(local).is_file():
            raise ProfileWorkerError(f"raw artifact {path} has no staged file")
        if sha256_file(Path(local)) != raw["sha256"]:
            raise ProfileWorkerError(f"raw artifact {path} sha256 does not match its bytes")
    # The bottleneck contract comes from the request: the controller pinned it
    # from the real mechanism_catalog.json and the gate re-checks the same
    # terms.  Judging against the fallback table in this file could only ever
    # reject evidence the controller already authorized.
    for bottleneck, metrics in parsed["required_metrics"].items():
        if bottleneck not in artifact["supported_bottlenecks"]:
            raise ProfileWorkerError(
                f"artifact does not declare the requested bottleneck {bottleneck!r}")
        missing = [metric for metric in metrics
                   if metric not in artifact["metrics"]
                   or artifact["metrics"][metric] is None]
        if missing:
            raise ProfileWorkerError(
                f"metrics for {bottleneck} are missing {missing}; the tool did not "
                "answer the question this diagnostic was opened to answer")
    try:
        canonical_json(artifact)
    except ValueError as exc:
        raise ProfileWorkerError(f"artifact is not canonically serialisable: {exc}") from exc


def artifact_basename(parsed: dict[str, Any]) -> str:
    """The filename the controller told the worker to write.

    ``_collect_worker_outputs`` looks for exactly this name in /output and
    falls back to guessing only when it is absent, so writing anything else
    is at best a guess the controller has to resolve.
    """
    return parsed["artifact_filename"]


def planned_placement(parsed: dict[str, Any]) -> dict[str, str]:
    """Where the artifact goes: /output/<name> now, the evidence root after.

    The controller does not read this -- it writes the finalized artifact to
    the path its own gate request named.  This is a statement of intent kept
    in the manifest so a reviewer can see the two sides agreed on the
    destination, and so a disagreement is visible instead of silent.
    """
    name = artifact_basename(parsed)
    return {"output_relpath": name,
            "repo_relpath": f"{parsed['evidence_root']}/{name}"}


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None or isinstance(value, (bool, str)):
            clean[key] = value
        elif isinstance(value, (int, float)):
            clean[key] = finite(value, f"metrics.{key}")
        elif isinstance(value, (list, dict)):
            clean[key] = json.loads(json.dumps(value, allow_nan=False))
        else:
            clean[key] = str(value)
    return clean


def run_profile(worker_request: dict[str, Any], output_dir: Path,
                request_path: Path) -> dict[str, Any]:
    """Produce the gate artifact.  Returns the controller-facing manifest."""
    started = time.time()
    parsed = parse_worker_request(worker_request)
    tool = parsed["tool"]
    notes: list[str] = list(parsed["request_notes"])
    degradations: list[str] = []
    if parsed["controller_notes"]:
        notes.append(f"controller: {parsed['controller_notes']}")

    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise ProfileWorkerError("--output must be an existing writable directory")
    if str(output_dir) != parsed["output_dir"]:
        # Not fatal: the self-test and any out-of-jail rehearsal run with a
        # temporary directory.  Inside the jail these are the same string.
        notes.append(
            f"--output {output_dir} is not the request's output_dir "
            f"{parsed['output_dir']}; running outside the controller's mount")

    # Refuse before doing any work if this request could never reconcile.
    if tool not in COLLECTORS:
        raise ProfileWorkerError(f"no collector implements tool {tool!r}")
    capable = COLLECTOR_METRICS[tool]
    for bottleneck, metrics in parsed["required_metrics"].items():
        unreachable = [metric for metric in metrics if metric not in capable]
        if unreachable:
            raise ProfileWorkerError(
                f"tool {tool!r} cannot produce {unreachable} for {bottleneck!r}; "
                "this diagnostic could never answer the question it was opened "
                "to answer")
    # The controller already checked tool-vs-bottleneck admissibility against
    # the live mechanism catalog and pinned the terms into required_metrics.
    # The table in this file is a cross-check that reports disagreement and
    # never overrides: refusing on a stale local copy could only reject work
    # the trusted side already authorized.
    catalog, catalog_source = load_catalog(parsed)
    for bottleneck in parsed["supported_bottlenecks"]:
        entry = catalog.get(bottleneck)
        if not isinstance(entry, dict):
            degradations.append(
                f"this worker's catalog copy does not know {bottleneck!r}; "
                "validated against the controller's pinned terms instead")
        elif tool not in entry.get("evidence_tools", []):
            degradations.append(
                f"this worker's catalog copy does not list {tool!r} as evidence "
                f"for {bottleneck!r}; the controller's catalog does, and it wins")
    verify_target(parsed)

    evidence = Evidence(output_dir, parsed["evidence_root"],
                        parsed["profile_record_id"])
    before = capture_machine_state("before", overrides=parsed["tool_paths"],
                                   search_paths=parsed["tool_search_paths"])

    collector = COLLECTORS[tool]
    if tool in CHILD_PROCESS_TOOLS:
        metrics, tool_version = collector(parsed, evidence, request_path, notes)
    else:
        metrics, tool_version = collector(parsed, evidence, notes)

    if parsed["sanitizer"]:
        try:
            metrics.update(collect_sanitizer(parsed, evidence, request_path, notes))
        except ProfileWorkerError as exc:
            degradations.append(f"compute-sanitizer pass skipped: {exc}")

    if parsed["incumbent_speedup"] is not None:
        metrics["incumbent_speedup"] = parsed["incumbent_speedup"]
        metrics["incumbent_speedup_source"] = "controller-supplied from durable state"
    elif parsed["measure_incumbent_speedup"]:
        try:
            ctx = build_route(parsed)
            run_route(ctx, parsed["warmup"])
            baseline_ms = _event_time_ms(ctx, ctx.baseline, parsed["iterations"])
            route_ms = _event_time_ms(ctx, ctx.model, parsed["iterations"])
            metrics["incumbent_speedup"] = baseline_ms / route_ms
            metrics["incumbent_speedup_source"] = "worker-measured-unprofiled"
            metrics["incumbent_speedup_authority"] = (
                "supporting only; promotion speedups come from the runner")
        except ProfileWorkerError as exc:
            degradations.append(f"incumbent speedup could not be measured: {exc}")
            metrics["incumbent_speedup_source"] = "unavailable"
    else:
        # The controller's flat request carries no incumbent_speedup field, so
        # this is the live path, and it is deliberate.  A diagnostic runs under
        # an instrument and states plainly that its timings are not performance
        # numbers, so it is the wrong place to decide what the champion is.
        # The gate reads the incumbent from its own durable state instead
        # (run_gate.py::_incumbent_speedup), where only an audited,
        # controller-measured, correct-and-clean run can write it.  Reporting
        # the absence honestly is therefore the complete and correct behaviour;
        # a "win" prediction citing this record is still validated, against a
        # number this worker cannot influence.
        metrics["incumbent_speedup_source"] = "unavailable"
        notes.append("incumbent_speedup absent (the controller's diagnostic "
                     "request does not carry one, by design); the gate takes "
                     "the incumbent from its own durable champion state, not "
                     "from this artifact")

    after = capture_machine_state("after", overrides=parsed["tool_paths"],
                                  search_paths=parsed["tool_search_paths"])
    # SUPPORTING machine state.  The artifact's machine_state_sha256 is the
    # controller's own capture -- it owns that field, writes the document into
    # the evidence directory itself, and refuses any artifact that contradicts
    # it.  This document is what the controller cannot see from outside the
    # jail, kept as one more hashed raw artifact under a name of its own.
    worker_machine_state = {
        "schema_version": 1,
        "worker_version": WORKER_VERSION,
        "authority": (
            "worker-observed supporting evidence only; the artifact's "
            "machine_state_sha256 is the controller's own capture, stored "
            f"beside this file as {CONTROLLER_MACHINE_STATE_NAME}"),
        "controller_machine_state_sha256": parsed["machine_state_sha256"],
        "profile_record_id": parsed["profile_record_id"],
        "request_id": parsed["request_id"],
        "campaign_id": parsed["campaign_id"],
        "shape_id": parsed["shape_id"],
        "route": parsed["route"],
        "question": parsed["question"],
        "tool": tool,
        "tool_version": tool_version,
        "before": before,
        "after": after,
    }
    worker_state_path = evidence.add_bytes(
        WORKER_MACHINE_STATE_NAME, canonical_json(worker_machine_state) + b"\n")

    metrics = _sanitize_metrics(metrics)
    metrics.setdefault("worker_notes", notes)
    metrics.setdefault("catalog_source", catalog_source)
    metrics["machine_state_authority"] = "trusted-controller"
    metrics["worker_machine_state_sha256"] = sha256_file(worker_state_path)
    if catalog_source != "mounted":
        degradations.append(
            "mechanism_catalog.json was not mounted (the controller does not "
            "mount it); bottleneck terms came from the controller's pinned "
            "required_metrics, and this worker's embedded table was used only "
            "as a cross-check")

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "profile_record_id": parsed["profile_record_id"],
        "request_id": parsed["request_id"],
        "campaign_id": parsed["campaign_id"],
        # The request's "shape" is the record; the artifact's is the id.
        "shape": parsed["shape_id"],
        "target_sha256": parsed["target_sha256"],
        "tool": tool,
        "tool_version": tool_version,
        "created_epoch": time.time(),
        "machine_state_sha256": parsed["machine_state_sha256"],
        "route": parsed["route"],
        "metrics": metrics,
        "supported_bottlenecks": list(parsed["supported_bottlenecks"]),
        "raw_artifacts": evidence.raw_artifacts(),
        "gate_request_sha256": parsed["gate_request_sha256"],
    }
    gate_equivalent_check(artifact, parsed, parsed["evidence_root"],
                          evidence.local_lookup())

    placement = planned_placement(parsed)
    artifact_bytes = canonical_json(artifact) + b"\n"
    write_exclusive(output_dir / placement["output_relpath"], artifact_bytes)
    return {
        "schema_version": 1,
        "worker_version": WORKER_VERSION,
        "status": "ok",
        "error": None,
        "profile_record_id": parsed["profile_record_id"],
        "request_id": parsed["request_id"],
        "campaign_id": parsed["campaign_id"],
        "shape": parsed["shape_id"],
        "tool": tool,
        "tool_version": tool_version,
        "supported_bottlenecks": list(parsed["supported_bottlenecks"]),
        "artifact": {
            "output_relpath": placement["output_relpath"],
            "repo_relpath": placement["repo_relpath"],
            "sha256": sha256_bytes(artifact_bytes),
            "bytes": len(artifact_bytes),
        },
        "raw_placements": evidence.placements,
        "evidence_root": parsed["evidence_root"],
        "diagnostic_profile_sha256": sha256_bytes(artifact_bytes),
        "machine_state": {
            "authoritative_sha256": parsed["machine_state_sha256"],
            "authority": "trusted-controller",
            "controller_document_name": CONTROLLER_MACHINE_STATE_NAME,
            "worker_document_name": WORKER_MACHINE_STATE_NAME,
            "worker_document_sha256": sha256_file(worker_state_path),
        },
        "catalog_source": catalog_source,
        "required_metrics": parsed["required_metrics"],
        "degradations": degradations,
        "notes": notes,
        "elapsed_seconds": time.time() - started,
    }


# --------------------------------------------------------------------------
# self-test (CPU only, no CUDA tooling required)
# --------------------------------------------------------------------------

NSYS_API_SUM_FIXTURE = """Processing [report.sqlite] with [.../cuda_api_sum.py]...

 Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
     61.4,       12500000,      820,  15243.9,  14900.0,  9100,  91000,   4021.7,cudaLaunchKernel
     21.3,        4300000,       20, 215000.0, 214000.0,190000,260000,  18000.0,cudaDeviceSynchronize
     12.1,        2450000,       20, 122500.0, 121000.0,110000,140000,   9000.0,cudaGraphLaunch
      5.2,        1050000,      100,  10500.0,  10400.0,  9000, 14000,    900.0,cudaMemcpyAsync
"""

NSYS_KERN_SUM_FIXTURE = """ Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
     55.0,        5500000,      400,  13750.0,  13700.0, 12000, 16000,    500.0,fused_qkv_kernel
     45.0,        4500000,      440,  10227.3,  10200.0,  9000, 12000,    400.0,softmax_kernel
"""

NSYS_GPU_TRACE_FIXTURE = """ Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MBps),SrcMemKd,DstMemKd,Device,Ctx,GreenCtx,Strm,Name
   1000,   1000,1,1,1,1,128,1,1,32,0.000,0.000,,,,,NVIDIA GeForce RTX 3060 Ti (0),1,,7,kernel_a
   3000,   1000,2,1,1,1,128,1,1,32,0.000,0.000,,,,,NVIDIA GeForce RTX 3060 Ti (0),1,,7,kernel_b
   3500,    500,3,1,1,1,128,1,1,32,0.000,0.000,,,,,NVIDIA GeForce RTX 3060 Ti (0),1,,8,kernel_c
   6000,   1000,4,1,1,1,128,1,1,32,0.000,0.000,,,,,NVIDIA GeForce RTX 3060 Ti (0),1,,7,kernel_d
"""

NCU_LONG_FIXTURE = '''"ID","Process ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1234","fused_kernel","Command line profiler metrics","dram__bytes.sum","byte","1,048,576"
"0","1234","fused_kernel","Command line profiler metrics","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","42.5"
"0","1234","fused_kernel","Command line profiler metrics","sm__warps_active.avg.pct_of_peak_sustained_active","%","31.0"
"0","1234","fused_kernel","Command line profiler metrics","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active","%","12.0"
"0","1234","fused_kernel","Command line profiler metrics","gpu__time_duration.sum","nsecond","1000"
"1","1234","softmax_kernel","Command line profiler metrics","dram__bytes.sum","byte","524,288"
"1","1234","softmax_kernel","Command line profiler metrics","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","10.5"
"1","1234","softmax_kernel","Command line profiler metrics","sm__warps_active.avg.pct_of_peak_sustained_active","%","55.0"
"1","1234","softmax_kernel","Command line profiler metrics","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active","%","0.0"
"1","1234","softmax_kernel","Command line profiler metrics","gpu__time_duration.sum","nsecond","3000"
'''

NCU_PERMISSION_FIXTURE = """==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
"""

SANITIZER_MEMCHECK_FIXTURE = """========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 4 bytes
=========     at 0x150 in fused_kernel
========= ERROR SUMMARY: 2 errors
"""

SANITIZER_CLEAN_FIXTURE = """========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
"""

SANITIZER_RACECHECK_FIXTURE = """========= COMPUTE-SANITIZER
========= RACECHECK SUMMARY: 3 hazards displayed (1 error, 2 warnings)
"""

TORCH_TRACE_FIXTURE = {
    "traceEvents": [
        {"ph": "X", "cat": "cpu_op", "name": "aten::linear", "ts": 100.0, "dur": 50.0},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 110.0, "dur": 5.0},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 130.0, "dur": 5.0},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": 150.0, "dur": 5.0},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "ts": 160.0, "dur": 5.0},
        {"ph": "X", "cat": "kernel", "name": "sgemm", "ts": 200.0, "dur": 10.0},
        {"ph": "X", "cat": "kernel", "name": "softmax", "ts": 215.0, "dur": 10.0},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoD", "ts": 220.0, "dur": 5.0},
        {"ph": "X", "cat": "kernel", "name": "gelu", "ts": 240.0, "dur": 10.0},
    ]
}


def _raises(callable_object: Callable[[], Any], make_dir: Path | None = None) -> bool:
    if make_dir is not None:
        Path(make_dir).mkdir(parents=True, exist_ok=True)
    try:
        callable_object()
    except ProfileWorkerError:
        return True
    except Exception:  # noqa: BLE001 - any refusal beats a fabricated success
        return True
    return False


def _independent_gate_audit(artifact: dict[str, Any], parsed: dict[str, Any],
                            lookup: dict[str, Path]) -> list[str]:
    """A second, deliberately naive reading of the two consumers' rules.

    Written from run_gate._reconcile_authority_diagnostic and
    trusted_controller._finalize_profile_artifact, from the flat request only,
    sharing no code with :func:`gate_equivalent_check`.
    """
    problems: list[str] = []
    required = {
        "schema_version", "profile_record_id", "request_id", "campaign_id",
        "shape", "target_sha256", "tool", "tool_version", "created_epoch",
        "machine_state_sha256", "route", "metrics", "supported_bottlenecks",
        "raw_artifacts", "gate_request_sha256",
    }
    if set(artifact) != required or artifact.get("schema_version") != 1:
        problems.append("unknown/missing field")
    bindings = {
        "profile_record_id": parsed["profile_record_id"],
        "request_id": parsed["request_id"],
        "campaign_id": parsed["campaign_id"],
        "shape": parsed["shape_id"],
        "target_sha256": parsed["target_sha256"],
        "tool": parsed["tool"], "route": parsed["route"],
        "supported_bottlenecks": parsed["supported_bottlenecks"],
        "gate_request_sha256": parsed["gate_request_sha256"],
        "machine_state_sha256": parsed["machine_state_sha256"],
    }
    if any(artifact.get(key) != value for key, value in bindings.items()):
        problems.append("binding disagreement")
    if (not isinstance(artifact["metrics"], dict) or not artifact["metrics"]
            or not isinstance(artifact["raw_artifacts"], list)
            or not isinstance(artifact["created_epoch"], (int, float))
            or isinstance(artifact["created_epoch"], bool)
            or not math.isfinite(float(artifact["created_epoch"]))
            or not HEX64.fullmatch(str(artifact["machine_state_sha256"]))):
        problems.append("malformed metrics/raw/machine state")
    for raw in artifact["raw_artifacts"]:
        if (not isinstance(raw, dict) or set(raw) != {"path", "sha256"}
                or not isinstance(raw.get("path"), str)
                or not HEX64.fullmatch(str(raw.get("sha256", "")))):
            problems.append("malformed raw reference")
            continue
        if not raw["path"].startswith(DEFAULT_EVIDENCE_ROOT + "/"):
            problems.append(f"raw escaped namespace: {raw['path']}")
        if Path(raw["path"]).name == CONTROLLER_MACHINE_STATE_NAME:
            problems.append("worker claimed the controller's reserved name")
        local = lookup.get(raw["path"])
        if (local is None or not Path(local).is_file()
                or sha256_file(Path(local)) != raw["sha256"]):
            problems.append(f"raw missing or changed: {raw['path']}")
    for bottleneck, metrics in parsed["required_metrics"].items():
        if bottleneck not in artifact["supported_bottlenecks"]:
            problems.append(f"undeclared bottleneck {bottleneck}")
        if any(metric not in artifact["metrics"] for metric in metrics):
            problems.append(f"insufficient evidence for {bottleneck}")
    return problems


def _selftest() -> int:  # noqa: PLR0915 - a linear test script reads better flat
    import tempfile

    failures: list[str] = []
    checks = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        if condition:
            print(f"  ok   {label}")
        else:
            print(f"  FAIL {label} {detail}")
            failures.append(label)

    print("interval merge")
    busy, span, count = merge_intervals([(0, 10), (5, 15), (20, 25)])
    check("union of overlapping intervals", (busy, span, count) == (20.0, 25.0, 3),
          f"got {(busy, span, count)}")
    check("idle fraction from busy/span", abs(idle_fraction(20.0, 25.0) - 0.2) < 1e-12)
    check("empty timeline refuses", _raises(lambda: idle_fraction(0.0, 0.0)))

    print("nsys parsers")
    api = parse_nsys_api_sum(NSYS_API_SUM_FIXTURE)
    check("cudaLaunchKernel count", api["kernel_launch_api_calls"] == 820, str(api))
    check("cudaGraphLaunch count", api["graph_launch_api_calls"] == 20, str(api))
    check("non-launch APIs excluded",
          api["cuda_api_calls_by_name"]["cudaDeviceSynchronize"] == 20)
    kern = parse_nsys_kern_sum(NSYS_KERN_SUM_FIXTURE)
    check("kernel instances summed", kern["kernel_instances"] == 840, str(kern))
    trace = parse_nsys_gpu_trace(NSYS_GPU_TRACE_FIXTURE)
    # busy = [1000,2000) + [3000,4000) + [6000,7000) = 3000; span = 6000
    check("gpu busy is the union, not the sum", trace["gpu_busy_ns"] == 3000.0, str(trace))
    check("gpu span", trace["gpu_span_ns"] == 6000.0, str(trace))
    check("gpu idle fraction", abs(trace["gpu_idle_fraction"] - 0.5) < 1e-12, str(trace))
    check("a renamed column is fatal, not zero",
          _raises(lambda: parse_nsys_api_sum(
              " Time (%),Total Time (ns),Calls,Name\n 1,1,1,cudaLaunchKernel\n")))
    check("no launch rows is fatal",
          _raises(lambda: parse_nsys_api_sum(
              " Num Calls,Name\n 5,cudaDeviceSynchronize\n")))

    print("torch profiler trace parser")
    torch_metrics = parse_torch_chrome_trace(TORCH_TRACE_FIXTURE)
    check("launch APIs counted", torch_metrics["kernel_launch_api_calls"] == 2,
          str(torch_metrics))
    check("graph launches counted", torch_metrics["graph_launch_api_calls"] == 1)
    check("sync APIs not counted as launches",
          "cudaStreamSynchronize" not in torch_metrics["cuda_api_calls_by_name"])
    # device busy: [200,210) [215,225) [220,225) [240,250) -> 10+10+10 = 30, span 50
    check("device busy union", torch_metrics["gpu_busy_us"] == 30.0, str(torch_metrics))
    check("device span", torch_metrics["gpu_span_us"] == 50.0)
    check("idle fraction", abs(torch_metrics["gpu_idle_fraction"] - 0.4) < 1e-12)
    check("a trace with no launches is fatal",
          _raises(lambda: parse_torch_chrome_trace({"traceEvents": [
              {"ph": "X", "cat": "kernel", "name": "k", "ts": 1.0, "dur": 1.0}]})))

    print("ncu parser")
    counters = parse_ncu_csv(NCU_LONG_FIXTURE)
    check("dram bytes per kernel", counters["dram__bytes.sum"] == [1048576.0, 524288.0],
          str(counters.get("dram__bytes.sum")))
    durations = counters["gpu__time_duration.sum"]
    weighted = weighted_mean(
        counters["sm__throughput.avg.pct_of_peak_sustained_elapsed"], durations)
    # (42.5*1000 + 10.5*3000) / 4000 = 18.5
    check("duration-weighted throughput", abs(weighted - 18.5) < 1e-9, str(weighted))
    check("thousands separators parsed", sum(counters["dram__bytes.sum"]) == 1572864.0)
    check("ERR_NVGPUCTRPERM detected", ncu_permission_denied(NCU_PERMISSION_FIXTURE))
    check("clean output not flagged as permission denied",
          not ncu_permission_denied(NCU_LONG_FIXTURE))
    wide = ('"Kernel Name","dram__bytes.sum","gpu__time_duration.sum"\n'
            '"k1","2048","100"\n"k2","4096","300"\n')
    wide_counters = parse_ncu_csv(wide)
    check("wide raw-page form parsed",
          wide_counters["dram__bytes.sum"] == [2048.0, 4096.0], str(wide_counters))

    print("compute-sanitizer parser")
    memcheck = parse_compute_sanitizer(SANITIZER_MEMCHECK_FIXTURE)
    check("memcheck error count", memcheck["sanitizer_errors"] == 2, str(memcheck))
    check("memcheck not clean", memcheck["sanitizer_clean"] is False)
    clean = parse_compute_sanitizer(SANITIZER_CLEAN_FIXTURE)
    check("clean run reported clean", clean["sanitizer_clean"] is True)
    race = parse_compute_sanitizer(SANITIZER_RACECHECK_FIXTURE)
    check("racecheck hazards", race["sanitizer_hazards_displayed"] == 3, str(race))
    check("racecheck errors", race["sanitizer_race_errors"] == 1, str(race))
    check("no summary is fatal", _raises(lambda: parse_compute_sanitizer("nothing here")))

    print("static route model")
    shape = {"id": 1, "batch_size": 64, "seq_len": 128, "d_model": 128,
             "num_heads": 4, "ffn_dim": 128, "num_layers": 4, "causal": True}
    model = static_route_model(shape, "float32")
    B, S, D, H, F, L = 64, 128, 128, 4, 128, 4
    check("score matrix bytes", model["materialized_bytes"] == B * H * S * S * 4,
          str(model["materialized_bytes"]))
    expected_flops = L * (4 * 2 * B * S * D * D + 2 * 2 * B * H * S * S * (D // H)
                          + 2 * 2 * B * S * D * F)
    check("matmul flops", model["matmul_flops"] == expected_flops,
          f"{model['matmul_flops']} != {expected_flops}")
    check("traffic exceeds parameter bytes", model["dram_bytes"] > model["parameter_bytes"])
    check("every traffic term is positive",
          all(term["bytes"] > 0 for term in model["traffic_terms_per_layer"]))
    check("causal shape gets the causal mask term",
          any(term["term"] == "causal_masked_fill_rw"
              for term in model["traffic_terms_per_layer"]))
    noncausal = static_route_model({**shape, "causal": False}, "float32")
    check("non-causal shape drops it",
          not any(term["term"] == "causal_masked_fill_rw"
                  for term in noncausal["traffic_terms_per_layer"]))
    check("fp16 halves the score matrix",
          static_route_model(shape, "float16")["materialized_bytes"]
          == model["materialized_bytes"] // 2)

    print("end-to-end artifact against a REAL controller-shaped request")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "target.py"
        target.write_text("# stand-in target bytes for the CPU self-test\n")
        alias = root / "candidate.py"
        alias.write_text(target.read_text())
        official = root / "official.py"
        official.write_text("# not imported by static-analysis\n")
        shapes = root / "shapes.json"
        shapes.write_text(json.dumps({"shapes": [shape]}))
        record_id = "profile-" + "ab" * 6
        bottlenecks = ["quadratic-materialization", "global-memory-traffic",
                       "memory-capacity"]
        # Exactly the flat document trusted_controller._profile_worker_request
        # builds.  The key set is asserted against CONTROLLER_REQUEST_KEYS
        # below, and integration_authority_test.py drives the REAL controller
        # function into this same worker so a fixture that drifts is caught
        # there rather than believed here.
        request_document = {
            "schema_version": 1,
            "operation": "diagnostic",
            "request_id": "req-" + "cd" * 8,
            "gate_request_sha256": "b" * 64,
            "profile_record_id": record_id,
            "campaign_id": "CAMP-SELFTEST",
            "shape_id": shape["id"],
            "shape": shape,
            "target_sha256": sha256_file(target),
            "target_path": str(target),
            "target_path_alias": str(alias),
            "official_path": str(official),
            "shapes_path": str(shapes),
            "official_sha256": sha256_file(official),
            "shapes_sha256": sha256_file(shapes),
            "output_dir": "/output",
            "artifact_filename": f"{record_id}.json",
            "artifact_fields": sorted(ARTIFACT_REQUIRED_KEYS),
            "tool": "static-analysis",
            "route": "dispatcher-shape-1",
            "question": "Does the dense route materialise an O(S^2) intermediate "
                        "large enough to bound this shape?",
            "supported_bottlenecks": bottlenecks,
            "required_metrics": {name: list(CATALOG_FALLBACK[name]["required_metrics"])
                                 for name in bottlenecks},
            "machine_state_sha256": "c" * 64,
            "tool_search_paths": ["/usr/local/cuda/bin", "/usr/bin"],
            "timeout_seconds": 10800,
            "is_performance_measurement": False,
            "notes": "Read-only diagnosis.",
        }
        check("the self-test fixture is the controller's exact key set",
              set(request_document) == set(CONTROLLER_REQUEST_KEYS),
              f"fixture_only={sorted(set(request_document) - CONTROLLER_REQUEST_KEYS)} "
              f"controller_only={sorted(CONTROLLER_REQUEST_KEYS - set(request_document))}")
        request_path = root / "request.json"
        request_path.write_text(json.dumps(request_document))
        output_dir = root / "out"
        output_dir.mkdir()
        parsed = parse_worker_request(request_document)
        check("the flat request parses", parsed["shape_id"] == shape["id"])
        check("the parsed shape record came from the request",
              parsed["shape"] == shape)
        manifest = run_profile(request_document, output_dir, request_path)
        check("worker reports ok", manifest["status"] == "ok", str(manifest.get("error")))
        artifact_file = output_dir / manifest["artifact"]["output_relpath"]
        check("artifact written under the controller's artifact_filename",
              artifact_file.is_file()
              and artifact_file.name == request_document["artifact_filename"])
        check("artifact sha matches the manifest",
              sha256_file(artifact_file) == manifest["artifact"]["sha256"])
        check("manifest exposes diagnostic_profile_sha256 for the controller",
              manifest["diagnostic_profile_sha256"] == manifest["artifact"]["sha256"])
        check("repo destination is inside the trusted evidence namespace",
              manifest["artifact"]["repo_relpath"]
              == f"{DEFAULT_EVIDENCE_ROOT}/{record_id}.json")
        artifact = json.loads(artifact_file.read_text())
        check("exact required key set", set(artifact) == set(ARTIFACT_REQUIRED_KEYS),
              str(sorted(set(artifact) ^ set(ARTIFACT_REQUIRED_KEYS))))
        expected_bindings = {
            "profile_record_id": record_id,
            "request_id": request_document["request_id"],
            "campaign_id": request_document["campaign_id"],
            "shape": request_document["shape_id"],
            "target_sha256": request_document["target_sha256"],
            "tool": request_document["tool"],
            "route": request_document["route"],
            "supported_bottlenecks": request_document["supported_bottlenecks"],
            "gate_request_sha256": request_document["gate_request_sha256"],
            "machine_state_sha256": request_document["machine_state_sha256"],
        }
        check("every authority binding is echoed",
              all(artifact[key] == expected_bindings[key] for key in GATE_BOUND_KEYS),
              str({key: artifact[key] for key in GATE_BOUND_KEYS
                   if artifact[key] != expected_bindings[key]}))
        check("the artifact's shape is the integer id, not the shape record",
              artifact["shape"] == shape["id"])
        check("machine_state_sha256 is the CONTROLLER's, not the worker's",
              artifact["machine_state_sha256"]
              == request_document["machine_state_sha256"])
        worker_state = [item for item in manifest["raw_placements"]
                        if item["output_relpath"].endswith(WORKER_MACHINE_STATE_NAME)]
        check("the worker's own machine state ships as a separate raw artifact",
              len(worker_state) == 1)
        check("and is referenced from metrics, never from the artifact hash",
              bool(worker_state)
              and artifact["metrics"]["worker_machine_state_sha256"]
              == worker_state[0]["sha256"]
              and artifact["metrics"]["worker_machine_state_sha256"]
              != artifact["machine_state_sha256"])
        check("metrics name the controller as the machine-state authority",
              artifact["metrics"]["machine_state_authority"] == "trusted-controller")
        check("no raw artifact uses the controller's reserved name",
              all(Path(entry["path"]).name != CONTROLLER_MACHINE_STATE_NAME
                  for entry in artifact["raw_artifacts"]))
        check("raw artifacts predict the controller's ingestion destination",
              all(entry["path"].startswith(
                  f"{DEFAULT_EVIDENCE_ROOT}/{record_id}.raw/{RAW_DIRNAME}/")
                  for entry in artifact["raw_artifacts"]))
        check("required metrics present for every bottleneck the request named",
              all(metric in artifact["metrics"]
                  for metrics in request_document["required_metrics"].values()
                  for metric in metrics))
        # The worker must never volunteer a champion number.  The gate reads
        # the incumbent from its own durable state, and an artifact that
        # asserted one here would be a jailed process describing the board.
        check("incumbent_speedup is honestly reported as unavailable",
              artifact["metrics"]["incumbent_speedup_source"] == "unavailable"
              and "incumbent_speedup" not in artifact["metrics"])
        check("raw artifact entries hold exactly path+sha256",
              all(set(entry) == {"path", "sha256"} for entry in artifact["raw_artifacts"]))
        check("created_epoch is fresh",
              0 <= time.time() - artifact["created_epoch"] < 3600)

        # Independent re-check with a from-scratch reading of the two consumers.
        lookup = {item["repo_relpath"]: output_dir / item["output_relpath"]
                  for item in manifest["raw_placements"]}
        problems = _independent_gate_audit(artifact, parsed, lookup)
        check("independent gate audit passes", not problems, "; ".join(problems))

        # Negative cases: the worker must refuse rather than emit junk.
        old_shape = json.loads(json.dumps(request_document))
        old_shape["gate_request"] = {"request_kind": "diagnostic",
                                     "mode": "diagnostic"}
        for key in ("shape_id", "artifact_filename", "required_metrics"):
            del old_shape[key]
        check("the pre-fix nested gate_request shape is refused",
              _raises(lambda: parse_worker_request(old_shape)))
        for key in sorted(CONTROLLER_REQUEST_KEYS - {"schema_version", "operation"}):
            dropped = json.loads(json.dumps(request_document))
            del dropped[key]
            if not _raises(lambda: parse_worker_request(dropped)):
                check(f"missing controller key {key} is refused", False)
                break
        else:
            check("every missing controller key is refused", True)
        bad_tool = json.loads(json.dumps(request_document))
        bad_tool["tool"] = "ncu"
        check("a tool that cannot produce the required metrics is refused",
              _raises(lambda: run_profile(bad_tool, root / "out2", request_path),
                      make_dir=root / "out2"))
        no_collector = json.loads(json.dumps(request_document))
        no_collector["tool"] = "tea-leaves"
        check("a tool with no collector is refused",
              _raises(lambda: run_profile(no_collector, root / "out3", request_path),
                      make_dir=root / "out3"))
        bad_sha = json.loads(json.dumps(request_document))
        bad_sha["target_sha256"] = "0" * 64
        check("target byte mismatch is refused",
              _raises(lambda: run_profile(bad_sha, root / "out4", request_path),
                      make_dir=root / "out4"))
        bad_official = json.loads(json.dumps(request_document))
        bad_official["official_sha256"] = "0" * 64
        check("official-script byte mismatch is refused",
              _raises(lambda: run_profile(bad_official, root / "out5", request_path),
                      make_dir=root / "out5"))
        bad_shapes = json.loads(json.dumps(request_document))
        bad_shapes["shapes_sha256"] = "0" * 64
        check("shapes.json byte mismatch is refused",
              _raises(lambda: run_profile(bad_shapes, root / "out6", request_path),
                      make_dir=root / "out6"))
        wrong_shape = json.loads(json.dumps(request_document))
        wrong_shape["shape"] = {**shape, "id": shape["id"] + 1}
        check("shape record that is not shape_id's record is refused",
              _raises(lambda: parse_worker_request(wrong_shape)))
        drifted_fields = json.loads(json.dumps(request_document))
        drifted_fields["artifact_fields"] = sorted(
            set(ARTIFACT_REQUIRED_KEYS) | {"invented_field"})
        check("artifact_fields drift between controller and worker is refused",
              _raises(lambda: parse_worker_request(drifted_fields)))
        perf = json.loads(json.dumps(request_document))
        perf["is_performance_measurement"] = True
        check("a request claiming to be a performance measurement is refused",
              _raises(lambda: parse_worker_request(perf)))
        mismatched_metrics = json.loads(json.dumps(request_document))
        mismatched_metrics["required_metrics"] = {"launch-overhead": ["kernel_launches"]}
        check("required_metrics that do not match the bottlenecks is refused",
              _raises(lambda: parse_worker_request(mismatched_metrics)))
        tampered = json.loads(json.dumps(artifact))
        tampered["metrics"]["dram_bytes"] = 1.0
        tampered["raw_artifacts"][0]["sha256"] = "f" * 64
        check("tampered raw hash fails the gate-equivalent check",
              _raises(lambda: gate_equivalent_check(
                  tampered, parsed, DEFAULT_EVIDENCE_ROOT, lookup)))
        extra_key = json.loads(json.dumps(artifact))
        extra_key["notes"] = "hello"
        check("an extra artifact key is fatal here, not at the gate",
              _raises(lambda: gate_equivalent_check(
                  extra_key, parsed, DEFAULT_EVIDENCE_ROOT, lookup)))
        stale = json.loads(json.dumps(artifact))
        stale["created_epoch"] = time.time() - 25 * 3600
        check("evidence outside the controller's time window is refused",
              _raises(lambda: gate_equivalent_check(
                  stale, parsed, DEFAULT_EVIDENCE_ROOT, lookup)))
        minted = json.loads(json.dumps(artifact))
        minted["machine_state_sha256"] = "d" * 64
        check("a worker-minted machine_state_sha256 is refused",
              _raises(lambda: gate_equivalent_check(
                  minted, parsed, DEFAULT_EVIDENCE_ROOT, lookup)))
        missing_metric = json.loads(json.dumps(artifact))
        del missing_metric["metrics"]["materialized_bytes"]
        check("a missing required metric is refused",
              _raises(lambda: gate_equivalent_check(
                  missing_metric, parsed, DEFAULT_EVIDENCE_ROOT, lookup)))

    print("request validation")
    check("a non-object request is refused",
          _raises(lambda: parse_worker_request(["not", "an", "object"])))
    check("a non-diagnostic operation is refused",
          _raises(lambda: parse_worker_request(
              {"schema_version": 1, "operation": "candidate"})))
    check("an empty request is refused",
          _raises(lambda: parse_worker_request({"schema_version": 1,
                                                "operation": "diagnostic"})))
    check("iterations out of bounds refused",
          _raises(lambda: bounded_int(0, "iterations", 1, 1000, 20)))
    check("planned_placement follows the controller's artifact_filename",
          planned_placement({"artifact_filename": "profile-x.json",
                             "evidence_root": DEFAULT_EVIDENCE_ROOT})
          == {"output_relpath": "profile-x.json",
              "repo_relpath": "Project/loop/profile_evidence/profile-x.json"})
    check("every collector has a declared metric capability",
          set(COLLECTORS) == set(COLLECTOR_METRICS),
          str(sorted(set(COLLECTORS) ^ set(COLLECTOR_METRICS))))
    check("every catalog metric this worker claims to support is reachable",
          all(any(metric in capable for capable in COLLECTOR_METRICS.values())
              for entry in CATALOG_FALLBACK.values()
              for metric in entry["required_metrics"]))
    print()
    print(f"{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("profile_worker self-test OK (CPU only; GPU paths are untested here)")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="profile_worker.py",
        description="Produce the diagnostic profile artifact the gate requires.")
    parser.add_argument("--request", help="worker request JSON (controller-authored)")
    parser.add_argument("--output", help="writable output directory")
    parser.add_argument("--summary-out", help="[--execute-route] route summary path")
    parser.add_argument("--execute-route", action="store_true",
                        help="internal: run the route once under an external "
                             "profiler (nsys / ncu / compute-sanitizer wrap this)")
    parser.add_argument("--selftest", action="store_true",
                        help="run the CPU-only parser and artifact self-test")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.selftest:
        return _selftest()
    if not args.request or not args.output:
        parser.error("--request and --output are required")

    request_path = Path(args.request).resolve()
    output_dir = Path(args.output).resolve()
    document = read_object(request_path, "worker request")

    if args.execute_route:
        parsed = parse_worker_request(document)
        summary_out = (Path(args.summary_out) if args.summary_out
                       else output_dir / RAW_DIRNAME / "route-summary.json")
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary = execute_route_child(parsed, summary_out)
        print(json.dumps({"status": "route-executed",
                          "iterations": summary["iterations"]}))
        return 0

    manifest_path = output_dir / MANIFEST_NAME
    try:
        manifest = run_profile(document, output_dir, request_path)
    except ProfileWorkerError as exc:
        failure = {
            "schema_version": 1,
            "worker_version": WORKER_VERSION,
            "status": "refused",
            "error": str(exc),
            "error_class": type(exc).__name__,
            "artifact": None,
            "raw_placements": [],
            "refused_epoch": time.time(),
        }
        try:
            write_exclusive(manifest_path, canonical_json(failure) + b"\n")
        except OSError:
            pass
        print(f"PROFILE_WORKER_REFUSED: {exc}", file=sys.stderr)
        # 3 == the tool itself is unavailable (a machine fact the controller
        # should report, not retry); 2 == the request or the evidence is bad.
        return 3 if isinstance(exc, ToolUnavailable) else 2
    write_exclusive(manifest_path, canonical_json(manifest) + b"\n")
    print(json.dumps({
        "status": "ok",
        "profile_record_id": manifest["profile_record_id"],
        "tool": manifest["tool"],
        "artifact_sha256": manifest["artifact"]["sha256"],
        "raw_artifacts": len(manifest["raw_placements"]),
        "degradations": manifest["degradations"],
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileWorkerError as exc:
        print(f"PROFILE_WORKER_REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
