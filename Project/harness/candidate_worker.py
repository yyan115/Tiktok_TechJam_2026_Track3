#!/usr/bin/env python3
"""Untrusted-namespace GPU evaluator.

The trusted controller launches this file through :mod:`sandbox`; it never has
the authority journal mounted.  This worker deliberately writes only a neutral
response and raw float32 challenge outputs.  Promotion, verdicts, evidence
selection, and durable state are controller responsibilities.

Candidate Python still shares this worker interpreter, so all timing emitted by
this file is *supporting* evidence.  The controller independently recomputes
reference outputs and requires an external clean-timing corroboration before a
result can be eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import secrets
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REQUEST_KEYS = {
    "schema_version",
    "request_id",
    "operation",
    "shape_id",
    "shape",
    "dtype",
    "seeds",
    "timing_args",
    "numerical",
    "candidate_sha256",
    "official_sha256",
    "shapes_sha256",
    "challenge_nonce",
}
OPERATIONS = {"candidate", "calibration", "diagnostic"}
NUMERICAL_REQUIRED = {
    "padding_ratio": 0.0,
    "input_scale": 1.0,
    "rtol": 0.02,
    "atol": 0.002,
    "matmul_precision": "high",
    "allow_tf32": True,
}


class WorkerError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise WorkerError(f"invalid {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerError(f"{name} must be an object")
    return value


def validate_request(
    request: dict[str, Any],
    *,
    candidate_path: Path,
    official_path: Path,
    shapes_path: Path,
) -> None:
    if set(request) != REQUEST_KEYS:
        raise WorkerError("request has unexpected or missing fields")
    if request["schema_version"] != SCHEMA_VERSION:
        raise WorkerError("unsupported request schema")
    if request["operation"] not in OPERATIONS:
        raise WorkerError("unsupported operation")
    for field in ("request_id", "challenge_nonce"):
        if not isinstance(request[field], str) or len(request[field]) < 16:
            raise WorkerError(f"{field} must be an unpredictable non-empty string")
    shape_id = request["shape_id"]
    if isinstance(shape_id, bool) or not isinstance(shape_id, int) or not 1 <= shape_id <= 14:
        raise WorkerError("shape_id must be 1..14")
    shapes_doc = read_object(shapes_path, "shapes file")
    official_shape = next(
        (item for item in shapes_doc.get("shapes", []) if item.get("id") == shape_id),
        None,
    )
    if official_shape is None or request["shape"] != official_shape:
        raise WorkerError("request shape does not exactly match shapes.json")
    if shape_id in {6, 14}:
        raise WorkerError("shapes 6 and 14 require their dedicated controller paths")
    if request["dtype"] not in {"float32", "float16", "bfloat16"}:
        raise WorkerError("unsupported dtype")
    seeds = request["seeds"]
    if (
        not isinstance(seeds, list)
        or not 5 <= len(seeds) <= 16
        or len(set(seeds)) != len(seeds)
        or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
    ):
        raise WorkerError("seeds must contain 5..16 unique integers")
    timing = request["timing_args"]
    if not isinstance(timing, dict) or set(timing) != {"warmup", "repeats", "rounds"}:
        raise WorkerError("timing_args schema mismatch")
    bounds = {"warmup": (1, 100), "repeats": (1, 2000), "rounds": (1, 20)}
    for field, (lower, upper) in bounds.items():
        value = timing[field]
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise WorkerError(f"timing_args.{field} outside safe bounds")
    numerical = request["numerical"]
    if numerical != NUMERICAL_REQUIRED:
        raise WorkerError("numerical settings must exactly match the official profile")
    if request["official_sha256"] != sha256_file(official_path):
        raise WorkerError("official benchmark SHA mismatch")
    if request["shapes_sha256"] != sha256_file(shapes_path):
        raise WorkerError("shapes SHA mismatch")
    candidate_sha = request["candidate_sha256"]
    if request["operation"] == "calibration":
        if candidate_sha is not None:
            raise WorkerError("calibration request cannot carry candidate bytes")
    elif not isinstance(candidate_sha, str) or candidate_sha != sha256_file(candidate_path):
        raise WorkerError("candidate SHA mismatch")


def load_module_exact(path: Path, name: str):
    source = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    code = compile(source, str(path), "exec")
    exec(code, module.__dict__)
    return module


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("official_worker_module", path)
    if spec is None or spec.loader is None:
        raise WorkerError("cannot create official module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def accuracy_dict(result: Any) -> dict[str, Any]:
    return {
        "passed": bool(result.passed),
        "total_elements": int(result.total_elements),
        "failed_elements": int(result.failed_elements),
        "max_abs_error": float(result.max_abs_error),
        "max_relative_error": float(result.max_relative_error),
        "mean_abs_error": float(result.mean_abs_error),
    }


def timing_dict(samples: list[float]) -> dict[str, Any]:
    if not samples or not all(math.isfinite(value) and value >= 0 for value in samples):
        raise WorkerError("timing produced invalid samples")
    ordered = sorted(samples)
    p90_index = max(0, min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": ordered[p90_index],
        "min_ms": min(samples),
        "n_samples": len(samples),
        "raw_samples_ms": samples,
    }


def write_exclusive(path: Path, data: bytes) -> None:
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


def write_output_tensor(output_dir: Path, tensor: Any, seed: int) -> dict[str, Any]:
    cpu = tensor.detach().float().contiguous().cpu()
    data = cpu.numpy().tobytes(order="C")
    digest = hashlib.sha256(data).hexdigest()
    filename = f"challenge-{seed}-{secrets.token_hex(8)}.f32"
    write_exclusive(output_dir / filename, data)
    return {
        "seed": seed,
        "filename": filename,
        "sha256": digest,
        "shape": list(cpu.shape),
        "dtype": "float32",
        "nbytes": len(data),
    }


def build_candidate(
    candidate_module: Any,
    official: Any,
    config: Any,
    baseline: Any,
    baseline_cpu_state: dict[str, Any],
):
    if hasattr(candidate_module, "build"):
        candidate = candidate_module.build(official, config)
    elif hasattr(candidate_module, "UserOptimizedTransformer"):
        candidate = candidate_module.UserOptimizedTransformer(config)
    else:
        raise WorkerError("candidate must define build() or UserOptimizedTransformer")
    if hasattr(candidate_module, "copy_weights"):
        candidate_module.copy_weights(baseline, candidate)
    else:
        incompatible = candidate.load_state_dict(baseline_cpu_state, strict=True)
        if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
            raise WorkerError("strict candidate weight copy failed")
    return candidate


def wall_per_iter(torch: Any, model: Any, x: Any, mask: Any, iterations: int) -> float:
    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            model(x, mask)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
    return (end - start) / 1e6 / iterations


def evaluate(request: dict[str, Any], candidate_path: Path, official_path: Path, output_dir: Path) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise WorkerError("CUDA is unavailable inside the candidate namespace")
    official = load_official(official_path)
    trusted = {
        name: getattr(official, name)
        for name in (
            "TransformerConfig",
            "BaselineTransformer",
            "resolve_dtype",
            "generate_random_case",
            "compare_outputs",
            "warmup_model",
            "benchmark_once",
        )
    }
    shape = request["shape"]
    config = trusted["TransformerConfig"](
        batch_size=shape["batch_size"],
        seq_len=shape["seq_len"],
        d_model=shape["d_model"],
        num_heads=shape["num_heads"],
        ffn_dim=shape["ffn_dim"],
        num_layers=shape["num_layers"],
        causal=shape["causal"],
    )
    config.validate()
    device = torch.device("cuda")
    dtype = trusted["resolve_dtype"](request["dtype"])
    numerical = request["numerical"]
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision(numerical["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = numerical["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = numerical["allow_tf32"]

    baseline = trusted["BaselineTransformer"](config).to(device=device, dtype=dtype).eval()
    baseline_cpu_state = {
        key: value.detach().cpu().clone() for key, value in baseline.state_dict().items()
    }

    # This probe and its expected output exist before untrusted candidate bytes
    # execute.  A second run after candidate construction catches mutation of
    # the official module, baseline parameters, or framework math.
    probe_x, probe_mask = trusted["generate_random_case"](
        config, device, dtype, 991_827_331,
        numerical["padding_ratio"], numerical["input_scale"],
    )
    with torch.inference_mode():
        probe_reference = baseline(probe_x, probe_mask).detach().clone()

    if request["operation"] == "calibration":
        candidate = trusted["BaselineTransformer"](config)
        candidate.load_state_dict(baseline_cpu_state, strict=True)
    else:
        candidate_module = load_module_exact(candidate_path, "untrusted_candidate")
        candidate = build_candidate(
            candidate_module, official, config, baseline, baseline_cpu_state
        )
    candidate = candidate.to(device=device, dtype=dtype).eval()

    trials: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    all_passed = True
    with torch.inference_mode():
        for seed in request["seeds"]:
            x, mask = trusted["generate_random_case"](
                config, device, dtype, seed,
                numerical["padding_ratio"], numerical["input_scale"],
            )
            x_snapshot = x.clone()
            mask_snapshot = mask.clone()
            reference = baseline(x, mask)
            result_tensor = candidate(x, mask)
            input_unchanged = torch.equal(x, x_snapshot) and torch.equal(mask, mask_snapshot)
            result = trusted["compare_outputs"](
                reference,
                result_tensor,
                rtol=numerical["rtol"],
                atol=numerical["atol"],
            )
            trial = accuracy_dict(result)
            trial["seed"] = seed
            trial["input_unchanged"] = input_unchanged
            trials.append(trial)
            outputs.append(write_output_tensor(output_dir, result_tensor, seed))
            all_passed &= result.passed and input_unchanged

        probe_now = baseline(probe_x, probe_mask)
        baseline_invariant = torch.equal(probe_now, probe_reference)
        all_passed &= baseline_invariant

    timing = request["timing_args"]
    timing_x, timing_mask = trusted["generate_random_case"](
        config, device, dtype, request["seeds"][0] + 1_000_000,
        numerical["padding_ratio"], numerical["input_scale"],
    )
    trusted["warmup_model"](baseline, timing_x, timing_mask, timing["warmup"], device)
    trusted["warmup_model"](candidate, timing_x, timing_mask, timing["warmup"], device)
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    for round_index in range(timing["rounds"]):
        order = (
            ((baseline, baseline_samples), (candidate, candidate_samples))
            if round_index % 2 == 0
            else ((candidate, candidate_samples), (baseline, baseline_samples))
        )
        for model, samples in order:
            samples.extend(trusted["benchmark_once"](model, timing_x, timing_mask, timing["repeats"], device))
    baseline_stats = timing_dict(baseline_samples)
    candidate_stats = timing_dict(candidate_samples)
    event_speedup = baseline_stats["median_ms"] / candidate_stats["median_ms"]
    wall_iters = max(10, min(100, timing["repeats"] // 2))
    baseline_wall = wall_per_iter(torch, baseline, timing_x, timing_mask, wall_iters)
    candidate_wall = wall_per_iter(torch, candidate, timing_x, timing_mask, wall_iters)
    wall_speedup = baseline_wall / candidate_wall
    agreement_ratio = max(event_speedup, wall_speedup) / max(
        min(event_speedup, wall_speedup), 1e-12
    )

    # Same-address fresh values: one output check per iteration.  This is
    # intentionally untimed as a correctness challenge; clean timing remains a
    # separate measurement.
    anti_cache_passed = True
    generator = torch.Generator(device=device)
    with torch.inference_mode():
        anti_x, anti_mask = trusted["generate_random_case"](
            config, device, dtype, request["seeds"][0] + 2_000_000,
            numerical["padding_ratio"], numerical["input_scale"],
        )
        for index in range(5):
            generator.manual_seed(request["seeds"][0] + 3_000_000 + index)
            anti_x.copy_(torch.randn(anti_x.shape, generator=generator, device=device, dtype=dtype))
            reference = baseline(anti_x, anti_mask)
            result_tensor = candidate(anti_x, anti_mask)
            result = trusted["compare_outputs"](
                reference, result_tensor,
                rtol=numerical["rtol"], atol=numerical["atol"],
            )
            anti_cache_passed &= result.passed
    all_passed &= anti_cache_passed

    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "challenge_nonce": request["challenge_nonce"],
        "candidate_sha256": request["candidate_sha256"],
        "official_sha256": request["official_sha256"],
        "shapes_sha256": request["shapes_sha256"],
        "shape_id": request["shape_id"],
        "correctness": {
            "passed": all_passed,
            "baseline_invariant": baseline_invariant,
            "anti_cache_passed": anti_cache_passed,
            "trials": trials,
        },
        "challenge_outputs": outputs,
        "supporting_timing": {
            "baseline": baseline_stats,
            "candidate": candidate_stats,
            "event_speedup": event_speedup,
            "baseline_wall_ms_per_iter": baseline_wall,
            "candidate_wall_ms_per_iter": candidate_wall,
            "wall_speedup": wall_speedup,
            "event_wall_speedup_agreement_ratio": agreement_ratio,
            "suspicious": agreement_ratio > 1.25,
            "authority": "supporting-worker-measurement",
        },
        "effective_numerical_state": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--official", required=True)
    parser.add_argument("--shapes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request_path = Path(args.request)
    candidate_path = Path(args.candidate)
    official_path = Path(args.official)
    shapes_path = Path(args.shapes)
    output_dir = Path(args.output)
    request = read_object(request_path, "request")
    validate_request(
        request,
        candidate_path=candidate_path,
        official_path=official_path,
        shapes_path=shapes_path,
    )
    response = evaluate(request, candidate_path, official_path, output_dir)
    response_bytes = canonical_json(response) + b"\n"
    write_exclusive(output_dir / "response.json", response_bytes)
    print(json.dumps({"request_id": request["request_id"], "status": "completed"}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkerError as exc:
        print(f"WORKER_REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
