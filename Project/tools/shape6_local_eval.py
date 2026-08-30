#!/usr/bin/env python3
"""Fail-closed, submission-bound evidence for official shape 6.

The dense B=10000 reference does not fit on the project GPU. Batch rows do
not interact, so correctness is computed against the pinned official model in
batch chunks while the exact generated submission receives the full official
input. Input generation, masks, numerical flags, tolerances, warmups, timing
repeats, and timing rounds mirror the pinned PyTorch benchmark.

This evaluator never promotes a result. It writes one immutable packet under
``Project/results_side`` for the independent auditor and final evidence map.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import secrets
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "Project"
SIDE_RESULTS = PROJECT / "results_side"
OFFICIAL_TORCH = ROOT / "torch_transformer_benchmark.py"
MANIFEST_PATH = PROJECT / "manifest.json"
SUBMISSION_FILE = (
    PROJECT / "submission" / "torch_transformer_benchmark_submission.py"
)

ATOL, RTOL = 0.002, 0.02
SEED0 = 1234
DEFAULT_SEEDS = tuple(SEED0 + index for index in range(5))
TIMING_SEED = SEED0 + 100000
CHUNK = 500
MEMORY_WARMUPS = 3
MEMORY_REPEATS = 10
OFFICIAL_WARMUPS = 20
OFFICIAL_REPEATS = 100
OFFICIAL_ROUNDS = 3
MIB = 2**20
MEMORY_LIMITS = {
    "allocated_slope_bytes_per_repeat": 16 * MIB,
    "reserved_slope_bytes_per_repeat": 64 * MIB,
    "allocated_end_growth_bytes": 64 * MIB,
    "reserved_end_growth_bytes": 256 * MIB,
    "allocated_max_growth_bytes": 64 * MIB,
    "reserved_max_growth_bytes": 256 * MIB,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_seeds(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers")
    try:
        seeds = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--seeds must be comma-separated integers"
        ) from exc
    if len(seeds) < 5:
        raise argparse.ArgumentTypeError("shape-6 evidence requires at least 5 seeds")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("shape-6 seeds must be unique")
    return seeds


def verify_official() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    mismatches = []
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(name)
    if mismatches:
        raise SystemExit(f"INTEGRITY FAILURE: official files changed: {mismatches}")
    return {
        "official_commit": manifest["official_commit"],
        "official_sha256": sha256_file(OFFICIAL_TORCH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
    }


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def configure_official_numerics(torch, seed: int) -> dict:
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    state = {
        "dtype": "float32",
        "matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": True,
        "padding_ratio": 0.0,
        "input_scale": 1.0,
        "atol": ATOL,
        "rtol": RTOL,
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get(
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
        ),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }
    if (
        state["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] is not None
        or state["NVIDIA_TF32_OVERRIDE"] is not None
    ):
        raise SystemExit("TF32 environment overrides make the numerical state non-official")
    return state


def official_case(otb, torch, config, device, seed):
    x, mask = otb.generate_random_case(
        config=config,
        device=device,
        dtype=torch.float32,
        seed=seed,
        padding_ratio=0.0,
        input_scale=1.0,
    )
    if mask.dtype != torch.bool or mask.device != device:
        raise AssertionError("official generator returned an invalid mask")
    if not bool(mask.all()):
        raise AssertionError("padding_ratio=0 must produce an all-true mask")
    return x, mask


def official_error_stats(torch, reference, candidate) -> dict:
    """Mirror finite AND (absolute OR relative) from the official script."""
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"shape mismatch: reference={tuple(reference.shape)}, "
            f"candidate={tuple(candidate.shape)}"
        )
    ref = reference.detach().float()
    opt = candidate.detach().float()
    finite = torch.isfinite(ref) & torch.isfinite(opt)
    absolute = (opt - ref).abs()
    passed = finite & ((absolute <= ATOL) | (absolute <= RTOL * ref.abs()))
    finite_absolute = absolute.masked_fill(~finite, float("-inf"))
    finite_max = finite_absolute.max().item()
    return {
        "violations": int((~passed).sum().item()),
        "nonfinite_elements": int((~finite).sum().item()),
        "max_finite_abs_error": None if finite_max == float("-inf") else finite_max,
        "elements": reference.numel(),
    }


def linear_slope(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    center_x = (len(values) - 1) / 2.0
    center_y = statistics.fmean(values)
    numerator = sum(
        (index - center_x) * (value - center_y)
        for index, value in enumerate(values)
    )
    denominator = sum((index - center_x) ** 2 for index in range(len(values)))
    return numerator / denominator


def memory_assessment(allocated: list[int], reserved: list[int]) -> dict:
    if len(allocated) != len(reserved) or len(allocated) < 3:
        raise ValueError("memory trend requires matching series with >=3 repeats")
    allocated_slope = linear_slope(allocated)
    reserved_slope = linear_slope(reserved)
    allocated_growth = allocated[-1] - allocated[0]
    reserved_growth = reserved[-1] - reserved[0]
    allocated_max_growth = max(allocated) - allocated[0]
    reserved_max_growth = max(reserved) - reserved[0]
    flat = (
        allocated_slope <= MEMORY_LIMITS["allocated_slope_bytes_per_repeat"]
        and reserved_slope <= MEMORY_LIMITS["reserved_slope_bytes_per_repeat"]
        and allocated_growth <= MEMORY_LIMITS["allocated_end_growth_bytes"]
        and reserved_growth <= MEMORY_LIMITS["reserved_end_growth_bytes"]
        and allocated_max_growth <= MEMORY_LIMITS["allocated_max_growth_bytes"]
        and reserved_max_growth <= MEMORY_LIMITS["reserved_max_growth_bytes"]
    )
    return {
        "flat": flat,
        "allocated_slope_bytes_per_repeat": allocated_slope,
        "reserved_slope_bytes_per_repeat": reserved_slope,
        "allocated_end_growth_bytes": allocated_growth,
        "reserved_end_growth_bytes": reserved_growth,
        "allocated_max_growth_bytes": allocated_max_growth,
        "reserved_max_growth_bytes": reserved_max_growth,
        "limits": MEMORY_LIMITS,
    }


def immutable_packet_path(prefix: str) -> Path:
    SIDE_RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return SIDE_RESULTS / f"{prefix}_{stamp}.json"


def new_entry_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def write_immutable_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def env_fingerprint(torch) -> dict:
    driver = "unknown"
    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip().splitlines()
        if rows:
            driver = rows[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "gpu": torch.cuda.get_device_name(0),
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "hostname": platform.node(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Submission-bound shape-6 evaluator")
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=list(DEFAULT_SEEDS),
        help="at least five unique comma-separated correctness seeds",
    )
    args = parser.parse_args()

    integrity = verify_official()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for shape-6 evidence")
    otb = load_module(OFFICIAL_TORCH, "shape6_official")
    submission = load_module(SUBMISSION_FILE, "shape6_exact_submission")
    if not hasattr(submission, "UserOptimizedTransformer"):
        raise SystemExit("generated submission has no UserOptimizedTransformer")

    device = torch.device("cuda")
    numerical_state = configure_official_numerics(torch, SEED0)
    config = otb.TransformerConfig(
        batch_size=10000,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    config.validate()

    baseline = otb.BaselineTransformer(config)
    state = {
        name: value.detach().clone()
        for name, value in baseline.state_dict().items()
    }
    candidate = submission.UserOptimizedTransformer(config)
    candidate.load_state_dict(state, strict=True)
    baseline = baseline.to(device=device, dtype=torch.float32).eval()
    candidate = candidate.to(device=device, dtype=torch.float32).eval()

    # Memory trend after compile/cache warmup, retaining both allocator views.
    memory_x, memory_mask = official_case(otb, torch, config, device, SEED0)
    with torch.inference_mode():
        for _ in range(MEMORY_WARMUPS):
            warm = candidate(memory_x, memory_mask)
            del warm
        torch.cuda.synchronize(device)
        peak_allocated = []
        peak_reserved = []
        settled_allocated = []
        settled_reserved = []
        for _ in range(MEMORY_REPEATS):
            torch.cuda.reset_peak_memory_stats(device)
            output = candidate(memory_x, memory_mask)
            torch.cuda.synchronize(device)
            peak_allocated.append(torch.cuda.max_memory_allocated(device))
            peak_reserved.append(torch.cuda.max_memory_reserved(device))
            del output
            torch.cuda.synchronize(device)
            settled_allocated.append(torch.cuda.memory_allocated(device))
            settled_reserved.append(torch.cuda.memory_reserved(device))
    memory_trend = memory_assessment(settled_allocated, settled_reserved)
    del memory_x, memory_mask
    torch.cuda.empty_cache()

    # Five or more official-generator trials, reference-computed in batch chunks.
    correctness_trials = []
    total_violations = 0
    total_nonfinite = 0
    global_max = -1.0
    with torch.inference_mode():
        for seed in args.seeds:
            x, valid_mask = official_case(otb, torch, config, device, seed)
            optimized = candidate(x, valid_mask)
            trial_violations = 0
            trial_nonfinite = 0
            trial_elements = 0
            trial_max = -1.0
            for start in range(0, config.batch_size, CHUNK):
                end = min(start + CHUNK, config.batch_size)
                reference = baseline(x[start:end], valid_mask[start:end])
                stats = official_error_stats(torch, reference, optimized[start:end])
                trial_violations += stats["violations"]
                trial_nonfinite += stats["nonfinite_elements"]
                trial_elements += stats["elements"]
                if stats["max_finite_abs_error"] is not None:
                    trial_max = max(trial_max, stats["max_finite_abs_error"])
                del reference
            correctness_trials.append(
                {
                    "seed": seed,
                    "violations": trial_violations,
                    "nonfinite_elements": trial_nonfinite,
                    "elements": trial_elements,
                    "max_finite_abs_error": None if trial_max < 0 else trial_max,
                    "passed": trial_violations == 0,
                }
            )
            total_violations += trial_violations
            total_nonfinite += trial_nonfinite
            global_max = max(global_max, trial_max)
            del optimized, x, valid_mask
            torch.cuda.empty_cache()

    # Candidate-only official timing protocol. A full dense baseline remains
    # infeasible, so this is latency evidence rather than a speedup claim.
    timing_x, timing_mask = official_case(otb, torch, config, device, TIMING_SEED)
    with torch.inference_mode():
        for _ in range(OFFICIAL_WARMUPS):
            candidate(timing_x, timing_mask)
        torch.cuda.synchronize(device)
        rounds = []
        raw_samples = []
        for round_index in range(OFFICIAL_ROUNDS):
            starts = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(OFFICIAL_REPEATS)
            ]
            ends = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(OFFICIAL_REPEATS)
            ]
            torch.cuda.synchronize(device)
            for index in range(OFFICIAL_REPEATS):
                starts[index].record()
                candidate(timing_x, timing_mask)
                ends[index].record()
            torch.cuda.synchronize(device)
            samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
            rounds.append({"round": round_index, "samples_ms": samples})
            raw_samples.extend(samples)
    median_ms = statistics.median(raw_samples)

    useful_gflop = 4 * (
        8 * 10000 * 128 * 128 * 128
        + 0.5 * 4 * 10000 * 128 * 128 * 128
        + 4 * 10000 * 128 * 128 * 128
    ) / 1e9
    submission_sha = sha256_file(SUBMISSION_FILE)
    evaluator_sha = sha256_file(Path(__file__).resolve())
    passed = (
        total_violations == 0
        and total_nonfinite == 0
        and memory_trend["flat"]
    )
    packet = {
        "schema_version": "shape6-submission-v2",
        "type": "shape6_submission_evaluation",
        "entry_id": new_entry_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "id": 6,
            "batch_size": 10000,
            "seq_len": 128,
            "d_model": 128,
            "num_heads": 4,
            "ffn_dim": 128,
            "num_layers": 4,
            "causal": True,
        },
        "binding": {
            "submission_sha256": submission_sha,
            "evaluator_sha256": evaluator_sha,
            "official_sha256": integrity["official_sha256"],
            "official_manifest_sha256": integrity["manifest_sha256"],
        },
        "candidate": {
            "path": str(SUBMISSION_FILE.relative_to(ROOT)),
            "sha256": submission_sha,
            "name": "UserOptimizedTransformer (exact generated submission)",
            "runtime_routes": candidate._sub_runtime_route_report(),
        },
        "official": integrity,
        "numerical_state": numerical_state,
        "env": env_fingerprint(torch),
        "correctness": {
            "passed": total_violations == 0 and total_nonfinite == 0,
            "reference": "batch-chunked pinned official baseline (batch-independent exact computation)",
            "input_generator": "pinned official CUDA-device Generator",
            "mask": "pinned official all-true CUDA mask",
            "predicate": "finite AND (abs <= atol OR abs <= rtol * abs(reference))",
            "seeds": args.seeds,
            "trials": correctness_trials,
            "violations": total_violations,
            "nonfinite_elements": total_nonfinite,
            "max_finite_abs_error": None if global_max < 0 else global_max,
            "tolerance": {"atol": ATOL, "rtol": RTOL},
        },
        "memory": {
            "warmups": MEMORY_WARMUPS,
            "repeats": MEMORY_REPEATS,
            "peak_allocated_bytes_per_repeat": peak_allocated,
            "peak_reserved_bytes_per_repeat": peak_reserved,
            "settled_allocated_bytes_per_repeat": settled_allocated,
            "settled_reserved_bytes_per_repeat": settled_reserved,
            **memory_trend,
        },
        "timing": {
            "protocol": "candidate-only official 20-warmup, 100-repeat, 3-round CUDA-event protocol",
            "timing_seed": TIMING_SEED,
            "warmups": OFFICIAL_WARMUPS,
            "repeats_per_round": OFFICIAL_REPEATS,
            "round_count": OFFICIAL_ROUNDS,
            "rounds": rounds,
            "raw_samples_ms": raw_samples,
            "median_ms": median_ms,
            "median_definition": "statistics.median over all retained CUDA-event samples",
            "useful_gflop": useful_gflop,
            "achieved_tf_s": useful_gflop / median_ms,
            "speedup_vs_baseline": None,
        },
        "passed": passed,
        "limitation": "full B=10000 dense timing is infeasible on the project GPU; no baseline speedup is claimed",
    }
    output_path = immutable_packet_path("shape6_submission")
    write_immutable_json(output_path, packet)
    print(
        json.dumps(
            {
                "passed": passed,
                "entry_id": packet["entry_id"],
                "correctness_passed": packet["correctness"]["passed"],
                "memory_flat": memory_trend["flat"],
                "median_ms": median_ms,
                "packet": str(output_path),
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
