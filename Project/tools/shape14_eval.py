#!/usr/bin/env python3
"""Independently pinned side evaluator for SHAPE 14 (B=32, S=100000, d=1024,
H=16, L=2, causal) — the shape the frozen runner refuses by design.

Contract (harness_v2_proposal Card 1, converged 29 Aug after 4 blind review
rounds):
- The FROZEN RUNNER IS NOT TOUCHED. This tool is separate, self-hashing, and
  writes immutable evidence packets to Project/results_side/ (never to the
  runner-owned results/JOURNAL.jsonl).
- `validate`: the streamed full-model oracle is checked against the PINNED
  official dense implementation (hash-verified via manifest) at feasible
  sequence lengths, full model, multi-seed. The oracle referees nothing
  until this is green.
- `eval`: candidate vs streamed oracle at arbitrary S with batch-streamed
  comparison, batch-microchunked candidate execution (no CUDA graphs at
  this scale), size-aware timing, and peak-memory accounting. Produces the
  evidence packet binding evaluator/candidate/official SHAs, config,
  environment, seeds, error statistics, raw timing samples, and peak
  allocated/reserved memory.

Memory discipline everywhere: nothing materializes more than a few GiB.
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
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "Project"
MANIFEST_PATH = PROJECT / "manifest.json"
OFFICIAL_TORCH = ROOT / "torch_transformer_benchmark.py"
SUBMISSION_FILE = (PROJECT / "submission" /
                   "torch_transformer_benchmark_submission.py")
SIDE_RESULTS = PROJECT / "results_side"

ATOL, RTOL = 0.002, 0.02
SEED0 = 1234
TIMING_SEED = SEED0 + 100000
OFFICIAL_BATCH = 32
OFFICIAL_SEQ = 100000
SLICE_SEED_FORMULA = "base_seed * 1000 + batch_index"
VALIDATION_SCHEMA = "shape14-oracle-validation-v2"
DECOMPOSITION_SCHEMA = "shape14-decomposition-v2"
EVALUATION_SCHEMA = "shape14-streamed-v2"
DEFAULT_DECOMP_MAX_ABS = 1e-5


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_official() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    mismatches = []
    for name, expected in manifest["files"].items():
        actual = sha256_file(ROOT / name)
        if actual != expected:
            mismatches.append(name)
    if mismatches:
        raise SystemExit(f"INTEGRITY FAILURE: official files changed: {mismatches}")
    return {"official_commit": manifest["official_commit"],
            "official_sha256": sha256_file(OFFICIAL_TORCH),
            "official_manifest_sha256": sha256_file(MANIFEST_PATH)}


def evidence_binding(integrity: dict) -> dict:
    return {
        "submission_sha256": sha256_file(SUBMISSION_FILE),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "official_sha256": integrity["official_sha256"],
        "official_manifest_sha256": integrity["official_manifest_sha256"],
    }


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


def require_bound_artifact(path_value: str, artifact_type: str,
                           schema: str, binding: dict) -> tuple[Path, dict]:
    unresolved = Path(path_value)
    if unresolved.is_symlink():
        raise SystemExit("artifact must not be a symlink")
    path = unresolved.resolve(strict=True)
    side_root = SIDE_RESULTS.resolve()
    if not path.is_relative_to(side_root) or not path.is_file():
        raise SystemExit(f"artifact must be an existing file under {SIDE_RESULTS}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid artifact {path}: {exc}") from exc
    if payload.get("type") != artifact_type or payload.get("schema_version") != schema:
        raise SystemExit(f"wrong artifact type/schema: {path}")
    if payload.get("passed") is not True:
        raise SystemExit(f"artifact did not pass: {path}")
    if payload.get("binding") != binding:
        raise SystemExit(f"artifact binding does not match current bytes: {path}")
    return path, payload


def load_official():
    import importlib.util
    spec = importlib.util.spec_from_file_location("otb", OFFICIAL_TORCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["otb"] = module
    spec.loader.exec_module(module)
    return module


def env_fingerprint(torch) -> dict:
    driver = "unknown"
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
    except Exception:
        pass
    # ship_manifest.REQUIRED_ENV_KEYS binds gpu/driver/torch/cuda/triton.  A
    # missing or "unknown" value makes the packet unshippable, so the version
    # is resolved here rather than left to the consumer.  The authored kernels
    # are Triton, so an absent Triton is a fatal evidence defect, not a note.
    triton_version = "unknown"
    try:
        import triton  # noqa: PLC0415 - optional at import time, required here
        triton_version = str(getattr(triton, "__version__", "")).strip() or "unknown"
    except Exception:
        pass
    return {
        "gpu": torch.cuda.get_device_name(0),
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton_version,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }


def streamed_attention(attn, x, causal, q_chunk, k_chunk):
    """Baseline-exact attention in fp32 without the S x S table: per (b, h),
    streaming softmax over (q-chunk x k-chunk) blocks. Dense-math identical
    up to fp32 reassociation."""
    import torch
    batch, seq_len, _ = x.shape
    q = attn._split_heads(attn.q_proj(x))
    k = attn._split_heads(attn.k_proj(x))
    v = attn._split_heads(attn.v_proj(x))
    B, H, S, D = q.shape
    out = torch.empty_like(q)
    for b in range(B):
        for h in range(H):
            qb, kb, vb = q[b, h], k[b, h], v[b, h]
            for qs in range(0, S, q_chunk):
                qe = min(qs + q_chunk, S)
                m = torch.full((qe - qs,), float("-inf"), device=x.device)
                l = torch.zeros(qe - qs, device=x.device)
                acc = torch.zeros(qe - qs, D, device=x.device)
                k_end = qe if causal else S
                for ks in range(0, k_end, k_chunk):
                    ke = min(ks + k_chunk, k_end)
                    scores = qb[qs:qe] @ kb[ks:ke].T * attn.scale
                    if causal:
                        qidx = torch.arange(qs, qe, device=x.device)[:, None]
                        kidx = torch.arange(ks, ke, device=x.device)[None, :]
                        scores = scores.masked_fill(kidx > qidx, float("-inf"))
                    m_new = torch.maximum(m, scores.amax(dim=1))
                    alpha = torch.exp(m - m_new)
                    p = torch.exp(scores - m_new[:, None])
                    l = l * alpha + p.sum(dim=1)
                    acc = acc * alpha[:, None] + p @ vb[ks:ke]
                    m = m_new
                out[b, h, qs:qe] = acc / l[:, None]
    context = out.transpose(1, 2).contiguous().view(batch, seq_len, attn.d_model)
    return attn.out_proj(context)


def build_oracle(otb, config, state_dict, device, q_chunk=2048, k_chunk=8192):
    """Baseline modules with attention.forward swapped for the streamed
    implementation — same weights, same math, no S x S table."""
    import types
    import torch  # noqa: F401
    oracle = otb.BaselineTransformer(config)
    oracle.load_state_dict(state_dict, strict=True)
    oracle = oracle.to(device=device).eval()
    for layer in oracle.layers:
        att = layer.attention
        att.forward = types.MethodType(
            lambda self, x, valid_token_mask=None, causal=True,
                   _qc=q_chunk, _kc=k_chunk:
                streamed_attention(self, x, causal, _qc, _kc),
            att)
    return oracle


def make_config(otb, batch, seq):
    cfg = otb.TransformerConfig(batch_size=batch, seq_len=seq, d_model=1024,
                                num_heads=16, ffn_dim=1024, num_layers=2,
                                causal=True)
    cfg.validate()
    return cfg


def official_case(otb, torch, cfg, device, seed):
    x, mask = otb.generate_random_case(
        config=cfg,
        device=device,
        dtype=torch.float32,
        seed=seed,
        padding_ratio=0.0,
        input_scale=1.0,
    )
    if mask.dtype != torch.bool or mask.device != device or not bool(mask.all()):
        raise AssertionError("official no-padding case must return an all-true CUDA mask")
    return x, mask


def cmd_validate(args) -> int:
    """Oracle vs PINNED OFFICIAL DENSE path: full model, multi-seed, at
    sequence lengths where dense attention fits. Gate for everything else."""
    if args.seeds < 3:
        raise SystemExit("--seeds must be at least 3")
    integrity = verify_official()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for shape-14 validation")
    otb = load_official()
    device = torch.device("cuda")
    numerical_state = configure_official_numerics(torch, SEED0)
    binding = evidence_binding(integrity)
    results = []
    ok = True
    for seq, batch in [(1024, 2), (2048, 2), (4096, 1)]:
        cfg = make_config(otb, batch, seq)
        dense = otb.BaselineTransformer(cfg).to(device).eval()
        state = {k: v.detach().clone() for k, v in dense.state_dict().items()}
        oracle = build_oracle(otb, cfg, state, device,
                              q_chunk=min(2048, seq), k_chunk=min(8192, seq))
        for trial in range(args.seeds):
            x, valid_mask = official_case(
                otb, torch, cfg, device, SEED0 + trial
            )
            with torch.inference_mode():
                ref = dense(x, valid_mask)
                out = oracle(x, valid_mask)
            maxerr = (out - ref).abs().max().item()
            bad = (~torch.isclose(out, ref, atol=1e-4, rtol=1e-4)).sum().item()
            results.append({"seq": seq, "batch": batch, "seed": SEED0 + trial,
                            "max_abs_err": maxerr, "mismatch_at_1e-4": bad})
            ok &= maxerr < 1e-4
        del dense, oracle
        torch.cuda.empty_cache()
    packet = {
        "schema_version": VALIDATION_SCHEMA,
        "type": "shape14_oracle_validation",
        "entry_id": new_entry_id(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "binding": binding,
        "official": integrity,
        "candidate": {
            "path": str(SUBMISSION_FILE.relative_to(ROOT)),
            "sha256": binding["submission_sha256"],
            "note": "bound for downstream evaluation; not executed by this oracle-only check",
        },
        "numerical_state": numerical_state,
        "env": env_fingerprint(torch),
        "criterion": "oracle must match pinned dense within 1e-4 abs (fp32 reassociation only)",
        "results": results,
        "passed": ok,
    }
    out_path = immutable_packet_path("shape14_validation")
    write_immutable_json(out_path, packet)
    print(json.dumps({"passed": ok, "entry_id": packet["entry_id"],
                      "packet": str(out_path),
                      "worst": max(r["max_abs_err"] for r in results)}, indent=2))
    return 0 if ok else 1


def load_submission():
    """Import the exact generated submission artifact without running main."""
    module_name = "track3_shape14_submission"
    spec = importlib.util.spec_from_file_location(module_name, SUBMISSION_FILE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import submission from {SUBMISSION_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not hasattr(module, "UserOptimizedTransformer"):
        raise SystemExit(f"{SUBMISSION_FILE} has no UserOptimizedTransformer")
    return module


def build_submission_candidate(submission, cfg, state, device):
    candidate = submission.UserOptimizedTransformer(cfg)
    candidate.load_state_dict(state, strict=True)
    return candidate.to(device=device).eval()


def fresh_slice(otb, torch, seq, device, base_seed, batch_index):
    """Generate one official B=1 CUDA-generator case for streamed evaluation."""
    cfg = make_config(otb, 1, seq)
    return official_case(
        otb, torch, cfg, device, base_seed * 1000 + batch_index
    )


def official_error_stats(torch, reference, optimized) -> dict:
    """Apply the official finite AND (absolute OR relative) predicate."""
    if reference.shape != optimized.shape:
        raise AssertionError(
            f"shape mismatch: reference={tuple(reference.shape)}, "
            f"candidate={tuple(optimized.shape)}"
        )
    ref = reference.detach().float()
    opt = optimized.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()
    abs_ok = abs_error <= ATOL
    rel_ok = abs_error <= RTOL * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed_mask = ~passed_mask
    failed_elements = int(failed_mask.sum().item())
    nonfinite_elements = int((~finite_mask).sum().item())
    if nonfinite_elements:
        max_abs_error = None
        abs_error_sum = None
    else:
        max_abs_error = abs_error.max().item()
        abs_error_sum = abs_error.sum(dtype=torch.float64).item()
    return {
        "violations": failed_elements,
        "nonfinite_elements": nonfinite_elements,
        "max_abs_error": max_abs_error,
        "abs_error_sum": abs_error_sum,
        "elements": reference.numel(),
    }


def utc_now():
    now = datetime.now(timezone.utc)
    return now.isoformat(), now.strftime("%Y%m%d-%H%M%S")


def cmd_eval(args) -> int:
    """Evaluate the official workload as 32 serial, independently staged B=1 slices."""
    if args.correctness_seeds < 5:
        raise SystemExit("--correctness-seeds must be at least 5")
    if args.timing_repeats < 3:
        raise SystemExit("--timing-repeats must be at least 3")
    if args.warmup < 3:
        raise SystemExit("--warmup must be at least 3")

    integrity = verify_official()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for shape-14 evaluation")
    otb = load_official()
    submission = load_submission()
    device = torch.device("cuda")
    numerical_state = configure_official_numerics(torch, SEED0)
    binding = evidence_binding(integrity)
    validation_path, _validation = require_bound_artifact(
        args.validation_packet,
        "shape14_oracle_validation",
        VALIDATION_SCHEMA,
        binding,
    )
    decomposition_path, _decomposition = require_bound_artifact(
        args.decomposition_packet,
        "shape14_batch_decomposition_check",
        DECOMPOSITION_SCHEMA,
        binding,
    )
    cfg = make_config(otb, OFFICIAL_BATCH, OFFICIAL_SEQ)

    template = otb.BaselineTransformer(cfg)
    state = {k: v.detach().clone() for k, v in template.state_dict().items()}
    del template

    submission_sha = sha256_file(SUBMISSION_FILE)
    candidate = build_submission_candidate(submission, cfg, state, device)
    oracle_cfg = make_config(otb, 1, OFFICIAL_SEQ)
    oracle = build_oracle(otb, oracle_cfg, state, device)

    correctness_seeds = [SEED0 + trial for trial in range(args.correctness_seeds)]
    trials = []
    total_violations = 0
    total_nonfinite = 0
    total_abs_error = 0.0
    total_elements = 0
    global_max_abs_error = -1.0
    worst_seed = None
    worst_slice_index = None
    with torch.inference_mode():
        for base_seed in correctness_seeds:
            trial_violations = 0
            trial_nonfinite = 0
            trial_abs_error = 0.0
            trial_elements = 0
            trial_max_abs_error = -1.0
            trial_worst_slice = None
            for batch_index in range(OFFICIAL_BATCH):
                x, valid_mask = fresh_slice(
                    otb, torch, OFFICIAL_SEQ, device, base_seed, batch_index
                )
                out = candidate(x, valid_mask)
                ref = oracle(x, valid_mask)
                stats = official_error_stats(torch, ref, out)
                trial_violations += stats["violations"]
                trial_nonfinite += stats["nonfinite_elements"]
                trial_elements += stats["elements"]
                if stats["abs_error_sum"] is not None:
                    trial_abs_error += stats["abs_error_sum"]
                if stats["nonfinite_elements"]:
                    if trial_nonfinite == stats["nonfinite_elements"]:
                        trial_worst_slice = batch_index
                elif (trial_nonfinite == 0 and
                      stats["max_abs_error"] > trial_max_abs_error):
                    trial_max_abs_error = stats["max_abs_error"]
                    trial_worst_slice = batch_index
                del stats, ref, out, x, valid_mask
                torch.cuda.empty_cache()

            trial_mean = (None if trial_nonfinite else
                          trial_abs_error / trial_elements)
            trials.append({
                "base_seed": base_seed,
                "violations": trial_violations,
                "nonfinite_elements": trial_nonfinite,
                "elements": trial_elements,
                "max_abs_error": None if trial_nonfinite else trial_max_abs_error,
                "mean_abs_error": trial_mean,
                "worst_slice_index": trial_worst_slice,
            })
            total_violations += trial_violations
            total_elements += trial_elements
            if not trial_nonfinite:
                total_abs_error += trial_abs_error
            if trial_nonfinite:
                if total_nonfinite == 0:
                    worst_seed = base_seed
                    worst_slice_index = trial_worst_slice
            elif total_nonfinite == 0 and trial_max_abs_error > global_max_abs_error:
                global_max_abs_error = trial_max_abs_error
                worst_seed = base_seed
                worst_slice_index = trial_worst_slice
            total_nonfinite += trial_nonfinite

        del oracle
        torch.cuda.empty_cache()

        # Size-aware warmup: individual B=1 calls only, never a full B=32 tensor.
        for warmup_index in range(args.warmup):
            batch_index = warmup_index % OFFICIAL_BATCH
            x, valid_mask = fresh_slice(
                otb, torch, OFFICIAL_SEQ, device, TIMING_SEED, batch_index
            )
            out = candidate(x, valid_mask)
            torch.cuda.synchronize()
            del out, x, valid_mask

        repeat_slice_times = []
        repeat_sums = []
        staging_inclusive_wall_times = []
        peak_allocated = []
        peak_reserved = []
        for _repeat in range(args.timing_repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            wall_start = time.perf_counter()
            slice_times = []
            for batch_index in range(OFFICIAL_BATCH):
                x, valid_mask = fresh_slice(
                    otb, torch, OFFICIAL_SEQ, device,
                    TIMING_SEED, batch_index
                )
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = candidate(x, valid_mask)
                end.record()
                end.synchronize()
                slice_times.append(start.elapsed_time(end))
                del out, x, valid_mask, start, end
            torch.cuda.synchronize()
            staging_inclusive_wall_times.append(
                (time.perf_counter() - wall_start) * 1000.0
            )
            repeat_slice_times.append(slice_times)
            repeat_sums.append(sum(slice_times))
            peak_allocated.append(torch.cuda.max_memory_allocated())
            peak_reserved.append(torch.cuda.max_memory_reserved())

    matrix_by_batch = [
        [repeat_slice_times[repeat][batch_index]
         for repeat in range(args.timing_repeats)]
        for batch_index in range(OFFICIAL_BATCH)
    ]
    median_of_sums = statistics.median(repeat_sums)
    all_pass = total_violations == 0 and total_nonfinite == 0
    mean_abs_error = (None if total_nonfinite else
                      total_abs_error / total_elements)
    max_abs_error = None if total_nonfinite else global_max_abs_error
    timestamp, _stamp = utc_now()
    evaluator_sha = sha256_file(Path(__file__).resolve())
    packet = {
        "schema_version": EVALUATION_SCHEMA,
        "type": "shape14_side_evaluation",
        "entry_id": new_entry_id(),
        "timestamp": timestamp,
        "shape": {"id": 14, "batch_size": cfg.batch_size, "seq_len": cfg.seq_len,
                  "d_model": 1024, "num_heads": 16, "ffn_dim": 1024,
                  "num_layers": 2, "causal": True},
        "config": {"batch_size": cfg.batch_size, "seq_len": cfg.seq_len,
                   "d_model": 1024, "num_heads": 16, "ffn_dim": 1024,
                   "num_layers": 2, "causal": True},
        "official": integrity,
        "candidate": {
            "path": str(SUBMISSION_FILE.relative_to(ROOT)),
            "sha256": submission_sha,
            "name": "UserOptimizedTransformer (generated submission)",
            "runtime_routes": candidate._sub_runtime_route_report(),
        },
        "binding": binding,
        "submission_sha256": submission_sha,
        "evaluator_sha256": evaluator_sha,
        "required_artifacts": {
            "oracle_validation": {
                "path": str(validation_path.relative_to(ROOT)),
                "sha256": sha256_file(validation_path),
                "schema_version": VALIDATION_SCHEMA,
            },
            "batch_decomposition": {
                "path": str(decomposition_path.relative_to(ROOT)),
                "sha256": sha256_file(decomposition_path),
                "schema_version": DECOMPOSITION_SCHEMA,
            },
        },
        "numerical_state": numerical_state,
        "env": env_fingerprint(torch),
        "seeds": correctness_seeds,
        "seeding": {
            "correctness_base_seeds": correctness_seeds,
            "timing_base_seed": TIMING_SEED,
            "slice_seed_formula": SLICE_SEED_FORMULA,
            "generator": "pinned official CUDA-device Generator per B=1 slice",
            "mask": "pinned official all-true CUDA mask per B=1 slice",
            "timing_repeat_policy": "same 32 deterministic slices in every repeat",
        },
        "predicate": "official-abs-OR-rel",
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "correctness": {
            "trials": trials,
            "passed": all_pass,
            "violations": total_violations,
            "nonfinite_elements": total_nonfinite,
            "elements": total_elements,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "worst_base_seed": worst_seed,
            "worst_slice_index": worst_slice_index,
            "predicate": "official-abs-OR-rel",
            "reference": "streamed fp32 oracle (validated vs pinned dense)",
        },
        "timing": {
            "protocol": "32 serial B=1 submission calls; CUDA events exclude input staging",
            "warmup_slices": args.warmup,
            "timing_repeats": args.timing_repeats,
            "slice_times_ms": {
                "orientation": "batch_index x timing_repeat",
                "values": matrix_by_batch,
            },
            "gpu_compute_sum_ms_per_repeat": [
                value for value in repeat_sums
            ],
            "gpu_compute_median_of_sums_ms": median_of_sums,
            "staging_inclusive_wall_ms_per_repeat": [
                value for value in staging_inclusive_wall_times
            ],
            "staging_inclusive_wall_median_ms": statistics.median(
                staging_inclusive_wall_times
            ),
        },
        "memory": {
            "peak_allocated_bytes_per_repeat": peak_allocated,
            "peak_reserved_bytes_per_repeat": peak_reserved,
            "max_peak_allocated_bytes": max(peak_allocated),
            "max_peak_reserved_bytes": max(peak_reserved),
            "max_peak_allocated_gib": max(peak_allocated) / 2**30,
            "max_peak_reserved_gib": max(peak_reserved) / 2**30,
        },
        "limitation": "official dense baseline infeasible at this shape; "
                      "reference is the validated streamed oracle; timing is "
                      "32 serial B=1 submission calls, not one literal B=32 call",
    }
    packet["passed"] = all_pass
    out_path = immutable_packet_path("shape14_streamed")
    write_immutable_json(out_path, packet)
    print(json.dumps({"passed": all_pass,
                      "entry_id": packet["entry_id"],
                      "median_of_sums_ms": median_of_sums,
                      "peak_gib": round(packet["memory"]["max_peak_allocated_gib"], 2),
                      "packet": str(out_path)}, indent=2))
    return 0 if all_pass else 1


def cmd_decomp_check(args) -> int:
    """Compare one reduced full-batch call with independent B=1 calls."""
    if args.batch <= 0 or args.seq <= 0:
        raise SystemExit("--batch and --seq must be positive")
    if args.max_abs_difference <= 0:
        raise SystemExit("--max-abs-difference must be positive")

    integrity = verify_official()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for shape-14 decomposition checking")
    otb = load_official()
    submission = load_submission()
    device = torch.device("cuda")
    numerical_state = configure_official_numerics(torch, args.seed)
    binding = evidence_binding(integrity)
    cfg = make_config(otb, args.batch, args.seq)

    template = otb.BaselineTransformer(cfg)
    state = {k: v.detach().clone() for k, v in template.state_dict().items()}
    del template
    full_candidate = build_submission_candidate(submission, cfg, state, device)
    slice_candidate = build_submission_candidate(submission, cfg, state, device)
    x, valid_mask = official_case(otb, torch, cfg, device, args.seed)

    with torch.inference_mode():
        full_output = full_candidate(x, valid_mask)
        slice_outputs = [
            slice_candidate(
                x[index:index + 1], valid_mask[index:index + 1]
            )
            for index in range(args.batch)
        ]
        decomposed_output = torch.cat(slice_outputs, dim=0)
        differences = (full_output.float() - decomposed_output.float()).abs()
        max_abs_difference = differences.max().item()
        stats = official_error_stats(torch, full_output, decomposed_output)
    torch.cuda.synchronize()
    passed = (
        stats["violations"] == 0
        and stats["nonfinite_elements"] == 0
        and max_abs_difference <= args.max_abs_difference
    )
    packet = {
        "schema_version": DECOMPOSITION_SCHEMA,
        "type": "shape14_batch_decomposition_check",
        "entry_id": new_entry_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "binding": binding,
        "official": integrity,
        "candidate": {
            "path": str(SUBMISSION_FILE.relative_to(ROOT)),
            "sha256": binding["submission_sha256"],
            "full_batch_runtime_routes": full_candidate._sub_runtime_route_report(),
            "slice_runtime_routes": slice_candidate._sub_runtime_route_report(),
        },
        "numerical_state": numerical_state,
        "batch_size": args.batch,
        "seq_len": args.seq,
        "seed": args.seed,
        "input_generator": "pinned official CUDA-device Generator",
        "mask": "pinned official all-true CUDA mask",
        "max_abs_difference": max_abs_difference,
        "max_abs_difference_limit": args.max_abs_difference,
        "official_predicate": stats,
        "submission_sha256": sha256_file(SUBMISSION_FILE),
        "passed": passed,
    }
    output_path = immutable_packet_path("shape14_decomposition")
    write_immutable_json(output_path, packet)
    print(json.dumps({
        "passed": passed,
        "entry_id": packet["entry_id"],
        "max_abs_difference": max_abs_difference,
        "packet": str(output_path),
    }, indent=2))
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Pinned side evaluator for shape 14")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="streamed oracle vs pinned dense (gate)")
    v.add_argument("--seeds", type=int, default=3)
    e = sub.add_parser("eval", help="streamed official-shape submission evaluation")
    e.add_argument("--validation-packet", required=True,
                   help="passed shape14-oracle-validation-v2 packet")
    e.add_argument("--decomposition-packet", required=True,
                   help="passed shape14-decomposition-v2 packet")
    e.add_argument("--correctness-seeds", type=int, default=5)
    e.add_argument("--timing-repeats", type=int, default=3)
    e.add_argument("--warmup", type=int, default=3)
    d = sub.add_parser("decomp-check", help="reduced full-batch vs B=1 equivalence")
    d.add_argument("--batch", type=int, default=8)
    d.add_argument("--seq", type=int, default=2048)
    d.add_argument("--seed", type=int, default=SEED0)
    d.add_argument("--max-abs-difference", type=float,
                   default=DEFAULT_DECOMP_MAX_ABS)
    args = ap.parse_args()
    if args.cmd == "validate":
        return cmd_validate(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    return cmd_decomp_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
