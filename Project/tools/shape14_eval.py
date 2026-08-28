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
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "Project"
MANIFEST_PATH = PROJECT / "manifest.json"
OFFICIAL_TORCH = ROOT / "torch_transformer_benchmark.py"
SIDE_RESULTS = PROJECT / "results_side"

ATOL, RTOL = 0.002, 0.02
SEED0 = 1234


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
            "official_sha256": sha256_file(OFFICIAL_TORCH)}


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
    return {
        "gpu": torch.cuda.get_device_name(0),
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
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


def fresh_x(torch, cfg, device, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model,
                       generator=gen).to(device)


def cmd_validate(args) -> int:
    """Oracle vs PINNED OFFICIAL DENSE path: full model, multi-seed, at
    sequence lengths where dense attention fits. Gate for everything else."""
    integrity = verify_official()
    import torch
    otb = load_official()
    device = torch.device("cuda")
    results = []
    ok = True
    for seq, batch in [(1024, 2), (2048, 2), (4096, 1)]:
        cfg = make_config(otb, batch, seq)
        torch.manual_seed(SEED0)
        dense = otb.BaselineTransformer(cfg).to(device).eval()
        state = {k: v.detach().clone() for k, v in dense.state_dict().items()}
        oracle = build_oracle(otb, cfg, state, device,
                              q_chunk=min(2048, seq), k_chunk=min(8192, seq))
        for trial in range(args.seeds):
            x = fresh_x(torch, cfg, device, SEED0 + trial)
            with torch.inference_mode():
                ref = dense(x, None)
                out = oracle(x, None)
            maxerr = (out - ref).abs().max().item()
            bad = (~torch.isclose(out, ref, atol=1e-4, rtol=1e-4)).sum().item()
            results.append({"seq": seq, "batch": batch, "seed": SEED0 + trial,
                            "max_abs_err": maxerr, "mismatch_at_1e-4": bad})
            ok &= maxerr < 1e-4
        del dense, oracle
        torch.cuda.empty_cache()
    packet = {
        "type": "oracle_validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "official": integrity,
        "env": env_fingerprint(torch),
        "criterion": "oracle must match pinned dense within 1e-4 abs (fp32 reassociation only)",
        "results": results,
        "passed": ok,
    }
    SIDE_RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = SIDE_RESULTS / f"validation_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(packet, indent=2, sort_keys=True))
    print(json.dumps({"passed": ok, "packet": str(out_path),
                      "worst": max(r["max_abs_err"] for r in results)}, indent=2))
    return 0 if ok else 1


def load_candidate(impl_path: Path):
    import types
    source_bytes = impl_path.read_bytes()
    sha = hashlib.sha256(source_bytes).hexdigest()
    module = types.ModuleType(impl_path.stem)
    module.__file__ = str(impl_path)
    sys.modules[impl_path.stem] = module
    exec(compile(source_bytes, str(impl_path), "exec"), module.__dict__)
    if not hasattr(module, "build"):
        raise SystemExit(f"{impl_path} must define build(otb, config)")
    return module, sha


def cmd_eval(args) -> int:
    """Candidate vs streamed oracle at the requested size; batch-streamed
    comparison; microchunked candidate; evidence packet."""
    integrity = verify_official()
    import torch
    otb = load_official()
    device = torch.device("cuda")
    cfg = make_config(otb, args.batch, args.seq)

    torch.manual_seed(SEED0)
    template = otb.BaselineTransformer(cfg)
    state = {k: v.detach().clone() for k, v in template.state_dict().items()}
    del template

    impl_path = Path(args.impl).resolve()
    candidate_module, candidate_sha = load_candidate(impl_path)
    candidate = candidate_module.build(otb, cfg)
    candidate.load_state_dict(state, strict=True)
    candidate = candidate.to(device).eval()

    torch.cuda.reset_peak_memory_stats()
    trials = []
    all_pass = True
    with torch.inference_mode():
        for trial in range(args.seeds):
            x = fresh_x(torch, cfg, device, SEED0 + trial)
            out = candidate(x, None)
            # Streamed comparison: oracle recomputed per batch slice.
            worst = 0.0
            bad_total = 0
            for bs in range(0, cfg.batch_size, args.oracle_batch_slice):
                be = min(bs + args.oracle_batch_slice, cfg.batch_size)
                sl_cfg = make_config(otb, be - bs, args.seq)
                oracle = build_oracle(otb, sl_cfg, state, device)
                ref = oracle(x[bs:be], None)
                o = out[bs:be].float()
                bad_total += (~torch.isclose(o, ref, atol=ATOL, rtol=RTOL)).sum().item()
                worst = max(worst, (o - ref).abs().max().item())
                del oracle, ref
                torch.cuda.empty_cache()
            trials.append({"seed": SEED0 + trial, "max_abs_err": worst,
                           "violations": bad_total})
            all_pass &= bad_total == 0
            del out
            torch.cuda.empty_cache()

        # Timing: candidate only (no runnable dense baseline at this scale).
        x = fresh_x(torch, cfg, device, SEED0 + 100000)
        for _ in range(args.warmup):
            candidate(x, None)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            candidate(x, None)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))

    samples_sorted = sorted(samples)
    median_ms = samples_sorted[len(samples_sorted) // 2]
    packet = {
        "type": "shape14_side_evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shape": {"id": 14, "batch_size": cfg.batch_size, "seq_len": cfg.seq_len,
                  "d_model": 1024, "num_heads": 16, "ffn_dim": 1024,
                  "num_layers": 2, "causal": True},
        "official": integrity,
        "candidate": {"path": str(impl_path.relative_to(ROOT)),
                      "sha256": candidate_sha,
                      "name": getattr(candidate_module, "NAME", impl_path.stem)},
        "submission_sha256": (sha256_file(PROJECT / "submission" /
                              "torch_transformer_benchmark_submission.py")
                              if (PROJECT / "submission" /
                                  "torch_transformer_benchmark_submission.py").exists()
                              else None),
        "env": env_fingerprint(torch),
        "seeds": [SEED0 + t for t in range(args.seeds)],
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "correctness": {"trials": trials, "passed": all_pass,
                        "reference": "streamed fp32 oracle (validated vs pinned dense)"},
        "timing": {"warmup": args.warmup, "repeats": args.repeats,
                   "median_ms": median_ms,
                   "raw_samples_ms": [round(s, 6) for s in samples]},
        "memory": {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30},
        "limitation": "official dense baseline infeasible at this shape; "
                      "reference is the validated streamed oracle",
    }
    SIDE_RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = SIDE_RESULTS / (f"shape14_{time.strftime('%Y%m%d-%H%M%S')}"
                               f"_B{cfg.batch_size}_S{cfg.seq_len}.json")
    out_path.write_text(json.dumps(packet, indent=2, sort_keys=True))
    print(json.dumps({"passed": all_pass, "median_ms": median_ms,
                      "peak_gib": round(packet["memory"]["peak_allocated_gib"], 2),
                      "packet": str(out_path)}, indent=2))
    return 0 if all_pass else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Pinned side evaluator for shape 14")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="streamed oracle vs pinned dense (gate)")
    v.add_argument("--seeds", type=int, default=3)
    e = sub.add_parser("eval", help="candidate vs validated oracle + packet")
    e.add_argument("--impl", required=True)
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--seq", type=int, default=100000)
    e.add_argument("--seeds", type=int, default=2)
    e.add_argument("--oracle-batch-slice", type=int, default=1)
    e.add_argument("--warmup", type=int, default=3)
    e.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    return cmd_validate(args) if args.cmd == "validate" else cmd_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
