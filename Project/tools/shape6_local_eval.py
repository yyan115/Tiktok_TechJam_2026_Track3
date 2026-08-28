#!/usr/bin/env python3
"""Shape-6 local candidate-only evidence (Card C2).

The official dense baseline OOMs on 8 GB at B=10000 (verified 28 Aug), but
the baseline is batch-independent, so the batch-chunked official computation
is an exact reference. This tool produces the evidence packet: repeated
no-graph memory flatness, correctness vs the chunked official baseline,
candidate-only timing, and TF/s vs peaks. Writes to Project/results_side/;
the frozen runner and its journal are untouched.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "Project"
SIDE_RESULTS = PROJECT / "results_side"
OFFICIAL_TORCH = ROOT / "torch_transformer_benchmark.py"
MANIFEST_PATH = PROJECT / "manifest.json"

ATOL, RTOL = 0.002, 0.02
SEED0 = 1234
CHUNK = 500


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    for name, expected in manifest["files"].items():
        if sha256_file(ROOT / name) != expected:
            raise SystemExit(f"INTEGRITY FAILURE: {name} changed")
    import torch
    spec = importlib.util.spec_from_file_location("otb", OFFICIAL_TORCH)
    otb = importlib.util.module_from_spec(spec)
    sys.modules["otb"] = otb
    spec.loader.exec_module(otb)

    dev = torch.device("cuda")
    cfg = otb.TransformerConfig(batch_size=10000, seq_len=128, d_model=128,
                                num_heads=4, ffn_dim=128, num_layers=4,
                                causal=True)
    cfg.validate()

    torch.manual_seed(SEED0)
    base = otb.BaselineTransformer(cfg).to(dev).eval()
    state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    impl_path = (ROOT / "Project/kernels/k015_shape6.py").resolve()
    source = impl_path.read_bytes()
    csha = hashlib.sha256(source).hexdigest()
    import types
    mod = types.ModuleType("k015")
    mod.__file__ = str(impl_path)
    sys.modules["k015"] = mod
    exec(compile(source, str(impl_path), "exec"), mod.__dict__)
    cand = mod.build(otb, cfg)
    cand.load_state_dict(state, strict=True)
    cand = cand.to(dev).eval()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED0)
    x = torch.randn(10000, 128, 128, generator=gen).to(dev)

    # 1) Repeated no-graph route: memory must be flat across repeats.
    peaks = []
    with torch.inference_mode():
        for i in range(10):
            torch.cuda.reset_peak_memory_stats()
            out = cand(x, None)
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() / 2**30)
            if i < 9:
                del out
    flat = max(peaks) - min(peaks) < 0.25

    # 2) Correctness vs batch-chunked official baseline (exact computation).
    bad, maxerr = 0, 0.0
    with torch.inference_mode():
        for s in range(0, 10000, CHUNK):
            ref = base(x[s:s + CHUNK], None).float()
            o = out[s:s + CHUNK].float()
            bad += (~torch.isclose(o, ref, atol=ATOL, rtol=RTOL)).sum().item()
            maxerr = max(maxerr, (o - ref).abs().max().item())
            del ref
    del out
    torch.cuda.empty_cache()

    # 3) Candidate-only timing (no runnable full-batch baseline locally).
    samples = []
    with torch.inference_mode():
        for _ in range(3):
            cand(x, None)
        torch.cuda.synchronize()
        for _ in range(20):
            s0 = torch.cuda.Event(enable_timing=True)
            s1 = torch.cuda.Event(enable_timing=True)
            s0.record()
            cand(x, None)
            s1.record()
            torch.cuda.synchronize()
            samples.append(s0.elapsed_time(s1))
    median_ms = sorted(samples)[len(samples) // 2]
    gflop = 4 * (8*10000*128*128*128 + 0.5*4*10000*128*128*128 + 4*10000*128*128*128) / 1e9
    tfs = gflop / median_ms
    driver = "unknown"
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader"], capture_output=True,
                                text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        pass

    packet = {
        "type": "shape6_local_candidate_evidence",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shape": {"id": 6, "batch_size": 10000, "seq_len": 128, "d_model": 128,
                  "num_heads": 4, "ffn_dim": 128, "num_layers": 4, "causal": True},
        "candidate": {"path": "Project/kernels/k015_shape6.py", "sha256": csha,
                      "name": getattr(mod, "NAME", "k015")},
        "official_sha256": sha256_file(OFFICIAL_TORCH),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "env": {"gpu": torch.cuda.get_device_name(0), "driver": driver,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "python": platform.python_version(), "hostname": platform.node()},
        "seed": SEED0,
        "memory": {"peaks_gib_per_repeat": [round(p, 3) for p in peaks],
                   "flat": flat},
        "correctness": {"reference": "batch-chunked official baseline (exact)",
                        "violations": bad, "max_abs_err": maxerr,
                        "tolerance": {"atol": ATOL, "rtol": RTOL}},
        "timing": {"median_ms": median_ms, "raw_samples_ms": [round(s, 4) for s in samples],
                   "useful_gflop": round(gflop, 2), "achieved_tf_s": round(tfs, 2),
                   "vs_fp16_roof_32_5": round(tfs / 32.5, 3)},
        "limitation": "official dense baseline OOMs at this shape on 8 GB; "
                      "reference is the batch-chunked official computation",
    }
    SIDE_RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = SIDE_RESULTS / f"shape6_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(packet, indent=2, sort_keys=True))
    ok = flat and bad == 0
    print(json.dumps({"passed": ok, "median_ms": round(median_ms, 3),
                      "achieved_tf_s": round(tfs, 2),
                      "roof_fraction": round(tfs / 32.5, 3),
                      "peaks_flat": flat, "packet": str(out_path)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
