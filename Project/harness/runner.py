#!/usr/bin/env python3
"""Trusted runner ("the referee"), v0.9.3.

v0.9.2/v0.9.3 hardening (Sol round-3 minors + codex handoff review): candidate
code executes from the exact hashed bytes; anti-cache pass re-randomizes values
before EVERY timed call; input-mutation tamper checks around candidate calls and
timing rounds; calibration/champion matching pinned to the exact runner sha;
malformed ledger lines warn loudly; evidence packets verify source-vs-journal
hashes; optional --ledger flag isolates test runs from the production journal.

Takes a candidate implementation + an official shape id, then:
  1. verifies the official benchmark files are untouched (sha256 vs manifest.json)
  2. checks correctness against the official baseline (multi-seed + tripwires)
  3. times both models the same way the official script does (CUDA events,
     alternating rounds), plus wall-clock and anti-cache cross-checks
  4. appends one machine-written record (with raw samples) to results/JOURNAL.jsonl
  5. regenerates results/LEADERBOARD.md from the journal

Hardening added in v0.9.1 after the Stage-1 Sol audit (RULE_VIOLATION verdict,
see Project/audits/stage1_review_raw.log):
  - candidate source is hashed BEFORE its module code executes
  - trusted callables are snapshotted and the baseline is built BEFORE any
    candidate code runs; a baseline-invariance probe (bitwise-identical output
    on a stashed input) runs after candidate load and after timing — in-process
    monkeypatching of the official module, the baseline, or torch math changes
    the probe output and aborts the run
  - anti-cache timed pass: inputs are re-randomized IN PLACE (same addresses,
    fresh values) every round and rotated across buffers; outputs are re-checked
    against freshly computed references, and the anti-cache median must agree
    with the official fixed-input median. Address-keyed or stale caches either
    produce wrong outputs or blow the timing-agreement bound.
  - same-address-new-values correctness tripwire (replaces the weak 0.1%
    perturbation)
  - promotion requires the primary profile: official-default dtype AND official
    timing arguments; calibration must match the full environment fingerprint
    and timing arguments
  - the runner records its own sha256 in every entry

Residual risk, documented deliberately: candidate code still executes in the
same OS process, so a pathological candidate could in principle subvert
measurement in ways the invariance probe does not observe (e.g. patching CUDA
event timing itself). Full subprocess isolation was considered and deferred —
the project trust model is cooperative (guards against mistakes, not malice;
Project/PLAN.md), candidate sources are short files reviewed at Sol audit
checkpoints, and git records their exact bytes. The user decides at freeze time
whether this residual is acceptable.

Candidate contract (a .py file, usually in Project/kernels/):
  build(otb, config) -> torch.nn.Module   # fresh model, CPU/fp32, param names
                                          # compatible with the baseline
  optional: copy_weights(baseline, candidate)  # if param names differ
  optional: NAME, DESCRIPTION strings

This file is part of the trusted evaluator. After the Stage-1 freeze it must
not be modified without user approval (see Project/PLAN.md).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HARNESS_VERSION = "0.9.3-unfrozen"

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "Project"
SHAPES_PATH = PROJECT / "shapes.json"
MANIFEST_PATH = PROJECT / "manifest.json"
RESULTS_DIR = PROJECT / "results"
JOURNAL_PATH = RESULTS_DIR / "JOURNAL.jsonl"
LEADERBOARD_PATH = RESULTS_DIR / "LEADERBOARD.md"
PACKETS_DIR = PROJECT / "audits" / "packets"
VERDICTS_PATH = PROJECT / "audits" / "verdicts.jsonl"
OFFICIAL_TORCH = ROOT / "torch_transformer_benchmark.py"

# Official script defaults, mirrored exactly (see torch_transformer_benchmark.py
# parse_args / main). The primary profile never deviates from these.
OFFICIAL_DEFAULTS = dict(
    dtype="float32",
    padding_ratio=0.0,
    input_scale=1.0,
    accuracy_trials=5,
    rtol=0.02,
    atol=0.002,
    seed=1234,
    warmup=20,
    repeats=100,
    benchmark_rounds=3,
    matmul_precision="high",
    allow_tf32=True,
)

INVARIANCE_SEED = 424242  # private probe input; never used for scoring

# A candidate is promoted only if its speedup exceeds
# 1 + max(PROMOTION_MIN_MARGIN, PROMOTION_NOISE_FACTOR * calibrated_noise).
PROMOTION_MIN_MARGIN = 0.03
PROMOTION_NOISE_FACTOR = 3.0

# Wall-clock cross-check: flag when per-iter event time is much smaller than
# per-iter wall time (work possibly hidden from the event timer).
WALL_SUSPICION_FACTOR = 1.75
WALL_SUSPICION_SLACK_MS = 0.10

# Anti-cache pass: fresh-values timing must agree with fixed-input timing.
ANTI_CACHE_ITERS = 40
ANTI_CACHE_MAX_RATIO = 1.25
ANTI_CACHE_SLACK_MS = 0.05


class TamperError(SystemExit):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def verify_hashes() -> Dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    mismatches = []
    for name, expected in manifest["files"].items():
        actual = sha256_file(ROOT / name)
        if actual != expected:
            mismatches.append({"file": name, "expected": expected, "actual": actual})
    if mismatches:
        raise SystemExit(
            "INTEGRITY FAILURE: official files changed since manifest was approved: "
            + json.dumps(mismatches)
        )
    return {"official_commit": manifest["official_commit"], "verified": True}


def load_official():
    spec = importlib.util.spec_from_file_location("otb", OFFICIAL_TORCH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["otb"] = module  # dataclasses on py3.14 needs the module registered
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def snapshot_trusted(otb) -> Dict[str, Any]:
    """Capture references to trusted callables BEFORE any candidate code runs."""
    return {
        "compare_outputs": otb.compare_outputs,
        "benchmark_once": otb.benchmark_once,
        "warmup_model": otb.warmup_model,
        "generate_random_case": otb.generate_random_case,
        "TimingResult": otb.TimingResult,
        "copy_model_weights": otb.copy_model_weights,
        "resolve_dtype": otb.resolve_dtype,
        "TransformerConfig": otb.TransformerConfig,
        "BaselineTransformer": otb.BaselineTransformer,
        "UserOptimizedTransformer": otb.UserOptimizedTransformer,
    }


def load_shape(shape_id: int) -> Dict[str, Any]:
    shapes = json.loads(SHAPES_PATH.read_text())["shapes"]
    for shape in shapes:
        if shape["id"] == shape_id:
            return shape
    raise SystemExit(f"shape id {shape_id} not found in shapes.json")


def load_candidate(impl_path: Path):
    # Read once, hash those bytes, and execute EXACTLY those bytes (no separate
    # re-read via a loader, no chance of stale .pyc reuse) — Sol audit round 3,
    # minor finding 1.
    import types  # noqa: PLC0415
    source_bytes = impl_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    module = types.ModuleType(impl_path.stem)
    module.__file__ = str(impl_path)
    sys.modules[impl_path.stem] = module
    code = compile(source_bytes, str(impl_path), "exec")
    exec(code, module.__dict__)
    if not hasattr(module, "build"):
        raise SystemExit(f"{impl_path} must define build(otb, config)")
    return module, source_sha


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def env_fingerprint(torch) -> Dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    driver = "unknown"
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
    except Exception:
        pass
    triton_version = "none"
    try:
        import triton  # noqa: PLC0415
        triton_version = triton.__version__
    except Exception:
        pass
    return {
        "gpu": gpu,
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton_version,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "harness_version": HARNESS_VERSION,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "git_rev": git_rev(),
    }


def calibration_match_key(entry: Dict[str, Any]) -> Any:
    env = entry.get("env", {})
    return (
        entry.get("shape_id"),
        entry.get("dtype"),
        json.dumps(entry.get("timing_args", {}), sort_keys=True),
        env.get("gpu"), env.get("driver"), env.get("torch"), env.get("cuda"),
        env.get("hostname"), env.get("harness_version"),
        env.get("runner_sha256"),  # codex handoff review: exact-artifact matching
    )


def timing_stats(trusted, samples: List[float]) -> Dict[str, Any]:
    result = trusted["TimingResult"](samples)
    return {
        "median_ms": result.median_ms,
        "mean_ms": result.mean_ms,
        "p90_ms": result.p90_ms,
        "min_ms": result.min_ms,
        "n_samples": len(samples),
        "raw_samples_ms": [round(s, 6) for s in samples],
    }


def accuracy_to_dict(result) -> Dict[str, Any]:
    return {
        "passed": result.passed,
        "failed_elements": result.failed_elements,
        "total_elements": result.total_elements,
        "max_abs_error": result.max_abs_error,
        "max_relative_error": result.max_relative_error,
        "mean_abs_error": result.mean_abs_error,
    }


class Evaluation:
    """Owns the trusted objects for one run. Order of operations matters:
    everything trusted is created before candidate code executes."""

    def __init__(self, shape: Dict[str, Any], args, torch):
        self.torch = torch
        self.args = args
        self.shape = shape
        self.otb = load_official()
        self.trusted = snapshot_trusted(self.otb)

        config = self.trusted["TransformerConfig"](
            batch_size=shape["batch_size"], seq_len=shape["seq_len"],
            d_model=shape["d_model"], num_heads=shape["num_heads"],
            ffn_dim=shape["ffn_dim"], num_layers=shape["num_layers"],
            causal=shape["causal"],
        )
        config.validate()
        self.config = config
        self.device = torch.device("cuda")
        self.dtype = self.trusted["resolve_dtype"](args.dtype)

        torch.manual_seed(OFFICIAL_DEFAULTS["seed"])
        torch.cuda.manual_seed_all(OFFICIAL_DEFAULTS["seed"])
        torch.set_float32_matmul_precision(OFFICIAL_DEFAULTS["matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = OFFICIAL_DEFAULTS["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = OFFICIAL_DEFAULTS["allow_tf32"]

        self.baseline = self.trusted["BaselineTransformer"](config)
        self.baseline_cpu_state = {
            k: v.detach().clone() for k, v in self.baseline.state_dict().items()
        }
        self.baseline = self.baseline.to(device=self.device, dtype=self.dtype).eval()

        # Invariance probe: computed before candidate code exists in-process.
        with torch.inference_mode():
            self.probe_x, self.probe_mask = self.trusted["generate_random_case"](
                config=config, device=self.device, dtype=self.dtype,
                seed=INVARIANCE_SEED,
                padding_ratio=OFFICIAL_DEFAULTS["padding_ratio"],
                input_scale=OFFICIAL_DEFAULTS["input_scale"],
            )
            self.probe_reference = self.baseline(self.probe_x, self.probe_mask).clone()

    def check_invariance(self, stage: str) -> None:
        with self.torch.inference_mode():
            now = self.baseline(self.probe_x, self.probe_mask)
            if not self.torch.equal(now, self.probe_reference):
                raise TamperError(
                    f"TAMPER DETECTED ({stage}): baseline output on the stashed "
                    "probe input changed after candidate code was loaded. The "
                    "candidate has modified trusted state (official module, "
                    "baseline, or torch math). Run aborted; nothing recorded as "
                    "a result."
                )

    def attach_candidate(self, candidate_module) -> None:
        if candidate_module is None:
            candidate = self.trusted["BaselineTransformer"](self.config)  # calibration twin
        else:
            candidate = candidate_module.build(self.otb, self.config)
        if candidate_module is not None and hasattr(candidate_module, "copy_weights"):
            candidate_module.copy_weights(self.baseline, candidate)
        else:
            missing = candidate.load_state_dict(self.baseline_cpu_state, strict=True)
            del missing
        self.candidate = candidate.to(device=self.device, dtype=self.dtype).eval()
        self.check_invariance("after candidate load")

    def fresh_case(self, seed: int):
        return self.trusted["generate_random_case"](
            config=self.config, device=self.device, dtype=self.dtype, seed=seed,
            padding_ratio=OFFICIAL_DEFAULTS["padding_ratio"],
            input_scale=OFFICIAL_DEFAULTS["input_scale"],
        )

    def _candidate_checked(self, x, mask):
        """Call the candidate and verify it did not mutate its inputs — a
        candidate that rewrites x or mask can corrupt later comparisons or
        timing (codex handoff review finding 1). Raises TamperError on mutation."""
        torch = self.torch
        x_snapshot = x.clone()
        mask_snapshot = mask.clone()
        out = self.candidate(x, mask)
        if not torch.equal(x, x_snapshot) or not torch.equal(mask, mask_snapshot):
            raise TamperError(
                "TAMPER DETECTED (input mutation): the candidate modified its "
                "input tensor or mask in place. Run aborted; nothing recorded "
                "as a result."
            )
        return out

    # ---------------- correctness ----------------

    def run_correctness(self) -> Dict[str, Any]:
        torch = self.torch
        trusted = self.trusted
        trials = []
        all_passed = True
        with torch.inference_mode():
            x = mask = None
            for trial in range(OFFICIAL_DEFAULTS["accuracy_trials"]):
                x, mask = self.fresh_case(OFFICIAL_DEFAULTS["seed"] + trial)
                reference = self.baseline(x, mask)
                output = self._candidate_checked(x, mask)
                result = trusted["compare_outputs"](
                    reference, output,
                    rtol=OFFICIAL_DEFAULTS["rtol"], atol=OFFICIAL_DEFAULTS["atol"],
                )
                trials.append(accuracy_to_dict(result))
                all_passed &= result.passed

            # Tripwire 1: SAME memory address, materially different values.
            # An address-keyed cache returns the stale answer and fails hard.
            gen = torch.Generator(device=self.device)
            gen.manual_seed(OFFICIAL_DEFAULTS["seed"] + 777000)
            x.copy_(torch.randn(x.shape, generator=gen, device=self.device, dtype=x.dtype))
            ref_same_addr = self.baseline(x, mask)
            out_same_addr = self._candidate_checked(x, mask)
            trip_same_addr = trusted["compare_outputs"](
                ref_same_addr, out_same_addr,
                rtol=OFFICIAL_DEFAULTS["rtol"], atol=OFFICIAL_DEFAULTS["atol"],
            )

            # Tripwire 2: same values, fresh memory address.
            x_clone = x.clone()
            ref_clone = self.baseline(x_clone, mask)
            out_clone = self._candidate_checked(x_clone, mask)
            trip_clone = trusted["compare_outputs"](
                ref_clone, out_clone,
                rtol=OFFICIAL_DEFAULTS["rtol"], atol=OFFICIAL_DEFAULTS["atol"],
            )

        all_passed &= trip_same_addr.passed and trip_clone.passed
        return {
            "passed": all_passed,
            "trials": trials,
            "tripwire_same_address_new_values": accuracy_to_dict(trip_same_addr),
            "tripwire_clone_fresh_address": accuracy_to_dict(trip_clone),
        }

    # ---------------- timing ----------------

    def _wall_per_iter_ms(self, model, x, mask, iterations: int) -> float:
        torch = self.torch
        with torch.inference_mode():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iterations):
                model(x, mask)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / iterations

    def _anti_cache_pass(self, static_median_ms: float) -> Dict[str, Any]:
        """Time the candidate with input values re-randomized IN PLACE before
        EVERY timed call (same address, values never repeat — Sol audit round 3,
        minor finding 2), with periodic output checks against freshly computed
        references. Any caching strategy — address-keyed, value-keyed, or
        adaptive — either produces wrong outputs here or a timing far from the
        fixed-input timing."""
        torch = self.torch
        trusted = self.trusted
        samples: List[float] = []
        checks_passed = True
        total_iters = ANTI_CACHE_ITERS
        gen = torch.Generator(device=self.device)
        with torch.inference_mode():
            x, mask = self.fresh_case(OFFICIAL_DEFAULTS["seed"] + 888000)
            for i in range(total_iters):
                # Fresh, never-before-seen values at the same address (untimed).
                gen.manual_seed(OFFICIAL_DEFAULTS["seed"] + 999000 + i)
                x.copy_(torch.randn(x.shape, generator=gen, device=self.device,
                                    dtype=x.dtype))
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                self.candidate(x, mask)
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end))

                # Periodic output check against a freshly computed reference.
                if i % 10 == 0:
                    ref = self.baseline(x, mask)
                    out = self._candidate_checked(x, mask)
                    check = trusted["compare_outputs"](
                        ref, out,
                        rtol=OFFICIAL_DEFAULTS["rtol"], atol=OFFICIAL_DEFAULTS["atol"],
                    )
                    checks_passed &= check.passed

        anti_median = trusted["TimingResult"](samples).median_ms
        ratio = anti_median / static_median_ms if static_median_ms > 0 else float("inf")
        suspicious = (
            anti_median > static_median_ms * ANTI_CACHE_MAX_RATIO + ANTI_CACHE_SLACK_MS
        ) or not checks_passed
        return {
            "median_ms": anti_median,
            "ratio_vs_static": ratio,
            "outputs_correct": checks_passed,
            "suspicious": suspicious,
            "raw_samples_ms": [round(s, 6) for s in samples],
        }

    def run_timing(self) -> Dict[str, Any]:
        torch = self.torch
        trusted = self.trusted
        args = self.args
        with torch.inference_mode():
            x, mask = self.fresh_case(OFFICIAL_DEFAULTS["seed"] + 100000)

        timing_x_snapshot = x.clone()
        timing_mask_snapshot = mask.clone()
        trusted["warmup_model"](self.baseline, x, mask, args.warmup, self.device)
        trusted["warmup_model"](self.candidate, x, mask, args.warmup, self.device)

        baseline_samples: List[float] = []
        candidate_samples: List[float] = []
        for round_index in range(args.rounds):
            order = (
                [(self.baseline, baseline_samples), (self.candidate, candidate_samples)]
                if round_index % 2 == 0
                else [(self.candidate, candidate_samples), (self.baseline, baseline_samples)]
            )
            for model, sink in order:
                sink.extend(
                    trusted["benchmark_once"](model, x, mask, args.repeats, self.device)
                )

        baseline_stats = timing_stats(trusted, baseline_samples)
        candidate_stats = timing_stats(trusted, candidate_samples)

        # Input-integrity check: the timing input must be unchanged after all
        # warmup and timed rounds (codex handoff review finding 1).
        if not torch.equal(x, timing_x_snapshot) or not torch.equal(mask, timing_mask_snapshot):
            raise TamperError(
                "TAMPER DETECTED (timing-input mutation): input tensor or mask "
                "changed during the timed rounds. Run aborted."
            )

        # Tripwire 3: wall-clock cross-check (catches work hidden from events).
        wall_iters = max(10, args.repeats // 2)
        baseline_wall = self._wall_per_iter_ms(self.baseline, x, mask, wall_iters)
        candidate_wall = self._wall_per_iter_ms(self.candidate, x, mask, wall_iters)
        wall_suspicious = (
            candidate_stats["median_ms"] * WALL_SUSPICION_FACTOR + WALL_SUSPICION_SLACK_MS
            < candidate_wall
        )

        # Tripwire 4: anti-cache pass (fresh values every round).
        anti_cache = self._anti_cache_pass(candidate_stats["median_ms"])

        self.check_invariance("after timing")

        speedup = baseline_stats["median_ms"] / candidate_stats["median_ms"]
        return {
            "baseline": baseline_stats,
            "candidate": candidate_stats,
            "speedup": speedup,
            "wall_check": {
                "baseline_wall_ms_per_iter": baseline_wall,
                "candidate_wall_ms_per_iter": candidate_wall,
                "wall_speedup": baseline_wall / candidate_wall if candidate_wall > 0 else 0.0,
                "suspicious": wall_suspicious,
            },
            "anti_cache_check": anti_cache,
        }


# ---------------- journal / leaderboard ----------------

def read_journal() -> List[Dict[str, Any]]:
    if not JOURNAL_PATH.exists():
        return []
    entries = []
    malformed = 0
    for line in JOURNAL_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1
    if malformed:
        # Never silently drop ledger lines (codex handoff review finding).
        print(f"[warning] JOURNAL.jsonl contains {malformed} malformed line(s) — "
              "investigate before trusting derived results", file=sys.stderr)
    return entries


def latest_calibration(entries, reference_entry) -> Optional[Dict[str, Any]]:
    want = calibration_match_key(reference_entry)
    for entry in reversed(entries):
        if entry.get("type") == "calibration" and calibration_match_key(entry) == want:
            return entry
    return None


def append_journal(entry: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def read_verdicts() -> Dict[str, str]:
    """audits/verdicts.jsonl lines are written by the audit RECORDER (not by Sol
    directly): {"entry_id": <journal entry id>, "verdict": <Sol verdict>,
    "source_log": <raw log path>}. Sol's own schema has no entry_id — the
    recorder binds each verdict to the entry it audited (codex review finding 8)."""
    verdicts: Dict[str, str] = {}
    malformed = 0
    if VERDICTS_PATH.exists():
        for line in VERDICTS_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                v = json.loads(line)
                verdicts[v["entry_id"]] = v["verdict"]
            except Exception:
                malformed += 1
    if malformed:
        print(f"[warning] verdicts.jsonl contains {malformed} malformed line(s)",
              file=sys.stderr)
    return verdicts


def regenerate_leaderboard() -> str:
    entries = read_journal()
    verdicts = read_verdicts()
    runs = [e for e in entries if e.get("type") == "candidate"]
    keys = sorted({(e["shape_id"], e["env"]["gpu"], e["dtype"]) for e in runs})

    lines = [
        "# LEADERBOARD (auto-generated by the trusted runner — do not edit)",
        "",
        f"Regenerated: {time.strftime('%Y-%m-%d %H:%M:%S')} | harness {HARNESS_VERSION}",
        "",
    ]
    for shape_id, gpu, dtype in keys:
        group = [e for e in runs if (e["shape_id"], e["env"]["gpu"], e["dtype"]) == (shape_id, gpu, dtype)]
        # Champion eligibility: promoted entries measured by EXACTLY this runner
        # file (version AND sha) — any harness edit retires prior champions to
        # "legacy" until re-measured (codex handoff review finding 9).
        current_runner_sha = sha256_file(Path(__file__).resolve())
        promoted = [
            e for e in group
            if e.get("promoted")
            and e.get("env", {}).get("harness_version") == HARNESS_VERSION
            and e.get("env", {}).get("runner_sha256") == current_runner_sha
        ]
        champion_id = None
        if promoted:
            champion_id = max(promoted, key=lambda e: e["timing"]["speedup"])["entry_id"]
        lines.append(f"## Shape {shape_id} | {gpu} | {dtype}")
        lines.append("")
        lines.append("| impl | speedup | base ms | cand ms | correct | promoted | audit | harness | entry |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for e in sorted(group, key=lambda e: -(e["timing"]["speedup"] if e.get("timing") else 0)):
            star = " ★" if e["entry_id"] == champion_id else ""
            timing = e.get("timing")
            lines.append(
                "| {impl}{star} | {sp} | {b} | {c} | {ok} | {prom} | {audit} | {hv} | {eid} |".format(
                    impl=e["impl"]["name"], star=star,
                    sp=f"{timing['speedup']:.3f}x" if timing else "-",
                    b=f"{timing['baseline']['median_ms']:.4f}" if timing else "-",
                    c=f"{timing['candidate']['median_ms']:.4f}" if timing else "-",
                    ok="PASS" if e["correctness"]["passed"] else "FAIL",
                    prom=(
                        "yes"
                        if e.get("promoted")
                        and e.get("env", {}).get("harness_version") == HARNESS_VERSION
                        and e.get("env", {}).get("runner_sha256") == current_runner_sha
                        else ("legacy" if e.get("promoted") else "no")
                    ),
                    audit=verdicts.get(e["entry_id"], "unaudited"),
                    hv=e.get("env", {}).get("harness_version", "?"),
                    eid=e["entry_id"],
                )
            )
        cal = [e for e in entries if e.get("type") == "calibration"
               and (e["shape_id"], e["env"]["gpu"], e["dtype"]) == (shape_id, gpu, dtype)]
        if cal:
            noise = cal[-1]["noise"]["noise"]
            lines.append("")
            lines.append(f"Noise floor (baseline vs itself): {noise:.4f} "
                         f"(promotion needs speedup > {1 + max(PROMOTION_MIN_MARGIN, PROMOTION_NOISE_FACTOR * noise):.3f}x)")
        lines.append("")

    text = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.write_text(text, encoding="utf-8")
    return text


def new_entry_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def is_primary(args) -> bool:
    return (
        args.dtype == OFFICIAL_DEFAULTS["dtype"]
        and args.warmup == OFFICIAL_DEFAULTS["warmup"]
        and args.repeats == OFFICIAL_DEFAULTS["repeats"]
        and args.rounds == OFFICIAL_DEFAULTS["benchmark_rounds"]
    )


def cmd_run(args) -> int:
    global JOURNAL_PATH
    test_ledger = getattr(args, "ledger", None)
    if test_ledger:
        JOURNAL_PATH = Path(test_ledger).resolve()
    integrity = verify_hashes()
    import torch  # noqa: PLC0415
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    shape = load_shape(args.shape)
    if shape["id"] == 14:
        raise SystemExit(
            "shape 14 has no evaluation path in this runner version: the official "
            "baseline cannot run it (multi-TB attention table) and the chunked "
            "reference oracle is not built yet (PLAN.md Stage 4). Refusing to "
            "pretend otherwise."
        )

    calibration_mode = args.impl is None
    evaluation = Evaluation(shape, args, torch)

    candidate_module = None
    impl_info: Dict[str, Any] = {"name": "__calibration__", "path": None, "sha256": None}
    if not calibration_mode:
        impl_path = Path(args.impl).resolve()
        candidate_module, source_sha = load_candidate(impl_path)
        impl_info = {
            "name": getattr(candidate_module, "NAME", impl_path.stem),
            "path": str(impl_path.relative_to(ROOT)),
            "sha256": source_sha,
            "description": getattr(candidate_module, "DESCRIPTION", ""),
        }
    evaluation.attach_candidate(candidate_module)

    correctness = evaluation.run_correctness()

    entry: Dict[str, Any] = {
        "entry_id": new_entry_id(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "type": "calibration" if calibration_mode else "candidate",
        "shape_id": shape["id"],
        "shape": {k: v for k, v in shape.items() if k != "notes"},
        "dtype": args.dtype,
        "profile": "primary" if is_primary(args) else "custom",
        "impl": impl_info,
        "official": {**integrity, "defaults": OFFICIAL_DEFAULTS},
        "timing_args": {"warmup": args.warmup, "repeats": args.repeats, "rounds": args.rounds},
        "env": env_fingerprint(torch),
        "correctness": correctness,
    }

    if not correctness["passed"] and not calibration_mode:
        entry["timing"] = None
        entry["promoted"] = False
        entry["note"] = "timing skipped: correctness failed"
        append_journal(entry)
        if not test_ledger:
            regenerate_leaderboard()
        print(json.dumps({"entry_id": entry["entry_id"], "correct": False,
                          "promoted": False, "note": entry["note"]}, indent=2))
        return 2

    timing = evaluation.run_timing()
    entry["timing"] = timing

    if calibration_mode:
        noise = abs(1.0 - timing["speedup"])
        entry["noise"] = {
            "noise": noise,
            "promotion_threshold": 1 + max(PROMOTION_MIN_MARGIN, PROMOTION_NOISE_FACTOR * noise),
        }
        entry["promoted"] = False
    else:
        clean = (
            not timing["wall_check"]["suspicious"]
            and not timing["anti_cache_check"]["suspicious"]
        )
        if entry["profile"] != "primary":
            entry["promoted"] = False
            entry["note"] = "not promotion-eligible: non-primary profile (custom dtype/timing args)"
        else:
            entries = read_journal()
            cal = latest_calibration(entries, entry)
            if cal is None:
                entry["promoted"] = False
                entry["note"] = ("no calibration matching this shape/env/timing-args — "
                                 "run calibrate first")
            else:
                threshold = cal["noise"]["promotion_threshold"]
                entry["calibration_ref"] = cal["entry_id"]
                entry["promotion_threshold"] = threshold
                entry["promoted"] = timing["speedup"] > threshold and clean
                if not clean:
                    entry["note"] = ("NOT promoted: cross-check flagged suspicious "
                                     "timing (wall or anti-cache)")

    append_journal(entry)
    if not test_ledger:
        regenerate_leaderboard()
    print(json.dumps({
        "entry_id": entry["entry_id"],
        "type": entry["type"],
        "shape": shape["id"],
        "impl": impl_info["name"],
        "correct": correctness["passed"],
        "speedup": timing["speedup"],
        "wall_suspicious": timing["wall_check"]["suspicious"],
        "anti_cache_suspicious": timing["anti_cache_check"]["suspicious"],
        "anti_cache_ratio": round(timing["anti_cache_check"]["ratio_vs_static"], 4),
        "promoted": entry.get("promoted", False),
        "note": entry.get("note", ""),
    }, indent=2))
    return 0


def cmd_packet(args) -> int:
    entries = read_journal()
    entry = next((e for e in entries if e["entry_id"] == args.id), None)
    if entry is None:
        raise SystemExit(f"entry {args.id} not found in journal")
    packet: Dict[str, Any] = {"entry": entry}
    impl_path = entry.get("impl", {}).get("path")
    if impl_path:
        current_sha = sha256_file(ROOT / impl_path)
        journaled_sha = entry.get("impl", {}).get("sha256")
        packet["candidate_source"] = (ROOT / impl_path).read_text()
        packet["candidate_source_sha256_now"] = current_sha
        packet["candidate_source_matches_journal"] = current_sha == journaled_sha
        if current_sha != journaled_sha:
            packet["warning"] = ("candidate file on disk differs from the version "
                                 "this journal entry measured — source above is "
                                 "the CURRENT file, not the measured one")
    if entry["type"] == "candidate":
        cal_id = entry.get("calibration_ref")
        packet["calibration"] = next((e for e in entries if e["entry_id"] == cal_id), None)
    packet["shapes_json"] = json.loads(SHAPES_PATH.read_text())
    packet["manifest"] = json.loads(MANIFEST_PATH.read_text())
    PACKETS_DIR.mkdir(parents=True, exist_ok=True)
    out = PACKETS_DIR / f"{args.id}.json"
    out.write_text(json.dumps(packet, indent=2, sort_keys=True))
    print(str(out))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Trusted runner for TechJam Track 3")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="print environment fingerprint")
    sub.add_parser("check", help="verify official file hashes")

    def add_run_args(p):
        p.add_argument("--shape", type=int, required=True)
        p.add_argument("--dtype", default=OFFICIAL_DEFAULTS["dtype"],
                       choices=("float32", "float16", "bfloat16"))
        p.add_argument("--warmup", type=int, default=OFFICIAL_DEFAULTS["warmup"])
        p.add_argument("--repeats", type=int, default=OFFICIAL_DEFAULTS["repeats"])
        p.add_argument("--rounds", type=int, default=OFFICIAL_DEFAULTS["benchmark_rounds"])
        p.add_argument("--ledger", default=None,
                       help="alternate journal path for test/red-team runs "
                            "(production leaderboard is not regenerated)")

    p_cal = sub.add_parser("calibrate", help="baseline-vs-itself noise floor")
    add_run_args(p_cal)

    p_run = sub.add_parser("run", help="evaluate a candidate implementation")
    add_run_args(p_run)
    p_run.add_argument("--impl", required=True, help="path to candidate .py file")

    sub.add_parser("leaderboard", help="regenerate LEADERBOARD.md")

    p_pk = sub.add_parser("packet", help="write a neutral evidence packet")
    p_pk.add_argument("--id", required=True)

    args = parser.parse_args()

    if args.cmd == "env":
        verify_hashes()
        import torch  # noqa: PLC0415
        print(json.dumps(env_fingerprint(torch), indent=2))
        return 0
    if args.cmd == "check":
        print(json.dumps(verify_hashes(), indent=2))
        return 0
    if args.cmd == "calibrate":
        args.impl = None
        return cmd_run(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "leaderboard":
        regenerate_leaderboard()
        print(str(LEADERBOARD_PATH))
        return 0
    if args.cmd == "packet":
        return cmd_packet(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
