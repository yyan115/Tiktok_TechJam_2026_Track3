# Amendment v1.1 — exact code (against runner.py v1.0.2, 937 lines)

Companion to amendment_v1.1_bundle.md. Every block below is complete and
insertion-ready; line anchors reference v1.0.2 exactly. The Bash guard
(correctly) refuses to let the agent copy or touch the runner, so the person
applying this (you, after approval) makes these edits, then: self-test,
Sol diff review, manifest pin update.

## 0. Version

Line 72: `HARNESS_VERSION = "1.0.2"` → `"1.1.0"`.

## 1. MFU reporting (insert after OFFICIAL_DEFAULTS, ~line 100)

```python
DEVICE_PEAKS_PATH = PROJECT / "device_peaks.json"


def model_flops(shape: Dict[str, Any]) -> float:
    """Model FLOPs per forward, matching the official architecture; causal
    halves the attention terms. LayerNorm/GELU excluded (stated)."""
    B, S, d = shape["batch_size"], shape["seq_len"], shape["d_model"]
    ffn, L = shape["ffn_dim"], shape["num_layers"]
    attn_factor = 0.5 if shape["causal"] else 1.0
    per_layer = (
        8.0 * B * S * d * d
        + attn_factor * 4.0 * B * S * S * d
        + 4.0 * B * S * d * ffn
    )
    return L * per_layer


def lookup_peak_tflops(env: Dict[str, Any]) -> Optional[float]:
    """fp32 peak for the current GPU from device_peaks.json (substring match);
    None when unmapped — mfu is then recorded as null, never guessed."""
    try:
        peaks = json.loads(DEVICE_PEAKS_PATH.read_text())["peaks_fp32_tflops"]
    except Exception:
        return None
    gpu = env.get("gpu", "")
    for name, tflops in peaks.items():
        if name in gpu:
            return float(tflops)
    return None
```

In `cmd_run`, insert directly after `entry["timing"] = timing` (line 770):

```python
    peak = lookup_peak_tflops(entry["env"])
    if peak and timing.get("candidate") and timing.get("baseline"):
        fl = model_flops(shape)
        timing["mfu"] = {
            "candidate": fl / (timing["candidate"]["median_ms"] / 1e3) / (peak * 1e12),
            "baseline": fl / (timing["baseline"]["median_ms"] / 1e3) / (peak * 1e12),
            "peak_tflops_fp32": peak,
            "flops_model": fl,
        }
    else:
        timing["mfu"] = None
```

Reporting only; no promotion logic reads it. (The oracle path in §3 computes
its candidate-only MFU the same way.)

## 2. `official` subcommand (append near cmd_packet, ~line 848)

Runs the SUBMISSION script (official file with only the designated
UserOptimizedTransformer region replaced — built and provenance-verified by
Project/tools/build_submission.py) under the shape's exact dials, records
stdout verbatim. The runner adds nothing to the measurement.

```python
SUBMISSION_PATH = PROJECT / "submission" / "torch_transformer_benchmark_submission.py"


def verify_submission_provenance() -> Dict[str, Any]:
    """The acceptance property: outside the single replaced region, the
    submission's bytes are identical to the hash-verified official script."""
    official = OFFICIAL_TORCH.read_text()
    lines = official.splitlines(keepends=True)
    start = end = None
    for i, line in enumerate(lines):
        if start is None and line.startswith("class UserOptimizedTransformer(BaselineTransformer):"):
            start = i
        elif start is not None and line.startswith("def copy_model_weights("):
            end = i
            break
    if start is None or end is None:
        raise SystemExit("markers not found in official script")
    prefix = "".join(lines[:start]).encode()
    suffix = "".join(lines[end:]).encode()
    sub = SUBMISSION_PATH.read_bytes()
    if not (sub.startswith(prefix) and sub.endswith(suffix)):
        raise SystemExit("PROVENANCE FAILURE: submission bytes outside the "
                         "designated region differ from the official script")
    return {"submission_sha256": hashlib.sha256(sub).hexdigest(),
            "outside_region_identical": True}


def cmd_official(args) -> int:
    shape = load_shape(args.shape)
    if shape["id"] == 14:
        raise SystemExit("shape 14: the official baseline is infeasible "
                         "(multi-TB attention table); use `run` (oracle path)")
    provenance = verify_submission_provenance()
    cmd = [sys.executable, str(SUBMISSION_PATH),
           "--batch-size", str(shape["batch_size"]),
           "--seq-len", str(shape["seq_len"]),
           "--d-model", str(shape["d_model"]),
           "--heads", str(shape["num_heads"]),
           "--ffn-dim", str(shape["ffn_dim"]),
           "--layers", str(shape["num_layers"])]
    if shape["causal"]:
        cmd.append("--causal")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    import torch  # noqa: PLC0415
    entry = {
        "entry_id": new_entry_id(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "type": "official_acceptance",
        "shape_id": shape["id"],
        "shape": {k: v for k, v in shape.items() if k != "notes"},
        "official": {**verify_hashes(), "defaults": OFFICIAL_DEFAULTS},
        "submission": provenance,
        "env": env_fingerprint(torch),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-20000:],
        "stderr_tail": proc.stderr[-5000:],
    }
    append_journal(entry)
    print(proc.stdout[-2000:])
    print(f"official acceptance recorded: {entry['entry_id']} "
          f"(returncode {proc.returncode})")
    return proc.returncode


# in main(), with the other subparsers:
    p_off = sub.add_parser("official",
                           help="final acceptance via the untouched official "
                                "code paths (submission script, exact dials)")
    p_off.add_argument("--shape", type=int, required=True)
# and in the dispatch chain:
    if args.cmd == "official":
        return cmd_official(args)
```

## 3. Shape-14 oracle path (replace the refusal at lines 717-723)

```python
    if shape["id"] == 14:
        return cmd_shape14_oracle(args, shape, integrity, torch)
```

New function (append after cmd_run). Correctness against a chunked fp32
streaming-softmax oracle (algorithm proven in
Project/tools/smokes/shape14_core_smoke.py); candidate-only timing; NEVER
promoted; the official-baseline limitation recorded verbatim from
shapes.json notes.

```python
def _oracle_attention_forward(attn, x, valid_token_mask=None, causal=True,
                              q_chunk=4096, k_chunk=8192):
    """Baseline-exact attention in fp32 without the S x S table (streaming
    softmax over row/key chunks). Only the dense causal case is needed for
    shape 14 (padding_ratio=0.0 in the official defaults)."""
    import torch  # noqa: PLC0415
    batch, seq_len, _ = x.shape
    q = attn._split_heads(attn.q_proj(x))
    k = attn._split_heads(attn.k_proj(x))
    v = attn._split_heads(attn.v_proj(x))
    B, H, S, D = q.shape
    out = torch.empty_like(q)
    for qs in range(0, S, q_chunk):
        qe = min(qs + q_chunk, S)
        m = torch.full((B, H, qe - qs), float("-inf"), device=x.device)
        l = torch.zeros(B, H, qe - qs, device=x.device)
        acc = torch.zeros(B, H, qe - qs, D, device=x.device)
        k_end = qe if causal else S
        for ks in range(0, k_end, k_chunk):
            ke = min(ks + k_chunk, k_end)
            scores = torch.einsum("bhqd,bhkd->bhqk", q[:, :, qs:qe], k[:, :, ks:ke]) * attn.scale
            if causal:
                qidx = torch.arange(qs, qe, device=x.device)[:, None]
                kidx = torch.arange(ks, ke, device=x.device)[None, :]
                scores = scores.masked_fill(kidx > qidx, float("-inf"))
            m_new = torch.maximum(m, scores.amax(dim=-1))
            alpha = torch.exp(m - m_new)
            p = torch.exp(scores - m_new[..., None])
            l = l * alpha + p.sum(dim=-1)
            acc = acc * alpha[..., None] + torch.einsum("bhqk,bhkd->bhqd", p, v[:, :, ks:ke])
            m = m_new
        out[:, :, qs:qe] = acc / l[..., None]
    context = out.transpose(1, 2).contiguous().view(batch, seq_len, attn.d_model)
    return attn.out_proj(context)


def cmd_shape14_oracle(args, shape, integrity, torch) -> int:
    """Shape-14 evaluation: candidate vs the chunked fp32 oracle (correctness)
    plus candidate-only timing. Requires --batch-slice on memory-limited
    cards; records the official-baseline limitation verbatim."""
    import types  # noqa: PLC0415
    evaluation = Evaluation(  # noqa: F841 — construction validates env/config
        {**shape, "batch_size": max(1, args.batch_slice)}, args, torch)
    ev = evaluation
    # Oracle model: baseline modules with attention.forward swapped for the
    # chunked implementation — same weights, same math, no S x S table.
    oracle = ev.trusted["BaselineTransformer"](ev.config)
    oracle.load_state_dict(ev.baseline_cpu_state, strict=True)
    oracle = oracle.to(device=ev.device, dtype=ev.dtype).eval()
    for layer in oracle.layers:
        layer.attention.forward = types.MethodType(_oracle_attention_forward,
                                                   layer.attention)
    candidate_module, source_sha = load_candidate(Path(args.impl).resolve())
    candidate = candidate_module.build(ev.otb, ev.config)
    candidate.load_state_dict(ev.baseline_cpu_state, strict=True)
    candidate = candidate.to(device=ev.device, dtype=ev.dtype).eval()

    trials = []
    all_passed = True
    with torch.inference_mode():
        for trial in range(OFFICIAL_DEFAULTS["accuracy_trials"]):
            x, mask = ev.fresh_case(OFFICIAL_DEFAULTS["seed"] + trial)
            ref = oracle(x, mask)
            out = candidate(x, mask)
            res = ev.trusted["compare_outputs"](
                ref, out, rtol=OFFICIAL_DEFAULTS["rtol"],
                atol=OFFICIAL_DEFAULTS["atol"])
            trials.append(accuracy_to_dict(res))
            all_passed &= res.passed

        # Candidate-only timing (no runnable baseline exists at this shape).
        ev.trusted["warmup_model"](candidate, x, mask, args.warmup, ev.device)
        samples = []
        for _ in range(args.rounds):
            samples.extend(ev.trusted["benchmark_once"](
                candidate, x, mask, args.repeats, ev.device))

    stats = timing_stats(ev.trusted, samples)
    peak = lookup_peak_tflops(env_fingerprint(torch))
    sliced_shape = {**shape, "batch_size": max(1, args.batch_slice)}
    entry = {
        "entry_id": new_entry_id(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "type": "oracle_reference",
        "shape_id": shape["id"],
        "shape": {k: v for k, v in sliced_shape.items() if k != "notes"},
        "batch_slice": max(1, args.batch_slice),
        "dtype": args.dtype,
        "impl": {"name": getattr(candidate_module, "NAME", "candidate"),
                 "path": args.impl, "sha256": source_sha},
        "official": {**integrity, "defaults": OFFICIAL_DEFAULTS},
        "env": env_fingerprint(torch),
        "correctness": {"passed": all_passed, "trials": trials,
                        "reference": "chunked fp32 streaming-softmax oracle"},
        "timing": {"candidate": stats, "baseline": None, "speedup": None,
                   "mfu": ({"candidate": model_flops(sliced_shape)
                            / (stats["median_ms"] / 1e3) / (peak * 1e12),
                            "peak_tflops_fp32": peak} if peak else None)},
        "promoted": False,
        "limitation": shape.get("notes", ""),
    }
    append_journal(entry)
    print(json.dumps({"entry_id": entry["entry_id"], "correct": all_passed,
                      "median_ms": stats["median_ms"],
                      "batch_slice": entry["batch_slice"]}, indent=2))
    return 0 if all_passed else 2


# in add_run_args (or run's parser): 
    p.add_argument("--batch-slice", type=int, default=32,
                   help="shape-14 oracle path only: evaluate this many batch "
                        "rows (full shape = 32; 8GB cards need 1-2)")
```

## Self-test after applying (before Sol review)

1. `runner.py check` — pin mismatch EXPECTED until manifest updated; update
   manifest per the freeze procedure, then re-run: green.
2. `runner.py calibrate --shape 3` + `runner.py run --shape 3 --impl
   Project/kernels/k009_fused_tuned.py` — every pre-v1.1 journal field
   byte-compatible; new `mfu` block present.
3. `runner.py official --shape 3` — official_acceptance entry, returncode 0.
4. `runner.py run --shape 14 --impl Project/kernels/k006_fp16_hd128.py
   --batch-slice 1` on the 3060 Ti — oracle path green locally.
