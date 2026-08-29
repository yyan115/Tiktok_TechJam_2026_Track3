# Amendment bundle v1.1 (PROPOSAL — frozen runner untouched)

> **SUPERSEDED 29 Aug: the frozen runner is NOT edited. Shape-14 evidence comes from the independently pinned side evaluator (Project/tools/shape14_eval.py) with packets in Project/results_side/. Kept for history only.**


Drafted 28 Aug 2026 (Day 1) so the timeboxed re-freeze session with the user
is fast: review, approve, apply, Sol re-audit, pin update. Per the freeze
checklist, nothing here counts until that procedure completes.

Target: `Project/harness/runner.py` v1.0.2 (937 lines, sha pinned in
manifest.json). Three additions, no changes to existing measurement paths.

---

## A. MFU per result (webinar: score = weighted sum of per-shape MFUs)

**What**: every `run` entry gains `timing["mfu"]` = model FLOPs / (median
candidate seconds × device peak FLOPs), plus the same for the baseline.

**Model FLOPs** (per forward, matching the official architecture; causal
halves the attention terms):

```python
def model_flops(shape: Dict[str, Any]) -> float:
    B, S, d = shape["batch_size"], shape["seq_len"], shape["d_model"]
    ffn, L = shape["ffn_dim"], shape["num_layers"]
    attn_factor = 0.5 if shape["causal"] else 1.0
    per_layer = (
        8.0 * B * S * d * d              # QKV + out projections (4 GEMMs)
        + attn_factor * 4.0 * B * S * S * d  # scores + probs@V
        + 4.0 * B * S * d * ffn          # FFN in + out
    )
    return L * per_layer                 # norms/GELU negligible, stated
```

**Device peak**: `Project/device_peaks.json` (CREATED 29 Aug, outside the
locked harness dir) mapping GPU name substring → fp32 peak TFLOPS (RTX 3060
Ti: 16.2 + common rental cards, spec-sheet sourced). If the device is
unmapped, record `"mfu": null` — never guess a peak.

**Insertion**: in `cmd_run` after line 770 (`entry["timing"] = timing`):

```python
    peak = lookup_peak_tflops(entry["env"])          # from device_peaks.json
    if peak:
        fl = model_flops(shape)
        timing["mfu"] = {
            "candidate": fl / (timing["candidate"]["median_ms"] / 1e3) / (peak * 1e12),
            "baseline": fl / (timing["baseline"]["median_ms"] / 1e3) / (peak * 1e12),
            "peak_tflops_fp32": peak,
            "flops_model": fl,
        }
```

MFU is REPORTING ONLY — no promotion logic reads it.

## B. `official` subcommand (final acceptance via the untouched script)

**What**: `runner.py official --shape N --impl path` runs the pinned official
`torch_transformer_benchmark.py` (hash re-verified first) as a subprocess with
that shape's exact dials, a temp copy of the candidate installed the way the
official script expects, and records stdout + exit code as an
`"official_acceptance"` journal entry. Refuses shape 14 (baseline infeasible —
limitation stated in the entry, per PLAN Stage 5).

Sketch: `verify_hashes()` → build CLI args from shapes.json (batch, seq,
d_model, heads, ffn, layers, causal flag, atol/rtol defaults untouched) →
`subprocess.run([sys.executable, OFFICIAL_PATH, ...])` → parse the script's
own pass/fail output; the runner adds nothing to the measurement.

## C. Shape-14 evaluation path (chunked fp32 oracle)

**What**: replace the hard refusal at lines 717-723 with a dedicated path,
used ONLY for shape 14 and clearly typed `"oracle_reference"` in the journal:

1. Correctness: candidate attention vs the chunked fp32 streaming-softmax
   oracle (algorithm proven in `Project/tools/smokes/shape14_core_smoke.py`,
   28 Aug: 0 violations at seq=100k, max err 6.99e-4). Full-scale on the
   rental card; B=1/H=1 slices locally.
2. Timing: candidate-only wall+event timing (no baseline exists); entry
   records `"baseline": null` and the official-baseline limitation string
   from shapes.json notes verbatim.
3. Promotion rules do not apply (nothing to beat); the entry exists for MFU
   and correctness evidence.

## Re-freeze procedure (unchanged from freeze checklist)

1. User reviews this document and says go.
2. Apply to a COPY (`runner_v1.1.0.py`), full self-test: calibrate + one
   known champion re-run must reproduce v1.0.2 numbers bit-for-bit on the
   journal fields both versions share.
3. Sol blind review of the diff (1-2 rounds, timeboxed per STATE).
4. User approves pin update in manifest.json; old runner archived; locks
   re-verified on the new file.

## Open questions for the user

- MFU peak convention: fp32 CUDA-core peak (16.2 TF) even for internally-fp16
  candidates, or per-dtype peaks? Proposal: fp32 peak for everything —
  conservative, single denominator, matches the official fp32 profile.
- Shape-14 local slice evidence vs rental full-scale: report both, or
  rental-only in the judge-facing numbers? Proposal: both, labeled.
