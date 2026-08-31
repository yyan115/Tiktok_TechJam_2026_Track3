"""fp16-route correctness proof: shape 8's stack against the official baseline.

Shape 8 (B=64, d=1024, H=4, S=128, 4 layers, causal) is the only shipped shape
that takes `_fp16_forward`, and two changes landed on that route back to back:

  1. the residual adds dropped their explicit `.float()`, relying on type
     promotion to widen fp16 -> fp32 during the add rather than materialising a
     full fp32 temporary first;
  2. `_sub_triton_attention` stopped repacking. q, k and v now arrive as the raw
     transposed views of the qkv projection, batch and head became separate grid
     axes, and the output is allocated in [B,S,H,D] and written through its
     [B,H,S,D] transpose.

Both are meant to be BIT-IDENTICAL to what the route did before, not
approximations: (1) because fp16 -> fp32 is lossless and the addition is fp32
either way, (2) because the kernel reads the same values through different
strides and does the same arithmetic on them. That is a strong claim and it is
why this file exists -- a stride mistake in (2) would not be a small error, it
would be silent garbage, and a shape that fails the precision predicate scores
zero no matter how fast it is. The diagnostic lane checks nothing: `run_route`
in `Project/harness/profile_worker.py` just calls the model in a loop.

Note the route is fp16, so it is NOT expected to be bit-identical to the fp32
baseline -- the pass/fail line is the official predicate, and the printed
max_abs_err against a 2e-3 budget is the number to watch.

Correctness only. Nothing here is timed and nothing here may be quoted as a
speed result; the numbers that count come from the runner.
"""
import importlib.util
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import torch_transformer_benchmark as otb  # noqa: E402

SUBMISSION = f"{REPO}/Project/submission/torch_transformer_benchmark_submission.py"
ATOL, RTOL = 2e-3, 2e-2
SHAPE8 = dict(batch_size=64, seq_len=128, d_model=1024, num_heads=4,
              ffn_dim=1024, num_layers=4, causal=True)


def official_error_stats(reference, candidate):
    """Mirror the official finite AND (absolute OR relative) predicate."""
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"shape mismatch: reference={tuple(reference.shape)}, "
            f"candidate={tuple(candidate.shape)}"
        )
    ref = reference.detach().float()
    opt = candidate.detach().float()
    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()
    abs_ok = abs_error <= ATOL
    rel_ok = abs_error <= RTOL * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)
    failed = int((~passed_mask).sum().item())
    max_abs = (float("inf") if not bool(finite_mask.all())
               else abs_error.max().item())
    return failed, max_abs


def load_submission():
    spec = importlib.util.spec_from_file_location("sub", SUBMISSION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    torch.manual_seed(0)
    # Mirror profile_worker.NUMERICAL (Project/harness/profile_worker.py:270-277)
    # so the baseline this compares against is the one the runner scores, not a
    # stricter fp32 model that would make the fp16 route look worse than it is.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dev = torch.device("cuda")
    cfg = otb.TransformerConfig(**SHAPE8)
    cfg.validate()

    sub = load_submission()
    base = otb.BaselineTransformer(cfg).to(dev).eval()
    cand = sub.UserOptimizedTransformer(cfg).to(dev).eval()
    otb.copy_model_weights(base, cand)

    # The runner's own input, not an ad-hoc randn: same generator, same seed,
    # same padding ratio and scale (0.0 / 1.0, profile_worker.py:271-272), so
    # the mask path is exercised exactly as it ships.
    x, mask = otb.generate_random_case(cfg, dev, torch.float32, 0, 0.0, 1.0)

    with torch.no_grad():
        ref = base(x, mask)
        opt = cand(x, mask)
    torch.cuda.synchronize()

    bad, maxerr = official_error_stats(ref, opt)
    print(f"shape8 fp16-route vs official baseline: violations={bad} "
          f"max_abs_err={maxerr:.3e} budget={ATOL:.1e}")
    print("RESULT:", "PASS" if bad == 0 else "FAIL")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
