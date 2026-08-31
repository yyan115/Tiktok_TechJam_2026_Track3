# Exact-erf GELU: what it costs, and what an approximation buys (measured 31 Aug 2026)

## Why this note exists

The baseline computes GELU as `0.5 * x * (1 + erf(x / sqrt(2)))`. Our fused
tail kernel `_sub_attn_block_tail` reproduced that with Triton's `tl.erf`.
Nobody had ever asked what that single call costs, so it had never been on
any candidate list. It turned out to be the third-largest item in the
kernel that owns 35% of device time.

## Sizing it first, with a deliberately wrong kernel

`Project/kernels/probe_gelu_cost.py` (sha `98580170…`) is a copy of the
shipped tail kernel with the entire GELU replaced by the identity. It is
**numerically wrong on purpose** and is marked never-screen, never-promote:
its only job is to put a number on the upper bound.

| build | `_block_tail` device time (shape 1, 80 launches) |
| --- | --- |
| shipped, `tl.erf` | 46.5 us |
| probe, GELU deleted entirely | 41.4 us |

**GELU costs 5.1 us, 11% of the kernel.** That is the ceiling on any GELU
work. Without this the next step is a guess about which of three mechanisms
to chase; with it, the question is only "how much of 5.1 us is recoverable".

PROCESS: this is the cheap version of the ablation. The diagnostic lane
costs no attempts and does not check correctness, which is exactly what
makes a knowingly-wrong probe safe there and dangerous anywhere else —
see LESSONS 56, where an autotuner in the same lane picked a half-correct
kernel because nothing in that lane could tell it not to.

## Abramowitz & Stegun 7.1.26

The classic 5-term rational-times-exponential approximation to erf:

    t = 1 / (1 + 0.3275911 * |z|)
    erf(|z|) ~= 1 - t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5)))) * exp(-z*z)
    a1..a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429

Stated max absolute error **1.5e-7**, odd-symmetric so the sign is carried
by `tl.where(z >= 0, e, -e)`. One `exp`, five FMAs, one reciprocal.

Measured as `k027_fast_erf.py` (sha `80218c22…`):

| build | `_block_tail` | vs shipped |
| --- | --- | --- |
| shipped `tl.erf` | 46.5 us | |
| A&S 7.1.26 | **43.3 us** | **-3.2 us, -6.9%** |
| GELU deleted (floor) | 41.4 us | -5.1 us |

So the approximation recovers **63% of the total available GELU time**.
End to end on shape 1: **8.7551x vs 8.5217x, +2.7%**, inside the predicted
band, `correct: true`.

## The accuracy argument, and why it was not a trade

Max abs error on shape 1 was **0.000906 - 0.001052 across seeds, unchanged
from the exact-erf build**, against a 2e-3 budget. It did not move because
1.5e-7 is four orders of magnitude below the error the fp32 pipeline
already carries; the GELU approximation is not the binding term and cannot
become one. This is the opposite of the k008 int8 result in
[quantization-tolerance.md], where the approximation error WAS the binding
term and predictably blew the budget. The rule of thumb from that note —
estimate the error before building — is what said this one was free.

## What generalises

- **A transcendental inside a fused elementwise tail is a real line item**,
  not noise: 11% of a kernel that is 35% of device time.
- **Ablate before you optimise.** A wrong-but-fast probe in a lane that
  cannot promote turns "which of three mechanisms" into one number.
- `tl.erf` lowers to something materially slower than seven arithmetic ops
  plus an `exp` on SM 8.6. Anywhere else we call a libdevice
  transcendental in an inner loop is now a candidate by the same argument.
- The saving is **per element of the FFN hidden tensor**, so it should scale
  with token count rather than dilute: shape 1 (128 tokens) is the weakest
  possible case for it, and it still paid 2.7% end to end.

## Not established

- `k025` (halved FFN hidden axis, 48.4 us) and `k026` (persistent stride
  loop, 48.4 us) both lost to the 46.5 us shipped kernel. That is evidence
  those two specific rewrites cost more than they saved; it is **not**
  evidence that register pressure or pipelining are irrelevant here.
  Settling that needs occupancy counters, and `ncu` requires root and is
  deliberately absent from the post-LOCK allowlist.
- Whether a shorter approximation (A&S 7.1.25, 3 terms, ~2.5e-5) is faster
  still. Untested; the accuracy headroom is there, the time is not measured.

## Sources

- Abramowitz & Stegun, *Handbook of Mathematical Functions*, eq. 7.1.26.
- Measured in-repo: `probe_gelu_cost.py`, `k027_fast_erf.py`, profiles
  `profile-a54e004df00d0722d9acd155` (probe) and
  `profile-37b0c3b6d2cdc69b05fb56be` (fast erf).
