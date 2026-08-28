"""RED TEAM test candidate — deliberately malicious, part of the evaluator's
test suite. It monkeypatches the baseline's attention forward to return zeros,
which would make any candidate 'match' a sabotaged reference.

Expected runner behavior: TAMPER DETECTED abort (invariance probe), nothing
recorded as a result.
"""

NAME = "rt01_monkeypatch"
DESCRIPTION = "Red-team: sabotages the baseline math after the hash check."


def build(otb, config):
    original_forward = otb.BaselineSelfAttention.forward

    def sabotaged_forward(self, x, valid_token_mask=None, causal=False):
        return original_forward(self, x, valid_token_mask, causal) * 0.0

    otb.BaselineSelfAttention.forward = sabotaged_forward
    return otb.UserOptimizedTransformer(config)
