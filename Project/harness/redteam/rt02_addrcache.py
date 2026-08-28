"""RED TEAM test candidate — deliberately malicious, part of the evaluator's
test suite. Implements the exact cheat from the Stage-1 audit: compute honestly
whenever the input's memory address is NEW, return the cached output whenever
the address repeats (which is what benchmark timing loops do).

Expected runner behavior: caught. Either the same-address-new-values tripwire
fails it on correctness, or the anti-cache timed pass flags it.
"""

NAME = "rt02_addrcache"
DESCRIPTION = "Red-team: address-keyed output cache (fake near-zero latency)."


def build(otb, config):
    class AddressCachedTransformer(otb.UserOptimizedTransformer):
        def forward(self, x, valid_token_mask=None):
            cache = getattr(self, "_cheat_cache", None)
            if cache is None:
                cache = {}
                object.__setattr__(self, "_cheat_cache", cache)
            key = (x.data_ptr(), x.shape)
            if key in cache:
                return cache[key]
            out = super().forward(x, valid_token_mask)
            cache[key] = out
            return out

    return AddressCachedTransformer(config)
