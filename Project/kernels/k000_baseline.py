"""k000: the unmodified baseline, run through the candidate pipeline.

Purpose: sanity-check the whole referee. Expected result: correctness PASS,
speedup ~1.00x, NOT promoted (below the noise threshold by construction).
"""

NAME = "k000_baseline"
DESCRIPTION = "Unmodified baseline as candidate; pipeline sanity check."


def build(otb, config):
    return otb.UserOptimizedTransformer(config)
