# SPDX-License-Identifier: Apache-2.0
"""Independent draft noise for probabilistic DFlash drafting.

vLLM 0.28 keys the DFlash draft walk's Gumbel noise by (request seed, position,
candidate id). The rejection sampler's residual resample uses the same key, so
with ``draft_sample_method: probabilistic`` draft and residual share noise and
the output is not an exact sample of the target (argmax under-sampled by
0.3-0.4 pp, 5-9 sigma over ~0.5M tokens). The draft walk gets the request seeds
XOR a constant; the target sampler's seeds are untouched. Measured afterwards:
no bias (z -1.3 / +0.1), no acceptance loss. Greedy drafting draws no noise and
is left alone.
"""

DRAFT_SEED_XOR = 0x5DEECE66D


def independent_draft_seeds(speculator, seeds, dummy_run):
    """Seeds for the draft walk; ``seeds`` itself is not modified."""
    if speculator.speculative_config.draft_sample_method != "probabilistic":
        return seeds
    buffer = getattr(speculator, "_raon_draft_seed_buffer", None)
    if buffer is None or buffer.shape != seeds.shape or buffer.device != seeds.device:
        # A persistent buffer keeps the address stable for the captured draft graph.
        buffer = seeds.clone()
        speculator._raon_draft_seed_buffer = buffer
    buffer.copy_(seeds)
    buffer.bitwise_xor_(DRAFT_SEED_XOR)
    if not dummy_run and not getattr(speculator, "_raon_draft_seed_announced", False):
        print(f"[raon1_mini] {type(speculator).__name__}.propose: draft seeds XOR "
              f"{DRAFT_SEED_XOR:#x}", flush=True)
        speculator._raon_draft_seed_announced = True
    return buffer
