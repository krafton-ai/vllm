"""Keep a TP1 DFlash core replicated while its target uses tensor parallelism.

The pinned vLLM loader otherwise constructs the draft with the target TP group,
even when draft_tensor_parallel_size=1. The five draft KV heads cannot shard
over TP2. Only draft construction/loading uses singleton groups: its embedding
and LM head are subsequently shared from the target by native vLLM, retaining
their normal TP collectives. Draft dense layers cache TP1 at construction.
"""

from contextlib import contextmanager

_SINGLETON = None


@contextmanager
def draft_group(state, singleton):
    target = state._TP
    state._TP = singleton
    try:
        yield
    finally:
        state._TP = target


def get_draft_model(get_model, *, vllm_config, model_config):
    """Build the DFlash draft; a TP1 draft under target TP uses singleton groups."""
    global _SINGLETON
    from vllm.config import replace
    from vllm.distributed import parallel_state as state

    target_tp = vllm_config.parallel_config.tensor_parallel_size
    spec = vllm_config.speculative_config
    if target_tp == 1 or spec.draft_parallel_config.tensor_parallel_size != 1:
        return get_model(vllm_config=vllm_config, model_config=model_config)
    if _SINGLETON is None:
        world = state.get_world_group()
        _SINGLETON = state.init_model_parallel_group(
            [[rank] for rank in range(world.world_size)], world.local_rank,
            "nccl", group_name="raon_draft_tp", use_device_communicator=False)
    draft_config = replace(vllm_config, parallel_config=replace(
        vllm_config.parallel_config, tensor_parallel_size=1))
    with draft_group(state, _SINGLETON):
        model = get_model(vllm_config=draft_config, model_config=model_config)
    if getattr(model, "has_own_embed_tokens", False) or getattr(model, "has_own_lm_head", False):
        raise ValueError("Replicated TP1 DFlash requires target-shared embedding and LM head")
    print(f"[raon-parallel] target TP{target_tp}; replicated TP1 draft core per rank; "
          "target-shared embedding/LM head retain target TP", flush=True)
    return model
