# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split mixed prefill + decode KDA batches so decode rows take the decode path.

VLLM_RAON_KDA_SPLIT_DECODE_SUFFIX=1 (default off; also selected by
VLLM_RAON_OPTIMIZATIONS on supported devices) applies to scheduler steps
that contain both ordinary decode requests and prefill chunks and no
speculative decodes. The original path sends every row through the varlen
prefill kernels. With the option, the leading plain-decode rows run through the
fused decode step (kda_sgl_fused.try_decode, in-place state update) and only
the remaining rows run through the original prefill path.

Everything that decides whether the split is legal is computed here, from host
metadata, when the step metadata is built:

* the batch is exactly ``[decode rows][prefill rows]`` with no speculative
  rows and no zero-length padding rows;
* every decode row has query length 1 and is not prefilling, so it owns an
  initialized recurrent/conv state;
* no later row looks like a decode row (the layout is not interleaved).

A step that does not satisfy this carries no plan and takes the original path
unchanged. ``try_decode`` returns False only before it launches any kernel, so
the layer may still fall back to the original full path afterwards without
updating any state twice.

The planning functions are plain Python so they can be tested without a GPU.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm.model_executor.layers import raon_optimizations
from vllm.platforms import current_platform

_ENV = "VLLM_RAON_KDA_SPLIT_DECODE_SUFFIX"
_PLAN_ATTR = "_raon_kda_split_plan"


def explicit() -> bool:
    """True when the option was requested by its own environment variable."""
    value = os.environ.get(_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{_ENV} must be 0 or 1; got {value!r}")
    return value == "1"


def fused_decode_armed(vllm_config) -> bool:
    """The fused KDA decode step is on and no speculative decoding is set."""
    if os.environ.get("RAON_KDA_SGL_FUSED", "0") != "1":
        return False
    spec_config = getattr(vllm_config, "speculative_config", None)
    spec_k = getattr(spec_config, "num_speculative_tokens", 0) or 0
    return spec_k == 0


def enabled(device_id: int | None = None) -> bool:
    """The option's own variable, or the device profile when it applies.

    The profile path is only a request: callers must still check that the
    fused decode step is armed and silently skip the split when it is not.
    """
    if explicit():
        return True
    if not current_platform.is_cuda():
        return False
    if device_id is None:
        device_id = torch.cuda.current_device()
    return raon_optimizations.enabled("raon_kda_split_decode_suffix", device_id)


@dataclass(frozen=True)
class SplitPlan:
    num_decodes: int  # leading plain-decode rows == leading decode tokens
    num_prefills: int
    num_prefill_tokens: int


def plan_split(
    query_start_loc: Sequence[int],
    is_prefilling: Sequence[bool] | None,
    num_actual_tokens: int,
    num_spec_decodes: int,
) -> SplitPlan | None:
    """Return the split of a ``[decode rows][prefill rows]`` batch, else None.

    ``query_start_loc`` is the host cumulative query length (num_reqs + 1
    entries); ``is_prefilling[i]`` tells whether request i had not finished its
    prompt before this step.
    """
    if num_spec_decodes != 0 or is_prefilling is None:
        return None
    num_reqs = len(query_start_loc) - 1
    if num_reqs < 2 or len(is_prefilling) < num_reqs:
        return None
    if query_start_loc[0] != 0 or query_start_loc[num_reqs] != num_actual_tokens:
        return None
    lengths = [query_start_loc[i + 1] - query_start_loc[i] for i in range(num_reqs)]
    if min(lengths) <= 0:
        return None

    def is_decode(i: int) -> bool:
        return lengths[i] == 1 and not is_prefilling[i]

    num_decodes = 0
    while num_decodes < num_reqs and is_decode(num_decodes):
        num_decodes += 1
    if num_decodes == 0 or num_decodes == num_reqs:
        return None
    if any(is_decode(i) for i in range(num_decodes, num_reqs)):
        return None  # interleaved layout: not decode-first
    return SplitPlan(
        num_decodes=num_decodes,
        num_prefills=num_reqs - num_decodes,
        num_prefill_tokens=num_actual_tokens - num_decodes,
    )


def attach_plan(builder, common_attn_metadata, metadata) -> None:
    """Attach the split plan and the prefill-only metadata to KDA metadata.

    Uses only host tensors already present on the common metadata; no device
    synchronization.
    """
    if not enabled():
        return
    if not explicit() and not fused_decode_armed(getattr(builder, "vllm_config", None)):
        return
    if (
        metadata.num_prefills <= 0
        or metadata.num_spec_decodes != 0
        or metadata.non_spec_query_start_loc is None
        or metadata.non_spec_state_indices_tensor is None
        or metadata.has_initial_state is None
    ):
        return
    is_prefilling = common_attn_metadata.is_prefilling
    if is_prefilling is None or is_prefilling.device.type != "cpu":
        return
    num_reqs = common_attn_metadata.num_reqs
    plan = plan_split(
        common_attn_metadata.query_start_loc_cpu[: num_reqs + 1].tolist(),
        is_prefilling[:num_reqs].tolist(),
        metadata.num_actual_tokens,
        metadata.num_spec_decodes,
    )
    if plan is None:
        return

    import dataclasses

    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    from . import kda_vsplit

    nd = plan.num_decodes
    qsl = metadata.non_spec_query_start_loc
    qsl_cpu = common_attn_metadata.query_start_loc_cpu[nd : num_reqs + 1] - nd
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        qsl_cpu, device=qsl.device
    )
    prefill_metadata = dataclasses.replace(
        metadata,
        num_prefills=plan.num_prefills,
        num_prefill_tokens=plan.num_prefill_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_actual_tokens=plan.num_prefill_tokens,
        has_initial_state=metadata.has_initial_state[nd:num_reqs],
        non_spec_query_start_loc=qsl[nd : num_reqs + 1] - nd,
        non_spec_state_indices_tensor=(
            metadata.non_spec_state_indices_tensor[nd:num_reqs]
        ),
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
    # The adaptive V-split mode was chosen for the whole step; redo it for the
    # rows the prefill kernels will actually see.
    if kda_vsplit.enabled():
        prefill_metadata._raon_kda_vsplit_mode = kda_vsplit.select_mode(
            qsl_cpu.diff().tolist(),
            0,
            builder.vllm_config.model_config.hf_config.linear_attn_config["num_heads"],
            builder._raon_kda_vsplit_sms,
        )
    setattr(metadata, _PLAN_ATTR, (plan, prefill_metadata))


def get_plan(metadata):
    """The (SplitPlan, prefill-only metadata) pair attached to this step."""
    return getattr(metadata, _PLAN_ATTR, None)
