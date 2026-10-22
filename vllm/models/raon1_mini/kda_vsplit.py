# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive FlashKDA value-split selection for pure-prefill KDA steps.

RAON_KDA_PREFILL_VSPLIT=adaptive (default off) chooses FlashKDA's existing
value-split (V-split) recurrence per scheduler step instead of FlashKDA's own
heuristic. The kernel source is FlashKDA with only that boolean exposed as an
argument (vendored/flashkda-vsplit/PROVENANCE.md); device arithmetic and launch
geometry are unchanged, so each forced variant is one of the two variants the
stock build can already run.

Decision, made once per KDA metadata build from host request lengths:
  chunks_i = ceil(len_i / 16) for each request with len_i > 0
  effective = sum(chunks) / max(chunks)   (sequences the recurrence can overlap)
  V-split on  iff 2 * heads * effective <= streaming multiprocessors
Steps that contain speculative decodes keep the stock FlashKDA call (mode -1).
The rule is benchmark-derived; re-measure before changing it.
"""

import os

_ENV = "RAON_KDA_PREFILL_VSPLIT"
_BUILD_DIR_ENV = "RAON_KDA_VSPLIT_BUILD_DIR"
_SOURCE = os.path.join(os.path.dirname(__file__), "vendored", "flashkda-vsplit")
_module_loaded = False
_seen: set[tuple[int, int, int]] = set()


def enabled() -> bool:
    value = os.environ.get(_ENV, "0")
    if value not in ("0", "adaptive"):
        raise ValueError(f"{_ENV} must be 0 or adaptive; got {value!r}")
    return value == "adaptive"


def select_mode(query_lengths, num_spec_decodes, heads, sms):
    """Return 1 (V-split), 0 (no V-split) or -1 (stock FlashKDA call)."""
    chunks = [(int(length) + 15) // 16 for length in query_lengths if length > 0]
    if not chunks or num_spec_decodes != 0:
        return -1
    effective = sum(chunks) / max(chunks)
    return int(2 * heads * effective <= sms)


def _load_module():
    """JIT-build the vendored op once per process, like the other CUDA extensions."""
    global _module_loaded
    if _module_loaded:
        return
    import importlib.util

    import torch
    from torch.utils.cpp_extension import load

    # FlashInfer's pinned CUTLASS headers; the op uses only cute/cutlass headers.
    flashinfer = importlib.util.find_spec("flashinfer")
    if flashinfer is None or not flashinfer.submodule_search_locations:
        raise RuntimeError(
            f"{_ENV}=adaptive needs installed FlashInfer CUTLASS headers"
        )
    cutlass = os.path.join(
        flashinfer.submodule_search_locations[0], "data", "cutlass", "include"
    )
    if not os.path.isdir(cutlass):
        raise RuntimeError(f"{_ENV}=adaptive needs CUTLASS headers at {cutlass}")
    build_dir = os.environ.get(_BUILD_DIR_ENV)
    if not build_dir:
        raise RuntimeError(
            f"{_ENV}=adaptive requires {_BUILD_DIR_ENV} (set by vllm.models.raon1_mini)"
        )
    major, minor = torch.cuda.get_device_capability()
    if major == 10:
        arch = "100f"
    elif (major, minor) in ((12, 0), (12, 1)):
        arch = f"12{minor}a"
    else:
        raise ValueError(
            f"{_ENV}=adaptive supports SM10x, SM120 and SM121, got SM{major}{minor}"
        )
    build_dir = os.path.join(build_dir, arch)
    os.makedirs(build_dir, exist_ok=True)
    flags = [
        "-O3",
        "-DNDEBUG",
        "-DUSE_CUDA",
        "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
    ]
    load(
        name="raon_flashkda_vsplit",
        sources=[
            os.path.join(_SOURCE, name)
            for name in (
                "registration.cpp",
                "flash_kda.cpp",
                os.path.join("smxx", "fwd_launch.cu"),
            )
        ],
        extra_include_paths=[_SOURCE, cutlass],
        extra_cflags=flags,
        extra_cuda_cflags=flags
        + [
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--threads=4",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ],
        build_directory=build_dir,
        is_python_module=False,
        verbose=os.environ.get("RAON_KDA_VSPLIT_VERBOSE", "0") == "1",
    )
    assert hasattr(torch.ops.raon_flashkda_vsplit, "fwd")
    _module_loaded = True


def attach_mode(builder, common_attn_metadata, metadata) -> None:
    """Attach the per-step mode to KDA metadata from host lengths (no GPU sync)."""
    if not enabled():
        return
    sms = getattr(builder, "_raon_kda_vsplit_sms", None)
    if sms is None:
        import torch

        device = common_attn_metadata.query_start_loc.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        builder._raon_kda_vsplit_sms = sms
    heads = builder.vllm_config.model_config.hf_config.linear_attn_config["num_heads"]
    lengths = common_attn_metadata.query_start_loc_cpu.diff().tolist()
    metadata._raon_kda_vsplit_mode = select_mode(
        lengths, metadata.num_spec_decodes, heads, sms
    )


def validate(vllm_config) -> None:
    """Fail at model construction when the qualified contract does not hold."""
    if not enabled():
        return
    if vllm_config.parallel_config.tensor_parallel_size != 1:
        raise ValueError(f"{_ENV}=adaptive is qualified only for TP1")
    _load_module()


def prefill(
    metadata,
    *,
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    lower_bound,
    initial_state,
    cu_seqlens,
    out,
    final_state,
    workspace,
    stock,
):
    """Drop-in for vLLM's _flashkda_prefill with the per-step forced variant."""
    mode = getattr(metadata, "_raon_kda_vsplit_mode", -1)
    if mode < 0:
        return stock(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            out=out,
            final_state=final_state,
            workspace=workspace,
        )
    import torch

    signature = (mode, q.shape[1], cu_seqlens.numel() - 1)
    if signature not in _seen and len(_seen) < 64:
        _seen.add(signature)
        print(f"[kda-vsplit] mode,tokens,sequences={signature}", flush=True)
    # Same argument preparation as vLLM's stock _flashkda_prefill.
    torch.ops.raon_flashkda_vsplit.fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta,
        q.shape[-1] ** -0.5,
        out,
        workspace,
        A_log.contiguous(),
        dt_bias.view(-1, q.shape[-1]).contiguous(),
        lower_bound,
        mode,
        initial_state.contiguous(),
        final_state,
        cu_seqlens.contiguous(),
    )
    return out, final_state
