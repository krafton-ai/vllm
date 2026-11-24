"""RAON_MOE_SITU_FAST -- the routed experts' SiTU-GLU activation on a faster
launch of the SAME arithmetic as vLLM's ``torch.ops._C.situ_and_mul``.

## What is replaced

The live MoE path (unquantized bf16, ``TritonExperts``) applies
the expert activation through
``FusedMoEExpertsModular.activation -> apply_moe_activation``
(``vllm/model_executor/layers/fused_moe/modular_kernel.py:892-907``,
``activation.py:248-259``), which for ``MoEActivation.SITU`` launches
``vllm::situ_and_mul_kernel<c10::BFloat16>`` on
``intermediate_cache1.view(-1, 2 * 1088)`` (rows = tokens x top-6).

That upstream kernel (vllm @ 2cf0a6915,
``csrc/libtorch_stable/activation_kernels.cu:477-497``, launcher ``:621-640``)
runs one 1024-thread block per row with scalar 2-byte loads; at d = 1088 the
second loop trip keeps 64 of 1024 threads busy.  Profile j647331 measured
1.48 ms per layer for a 32K prefill (about 9x the 8 TB/s byte floor).

``moe_situ_fast_kernel.cu`` keeps the per-element expression character for
character (fp32 temporaries, accurate ``tanhf``/``expf``/IEEE division -- no
``--use_fast_math`` -- and the same ``c10::BFloat16`` conversion) and only
re-tiles the launch: flattened rows, 8 elements per thread, 16-byte
loads/stores.  Each output element is a pure function of (gate, up, beta,
linear_beta), so the tiling cannot move a bit; the bench
checks it over all 2^32 bf16 (gate, up) pairs.

## Switch

``RAON_MOE_SITU_FAST`` (raon-1-mini production default ``1``; ``0`` keeps vLLM's
kernel).  ``apply_moe_activation``'s SITU branch calls ``try_situ_and_mul``:
SITU calls on contiguous CUDA bf16/fp16 tensors go to
``torch.ops.raon_situ.situ_and_mul``, every other call runs the stock kernel.
The extension is JIT-built once per node into ``RAON_MOE_SITU_FAST_BUILD_DIR``
(the package sets a per-host directory under the vLLM cache root).

Witness: ``[raon-situ-fast]`` log lines on the first engaged call
(with its shape) and at engaged-call counts 10, 100, 1000, ... .  Calls made
during CUDA-graph capture are counted once at capture; replays run the
captured kernel.  The kernel name in a profile is
``raon_situ::situ_and_mul_vec_kernel``.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LEVER = "RAON_MOE_SITU_FAST"
_BUILD_DIR_ENV = "RAON_MOE_SITU_FAST_BUILD_DIR"
_DEFAULT_BUILD_DIR = "/var/tmp/{user}/raon-situ-fast-build"  # vllm.models.raon1_mini sets a per-host cache dir

_loaded = False
_load_failed = False
_installed = False
_calls = {"n": 0, "next": 1}


def lever_on() -> bool:
    return os.environ.get(_LEVER, "0") == "1"


def load() -> bool:
    """Compile (or load the cached build of) the kernel and register
    ``torch.ops.raon_situ.situ_and_mul``. Idempotent.

    Returns False if the extension cannot be built (for example when the CUDA
    development headers are missing); the failure is remembered and callers
    then use the stock kernel.
    """
    global _loaded, _load_failed
    if _loaded or _load_failed:
        return _loaded
    from torch.utils.cpp_extension import load as _cpp_load

    src = os.path.join(os.path.dirname(__file__), "moe_situ_fast_kernel.cu")
    build_dir = os.environ.get(
        _BUILD_DIR_ENV,
        _DEFAULT_BUILD_DIR.format(user=os.environ.get("USER", "unknown")),
    )
    os.makedirs(build_dir, exist_ok=True)
    from .jit_arch import current_device_arch

    # No --use_fast_math: tanhf/expf/division must stay the accurate
    # realizations upstream's (non-fast-math) build emits.
    try:
        with current_device_arch():
            _cpp_load(
                name="raon_situ_fast",
                sources=[src],
                build_directory=build_dir,
                extra_cuda_cflags=["-O3"],
                is_python_module=False,
                verbose=os.environ.get("RAON_MOE_SITU_FAST_VERBOSE", "0") == "1",
            )
    except Exception as error:  # noqa: BLE001 - any build failure falls back
        _load_failed = True
        logger.warning(
            "%s: kernel build failed (%r); using the stock situ_and_mul kernel",
            _LEVER,
            error,
        )
        return False
    _loaded = True
    return True


def situ_and_mul(out: torch.Tensor, x: torch.Tensor, beta: float,
                 linear_beta: float) -> None:
    """Drop-in for ``torch.ops._C.situ_and_mul(out, x, beta, linear_beta)``."""
    load()
    torch.ops.raon_situ.situ_and_mul(out, x, float(beta), float(linear_beta))


def _eligible(output: torch.Tensor, input: torch.Tensor) -> bool:
    return (input.is_cuda and input.dtype in (torch.bfloat16, torch.float16)
            and output.dtype == input.dtype
            and input.is_contiguous() and output.is_contiguous())


def _witness(output: torch.Tensor, input: torch.Tensor) -> None:
    _calls["n"] += 1
    n = _calls["n"]
    if n == 1:
        print(f"[raon-situ-fast] engaged: first SITU call rows={input.shape[0]} "
              f"2d={input.shape[-1]} dtype={input.dtype} -> "
              "torch.ops.raon_situ.situ_and_mul", flush=True)
    if n >= _calls["next"]:
        print(f"[raon-situ-fast] engaged calls={n}", flush=True)
        _calls["next"] *= 10


def try_situ_and_mul(output: torch.Tensor, input: torch.Tensor, beta: float, linear_beta: float) -> bool:
    """Called by vLLM's apply_moe_activation SITU branch: run the faster kernel when the lever is on.

    Returns False (the caller runs torch.ops._C.situ_and_mul) when RAON_MOE_SITU_FAST
    is off, the tensors are not eligible or the kernel cannot be built. The kernel is
    built on the first engaged call, vLLM's eager profile run, before any CUDA-graph
    capture; processes that never run the MoE never compile it.
    """
    if not lever_on() or not _eligible(output, input) or not load():
        return False
    situ_and_mul(output, input, beta, linear_beta)
    _witness(output, input)
    return True
