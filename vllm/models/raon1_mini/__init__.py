# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""raon-1-mini: KDA + MLA hybrid with mHC residual streams and an MXFP8 SiTU-GLU MoE.

The modules read their kernel selections from ``RAON_*`` environment variables.
The qualified production configuration (B300, TP1, DFlash k3) is the default:
importing this package fills in every selection the caller has not set, so
``vllm serve`` needs no environment. An explicitly set variable still wins.

The routed-expert selections depend on the checkpoint: MXFP8 (the HF release)
runs the grouped W8A8 mode-4 kernels, ModelOpt NVFP4 runs vLLM's Marlin MoE.
Raon1MiniForCausalLMConfig records the profile in ``RAON_PRECISION_PROFILE``
before the engine processes start; ``apply_precision_profile`` then sets or
withdraws the MXFP8-only defaults.
"""

import os
import socket

# The qualified serving configuration.
# Routed experts run W8A8 MXFP8; MLA projections and shared experts stay BF16.
PRODUCTION_ENV = {
    "RAON_RDA_FUSED": "1",
    "RAON_BATCH_INVARIANT": "0",
    "RAON_KDA_PREFILL": "flashkda",
    "RAON_KDA_DECODE": "fused",
    "RAON_KDA_SGL_FUSED": "1",
    "RAON_MHC_PHI_GEMM_DIRECT": "all-pad",
    "RAON_KDA_PREFILL_FRONT_FUSED": "1",
    "RAON_KDA_PREFILL_VSPLIT": "adaptive",
    "RAON_MHC_MULTI_TOKEN_MIX": "1",
    "RAON_MHC_HEAD_RMS": "1",
    "RAON_MHC_HEAD_CHART": "1",
    "RAON_MHC_SPLIT_MAX_T": "32",
    "RAON_MHC_SPLIT_HEAD_BC": "512",
    "RAON_MHC_SPLIT_MIX_BC": "128",
    "RAON_DFLASH_CACHE_ALIGNMENT": "1",
    "RAON_MOE_SHARED_SIDE_STREAM": "1",
    "RAON_EXPERIMENT_SHARED_GRAPH": "1",
    # Derived from the precision policy: MLA projections and shared experts
    # are not MXFP8 in either profile.
    "RAON_MLA_MXFP8_REALIZED": "0",
    "RAON_MOE_SHARED_MXFP8_REALIZED": "0",
}

# Routed experts as preconverted MXFP8 on the grouped W8A8 mode-4 kernels.
# The fused GEMM13 SiTU epilogue is required:
# the unfused chain misreads the reused MoE arena.
MXFP8_ENV = {
    "RAON_MOE_TRAINER_REALIZED": "1",
    "RAON_MOE_MXFP8_W8A8_TRITON": "1",
    "RAON_MOE_MXFP8_W8A8_GEMM_MODE": "4",
    "RAON_MOE_MXFP8_W8A8_FINALIZE": "1",
    "RAON_MXFP8_MIN_M": "0",
    "RAON_PREFILL_ALIGNED": "0",
    "RAON_MXFP8_COMPACT_LOAD": "1",
    "RAON_MXFP8_ARENA_PIN_MAX_TOKENS": "512",
    "RAON_SWAB_ROUTE_PACK_PREFILL": "1",
    "RAON_SWAB_SMALL_ROUTE": "3",
    "RAON_SWAB_SMALL_GEMM_SHAPE_MAP": "f32=128,16,1,1;bf16=128,16,1,1",
    "RAON_SWAB_SITU_EPILOGUE": "1",
    "RAON_MOE_EXPERT_MXFP8_REALIZED": "1",
}
# ModelOpt NVFP4 checkpoint: Marlin NVFP4 MoE and linears.
NVFP4_ENV = {
    "RAON_MOE_EXPERT_MXFP8_REALIZED": "0",
}
PROFILE_ENV = {"mxfp8": MXFP8_ENV, "nvfp4": NVFP4_ENV}
PROFILE_KEY = "RAON_PRECISION_PROFILE"


def _build_dirs() -> dict[str, str]:
    """JIT build directories for the CUDA extensions, one tree per host.

    A shared filesystem's file locks do not exclude another node, so two hosts
    must never build into the same directory.
    """
    from vllm import envs

    root = os.path.join(envs.VLLM_CACHE_ROOT, "raon1_mini", "build", socket.gethostname())
    return {
        "RAON_KDA_SGL_FUSED_BUILD_DIR": os.path.join(root, "sgl-kda-fused"),
        "RAON_KDA_SGL_FUSED_SPEC_BUILD_DIR": os.path.join(root, "sgl-kda-fused-spec"),
        "RAON_KDA_VSPLIT_BUILD_DIR": os.path.join(root, "kda-vsplit"),
    }


# Keys this package filled in, so a later profile switch withdraws only those.
_DEFAULTED: dict[str, str] = {}


def _set_default(key: str, value: str) -> None:
    if key not in os.environ:
        os.environ[key] = value
        _DEFAULTED[key] = value


def apply_precision_profile(profile: str) -> None:
    """Fill in ``profile``'s defaults, withdrawing the other profile's ones."""
    if profile not in PROFILE_ENV:
        raise ValueError(f"Unknown raon-1-mini precision profile: {profile}")
    os.environ[PROFILE_KEY] = profile
    for other, env in PROFILE_ENV.items():
        if other == profile:
            continue
        for key, value in env.items():
            if _DEFAULTED.get(key) == value and os.environ.get(key) == value:
                del os.environ[key]
                del _DEFAULTED[key]
    for key, value in PROFILE_ENV[profile].items():
        _set_default(key, value)


for _key, _value in {**PRODUCTION_ENV, **_build_dirs()}.items():
    _set_default(_key, _value)
del _key, _value
apply_precision_profile(os.environ.get(PROFILE_KEY, "mxfp8"))
