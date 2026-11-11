# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register Nano and its vLLM compatibility adapters without initializing CUDA."""


def register():
    from transformers import AutoConfig

    from vllm import ModelRegistry
    from vllm.models.raon1_nano import raon_phase
    from vllm.models.raon1_nano.config import Raon1NanoConfig
    from vllm.transformers_utils import model_arch_config_convertor as mac

    raon_phase.preflight()
    raon_phase.install_metadata_adapter()
    raon_phase.install_graph_dispatch_adapter()
    raon_phase.install_mla_phase_adapter()
    raon_phase.announce_installed()
    AutoConfig.register("raon1_nano", Raon1NanoConfig, exist_ok=True)

    class _RaonModelArchConfigConvertor(mac.ModelArchConfigConvertorBase):
        def is_deepseek_mla(self) -> bool:
            return getattr(self.hf_text_config, "kv_lora_rank", None) is not None

    mac.MODEL_ARCH_CONFIG_CONVERTORS["raon1_nano"] = _RaonModelArchConfigConvertor
    ModelRegistry.register_model(
        "Raon1NanoForCausalLM", "vllm.models.raon1_nano.nano:Raon1NanoForCausalLM"
    )
    mac.MODEL_ARCH_CONFIG_CONVERTORS["raon1_nano_mtp"] = _RaonModelArchConfigConvertor
    ModelRegistry.register_model(
        "Raon1NanoMTP", "vllm.models.raon1_nano.mtp:Raon1NanoMTP"
    )
    _install_mamba_index_dtype_guard()
    _install_mla_batch_invariant_decode()
    _install_mla_prefill_align()
    _install_kda_autotune_pin()


def _install_kda_autotune_pin() -> None:
    """Pin upstream Triton KDA autotuners to a repeatable launch configuration."""
    import logging

    log = logging.getLogger(__name__)
    try:
        from vllm.models.raon1_nano.kda_autotune import pin_kda_autotune_configs

        pin_kda_autotune_configs()
    except Exception as exc:
        log.warning(
            "raon_vllm: KDA autotune pin SKIPPED -- %s; KDA process/batch "
            "invariance is NOT guaranteed on this build",
            exc,
        )


def _install_mla_batch_invariant_decode() -> None:
    """Use one KV split for Triton MLA; leave FlashInfer unchanged."""
    import logging
    import os

    log = logging.getLogger(__name__)
    if os.environ.get("RAON_MLA_BATCH_INVARIANT", "1") == "0":
        log.info(
            "raon_vllm: MLA batch-invariant decode pin disabled "
            "(RAON_MLA_BATCH_INVARIANT=0)"
        )
        return
    try:
        from vllm.v1.attention.backends.mla import triton_mla
    except Exception as exc:
        log.warning(
            "raon_vllm: MLA batch-invariant pin SKIPPED -- could not import "
            "vllm...mla.triton_mla (%s); zero-mismatch MLA parity is NOT "
            "guaranteed on this build",
            exc,
        )
        return
    if getattr(triton_mla, "_raon_mla_fixed_split", False):
        return
    if not hasattr(triton_mla, "_compute_num_kv_splits"):
        log.warning(
            "raon_vllm: MLA batch-invariant pin SKIPPED -- "
            "vllm...mla.triton_mla._compute_num_kv_splits not found (vLLM version"
            " drift); validate the split knob against this vLLM before claiming "
            "MLA zero-mismatch"
        )
        return

    def _fixed_num_kv_splits(max_seq_len, sm_count):
        return 1

    triton_mla._compute_num_kv_splits = _fixed_num_kv_splits
    triton_mla._raon_mla_fixed_split = True  # type: ignore[attr-defined]
    log.info(
        "raon_vllm: MLA decode pinned to num_kv_splits=1 (fixed split-KV, fp32 "
        "reduction)"
    )


def _install_mla_prefill_align() -> None:
    """Install the optional attention prefill normalization adapters."""
    import logging

    log = logging.getLogger(__name__)
    try:
        from vllm.models.raon1_nano import mla_align

        mla_align.install()
    except Exception as exc:
        from vllm.models.raon1_nano import raon_phase

        if raon_phase.PREFILL_ALIGNED:
            raise RuntimeError(
                f"RAON_PREFILL_ALIGNED=1: MLA prefill alignment failed ({exc!r})."
            ) from exc
        log.warning(
            "raon_vllm: MLA prefill align SKIPPED -- %s; trainer-paired MLA "
            "prefill is NOT guaranteed on this build",
            exc,
        )


def _install_mamba_index_dtype_guard() -> None:
    """Keep postprocess state indices int64 as required by index_fill_."""
    import torch

    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    if getattr(MambaHybridModelState, "_raon_int64_guard", False):
        return
    original = MambaHybridModelState.postprocess_state

    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
        if isinstance(num_sampled, int) and idx_mapping.dtype != torch.int64:
            idx_mapping = idx_mapping.to(dtype=torch.int64)
        return original(self, idx_mapping, num_sampled, num_computed_tokens)

    MambaHybridModelState.postprocess_state = postprocess_state  # type: ignore[method-assign]
    MambaHybridModelState._raon_int64_guard = True  # type: ignore[attr-defined]
