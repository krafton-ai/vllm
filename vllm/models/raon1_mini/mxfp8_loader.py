"""Load immutable routed MXFP8 tensors without allocating BF16 expert storage.

Two sources hold the same bytes:

* the checkpoint itself (the HF release): each routed expert stores ``w{1,2,3}``
  as F8_E4M3 ``weight`` [out, in] with E8M0 ``weight_scale_inv`` bytes
  [out, in/32]. They are stacked per layer and the scales packed into the
  CUTLASS MMA layout after loading.
* a preconverted ``raon-mxfp8`` sidecar named by ``RAON_MXFP8_WEIGHTS``, already
  stacked and packed, next to a BF16 checkpoint.

Packing the release's layer tensors reproduces the sidecar files byte for byte.
"""

import os


def compact_loading_enabled():
    return os.environ.get("RAON_MXFP8_COMPACT_LOAD", "0") == "1"


def validate_compact_loading():
    if not compact_loading_enabled():
        return
    required = {
        "RAON_MOE_EXPERT_MXFP8_REALIZED": "1",
        "RAON_MOE_TRAINER_REALIZED": "1",
        "RAON_MOE_MXFP8_W8A8_TRITON": "1",
        "RAON_MOE_MXFP8_W8A8_GEMM_MODE": "4",
        "RAON_MXFP8_MIN_M": "0",
        "RAON_PREFILL_ALIGNED": "0",
    }
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise ValueError(f"Compact MXFP8 loading requires {name}={expected}")
    if os.environ.get("RAON_MOE_MXFP8_SHADOW", "0") != "0":
        raise ValueError("Compact MXFP8 loading cannot run the BF16 shadow path")


# Routed-expert storage declared by the HF release's quantization_config.
_CHECKPOINT_QUANTIZATION = {"quant_method": "mxfp8", "weight_block_size": [1, 32]}
# Per-layer staging parameters for checkpoint loading, by MoE parameter name.
CHECKPOINT_PARAMS = {
    "w13_weight": "w13_weight_mxfp8",
    "w2_weight": "w2_weight_mxfp8",
}
CHECKPOINT_SCALES = ("w13_weight_scale_inv", "w2_weight_scale_inv")


def _from_checkpoint(checkpoint_quantization):
    """True to read MXFP8 experts from the checkpoint, False for the sidecar."""
    if os.environ.get("RAON_MXFP8_WEIGHTS"):
        return False
    if checkpoint_quantization is None:
        raise ValueError(
            "raon-1-mini serves routed experts in MXFP8: load the MXFP8 HF release, "
            "or set RAON_MXFP8_WEIGHTS to the preconverted weights of a BF16 checkpoint")
    for key, expected in _CHECKPOINT_QUANTIZATION.items():
        if checkpoint_quantization.get(key) != expected:
            raise ValueError(f"Unsupported routed-expert quantization: {key}="
                             f"{checkpoint_quantization.get(key)!r}, expected {expected!r}")
    return True


def make_quant_config(layer_idx, checkpoint_quantization=None):
    """Use vLLM's MoE construction contract; shared/dense linears stay BF16."""
    import torch
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.utils import set_weight_attrs

    validate_compact_loading()
    from_checkpoint = _from_checkpoint(checkpoint_quantization)

    class PreconvertedMoEMethod(FusedMoEMethodBase):
        def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition,
                               act_dtype, moe_parallel_config):
            if from_checkpoint:
                if moe_parallel_config.tp_size > 1 or moe_parallel_config.ep_size > 1:
                    raise ValueError("Checkpoint MXFP8 experts are qualified for TP=1 only; "
                                     "use preconverted RAON_MXFP8_WEIGHTS to shard them")
                if hidden_size % 32 or intermediate_size_per_partition % 32:
                    raise ValueError("MXFP8 experts require 32-channel blocks")
                return hidden_size, intermediate_size_per_partition
            from .mxfp8_sharding import source_shapes

            intermediate = source_shapes(layer_idx)[1][-1]
            # Every rank uses the same padded shape, but owns complete blocks.
            blocks_per_rank = (intermediate // 32 + moe_parallel_config.tp_size - 1) // moe_parallel_config.tp_size
            local = max(intermediate_size_per_partition, blocks_per_rank * 32)
            return hidden_size, (local + 31) // 32 * 32

        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype, **attrs):
            parallel = self.moe.moe_parallel_config
            if parallel.enable_eplb or self.moe.has_bias:
                raise ValueError("Compact MXFP8 serving does not support expert bias or EPLB")
            if num_experts < 1 or num_experts & (num_experts - 1):
                raise ValueError("The MXFP8 expert kernel requires a power-of-two local expert count")
            if params_dtype != torch.bfloat16:
                raise ValueError("Compact MXFP8 serving requires BF16 shape anchors")
            # Shape/loader anchors preserve the existing expert dispatch ABI.
            # They have no storage, on either GPU or CPU.
            self.target_device = torch.empty(0).device
            shapes = {"w13_weight": (num_experts, 2 * intermediate_size_per_partition, hidden_size),
                      "w2_weight": (num_experts, hidden_size, intermediate_size_per_partition)}
            for name, shape in shapes.items():
                param = torch.nn.Parameter(torch.empty(shape, dtype=params_dtype, device="meta"),
                                           requires_grad=False)
                set_weight_attrs(param, attrs)
                layer.register_parameter(name, param)
            layer._raon_preconverted_mxfp8 = True
            if from_checkpoint:
                self._create_checkpoint_weights(layer, num_experts, hidden_size,
                                                intermediate_size_per_partition)

        def _create_checkpoint_weights(self, layer, num_experts, hidden, intermediate):
            """Staging storage for the checkpoint's FP8 experts and E8M0 scales."""
            shapes = {
                "w13_weight_mxfp8": ((num_experts, 2 * intermediate, hidden), torch.float8_e4m3fn),
                "w2_weight_mxfp8": ((num_experts, hidden, intermediate), torch.float8_e4m3fn),
                "w13_weight_scale_inv": ((num_experts, 2 * intermediate, hidden // 32), torch.uint8),
                "w2_weight_scale_inv": ((num_experts, hidden, intermediate // 32), torch.uint8),
            }
            for name, (shape, dtype) in shapes.items():
                param = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
                param.weight_loader = _load_checkpoint_expert
                layer.register_parameter(name, param)

        def get_fused_moe_quant_config(self, layer):
            # Only routing metadata consumes this object; the Raon mode-4
            # forward owns packed-weight GEMMs and activation quantization.
            return FusedMoEQuantConfig.make()

        def process_weights_after_loading(self, layer):
            from .mxfp8_artifact import load
            from .mxfp8_w8a8_cutlass import _WQ_CACHE, adopt_fc1_layout
            from .mxfp8_align import _record_scope

            key = id(layer.w13_weight)
            if key in _WQ_CACHE:
                raise RuntimeError("Compact MXFP8 weights cannot be refitted")
            parallel = self.moe.moe_parallel_config
            distributed = parallel.tp_size > 1 or parallel.ep_size > 1
            if from_checkpoint:
                from .mxfp8_sharding import pack_scales

                # Zero pad bytes, as in the preconverted files: the packed
                # scales then match them byte for byte.
                tensors = {
                    "q13": layer.w13_weight_mxfp8.data,
                    "q2": layer.w2_weight_mxfp8.data,
                    "s13_mma": pack_scales(layer.w13_weight_scale_inv.data, fill=0),
                    "s2_mma": pack_scales(layer.w2_weight_scale_inv.data, fill=0),
                }
                for name in (*CHECKPOINT_PARAMS.values(), *CHECKPOINT_SCALES):
                    delattr(layer, name)
            elif distributed:
                from .mxfp8_sharding import shard, source_shapes

                shapes = source_shapes(layer_idx)
                expert_ids = [i for i in range(shapes[0][0])
                              if layer._map_global_expert_id_to_local_expert_id(i) != -1]
                expert_ids.sort(key=layer._map_global_expert_id_to_local_expert_id)
                if len(expert_ids) != layer.w13_weight.shape[0]:
                    raise ValueError("MXFP8 expert placement does not match local weight storage")
                tensors = load(f"layer-{layer_idx}-routed", shapes, "cpu")
                tensors = shard(tensors, expert_ids, parallel.tp_size, parallel.tp_rank,
                                layer.w2_weight.shape[-1])
                tensors = {name: tensor.to(self.target_device) for name, tensor in tensors.items()}
            else:
                shapes = [layer.w13_weight.shape, layer.w2_weight.shape]
                tensors = load(f"layer-{layer_idx}-routed", shapes, self.target_device)
            if set(tensors) != {"q13", "q2", "s13_mma", "s2_mma"}:
                raise ValueError("Unexpected routed MXFP8 tensor set")
            for name, tensor in tensors.items():
                expected = torch.float8_e4m3fn if name in {"q13", "q2"} else torch.uint8
                if tensor.dtype != expected:
                    raise ValueError(f"Invalid routed MXFP8 dtype: {name}={tensor.dtype}")
            for name, tensor in tensors.items():
                layer.register_buffer("raon_mxfp8_" + name, tensor, persistent=False)
            tensors["_refs"] = (layer.w13_weight, layer.w2_weight)
            _WQ_CACHE[key] = adopt_fc1_layout(tensors)
            _record_scope(layer_idx, ("routed_fc1", "routed_fc2"))
            source = "checkpoint" if from_checkpoint else "preconverted"
            print(f"[mxfp8-loader] layer={layer_idx} routed BF16 storage=0; packed {source} weights loaded",
                  flush=True)

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None,
                  shared_experts_input=None):
            from .model import _moe_expert_chain_mxfp8

            if layer.expert_map is not None:
                from .ep_routes import local_routes

                topk_ids, topk_weights = local_routes(
                    layer.expert_map, topk_ids, topk_weights,
                    layer.w13_weight.shape[0])
            return _moe_expert_chain_mxfp8(
                x, topk_weights, topk_ids, layer.w13_weight, layer.w2_weight,
                self.moe.activation_situ_beta, self.moe.activation_situ_linear_beta,
                layer_idx)

        def apply_monolithic(self, *args, **kwargs):
            raise RuntimeError("Compact MXFP8 weights cannot run a stock BF16 MoE kernel")

    class PreconvertedConfig:
        def get_name(self):
            return "raon_preconverted_mxfp8"

        def get_quant_method(self, layer, prefix):
            if isinstance(layer, RoutedExperts):
                return PreconvertedMoEMethod(layer.moe_config)
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return None

    return PreconvertedConfig()


def _load_checkpoint_expert(param, loaded_weight, weight_name, expert_id, shard_id):
    """Copy one expert's FP8 projection or its E8M0 scales into the layer stack.

    ``w13`` stacks the gate (``w1``) rows before the up (``w3``) rows.
    """
    if loaded_weight.dtype != param.dtype:
        raise ValueError(f"{weight_name}: checkpoint dtype {loaded_weight.dtype}, expected {param.dtype}")
    target = param.data[expert_id]
    if shard_id in ("w1", "w3"):
        rows = target.shape[0] // 2
        target = target[:rows] if shard_id == "w1" else target[rows:]
    if loaded_weight.shape != target.shape:
        raise ValueError(f"{weight_name}: checkpoint shape {tuple(loaded_weight.shape)}, "
                         f"expected {tuple(target.shape)}")
    target.copy_(loaded_weight)
