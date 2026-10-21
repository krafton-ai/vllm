"""Load-time guards for prequantized ModelOpt NVFP4 linears.

vLLM 0.28's NVFP4 linear methods allocate the per-partition global scales
(``weight_scale_2``, ``input_scale``) with ``torch.empty`` and fold them with
``.max()`` in ``process_weights_after_loading``. A partition without a
checkpoint tensor keeps uninitialized memory, which can become the layer's
global scale. Raon's KDA ``in_proj_qkvgfab`` has exactly one such partition:
the alignment padding appended by the parent ``RaonRDABase`` (rda_base.py).

``fill_padding_partition`` gives that padding partition valid values from
the KDA layer that created it (kda.py). ``install_strict_scale_check`` makes
any other never-loaded partition a load error instead of silent corruption:
the scales start as NaN and must all be finite before the fold.
"""

import torch

_SCALE_NAMES = ("weight_scale_2", "input_scale")


def fill_padding_partition(layer: torch.nn.Module, pad_rows: int) -> list:
    """Zero the last ``pad_rows`` rows and fill the last partition's scales.

    The padding partition is the last logical partition. Its rows get zero
    packed weights and zero block scales (outputs are split off in forward),
    and each per-partition global scale copies the max of the loaded ones, so
    the ``.max()`` fold equals the max over the real partitions.
    """
    if pad_rows <= 0:  # x[-0:] is the whole tensor
        return []
    weight = layer.weight
    weight.data[-pad_rows:].view(torch.uint8).zero_()
    block_scale = getattr(layer, "weight_scale", None)
    if (block_scale is not None and block_scale.dim() == 2
            and block_scale.shape[0] == weight.shape[0]):
        block_scale.data[-pad_rows:].view(torch.uint8).zero_()
    partitions = len(getattr(layer, "logical_widths", None) or ())
    filled = []
    for name in _SCALE_NAMES:
        scale = getattr(layer, name, None)
        if (scale is None or scale.dim() != 1 or partitions < 2
                or scale.numel() != partitions):
            continue
        scale.data[-1] = scale.data[:-1].max()
        filled.append((name, float(scale.data[-1])))
    return filled


def install_strict_scale_check() -> None:
    """NaN-initialize NVFP4 per-partition scales; refuse any left unloaded."""
    from vllm.model_executor.layers.quantization import modelopt

    for cls in (modelopt.ModelOptNvFp4LinearMethod,
                modelopt.ModelOptNvFp4W4A16LinearMethod):
        if getattr(cls, "_raon_strict_scales", False):
            continue
        create, process = cls.create_weights, cls.process_weights_after_loading

        def create_weights(self, layer, *args, _create=create, **kwargs):
            _create(self, layer, *args, **kwargs)
            for name in _SCALE_NAMES:
                scale = getattr(layer, name, None)
                if scale is not None and not scale.is_meta:
                    scale.data.fill_(float("nan"))

        def process_weights_after_loading(self, layer, _process=process):
            for name in _SCALE_NAMES:
                scale = getattr(layer, name, None)
                if scale is None or scale.is_meta:
                    continue
                missing = torch.isnan(scale.data).nonzero().flatten().tolist()
                if missing:
                    raise RuntimeError(
                        f"NVFP4 {name} was never loaded for partition(s) "
                        f"{missing} (widths {getattr(layer, 'logical_widths', None)}, "
                        f"layer {getattr(layer, 'prefix', '<unknown>')})")
            return _process(self, layer)

        cls.create_weights = create_weights
        cls.process_weights_after_loading = process_weights_after_loading
        cls._raon_strict_scales = True
