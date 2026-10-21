"""Keep Kimi-K3 MLA reciprocal-scale storage stable across online refits."""

import torch


class StableMLAScaleBuffers:
    """Mixin: preserve only the two upstream-derived inverse-scale buffers.

    Kimi-K3's post-load hook re-registers reciprocal tensors on every call.
    After full graph capture that frees addresses the graph may still read.
    Preserve Module registration semantics for every other buffer; shape,
    dtype or device changes of these fixed scalars must fail explicitly.
    """

    def register_buffer(self, name, tensor, persistent=True):
        old = getattr(self, "_buffers", {}).get(name)
        if name in ("_q_scale_inv", "_k_scale_inv") and old is not None:
            if tensor is None or (
                old.shape != tensor.shape or old.dtype != tensor.dtype
                or old.device != tensor.device
            ):
                raise RuntimeError(f"MLA inverse-scale layout changed during refit: {name}")
            with torch.no_grad():
                old.copy_(tensor)
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)
            return
        return super().register_buffer(name, tensor, persistent=persistent)
