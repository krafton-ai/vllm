"""Reuse the exact FP32 widening of a versioned inference head weight."""
import torch


class FP32HeadWeightCache:
    def __init__(self):
        self.clear()

    def clear(self):
        self.source = None
        self.key = None
        self.value = None

    def get(self, weight):
        # Inference tensors have no mutation counter. Never silently reuse a
        # potentially stale value when mutation tracking is unavailable.
        try:
            version = weight._version
        except RuntimeError:
            self.clear()
            return weight.to(torch.float32)
        key = (version, weight.data_ptr(), weight.shape, weight.stride(),
               weight.dtype, weight.device)
        if self.source is not weight or self.key != key:
            self.value = weight.to(torch.float32)
            self.source = weight
            self.key = key
        return self.value
