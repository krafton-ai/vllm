# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build JIT CUDA extensions for the current GPU only."""

import contextlib
import os

import torch


@contextlib.contextmanager
def current_device_arch():
    """Restrict TORCH_CUDA_ARCH_LIST to the current GPU while building.

    Images often set TORCH_CUDA_ARCH_LIST to many architectures; kernels that
    use newer PTX features then fail to build for the oldest one even though
    only the current GPU runs them.
    """
    major, minor = torch.cuda.get_device_capability()
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous
