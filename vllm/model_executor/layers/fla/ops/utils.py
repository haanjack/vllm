# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import contextlib
import functools
import logging
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton

logger = logging.getLogger(__name__)

COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_GDN_FIX_BT = os.getenv("FLA_GDN_FIX_BT", "0") == "1"

SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 4). When the cache is full, the oldest entry will be removed.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache_entries: tuple[tuple | None, dict | None, Any] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(a is b for a, b in zip(args, last_args))
                and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                )
            ):
                cache_entries = (
                    cache_entries[:i]
                    + cache_entries[i + 1 :]
                    + [(args, kwargs, last_result)]
                )
                return last_result

        result = fn(*args, **kwargs)

        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (
            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
        )
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.cuda.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


@functools.cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except (RuntimeError, AttributeError):
        return "cpu"


@functools.cache
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    device = get_available_device()
    mapping = {
        "cuda": "nvidia",
        "hip": "amd",
        "xpu": "intel",
    }
    # return the mapped value, or the original if not found
    return mapping.get(device, device)


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
device = "cuda" if current_platform.is_cuda_alike() else get_available_device()
device_torch_lib = getattr(torch, device, None)
device_platform = _check_platform()

is_amd = device_platform == "amd"
is_intel = device_platform == "intel"
is_nvidia = device_platform == "nvidia"
is_intel_alchemist = is_intel and "Intel(R) Arc(TM) A" in torch.xpu.get_device_name(0)
is_nvidia_hopper = is_nvidia and (
    "NVIDIA H" in torch.cuda.get_device_name(0)
    or torch.cuda.get_device_capability()[0] >= 9
)
use_cuda_graph = is_nvidia and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"

# AMD GPU architecture detection
# CDNA3: MI300X, MI325X - 64KB LDS, wavefront64, max 40 warps/XCD
# CDNA4: MI350X, MI355X - 160KB LDS, wavefront64, max 40 warps/XCD
_amd_device_name = torch.cuda.get_device_name(0) if is_amd else ""
is_amd_cdna3 = is_amd and any(
    x in _amd_device_name for x in ["MI300X", "MI325X", "MI300A"]
)
is_amd_cdna4 = is_amd and any(x in _amd_device_name for x in ["MI350X", "MI355X"])
is_gather_supported = hasattr(triton.language, "gather")
is_tma_supported = (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 9) and (
    hasattr(triton.language, "_experimental_make_tensor_descriptor")
    or hasattr(triton.language, "make_tensor_descriptor")
)


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)[
                "max_shared_mem"
            ]
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        return [-1]


class Backend(Enum):
    # NVIDIA GPUs
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    # AMD GPUs - CDNA3 (MI300X, MI325X): 64KB LDS per workgroup
    # wavefront size: 64, max 40 warps per XCD
    CDNA3 = 65536
    MI300X = 65536
    MI325X = 65536
    MI300A = 65536
    # AMD GPUs - CDNA4 (MI350X, MI355X): 160KB LDS per workgroup
    # wavefront size: 64, max 40 warps per XCD
    CDNA4 = 163840
    MI350X = 163840
    MI355X = 163840
    # Default fallback
    DEFAULT = 102400

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


def get_amd_arch_name() -> str:
    """
    Get the AMD architecture name for the current device.

    Returns:
        str: Architecture name ('CDNA4', 'CDNA3', or 'UNKNOWN')
    """
    if is_amd_cdna4:
        return "CDNA4"
    elif is_amd_cdna3:
        return "CDNA3"
    return "UNKNOWN"


def get_amd_lds_size() -> int:
    """
    Get the LDS (Local Data Share) size in bytes for the current AMD GPU.

    Returns:
        int: LDS size in bytes (160KB for CDNA4, 64KB for CDNA3, 64KB default)
    """
    if is_amd_cdna4:
        return Backend.CDNA4.value  # 160KB
    elif is_amd_cdna3:
        return Backend.CDNA3.value  # 64KB
    return 65536  # Default 64KB


def get_num_warps_for_amd() -> list[int]:
    """
    Get recommended num_warps values for AMD GPU auto-tuning.

    AMD CDNA architecture uses wavefront64 (64 threads per warp vs 32 on NVIDIA).
    Max 40 warps per XCD on MI300X/MI325X/MI350X/MI355X.

    For effective occupancy, we use lower warp counts compared to NVIDIA
    since each AMD warp processes twice as many threads.

    Returns:
        list[int]: List of num_warps values for auto-tuning
    """
    # AMD wavefront64 means fewer warps needed for same thread count
    # Max 40 warps/XCD, but practical limits are lower for register pressure
    return [4, 8, 16, 32]


def get_num_stages_for_amd() -> list[int]:
    """
    Get recommended num_stages values for AMD GPU auto-tuning.

    AMD GPUs have different memory hierarchy and prefetch behavior
    than NVIDIA GPUs. Fewer stages often work better.

    Returns:
        list[int]: List of num_stages values for auto-tuning
    """
    return [1, 2, 3]


def get_block_sizes_for_amd() -> list[int]:
    """
    Get recommended block sizes for AMD GPU auto-tuning.

    CDNA4 (160KB LDS) can support larger blocks than CDNA3 (64KB LDS).

    Returns:
        list[int]: List of block sizes for auto-tuning
    """
    if is_amd_cdna4:
        # 160KB LDS allows larger blocks
        return [64, 128, 192]
    elif is_amd_cdna3:
        # 64KB LDS, more conservative
        return [32, 64, 128]
    return [32, 64]  # Default
