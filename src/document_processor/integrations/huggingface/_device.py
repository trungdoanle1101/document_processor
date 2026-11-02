from typing import Literal

import torch

type Device = Literal["auto", "cuda", "mps", "cpu"]
type ResolvedDevice = Literal["cuda", "mps", "cpu"]


def resolve_device(device: Device) -> ResolvedDevice:
    cuda_available = torch.cuda.is_available()
    if device == "cuda":
        if not cuda_available:
            raise ValueError(
                "device 'cuda' was asked for, but PyTorch found no CUDA-compatible GPU; use 'auto' or 'cpu'"
            )
        return "cuda"

    mps_available = torch.backends.mps.is_available()
    if device == "mps":
        if not mps_available:
            raise ValueError(
                "device 'mps' was asked for, but PyTorch found no MPS-compatible GPU; use 'auto' or 'cpu'"
            )
        return "mps"

    if device == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
    return "cpu"
