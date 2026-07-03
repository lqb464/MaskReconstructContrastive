from __future__ import annotations

import torch
import torch.nn as nn


def flip_lr(x: torch.Tensor) -> torch.Tensor:
    """Left-right flip on width dimension for NCHW tensor."""
    return torch.flip(x, dims=[-1])


def flip_lr_nhwc(x: torch.Tensor) -> torch.Tensor:
    """Left-right flip on width dimension for NHWC tensor."""
    return torch.flip(x, dims=[2])


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


__all__ = [
    "count_parameters",
    "flip_lr",
    "flip_lr_nhwc",
    "nchw_to_nhwc",
    "nhwc_to_nchw",
]
