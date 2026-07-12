from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config.experiment import ExperimentConfig
from ..models.factory import build_model
from ..models.model_utils import flip_lr


def build_downstream_model(
    cfg: ExperimentConfig,
    *,
    out_ch: int,
    enable_reconstruct: bool = True,
    single_view: bool = False,
) -> nn.Module:
    cfg.training.enable_reconstruct = bool(enable_reconstruct)
    cfg.training.single_view = bool(single_view)
    return build_model(cfg, out_ch=int(out_ch))


def replace_output_channels(model: nn.Module, num_classes: int) -> None:
    if int(num_classes) < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if hasattr(model, "replace_output_channels"):
        model.replace_output_channels(int(num_classes))
        return
    raise TypeError(f"Model {type(model).__name__} does not support replace_output_channels")


def forward_dual_recon(
    model: nn.Module,
    x: torch.Tensor,
    plane_one_hot: torch.Tensor,
    pixel_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    recon1, recon2, _, _ = model(x, pixel_mask, plane_one_hot)
    if recon1 is None:
        raise RuntimeError("Model returned recon1=None; enable_reconstruct must be True for downstream recon tasks.")
    if recon2 is None:
        raise RuntimeError(
            "Model returned recon2=None; downstream tasks require dual-view (single_view=False)."
        )
    return recon1, recon2


def pool_bottleneck(
    model: nn.Module,
    x: torch.Tensor,
    plane_one_hot: torch.Tensor,
    *,
    view: int = 1,
) -> torch.Tensor:
    feat = model.encode_bottleneck(x, plane_one_hot, view=int(view), pixel_mask=None)
    if feat.ndim != 4:
        raise ValueError(f"encode_bottleneck must return [B,H,W,C], got {tuple(feat.shape)}")
    return feat.mean(dim=(1, 2))


def encode_dual_bottleneck(
    model: nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
    plane_one_hot: torch.Tensor,
    *,
    mode: str = "bottleneck_concat",
) -> torch.Tensor:
    h1 = pool_bottleneck(model, x1, plane_one_hot, view=1)
    h2 = pool_bottleneck(model, x2, plane_one_hot, view=2)
    if mode == "bottleneck":
        return 0.5 * (h1 + h2)
    if mode == "bottleneck_concat":
        return torch.cat([h1, h2], dim=-1)
    raise ValueError(f"Unsupported classifier feature mode: {mode!r}. Use bottleneck or bottleneck_concat.")


__all__ = [
    "build_downstream_model",
    "replace_output_channels",
    "forward_dual_recon",
    "pool_bottleneck",
    "encode_dual_bottleneck",
]
