from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

from .ae_blocks import ConvDecoder, ConvEncoder, apply_pixel_mask
from .model_utils import count_parameters, flip_lr


class VanillaAEBase(nn.Module, ABC):
    """
    Abstract vanilla conv AutoEncoder base (no skip connections).
    Not exposed via CLI; MAE and VAE build on this interface.
    """

    uses_pixel_mask: bool = False
    vis_mode: Literal["masked", "full"] = "full"

    def __init__(
        self,
        *,
        in_ch: int = 1,
        base_ch: int = 32,
        out_ch: int = 1,
        use_gn: bool = False,
        enable_reconstruct: bool = True,
        single_view: bool = False,
    ):
        super().__init__()
        self.enable_reconstruct = bool(enable_reconstruct)
        self.single_view = bool(single_view)
        self.in_ch = int(in_ch)
        self.base_ch = int(base_ch)

        if int(out_ch) < 1:
            raise ValueError(f"out_ch must be >=1, got {out_ch}")

        self.encoder = ConvEncoder(self.in_ch, self.base_ch, use_gn=use_gn)
        self.decoder = ConvDecoder(self.base_ch, int(out_ch), use_gn=use_gn)

    def _encode_decode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        z = self.encoder(x)
        logits = self.decoder(z, target_size=(h, w))
        return logits, z

    def encoder_state_dict_prefixes(self) -> tuple[str, ...]:
        return ("encoder",)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = bool(trainable)

    def reset_contrastive_projection_heads(self) -> None:
        return None

    def param_count_breakdown(self) -> Dict[str, int]:
        total = count_parameters(self)
        enc = count_parameters(self.encoder)
        dec = count_parameters(self.decoder)
        return {
            "total": total,
            "enc_early_view1": enc,
            "enc_early_view2": 0,
            "saca": 0,
            "enc_shared_trunk": 0,
            "contrastive_head": 0,
            "decoder_shared_up2": dec,
            "decoder_branch_v1": 0,
            "decoder_branch_v2": 0,
            "recon_heads": 0,
            "check_sum": enc + dec,
            "delta_total_minus_check": total - (enc + dec),
        }

    @torch.no_grad()
    def encode_bottleneck(
        self,
        x: torch.Tensor,
        plane_one_hot: torch.Tensor,
        view: int = 1,
        *,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del plane_one_hot
        x_in = self._prepare_input(x, pixel_mask)
        if int(view) == 2:
            x_in = flip_lr(x_in)
        z = self.encoder(x_in)
        return z.permute(0, 2, 3, 1).contiguous()

    def _prepare_input(self, x: torch.Tensor, pixel_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.uses_pixel_mask and pixel_mask is not None:
            return apply_pixel_mask(x, pixel_mask)
        return x

    @abstractmethod
    def _forward_one(self, x: torch.Tensor, pixel_mask: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        pixel_mask: Optional[torch.Tensor],
        plane_one_hot: torch.Tensor,
    ):
        if plane_one_hot.shape[0] != x.shape[0]:
            raise ValueError(
                f"plane_one_hot batch ({plane_one_hot.shape[0]}) must match input batch ({x.shape[0]})."
            )

        if not self.enable_reconstruct:
            return None, None, None, None

        recon_raw_orig = self._forward_one(x, pixel_mask)

        if self.single_view:
            return recon_raw_orig, None, None, None

        recon_raw_flip = self._forward_one(flip_lr(x), pixel_mask)
        return recon_raw_orig, recon_raw_flip, None, None


__all__ = ["VanillaAEBase"]
