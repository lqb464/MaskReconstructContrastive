from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .ae_blocks import (
    apply_pixel_mask,
    downsample_mask_to_patches,
    patchify,
    unpatchify,
)
from .model_utils import count_parameters, flip_lr


def _build_recon_head(out_ch: int) -> nn.Sequential:
    """3-layer conv recon head (32 -> 24 -> 16 -> out_ch)."""
    return nn.Sequential(
        nn.Conv2d(32, 24, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(24, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, out_ch, kernel_size=1),
    )


class _PatchEncoder(nn.Module):
    """Encode visible patch tokens; masked positions use a learnable mask token."""

    def __init__(self, patch_dim: int, embed_dim: int, depth: int = 4):
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layers = []
        for _ in range(depth):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=max(1, embed_dim // 32),
                    dim_feedforward=embed_dim * 4,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
            )
        self.transformer = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, L, patch_dim]
        patch_mask: [B, L] with 1 = masked
        """
        x = self.proj(tokens)
        mask = patch_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * (1.0 - mask) + self.mask_token.to(dtype=x.dtype) * mask
        x = self.transformer(x)
        return self.norm(x)


class _PatchDecoder(nn.Module):
    """Decode patch tokens back to pixel patches."""

    def __init__(self, embed_dim: int, patch_dim: int, depth: int = 2):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=max(1, embed_dim // 32),
                    dim_feedforward=embed_dim * 4,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
            )
        self.transformer = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.transformer(tokens)
        x = self.norm(x)
        return self.head(x)


class MAEDualViewSSL(nn.Module):
    """
    Masked AutoEncoder backbone (patch-token transformer encoder/decoder).
    Input pixels are zeroed in masked regions before patchify (dual-view SSL protocol).
    """

    def __init__(
        self,
        *,
        in_ch: int = 1,
        image_size: int = 192,
        patch_size: int = 16,
        embed_dim: int = 256,
        enc_depth: int = 4,
        dec_depth: int = 2,
        out_ch: int = 1,
        enable_reconstruct: bool = True,
        enable_contrastive: bool = False,
        single_view: bool = False,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.enable_reconstruct = bool(enable_reconstruct)
        self.enable_contrastive = bool(enable_contrastive)
        self.single_view = bool(single_view)

        if self.enable_contrastive:
            raise ValueError("MAEDualViewSSL currently supports reconstruction-only (enable_contrastive=False).")
        if int(out_ch) != int(in_ch):
            raise ValueError(f"MAEDualViewSSL expects out_ch == in_ch, got out_ch={out_ch}, in_ch={in_ch}")
        if (self.image_size % self.patch_size) != 0:
            raise ValueError(f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})")

        patch_dim = self.in_ch * self.patch_size * self.patch_size
        self.patch_encoder = _PatchEncoder(patch_dim, self.embed_dim, depth=enc_depth)
        self.patch_decoder = _PatchDecoder(self.embed_dim, patch_dim, depth=dec_depth)
        self.to_recon_feat = nn.Conv2d(self.in_ch, 32, kernel_size=3, padding=1)
        self.recon_head = _build_recon_head(int(out_ch))

    def _forward_one(self, x: torch.Tensor, pixel_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        tokens = patchify(x, self.patch_size)
        if pixel_mask is None:
            patch_mask = torch.zeros((b, tokens.shape[1]), device=x.device, dtype=x.dtype)
        else:
            patch_mask = downsample_mask_to_patches(pixel_mask, self.patch_size)

        latent = self.patch_encoder(tokens, patch_mask)
        pred_tokens = self.patch_decoder(latent)
        recon_feat = unpatchify(pred_tokens, self.patch_size, h, w)
        recon_feat = self.to_recon_feat(recon_feat)
        recon = self.recon_head(recon_feat)
        return recon, latent

    def encoder_state_dict_prefixes(self) -> tuple[str, ...]:
        return ("patch_encoder",)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for p in self.patch_encoder.parameters():
            p.requires_grad = bool(trainable)

    def reset_contrastive_projection_heads(self) -> None:
        return None

    def param_count_breakdown(self) -> Dict[str, int]:
        total = count_parameters(self)
        enc = count_parameters(self.patch_encoder)
        dec = (
            count_parameters(self.patch_decoder)
            + count_parameters(self.to_recon_feat)
            + count_parameters(self.recon_head)
        )
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
        if int(view) == 2:
            x = flip_lr(x)
        x_in = apply_pixel_mask(x, pixel_mask)
        b, _, h, w = x_in.shape
        tokens = patchify(x_in, self.patch_size)
        if pixel_mask is None:
            patch_mask = torch.zeros((b, tokens.shape[1]), device=x.device, dtype=x.dtype)
        else:
            patch_mask = downsample_mask_to_patches(pixel_mask, self.patch_size)
        latent = self.patch_encoder(tokens, patch_mask)
        gh, gw = h // self.patch_size, w // self.patch_size
        return latent.reshape(b, gh, gw, self.embed_dim)

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

        x1 = apply_pixel_mask(x, pixel_mask)
        recon_raw_orig, _ = self._forward_one(x1, pixel_mask=pixel_mask)

        if self.single_view:
            return recon_raw_orig, None, None, None

        x2 = apply_pixel_mask(flip_lr(x), pixel_mask)
        recon_raw_flip, _ = self._forward_one(x2, pixel_mask=pixel_mask)
        return recon_raw_orig, recon_raw_flip, None, None


__all__ = ["MAEDualViewSSL"]
