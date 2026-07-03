from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ae_blocks import apply_pixel_mask, kl_divergence, reparameterize
from .model_utils import count_parameters, flip_lr
from .unet_dualview_ssl import UNetDualViewSSL, _ResBlock, _UpBlock


class VAEDualViewSSL(nn.Module):
    """
    Variational AutoEncoder with UNet encoder/decoder (skip connections + skip masking).
    Exposes `last_kl_loss` after each forward pass for the trainer to consume.
    """

    def __init__(
        self,
        *,
        in_ch: int = 1,
        base_ch: int = 16,
        out_ch: int = 1,
        latent_dim: int = 256,
        use_gn: bool = False,
        use_se: bool = False,
        enable_reconstruct: bool = True,
        enable_contrastive: bool = False,
        single_view: bool = False,
    ):
        super().__init__()
        self.enable_reconstruct = bool(enable_reconstruct)
        self.enable_contrastive = bool(enable_contrastive)
        self.single_view = bool(single_view)
        self.latent_dim = int(latent_dim)
        self.last_kl_loss: torch.Tensor | None = None

        if self.enable_contrastive:
            raise ValueError("VAEDualViewSSL currently supports reconstruction-only (enable_contrastive=False).")
        if int(out_ch) < 1:
            raise ValueError(f"out_ch must be >=1, got {out_ch}")

        self.enc1 = _ResBlock(in_ch, base_ch, use_gn=use_gn, se=False)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ResBlock(base_ch, base_ch * 2, use_gn=use_gn, se=False)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _ResBlock(base_ch * 2, base_ch * 4, use_gn=use_gn, se=False)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _ResBlock(base_ch * 4, base_ch * 8, use_gn=use_gn, se=use_se)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = _ResBlock(base_ch * 8, base_ch * 8, use_gn=use_gn, se=use_se)

        self.mu_head = nn.Conv2d(base_ch * 8, self.latent_dim, kernel_size=1)
        self.logvar_head = nn.Conv2d(base_ch * 8, self.latent_dim, kernel_size=1)
        self.latent_to_spatial = nn.Conv2d(self.latent_dim, base_ch * 8, kernel_size=1)

        self.up1 = _UpBlock(base_ch * 8, base_ch * 8, base_ch * 4, use_gn=use_gn)
        self.up2 = _UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, use_gn=use_gn)
        self.up3 = _UpBlock(base_ch * 2, base_ch * 2, base_ch, use_gn=use_gn)
        self.up4 = _UpBlock(base_ch, base_ch, base_ch, use_gn=use_gn)
        self.out_conv = nn.Conv2d(base_ch, int(out_ch), kernel_size=1)

    @staticmethod
    def _apply_pixel_mask(x: torch.Tensor, pixel_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return UNetDualViewSSL._apply_pixel_mask(x, pixel_mask)

    def _encode(self, x: torch.Tensor):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        b = self.bottleneck(self.pool4(s4))
        return s1, s2, s3, s4, b

    def _decode(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        s3: torch.Tensor,
        s4: torch.Tensor,
        b: torch.Tensor,
        pixel_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        m1 = m2 = m3 = m4 = None
        if pixel_mask is not None:
            m4 = F.interpolate(pixel_mask, size=s4.shape[-2:], mode="nearest")
            m3 = F.interpolate(pixel_mask, size=s3.shape[-2:], mode="nearest")
            m2 = F.interpolate(pixel_mask, size=s2.shape[-2:], mode="nearest")
            m1 = F.interpolate(pixel_mask, size=s1.shape[-2:], mode="nearest")
        x = self.up1(b, s4, skip_mask=m4)
        x = self.up2(x, s3, skip_mask=m3)
        x = self.up3(x, s2, skip_mask=m2)
        x = self.up4(x, s1, skip_mask=m1)
        return self.out_conv(x)

    def _encode_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        s1, s2, s3, s4, b = self._encode(x)
        mu_map = self.mu_head(b)
        logvar_map = self.logvar_head(b)
        mu = mu_map.mean(dim=(2, 3))
        logvar = logvar_map.mean(dim=(2, 3))
        z_map = reparameterize(mu_map, logvar_map)
        z = self.latent_to_spatial(z_map)
        skips = (s1, s2, s3, s4)
        return z, mu, logvar, skips

    def _forward_one(
        self,
        x: torch.Tensor,
        pixel_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar, skips = self._encode_latent(x)
        s1, s2, s3, s4 = skips
        logits = self._decode(s1, s2, s3, s4, z, pixel_mask=pixel_mask)
        kl = kl_divergence(mu, logvar)
        return logits, z, kl

    def encoder_state_dict_prefixes(self) -> tuple[str, ...]:
        return (
            "enc1",
            "pool1",
            "enc2",
            "pool2",
            "enc3",
            "pool3",
            "enc4",
            "pool4",
            "bottleneck",
            "mu_head",
            "logvar_head",
        )

    def set_encoder_trainable(self, trainable: bool) -> None:
        prefixes = self.encoder_state_dict_prefixes()
        for name, p in self.named_parameters():
            if name.startswith(prefixes):
                p.requires_grad = bool(trainable)

    def reset_contrastive_projection_heads(self) -> None:
        return None

    def param_count_breakdown(self) -> Dict[str, int]:
        total = count_parameters(self)
        enc = sum(
            count_parameters(m)
            for m in [
                self.enc1,
                self.enc2,
                self.enc3,
                self.enc4,
                self.bottleneck,
                self.mu_head,
                self.logvar_head,
            ]
        )
        dec = sum(
            count_parameters(m)
            for m in [self.latent_to_spatial, self.up1, self.up2, self.up3, self.up4, self.out_conv]
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
        x = self._apply_pixel_mask(x, pixel_mask)
        if int(view) == 2:
            x = flip_lr(x)
        z, _, _, _ = self._encode_latent(x)
        return z.permute(0, 2, 3, 1).contiguous()

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
            self.last_kl_loss = None
            return None, None, None, None

        x1 = self._apply_pixel_mask(x, pixel_mask)
        recon_raw_orig, _, kl_orig = self._forward_one(x1, pixel_mask=pixel_mask)

        if self.single_view:
            self.last_kl_loss = kl_orig
            return recon_raw_orig, None, None, None

        x2 = self._apply_pixel_mask(flip_lr(x), pixel_mask)
        recon_raw_flip, _, kl_flip = self._forward_one(x2, pixel_mask=pixel_mask)
        self.last_kl_loss = kl_orig + kl_flip
        return recon_raw_orig, recon_raw_flip, None, None


__all__ = ["VAEDualViewSSL"]
