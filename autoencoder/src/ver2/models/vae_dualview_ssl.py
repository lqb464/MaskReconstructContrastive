from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .ae_blocks import kl_divergence, reparameterize
from .model_utils import count_parameters
from .vanilla_ae_base import VanillaAEBase


class VAEDualViewSSL(VanillaAEBase):
    """
    Variational AutoEncoder on full images (no pixel masking).
    """

    uses_pixel_mask = False
    vis_mode = "full"

    def __init__(
        self,
        *,
        in_ch: int = 1,
        base_ch: int = 32,
        out_ch: int = 1,
        latent_dim: int = 256,
        use_gn: bool = False,
        enable_reconstruct: bool = True,
        single_view: bool = False,
    ):
        super().__init__(
            in_ch=in_ch,
            base_ch=base_ch,
            out_ch=out_ch,
            use_gn=use_gn,
            enable_reconstruct=enable_reconstruct,
            single_view=single_view,
        )
        self.latent_dim = int(latent_dim)
        self.last_kl_loss: torch.Tensor | None = None

        self.mu_head = nn.Conv2d(base_ch * 8, self.latent_dim, kernel_size=1)
        self.logvar_head = nn.Conv2d(base_ch * 8, self.latent_dim, kernel_size=1)
        self.latent_to_spatial = nn.Conv2d(self.latent_dim, base_ch * 8, kernel_size=1)

    def _encode_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        mu_map = self.mu_head(feat)
        logvar_map = self.logvar_head(feat)
        mu = mu_map.mean(dim=(2, 3))
        logvar = logvar_map.mean(dim=(2, 3))
        z_map = reparameterize(mu_map, logvar_map)
        z = self.latent_to_spatial(z_map)
        return z, mu, logvar, feat

    def _forward_one(self, x: torch.Tensor, pixel_mask: Optional[torch.Tensor]) -> torch.Tensor:
        del pixel_mask
        h, w = x.shape[-2:]
        z, mu, logvar, _ = self._encode_latent(x)
        logits = self.decoder(z, target_size=(h, w))
        self._last_kl = kl_divergence(mu, logvar)
        return logits

    def encoder_state_dict_prefixes(self) -> tuple[str, ...]:
        return ("encoder", "mu_head", "logvar_head")

    def set_encoder_trainable(self, trainable: bool) -> None:
        for module in (self.encoder, self.mu_head, self.logvar_head):
            for p in module.parameters():
                p.requires_grad = bool(trainable)

    def param_count_breakdown(self) -> Dict[str, int]:
        total = count_parameters(self)
        enc = count_parameters(self.encoder) + count_parameters(self.mu_head) + count_parameters(self.logvar_head)
        dec = count_parameters(self.latent_to_spatial) + count_parameters(self.decoder)
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

    def forward(
        self,
        x: torch.Tensor,
        pixel_mask: Optional[torch.Tensor],
        plane_one_hot: torch.Tensor,
    ):
        del pixel_mask
        if plane_one_hot.shape[0] != x.shape[0]:
            raise ValueError(
                f"plane_one_hot batch ({plane_one_hot.shape[0]}) must match input batch ({x.shape[0]})."
            )

        if not self.enable_reconstruct:
            self.last_kl_loss = None
            return None, None, None, None

        recon_raw_orig = self._forward_one(x, None)
        kl_orig = self._last_kl

        if self.single_view:
            self.last_kl_loss = kl_orig
            return recon_raw_orig, None, None, None

        from .model_utils import flip_lr

        recon_raw_flip = self._forward_one(flip_lr(x), None)
        kl_flip = self._last_kl
        self.last_kl_loss = kl_orig + kl_flip
        return recon_raw_orig, recon_raw_flip, None, None


__all__ = ["VAEDualViewSSL"]
