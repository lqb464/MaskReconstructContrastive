from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .ae_blocks import downsample_mask_to_patches, patchify, unpatchify
from .model_utils import count_parameters, flip_lr


def _init_pos_embed(pos_embed: nn.Parameter, std: float = 0.02) -> None:
    nn.init.trunc_normal_(pos_embed, std=std)


class _PatchEncoder(nn.Module):
    """Encode patch tokens; masked positions use a learnable mask token."""

    def __init__(self, patch_dim: int, embed_dim: int, depth: int = 4):
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
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

    def forward(
        self,
        tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        x = self.proj(tokens) + pos_embed
        mask = patch_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * (1.0 - mask) + self.mask_token.to(dtype=x.dtype) * mask
        x = self.transformer(x)
        return self.norm(x)


class _PatchDecoder(nn.Module):
    """Lightweight transformer decoder for patch reconstruction."""

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

    def forward(self, tokens: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        x = tokens + pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x)


class MAEDualViewSSL(nn.Module):
    """
    Masked AutoEncoder with patch transformer, mask token, and positional embeddings.
    Training masks use hemisphere anti-mirror patches (swin_unet-compatible), applied at token level.
    """

    uses_pixel_mask = True
    vis_mode = "masked"

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
        single_view: bool = False,
        base_ch: int = 32,
        use_gn: bool = False,
    ):
        super().__init__()
        del base_ch, use_gn
        self.enable_reconstruct = bool(enable_reconstruct)
        self.single_view = bool(single_view)
        self.in_ch = int(in_ch)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)

        if int(out_ch) != int(in_ch):
            raise ValueError(f"MAEDualViewSSL expects out_ch == in_ch, got out_ch={out_ch}, in_ch={in_ch}")
        if (self.image_size % self.patch_size) != 0:
            raise ValueError(f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})")

        gh = self.image_size // self.patch_size
        gw = self.image_size // self.patch_size
        num_patches = gh * gw

        patch_dim = self.in_ch * self.patch_size * self.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        _init_pos_embed(self.pos_embed)

        self.patch_encoder = _PatchEncoder(patch_dim, self.embed_dim, depth=enc_depth)
        self.patch_decoder = _PatchDecoder(self.embed_dim, patch_dim, depth=dec_depth)
        self.recon_head = nn.Conv2d(self.in_ch, int(out_ch), kernel_size=1)

    def _pos_embed_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.pos_embed.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def _forward_one(self, x: torch.Tensor, pixel_mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, _, h, w = x.shape
        tokens = patchify(x, self.patch_size)
        if pixel_mask is None:
            patch_mask = torch.zeros((b, tokens.shape[1]), device=x.device, dtype=x.dtype)
        else:
            patch_mask = downsample_mask_to_patches(pixel_mask, self.patch_size)

        pos = self._pos_embed_batch(b, x.device, x.dtype)
        latent = self.patch_encoder(tokens, patch_mask, pos)
        pred_tokens = self.patch_decoder(latent, pos)
        recon = unpatchify(pred_tokens, self.patch_size, h, w)
        return self.recon_head(recon)

    def encoder_state_dict_prefixes(self) -> tuple[str, ...]:
        return ("patch_encoder", "pos_embed")

    def set_encoder_trainable(self, trainable: bool) -> None:
        for module in (self.patch_encoder,):
            for p in module.parameters():
                p.requires_grad = bool(trainable)
        self.pos_embed.requires_grad = bool(trainable)

    def param_count_breakdown(self) -> Dict[str, int]:
        total = count_parameters(self)
        enc = count_parameters(self.patch_encoder) + count_parameters(self.pos_embed)
        dec = count_parameters(self.patch_decoder) + count_parameters(self.recon_head)
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
        b, _, h, w = x.shape
        tokens = patchify(x, self.patch_size)
        if pixel_mask is None:
            patch_mask = torch.zeros((b, tokens.shape[1]), device=x.device, dtype=x.dtype)
        else:
            patch_mask = downsample_mask_to_patches(pixel_mask, self.patch_size)
        pos = self._pos_embed_batch(b, x.device, x.dtype)
        latent = self.patch_encoder(tokens, patch_mask, pos)
        gh, gw = h // self.patch_size, w // self.patch_size
        return latent.reshape(b, gh, gw, self.embed_dim)

    def reset_contrastive_projection_heads(self) -> None:
        return None

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


__all__ = ["MAEDualViewSSL"]
