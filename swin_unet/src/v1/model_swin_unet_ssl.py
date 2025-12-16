# =============================================
# File: model_swin_unet_ssl.py
# Swin Transformer encoder + CNN decoder
# Phase 1: MIM + Contrastive (dual-view, shared weights)
# =============================================

from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Utils
# -------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    x: [B, H, W, C]
    return: [num_windows*B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
         .contiguous()
         .view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    windows: [num_windows*B, window_size, window_size, C]
    return: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    x = (
        x.permute(0, 1, 3, 2, 4, 5)
         .contiguous()
         .view(B, H, W, -1)
    )
    return x


# -------------------------------------------------
# MLP
# -------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -------------------------------------------------
# Window Attention with relative position bias
# -------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # relative position bias
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )  # [2, Ws, Ws]
        coords_flat = coords.flatten(1)  # [2, Ws*Ws]
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        rel_coords[:, :, 0] += window_size - 1
        rel_coords[:, :, 1] += window_size - 1
        rel_coords[:, :, 0] *= 2 * window_size - 1
        rel_position_index = rel_coords.sum(-1)

        self.register_buffer("rel_position_index", rel_position_index)
        self.rel_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )

        nn.init.trunc_normal_(self.rel_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [num_windows*B, Ws*Ws, C]
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel_bias = self.rel_position_bias_table[self.rel_position_index.view(-1)]
        rel_bias = rel_bias.view(N, N, -1).permute(2, 0, 1)
        attn = attn + rel_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out


# -------------------------------------------------
# Swin Block
# -------------------------------------------------

class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        x_windows = window_partition(x, self.attn.window_size)
        x_windows = x_windows.view(-1, self.attn.window_size ** 2, C)

        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(
            -1,
            self.attn.window_size,
            self.attn.window_size,
            C,
        )

        x = window_reverse(
            attn_windows,
            self.attn.window_size,
            H,
            W,
        )
        x = x.view(B, H * W, C)
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        return x


# -------------------------------------------------
# Patch Embedding
# -------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)  # [B, C, H', W']
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


# -------------------------------------------------
# CNN Decoder (light UNet-style)
# -------------------------------------------------

class CNNDecoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim // 4, 3, padding=1)
        self.conv3 = nn.Conv2d(embed_dim // 4, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv3(x)
        return x


# -------------------------------------------------
# Swin UNet SSL (Phase 1)
# -------------------------------------------------

class SwinUNetSSL(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        window_size: int = 8,
        bottleneck_dim: int = 128,
        proj_dim: int = 128,
        plane_dim: int = 2,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim
        )

        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # meta embedding (x,y + plane one-hot)
        self.meta_fc = nn.Sequential(
            nn.Linear(2 + plane_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        self.decoder = CNNDecoder(embed_dim)

        # projection heads
        self.embed_fc = nn.Linear(embed_dim, bottleneck_dim)
        self.proj = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim, bias=False),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, proj_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        meta_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B,1,256,256] (masked already)
        meta_vec: [B, 2 + plane_dim]
        """
        B = x.size(0)
        x, H, W = self.patch_embed(x)

        for blk in self.blocks:
            x = blk(x, H, W)

        x = self.norm(x)
        feat = x.mean(dim=1)

        if meta_vec is not None:
            feat = feat + self.meta_fc(meta_vec)

        # reconstruction
        feat_map = x.transpose(1, 2).view(B, -1, H, W)
        recon = self.decoder(feat_map)
        recon = F.interpolate(
            recon,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        return recon

    def encoder_embed(
        self,
        x: torch.Tensor,
        meta_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, H, W = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.norm(x)
        h = x.mean(dim=1)
        if meta_vec is not None:
            h = h + self.meta_fc(meta_vec)
        h = self.embed_fc(h)
        z = F.normalize(self.proj(h), dim=-1)
        return z, h
