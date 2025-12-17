
# =============================================
# File: model_phase1.py
# SwinUNet Dual View SSL with Plane Conditioning (Phase 1)
# - Dual view: view2 is flip_lr(x) with SAME pixel_mask (mask not flipped)
# - Split weights up to Stage1 (PatchEmbed, Stage0, Merge0, Stage1 are separate)
# - Shared trunk from Stage2 (Merge1, Stage2, Merge2, Stage3 shared)
# - Decoder exists ONLY for view1 (masked reconstruction head)
# - Contrastive head uses pooled bottleneck AFTER shared trunk (InfoNCE outside)
#
# Notes:
# - Phase 1: no SACA / no SAH implemented here.
# - Uses only torch + einops (no timm dependency).
# - Tensors are channel-last inside transformer blocks: [B, H, W, C]
# =============================================
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -------------------------
# Utility
# -------------------------
def _to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))


def flip_lr(x: torch.Tensor) -> torch.Tensor:
    """Left-right flip on width dimension for NCHW tensor."""
    return torch.flip(x, dims=[-1])


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()


# -------------------------
# Patch operations
# -------------------------
class PatchEmbed(nn.Module):
    """Conv patch embedding: NCHW -> NHWC tokens."""
    def __init__(self, in_ch: int = 1, embed_dim: int = 96, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B, H/P, W/P, C0]
        x = self.proj(x)
        x = nchw_to_nhwc(x)
        return x


class PatchMerging(nn.Module):
    """Downsample by 2x: [B,H,W,C] -> [B,H/2,W/2,2C]"""
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        # pad if odd
        if (H % 2) == 1:
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
            H += 1
        if (W % 2) == 1:
            x = F.pad(x, (0, 0, 0, 1, 0, 0))
            W += 1
        x0 = x[:, 0::2, 0::2, :]  # [B,H/2,W/2,C]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B,H/2,W/2,4C]
        x = self.norm(x)
        x = self.reduction(x)  # [B,H/2,W/2,2C]
        return x


class PatchExpand(nn.Module):
    """Upsample by 2x: [B,H,W,C] -> [B,2H,2W,C/2]"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = self.expand(x)  # [B,H,W,2C]
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C // 2)
        x = self.norm(x)
        return x


class FinalPatchExpand(nn.Module):
    """Upsample by patch_size: [B,H/P,W/P,C] -> [B,H,W,C_out]"""
    def __init__(self, dim: int, patch_size: int, out_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.proj = nn.Linear(dim, (patch_size * patch_size) * out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        p = self.patch_size
        x = self.proj(x)  # [B,H,W,p*p*out_dim]
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=p, p2=p, c=self.out_dim)
        return x


# -------------------------
# Window attention (Swin)
# -------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """[B,H,W,C] -> [num_windows*B, window, window, C]"""
    B, H, W, C = x.shape
    x = rearrange(x, "b (nh ws1) (nw ws2) c -> (b nh nw) ws1 ws2 c",
                  ws1=window_size, ws2=window_size)
    return x


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
    """[num_windows*B, ws, ws, C] -> [B,H,W,C]"""
    x = rearrange(windows, "(b nh nw) ws1 ws2 c -> b (nh ws1) (nw ws2) c",
                  b=B, ws1=window_size, ws2=window_size, nh=H // window_size, nw=W // window_size)
    return x


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # relative position bias table
        ws = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )
        # relative position index
        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, ws, ws]
        coords_flatten = torch.flatten(coords, 1)  # [2, ws*ws]
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, ws*ws, ws*ws]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()  # [ws*ws, ws*ws, 2]
        rel_coords[:, :, 0] += ws - 1
        rel_coords[:, :, 1] += ws - 1
        rel_coords[:, :, 0] *= 2 * ws - 1
        rel_pos_index = rel_coords.sum(-1)  # [ws*ws, ws*ws]
        self.register_buffer("relative_position_index", rel_pos_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [nW*B, N, C], N=ws*ws
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, Bn, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [Bn, heads, N, N]

        # add relative bias
        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()  # heads, N, N
        attn = attn + rel_bias.unsqueeze(0)

        if attn_mask is not None:
            # attn_mask: [nW, N, N]
            nW = attn_mask.size(0)
            attn = attn.view(Bn // nW, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def compute_attn_mask(H: int, W: int, window_size: int, shift_size: int, device: torch.device) -> Optional[torch.Tensor]:
    """Compute attention mask for shifted windows."""
    if shift_size == 0:
        return None
    img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
    cnt = 0
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws, ws, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,W,C]
        B, H, W, C = x.shape
        ws = self.window_size
        ss = self.shift_size

        # pad to window
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        attn_mask = compute_attn_mask(Hp, Wp, ws, ss, x.device)

        shortcut = x
        x = self.norm1(x)
        if ss > 0:
            x = torch.roll(x, shifts=(-ss, -ss), dims=(1, 2))

        x_windows = window_partition(x, ws)  # [nW*B, ws, ws, C]
        x_windows = x_windows.view(-1, ws * ws, C)
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)
        attn_windows = attn_windows.view(-1, ws, ws, C)
        x = window_reverse(attn_windows, ws, Hp, Wp, B)

        if ss > 0:
            x = torch.roll(x, shifts=(ss, ss), dims=(1, 2))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        # remove padding
        x = x[:, :H, :W, :].contiguous()
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int):
        super().__init__()
        blocks = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            blocks.append(SwinTransformerBlock(dim, num_heads, window_size=window_size, shift_size=shift))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# -------------------------
# Plane conditioning
# -------------------------
class PlaneCondition(nn.Module):
    """
    plane_one_hot: [B,2] -> p: [B,C2]
    Injection at Stage2 entry:
      - add:  f = f + broadcast(p)
      - film: f = f * gamma(p) + beta(p)
    """
    def __init__(self, in_dim: int, feat_dim: int, method: str = "film", hidden: int = 128):
        super().__init__()
        self.method = method.lower().strip()
        if self.method not in {"film", "add"}:
            raise ValueError(f"PlaneCondition method must be 'film' or 'add', got {method}")

        if self.method == "add":
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, feat_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2 * feat_dim),
            )

    def forward(self, f: torch.Tensor, plane_one_hot: torch.Tensor) -> torch.Tensor:
        # f: [B,H,W,C]
        B, H, W, C = f.shape
        p = self.mlp(plane_one_hot)  # [B, C] or [B,2C]
        if self.method == "add":
            p = p.view(B, 1, 1, C)
            return f + p
        gamma, beta = p.chunk(2, dim=-1)
        gamma = gamma.view(B, 1, 1, C)
        beta = beta.view(B, 1, 1, C)
        return f * (1.0 + gamma) + beta


# -------------------------
# Decoder blocks
# -------------------------
class SwinUpBlock(nn.Module):
    """
    Upsample by 2x, concat skip, linear proj, then Swin blocks.
    """
    def __init__(self, in_dim: int, skip_dim: int, out_dim: int, depth: int, num_heads: int, window_size: int):
        super().__init__()
        self.up = PatchExpand(in_dim)  # -> out_dim (since out_dim = in_dim//2 in typical)
        self.proj = nn.Linear((in_dim // 2) + skip_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.layer = BasicLayer(out_dim, depth=depth, num_heads=num_heads, window_size=window_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad if mismatch
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            dh = skip.shape[1] - x.shape[1]
            dw = skip.shape[2] - x.shape[2]
            x = F.pad(x, (0, 0, 0, max(dw, 0), 0, max(dh, 0)))
            x = x[:, :skip.shape[1], :skip.shape[2], :]

        x = torch.cat([x, skip], dim=-1)
        x = self.proj(x)
        x = self.norm(x)
        x = self.layer(x)
        return x


# -------------------------
# Projection head for contrastive
# -------------------------
class ProjectionHead(nn.Module):
    """2-layer MLP projection head -> normalized output."""
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.normalize(x, dim=-1)
        return x


# -------------------------
# Main model
# -------------------------
class SwinUNetDualViewSSLPhase1(nn.Module):
    """
    Phase 1 model:
      - dual view forward (masked + flipped-masked)
      - reconstruction from view1 only
      - contrastive embeddings from bottlenecks b1, b2 after shared trunk
    """
    def __init__(
        self,
        in_ch: int = 1,
        image_size: int = 192,
        patch_size: int = 16,
        embed_dim: int = 96,
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: int = 7,
        proj_dim: int = 128,
        plane_inject_method: str = "film",
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size

        # channels: C0, C1=2C0, C2=4C0, C3=8C0
        C0 = embed_dim
        C1 = 2 * C0
        C2 = 2 * C1
        C3 = 2 * C2

        # -----------------
        # Early encoders (separate weights) up to Stage1
        # -----------------
        self.patch_embed_1 = PatchEmbed(in_ch=in_ch, embed_dim=C0, patch_size=patch_size)
        self.stage0_1 = BasicLayer(dim=C0, depth=depths[0], num_heads=num_heads[0], window_size=window_size)
        self.merge0_1 = PatchMerging(dim=C0)
        self.stage1_1 = BasicLayer(dim=C1, depth=depths[1], num_heads=num_heads[1], window_size=window_size)

        self.patch_embed_2 = PatchEmbed(in_ch=in_ch, embed_dim=C0, patch_size=patch_size)
        self.stage0_2 = BasicLayer(dim=C0, depth=depths[0], num_heads=num_heads[0], window_size=window_size)
        self.merge0_2 = PatchMerging(dim=C0)
        self.stage1_2 = BasicLayer(dim=C1, depth=depths[1], num_heads=num_heads[1], window_size=window_size)

        # -----------------
        # Shared trunk from Stage2
        # -----------------
        self.merge1 = PatchMerging(dim=C1)
        self.plane_cond = PlaneCondition(in_dim=2, feat_dim=C2, method=plane_inject_method)
        self.stage2 = BasicLayer(dim=C2, depth=depths[2], num_heads=num_heads[2], window_size=window_size)

        self.merge2 = PatchMerging(dim=C2)
        self.stage3 = BasicLayer(dim=C3, depth=depths[3], num_heads=num_heads[3], window_size=window_size)

        # -----------------
        # Contrastive head (shared)
        # -----------------
        self.proj = ProjectionHead(in_dim=C3, proj_dim=proj_dim)

        # -----------------
        # Decoder (view1 only)
        # Using mirrored depths/heads for up blocks (lightweight)
        # -----------------
        # Up from C3 -> C2, C2 -> C1, C1 -> C0
        self.up2 = SwinUpBlock(in_dim=C3, skip_dim=C2, out_dim=C2, depth=2, num_heads=num_heads[2], window_size=window_size)
        self.up1 = SwinUpBlock(in_dim=C2, skip_dim=C1, out_dim=C1, depth=2, num_heads=num_heads[1], window_size=window_size)
        self.up0 = SwinUpBlock(in_dim=C1, skip_dim=C0, out_dim=C0, depth=2, num_heads=num_heads[0], window_size=window_size)

        self.final_up = FinalPatchExpand(dim=C0, patch_size=patch_size, out_dim=32)
        self.recon_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    # --------
    # Encoder pieces
    # --------
    def _early_encode_view1(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = self.patch_embed_1(x)          # [B,H/P,W/P,C0]
        s0 = self.stage0_1(f0)              # skip0
        f1 = self.merge0_1(s0)              # [B,H/2P,W/2P,C1]
        s1 = self.stage1_1(f1)              # skip1
        return s0, s1, s1

    def _early_encode_view2(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f0 = self.patch_embed_2(x)
        s0 = self.stage0_2(f0)
        f1 = self.merge0_2(s0)
        s1 = self.stage1_2(f1)
        return s0, s1

    def _shared_trunk(self, s1: torch.Tensor, plane_one_hot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u2 = self.merge1(s1)                # [B,H/4P,W/4P,C2]
        u2 = self.plane_cond(u2, plane_one_hot)
        s2 = self.stage2(u2)                # skip2
        u3 = self.merge2(s2)                # [B,H/8P,W/8P,C3]
        b = self.stage3(u3)                 # bottleneck
        return s2, b, s2

    # --------
    # Public APIs
    # --------
    def encode_bottleneck(self, x_masked: torch.Tensor, plane_one_hot: torch.Tensor, view: int = 1) -> torch.Tensor:
        """
        Encode one view to bottleneck b: [B,H/8P,W/8P,C3]
        x_masked: NCHW
        """
        if view == 1:
            f0 = self.patch_embed_1(x_masked)
            s0 = self.stage0_1(f0)
            f1 = self.merge0_1(s0)
            s1 = self.stage1_1(f1)
        else:
            f0 = self.patch_embed_2(x_masked)
            s0 = self.stage0_2(f0)
            f1 = self.merge0_2(s0)
            s1 = self.stage1_2(f1)
        _, b, _ = self._shared_trunk(s1, plane_one_hot)
        return b

    def forward(
        self,
        x: torch.Tensor,
        pixel_mask: torch.Tensor,
        plane_one_hot: torch.Tensor,
        return_embeddings: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs:
          x: [B,1,H,W]
          pixel_mask: [B,1,H,W], 1 = masked
          plane_one_hot: [B,2]
        Returns:
          recon_view1: [B,1,H,W]
          z1: [B,D] (proj)
          z2: [B,D] (proj)
        """
        # construct views using SAME mask; mask is NOT flipped
        x1_masked = x * (1.0 - pixel_mask)
        x2_masked = flip_lr(x) * (1.0 - pixel_mask)

        # -----------------
        # View 1 encode (save skips for decoder)
        # -----------------
        f0_1 = self.patch_embed_1(x1_masked)
        s0_1 = self.stage0_1(f0_1)           # skip0_1
        f1_1 = self.merge0_1(s0_1)
        s1_1 = self.stage1_1(f1_1)           # skip1_1

        s2_1, b1, _ = self._shared_trunk(s1_1, plane_one_hot)  # skip2_1, bottleneck

        # -----------------
        # View 2 encode (shared trunk)
        # -----------------
        f0_2 = self.patch_embed_2(x2_masked)
        s0_2 = self.stage0_2(f0_2)
        f1_2 = self.merge0_2(s0_2)
        s1_2 = self.stage1_2(f1_2)

        _, b2, _ = self._shared_trunk(s1_2, plane_one_hot)

        # -----------------
        # Contrastive embeddings (after shared bottleneck)
        # -----------------
        if return_embeddings:
            z1 = b1.mean(dim=(1, 2))  # GAP over H,W
            z2 = b2.mean(dim=(1, 2))
            z1 = self.proj(z1)
            z2 = self.proj(z2)
        else:
            z1 = torch.empty((x.size(0), 0), device=x.device)
            z2 = torch.empty((x.size(0), 0), device=x.device)

        # -----------------
        # Decoder (view1 only) to reconstruct full resolution
        # -----------------
        d2 = self.up2(b1, s2_1)
        d1 = self.up1(d2, s1_1)
        d0 = self.up0(d1, s0_1)

        feat_full = self.final_up(d0)  # [B,H,W,32]
        recon = self.recon_head(nhwc_to_nchw(feat_full))
        return recon, z1, z2


__all__ = [
    "SwinUNetDualViewSSLPhase1",
]
