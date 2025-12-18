
# =============================================
# File: model.py
# SwinUNet Dual View SSL with Plane Conditioning (Phase 1 + Phase A)
#
# Phase 1 baseline:
# - Dual view: view1 masked, view2 flip_lr(x) masked with SAME mask (mask NOT flipped)
# - Split weights up to Stage1 (PatchEmbed, Stage0, Merge0, Stage1 separate per view)
# - Shared trunk from Stage2 (Merge1, PlaneCondition, Stage2, Merge2, Stage3 shared)
# - Contrastive head uses pooled bottleneck AFTER shared trunk (InfoNCE outside)
#
# Phase A extension:
# - Dual Reconstruction Heads (Original + Flip)
#   * Decoder trunk is shared (view1 only) producing a feature map at full resolution.
#   * Two lightweight reconstruction heads run in parallel on the SAME decoder feature map:
#       - recon_head_orig -> logits for original image
#       - recon_head_flip -> logits for flipped image
#
# Outputs:
#   recon_raw_orig: [B,1,H,W] logits
#   recon_raw_flip: [B,1,H,W] logits
#   z1, z2: [B,D] embeddings (contrastive) unchanged
# =============================================
from __future__ import annotations

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -------------------------
# Utility
# -------------------------
def flip_lr(x: torch.Tensor) -> torch.Tensor:
    """Left-right flip on width dimension for NCHW tensor."""
    return torch.flip(x, dims=[-1])


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous()


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2).contiguous()


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# -------------------------
# Patch ops
# -------------------------
class PatchEmbed(nn.Module):
    """Conv patch embedding: NCHW -> NHWC tokens."""
    def __init__(self, in_ch: int = 1, embed_dim: int = 96, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return nchw_to_nhwc(x)


class PatchMerging(nn.Module):
    """Downsample by 2x: [B,H,W,C] -> [B,H/2,W/2,2C]"""
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        if (H % 2) == 1:
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
            H += 1
        if (W % 2) == 1:
            x = F.pad(x, (0, 0, 0, 1, 0, 0))
            W += 1

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
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
        x = self.expand(x)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C // 2)
        x = self.norm(x)
        return x


class FinalPatchExpand(nn.Module):
    """Upsample by patch_size: [B,H/P,W/P,C] -> [B,H,W,out_dim] (channel-last)"""
    def __init__(self, dim: int, patch_size: int, out_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.proj = nn.Linear(dim, (patch_size * patch_size) * out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        p = self.patch_size
        x = self.proj(x)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=p, p2=p, c=self.out_dim)
        return x


# -------------------------
# Swin blocks (minimal)
# -------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    return rearrange(x, "b (nh ws1) (nw ws2) c -> (b nh nw) ws1 ws2 c", ws1=window_size, ws2=window_size)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
    return rearrange(
        windows,
        "(b nh nw) ws1 ws2 c -> b (nh ws1) (nw ws2) c",
        b=B,
        ws1=window_size,
        ws2=window_size,
        nh=H // window_size,
        nw=W // window_size,
    )


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
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

        ws = window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads))

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        rel_coords[:, :, 0] += ws - 1
        rel_coords[:, :, 1] += ws - 1
        rel_coords[:, :, 0] *= 2 * ws - 1
        rel_pos_index = rel_coords.sum(-1)
        self.register_buffer("relative_position_index", rel_pos_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        if attn_mask is not None:
            nW = attn_mask.size(0)
            attn = attn.view(Bn // nW, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        out = self.proj_drop(self.proj(out))
        return out


def compute_attn_mask(H: int, W: int, window_size: int, shift_size: int, device: torch.device) -> Optional[torch.Tensor]:
    if shift_size == 0:
        return None
    img_mask = torch.zeros((1, H, W, 1), device=device)
    cnt = 0
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        ws = self.window_size
        ss = self.shift_size

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

        x_windows = window_partition(x, ws).view(-1, ws * ws, C)
        attn_windows = self.attn(x_windows, attn_mask=attn_mask).view(-1, ws, ws, C)
        x = window_reverse(attn_windows, ws, Hp, Wp, B)

        if ss > 0:
            x = torch.roll(x, shifts=(ss, ss), dims=(1, 2))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        x = x[:, :H, :W, :].contiguous()
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0 if (i % 2 == 0) else window_size // 2),
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# -------------------------
# Plane conditioning
# -------------------------
class PlaneCondition(nn.Module):
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
        B, H, W, C = f.shape
        p = self.mlp(plane_one_hot)
        if self.method == "add":
            return f + p.view(B, 1, 1, C)
        gamma, beta = p.chunk(2, dim=-1)
        return f * (1.0 + gamma.view(B, 1, 1, C)) + beta.view(B, 1, 1, C)


# -------------------------
# Decoder up blocks
# -------------------------
class SwinUpBlock(nn.Module):
    def __init__(self, in_dim: int, skip_dim: int, out_dim: int, depth: int, num_heads: int, window_size: int):
        super().__init__()
        self.up = PatchExpand(in_dim)
        self.proj = nn.Linear((in_dim // 2) + skip_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.layer = BasicLayer(out_dim, depth=depth, num_heads=num_heads, window_size=window_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # align
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            dh = skip.shape[1] - x.shape[1]
            dw = skip.shape[2] - x.shape[2]
            x = F.pad(x, (0, 0, 0, max(dw, 0), 0, max(dh, 0)))
            x = x[:, :skip.shape[1], :skip.shape[2], :]
        x = torch.cat([x, skip], dim=-1)
        x = self.norm(self.proj(x))
        x = self.layer(x)
        return x


class ProjectionHead(nn.Module):
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
        return F.normalize(x, dim=-1)


# -------------------------
# Main model
# -------------------------
class SwinUNetDualViewSSLPhase1(nn.Module):
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

        C0 = embed_dim
        C1 = 2 * C0
        C2 = 2 * C1
        C3 = 2 * C2

        # Early encoders (separate)
        self.patch_embed_1 = PatchEmbed(in_ch=in_ch, embed_dim=C0, patch_size=patch_size)
        self.stage0_1 = BasicLayer(dim=C0, depth=depths[0], num_heads=num_heads[0], window_size=window_size)
        self.merge0_1 = PatchMerging(dim=C0)
        self.stage1_1 = BasicLayer(dim=C1, depth=depths[1], num_heads=num_heads[1], window_size=window_size)

        self.patch_embed_2 = PatchEmbed(in_ch=in_ch, embed_dim=C0, patch_size=patch_size)
        self.stage0_2 = BasicLayer(dim=C0, depth=depths[0], num_heads=num_heads[0], window_size=window_size)
        self.merge0_2 = PatchMerging(dim=C0)
        self.stage1_2 = BasicLayer(dim=C1, depth=depths[1], num_heads=num_heads[1], window_size=window_size)

        # Shared trunk (Stage2+)
        self.merge1 = PatchMerging(dim=C1)
        self.plane_cond = PlaneCondition(in_dim=2, feat_dim=C2, method=plane_inject_method)
        self.stage2 = BasicLayer(dim=C2, depth=depths[2], num_heads=num_heads[2], window_size=window_size)
        self.merge2 = PatchMerging(dim=C2)
        self.stage3 = BasicLayer(dim=C3, depth=depths[3], num_heads=num_heads[3], window_size=window_size)

        # Contrastive head (unchanged)
        self.proj = ProjectionHead(in_dim=C3, proj_dim=proj_dim)

        # Decoder trunk (view1 only), produces feat_full (shared for both recon heads)
        self.up2 = SwinUpBlock(in_dim=C3, skip_dim=C2, out_dim=C2, depth=2, num_heads=num_heads[2], window_size=window_size)
        self.up1 = SwinUpBlock(in_dim=C2, skip_dim=C1, out_dim=C1, depth=2, num_heads=num_heads[1], window_size=window_size)
        self.up0 = SwinUpBlock(in_dim=C1, skip_dim=C0, out_dim=C0, depth=2, num_heads=num_heads[0], window_size=window_size)

        self.final_up = FinalPatchExpand(dim=C0, patch_size=patch_size, out_dim=32)

        # Phase A: two parallel lightweight reconstruction heads
        self.recon_head_orig = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.recon_head_flip = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def _shared_trunk(self, s1: torch.Tensor, plane_one_hot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u2 = self.merge1(s1)
        u2 = self.plane_cond(u2, plane_one_hot)
        s2 = self.stage2(u2)
        u3 = self.merge2(s2)
        b = self.stage3(u3)
        return s2, b

    @torch.no_grad()
    def encode_bottleneck(self, x: torch.Tensor, plane_one_hot: torch.Tensor, view: int = 1) -> torch.Tensor:
        if view == 1:
            f0 = self.patch_embed_1(x)
            s0 = self.stage0_1(f0)
            f1 = self.merge0_1(s0)
            s1 = self.stage1_1(f1)
        else:
            f0 = self.patch_embed_2(x)
            s0 = self.stage0_2(f0)
            f1 = self.merge0_2(s0)
            s1 = self.stage1_2(f1)
        _, b = self._shared_trunk(s1, plane_one_hot)
        return b

    def param_count_breakdown(self) -> Dict[str, int]:
        """Return parameter counts for logging."""
        encoder_modules = [
            self.patch_embed_1, self.stage0_1, self.merge0_1, self.stage1_1,
            self.patch_embed_2, self.stage0_2, self.merge0_2, self.stage1_2,
            self.merge1, self.plane_cond, self.stage2, self.merge2, self.stage3,
            self.proj,
        ]
        decoder_trunk_modules = [self.up2, self.up1, self.up0, self.final_up]
        head_modules = [self.recon_head_orig, self.recon_head_flip]

        enc = sum(count_parameters(m) for m in encoder_modules)
        dec = sum(count_parameters(m) for m in decoder_trunk_modules)
        heads = sum(count_parameters(m) for m in head_modules)
        total = count_parameters(self)
        return {
            "total": total,
            "encoder": enc,
            "decoder_trunk": dec,
            "recon_heads": heads,
        }

    def forward(
        self,
        x: torch.Tensor,
        pixel_mask: torch.Tensor,
        plane_one_hot: torch.Tensor,
        return_embeddings: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          recon_raw_orig: [B,1,H,W] logits
          recon_raw_flip: [B,1,H,W] logits
          z1, z2: [B,D] embeddings (if return_embeddings else empty)
        """
        # dual views for contrastive (unchanged)
        x1_masked = x * (1.0 - pixel_mask)
        x2_masked = flip_lr(x) * (1.0 - pixel_mask)  # mask NOT flipped

        # view1 early encode
        f0_1 = self.patch_embed_1(x1_masked)
        s0_1 = self.stage0_1(f0_1)
        f1_1 = self.merge0_1(s0_1)
        s1_1 = self.stage1_1(f1_1)

        s2_1, b1 = self._shared_trunk(s1_1, plane_one_hot)

        # view2 early encode
        f0_2 = self.patch_embed_2(x2_masked)
        s0_2 = self.stage0_2(f0_2)
        f1_2 = self.merge0_2(s0_2)
        s1_2 = self.stage1_2(f1_2)
        _, b2 = self._shared_trunk(s1_2, plane_one_hot)

        # contrastive embeddings
        if return_embeddings:
            z1 = self.proj(b1.mean(dim=(1, 2)))
            z2 = self.proj(b2.mean(dim=(1, 2)))
        else:
            z1 = torch.empty((x.size(0), 0), device=x.device)
            z2 = torch.empty((x.size(0), 0), device=x.device)

        # decoder trunk (view1 only) -> shared feature map
        d2 = self.up2(b1, s2_1)
        d1 = self.up1(d2, s1_1)
        d0 = self.up0(d1, s0_1)
        feat_full = self.final_up(d0)                 # [B,H,W,32]
        feat_full_nchw = nhwc_to_nchw(feat_full)      # [B,32,H,W]

        # Phase A: two parallel heads (logits)
        recon_raw_orig = self.recon_head_orig(feat_full_nchw)
        recon_raw_flip = self.recon_head_flip(feat_full_nchw)

        return recon_raw_orig, recon_raw_flip, z1, z2


__all__ = [
    "SwinUNetDualViewSSLPhase1",
    "flip_lr",
]
