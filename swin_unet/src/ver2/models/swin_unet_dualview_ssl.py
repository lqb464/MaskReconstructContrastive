# =============================================
# File: model.py
# SwinUNet Dual View SSL with Plane Conditioning (Phase 1 + Phase A)
#
# Option A extension:
# - Add SACA-style window cross attention between s1_1 and s1_2 (stage1 tokens)
# - Align view2 tokens by flipping width in token space before cross-attn,
#   then flip back to preserve view2 native orientation for subsequent path.
# =============================================
from __future__ import annotations

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .model_utils import (
    flip_lr,
    flip_lr_nhwc,
    nchw_to_nhwc,
    nhwc_to_nchw,
    count_parameters,
    window_partition,
    window_reverse,
    compute_attn_mask,
)

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
# SACA-style window cross attention
# -------------------------
class WindowCrossAttention(nn.Module):
    """
    Window-based cross attention:
      Q from x_q, K/V from x_kv, both are [Bn, N, C] where N=ws*ws.
    Uses the same relative position bias scheme as WindowAttention.
    """
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

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

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        Bn, N, C = x_q.shape

        q = self.q(x_q).reshape(Bn, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x_kv).reshape(Bn, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        out = self.proj_drop(self.proj(out))
        return out


class SACA(nn.Module):
    """
    SACA-style symmetric cross attention between view1 and view2 stage1 tokens.
    Input/Output: NHWC tensors [B,H,W,C] for each view.

    Steps:
      - Align view2 tokens into view1 coordinate system by flipping width in token space
      - Window cross attention: v1 attends to v2_aligned, and v2_aligned attends to v1
      - Residual with learnable gate
      - Flip back to keep view2 native orientation for downstream branches
    """
    def __init__(self, dim: int, window_size: int, num_heads: int, mlp_ratio: float = 4.0, gate_init: float = 0.0):
        super().__init__()
        self.window_size = window_size

        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_kv1 = nn.LayerNorm(dim)
        self.xattn_12 = WindowCrossAttention(dim=dim, window_size=window_size, num_heads=num_heads)

        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_kv2 = nn.LayerNorm(dim)
        self.xattn_21 = WindowCrossAttention(dim=dim, window_size=window_size, num_heads=num_heads)

        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        self.norm2_1 = nn.LayerNorm(dim)
        self.mlp1 = Mlp(dim, mlp_ratio=mlp_ratio)

        self.norm2_2 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(dim, mlp_ratio=mlp_ratio)

    @staticmethod
    def _pad_to_window(x: torch.Tensor, ws: int) -> Tuple[torch.Tensor, int, int]:
        B, H, W, C = x.shape
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        return x, pad_b, pad_r

    def forward(self, s1_1: torch.Tensor, s1_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Align view2 tokens to view1 coordinates (token-space flip)
        s1_2a = flip_lr_nhwc(s1_2)

        ws = self.window_size

        # Pad both to window
        x1p, pad_b1, pad_r1 = self._pad_to_window(s1_1, ws)
        x2p, pad_b2, pad_r2 = self._pad_to_window(s1_2a, ws)

        # Require same padded spatial size for window pairing
        Hp = max(x1p.shape[1], x2p.shape[1])
        Wp = max(x1p.shape[2], x2p.shape[2])

        if x1p.shape[1] != Hp or x1p.shape[2] != Wp:
            x1p = F.pad(x1p, (0, 0, 0, Wp - x1p.shape[2], 0, Hp - x1p.shape[1]))
        if x2p.shape[1] != Hp or x2p.shape[2] != Wp:
            x2p = F.pad(x2p, (0, 0, 0, Wp - x2p.shape[2], 0, Hp - x2p.shape[1]))

        B, H1, W1, C = x1p.shape

        # Window partition
        w1 = window_partition(self.norm_q1(x1p), ws).view(-1, ws * ws, C)
        w2 = window_partition(self.norm_kv1(x2p), ws).view(-1, ws * ws, C)
        delta1 = self.xattn_12(w1, w2).view(-1, ws, ws, C)
        delta1 = window_reverse(delta1, ws, H1, W1, B)

        w2q = window_partition(self.norm_q2(x2p), ws).view(-1, ws * ws, C)
        w1kv = window_partition(self.norm_kv2(x1p), ws).view(-1, ws * ws, C)
        delta2 = self.xattn_21(w2q, w1kv).view(-1, ws, ws, C)
        delta2 = window_reverse(delta2, ws, H1, W1, B)

        # Residual with gate
        x1 = x1p + self.gate * delta1
        x2 = x2p + self.gate * delta2

        # Per-view MLP refinement (keeps same style as Swin blocks)
        x1 = x1 + self.mlp1(self.norm2_1(x1))
        x2 = x2 + self.mlp2(self.norm2_2(x2))

        # Crop back to original s1 sizes (before padding)
        H0, W0 = s1_1.shape[1], s1_1.shape[2]
        x1 = x1[:, :H0, :W0, :].contiguous()

        H2, W2 = s1_2a.shape[1], s1_2a.shape[2]
        x2 = x2[:, :H2, :W2, :].contiguous()

        # Flip back to keep view2 native orientation
        s1_2_out = flip_lr_nhwc(x2)
        return x1, s1_2_out


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
    def __init__(self, in_dim: int, proj_dim: int, *, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.LayerNorm(in_dim), # nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x



# -------------------------
# Main model
# -------------------------
class SwinUNetDualViewSSL(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        image_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 96,
        enc_depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        dec_depths: Tuple[int, int, int] = (6, 2, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: int = 7,
        proj_dim: int = 128,
        plane_inject_method: str = "film",
        enable_saca: bool = True,
        saca_position: str = "after_stage1",
        saca_gate_init: float = 0.0,
        saca_warmup_epochs: int = 0,
        enable_reconstruct: bool = True,
        enable_contrastive: bool = True,
        contrastive_loss_type: str = "infonce"
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.enc_depths = enc_depths
        self.dec_depths = dec_depths
        self.num_heads = num_heads
        self.window_size = window_size

        # ---- SACA config ----
        self.enable_saca = enable_saca
        self.saca_position = saca_position
        self.saca_warmup_epochs = saca_warmup_epochs
        self.current_epoch = 0  # safe default
        
        # ---- Training mode ----
        self.enable_reconstruct = enable_reconstruct
        self.enable_contrastive = enable_contrastive

        C0 = embed_dim
        C1 = 2 * C0
        C2 = 2 * C1
        C3 = 2 * C2

        # ---- Validate SACA config early (fail-fast) ----
        self._validate_saca_config(C0, C1, num_heads)

        # ---- Early encoders (unchanged) ----
        self.patch_embed_1 = PatchEmbed(in_ch=in_ch, embed_dim=C0, patch_size=patch_size)
        self.stage0_1 = BasicLayer(dim=C0, depth=enc_depths[0], num_heads=num_heads[0], window_size=window_size)
        self.merge0_1 = PatchMerging(dim=C0)
        self.stage1_1 = BasicLayer(dim=C1, depth=enc_depths[1], num_heads=num_heads[1], window_size=window_size)

        self.patch_embed_2 = PatchEmbed(in_ch=in_ch, embed_dim=C0, patch_size=patch_size)
        self.stage0_2 = BasicLayer(dim=C0, depth=enc_depths[0], num_heads=num_heads[0], window_size=window_size)
        self.merge0_2 = PatchMerging(dim=C0)
        self.stage1_2 = BasicLayer(dim=C1, depth=enc_depths[1], num_heads=num_heads[1], window_size=window_size)

        # ---- SACA instances ----
        if self.enable_saca:
            self.saca_c0 = SACA(
                dim=C0,
                window_size=window_size,
                num_heads=num_heads[0],
                mlp_ratio=4.0,
                gate_init=saca_gate_init,
            )

            self.saca_c1 = SACA(
                dim=C1,
                window_size=window_size,
                num_heads=num_heads[1],
                mlp_ratio=4.0,
                gate_init=saca_gate_init,
            )
            
        else:
            self.saca_c0 = None 
            self.saca_c1 = None 

        # ---- Shared trunk  ----
        self.merge1 = PatchMerging(dim=C1)
        self.plane_cond = PlaneCondition(in_dim=2, feat_dim=C2, method=plane_inject_method)
        self.stage2 = BasicLayer(dim=C2, depth=enc_depths[2], num_heads=num_heads[2], window_size=window_size)
        self.merge2 = PatchMerging(dim=C2)
        self.stage3 = BasicLayer(dim=C3, depth=enc_depths[3], num_heads=num_heads[3], window_size=window_size)

        # ---- Contrastive Head ----
        if self.enable_contrastive:
            proj_normalize = (contrastive_loss_type.lower().strip() == "infonce")
            self.proj = ProjectionHead(in_dim=C3, proj_dim=proj_dim, normalize=proj_normalize)
        else:
            self.proj = None

        # ---- Decoder ----
        if self.enable_reconstruct:
            self.up2_shared = SwinUpBlock(
                in_dim=C3,
                skip_dim=C2,
                out_dim=C2,
                depth=dec_depths[0],
                num_heads=num_heads[2],
                window_size=window_size,
            )

            self.up1_v1 = SwinUpBlock(
                in_dim=C2,
                skip_dim=C1,
                out_dim=C1,
                depth=dec_depths[1],
                num_heads=num_heads[1],
                window_size=window_size,
            )
            self.up0_v1 = SwinUpBlock(
                in_dim=C1,
                skip_dim=C0,
                out_dim=C0,
                depth=dec_depths[2],
                num_heads=num_heads[0],
                window_size=window_size,
            )
            self.final_up_v1 = FinalPatchExpand(dim=C0, patch_size=patch_size, out_dim=32)
            self.recon_head_v1 = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
            )

            self.up1_v2 = SwinUpBlock(
                in_dim=C2,
                skip_dim=C1,
                out_dim=C1,
                depth=dec_depths[1],
                num_heads=num_heads[1],
                window_size=window_size,
            )
            self.up0_v2 = SwinUpBlock(
                in_dim=C1,
                skip_dim=C0,
                out_dim=C0,
                depth=dec_depths[2],
                num_heads=num_heads[0],
                window_size=window_size,
            )
            self.final_up_v2 = FinalPatchExpand(dim=C0, patch_size=patch_size, out_dim=32)
            self.recon_head_v2 = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
            )
        else:
            self.up2_shared = None
            self.up1_v1 = None
            self.up0_v1 = None
            self.final_up_v1 = None
            self.recon_head_v1 = None

            self.up1_v2 = None
            self.up0_v2 = None
            self.final_up_v2 = None
            self.recon_head_v2 = None

        if self.enable_contrastive and (self.proj is None):
            raise RuntimeError("enable_contrastive=True but projection head is not initialized")

        if self.enable_reconstruct and (self.up2_shared is None):
            raise RuntimeError("enable_reconstruct=True but decoder is not initialized")
        
        print(self.get_saca_debug_info())

    def get_saca_debug_info(self) -> Dict[str, float]:
        """
        Lightweight debug info for logging.
        Safe to call every epoch.
        """
        
        print("="*100)
        info = {
            "saca_enable": bool(self.enable_saca),
            "saca_position": self.saca_position if self.enable_saca else "disabled",
            "saca_warmup_epochs": float(self.saca_warmup_epochs),
            "current_epoch": float(self.current_epoch),
        }

        if self.enable_saca:
            info["saca_gate_c0"] = float(self.saca_c0.gate.detach().cpu())
            info["saca_gate_c1"] = float(self.saca_c1.gate.detach().cpu())
        return info

    
    def _validate_saca_config(self, C0: int, C1: int, num_heads):
        if not self.enable_saca:
            return

        valid_positions = {"after_patch_embed", "after_merge0", "after_stage1"}
        if self.saca_position not in valid_positions:
            raise ValueError(
                f"saca_position must be one of {valid_positions}, got {self.saca_position}"
            )

        if C0 % num_heads[0] != 0:
            raise ValueError(f"C0 ({C0}) must be divisible by num_heads[0] ({num_heads[0]})")

        if C1 % num_heads[1] != 0:
            raise ValueError(f"C1 ({C1}) must be divisible by num_heads[1] ({num_heads[1]})")

    def maybe_saca(
        self,
        point: str,
        f1: torch.Tensor,
        f2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 1: helper only, not wired into forward yet.
        """
        if not self.enable_saca:
            return f1, f2

        if point != self.saca_position:
            return f1, f2

        if self.current_epoch < self.saca_warmup_epochs:
            return f1, f2

        if point == "after_patch_embed":
            return self.saca_c0(f1, f2)

        # after_merge0 or after_stage1
        return self.saca_c1(f1, f2)

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

        if self.enable_saca:
            # For encode_bottleneck we do not have the other view, so we skip SACA.
            pass

        _, b = self._shared_trunk(s1, plane_one_hot)
        return b

    def param_count_breakdown(self) -> Dict[str, int]:
        early_view1 = [self.patch_embed_1, self.stage0_1, self.merge0_1, self.stage1_1]
        early_view2 = [self.patch_embed_2, self.stage0_2, self.merge0_2, self.stage1_2]
        shared_trunk = [self.merge1, self.plane_cond, self.stage2, self.merge2, self.stage3]
        contrastive_head = [self.proj]
        decoder_shared = [self.up2_shared]
        decoder_branch_v1 = [self.up1_v1, self.up0_v1, self.final_up_v1]
        decoder_branch_v2 = [self.up1_v2, self.up0_v2, self.final_up_v2]
        recon_heads = [self.recon_head_v1, self.recon_head_v2]
        saca = [self.saca_c0, self.saca_c1]

        def _count(mods) -> int:
            mods = [m for m in mods if m is not None]
            return sum(count_parameters(m) for m in mods)

        return {
            "total": count_parameters(self),
            "enc_early_view1": _count(early_view1),
            "enc_early_view2": _count(early_view2),
            "saca": _count(saca),
            "enc_shared_trunk": _count(shared_trunk),
            "contrastive_head": _count(contrastive_head),
            "decoder_shared_up2": _count(decoder_shared),
            "decoder_branch_v1": _count(decoder_branch_v1),
            "decoder_branch_v2": _count(decoder_branch_v2),
            "recon_heads": _count(recon_heads),
        }
        
    def forward(
        self,
        x: torch.Tensor,
        pixel_mask: torch.Tensor,
        plane_one_hot: torch.Tensor,
    ):
        x1_masked = x * (1.0 - pixel_mask)
        # NOTE:
        # View2 uses a flipped image but the SAME pixel mask (mask is NOT flipped).
        # This is intentional to create mask diversity between views,
        # encouraging invariance beyond exact masked-region alignment.
        x2_masked = flip_lr(x) * (1.0 - pixel_mask)


        # ---- patch embed ----
        f0_1 = self.patch_embed_1(x1_masked)
        f0_2 = self.patch_embed_2(x2_masked)

        # SACA: after_patch_embed
        f0_1, f0_2 = self.maybe_saca("after_patch_embed", f0_1, f0_2)

        # ---- stage0 ----
        s0_1 = self.stage0_1(f0_1)
        s0_2 = self.stage0_2(f0_2)

        # ---- merge0 ----
        f1_1 = self.merge0_1(s0_1)
        f1_2 = self.merge0_2(s0_2)

        # SACA: after_merge0
        f1_1, f1_2 = self.maybe_saca("after_merge0", f1_1, f1_2)

        # ---- stage1 ----
        s1_1 = self.stage1_1(f1_1)
        s1_2 = self.stage1_2(f1_2)

        # SACA: after_stage1
        s1_1, s1_2 = self.maybe_saca("after_stage1", s1_1, s1_2)

        # ---- shared trunk ----
        s2_1, b1 = self._shared_trunk(s1_1, plane_one_hot)
        s2_2, b2 = self._shared_trunk(s1_2, plane_one_hot)

        # ---- contrastive head ----
        if self.enable_contrastive:
            z1 = self.proj(b1.mean(dim=(1, 2)))
            z2 = self.proj(b2.mean(dim=(1, 2)))
        else:
            z1, z2 = None, None
            
        if not self.enable_reconstruct:
            recon_raw_orig = None
            recon_raw_flip = None
            return recon_raw_orig, recon_raw_flip, z1, z2

        # ---- shared up2 ----
        d2_1 = self.up2_shared(b1, s2_1)
        d2_2 = self.up2_shared(b2, s2_2)

        # ---- decoder view1 ----
        d1_1 = self.up1_v1(d2_1, s1_1)
        d0_1 = self.up0_v1(d1_1, s0_1)
        feat1 = self.final_up_v1(d0_1)
        recon_raw_orig = self.recon_head_v1(nhwc_to_nchw(feat1))

        # ---- decoder view2 ----
        d1_2 = self.up1_v2(d2_2, s1_2)
        d0_2 = self.up0_v2(d1_2, s0_2)
        feat2 = self.final_up_v2(d0_2)
        recon_raw_flip = self.recon_head_v2(nhwc_to_nchw(feat2))

        return recon_raw_orig, recon_raw_flip, z1, z2

__all__ = [
    "SwinUNetDualViewSSL",
    "flip_lr",
]
