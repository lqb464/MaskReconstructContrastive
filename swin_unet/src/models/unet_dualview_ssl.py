from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import count_parameters, flip_lr
from .swin_unet_dualview_ssl import SACA

def _norm2d(num_channels: int, use_gn: bool = False, num_groups: int = 8) -> nn.Module:
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels) if use_gn else nn.BatchNorm2d(num_channels)

class _SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=False)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_gn: bool = False, se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n1 = _norm2d(out_ch, use_gn)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = _norm2d(out_ch, use_gn)
        self.act = nn.ReLU(inplace=False)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.se = _SEBlock(out_ch) if se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv1(x)
        out = self.n1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.n2(out)
        out = self.se(out)
        out = self.act(out + identity)
        return out

class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_gn: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _ResBlock(out_ch + skip_ch, out_ch, use_gn=use_gn, se=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, skip_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            x = F.pad(x, (0, dw, 0, dh))
        if skip_mask is not None:
            skip = skip * (1.0 - skip_mask)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNetDualViewSSL(nn.Module):
    """
    UNet backbone adapter for the trainer interface.
    Reconstruction-only path with dual-view outputs:
    - view1: original input
    - view2: left-right flipped input
    """

    def __init__(
        self,
        *,
        in_ch: int = 1,
        base_ch: int = 16,
        out_ch: int = 1,
        use_gn: bool = False,
        use_se: bool = False,
        enable_reconstruct: bool = True,
        enable_contrastive: bool = False,
        single_view: bool = False,
        enable_saca: bool = False,
        saca_position: str = "after_stage1",
        saca_positions: Optional[list[str]] = None,
        saca_gate_init: float = 0.0,
        saca_warmup_epochs: int = 0,
    ):
        super().__init__()
        self.enable_reconstruct = bool(enable_reconstruct)
        self.enable_contrastive = bool(enable_contrastive)
        self.single_view = bool(single_view)

        if self.enable_contrastive:
            raise ValueError("UNetDualViewSSL currently supports reconstruction-only (enable_contrastive=False).")

        self.enable_saca = bool(enable_saca)
        self.saca_position = str(saca_position)
        if saca_positions is not None:
            self.saca_positions = list(saca_positions)
        else:
            self.saca_positions = [p.strip() for p in saca_position.split(",") if p.strip()]
        self.saca_warmup_epochs = int(saca_warmup_epochs)
        self.current_epoch = 0

        self.enc1 = _ResBlock(in_ch, base_ch, use_gn=use_gn, se=False)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ResBlock(base_ch, base_ch * 2, use_gn=use_gn, se=False)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _ResBlock(base_ch * 2, base_ch * 4, use_gn=use_gn, se=False)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _ResBlock(base_ch * 4, base_ch * 8, use_gn=use_gn, se=use_se)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = _ResBlock(base_ch * 8, base_ch * 8, use_gn=use_gn, se=use_se)

        self.saca_modules = nn.ModuleDict()
        if self.enable_saca and self.saca_positions:
            for pos in self.saca_positions:
                if pos == "after_patch_embed":
                    dim = base_ch
                elif pos == "after_stage0":
                    dim = base_ch * 2
                elif pos == "after_merge0":
                    dim = base_ch * 4
                elif pos == "after_stage1":
                    dim = base_ch * 8
                else:
                    raise ValueError(f"Unknown SACA position for UNet: {pos}")

                heads = 4
                while dim % heads != 0 and heads > 1:
                    heads //= 2

                self.saca_modules[pos] = SACA(
                    dim=dim,
                    window_size=8,
                    num_heads=heads,
                    mlp_ratio=4.0,
                    gate_init=saca_gate_init,
                )

        self.up1 = _UpBlock(base_ch * 8, base_ch * 8, base_ch * 4, use_gn=use_gn)
        self.up2 = _UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, use_gn=use_gn)
        self.up3 = _UpBlock(base_ch * 2, base_ch * 2, base_ch, use_gn=use_gn)
        self.up4 = _UpBlock(base_ch, base_ch, base_ch, use_gn=use_gn)
        if int(out_ch) < 1:
            raise ValueError(f"out_ch must be >=1, got {out_ch}")
        self.out_conv = nn.Conv2d(base_ch, int(out_ch), kernel_size=1)

    @staticmethod
    def _apply_pixel_mask(x: torch.Tensor, pixel_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pixel_mask is None:
            return x
        if pixel_mask.ndim != 4:
            raise ValueError(f"pixel_mask must be 4D [B,1,H,W] or [B,C,H,W], got shape {tuple(pixel_mask.shape)}")
        if pixel_mask.shape[0] != x.shape[0] or pixel_mask.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"pixel_mask shape {tuple(pixel_mask.shape)} is incompatible with input shape {tuple(x.shape)}"
            )
        if pixel_mask.shape[1] not in {1, x.shape[1]}:
            raise ValueError(
                f"pixel_mask channel dimension must be 1 or match input channels ({x.shape[1]}), got {pixel_mask.shape[1]}"
            )
        return x * (1.0 - pixel_mask.to(dtype=x.dtype))

    def _encode(self, x: torch.Tensor):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        b = self.bottleneck(self.pool4(s4))
        return s1, s2, s3, s4, b

    def _decode(self, s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor, s4: torch.Tensor, b: torch.Tensor, pixel_mask: Optional[torch.Tensor]) -> torch.Tensor:
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

    def _forward_one(self, x: torch.Tensor, pixel_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        s1, s2, s3, s4, b = self._encode(x)
        logits = self._decode(s1, s2, s3, s4, b, pixel_mask=pixel_mask)
        return logits, b

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
        enc = sum(count_parameters(m) for m in [self.enc1, self.enc2, self.enc3, self.enc4, self.bottleneck])
        dec = sum(count_parameters(m) for m in [self.up1, self.up2, self.up3, self.up4, self.out_conv])
        saca_cnt = sum(count_parameters(m) for m in self.saca_modules.values())
        return {
            "total": total,
            "enc_early_view1": enc,
            "enc_early_view2": 0,
            "saca": saca_cnt,
            "enc_shared_trunk": 0,
            "contrastive_head": 0,
            "decoder_shared_up2": dec,
            "decoder_branch_v1": 0,
            "decoder_branch_v2": 0,
            "recon_heads": 0,
            "check_sum": total,
            "delta_total_minus_check": 0,
        }

    def maybe_saca(
        self,
        point: str,
        f1: torch.Tensor,
        f2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_saca:
            return f1, f2

        if point not in self.saca_modules:
            return f1, f2

        if self.current_epoch < self.saca_warmup_epochs:
            return f1, f2

        # Convert NCHW to NHWC
        f1_nhwc = f1.permute(0, 2, 3, 1).contiguous()
        f2_nhwc = f2.permute(0, 2, 3, 1).contiguous()

        saca = self.saca_modules[point]
        f1_out_nhwc, f2_out_nhwc = saca(f1_nhwc, f2_nhwc)

        # Convert back to NCHW
        f1_out = f1_out_nhwc.permute(0, 3, 1, 2).contiguous()
        f2_out = f2_out_nhwc.permute(0, 3, 1, 2).contiguous()

        return f1_out, f2_out

    def encode_dual_views(self, x1: torch.Tensor, x2: torch.Tensor):
        s1_1 = self.enc1(x1)
        s1_2 = self.enc1(x2)
        s1_1, s1_2 = self.maybe_saca("after_patch_embed", s1_1, s1_2)

        s2_1 = self.enc2(self.pool1(s1_1))
        s2_2 = self.enc2(self.pool1(s1_2))
        s2_1, s2_2 = self.maybe_saca("after_stage0", s2_1, s2_2)

        s3_1 = self.enc3(self.pool2(s2_1))
        s3_2 = self.enc3(self.pool2(s2_2))
        s3_1, s3_2 = self.maybe_saca("after_merge0", s3_1, s3_2)

        s4_1 = self.enc4(self.pool3(s3_1))
        s4_2 = self.enc4(self.pool3(s3_2))
        s4_1, s4_2 = self.maybe_saca("after_stage1", s4_1, s4_2)

        b1 = self.bottleneck(self.pool4(s4_1))
        b2 = self.bottleneck(self.pool4(s4_2))

        return (s1_1, s2_1, s3_1, s4_1, b1), (s1_2, s2_2, s3_2, s4_2, b2)

    def get_saca_debug_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {
            "saca_enable": bool(self.enable_saca),
            "saca_position": self.saca_position if self.enable_saca else "disabled",
            "saca_positions": list(self.saca_positions) if self.enable_saca else [],
            "saca_warmup_epochs": float(self.saca_warmup_epochs),
            "current_epoch": float(self.current_epoch),
        }
        if self.enable_saca:
            for pos, mod in self.saca_modules.items():
                gate = torch.tanh(mod.gate)
                info[f"saca_gate_{pos}_mean"] = float(gate.mean().cpu())
                info[f"saca_gate_{pos}_absmax"] = float(gate.abs().max().cpu())
        return info

    def get_saca_debug_string(self) -> str:
        info = self.get_saca_debug_info()
        return " | ".join(f"{k}={v}" for k, v in info.items())

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
        if self.enable_saca:
            x1 = self._apply_pixel_mask(x, pixel_mask)
            x2 = self._apply_pixel_mask(flip_lr(x), pixel_mask)
            (s1_1, s2_1, s3_1, s4_1, b1), (s1_2, s2_2, s3_2, s4_2, b2) = self.encode_dual_views(x1, x2)
            b = b1 if int(view) == 1 else b2
        else:
            x = self._apply_pixel_mask(x, pixel_mask)
            if int(view) == 2:
                x = flip_lr(x)
            _, _, _, _, b = self._encode(x)
        return b.permute(0, 2, 3, 1).contiguous()

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

        x1 = self._apply_pixel_mask(x, pixel_mask)

        if self.single_view:
            if self.enable_saca:
                raise ValueError("SACA requires dual-view. Disable SACA or use dual-view mode.")
            s1, s2, s3, s4, b = self._encode(x1)
            recon_raw_orig = self._decode(s1, s2, s3, s4, b, pixel_mask=pixel_mask)
            return recon_raw_orig, None, None, None

        x2 = self._apply_pixel_mask(flip_lr(x), pixel_mask)

        (s1_1, s2_1, s3_1, s4_1, b1), (s1_2, s2_2, s3_2, s4_2, b2) = self.encode_dual_views(x1, x2)

        recon_raw_orig = self._decode(s1_1, s2_1, s3_1, s4_1, b1, pixel_mask=pixel_mask)
        recon_raw_flip = self._decode(s1_2, s2_2, s3_2, s4_2, b2, pixel_mask=pixel_mask)

        return recon_raw_orig, recon_raw_flip, None, None

__all__ = ["UNetDualViewSSL"]
