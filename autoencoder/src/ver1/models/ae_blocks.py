from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm2d(num_channels: int, use_gn: bool = False, num_groups: int = 8) -> nn.Module:
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels) if use_gn else nn.BatchNorm2d(num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, use_gn: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n1 = norm2d(out_ch, use_gn)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = norm2d(out_ch, use_gn)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.act(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return self.act(out + identity)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, use_gn: bool = False):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, use_gn=use_gn)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return feat, self.pool(feat)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, use_gn: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch, out_ch, use_gn=use_gn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class ConvEncoder(nn.Module):
    """4-stage convolutional encoder without skip connections."""

    def __init__(self, in_ch: int, base_ch: int, *, use_gn: bool = False):
        super().__init__()
        self.down1 = DownBlock(in_ch, base_ch, use_gn=use_gn)
        self.down2 = DownBlock(base_ch, base_ch * 2, use_gn=use_gn)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, use_gn=use_gn)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8, use_gn=use_gn)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8, use_gn=use_gn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, x = self.down1(x)
        _, x = self.down2(x)
        _, x = self.down3(x)
        _, x = self.down4(x)
        return self.bottleneck(x)


class ConvDecoder(nn.Module):
    """4-stage convolutional decoder (pure AE, no skip connections)."""

    def __init__(self, base_ch: int, out_ch: int, *, use_gn: bool = False):
        super().__init__()
        self.up1 = UpBlock(base_ch * 8, base_ch * 4, use_gn=use_gn)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, use_gn=use_gn)
        self.up3 = UpBlock(base_ch * 2, base_ch, use_gn=use_gn)
        self.up4 = UpBlock(base_ch, base_ch, use_gn=use_gn)
        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, z: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        if x.shape[-2:] != target_size:
            dh = target_size[0] - x.shape[-2]
            dw = target_size[1] - x.shape[-1]
            x = F.pad(x, (0, max(dw, 0), 0, max(dh, 0)))
            x = x[..., : target_size[0], : target_size[1]]
        return self.out_conv(x)


def apply_pixel_mask(x: torch.Tensor, pixel_mask: torch.Tensor | None) -> torch.Tensor:
    if pixel_mask is None:
        return x
    if pixel_mask.ndim != 4:
        raise ValueError(f"pixel_mask must be 4D, got shape {tuple(pixel_mask.shape)}")
    if pixel_mask.shape[0] != x.shape[0] or pixel_mask.shape[-2:] != x.shape[-2:]:
        raise ValueError(
            f"pixel_mask shape {tuple(pixel_mask.shape)} is incompatible with input shape {tuple(x.shape)}"
        )
    if pixel_mask.shape[1] not in {1, x.shape[1]}:
        raise ValueError(
            f"pixel_mask channel dimension must be 1 or match input channels ({x.shape[1]}), "
            f"got {pixel_mask.shape[1]}"
        )
    return x * (1.0 - pixel_mask.to(dtype=x.dtype))


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert NCHW image to patch tokens N L C where L = (H/ps)*(W/ps)."""
    b, c, h, w = x.shape
    if (h % patch_size) != 0 or (w % patch_size) != 0:
        raise ValueError(f"image size ({h}, {w}) must be divisible by patch_size={patch_size}")
    gh, gw = h // patch_size, w // patch_size
    x = x.reshape(b, c, gh, patch_size, gw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(b, gh * gw, c * patch_size * patch_size)


def unpatchify(tokens: torch.Tensor, patch_size: int, h: int, w: int) -> torch.Tensor:
    """Convert patch tokens back to NCHW image."""
    b, _, patch_dim = tokens.shape
    gh, gw = h // patch_size, w // patch_size
    c = patch_dim // (patch_size * patch_size)
    x = tokens.reshape(b, gh, gw, c, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(b, c, h, w)


def downsample_mask_to_patches(pixel_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Downsample pixel mask [B,1,H,W] to patch mask [B,L] (1 = masked)."""
    b, _, h, w = pixel_mask.shape
    gh, gw = h // patch_size, w // patch_size
    m = pixel_mask.reshape(b, 1, gh, patch_size, gw, patch_size)
    m = m.amax(dim=(3, 5)).reshape(b, gh * gw)
    return (m > 0.5).to(dtype=pixel_mask.dtype)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard VAE KL divergence against N(0,1), averaged over batch."""
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


__all__ = [
    "ConvBlock",
    "DownBlock",
    "UpBlock",
    "ConvEncoder",
    "ConvDecoder",
    "apply_pixel_mask",
    "patchify",
    "unpatchify",
    "downsample_mask_to_patches",
    "kl_divergence",
    "reparameterize",
]
