# =============================================
# File: preprocessing.py
# All image preprocessing functions
# =============================================
from __future__ import annotations

import torch
import torch.nn.functional as F


def otsu_threshold(x: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Compute Otsu threshold for each image in batch"""
    B = x.size(0)
    thresholds = []
    for b in range(B):
        hist = torch.histc(x[b].flatten(), bins=bins, min=0.0, max=1.0)
        p = hist / hist.sum().clamp(min=1.0)
        omega = torch.cumsum(p, 0)
        mu = torch.cumsum(p * torch.arange(bins, device=x.device), 0)
        mu_t = mu[-1]
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega)).clamp(min=1e-8)
        sigma_b2[torch.isnan(sigma_b2)] = -1
        t = torch.argmax(sigma_b2).item()
        thresholds.append((t + 0.5) / bins)
    return torch.tensor(thresholds, device=x.device, dtype=x.dtype).view(B, 1, 1, 1)


def brain_mask(x: torch.Tensor) -> torch.Tensor:
    """Generate brain mask using Otsu thresholding and morphological operations"""
    thr = otsu_threshold(x)
    m = (x > thr).float()
    m_blur = F.avg_pool2d(m, kernel_size=7, stride=1, padding=3)
    m = (m_blur > 0.2).float()
    return m


def bias_field_lite(x: torch.Tensor, kernel: int = 31) -> torch.Tensor:
    """Simple bias field correction using division by blurred image"""
    blur = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    blur = blur.clamp(min=1e-3)
    x_corr = x / blur
    x_corr = x_corr - x_corr.amin(dim=(2,3), keepdim=True)
    x_corr = x_corr / x_corr.amax(dim=(2,3), keepdim=True).clamp(min=1e-6)
    return x_corr


def tight_crop_and_resize(x: torch.Tensor, mask: torch.Tensor, out_hw: int) -> torch.Tensor:
    """Crop to brain bounding box and resize to target size"""
    B, _, H, W = x.shape
    out = []
    for b in range(B):
        ys, xs = torch.where(mask[b, 0] > 0.0)
        if ys.numel() == 0:
            out.append(F.interpolate(x[b:b+1], size=(out_hw, out_hw), mode="bilinear", align_corners=False))
            continue
        y1, y2 = ys.min().item(), ys.max().item()
        x1, x2 = xs.min().item(), xs.max().item()
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        side = max(h, w)
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2
        y1s = max(0, cy - side // 2)
        x1s = max(0, cx - side // 2)
        y2s = min(H, y1s + side)
        x2s = min(W, x1s + side)
        crop = x[b:b+1, :, y1s:y2s, x1s:x2s]
        out.append(F.interpolate(crop, size=(out_hw, out_hw), mode="bilinear", align_corners=False))
    return torch.cat(out, dim=0)


def align_midline(x: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
    """Align brain hemispheres by maximizing left-right symmetry"""
    B, C, H, W = x.shape
    best = []
    for b in range(B):
        xb = x[b:b+1]
        best_score = -1e9
        best_img = xb
        for d in range(-max_shift, max_shift + 1):
            if d < 0:
                pad = (0, -d, 0, 0)
                xs = F.pad(xb, pad, mode="replicate")[..., :W]
            elif d > 0:
                pad = (d, 0, 0, 0)
                xs = F.pad(xb, pad, mode="replicate")[..., -W:]
            else:
                xs = xb
            left = xs[..., :W//2]
            right = torch.flip(xs[..., W//2:], dims=[-1])
            score = (left * right).mean()
            if score > best_score:
                best_score = score
                best_img = xs
        best.append(best_img)
    return torch.cat(best, dim=0)


def intensity_normalize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalize intensity using quantile clipping within brain mask"""
    B = x.size(0)
    flat = x.view(B, -1)
    flat_m = mask.view(B, -1)
    out = []
    for b in range(B):
        vals = flat[b][flat_m[b] > 0]
        if vals.numel() > 0:
            lo = torch.quantile(vals, 0.01)
            hi = torch.quantile(vals, 0.99)
            xb = x[b:b+1].clamp(min=lo.item(), max=hi.item())
            xb = (xb - xb.mean()) / (xb.std().clamp(min=1e-6))
            xb = (xb - xb.amin()) / (xb.amax().clamp(min=1e-6))
        else:
            xb = x[b:b+1]
        out.append(xb)
    return torch.cat(out, dim=0)


def preprocess_batch(x: torch.Tensor, config) -> torch.Tensor:
    """Apply full preprocessing pipeline based on config"""
    # Bias field correction
    if getattr(config, "pre_bias", False):
        x = bias_field_lite(x, kernel=31)
    
    # Generate mask if needed for normalization or cropping
    if getattr(config, "pre_norm", False) or getattr(config, "pre_crop", False):
        m = brain_mask(x)
    
    # Intensity normalization
    if getattr(config, "pre_norm", False):
        x = intensity_normalize(x, m)
    
    # Tight cropping to brain
    if getattr(config, "pre_crop", False):
        image_size = getattr(config, "image_size", 192)
        x = tight_crop_and_resize(x, m, out_hw=image_size)
    
    # Midline alignment
    if getattr(config, "pre_align", False):
        x = align_midline(x, max_shift=4)
    
    return x.clamp(0.0, 1.0)


__all__ = [
    "otsu_threshold",
    "brain_mask",
    "bias_field_lite",
    "tight_crop_and_resize",
    "align_midline",
    "intensity_normalize",
    "preprocess_batch",
]