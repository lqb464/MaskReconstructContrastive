# =============================================
# File: losses.py
# All loss functions and metrics
# =============================================
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Tuple


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None):
    """Generate 2D Gaussian window for SSIM computation"""
    half = window_size // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    g = (gauss / gauss.sum()).unsqueeze(0)
    kernel2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return kernel2d


def ssim_index(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two images
    
    Args:
        x: First image tensor (B, 1, H, W)
        y: Second image tensor (B, 1, H, W)
        window_size: Size of Gaussian window
        sigma: Standard deviation of Gaussian
        
    Returns:
        SSIM values per image (B,)
    """
    assert x.shape == y.shape and x.dim() == 4 and x.size(1) == 1
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    kernel = _gaussian_window(window_size, sigma, device=x.device, dtype=x.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=1)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=1) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=1) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=1) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den.clamp_min(1e-8)
    return ssim_map.mean(dim=(1, 2, 3))


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
    """
    L1 loss computed only on masked regions
    
    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)
        pixel_mask: Binary mask (B, 1, H, W), 1 = masked
        
    Returns:
        Scalar loss
    """
    diff = torch.abs(pred - target) * pixel_mask
    denom = pixel_mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def mixed_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
    alpha_mask: float = 1.0,
    beta_unmask: float = 0.2
) -> torch.Tensor:
    """
    Weighted L1 loss on both masked and unmasked regions
    
    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)
        pixel_mask: Binary mask (B, 1, H, W), 1 = masked
        alpha_mask: Weight for masked region
        beta_unmask: Weight for unmasked region
        
    Returns:
        Scalar loss
    """
    diff = torch.abs(pred - target)
    m = pixel_mask
    um = 1.0 - m

    # L1 on masked region
    masked_denom = m.sum().clamp(min=1.0)
    masked_l1 = (diff * m).sum() / masked_denom

    # L1 on unmasked region
    unmasked_denom = um.sum().clamp(min=1.0)
    unmasked_l1 = (diff * um).sum() / unmasked_denom

    # Weighted combination
    return alpha_mask * masked_l1 + beta_unmask * unmasked_l1


def _foreground_weighted_bce_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    fg_eps: float = 0.02,
    fg_weight: float = 10.0,
) -> torch.Tensor:
    """Weighted BCEWithLogits where pixels with target > fg_eps get larger weight.

    Target is expected in [0, 1]. Returns unreduced loss map.
    """
    with torch.no_grad():
        w = torch.ones_like(target)
        w = torch.where(target > fg_eps, torch.full_like(w, fg_weight), w)
    return F.binary_cross_entropy_with_logits(logits, target, weight=w, reduction="none")


def masked_bce_logits_weighted(
    logits: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
    fg_eps: float = 0.02,
    fg_weight: float = 10.0,
) -> torch.Tensor:
    """BCE logits computed only on masked region (pixel_mask==1)."""
    loss_map = _foreground_weighted_bce_logits(logits, target, fg_eps=fg_eps, fg_weight=fg_weight)
    m = pixel_mask
    denom = m.sum().clamp(min=1.0)
    return (loss_map * m).sum() / denom


def mixed_bce_logits_weighted(
    logits: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
    fg_eps: float = 0.02,
    fg_weight: float = 10.0,
    alpha_mask: float = 1.0,
    beta_unmask: float = 0.2,
) -> torch.Tensor:
    """Weighted BCE logits computed on both masked and unmasked, with different weights."""
    loss_map = _foreground_weighted_bce_logits(logits, target, fg_eps=fg_eps, fg_weight=fg_weight)
    m = pixel_mask
    um = 1.0 - m
    masked = (loss_map * m).sum() / m.sum().clamp(min=1.0)
    unmasked = (loss_map * um).sum() / um.sum().clamp(min=1.0)
    return alpha_mask * masked + beta_unmask * unmasked


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning
    Used in SimCLR

    Args:
        z1: First set of embeddings (B, D)
        z2: Second set of embeddings (B, D)
        temperature: Temperature parameter for softmax

    Returns:
        Scalar loss
    """
    # L2-normalize embeddings to make similarities scale-stable.
    # This matches the intended "Normalized" part of NT-Xent.
    z1 = F.normalize(z1, dim=1, eps=1e-8)
    z2 = F.normalize(z2, dim=1, eps=1e-8)

    B, _ = z1.size()
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.t()) / temperature  # (2B, 2B)
    sim = sim.to(torch.float32)

    # Mask out self-similarity
    diag = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -float("inf"))

    # Positive pairs: (i, B+i) and (B+i, i)
    pos = torch.cat(
        [
            torch.arange(B, 2 * B, device=sim.device),
            torch.arange(0, B, device=sim.device),
        ],
        dim=0,
    )
    labels = pos

    loss = F.cross_entropy(sim, labels)
    return loss


def compute_embedding_variance(z_list: List[torch.Tensor]) -> Tuple[float, float]:
    """
    Compute variance statistics of embeddings
    Used to monitor embedding collapse
    
    Args:
        z_list: List of embedding tensors
        
    Returns:
        (mean_variance, min_variance)
    """
    if len(z_list) == 0:
        return 0.0, 0.0
    Z = torch.cat(z_list, dim=0)
    var = Z.var(dim=0, unbiased=False)
    return var.mean().item(), var.min().item()


def vicreg_loss(
    *,
    z1: torch.Tensor,
    z2: torch.Tensor,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    variance_eps: float = 1e-4,
    target_std: float = 1.0,
) -> torch.Tensor:
    """
    VICReg loss (invariance + variance + covariance).

    Args:
        z1, z2: embeddings (B, D)
        invariance_weight: weight for MSE(z1, z2)
        variance_weight: weight for variance regularization
        covariance_weight: weight for covariance regularization
        variance_eps: epsilon inside sqrt for numerical stability
        target_std: desired per-dimension std (typically 1.0)

    Returns:
        scalar loss
    """
    assert z1.dim() == 2 and z2.dim() == 2 and z1.shape == z2.shape, f"z1,z2 must be (B,D) same shape, got {z1.shape} and {z2.shape}"
    B, D = z1.shape

    # 1) Invariance term
    inv = F.mse_loss(z1, z2)

    # Center features for var and cov terms
    z1c = z1 - z1.mean(dim=0, keepdim=True)
    z2c = z2 - z2.mean(dim=0, keepdim=True)

    # 2) Variance term
    # Penalize dimensions whose std is below target_std
    std_z1 = torch.sqrt(z1c.var(dim=0, unbiased=False) + variance_eps)
    std_z2 = torch.sqrt(z2c.var(dim=0, unbiased=False) + variance_eps)
    var = (F.relu(target_std - std_z1).mean() + F.relu(target_std - std_z2).mean())

    # 3) Covariance term
    # Penalize off-diagonal covariance
    if B > 1:
        cov_z1 = (z1c.T @ z1c) / (B - 1)
        cov_z2 = (z2c.T @ z2c) / (B - 1)
    else:
        # Degenerate batch, avoid divide by zero
        cov_z1 = z1c.T @ z1c
        cov_z2 = z2c.T @ z2c

    diag = torch.eye(D, device=z1.device, dtype=torch.bool)
    cov_off_1 = cov_z1.masked_fill(diag, 0.0)
    cov_off_2 = cov_z2.masked_fill(diag, 0.0)
    cov = (cov_off_1.pow(2).sum() / D) + (cov_off_2.pow(2).sum() / D)

    loss = (invariance_weight * inv) + (variance_weight * var) + (covariance_weight * cov)
    
    return loss


__all__ = [
    "ssim_index",
    "masked_l1_loss",
    "mixed_l1_loss",
    "nt_xent_loss",
    "compute_embedding_variance",
    "masked_bce_logits_weighted",
    "mixed_bce_logits_weighted",
    "vicreg_loss",
]