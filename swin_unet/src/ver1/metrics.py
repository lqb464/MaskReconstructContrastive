# =============================================
# File: metrics.py
# Metric computation and tracking
# =============================================
from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReconstructionMetrics:
    """Metrics for reconstruction quality"""
    masked_l1: float = 0.0
    unmasked_l1: float = 0.0
    total_l1: float = 0.0
    ssim: float = 0.0
    
    def __str__(self) -> str:
        return (
            f"L1(M/U/T): {self.masked_l1:.4f}/{self.unmasked_l1:.4f}/{self.total_l1:.4f} | "
            f"SSIM: {self.ssim:.4f}"
        )


@dataclass
class ContrastiveMetrics:
    """Metrics for contrastive learning"""
    loss: float = 0.0
    mean_var: float = 0.0
    min_var: float = 0.0
    
    def __str__(self) -> str:
        return f"Con Loss: {self.loss:.4f} | Var(mean/min): {self.mean_var:.6f}/{self.min_var:.6f}"


class MetricsAccumulator:
    """Accumulates metrics over multiple batches"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulators"""
        # L1 components
        self.mask_num = 0.0
        self.mask_den = 0.0
        self.unmask_num = 0.0
        self.unmask_den = 0.0
        self.total_num = 0.0
        self.total_den = 0.0
        
        # SSIM
        self.ssim_sum = 0.0
        self.img_count = 0
    
    def update(
        self, 
        diff: torch.Tensor, 
        mask: torch.Tensor,
        ssim_sum: Optional[float] = None
    ):
        """
        Update accumulators with batch results
        
        Args:
            diff: Absolute difference tensor (B, C, H, W)
            mask: Binary mask (B, 1, H, W), 1 = masked
            ssim_sum: Optional SSIM sum over batch
        """
        m = mask
        um = 1.0 - m
        
        self.mask_num += (diff * m).sum().item()
        self.mask_den += m.sum().item()
        self.unmask_num += (diff * um).sum().item()
        self.unmask_den += um.sum().item()
        self.total_num += diff.sum().item()
        self.total_den += diff.numel()
        
        if ssim_sum is not None:
            self.ssim_sum += ssim_sum
            self.img_count += diff.size(0)
    
    def compute(self) -> ReconstructionMetrics:
        """Compute final metrics from accumulated values"""
        return ReconstructionMetrics(
            masked_l1=self.mask_num / max(self.mask_den, 1.0),
            unmasked_l1=self.unmask_num / max(self.unmask_den, 1.0),
            total_l1=self.total_num / max(self.total_den, 1.0),
            ssim=self.ssim_sum / max(self.img_count, 1),
        )


@dataclass
class EpochMetrics:
    """Complete metrics for one epoch"""
    epoch: int
    train: ReconstructionMetrics
    val: ReconstructionMetrics
    contrastive: Optional[ContrastiveMetrics] = None
    
    def to_csv_row(self) -> str:
        """Convert to CSV row string"""
        return (
            f"{self.epoch},"
            f"{self.train.masked_l1:.6f},{self.train.unmasked_l1:.6f},{self.train.total_l1:.6f},"
            f"{self.val.masked_l1:.6f},{self.val.unmasked_l1:.6f},{self.val.total_l1:.6f},"
            f"{self.train.ssim:.6f},{self.val.ssim:.6f}\n"
        )
    
    @staticmethod
    def csv_header() -> str:
        """Get CSV header string"""
        return (
            "epoch,"
            "train_recon_masked,train_recon_unmasked,train_recon_total,"
            "val_recon_masked,val_recon_unmasked,val_recon_total,"
            "train_ssim,val_ssim\n"
        )
    
    def __str__(self) -> str:
        s = f"Epoch {self.epoch:03d}\n"
        s += f"  Train: {self.train}\n"
        s += f"  Val:   {self.val}"
        if self.contrastive:
            s += f"\n  {self.contrastive}"
        return s


class MetricsLogger:
    """Handles metric logging to files"""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.first_write = True
    
    def log(self, metrics: EpochMetrics):
        """Append epoch metrics to CSV file"""
        mode = 'w' if self.first_write else 'a'
        with open(self.log_path, mode) as f:
            if self.first_write:
                f.write(EpochMetrics.csv_header())
                self.first_write = False
            f.write(metrics.to_csv_row())


__all__ = [
    "ReconstructionMetrics",
    "ContrastiveMetrics",
    "MetricsAccumulator",
    "EpochMetrics",
    "MetricsLogger",
]