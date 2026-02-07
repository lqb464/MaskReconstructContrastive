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
        self.mask_num = None
        self.mask_den = None
        self.unmask_num = None
        self.unmask_den = None
        self.total_num = None
        self.total_den = 0
        
        # SSIM
        self.ssim_sum = None
        self.img_count = 0

    def _init_accumulators(self, device: torch.device, dtype: torch.dtype):
        def _z():
            return torch.zeros((), device=device, dtype=dtype)
        self.mask_num = _z()
        self.mask_den = _z()
        self.unmask_num = _z()
        self.unmask_den = _z()
        self.total_num = _z()
        self.ssim_sum = _z()
    
    def update(
        self, 
        diff: torch.Tensor, 
        mask: torch.Tensor,
        ssim_sum: Optional[float | torch.Tensor] = None
    ):
        """
        Update accumulators with batch results
        
        Args:
            diff: Absolute difference tensor (B, C, H, W)
            mask: Binary mask (B, 1, H, W), 1 = masked
            ssim_sum: Optional SSIM sum over batch
        """
        if self.mask_num is None:
            self._init_accumulators(diff.device, diff.dtype)
        m = mask
        um = 1.0 - m
        
        self.mask_num += (diff * m).sum()
        self.mask_den += m.sum()
        self.unmask_num += (diff * um).sum()
        self.unmask_den += um.sum()
        self.total_num += diff.sum()
        self.total_den += diff.numel()
        
        if ssim_sum is not None:
            if torch.is_tensor(ssim_sum):
                ssim_val = ssim_sum.detach()
                if ssim_val.device != self.ssim_sum.device or ssim_val.dtype != self.ssim_sum.dtype:
                    ssim_val = ssim_val.to(device=self.ssim_sum.device, dtype=self.ssim_sum.dtype)
            else:
                ssim_val = torch.as_tensor(ssim_sum, device=self.ssim_sum.device, dtype=self.ssim_sum.dtype)
            self.ssim_sum += ssim_val
            self.img_count += int(diff.size(0))
    
    def compute(self) -> ReconstructionMetrics:
        """Compute final metrics from accumulated values"""
        if self.mask_num is None:
            return ReconstructionMetrics()
        one = self.mask_den.new_tensor(1.0)
        total_den = max(self.total_den, 1)
        img_count = max(self.img_count, 1)
        return ReconstructionMetrics(
            masked_l1=(self.mask_num / torch.maximum(self.mask_den, one)).item(),
            unmasked_l1=(self.unmask_num / torch.maximum(self.unmask_den, one)).item(),
            total_l1=(self.total_num / float(total_den)).item(),
            ssim=(self.ssim_sum / float(img_count)).item(),
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
