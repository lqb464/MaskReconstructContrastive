from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

class EpochCSVLogger:
    """Write epoch-level summary CSV with the same schema as legacy trainer."""

    def __init__(self, path: Path):
        self.path = path
        self._init_file()

    def _init_file(self) -> None:
        if self.path.exists():
            return
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch",
                "train_loss",
                "train_recon_total",
                "train_recon_masked",
                "train_recon_unmasked",
                "train_ssim",
                "train_loss_contrast",
                "train_embed_var_mean",
                "train_embed_var_min",
                "val_loss",
                "val_recon_total",
                "val_recon_masked",
                "val_recon_unmasked",
                "val_ssim",
            ])

    def append(self, row: Dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                row["epoch"],
                row["train_loss"],
                row["train_recon_total"],
                row["train_recon_masked"],
                row["train_recon_unmasked"],
                row["train_ssim"],
                row["train_loss_contrast"],
                row["train_embed_var_mean"],
                row["train_embed_var_min"],
                row["val_loss"],
                row["val_recon_total"],
                row["val_recon_masked"],
                row["val_recon_unmasked"],
                row["val_ssim"],
            ])

class LossDecompCSVLogger:
    """Write explicit loss decomposition CSV (train/val)."""

    def __init__(self, path: Path):
        self.path = path
        self._init_file()

    def _init_file(self) -> None:
        if self.path.exists():
            return
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch",
                "split",
                "loss_recon_orig",
                "loss_recon_flip",
                "loss_recon_total",
                "loss_contrastive",
                "loss_total",
            ])

    def append(self, epoch: int, split: str, d: Dict[str, float]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                split,
                d.get("loss_recon_orig", 0.0),
                d.get("loss_recon_flip", 0.0),
                d.get("loss_recon_total", 0.0),
                d.get("loss_contrastive", 0.0),
                d.get("loss_total", 0.0),
            ])
