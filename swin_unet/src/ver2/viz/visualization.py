# =============================================
# File: visualization.py
# Visualization utilities for training monitoring
# =============================================
from __future__ import annotations

import os
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.manifold import TSNE


def save_image_grid(
    tensors: List[torch.Tensor],
    titles: List[str],
    out_path: str,
    annotations: Optional[Dict[int, List[str]]] = None,
    panel_vmax: Optional[Dict[int, float]] = None,
):
    """
    Save a grid of images for visualization
    
    Args:
        tensors: List of image tensors, each (B, C, H, W) or (B, H, W)
        titles: List of titles for each column
        out_path: Output file path
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with torch.no_grad():
        panels = []
        for t in tensors:
            if t.dim() == 4 and t.size(1) == 1:
                t = t.squeeze(1)
            panels.append(t)
        
        b = min(p.size(0) for p in panels)
        fig, axes = plt.subplots(b, len(panels), figsize=(3 * len(panels), 3 * b))
        
        if b == 1:
            axes = np.expand_dims(axes, 0)
        
        for i in range(b):
            for j, p in enumerate(panels):
                ax = axes[i, j]
                img = p[i].detach().cpu().numpy()
                vmax = 1.0
                if panel_vmax is not None and j in panel_vmax:
                    vmax = float(panel_vmax[j])
                ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)

                if annotations is not None and j in annotations:
                    ann_list = annotations[j]
                    if i < len(ann_list):
                        ax.text(
                            0.02,
                            0.02,
                            str(ann_list[i]),
                            transform=ax.transAxes,
                            fontsize=8,
                            color="white",
                            bbox=dict(facecolor="black", alpha=0.5, pad=2),
                            verticalalignment="bottom",
                            horizontalalignment="left",
                        )
                if i == 0 and j < len(titles):
                    ax.set_title(titles[j], fontsize=10)
                ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_training_curves(log_csv_path: Path, output_dir: Path):
    """
    Generate training curve plots from logged metrics
    
    Args:
        log_csv_path: Path to epoch_log.csv
        output_dir: Directory to save plots
    """
    if not log_csv_path.exists():
        return
    
    df = pd.read_csv(log_csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Total reconstruction L1
    plt.figure(figsize=(6, 4))
    plt.plot(df['epoch'], df['train_recon_total'], label='train_total')
    plt.plot(df['epoch'], df['val_recon_total'], label='val_total')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss (Whole Image)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'recon_total_curves.png', dpi=150)
    plt.close()
    
    # Masked region L1
    plt.figure(figsize=(6, 4))
    plt.plot(df['epoch'], df['train_recon_masked'], label='train_masked')
    plt.plot(df['epoch'], df['val_recon_masked'], label='val_masked')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss (Masked Region)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'recon_masked_curves.png', dpi=150)
    plt.close()
    
    # Unmasked region L1
    plt.figure(figsize=(6, 4))
    plt.plot(df['epoch'], df['train_recon_unmasked'], label='train_unmasked')
    plt.plot(df['epoch'], df['val_recon_unmasked'], label='val_unmasked')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss (Unmasked Region)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'recon_unmasked_curves.png', dpi=150)
    plt.close()
    
    # SSIM curves
    plt.figure(figsize=(6, 4))
    plt.plot(df['epoch'], df['train_ssim'], label='train_ssim')
    plt.plot(df['epoch'], df['val_ssim'], label='val_ssim')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'ssim_curves.png', dpi=150)
    plt.close()


def run_tsne_visualization(
    model,
    loader,
    device: torch.device,
    out_prefix: str,
    max_items: int = 1000,
    label_val: str = "label",
    data_module=None
):
    """
    Generate t-SNE visualizations of learned embeddings
    
    Args:
        model: Trained model with encoder_embed method
        loader: DataLoader
        device: Computing device
        out_prefix: Output file prefix
        max_items: Maximum number of samples
        label_val: Label field name in batch
        data_module: Data module containing label mappings
    """
    model.eval()
    
    modes = ["s4", "bottleneck"]
    if getattr(model, "use_multiscale", False):
        modes.append("multiscale")
    
    def collect(mode: str):
        embs, labels = [], []
        count = 0

        with torch.no_grad():   # <-- QUAN TRỌNG
            for batch in loader:
                x = batch["input"].to(device, non_blocking=True)
                _, h = model.encoder_embed(x, mode=mode)

                h = F.normalize(h, dim=-1)
                embs.append(h.cpu().numpy())

                labels.append(
                    batch.get(label_val, torch.zeros(x.size(0), dtype=torch.long))
                    .cpu()
                    .numpy()
                )

                count += x.size(0)
                if count >= max_items:
                    break

        if not embs:
            return None, None

        return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)
    
    for mode in modes:
        X, y = collect(mode)
        if X is None:
            continue
        
        tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
        X2 = tsne.fit_transform(X)
        
        plt.figure(figsize=(6, 6))
        uniq = sorted(set(list(y)))
        
        for lbl in uniq:
            key = str(int(lbl)) if str(lbl).isdigit() else str(lbl)
            
            # Get label name and color from data module if available
            if data_module:
                if label_val == "label_1":
                    name_id = data_module.mindset_idx_map_label_1.get(key, key)
                    name = data_module.mindset_label_map_idx_1.get(name_id, name_id)
                    color = data_module.mindset_colors_1.get(name, "#888888")
                elif label_val == "label_2":
                    name_id = data_module.mindset_idx_map_label_2.get(key, key)
                    name = data_module.mindset_label_map_idx_2.get(name_id, name_id)
                    color = data_module.mindset_colors_2.get(name, "#888888")
                else:
                    name = data_module.hf_idx_map_label.get(key, key)
                    color = data_module.hf_demantia_colors.get(name, "#888888")
            else:
                name = key
                color = "#888888"
            
            mask = (y == lbl)
            plt.scatter(X2[mask, 0], X2[mask, 1], s=10, c=color, label=name, alpha=0.9)
        
        plt.legend(loc="best", fontsize=8, frameon=False)
        plt.tight_layout()
        
        path = f"{out_prefix}_enc_{mode}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()


def plot_loss_decomposition_curves(loss_decomp_csv_path: Path, output_dir: Path):
    """
    Tạo 3 plot:
    1. Reconstruction loss (orig / flip / total) cho train + val trong 1 ảnh
    2. Contrastive loss cho train + val trong 1 ảnh
    3. Total loss (recon + contrast) cho train + val trong 1 ảnh
    """
    if not loss_decomp_csv_path.exists():
        return

    df = pd.read_csv(loss_decomp_csv_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 1) Reconstruction losses
    # =====================================================
    plt.figure(figsize=(10, 6))
    for split in ["train", "val"]:
        sub = df[df["split"] == split]
        plt.plot(sub["epoch"], sub["loss_recon_orig"],
                 label=f"{split}-recon-orig")
        plt.plot(sub["epoch"], sub["loss_recon_flip"],
                 label=f"{split}-recon-flip")
        plt.plot(sub["epoch"], sub["loss_recon_total"],
                 label=f"{split}-recon-total")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss (orig / flip / total)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_loss_curves.png")
    plt.close()

    # =====================================================
    # 2) Contrastive loss
    # =====================================================
    plt.figure(figsize=(8, 5))
    for split in ["train", "val"]:
        sub = df[df["split"] == split]
        plt.plot(sub["epoch"], sub["loss_contrastive"],
                 label=f"{split}-contrastive")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Contrastive Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "contrastive_loss_curves.png")
    plt.close()

    # =====================================================
    # 3) Total loss
    # =====================================================
    plt.figure(figsize=(8, 5))
    for split in ["train", "val"]:
        sub = df[df["split"] == split]
        plt.plot(sub["epoch"], sub["loss_total"],
                 label=f"{split}-total")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss (recon + contrast)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "total_loss_curves.png")
    plt.close()

__all__ = [
    "save_image_grid",
    "plot_training_curves",
    "run_tsne_visualization",
    "plot_loss_decomposition_curves",
]