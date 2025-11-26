from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import os
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from model import SmallUNetSSL
from train import preprocess_batch, set_seed

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alzheimer_unet_data import (
    mindset_idx_map_label_1,
    mindset_label_map_idx_1,
)


# -----------------------------
# Small helper for preprocessing
# -----------------------------
@dataclass
class PreprocessConfig:
    pre_bias: bool = True
    pre_norm: bool = True
    pre_crop: bool = True
    pre_align: bool = True
    image_size: int = 192


# -----------------------------
# Dataset
# -----------------------------
class MindsetMRIDataset(Dataset):
    """
    Dataset that reads the CSV with columns:
        - img_path: image file name
        - abnormal_type: string used with mindset_label_map_idx_1
        - set: "train" or "test"
    """
    def __init__(self, df: pd.DataFrame, images_root: Path):
        self.df = df.reset_index(drop=True)
        self.images_root = Path(images_root)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_file = self.images_root / row["img_path"]

        if not img_file.is_file():
            raise FileNotFoundError(f"Image file not found: {img_file}")

        # Load as single-channel (grayscale) and convert to tensor in [0,1]
        img = Image.open(img_file).convert("L")
        img = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).unsqueeze(0)  # [1,H,W]

        abnormal_type = row["abnormal_type"]
        if abnormal_type not in mindset_label_map_idx_1:
            raise KeyError(f"Unknown abnormal_type: {abnormal_type}")
        label = mindset_label_map_idx_1[abnormal_type]

        return img, label


# -----------------------------
# Classifier on top of bottleneck features
# -----------------------------
class UNetBottleneckClassifier(nn.Module):
    """
    Wraps a frozen SmallUNetSSL and adds a simple Linear head on top of
    the bottleneck embedding (encoder_embed with mode='bottleneck').
    """
    def __init__(self, backbone: SmallUNetSSL, num_classes: int = 4, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # bottleneck_dim is the output dim of the bottleneck head in embed_fc
        bottleneck_dim = self.backbone.embed_fc["bottleneck"].out_features
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute bottleneck embedding without tracking gradients for the backbone
        with torch.no_grad():
            _, h = self.backbone.encoder_embed(x, mode="bottleneck")
        logits = self.classifier(h)
        return logits


# -----------------------------
# Utilities
# -----------------------------
def build_backbone_from_checkpoint(ckpt_path: Path, device: torch.device) -> SmallUNetSSL:
    ckpt = torch.load(str(ckpt_path), map_location=device)

    # Recreate the backbone with the same hyperparameters used in SSL training
    ssl_args = ckpt.get("args", {})
    model_kwargs = {
        "in_ch": 1,
        "base_ch": ssl_args.get("base_ch", 16),
        "bottleneck_dim": ssl_args.get("bottleneck_dim", 128),
        "proj_dim": ssl_args.get("proj_dim", 128),
        "use_gn": ssl_args.get("use_gn", False),
        "use_se": ssl_args.get("use_se", False),
        "use_multiscale": ssl_args.get("use_multiscale", True),
    }
    backbone = SmallUNetSSL(**model_kwargs)
    backbone.load_state_dict(ckpt["model"], strict=True)
    backbone.to(device)
    backbone.eval()
    return backbone


def create_dataloaders(
    csv_path: Path,
    images_root: Path,
    batch_size: int,
    val_ratio: float = 0.2,
    num_workers: int = 4,
):
    df = pd.read_csv(csv_path)

    # Use "train" rows for train/val splits and keep "test" as held-out
    train_df = df[df["set"] == "train"].reset_index(drop=True)
    test_df = df[df["set"] == "test"].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("CSV contains no rows with set == 'train'")
    if len(test_df) == 0:
        print("Warning: CSV contains no rows with set == 'test'. Test metrics will be skipped.")

    full_train_dataset = MindsetMRIDataset(train_df, images_root)
    test_dataset = MindsetMRIDataset(test_df, images_root) if len(test_df) > 0 else None

    # Split train into train and val
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(num_train * (1.0 - val_ratio))
    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: UNetBottleneckClassifier,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    preprocess_cfg: PreprocessConfig,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)  # [B,1,H,W] in [0,1]
        labels = labels.to(device, dtype=torch.long)

        # Apply same preprocessing as SSL training (bias, crop, align, etc.)
        imgs = preprocess_batch(imgs, preprocess_cfg)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: UNetBottleneckClassifier,
    loader: DataLoader,
    device: torch.device,
    preprocess_cfg: PreprocessConfig,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device, dtype=torch.long)

        imgs = preprocess_batch(imgs, preprocess_cfg)

        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

def build_argparser():
    p = argparse.ArgumentParser(
        "Train 4-class classifier (Normal / MTL / Other / WMH) on top of SmallUNetSSL bottleneck"
    )
    p.add_argument("--csv-path", type=str, default="data_description.csv")
    p.add_argument("--images-root", type=str, required=True, help="Directory containing image files")
    p.add_argument("--ckpt-path", type=str, required=True, help="Path to SSL UNet checkpoint (ckpt_best.pt)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--outdir", type=str, default="runs_bottleneck_cls")
    
    return p

# -----------------------------
# Main
# -----------------------------
def main():
    
    args = build_argparser().parse_args()

    # Resolve paths
    csv_path = Path(args.csv_path)
    images_root = Path(args.images_root)
    ckpt_path = Path(args.ckpt_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Reproducibility
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Preprocess config (mirrors SSL preprocessing pipeline)
    preprocess_cfg = PreprocessConfig(
        pre_bias=True,
        pre_norm=True,
        pre_crop=True,
        pre_align=True,
        image_size=args.image_size,
    )

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=csv_path,
        images_root=images_root,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )

    # Backbone + classifier
    backbone = build_backbone_from_checkpoint(ckpt_path, device)
    model = UNetBottleneckClassifier(backbone, num_classes=4, freeze_backbone=True).to(device)

    # Only train the classifier parameters
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, preprocess_cfg)
        val_loss, val_acc = evaluate(model, val_loader, device, preprocess_cfg)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        # Save best checkpoint based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "backbone_ckpt": str(ckpt_path),
                "val_acc": val_acc,
                "args": vars(args),
                "label_map": mindset_idx_map_label_1,
            }
            torch.save(ckpt, str(outdir / "cls_ckpt_best.pt"))

    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # Final test evaluation
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, device, preprocess_cfg)
        print(f"Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")
    else:
        print("No test set found in CSV (set == 'test'), skipping test evaluation.")


if __name__ == "__main__":
    main()
