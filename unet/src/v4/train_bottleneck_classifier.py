from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
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
from collections import Counter



# -----------------------------
# Label mappings
# -----------------------------
mindset_idx_map_label_1 = {
    0: "Normal",
    1: "MTL",
    2: "Other",
    3: "WMH",
}

mindset_label_map_idx_1 = {
    "mtl_atrophy": 1,
    "mtl_atrophy,other_atrophy": 1,
    "mtl_atrophy,wmh": 1,
    "normal": 0,
    "other_atrophy": 2,
    "wmh": 3,
    "wmh,other_atrophy": 3,
}


# -----------------------------
# Preprocess config for preprocess_batch
# -----------------------------
@dataclass
class PreprocessConfig:
    pre_bias: bool = False
    pre_norm: bool = True
    pre_crop: bool = True
    pre_align: bool = True
    image_size: int = 192

@torch.no_grad()
def debug_predictions(model, loader, device, preprocess_cfg, label_map):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        imgs = preprocess_batch(imgs, preprocess_cfg)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    print("Label counts:", torch.bincount(all_labels, minlength=4))
    print("Pred counts:", torch.bincount(all_preds, minlength=4))

    # Optional: confusion matrix
    cm = torch.zeros(4, 4, dtype=torch.int64)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


# -----------------------------
# Dataset
# -----------------------------
class MindsetMRIDataset(Dataset):
    """
    Dataset that reads a CSV with columns:
      - img_path: file name under images_root
      - abnormal_type: string label used with mindset_label_map_idx_1
      - set: "train" or "test" (used only to split in create_dataloaders)
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

        # Load grayscale, convert to float tensor in [0,1], shape [1,H,W]
        img = Image.open(img_file).convert("L")
        np_img = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(np_img).unsqueeze(0).contiguous()

        abnormal_type = row["abnormal_type"]
        if abnormal_type not in mindset_label_map_idx_1:
            raise KeyError(f"Unknown abnormal_type: {abnormal_type}")
        label = mindset_label_map_idx_1[abnormal_type]

        return img_t, int(label)


# -----------------------------
# Backbone + classifier
# -----------------------------
class UNetBottleneckClassifier(nn.Module):
    """
    Wrap a frozen SmallUNetSSL and train a linear classifier on top of the
    bottleneck embedding returned by encoder_embed(mode='bottleneck').
    """
    def __init__(
        self, 
        backbone: SmallUNetSSL, 
        num_classes: int = 4, 
        freeze_backbone: bool = True,
        unfreeze_last: bool = False,
    ):
        super().__init__()
        self.backbone = backbone

        # Freeze everything first
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_last:
            # Example: unfreeze the deepest encoder block and bottleneck conv
            for p in self.backbone.enc4.parameters():
                p.requires_grad = True
            for p in self.backbone.bottleneck.parameters():
                p.requires_grad = True
            # Also unfreeze the bottleneck embedding MLP
            for p in self.backbone.embed_fc["bottleneck"].parameters():
                p.requires_grad = True

        bottleneck_dim = self.backbone.embed_fc["bottleneck"].out_features
        # stronger head
        self.classifier = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, h = self.backbone.encoder_embed(x, mode="bottleneck")
        logits = self.classifier(h)
        return logits


def build_backbone_from_checkpoint(ckpt_path: Path, device: torch.device) -> SmallUNetSSL:
    ckpt = torch.load(str(ckpt_path), map_location=device)

    ssl_args = ckpt.get("args", {})

    def get_param(name, default=None):
        if name in ssl_args:
            return ssl_args[name]
        if name in ckpt:
            return ckpt[name]
        if default is not None:
            return default
        raise KeyError(f"Missing `{name}` in checkpoint args/ckpt")

    model_kwargs = {
        "in_ch": 1,
        "base_ch":        get_param("base_ch", 16),
        "bottleneck_dim": get_param("bottleneck_dim", 128),
        "proj_dim":       get_param("proj_dim", 128),
        "use_gn":         get_param("use_gn", False),
        "use_se":         get_param("use_se", False),
        "use_multiscale": get_param("use_multiscale", True),
    }

    backbone = SmallUNetSSL(**model_kwargs)
    backbone.load_state_dict(ckpt["model"], strict=True)
    backbone.to(device)
    backbone.eval()
    return backbone


# -----------------------------
# Dataloaders with custom collate_fn
# -----------------------------
def _mri_collate(batch):
    # batch: list of (img_t, label_int)
    imgs, labels = zip(*batch)

    # Get max H, W in this batch
    Hs = [img.shape[-2] for img in imgs]
    Ws = [img.shape[-1] for img in imgs]
    maxH, maxW = max(Hs), max(Ws)

    padded_imgs = []
    for img in imgs:
        # img: [1, H, W]
        C, H, W = img.shape
        # zero pad to [1, maxH, maxW]
        pad = torch.zeros((C, maxH, maxW), dtype=img.dtype)
        pad[:, :H, :W] = img
        padded_imgs.append(pad)

    imgs_tensor = torch.stack(padded_imgs, dim=0)         # [B, 1, maxH, maxW]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return imgs_tensor, labels_tensor



def create_dataloaders(
    csv_path: Path,
    images_root: Path,
    batch_size: int,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    df = pd.read_csv(csv_path)

    train_df = df[df["set"] == "train"].reset_index(drop=True)
    test_df = df[df["set"] == "test"].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("CSV contains no rows with set == 'train'")

    full_train_dataset = MindsetMRIDataset(train_df, images_root)
    test_dataset = MindsetMRIDataset(test_df, images_root) if len(test_df) > 0 else None

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
        pin_memory=pin_memory,
        collate_fn=_mri_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_mri_collate,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_mri_collate,
        )

    return train_loader, val_loader, test_loader


# -----------------------------
# Train / eval loops
# -----------------------------
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
        imgs = imgs.to(device)
        labels = labels.to(device, dtype=torch.long)

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


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        "Train 4-class classifier (Normal / MTL / Other / WMH) on top of UNet bottleneck"
    )
    parser.add_argument("--csv-path", type=str, default="data_description.csv")
    parser.add_argument("--images-root", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, default="runs_bottleneck_cls")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    images_root = Path(args.images_root)
    ckpt_path = Path(args.ckpt_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    preprocess_cfg = PreprocessConfig(
        pre_bias=False,
        pre_norm=True,
        pre_crop=True,
        pre_align=True,
        image_size=args.image_size,
    )

    pin_memory = device.type == "cuda"

    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=csv_path,
        images_root=images_root,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    backbone = build_backbone_from_checkpoint(ckpt_path, device)
    model = UNetBottleneckClassifier(
        backbone, 
        num_classes=4, 
        freeze_backbone=True,
        unfreeze_last=True,
    ).to(device)
    
    # Create param groups with different lrs
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)



    optimizer = optim.AdamW(
        [
            {"params": head_params, "lr": args.lr},            # e.g. 1e3
            {"params": backbone_params, "lr": args.lr * 1e2}, # e.g. 1e5
        ],
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, preprocess_cfg)
        val_loss, val_acc = evaluate(model, val_loader, device, preprocess_cfg)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        
        debug_predictions(model, val_loader, device, preprocess_cfg, mindset_idx_map_label_1)

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

    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, device, preprocess_cfg)
        print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")
    else:
        print("No test set with set == 'test' found in CSV; skipping test evaluation.")


if __name__ == "__main__":
    main()
