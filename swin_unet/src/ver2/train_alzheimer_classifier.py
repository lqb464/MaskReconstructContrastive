# train_alzheimer_classifier.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_dataset
from sklearn.metrics import f1_score, confusion_matrix

from models.swin_unet_dualview_ssl import SwinUNetDualViewSSL
from training.ckpt_io import load_checkpoint_weights_filtered

# -------------------------
# Focal Loss (multi-class)
# -------------------------
class FocalLoss(nn.Module):
    """
    Multi-class focal loss on logits.
    - logits: [B, K]
    - target: [B] with class indices
    alpha:
      - None: no class weighting
      - float: scalar alpha applied uniformly
      - list/tuple/tensor of shape [K]: per-class alpha
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)              # [B,K]
        p = logp.exp()                                   # [B,K]
        tgt = target.long()

        logp_t = logp.gather(dim=1, index=tgt.view(-1, 1)).squeeze(1)  # [B]
        p_t = p.gather(dim=1, index=tgt.view(-1, 1)).squeeze(1)        # [B]

        focal = (1.0 - p_t).pow(self.gamma)

        if self.alpha is None:
            loss = -focal * logp_t
        else:
            if self.alpha.numel() == 1:
                a_t = self.alpha.view(1).expand_as(p_t)
            else:
                a_t = self.alpha.gather(dim=0, index=tgt)
            loss = -a_t * focal * logp_t

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


# -------------------------
# Small classifier head
# -------------------------
class EncoderClassifier(nn.Module):
    def __init__(self, encoder: SwinUNetDualViewSSL, num_classes: int = 4, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        # bottleneck channel dim = C3 = 8 * embed_dim (per your model)
        c3 = 8 * int(getattr(encoder, "embed_dim", 96))
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor, plane_one_hot: torch.Tensor) -> torch.Tensor:
        # Use encode_bottleneck to avoid masking + flip logic in forward()
        # encode_bottleneck returns NHWC [B,H',W',C3]
        b = self.encoder.encode_bottleneck(x, plane_one_hot, view=1)
        h = b.mean(dim=(1, 2))  # GAP over H,W -> [B,C3]
        h = self.dropout(h)
        return self.fc(h)


# -------------------------
# Dataset wrapper
# -------------------------
class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, tfm):
        self.ds = hf_ds
        self.tfm = tfm

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        img = item["image"]  # PIL
        y = int(item["label"])
        x = self.tfm(img)    # [1,H,W] float32
        return x, y


def build_plane_one_hot(plane: str, batch_size: int, device: torch.device) -> torch.Tensor:
    plane = plane.lower().strip()
    if plane == "axial":
        v = torch.tensor([0.0, 1.0], device=device)
    elif plane == "coronal":
        v = torch.tensor([1.0, 0.0], device=device)
    else:
        raise ValueError("plane must be axial or coronal")
    return v.view(1, 2).repeat(batch_size, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        plane = build_plane_one_hot("axial", x.size(0), device)
        logits = model(x, plane)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        loss_sum += float(loss.item()) * float(y.numel())

    return {
        "acc": float(correct) / float(max(total, 1)),
        "ce": float(loss_sum) / float(max(total, 1)),
    }

def save_confusion_matrix_png(
    cm,
    class_names,
    out_path: Path,
    title: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() * 0.5 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            plt.text(
                j,
                i,
                str(v),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="runs/alzheimer_cls")
    p.add_argument("--resume-ckpt", type=str, default="")
    p.add_argument(
        "--ckpt-load-mode", 
        type=str, 
        default="encoder_only",           
        choices=["none", "full", "encoder_only"],
    )
    p.add_argument("--in-ch", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--enc-depths", type=int, nargs=4, default=[2, 2, 6, 2])
    p.add_argument("--dec-depths", type=int, nargs=3, default=[6, 2, 2])
    p.add_argument("--num-heads", type=int, nargs=4, default=[3, 6, 12, 24])
    p.add_argument("--window-size", type=int, default=7)

    p.add_argument("--bottleneck-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)
    
    # Plane conditioning
    p.add_argument("--plane-inject-method", type=str, default="film", choices=["film", "add"])
    
    # SACA 
    p.add_argument("--enable_saca", action="store_true")
    p.add_argument("--saca_position", type=str, default="after_stage1", choices=["after_patch_embed", "after_merge0", "after_stage1"])
    p.add_argument("--saca_gate_init", type=float, default=0.0)
    p.add_argument("--saca_warmup_epochs", type=int, default=0)
    
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    
    p.add_argument("--enable-contrastive", action="store_true")
    p.add_argument("--disable-contrastive", dest="enable_contrastive", action="store_false")
    p.set_defaults(enable_contrastive=True)

    p.add_argument("--freeze_encoder_epochs", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.0)

    # focal loss params
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=str, default="")  # "", "scalar:0.25", "list:0.5,1,1,1"

    return p
    

def main():
    parser = build_argparser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)


    # -------------------------
    # Load HF dataset
    # -------------------------
    ds = load_dataset("Falah/Alzheimer_MRI")
    # splits: train, test (per dataset card)
    train_ds = ds["train"]
    test_ds = ds["test"]

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),  # [0,1]
    ])

    train_pt = HFDataset(train_ds, tfm)
    test_pt = HFDataset(test_ds, tfm)

    train_loader = DataLoader(
        train_pt,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_pt,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # -------------------------
    # Build encoder in "encoder_only + no reconstruct"
    # -------------------------
    encoder = SwinUNetDualViewSSL(
        in_ch=args.in_ch,
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        enc_depths=tuple(args.enc_depths),
        dec_depths=tuple(args.dec_depths),
        num_heads=tuple(args.num_heads),
        window_size=args.window_size,
        proj_dim=args.proj_dim,
        plane_inject_method=args.plane_inject_method,
        enable_saca=args.enable_saca,
        saca_position=args.saca_position,
        saca_gate_init=args.saca_gate_init,
        saca_warmup_epochs=args.saca_warmup_epochs,
        enable_reconstruct=False,
        enable_contrastive=args.enable_contrastive,  
        contrastive_loss_type="infonce",
        contrastive_position="bottleneck",
    ).to(device)

    # Load pretrained encoder weights
    if args.resume_ckpt and args.ckpt_load_mode == "encoder_only":
        if load_checkpoint_weights_filtered is None:
            raise RuntimeError("training.ckpt_io.load_checkpoint_weights_filtered is not available in your environment")
        ckpt_path = Path(args.resume_ckpt)
        _ = load_checkpoint_weights_filtered(
            ckpt_path=ckpt_path,
            device=device,
            model=encoder,
            include_prefixes=encoder.encoder_state_dict_prefixes(),
            exclude_prefixes=("proj_c1", "proj_c2", "proj_c3", "proj"),
        )
        print("[ckpt] loaded encoder_only from:", str(ckpt_path))

    model = EncoderClassifier(encoder=encoder, num_classes=4, dropout=args.dropout).to(device)

    # -------------------------
    # Focal loss alpha parsing
    # -------------------------
    alpha = None
    if args.focal_alpha:
        s = args.focal_alpha.strip()
        if s.startswith("scalar:"):
            alpha = torch.tensor([float(s.split("scalar:")[1])], dtype=torch.float32)
        elif s.startswith("list:"):
            vals = s.split("list:")[1].split(",")
            alpha = torch.tensor([float(v) for v in vals], dtype=torch.float32)
        else:
            raise ValueError("focal_alpha format must be '', 'scalar:0.25', or 'list:a,b,c,d'")

    criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha, reduction="mean").to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        
        # freeze encoder for first N epochs, then unfreeze
        freeze_n = int(getattr(args, "freeze_encoder_epochs", 0))
        encoder_trainable = not (epoch <= freeze_n)
        encoder.set_encoder_trainable(trainable=encoder_trainable)

        if epoch == 1 and freeze_n > 0:
            print(f"[train] freeze encoder for first {freeze_n} epochs")
        if epoch == freeze_n + 1 and freeze_n > 0:
            print("[train] encoder unfrozen")
        
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            plane = build_plane_one_hot("axial", x.size(0), device)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=(args.amp and device.type == "cuda")):
                logits = model(x, plane)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            loss_sum += float(loss.item()) * float(y.numel())

        train_acc = float(correct) / float(max(total, 1))
        train_loss = float(loss_sum) / float(max(total, 1))

        # quick eval (test)
        model.eval()
        t_correct = 0
        t_total = 0
        t_loss_sum = 0.0
        all_pred = []
        all_true = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                plane = build_plane_one_hot("axial", x.size(0), device)

                logits = model(x, plane)
                loss = criterion(logits, y)

                pred = logits.argmax(dim=-1)

                t_correct += int((pred == y).sum().item())
                t_total += int(y.numel())
                t_loss_sum += float(loss.item()) * float(y.numel())

                all_pred.append(pred.detach().cpu())
                all_true.append(y.detach().cpu())

        test_acc = float(t_correct) / float(max(t_total, 1))
        test_loss = float(t_loss_sum) / float(max(t_total, 1))

        y_true = torch.cat(all_true, dim=0).numpy()
        y_pred = torch.cat(all_pred, dim=0).numpy()

        # choose macro for balanced view across classes
        test_f1 = float(f1_score(y_true, y_pred, average="macro"))
        
        cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))

        # try to read label names from HF dataset, fallback to indices
        try:
            class_names = list(train_ds.features["label"].names)
        except Exception:
            class_names = [str(i) for i in range(4)]

        # save confusion matrix for this epoch
        save_confusion_matrix_png(
            cm=cm,
            class_names=class_names,
            out_path=plots_dir / f"confusion_matrix_epoch_{epoch:03d}.png",
            title=f"Confusion Matrix (epoch {epoch:03d})",
        )

        # also save the best confusion matrix snapshot when best_f1 updates

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} " 
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
        )

        # save best by F1
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(
                {
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "test_acc_at_best_f1": test_acc,
                    "encoder_state": encoder.state_dict(),
                    "clf_state": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                },
                ckpt_dir / "best_cls.pt",
            )
            
            save_confusion_matrix_png(
                cm=cm,
                class_names=class_names,
                out_path=plots_dir / "confusion_matrix_best_f1.png",
                title=f"Confusion Matrix (best_f1={best_f1:.4f})",
            )


        # save latest
        torch.save(
            {
                "epoch": epoch,
                "best_f1": best_f1,
                "encoder_state": encoder.state_dict(),
                "clf_state": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
            },
            ckpt_dir / "latest_cls.pt",
        )

    print("[done] best_f1:", best_f1)


if __name__ == "__main__":
    main()
