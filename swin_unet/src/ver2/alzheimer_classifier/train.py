from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score

from models.model_utils import flip_lr
from models.swin_unet_dualview_ssl import SwinUNetDualViewSSL
from training.ckpt_io import load_checkpoint_weights_filtered

from .io import ensure_dir, save_json


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
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        tgt = target.long()

        logp_t = logp.gather(dim=1, index=tgt.view(-1, 1)).squeeze(1)
        p_t = p.gather(dim=1, index=tgt.view(-1, 1)).squeeze(1)

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


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: SwinUNetDualViewSSL, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        c3 = 8 * int(getattr(encoder, "embed_dim", 96))
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, plane_one_hot: torch.Tensor) -> torch.Tensor:
        b1 = self.encoder.encode_bottleneck(x1, plane_one_hot, view=1)
        b2 = self.encoder.encode_bottleneck(x2, plane_one_hot, view=2)
        h1 = b1.mean(dim=(1, 2))
        h2 = b2.mean(dim=(1, 2))
        h = 0.5 * (h1 + h2)
        h = self.dropout(h)
        return self.fc(h)


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, tfm):
        if not hasattr(hf_ds, "features") or "label" not in hf_ds.features:
            raise ValueError("dataset is missing required 'label' feature")
        self.ds = hf_ds
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        if "label" not in item:
            raise ValueError("dataset item missing 'label'")
        y = item["label"]
        if y is None:
            raise ValueError("dataset item label is None")
        img = item["image"]
        x1 = self.tfm(img)
        x2 = flip_lr(x1)
        return x1, x2, int(y)


def prepare_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) != 3:
        raise ValueError("batch must be (x1, x2, y); masking is not allowed")
    x1, x2, y = batch
    x1 = x1.to(device, non_blocking=True)
    x2 = x2.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if x1.shape != x2.shape:
        raise ValueError("view shapes must match for classifier training")
    if x1.dtype != x2.dtype:
        raise ValueError("view dtypes must match for classifier training")
    return x1, x2, y


def build_plane_one_hot(plane: str, batch_size: int, device: torch.device) -> torch.Tensor:
    plane = plane.lower().strip()
    if plane == "axial":
        v = torch.tensor([0.0, 1.0], device=device)
    elif plane == "coronal":
        v = torch.tensor([1.0, 0.0], device=device)
    else:
        raise ValueError("plane must be axial or coronal")
    return v.view(1, 2).repeat(batch_size, 1)


def save_confusion_matrix_png(cm, class_names, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ensure_dir(out_path.parent)

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


def parse_focal_alpha(value: str, num_classes: int) -> Optional[torch.Tensor]:
    if not value:
        return None
    s = value.strip()
    if s.startswith("scalar:"):
        return torch.tensor([float(s.split("scalar:")[1])], dtype=torch.float32)
    if s.startswith("list:"):
        vals = [float(v) for v in s.split("list:")[1].split(",")]
        if len(vals) != num_classes:
            raise ValueError("focal_alpha list size must match num_classes")
        return torch.tensor(vals, dtype=torch.float32)
    raise ValueError("focal_alpha format must be '', 'scalar:0.25', or 'list:a,b,c,d'")


def parse_ce_class_weights(value: str, num_classes: int) -> torch.Tensor:
    if not value:
        raise ValueError("ce_class_weights is required for loss_type=wce (format: list:w0,w1,...)")
    s = value.strip()
    if not s.startswith("list:"):
        raise ValueError("ce_class_weights format must be 'list:w0,w1,...'")
    vals = [float(v) for v in s.split("list:")[1].split(",")]
    if len(vals) != num_classes:
        raise ValueError("ce_class_weights list size must match num_classes")
    return torch.tensor(vals, dtype=torch.float32)


def print_label_map(label_names: List[str]) -> None:
    for i, name in enumerate(label_names):
        print(f"[label_map] {i} -> {name}")


def normalize_label_name(name: str) -> str:
    s = name.lower().strip().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    if "moderate" in s:
        return "moderate demented"
    if "very mild" in s:
        return "very mild demented"
    if "mild" in s:
        return "mild demented"
    if "non" in s:
        return "non-demented"
    return s


def infer_num_classes(ds) -> int:
    try:
        names = list(ds.features["label"].names)
        return len(names)
    except Exception:
        labels = []
        for i in range(len(ds)):
            item = ds[i]
            if "label" not in item:
                raise ValueError("labels missing from dataset")
            if item["label"] is None:
                raise ValueError("labels missing from dataset")
            labels.append(int(item["label"]))
        return int(max(labels)) + 1 if labels else 0


def build_model(args: argparse.Namespace, device: torch.device) -> SwinUNetDualViewSSL:
    if getattr(args, "enable_masking", False):
        raise RuntimeError("masking must be disabled for classifier training")
    if getattr(args, "enable_contrastive", False):
        raise RuntimeError("contrastive must be disabled for classifier training")
    if getattr(args, "enable_reconstruct", False):
        raise RuntimeError("reconstruct must be disabled for classifier training")

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
        enable_contrastive=False,
    ).to(device)

    if encoder.enable_reconstruct or encoder.enable_contrastive:
        raise RuntimeError("encoder must be encoder-only (no reconstruct/contrastive)")
    if getattr(encoder, "proj", None) is not None:
        raise RuntimeError("projection head must be disabled for classifier training")

    return encoder


def maybe_load_encoder_weights(args: argparse.Namespace, encoder: SwinUNetDualViewSSL, device: torch.device) -> None:
    if not args.resume_ckpt:
        return
    if args.ckpt_load_mode != "encoder_only":
        return
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, object]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    all_pred: List[torch.Tensor] = []
    all_true: List[torch.Tensor] = []

    for batch in loader:
        x1, x2, y = prepare_batch(batch, device)
        plane = build_plane_one_hot("axial", x1.size(0), device)
        logits = model(x1, x2, plane)
        loss = criterion(logits, y)

        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        loss_sum += float(loss.item()) * float(y.numel())

        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    y_true = torch.cat(all_true, dim=0).numpy() if all_true else []
    y_pred = torch.cat(all_pred, dim=0).numpy() if all_pred else []

    return {
        "acc": float(correct) / float(max(total, 1)),
        "ce": float(loss_sum) / float(max(total, 1)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0,
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")) if len(y_true) else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_fold(
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    out_dir: Path,
    class_names: List[str],
) -> Dict[str, object]:
    ensure_dir(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    ensure_dir(ckpt_dir)
    ensure_dir(plots_dir)

    encoder = build_model(args, device)
    maybe_load_encoder_weights(args, encoder, device)

    model = EncoderClassifier(encoder=encoder, num_classes=num_classes, dropout=args.dropout).to(device)
    if model.fc.out_features != num_classes:
        raise RuntimeError("classifier head size must match num_classes")

    if args.loss_type == "focal":
        alpha = parse_focal_alpha(args.focal_alpha, num_classes)
        if alpha is not None:
            alpha_vec = alpha if alpha.numel() > 1 else alpha.repeat(num_classes)
            for i, name in enumerate(class_names):
                print(f"[focal_alpha] idx={i} name={name} w={float(alpha_vec[i].item())}")
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha, reduction="mean").to(device)
    elif args.loss_type == "wce":
        weights = parse_ce_class_weights(args.ce_class_weights, num_classes)
        for i, name in enumerate(class_names):
            print(f"[ce_weights] idx={i} name={name} w={float(weights[i].item())}")
        criterion = nn.CrossEntropyLoss(weight=weights.to(device=device, dtype=torch.float32)).to(device)
    else:
        raise ValueError(f"unsupported loss_type: {args.loss_type}")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_f1 = -1.0
    best_epoch = -1
    best_test = None

    for epoch in range(1, args.epochs + 1):
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
        train_pred = []
        train_true = []

        for batch in train_loader:
            x1, x2, y = prepare_batch(batch, device)
            opt.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=(args.amp and device.type == "cuda")):
                plane = build_plane_one_hot("axial", x1.size(0), device)
                logits = model(x1, x2, plane)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            loss_sum += float(loss.item()) * float(y.numel())
            train_pred.append(pred.detach().cpu())
            train_true.append(y.detach().cpu())

        train_acc = float(correct) / float(max(total, 1))
        train_loss = float(loss_sum) / float(max(total, 1))
        test_metrics = evaluate(model, test_loader, device, criterion)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_metrics['ce']:.4f} test_acc={test_metrics['acc']:.4f} "
            f"test_f1={test_metrics['f1_macro']:.4f}"
        )

        score = test_metrics["f1_macro"]
        if score > best_f1:
            best_f1 = float(score)
            best_epoch = epoch
            best_test = test_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "test_acc_at_best_f1": test_metrics["acc"],
                    "encoder_state": encoder.state_dict(),
                    "clf_state": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                },
                ckpt_dir / "best_cls.pt",
            )

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

    return {
        "best_epoch": best_epoch,
        "best_score": best_f1,
        "test": best_test,
    }


def run_single_split(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    ds = load_dataset("Falah/Alzheimer_MRI")
    train_ds = ds["train"]
    test_ds = ds["test"]

    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    train_pt = HFDataset(train_ds, tfm)
    test_pt = HFDataset(test_ds, tfm)

    try:
        label_names = list(train_ds.features["label"].names)
    except Exception as exc:
        raise RuntimeError("dataset label names are required for classifier training") from exc

    expected_order = [
        "non-demented",
        "very mild demented",
        "mild demented",
        "moderate demented",
    ]
    detected = [normalize_label_name(name) for name in label_names]
    if len(detected) != len(expected_order):
        raise RuntimeError(
            f"label count mismatch: detected={label_names} expected={expected_order}"
        )
    if detected != expected_order:
        raise RuntimeError(
            f"label order mismatch: detected={label_names} expected={expected_order}"
        )

    class_names = [
        "Non-Demented",
        "Very Mild Demented",
        "Mild Demented",
        "Moderate Demented",
    ]
    num_classes = 4
    print_label_map(class_names)

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

    metrics = train_one_fold(
        args=args,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        out_dir=out_dir,
        class_names=class_names,
    )
    best_test = metrics.get("test") or {}
    record = {
        "best_epoch": int(metrics.get("best_epoch", -1)),
        "best_score": float(metrics.get("best_score", -1.0)),
        "test_acc": float(best_test.get("acc", 0.0)),
        "test_ce": float(best_test.get("ce", 0.0)),
        "test_f1_macro": float(best_test.get("f1_macro", 0.0)),
        "test_f1_weighted": float(best_test.get("f1_weighted", 0.0)),
    }
    save_json(out_dir / "metrics" / "single_split_metrics.json", record)
    print("[done] best_f1:", metrics["best_score"])


def run(args: argparse.Namespace) -> None:
    run_single_split(args)
