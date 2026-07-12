from __future__ import annotations

"""
Alzheimer classifier training on MAE/VAE encoder bottleneck features.
Supports classification_default and classification_bottleneck_concat only.
"""

import argparse
import csv
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

from ..config.experiment import ExperimentConfig
from ..downstream.model_utils import build_downstream_model, encode_dual_bottleneck, pool_bottleneck
from ..models.model_utils import flip_lr
from ..training.ckpt_io import load_checkpoint_weights_filtered
from ..training.utils import get_device

from .classification_head import ClassificationHead
from .io import ensure_dir, save_json

class FocalLoss(nn.Module):
    """
    Multi-class focal loss on logits.
    - logits: [B, K]
    - target: [B] with class indices
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

class HFDataset(torch.utils.data.Dataset):
    """Wrap HF dataset -> (x1, x2, y) with dual view (flip_lr)."""

    def __init__(self, hf_ds, tfm):
        if not hasattr(hf_ds, "features") or "label" not in hf_ds.features:
            raise ValueError("dataset is missing required 'label' feature")
        self.ds = hf_ds
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        if "label" not in item or item["label"] is None:
            raise ValueError("dataset item missing 'label'")
        y = int(item["label"])
        img = item["image"]
        x1 = self.tfm(img)
        x2 = flip_lr(x1)
        return x1, x2, y

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

@dataclass
class ClsConfig:
    num_classes: int
    class_names: List[str]
    classification_mode: str
    fusion: str
    clf_hidden_dim: int
    clf_dropout: float
    clf_activation: str
    clf_layernorm: bool
    amp: bool
    device: torch.device
    loss_type: str
    focal_gamma: float
    focal_alpha: Optional[torch.Tensor]
    ce_weights: Optional[torch.Tensor]
    freeze_encoder_epochs: int
    view_mode: str
    head_in_dim: int
    backbone: str

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

def _args_to_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    backbone = "vae" if getattr(args, "vae", False) else "mae"
    ns = argparse.Namespace(
        data_root="",
        preprocessed_dir="",
        train_mod=1.0,
        image_size=args.image_size,
        plane="axial",
        val_ratio=0.2,
        test_ratio=0.0,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        split_test=False,
        label_csv="",
        label_path_col="image_path",
        label_col="label",
        in_ch=args.in_ch,
        patch_size=args.patch_size,
        mask_ratio=0.35,
        enable_masking=False,
        mae=args.mae or backbone == "mae",
        vae=args.vae or backbone == "vae",
        base_ch=args.base_ch,
        use_gn=args.use_gn,
        embed_dim=args.embed_dim,
        mae_enc_depth=args.mae_enc_depth,
        mae_dec_depth=args.mae_dec_depth,
        latent_dim=args.latent_dim,
        enable_reconstruct=False,
        disable_reconstruct=True,
        enable_contrastive=False,
        single_view=args.view_mode != "two",
        enable_masked_loss=False,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_recon=1.0,
        lambda_contrast=0.0,
        lambda_kl=1e-4,
        temperature=0.2,
        seed=args.seed,
        cpu=args.cpu,
        amp=args.amp,
        resume_ckpt=args.resume_ckpt,
        ckpt_load_mode=args.ckpt_load_mode,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        reset_contrastive_proj_head=True,
        aug_p_noise=0.7,
        aug_p_jitter=0.7,
        aug_p_blur=0.2,
        aug_noise_std=0.02,
        aug_jitter_strength=0.1,
        aug_blur_kernel=3,
        recon_loss="weighted_bce_logits",
        fg_eps=0.02,
        fg_weight=10.0,
        out_dir=args.out_dir,
        run_name="",
        ckpt_dir="",
        vis_every=20,
        save_latest_every=1,
        save_best_after_epoch=0,
        save_best_every=20,
        enable_tsne=False,
        tsne_only_if_labeled=True,
        tsne_every=20,
        tsne_max_items=1000,
        contrastive_loss_type="infonce",
        contrastive_position="bottleneck",
        vicreg_invariance_weight=25.0,
        vicreg_variance_weight=25.0,
        vicreg_covariance_weight=1.0,
        vicreg_variance_eps=1e-4,
        vicreg_target_std=1.0,
    )
    return ExperimentConfig.from_args(ns)


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    cfg = _args_to_experiment_config(args)
    model = build_downstream_model(
        cfg,
        out_ch=1,
        enable_reconstruct=False,
        single_view=args.view_mode != "two",
    ).to(device)
    return model

@torch.no_grad()
def infer_feature_dims(model: nn.Module, args: argparse.Namespace, device: torch.device) -> Dict[str, int]:
    model.eval()
    b = 1
    dummy = torch.zeros(b, args.in_ch, args.image_size, args.image_size, device=device)
    plane = torch.zeros(b, 2, device=device)
    h = pool_bottleneck(model, dummy, plane, view=1)
    return {"bottleneck": int(h.shape[-1])}


def compute_head_in_dim(args: argparse.Namespace, dims: Dict[str, int]) -> int:
    mode = args.classification_mode
    fusion = args.fusion
    view_mode = args.view_mode
    d = dims["bottleneck"]

    if mode == "classification_bottleneck_concat":
        if view_mode != "two":
            return d
        return 2 * d

    if mode == "classification_default":
        if view_mode != "two":
            return d
        return 2 * d if fusion == "concat" else d

    raise ValueError(f"unsupported classification_mode: {mode}")

def build_head(
    args: argparse.Namespace,
    num_classes: int,
    head_in_dim: int,
    device: torch.device,
) -> ClassificationHead:
    hidden = args.clf_hidden_dim
    head = ClassificationHead(
        in_dim=head_in_dim,
        num_classes=num_classes,
        hidden_dim=hidden,
        dropout=args.clf_dropout,
        activation=args.clf_activation,
        use_layernorm=not getattr(args, "no_clf_layernorm", False),
    ).to(device)
    return head

def build_dataloaders(args: argparse.Namespace, device: torch.device) -> Tuple[DataLoader, DataLoader, List[str]]:
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

    class_names = normalize_class_names(args, label_names)
    num_classes = len(class_names)

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
    return train_loader, test_loader, class_names

def load_checkpoint(
    args: argparse.Namespace,
    model: nn.Module,
    head: ClassificationHead,
    device: torch.device,
) -> Dict[str, object]:
    if not args.resume_ckpt or args.ckpt_load_mode == "none":
        return {"start_epoch": 1, "best_f1": -1.0}

    ckpt_path = Path(args.resume_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"resume_ckpt not found: {ckpt_path}")

    if args.ckpt_load_mode == "encoder_only":
        _ = load_checkpoint_weights_filtered(
            ckpt_path=ckpt_path,
            device=device,
            model=model,
            include_prefixes=model.encoder_state_dict_prefixes(),
            exclude_prefixes=(),
        )
        print("[ckpt] loaded encoder_only weights from", ckpt_path)
        return {"start_epoch": 1, "best_f1": -1.0}

    obj = torch.load(ckpt_path, map_location=device)
    if "encoder_state" in obj:
        model.load_state_dict(obj["encoder_state"], strict=False)
        if "head_state" in obj:
            head.load_state_dict(obj["head_state"], strict=False)
        print("[ckpt] loaded structured checkpoint", ckpt_path)
        return {
            "start_epoch": int(obj.get("epoch", 0)) + 1,
            "best_f1": float(obj.get("best_f1", -1.0)),
        }

    if "model" in obj:
        model.load_state_dict(obj["model"], strict=False)
        print("[ckpt] loaded model state from", ckpt_path)
    return {"start_epoch": 1, "best_f1": -1.0}

def maybe_freeze_after_load(args: argparse.Namespace, model: nn.Module) -> None:
    if args.ckpt_load_mode != "full":
        return
    for p in model.parameters():
        p.requires_grad = False
    print("[freeze] all model params frozen after full checkpoint load")

def fuse_features(h1: torch.Tensor, h2: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "avg":
        return 0.5 * (h1 + h2)
    if mode == "max":
        return torch.max(h1, h2)
    if mode == "concat":
        return torch.cat([h1, h2], dim=1)
    raise ValueError("fusion mode must be avg, max, or concat")

def compute_metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        acc = float((pred == target).sum().item()) / float(max(target.numel(), 1))
    return {"acc": acc}

def extract_fused_feature(
    cfg: ClsConfig,
    model: nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
    plane: torch.Tensor,
) -> torch.Tensor:
    mode = cfg.classification_mode

    if mode == "classification_bottleneck_concat":
        if cfg.view_mode != "two":
            view = 1 if cfg.view_mode == "one_v1" else 2
            x = x1 if view == 1 else x2
            return pool_bottleneck(model, x, plane, view=view)
        return encode_dual_bottleneck(model, x1, x2, plane, mode="bottleneck_concat")

    if mode == "classification_default":
        if cfg.view_mode != "two":
            view = 1 if cfg.view_mode == "one_v1" else 2
            x = x1 if view == 1 else x2
            return pool_bottleneck(model, x, plane, view=view)
        h1 = pool_bottleneck(model, x1, plane, view=1)
        h2 = pool_bottleneck(model, x2, plane, view=2)
        return fuse_features(h1, h2, cfg.fusion)

    raise ValueError(f"unsupported classification_mode: {mode}")

def write_epoch_csv(
    path: Path,
    epoch: int,
    train_loss: float,
    train_f1: float,
    val_loss: float,
    val_f1: float,
) -> None:
    ensure_dir(path.parent)
    header = ["epoch", "train_loss", "train_f1_macro", "val_loss", "val_f1_macro"]
    row = [epoch, train_loss, train_f1, val_loss, val_f1]
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

def train_one_epoch(
    args: argparse.Namespace,
    cfg: ClsConfig,
    model: nn.Module,
    head: ClassificationHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    head.train()

    total = 0
    correct = 0
    loss_sum = 0.0
    all_true = []
    all_pred = []

    if cfg.loss_type == "focal":
        loss_fn = FocalLoss(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha, reduction="mean").to(device)
    elif cfg.loss_type == "wce":
        loss_fn = nn.CrossEntropyLoss(weight=cfg.ce_weights).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    for batch in loader:
        x1, x2, y = prepare_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
            plane = build_plane_one_hot("axial", x1.size(0), device)
            fused = extract_fused_feature(cfg, model, x1, x2, plane)
            logits = head(fused)

            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        loss_sum += float(loss.item()) * float(y.numel())
        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    y_true = torch.cat(all_true, dim=0).numpy() if all_true else []
    y_pred = torch.cat(all_pred, dim=0).numpy() if all_pred else []

    f1_macro = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0

    return {
        "loss": float(loss_sum) / float(max(total, 1)),
        "acc": float(correct) / float(max(total, 1)),
        "f1_macro": f1_macro,
        "y_true": y_true,
        "y_pred": y_pred,
    }

@torch.no_grad()
def validate_one_epoch(
    args: argparse.Namespace,
    cfg: ClsConfig,
    model: nn.Module,
    head: ClassificationHead,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    head.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    all_true = []
    all_pred = []

    if cfg.loss_type == "focal":
        loss_fn = FocalLoss(gamma=cfg.focal_gamma, alpha=cfg.focal_alpha, reduction="mean").to(device)
    elif cfg.loss_type == "wce":
        loss_fn = nn.CrossEntropyLoss(weight=cfg.ce_weights).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    for batch in loader:
        x1, x2, y = prepare_batch(batch, device)
        plane = build_plane_one_hot("axial", x1.size(0), device)

        fused = extract_fused_feature(cfg, model, x1, x2, plane)
        logits = head(fused)
        loss = loss_fn(logits, y)

        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        loss_sum += float(loss.item()) * float(y.numel())
        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    y_true = torch.cat(all_true, dim=0).numpy() if all_true else []
    y_pred = torch.cat(all_pred, dim=0).numpy() if all_pred else []

    f1_macro = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0

    return {
        "loss": float(loss_sum) / float(max(total, 1)),
        "acc": float(correct) / float(max(total, 1)),
        "f1_macro": f1_macro,
        "y_true": y_true,
        "y_pred": y_pred,
    }

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

def normalize_class_names(args: argparse.Namespace, label_names: List[str]) -> List[str]:
    detected = [normalize_label_name(name) for name in label_names]
    if getattr(args, "label_order", ""):
        raw = [part.strip() for part in str(args.label_order).split(",") if part.strip()]
        expected_order = [normalize_label_name(name) for name in raw]
    else:
        expected_order = [
            "mild demented",
            "moderate demented",
            "non-demented",
            "very mild demented",
        ]

    if len(detected) != len(expected_order):
        raise RuntimeError(f"label count mismatch: detected={detected} expected={expected_order}")
    if detected != expected_order:
        raise RuntimeError(f"label order mismatch: detected={detected} expected={expected_order}")

    if getattr(args, "label_order", ""):
        return [name.strip() for name in str(args.label_order).split(",") if name.strip()]
    return [
        "Mild Demented",
        "Moderate Demented",
        "Non-Demented",
        "Very Mild Demented",
    ]

def build_loss_and_cfg(
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int,
    class_names: List[str],
) -> ClsConfig:
    focal_alpha = parse_focal_alpha(args.focal_alpha, num_classes) if args.loss_type == "focal" else None
    ce_weights = parse_ce_class_weights(args.ce_class_weights, num_classes).to(device) if args.loss_type == "wce" else None

    return ClsConfig(
        num_classes=num_classes,
        class_names=class_names,
        classification_mode=args.classification_mode,
        fusion=args.fusion,
        clf_hidden_dim=args.clf_hidden_dim,
        clf_dropout=args.clf_dropout,
        clf_activation=args.clf_activation,
        clf_layernorm=not getattr(args, "no_clf_layernorm", False),
        amp=args.amp,
        device=device,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha,
        ce_weights=ce_weights,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        view_mode=args.view_mode,
        head_in_dim=args.head_in_dim,
        backbone="vae" if getattr(args, "vae", False) else "mae",
    )

def save_checkpoint(
    path: Path,
    epoch: int,
    best_f1: float,
    model: nn.Module,
    head: ClassificationHead,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
) -> None:
    obj = {
        "epoch": epoch,
        "best_f1": float(best_f1),
        "encoder_state": model.state_dict(),
        "head_state": head.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }
    torch.save(obj, path)

def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(bool(args.cpu))

    if not hasattr(args, "fusion") and hasattr(args, "fusion_mode"):
        args.fusion = args.fusion_mode
    args.fusion = getattr(args, "fusion", "avg")
    if not hasattr(args, "clf_dropout"):
        args.clf_dropout = getattr(args, "dropout", 0.1)
    if not hasattr(args, "clf_hidden_dim"):
        args.clf_hidden_dim = 256
    if not hasattr(args, "clf_activation"):
        args.clf_activation = "gelu"
    if not hasattr(args, "classification_mode"):
        args.classification_mode = "classification_default"

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    train_loader, test_loader, class_names = build_dataloaders(args, device)
    num_classes = len(class_names)

    model = build_model(args, device)
    dims = infer_feature_dims(model, args, device)
    args.head_in_dim = compute_head_in_dim(args, dims)
    cfg = build_loss_and_cfg(args, device, num_classes, class_names)
    head = build_head(args, num_classes, args.head_in_dim, device)

    ckpt_info = load_checkpoint(args, model, head, device)
    maybe_freeze_after_load(args, model)

    params = list(model.parameters()) + list(head.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_f1 = ckpt_info.get("best_f1", -1.0)
    best_epoch = -1
    start_epoch = ckpt_info.get("start_epoch", 1)

    print("[label_map]")
    for i, name in enumerate(class_names):
        print(f"  {i} -> {name}")

    for epoch in range(start_epoch, args.epochs + 1):
        freeze_n = int(getattr(args, "freeze_encoder_epochs", 0))
        encoder_trainable = not (epoch <= freeze_n)
        model.set_encoder_trainable(trainable=encoder_trainable)
        if epoch == 1 and freeze_n > 0:
            print(f"[train] freeze encoder for first {freeze_n} epochs")
        if epoch == freeze_n + 1 and freeze_n > 0:
            print("[train] encoder unfrozen")

        train_metrics = train_one_epoch(args, cfg, model, head, train_loader, optimizer, scaler, device)
        val_metrics = validate_one_epoch(args, cfg, model, head, test_loader, device)

        write_epoch_csv(out_dir / "epoch_log.csv", epoch, train_metrics["loss"], train_metrics["f1_macro"], val_metrics["loss"], val_metrics["f1_macro"])

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_f1={train_metrics['f1_macro']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )

        train_cm = confusion_matrix(train_metrics["y_true"], train_metrics["y_pred"], labels=list(range(num_classes)))
        val_cm = confusion_matrix(val_metrics["y_true"], val_metrics["y_pred"], labels=list(range(num_classes)))
        print(f"[epoch {epoch:03d}] train_cm:\n{train_cm}")
        print(f"[epoch {epoch:03d}] val_cm:\n{val_cm}")

        ckpt_dir = out_dir / "checkpoints"
        ensure_dir(ckpt_dir)
        save_checkpoint(ckpt_dir / "latest_cls.pt", epoch, best_f1, model, head, optimizer, scaler, args)

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = float(val_metrics["f1_macro"])
            best_epoch = epoch
            save_checkpoint(ckpt_dir / "best_cls.pt", epoch, best_f1, model, head, optimizer, scaler, args)
            print(f"[best_f1] epoch={epoch:03d} val_f1={best_f1:.4f}")

    record = {
        "best_epoch": int(best_epoch if best_epoch != -1 else epoch),
        "best_score": float(best_f1),
    }
    save_json(out_dir / "metrics" / "single_split_metrics.json", record)
    print("[done] best_f1:", best_f1)
