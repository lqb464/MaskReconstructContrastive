# =============================================
# File: train_phase1_swin.py
# Phase 1 SSL training:
#   - Dual-view: masked original + flipped (copy mask, no flip)
#   - Loss: reconstruction + InfoNCE
# Backbone: Swin Transformer encoder + CNN decoder
# =============================================

from __future__ import annotations

import os
import time
import math
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_png_folder import PngFolderDataset
from masking import MaskSpec, sample_masks_anti_mirror
from model_swin_unet_ssl import SwinUNetSSL


# -------------------------------------------------
# Seed
# -------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------
# Meta vector: Option 1 (grid coordinate normalized + plane one-hot)
# -------------------------------------------------

def build_meta_vec(
    batch_size: int,
    plane: str,
    device: torch.device,
) -> torch.Tensor:
    """
    meta_vec = [x_norm, y_norm, onehot_plane(2)]
    - Option 1: x,y are normalized grid coordinates (no slice index assumed)
    - For a single 2D slice, we use fixed center coordinates (0,0) by default.
    """
    # fixed center coordinate for now
    x_norm = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
    y_norm = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)

    plane = str(plane).lower().strip()
    if plane == "axial":
        onehot = torch.tensor([1.0, 0.0], device=device).view(1, 2).repeat(batch_size, 1)
    elif plane == "coronal":
        onehot = torch.tensor([0.0, 1.0], device=device).view(1, 2).repeat(batch_size, 1)
    else:
        raise ValueError("plane must be 'axial' or 'coronal'")

    return torch.cat([x_norm, y_norm, onehot], dim=1)


# -------------------------------------------------
# Losses (reuse from codebase spirit)
# -------------------------------------------------

def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - target) * pixel_mask
    denom = pixel_mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def mixed_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: torch.Tensor,
    alpha_mask: float = 1.0,
    beta_unmask: float = 0.2,
) -> torch.Tensor:
    diff = torch.abs(pred - target)
    m = pixel_mask
    um = 1.0 - m

    masked_denom = m.sum().clamp(min=1.0)
    masked_l1 = (diff * m).sum() / masked_denom

    unmasked_denom = um.sum().clamp(min=1.0)
    unmasked_l1 = (diff * um).sum() / unmasked_denom

    return alpha_mask * masked_l1 + beta_unmask * unmasked_l1


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Standard SimCLR NT-Xent.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    z = F.normalize(z, dim=-1)

    sim = torch.matmul(z, z.t()) / temperature
    sim = sim.to(torch.float32)

    # mask self
    diag = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -float("inf"))

    # positives: i <-> i+B
    pos = torch.cat(
        [torch.arange(B, 2 * B, device=sim.device), torch.arange(0, B, device=sim.device)],
        dim=0,
    )
    loss = F.cross_entropy(sim, pos)
    return loss


# -------------------------------------------------
# Training
# -------------------------------------------------

def train_one_epoch(
    model: SwinUNetSSL,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    args,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_con = 0.0
    n_items = 0

    spec = MaskSpec(
        patch_size=args.patch_size,
        image_size=args.image_size,
        mask_ratio_side=args.mask_ratio,
    )

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)  # [B,1,256,256]
        B = x.size(0)

        # ---- meta vector (option 1) ----
        meta = build_meta_vec(B, args.plane, device)

        # ---- mask sampling (anti-mirror) ----
        pixel_mask = sample_masks_anti_mirror(B, spec, device)

        # ---- view 1: original masked ----
        x1_masked = x * (1.0 - pixel_mask)

        # ---- view 2: flip image, copy mask without flipping ----
        x2 = torch.flip(x, dims=[-1])  # horizontal flip
        x2_masked = x2 * (1.0 - pixel_mask)  # IMPORTANT: reuse mask1 directly

        # ---- forward recon ----
        recon1 = model.forward(x1_masked, meta_vec=meta)
        recon2 = model.forward(x2_masked, meta_vec=meta)

        if args.recon_loss == "masked_l1":
            loss_r1 = masked_l1_loss(recon1, x, pixel_mask)
            loss_r2 = masked_l1_loss(recon2, x2, pixel_mask)
        else:
            loss_r1 = mixed_l1_loss(recon1, x, pixel_mask, alpha_mask=1.0, beta_unmask=args.beta_unmask)
            loss_r2 = mixed_l1_loss(recon2, x2, pixel_mask, alpha_mask=1.0, beta_unmask=args.beta_unmask)

        loss_recon = 0.5 * (loss_r1 + loss_r2)

        # ---- contrastive embeddings ----
        z1, _ = model.encoder_embed(x1_masked, meta_vec=meta)
        z2, _ = model.encoder_embed(x2_masked, meta_vec=meta)
        loss_con = nt_xent_loss(z1, z2, temperature=args.temperature)

        loss = loss_recon + args.lambda_contrast * loss_con

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()

        total_loss += loss.item() * B
        total_recon += loss_recon.item() * B
        total_con += loss_con.item() * B
        n_items += B

    return {
        "loss": total_loss / max(n_items, 1),
        "loss_recon": total_recon / max(n_items, 1),
        "loss_contrast": total_con / max(n_items, 1),
    }


@torch.no_grad()
def validate_recon(
    model: SwinUNetSSL,
    loader: DataLoader,
    device: torch.device,
    args,
) -> float:
    model.eval()

    spec = MaskSpec(
        patch_size=args.patch_size,
        image_size=args.image_size,
        mask_ratio_side=args.mask_ratio,
    )

    losses = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        B = x.size(0)
        meta = build_meta_vec(B, args.plane, device)

        pixel_mask = sample_masks_anti_mirror(B, spec, device)
        x_masked = x * (1.0 - pixel_mask)
        recon = model.forward(x_masked, meta_vec=meta)

        if args.recon_loss == "masked_l1":
            loss = masked_l1_loss(recon, x, pixel_mask)
        else:
            loss = mixed_l1_loss(recon, x, pixel_mask, alpha_mask=1.0, beta_unmask=args.beta_unmask)

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def main():
    parser = argparse.ArgumentParser("Phase 1 SSL training with SwinUNet (dual-view)")

    # data
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--label", type=int, required=True, help="constant label from arg parser")
    parser.add_argument("--plane", type=str, default="axial", choices=["axial", "coronal"])
    parser.add_argument("--image-size", type=int, default=256)

    # mask
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.35)

    # model
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=128)

    # optimization
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # losses
    parser.add_argument("--recon-loss", type=str, default="mixed_l1", choices=["masked_l1", "mixed_l1"])
    parser.add_argument("--beta-unmask", type=float, default=0.2)
    parser.add_argument("--lambda-contrast", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.2)

    # misc
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="./runs_phase1_swin")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--save-every", type=int, default=1)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # dataset
    dataset = PngFolderDataset(
        root_dir=args.data_root,
        image_size=args.image_size,
        label=args.label,
    )

    # simple split: 95/5 train/val
    n = len(dataset)
    n_val = max(1, int(0.05 * n))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SwinUNetSSL(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=1,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        window_size=args.window_size,
        bottleneck_dim=args.bottleneck_dim,
        proj_dim=args.proj_dim,
        plane_dim=2,
    ).to(device)
    
    print("Dataset size:", len(dataset))
    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))


    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # output
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name if args.run_name else f"{ts}_img{args.image_size}_swin"
    out_dir = Path(args.out_dir) / run_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_loader, opt, device, args)
        val_recon = validate_recon(model, val_loader, device, args)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={stats['loss']:.4f} | "
            f"recon={stats['loss_recon']:.4f} | "
            f"con={stats['loss_contrast']:.4f} | "
            f"val_recon={val_recon:.4f}"
        )

        # save best
        if val_recon < best_val:
            best_val = val_recon
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val_recon": best_val,
                },
                str(ckpt_dir / "best.pt"),
            )

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_recon": val_recon,
                },
                str(ckpt_dir / f"epoch_{epoch:03d}.pt"),
            )

    print(f"Done. Best val recon: {best_val:.4f}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
