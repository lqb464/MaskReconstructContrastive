# =============================================
# File: eval.py
# Eval helpers for Phase 1 SwinUNet SSL (dual-view)
# - Reconstruction eval (masked region)
# - tSNE on embeddings
# =============================================
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Your refactored modules ----
from dataset_png_folder import PngFolderDataset
from masking import sample_anti_mirror_pixel_mask
from model_swin import SwinUNetSSLPhase1


# -----------------------------
# Meta vector: [x, y, onehot_plane]
# x,y are placeholders here because Swin has relative position bias internally.
# If you later implement explicit (x,y) conditioning, update this.
# axial    -> [0, 0, 1, 0]
# coronal  -> [0, 0, 0, 1]
# -----------------------------
def build_meta_vec(batch_size: int, plane: str, device: torch.device) -> torch.Tensor:
    plane = plane.lower()
    if plane not in ["axial", "coronal"]:
        raise ValueError(f"plane must be axial or coronal, got: {plane}")
    if plane == "axial":
        v = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
    else:
        v = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    return v.unsqueeze(0).repeat(batch_size, 1)


@dataclass
class EvalSpec:
    image_size: int = 256
    patch_size: int = 16
    mask_ratio: float = 0.35


@torch.no_grad()
def evaluate_recon_phase1(
    model: SwinUNetSSLPhase1,
    loader: DataLoader,
    device: torch.device,
    spec: EvalSpec,
    plane: str,
) -> Dict[str, float]:
    model.eval()

    num = 0.0
    den = 0.0

    num1 = 0.0
    den1 = 0.0

    num2 = 0.0
    den2 = 0.0

    for batch in loader:
        # dataset returns dict-like
        x = batch["input"].to(device, non_blocking=True)  # [B,1,256,256]
        B = x.size(0)

        # flip view 2
        x_flip = torch.flip(x, dims=[-1])

        # anti-mirror mask from view 1 (pixel mask), do NOT flip it
        pixel_mask = sample_anti_mirror_pixel_mask(
            batch_size=B,
            image_size=spec.image_size,
            patch_size=spec.patch_size,
            mask_ratio_side=spec.mask_ratio,
            device=device,
        )  # [B,1,H,W] with 1 on masked pixels

        # apply same mask to both views
        x1_masked = x * (1.0 - pixel_mask)
        x2_masked = x_flip * (1.0 - pixel_mask)

        meta = build_meta_vec(B, plane=plane, device=device)

        # forward: should return recon1, recon2 (phase1)
        # If your forward returns extra outputs, keep the first two.
        out = model.forward(
            x1_masked,
            x2_masked,
            pixel_mask=pixel_mask,
            meta=meta,
        )

        if isinstance(out, (tuple, list)) and len(out) >= 2:
            recon1, recon2 = out[0], out[1]
        else:
            raise RuntimeError("model.forward must return (recon1, recon2, ...) at minimum")

        # masked region L1 for each branch
        diff1 = torch.abs(recon1 - x) * pixel_mask
        diff2 = torch.abs(recon2 - x_flip) * pixel_mask

        n1 = diff1.sum().item()
        n2 = diff2.sum().item()
        d = pixel_mask.sum().item()

        num1 += n1
        num2 += n2
        den1 += d
        den2 += d

        num += (n1 + n2)
        den += (d + d)

    eps = 1e-8
    return {
        "recon_masked_mean": float(num / (den + eps)),
        "recon_masked_view1": float(num1 / (den1 + eps)),
        "recon_masked_view2": float(num2 / (den2 + eps)),
    }


@torch.no_grad()
def collect_embeddings(
    model: SwinUNetSSLPhase1,
    loader: DataLoader,
    device: torch.device,
    plane: str,
    max_items: int,
    mode: str = "bottleneck",
    view: str = "view1",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()

    embs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    count = 0

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch.get("label", None)
        if y is None:
            y = torch.zeros(x.size(0), dtype=torch.long)
        y = y.cpu().numpy()

        if view == "view2":
            x = torch.flip(x, dims=[-1])

        B = x.size(0)
        meta = build_meta_vec(B, plane=plane, device=device)

        # encoder_embed should return (z, h) similar to previous code style
        z, h = model.encoder_embed(x, meta=meta, mode=mode)

        # use h for tsne, normalized
        h = F.normalize(h, dim=-1).detach().cpu().numpy()

        embs.append(h)
        labels.append(y)

        count += B
        if count >= max_items:
            break

    if not embs:
        return None, None

    X = np.concatenate(embs, axis=0)[:max_items]
    Y = np.concatenate(labels, axis=0)[:max_items]
    return X, Y


@torch.no_grad()
def run_tsne(
    model: SwinUNetSSLPhase1,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
    plane: str,
    max_items: int = 1000,
    mode: str = "bottleneck",
    view: str = "view1",
):
    X, y = collect_embeddings(
        model=model,
        loader=loader,
        device=device,
        plane=plane,
        max_items=max_items,
        mode=mode,
        view=view,
    )
    if X is None:
        return

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    X2 = tsne.fit_transform(X)

    plt.figure(figsize=(6, 6))
    uniq = sorted(set(list(y)))
    for lbl in uniq:
        m = (y == lbl)
        plt.scatter(X2[m, 0], X2[m, 1], s=10, label=str(lbl), alpha=0.9)

    plt.legend(loc="best", fontsize=8, frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _load_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[SwinUNetSSLPhase1, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("args", {})

    model = SwinUNetSSLPhase1(
        img_size=int(cfg.get("image_size", 256)),
        patch_size=int(cfg.get("patch_size", 16)),
        in_chans=1,
        embed_dim=int(cfg.get("embed_dim", 96)),
        depths=tuple(cfg.get("depths", (2, 2, 6, 2))),
        num_heads=tuple(cfg.get("num_heads", (3, 6, 12, 24))),
        window_size=int(cfg.get("window_size", 8)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        proj_dim=int(cfg.get("proj_dim", 128)),
        meta_dim=int(cfg.get("meta_dim", 4)),
        decoder_channels=tuple(cfg.get("decoder_channels", (256, 128, 64, 32))),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


def build_argparser():
    p = argparse.ArgumentParser("Eval Phase 1 SwinUNet SSL (dual-view)")
    p.add_argument("--data-root", type=str, required=True, help="root folder with many subfolders containing .png")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--plane", type=str, required=True, choices=["axial", "coronal"])

    # label is passed from user input (as you requested)
    p.add_argument("--label", type=int, default=0, help="label id for all samples if dataset has no labels")

    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--mask-ratio", type=float, default=0.35)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--out-dir", type=str, default="runs_eval")

    # optional tsne
    p.add_argument("--tsne", action="store_true")
    p.add_argument("--tsne-max-items", type=int, default=1000)
    p.add_argument("--tsne-mode", type=str, default="bottleneck", choices=["bottleneck", "s4", "multiscale"])
    p.add_argument("--tsne-view", type=str, default="view1", choices=["view1", "view2"])

    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = _load_from_ckpt(args.ckpt, device)

    # dataset
    ds = PngFolderDataset(
        root_dir=args.data_root,
        image_size=args.image_size,
        label=args.label,
        return_dict=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    spec = EvalSpec(
        image_size=args.image_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    metrics = evaluate_recon_phase1(
        model=model,
        loader=loader,
        device=device,
        spec=spec,
        plane=args.plane,
    )

    print("Reconstruction (masked region) metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    if args.tsne:
        out_path = os.path.join(
            args.out_dir,
            f"tsne_{args.plane}_{args.tsne_view}_{args.tsne_mode}.png",
        )
        run_tsne(
            model=model,
            loader=loader,
            device=device,
            out_path=out_path,
            plane=args.plane,
            max_items=args.tsne_max_items,
            mode=args.tsne_mode,
            view=args.tsne_view,
        )
        print(f"Saved tSNE: {out_path}")


if __name__ == "__main__":
    main()
