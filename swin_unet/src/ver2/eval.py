import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from config import load_config
from data import build_dataloader
from model import build_model, flip_lr
from metrics import MetricsAccumulator
from losses import masked_l1_loss, mixed_l1_loss, ssim_index
from visualization import visualize_once

# ============================================================
# SHARED LOSSES WITH TRAINER (ONLY CHANGE IN STAGE 3)
# ============================================================
try:
    # package-style run
    from phase1.training.recon_losses import (
        masked_bce_logits_weighted,
        mixed_bce_logits_weighted,
    )
except Exception:
    # legacy run inside src/phase1
    from training.recon_losses import (
        masked_bce_logits_weighted,
        mixed_bce_logits_weighted,
    )


# ============================================================
# Utilities (kept unchanged)
# ============================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def dataclass_from_dict(klass, dikt):
    try:
        fieldtypes = {f.name: f.type for f in klass.__dataclass_fields__.values()}
        return klass(**{f: dataclass_from_dict(fieldtypes[f], dikt[f]) for f in dikt})
    except Exception:
        return dikt


# ============================================================
# Main evaluation loop (UNCHANGED)
# ============================================================
@torch.no_grad()
def run_eval(
    model,
    loader,
    device,
    cfg,
    visualize=False,
    vis_dir=None,
):
    model.eval()

    meter = MetricsAccumulator()
    loss_recon_meter = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Eval"):
        x = batch["image"].to(device)

        if "plane_one_hot" in batch:
            plane = batch["plane_one_hot"].to(device)
        else:
            plane = torch.tensor([[0.0, 1.0]], device=device).repeat(x.size(0), 1)

        if cfg.mask.enable:
            from augmentation import sample_masks_anti_mirror
            pixel_mask = sample_masks_anti_mirror(
                x.shape,
                device=device,
                **cfg.mask.kwargs,
            )
        else:
            pixel_mask = None

        x_flip = flip_lr(x)

        recon_raw_orig, recon_raw_flip, *_ = model(
            x,
            plane,
            pixel_mask=pixel_mask,
            return_embeddings=False,
        )

        # ------------------------------------------------------------
        # Recon loss (logic unchanged, only loss source shared)
        # ------------------------------------------------------------
        if cfg.training.recon_loss_type == "weighted_bce_logits":
            if cfg.training.enable_masked_loss:
                loss_orig = masked_bce_logits_weighted(
                    recon_raw_orig, x, pixel_mask
                )
                loss_flip = masked_bce_logits_weighted(
                    recon_raw_flip, x_flip, pixel_mask
                )
            else:
                loss_orig = mixed_bce_logits_weighted(
                    recon_raw_orig, x, pixel_mask
                )
                loss_flip = mixed_bce_logits_weighted(
                    recon_raw_flip, x_flip, pixel_mask
                )
        else:
            recon_orig = torch.sigmoid(recon_raw_orig).clamp(0, 1)
            recon_flip = torch.sigmoid(recon_raw_flip).clamp(0, 1)

            if cfg.training.enable_masked_loss:
                loss_orig = masked_l1_loss(
                    recon_orig, x, pixel_mask
                )
                loss_flip = masked_l1_loss(
                    recon_flip, x_flip, pixel_mask
                )
            else:
                loss_orig = mixed_l1_loss(
                    recon_orig, x, pixel_mask
                )
                loss_flip = mixed_l1_loss(
                    recon_flip, x_flip, pixel_mask
                )

        loss_recon = 0.5 * (loss_orig + loss_flip)
        loss_recon_meter += loss_recon.item()
        n_batches += 1

        # ------------------------------------------------------------
        # Metrics (UNCHANGED)
        # ------------------------------------------------------------
        recon_orig_m = torch.sigmoid(recon_raw_orig).clamp(0, 1)
        recon_flip_m = torch.sigmoid(recon_raw_flip).clamp(0, 1)

        diff_orig = (recon_orig_m - x).abs()
        diff_flip = (recon_flip_m - x_flip).abs()
        diff_total = 0.5 * (diff_orig + diff_flip)

        ssim_orig = ssim_index(recon_orig_m, x)
        ssim_flip = ssim_index(recon_flip_m, x_flip)
        ssim_sum = 0.5 * (ssim_orig + ssim_flip)

        meter.update(diff_total, pixel_mask, ssim_sum=ssim_sum)

        if visualize:
            visualize_once(
                x,
                recon_orig_m,
                recon_flip_m,
                pixel_mask,
                vis_dir,
            )

    return {
        "loss_recon": loss_recon_meter / max(n_batches, 1),
        **meter.compute(),
    }


# ============================================================
# Entry
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--out_dir", default="eval_outputs")

    args = parser.parse_args()

    cfg = load_config(args.config)

    device = get_device()

    loader = build_dataloader(
        cfg,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    ensure_dir(args.out_dir)

    results = run_eval(
        model,
        loader,
        device,
        cfg,
        visualize=args.visualize,
        vis_dir=args.out_dir,
    )

    print("Eval results:")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
