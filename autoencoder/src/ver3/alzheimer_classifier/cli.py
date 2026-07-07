from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Alzheimer classifier on MAE/VAE encoder (bottleneck features only).",
    )
    p.add_argument("--out_dir", type=str, default="runs/alzheimer_cls")
    p.add_argument("--resume-ckpt", type=str, default="")
    p.add_argument(
        "--ckpt-load-mode",
        type=str,
        default="encoder_only",
        choices=["none", "full", "encoder_only"],
    )
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--in-ch", type=int, default=1)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--mae-enc-depth", type=int, default=4)
    p.add_argument("--mae-dec-depth", type=int, default=2)
    p.add_argument("--use-gn", action="store_true")

    backbone = p.add_mutually_exclusive_group()
    backbone.add_argument("--mae", action="store_true", help="Use MAE backbone (default).")
    backbone.add_argument("--vae", action="store_true", help="Use VAE backbone.")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--classification_mode",
        type=str,
        default="classification_default",
        choices=[
            "classification_default",
            "classification_bottleneck_concat",
        ],
        help="Bottleneck-only feature modes for MAE/VAE",
    )
    p.add_argument(
        "--fusion",
        type=str,
        default="avg",
        choices=["avg", "concat", "max"],
        help="Fusion strategy for dual-view bottleneck features",
    )
    p.add_argument("--clf_hidden_dim", type=int, default=256)
    p.add_argument("--clf_dropout", type=float, default=0.1)
    p.add_argument("--clf_activation", type=str, default="gelu", choices=["gelu", "relu"])
    p.add_argument("--no-clf-layernorm", action="store_true")

    p.add_argument("--freeze_encoder_epochs", type=int, default=0)

    p.add_argument(
        "--view_mode",
        type=str,
        default="two",
        choices=["two", "one_v1", "one_v2"],
    )
    p.add_argument("--label_order", type=str, default="")
    p.add_argument("--loss_type", type=str, default="focal", choices=["focal", "wce", "ce"])
    p.add_argument("--ce_class_weights", type=str, default="")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=str, default="")
    return p


__all__ = ["build_argparser"]
