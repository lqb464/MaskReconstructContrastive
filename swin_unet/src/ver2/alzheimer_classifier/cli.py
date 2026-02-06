from __future__ import annotations

import argparse

from .train import run


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="runs/alzheimer_cls")
    p.add_argument("--resume-ckpt", type=str, default="")
    p.add_argument(
        "--ckpt-load-mode",
        type=str,
        default="encoder_only",
        choices=["none", "full", "encoder_only"],
    )
    p.add_argument(
        "--freeze-recon",
        action="store_true",
        help="When ckpt-load-mode=full, freeze entire encoder/decoder after loading",
    )
    p.add_argument(
        "--freeze-decoder-recon",
        action="store_true",
        help="When ckpt-load-mode=full, freeze decoder/reconstruction blocks after loading",
    )
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--in-ch", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--enc-depths", type=int, nargs=4, default=[2, 2, 6, 2])
    p.add_argument("--dec-depths", type=int, nargs=3, default=[6, 2, 2])
    p.add_argument("--num-heads", type=int, nargs=4, default=[3, 6, 12, 24])
    p.add_argument("--window-size", type=int, default=7)

    p.add_argument("--bottleneck-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)

    p.add_argument("--plane-inject-method", type=str, default="film", choices=["film", "add"])

    p.add_argument("--enable_saca", action="store_true")
    p.add_argument(
        "--saca_position",
        type=str,
        default="after_stage1",
        choices=["after_patch_embed", "after_stage0", "after_merge0", "after_stage1"],
    )
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

    p.add_argument(
        "--classification_mode",
        type=str,
        default="classification_default",
        choices=[
            "classification_default",
            "classification_bottleneck_concat",
            "classification_stage2_fusion",
            "classification_multiscale",
        ],
        help="Select classification pipeline variant",
    )
    p.add_argument(
        "--feature_level",
        type=str,
        default="bottleneck",
        choices=["stage1", "stage2", "bottleneck"],
        help="Feature level for classification pooling",
    )
    p.add_argument(
        "--fusion",
        type=str,
        default="avg",
        choices=["avg", "concat", "max"],
        help="Fusion strategy for dual-view features",
    )
    p.add_argument("--clf_hidden_dim", type=int, default=256, help="Hidden dim for 2-layer classification head")
    p.add_argument("--clf_dropout", type=float, default=0.1, help="Dropout after first layer in classification head")
    p.add_argument("--clf_activation", type=str, default="gelu", choices=["gelu", "relu"], help="Activation in classification head")
    p.add_argument("--no-clf-layernorm", action="store_true", help="Disable LayerNorm in classification head hidden")

    p.add_argument("--freeze_encoder_epochs", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.0)  # legacy head dropout (fallback for clf_dropout)

    p.add_argument(
        "--view_mode",
        type=str,
        default="two",
        choices=["two", "one_v1", "one_v2"],
        help="two=use both views, one_v1=use view1 only, one_v2=use view2 only",
    )
    p.add_argument(
        "--label_order",
        type=str,
        default="",
        help="Optional comma-separated label order override (e.g., 'Non_Demented,Very_Mild_Demented,Mild_Demented,Moderate_Demented')",
    )
    p.add_argument("--loss_type", type=str, default="focal", choices=["focal", "wce", "ce"])
    p.add_argument("--ce_class_weights", type=str, default="")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=str, default="")

    return p
