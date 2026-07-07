from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from .base import (
    ContrastiveLossConfig,
    DataConfig,
    LoggingConfig,
    MaskConfig,
    TrainingConfig,
)


@dataclass
class ModelConfig:
    """Model architecture configuration (MAE / VAE dual-view SSL)."""
    backbone: str = "mae"
    in_ch: int = 1

    base_ch: int = 32
    use_gn: bool = False

    patch_size: int = 16
    embed_dim: int = 256
    mae_enc_depth: int = 4
    mae_dec_depth: int = 2

    latent_dim: int = 256


@dataclass
class AETrainingConfig(TrainingConfig):
    """Training config extended with VAE KL weight."""
    lambda_kl: float = 1e-4


@dataclass
class ExperimentConfig:
    """Complete experiment configuration for autoencoder ver2."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: AETrainingConfig = field(default_factory=AETrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    contrast_loss: ContrastiveLossConfig = field(default_factory=ContrastiveLossConfig)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        backbone = str(getattr(args, "backbone", "mae")).lower()
        if getattr(args, "mae", False):
            backbone = "mae"
        elif getattr(args, "vae", False):
            backbone = "vae"

        vis_mask_seed = int(getattr(args, "vis_mask_seed", -1))
        if vis_mask_seed < 0:
            vis_mask_seed = int(args.seed)

        cfg = cls(
            model=ModelConfig(
                backbone=backbone,
                in_ch=args.in_ch,
                base_ch=getattr(args, "base_ch", 32),
                use_gn=getattr(args, "use_gn", False),
                patch_size=args.patch_size,
                embed_dim=getattr(args, "embed_dim", 256),
                mae_enc_depth=getattr(args, "mae_enc_depth", 4),
                mae_dec_depth=getattr(args, "mae_dec_depth", 2),
                latent_dim=getattr(args, "latent_dim", 256),
            ),
            training=AETrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lambda_recon=args.lambda_recon,
                lambda_contrast=args.lambda_contrast,
                lambda_kl=getattr(args, "lambda_kl", 1e-4),
                temperature=args.temperature,
                seed=args.seed,
                amp=args.amp,
                cpu=args.cpu,
                enable_reconstruct=args.enable_reconstruct,
                enable_contrastive=args.enable_contrastive,
                single_view=args.single_view,
                resume_ckpt=args.resume_ckpt,
                ckpt_load_mode=args.ckpt_load_mode,
                freeze_encoder_epochs=args.freeze_encoder_epochs,
                reset_contrastive_proj_head=args.reset_contrastive_proj_head,
                enable_masked_loss=args.enable_masked_loss,
                aug_p_noise=args.aug_p_noise,
                aug_p_jitter=args.aug_p_jitter,
                aug_p_blur=args.aug_p_blur,
                aug_noise_std=args.aug_noise_std,
                aug_jitter_strength=args.aug_jitter_strength,
                aug_blur_kernel=args.aug_blur_kernel,
                recon_loss=args.recon_loss,
                fg_eps=args.fg_eps,
                fg_weight=args.fg_weight,
                dice_loss_weight=getattr(args, "dice_loss_weight", 0.0),
                dice_mode=getattr(args, "dice_mode", "fg"),
                dice_smooth=getattr(args, "dice_smooth", 1e-6),
                grad_clip=getattr(args, "grad_clip", 1.0),
                warmup_epochs=getattr(args, "warmup_epochs", 5),
                min_lr=getattr(args, "min_lr", 1e-6),
                torch_compile=getattr(args, "torch_compile", False),
            ),
            data=DataConfig(
                data_root=args.data_root,
                preprocessed_dir=getattr(args, "preprocessed_dir", ""),
                train_mod=args.train_mod,
                image_size=args.image_size,
                plane=args.plane,
                skip_resize_in_loader=getattr(args, "skip_resize_in_loader", False),
                label_csv=args.label_csv,
                label_path_col=args.label_path_col,
                label_col=args.label_col,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                drop_last=args.drop_last,
                split_test=args.split_test,
            ),
            mask=MaskConfig(
                patch_size=args.patch_size,
                mask_ratio_side=args.mask_ratio,
                image_size=args.image_size,
                enable_masking=args.enable_masking,
                mae_mask_ratio=getattr(args, "mae_mask_ratio", 0.75),
            ),
            logging=LoggingConfig(
                out_dir=args.out_dir,
                run_name=args.run_name,
                ckpt_dir=args.ckpt_dir,
                vis_every=args.vis_every,
                save_latest_every=args.save_latest_every,
                save_best_after_epoch=args.save_best_after_epoch,
                save_best_every=args.save_best_every,
                vis_batch_index=getattr(args, "vis_batch_index", 0),
                vis_mask_seed=vis_mask_seed,
                vis_manifest=getattr(args, "vis_manifest", True),
                enable_tsne=args.enable_tsne,
                tsne_only_if_labeled=args.tsne_only_if_labeled,
                tsne_every=args.tsne_every,
                tsne_max_items=args.tsne_max_items,
            ),
            contrast_loss=ContrastiveLossConfig(
                contrastive_loss_type=args.contrastive_loss_type,
                contrastive_position=args.contrastive_position,
                vicreg_invariance_weight=args.vicreg_invariance_weight,
                vicreg_variance_weight=args.vicreg_variance_weight,
                vicreg_covariance_weight=args.vicreg_covariance_weight,
                vicreg_variance_eps=args.vicreg_variance_eps,
                vicreg_target_std=args.vicreg_target_std,
            ),
        )
        return cfg


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("MAE / VAE Dual View SSL v2 (brain MRI)")

    p.add_argument("--data-root", type=str, required=True, help="Root folder containing subfolders of images")
    p.add_argument("--preprocessed-dir", type=str, default="", help="Optional root/folder of offline preprocessed data.")
    p.add_argument(
        "--train-mod",
        type=float,
        default=1.0,
        help=(
            "Train subsampling factor (>=1). "
            "Integer k keeps indices 0,k,2k,... ; "
            "float (e.g. 2.5) keeps ~1/2.5 samples with deterministic interleaving."
        ),
    )
    p.add_argument("--image-size", type=int, default=192)
    p.add_argument("--plane", type=str, default="axial", choices=["axial", "coronal", "auto"])
    p.add_argument("--skip-resize-in-loader", action="store_true", help="Skip resize work in dataset loader.")
    p.add_argument("--no-skip-resize-in-loader", dest="skip_resize_in_loader", action="store_false")
    p.set_defaults(skip_resize_in_loader=False)

    p.add_argument("--label-csv", type=str, default="", help="CSV mapping image_path -> label (optional)")
    p.add_argument("--label-path-col", type=str, default="image_path")
    p.add_argument("--label-col", type=str, default="label")

    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.0)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=True)
    p.add_argument("--drop-last", action="store_true")
    p.add_argument("--split_test", action="store_true")
    p.add_argument("--no-drop-last", dest="drop_last", action="store_false")
    p.set_defaults(drop_last=True)

    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument(
        "--mask-ratio",
        type=float,
        default=0.35,
        help="Per-hemisphere patch mask ratio for MAE (anti-mirror, same style as swin_unet).",
    )
    p.add_argument(
        "--mae-mask-ratio",
        type=float,
        default=0.75,
        help="Deprecated in v2; MAE uses --mask-ratio hemisphere masking instead.",
    )

    p.add_argument("--enable-masking", action="store_true")
    p.add_argument("--disable-masking", dest="enable_masking", action="store_false")
    p.set_defaults(enable_masking=True)

    backbone = p.add_mutually_exclusive_group()
    backbone.add_argument("--backbone", type=str, default="mae", choices=["mae", "vae"])
    backbone.add_argument("--mae", action="store_true", help="Use Masked AutoEncoder backbone (default).")
    backbone.add_argument("--vae", action="store_true", help="Use Variational AutoEncoder backbone.")

    p.add_argument("--in-ch", type=int, default=1)
    p.add_argument("--base-ch", type=int, default=32, help="Base channels for VAE conv AE backbone.")
    p.add_argument("--use-gn", action="store_true", help="Use GroupNorm in conv blocks (VAE).")
    p.add_argument("--embed-dim", type=int, default=256, help="Token embedding dim for MAE.")
    p.add_argument("--mae-enc-depth", type=int, default=4, help="Transformer encoder depth for MAE.")
    p.add_argument("--mae-dec-depth", type=int, default=2, help="Transformer decoder depth for MAE.")
    p.add_argument("--latent-dim", type=int, default=256, help="Latent dimension for VAE.")

    p.add_argument("--enable-reconstruct", action="store_true")
    p.add_argument("--disable-reconstruct", dest="enable_reconstruct", action="store_false")
    p.set_defaults(enable_reconstruct=True)

    p.add_argument("--enable-contrastive", action="store_true")
    p.add_argument("--disable-contrastive", dest="enable_contrastive", action="store_false")
    p.set_defaults(enable_contrastive=False)

    p.add_argument("--single-view", action="store_true")
    p.add_argument("--dual-view", dest="single_view", action="store_false")
    p.set_defaults(single_view=False)

    p.add_argument("--enable-masked-loss", action="store_true", help="Use masked-only loss (default for MAE in main.py)")
    p.add_argument("--enable-mixed-loss", dest="enable_masked_loss", action="store_false")
    p.set_defaults(enable_masked_loss=True)

    p.add_argument(
        "--recon-loss",
        type=str,
        default="weighted_bce_logits",
        choices=["weighted_bce_logits", "l1_sigmoid"],
    )
    p.add_argument("--fg-eps", type=float, default=0.02)
    p.add_argument("--fg-weight", type=float, default=10.0)

    p.add_argument("--dice-loss-weight", type=float, default=0.0)
    p.add_argument("--dice-mode", type=str, default="fg", choices=["total", "fg"])
    p.add_argument("--dice-smooth", type=float, default=1e-6)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-contrast", type=float, default=0.0)
    p.add_argument("--lambda-kl", type=float, default=1e-4, help="KL divergence weight for VAE backbone.")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--torch-compile", dest="torch_compile", action="store_true")
    p.add_argument("--no-torch-compile", dest="torch_compile", action="store_false")
    p.set_defaults(torch_compile=False)

    p.add_argument("--resume-ckpt", type=str, default="")
    p.add_argument(
        "--ckpt-load-mode",
        type=str,
        default="none",
        choices=["none", "full", "encoder_only"],
    )
    p.add_argument("--freeze-encoder-epochs", type=int, default=0)
    p.add_argument("--reset-proj-head", action="store_true")
    p.add_argument("--no-reset-proj-head", dest="reset_contrastive_proj_head", action="store_false")
    p.set_defaults(reset_contrastive_proj_head=True)

    p.add_argument("--aug-p-noise", type=float, default=0.7)
    p.add_argument("--aug-p-jitter", type=float, default=0.7)
    p.add_argument("--aug-p-blur", type=float, default=0.2)
    p.add_argument("--aug-noise-std", type=float, default=0.02)
    p.add_argument("--aug-jitter-strength", type=float, default=0.1)
    p.add_argument("--aug-blur-kernel", type=int, default=3)

    p.add_argument("--out-dir", type=str, default="runs_ssl_ae_v2")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--ckpt-dir", type=str, default="")
    p.add_argument("--vis-every", type=int, default=20)
    p.add_argument("--vis-n-results", type=int, default=8)
    p.add_argument("--vis-batch-index", type=int, default=0)
    p.add_argument("--vis-mask-seed", type=int, default=-1, help="RNG seed for viz masks; default = --seed")
    p.add_argument("--vis-manifest", action="store_true")
    p.add_argument("--no-vis-manifest", dest="vis_manifest", action="store_false")
    p.set_defaults(vis_manifest=True)
    p.add_argument("--save-latest-every", type=int, default=1)
    p.add_argument("--save-best-after-epoch", type=int, default=0)
    p.add_argument("--save-best-every", type=int, default=1)
    p.add_argument("--dump-val-paths", action="store_true")
    p.add_argument("--dump-val-paths-only", action="store_true")

    p.add_argument("--contrastive_loss_type", type=str, default="infonce", choices=["infonce", "vicreg"])
    p.add_argument("--contrastive_position", type=str, default="bottleneck", choices=["bottleneck", "stage1", "stage2"])
    p.add_argument("--vicreg_invariance_weight", type=float, default=25.0)
    p.add_argument("--vicreg_variance_weight", type=float, default=25.0)
    p.add_argument("--vicreg_covariance_weight", type=float, default=1.0)
    p.add_argument("--vicreg_variance_eps", type=float, default=1e-4)
    p.add_argument("--vicreg_target_std", type=float, default=1.0)

    p.add_argument("--enable-tsne", action="store_true")
    p.add_argument("--tsne-only-if-labeled", action="store_true")
    p.add_argument("--tsne-even-if-unlabeled", dest="tsne_only_if_labeled", action="store_false")
    p.set_defaults(tsne_only_if_labeled=True)
    p.add_argument("--tsne-every", type=int, default=20)
    p.add_argument("--tsne-max-items", type=int, default=1000)

    return p


__all__ = [
    "ModelConfig",
    "AETrainingConfig",
    "ExperimentConfig",
    "build_argparser",
]
