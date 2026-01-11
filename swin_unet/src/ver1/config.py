
# =============================================
# File: config_phase1.py
# Centralized configuration and argument parsing (Phase 1)
# - Folder dataset with optional CSV labels
# - No preprocessing pipeline (removed)
# - Keeps mask config (patch masking) and logging config
# =============================================
from __future__ import annotations

import argparse
from dataclasses import dataclass, field


# -------------------------
# Core configs
# -------------------------
@dataclass
class MaskConfig:
    """Masking configuration (patch-level expanded to pixels)"""
    patch_size: int = 16
    mask_ratio_side: float = 0.35
    image_size: int = 192

    def grid_size(self) -> tuple[int, int]:
        gh = self.image_size // self.patch_size
        gw = self.image_size // self.patch_size
        return gh, gw

    def half_grid_w(self) -> int:
        return (self.image_size // 2) // self.patch_size

    def num_patches_side(self) -> int:
        gh = self.image_size // self.patch_size
        hw = self.half_grid_w()
        return gh * hw


@dataclass
class ModelConfig:
    """Model architecture configuration (Phase 1: SwinUNet dual-view SSL)"""
    in_ch: int = 1

    # Swin/UNet knobs (Phase 1 implementation)
    patch_size: int = 16
    embed_dim: int = 96
    enc_depths: tuple[int, int, int, int] = (2, 2, 6, 2)
    dec_depths: tuple[int, int, int] = (6, 2, 2)
    num_heads: tuple[int, int, int, int] = (3, 6, 12, 24)
    window_size: int = 7

    # SSL heads
    proj_dim: int = 128
    bottleneck_dim: int = 256

    # Dual-view split/share
    split_to_stage: int = 1
    shared_from_stage: int = 2

    # Plane conditioning
    plane_in_dim: int = 2
    plane_inject_stage: int = 2
    plane_inject_method: str = "film"  # "film" or "add"
    enable_saca_stage1: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_recon: float = 1.0
    lambda_contrast: float = 1.0
    temperature: float = 0.2
    seed: int = 42
    amp: bool = False
    cpu: bool = False

    # Loss options
    enable_contrastive: bool = True
    enable_masked_loss: bool = False

    # Recon loss stabilization (recommended for sparse medical images)
    recon_loss: str = "weighted_bce_logits"  # "weighted_bce_logits" or "l1_sigmoid"
    fg_eps: float = 0.02
    fg_weight: float = 10.0

    # Data augmentation (contrastive)
    aug_p_noise: float = 0.7
    aug_p_jitter: float = 0.7
    aug_p_blur: float = 0.2
    aug_noise_std: float = 0.02
    aug_jitter_strength: float = 0.1
    aug_blur_kernel: int = 3


@dataclass
class DataConfig:
    """Data loading configuration (Phase 1 folder dataset)"""
    data_root: str = ""
    image_size: int = 192
    plane: str = "axial"  # "axial" | "coronal" | "auto"

    # Optional labels
    label_csv: str = ""
    label_path_col: str = "image_path"
    label_col: str = "label"

    # Random split ratios
    val_ratio: float = 0.2
    test_ratio: float = 0.0

    # Dataloader
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True


@dataclass
class LoggingConfig:
    """Logging and visualization"""
    out_dir: str = "runs_ssl_swinunet"
    run_name: str = ""
    ckpt_dir: str = ""
    vis_every: int = 20

    # t-SNE settings
    enable_tsne: bool = False
    tsne_only_if_labeled: bool = True
    tsne_every: int = 20
    tsne_max_items: int = 1000


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        """Create config from argparse namespace"""
        return cls(
            model=ModelConfig(
                in_ch=args.in_ch,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                enc_depths=tuple(args.enc_depths),
                dec_depths=tuple(args.dec_depths),
                num_heads=tuple(args.num_heads),
                window_size=args.window_size,
                proj_dim=args.proj_dim,
                bottleneck_dim=args.bottleneck_dim,
                split_to_stage=args.split_to_stage,
                shared_from_stage=args.shared_from_stage,
                plane_inject_method=args.plane_inject_method,
                enable_saca_stage1= args.enable_saca_stage1,
            ),
            training=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lambda_recon=args.lambda_recon,
                lambda_contrast=args.lambda_contrast,
                temperature=args.temperature,
                seed=args.seed,
                amp=args.amp,
                cpu=args.cpu,
                enable_contrastive=args.enable_contrastive,
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
            ),
            data=DataConfig(
                data_root=args.data_root,
                image_size=args.image_size,
                plane=args.plane,
                label_csv=args.label_csv,
                label_path_col=args.label_path_col,
                label_col=args.label_col,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                drop_last=args.drop_last,
            ),
            mask=MaskConfig(
                patch_size=args.patch_size,
                mask_ratio_side=args.mask_ratio,
                image_size=args.image_size,
            ),
            logging=LoggingConfig(
                out_dir=args.out_dir,
                run_name=args.run_name,
                ckpt_dir=args.ckpt_dir,
                vis_every=args.vis_every,
                enable_tsne=args.enable_tsne,
                tsne_only_if_labeled=args.tsne_only_if_labeled,
                tsne_every=args.tsne_every,
                tsne_max_items=args.tsne_max_items,
            ),
        )



# -------------------------
# Argparse
# -------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Phase 1: SwinUNet Dual View SSL (MIM + Contrastive)")

    # Data (folder dataset)
    p.add_argument("--data-root", type=str, required=True, help="Root folder containing subfolders of images")
    p.add_argument("--image-size", type=int, default=192)
    p.add_argument("--plane", type=str, default="axial", choices=["axial", "coronal", "auto"])

    # Optional label CSV mapping
    p.add_argument("--label-csv", type=str, default="", help="CSV mapping image_path -> label (optional)")
    p.add_argument("--label-path-col", type=str, default="image_path")
    p.add_argument("--label-col", type=str, default="label")

    # Split ratios
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.0)

    # Dataloader
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=True)
    p.add_argument("--drop-last", action="store_true")
    p.add_argument("--no-drop-last", dest="drop_last", action="store_false")
    p.set_defaults(drop_last=True)

    # Masking
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--mask-ratio", type=float, default=0.35)

    # Model
    p.add_argument("--in-ch", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--enc-depths", type=int, nargs=4, default=[2, 2, 6, 2])
    p.add_argument("--dec-depths", type=int, nargs=3, default=[6, 2, 2])
    p.add_argument("--num-heads", type=int, nargs=4, default=[3, 6, 12, 24])
    p.add_argument("--window-size", type=int, default=7)

    p.add_argument("--bottleneck-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)

    # Dual view split/share
    p.add_argument("--split-to-stage", type=int, default=1)
    p.add_argument("--shared-from-stage", type=int, default=2)

    # Plane conditioning
    p.add_argument("--plane-inject-method", type=str, default="film", choices=["film", "add"])
    p.add_argument("--enable_saca_stage1", action="store_true")

    # Training
    p.add_argument("--enable-contrastive", action="store_true")
    p.add_argument("--ramp-contrastive", type=int, default=20)
    p.add_argument("--disable-contrastive", dest="enable_contrastive", action="store_false")
    p.set_defaults(enable_contrastive=True)

    p.add_argument("--enable-masked-loss", action="store_true", help="Use masked-only loss instead of mixed loss")
    p.add_argument("--recon-loss", type=str, default="weighted_bce_logits", choices=["weighted_bce_logits", "l1_sigmoid"],
                   help="Reconstruction loss type. weighted_bce_logits is recommended to avoid all-zero collapse.")
    p.add_argument("--fg-eps", type=float, default=0.02, help="Foreground threshold for weighted_bce_logits")
    p.add_argument("--fg-weight", type=float, default=10.0, help="Extra foreground weight for weighted_bce_logits")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-recon", type=float, default=0.1)
    p.add_argument("--lambda-contrast", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true")

    # Augmentation
    p.add_argument("--aug-p-noise", type=float, default=0.7)
    p.add_argument("--aug-p-jitter", type=float, default=0.7)
    p.add_argument("--aug-p-blur", type=float, default=0.2)
    p.add_argument("--aug-noise-std", type=float, default=0.02)
    p.add_argument("--aug-jitter-strength", type=float, default=0.1)
    p.add_argument("--aug-blur-kernel", type=int, default=3)

    # Logging / outputs
    p.add_argument("--out-dir", type=str, default="runs_ssl_swinunet")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--ckpt-dir", type=str, default="")
    p.add_argument("--vis-every", type=int, default=20)

    # t-SNE gating
    p.add_argument("--enable-tsne", action="store_true")
    p.add_argument("--tsne-only-if-labeled", action="store_true")
    p.add_argument("--tsne-even-if-unlabeled", dest="tsne_only_if_labeled", action="store_false")
    p.set_defaults(tsne_only_if_labeled=True)
    p.add_argument("--tsne-every", type=int, default=20)
    p.add_argument("--tsne-max-items", type=int, default=1000)

    return p


__all__ = [
    "MaskConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "LoggingConfig",
    "ExperimentConfig",
    "build_argparser",
]