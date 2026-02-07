
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
    
    # This only change when adapt new tabs
    enable_masking: bool = True

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
    """Model architecture configuration (SwinUNet dual-view SSL)"""
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
    enable_saca: bool = False
    saca_position: str = "after_stage1" # after_patch_embed | after_stage0 | after_merge0 | after_stage1
    saca_positions: list[str] = field(default_factory=list)
    saca_gate_init: float = 0.0 
    saca_warmup_epochs: int = 5 
    

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 200
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lambda_recon: float = 0.0
    lambda_contrast: float = 0.0
    temperature: float = 0.2
    seed: int = 42
    amp: bool = True
    cpu: bool = False
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Run mode
    enable_reconstruct: bool = False
    enable_contrastive: bool = False
    single_view: bool = False
    
    # Checkpoint / pretrained loading
    resume_ckpt: str = ""  # path to .pt
    ckpt_load_mode: str = "none"  # "none" | "full" | "encoder_only"
    freeze_encoder_epochs: int = 0  # freeze encoder for first N epochs
    reset_contrastive_proj_head: bool = True  # always re-init projection head when loading pretrained encoder

    # Loss options
    enable_masked_loss: bool = False

    # Recon loss stabilization 
    recon_loss: str = "weighted_bce_logits"  # "weighted_bce_logits" or "l1_sigmoid"
    fg_eps: float = 0.02
    fg_weight: float = 10.0
    dice_loss_weight: float = 0.2
    dice_mode: str = "fg"
    dice_smooth: float = 1e-6

    # Data augmentation (contrastive)
    aug_p_noise: float = 0.7
    aug_p_jitter: float = 0.7
    aug_p_blur: float = 0.2
    aug_noise_std: float = 0.02
    aug_jitter_strength: float = 0.1
    aug_blur_kernel: int = 3

    # Optional dice auxiliary for supervised segmentation
    dice_loss_weight: float = 0.0
    dice_mode: str = "total"  # "total" | "fg"
    dice_smooth: float = 1e-6


@dataclass
class DataConfig:
    """Data loading configuration"""
    data_root: str = ""
    train_mod: int = 1
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
    split_test: bool = False


@dataclass
class LoggingConfig:
    """Logging and visualization"""
    out_dir: str = "runs_ssl_swinunet"
    run_name: str = ""
    ckpt_dir: str = ""
    vis_every: int = 20
    vis_n_results: int = 8
    save_latest_every: int = 1
    save_best_after_epoch: int = 0
    save_best_every: int = 20

    # t-SNE settings
    enable_tsne: bool = False
    tsne_only_if_labeled: bool = True
    tsne_every: int = 20
    tsne_max_items: int = 1000
    
@dataclass
class ContrastiveLossConfig:
    "Contrastive Loss options"
    contrastive_loss_type: str = "infonce" # "infonce" or "vicreg"
    contrastive_position: str = "bottleneck" # "bottleneck" or "stage1" or "stage2"
    
    # VICReg options
    vicreg_invariance_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    vicreg_variance_eps: float = 1e-4 
    vicreg_target_std: float = 1.0

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    contrast_loss: ContrastiveLossConfig = field(default_factory=ContrastiveLossConfig)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        """Create config from argparse namespace"""
        cfg = cls(
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
                enable_saca=args.enable_saca,
                saca_position=args.saca_position,
                saca_positions=[],
                saca_gate_init=args.saca_gate_init,
                saca_warmup_epochs=args.saca_warmup_epochs,
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
                dice_loss_weight=getattr(args, "dice_loss_weight", 0.2),
                dice_mode=getattr(args, "dice_mode", "fg"),
                dice_smooth=getattr(args, "dice_smooth", 1e-6),
                grad_clip=getattr(args, "grad_clip", 1.0),
                warmup_epochs=getattr(args, "warmup_epochs", 5),
                min_lr=getattr(args, "min_lr", 1e-6),
            ),
            data=DataConfig(
                data_root=args.data_root,
                train_mod=args.train_mod,
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
                split_test=args.split_test,
            ),
            mask=MaskConfig(
                patch_size=args.patch_size,
                mask_ratio_side=args.mask_ratio,
                image_size=args.image_size,
                enable_masking=args.enable_masking,
            ),
            logging=LoggingConfig(
                out_dir=args.out_dir,
                run_name=args.run_name,
                ckpt_dir=args.ckpt_dir,
                vis_every=args.vis_every,
                save_latest_every=args.save_latest_every,
                save_best_after_epoch=args.save_best_after_epoch,
                save_best_every=args.save_best_every,
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
            )
        )
        valid_positions = {"after_patch_embed", "after_stage0", "after_merge0", "after_stage1"}
        if args.saca_positions:
            positions = [p.strip() for p in args.saca_positions.split(",") if p.strip()]
            cfg.model.saca_positions = positions
            if positions:
                cfg.model.saca_position = ",".join(positions)
        elif args.saca_position:
            cfg.model.saca_positions = [args.saca_position]
        else:
            cfg.model.saca_positions = []

        invalid = [p for p in cfg.model.saca_positions if p not in valid_positions]
        if invalid:
            raise ValueError(
                f"saca_positions must be subset of {valid_positions}, got {invalid}"
            )
        return cfg


# -------------------------
# Argparse
# -------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("SwinUNet Dual View SSL (MIM + Contrastive)")

    # Data (folder dataset)
    p.add_argument("--data-root", type=str, required=True, help="Root folder containing subfolders of images")
    p.add_argument("--train-mod", type=int, default=1, help="Use only items where index%%train_mod==0")
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
    p.add_argument("--split_test", action="store_true")
    p.add_argument("--no-drop-last", dest="drop_last", action="store_false")
    p.set_defaults(drop_last=True)

    # Masking
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--mask-ratio", type=float, default=0.35)
    
    p.add_argument("--enable-masking", action="store_true")
    p.add_argument("--disable-masking", dest="enable_masking", action="store_false")
    p.set_defaults(enable_masking=True)

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
    
    # SACA 
    p.add_argument("--enable_saca", action="store_true")
    p.add_argument("--saca_position", type=str, default="after_stage1", choices=["after_patch_embed", "after_stage0", "after_merge0", "after_stage1"])
    p.add_argument("--saca_positions", type=str, default="")
    p.add_argument("--saca_gate_init", type=float, default=0.0)
    p.add_argument("--saca_warmup_epochs", type=int, default=5)

    # Training
    p.add_argument("--enable-reconstruct", action="store_true")
    p.add_argument("--disable-reconstruct", dest="enable_reconstruct", action="store_false")
    p.set_defaults(enable_reconstruct=True)
    
    p.add_argument("--enable-contrastive", action="store_true")
    p.add_argument("--disable-contrastive", dest="enable_contrastive", action="store_false")
    p.set_defaults(enable_contrastive=True)

    p.add_argument("--single-view", action="store_true")
    p.add_argument("--dual-view", dest="single_view", action="store_false")
    p.set_defaults(single_view=False)
    
    p.add_argument("--ramp-contrastive", type=int, default=20)

    p.add_argument("--enable-masked-loss", action="store_true", help="Use masked-only loss instead of mixed loss")
    p.add_argument("--recon-loss", type=str, default="weighted_bce_logits", choices=["weighted_bce_logits", "l1_sigmoid"],
                   help="Reconstruction loss type. weighted_bce_logits is recommended to avoid all-zero collapse.")
    p.add_argument("--fg-eps", type=float, default=0.02, help="Foreground threshold for weighted_bce_logits")
    p.add_argument("--fg-weight", type=float, default=10.0, help="Extra foreground weight for weighted_bce_logits")
    
    # Dice auxiliary (segmentation)
    p.add_argument("--dice-loss-weight", type=float, default=0.2, help="Weight for auxiliary soft dice loss")
    p.add_argument("--dice-mode", type=str, default="fg", choices=["total", "fg"], help="Dice over total image or fg only")
    p.add_argument("--dice-smooth", type=float, default=1e-6, help="Smoothing epsilon for dice")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-recon", type=float, default=0.0)
    p.add_argument("--lambda-contrast", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    
    # Checkpoint / pretrained
    p.add_argument("--resume-ckpt", type=str, default="", help="Path to checkpoint .pt")
    p.add_argument(
        "--ckpt-load-mode",
        type=str,
        default="none",
        choices=["none", "full", "encoder_only"],
        help="Checkpoint loading mode",
    )
    p.add_argument("--freeze-encoder-epochs", type=int, default=0, help="Freeze encoder for first N epochs")
    p.add_argument("--reset-proj-head", action="store_true", help="Re-init projection head after loading encoder")
    p.add_argument("--no-reset-proj-head", dest="reset_contrastive_proj_head", action="store_false")
    p.set_defaults(reset_contrastive_proj_head=True)


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
    p.add_argument("--vis-n-results", type=int, default=8)
    p.add_argument("--save-latest-every", type=int, default=1, help='Save "latest" checkpoint every N epochs')
    p.add_argument("--save-best-after-epoch", type=int, default=0, help='Start saving "best" checkpoints from this epoch')
    p.add_argument("--save-best-every", type=int, default=1, help="Only evaluate best saving every N epochs")
    
    # Contrast Loss options
    p.add_argument("--contrastive_loss_type", type=str, default="infonce", choices=["infonce", "vicreg"])
    p.add_argument("--contrastive_position", type=str, default="bottleneck", choices=["bottleneck", "stage1", "stage2"])
    p.add_argument("--vicreg_invariance_weight", type=float, default=25.0)
    p.add_argument("--vicreg_variance_weight", type=float, default=25.0)
    p.add_argument("--vicreg_covariance_weight", type=float, default=1.0)
    p.add_argument("--vicreg_variance_eps", type=float, default=1e-4)
    p.add_argument("--vicreg_target_std", type=float, default=1.0)

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
