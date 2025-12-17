# =============================================
# File: config.py
# Centralized configuration and argument parsing
# =============================================
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreprocessConfig:
    """Preprocessing options"""
    pre_norm: bool = False
    pre_crop: bool = False
    pre_bias: bool = False
    pre_align: bool = False


@dataclass
class MaskConfig:
    """Masking configuration"""
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
    """Model architecture configuration"""
    in_ch: int = 1
    base_ch: int = 16
    bottleneck_dim: int = 128
    proj_dim: int = 128
    use_gn: bool = False
    use_se: bool = False
    use_multiscale: bool = True


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
    enable_contrastive: bool = False
    enable_masked_loss: bool = False
    
    # Data augmentation
    aug_p_noise: float = 0.7
    aug_p_jitter: float = 0.7
    aug_p_blur: float = 0.2
    aug_noise_std: float = 0.02
    aug_jitter_strength: float = 0.1
    aug_blur_kernel: int = 3


@dataclass
class DataConfig:
    """Data loading configuration"""
    data_source: str = "hf"
    adni_path: str = ""
    adni_preproc_path: str = ""
    image_type: str = "axial"
    image_size: int = 192
    val_size: float = 0.2
    num_workers: int = 4
    apply_unsharp: bool = True
    pin_memory: bool = True


@dataclass
class LoggingConfig:
    """Logging and visualization"""
    out_dir: str = "runs_ssl_unet"
    run_name: str = ""
    ckpt_dir: str = ""
    vis_every: int = 20
    tsne_every: int = 20
    tsne_max_items: int = 1000


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ExperimentConfig':
        """Create config from argparse namespace"""
        return cls(
            model=ModelConfig(
                base_ch=args.base_ch,
                bottleneck_dim=args.bottleneck_dim,
                proj_dim=args.proj_dim,
                use_gn=args.use_gn,
                use_se=args.use_se,
                use_multiscale=args.use_multiscale,
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
            ),
            data=DataConfig(
                data_source=args.data_source,
                adni_path=args.adni_path,
                adni_preproc_path=args.adni_preproc_path,
                image_type=args.image_type,
                image_size=args.image_size,
                val_size=args.val_size,
                num_workers=args.num_workers,
            ),
            mask=MaskConfig(
                patch_size=args.patch_size,
                mask_ratio_side=args.mask_ratio,
                image_size=args.image_size,
            ),
            preprocess=PreprocessConfig(
                pre_norm=args.pre_norm,
                pre_crop=args.pre_crop,
                pre_bias=args.pre_bias,
                pre_align=args.pre_align,
            ),
            logging=LoggingConfig(
                out_dir=args.out_dir,
                run_name=args.run_name,
                ckpt_dir=args.ckpt_dir,
                vis_every=args.vis_every,
                tsne_every=args.tsne_every,
                tsne_max_items=args.tsne_max_items,
            ),
        )


def build_argparser() -> argparse.ArgumentParser:
    """Build argument parser for training"""
    p = argparse.ArgumentParser("Self supervised UNet training")
    
    # Data source
    p.add_argument("--data-source", type=str, choices=["hf", "adni", "adni_preproc"])
    p.add_argument("--adni-path", type=str, default="")
    p.add_argument("--adni-preproc-path", type=str, default="")
    p.add_argument("--image-type", type=str, default="axial", choices=["axial", "coronal"])
     
    # Size and data
    p.add_argument("--image-size", type=int, default=192)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--mask-ratio", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)

    # Model
    p.add_argument("--base-ch", type=int, default=16)
    p.add_argument("--bottleneck-dim", type=int, default=128)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--use-gn", action="store_true")
    p.add_argument("--use-se", action="store_true")
    p.add_argument("--use-multiscale", action="store_true")

    # Training
    p.add_argument("--enable-contrastive", action="store_true")
    p.add_argument("--enable-masked-loss", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-contrast", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="runs_ssl_unet")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--ckpt-dir", type=str, default="")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--vis-every", type=int, default=20)
    p.add_argument("--tsne-every", type=int, default=20)
    p.add_argument("--tsne-max-items", type=int, default=1000)

    # Preprocessing
    p.add_argument("--pre-norm", action="store_true")
    p.add_argument("--pre-crop", action="store_true")
    p.add_argument("--pre-bias", action="store_true")
    p.add_argument("--pre-align", action="store_true")
    
    return p