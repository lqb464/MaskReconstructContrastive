from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MaskConfig:
    """Masking configuration (patch-level expanded to pixels)"""
    patch_size: int = 16
    mask_ratio_side: float = 0.35
    image_size: int = 192

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

    enable_reconstruct: bool = False
    enable_contrastive: bool = False
    single_view: bool = False

    resume_ckpt: str = ""
    ckpt_load_mode: str = "none"
    freeze_encoder_epochs: int = 0
    reset_contrastive_proj_head: bool = True

    enable_masked_loss: bool = False

    recon_loss: str = "weighted_bce_logits"
    fg_eps: float = 0.02
    fg_weight: float = 10.0

    aug_p_noise: float = 0.7
    aug_p_jitter: float = 0.7
    aug_p_blur: float = 0.2
    aug_noise_std: float = 0.02
    aug_jitter_strength: float = 0.1
    aug_blur_kernel: int = 3

    dice_loss_weight: float = 0.0
    dice_mode: str = "fg"
    dice_smooth: float = 1e-6

    torch_compile: bool = False


@dataclass
class DataConfig:
    """Data loading configuration"""
    data_root: str = ""
    preprocessed_dir: str = ""
    train_mod: float = 1.0
    image_size: int = 192
    plane: str = "axial"
    skip_resize_in_loader: bool = False

    label_csv: str = ""
    label_path_col: str = "image_path"
    label_col: str = "label"

    val_ratio: float = 0.2
    test_ratio: float = 0.0

    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    split_test: bool = False


@dataclass
class LoggingConfig:
    """Logging and visualization"""
    out_dir: str = "runs_ssl_ae"
    run_name: str = ""
    ckpt_dir: str = ""
    vis_every: int = 20
    vis_n_results: int = 8
    save_latest_every: int = 1
    save_best_after_epoch: int = 0
    save_best_every: int = 20

    enable_tsne: bool = False
    tsne_only_if_labeled: bool = True
    tsne_every: int = 20
    tsne_max_items: int = 1000


@dataclass
class ContrastiveLossConfig:
    "Contrastive Loss options"
    contrastive_loss_type: str = "infonce"
    contrastive_position: str = "bottleneck"

    vicreg_invariance_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    vicreg_variance_eps: float = 1e-4
    vicreg_target_std: float = 1.0


__all__ = [
    "MaskConfig",
    "TrainingConfig",
    "DataConfig",
    "LoggingConfig",
    "ContrastiveLossConfig",
]
