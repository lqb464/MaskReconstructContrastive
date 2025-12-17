# =============================================
# File: trainer.py
# Main training orchestration
# =============================================
from __future__ import annotations

import time
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from config import ExperimentConfig, build_argparser
from preprocessing import preprocess_batch
from augmentation import sample_masks_anti_mirror, HalfAug
from losses import masked_l1_loss, mixed_l1_loss, nt_xent_loss, compute_embedding_variance, ssim_index
from metrics import MetricsAccumulator, EpochMetrics, MetricsLogger, ContrastiveMetrics
from visualization import save_image_grid, plot_training_curves, run_tsne_visualization
from model import SmallUNetSSL


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SSLTrainer:
    """Self-supervised learning trainer for UNet"""
    
    def __init__(self, config: ExperimentConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Setup directories
        self._setup_directories()
        
        # Build model
        self.model = self._build_model()
        
        # Setup optimizer and scaler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and config.training.amp))
        
        # Setup augmentation
        self.augmenter = HalfAug(
            p_noise=config.training.aug_p_noise,
            p_jitter=config.training.aug_p_jitter,
            p_blur=config.training.aug_p_blur,
            noise_std=config.training.aug_noise_std,
            jitter_strength=config.training.aug_jitter_strength,
            blur_kernel=config.training.aug_blur_kernel
        )
        
        # Setup logging
        self.metrics_logger = MetricsLogger(str(self.logs_dir / 'epoch_log.csv'))
        
        # Best validation tracking
        self.best_val = float("inf")
    
    def _setup_directories(self):
        """Create output directories"""
        ts = time.strftime('%Y%m%d-%H%M%S')
        enc_tag = 'ms' if self.config.model.use_multiscale else 'bn'
        norm_tag = 'GN' if self.config.model.use_gn else 'BN'
        se_tag = '_SE' if self.config.model.use_se else ''
        
        run_name = self.config.logging.run_name
        if not run_name:
            run_name = f"{ts}_img{self.config.data.image_size}_b{self.config.model.base_ch}_{enc_tag}_{norm_tag}{se_tag}"
        
        self.base_out = Path(self.config.logging.out_dir) / run_name
        self.ckpt_dir = Path(self.config.logging.ckpt_dir) if self.config.logging.ckpt_dir else self.base_out / 'checkpoints'
        self.vis_dir = self.base_out / 'vis'
        self.tsne_dir = self.base_out / 'tsne'
        self.plots_dir = self.base_out / 'plots'
        self.logs_dir = self.base_out / 'logs'
        
        for d in [self.base_out, self.ckpt_dir, self.vis_dir, self.tsne_dir, self.plots_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _build_model(self) -> SmallUNetSSL:
        """Build and initialize model"""
        model = SmallUNetSSL(
            in_ch=self.config.model.in_ch,
            base_ch=self.config.model.base_ch,
            bottleneck_dim=self.config.model.bottleneck_dim,
            proj_dim=self.config.model.proj_dim,
            use_gn=self.config.model.use_gn,
            use_se=self.config.model.use_se,
            use_multiscale=self.config.model.use_multiscale
        ).to(self.device)
        
        print(f"Model parameters: {count_params(model) / 1e6:.2f}M")
        return model
    
    def train_epoch(self, loader) -> tuple[MetricsAccumulator, ContrastiveMetrics]:
        """Train for one epoch"""
        self.model.train()
        acc = MetricsAccumulator()
        con_loss_sum = 0.0
        con_count = 0
        all_z = []
        
        for batch in loader:
            x = batch["input"].to(self.device, non_blocking=True)
            x = preprocess_batch(x, self.config.preprocess)
            
            # Forward pass with reconstruction
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.config.training.amp)):
                pixel_mask = sample_masks_anti_mirror(x.size(0), self.config.mask, self.device)
                x_masked = x * (1.0 - pixel_mask)
                recon, _ = self.model.forward(x_masked, pixel_mask=pixel_mask)
                
                # Reconstruction loss
                if self.config.training.enable_masked_loss:
                    loss_recon = masked_l1_loss(recon, x, pixel_mask)
                else:
                    loss_recon = mixed_l1_loss(recon, x, pixel_mask)
            
            # Compute metrics (no autocast for SSIM)
            with torch.amp.autocast("cuda", enabled=False):
                ssim_sum = ssim_index(x.float(), recon.float()).sum()
            
            diff = torch.abs(recon - x)
            acc.update(diff, pixel_mask, float(ssim_sum.item()))
            
            # Contrastive loss
            if self.config.training.enable_contrastive:
                B, C, H, W = x.size()
                mid = W // 2
                left = x[..., :mid]
                right = x[..., mid:]
                left_aug = self.augmenter(left.clone())
                right_aug = self.augmenter(right.clone())
                
                mode = "multiscale" if self.config.model.use_multiscale else "bottleneck"
                zL, _ = self.model.encoder_embed(left_aug, mode=mode)
                zR, _ = self.model.encoder_embed(right_aug, mode=mode)
                loss_con = nt_xent_loss(zL, zR, temperature=self.config.training.temperature)
                
                con_loss_sum += loss_con.item()
                con_count += 1
                all_z.extend([zL.detach(), zR.detach()])
            else:
                loss_con = torch.tensor(0.0, device=x.device)
            
            # Combined loss
            loss = self.config.training.lambda_recon * loss_recon + \
                   self.config.training.lambda_contrast * loss_con
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        # Compute contrastive metrics
        if self.config.training.enable_contrastive and con_count > 0:
            mean_var, min_var = compute_embedding_variance(all_z)
            con_metrics = ContrastiveMetrics(
                loss=con_loss_sum / con_count,
                mean_var=mean_var,
                min_var=min_var
            )
        else:
            con_metrics = None
        
        return acc, con_metrics
    
    @torch.no_grad()
    def validate_epoch(self, loader) -> MetricsAccumulator:
        """Validate for one epoch"""
        self.model.eval()
        acc = MetricsAccumulator()
        
        for batch in loader:
            x = batch['input'].to(self.device, non_blocking=True)
            x = preprocess_batch(x, self.config.preprocess)
            
            pixel_mask = sample_masks_anti_mirror(x.size(0), self.config.mask, self.device)
            x_masked = x * (1.0 - pixel_mask)
            recon, _ = self.model.forward(x_masked, pixel_mask=pixel_mask)
            
            diff = torch.abs(recon - x)
            
            with torch.amp.autocast("cuda", enabled=False):
                ssim_sum = float(ssim_index(x.float(), recon.float()).sum().item())
            
            acc.update(diff, pixel_mask, ssim_sum)
        
        return acc
    
    def visualize_reconstruction(self, loader, epoch: int):
        """Save reconstruction visualization"""
        self.model.eval()
        batch = next(iter(loader))
        x = batch['input'].to(self.device, non_blocking=True)
        x = preprocess_batch(x, self.config.preprocess)
        
        with torch.no_grad():
            pixel_mask = sample_masks_anti_mirror(x.size(0), self.config.mask, self.device)
            x_masked = x * (1.0 - pixel_mask)
            recon, _ = self.model.forward(x_masked, pixel_mask=pixel_mask)
        
        out_path = str(self.vis_dir / f'epoch_{epoch:03d}.png')
        
        if self.config.training.enable_masked_loss:
            recon_full = pixel_mask * recon + (1.0 - pixel_mask) * x
            residual = torch.abs(x - recon_full).clamp(0, 1)
            save_image_grid(
                [x, pixel_mask, x_masked, recon_full.clamp(0, 1), residual],
                ['target', 'mask', 'masked', 'recon_full', 'residual'],
                out_path
            )
        else:
            residual = torch.abs(x - recon).clamp(0, 1)
            save_image_grid(
                [x, pixel_mask, x_masked, recon.clamp(0, 1), residual],
                ['target', 'mask', 'masked', 'recon', 'residual'],
                out_path
            )
    
    def save_checkpoint(self, epoch: int, val_metric: float):
        """Save model checkpoint"""
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "args": vars(self.config),
            "val_recon": val_metric,
            "base_ch": self.config.model.base_ch,
            "bottleneck_dim": self.config.model.bottleneck_dim,
            "use_gn": self.config.model.use_gn,
            "use_se": self.config.model.use_se,
            "use_multiscale": self.config.model.use_multiscale
        }
        
        if val_metric < self.best_val:
            self.best_val = val_metric
            torch.save(ckpt, str(self.ckpt_dir / "ckpt_best.pt"))
    
    def train(self, train_loader, val_loader, test_loader):
        """Main training loop"""
        for epoch in tqdm(range(1, self.config.training.epochs + 1), desc="Training"):
            # Train
            train_acc, con_metrics = self.train_epoch(train_loader)
            train_metrics = train_acc.compute()
            
            # Validate
            val_acc = self.validate_epoch(val_loader)
            val_metrics = val_acc.compute()
            
            # Log metrics
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train=train_metrics,
                val=val_metrics,
                contrastive=con_metrics
            )
            print(f"\n{epoch_metrics}")
            self.metrics_logger.log(epoch_metrics)
            
            # Visualization
            if epoch % self.config.logging.vis_every == 0:
                self.visualize_reconstruction(val_loader, epoch)
            
            # t-SNE
            if epoch % self.config.logging.tsne_every == 0:
                tsne_prefix = str(self.tsne_dir / f"tsne_epoch{epoch:03d}")
                try:
                    run_tsne_visualization(
                        self.model, 
                        val_loader, 
                        self.device, 
                        tsne_prefix,
                        max_items=self.config.logging.tsne_max_items
                    )
                except Exception as e:
                    print(f"t-SNE failed: {e}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics.total_l1)
        
        # Final test evaluation
        test_acc = self.validate_epoch(test_loader)
        test_metrics = test_acc.compute()
        print(f"\nFinal Test Metrics: {test_metrics}")
        
        # Generate training curves
        plot_training_curves(self.logs_dir / 'epoch_log.csv', self.plots_dir)


def main():
    """Main entry point"""
    args = build_argparser().parse_args()
    config = ExperimentConfig.from_args(args)
    
    # Setup
    set_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.training.cpu else "cpu")
    
    # Load data
    from data import create_unet_dataloaders
    train_loader, val_loader, test_loader = create_unet_dataloaders(
        image_size=config.data.image_size,
        batch_size=config.training.batch_size,
        val_size=config.data.val_size,
        num_workers=config.data.num_workers,
        apply_unsharp=config.data.apply_unsharp,
        pin_memory=config.data.pin_memory,
        data_source=config.data.data_source,
        adni_path=config.data.adni_path,
        adni_image_type=config.data.image_type,
        adni_preproc_path=config.data.adni_preproc_path
    )
    
    # Train
    trainer = SSLTrainer(config, device)
    trainer.train(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()