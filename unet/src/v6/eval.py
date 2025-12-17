# =============================================
# File: eval.py
# Evaluation helpers and CLI for trained models
# =============================================
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import MaskConfig, PreprocessConfig
from preprocessing import preprocess_batch
from augmentation import sample_masks_anti_mirror
from losses import masked_l1_loss
from metrics import MetricsAccumulator
from visualization import run_tsne_visualization
from model import SmallUNetSSL

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import (
    create_unet_dataloader_from_folder_csv,
    mindset_colors_1,
    mindset_colors_2,
    mindset_idx_map_label_1,
    mindset_idx_map_label_2,
    mindset_label_map_idx_1,
    mindset_label_map_idx_2,
    hf_idx_map_label,
    hf_demantia_colors,
)


class DataModule:
    """Wrapper to provide data label mappings for visualization"""
    mindset_colors_1 = mindset_colors_1
    mindset_colors_2 = mindset_colors_2
    mindset_idx_map_label_1 = mindset_idx_map_label_1
    mindset_idx_map_label_2 = mindset_idx_map_label_2
    mindset_label_map_idx_1 = mindset_label_map_idx_1
    mindset_label_map_idx_2 = mindset_label_map_idx_2
    hf_idx_map_label = hf_idx_map_label
    hf_demantia_colors = hf_demantia_colors


def load_checkpoint(ckpt_path: str, device: torch.device) -> tuple[SmallUNetSSL, dict]:
    """
    Load model from checkpoint
    
    Args:
        ckpt_path: Path to checkpoint file
        device: Target device
        
    Returns:
        (model, config_dict)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("args", {})
    
    # Handle both old dict format and new config format
    if isinstance(cfg, dict):
        model_cfg = {
            "in_ch": 1,
            "base_ch": cfg.get("base_ch", 16),
            "bottleneck_dim": cfg.get("bottleneck_dim", 128),
            "proj_dim": cfg.get("proj_dim", 128),
            "use_gn": cfg.get("use_gn", False),
            "use_se": cfg.get("use_se", False),
            "use_multiscale": cfg.get("use_multiscale", True),
        }
    else:
        # New config format
        model_cfg = {
            "in_ch": 1,
            "base_ch": cfg.model.base_ch,
            "bottleneck_dim": cfg.model.bottleneck_dim,
            "proj_dim": cfg.model.proj_dim,
            "use_gn": cfg.model.use_gn,
            "use_se": cfg.model.use_se,
            "use_multiscale": cfg.model.use_multiscale,
        }
    
    model = SmallUNetSSL(**model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    
    return model, cfg


@torch.no_grad()
def evaluate_reconstruction(
    model: SmallUNetSSL,
    loader: DataLoader,
    device: torch.device,
    mask_spec: MaskConfig,
    preprocess_cfg: PreprocessConfig
) -> dict:
    """
    Evaluate reconstruction quality on a dataset
    
    Args:
        model: Trained model
        loader: DataLoader
        device: Computing device
        mask_spec: Masking configuration
        preprocess_cfg: Preprocessing configuration
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    acc = MetricsAccumulator()
    
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        x = preprocess_batch(x, preprocess_cfg)
        
        pixel_mask = sample_masks_anti_mirror(x.size(0), mask_spec, device)
        x_masked = x * (1 - pixel_mask)
        recon, _ = model(x_masked, pixel_mask=pixel_mask)
        
        diff = torch.abs(recon - x)
        acc.update(diff, pixel_mask)
    
    metrics = acc.compute()
    
    return {
        "masked_l1": metrics.masked_l1,
        "unmasked_l1": metrics.unmasked_l1,
        "total_l1": metrics.total_l1,
    }


def run_tsne_multi_label(
    model: SmallUNetSSL,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_items: int = 1000
):
    """
    Generate t-SNE visualizations for multiple label types
    
    Args:
        model: Trained model
        loader: DataLoader
        device: Computing device
        out_dir: Output directory
        max_items: Maximum samples to use
    """
    data_module = DataModule()
    
    # Generate t-SNE for label_1
    tsne_prefix_1 = str(out_dir / "tsne_label_1")
    try:
        run_tsne_visualization(
            model=model,
            loader=loader,
            device=device,
            out_prefix=tsne_prefix_1,
            max_items=max_items,
            label_val="label_1",
            data_module=data_module
        )
        print(f"✓ Saved t-SNE for label_1 to {tsne_prefix_1}_enc_*.png")
    except Exception as e:
        print(f"✗ Failed to generate t-SNE for label_1: {e}")
    
    # Generate t-SNE for label_2
    tsne_prefix_2 = str(out_dir / "tsne_label_2")
    try:
        run_tsne_visualization(
            model=model,
            loader=loader,
            device=device,
            out_prefix=tsne_prefix_2,
            max_items=max_items,
            label_val="label_2",
            data_module=data_module
        )
        print(f"✓ Saved t-SNE for label_2 to {tsne_prefix_2}_enc_*.png")
    except Exception as e:
        print(f"✗ Failed to generate t-SNE for label_2: {e}")


def build_eval_argparser() -> argparse.ArgumentParser:
    """Build argument parser for evaluation"""
    p = argparse.ArgumentParser(
        description="Evaluate trained SSL UNet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to model checkpoint")
    p.add_argument("--image-dir", type=str, required=True,
                   help="Path to image directory")
    p.add_argument("--csv-map", type=str, required=True,
                   help="Path to CSV mapping file")
    
    # Data arguments
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                   help="Which split to evaluate")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for evaluation")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Number of data loading workers")
    
    # Evaluation options
    p.add_argument("--compute-recon", action="store_true",
                   help="Compute reconstruction metrics")
    p.add_argument("--tsne", action="store_true",
                   help="Generate t-SNE visualizations")
    p.add_argument("--tsne-max-items", type=int, default=1000,
                   help="Maximum samples for t-SNE")
    
    # Output
    p.add_argument("--out-dir", type=str, default="runs_eval",
                   help="Output directory for results")
    
    # Preprocessing (override checkpoint config if specified)
    p.add_argument("--pre-norm", action="store_true",
                   help="Enable intensity normalization")
    p.add_argument("--pre-crop", action="store_true",
                   help="Enable tight brain cropping")
    p.add_argument("--pre-bias", action="store_true",
                   help="Enable bias field correction")
    p.add_argument("--pre-align", action="store_true",
                   help="Enable midline alignment")
    
    return p


def main():
    """Main evaluation entry point"""
    args = build_eval_argparser().parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.ckpt}")
    model, cfg = load_checkpoint(args.ckpt, device)
    print("✓ Model loaded successfully")
    
    # Extract config from checkpoint
    if isinstance(cfg, dict):
        # Old dict format
        image_size = cfg.get("image_size", 192)
        patch_size = cfg.get("patch_size", 16)
        mask_ratio = cfg.get("mask_ratio", 0.35)
        
        # Preprocessing config (use checkpoint config or CLI override)
        preprocess_cfg = PreprocessConfig(
            pre_norm=args.pre_norm if args.pre_norm else cfg.get("pre_norm", False),
            pre_crop=args.pre_crop if args.pre_crop else cfg.get("pre_crop", False),
            pre_bias=args.pre_bias if args.pre_bias else cfg.get("pre_bias", False),
            pre_align=args.pre_align if args.pre_align else cfg.get("pre_align", False),
        )
    else:
        # New config format
        image_size = cfg.data.image_size
        patch_size = cfg.mask.patch_size
        mask_ratio = cfg.mask.mask_ratio_side
        
        # Use CLI args if provided, otherwise use checkpoint config
        preprocess_cfg = PreprocessConfig(
            pre_norm=args.pre_norm if args.pre_norm else cfg.preprocess.pre_norm,
            pre_crop=args.pre_crop if args.pre_crop else cfg.preprocess.pre_crop,
            pre_bias=args.pre_bias if args.pre_bias else cfg.preprocess.pre_bias,
            pre_align=args.pre_align if args.pre_align else cfg.preprocess.pre_align,
        )
    
    mask_spec = MaskConfig(
        patch_size=patch_size,
        mask_ratio_side=mask_ratio,
        image_size=image_size
    )
    
    print(f"\nConfiguration:")
    print(f"  Image size: {image_size}")
    print(f"  Patch size: {patch_size}")
    print(f"  Mask ratio: {mask_ratio}")
    print(f"  Preprocessing: norm={preprocess_cfg.pre_norm}, crop={preprocess_cfg.pre_crop}, "
          f"bias={preprocess_cfg.pre_bias}, align={preprocess_cfg.pre_align}")
    
    # Load data
    print(f"\nLoading {args.split} data from: {args.image_dir}")
    loader = create_unet_dataloader_from_folder_csv(
        image_dir=args.image_dir,
        csv_map=args.csv_map,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        apply_unsharp=True,
        pin_memory=True,
    )
    print(f"✓ Loaded {len(loader.dataset)} samples")
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate reconstruction
    if args.compute_recon:
        print("\n" + "="*60)
        print("RECONSTRUCTION EVALUATION")
        print("="*60)
        
        metrics = evaluate_reconstruction(
            model=model,
            loader=loader,
            device=device,
            mask_spec=mask_spec,
            preprocess_cfg=preprocess_cfg
        )
        
        print(f"\n{args.split.upper()} Results:")
        print(f"  Masked L1:   {metrics['masked_l1']:.4f}")
        print(f"  Unmasked L1: {metrics['unmasked_l1']:.4f}")
        print(f"  Total L1:    {metrics['total_l1']:.4f}")
        
        # Save results to file
        results_file = out_dir / f"eval_results_{args.split}.txt"
        with open(results_file, 'w') as f:
            f.write(f"Evaluation Results - {args.split}\n")
            f.write("="*60 + "\n")
            f.write(f"Checkpoint: {args.ckpt}\n")
            f.write(f"Image dir: {args.image_dir}\n")
            f.write(f"CSV map: {args.csv_map}\n")
            f.write("\nMetrics:\n")
            f.write(f"  Masked L1:   {metrics['masked_l1']:.4f}\n")
            f.write(f"  Unmasked L1: {metrics['unmasked_l1']:.4f}\n")
            f.write(f"  Total L1:    {metrics['total_l1']:.4f}\n")
        
        print(f"\n✓ Results saved to: {results_file}")
    
    # Generate t-SNE
    if args.tsne:
        print("\n" + "="*60)
        print("t-SNE VISUALIZATION")
        print("="*60)
        print(f"Generating t-SNE with max {args.tsne_max_items} samples...")
        
        run_tsne_multi_label(
            model=model,
            loader=loader,
            device=device,
            out_dir=out_dir,
            max_items=args.tsne_max_items
        )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"All results saved to: {out_dir}")


if __name__ == "__main__":
    main()