from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from models.swin_unet_dualview_ssl import SwinUNetDualViewSSL
from training.utils import get_device, set_seed, ensure_dir
from data.dataset import split_indices

from .dataset import MaskReconstructionDataset
from .trainer import MaskReconstructionTrainer, RunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised mask reconstruction with Swin-UNet dual view")
    parser.add_argument("--data_dir", type=str, default="", help="[deprecated] Single folder; use --train_dir/--val_dir instead")
    parser.add_argument("--train_dir", type=str, default="", help="Folder with training *.png and *_mask.npz")
    parser.add_argument("--val_dir", type=str, default="", help="Optional folder with validation *.png and *_mask.npz")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for logs and checkpoints")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--amp", type=int, default=0, help="Use mixed precision (1) or not (0)")
    parser.add_argument("--mask_key", type=str, default=None, help="Optional key inside NPZ mask file")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold for dice metric")
    parser.add_argument("--strict_pairs", type=int, default=1, help="1: error on missing mask, 0: skip missing")
    parser.add_argument("--save_best_only", type=int, default=0, help="1: only keep best checkpoint")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (0-1)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--vis-every", type=int, default=0, help="Save validation visualizations every N epochs (0=off)")
    parser.add_argument("--vis-num", type=int, default=4, help="Number of validation samples to visualize")
    parser.add_argument("--vis-threshold", type=float, default=0.5, help="Threshold for visualization binarization")
    return parser.parse_args()


def make_dataloaders(
    train_ds: MaskReconstructionDataset,
    args: argparse.Namespace,
    device: torch.device,
    val_ds: MaskReconstructionDataset | None = None,
):
    def _loader(dataset, shuffle: bool) -> DataLoader:
        extra = {}
        if args.num_workers > 0:
            extra["persistent_workers"] = True
            extra["prefetch_factor"] = 2
        return DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=shuffle,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            **extra,
        )

    if val_ds is not None:
        return _loader(train_ds, shuffle=True), _loader(val_ds, shuffle=False)

    train_idx, val_idx, _ = split_indices(
        n=len(train_ds),
        val_ratio=float(args.val_ratio),
        test_ratio=0.0,
        seed=int(args.seed),
    )

    # Guarantee a non-empty val split when dataset is tiny
    if len(val_idx) == 0 and len(train_idx) > 1:
        val_idx = [train_idx.pop()]  # move one sample to val

    return _loader(Subset(train_ds, train_idx), shuffle=True), _loader(Subset(train_ds, val_idx), shuffle=False)


def build_model() -> SwinUNetDualViewSSL:
    """
    Instantiate Swin-UNet in dual-view reconstruction-only mode.
    SACA and contrastive heads are disabled for this supervised task.
    """
    model = SwinUNetDualViewSSL(
        in_ch=1,
        image_size=192,
        patch_size=16,
        embed_dim=96,
        enc_depths=(2, 2, 6, 2),
        dec_depths=(6, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        proj_dim=128,
        plane_inject_method="film",
        enable_saca=False,
        saca_position="after_stage1",
        enable_reconstruct=True,
        enable_contrastive=False,
        single_view=False,
    )
    return model


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    device = get_device(cpu=bool(args.cpu))
    print(f"[device] using {device}")

    effective_train_dir = args.train_dir if args.train_dir else args.data_dir
    if not effective_train_dir:
        raise ValueError("Provide --train_dir (recommended) or --data_dir (deprecated).")

    val_dir = args.val_dir

    train_ds = MaskReconstructionDataset(
        data_dir=effective_train_dir,
        strict_pairs=bool(args.strict_pairs),
        mask_key=args.mask_key,
    )
    val_ds = None
    if val_dir:
        val_ds = MaskReconstructionDataset(
            data_dir=val_dir,
            strict_pairs=bool(args.strict_pairs),
            mask_key=args.mask_key,
        )
    train_loader, val_loader = make_dataloaders(train_ds, args, device, val_ds=val_ds)

    train_pairs = len(train_loader.dataset)
    val_pairs = len(val_loader.dataset)
    print(
        f"[data] train_dir={Path(effective_train_dir).resolve()} train_pairs={train_pairs}"
        + (
            f" | val_dir={Path(val_dir).resolve()} val_pairs={val_pairs}"
            if val_ds is not None
            else f" | val_split_ratio={args.val_ratio} val_pairs={val_pairs}"
        )
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    out_dir = ensure_dir(Path(args.out_dir))
    if args.vis_every > 0:
        ensure_dir(out_dir / "vis")
    run_cfg = RunConfig(
        data_dir=str(Path(effective_train_dir).resolve()),
        out_dir=str(out_dir.resolve()),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        amp=bool(args.amp),
        mask_key=args.mask_key,
        threshold=args.threshold,
        strict_pairs=bool(args.strict_pairs),
        save_best_only=bool(args.save_best_only),
        vis_every=int(args.vis_every),
        vis_num=int(args.vis_num),
        vis_threshold=float(args.vis_threshold),
    )

    trainer = MaskReconstructionTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        out_dir=out_dir,
        run_cfg=run_cfg,
        threshold=args.threshold,
        save_best_only=bool(args.save_best_only),
        align_flip_target=True,
        vis_every=int(args.vis_every),
        vis_num=int(args.vis_num),
        vis_threshold=float(args.vis_threshold),
    )
    trainer.fit(train_loader, val_loader, epochs=int(args.epochs))


if __name__ == "__main__":
    main()
