from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .common.cli_utils import run_entrypoint
from .config.experiment import ExperimentConfig, build_argparser
from .data.dataset import create_dataloaders_from_folder
from .training.utils import (
    copy_images_to_dir,
    ensure_dir,
    extract_dataset_paths,
    get_device,
    set_seed,
    write_path_list,
)
from .trainer import Trainer


def run(args) -> None:
    """
    Train MAE or VAE (ver2).

    Fair comparison with swin_unet on the same val images:
      - Use identical --seed, --data-root, --train-mod, --val-ratio, --image-size
      - Run with --dump-val-paths to write val_paths_ssl.txt
      - Use --vis-batch-index 0 and --vis-mask-seed <int> for reproducible viz masks (MAE)
    """
    cfg = ExperimentConfig.from_args(args)

    cfg.training.enable_reconstruct = True
    cfg.training.enable_contrastive = False
    if float(cfg.training.lambda_recon) == 0.0:
        cfg.training.lambda_recon = 1.0

    backbone = str(cfg.model.backbone).lower()
    if backbone == "vae":
        cfg.training.enable_masked_loss = False
        cfg.mask.enable_masking = False
    elif backbone == "mae":
        cfg.training.enable_masked_loss = bool(args.enable_masked_loss)

    print("=" * 100)
    print("Configuration:")
    print(cfg)
    print("=" * 100)

    print("Loss Function:")
    print("Backbone:", cfg.model.backbone)
    if backbone == "vae":
        print("L =", cfg.training.lambda_recon, "* L_recon +", cfg.training.lambda_kl, "* L_kl")
    else:
        print("L =", cfg.training.lambda_recon, "* L_recon (masked-only default for MAE)")

    if cfg.training.single_view and not cfg.training.enable_reconstruct:
        raise Exception("[Error] single_view requires --enable-reconstruct")

    set_seed(cfg.training.seed)
    device = get_device(cfg.training.cpu)

    train_loader, val_loader, _, _ = create_dataloaders_from_folder(
        data_root=cfg.data.data_root,
        train_mod=cfg.data.train_mod,
        image_size=cfg.data.image_size,
        plane=cfg.data.plane,
        label_csv=cfg.data.label_csv if cfg.data.label_csv else None,
        label_path_col=cfg.data.label_path_col,
        label_col=cfg.data.label_col,
        batch_size=cfg.training.batch_size,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        seed=cfg.training.seed,
        drop_last=cfg.data.drop_last,
        split_test=cfg.data.split_test,
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    if bool(getattr(args, "dump_val_paths", False)) or bool(getattr(args, "dump_val_paths_only", False)):
        out_dir = Path(cfg.logging.out_dir)
        if cfg.logging.run_name:
            out_dir = out_dir / cfg.logging.run_name
        out_dir = ensure_dir(out_dir)

        val_paths = extract_dataset_paths(val_loader.dataset)
        val_path_file = write_path_list(val_paths, out_dir / "val_paths_ssl.txt")
        copied_n = copy_images_to_dir(val_paths, out_dir / "val_images_ssl")
        print(f"[val_paths] task=ssl count={len(val_paths)} file={val_path_file}")
        print(f"[val_paths] copied_images={copied_n} dir={out_dir / 'val_images_ssl'}")
        for p in val_paths:
            print(p)

        if bool(getattr(args, "dump_val_paths_only", False)):
            print("[val_paths] dump_val_paths_only=1; exiting before training.")
            return

    trainer = Trainer(cfg, device)
    trainer.fit(train_loader, val_loader)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_entrypoint(build_argparser, run, argv=argv)


if __name__ == "__main__":
    main()
