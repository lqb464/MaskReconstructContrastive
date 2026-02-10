from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence
import torch


from ..common.cli_utils import run_entrypoint
from .experiment import ExperimentConfig, build_argparser, enforce_recon_only_args
from .dataset import MaskReconstructionDataset
from ..models.swin_unet_dualview_ssl import SwinUNetDualViewSSL

PREPROCESS_META_FILENAME = "preprocess_meta.json"



def build_mask_argparser() -> argparse.ArgumentParser:
    # Reuse full ver2 CLI
    parser = build_argparser()

    # Mask reconstruction specific group
    grp = parser.add_argument_group("mask_reconstruction dataset")
    grp.add_argument("--train_dir", type=str, required=True, help="Folder with training *.png and mask files")
    grp.add_argument("--val_dir", type=str, default="", help="Optional folder with validation *.png and mask files")
    grp.add_argument("--image_ext", type=str, default=".png", help="Image extension for inputs")
    grp.add_argument("--mask_suffix", type=str, default="_mask.npz", help="Suffix appended to image stem to find mask")
    grp.add_argument("--mask_key", type=str, default="", help="Optional key inside NPZ mask file")
    grp.add_argument("--threshold", type=float, default=None, help="Optional threshold for dice metric")
    grp.add_argument("--strict_pairs", type=int, default=1, help="1: error on missing mask, 0: skip missing")
    grp.add_argument("--vis-num", type=int, default=4, help="Number of validation samples to visualize")
    grp.add_argument("--vis-threshold", type=float, default=0.5, help="Threshold for visualization binarization")
    grp.add_argument("--no-tqdm", type=int, default=0, help="Disable progress bars (default off)")
    grp.add_argument("--target-size", type=int, default=0, help="Force square resize to this size (0 keeps original)")
    grp.add_argument("--resize-mode", type=str, default="letterbox", choices=["letterbox", "direct"], help="Resize strategy for image/mask pair")
    grp.add_argument("--debug-shapes", type=int, default=0, help="Log sample shapes for debugging (0/1)")
    grp.add_argument("--binarize-target", action="store_true", help="Binarize target mask with (y > 0).float()")
    grp.add_argument("--preprocessed_dir", dest="preprocessed_dir", type=str, default="", help="Optional preprocessed train folder (offline-resized pairs).")
    grp.add_argument("--skip_resize_in_loader", dest="skip_resize_in_loader", action="store_true", help="Skip resize in dataset loader (use with preprocessed data).")

    # Make base data-root optional by clearing required flag to allow train_dir-only workflows
    for action in parser._actions:
        if action.dest == "data_root":
            action.required = False
            if action.default is None:
                action.default = ""
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_mask_argparser()
    args = parser.parse_args(argv)
    enforce_recon_only_args(args)
    return args


def make_dataloaders(
    train_ds: MaskReconstructionDataset,
    cfg: ExperimentConfig,
    device: torch.device,
    val_ds: MaskReconstructionDataset | None = None,
):
    from torch.utils.data import DataLoader, Subset

    from ..data.dataset import split_indices
    from .utils import make_worker_init_fn

    seed_workers = bool(int(os.getenv("MASK_RECON_SEED_WORKERS", "0")))
    worker_init_fn = make_worker_init_fn(int(cfg.training.seed)) if seed_workers else None

    def _loader(dataset, shuffle: bool) -> DataLoader:
        extra = {}
        if cfg.data.num_workers > 0:
            extra["persistent_workers"] = True
            extra["prefetch_factor"] = 2
            if worker_init_fn is not None:
                extra["worker_init_fn"] = worker_init_fn
        return DataLoader(
            dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=shuffle,
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory) and device.type == "cuda",
            drop_last=bool(cfg.data.drop_last) if shuffle else False,
            **extra,
        )

    if val_ds is not None:
        # Apply train_mod only to train set
        if int(cfg.data.train_mod) > 1:
            idx_train = [i for i in range(len(train_ds)) if (i % int(cfg.data.train_mod)) == 0]
            train_ds = Subset(train_ds, idx_train)
        return _loader(train_ds, shuffle=True), _loader(val_ds, shuffle=False)

    # Apply train_mod before split
    if int(cfg.data.train_mod) > 1:
        idx_train = [i for i in range(len(train_ds)) if (i % int(cfg.data.train_mod)) == 0]
        train_ds = Subset(train_ds, idx_train)

    train_idx, val_idx, _ = split_indices(
        n=len(train_ds),
        val_ratio=float(cfg.data.val_ratio),
        test_ratio=0.0,
        seed=int(cfg.training.seed),
    )

    # Guarantee a non-empty val split when dataset is tiny
    if len(val_idx) == 0 and len(train_idx) > 1:
        val_idx = [train_idx.pop()]  # move one sample to val

    return _loader(Subset(train_ds, train_idx), shuffle=True), _loader(Subset(train_ds, val_idx), shuffle=False)


def build_model(cfg: ExperimentConfig) -> SwinUNetDualViewSSL:
    """Instantiate Swin-UNet using shared config to honor SACA/contrastive flags."""
    from ..models.swin_unet_dualview_ssl import SwinUNetDualViewSSL

    mcfg = cfg.model
    tcfg = cfg.training
    if bool(getattr(tcfg, "enable_contrastive", False)):
        raise ValueError("mask_reconstruction entrypoint forbids contrastive mode.")
    if bool(getattr(cfg.mask, "enable_masking", False)):
        raise ValueError("mask_reconstruction entrypoint forbids masking mode.")
    model = SwinUNetDualViewSSL(
        in_ch=mcfg.in_ch,
        image_size=cfg.data.image_size,
        patch_size=mcfg.patch_size,
        embed_dim=mcfg.embed_dim,
        enc_depths=tuple(mcfg.enc_depths),
        dec_depths=tuple(mcfg.dec_depths),
        num_heads=tuple(mcfg.num_heads),
        window_size=mcfg.window_size,
        proj_dim=mcfg.proj_dim,
        plane_inject_method=mcfg.plane_inject_method,
        enable_saca=mcfg.enable_saca,
        saca_position=mcfg.saca_position,
        saca_positions=mcfg.saca_positions,
        saca_gate_init=mcfg.saca_gate_init,
        saca_warmup_epochs=mcfg.saca_warmup_epochs,
        enable_reconstruct=tcfg.enable_reconstruct,
        enable_contrastive=False,
        contrastive_loss_type=cfg.contrast_loss.contrastive_loss_type,
        contrastive_position=cfg.contrast_loss.contrastive_position,
        single_view=tcfg.single_view,
    )
    return model


def _load_preprocess_meta_or_none(data_dir: str | Path) -> dict | None:
    meta_path = Path(data_dir).expanduser() / PREPROCESS_META_FILENAME
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_preprocess_meta(meta: dict, *, expected_image_size: int, data_dir: str | Path) -> None:
    image_size = meta.get("image_size")
    if not isinstance(image_size, dict):
        return
    try:
        h = int(image_size.get("height"))
        w = int(image_size.get("width"))
    except Exception:
        return

    if h != w:
        raise ValueError(
            f"Preprocessed data in {Path(data_dir).resolve()} has non-square image_size=({h}, {w}). "
            "Current mask reconstruction model expects square --image-size."
        )
    if int(expected_image_size) > 0 and int(expected_image_size) != h:
        raise ValueError(
            f"Preprocessed metadata mismatch for {Path(data_dir).resolve()}: "
            f"meta image_size={h} but runtime --image-size={expected_image_size}. "
            "Use matching --image-size or regenerate preprocessed data."
        )


def _resolve_dataset_file_pattern(
    meta: dict | None,
    *,
    image_ext_fallback: str,
    mask_suffix_fallback: str,
) -> tuple[str, str]:
    if not meta:
        return image_ext_fallback, mask_suffix_fallback
    meta_image_ext = str(meta.get("image_ext", image_ext_fallback) or image_ext_fallback)
    if meta_image_ext == "mixed_by_source":
        meta_image_ext = image_ext_fallback
    meta_mask_suffix = str(meta.get("mask_suffix", mask_suffix_fallback) or mask_suffix_fallback)
    return meta_image_ext, meta_mask_suffix


def run(args: argparse.Namespace) -> None:
    import torch

    from ..training.utils import ensure_dir, get_device
    from .dataset import MaskReconstructionDataset
    from .plotting import generate_plots
    from .trainer import MaskReconstructionTrainer
    from .utils import set_seed

    enforce_recon_only_args(args)
    # Map train_dir to data_root for shared config compatibility if not provided
    if not getattr(args, "data_root", ""):
        args.data_root = args.train_dir
    cfg = ExperimentConfig.from_args(args)
    # Hard guardrails for this task-specific entrypoint.
    cfg.training.enable_contrastive = False
    # cfg.mask.enable_masking = False
    # if bool(cfg.training.enable_contrastive) or bool(cfg.mask.enable_masking):
    #     raise ValueError("mask_reconstruction/main.py enforces reconstruction-only (no masking, no contrastive).")
    if getattr(args, "preprocessed_dir", ""):
        cfg.data.skip_resize_in_loader = True
    set_seed(int(cfg.training.seed))
    
    print("[config] Loaded experiment configuration:")
    print(cfg)

    device = get_device(cpu=bool(cfg.training.cpu))
    print(f"[device] using {device}")

    use_preprocessed_dir = bool(getattr(args, "preprocessed_dir", ""))
    preprocessed_mode = bool(cfg.data.skip_resize_in_loader)
    train_dir = args.preprocessed_dir if use_preprocessed_dir else args.train_dir
    val_dir = args.val_dir
    print(
        f"[data] loader_mode: skip_resize_in_loader={bool(cfg.data.skip_resize_in_loader)} "
        f"use_preprocessed_dir={use_preprocessed_dir} preprocessed_mode={preprocessed_mode}"
    )

    train_meta = _load_preprocess_meta_or_none(train_dir)
    if preprocessed_mode and train_meta is None:
        raise FileNotFoundError(
            f"Expected preprocessed metadata at {Path(train_dir).resolve() / PREPROCESS_META_FILENAME}. "
            "When --skip_resize_in_loader is enabled, train_dir (or --preprocessed_dir if provided) must be preprocessed."
        )
    if train_meta is not None:
        _validate_preprocess_meta(train_meta, expected_image_size=int(cfg.data.image_size), data_dir=train_dir)

    val_meta = _load_preprocess_meta_or_none(val_dir) if val_dir else None
    if preprocessed_mode and val_dir and val_meta is None:
        raise FileNotFoundError(
            f"Expected preprocessed metadata at {Path(val_dir).resolve() / PREPROCESS_META_FILENAME}. "
            "When --skip_resize_in_loader is enabled, val_dir must be a preprocessed folder."
        )
    if val_meta is not None:
        _validate_preprocess_meta(val_meta, expected_image_size=int(cfg.data.image_size), data_dir=val_dir)

    train_image_ext, train_mask_suffix = _resolve_dataset_file_pattern(
        train_meta,
        image_ext_fallback=args.image_ext,
        mask_suffix_fallback=args.mask_suffix,
    )
    val_image_ext, val_mask_suffix = _resolve_dataset_file_pattern(
        val_meta,
        image_ext_fallback=args.image_ext,
        mask_suffix_fallback=args.mask_suffix,
    )

    train_ds = MaskReconstructionDataset(
        data_dir=train_dir,
        image_ext=train_image_ext,
        mask_suffix=train_mask_suffix,
        strict_pairs=bool(args.strict_pairs),
        mask_key=(args.mask_key or None),
        image_size=int(cfg.data.image_size) if cfg.data.image_size else None,
        target_size=int(args.target_size),
        resize_mode=args.resize_mode,
        debug_shapes=bool(args.debug_shapes),
        plane=args.plane,
        binarize_target=bool(args.binarize_target),
        preprocessed=preprocessed_mode,
        skip_resize_in_loader=bool(cfg.data.skip_resize_in_loader),
    )
    val_ds = None
    if val_dir:
        val_ds = MaskReconstructionDataset(
            data_dir=val_dir,
            image_ext=val_image_ext,
            mask_suffix=val_mask_suffix,
            strict_pairs=bool(args.strict_pairs),
            mask_key=(args.mask_key or None),
            image_size=int(cfg.data.image_size) if cfg.data.image_size else None,
            target_size=int(args.target_size),
            resize_mode=args.resize_mode,
            debug_shapes=bool(args.debug_shapes),
            plane=args.plane,
            binarize_target=bool(args.binarize_target),
            preprocessed=preprocessed_mode,
            skip_resize_in_loader=bool(cfg.data.skip_resize_in_loader),
        )
    train_loader, val_loader = make_dataloaders(train_ds, cfg, device, val_ds=val_ds)

    train_pairs = len(train_loader.dataset)
    val_pairs = len(val_loader.dataset)
    print(
        f"[data] train_dir={Path(train_dir).resolve()} train_pairs={train_pairs}"
        + (
            f" | val_dir={Path(val_dir).resolve()} val_pairs={val_pairs}"
            if val_ds is not None
            else f" | val_split_ratio={cfg.data.val_ratio} val_pairs={val_pairs}"
        )
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    # Smoke snippet (manual quick check):
    # dummy_x = torch.randn(2, 1, cfg.data.image_size, cfg.data.image_size, device=device)
    # dummy_y = torch.rand_like(dummy_x)
    # dummy_plane = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    # r1, r2, _, _ = model(dummy_x, None, dummy_plane)  # dual view path
    # assert r1.shape == dummy_y.shape and (r2 is None or r2.shape == dummy_y.shape)

    out_dir = Path(cfg.logging.out_dir)
    if cfg.logging.run_name:
        out_dir = out_dir / cfg.logging.run_name
    out_dir = ensure_dir(out_dir)
    if cfg.logging.vis_every > 0:
        ensure_dir(out_dir / "vis")

    trainer = MaskReconstructionTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        out_dir=out_dir,
        cfg=cfg,
        threshold=args.threshold if hasattr(args, "threshold") else None,
        save_best_only=False,
        align_flip_target=True,
        vis_every=int(cfg.logging.vis_every),
        vis_num=int(getattr(args, "vis_n_results", args.vis_num)),
        vis_threshold=float(args.vis_threshold),
        disable_tqdm=bool(args.no_tqdm),
        train_step_dice=False,
    )
    trainer.fit(train_loader, val_loader, epochs=int(cfg.training.epochs))
    generate_plots(out_dir / "epoch_log.csv", out_dir / "plot")


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_entrypoint(build_mask_argparser, run, argv=argv)


if __name__ == "__main__":
    main()
