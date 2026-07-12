from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from ..common.cli_utils import run_entrypoint
from ..models.swin_unet_dualview_ssl import SwinUNetDualViewSSL
from ..models.unet_dualview_ssl import UNetDualViewSSL
from ..training.utils import copy_images_to_dir, ensure_dir, extract_dataset_paths, get_device, write_path_list
from ..data.dataset import select_indices_by_train_mod
from .dataset import TissueSegmentationDataset
from .experiment import ExperimentConfig, build_argparser, enforce_tissue_args
from .io import (
    assert_encoding_deterministic,
    build_image_index,
    build_label_encoding_info,
    identify_special_ids,
    parse_seg_labels_txt,
    read_scan_list,
)
from .plotting import generate_plots
from .trainer import TissueSegmentationTrainer

def build_tissue_argparser() -> argparse.ArgumentParser:
    return build_argparser()

def _replace_recon_head_out_channels(model: SwinUNetDualViewSSL, num_classes: int) -> None:
    import torch.nn as nn

    if int(num_classes) < 2:
        raise ValueError(f"num_classes must be >=2, got {num_classes}")

    for attr in ("recon_head_v1", "recon_head_v2"):
        head = getattr(model, attr, None)
        if head is None:
            raise RuntimeError(f"Model is missing {attr}; enable_reconstruct must be True.")
        if not isinstance(head, nn.Sequential) or len(head) < 1 or not isinstance(head[-1], nn.Conv2d):
            raise RuntimeError(f"Unexpected {attr} structure: expected nn.Sequential(..., Conv2d)")

        last_conv = head[-1]
        if int(last_conv.out_channels) == int(num_classes):
            continue
        head[-1] = nn.Conv2d(last_conv.in_channels, int(num_classes), kernel_size=1)

def build_model(cfg: ExperimentConfig, *, num_classes: int):
    mcfg = cfg.model
    tcfg = cfg.training

    if bool(getattr(tcfg, "enable_contrastive", False)):
        raise ValueError("tissue_segmentation entrypoint forbids contrastive mode.")

    single_view = bool(getattr(tcfg, "single_view", False))
    if single_view and bool(getattr(mcfg, "enable_saca", False)):
        raise ValueError(
            "tissue_segmentation: --single-view is incompatible with SACA. "
            "Disable SACA or use dual-view."
        )

    backbone = str(getattr(mcfg, "backbone", "swin")).lower()
    if backbone == "unet":
        return UNetDualViewSSL(
            in_ch=mcfg.in_ch,
            base_ch=int(getattr(mcfg, "unet_base_ch", 16)),
            out_ch=int(num_classes),
            use_gn=bool(getattr(mcfg, "unet_use_gn", False)),
            use_se=bool(getattr(mcfg, "unet_use_se", False)),
            enable_reconstruct=True,
            enable_contrastive=False,
            single_view=single_view,
        )

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
        enable_reconstruct=True,
        enable_contrastive=False,
        contrastive_loss_type=cfg.contrast_loss.contrastive_loss_type,
        contrastive_position=cfg.contrast_loss.contrastive_position,
        single_view=single_view,
    )
    _replace_recon_head_out_channels(model, num_classes=num_classes)
    return model

def make_dataloaders(
    train_ds: TissueSegmentationDataset,
    eval_ds: TissueSegmentationDataset,
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    def _loader(dataset, shuffle: bool) -> DataLoader:
        extra = {}
        if cfg.data.num_workers > 0:
            extra["persistent_workers"] = True
            extra["prefetch_factor"] = 2

        return DataLoader(
            dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=shuffle,
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory) and device.type == "cuda",
            drop_last=bool(cfg.data.drop_last) if shuffle else False,
            **extra,
        )

    return _loader(train_ds, shuffle=True), _loader(eval_ds, shuffle=False)

def run(args: argparse.Namespace) -> None:
    enforce_tissue_args(args)

    if not getattr(args, "data_root", ""):
        args.data_root = args.train_root

    cfg = ExperimentConfig.from_args(args)
    cfg.training.enable_contrastive = False
    cfg.training.enable_reconstruct = True
    cfg.mask.enable_masking = False
    # Honor --single-view / --dual-view from CLI (do not force dual-view).

    print("[config] Loaded experiment configuration:")
    print(cfg)
    print(f"[view] mode={'single' if bool(cfg.training.single_view) else 'dual'}")

    device = get_device(cpu=bool(cfg.training.cpu))
    print(f"[device] using {device}")

    seg_labels = parse_seg_labels_txt(cfg.tissue.seg_labels)
    unknown_ids, non_brain_ids = identify_special_ids(seg_labels)
    if not unknown_ids:
        print("[labels] WARNING: no 'unknown' ids detected from seg_labels names")
    if not non_brain_ids:
        print("[labels] WARNING: no 'non brain' ids detected from seg_labels names")

    encoding_info = build_label_encoding_info(
        mode=cfg.tissue.label_mode,
        id_to_name=seg_labels,
        unknown_ids=unknown_ids,
        non_brain_ids=non_brain_ids,
        num_classes_override=cfg.tissue.num_classes,
        require_special_ids=True,
    )
    assert_encoding_deterministic(encoding_info)
    cfg.data.num_classes = int(encoding_info.num_classes)
    cfg.model.num_classes = int(encoding_info.num_classes)

    one_token = str(getattr(cfg.tissue, "one", "")).strip()
    one_mode = bool(one_token)
    if one_mode:
        train_tokens = [one_token]
        eval_tokens = [one_token]
        cfg.tissue.eval_root = cfg.tissue.train_root
        cfg.tissue.eval_label = cfg.tissue.train_label
        train_token_count_raw = 1
    else:
        train_tokens = read_scan_list(cfg.tissue.train_list)
        eval_tokens = read_scan_list(cfg.tissue.eval_list)
        if not train_tokens:
            raise RuntimeError(f"Train list has no usable scan tokens: {cfg.tissue.train_list}")
        if not eval_tokens:
            raise RuntimeError(f"Eval list has no usable scan tokens: {cfg.tissue.eval_list}")

        train_token_count_raw = len(train_tokens)
        keep_idx = select_indices_by_train_mod(train_token_count_raw, float(cfg.data.train_mod))
        train_tokens = [train_tokens[i] for i in keep_idx]
        if not train_tokens:
            raise RuntimeError(
                f"No train scans selected after applying train_mod={cfg.data.train_mod} "
                f"on train_list={cfg.tissue.train_list}"
            )

    train_image_index = build_image_index(image_root=cfg.tissue.train_root, image_ext=cfg.tissue.image_ext)
    eval_root_resolved = Path(cfg.tissue.eval_root).expanduser().resolve()
    train_root_resolved = Path(cfg.tissue.train_root).expanduser().resolve()
    if eval_root_resolved == train_root_resolved:
        eval_image_index = train_image_index
    else:
        eval_image_index = build_image_index(image_root=cfg.tissue.eval_root, image_ext=cfg.tissue.image_ext)

    train_ds = TissueSegmentationDataset(
        image_root=cfg.tissue.train_root,
        label_root=cfg.tissue.train_label,
        scan_tokens=train_tokens,
        encoding_info=encoding_info,
        image_ext=cfg.tissue.image_ext,
        label_suffix=cfg.tissue.label_suffix,
        label_key=cfg.tissue.label_key or None,
        image_size=int(cfg.data.image_size) if cfg.data.image_size else None,
        target_size=int(cfg.tissue.target_size),
        resize_mode=cfg.tissue.resize_mode,
        plane=cfg.data.plane,
        strict_pairs=bool(cfg.tissue.strict_pairs),
        strict_label_ids=bool(cfg.tissue.strict_label_ids),
        allow_unknown_label_ids=bool(cfg.tissue.allow_unknown_label_ids),
        debug_shapes=bool(cfg.tissue.debug_shapes),
        image_index=train_image_index,
    )
    eval_ds = TissueSegmentationDataset(
        image_root=cfg.tissue.eval_root,
        label_root=cfg.tissue.eval_label,
        scan_tokens=eval_tokens,
        encoding_info=encoding_info,
        image_ext=cfg.tissue.image_ext,
        label_suffix=cfg.tissue.label_suffix,
        label_key=cfg.tissue.label_key or None,
        image_size=int(cfg.data.image_size) if cfg.data.image_size else None,
        target_size=int(cfg.tissue.target_size),
        resize_mode=cfg.tissue.resize_mode,
        plane=cfg.data.plane,
        strict_pairs=bool(cfg.tissue.strict_pairs),
        strict_label_ids=bool(cfg.tissue.strict_label_ids),
        allow_unknown_label_ids=bool(cfg.tissue.allow_unknown_label_ids),
        debug_shapes=bool(cfg.tissue.debug_shapes),
        image_index=eval_image_index,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty after list filtering and label pairing.")
    if len(eval_ds) == 0:
        raise RuntimeError("Eval dataset is empty after list filtering and label pairing.")

    train_loader, eval_loader = make_dataloaders(train_ds, eval_ds, cfg, device)

    print(f"[data] train_pairs={len(train_ds)} eval_pairs={len(eval_ds)}")
    print(
        "[data/train] "
        f"resolved_images={train_ds.num_images_resolved} "
        f"labeled_samples={train_ds.num_labeled_samples} "
        f"filtered_missing_labels={train_ds.num_missing_labels}"
    )
    print(
        "[data/eval] "
        f"resolved_images={eval_ds.num_images_resolved} "
        f"labeled_samples={eval_ds.num_labeled_samples} "
        f"filtered_missing_labels={eval_ds.num_missing_labels}"
    )
    print(
        f"[split] fixed lists (no random split): train_list={cfg.tissue.train_list} "
        f"eval_list={cfg.tissue.eval_list}"
    )
    if one_mode:
        print(
            f"[one] enabled token={one_token} "
            f"train_root={cfg.tissue.train_root} eval_root={cfg.tissue.eval_root}"
        )
    print(
        f"[subsample] train_mod={float(cfg.data.train_mod):.4f} "
        f"train_scans={len(train_tokens)}/{train_token_count_raw} (eval unchanged)"
    )
    print(
        f"[roots] train_root={cfg.tissue.train_root} eval_root={cfg.tissue.eval_root} "
        f"train_label={cfg.tissue.train_label} eval_label={cfg.tissue.eval_label}"
    )
    print(f"[plane] mode={cfg.data.plane} (auto infers from image path tokens: axial/coronal)")
    print(
        f"[labels] mode={encoding_info.mode} num_classes={encoding_info.num_classes} "
        f"unknown_ids={sorted(encoding_info.unknown_ids)} non_brain_ids={sorted(encoding_info.non_brain_ids)}"
    )
    print(
        f"[labels] strict_label_ids={bool(cfg.tissue.strict_label_ids)} "
        f"allow_unknown_label_ids={bool(cfg.tissue.allow_unknown_label_ids)}"
    )
    print(
        f"[dice] include_bg={bool(cfg.tissue.dice_include_bg)} "
        f"empty_handling={'one' if bool(cfg.tissue.dice_empty_as_one) else 'exclude'}"
    )

    out_dir = Path(cfg.logging.out_dir)
    if cfg.logging.run_name:
        out_dir = out_dir / cfg.logging.run_name
    out_dir = ensure_dir(out_dir)

    if bool(getattr(args, "dump_val_paths", False)) or bool(getattr(args, "dump_val_paths_only", False)):
        eval_paths = extract_dataset_paths(eval_loader.dataset)
        eval_path_file = write_path_list(eval_paths, out_dir / "val_paths_tissue_segmentation.txt")
        copied_n = copy_images_to_dir(eval_paths, out_dir / "val_images_tissue_segmentation")
        print(f"[val_paths] task=tissue_segmentation count={len(eval_paths)} file={eval_path_file}")
        print(
            f"[val_paths] copied_images={copied_n} "
            f"dir={out_dir / 'val_images_tissue_segmentation'}"
        )
        for p in eval_paths:
            print(p)
        if bool(getattr(args, "dump_val_paths_only", False)):
            print("[val_paths] dump_val_paths_only=1; exiting before training.")
            return

    model = build_model(cfg, num_classes=encoding_info.num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    plot_dir = ensure_dir(out_dir / "plot")
    ensure_dir(out_dir / "reports")

    trainer = TissueSegmentationTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        out_dir=out_dir,
        cfg=cfg,
        num_classes=encoding_info.num_classes,
        class_names=encoding_info.encoded_id_to_name,
        enc_to_orig_map=encoding_info.decode_map,
        dice_include_bg=bool(cfg.tissue.dice_include_bg),
        vis_every=int(cfg.logging.vis_every),
        vis_num=min(4, int(cfg.tissue.vis_num)),
        disable_tqdm=bool(cfg.tissue.no_tqdm),
        val_every=max(1, int(cfg.tissue.val_every)),
    )
    trainer.fit(train_loader, eval_loader, epochs=int(cfg.training.epochs))

    generate_plots(out_dir / "epoch_log.csv", plot_dir)

def main(argv: Optional[Sequence[str]] = None) -> None:
    run_entrypoint(build_tissue_argparser, run, argv=argv)

if __name__ == "__main__":
    main()
