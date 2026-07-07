"""
main.py

Entrypoint for BraTS 2021 tumor segmentation fine-tuning.

Key differences from tissue_segmentation/main.py:
  - build_label_encoding_info called with require_special_ids=False
    (BraTS seg_labels.txt has no 'Unknown' / 'Non-Brain' entries)
  - Default label_mode=3 (contiguous remap: {0,1,2,4} -> {0,1,2,3})
  - Uses TumorSegmentationTrainer (adds WT/TC/ET region dice logging)
  - Default out_dir: autoencoder/outputs/brats2021/tumor_seg/

Usage:
    python -m autoencoder.src.ver3.tumor_segmentation.main \\
        --backbone vae \\
        --train-root /data/brats2021_2d/images \\
        --train-label /data/brats2021_2d/labels \\
        --eval-root  /data/brats2021_2d/images \\
        --eval-label /data/brats2021_2d/labels \\
        --train-list /data/brats2021_2d/train_list.txt \\
        --eval-list  /data/brats2021_2d/eval_list.txt \\
        --epochs 50 \\
        --ce-class-weights "1,5,3,8" \\
        --out-dir autoencoder/outputs/brats2021/tumor_seg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from ..common.cli_utils import run_entrypoint
from ..downstream.ckpt_load import load_pretrained_for_downstream
from ..downstream.model_utils import build_downstream_model, replace_output_channels
from ..training.utils import copy_images_to_dir, ensure_dir, extract_dataset_paths, get_device, write_path_list
from ..data.dataset import select_indices_by_train_mod
from ..tissue_segmentation.dataset import TissueSegmentationDataset
from ..tissue_segmentation.io import (
    assert_encoding_deterministic,
    build_image_index,
    build_label_encoding_info,
    identify_special_ids,
    parse_seg_labels_txt,
)
from ..tissue_segmentation.plotting import generate_plots
from .experiment import ExperimentConfig, build_argparser, enforce_tumor_args, resolve_seg_labels_path
from .scan_lists import resolve_train_eval_tokens
from .trainer import TumorSegmentationTrainer


def build_tumor_argparser() -> argparse.ArgumentParser:
    return build_argparser()


def build_model(cfg: ExperimentConfig, *, num_classes: int):
    if bool(getattr(cfg.training, "enable_contrastive", False)):
        raise ValueError("tumor_segmentation entrypoint forbids contrastive mode.")

    backbone = str(getattr(cfg.model, "backbone", "mae")).lower()
    if backbone not in {"mae", "vae"}:
        raise ValueError(f"tumor_segmentation supports backbone mae|vae, got {backbone!r}")

    model = build_downstream_model(
        cfg,
        out_ch=int(num_classes),
        enable_reconstruct=True,
        single_view=False,
    )
    replace_output_channels(model, int(num_classes))
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
    enforce_tumor_args(args)

    if not getattr(args, "data_root", ""):
        args.data_root = args.train_root

    cfg = ExperimentConfig.from_args(args)
    cfg.training.enable_contrastive = False
    cfg.training.enable_reconstruct = True
    cfg.training.single_view = False
    cfg.mask.enable_masking = False

    # Proxy tumor config onto cfg.tissue so that TissueSegmentationTrainer can
    # read standard fields (dice_include_bg, ce_class_weights, etc.)
    cfg.tissue = cfg.tumor  # type: ignore[assignment]

    print("[config] Loaded experiment configuration:")
    print(cfg)

    device = get_device(cpu=bool(cfg.training.cpu))
    print(f"[device] using {device}")

    seg_labels_path = resolve_seg_labels_path(cfg.tumor.seg_labels)
    cfg.tumor.seg_labels = seg_labels_path
    seg_labels = parse_seg_labels_txt(seg_labels_path)
    unknown_ids, non_brain_ids = identify_special_ids(seg_labels)

    print(
        f"[labels] seg_labels parsed: {len(seg_labels)} entries  "
        f"unknown_ids={sorted(unknown_ids)}  non_brain_ids={sorted(non_brain_ids)}"
    )
    if not unknown_ids:
        print("[labels] INFO: no 'unknown' ids — expected for BraTS (mode 3 does not require them)")
    if not non_brain_ids:
        print("[labels] INFO: no 'non-brain' ids — expected for BraTS (mode 3 does not require them)")

    # require_special_ids=False: BraTS has no Unknown/Non-Brain labels
    encoding_info = build_label_encoding_info(
        mode=cfg.tumor.label_mode,
        id_to_name=seg_labels,
        unknown_ids=unknown_ids,
        non_brain_ids=non_brain_ids,
        num_classes_override=cfg.tumor.num_classes,
        require_special_ids=False,
    )
    assert_encoding_deterministic(encoding_info)
    cfg.data.num_classes = int(encoding_info.num_classes)
    cfg.model.num_classes = int(encoding_info.num_classes)

    print(
        f"[labels] mode={encoding_info.mode} num_classes={encoding_info.num_classes} "
        f"encode_map={dict(encoding_info.encode_map)}"
    )

    one_token = str(getattr(cfg.tumor, "one", "")).strip()
    one_mode = bool(one_token)
    if one_mode:
        train_tokens = [one_token]
        eval_tokens = [one_token]
        cfg.tumor.eval_root = cfg.tumor.train_root
        cfg.tumor.eval_label = cfg.tumor.train_label
        train_token_count_raw = 1
    else:
        train_tokens, eval_tokens, train_list_path, eval_list_path = resolve_train_eval_tokens(
            train_list=cfg.tumor.train_list,
            eval_list=cfg.tumor.eval_list,
            image_root=cfg.tumor.train_root,
            image_ext=cfg.tumor.image_ext,
            val_ratio=float(cfg.data.val_ratio),
            seed=int(cfg.training.seed),
            write_lists=True,
        )
        cfg.tumor.train_list = train_list_path
        cfg.tumor.eval_list = eval_list_path

        train_token_count_raw = len(train_tokens)
        keep_idx = select_indices_by_train_mod(train_token_count_raw, float(cfg.data.train_mod))
        train_tokens = [train_tokens[i] for i in keep_idx]
        if not train_tokens:
            raise RuntimeError(
                f"No train scans selected after applying train_mod={cfg.data.train_mod}"
            )

    train_image_index = build_image_index(image_root=cfg.tumor.train_root, image_ext=cfg.tumor.image_ext)
    eval_root_resolved  = Path(cfg.tumor.eval_root).expanduser().resolve()
    train_root_resolved = Path(cfg.tumor.train_root).expanduser().resolve()
    if eval_root_resolved == train_root_resolved:
        eval_image_index = train_image_index
    else:
        eval_image_index = build_image_index(image_root=cfg.tumor.eval_root, image_ext=cfg.tumor.image_ext)

    ds_kwargs = dict(
        encoding_info=encoding_info,
        image_ext=cfg.tumor.image_ext,
        label_suffix=cfg.tumor.label_suffix,
        label_key=cfg.tumor.label_key or None,
        image_size=int(cfg.data.image_size) if cfg.data.image_size else None,
        target_size=int(cfg.tumor.target_size),
        resize_mode=cfg.tumor.resize_mode,
        plane=cfg.data.plane,
        strict_pairs=bool(cfg.tumor.strict_pairs),
        strict_label_ids=bool(cfg.tumor.strict_label_ids),
        allow_unknown_label_ids=bool(cfg.tumor.allow_unknown_label_ids),
        debug_shapes=bool(cfg.tumor.debug_shapes),
    )

    train_ds = TissueSegmentationDataset(
        image_root=cfg.tumor.train_root,
        label_root=cfg.tumor.train_label,
        scan_tokens=train_tokens,
        image_index=train_image_index,
        **ds_kwargs,
    )
    eval_ds = TissueSegmentationDataset(
        image_root=cfg.tumor.eval_root,
        label_root=cfg.tumor.eval_label,
        scan_tokens=eval_tokens,
        image_index=eval_image_index,
        **ds_kwargs,
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
    if one_mode:
        print(f"[one] enabled token={one_token}")
    print(
        f"[subsample] train_mod={float(cfg.data.train_mod):.4f} "
        f"train_scans={len(train_tokens)}/{train_token_count_raw}"
    )
    print(
        f"[roots] train_root={cfg.tumor.train_root} eval_root={cfg.tumor.eval_root} "
        f"train_label={cfg.tumor.train_label} eval_label={cfg.tumor.eval_label}"
    )
    print(f"[plane] mode={cfg.data.plane}")
    print(
        f"[dice] include_bg={bool(cfg.tumor.dice_include_bg)} "
        f"empty_handling={'one' if bool(cfg.tumor.dice_empty_as_one) else 'exclude'}"
    )
    print(
        f"[region_dice] enabled={bool(cfg.tumor.enable_region_dice)} "
        "(WT=classes{1,2,3} TC=classes{1,3} ET=class{3})"
    )

    out_dir = Path(cfg.logging.out_dir)
    if cfg.logging.run_name:
        out_dir = out_dir / cfg.logging.run_name
    out_dir = ensure_dir(out_dir)

    if bool(getattr(args, "dump_val_paths", False)) or bool(getattr(args, "dump_val_paths_only", False)):
        eval_paths = extract_dataset_paths(eval_loader.dataset)
        eval_path_file = write_path_list(eval_paths, out_dir / "val_paths_tumor_segmentation.txt")
        copied_n = copy_images_to_dir(eval_paths, out_dir / "val_images_tumor_segmentation")
        print(f"[val_paths] task=tumor_segmentation count={len(eval_paths)} file={eval_path_file}")
        print(f"[val_paths] copied_images={copied_n}")
        for p in eval_paths:
            print(p)
        if bool(getattr(args, "dump_val_paths_only", False)):
            print("[val_paths] dump_val_paths_only=1; exiting before training.")
            return

    model = build_model(cfg, num_classes=encoding_info.num_classes).to(device)
    load_pretrained_for_downstream(model, cfg, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    ensure_dir(out_dir / "plot")
    ensure_dir(out_dir / "reports")

    trainer = TumorSegmentationTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        out_dir=out_dir,
        cfg=cfg,
        num_classes=encoding_info.num_classes,
        class_names=encoding_info.encoded_id_to_name,
        enc_to_orig_map=encoding_info.decode_map,
        dice_include_bg=bool(cfg.tumor.dice_include_bg),
        vis_every=int(cfg.logging.vis_every),
        vis_num=min(4, int(cfg.tumor.vis_num)),
        disable_tqdm=bool(cfg.tumor.no_tqdm),
        val_every=max(1, int(cfg.tumor.val_every)),
        enable_region_dice=bool(cfg.tumor.enable_region_dice),
    )
    trainer.fit(train_loader, eval_loader, epochs=int(cfg.training.epochs))

    generate_plots(out_dir / "epoch_log.csv", out_dir / "plot")


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_entrypoint(build_tumor_argparser, run, argv=argv)


if __name__ == "__main__":
    main()
