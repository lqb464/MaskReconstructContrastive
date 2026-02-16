from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from ..config.experiment import ExperimentConfig as _BaseExperimentConfig
from ..config.experiment import build_argparser as _build_argparser


@dataclass
class TissueTaskConfig:
    image_root: str = ""
    label_root: str = ""
    train_list: str = ""
    eval_list: str = ""
    seg_labels: str = ""
    image_ext: str = ".png"
    label_suffix: str = "_label.npz"
    label_key: str = ""
    label_mode: int = 1
    num_classes: int = 0
    dice_include_bg: bool = False
    dice_empty_as_one: bool = False
    ignore_index: int = -100
    ce_class_weights: str = ""
    target_size: int = 0
    resize_mode: str = "letterbox"
    strict_pairs: bool = False
    val_every: int = 1
    vis_num: int = 4
    vis_threshold: float = 0.5
    no_tqdm: bool = False
    debug_shapes: bool = False


@dataclass
class ExperimentConfig(_BaseExperimentConfig):
    tissue: TissueTaskConfig = field(default_factory=TissueTaskConfig)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        base = _BaseExperimentConfig.from_args(args)
        tissue = TissueTaskConfig(
            image_root=args.image_root,
            label_root=args.label_root,
            train_list=args.train_list,
            eval_list=args.eval_list,
            seg_labels=args.seg_labels,
            image_ext=args.image_ext,
            label_suffix=args.label_suffix,
            label_key=getattr(args, "label_key", ""),
            label_mode=int(args.label_mode),
            num_classes=int(args.num_classes),
            dice_include_bg=bool(args.dice_include_bg),
            dice_empty_as_one=bool(args.dice_empty_as_one),
            ignore_index=int(getattr(args, "ignore_index", -100)),
            ce_class_weights=str(getattr(args, "ce_class_weights", "")),
            target_size=int(args.target_size),
            resize_mode=str(args.resize_mode),
            strict_pairs=bool(args.strict_pairs),
            val_every=max(1, int(args.val_every)),
            vis_num=max(1, int(args.vis_num)),
            vis_threshold=float(args.vis_threshold),
            no_tqdm=bool(args.no_tqdm),
            debug_shapes=bool(args.debug_shapes),
        )
        return cls(
            model=base.model,
            training=base.training,
            data=base.data,
            mask=base.mask,
            logging=base.logging,
            contrast_loss=base.contrast_loss,
            tissue=tissue,
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = _build_argparser()

    parser.set_defaults(enable_contrastive=False, enable_masking=False, enable_reconstruct=True)

    grp = parser.add_argument_group("tissue_segmentation dataset")
    grp.add_argument("--image-root", type=str, required=True, help="Root folder containing input images.")
    grp.add_argument("--label-root", type=str, required=True, help="Root folder containing segmentation labels.")
    grp.add_argument("--train-list", type=str, required=True, help="Path to scans list for train split.")
    grp.add_argument("--eval-list", type=str, required=True, help="Path to scans list for eval/test split.")
    grp.add_argument(
        "--seg-labels",
        type=str,
        default=str((Path(__file__).resolve().parent / "txt" / "seg_labels.txt")),
        help="Path to seg_labels.txt.",
    )

    grp.add_argument("--image-ext", type=str, default=".png", help="Image extension for inputs.")
    grp.add_argument("--label-suffix", type=str, default="_label.npz", help="Label filename suffix.")
    grp.add_argument("--label-key", type=str, default="", help="Optional key inside NPZ label file.")

    grp.add_argument("--label-mode", type=int, choices=[1, 2, 3, 4], default=1, help="Label encoding mode.")
    grp.add_argument("--num-classes", type=int, default=0, help="Optional override for output classes (0=infer).")
    grp.add_argument("--dice-include-bg", action="store_true", help="Include background class in macro dice.")
    grp.add_argument("--no-dice-include-bg", dest="dice_include_bg", action="store_false")
    grp.add_argument(
        "--dice-empty-as-one",
        action="store_true",
        help="When a class has zero denominator for the whole epoch, treat its Dice as 1.0 (instead of excluding).",
    )
    grp.add_argument("--no-dice-empty-as-one", dest="dice_empty_as_one", action="store_false")
    grp.add_argument(
        "--ignore-index",
        type=int,
        default=-100,
        help="Explicit ignore_index used by CrossEntropyLoss (default -100).",
    )
    grp.add_argument(
        "--ce-class-weights",
        type=str,
        default="",
        help="Optional comma-separated class weights for CrossEntropyLoss (length must equal num_classes).",
    )

    grp.add_argument("--target-size", type=int, default=0, help="Force square resize to this size (0 uses --image-size).")
    grp.add_argument(
        "--resize-mode",
        type=str,
        default="letterbox",
        choices=["letterbox", "direct"],
        help="Resize strategy for image/label pair.",
    )
    grp.add_argument("--strict-pairs", type=int, default=0, help="1: error if image has no label; 0: filter out.")
    grp.add_argument("--val-every", type=int, default=1, help="Run eval every N epochs.")
    grp.add_argument("--vis-num", type=int, default=4, help="Number of eval samples to visualize.")
    grp.add_argument("--vis-threshold", type=float, default=0.5, help="Reserved visualization threshold argument.")
    grp.add_argument("--no-tqdm", type=int, default=0, help="Disable progress bars.")
    grp.add_argument("--debug-shapes", type=int, default=0, help="Log sample tensor shapes for debugging.")
    parser.set_defaults(dice_include_bg=False, dice_empty_as_one=False)

    # Keep base parser compatibility: make data-root optional and bind it from image-root in main.
    for action in parser._actions:
        if action.dest == "data_root":
            action.required = False
            if action.default is None:
                action.default = ""

    return parser


def enforce_tissue_args(args: argparse.Namespace) -> None:
    if bool(getattr(args, "enable_contrastive", False)):
        raise ValueError(
            "tissue_segmentation task is supervised segmentation-only. "
            "Disable contrastive with --disable-contrastive."
        )
    if int(getattr(args, "label_mode", 0)) not in {1, 2, 3, 4}:
        raise ValueError("--label-mode must be one of {1,2,3,4}")


__all__ = ["ExperimentConfig", "TissueTaskConfig", "build_argparser", "enforce_tissue_args"]
