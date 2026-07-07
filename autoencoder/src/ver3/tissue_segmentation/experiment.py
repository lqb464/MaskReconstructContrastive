from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from ..config.experiment import ExperimentConfig as _BaseExperimentConfig
from ..config.experiment import build_argparser as _build_argparser

@dataclass
class TissueTaskConfig:
    train_root: str = ""
    eval_root: str = ""
    train_label: str = ""
    eval_label: str = ""
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
    strict_label_ids: bool = False
    allow_unknown_label_ids: bool = True
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
    one: str = ""
    primary_metric: str = "pc_macro_dice"
    presence_policy: str = "target_present"
    aggregation_level: str = "scan"

@dataclass
class ExperimentConfig(_BaseExperimentConfig):
    tissue: TissueTaskConfig = field(default_factory=TissueTaskConfig)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        base = _BaseExperimentConfig.from_args(args)
        tissue = TissueTaskConfig(
            train_root=args.train_root,
            eval_root=args.eval_root,
            train_label=args.train_label,
            eval_label=args.eval_label,
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
            strict_label_ids=bool(getattr(args, "strict_label_ids", False)),
            allow_unknown_label_ids=bool(getattr(args, "allow_unknown_label_ids", True)),
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
            one=str(getattr(args, "one", "")).strip(),
            primary_metric=str(getattr(args, "primary_metric", "pc_macro_dice")),
            presence_policy=str(getattr(args, "presence_policy", "target_present")),
            aggregation_level=str(getattr(args, "aggregation_level", "scan")),
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

    parser.set_defaults(enable_contrastive=False, enable_masking=False, enable_reconstruct=True, plane="auto")

    grp = parser.add_argument_group("tissue_segmentation dataset")
    grp.add_argument("--train-root", "--train-dir", dest="train_root", type=str, default="", help="Root folder containing train input images.")
    grp.add_argument("--eval-root", "--eval-dir", dest="eval_root", type=str, default="", help="Root folder containing eval input images.")
    grp.add_argument("--train-label", type=str, default="", help="Root folder containing train segmentation labels.")
    grp.add_argument("--eval-label", type=str, default="", help="Root folder containing eval segmentation labels.")
    grp.add_argument("--train-list", type=str, default="", help="Path to scans list for train split.")
    grp.add_argument("--eval-list", type=str, default="", help="Path to scans list for eval/test split.")
    grp.add_argument(
        "--one",
        type=str,
        default="",
        help=(
            "Single-image mode: train/eval on one scan token or path. "
            "This sample is resolved from --train-root and labeled from --train-label. "
            "When set, eval root/label/list are optional and default to train root/label."
        ),
    )
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
        "--strict-label-ids",
        action="store_true",
        help="Require every id in label files to exist in seg_labels mapping (default: true).",
    )
    grp.add_argument("--no-strict-label-ids", dest="strict_label_ids", action="store_false")
    grp.add_argument(
        "--allow-unknown-label-ids",
        action="store_true",
        help="When strict id checks are disabled, map unknown ids to class 0.",
    )
    grp.add_argument("--no-allow-unknown-label-ids", dest="allow_unknown_label_ids", action="store_false")
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
    grp.add_argument(
        "--primary-metric",
        type=str,
        default="pc_macro_dice",
        choices=["pc_macro_dice", "macro_dice"],
        help="Primary metric used for checkpoint selection and summary logging.",
    )
    grp.add_argument(
        "--presence-policy",
        type=str,
        default="target_present",
        choices=["target_present", "all"],
        help="Class presence policy for PC-MDice reduction.",
    )
    grp.add_argument(
        "--aggregation-level",
        type=str,
        default="scan",
        choices=["scan", "epoch"],
        help="Presence aggregation level (currently target-presence reduction scope).",
    )
    parser.set_defaults(
        dice_include_bg=False,
        dice_empty_as_one=False,
        strict_label_ids=False,
        allow_unknown_label_ids=True,
    )

    for action in parser._actions:
        if action.dest == "data_root":
            action.required = False
            if action.default is None:
                action.default = ""

    return parser

def enforce_tissue_args(args: argparse.Namespace) -> None:
    one_token = str(getattr(args, "one", "")).strip()
    one_mode = bool(one_token)

    if not str(getattr(args, "train_root", "")).strip():
        raise ValueError("--train-root is required.")
    if not str(getattr(args, "train_label", "")).strip():
        raise ValueError("--train-label is required.")

    if one_mode:
        if not str(getattr(args, "eval_root", "")).strip():
            args.eval_root = args.train_root
        if not str(getattr(args, "eval_label", "")).strip():
            args.eval_label = args.train_label
    else:
        if not str(getattr(args, "eval_root", "")).strip():
            raise ValueError("--eval-root is required when --one is not set.")
        if not str(getattr(args, "eval_label", "")).strip():
            raise ValueError("--eval-label is required when --one is not set.")
        if not str(getattr(args, "train_list", "")).strip():
            raise ValueError("--train-list is required when --one is not set.")
        if not str(getattr(args, "eval_list", "")).strip():
            raise ValueError("--eval-list is required when --one is not set.")

    if bool(getattr(args, "enable_contrastive", False)):
        raise ValueError(
            "tissue_segmentation task is supervised segmentation-only. "
            "Disable contrastive with --disable-contrastive."
        )
    if not bool(getattr(args, "enable_reconstruct", True)):
        raise ValueError(
            "tissue_segmentation task requires reconstruction path enabled. "
            "Remove --disable-reconstruct (or pass --enable-reconstruct)."
        )
    if int(getattr(args, "label_mode", 0)) not in {1, 2, 3, 4}:
        raise ValueError("--label-mode must be one of {1,2,3,4}")
    if bool(getattr(args, "strict_label_ids", False)) and bool(getattr(args, "allow_unknown_label_ids", True)):
        raise ValueError(
            "--allow-unknown-label-ids cannot be used while strict label id checking is enabled. "
            "Use --no-strict-label-ids together with --allow-unknown-label-ids."
        )
    if str(getattr(args, "primary_metric", "pc_macro_dice")) not in {"pc_macro_dice", "macro_dice"}:
        raise ValueError("--primary-metric must be one of {pc_macro_dice,macro_dice}")
    if str(getattr(args, "presence_policy", "target_present")) not in {"target_present", "all"}:
        raise ValueError("--presence-policy must be one of {target_present,all}")
    if str(getattr(args, "aggregation_level", "scan")) not in {"scan", "epoch"}:
        raise ValueError("--aggregation-level must be one of {scan,epoch}")

__all__ = ["ExperimentConfig", "TissueTaskConfig", "build_argparser", "enforce_tissue_args"]
