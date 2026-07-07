from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from ..config.experiment import ExperimentConfig as _BaseExperimentConfig
from ..config.experiment import build_argparser as _build_argparser
from ..tissue_segmentation.experiment import TissueTaskConfig

_MODULE_DIR = Path(__file__).resolve().parent
_BUNDLED_SEG_LABELS = _MODULE_DIR / "txt" / "seg_labels.txt"

# Embedded fallback so training works even when txt/ is not shipped with the repo.
_DEFAULT_BRATS_SEG_LABELS_TEXT = """\
#No. Label Name:                            R   G   B   A
# BraTS 2021 segmentation labels.
# Original IDs: 0 (background), 1 (necrotic core), 2 (edema), 4 (enhancing tumor).
# Note: ID 3 is intentionally absent. Use label_mode=3 for contiguous remapping:
#   0->0, 1->1, 2->2, 4->3  =>  4 classes total.
#
# BraTS standard evaluation regions (using encoded class IDs after mode-3 mapping):
#   WT (Whole Tumor)   = classes {1, 2, 3}  (original BraTS labels {1, 2, 4})
#   TC (Tumor Core)    = classes {1, 3}     (original BraTS labels {1, 4})
#   ET (Enhancing Tumor) = class {3}        (original BraTS label  {4})

0   Background        0   0   0   0
1   Necrotic-Core   128   0   0   0
2   Edema             0 128   0   0
4   Enhancing-Tumor 255 128   0   0
"""


def bundled_seg_labels_path() -> Path:
    """Return the path to the bundled BraTS seg_labels.txt."""
    return _BUNDLED_SEG_LABELS


def ensure_bundled_seg_labels() -> Path:
    """Create the bundled seg_labels.txt from embedded defaults if missing."""
    path = bundled_seg_labels_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_BRATS_SEG_LABELS_TEXT, encoding="utf-8")
        print(f"[labels] created bundled seg_labels.txt at {path}")
    return path


def resolve_seg_labels_path(path: str | Path) -> str:
    """
    Resolve seg_labels.txt path for tumor segmentation.

    If the requested path is the bundled default but the file is missing
    (e.g. txt/ was not committed to git), auto-create it from embedded content.
    """
    requested = Path(path).expanduser()
    bundled = bundled_seg_labels_path()

    if requested.exists():
        return str(requested.resolve())

    requested_resolved = requested.resolve()
    bundled_resolved = bundled.resolve()
    is_bundled_default = (
        requested_resolved == bundled_resolved
        or (requested.name == "seg_labels.txt" and requested.parent.name == "txt")
    )
    if is_bundled_default:
        resolved = ensure_bundled_seg_labels()
        if requested_resolved != bundled_resolved:
            print(
                f"[labels] seg_labels not found at {requested}; "
                f"using bundled default {resolved}"
            )
        return str(resolved)

    raise FileNotFoundError(
        f"seg_labels.txt not found: {requested}. "
        f"Provide --seg-labels or use the bundled default at {bundled}."
    )


@dataclass
class TumorTaskConfig(TissueTaskConfig):
    """
    Configuration for BraTS 2021 tumor segmentation.

    Inherits all fields from TissueTaskConfig and overrides defaults for BraTS:
      - label_mode=3  (contiguous remap, no 'unknown'/'non-brain' special IDs required)
      - num_classes=4 (background, necrotic core, edema, enhancing tumor)
      - require_special_ids=False (BraTS has no 'Unknown'/'Non-Brain' labels)
      - enable_region_dice=True  (compute WT/TC/ET region-level dice)
    """

    require_special_ids: bool = False
    enable_region_dice: bool = True


@dataclass
class ExperimentConfig(_BaseExperimentConfig):
    tumor: TumorTaskConfig = field(default_factory=TumorTaskConfig)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        base = _BaseExperimentConfig.from_args(args)
        tumor = TumorTaskConfig(
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
            require_special_ids=bool(getattr(args, "require_special_ids", False)),
            enable_region_dice=bool(getattr(args, "enable_region_dice", True)),
        )
        return cls(
            model=base.model,
            training=base.training,
            data=base.data,
            mask=base.mask,
            logging=base.logging,
            contrast_loss=base.contrast_loss,
            tumor=tumor,
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = _build_argparser()

    parser.set_defaults(
        enable_contrastive=False,
        enable_masking=False,
        enable_reconstruct=True,
        plane="axial",
    )

    grp = parser.add_argument_group("tumor_segmentation dataset")
    grp.add_argument("--train-root", "--train-dir", dest="train_root", type=str, default="", help="Root folder containing train input images.")
    grp.add_argument("--eval-root", "--eval-dir", dest="eval_root", type=str, default="", help="Root folder containing eval input images.")
    grp.add_argument("--train-label", type=str, default="", help="Root folder containing train segmentation labels.")
    grp.add_argument("--eval-label", type=str, default="", help="Root folder containing eval segmentation labels.")
    grp.add_argument("--train-list", type=str, default="", help="Path to scan list for train split. If missing, patients are auto-discovered from --train-root.")
    grp.add_argument("--eval-list", type=str, default="", help="Path to scan list for eval/test split. If missing, patients are auto-discovered/split from --train-root.")
    grp.add_argument(
        "--one",
        type=str,
        default="",
        help="Single-image mode: train/eval on one scan token or path.",
    )
    grp.add_argument(
        "--seg-labels",
        type=str,
        default=str(_BUNDLED_SEG_LABELS),
        help="Path to seg_labels.txt. Defaults to the BraTS 2021 label file bundled with this module.",
    )
    grp.add_argument("--image-ext", type=str, default=".png", help="Image extension.")
    grp.add_argument("--label-suffix", type=str, default="_label.npz", help="Label filename suffix.")
    grp.add_argument("--label-key", type=str, default="", help="Optional key inside NPZ label file.")
    grp.add_argument(
        "--label-mode",
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
        help=(
            "Label encoding mode. Default=3 (contiguous remap, no special-id merging). "
            "BraTS IDs {0,1,2,4} are remapped to {0,1,2,3}."
        ),
    )
    grp.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of output classes. Default=4 for BraTS 2021 (bg + NCR + edema + ET).",
    )
    grp.add_argument("--dice-include-bg", action="store_true", help="Include background in macro Dice.")
    grp.add_argument("--no-dice-include-bg", dest="dice_include_bg", action="store_false")
    grp.add_argument("--dice-empty-as-one", action="store_true")
    grp.add_argument("--no-dice-empty-as-one", dest="dice_empty_as_one", action="store_false")
    grp.add_argument("--strict-label-ids", action="store_true")
    grp.add_argument("--no-strict-label-ids", dest="strict_label_ids", action="store_false")
    grp.add_argument("--allow-unknown-label-ids", action="store_true")
    grp.add_argument("--no-allow-unknown-label-ids", dest="allow_unknown_label_ids", action="store_false")
    grp.add_argument("--ignore-index", type=int, default=-100)
    grp.add_argument(
        "--ce-class-weights",
        type=str,
        default="",
        help=(
            "Comma-separated CE class weights. BraTS has heavy class imbalance; "
            "e.g. '1,5,3,8' upweights tumor classes."
        ),
    )
    grp.add_argument("--target-size", type=int, default=0)
    grp.add_argument(
        "--resize-mode",
        type=str,
        default="letterbox",
        choices=["letterbox", "direct"],
    )
    grp.add_argument("--strict-pairs", type=int, default=0)
    grp.add_argument("--val-every", type=int, default=1)
    grp.add_argument("--vis-num", type=int, default=4)
    grp.add_argument("--vis-threshold", type=float, default=0.5)
    grp.add_argument("--no-tqdm", type=int, default=0)
    grp.add_argument("--debug-shapes", type=int, default=0)
    grp.add_argument(
        "--primary-metric",
        type=str,
        default="pc_macro_dice",
        choices=["pc_macro_dice", "macro_dice"],
    )
    grp.add_argument(
        "--presence-policy",
        type=str,
        default="target_present",
        choices=["target_present", "all"],
    )
    grp.add_argument(
        "--aggregation-level",
        type=str,
        default="scan",
        choices=["scan", "epoch"],
    )
    grp.add_argument(
        "--enable-region-dice",
        action="store_true",
        default=True,
        help="Compute BraTS region dice (WT/TC/ET) during evaluation (default: enabled).",
    )
    grp.add_argument(
        "--no-region-dice",
        dest="enable_region_dice",
        action="store_false",
        help="Disable BraTS region dice computation.",
    )
    grp.add_argument(
        "--require-special-ids",
        action="store_true",
        default=False,
        help="Require 'unknown'/'non-brain' label IDs in seg_labels.txt (not needed for BraTS).",
    )

    parser.set_defaults(
        dice_include_bg=False,
        dice_empty_as_one=False,
        strict_label_ids=False,
        allow_unknown_label_ids=True,
        enable_region_dice=True,
        require_special_ids=False,
    )

    for action in parser._actions:
        if action.dest == "data_root":
            action.required = False
            if action.default is None:
                action.default = ""

    return parser


def enforce_tumor_args(args: argparse.Namespace) -> None:
    """Validate required arguments for tumor segmentation."""
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
        # train_list / eval_list are optional: missing files are auto-discovered from images.

    if bool(getattr(args, "enable_contrastive", False)):
        raise ValueError(
            "tumor_segmentation task is supervised segmentation-only. "
            "Disable contrastive with --disable-contrastive."
        )
    if not bool(getattr(args, "enable_reconstruct", True)):
        raise ValueError(
            "tumor_segmentation task requires reconstruction path enabled. "
            "Remove --disable-reconstruct."
        )
    label_mode = int(getattr(args, "label_mode", 3))
    if label_mode not in {1, 2, 3, 4}:
        raise ValueError("--label-mode must be one of {1,2,3,4}")
    if label_mode in {1, 2}:
        print(
            f"[labels] WARNING: label_mode={label_mode} requires 'unknown' and 'non-brain' label IDs. "
            "BraTS 2021 seg_labels.txt does not define these. "
            "Consider using --label-mode 3 (default) instead."
        )


__all__ = [
    "ExperimentConfig",
    "TumorTaskConfig",
    "build_argparser",
    "enforce_tumor_args",
    "bundled_seg_labels_path",
    "ensure_bundled_seg_labels",
    "resolve_seg_labels_path",
]
