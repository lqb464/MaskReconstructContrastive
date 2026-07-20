"""
scan_lists.py

Helpers for resolving BraTS train/eval scan token lists.

If train_list.txt / eval_list.txt are missing (common when only images/labels
were copied without running prepare_brats2021.py), patient tokens are
auto-discovered from image filenames and split by val_ratio.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ..tissue_segmentation.io import read_scan_list

_BRATS_SLICE_RE = re.compile(r"^(.+)_z\d+$", re.IGNORECASE)
# Match modality suffix; t1ce before t1 so "…_t1ce" is not split as "…_t1".
_MODALITY_SUFFIX_RE = re.compile(r"_(t1ce|flair|t1|t2)$", re.IGNORECASE)
ALL_MODALITIES: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair")


def patient_token_from_stem(stem: str) -> str:
    """
    Extract patient-level token from a 2D slice stem.

    Supports:
      - BraTS2021_00000_z0080       -> BraTS2021_00000
      - BraTS2021_00000_flair_z0080 -> BraTS2021_00000
    Falls back to the full stem when the pattern does not match.
    """
    match = _BRATS_SLICE_RE.match(stem)
    if not match:
        return stem
    token = match.group(1)
    patient, _mod = split_patient_and_modality(token)
    return patient


def split_patient_and_modality(token: str) -> Tuple[str, Optional[str]]:
    """Split 'BraTS2021_00000_flair' -> ('BraTS2021_00000', 'flair')."""
    raw = str(token).strip()
    if not raw:
        return "", None
    match = _MODALITY_SUFFIX_RE.search(raw)
    if match:
        return raw[: match.start()], match.group(1).lower()
    return raw, None


def parse_modality_arg(modality_arg: str | None) -> Optional[List[str]]:
    """
    Parse --modality CLI value.

    Returns:
      None  -> keep all modalities (no token rewrite)
      list  -> ordered unique modalities to keep/expand to
    """
    raw = str(modality_arg or "").strip().lower()
    if not raw or raw in {"all", "*"}:
        return None
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    unknown = [p for p in parts if p not in ALL_MODALITIES]
    if unknown:
        raise ValueError(
            f"Unknown --modality value(s): {unknown}. "
            f"Choose from {list(ALL_MODALITIES)}, comma-separated list, or 'all'."
        )
    seen: set[str] = set()
    ordered: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def apply_modality_filter(
    tokens: Iterable[str],
    modalities: Optional[Sequence[str]],
) -> List[str]:
    """
    Rewrite scan tokens so image resolution only hits selected modalities.

    - Patient token 'BraTS2021_00000' + modalities=['flair']
        -> ['BraTS2021_00000_flair']
    - Already modality-scoped token kept only if its modality is selected
    - modalities=None -> tokens unchanged (all modalities via patient prefix)
    """
    if modalities is None:
        return [str(t).strip() for t in tokens if str(t).strip()]

    mods = [str(m).lower() for m in modalities]
    out: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        patient, existing = split_patient_and_modality(token)
        if not patient:
            continue
        if existing is not None:
            if existing not in mods:
                continue
            scoped = f"{patient}_{existing}"
            if scoped not in seen:
                seen.add(scoped)
                out.append(scoped)
            continue
        for mod in mods:
            scoped = f"{patient}_{mod}"
            if scoped not in seen:
                seen.add(scoped)
                out.append(scoped)
    return out


def discover_patient_tokens_from_images(
    *,
    image_root: str | Path,
    image_ext: str = ".png",
) -> list[str]:
    """
    Discover unique patient tokens by scanning image files under image_root.
    """
    root = Path(image_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"image_root not found: {root}")

    ext = image_ext.lower()
    patients: set[str] = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ext:
            continue
        patients.add(patient_token_from_stem(path.stem))

    tokens = sorted(patients)
    if not tokens:
        raise RuntimeError(
            f"No patient tokens discovered under {root} with extension {ext}. "
            "Check --train-root and --image-ext."
        )
    return tokens


def split_patient_tokens(
    tokens: list[str],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split patient tokens into train/eval lists (patient-level, deterministic)."""
    if not tokens:
        raise ValueError("Cannot split an empty patient token list.")

    val_ratio = float(val_ratio)
    if val_ratio <= 0.0:
        return list(tokens), []
    if val_ratio >= 1.0:
        return [], list(tokens)

    ordered = list(tokens)
    rng = random.Random(int(seed))
    rng.shuffle(ordered)

    n_eval = max(1, int(round(len(ordered) * val_ratio)))
    if n_eval >= len(ordered):
        n_eval = max(1, len(ordered) - 1)

    eval_tokens = sorted(ordered[:n_eval])
    train_tokens = sorted(ordered[n_eval:])
    if not train_tokens:
        raise RuntimeError(
            f"Auto split produced an empty train set (patients={len(ordered)}, "
            f"val_ratio={val_ratio}). Reduce --val-ratio or add more patients."
        )
    return train_tokens, eval_tokens


def _write_scan_list(path: Path, tokens: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def resolve_train_eval_tokens(
    *,
    train_list: str,
    eval_list: str,
    image_root: str | Path,
    image_ext: str = ".png",
    val_ratio: float = 0.15,
    seed: int = 42,
    write_lists: bool = True,
) -> tuple[list[str], list[str], str, str]:
    """
    Resolve train/eval scan tokens from list files or auto-discovery.

    Returns:
        (train_tokens, eval_tokens, train_list_path, eval_list_path)
    """
    image_root = Path(image_root).expanduser().resolve()
    default_dir = image_root.parent

    train_list_path = Path(train_list).expanduser() if str(train_list).strip() else default_dir / "train_list.txt"
    eval_list_path = Path(eval_list).expanduser() if str(eval_list).strip() else default_dir / "eval_list.txt"

    train_exists = train_list_path.exists()
    eval_exists = eval_list_path.exists()

    if train_exists and eval_exists:
        train_tokens = read_scan_list(train_list_path)
        eval_tokens = read_scan_list(eval_list_path)
        if not train_tokens:
            raise RuntimeError(f"Train list has no usable scan tokens: {train_list_path}")
        if not eval_tokens:
            raise RuntimeError(f"Eval list has no usable scan tokens: {eval_list_path}")
        return train_tokens, eval_tokens, str(train_list_path), str(eval_list_path)

    all_patients = discover_patient_tokens_from_images(
        image_root=image_root,
        image_ext=image_ext,
    )

    if train_exists and not eval_exists:
        train_tokens = read_scan_list(train_list_path)
        if not train_tokens:
            raise RuntimeError(f"Train list has no usable scan tokens: {train_list_path}")
        eval_set = set(all_patients) - set(train_tokens)
        eval_tokens = sorted(eval_set)
        if not eval_tokens:
            train_tokens, eval_tokens = split_patient_tokens(
                all_patients, val_ratio=val_ratio, seed=seed
            )
            print(
                "[data] eval_list missing and train_list covers all patients; "
                f"re-split {len(all_patients)} patients with val_ratio={val_ratio:.2f}"
            )
        else:
            print(
                f"[data] eval_list missing; derived {len(eval_tokens)} eval patients "
                f"not present in train_list"
            )
    elif eval_exists and not train_exists:
        eval_tokens = read_scan_list(eval_list_path)
        if not eval_tokens:
            raise RuntimeError(f"Eval list has no usable scan tokens: {eval_list_path}")
        train_set = set(all_patients) - set(eval_tokens)
        train_tokens = sorted(train_set)
        if not train_tokens:
            raise RuntimeError(
                f"Train list missing and all discovered patients are in eval_list: {eval_list_path}"
            )
        print(
            f"[data] train_list missing; derived {len(train_tokens)} train patients "
            f"from discovered images"
        )
    else:
        print(
            f"[data] scan list files not found at {train_list_path} / {eval_list_path}; "
            f"auto-discovering {len(all_patients)} patients from {image_root}"
        )
        train_tokens, eval_tokens = split_patient_tokens(
            all_patients,
            val_ratio=val_ratio,
            seed=seed,
        )
        print(
            f"[data] auto-split patients: train={len(train_tokens)} eval={len(eval_tokens)} "
            f"(val_ratio={val_ratio:.2f}, seed={seed})"
        )

    if write_lists:
        if not train_exists:
            _write_scan_list(train_list_path, train_tokens)
            print(f"[data] wrote train_list ({len(train_tokens)} patients) -> {train_list_path}")
        if not eval_exists:
            _write_scan_list(eval_list_path, eval_tokens)
            print(f"[data] wrote eval_list ({len(eval_tokens)} patients) -> {eval_list_path}")

    return train_tokens, eval_tokens, str(train_list_path), str(eval_list_path)


__all__ = [
    "ALL_MODALITIES",
    "patient_token_from_stem",
    "split_patient_and_modality",
    "parse_modality_arg",
    "apply_modality_filter",
    "discover_patient_tokens_from_images",
    "split_patient_tokens",
    "resolve_train_eval_tokens",
]
