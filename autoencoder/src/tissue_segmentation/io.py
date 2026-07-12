from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

@dataclass(frozen=True)
class LabelEncodingInfo:
    """Resolved label encoding metadata used by dataset/trainer."""

    mode: int
    original_id_to_name: Dict[int, str]
    unknown_ids: set[int]
    non_brain_ids: set[int]
    encode_map: Dict[int, int]
    decode_map: Dict[int, list[int]]
    encoded_id_to_name: Dict[int, str]
    num_classes: int

@dataclass(frozen=True)
class ImageIndex:
    """
    Pre-built image lookup tables for scan-token resolution.
    Built once per image_root/image_ext to avoid repeated scans.
    """

    root: Path
    image_ext: str
    all_images: list[Path]
    path_set: set[Path]
    stem_index: Dict[str, list[Path]]
    basename_index: Dict[str, list[Path]]
    rel_index: Dict[str, Path]
    token_prefix_index: Dict[str, list[Path]]

def _first_npz_array(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    if len(npz.files) == 0:
        raise ValueError("NPZ file contains no arrays.")
    first_key = npz.files[0]
    return npz[first_key]

def _ensure_hw_int_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unexpected label shape {arr.shape}; expected [H,W] or [1,H,W] or [H,W,1].")
    else:
        raise ValueError(f"Unexpected label ndim={arr.ndim}; expected 2 or 3.")

    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.rint(arr).astype(np.int64)
    else:
        arr = arr.astype(np.int64, copy=False)
    return arr

def load_label_array(path: str | Path, key: Optional[str] = None) -> np.ndarray:
    """
    Load segmentation label array as integer [H,W].
    Supported formats:
      - .npz (first array or explicit key)
      - .npy
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npz":
        with np.load(p) as data:
            arr = data[key] if key is not None else _first_npz_array(data)
        return _ensure_hw_int_array(arr)
    if suffix == ".npy":
        arr = np.load(p, allow_pickle=False)
        return _ensure_hw_int_array(arr)
    raise ValueError(f"Unsupported label extension '{p.suffix}' for {p}. Expected .npz or .npy.")

def _normalize_name(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())

def parse_seg_labels_txt(path: str | Path) -> Dict[int, str]:
    """
    Parse seg_labels.txt into {id: label_name}.
    Only rows with trailing RGBA integer columns are considered valid labels.
    Accepts lines in forms like:
      100 Non-Brain-1 92 75 81 0
      0   Unknown
    Ignores comments (#...) and blank lines.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"seg_labels.txt not found: {p}")

    id_to_name: Dict[int, str] = {}
    for ln, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            label_id = int(parts[0])
        except ValueError:
            continue

        trailing_rgba = False
        if len(parts) >= 6:
            tail = parts[-4:]
            trailing_rgba = all(re.fullmatch(r"-?\d+", t) is not None for t in tail)
        if not trailing_rgba:
            continue

        name_tokens = parts[1:-4]
        if not name_tokens:
            raise ValueError(f"Invalid seg label line {ln}: '{raw_line}'")
        label_name = " ".join(name_tokens).strip()
        if label_id in id_to_name:
            raise ValueError(
                f"Duplicate label id {label_id} found in {p} at line {ln}. "
                "Each label id must appear once."
            )
        id_to_name[label_id] = label_name

    if not id_to_name:
        raise ValueError(f"No labels parsed from {p}")
    return id_to_name

def identify_special_ids(id_to_name: Dict[int, str]) -> tuple[set[int], set[int]]:
    """
    Identify unknown + non-brain ids by label-name tokens.
    """
    unknown_ids: set[int] = set()
    non_brain_ids: set[int] = set()

    for idx, name in id_to_name.items():
        norm = _normalize_name(name)
        tokens = set(norm.split())

        if "unknown" in tokens:
            unknown_ids.add(idx)

        has_non = "non" in tokens
        has_brain = "brain" in tokens
        has_non_brain_joined = "nonbrain" in norm.replace(" ", "")
        if (has_non and has_brain) or has_non_brain_joined:
            non_brain_ids.add(idx)

    return unknown_ids, non_brain_ids

def _invert_mapping(mapping: Dict[int, int]) -> Dict[int, list[int]]:
    out: Dict[int, list[int]] = {}
    for src, dst in mapping.items():
        out.setdefault(dst, []).append(src)
    for dst in out:
        out[dst] = sorted(out[dst])
    return out

def _build_encoded_names(
    decode_map: Dict[int, list[int]],
    id_to_name: Dict[int, str],
    *,
    special_bg_label: str,
) -> Dict[int, str]:
    encoded_to_name: Dict[int, str] = {}
    for enc_id, src_ids in decode_map.items():
        if enc_id == 0 and len(src_ids) > 1:
            encoded_to_name[enc_id] = special_bg_label
            continue
        if len(src_ids) == 1:
            encoded_to_name[enc_id] = id_to_name.get(src_ids[0], f"id_{src_ids[0]}")
            continue
        names = [id_to_name.get(x, f"id_{x}") for x in src_ids]
        encoded_to_name[enc_id] = " | ".join(names)
    return encoded_to_name

def build_label_encoding_info(
    *,
    mode: int,
    id_to_name: Dict[int, str],
    unknown_ids: set[int],
    non_brain_ids: set[int],
    num_classes_override: int = 0,
    require_special_ids: bool = True,
) -> LabelEncodingInfo:
    """
    Build mode-specific encoding metadata.

    Mode 1:
      unknown + non-brain -> 0
      all others remapped to contiguous 1..K

    Mode 2:
      unknown + non-brain -> 0
      all others keep original ids

    Mode 3:
      no merge; all ids remapped to contiguous 0..K-1 (unknown/non-brain stay distinct)

    Mode 4:
      identity mapping (no merge, no remap)
    """
    if mode not in {1, 2, 3, 4}:
        raise ValueError(f"label mode must be one of { 1,2,3,4} , got {mode}")

    all_ids = sorted(id_to_name.keys())
    special_ids = set(unknown_ids) | set(non_brain_ids)
    if mode in {1, 2} and require_special_ids:
        if not unknown_ids:
            raise ValueError(
                "Label mode requires 'unknown' ids, but none were detected in seg_labels names. "
                "Fix seg_labels.txt naming or choose a mode that does not merge special ids."
            )
        if not non_brain_ids:
            raise ValueError(
                "Label mode requires 'non brain' ids, but none were detected in seg_labels names. "
                "Fix seg_labels.txt naming or choose a mode that does not merge special ids."
            )
    encode_map: Dict[int, int] = {}

    if mode == 1:
        foreground_ids = [x for x in all_ids if x not in special_ids]
        for sid in sorted(special_ids):
            encode_map[sid] = 0
        for i, oid in enumerate(foreground_ids, start=1):
            encode_map[oid] = i
        num_classes = len(foreground_ids) + 1

    elif mode == 2:
        for oid in all_ids:
            encode_map[oid] = 0 if oid in special_ids else oid
        num_classes = max(encode_map.values()) + 1

    elif mode == 3:
        for i, oid in enumerate(all_ids):
            encode_map[oid] = i
        num_classes = len(all_ids)

    else:
        for oid in all_ids:
            encode_map[oid] = oid
        num_classes = max(all_ids) + 1

    if int(num_classes_override) > 0:
        if int(num_classes_override) < int(num_classes):
            raise ValueError(
                f"Provided --num-classes={num_classes_override} is smaller than required classes={num_classes}."
            )
        num_classes = int(num_classes_override)

    if num_classes < 2:
        raise ValueError(f"Computed num_classes={num_classes}, expected >= 2.")

    decode_map = _invert_mapping(encode_map)
    encoded_id_to_name = _build_encoded_names(
        decode_map,
        id_to_name,
        special_bg_label="background(unknown+non_brain)",
    )

    return LabelEncodingInfo(
        mode=int(mode),
        original_id_to_name=dict(id_to_name),
        unknown_ids=set(unknown_ids),
        non_brain_ids=set(non_brain_ids),
        encode_map=encode_map,
        decode_map=decode_map,
        encoded_id_to_name=encoded_id_to_name,
        num_classes=int(num_classes),
    )

def encode_label_array(
    label_arr: np.ndarray,
    info: LabelEncodingInfo,
    *,
    strict_label_ids: bool = True,
    allow_unknown_label_ids: bool = False,
    unknown_fallback_id: int = 0,
) -> np.ndarray:
    """
    Encode label ids according to LabelEncodingInfo.
    Raises if unseen label ids exist in input.
    """
    src = _ensure_hw_int_array(label_arr)
    uniq = np.unique(src)
    missing = [int(x) for x in uniq.tolist() if int(x) not in info.encode_map]
    if missing:
        if strict_label_ids:
            raise ValueError(
                f"Label array has ids absent from seg_labels mapping: {sorted(missing)}. "
                "Update seg_labels.txt or disable strict checks explicitly."
            )
        if not allow_unknown_label_ids:
            raise ValueError(
                f"Unknown label ids found: {sorted(missing)}. "
                "Use --allow-unknown-label-ids to map unknown ids to class 0."
            )
        if int(unknown_fallback_id) < 0 or int(unknown_fallback_id) >= int(info.num_classes):
            raise ValueError(
                f"unknown_fallback_id={unknown_fallback_id} is out of [0, {info.num_classes - 1}]"
            )

    out = np.empty(src.shape, dtype=np.int64)
    for sid in uniq.tolist():
        sid_int = int(sid)
        mapped = info.encode_map.get(sid_int, int(unknown_fallback_id))
        out[src == sid_int] = int(mapped)

    vmin = int(out.min()) if out.size > 0 else 0
    vmax = int(out.max()) if out.size > 0 else 0
    if vmin < 0 or vmax >= int(info.num_classes):
        raise ValueError(
            f"Encoded labels out of range [0,{info.num_classes - 1}]: min={vmin}, max={vmax}"
        )

    return out

def assert_encoding_deterministic(info: LabelEncodingInfo) -> None:
    """
    Lightweight internal self-check to guarantee deterministic mapping.
    """
    keys = sorted(info.encode_map.keys())
    vals = [info.encode_map[k] for k in keys]
    if any((not isinstance(v, int)) for v in vals):
        raise AssertionError("encode_map contains non-int values")
    if any(v < 0 or v >= info.num_classes for v in vals):
        raise AssertionError("encode_map has ids outside num_classes range")

def read_scan_list(path: str | Path) -> list[str]:
    """
    Read scan list text file.
    - ignores blank lines
    - ignores comments prefixed with '#'
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"scan list file not found: {p}")

    items: list[str] = []
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            items.append(line)
    return items

def _path_has_sep(token: str) -> bool:
    return ("/" in token) or ("\\" in token)

def build_image_index(*, image_root: str | Path, image_ext: str) -> ImageIndex:
    root = Path(image_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"image_root not found: {root}")

    ext = image_ext.lower()
    all_images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ext)
    if not all_images:
        raise RuntimeError(f"No images with extension {ext} found under {root}")

    path_set = {p.resolve() for p in all_images}
    stem_index: Dict[str, list[Path]] = {}
    basename_index: Dict[str, list[Path]] = {}
    rel_index: Dict[str, Path] = {}
    token_prefix_index: Dict[str, list[Path]] = {}
    for p in all_images:
        stem_l = p.stem.lower()
        stem_index.setdefault(stem_l, []).append(p)
        basename_index.setdefault(p.name.lower(), []).append(p)
        rel_index[str(p.relative_to(root)).replace("\\", "/").lower()] = p

        parts = [x for x in stem_l.split("_") if x]
        if parts:
            for i in range(1, len(parts) + 1):
                key = "_".join(parts[:i])
                token_prefix_index.setdefault(key, []).append(p)

    for key in stem_index:
        stem_index[key] = sorted(stem_index[key])
    for key in basename_index:
        basename_index[key] = sorted(basename_index[key])
    for key in token_prefix_index:
        token_prefix_index[key] = sorted(token_prefix_index[key])

    return ImageIndex(
        root=root,
        image_ext=ext,
        all_images=all_images,
        path_set=path_set,
        stem_index=stem_index,
        basename_index=basename_index,
        rel_index=rel_index,
        token_prefix_index=token_prefix_index,
    )

def resolve_scan_tokens_to_images(
    *,
    image_root: str | Path,
    tokens: Iterable[str],
    image_ext: str,
    image_index: ImageIndex | None = None,
) -> list[Path]:
    """
    Resolve scan tokens to concrete image paths.

    Resolution policy:
    1) If token looks like a path (contains / or \\), treat as relative to image_root
       (or absolute path if already absolute), with optional image_ext completion.
    2) Otherwise treat token as basename stem; use indexed stem matches under image_root.
       If multiple matches exist, keep deterministic first path in sorted order.
    """
    idx = image_index if image_index is not None else build_image_index(image_root=image_root, image_ext=image_ext)
    root = idx.root
    ext = idx.image_ext
    path_set = idx.path_set
    stem_index = idx.stem_index
    basename_index = idx.basename_index
    rel_index = idx.rel_index
    token_prefix_index = idx.token_prefix_index

    resolved: list[Path] = []
    for token_raw in tokens:
        token = token_raw.strip()
        if not token:
            continue

        matched: Optional[Path] = None

        if _path_has_sep(token):
            tok = token.replace("\\", "/")
            tok_path = Path(tok)
            candidate = tok_path if tok_path.is_absolute() else (root / tok_path)

            if candidate.suffix == "":
                candidate_with_ext = candidate.with_suffix(ext)
            else:
                candidate_with_ext = candidate

            if candidate_with_ext.exists() and candidate_with_ext.resolve() in path_set:
                matched = candidate_with_ext.resolve()
            else:
                rel_candidates: list[str] = []
                try:
                    rel_candidates.append(str(candidate_with_ext.relative_to(root)).replace("\\", "/").lower())
                except Exception:
                    pass
                rel_candidates.append(tok.lower())
                if not tok.lower().endswith(ext):
                    rel_candidates.append(f"{tok.lower()}{ext}")
                for key in rel_candidates:
                    if key in rel_index:
                        matched = rel_index[key]
                        break
        else:
            key = token.lower()
            if key in basename_index:
                matched = basename_index[key][0]
            elif key in stem_index:
                matched = stem_index[key][0]
            elif f"{key}{ext}" in basename_index:
                matched = basename_index[f"{key}{ext}"][0]
            elif key in token_prefix_index:
                resolved.extend(token_prefix_index[key])
                continue

        if matched is not None:
            resolved.append(matched)

    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in resolved:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq

__all__ = [
    "ImageIndex",
    "LabelEncodingInfo",
    "load_label_array",
    "parse_seg_labels_txt",
    "identify_special_ids",
    "build_label_encoding_info",
    "encode_label_array",
    "assert_encoding_deterministic",
    "read_scan_list",
    "build_image_index",
    "resolve_scan_tokens_to_images",
]
