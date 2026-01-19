from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _has_all_classes(labels: np.ndarray, idx: np.ndarray, classes: Sequence[int]) -> bool:
    present = set(labels[idx].tolist()) if idx.size else set()
    return all(c in present for c in classes)


def _possible_val_all_classes(
    class_counts: np.ndarray, n_splits: int, val_ratio: float
) -> bool:
    if val_ratio <= 0.0:
        return True
    for count in class_counts:
        test_max = int(np.ceil(count / float(n_splits)))
        train_val = int(count) - test_max
        if train_val < 2:
            return False
    return True


def _make_val_split(
    labels: np.ndarray,
    train_val_idx: np.ndarray,
    val_ratio: float,
    seed: int,
    groups: Optional[np.ndarray],
    classes: Sequence[int],
    require_all_classes: bool,
    max_tries: int,
) -> tuple[np.ndarray, np.ndarray]:
    if val_ratio <= 0.0:
        return train_val_idx, np.array([], dtype=np.int64)

    sub_labels = labels[train_val_idx]
    sub_groups = groups[train_val_idx] if groups is not None else None

    last_ok = False
    for attempt in range(max_tries):
        rng_seed = seed + attempt
        if sub_groups is None:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=rng_seed)
            train_sub, val_sub = next(splitter.split(np.zeros_like(sub_labels), sub_labels))
        else:
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=rng_seed)
            train_sub, val_sub = next(splitter.split(np.zeros_like(sub_labels), sub_labels, groups=sub_groups))

        train_idx = train_val_idx[train_sub]
        val_idx = train_val_idx[val_sub]
        if not require_all_classes:
            return train_idx, val_idx
        if _has_all_classes(labels, train_idx, classes) and _has_all_classes(labels, val_idx, classes):
            return train_idx, val_idx

    return train_val_idx, np.array([], dtype=np.int64)


def make_kfold_splits(
    labels: Iterable[int],
    n_splits: int,
    seed: int,
    *,
    val_ratio: float = 0.1,
    groups: Optional[Iterable[int]] = None,
    max_tries: int = 50,
) -> List[FoldSplit]:
    labels = np.asarray(list(labels), dtype=np.int64)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for k-fold")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")

    groups_arr = np.asarray(list(groups), dtype=np.int64) if groups is not None else None
    classes = sorted(set(labels.tolist()))
    class_counts = np.array([np.sum(labels == c) for c in classes], dtype=np.int64)
    require_all_classes = _possible_val_all_classes(class_counts, n_splits, val_ratio)

    splits: List[FoldSplit] = []

    for attempt in range(max_tries):
        if groups_arr is None:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + attempt)
            fold_iter = splitter.split(np.zeros_like(labels), labels)
        else:
            if StratifiedGroupKFold is not None:
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + attempt)
                fold_iter = splitter.split(np.zeros_like(labels), labels, groups=groups_arr)
            else:
                splitter = GroupKFold(n_splits=n_splits)
                fold_iter = splitter.split(np.zeros_like(labels), labels, groups=groups_arr)

        splits = []
        ok = True
        for fold_idx, (train_val_idx, test_idx) in enumerate(fold_iter):
            train_idx, val_idx = _make_val_split(
                labels=labels,
                train_val_idx=train_val_idx,
                val_ratio=val_ratio,
                seed=seed + attempt,
                groups=groups_arr,
                classes=classes,
                require_all_classes=require_all_classes,
                max_tries=max_tries,
            )

            if require_all_classes:
                if not _has_all_classes(labels, train_idx, classes):
                    ok = False
                    break
                if val_ratio > 0.0 and not _has_all_classes(labels, val_idx, classes):
                    ok = False
                    break

            splits.append(
                FoldSplit(
                    fold_index=fold_idx,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                )
            )

        last_ok = ok and bool(splits)
        if last_ok:
            return splits

    if require_all_classes:
        raise RuntimeError(
            "unable to create k-fold splits with all classes present in train/val; "
            "check class counts or reduce n_splits/val_ratio"
        )
    return splits
