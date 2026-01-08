from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2


def _robust_scale_01(x: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    if p99 <= p1:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - p1) / (p99 - p1)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _brain_mask_approx(x01: np.ndarray) -> np.ndarray:
    # x01 in [0,1]
    x = (x01 * 255).astype(np.uint8)
    x_blur = cv2.GaussianBlur(x, (5, 5), 0)
    thr = max(int(np.mean(x_blur) * 0.8), 1)
    mask = (x_blur > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask.astype(np.float32)


def _vertical_symmetry_mse(mask: np.ndarray) -> float:
    # Compare left and right halves for symmetry
    h, w = mask.shape
    mid = w // 2
    left = mask[:, :mid]
    right = mask[:, w - mid : w]
    right_flip = np.fliplr(right)
    if left.shape != right_flip.shape:
        m = min(left.shape[1], right_flip.shape[1])
        left = left[:, :m]
        right_flip = right_flip[:, :m]
    diff = left - right_flip
    return float(np.mean(diff * diff))


def ensure_vertical_orientation(slice_2d: np.ndarray) -> np.ndarray:
    # Try rotations: 0, 90, 270 degrees and pick the best symmetry score
    candidates = [
        slice_2d,
        np.rot90(slice_2d, 1),
        np.rot90(slice_2d, 3),
    ]

    best = None
    best_score = None
    for cand in candidates:
        x01 = _robust_scale_01(cand.astype(np.float32))
        mask = _brain_mask_approx(x01)
        score = _vertical_symmetry_mse(mask)
        if best_score is None or score < best_score:
            best_score = score
            best = cand

    return best.astype(np.float32)
