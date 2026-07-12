from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_image_pil(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


__all__ = ["load_image_pil"]
