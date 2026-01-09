"""Backward-compatible wrapper for legacy imports.

Keeps old entrypoints working:
  from augmentation import sample_masks_anti_mirror, HalfAug

Do not modify training/eval logic; this module only re-exports symbols.
"""

try:
    from phase1.data.augmentation import *  # type: ignore
except Exception:  # pragma: no cover
    from data.augmentation import *  # type: ignore
