"""Backward-compatible wrapper for legacy imports.

Keeps old entrypoints working:
  from data import create_dataloaders_from_folder, ...

Do not modify training/eval logic; this module only re-exports symbols.
"""

try:
    from phase1.data.dataset import *  # type: ignore
except Exception:  # pragma: no cover
    from data.dataset import *  # type: ignore
