"""Backward-compatible wrapper for legacy imports.

Keeps old entrypoints working:
  from visualization import save_image_grid, plot_training_curves, ...

Do not modify training/eval logic; this module only re-exports symbols.
"""

try:
    from phase1.viz.visualization import *  # type: ignore
except Exception:  # pragma: no cover
    from viz.visualization import *  # type: ignore
