"""Backward-compatible wrapper for legacy imports.

Keeps old entrypoints working:
  from model import SwinUNetDualViewSSLPhase1, flip_lr

Do not modify training/eval logic; this module only re-exports symbols.
"""

try:
    from phase1.models.swin_unet_dualview_ssl import *  # type: ignore
except Exception:  # pragma: no cover
    from models.swin_unet_dualview_ssl import *  # type: ignore
