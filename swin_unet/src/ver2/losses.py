"""Backward-compatible wrapper for legacy imports.

Keeps old entrypoints working:
  from losses import masked_l1_loss, nt_xent_loss, ...

Do not modify training/eval logic; this module only re-exports symbols.
"""

try:
    from phase1.common.losses import *  # type: ignore
except Exception:  # pragma: no cover
    from common.losses import *  # type: ignore
