"""Backward-compatible wrapper for legacy imports.

Keeps old entrypoints working:
  from metrics import MetricsAccumulator, EpochMetrics, ...

Do not modify training/eval logic; this module only re-exports symbols.
"""

try:
    from phase1.common.metrics import *  # type: ignore
except Exception:  # pragma: no cover
    from common.metrics import *  # type: ignore
