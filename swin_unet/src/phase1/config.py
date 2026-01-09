"""Backward-compatible wrapper for legacy imports.

Keeps old entrypoints working:
  from config import ExperimentConfig, build_argparser, ...

Do not modify training/eval logic; this module only re-exports symbols.
"""

try:
    # If imported as a package (PYTHONPATH includes src)
    from phase1.config.experiment import *  # type: ignore
except Exception:  # pragma: no cover
    # If running from within src/phase1 (legacy style)
    from config.experiment import *  # type: ignore
