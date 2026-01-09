"""Legacy entrypoint wrapper.

This file keeps backward compatibility for code that runs `phase1/trainer.py`.
Training logic lives in `phase1/training/trainer.py`.
"""

from __future__ import annotations

# Import using local package path (works when PYTHONPATH includes src)
try:
    from phase1.training.trainer import main
except Exception:  # pragma: no cover
    # Fallback when running from within src/phase1 directory
    from training.trainer import main


if __name__ == "__main__":
    main()
