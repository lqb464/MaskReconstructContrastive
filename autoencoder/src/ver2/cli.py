from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, Optional, Sequence

from .common.cli_utils import dispatch_subcommand


def _train_ssl(argv: Sequence[str]) -> object:
    from . import main as ssl_train_entry

    return ssl_train_entry.main(argv)


def _eval_ssl(argv: Sequence[str]) -> object:
    from . import eval as ssl_eval_entry

    return ssl_eval_entry.main(argv)


COMMAND_HANDLERS: Dict[str, Callable[[Sequence[str]], object]] = {
    "train-ssl": _train_ssl,
    "eval-ssl": _eval_ssl,
}

COMPAT_ALIASES: Dict[str, str] = {
    "train": "train-ssl",
    "eval": "eval-ssl",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autoencoder.src.ver2.cli",
        description="MAE / VAE dual-view SSL v2 (proper AE definitions).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Subcommand: train-ssl | eval-ssl",
    )
    parser.add_argument("command_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> object:
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    if not raw_argv or raw_argv[0] in {"-h", "--help"}:
        parser = build_argparser()
        parser.print_help()
        print("\nAliases:")
        print("  train -> train-ssl")
        print("  eval  -> eval-ssl")
        print("\nBackbones:")
        print("  --mae  Masked AutoEncoder (random patch mask, default)")
        print("  --vae  Variational AutoEncoder (full image, no mask)")
        return None

    cmd = COMPAT_ALIASES.get(raw_argv[0], raw_argv[0])
    if cmd not in COMMAND_HANDLERS:
        parser = build_argparser()
        parser.error(f"Unknown command: {raw_argv[0]}")
    return dispatch_subcommand([cmd, *raw_argv[1:]], handlers=COMMAND_HANDLERS)


if __name__ == "__main__":
    main()
