from __future__ import annotations

from typing import Optional, Sequence

from ..common.cli_utils import run_entrypoint
from .cli import build_argparser


def main(argv: Optional[Sequence[str]] = None):
    def _run(args):
        from .train import run

        return run(args)

    run_entrypoint(build_argparser, _run, argv=argv)


if __name__ == "__main__":
    main()
