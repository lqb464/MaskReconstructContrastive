from __future__ import annotations

import argparse
from typing import Callable, Mapping, Optional, Sequence


ParserBuilder = Callable[[], argparse.ArgumentParser]
ArgsRunner = Callable[[argparse.Namespace], object]


def parse_with_builder(
    build_parser: ParserBuilder,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def run_entrypoint(
    build_parser: ParserBuilder,
    run_from_args: ArgsRunner,
    argv: Optional[Sequence[str]] = None,
) -> object:
    args = parse_with_builder(build_parser, argv=argv)
    return run_from_args(args)


def dispatch_subcommand(
    argv: Sequence[str],
    handlers: Mapping[str, Callable[[Sequence[str]], object]],
) -> object:
    if not argv:
        raise SystemExit(2)
    cmd = argv[0]
    if cmd not in handlers:
        raise SystemExit(2)
    return handlers[cmd](argv[1:])
