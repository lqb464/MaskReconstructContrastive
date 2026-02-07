from __future__ import annotations

from .train import run
from .cli import build_argparser

def main():
    parser = build_argparser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

