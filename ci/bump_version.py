#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
VERSION_LINE_PATTERN = re.compile(r'(?m)^__version__ = "[^"]+"$')
DEFAULT_TARGET = Path(__file__).resolve().parent.parent / "awex" / "reader" / "__init__.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update awex.reader.__version__ for release builds."
    )
    parser.add_argument("version", help="Version string in x.y.z format")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_TARGET,
        help="Path to the module file containing __version__",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not VERSION_PATTERN.fullmatch(args.version):
        raise SystemExit(f"Version must match x.y.z, got: {args.version}")

    target = args.path
    content = target.read_text()
    updated, count = VERSION_LINE_PATTERN.subn(
        f'__version__ = "{args.version}"',
        content,
        count=1,
    )

    if count != 1:
        raise SystemExit(f"Failed to rewrite __version__ in {target}")

    target.write_text(updated)
    print(f'__version__ = "{args.version}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
