#!/usr/bin/env python3
"""Regenerate the generated parts of docs/USER_GUIDE.md (the key table, from runfleet.ui.keymap).

    uv run python scripts/gen_docs.py           # rewrite the block in place
    uv run python scripts/gen_docs.py --check   # exit 1 if the guide is out of date (CI)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "USER_GUIDE.md"
BLOCK = re.compile(r"(<!-- keymap:begin -->\n)(.*?)(<!-- keymap:end -->)", re.S)


def render(guide_text: str) -> str:
    from runfleet.ui import keymap

    table = keymap.markdown().strip() + "\n"
    if not BLOCK.search(guide_text):
        raise SystemExit(f"{GUIDE}: keymap markers not found")
    return BLOCK.sub(lambda m: m.group(1) + table + m.group(3), guide_text, count=1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="do not write; fail if the guide is out of date")
    args = p.parse_args(argv)
    current = GUIDE.read_text(encoding="utf-8")
    fresh = render(current)
    if args.check:
        if fresh != current:
            print(f"{GUIDE.name} is out of date: run `uv run python scripts/gen_docs.py`", file=sys.stderr)
            return 1
        return 0
    if fresh != current:
        GUIDE.write_text(fresh, encoding="utf-8", newline="\n")   # LF on Windows too (.gitattributes: -text)
        print(f"updated {GUIDE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
