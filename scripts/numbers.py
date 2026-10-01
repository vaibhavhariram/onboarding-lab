#!/usr/bin/env python3
"""Rewrite the README's numbers block from a committed run.

The only thing permitted to write a figure into a document. Fails loudly rather
than leaving a stale number in place, because a stale number is worse than no
number.

Phase 3 fills in the aggregate reading; the marker contract is enforced now so
the README can never drift into hand-typed figures.
"""

from __future__ import annotations

import sys

# This file is named `numbers.py` because `make numbers` is the interface, but
# that shadows the stdlib `numbers` module for anything imported afterwards --
# matplotlib imports it, and Phase 3 imports matplotlib from here. Running
# `python scripts/numbers.py` puts this directory at sys.path[0], so drop it
# before importing anything else.
if sys.path and sys.path[0].endswith("scripts"):
    sys.path.pop(0)

from pathlib import Path

START = "<!-- numbers:start -->"
END = "<!-- numbers:end -->"


def replace_block(readme: Path, body: str) -> None:
    text = readme.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise SystemExit(
            f"{readme}: expected exactly one {START} and one {END}. "
            f"Refusing to write numbers into a document without them."
        )
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    readme.write_text(f"{head}{START}\n{body.strip()}\n{END}{tail}")


def main() -> int:
    readme = Path("README.md")
    if not readme.is_file():
        raise SystemExit("README.md not found")
    # Validate the marker contract even before there is an aggregate to read, so
    # a broken README fails here rather than at the end of a paid run.
    replace_block(readme, readme.read_text().split(START)[1].split(END)[0])
    print(
        "numbers.py: marker contract OK. Aggregate reading lands in Phase 3; no figures written.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
