#!/usr/bin/env python3
"""Rewrite the README's numbers block from a committed run.

Not named ``numbers.py``: that shadows the stdlib ``numbers`` module, which
matplotlib imports, for anything that puts this directory on ``sys.path`` --
which is any script run from here or any shell with this as its cwd. ``make
numbers`` is still the interface. ``tests/test_scripts.py`` enforces the rule.

The only thing permitted to write a figure into a document. Fails loudly rather
than leaving a stale number in place, because a stale number is worse than no
number.

Phase 3 fills in the aggregate reading; the marker contract is enforced now so
the README can never drift into hand-typed figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

START = "<!-- numbers:start -->"
END = "<!-- numbers:end -->"


def _split(readme: Path) -> tuple[str, str, str]:
    """Return (head, current body, tail), or refuse.

    Refusing is the point: a stale number left in place is worse than no number,
    so a README without both markers is an error rather than a no-op.
    """
    text = readme.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise SystemExit(
            f"{readme}: expected exactly one {START} and one {END}. "
            f"Refusing to write numbers into a document without them."
        )
    head, _, rest = text.partition(START)
    body, _, tail = rest.partition(END)
    return head, body, tail


def read_block(readme: Path) -> str:
    return _split(readme)[1]


def replace_block(readme: Path, body: str) -> None:
    head, _, tail = _split(readme)
    readme.write_text(f"{head}{START}\n{body.strip()}\n{END}{tail}")


def main() -> int:
    readme = Path("README.md")
    if not readme.is_file():
        raise SystemExit("README.md not found")
    # Validate the marker contract even before there is an aggregate to read, so
    # a broken README fails here rather than at the end of a paid run.
    replace_block(readme, read_block(readme))
    print(
        "regen_numbers.py: marker contract OK. Aggregate reading lands in "
        "Phase 3; no figures written.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
