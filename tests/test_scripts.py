"""Guards on the `scripts/` directory.

A script whose basename matches a stdlib module shadows it for anything that
puts this directory on `sys.path` — which is any script run from here, and any
shell whose cwd is here. `numbers.py` was the original name and matplotlib
imports `numbers`, so this is not hypothetical.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path("scripts")


def test_no_script_shadows_a_stdlib_module() -> None:
    offenders = sorted(p.name for p in SCRIPTS.glob("*.py") if p.stem in sys.stdlib_module_names)
    assert not offenders, (
        f"{offenders} shadow stdlib modules for anything importing from scripts/. Rename them."
    )


def test_regen_numbers_has_no_sys_path_manipulation() -> None:
    """The rename is the fix; the old workaround should not creep back."""
    source = (SCRIPTS / "regen_numbers.py").read_text()
    assert "sys.path.pop" not in source
    assert "sys.path.insert" not in source


def test_regen_numbers_refuses_a_readme_without_markers(tmp_path: Path) -> None:
    """No figure is ever written into a document that lacks the markers."""
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n")
    result = subprocess.run(
        [sys.executable, str(Path("scripts/regen_numbers.py").resolve())],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "numbers:start" in result.stdout + result.stderr


def test_make_numbers_points_at_the_renamed_script() -> None:
    makefile = Path("Makefile").read_text()
    assert "scripts/regen_numbers.py" in makefile
    assert "scripts/numbers.py" not in makefile
