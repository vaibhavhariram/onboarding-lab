"""Content hashing and code-version stamping.

Every artifact identity in the lab is a hash of its inputs, so canonicalisation
has to be stable: sorted keys, no incidental whitespace, UTF-8.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any


def canonical_json(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, compact separators, UTF-8 preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def short_hash(text: str, n: int = 6) -> str:
    return sha256_hex(text)[:n]


def code_version() -> str:
    """Git sha, suffixed ``-dirty`` when the tree has changes; ``dev`` with no commits."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "dev"
    if not sha:
        return "dev"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return sha
    return f"{sha}-dirty" if dirty else sha
