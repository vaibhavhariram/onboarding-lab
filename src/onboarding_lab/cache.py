"""Content-hash cache for model calls.

The cache is what makes *reports* reproducible: identical inputs regenerate
identical artifacts, so a number can be re-derived on a clean checkout. It does
not make model behaviour deterministic — nothing does, now that sampling
parameters are gone. That job belongs to the measured noise floor (PLAN.md P5).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .hashing import canonical_json, sha256_hex

CACHE_ROOT = Path(".cache") / "llm"


def cache_enabled() -> bool:
    return os.environ.get("LAB_CACHE", "1") != "0"


def cache_key(
    *,
    role: str,
    system: str,
    messages: list[dict[str, Any]],
    json_schema: dict | None,
    model: str,
    max_tokens: int,
    thinking: str,
    effort: str,
    nonce: str | None = None,
) -> str:
    """Stable key over everything that can change the response.

    No ``temperature`` and no ``seed``: neither reaches the API (PLAN.md P8).
    ``nonce`` is included so floor runs can bypass the cache deliberately.
    """
    return sha256_hex(
        canonical_json(
            {
                "role": role,
                "system": system,
                "messages": messages,
                "json_schema": json_schema,
                "model": model,
                "max_tokens": max_tokens,
                "thinking": thinking,
                "effort": effort,
                "nonce": nonce,
            }
        )
    )


def _path(key: str, root: Path | None = None) -> Path:
    return (root or CACHE_ROOT) / f"{key}.json"


def read(key: str, root: Path | None = None) -> dict | None:
    if not cache_enabled():
        return None
    p = _path(key, root)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write(key: str, payload: dict, root: Path | None = None) -> None:
    if not cache_enabled():
        return
    p = _path(key, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(p)
