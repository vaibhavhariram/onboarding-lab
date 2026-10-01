"""The content-hash cache.

The cache is what makes reports reproducible; it does not make model behaviour
deterministic. The nonce exists so floor runs can bypass it deliberately.
"""

from __future__ import annotations

from pathlib import Path

from onboarding_lab import cache

BASE = {
    "role": "extract",
    "system": "sys",
    "messages": [{"role": "user", "content": "hello"}],
    "json_schema": None,
    "model": "m",
    "max_tokens": 4000,
    "thinking": "between_tools",
    "effort": "low",
}


def test_identical_inputs_hit(tmp_path: Path) -> None:
    key = cache.cache_key(**BASE)
    assert cache.read(key, tmp_path) is None
    cache.write(key, {"status": "ok", "text": "v"}, tmp_path)
    assert cache.read(key, tmp_path) == {"status": "ok", "text": "v"}
    assert cache.cache_key(**BASE) == key


def test_key_order_independence() -> None:
    reordered = {k: BASE[k] for k in reversed(list(BASE))}
    assert cache.cache_key(**reordered) == cache.cache_key(**BASE)


def test_changed_parameter_misses() -> None:
    key = cache.cache_key(**BASE)
    for field, value in [
        ("role", "aligner"),
        ("system", "other"),
        ("model", "other"),
        ("max_tokens", 10),
        ("thinking", "disabled"),
        ("effort", "high"),
        ("json_schema", {"type": "object"}),
        ("messages", [{"role": "user", "content": "different"}]),
    ]:
        assert cache.cache_key(**{**BASE, field: value}) != key, field


def test_nonce_bypasses(tmp_path: Path) -> None:
    """Floor runs vary only by nonce, so each gets its own cache slot."""
    plain = cache.cache_key(**BASE)
    keys = {cache.cache_key(**BASE, nonce=f"floor{k}") for k in range(5)}
    assert plain not in keys
    assert len(keys) == 5


def test_disabled_by_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LAB_CACHE", "0")
    key = cache.cache_key(**BASE)
    cache.write(key, {"status": "ok"}, tmp_path)
    assert cache.read(key, tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_corrupt_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    key = cache.cache_key(**BASE)
    (tmp_path / f"{key}.json").write_text("{not json")
    assert cache.read(key, tmp_path) is None
