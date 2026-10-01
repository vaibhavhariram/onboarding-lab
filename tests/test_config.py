"""Credential handling and run configuration.

The key rule exists because ``ANTHROPIC_API_KEY`` is consumed by other tooling in
preference to a subscription, so reading it would bill per token.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from onboarding_lab.config import API_KEY_VAR, FORBIDDEN_KEY_VAR, Config, api_key, load_dotenv


def test_never_reads_the_forbidden_variable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(FORBIDDEN_KEY_VAR, "sk-ant-should-never-be-used")
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    assert api_key(required=False) is None


def test_requires_lab_specific_variable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(FORBIDDEN_KEY_VAR, "sk-ant-should-never-be-used")
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    with pytest.raises(RuntimeError, match=API_KEY_VAR):
        api_key()


def test_reads_lab_variable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_VAR, "sk-lab-test")
    assert api_key() == "sk-lab-test"


def test_dotenv_loading(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    Path(".env").write_text(f'# comment\n{API_KEY_VAR}="sk-from-dotenv"\nEMPTY\n')
    assert api_key() == "sk-from-dotenv"


def test_dotenv_does_not_override_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_VAR, "sk-from-env")
    Path(".env").write_text(f"{API_KEY_VAR}=sk-from-dotenv\n")
    assert load_dotenv() == {API_KEY_VAR: "sk-from-dotenv"}
    assert api_key() == "sk-from-env"


def test_missing_dotenv_is_not_an_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_dotenv() == {}


# -- run config ---------------------------------------------------------------


def test_shipped_configs_load() -> None:
    full = Config.load("lab.yaml")
    assert full.personas == 12
    assert full.noise_rates == [0.0, 0.05, 0.1, 0.2]
    assert full.floor_runs >= 5, "a sigma estimate from fewer than 5 runs is too noisy to gate on"
    dev = Config.load("lab.dev.yaml")
    assert dev.personas == 3
    assert dev.noise_rates == [0.0]


def test_user_agent_is_the_cached_prefix() -> None:
    """The one role whose system prompt is byte-stable across many calls."""
    cfg = Config.load("lab.yaml")
    assert cfg.role("user_agent").cache_system is True
    assert cfg.role("extract").cache_system is False


def test_aligner_gets_more_effort_than_the_simulator() -> None:
    cfg = Config.load("lab.yaml")
    assert cfg.role("aligner").effort == "medium"
    assert cfg.role("user_agent").effort == "low"


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("personas: 3\ntypo_key: 1\n")
    with pytest.raises(ValueError, match="typo_key"):
        Config.load(p)
