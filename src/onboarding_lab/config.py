"""Run configuration and credential loading.

**The lab never reads ``ANTHROPIC_API_KEY``.** That variable is consumed by other
tooling in this environment in preference to a subscription, so reading it would
bill per token. The lab reads ``LAB_ANTHROPIC_API_KEY`` from a gitignored
``.env`` and passes it to the SDK explicitly (CLAUDE.md rule 1).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

#: The only credential variable this project reads.
API_KEY_VAR = "LAB_ANTHROPIC_API_KEY"

#: Deliberately *not* read. Named here so the prohibition is greppable and the
#: test suite can assert we never touch it.
FORBIDDEN_KEY_VAR = "ANTHROPIC_API_KEY"


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Minimal ``.env`` reader: ``KEY=value`` lines, ``#`` comments, optional quotes.

    Hand-rolled rather than adding a dependency for fifteen lines (CLAUDE.md
    rule 12). Returns what it set.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    loaded: dict[str, str] = {}
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def api_key(*, required: bool = True) -> str | None:
    """Read the lab's API key, loading ``.env`` first."""
    load_dotenv()
    key = os.environ.get(API_KEY_VAR, "").strip()
    if key:
        return key
    if required:
        raise RuntimeError(
            f"{API_KEY_VAR} is not set. Put it in a gitignored .env file "
            f"(see .env.example). The lab deliberately does not read {FORBIDDEN_KEY_VAR}."
        )
    return None


class RoleConfig(BaseModel):
    """Per-role model parameters, recorded in provenance and the cache key."""

    model_config = ConfigDict(extra="forbid")

    #: Lowest thinking setting. ``disabled`` is rejected on some models and
    #: ``between_tools`` on others, so it is configuration, not a constant, and
    #: ``make smoke`` decides which this model accepts (PLAN.md amendment 1).
    thinking: str = "between_tools"
    effort: Literal["low", "medium", "high"] = "low"
    max_tokens: int = 4000
    #: Put a cache breakpoint after the system prompt. Worth it only where the
    #: prefix is byte-stable across many calls — the user agent's fact sheet
    #: (PLAN.md P7).
    cache_system: bool = False


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str = "claude-sonnet-5-5"
    #: Used when the smoke test shows the default is unavailable or refuses too
    #: often (PLAN.md amendment 1).
    fallback: str = "claude-sonnet-5"
    fallback_thinking: str = "disabled"


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = "prompts/extract.txt"
    schema_path: str = Field(default="schema.json", alias="schema")
    tag: str = "v1"


class SimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 1


class StabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Derived from the floor runs (pairwise Jaccard), so there is no extra pass.
    enabled: bool = True


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    run_name: str = "default"
    personas: int = 12
    seed: int = 7
    script: str = "questions.yaml"
    noise_rates: list[float] = [0.0, 0.05, 0.1, 0.2]
    concurrency: int = 8
    models: ModelsConfig = ModelsConfig()
    roles: dict[str, RoleConfig] = {}
    sim: SimConfig = SimConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    stability: StabilityConfig = StabilityConfig()
    floor_runs: int = 5

    def role(self, name: str) -> RoleConfig:
        return self.roles.get(name, RoleConfig())

    @classmethod
    def load(cls, path: str | Path) -> Config:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)
