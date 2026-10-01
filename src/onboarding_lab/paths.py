"""Run identity and artifact paths.

One place computes every artifact path, because the spec's layout had two bugs
that silently corrupt results:

- Floor runs reuse one directory, so three runs overwrite each other and the
  standard deviation is computed over a single survivor (PLAN.md A1).
- Noise rates appeared in filenames as dotted floats (``p001.s1.n0.10.json``),
  which is ambiguous against the ``.`` separator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .hashing import canonical_json, short_hash

RUNS_ROOT = Path("runs")


def noise_token(rate: float) -> str:
    """``0.1 -> 'n010'``. No dots, fixed width, sorts correctly."""
    return f"n{round(rate * 100):03d}"


def transcript_stem(persona_id: str, sim_seed: int, rate: float, *, shuffled: bool = False) -> str:
    parts = [persona_id, f"s{sim_seed}"]
    if shuffled:
        parts.append("shuf")
    parts.append(noise_token(rate))
    return "__".join(parts)


def floor_tag(tag: str, k: int) -> str:
    """Floor run ``k`` gets its own tag, so artifacts cannot collide."""
    return f"{tag}.floor{k}"


def run_id(config_obj: object, *, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M")
    return f"{stamp}-{short_hash(canonical_json(config_obj))}"


@dataclass(frozen=True)
class RunPaths:
    """Every path under one run. Created lazily; nothing is written on access."""

    root: Path

    @classmethod
    def for_run(cls, rid: str, *, runs_root: Path | None = None) -> RunPaths:
        return cls((runs_root or RUNS_ROOT) / rid)

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def failures(self) -> Path:
        """Append-only log, so a dropped persona can never vanish silently."""
        return self.root / "failures.jsonl"

    @property
    def personas(self) -> Path:
        return self.root / "personas"

    @property
    def transcripts(self) -> Path:
        return self.root / "transcripts"

    def extractions(self, tag: str) -> Path:
        return self.root / "extractions" / tag

    def scores(self, tag: str) -> Path:
        return self.root / "scores" / tag

    def aggregate(self, tag: str) -> Path:
        return self.scores(tag) / "aggregate.json"

    def audit(self, tag: str) -> Path:
        return self.root / "audit" / tag

    def report(self, tag: str) -> Path:
        return self.root / "report" / tag

    def mkdirs(self) -> None:
        for p in (self.root, self.personas, self.transcripts):
            p.mkdir(parents=True, exist_ok=True)


def latest_run(runs_root: Path | None = None) -> str | None:
    """Newest run directory, for ``--run``'s default."""
    root = runs_root or RUNS_ROOT
    if not root.is_dir():
        return None
    runs = sorted((p.name for p in root.iterdir() if p.is_dir()), reverse=True)
    return runs[0] if runs else None


def append_failure(paths: RunPaths, record: dict) -> None:
    """Record a failed persona, simulation, or extraction.

    Silently dropping items changes every number in the report, so failures are
    recorded and counted rather than skipped (PLAN.md A2).
    """
    paths.root.mkdir(parents=True, exist_ok=True)
    with paths.failures.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
