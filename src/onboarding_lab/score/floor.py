"""The measured noise floor, and the diff gate built on it.

Sampling parameters do not exist on the models this harness runs against
(PLAN.md amendment 1), so there is no ``temperature: 0`` to hide behind and the
harness does **not** assume deterministic output. Reproducibility rests on the
content-hash cache plus this measured floor (PLAN.md P5), which makes the floor
the only thing that makes the diff gate sound — hence ``floor_runs: 5`` rather
than 3, because a standard deviation from three observations is a poor estimate
of the variance a gate is about to trust.

``floor[metric] = max(2 * sigma, 0.005)`` over the floor runs, where sigma is
the run-to-run standard deviation of that metric with only the extraction nonce
changed. The 0.5-percentage-point minimum matters because under ``FakeProvider``
variance is exactly zero (PLAN.md D3), and a zero floor would fail a candidate
on a rounding difference.

Only ``metrics_schema.GATED_KEYS`` gate, and only at noise rate 0.0 (PLAN.md
M15): a regression at 0.2 noise is as likely to be the noise injector as the
system under test, so other levels are reported un-gated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .. import metrics_schema
from ..metrics_schema import GATED_KEYS
from .stats import MIN_FLOOR, Tally, mean, pairwise_jaccard, stdev

#: Exit codes for ``lab diff`` (SPEC §3.15).
EXIT_PASS = 0
EXIT_REGRESSION = 1
EXIT_INVALID = 2

#: The noise level the gate runs at. Everything else is reported, not gated.
GATED_NOISE_RATE = 0.0

#: The vocabulary the report renders: ``regression`` for a gated drop beyond the
#: floor, ``ungated`` for a metric reported but not gated (including every metric
#: at a non-zero noise rate), and ``missing`` where one side has no value.
Verdict = Literal["pass", "regression", "ungated", "missing"]


def stability_extraction(captured_sets: Sequence[frozenset[str] | set[str]]) -> Tally:
    """Pairwise Jaccard of captured fact-id sets across the floor runs.

    No extra pass: the floor runs already re-extract the same transcript with
    different nonces, which is exactly the comparison a stability number wants.
    Needs at least two runs with a non-empty union, or the denominator is zero
    and the key is omitted.
    """
    return pairwise_jaccard(captured_sets)


@dataclass(frozen=True)
class MetricFloor:
    """One metric's run-to-run variance, and the floor derived from it."""

    metric: str
    baseline_mean: float | None
    sigma: float | None
    floor: float
    runs: int

    @property
    def measured(self) -> bool:
        """False when the floor fell back to its minimum for want of data.

        A gate built on an unmeasured floor is still sound — the minimum is the
        conservative end — but the report says which it is rather than letting
        0.005 pass for a measurement.
        """
        return self.sigma is not None


def compute_floor(
    runs: Sequence[Mapping[str, float | None]],
    *,
    keys: Iterable[str] | None = None,
) -> dict[str, MetricFloor]:
    """Per-metric floor over the floor runs' aggregates.

    Each element of ``runs`` is one floor run's pooled metrics. ``None`` values
    (an undefined rate) are not observations and are skipped rather than read as
    zero; a metric with fewer than two observations gets the minimum floor and
    ``sigma is None``.
    """
    wanted = set(keys) if keys is not None else {k for run in runs for k in run}
    unknown = sorted(k for k in wanted if not metrics_schema.is_known(k))
    if unknown:
        raise ValueError(f"metric keys not in metrics_schema: {unknown}")

    out: dict[str, MetricFloor] = {}
    for key in sorted(wanted):
        values = [float(v) for v in (run.get(key) for run in runs) if v is not None]
        sigma = stdev(values)
        out[key] = MetricFloor(
            metric=key,
            baseline_mean=mean(values),
            sigma=sigma,
            floor=max(2.0 * sigma, MIN_FLOOR) if sigma is not None else MIN_FLOOR,
            runs=len(values),
        )
    return out


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DiffRow:
    """One row of the diff table. Track D renders it; nothing here formats."""

    metric: str
    baseline_mean: float | None
    candidate: float | None
    delta: float | None
    floor: float | None
    verdict: Verdict
    gated: bool

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "baseline_mean": self.baseline_mean,
            "candidate": self.candidate,
            "delta": self.delta,
            "floor": self.floor,
            "verdict": self.verdict,
            "gated": self.gated,
        }


@dataclass(frozen=True)
class DiffResult:
    rows: tuple[DiffRow, ...]
    exit_code: int
    noise_rate: float
    gated: bool
    invalid_reason: str | None = None

    @property
    def regressions(self) -> tuple[DiffRow, ...]:
        return tuple(r for r in self.rows if r.verdict == "regression")

    def table(self) -> list[dict]:
        return [r.as_dict() for r in self.rows]

    def to_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "noise_rate": self.noise_rate,
            "gated": self.gated,
            "invalid_reason": self.invalid_reason,
            "rows": self.table(),
        }


def diff(
    *,
    baseline: Mapping[str, float | None],
    candidate: Mapping[str, float | None],
    floors: Mapping[str, MetricFloor] | Mapping[str, float],
    noise_rate: float = GATED_NOISE_RATE,
    gated_keys: Sequence[str] = GATED_KEYS,
) -> DiffResult:
    """Compare two tags' pooled metrics against the measured floor.

    Exit 0 pass, 1 regression beyond the floor on a gated metric, 2 invalid
    input. Invalid means the comparison could not be made — an empty side, an
    unknown metric key, or a gated metric missing its value or its floor at the
    gated noise rate. It is deliberately not a pass: a gate that cannot see its
    metric has not cleared it.

    ``baseline`` is the mean over the floor runs, which is what the floor was
    measured against; passing a single run's metrics compares against one draw
    from the distribution the floor describes.
    """
    gated = noise_rate == GATED_NOISE_RATE
    reason = _invalid_reason(baseline, candidate, floors, gated_keys, gated=gated)
    if reason is not None:
        return DiffResult(
            rows=(),
            exit_code=EXIT_INVALID,
            noise_rate=noise_rate,
            gated=gated,
            invalid_reason=reason,
        )

    gated_set = set(gated_keys)
    rows: list[DiffRow] = []
    for metric in sorted(set(baseline) | set(candidate)):
        base = baseline.get(metric)
        cand = candidate.get(metric)
        floor = _floor_value(floors, metric)
        is_gated = gated and metric in gated_set
        if base is None or cand is None:
            rows.append(DiffRow(metric, base, cand, None, floor, "missing", is_gated))
            continue
        delta = cand - base
        if not is_gated:
            verdict: Verdict = "ungated"
        elif floor is not None and cand < base - floor:
            verdict = "regression"
        else:
            verdict = "pass"
        rows.append(DiffRow(metric, base, cand, delta, floor, verdict, is_gated))

    regressed = any(r.verdict == "regression" for r in rows)
    return DiffResult(
        rows=tuple(rows),
        exit_code=EXIT_REGRESSION if regressed else EXIT_PASS,
        noise_rate=noise_rate,
        gated=gated,
    )


def _floor_value(
    floors: Mapping[str, MetricFloor] | Mapping[str, float], metric: str
) -> float | None:
    value = floors.get(metric)
    if value is None:
        return None
    return value.floor if isinstance(value, MetricFloor) else float(value)


def _invalid_reason(
    baseline: Mapping[str, float | None],
    candidate: Mapping[str, float | None],
    floors: Mapping[str, MetricFloor] | Mapping[str, float],
    gated_keys: Sequence[str],
    *,
    gated: bool,
) -> str | None:
    if not baseline:
        return "baseline has no metrics"
    if not candidate:
        return "candidate has no metrics"
    unknown = sorted(k for k in set(baseline) | set(candidate) if not metrics_schema.is_known(k))
    if unknown:
        return f"metric keys not in metrics_schema: {unknown}"
    if not gated:
        return None
    for key in gated_keys:
        if baseline.get(key) is None:
            return f"gated metric {key!r} is undefined in the baseline"
        if candidate.get(key) is None:
            return f"gated metric {key!r} is undefined in the candidate"
        if _floor_value(floors, key) is None:
            return f"gated metric {key!r} has no measured floor"
    return None
