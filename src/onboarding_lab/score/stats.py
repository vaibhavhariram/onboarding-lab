"""Estimators for the metrics layer: tallies, Wilson intervals, pooling.

Kept apart from ``metrics.py`` because the choice of estimator is a decision in
its own right (PLAN.md M13). Two rules are enforced here rather than left to
each call site, because a call site is where they get silently broken:

- **A metric is a numerator over a denominator, never a bare float.** Pooling
  across personas sums the parts; averaging per-persona ratios would weight a
  persona with two claims the same as one with thirty.
- **An empty sample has no point estimate.** ``Tally.value`` is ``None`` when
  the denominator is zero, and the caller omits the key from ``Scores.metrics``
  with its ``0`` recorded in ``Scores.n`` (PLAN.md M12). ``0.0`` there reads as
  "we got everything wrong".

Wilson applies only to proportions, so ``wilson_interval`` is never called for a
``ratio`` key such as ``yield_per_minute`` — see ``metrics_schema.supports_wilson``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..metrics_schema import MetricKind

#: Two-sided 95%. Spelled out rather than imported so the interval in the report
#: cannot change because a dependency changed its default.
Z95 = 1.959963984540054

#: PLAN.md §3.12: ``floor[metric] = max(2 * sigma, 0.005)``.
MIN_FLOOR = 0.005


@dataclass(frozen=True)
class Tally:
    """One metric's raw parts.

    ``count`` keys carry only a numerator; ``rate`` and ``ratio`` keys carry a
    denominator that goes into ``Scores.n``. A ``rate`` is a proportion, so its
    numerator cannot exceed its denominator — violating that means the wrong
    denominator was used, which is the bug class this module exists to catch.
    """

    kind: MetricKind
    numerator: float
    denominator: int = 0

    def __post_init__(self) -> None:
        if self.denominator < 0:
            raise ValueError(f"negative denominator: {self.denominator}")
        if self.numerator < 0:
            raise ValueError(f"negative numerator: {self.numerator}")
        if self.kind == "count" and self.denominator:
            raise ValueError("a count has no denominator")
        if self.kind == "rate" and self.numerator > self.denominator:
            raise ValueError(
                f"proportion numerator {self.numerator} exceeds denominator {self.denominator}"
            )

    @property
    def defined(self) -> bool:
        """Counts are always defined; a rate or ratio needs a denominator."""
        return self.kind == "count" or self.denominator > 0

    @property
    def value(self) -> float | None:
        """The reportable number, or ``None`` when there is nothing to report."""
        if self.kind == "count":
            return float(self.numerator)
        if self.denominator == 0:
            return None
        return self.numerator / self.denominator

    def interval(self, *, z: float = Z95) -> tuple[float, float] | None:
        """Wilson interval, for proportions only."""
        if self.kind != "rate":
            return None
        return wilson_interval(self.numerator, self.denominator, z=z)


def count(n: float) -> Tally:
    return Tally("count", n)


def rate(numerator: float, denominator: int) -> Tally:
    return Tally("rate", numerator, denominator)


def ratio(numerator: float, denominator: int) -> Tally:
    return Tally("ratio", numerator, denominator)


def wilson_interval(successes: float, n: int, *, z: float = Z95) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion, or ``None`` when ``n == 0``.

    Wilson rather than normal-approximation because the rates of interest sit
    near 0 or 1 at the persona counts this harness runs, where the normal
    interval leaves the unit range. The interval ignores persona clustering;
    the README says so.
    """
    if n <= 0:
        return None
    if successes > n:
        raise ValueError(f"successes {successes} exceeds n {n}")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def pool(tallies: Iterable[Tally]) -> Tally:
    """Sum the parts. This is what "pooled, not a mean of ratios" means (M13)."""
    items = list(tallies)
    if not items:
        raise ValueError("cannot pool an empty sequence")
    kinds = {t.kind for t in items}
    if len(kinds) > 1:
        raise ValueError(f"cannot pool mixed kinds: {sorted(kinds)}")
    return Tally(
        items[0].kind,
        math.fsum(t.numerator for t in items),
        sum(t.denominator for t in items),
    )


def mean(values: Iterable[float]) -> float | None:
    """``None`` for an empty sample rather than a fabricated zero."""
    items = list(values)
    if not items:
        return None
    return math.fsum(items) / len(items)


def stdev(values: Iterable[float]) -> float | None:
    """Sample standard deviation, ``None`` below two observations.

    Sample (``n - 1``) rather than population: the floor runs are a sample of
    the harness's run-to-run variance, not the whole of it, and the unbiased
    estimator is the conservative choice for a gate.
    """
    items = list(values)
    if len(items) < 2:
        return None
    mu = math.fsum(items) / len(items)
    return math.sqrt(math.fsum((v - mu) ** 2 for v in items) / (len(items) - 1))


def pairwise_jaccard(sets: Sequence[frozenset[str] | set[str]]) -> Tally:
    """Pooled Jaccard agreement over every unordered pair.

    Pooled — intersections summed over unions summed — rather than a mean of
    per-pair ratios, for the same reason every other rate is pooled (M13), and
    it makes the result a proportion that takes a Wilson interval. Pairs whose
    union is empty are skipped: 0/0 is not agreement, it is no observation.
    """
    intersections = 0
    unions = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                continue
            intersections += len(sets[i] & sets[j])
            unions += len(union)
    return Tally("rate", intersections, unions)
