"""Canonical registry of metric keys.

Single source of truth so ``test_e2e`` can assert aggregate completeness instead
of hardcoding key strings in two places (PLAN.md M12).

Three kinds, because they aggregate and interval differently (PLAN.md M13):

- ``rate``  — a proportion with a denominator in ``Scores.n``. Pooled across
  personas, 95% Wilson interval printed.
- ``count`` — an integer total. Summed across personas, no interval.
- ``ratio`` — a quotient that is not a proportion (facts per minute). Pooled as
  numerator-sum / denominator-sum. **No Wilson interval**; Wilson does not apply.

A rate whose denominator is zero is **omitted** from ``Scores.metrics`` with its
``0`` recorded in ``Scores.n`` — never ``0.0``, never ``NaN``.
"""

from __future__ import annotations

from typing import Literal

MetricKind = Literal["rate", "count", "ratio"]

#: Metrics the CI diff gate fails on when a candidate drops below
#: ``baseline_mean - floor``. Everything else is reported but not gated.
#: ``recall_elicited`` rather than ``recall``: the gate should measure extraction,
#: not whether the interview happened to elicit the fact (PLAN.md M6).
GATED_KEYS: tuple[str, ...] = (
    "precision",
    "precision_dealbreakers",
    "recall_elicited",
    "recall_dealbreakers",
)

STATIC_RATES: tuple[str, ...] = (
    "precision",
    "hallucination_rate",
    "precision_dealbreakers",
    "recall",
    "recall_elicited",
    "recall_dealbreakers",
    "recall_exact",
    "offsheet_rate",
    "abstention_rate",
    "followup_share",
    "judge_error_rate",
    "span_invalid_rate_first_pass",
    "span_invalid_rate_final",
    "stability_extraction",
    "disclosure_mismatch",
)

STATIC_COUNTS: tuple[str, ...] = (
    "untagged_capture",
    "claims_deduped",
)

#: ``prefix -> kind``. A key is ``"<prefix>.<suffix>"`` with a non-empty suffix.
DYNAMIC_PREFIXES: dict[str, MetricKind] = {
    "recall_by_disclosure": "rate",
    "recall_by_field": "rate",
    "precision_by_field": "rate",
    "recall_exact_by_field": "rate",
    "exact_floor": "rate",
    "followup_share_by_style": "rate",
    "yield": "count",
    "yield_per_minute": "ratio",
}


def classify(key: str) -> MetricKind:
    """Return the kind of a metric key. Raises ``KeyError`` on an unknown key."""
    if key in STATIC_RATES:
        return "rate"
    if key in STATIC_COUNTS:
        return "count"
    prefix, _, suffix = key.partition(".")
    if suffix and prefix in DYNAMIC_PREFIXES:
        return DYNAMIC_PREFIXES[prefix]
    raise KeyError(f"unknown metric key: {key!r}")


def is_known(key: str) -> bool:
    try:
        classify(key)
    except KeyError:
        return False
    return True


def needs_denominator(key: str) -> bool:
    """Rates and ratios carry a denominator in ``Scores.n``; counts do not."""
    return classify(key) in ("rate", "ratio")


def supports_wilson(key: str) -> bool:
    """Only proportions get a Wilson interval (PLAN.md M13)."""
    return classify(key) == "rate"


def expected_keys(
    *,
    fields: tuple[str, ...],
    exact_fields: tuple[str, ...],
    disclosures: tuple[str, ...],
    styles: tuple[str, ...],
    question_ids: tuple[str, ...],
) -> frozenset[str]:
    """Every key an aggregate is expected to carry, for ``test_e2e``.

    Per-persona score files legitimately omit zero-denominator rates; the
    aggregate carries the full set so the report never has a missing row.
    """
    keys: set[str] = set(STATIC_RATES) | set(STATIC_COUNTS)
    for f in fields:
        keys.add(f"recall_by_field.{f}")
        keys.add(f"precision_by_field.{f}")
    for f in exact_fields:
        keys.add(f"exact_floor.{f}")
        keys.add(f"recall_exact_by_field.{f}")
    for d in disclosures:
        keys.add(f"recall_by_disclosure.{d}")
    for s in styles:
        keys.add(f"followup_share_by_style.{s}")
    for q in question_ids:
        keys.add(f"yield.{q}")
        keys.add(f"yield_per_minute.{q}")
    return frozenset(keys)
