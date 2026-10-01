"""The measured noise floor and the diff gate.

There is no ``temperature: 0`` on the models this harness runs against, so the
gate's soundness rests entirely on this floor (PLAN.md P5). These tests pin the
floor arithmetic and all three exit codes.
"""

from __future__ import annotations

import pytest

from onboarding_lab.metrics_schema import GATED_KEYS
from onboarding_lab.score import floor as fl
from onboarding_lab.score.stats import MIN_FLOOR


def runs(**series: list[float | None]) -> list[dict[str, float | None]]:
    """Transpose ``metric -> per-run values`` into one mapping per floor run."""
    length = {len(v) for v in series.values()}
    assert len(length) == 1, "every metric needs a value slot per run"
    return [{k: v[i] for k, v in series.items()} for i in range(length.pop())]


# -- floor math ---------------------------------------------------------------


def test_floor_is_two_sigma() -> None:
    measured = fl.compute_floor(runs(precision=[0.8, 0.9]))["precision"]
    assert measured.baseline_mean == pytest.approx(0.85)
    assert measured.sigma == pytest.approx(0.0707106781, abs=1e-9)
    assert measured.floor == pytest.approx(2 * 0.0707106781, abs=1e-9)
    assert measured.runs == 2
    assert measured.measured


def test_floor_never_drops_below_its_minimum() -> None:
    """Under FakeProvider variance is exactly zero (PLAN.md D3)."""
    measured = fl.compute_floor(runs(precision=[0.8, 0.8, 0.8, 0.8, 0.8]))["precision"]
    assert measured.sigma == 0.0
    assert measured.floor == MIN_FLOOR


def test_an_undefined_run_value_is_skipped_not_read_as_zero() -> None:
    """A rate omitted for a zero denominator is no observation, not a zero."""
    measured = fl.compute_floor(runs(precision=[0.8, None, 0.9]))["precision"]
    assert measured.runs == 2
    assert measured.baseline_mean == pytest.approx(0.85)


def test_one_observation_gives_the_minimum_floor_and_no_sigma() -> None:
    measured = fl.compute_floor(runs(precision=[0.8, None]))["precision"]
    assert measured.sigma is None
    assert measured.floor == MIN_FLOOR
    assert not measured.measured, "the report must not print 0.005 as a measurement"


def test_a_metric_absent_from_every_run_is_still_reported_as_unmeasured() -> None:
    measured = fl.compute_floor(runs(precision=[0.8, 0.9]), keys=["recall"])
    assert measured["recall"].baseline_mean is None
    assert measured["recall"].floor == MIN_FLOOR
    assert "precision" not in measured


def test_unknown_metric_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="not in metrics_schema"):
        fl.compute_floor([{"made_up": 0.5}])


# -- stability from the floor runs -------------------------------------------


def test_stability_is_pairwise_jaccard_of_captured_sets() -> None:
    """Derived from the floor runs, so no extra extraction pass is needed."""
    tally = fl.stability_extraction(
        [frozenset({"f01", "f02"}), frozenset({"f01", "f02"}), frozenset({"f01", "f03"})]
    )
    assert tally.numerator == 4
    assert tally.denominator == 8


def test_perfect_agreement_is_one() -> None:
    sets = [frozenset({"f01", "f02"})] * 5
    assert fl.stability_extraction(sets).value == 1.0


def test_a_single_run_has_no_stability_number() -> None:
    assert fl.stability_extraction([frozenset({"f01"})]).value is None


# -- diff ---------------------------------------------------------------------

BASE = dict.fromkeys(GATED_KEYS, 0.80)
FLOORS = dict.fromkeys(GATED_KEYS, 0.01)


def test_an_unchanged_candidate_passes() -> None:
    result = fl.diff(baseline=BASE, candidate=dict(BASE), floors=FLOORS)
    assert result.exit_code == fl.EXIT_PASS
    assert {r.verdict for r in result.rows} == {"pass"}


def test_a_drop_inside_the_floor_passes() -> None:
    candidate = {**BASE, "precision": 0.795}
    result = fl.diff(baseline=BASE, candidate=candidate, floors=FLOORS)
    assert result.exit_code == fl.EXIT_PASS


def test_a_drop_beyond_the_floor_exits_one() -> None:
    candidate = {**BASE, "precision": 0.60}
    result = fl.diff(baseline=BASE, candidate=candidate, floors=FLOORS)
    assert result.exit_code == fl.EXIT_REGRESSION
    assert [r.metric for r in result.regressions] == ["precision"]
    row = next(r for r in result.rows if r.metric == "precision")
    assert row.delta == pytest.approx(-0.20)
    assert row.floor == 0.01
    assert row.gated


def test_an_ungated_metric_is_reported_not_gated() -> None:
    """``recall`` regressing is a finding, not a failure: the gate uses
    ``recall_elicited`` so it measures extraction, not the interview (M6)."""
    base = {**BASE, "recall": 0.9}
    candidate = {**BASE, "recall": 0.1}
    result = fl.diff(baseline=base, candidate=candidate, floors={**FLOORS, "recall": 0.01})
    assert result.exit_code == fl.EXIT_PASS
    assert next(r for r in result.rows if r.metric == "recall").verdict == "ungated"


def test_other_noise_levels_are_reported_un_gated() -> None:
    """A regression under noise is as likely to be the injector as the SUT (M15)."""
    candidate = {**BASE, "precision": 0.10}
    result = fl.diff(baseline=BASE, candidate=candidate, floors=FLOORS, noise_rate=0.2)
    assert result.exit_code == fl.EXIT_PASS
    assert not result.gated
    assert {r.verdict for r in result.rows} == {"ungated"}


def test_measured_floors_are_accepted_in_place_of_plain_numbers() -> None:
    floors = fl.compute_floor([dict.fromkeys(GATED_KEYS, v) for v in (0.80, 0.82)])
    result = fl.diff(baseline=BASE, candidate={**BASE, "precision": 0.75}, floors=floors)
    assert result.exit_code == fl.EXIT_REGRESSION


# -- exit 2: the comparison could not be made --------------------------------


def test_an_empty_side_exits_two() -> None:
    assert fl.diff(baseline={}, candidate=dict(BASE), floors=FLOORS).exit_code == fl.EXIT_INVALID
    assert fl.diff(baseline=BASE, candidate={}, floors=FLOORS).exit_code == fl.EXIT_INVALID


def test_an_unknown_metric_key_exits_two() -> None:
    result = fl.diff(baseline=BASE, candidate={**BASE, "made_up": 0.5}, floors=FLOORS)
    assert result.exit_code == fl.EXIT_INVALID
    assert "metrics_schema" in (result.invalid_reason or "")


def test_a_gated_metric_without_a_value_exits_two_rather_than_passing() -> None:
    """A gate that cannot see its metric has not cleared it."""
    candidate = {k: v for k, v in BASE.items() if k != "precision"}
    result = fl.diff(baseline=BASE, candidate=candidate, floors=FLOORS)
    assert result.exit_code == fl.EXIT_INVALID
    assert "precision" in (result.invalid_reason or "")


def test_a_gated_metric_without_a_floor_exits_two() -> None:
    floors = {k: v for k, v in FLOORS.items() if k != "recall_elicited"}
    result = fl.diff(baseline=BASE, candidate=dict(BASE), floors=floors)
    assert result.exit_code == fl.EXIT_INVALID
    assert "recall_elicited" in (result.invalid_reason or "")


def test_a_missing_ungated_metric_is_a_row_not_an_exit_code() -> None:
    base = {**BASE, "recall": 0.9}
    result = fl.diff(baseline=base, candidate=dict(BASE), floors=FLOORS)
    assert result.exit_code == fl.EXIT_PASS
    assert next(r for r in result.rows if r.metric == "recall").verdict == "missing"


# -- the table Track D renders -----------------------------------------------


def test_the_table_carries_the_six_spec_columns() -> None:
    result = fl.diff(baseline=BASE, candidate=dict(BASE), floors=FLOORS)
    row = result.table()[0]
    assert set(row) == {
        "metric",
        "baseline_mean",
        "candidate",
        "delta",
        "floor",
        "verdict",
        "gated",
    }
    assert "rows" in result.to_dict()
