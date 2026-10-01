"""Report, chart, and diff rendering.

Everything here runs against ``tests/fixtures/aggregate_synthetic.json``, which
is a **test fixture only**: every number in it is hand-made to exercise a code
path and none of it is a measurement. It must never reach README.md or the
committed demo report, which render from a real run's artifacts
(CLAUDE.md rule 5). ``test_the_fixture_says_it_is_synthetic`` is what keeps that
honest.

The rules worth regression-testing are the ones that are easy to break quietly:
a placeholder where a number was never computed, a fabricated audit agreement, an
interval on something that is not a proportion, and a gated metric printed
without the measured floor that makes gating it sound.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import pytest

# Matplotlib warns when it cannot write its font cache, and pytest is configured
# to turn UserWarning into an error — so the cache directory is pinned before
# anything imports matplotlib. The Makefile exports the same variable; this keeps
# a bare `pytest` green too.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

from onboarding_lab import metrics_schema
from onboarding_lab.models import EXACT_FIELDS, SCORED_FIELDS
from onboarding_lab.report import charts, render

FIXTURE = Path(__file__).parent / "fixtures" / "aggregate_synthetic.json"


@pytest.fixture(scope="module")
def aggregate() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def diff_data(aggregate: dict) -> dict:
    return aggregate["_diff_synthetic"]


@pytest.fixture(scope="module")
def html(aggregate: dict) -> str:
    return render.render_html(aggregate)


@pytest.fixture(scope="module")
def summary(aggregate: dict) -> str:
    return render.render_summary(aggregate)


# -- the fixture itself -------------------------------------------------------


def test_the_fixture_says_it_is_synthetic(aggregate: dict) -> None:
    """It must be impossible to mistake this file for a measurement."""
    note = aggregate["_note"]
    assert "SYNTHETIC TEST FIXTURE" in note
    assert "README.md" in note and "NEVER" in note


def test_every_fixture_metric_key_is_in_the_registry(aggregate: dict) -> None:
    """Otherwise the fixture drifts away from what the scorer can actually emit."""
    unknown = [
        key
        for key in list(aggregate["metrics"]) + list(aggregate["n"])
        if not metrics_schema.is_known(key)
    ]
    assert unknown == []


# -- rendering ----------------------------------------------------------------


def test_index_and_summary_render_to_disk(aggregate: dict, tmp_path: Path) -> None:
    written = render.write_report(aggregate, tmp_path)
    assert set(written) == {"index.html", "summary.md"}
    assert written["index.html"].read_text().startswith("<!DOCTYPE html>")
    assert "# Extraction quality summary" in written["summary.md"].read_text()


def test_render_takes_a_loaded_aggregate_not_a_path(aggregate: dict) -> None:
    """`make demo` must work from committed artifacts with no provider, no key,
    and no network (PLAN.md D1), so the entry points take dicts."""
    assert render.render_html(aggregate)
    assert render.render_summary(aggregate)


def test_all_seven_sections_are_present_in_order(html: str) -> None:
    headings = re.findall(r"<h2>.*?</h2>", html, flags=re.DOTALL)
    assert len(headings) == 7
    expected = [
        "Headline numbers",
        "Per-field precision and recall",
        "Per-question yield",
        "transcription noise",
        "Recall by disclosure type",
        "Hallucination examples",
        "Run manifest",
    ]
    for heading, want in zip(headings, expected, strict=True):
        assert want in heading


def test_output_contains_no_javascript_and_no_external_url(html: str) -> None:
    """One self-contained file: inline CSS, inline SVG, nothing fetched."""
    assert not re.search(r"<\s*script", html, flags=re.IGNORECASE)
    assert "javascript:" not in html.lower()
    assert re.findall(r"\son[a-z]+\s*=", html) == []
    assert "@import" not in html
    # The only remaining http(s) strings are the SVG/xlink XML namespace
    # declarations, which name a spec and are never fetched.
    assert set(re.findall(r"https?://[^\"'\s<>)]+", html)) == {
        "http://www.w3.org/1999/xlink",
        "http://www.w3.org/2000/svg",
    }
    for attr in ("src", "href"):
        assert re.findall(rf'{attr}="https?:', html) == []


def test_deferred_metrics_are_not_rendered(html: str, summary: str) -> None:
    """`decoy_rate` and `specificity_match` are cut to v1.1 (PLAN.md amendment 7)."""
    for key in ("decoy_rate", "specificity_match"):
        assert key not in html
        assert key not in summary


# -- the degraded banner ------------------------------------------------------


def _with_judge(aggregate: dict, *, judged: int, errors: int, span_invalid: int = 0) -> dict:
    data = deepcopy(aggregate)
    data["judge"] = {
        "calls": judged,
        "judged_claims": judged,
        "errors": errors,
        "span_invalid": span_invalid,
        "span_invalid_first_pass": span_invalid,
    }
    return data


def test_degraded_banner_appears_above_five_percent(aggregate: dict) -> None:
    html = render.render_html(_with_judge(aggregate, judged=100, errors=6))
    assert "DEGRADED" in html
    assert "above the 5% threshold" in html
    assert "DEGRADED" in render.render_summary(_with_judge(aggregate, judged=100, errors=6))


def test_degraded_banner_absent_at_or_below_five_percent(aggregate: dict) -> None:
    """The gate is strictly greater than 5%, pooled over judged claims (M11)."""
    at_threshold = render.render_html(_with_judge(aggregate, judged=100, errors=5))
    assert "DEGRADED" not in at_threshold
    assert "DEGRADED" not in render.render_html(_with_judge(aggregate, judged=100, errors=0))


def test_judge_error_rate_denominator_is_judged_claims(aggregate: dict) -> None:
    """A denominator including exact-method claims the judge never saw would
    deflate the rate and weaken the 5% gate (PLAN.md M11)."""
    data = _with_judge(aggregate, judged=50, errors=2, span_invalid=1)
    judge = render.build_context(data)["judge"]
    assert judge["metric"].value == pytest.approx(3 / 50)
    assert judge["metric"].n == 50


def test_judge_error_rate_is_absent_when_nothing_was_judged(aggregate: dict) -> None:
    """Zero judged claims is not a zero error rate (CLAUDE.md rule 6)."""
    data = _with_judge(aggregate, judged=0, errors=0)
    judge = render.build_context(data)["judge"]
    assert judge["metric"].value is None
    assert judge["degraded"] is False
    assert "DEGRADED" not in render.render_html(data)


# -- audit labels (D2) --------------------------------------------------------


def test_no_audit_labels_prints_the_red_banner_and_no_number(
    html: str, summary: str, aggregate: dict
) -> None:
    assert render.NOT_AUDITED == "judge not yet audited"
    for text in (html, summary):
        assert render.NOT_AUDITED in text
    # Red, and structurally so: the class the stylesheet colours with --danger.
    assert f'class="absent">{render.NOT_AUDITED}' in html
    # And no fabricated agreement anywhere.
    assert render.build_context(aggregate)["audit"] is None
    assert "Audit agreement from" not in html


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param({}, id="empty-file"),
        pytest.param({"a1b2": None, "c3d4": None}, id="sampled-but-unlabelled"),
    ],
)
def test_unfilled_labels_are_not_counted_as_disagreement(
    aggregate: dict, labels: dict[str, bool | None]
) -> None:
    """`lab audit` writes one `agree: null` per item; counting nulls as false
    would turn an unfinished label pass into a bad judge."""
    assert render.audit_agreement(labels) is None
    assert render.NOT_AUDITED in render.render_html(aggregate, audit_labels=labels)


def test_audit_agreement_renders_with_n_and_an_interval(aggregate: dict) -> None:
    labels = {"h1": True, "h2": True, "h3": False, "h4": None}
    agreement = render.audit_agreement(labels, source="committed fixture (test)")
    assert agreement is not None
    assert (agreement["agreed"], agreement["labeled"], agreement["sampled"]) == (2, 3, 4)
    assert agreement["metric"].value == pytest.approx(2 / 3)
    assert agreement["metric"].interval is not None

    html = render.render_html(aggregate, audit_labels=labels, audit_source="run-local (test)")
    assert render.NOT_AUDITED not in html
    assert "Audit agreement from 2/3 hand-filled labels" in html


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("labels:\n  deadbeef:\n    agree: true\n", id="lab-audit-spelling"),
        pytest.param("deadbeef: true\n", id="hand-shortened-spelling"),
    ],
)
def test_labels_load_from_either_spelling(tmp_path: Path, body: str) -> None:
    path = tmp_path / "labels.yaml"
    path.write_text(body)
    labels, source = render.load_audit_labels(run_local=path, committed=None)
    assert labels == {"deadbeef": True}
    assert source is not None and "run-local" in source


def test_run_local_labels_win_over_the_committed_fixture(tmp_path: Path) -> None:
    """Preference order, with the committed copy as the fallback that survives
    `make clean` (PLAN.md D2)."""
    run_local = tmp_path / "run.yaml"
    committed = tmp_path / "fixture.yaml"
    run_local.write_text("h1: true\n")
    committed.write_text("h1: false\n")

    labels, source = render.load_audit_labels(run_local=run_local, committed=committed)
    assert labels == {"h1": True} and "run-local" in source

    labels, source = render.load_audit_labels(run_local=tmp_path / "gone.yaml", committed=committed)
    assert labels == {"h1": False} and "committed fixture" in source

    labels, source = render.load_audit_labels(
        run_local=tmp_path / "gone.yaml", committed=tmp_path / "also-gone.yaml"
    )
    assert labels == {} and source is None


def test_labels_are_keyed_by_content_hash_not_position() -> None:
    """Position keys would silently re-point every label at a different alignment
    after a resample, which is why D2 specifies content hashes."""
    labels, _ = render.load_audit_labels(run_local=None, committed=render.COMMITTED_AUDIT_LABELS)
    assert all(not key.isdigit() for key in labels)


# -- absent is not zero -------------------------------------------------------


def test_an_absent_metric_renders_as_n_a_not_zero(aggregate: dict) -> None:
    """The fixture omits `precision_by_field.communication_style` with n=0: a
    zero-denominator rate is omitted upstream, and 0.0 would read as
    'we got everything wrong' (CLAUDE.md rule 6)."""
    assert "precision_by_field.communication_style" not in aggregate["metrics"]
    assert aggregate["n"]["precision_by_field.communication_style"] == 0

    metric = render.Aggregate(aggregate).metric(
        "precision_by_field.communication_style", "communication_style"
    )
    assert metric.value is None
    assert metric.present is False
    assert metric.value_text == "n/a"
    assert metric.interval is None


def test_an_absent_metric_is_marked_absent_in_the_html(aggregate: dict) -> None:
    rows = render.build_context(aggregate)["field_rows"]
    row = next(r for r in rows if r["field"] == "communication_style")
    assert row["precision"].value is None
    assert row["recall"].value is not None  # only the precision side is absent
    html = render.render_html(aggregate)
    assert 'class="absent"' in html


def test_every_scored_field_gets_a_row_even_when_both_rates_are_absent() -> None:
    """A field the extraction never produced must be visible, not missing."""
    rows = render.build_context({"metrics": {}, "n": {}})["field_rows"]
    assert [r["field"] for r in rows] == list(SCORED_FIELDS)
    assert all(r["precision"].value is None and r["recall"].value is None for r in rows)


def test_an_empty_aggregate_renders_without_inventing_anything() -> None:
    html = render.render_html({})
    summary = render.render_summary({})
    assert "n/a" in html and "n/a" in summary
    assert render.NOT_AUDITED in html
    assert "0.0%" not in html


# -- intervals ----------------------------------------------------------------


def test_wilson_intervals_appear_for_rates(aggregate: dict) -> None:
    precision = next(m for m in render.build_context(aggregate)["headline"] if m.key == "precision")
    assert metrics_schema.supports_wilson("precision")
    assert precision.interval is not None
    lo, hi = precision.interval
    assert 0.0 <= lo < precision.value < hi <= 1.0
    assert "95% CI" in precision.interval_text


def test_no_wilson_interval_for_yield_per_minute(aggregate: dict) -> None:
    """It is a ratio, not a proportion; Wilson does not apply (PLAN.md M13)."""
    rows = render.build_context(aggregate)["yield_section"]["rows"]
    assert rows, "the fixture must carry per-question yield"
    for row in rows:
        assert row["per_minute"].kind == "ratio"
        assert metrics_schema.supports_wilson(row["per_minute"].key) is False
        assert row["per_minute"].interval is None
        assert row["per_minute"].interval_text == ""


def test_no_wilson_interval_for_counts(aggregate: dict) -> None:
    for key in ("untagged_capture", "claims_deduped"):
        metric = render.Aggregate(aggregate).metric(key, key)
        assert metric.kind == "count"
        assert metric.interval is None
        assert metric.n_cell == "—"


def test_wilson_is_none_for_a_zero_denominator() -> None:
    assert render.wilson(0.5, 0) is None
    lo, hi = render.wilson(1.0, 10)
    assert hi == pytest.approx(1.0) and lo < 1.0
    lo, hi = render.wilson(0.0, 10)
    assert lo == 0.0 and hi > 0.0


# -- the measured floor (P5) --------------------------------------------------


def test_the_measured_floor_appears_beside_every_gated_metric(aggregate: dict) -> None:
    """The harness does not claim deterministic output; it measures run-to-run
    variance and gates only beyond it, so the measurement has to be visible."""
    headline = render.build_context(aggregate)["headline"]
    gated = [m for m in headline if m.gated]
    assert {m.key for m in gated} == set(metrics_schema.GATED_KEYS)
    for metric in gated:
        assert metric.floor is not None, metric.key
        assert metric.floor_text.endswith("pp")


def test_a_gated_metric_without_a_measured_floor_says_so(aggregate: dict) -> None:
    """Silently omitting the floor would make the gate look unconditional."""
    data = deepcopy(aggregate)
    data["floor"] = {}
    html = render.render_html(data)
    assert "floor n/a" in html
    assert "no measured floor in the aggregate" in html


def test_floor_is_printed_in_the_summary_too(summary: str) -> None:
    assert "measured floor" in summary
    assert "pp" in summary


# -- charts -------------------------------------------------------------------


def _svgs(html: str) -> list[str]:
    return re.findall(r"<svg.*?</svg>", html, flags=re.DOTALL)


def test_both_charts_are_valid_svg_embedded_inline(html: str) -> None:
    found = _svgs(html)
    assert len(found) == 2, "exactly the two charts §3.14 specifies"
    for svg in found:
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")
        # viewBox kept, width/height dropped, so it scales on screen and in print.
        assert root.get("viewBox")
        assert root.get("width") is None and root.get("height") is None
        assert root.get("{http://www.w3.org/2000/svg}role") or root.get("role") == "img"
        assert root.get("aria-label")
    assert "<?xml" not in html and "<!DOCTYPE svg" not in html


def test_chart_ids_do_not_collide_between_figures(html: str) -> None:
    """Matplotlib numbers ids from 1 per figure, so two inline figures would
    otherwise share clip-path ids and the second would clip against the first."""
    first, second = (set(re.findall(r'id="([^"]+)"', svg)) for svg in _svgs(html))
    assert first and second
    assert not (first & second)
    # Every internal reference resolves inside its own figure.
    for svg, ids in zip(_svgs(html), (first, second), strict=True):
        for ref in re.findall(r"url\(#([^)]+)\)", svg):
            assert ref in ids


def test_charts_are_deterministic() -> None:
    points = [("q01", 1.25), ("q02", 0.80)]
    assert charts.yield_per_minute_chart(points) == charts.yield_per_minute_chart(points)
    curve = [(0.0, 0.9, 0.8), (0.1, 0.8, 0.7)]
    assert charts.noise_curve_chart(curve) == charts.noise_curve_chart(curve)


def test_the_noise_chart_skips_an_absent_level_instead_of_plotting_zero() -> None:
    both = charts.noise_curve_chart([(0.0, 0.9, 0.8), (0.1, 0.8, 0.7)])
    missing = charts.noise_curve_chart([(0.0, 0.9, 0.8), (0.1, 0.8, None)])
    assert both != missing
    assert charts.noise_curve_chart([(0.0, None, None)])  # nothing to plot, still valid


def test_an_empty_chart_input_is_an_error_not_an_empty_figure() -> None:
    with pytest.raises(ValueError, match="at least one question"):
        charts.yield_per_minute_chart([])
    with pytest.raises(ValueError, match="at least one noise level"):
        charts.noise_curve_chart([])


def test_the_follow_up_overlay_is_only_drawn_where_a_share_was_recorded(
    aggregate: dict,
) -> None:
    """Nothing is estimated: no recorded share means no subdivided bar."""
    context = render.build_context(aggregate)
    assert context["yield_section"]["followup_marked"] is True
    assert "that question's own follow-up share" in render.render_html(aggregate)

    without = deepcopy(aggregate)
    del without["followup_share_by_question"]
    assert render.build_context(without)["yield_section"]["followup_marked"] is False
    assert "the bars are not subdivided" in render.render_html(without)


def test_every_chart_has_a_data_table_fallback(html: str) -> None:
    """The report must degrade gracefully if the SVG does not render."""
    assert html.count("also the fallback if the SVG does not render") == 2
    for value in ("1.57", "87.3%"):  # a yield ratio and a noise-curve precision
        assert value in html


def test_a_missing_chart_section_says_so_rather_than_breaking(aggregate: dict) -> None:
    data = deepcopy(aggregate)
    data["noise_curve"] = []
    data["metrics"] = {k: v for k, v in data["metrics"].items() if "yield" not in k}
    data["n"] = {k: v for k, v in data["n"].items() if "yield" not in k}
    html = render.render_html(data)
    assert _svgs(html) == []
    assert "nothing to chart" in html


# -- hallucination examples ---------------------------------------------------


def test_hallucination_examples_cap_at_five(aggregate: dict, html: str, summary: str) -> None:
    assert len(aggregate["hallucinations"]) > render.MAX_HALLUCINATION_EXAMPLES
    examples = render.build_context(aggregate)["hallucinations"]
    assert len(examples) == render.MAX_HALLUCINATION_EXAMPLES == 5
    shown = aggregate["hallucinations"][:5]
    dropped = aggregate["hallucinations"][5:]
    for item in shown:
        assert item["claimed_value"] in html
        assert item["claimed_value"] in summary
    for item in dropped:
        assert item["claimed_value"] not in html


def test_each_example_carries_field_value_and_the_closest_turn(aggregate: dict) -> None:
    for example in render.build_context(aggregate)["hallucinations"]:
        assert example["field"] and example["claimed_value"]
        assert example["turn_text"], "a hallucination is only checkable beside the transcript"


def test_a_claim_with_no_nearby_turn_says_so(aggregate: dict) -> None:
    data = deepcopy(aggregate)
    data["hallucinations"] = [{"field": "occupation", "claimed_value": "x", "turn_text": None}]
    assert "no nearby user turn recorded" in render.render_html(data)


def test_no_examples_is_not_presented_as_proof_of_none(aggregate: dict) -> None:
    data = deepcopy(aggregate)
    data["hallucinations"] = []
    html = render.render_html(data)
    assert "absence of recorded" in html


# -- exact-field floor and lift (M9) -----------------------------------------


def test_exact_field_recall_is_printed_beside_its_majority_class_floor(aggregate: dict) -> None:
    rows = render.build_context(aggregate)["exact_floor_rows"]
    assert [r["field"] for r in rows] == list(EXACT_FIELDS)
    scored = [r for r in rows if r["lift"] is not None]
    assert scored, "the fixture must carry at least one measured floor"
    for row in scored:
        assert row["lift"] == pytest.approx(row["recall"].value - row["floor"].value)
        assert row["lift_text"].endswith("pp")
        assert row["lift_text"][0] in "+-"


def test_a_field_with_no_floor_gets_no_invented_lift(aggregate: dict) -> None:
    """`age` is exact-scored but has no class set, so it has no majority class."""
    rows = render.build_context(aggregate)["exact_floor_rows"]
    age = next(r for r in rows if r["field"] == "age")
    assert age["floor"].value is None
    assert age["lift"] is None
    assert age["lift_text"] == "n/a"


# -- health statistics, not gates (M10) --------------------------------------


def test_disclosure_health_is_presented_beside_judge_health_not_as_a_gate(
    aggregate: dict, html: str
) -> None:
    keys = {m.key for m in render.build_context(aggregate)["health"]}
    assert {"disclosure_mismatch", "untagged_capture"} <= keys
    assert not any(k in metrics_schema.GATED_KEYS for k in keys)
    assert "Health statistics" in html and "not gates" in html
    assert "none of them fails a build" in html


def test_disclosure_mismatch_carries_its_n_and_untagged_capture_is_a_count(
    aggregate: dict,
) -> None:
    agg = render.Aggregate(aggregate)
    mismatch = agg.metric("disclosure_mismatch", "x")
    assert mismatch.kind == "rate" and mismatch.n is not None
    assert agg.metric("untagged_capture", "x").kind == "count"


# -- recall stages, abstention, manifest -------------------------------------


def test_both_recall_stages_are_labelled_unambiguously(aggregate: dict, html: str) -> None:
    labels = {m.key: m.label for m in render.build_context(aggregate)["headline"]}
    assert labels["recall"] == "Recall (pipeline)"
    assert labels["recall_elicited"] == "Recall (extraction only)"
    assert "includes facts the interview never elicited" in html
    assert "the metric the diff gate uses" in html


def test_abstention_rate_is_a_headline_row(aggregate: dict) -> None:
    keys = [m.key for m in render.build_context(aggregate)["headline"]]
    assert "abstention_rate" in keys


def test_the_schema_invalid_exclusion_is_printed(aggregate: dict, html: str, summary: str) -> None:
    text = render.build_context(aggregate)["manifest"]["excluded_text"]
    assert text == "1 personas excluded: schema_invalid"
    for rendered in (html, summary):
        assert "personas excluded: schema_invalid" in rendered


def test_no_exclusion_line_when_none_were_excluded(aggregate: dict) -> None:
    data = deepcopy(aggregate)
    data["manifest"]["personas"] = {"requested": 12, "ok": 12, "failed": 0, "schema_invalid": 0}
    assert render.build_context(data)["manifest"]["excluded_text"] is None
    assert "personas excluded: schema_invalid" not in render.render_html(data)


def test_the_manifest_carries_everything_section_seven_promises(html: str) -> None:
    for expected in (
        "fake-model",
        "effort=low",
        "cache_read_input_tokens",
        "912,300",
        "0000000-synthetic",
        "2026-09-30T12:18:42+00:00",
        "audit_seed",
        "terse",
    ):
        assert expected in html, expected


def test_a_failed_persona_is_counted_in_the_manifest(aggregate: dict) -> None:
    personas = render.build_context(aggregate)["manifest"]["personas"]
    assert personas["requested"] == 12
    assert personas["ok"] + personas["failed"] == personas["requested"]


# -- diff (§3.15) -------------------------------------------------------------


def test_the_diff_table_renders_with_all_six_columns(diff_data: dict) -> None:
    markdown = render.render_diff(diff_data)
    header = "| metric | baseline mean | candidate | delta | floor | verdict |"
    assert header in markdown
    for line in markdown.splitlines():
        if line.startswith("|") and "---" not in line:
            assert line.count("|") == 7, line  # six columns, seven pipes


def test_the_diff_marks_the_regression_and_the_gated_keys(diff_data: dict) -> None:
    markdown = render.render_diff(diff_data)
    assert "REGRESSION" in markdown
    assert "**precision**" in markdown  # gated keys are bold
    assert "hallucination_rate" in markdown and "**hallucination_rate**" not in markdown
    rows = {r["metric"]: r for r in render.diff_rows(diff_data)}
    assert rows["precision"]["verdict"] == "REGRESSION"
    assert rows["yield_per_minute.q07"]["verdict"] == "not gated"


def test_the_diff_derives_a_missing_verdict_rather_than_leaving_a_blank(diff_data: dict) -> None:
    """Track C owns the verdict; this is the fallback, not a second opinion."""
    row = next(r for r in diff_data["rows"] if r["metric"] == "recall_dealbreakers")
    assert "verdict" not in row
    rendered = next(r for r in render.diff_rows(diff_data) if r["metric"] == "recall_dealbreakers")
    # 0.7917 is above 0.8125 - 0.0213, so it is inside the measured floor.
    assert rendered["verdict"] == "pass"
    assert rendered["delta"] == "-2.1 pp"


def test_the_diff_does_not_invent_a_missing_baseline(diff_data: dict) -> None:
    row = next(
        r
        for r in render.diff_rows(diff_data)
        if r["metric"] == "precision_by_field.communication_style"
    )
    assert row["baseline_mean"] == "n/a"
    assert row["delta"] == "n/a"
    assert row["floor"] == "n/a"


def test_the_diff_states_that_the_floor_is_measured(diff_data: dict) -> None:
    markdown = render.render_diff(diff_data)
    assert "**measured** run-to-run variance" in markdown
    assert "not an" in markdown and "assumed zero" in markdown


def test_write_diff_names_the_file_after_both_tags(diff_data: dict, tmp_path: Path) -> None:
    path = render.write_diff(diff_data, tmp_path)
    assert path.name == "diff-v1-v2.md"
    assert path.read_text().startswith("# Diff: `v1` → `v2`")


def test_an_empty_diff_renders_a_table_rather_than_crashing(tmp_path: Path) -> None:
    markdown = render.render_diff({"baseline": "v1", "candidate": "v2", "rows": []})
    assert "_no metrics compared_" in markdown
    assert render.write_diff({"rows": []}, tmp_path).name == "diff-baseline-candidate.md"
