"""Two personas through the whole pipeline on FakeProvider.

No network, no API key. This is the integration gate: every artifact must exist,
validate against the models, and join by id.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from onboarding_lab import llm as llm_module
from onboarding_lab import metrics_schema
from onboarding_lab.config import Config
from onboarding_lab.models import (
    EXACT_FIELDS,
    SCORED_FIELDS,
    Extraction,
    Scores,
    Transcript,
    TruthSheet,
)
from onboarding_lab.pipeline import run_pipeline
from onboarding_lab.providers.fake import FakeProvider

UTTERANCE = (
    "I am a nurse and I have been doing it for nine years now. "
    "Most weeks run together, honestly, but I like the people I work with."
)

PERSONA_FACTS = {
    "facts": [
        {
            "field": "occupation",
            "value": "nurse",
            "specificity": "specific",
            "disclosure": "volunteer",
            "importance": "core",
        },
        *[
            {
                "field": f,
                "value": f"{f} value",
                "specificity": "generic",
                "disclosure": "volunteer",
                "importance": "color",
            }
            for f in (
                "location",
                "values",
                "interests",
                "life_events",
                "dealbreakers",
                "partner_preferences",
                "communication_style",
            )
        ],
    ],
    "bio": "I am a nurse in a mid-sized city and I like my work. " * 6,
}

# Claim ids follow SCORED_FIELDS order in decompose(), so with this profile
# wants_kids is c01 and occupation is c02.
PROFILE = {"wants_kids": "yes", "occupation": "nurse"}
ALIGNER = [
    {
        "claim_id": "c02",
        "matched_fact_id": "f06",
        "span_turn_id": "t002",
        "span_text": "I am a nurse",
        "specificity": "specific",
    }
]


@pytest.fixture(autouse=True)
def _fast_and_uncached(monkeypatch):
    monkeypatch.setenv("LAB_CACHE", "0")

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(llm_module.asyncio, "sleep", _instant)


def provider() -> FakeProvider:
    return FakeProvider(
        {
            "role:persona_gen": PERSONA_FACTS,
            "role:user_agent": {"utterance": UTTERANCE, "disclosed_fact_ids": ["f06"]},
            "role:interviewer_followup": {"targets_addressed": [], "followup": None},
            "role:extract": PROFILE,
            "role:aligner": ALIGNER,
        }
    )


@pytest.fixture
def outcome(tmp_path):
    cfg = Config.load("lab.yaml").model_copy(
        update={"personas": 2, "noise_rates": [0.0, 0.1], "concurrency": 2}
    )
    return asyncio.run(
        run_pipeline(
            cfg,
            provider=provider(),
            model="fake-model",
            runs_root=tmp_path,
            code_version="test",
        )
    )


def test_every_artifact_exists_and_validates(outcome) -> None:
    paths = outcome.paths
    assert paths.manifest.is_file()

    personas = sorted(paths.personas.glob("*.json"))
    assert len(personas) == 2
    for p in personas:
        TruthSheet.model_validate_json(p.read_text())

    transcripts = sorted(paths.transcripts.glob("*.json"))
    clean = [p for p in transcripts if p.name.endswith("n000.json")]
    assert len(clean) == 2
    for p in clean:
        Transcript.model_validate_json(p.read_text())


def test_noised_transcripts_point_back_at_their_clean_source(outcome) -> None:
    """Time-based metrics must read word counts from the clean source."""
    noised = [
        Transcript.model_validate_json(p.read_text())
        for p in outcome.paths.transcripts.glob("*n010.json")
    ]
    assert noised
    for t in noised:
        assert t.noise_rate == pytest.approx(0.1)
        assert t.noise_seed is not None
        assert t.source_transcript_id is not None
        assert t.source_transcript_id.endswith("n000")


def test_extractions_and_scores_validate(outcome) -> None:
    for tag in outcome.aggregates:
        extractions = sorted(outcome.paths.extractions(tag).glob("*.json"))
        scores = sorted(outcome.paths.scores(tag).glob("*.json"))
        assert len(extractions) == 2
        assert len([s for s in scores if s.name != "aggregate.json"]) == 2
        for p in extractions:
            assert Extraction.model_validate_json(p.read_text()).status == "ok"
        for p in scores:
            if p.name != "aggregate.json":
                Scores.model_validate_json(p.read_text())


def test_artifacts_join_by_id_not_position(outcome) -> None:
    sheets = {s.persona_id: s for s in outcome.sheets}
    for tag, scores in outcome.scores.items():
        for s in scores:
            transcript_id, _, score_tag = s.extraction_id.rpartition(":")
            assert score_tag == tag
            persona_id = transcript_id.split("__")[0]
            assert persona_id in sheets
            fact_ids = {f.fact_id for f in sheets[persona_id].facts}
            for c in s.coverage:
                assert c.fact_id in fact_ids


def test_aggregate_carries_every_expected_metric_key(outcome) -> None:
    script_ids = tuple(f"q{i:02d}" for i in range(1, 11))
    expected = metrics_schema.expected_keys(
        fields=SCORED_FIELDS,
        exact_fields=EXACT_FIELDS,
        disclosures=("volunteer", "needs_followup", "hedge"),
        styles=("terse", "balanced", "rambling", "tangential"),
        question_ids=script_ids,
    )
    for tag in outcome.aggregates:
        agg = json.loads(outcome.paths.aggregate(tag).read_text())
        present = set(agg["metrics"])
        missing = {k for k in expected if k in agg["kinds"]} - present
        assert not missing, f"{tag}: aggregate missing {sorted(missing)[:5]}"


def test_undefined_rates_are_null_never_zero(outcome) -> None:
    """0.0 would read as 'we got everything wrong'."""
    for tag in outcome.aggregates:
        agg = json.loads(outcome.paths.aggregate(tag).read_text())
        for key, value in agg["metrics"].items():
            if value is None:
                assert agg["n"].get(key, 0) == 0, f"{key} is null but has a denominator"


def test_manifest_accounts_for_every_persona(outcome) -> None:
    manifest = json.loads(outcome.paths.manifest.read_text())
    assert manifest["n_personas_requested"] == 2
    assert manifest["n_ok"] + manifest["n_failed"] == 2
    assert manifest["code_version"] == "test"
    assert sum(manifest["style_mix"].values()) == manifest["n_ok"]


def test_judge_health_is_pooled(outcome) -> None:
    for tag in outcome.aggregates:
        agg = json.loads(outcome.paths.aggregate(tag).read_text())
        judge = agg["judge"]
        assert judge["judged_claims"] >= judge["errors"] + judge["span_invalid"]
        assert isinstance(judge["degraded"], bool)


def test_report_renders_from_the_aggregate(outcome, tmp_path) -> None:
    from onboarding_lab.report.render import load_aggregate, write_report

    tag = sorted(outcome.aggregates)[0]
    written = write_report(load_aggregate(outcome.paths.aggregate(tag)), tmp_path / "report")
    html = written["index.html"].read_text()
    assert "<html" in html.lower()
    assert "<script" not in html.lower(), "the report must contain no JavaScript"
    assert written["summary.md"].read_text().strip()


# -- the remaining merge gates ------------------------------------------------


def run_with(provider_obj, tmp_path, **overrides):
    cfg = Config.load("lab.yaml").model_copy(
        update={"personas": 2, "noise_rates": [0.0], "concurrency": 2, **overrides}
    )
    return asyncio.run(
        run_pipeline(
            cfg,
            provider=provider_obj,
            model="fake-model",
            runs_root=tmp_path,
            code_version="test",
        )
    )


def test_a_dropped_persona_is_recorded_not_silently_skipped(tmp_path) -> None:
    """A silent drop changes every number in the report."""
    broken = FakeProvider(
        {"role:persona_gen": {"__error__": True, "reason": "refusal", "error": "declined"}}
    )
    outcome = run_with(broken, tmp_path)
    manifest = json.loads(outcome.paths.manifest.read_text())
    assert manifest["n_ok"] == 0
    assert manifest["n_failed"] == 2
    assert outcome.paths.failures.is_file()
    records = [json.loads(line) for line in outcome.paths.failures.read_text().splitlines()]
    assert {r["status"] for r in records} == {"refusal"}
    assert {r["persona_id"] for r in records} == {"p001", "p002"}


def test_judge_errors_are_excluded_from_denominators(tmp_path) -> None:
    """A failed judge call is a state, never a verdict about the claim."""
    failing_judge = FakeProvider(
        {
            "role:persona_gen": PERSONA_FACTS,
            "role:user_agent": {"utterance": UTTERANCE, "disclosed_fact_ids": ["f06"]},
            "role:interviewer_followup": {"targets_addressed": [], "followup": None},
            "role:extract": PROFILE,
            "role:aligner": {"__error__": True, "reason": "provider_error", "error": "boom"},
        }
    )
    outcome = run_with(failing_judge, tmp_path)
    tag = sorted(outcome.aggregates)[0]
    agg = json.loads(outcome.paths.aggregate(tag).read_text())

    assert agg["judge"]["errors"] > 0
    # Every judged claim errored, so only the exact-method claim can remain in
    # the precision denominator.
    assert agg["n"]["precision"] == agg["judge"]["judged_claims"] - agg["judge"]["errors"] + 2
    assert agg["judge"]["degraded"] is True


def test_diff_exits_one_when_the_candidate_regresses() -> None:
    """Fake variance is zero, so the floor sits at its 0.005 minimum; the
    candidate is made far worse so the gate cannot flake."""
    from onboarding_lab.score.floor import diff as compute_diff

    baseline = {k: 0.90 for k in metrics_schema.GATED_KEYS}
    candidate = {k: 0.70 for k in metrics_schema.GATED_KEYS}
    floors = dict.fromkeys(metrics_schema.GATED_KEYS, 0.005)

    regressed = compute_diff(baseline=baseline, candidate=candidate, floors=floors)
    assert regressed.exit_code == 1

    held = compute_diff(baseline=baseline, candidate=dict(baseline), floors=floors)
    assert held.exit_code == 0

    invalid = compute_diff(baseline={}, candidate={}, floors={})
    assert invalid.exit_code == 2
