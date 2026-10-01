"""The judge and its verification.

The span rule is what keeps the judge from being an unchecked oracle, so these
tests care most about what happens when the judge is wrong.
"""

from __future__ import annotations

import asyncio

import pytest

from onboarding_lab import llm as llm_module
from onboarding_lab.models import Claim
from onboarding_lab.providers.fake import FakeProvider
from onboarding_lab.score.align import align_claims
from onboarding_lab.transcript_text import render_transcript

CLAIMS = [
    Claim(claim_id="c01", field="occupation", value="hospital systems administrator"),
    Claim(claim_id="c02", field="interests", value="restoring a 1972 motorcycle"),
]


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(llm_module.asyncio, "sleep", _instant)


def entry(claim_id, fact_id, turn_id, span, specificity="specific") -> dict:
    return {
        "claim_id": claim_id,
        "matched_fact_id": fact_id,
        "span_turn_id": turn_id,
        "span_text": span,
        "specificity": specificity,
    }


def run(sheet, transcript, claims, canned, tmp_path, **kw):
    provider = FakeProvider({"role:aligner": canned})
    result = asyncio.run(
        align_claims(
            sheet,
            transcript,
            claims,
            provider=provider,
            model="fake-model",
            cache_root=tmp_path,
            **kw,
        )
    )
    return result, provider


def test_valid_span_and_fact_match_is_supported(truth_sheet, transcript, tmp_path) -> None:
    canned = [entry("c01", "f06", "t002", "hospital systems administrator")]
    result, _ = run(truth_sheet, transcript, CLAIMS[:1], canned, tmp_path)
    [a] = result.alignments
    assert a.verdict == "supported"
    assert a.method == "judge"
    assert a.matched_fact_id == "f06"
    assert result.span_invalid == 0


def test_valid_span_without_a_fact_match_is_offsheet(truth_sheet, transcript, tmp_path) -> None:
    """Faithful to the transcript but matching nothing on the sheet: the
    simulator leaked. A sim-quality problem, not an extraction hallucination."""
    canned = [entry("c01", None, "t002", "hospital systems administrator")]
    result, _ = run(truth_sheet, transcript, CLAIMS[:1], canned, tmp_path)
    [a] = result.alignments
    assert a.verdict == "supported_offsheet"
    assert a.matched_fact_id is None


def test_no_span_is_unsupported(truth_sheet, transcript, tmp_path) -> None:
    canned = [entry("c01", None, None, None)]
    result, _ = run(truth_sheet, transcript, CLAIMS[:1], canned, tmp_path)
    assert result.alignments[0].verdict == "unsupported"


def test_paraphrased_span_is_span_invalid_not_a_verdict(truth_sheet, transcript, tmp_path) -> None:
    canned = [entry("c01", "f06", "t002", "works in hospital IT")]
    result, _ = run(truth_sheet, transcript, CLAIMS[:1], canned, tmp_path)
    [a] = result.alignments
    assert a.verdict == "span_invalid"
    # The misquote is retained for the audit sample.
    assert a.span_text == "works in hospital IT"
    assert result.span_invalid_first_pass == 1


def test_interviewer_citation_is_span_invalid(truth_sheet, transcript, tmp_path) -> None:
    canned = [entry("c01", "f06", "t001", "What do you do for work")]
    result, _ = run(truth_sheet, transcript, CLAIMS[:1], canned, tmp_path)
    assert result.alignments[0].verdict == "span_invalid"


def test_failed_judge_call_is_judge_error_for_every_claim(
    truth_sheet, transcript, tmp_path
) -> None:
    canned = {"__error__": True, "reason": "provider_error", "error": "boom"}
    result, _ = run(truth_sheet, transcript, CLAIMS, canned, tmp_path)
    assert [a.verdict for a in result.alignments] == ["judge_error", "judge_error"]
    assert result.errors == 2


def test_skipped_claim_is_judge_error_not_unsupported(truth_sheet, transcript, tmp_path) -> None:
    """Missing output is an error state; it must not imply a hallucination."""
    canned = [entry("c01", "f06", "t002", "hospital systems administrator")]
    result, _ = run(truth_sheet, transcript, CLAIMS, canned, tmp_path)
    verdicts = {a.claim_id: a.verdict for a in result.alignments}
    assert verdicts["c01"] == "supported"
    assert verdicts["c02"] == "judge_error"
    assert result.errors == 1


def test_claims_are_joined_by_id_not_position(truth_sheet, transcript, tmp_path) -> None:
    """The judge's array order is not trusted."""
    turn = transcript.turns[1].model_copy(
        update={
            "text": "I am a hospital systems administrator restoring a 1972 motorcycle.",
            "word_count": 9,
            "disclosed_fact_ids": ["f06", "f07"],
        }
    )
    t = transcript.model_copy(
        update={"turns": [transcript.turns[0], turn], "simulated_minutes": round(14 / 150, 4)}
    )
    canned = [
        entry("c02", "f07", "t002", "restoring a 1972 motorcycle"),
        entry("c01", "f06", "t002", "hospital systems administrator"),
    ]
    result, _ = run(truth_sheet, t, CLAIMS, canned, tmp_path)
    by_id = {a.claim_id: a for a in result.alignments}
    assert by_id["c01"].matched_fact_id == "f06"
    assert by_id["c02"].matched_fact_id == "f07"


def test_fact_id_for_the_wrong_field_is_not_a_match(truth_sheet, transcript, tmp_path) -> None:
    """f07 is an interests fact; c01 is an occupation claim."""
    canned = [entry("c01", "f07", "t002", "hospital systems administrator")]
    result, _ = run(truth_sheet, transcript, CLAIMS[:1], canned, tmp_path)
    assert result.alignments[0].verdict == "supported_offsheet"
    assert result.invalid_fact_refs == 1


def test_nonexistent_fact_id_is_not_a_match(truth_sheet, transcript, tmp_path) -> None:
    canned = [entry("c01", "f99", "t002", "hospital systems administrator")]
    result, _ = run(truth_sheet, transcript, CLAIMS[:1], canned, tmp_path)
    assert result.alignments[0].verdict == "supported_offsheet"
    assert result.invalid_fact_refs == 1


def test_exact_fields_never_reach_the_judge(truth_sheet, transcript, tmp_path) -> None:
    exact_claims = [Claim(claim_id="c01", field="wants_kids", value="yes")]
    result, provider = run(truth_sheet, transcript, exact_claims, [], tmp_path)
    assert result.alignments == []
    assert result.judged_claims == 0
    assert provider.calls == [], "no judge call should have been made"


def test_span_retry_recovers_and_both_rates_are_reported(truth_sheet, transcript, tmp_path) -> None:
    """Models paraphrase, so one corrective pass is worth it; the gap between
    first-pass and final rates is a property of the prompt (amendment 5)."""
    bad = [entry("c01", "f06", "t002", "works in hospital IT")]
    good = [entry("c01", "f06", "t002", "hospital systems administrator")]
    provider = FakeProvider({"role:aligner": bad})
    calls = {"n": 0}
    original = provider.complete

    async def flaky(**kwargs):
        calls["n"] += 1
        provider.responses["role:aligner"] = good if calls["n"] > 1 else bad
        return await original(**kwargs)

    provider.complete = flaky
    result = asyncio.run(
        align_claims(
            truth_sheet,
            transcript,
            CLAIMS[:1],
            provider=provider,
            model="fake-model",
            cache_root=tmp_path,
        )
    )
    assert result.span_invalid_first_pass == 1
    assert result.span_invalid == 0
    assert result.alignments[0].verdict == "supported"
    assert result.calls == 2


def test_no_retry_when_every_span_is_valid(truth_sheet, transcript, tmp_path) -> None:
    canned = [entry("c01", "f06", "t002", "hospital systems administrator")]
    result, _ = run(truth_sheet, transcript, CLAIMS[:1], canned, tmp_path)
    assert result.calls == 1


def test_transcript_rendering_carries_turn_ids(transcript) -> None:
    rendered = render_transcript(transcript)
    assert "[t001 interviewer]" in rendered
    assert "[t002 user]" in rendered
