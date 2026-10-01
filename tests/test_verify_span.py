"""Span verification. The judge's claim about its evidence is not evidence."""

from __future__ import annotations

from onboarding_lab.score.verify_span import check_span, normalize


def test_normalization_folds_case_punctuation_and_whitespace() -> None:
    assert normalize("  Hospital, Systems!  Administrator.  ") == "hospital systems administrator"


def test_normalization_folds_unicode_lookalikes() -> None:
    """A curly apostrophe or non-breaking space in the quote must not read as a
    mismatch against straight ASCII in the transcript."""
    curly_apostrophe = chr(0x2019)
    no_break_space = chr(0x00A0)
    assert normalize(f"I{curly_apostrophe}m a nurse") == normalize("I'm a nurse")
    assert normalize(f"a{no_break_space}nurse") == normalize("a nurse")


def test_exact_quote_is_valid(transcript) -> None:
    assert check_span(transcript, "t002", "hospital systems administrator").valid


def test_quote_differing_only_in_case_and_punctuation_is_valid(transcript) -> None:
    assert check_span(transcript, "t002", "Hospital Systems Administrator!").valid


def test_paraphrase_is_span_invalid(transcript) -> None:
    """The verbatim rule exists precisely to catch this."""
    check = check_span(transcript, "t002", "works in hospital IT")
    assert not check.valid
    assert "verbatim" in check.reason


def test_interviewer_turns_cannot_be_cited(transcript) -> None:
    """Otherwise the question's own wording could support a claim."""
    check = check_span(transcript, "t001", "What do you do for work")
    assert not check.valid
    assert "interviewer" in check.reason


def test_unknown_turn_is_span_invalid(transcript) -> None:
    assert not check_span(transcript, "t999", "anything").valid


def test_overlong_quote_is_span_invalid(transcript) -> None:
    long_quote = " ".join(["word"] * 13)
    check = check_span(transcript, "t002", long_quote)
    assert not check.valid
    assert "words" in check.reason


def test_word_cap_boundary(transcript) -> None:
    text = " ".join(f"w{i}" for i in range(12))
    turn = transcript.turns[1].model_copy(update={"text": text, "word_count": 12})
    t = transcript.model_copy(update={"turns": [transcript.turns[0], turn]})
    assert check_span(t, "t002", text).valid
    assert not check_span(t, "t002", text + " extra").valid


def test_no_span_cited_is_not_an_invalid_span(transcript) -> None:
    """No quote means the claim is unsupported; the caller decides that."""
    check = check_span(transcript, None, None)
    assert not check.valid
    assert check.reason == "no span cited"


def test_empty_quote_is_span_invalid(transcript) -> None:
    assert not check_span(transcript, "t002", "   ...   ").valid
