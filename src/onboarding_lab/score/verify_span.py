"""Deterministic verification of the judge's cited spans.

The judge is asked to quote the words that support each claim; this module checks
that the quote is real. Nothing here consults a model, which is the point: the
judge's own claim about its evidence is not evidence.

A span fails verification when it does not occur in the named turn after
normalisation, when the turn is not a user turn, or when it exceeds the word cap.
Each failure is ``span_invalid`` — a distinct state, never a verdict about the
claim (PLAN.md: errors are states, not verdicts).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..models import Transcript
from ..role_schemas import MAX_SPAN_WORDS

#: Normalisation is deliberately narrow: whitespace, case, and punctuation only.
#: Anything looser would start accepting paraphrase, which is exactly what the
#: verbatim rule exists to catch.
_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace.

    Unicode is NFKC-folded first so a curly apostrophe or a non-breaking space
    in the model's quote does not read as a mismatch against straight ASCII in
    the transcript.
    """
    folded = unicodedata.normalize("NFKC", text)
    stripped = _PUNCTUATION.sub(" ", folded)
    return " ".join(stripped.split()).casefold()


@dataclass(frozen=True)
class SpanCheck:
    valid: bool
    reason: str | None = None


def check_span(
    transcript: Transcript,
    span_turn_id: str | None,
    span_text: str | None,
    *,
    max_words: int = MAX_SPAN_WORDS,
) -> SpanCheck:
    if span_turn_id is None or span_text is None:
        # No span cited at all is not an invalid span: it means the claim is
        # unsupported, which the caller decides.
        return SpanCheck(False, "no span cited")

    normalized_span = normalize(span_text)
    if not normalized_span:
        return SpanCheck(False, "span is empty after normalisation")
    if len(normalized_span.split()) > max_words:
        return SpanCheck(False, f"span exceeds {max_words} words")

    try:
        turn = transcript.turn(span_turn_id)
    except KeyError:
        return SpanCheck(False, f"no such turn: {span_turn_id}")

    if turn.speaker != "user":
        # Quoting the interviewer would let the question's own wording support a
        # claim the user never made.
        return SpanCheck(False, f"{span_turn_id} is an interviewer turn")

    if normalized_span not in normalize(turn.text):
        return SpanCheck(False, "span is not verbatim in the cited turn")

    return SpanCheck(True)
