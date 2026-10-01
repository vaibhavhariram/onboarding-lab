"""Derive fact coverage from alignments and the transcript.

A fact is captured when some ``supported`` claim matched it. The eliciting turn
is the first user turn whose ``disclosed_fact_ids`` contains the fact, and the
question id and follow-up flag come from that turn — which is what makes
per-question yield and follow-up value computable rather than estimated.

``disclosed_turn_id`` being ``None`` is meaningful, not missing data: it means
the interview never elicited the fact, so extraction could not have captured it.
Separating that case from "extraction missed it" is the whole point of reporting
pipeline recall and elicited recall apart (PLAN.md M6).
"""

from __future__ import annotations

from ..models import Alignment, Fact, FactCoverage, Transcript


def first_disclosure(transcript: Transcript, fact_id: str):
    """The earliest user turn that stated this fact's true value."""
    for turn in transcript.turns:
        if turn.speaker == "user" and fact_id in turn.disclosed_fact_ids:
            return turn
    return None


def build_coverage(
    facts: list[Fact], alignments: list[Alignment], transcript: Transcript
) -> list[FactCoverage]:
    # Join by id. First supported claim wins, so a duplicated match cannot make
    # one fact look captured twice.
    captured_by: dict[str, str] = {}
    for a in alignments:
        if a.verdict == "supported" and a.matched_fact_id is not None:
            captured_by.setdefault(a.matched_fact_id, a.claim_id)

    coverage: list[FactCoverage] = []
    for fact in facts:
        turn = first_disclosure(transcript, fact.fact_id)
        claim_id = captured_by.get(fact.fact_id)
        coverage.append(
            FactCoverage(
                fact_id=fact.fact_id,
                captured=claim_id is not None,
                by_claim_id=claim_id,
                disclosed_turn_id=turn.turn_id if turn else None,
                question_id=turn.question_id if turn else None,
                via_followup=turn.is_followup if turn else None,
            )
        )
    return coverage
