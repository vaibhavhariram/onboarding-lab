"""The judge, and the verification that keeps it honest.

Judged fields are free text, so a model decides whether an extracted claim means
the same thing as a truth fact. What stops that from being an unchecked oracle is
the span rule: the judge must quote the words that support the claim, and
:mod:`..score.verify_span` checks the quote is verbatim in the named user turn.

Models paraphrase, so a first prompt can return a high ``span_invalid`` rate. The
aligner therefore re-asks **once**, with only the failing claims and the exact
text of each cited turn; a second failure stays ``span_invalid``. Both the
first-pass and the final rates are reported, because the gap between them is a
property of the prompt worth seeing (PLAN.md amendment 5).

Claims are matched to judge output **by claim id**, never by list position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import RoleConfig
from ..llm import call
from ..models import JUDGED_FIELDS, Alignment, Claim, Transcript, TruthSheet
from ..providers.base import Message, Provider
from ..role_schemas import ALIGNER_SCHEMA, MAX_SPAN_WORDS
from ..transcript_text import render_transcript, render_user_turns
from .verify_span import check_span

PROMPT_PATH = Path("prompts/aligner.txt")


@dataclass
class AlignResult:
    alignments: list[Alignment] = field(default_factory=list)
    calls: int = 0
    judged_claims: int = 0
    errors: int = 0
    span_invalid: int = 0
    span_invalid_first_pass: int = 0
    #: The judge named a fact id that does not exist, or belongs to another
    #: field. Treated as "no match" rather than as a judge error, so the claim
    #: stays in the precision denominator, but counted so the judge's own
    #: reliability stays visible.
    invalid_fact_refs: int = 0


def _facts_payload(sheet: TruthSheet) -> list[dict]:
    return [
        {"fact_id": f.fact_id, "field": f.field, "value": f.value}
        for f in sheet.facts
        if f.field in JUDGED_FIELDS
    ]


def _claims_payload(claims: list[Claim]) -> list[dict]:
    return [
        {"claim_id": c.claim_id, "field": c.field, "value": c.value}
        for c in claims
        if c.field in JUDGED_FIELDS
    ]


def render_prompt(
    sheet: TruthSheet, transcript: Transcript, claims: list[Claim], template: str
) -> str:
    import json

    return template.format(
        facts=json.dumps(_facts_payload(sheet), indent=2, default=str),
        claims=json.dumps(_claims_payload(claims), indent=2, default=str),
        transcript=render_transcript(transcript),
    )


def _verdict_for(
    entry: dict,
    sheet: TruthSheet,
    transcript: Transcript,
    claim: Claim,
    result: AlignResult,
) -> Alignment:
    span_turn_id = entry.get("span_turn_id")
    span_text = entry.get("span_text")
    specificity = entry.get("specificity")

    matched = entry.get("matched_fact_id")
    if matched is not None:
        fact = next((f for f in sheet.facts if f.fact_id == matched), None)
        if fact is None or fact.field != claim.field:
            result.invalid_fact_refs += 1
            matched = None

    if span_turn_id is None or span_text is None:
        # Nothing cited: the claim is unsupported. This is the hallucination
        # case, and it is a verdict, not an error.
        return Alignment(
            claim_id=claim.claim_id,
            verdict="unsupported",
            method="judge",
            specificity=specificity,
        )

    check = check_span(transcript, span_turn_id, span_text, max_words=MAX_SPAN_WORDS)
    if not check.valid:
        return Alignment(
            claim_id=claim.claim_id,
            verdict="span_invalid",
            method="judge",
            span_turn_id=span_turn_id,
            span_text=span_text,
            specificity=specificity,
        )

    if matched is not None:
        return Alignment(
            claim_id=claim.claim_id,
            verdict="supported",
            method="judge",
            matched_fact_id=matched,
            span_turn_id=span_turn_id,
            span_text=span_text,
            specificity=specificity,
        )
    # Faithful to the transcript but matching no fact on the sheet: the
    # simulator leaked something. That is a sim-quality problem, not an
    # extraction hallucination, so it gets its own verdict.
    return Alignment(
        claim_id=claim.claim_id,
        verdict="supported_offsheet",
        method="judge",
        span_turn_id=span_turn_id,
        span_text=span_text,
        specificity=specificity,
    )


async def align_claims(
    sheet: TruthSheet,
    transcript: Transcript,
    claims: list[Claim],
    *,
    provider: Provider,
    model: str,
    role_cfg: RoleConfig | None = None,
    template: str | None = None,
    nonce: str | None = None,
    cache_root=None,
) -> AlignResult:
    cfg = role_cfg or RoleConfig()
    tmpl = template if template is not None else PROMPT_PATH.read_text()
    judged = [c for c in claims if c.field in JUDGED_FIELDS]
    result = AlignResult(judged_claims=len(judged))
    if not judged:
        return result

    system = render_prompt(sheet, transcript, judged, tmpl)
    completion = await call(
        role="aligner",
        provider=provider,
        system=system,
        messages=[Message(role="user", content="Judge every claim.")],
        model=model,
        role_cfg=cfg,
        json_schema=ALIGNER_SCHEMA,
        nonce=nonce,
        cache_root=cache_root,
    )
    result.calls += 1

    if not completion.ok:
        # A failed judge call is its own state for every claim it would have
        # covered. It is never a verdict about those claims.
        result.errors = len(judged)
        result.alignments = [
            Alignment(claim_id=c.claim_id, verdict="judge_error", method="judge") for c in judged
        ]
        return result

    entries = {e.get("claim_id"): e for e in (completion.parsed or []) if isinstance(e, dict)}
    by_claim: dict[str, Alignment] = {}
    for claim in judged:
        entry = entries.get(claim.claim_id)
        if entry is None:
            # The judge skipped this claim. Missing output is an error state,
            # not an implied "unsupported".
            result.errors += 1
            by_claim[claim.claim_id] = Alignment(
                claim_id=claim.claim_id, verdict="judge_error", method="judge"
            )
            continue
        by_claim[claim.claim_id] = _verdict_for(entry, sheet, transcript, claim, result)

    result.span_invalid_first_pass = sum(
        1 for a in by_claim.values() if a.verdict == "span_invalid"
    )

    retryable = [c for c in judged if by_claim[c.claim_id].verdict == "span_invalid"]
    if retryable:
        await _retry_spans(
            sheet,
            transcript,
            retryable,
            by_claim,
            tmpl,
            provider,
            model,
            cfg,
            nonce,
            cache_root,
            result,
        )

    result.span_invalid = sum(1 for a in by_claim.values() if a.verdict == "span_invalid")
    result.alignments = [by_claim[c.claim_id] for c in judged]
    return result


async def _retry_spans(
    sheet: TruthSheet,
    transcript: Transcript,
    retryable: list[Claim],
    by_claim: dict[str, Alignment],
    template: str,
    provider: Provider,
    model: str,
    cfg: RoleConfig,
    nonce: str | None,
    cache_root,
    result: AlignResult,
) -> None:
    """One more attempt, showing the judge the turns it misquoted."""
    cited_turns = [
        by_claim[c.claim_id].span_turn_id
        for c in retryable
        if by_claim[c.claim_id].span_turn_id is not None
    ]
    system = render_prompt(sheet, transcript, retryable, template)
    reminder = (
        "Your previous quotes were not verbatim. Here are the exact user turns "
        f"you cited:\n{render_user_turns(transcript, cited_turns)}\n\n"
        f"Re-judge only these claims. Copy at most {MAX_SPAN_WORDS} words "
        "character-for-character from a user turn, or return null for the span."
    )
    completion = await call(
        role="aligner",
        provider=provider,
        system=system,
        messages=[Message(role="user", content=reminder)],
        model=model,
        role_cfg=cfg,
        json_schema=ALIGNER_SCHEMA,
        nonce=f"{nonce or ''}span-retry",
        cache_root=cache_root,
    )
    result.calls += 1
    if not completion.ok:
        # The retry failing leaves the first-pass span_invalid verdicts in
        # place; it does not convert them into judge errors.
        return

    entries = {e.get("claim_id"): e for e in (completion.parsed or []) if isinstance(e, dict)}
    for claim in retryable:
        entry = entries.get(claim.claim_id)
        if entry is None:
            continue
        by_claim[claim.claim_id] = _verdict_for(entry, sheet, transcript, claim, result)
