"""Audit of the user agent's own disclosure self-report (PLAN.md M10).

``Turn.disclosed_fact_ids`` is the simulator telling us what it just said, and it
is the ground truth behind every recall and yield number in the report. The spec
applies verbatim-span skepticism to the judge and then takes the simulator's word
for this. These two checks close that gap with no extra API calls:

- ``disclosure_mismatch`` — the agent tagged a fact but its text shows no trace
  of it. Judged-field ``specific`` facts only: an enum or boolean value cannot be
  looked for in prose. Run on the **clean** transcript, because ASR noise drops
  and mangles exactly the words being looked for and would inflate the rate.
- ``untagged_capture`` — the reverse: extraction captured a fact, with a valid
  judge span, that no turn ever tagged, or whose cited span sits before the first
  tagged turn. As specified these drop out of ``yield`` but stay in the
  ``followup_share`` denominator; they are excluded from both.

Both are **health statistics printed beside judge health, not gates.** A
mismatch is a simulator-quality problem, not an extraction failure, and gating on
it would fail a candidate for its baseline's noise.

Matching is on **anchor tokens** — numbers, capitalised words, and rare content
words — rather than full-value overlap, because the agent is instructed to state
facts in its own voice and a paraphrase is not a mismatch. The normalisation is
deliberately crude (case, punctuation, plural ``s``/``es``): the check is for a
fact that left no trace at all, so recall of the matcher matters and precision
does not.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..models import JUDGED_FIELDS, Alignment, FactCoverage, Transcript, TruthSheet
from .stats import Tally, rate

#: Words too common to anchor on. Short enough to read, long enough that a value
#: made only of these is reported as unverifiable rather than as a mismatch.
_STOPWORD_TEXT = """
    a an and are as at be been being but by can could did do does for from had has have
    he her him his how i if in into is it its me more most my no not of on one or our out
    she so some that the their them then there these they this those to too up us very
    was we were what when where which while who why will with would you your am been
    about also just like really get got go going went make made take took thing things
    lot bit kind sort much many own new old time times way ways
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())

#: Runs of letters and digits. Apostrophes and hyphens split rather than join,
#: which costs nothing: both sides of the comparison tokenise identically, and
#: the fragments an apostrophe leaves behind are too short to anchor on anyway.
_WORD = re.compile(r"[0-9A-Za-z]+")
_MIN_ANCHOR_LEN = 4


def _normalise(token: str) -> str:
    """Lowercase and strip a plural suffix."""
    t = token.lower()
    if len(t) > 4 and t.endswith("es"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


def anchor_tokens(value: object) -> frozenset[str]:
    """Tokens whose presence is evidence the value was actually stated.

    A token qualifies when it contains a digit, is capitalised mid-phrase (a
    short proper noun that the length rule would otherwise drop), or is a
    content word of at least four characters. Returns an empty set for a value
    with nothing to anchor on, which the caller counts as unverifiable rather
    than as a mismatch.
    """
    if not isinstance(value, str):
        return frozenset()
    raw = _WORD.findall(value)
    anchors: set[str] = set()
    for index, token in enumerate(raw):
        norm = _normalise(token)
        if not norm or norm in STOPWORDS:
            continue
        numeric = any(ch.isdigit() for ch in token)
        proper = index > 0 and token[0].isupper()
        if numeric or proper or len(norm) >= _MIN_ANCHOR_LEN:
            anchors.add(norm)
    return frozenset(anchors)


def _token_set(text: str) -> frozenset[str]:
    return frozenset(_normalise(t) for t in _WORD.findall(text))


@dataclass(frozen=True)
class DisclosureReport:
    """Health statistics on the simulator's self-report, with their ``n``."""

    #: Judged-field ``specific`` facts that were tagged and carry an anchor.
    checked: tuple[str, ...] = ()
    #: Of those, the ones whose tagging turns contain none of their anchors.
    mismatched: tuple[str, ...] = ()
    #: Tagged but with no anchor to look for, so neither matched nor mismatched.
    unverifiable: tuple[str, ...] = ()
    #: Captured with a valid judge span but never tagged, or cited before the
    #: first tagged turn. Excluded from ``yield`` and from the
    #: ``followup_share`` denominator.
    untagged: tuple[str, ...] = ()
    #: Index of the first user turn carrying any tag, for the second clause.
    first_tagged_index: int | None = None
    #: ``fact_id -> why``, for the report's detail table.
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def mismatch(self) -> Tally:
        return rate(len(self.mismatched), len(self.checked))

    @property
    def untagged_fact_ids(self) -> frozenset[str]:
        return frozenset(self.untagged)

    @property
    def untagged_count(self) -> int:
        return len(self.untagged)


def check_disclosure_mismatch(
    *,
    truth: TruthSheet,
    clean_transcript: Transcript,
) -> DisclosureReport:
    """Anchor-token audit of every tagged, judged-field, ``specific`` fact.

    ``clean_transcript`` must be the noise-0 transcript; passing the noised copy
    measures the noise injector, not the simulator.
    """
    _require_clean(clean_transcript)
    tagged_text: dict[str, list[str]] = {}
    for turn in clean_transcript.turns:
        for fact_id in turn.disclosed_fact_ids:
            tagged_text.setdefault(fact_id, []).append(turn.text)

    judged = set(JUDGED_FIELDS)
    checked: list[str] = []
    mismatched: list[str] = []
    unverifiable: list[str] = []
    notes: dict[str, str] = {}
    for fact in truth.facts:
        if fact.field not in judged or fact.specificity != "specific":
            continue
        texts = tagged_text.get(fact.fact_id)
        if not texts:
            continue
        anchors = anchor_tokens(fact.value)
        if not anchors:
            unverifiable.append(fact.fact_id)
            notes[fact.fact_id] = "no anchor token in the fact value"
            continue
        checked.append(fact.fact_id)
        spoken = frozenset().union(*(_token_set(t) for t in texts))
        if not anchors & spoken:
            mismatched.append(fact.fact_id)
            notes[fact.fact_id] = (
                f"tagged as disclosed but the turn text contains none of {sorted(anchors)}"
            )
    return DisclosureReport(
        checked=tuple(checked),
        mismatched=tuple(mismatched),
        unverifiable=tuple(unverifiable),
        notes=notes,
    )


def check_untagged_capture(
    *,
    transcript: Transcript,
    coverage: Sequence[FactCoverage],
    alignments: Sequence[Alignment],
) -> DisclosureReport:
    """Captures the self-report never claimed.

    ``transcript`` is the transcript the spans were cited against; turn ids are
    preserved by the noise injector, so either copy gives the same ordering.
    """
    order = {turn.turn_id: i for i, turn in enumerate(transcript.turns)}
    tagged: set[str] = set()
    first_tagged: int | None = None
    for i, turn in enumerate(transcript.turns):
        if turn.disclosed_fact_ids:
            tagged.update(turn.disclosed_fact_ids)
            if first_tagged is None:
                first_tagged = i

    by_claim = {a.claim_id: a for a in alignments}
    untagged: list[str] = []
    notes: dict[str, str] = {}
    for cov in coverage:
        if not cov.captured or cov.by_claim_id is None:
            continue
        alignment = by_claim.get(cov.by_claim_id)
        if alignment is None or alignment.method != "judge":
            continue
        if alignment.verdict != "supported" or alignment.span_turn_id is None:
            continue
        if cov.fact_id not in tagged:
            untagged.append(cov.fact_id)
            notes[cov.fact_id] = "captured with a valid span but no turn tagged it"
            continue
        cited = order.get(alignment.span_turn_id)
        if cited is None:
            raise ValueError(
                f"alignment for claim {cov.by_claim_id} cites turn "
                f"{alignment.span_turn_id!r}, which is not in transcript "
                f"{transcript.transcript_id}"
            )
        if first_tagged is not None and cited < first_tagged:
            untagged.append(cov.fact_id)
            notes[cov.fact_id] = "cited span precedes the first tagged turn"
    return DisclosureReport(
        untagged=tuple(untagged),
        first_tagged_index=first_tagged,
        notes=notes,
    )


def check_disclosures(
    *,
    truth: TruthSheet,
    clean_transcript: Transcript,
    coverage: Sequence[FactCoverage] = (),
    alignments: Sequence[Alignment] = (),
    span_transcript: Transcript | None = None,
) -> DisclosureReport:
    """Both checks in one report, which is how the scorer consumes them."""
    mismatch = check_disclosure_mismatch(truth=truth, clean_transcript=clean_transcript)
    untagged = check_untagged_capture(
        transcript=span_transcript or clean_transcript,
        coverage=coverage,
        alignments=alignments,
    )
    return DisclosureReport(
        checked=mismatch.checked,
        mismatched=mismatch.mismatched,
        unverifiable=mismatch.unverifiable,
        untagged=untagged.untagged,
        first_tagged_index=untagged.first_tagged_index,
        notes={**mismatch.notes, **untagged.notes},
    )


def _require_clean(transcript: Transcript) -> None:
    if transcript.noise_rate or transcript.source_transcript_id is not None:
        raise ValueError(
            f"transcript {transcript.transcript_id} is a noised copy; the disclosure "
            f"check runs on the clean source or it measures the noise injector (M10)"
        )
