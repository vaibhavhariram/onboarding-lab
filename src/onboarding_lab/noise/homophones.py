"""Near-phonetic word substitutions for the ASR noise injector.

A real speech recogniser's most damaging error is not a garbled word — a garbled
word is visibly garbled, and extraction skips it. The damaging error is a
*plausible* word: ``their`` for ``there``, ``weak`` for ``week``. Those survive
into the transcript as fluent text and extraction reads them as meant.

So the swap has two tiers:

- a small curated dictionary of true homophones and near-homophones, which is
  where the plausible-word failure actually lives;
- a character-level fallback for every word the dictionary does not cover, so
  the swap operation is never a silent no-op on ordinary vocabulary.

Nothing here is random on its own: every choice is drawn from a caller-supplied
``random.Random``, which ``inject`` derives per turn.
"""

from __future__ import annotations

import random
import re
from typing import Literal, NamedTuple

#: Curated homophones and near-homophones, keyed by lowercase word. Values never
#: contain the key, so a dictionary swap always changes the word. Kept small on
#: purpose: a long list of rare pairs would not fire on conversational text.
HOMOPHONES: dict[str, tuple[str, ...]] = {
    "accept": ("except",),
    "affect": ("effect",),
    "allowed": ("aloud",),
    "aloud": ("allowed",),
    "ant": ("aunt",),
    "ate": ("eight",),
    "aunt": ("ant",),
    "bare": ("bear",),
    "bear": ("bare",),
    "board": ("bored",),
    "bored": ("board",),
    "brake": ("break",),
    "break": ("brake",),
    "buy": ("by", "bye"),
    "by": ("buy", "bye"),
    "bye": ("by", "buy"),
    "cell": ("sell",),
    "cent": ("sent", "scent"),
    "coarse": ("course",),
    "complement": ("compliment",),
    "compliment": ("complement",),
    "course": ("coarse",),
    "dear": ("deer",),
    "deer": ("dear",),
    "die": ("dye",),
    "dye": ("die",),
    "effect": ("affect",),
    "eight": ("ate",),
    "except": ("accept",),
    "fair": ("fare",),
    "fare": ("fair",),
    "flour": ("flower",),
    "flower": ("flour",),
    "for": ("four", "fore"),
    "fore": ("for", "four"),
    "four": ("for", "fore"),
    "guessed": ("guest",),
    "guest": ("guessed",),
    "heard": ("herd",),
    "hear": ("here",),
    "herd": ("heard",),
    "here": ("hear",),
    "hole": ("whole",),
    "hour": ("our",),
    "it's": ("its",),
    "its": ("it's",),
    "knew": ("new",),
    "knight": ("night",),
    "know": ("no",),
    "knows": ("nose",),
    "lessen": ("lesson",),
    "lesson": ("lessen",),
    "mail": ("male",),
    "male": ("mail",),
    "meat": ("meet",),
    "medal": ("meddle", "metal"),
    "meet": ("meat",),
    "metal": ("medal",),
    "morning": ("mourning",),
    "mourning": ("morning",),
    "new": ("knew",),
    "night": ("knight",),
    "no": ("know",),
    "nose": ("knows",),
    "one": ("won",),
    "our": ("hour",),
    "pair": ("pear", "pare"),
    "passed": ("past",),
    "past": ("passed",),
    "patience": ("patients",),
    "patients": ("patience",),
    "peace": ("piece",),
    "pear": ("pair",),
    "piece": ("peace",),
    "plain": ("plane",),
    "plane": ("plain",),
    "poor": ("pour", "pore"),
    "pour": ("poor",),
    "principal": ("principle",),
    "principle": ("principal",),
    "rain": ("reign", "rein"),
    "reign": ("rain",),
    "right": ("write", "rite"),
    "rite": ("right",),
    "road": ("rode", "rowed"),
    "rode": ("road",),
    "role": ("roll",),
    "roll": ("role",),
    "sail": ("sale",),
    "sale": ("sail",),
    "scene": ("seen",),
    "scent": ("sent", "cent"),
    "sea": ("see",),
    "see": ("sea",),
    "seen": ("scene",),
    "sell": ("cell",),
    "sent": ("cent", "scent"),
    "sew": ("so", "sow"),
    "so": ("sew", "sow"),
    "some": ("sum",),
    "son": ("sun",),
    "steal": ("steel",),
    "steel": ("steal",),
    "sum": ("some",),
    "sun": ("son",),
    "tail": ("tale",),
    "tale": ("tail",),
    "than": ("then",),
    "their": ("there", "they're"),
    "then": ("than",),
    "there": ("their", "they're"),
    "they're": ("their", "there"),
    "threw": ("through",),
    "through": ("threw",),
    "to": ("too", "two"),
    "too": ("to", "two"),
    "two": ("to", "too"),
    "waist": ("waste",),
    "wait": ("weight",),
    "waste": ("waist",),
    "weak": ("week",),
    "wear": ("where", "ware"),
    "weather": ("whether",),
    "week": ("weak",),
    "weight": ("wait",),
    "we're": ("were",),
    "were": ("we're", "where"),
    "where": ("wear", "were"),
    "whether": ("weather",),
    "which": ("witch",),
    "whole": ("hole",),
    "witch": ("which",),
    "won": ("one",),
    "wood": ("would",),
    "would": ("wood",),
    "write": ("right",),
    "your": ("you're",),
    "you're": ("your",),
}

#: Substring rewrites standing in for a recogniser's acoustic confusions. Every
#: pair rewrites to something different, so applying one always changes the
#: word — that is what lets the fallback report ``character`` honestly.
CHARACTER_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("ck", "k"),
    ("ph", "f"),
    ("th", "f"),
    ("ing", "in"),
    ("er", "a"),
    ("c", "k"),
    ("k", "c"),
    ("s", "z"),
    ("z", "s"),
    ("f", "ph"),
    ("i", "e"),
    ("e", "i"),
    ("a", "u"),
    ("u", "a"),
    ("o", "a"),
    ("t", "d"),
    ("d", "t"),
    ("v", "b"),
    ("b", "v"),
    ("n", "m"),
    ("m", "n"),
    ("g", "j"),
    ("y", "i"),
)

#: Shortest word the last-resort clipping applies to. Below it, clipping would
#: leave a one- or two-character stub that reads as corruption rather than as a
#: misrecognition.
MIN_CLIP_LENGTH = 4

SwapSource = Literal["dictionary", "character", "unchanged"]

#: Leading and trailing non-word characters are peeled off so that quoting and
#: sentence punctuation survive a swap: the punctuation operation owns
#: punctuation, this one owns words.
_TOKEN = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


class Swap(NamedTuple):
    """The replacement token and which tier produced it (recorded in the edit log)."""

    token: str
    source: SwapSource


def split_token(token: str) -> tuple[str, str, str]:
    """``'there.' -> ('', 'there', '.')``. Word-internal apostrophes stay in the core."""
    match = _TOKEN.match(token)
    if match is None:  # pragma: no cover - the pattern matches every string
        return "", token, ""
    return match.group(1), match.group(2), match.group(3)


def _match_case(core: str, replacement: str) -> str:
    if core.isupper() and len(core) > 1:
        return replacement.upper()
    if core[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def character_variant(core: str, rng: random.Random) -> Swap:
    """Perturb one substring of ``core``, or clip its last character.

    Used for every word the curated dictionary misses, which on real
    conversational text is most of them.
    """
    lowered = core.lower()
    sites = [
        (index, pattern, replacement)
        for pattern, replacement in CHARACTER_SUBSTITUTIONS
        for index in range(len(lowered) - len(pattern) + 1)
        if lowered.startswith(pattern, index)
    ]
    if sites:
        index, pattern, replacement = rng.choice(sites)
        return Swap(core[:index] + replacement + core[index + len(pattern) :], "character")
    if len(core) >= MIN_CLIP_LENGTH:
        return Swap(core[:-1], "character")
    # Digits and very short symbol-only tokens have no plausible mishearing, so
    # the swap leaves them alone. The edit is still counted (PLAN.md D4).
    return Swap(core, "unchanged")


def swap(token: str, rng: random.Random) -> Swap:
    """Replace one token with a plausible mishearing of it.

    Case and surrounding punctuation are preserved, so a swap is visible only as
    a different word.
    """
    lead, core, trail = split_token(token)
    if not core:
        return Swap(token, "unchanged")
    candidates = HOMOPHONES.get(core.lower())
    if candidates:
        return Swap(lead + _match_case(core, rng.choice(candidates)) + trail, "dictionary")
    replaced = character_variant(core, rng)
    return Swap(lead + replaced.token + trail, replaced.source)
