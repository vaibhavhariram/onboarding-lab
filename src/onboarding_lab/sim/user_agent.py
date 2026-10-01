"""The simulated user.

Its system prompt carries the bio and the fact sheet and is **byte-stable for the
whole persona**: facts sorted by id, no timestamps, no per-turn interpolation.
That is a hard requirement, not tidiness — it is the one prefix in the pipeline
worth a cache breakpoint, and it is the bulk of all input tokens in a run
(PLAN.md P7). Any volatile byte in here silently costs several times the
simulation bill.

The conversation is replayed as alternating roles (interviewer questions as
``user``, the persona's own answers as ``assistant``) so the cached prefix grows
monotonically instead of being rewritten each turn.
"""

from __future__ import annotations

from pathlib import Path

from ..config import RoleConfig
from ..hashing import sha256_hex
from ..llm import call
from ..models import Fact, Style, TruthSheet, Turn
from ..providers.base import Message, Provider
from ..role_schemas import USER_AGENT_SCHEMA

PROMPT_PATH = Path("prompts/user_agent.txt")

#: Style word ranges. These exist to stress the interview policy: a terse
#: persona sits below the 40-word follow-up trigger and so provokes a follow-up
#: on nearly every question, while a rambling one never trips it on length.
#: That is why followup_share is also reported broken down by style -- the
#: headline number is largely an artifact of the sampled style mix
#: (PLAN.md M15).
STYLE_WORD_RANGES: dict[Style, tuple[int, int]] = {
    "terse": (15, 40),
    "balanced": (40, 120),
    "rambling": (120, 250),
    "tangential": (120, 250),
}

_STYLE_NOTES: dict[Style, str] = {
    "tangential": "- Include at least one digression that has nothing to do with the question.",
}


def render_fact_sheet(facts: list[Fact]) -> str:
    """One line per fact, sorted by id. Deterministic by construction."""
    lines = []
    for f in sorted(facts, key=lambda f: f.fact_id):
        lines.append(f"{f.fact_id} | {f.field} | {f.value} | {f.disclosure} | {f.specificity}")
    return "\n".join(lines)


def render_system(sheet: TruthSheet, template: str) -> str:
    low, high = STYLE_WORD_RANGES[sheet.style]
    return template.format(
        bio=sheet.bio.strip(),
        fact_sheet=render_fact_sheet(sheet.facts),
        style=sheet.style,
        min_words=low,
        max_words=high,
        style_note=_STYLE_NOTES.get(sheet.style, ""),
    )


def conversation_messages(turns: list[Turn]) -> list[Message]:
    """Replay the transcript from the persona's point of view."""
    return [
        Message(role="user" if t.speaker == "interviewer" else "assistant", content=t.text)
        for t in turns
    ]


class UserAgent:
    """Answers questions as one persona, reporting what it disclosed."""

    def __init__(
        self,
        sheet: TruthSheet,
        *,
        provider: Provider,
        model: str,
        role_cfg: RoleConfig | None = None,
        template: str | None = None,
        cache_root=None,
    ) -> None:
        self.sheet = sheet
        self.provider = provider
        self.model = model
        self.cfg = role_cfg or RoleConfig()
        tmpl = template if template is not None else PROMPT_PATH.read_text()
        self.prompt_hash = sha256_hex(tmpl)
        # Rendered once per persona and reused for every turn, which is what
        # makes the cached prefix a cache *hit* rather than a re-creation.
        self.system = render_system(sheet, tmpl)
        self.cache_root = cache_root
        self._valid_fact_ids = {f.fact_id for f in sheet.facts}
        #: Fact ids the model reported that do not exist on the sheet. Dropped
        #: rather than trusted, and surfaced so the simulator's own reliability
        #: is visible instead of silently corrupting recall.
        self.invented_fact_ids: list[str] = []

    async def answer(self, turns: list[Turn]) -> tuple[str | None, list[str], str | None]:
        """Return ``(utterance, disclosed_fact_ids, error)``."""
        result = await call(
            role="user_agent",
            provider=self.provider,
            system=self.system,
            messages=conversation_messages(turns),
            model=self.model,
            role_cfg=self.cfg,
            json_schema=USER_AGENT_SCHEMA,
            cache_root=self.cache_root,
        )
        if not result.ok:
            return None, [], result.error or result.reason
        payload = result.parsed or {}
        utterance = (payload.get("utterance") or "").strip()
        if not utterance:
            return None, [], "user agent returned an empty utterance"

        reported = payload.get("disclosed_fact_ids") or []
        disclosed: list[str] = []
        for fact_id in reported:
            if fact_id in self._valid_fact_ids:
                if fact_id not in disclosed:
                    disclosed.append(fact_id)
            else:
                self.invented_fact_ids.append(fact_id)
        return utterance, disclosed, None
