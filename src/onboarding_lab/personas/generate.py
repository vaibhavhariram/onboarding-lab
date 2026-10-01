"""Turn skeletons into truth sheets.

The model writes only the judged-field facts and the bio; every exact-scored
value arrives fixed from the skeleton and is passed through untouched
(PLAN.md M2). ``fact_id``s are assigned here, not by the model: ids are the only
join key in the lab, so a model that repeated or skipped one would corrupt every
downstream join.

A persona that cannot be generated is a recorded failure, never a silent drop —
dropping personas changes every number in the report (PLAN.md A2).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from ..config import RoleConfig
from ..hashing import sha256_hex
from ..llm import call
from ..models import Fact, ProvenanceStamp, TruthSheet
from ..providers.base import Message, Provider
from ..role_schemas import PERSONA_GEN_SCHEMA
from .sample_skeleton import Skeleton

PROMPT_PATH = Path("prompts/persona_gen.txt")

PersonaStatus = Literal["ok", "schema_invalid", "provider_error", "refusal", "truncated"]

#: Provider error reasons mapped onto persona statuses. A refusal and a
#: truncation are distinct states, not generic failures: a refusal says the
#: content tripped a classifier, a truncation says max_tokens was too low. The
#: report distinguishes them because the fixes differ.
_STATUS_BY_REASON: dict[str, PersonaStatus] = {
    "schema_invalid": "schema_invalid",
    "refusal": "refusal",
    "max_tokens": "truncated",
    "model_context_window_exceeded": "truncated",
    "provider_error": "provider_error",
}


def _status_for(reason: str | None) -> PersonaStatus:
    return _STATUS_BY_REASON.get(reason or "", "provider_error")


@dataclass(frozen=True)
class PersonaResult:
    """One attempt. ``sheet`` is set only when ``status == "ok"``."""

    persona_id: str
    status: PersonaStatus
    sheet: TruthSheet | None = None
    detail: str | None = None

    def as_failure_record(self) -> dict:
        return {
            "stage": "gen",
            "persona_id": self.persona_id,
            "status": self.status,
            "detail": self.detail,
        }


def render_prompt(skeleton: Skeleton, template: str) -> str:
    import json

    return template.format(
        skeleton=json.dumps(skeleton.as_prompt_payload(), indent=2, sort_keys=True),
        n_specific=skeleton.n_specific,
        n_volunteer=skeleton.n_volunteer,
        n_followup=skeleton.n_followup,
        n_hedge=skeleton.n_hedge,
    )


def build_truth_sheet(
    skeleton: Skeleton,
    payload: dict,
    *,
    model: str,
    prompt_hash: str,
    params: dict,
    code_version: str,
    now: datetime | None = None,
) -> TruthSheet:
    """Assemble the sheet. Raises ``ValidationError`` if the model's half is unusable."""
    judged: list[Fact] = []
    next_index = skeleton.next_fact_index()
    for offset, raw in enumerate(payload.get("facts", [])):
        judged.append(
            Fact(
                fact_id=f"f{next_index + offset:02d}",
                field=raw["field"],
                value=raw["value"],
                specificity=raw["specificity"],
                disclosure=raw["disclosure"],
                importance=raw["importance"],
            )
        )
    return TruthSheet(
        persona_id=skeleton.persona_id,
        seed=skeleton.seed,
        style=skeleton.style,
        demographics=skeleton.demographics,
        facts=[*skeleton.exact_facts, *judged],
        bio=payload.get("bio", ""),
        provenance=ProvenanceStamp(
            model=model,
            prompt_hash=prompt_hash,
            params=params,
            code_version=code_version,
            created_at=now or datetime.now(UTC),
        ),
    )


async def generate_one(
    skeleton: Skeleton,
    *,
    provider: Provider,
    model: str,
    role_cfg: RoleConfig | None = None,
    template: str | None = None,
    code_version: str = "dev",
    nonce: str | None = None,
    cache_root=None,
    now: datetime | None = None,
) -> PersonaResult:
    cfg = role_cfg or RoleConfig()
    tmpl = template if template is not None else PROMPT_PATH.read_text()
    # Hash the template, not the rendered instance, so every persona in a run
    # shares one prompt identity -- that is what tag-collision detection compares
    # (PLAN.md A3).
    prompt_hash = sha256_hex(tmpl)
    params = {"effort": cfg.effort, "thinking": cfg.thinking, "max_tokens": cfg.max_tokens}
    if nonce is not None:
        params["nonce"] = nonce

    system = render_prompt(skeleton, tmpl)
    attempt_note = ""

    # One corrective attempt: llm.call already retries a JSON Schema violation,
    # but a payload can satisfy the schema and still fail the Pydantic contract
    # (a field outside the judged set, a bio of the wrong length). Spec 3.9 asks
    # for one regeneration with the error appended, then a logged failure.
    for attempt in range(2):
        result = await call(
            role="persona_gen",
            provider=provider,
            system=system,
            messages=[
                Message(
                    role="user",
                    content=("Write the truth sheet." + attempt_note),
                )
            ],
            model=model,
            role_cfg=cfg,
            json_schema=PERSONA_GEN_SCHEMA,
            nonce=nonce if attempt == 0 else f"{nonce or ''}retry{attempt}",
            cache_root=cache_root,
        )
        if not result.ok:
            return PersonaResult(
                persona_id=skeleton.persona_id,
                status=_status_for(result.reason),
                detail=result.error,
            )
        try:
            sheet = build_truth_sheet(
                skeleton,
                result.parsed or {},
                model=result.model or model,
                prompt_hash=prompt_hash,
                params=params,
                code_version=code_version,
                now=now,
            )
        except (ValidationError, KeyError, TypeError) as exc:
            if attempt == 0:
                attempt_note = f"\n\nThe previous attempt was rejected: {exc}. Correct it."
                continue
            return PersonaResult(
                persona_id=skeleton.persona_id,
                status="schema_invalid",
                detail=str(exc),
            )
        return PersonaResult(persona_id=skeleton.persona_id, status="ok", sheet=sheet)

    raise AssertionError("unreachable")


async def generate_all(
    skeletons: list[Skeleton],
    *,
    provider: Provider,
    model: str,
    role_cfg: RoleConfig | None = None,
    concurrency: int = 8,
    code_version: str = "dev",
    cache_root=None,
    now: datetime | None = None,
) -> list[PersonaResult]:
    """Generate every persona under a bounded-concurrency gate.

    Results come back in skeleton order regardless of completion order: joining
    by position after a gather would be exactly the misalignment bug this project
    is built to avoid.
    """
    template = PROMPT_PATH.read_text()
    gate = asyncio.Semaphore(concurrency)

    async def one(sk: Skeleton) -> PersonaResult:
        async with gate:
            return await generate_one(
                sk,
                provider=provider,
                model=model,
                role_cfg=role_cfg,
                template=template,
                code_version=code_version,
                cache_root=cache_root,
                now=now,
            )

    results = await asyncio.gather(*(one(sk) for sk in skeletons))
    by_id = {r.persona_id: r for r in results}
    return [by_id[sk.persona_id] for sk in skeletons]
