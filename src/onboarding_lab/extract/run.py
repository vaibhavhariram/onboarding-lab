"""The extraction runner: the system under test.

Everything product-specific here is a file — the prompt, the schema, the model —
so a team swaps in their own and the numbers become theirs. The prompt and schema
hashes travel with every artifact, which is what lets a tag collision be caught
rather than silently compared (PLAN.md A3).

A refusal, a truncation, a provider error, and a schema violation are four
distinct statuses. None of them produces claims, and none is ever scored as a
wrong answer.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from ..config import RoleConfig
from ..hashing import canonical_json, sha256_hex
from ..llm import call
from ..models import Extraction, ExtractionStatus, ProvenanceStamp, Transcript
from ..providers.base import Message, Provider
from ..role_schemas import load_profile_schema
from ..transcript_text import render_transcript
from .decompose import Decomposition, decompose

DEFAULT_PROMPT = "prompts/extract.txt"
DEFAULT_SCHEMA = "schema.json"

_STATUS_BY_REASON: dict[str, ExtractionStatus] = {
    "schema_invalid": "schema_invalid",
    "refusal": "refusal",
    "max_tokens": "truncated",
    "model_context_window_exceeded": "truncated",
    "provider_error": "provider_error",
}


def extraction_id(transcript_id: str, tag: str) -> str:
    return f"{transcript_id}:{tag}"


def render_prompt(transcript: Transcript, schema: dict, template: str) -> str:
    return template.format(
        transcript=render_transcript(transcript),
        schema=json.dumps(schema, indent=2),
    )


async def extract_one(
    transcript: Transcript,
    *,
    provider: Provider,
    model: str,
    tag: str = "v1",
    role_cfg: RoleConfig | None = None,
    template: str | None = None,
    schema: dict | None = None,
    prompt_path: str = DEFAULT_PROMPT,
    schema_path: str = DEFAULT_SCHEMA,
    code_version: str = "dev",
    nonce: str | None = None,
    cache_root=None,
    now: datetime | None = None,
) -> tuple[Extraction, Decomposition | None]:
    cfg = role_cfg or RoleConfig()
    tmpl = template if template is not None else Path(prompt_path).read_text()
    profile_schema = schema if schema is not None else load_profile_schema(schema_path)

    prompt_hash = sha256_hex(tmpl)
    schema_hash = sha256_hex(canonical_json(profile_schema))
    params = {"effort": cfg.effort, "thinking": cfg.thinking, "max_tokens": cfg.max_tokens}
    if nonce is not None:
        params["nonce"] = nonce

    completion = await call(
        role="extract",
        provider=provider,
        system=render_prompt(transcript, profile_schema, tmpl),
        messages=[Message(role="user", content="Extract the profile.")],
        model=model,
        role_cfg=cfg,
        json_schema=profile_schema,
        nonce=nonce,
        cache_root=cache_root,
    )

    stamp = ProvenanceStamp(
        model=completion.model or model,
        prompt_hash=prompt_hash,
        params=params,
        code_version=code_version,
        created_at=now or datetime.now(UTC),
    )
    base = {
        "extraction_id": extraction_id(transcript.transcript_id, tag),
        "transcript_id": transcript.transcript_id,
        "tag": tag,
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "model": completion.model or model,
        "provenance": stamp,
    }

    if not completion.ok:
        return (
            Extraction(
                **base,
                status=_STATUS_BY_REASON.get(completion.reason or "", "provider_error"),
                profile=None,
                claims=[],
                raw=completion.text or "",
            ),
            None,
        )

    profile = completion.parsed
    if not isinstance(profile, dict):
        return (
            Extraction(
                **base,
                status="schema_invalid",
                profile=None,
                claims=[],
                raw=completion.text or "",
            ),
            None,
        )

    decomposition = decompose(profile)
    return (
        Extraction(
            **base,
            status="ok",
            profile=profile,
            claims=decomposition.claims,
            raw=completion.text or "",
        ),
        decomposition,
    )


async def extract_all(
    transcripts: list[Transcript],
    *,
    provider: Provider,
    model: str,
    tag: str = "v1",
    concurrency: int = 8,
    **kwargs,
) -> list[tuple[Extraction, Decomposition | None]]:
    """Extraction is embarrassingly parallel, unlike simulation.

    Results are re-ordered by transcript id rather than returned in completion
    order.
    """
    template = Path(kwargs.pop("prompt_path", DEFAULT_PROMPT)).read_text()
    schema = load_profile_schema(kwargs.pop("schema_path", DEFAULT_SCHEMA))
    gate = asyncio.Semaphore(concurrency)

    async def one(t: Transcript):
        async with gate:
            return await extract_one(
                t,
                provider=provider,
                model=model,
                tag=tag,
                template=template,
                schema=schema,
                **kwargs,
            )

    results = await asyncio.gather(*(one(t) for t in transcripts))
    by_id = {e.transcript_id: (e, d) for e, d in results}
    return [by_id[t.transcript_id] for t in transcripts]
