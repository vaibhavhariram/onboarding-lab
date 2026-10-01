"""The single entry point for every model call.

Applies the content-hash cache, bounded retries, and schema validation, and
returns a :class:`Completion` with ``status="error"`` rather than raising. A
failed call is a state the pipeline records, not an exception that aborts a run.

Retry policy: only transport-level failures are retried. A refusal is not
retried (it would refuse again) and a truncation is not retried (it needs a
different ``max_tokens``, which would change the recorded parameters).
"""

from __future__ import annotations

import asyncio
import random

import jsonschema

from . import cache
from .config import RoleConfig
from .providers.base import Completion, Message, Provider
from .providers.fake import role_marker

#: Only these are worth another attempt.
RETRYABLE: frozenset[str] = frozenset({"provider_error"})


def build_system(role: str, system: str) -> str:
    """Prepend the role marker.

    It lets ``FakeProvider`` dispatch on role without the caller threading it
    separately, and it makes the role part of the cache key by construction.
    """
    return f"{role_marker(role)}\n{system}"


async def call(
    *,
    role: str,
    provider: Provider,
    system: str,
    messages: list[Message],
    model: str,
    role_cfg: RoleConfig | None = None,
    json_schema: dict | None = None,
    nonce: str | None = None,
    cache_root=None,
    max_attempts: int = 3,
) -> Completion:
    cfg = role_cfg or RoleConfig()
    full_system = build_system(role, system)

    key = cache.cache_key(
        role=role,
        system=full_system,
        messages=[m.model_dump() for m in messages],
        json_schema=json_schema,
        model=model,
        max_tokens=cfg.max_tokens,
        thinking=cfg.thinking,
        effort=cfg.effort,
        nonce=nonce,
    )
    hit = cache.read(key, cache_root)
    if hit is not None:
        return Completion.model_validate({**hit, "cached": True})

    attempt_messages = list(messages)
    schema_retried = False
    last: Completion | None = None

    for attempt in range(max_attempts):
        # `nonce` is deliberately not passed to the provider: it exists to vary
        # the cache key for floor runs, not to change the request. The floor
        # measures the model's own run-to-run variance on an identical request.
        result = await provider.complete(
            system=full_system,
            messages=attempt_messages,
            json_schema=json_schema,
            max_tokens=cfg.max_tokens,
            thinking=cfg.thinking,
            effort=cfg.effort,
            cache_system=cfg.cache_system,
        )
        last = result

        if result.ok and json_schema is not None:
            payload = result.parsed
            if payload is None:
                result = result.model_copy(
                    update={
                        "status": "error",
                        "reason": "schema_invalid",
                        "error": "provider returned no parsed JSON",
                    }
                )
                last = result
            else:
                try:
                    jsonschema.validate(payload, json_schema)
                except jsonschema.ValidationError as exc:
                    if not schema_retried:
                        # One corrective attempt with the validation error fed back.
                        schema_retried = True
                        attempt_messages = [
                            *attempt_messages,
                            Message(role="assistant", content=result.text or ""),
                            Message(
                                role="user",
                                content=(
                                    "That response failed schema validation: "
                                    f"{exc.message}. Return corrected JSON only."
                                ),
                            ),
                        ]
                        continue
                    result = result.model_copy(
                        update={
                            "status": "error",
                            "reason": "schema_invalid",
                            "error": f"schema validation failed: {exc.message}",
                        }
                    )
                    last = result

        if result.ok:
            cache.write(key, result.model_dump(mode="json"), cache_root)
            return result

        if result.reason not in RETRYABLE or attempt == max_attempts - 1:
            return result

        # Exponential backoff with jitter. Uses the ambient RNG deliberately:
        # retry timing must not be reproducible, and no seeded stream should be
        # perturbed by how many times the network happened to fail.
        await asyncio.sleep(0.5 * (2**attempt) + random.uniform(0, 0.25))

    assert last is not None
    return last
