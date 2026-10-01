"""The provider contract.

``complete`` is ``async`` because the pipeline runs under an asyncio semaphore at
``concurrency`` (PLAN.md P1); a sync method cannot be awaited under a gate.

There are no sampling parameters. ``temperature``/``top_p``/``top_k`` are removed
on current models and return HTTP 400, and the Messages API has no ``seed``
(PLAN.md P2). "Seed" in this codebase always means a Python-side
``random.Random(seed)`` and is never sent to a model.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

#: Why a call failed. A refusal arrives as HTTP 200 with ``stop_reason="refusal"``,
#: so it must be detected explicitly or its prose gets parsed as output
#: (PLAN.md amendment 3).
ErrorReason = Literal[
    "provider_error",
    "refusal",
    "max_tokens",
    "model_context_window_exceeded",
    "schema_invalid",
]

#: Effort levels valid alongside the lowest thinking setting.
Effort = Literal["low", "medium", "high"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class Usage(BaseModel):
    """Token accounting. Recorded in the run manifest; no hand-typed costs."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
        )


class Completion(BaseModel):
    """One model call's result.

    ``status="error"`` is returned rather than raised, so a failed call is a
    state the pipeline records instead of an exception that aborts a run.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    text: str | None = None
    #: Parsed and schema-validated when ``json_schema`` was given. Named
    #: ``parsed``, not ``json``: a field named ``json`` shadows
    #: ``BaseModel.json`` and warns on import (PLAN.md P9).
    parsed: dict | list | None = None
    model: str = ""
    usage: Usage = Usage()
    latency_ms: int = 0
    error: str | None = None
    reason: ErrorReason | None = None
    stop_reason: str | None = None
    stop_details: dict | None = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        json_schema: dict | None = None,
        max_tokens: int = 4000,
        thinking: str = "between_tools",
        effort: Effort = "low",
        nonce: str | None = None,
        cache_system: bool = False,
    ) -> Completion: ...
