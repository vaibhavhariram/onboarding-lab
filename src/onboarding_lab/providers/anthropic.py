"""Anthropic adapter.

Parameter shapes worth not re-deriving from memory (PLAN.md amendment 1):

- No sampling parameters. ``temperature``/``top_p``/``top_k`` return HTTP 400.
- No assistant prefill. Returns HTTP 400.
- Structured output goes through ``output_config.format``, not a legacy
  ``output_format``.
- ``thinking`` is configuration, not a constant: the lowest setting differs by
  model, so ``make smoke`` establishes which this model accepts.
- Response content is read **by block type**, never ``content[0]``.
- ``stop_reason == "refusal"`` arrives as **HTTP 200**. Unguarded, its prose
  gets parsed as output or recorded as a schema failure.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .base import Completion, Effort, Message, Usage

#: Terminal stop reasons mapped to error states rather than parsed as output.
_STOP_REASON_ERRORS: dict[str, str] = {
    "refusal": "refusal",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "model_context_window_exceeded",
}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, api_key: str, model: str) -> None:
        # Imported here so the package imports without the SDK configured and so
        # tests never touch it.
        from anthropic import AsyncAnthropic

        # Explicit api_key: never the zero-arg constructor, which would pick up
        # ANTHROPIC_API_KEY from the environment (CLAUDE.md rule 1).
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model

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
    ) -> Completion:
        system_block: dict[str, Any] = {"type": "text", "text": system}
        if cache_system:
            system_block["cache_control"] = {"type": "ephemeral"}

        output_config: dict[str, Any] = {"effort": effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [system_block],
            "messages": [m.model_dump() for m in messages],
            "output_config": output_config,
        }
        if thinking:
            request["thinking"] = {"type": thinking}

        started = time.perf_counter()
        try:
            response = await self._client.messages.create(**request)
        except Exception as exc:  # every provider failure is a recorded state, never a raise
            return Completion(
                status="error",
                reason="provider_error",
                error=f"{type(exc).__name__}: {exc}",
                model=self.model,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        latency = int((time.perf_counter() - started) * 1000)

        usage = _usage(response)
        stop_reason = getattr(response, "stop_reason", None)
        stop_details = _as_dict(getattr(response, "stop_details", None))

        if stop_reason in _STOP_REASON_ERRORS:
            return Completion(
                status="error",
                reason=_STOP_REASON_ERRORS[stop_reason],
                error=f"stop_reason={stop_reason}",
                stop_reason=stop_reason,
                stop_details=stop_details,
                model=self.model,
                usage=usage,
                latency_ms=latency,
            )

        text = _text_from_blocks(response)
        parsed: dict | list | None = None
        if json_schema is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                return Completion(
                    status="error",
                    reason="schema_invalid",
                    error=f"response was not valid JSON: {exc}",
                    text=text,
                    stop_reason=stop_reason,
                    model=self.model,
                    usage=usage,
                    latency_ms=latency,
                )

        return Completion(
            status="ok",
            text=text,
            parsed=parsed,
            model=getattr(response, "model", self.model),
            usage=usage,
            latency_ms=latency,
            stop_reason=stop_reason,
            stop_details=stop_details,
        )


def _text_from_blocks(response: Any) -> str:
    """Concatenate ``text`` blocks only.

    Thinking blocks and any future block type are skipped; ``content[0]`` would
    pick up whatever happens to come first.
    """
    out: list[str] = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            out.append(getattr(block, "text", "") or "")
    return "".join(out)


def _usage(response: Any) -> Usage:
    raw = getattr(response, "usage", None)
    if raw is None:
        return Usage()
    return Usage(
        input_tokens=getattr(raw, "input_tokens", 0) or 0,
        output_tokens=getattr(raw, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
    )


def _as_dict(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    return {"value": str(value)}
