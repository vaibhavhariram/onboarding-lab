"""Deterministic provider backing every test.

``make test`` must pass on a clean checkout with no environment variables and no
network (CLAUDE.md rule 2), so this is the only provider the test suite sees.

The keying is a frozen contract (PLAN.md D3): dispatch is on
``(role, prompt_hash, messages_hash)``. Canned responses live in a JSON file so
fixtures can be committed and reviewed rather than buried in code.

Variance here is exactly zero, which matters for the diff test: the floor
collapses to its 0.005 minimum, so a fake "worse" candidate must be worse by a
wide margin for ``lab diff`` to exit 1 without flaking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..hashing import canonical_json, sha256_hex
from .base import Completion, Effort, Message, Usage


def response_key(role: str, prompt_hash: str, messages_hash: str) -> str:
    """The frozen dispatch key. Short hashes keep committed fixtures readable."""
    return f"{role}:{prompt_hash[:12]}:{messages_hash[:12]}"


class FakeProvider:
    """Canned, deterministic, and call-recording.

    ``responses`` maps :func:`response_key` to either a JSON-serialisable object
    (returned as ``parsed``) or a string (returned as ``text``). A
    ``role:<role>`` entry is the fallback for any unmatched request in that role,
    which keeps fixtures small while still being explicit.
    """

    name = "fake"

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        *,
        responses_path: Path | None = None,
        model: str = "fake-model",
    ) -> None:
        self.model = model
        self.responses: dict[str, Any] = dict(responses or {})
        if responses_path is not None:
            self.responses.update(json.loads(Path(responses_path).read_text()))
        #: Every call, in order, for assertions about caching and concurrency.
        self.calls: list[dict[str, Any]] = []

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def messages_hash(messages: list[Message]) -> str:
        return sha256_hex(canonical_json([m.model_dump() for m in messages]))

    def _lookup(self, role: str, prompt_hash: str, messages_hash: str) -> Any:
        key = response_key(role, prompt_hash, messages_hash)
        if key in self.responses:
            return self.responses[key]
        fallback = f"role:{role}"
        if fallback in self.responses:
            return self.responses[fallback]
        return None

    # -- provider contract ------------------------------------------------- #

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
        started = time.perf_counter()
        role = _role_from_system(system)
        prompt_hash = sha256_hex(system)
        mhash = self.messages_hash(messages)
        self.calls.append(
            {
                "role": role,
                "system": system,
                "messages": [m.model_dump() for m in messages],
                "json_schema": json_schema,
                "max_tokens": max_tokens,
                "thinking": thinking,
                "effort": effort,
                "nonce": nonce,
                "cache_system": cache_system,
            }
        )
        canned = self._lookup(role, prompt_hash, mhash)
        latency = int((time.perf_counter() - started) * 1000)

        if canned is None:
            missing = response_key(role, prompt_hash, mhash)
            return Completion(
                status="error",
                reason="provider_error",
                error=f"FakeProvider has no canned response for {missing}",
                model=self.model,
                latency_ms=latency,
            )

        if isinstance(canned, dict) and canned.get("__error__"):
            return Completion(
                status="error",
                reason=canned.get("reason", "provider_error"),
                error=canned.get("error", "canned error"),
                stop_reason=canned.get("stop_reason"),
                stop_details=canned.get("stop_details"),
                model=self.model,
                latency_ms=latency,
            )

        if isinstance(canned, str):
            return Completion(
                status="ok",
                text=canned,
                model=self.model,
                usage=Usage(input_tokens=len(system.split()), output_tokens=len(canned.split())),
                latency_ms=latency,
                stop_reason="end_turn",
            )

        text = json.dumps(canned)
        return Completion(
            status="ok",
            text=text,
            parsed=canned,
            model=self.model,
            usage=Usage(
                input_tokens=len(system.split()),
                output_tokens=len(text.split()),
                # Non-zero on repeat calls so cache assertions have something to
                # read; the real provider reports this for real.
                cache_read_input_tokens=len(system.split()) if cache_system else 0,
            ),
            latency_ms=latency,
            stop_reason="end_turn",
        )


#: Marker the system prompt carries so the fake can dispatch on role without the
#: caller threading it separately. ``llm.call`` always prepends it.
ROLE_MARKER = "[lab-role:"


def role_marker(role: str) -> str:
    return f"{ROLE_MARKER}{role}]"


def _role_from_system(system: str) -> str:
    start = system.find(ROLE_MARKER)
    if start == -1:
        return "unknown"
    end = system.find("]", start)
    if end == -1:
        return "unknown"
    return system[start + len(ROLE_MARKER) : end]
