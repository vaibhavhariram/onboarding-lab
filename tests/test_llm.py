"""The call path: caching, bounded retries, and errors as states.

A failed call must come back as a `Completion` the pipeline records, never as an
exception that aborts a run and never as text that gets parsed as output.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from onboarding_lab import llm as llm_module
from onboarding_lab.llm import call
from onboarding_lab.providers.base import Completion, Effort, Message, Usage
from onboarding_lab.providers.fake import FakeProvider, role_marker

MESSAGES = [Message(role="user", content="hello")]
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["n"],
    "properties": {"n": {"type": "integer"}},
}


class ScriptedProvider:
    """Returns a queued sequence of completions and counts attempts."""

    name = "scripted"

    def __init__(self, queue: list[Completion]) -> None:
        self.queue = queue
        self.attempts = 0

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
        self.attempts += 1
        return self.queue[min(self.attempts - 1, len(self.queue) - 1)]


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Skip the retry backoff so the suite stays fast.

    The backoff itself is deliberately unseeded (retry timing must not be
    reproducible); what the tests care about is the attempt count.
    """

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(llm_module.asyncio, "sleep", _instant)


def err(reason: str) -> Completion:
    return Completion(status="error", reason=reason, error=f"simulated {reason}")


def ok_json(payload: dict) -> Completion:
    return Completion(status="ok", text=json.dumps(payload), parsed=payload, usage=Usage())


def test_role_marker_is_in_the_system_prompt() -> None:
    assert role_marker("extract") == "[lab-role:extract]"


def test_cache_hit_avoids_a_second_provider_call(tmp_path: Path) -> None:
    provider = FakeProvider({"role:extract": {"n": 1}})
    kwargs = dict(
        role="extract",
        provider=provider,
        system="sys",
        messages=MESSAGES,
        model="m",
        cache_root=tmp_path,
    )
    first = asyncio.run(call(**kwargs))
    assert first.ok and first.cached is False
    second = asyncio.run(call(**kwargs))
    assert second.ok and second.cached is True
    assert len(provider.calls) == 1, "second call should have been served from cache"
    assert second.parsed == first.parsed


def test_nonce_forces_a_fresh_call(tmp_path: Path) -> None:
    provider = FakeProvider({"role:extract": {"n": 1}})
    kwargs = dict(
        role="extract",
        provider=provider,
        system="sys",
        messages=MESSAGES,
        model="m",
        cache_root=tmp_path,
    )
    asyncio.run(call(**kwargs))
    asyncio.run(call(**kwargs, nonce="floor1"))
    assert len(provider.calls) == 2


def test_provider_error_is_retried_then_succeeds(tmp_path: Path) -> None:
    provider = ScriptedProvider([err("provider_error"), err("provider_error"), ok_json({"n": 1})])
    result = asyncio.run(
        call(
            role="extract",
            provider=provider,
            system="sys",
            messages=MESSAGES,
            model="m",
            cache_root=tmp_path,
        )
    )
    assert result.ok
    assert provider.attempts == 3


def test_retries_are_bounded(tmp_path: Path) -> None:
    provider = ScriptedProvider([err("provider_error")])
    result = asyncio.run(
        call(
            role="extract",
            provider=provider,
            system="sys",
            messages=MESSAGES,
            model="m",
            cache_root=tmp_path,
        )
    )
    assert result.status == "error"
    assert result.reason == "provider_error"
    assert provider.attempts == 3


@pytest.mark.parametrize("reason", ["refusal", "max_tokens", "model_context_window_exceeded"])
def test_terminal_errors_are_not_retried(tmp_path: Path, reason: str) -> None:
    """A refusal would refuse again; a truncation needs different parameters."""
    provider = ScriptedProvider([err(reason)])
    result = asyncio.run(
        call(
            role="extract",
            provider=provider,
            system="sys",
            messages=MESSAGES,
            model="m",
            cache_root=tmp_path,
        )
    )
    assert result.status == "error"
    assert result.reason == reason
    assert provider.attempts == 1, "terminal errors must not be retried"


def test_errors_are_never_cached(tmp_path: Path) -> None:
    provider = ScriptedProvider([err("refusal")])
    asyncio.run(
        call(
            role="extract",
            provider=provider,
            system="sys",
            messages=MESSAGES,
            model="m",
            cache_root=tmp_path,
        )
    )
    assert not list(tmp_path.rglob("*.json")), "a failed call must not poison the cache"


def test_schema_violation_retries_once_then_errors(tmp_path: Path) -> None:
    provider = ScriptedProvider([ok_json({"n": "not-an-integer"})])
    result = asyncio.run(
        call(
            role="extract",
            provider=provider,
            system="sys",
            messages=MESSAGES,
            model="m",
            json_schema=SCHEMA,
            cache_root=tmp_path,
        )
    )
    assert result.status == "error"
    assert result.reason == "schema_invalid"
    assert provider.attempts == 2, "one corrective attempt, then give up"


def test_schema_violation_recovers_on_the_corrective_attempt(tmp_path: Path) -> None:
    provider = ScriptedProvider([ok_json({"n": "bad"}), ok_json({"n": 7})])
    result = asyncio.run(
        call(
            role="extract",
            provider=provider,
            system="sys",
            messages=MESSAGES,
            model="m",
            json_schema=SCHEMA,
            cache_root=tmp_path,
        )
    )
    assert result.ok
    assert result.parsed == {"n": 7}


def test_missing_canned_response_is_an_error_not_a_crash(tmp_path: Path) -> None:
    result = asyncio.run(
        call(
            role="extract",
            provider=FakeProvider({}),
            system="sys",
            messages=MESSAGES,
            model="m",
            cache_root=tmp_path,
        )
    )
    assert result.status == "error"
    assert result.reason == "provider_error"
