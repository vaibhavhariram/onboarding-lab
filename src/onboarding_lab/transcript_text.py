"""Render a transcript for a prompt.

Turn ids are part of the rendering so span citation has a stable anchor: the
judge must name the turn it quoted, and verification looks that turn up by id.
"""

from __future__ import annotations

from .models import Transcript, Turn


def render_turn(turn: Turn) -> str:
    return f"[{turn.turn_id} {turn.speaker}] {turn.text}"


def render_transcript(transcript: Transcript) -> str:
    return "\n".join(render_turn(t) for t in transcript.turns)


def render_user_turns(transcript: Transcript, turn_ids: list[str]) -> str:
    """Only the named turns, for the aligner's second pass."""
    wanted = set(turn_ids)
    return "\n".join(
        render_turn(t) for t in transcript.turns if t.turn_id in wanted and t.speaker == "user"
    )
