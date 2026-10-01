"""JSON Schemas for each role's structured output.

Passed as ``output_config.format`` so the provider validates server-side, with
``jsonschema`` re-validating locally — the local pass is what produces the
``schema_invalid`` state, and that state is the point (PLAN.md P6).

``fact_id`` is assigned in Python, not by the model: ids are the only join key in
the lab, so a model that repeats or skips one would corrupt every downstream
join.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .models import ARRAY_FIELDS, JUDGED_FIELDS

#: Persona generation returns only the judged-field facts and the bio. Every
#: exact-scored field is fixed by the Python skeleton sampler (PLAN.md M2).
PERSONA_GEN_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts", "bio"],
    "properties": {
        "facts": {
            "type": "array",
            "minItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "value", "specificity", "disclosure", "importance"],
                "properties": {
                    "field": {"enum": list(JUDGED_FIELDS)},
                    "value": {"type": "string"},
                    "specificity": {"enum": ["generic", "specific"]},
                    # `contradict` is deferred to v1.1 and is deliberately not
                    # offered here, so the simulator cannot produce decoys the
                    # scorer does not yet account for.
                    "disclosure": {"enum": ["volunteer", "needs_followup", "hedge"]},
                    "importance": {"enum": ["dealbreaker", "core", "color"]},
                },
            },
        },
        "bio": {"type": "string"},
    },
}

USER_AGENT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["utterance", "disclosed_fact_ids"],
    "properties": {
        "utterance": {"type": "string"},
        "disclosed_fact_ids": {"type": "array", "items": {"type": "string"}},
    },
}

INTERVIEWER_FOLLOWUP_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["targets_addressed", "followup"],
    "properties": {
        "targets_addressed": {"type": "array", "items": {"type": "string"}},
        "followup": {"type": ["string", "null"]},
    },
}

ALIGNER_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "claim_id",
            "matched_fact_id",
            "span_turn_id",
            "span_text",
            "specificity",
        ],
        "properties": {
            "claim_id": {"type": "string"},
            "matched_fact_id": {"type": ["string", "null"]},
            "span_turn_id": {"type": ["string", "null"]},
            # 12 words is a cap the judge is told about and the verifier
            # enforces; long quotes are where paraphrase creeps in.
            "span_text": {"type": ["string", "null"]},
            "specificity": {"enum": ["generic", "specific"]},
        },
    },
}

#: Enforced by ``verify_span``, and stated in ``aligner.txt``.
MAX_SPAN_WORDS = 12


@lru_cache(maxsize=8)
def load_profile_schema(path: str = "schema.json") -> dict:
    """Load the profile schema under test.

    Guards the one-word mistake with the widest blast radius: a ``required``
    array would turn every incomplete profile into ``schema_invalid`` and
    collapse the eval (PLAN.md M15).
    """
    schema = json.loads(Path(path).read_text())
    required = set(schema.get("required", []))
    if required - set(ARRAY_FIELDS) - {"summary"}:
        raise ValueError(
            f"{path}: 'required' must be empty (or only 'summary'); a missing field has to be "
            f"allowed, or every incomplete profile becomes schema_invalid. Got: {sorted(required)}"
        )
    return schema
