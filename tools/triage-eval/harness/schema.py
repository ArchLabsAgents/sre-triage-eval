"""JSON schema for the agent's action output, shared by config B and config C.

This is what schema-constrained decoding constrains against. The schema itself is intentionally generic across
diagnose/act loops; a specific benchmark adapter narrows `action_type` and `target` further.
"""

from __future__ import annotations

import copy
from typing import Any

import jsonschema

from .fault_table import CAUSE_CLASSES

AGENT_ACTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["diagnosis", "cause_class", "action_type", "target", "confidence"],
    "properties": {
        "diagnosis": {
            "type": "string",
            "minLength": 1,
            "description": "Root cause hypothesis in plain language.",
        },
        # The structured cause the TriageRecord carries in causa_provavel.classe. Before this
        # field the model only gave free text and the assembly marked the record unclassified (or
        # unknown when it escalated), so the scorer never saw a class hit. The enum comes from
        # fault_table (derived, never copied); UNCLASSIFIED_CLASS is left out because it is a
        # harness marker, not a model answer. Required on every turn: the decoder constraint
        # needs it in the grammar, and the assembly reads the class from the final action.
        "cause_class": {
            "type": "string",
            "enum": list(CAUSE_CLASSES),
            "description": "Most likely root cause, from the closed taxonomy. Use unknown when the evidence does not support any class.",
        },
        "action_type": {
            "type": "string",
            "enum": ["inspect", "remediate", "escalate", "submit"],
        },
        "target": {
            "type": "string",
            "minLength": 1,
            "description": "Resource the action applies to (pod, service, deployment, ...).",
        },
        "command": {
            "type": ["string", "null"],
            "description": "Concrete command to run, required when action_type is inspect or remediate.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "additionalProperties": False,
    # inspect/remediate resolve to exec_shell in the adapter (_format_action), which needs a
    # concrete command; without this conditional the schema accepted command=null/missing, passed
    # validation and only blew up with ValueError inside the adapter, taking down the whole
    # episode instead of becoming retry feedback for the model like any other malformed output.
    "if": {"properties": {"action_type": {"enum": ["inspect", "remediate"]}}},
    "then": {
        "required": ["command"],
        "properties": {"command": {"type": "string", "minLength": 1}},
    },
}


class SchemaValidationError(ValueError):
    """Raised when a model output fails the agent action schema."""


def validate_action(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a decoded action against AGENT_ACTION_SCHEMA.

    Returns the payload unchanged on success. Raises SchemaValidationError on failure, with
    the underlying jsonschema message preserved so config_a runs (no constrained decoding) can
    still report *how* the output was malformed, not just that it was.
    """
    try:
        jsonschema.validate(instance=payload, schema=AGENT_ACTION_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise SchemaValidationError(str(exc)) from exc
    return payload


def build_decoding_schema(validation_schema: dict[str, Any]) -> dict[str, Any]:
    """Derive the schema handed to the constrained decoder from the validation schema.

    The validator enforces "command is a non-empty string when action_type is inspect or
    remediate" through if/then, but Outlines compiles only the base structure of a schema
    and ignores if/then (measured on the first config matrix: config_c produced command=null on
    remediate 6 times, each one rejected post-hoc by validate_action and burned as a retry
    turn). Expressing the same rule as anyOf over two closed variants, one per action_type
    group, puts it inside the grammar the decoder actually enforces. Derived from the
    validation schema, never hand-written, so the two cannot drift apart: whatever the
    decoder can emit, the validator accepts (tests/triage_eval/test_schema.py).
    """
    base = copy.deepcopy(validation_schema)
    for key in ("$schema", "if", "then"):
        base.pop(key, None)
    required = list(base["required"])
    if "command" not in required:
        required.append("command")

    def variant(action_types: list[str], command: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(base)
        out["properties"]["action_type"] = {"type": "string", "enum": action_types}
        out["properties"]["command"] = command
        out["required"] = required
        return out

    return {
        "anyOf": [
            variant(["inspect", "remediate"], {"type": "string", "minLength": 1}),
            variant(["escalate", "submit"], {"type": "null"}),
        ]
    }


AGENT_ACTION_DECODING_SCHEMA: dict[str, Any] = build_decoding_schema(AGENT_ACTION_SCHEMA)
