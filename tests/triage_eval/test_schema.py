import pytest

from harness.fault_table import CAUSE_CLASSES, UNCLASSIFIED_CLASS
from harness.schema import AGENT_ACTION_DECODING_SCHEMA, SchemaValidationError, validate_action


def test_valid_action_passes():
    action = {
        "diagnosis": "checkout service returning 500s under load",
        "cause_class": "unknown",
        "action_type": "remediate",
        "target": "checkout-deployment",
        "command": "kubectl rollout restart deployment/checkout",
        "confidence": 0.8,
    }
    assert validate_action(action) == action


def test_missing_required_field_rejected():
    action = {
        "diagnosis": "no target given",
        "cause_class": "unknown",
        "action_type": "inspect",
        "confidence": 0.5,
    }
    with pytest.raises(SchemaValidationError):
        validate_action(action)


def test_invalid_action_type_rejected():
    action = {
        "diagnosis": "x",
        "cause_class": "unknown",
        "action_type": "reboot_the_universe",
        "target": "x",
        "confidence": 0.5,
    }
    with pytest.raises(SchemaValidationError):
        validate_action(action)


def test_confidence_out_of_range_rejected():
    action = {
        "diagnosis": "x",
        "cause_class": "unknown",
        "action_type": "submit",
        "target": "x",
        "confidence": 1.5,
    }
    with pytest.raises(SchemaValidationError):
        validate_action(action)


def test_inspect_without_command_rejected():
    action = {
        "diagnosis": "x",
        "cause_class": "unknown",
        "action_type": "inspect",
        "target": "checkout-deployment",
        "confidence": 0.5,
    }
    with pytest.raises(SchemaValidationError):
        validate_action(action)


def test_remediate_with_null_command_rejected():
    action = {
        "diagnosis": "x",
        "cause_class": "unknown",
        "action_type": "remediate",
        "target": "checkout-deployment",
        "command": None,
        "confidence": 0.5,
    }
    with pytest.raises(SchemaValidationError):
        validate_action(action)


def test_remediate_with_empty_command_rejected():
    action = {
        "diagnosis": "x",
        "cause_class": "unknown",
        "action_type": "remediate",
        "target": "checkout-deployment",
        "command": "",
        "confidence": 0.5,
    }
    with pytest.raises(SchemaValidationError):
        validate_action(action)


def test_submit_without_command_still_accepted():
    action = {
        "diagnosis": "x",
        "cause_class": "unknown",
        "action_type": "submit",
        "target": "checkout-deployment",
        "confidence": 0.5,
    }
    assert validate_action(action) == action


def test_additional_properties_rejected():
    action = {
        "diagnosis": "x",
        "cause_class": "unknown",
        "action_type": "submit",
        "target": "x",
        "confidence": 0.5,
        "extra_field_not_in_schema": True,
    }
    with pytest.raises(SchemaValidationError):
        validate_action(action)


# --- decoding schema (config B/C): the if/then becomes an anyOf that Outlines compiles ---

from harness.schema import AGENT_ACTION_DECODING_SCHEMA, AGENT_ACTION_SCHEMA, build_decoding_schema  # noqa: E402

_DECODING_SAMPLES = [
    ({"diagnosis": "x", "cause_class": "unknown", "action_type": "inspect", "target": "t", "command": "kubectl get pods", "confidence": 0.5}, True),
    ({"diagnosis": "x", "cause_class": "unknown", "action_type": "remediate", "target": "t", "command": "kubectl apply -f a.yaml", "confidence": 0.5}, True),
    ({"diagnosis": "x", "cause_class": "unknown", "action_type": "inspect", "target": "t", "command": None, "confidence": 0.5}, False),
    ({"diagnosis": "x", "cause_class": "unknown", "action_type": "remediate", "target": "t", "command": "", "confidence": 0.5}, False),
    ({"diagnosis": "x", "cause_class": "unknown", "action_type": "submit", "target": "t", "command": None, "confidence": 0.9}, True),
    ({"diagnosis": "x", "cause_class": "unknown", "action_type": "escalate", "target": "t", "command": None, "confidence": 0.9}, True),
    ({"diagnosis": "x", "cause_class": "unknown", "action_type": "submit", "target": "t", "command": "ls", "confidence": 0.9}, False),
    ({"diagnosis": "x", "cause_class": "unknown", "action_type": "get_traces", "target": "t", "command": None, "confidence": 0.9}, False),
]


def _decoding_accepts(payload) -> bool:
    import jsonschema

    try:
        jsonschema.validate(instance=payload, schema=AGENT_ACTION_DECODING_SCHEMA)
    except jsonschema.ValidationError:
        return False
    return True


@pytest.mark.parametrize("payload,expected", _DECODING_SAMPLES)
def test_decoding_schema_requires_command_by_action_type(payload, expected):
    assert _decoding_accepts(payload) is expected


@pytest.mark.parametrize("payload,expected", _DECODING_SAMPLES)
def test_whatever_the_decoder_can_emit_the_validator_accepts(payload, expected):
    # Decoding is a subset of validation: config B/C never produces output the post-hoc
    # validator rejects (the measured gap was command=null on remediate, 6 times on config C).
    if expected:
        assert validate_action(payload) == payload


def test_decoding_schema_is_derived_not_hand_written():
    assert build_decoding_schema(AGENT_ACTION_SCHEMA) == AGENT_ACTION_DECODING_SCHEMA
    for variant in AGENT_ACTION_DECODING_SCHEMA["anyOf"]:
        assert "if" not in variant and "then" not in variant
        assert variant["additionalProperties"] is False
        assert "command" in variant["required"]


def test_outlines_grammar_enforces_command_rule():
    # Runs only where outlines_core is installed (the harness venv); it proves the rule made it
    # into the GRAMMAR, not only into the jsonschema.
    outlines_json = pytest.importorskip("outlines_core.json_schema")
    import json
    import re

    regex = re.compile(outlines_json.build_regex_from_schema(json.dumps(AGENT_ACTION_DECODING_SCHEMA)))
    for payload, expected in _DECODING_SAMPLES:
        assert bool(regex.fullmatch(json.dumps(payload))) is expected, payload


def _action_with_cause(cause_class):
    action = {
        "diagnosis": "x",
        "action_type": "submit",
        "target": "t",
        "command": None,
        "confidence": 0.9,
    }
    if cause_class is not None:
        action["cause_class"] = cause_class
    return action


@pytest.mark.parametrize("cause_class", list(CAUSE_CLASSES))
def test_every_taxonomy_class_is_accepted(cause_class):
    validate_action(_action_with_cause(cause_class))


@pytest.mark.parametrize("cause_class", [None, "", "crashloop", UNCLASSIFIED_CLASS])
def test_cause_class_missing_or_outside_the_taxonomy_rejected(cause_class):
    """UNCLASSIFIED_CLASS is the harness marker for "model gave no class", never a model answer:
    accepting it here would let the model opt out of classifying while looking classified."""
    with pytest.raises(SchemaValidationError):
        validate_action(_action_with_cause(cause_class))


def test_decoding_schema_constrains_cause_class_in_every_variant():
    """The enum has to be inside the grammar the decoder enforces, not only in the post-hoc
    validator, or constrained configs would burn retry turns on a free-text class."""
    for variant in AGENT_ACTION_DECODING_SCHEMA["anyOf"]:
        assert "cause_class" in variant["required"]
        assert variant["properties"]["cause_class"]["enum"] == list(CAUSE_CLASSES)
