"""Unit tests for adapters.anthropic_model_client with the HTTP transport mocked (no network, no
key, no spend), same shape as test_vllm_model_client.py."""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message
from unittest.mock import patch

import pytest

from adapters.anthropic_model_client import (
    PINNED_MODEL,
    VOLATILE_MARKERS,
    AnthropicModelClient,
    AnthropicModelConfig,
    build_output_schema,
    request_cost_usd,
    split_cacheable_prefix,
)
from adapters.vllm_model_client import ModelUnavailable
from harness.agent import Agent, AgentConfig, TurnResult
from harness.schema import SchemaValidationError, validate_action


def _config(**overrides) -> AnthropicModelConfig:
    defaults = dict(api_key="sk-test", max_output_tokens=2048, timeout_seconds=5, connect_retries=1)
    defaults.update(overrides)
    return AnthropicModelConfig(**defaults)


def _client(**overrides) -> AnthropicModelClient:
    client = AnthropicModelClient(_config(**overrides))
    client._sleep = lambda _seconds: None
    return client


class _FakeResponse:
    def __init__(self, body: dict, request_id: str | None = "req_1"):
        self._body = json.dumps(body).encode("utf-8")
        self.status = 200
        self.headers = {"request-id": request_id} if request_id else {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _message(text: str = '{"ok": true}', **usage) -> dict:
    base = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    base.update(usage)
    return {
        "model": PINNED_MODEL,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": base,
    }


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["retry-after"] = retry_after
    return urllib.error.HTTPError("u", code, "err", headers, io.BytesIO(b'{"error":"x"}'))


def _capture(responses):
    sent = []
    queue = list(responses)

    def fake_urlopen(request, timeout=None):
        sent.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return sent, fake_urlopen


def test_config_refuses_any_model_but_the_pinned_one():
    with pytest.raises(ValueError, match="pinned"):
        _config(model="claude-opus-5-5")


def test_config_refuses_an_empty_key():
    with pytest.raises(ValueError, match="api_key"):
        _config(api_key="")


def test_config_refuses_disabled_thinking_which_the_model_rejects():
    with pytest.raises(ValueError, match="thinking"):
        _config(thinking="disabled")


def test_generate_posts_pinned_model_with_key_version_and_no_sampling_params():
    sent, fake = _capture([_FakeResponse(_message("hello"))])
    with patch("urllib.request.urlopen", fake):
        out = _client().generate("diagnose", schema_constrained=False)

    request = sent[0]
    body = json.loads(request.data.decode("utf-8"))
    assert out == "hello"
    assert request.full_url == "https://api.anthropic.com/v1/messages"
    assert request.get_header("X-api-key") == "sk-test"
    assert request.get_header("Anthropic-version") == "2023-06-01"
    assert body["model"] == "claude-sonnet-5-5"
    assert body["max_tokens"] == 2048
    assert body["thinking"] == {"type": "between_tools"}
    assert "temperature" not in body and "output_config" not in body


def test_schema_constrained_sends_a_json_schema_output_format():
    sent, fake = _capture([_FakeResponse(_message())])
    with patch("urllib.request.urlopen", fake):
        _client().generate("x", schema_constrained=True)

    body = json.loads(sent[0].data.decode("utf-8"))
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": build_output_schema()}}


def test_output_schema_is_a_plain_object_derived_from_the_validation_schema():
    schema = build_output_schema()
    assert schema["type"] == "object" and "anyOf" not in schema and "if" not in schema
    assert schema["additionalProperties"] is False
    assert "command" in schema["required"]
    assert schema["properties"]["command"] == {"type": ["string", "null"]}
    wire = json.dumps(schema)
    for keyword in ("minLength", "maxLength", "minimum", "maximum"):
        assert keyword not in wire


def test_a_reply_the_wire_schema_allows_but_the_harness_schema_rejects_is_still_malformed():
    # The API schema drops the conditional; the harness validator is the backstop.
    reply = {
        "diagnosis": "d",
        "cause_class": "unknown",
        "action_type": "inspect",
        "target": "t",
        "command": None,
        "confidence": 0.5,
    }
    with pytest.raises(SchemaValidationError):
        validate_action(reply)


def test_prompt_is_split_into_a_cached_prefix_and_a_volatile_tail():
    prompt = "SYSTEM\nTask:\nbroken pod\nLast observation:\npod pending"
    sent, fake = _capture([_FakeResponse(_message())])
    with patch("urllib.request.urlopen", fake):
        _client().generate(prompt, schema_constrained=False)

    blocks = json.loads(sent[0].data.decode("utf-8"))["messages"][0]["content"]
    assert blocks[0] == {
        "type": "text",
        "text": "SYSTEM\nTask:\nbroken pod",
        "cache_control": {"type": "ephemeral"},
    }
    assert blocks[1] == {"type": "text", "text": "\nLast observation:\npod pending"}
    assert "".join(b["text"] for b in blocks) == prompt


def test_prompt_without_a_volatile_section_is_one_cached_block():
    assert split_cacheable_prefix("SYSTEM\nTask:\nx") == ("SYSTEM\nTask:\nx", "")


def test_volatile_markers_match_what_the_agent_really_builds():
    agent = Agent(model=None, config=AgentConfig(schema_constrained_decoding=True, memory_enabled=False))
    history = [TurnResult(output_tokens=1, schema_valid=True, action={"a": 1}, solved=False, observation="o")]
    prompt = agent._build_prompt("the task", "latest obs", history=history)

    stable, volatile = split_cacheable_prefix(prompt)

    assert stable.rstrip("\n").endswith("Task:\nthe task")
    assert volatile.startswith(VOLATILE_MARKERS[1])  # history comes before the last observation
    assert "Last observation:\nlatest obs" in volatile


def test_cost_uses_the_four_token_prices():
    cost = request_cost_usd(
        input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=1_000_000, cache_write_tokens=1_000_000
    )
    assert cost == pytest.approx(2.00 + 10.00 + 0.20 + 2.50)


def test_usage_is_logged_per_request_and_accumulated(tmp_path):
    log = tmp_path / "usage.jsonl"
    usage = dict(input_tokens=1000, output_tokens=200, cache_read_input_tokens=5000, cache_creation_input_tokens=0)
    sent, fake = _capture([_FakeResponse(_message(**usage), "req_a"), _FakeResponse(_message(**usage), "req_b")])
    client = _client(usage_log_path=log)
    with patch("urllib.request.urlopen", fake):
        client.generate("x", schema_constrained=False)
        client.generate("x", schema_constrained=False)

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    expected = (1000 * 2.0 + 200 * 10.0 + 5000 * 0.2) / 1_000_000
    assert [r["request_id"] for r in rows] == ["req_a", "req_b"]
    assert rows[0]["cost_usd"] == pytest.approx(expected)
    assert rows[0]["cache_read_input_tokens"] == 5000
    assert client.totals.requests == 2
    assert client.totals.cost_usd == pytest.approx(2 * expected)
    assert client.stats.calls == 2
    assert client.stats.prompt_tokens == 2 * 6000  # everything read, cached or not
    assert client.stats.completion_tokens == 400


def test_response_without_usage_is_charged_a_worst_case_estimate(tmp_path):
    body = _message()
    del body["usage"]
    sent, fake = _capture([_FakeResponse(body)])
    log = tmp_path / "usage.jsonl"
    client = _client(usage_log_path=log)
    with patch("urllib.request.urlopen", fake):
        client.generate("x", schema_constrained=False)

    assert client.stats.calls_without_usage == 1
    assert client.totals.requests == 0
    # At least the full max_tokens as output, so it can never be cheaper than the real request.
    assert client.totals.cost_usd >= 2048 * 10.0 / 1_000_000
    (record,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert record["usage_estimated"] is True
    assert record["cost_usd"] == client.totals.cost_usd


def test_usage_less_responses_still_count_against_the_budget_cap():
    body = _message()
    del body["usage"]
    sent, fake = _capture([_FakeResponse(body)])
    client = _client(max_cost_usd=0.01)  # one worst-case estimate (>= $0.02) spends it
    with patch("urllib.request.urlopen", fake):
        client.generate("x", schema_constrained=False)
        with pytest.raises(ModelUnavailable, match="budget"):
            client.generate("x", schema_constrained=False)

    assert len(sent) == 1


class _RawResponse(_FakeResponse):
    def __init__(self, raw: bytes):
        super().__init__({})
        self._body = raw


def test_truncated_json_body_is_retried_then_becomes_model_unavailable():
    sent, fake = _capture([_RawResponse(b'{"content": [{"ty'), _RawResponse(b"<html>")])
    client = _client()
    with patch("urllib.request.urlopen", fake):
        with pytest.raises(ModelUnavailable, match="2 attempt"):
            client.generate("x", schema_constrained=False)

    assert len(sent) == 2
    # Each broken body was probably billed, so both count against the cap.
    assert client.totals.cost_usd >= 2 * 2048 * 10.0 / 1_000_000


def test_read_timeout_after_send_is_charged_the_estimate():
    sent, fake = _capture([TimeoutError("read timed out"), _FakeResponse(_message())])
    client = _client()
    with patch("urllib.request.urlopen", fake):
        client.generate("x", schema_constrained=False)

    assert client.totals.cost_usd >= 2048 * 10.0 / 1_000_000


def test_connect_phase_timeout_wrapped_in_urlerror_is_not_charged():
    sent, fake = _capture([urllib.error.URLError(TimeoutError("connect")), _FakeResponse(_message())])
    client = _client()
    with patch("urllib.request.urlopen", fake):
        client.generate("x", schema_constrained=False)

    assert client.totals.cost_usd < 2048 * 10.0 / 1_000_000


def test_connection_error_without_a_body_is_not_charged():
    sent, fake = _capture([urllib.error.URLError("down"), _FakeResponse(_message())])
    client = _client()
    with patch("urllib.request.urlopen", fake):
        client.generate("x", schema_constrained=False)

    assert client.totals.cost_usd == pytest.approx(request_cost_usd(
        input_tokens=100, output_tokens=20, cache_read_tokens=0, cache_write_tokens=0
    ))


def test_incomplete_read_is_retried_like_a_dropped_connection():
    import http.client

    sent, fake = _capture([http.client.IncompleteRead(b"partial"), _FakeResponse(_message("ok"))])
    client = _client()
    with patch("urllib.request.urlopen", fake):
        assert client.generate("x", schema_constrained=False) == "ok"

    assert client.stats.retries == 1


def test_usage_log_write_failure_keeps_the_paid_reply(tmp_path):
    sent, fake = _capture([_FakeResponse(_message("kept"))])
    client = _client(usage_log_path=tmp_path / "missing-dir" / "u.jsonl")
    with patch("urllib.request.urlopen", fake):
        assert client.generate("x", schema_constrained=False) == "kept"

    assert client.totals.requests == 1


def test_thinking_blocks_are_not_part_of_the_returned_text():
    body = _message()
    body["content"] = [
        {"type": "thinking", "thinking": "", "signature": "s"},
        {"type": "text", "text": '{"a": 1}'},
    ]
    sent, fake = _capture([_FakeResponse(body)])
    with patch("urllib.request.urlopen", fake):
        assert _client().generate("x", schema_constrained=False) == '{"a": 1}'


def test_retryable_status_is_retried_then_succeeds():
    sent, fake = _capture([_http_error(529), _FakeResponse(_message("ok"))])
    client = _client()
    with patch("urllib.request.urlopen", fake):
        assert client.generate("x", schema_constrained=False) == "ok"

    assert len(sent) == 2
    assert client.stats.retries == 1


def test_retry_after_header_sets_the_backoff():
    waits = []
    sent, fake = _capture([_http_error(429, retry_after="7"), _FakeResponse(_message())])
    client = _client()
    client._sleep = waits.append
    with patch("urllib.request.urlopen", fake):
        client.generate("x", schema_constrained=False)

    assert waits == [7.0]


def test_client_error_is_not_retried_and_keeps_the_body():
    sent, fake = _capture([_http_error(400)])
    with patch("urllib.request.urlopen", fake):
        with pytest.raises(ModelUnavailable, match=r"HTTP 400.*\"error\""):
            _client().generate("x", schema_constrained=False)

    assert len(sent) == 1


def test_transport_failure_becomes_model_unavailable_after_retries():
    sent, fake = _capture([urllib.error.URLError("down"), urllib.error.URLError("down")])
    with patch("urllib.request.urlopen", fake):
        with pytest.raises(ModelUnavailable, match="2 attempt"):
            _client().generate("x", schema_constrained=False)


def test_budget_cap_stops_the_arm_before_sending_another_request():
    usage = dict(input_tokens=1_000_000, output_tokens=0)  # $2.00
    sent, fake = _capture([_FakeResponse(_message(**usage))])
    client = _client(max_cost_usd=1.0)
    with patch("urllib.request.urlopen", fake):
        client.generate("x", schema_constrained=False)
        with pytest.raises(ModelUnavailable, match="budget"):
            client.generate("x", schema_constrained=False)

    assert len(sent) == 1


def test_usage_block_with_only_malformed_counts_is_charged_the_estimate():
    body = _message()
    body["usage"] = {"input_tokens": "lots", "output_tokens": -1}
    sent, fake = _capture([_FakeResponse(body)])
    client = _client()
    with patch("urllib.request.urlopen", fake):
        client.generate("x", schema_constrained=False)

    assert client.stats.calls_without_usage == 1
    assert client.totals.cost_usd >= 2048 * 10.0 / 1_000_000
