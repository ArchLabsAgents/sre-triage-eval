"""Unit tests for adapters.vllm_model_client, with the HTTP transport mocked (no real vLLM
server, no network). Mirrors tests/triage_eval/test_mlx_model_client.py's shape: exercise the
adapter's own logic, not a real backend.
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

import pytest

from adapters.vllm_model_client import ModelUnavailable, VllmModelClient, VllmModelConfig
from harness.schema import AGENT_ACTION_DECODING_SCHEMA


def _config(**overrides) -> VllmModelConfig:
    defaults = dict(
        base_url="http://127.0.0.1:8000/v1",
        model="Qwen2.5-7B-Instruct",
        max_output_tokens=2048,
        temperature=0.2,
        timeout_seconds=5,
        connect_retries=1,
    )
    defaults.update(overrides)
    return VllmModelConfig(**defaults)


class _FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self._body = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_generate_posts_to_chat_completions_with_chat_message(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(_chat_response('{"ok": true}'))

    with patch("urllib.request.urlopen", fake_urlopen):
        client = VllmModelClient(_config())
        out = client.generate("diagnose the pod", schema_constrained=False)

    assert out == '{"ok": true}'
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["body"]["messages"] == [{"role": "user", "content": "diagnose the pod"}]
    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["max_tokens"] == 2048
    assert captured["timeout"] == 5


def test_unconstrained_generate_sends_no_guided_field(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(_chat_response("plain text"))

    with patch("urllib.request.urlopen", fake_urlopen):
        VllmModelClient(_config()).generate("x", schema_constrained=False)

    assert "guided_json" not in captured["body"]
    assert "guided_decoding_backend" not in captured["body"]


def test_constrained_generate_sends_decoding_schema_by_identity(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(_chat_response('{"command": null}'))

    with patch("urllib.request.urlopen", fake_urlopen):
        VllmModelClient(_config()).generate("x", schema_constrained=True)

    assert captured["body"]["guided_decoding_backend"] == "outlines"
    # By VALUE equality after the JSON round trip (identity does not survive json.dumps), but
    # the source object sent is AGENT_ACTION_DECODING_SCHEMA itself -- see the next test.
    assert captured["body"]["guided_json"] == AGENT_ACTION_DECODING_SCHEMA


def test_constrained_generate_sends_the_imported_object_not_a_copy():
    """An early run lost six turns to command=null because the validator and the decoder disagreed
    (see the module docstring). The regression test for that is identity, not equality: a
    hand-maintained copy of the schema would pass the equality test above and still reopen the
    hole the day one schema changes and the other doesn't."""
    captured_body = {}
    real_dumps = json.dumps

    def fake_dumps(obj, *args, **kwargs):
        if isinstance(obj, dict) and "guided_json" in obj:
            captured_body["guided_json"] = obj["guided_json"]
        return real_dumps(obj, *args, **kwargs)

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_chat_response("{}"))

    with patch("adapters.vllm_model_client.json.dumps", fake_dumps), patch(
        "urllib.request.urlopen", fake_urlopen
    ):
        VllmModelClient(_config()).generate("x", schema_constrained=True)

    assert captured_body["guided_json"] is AGENT_ACTION_DECODING_SCHEMA


def test_returns_raw_content_verbatim_no_repair(monkeypatch):
    raw = "not valid json at all ```"

    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_chat_response(raw))

    with patch("urllib.request.urlopen", fake_urlopen):
        out = VllmModelClient(_config()).generate("x", schema_constrained=True)

    assert out == raw


def test_api_key_sends_bearer_auth_header(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["headers"] = dict(request.header_items())
        return _FakeResponse(_chat_response("ok"))

    with patch("urllib.request.urlopen", fake_urlopen):
        VllmModelClient(_config(api_key="s3cr3t")).generate("x", schema_constrained=False)

    assert captured["headers"]["Authorization"] == "Bearer s3cr3t"


def test_no_api_key_sends_no_authorization_header(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["headers"] = dict(request.header_items())
        return _FakeResponse(_chat_response("ok"))

    with patch("urllib.request.urlopen", fake_urlopen):
        VllmModelClient(_config()).generate("x", schema_constrained=False)

    assert "Authorization" not in captured["headers"]


def _http_error(status: int, reason: str, body: bytes) -> urllib.error.HTTPError:
    import io

    return urllib.error.HTTPError(
        url="http://127.0.0.1:8000/v1/chat/completions",
        code=status,
        msg=reason,
        hdrs=None,
        fp=io.BytesIO(body),
    )


def test_http_error_body_surfaces_in_model_unavailable_message(monkeypatch):
    # 2026-09-17 hyperstack-a6000: two config_b scenarios died with only "HTTP Error 400: Bad
    # Request" in model_error, because HTTPError (a URLError subclass) fell into the generic
    # transport-failure branch and its response body -- vLLM's actual rejection reason -- was
    # never read. This is the regression test for that gap being closed.
    def fake_urlopen(request, timeout=None):
        raise _http_error(400, "Bad Request", b'{"error": "guided_json schema failed to compile"}')

    with patch("urllib.request.urlopen", fake_urlopen):
        client = VllmModelClient(_config(connect_retries=0))
        with pytest.raises(ModelUnavailable) as exc_info:
            client.generate("x", schema_constrained=True)

    message = str(exc_info.value)
    assert "HTTP 400 Bad Request" in message
    assert "guided_json schema failed to compile" in message


def test_http_error_with_unreadable_body_still_raises_model_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        exc = _http_error(400, "Bad Request", b"")
        exc.read = lambda: (_ for _ in ()).throw(OSError("body already consumed"))
        raise exc

    with patch("urllib.request.urlopen", fake_urlopen):
        client = VllmModelClient(_config(connect_retries=0))
        with pytest.raises(ModelUnavailable) as exc_info:
            client.generate("x", schema_constrained=False)

    assert "<no response body>" in str(exc_info.value)


def test_http_error_retries_like_other_transport_failures(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(400, "Bad Request", b"first attempt rejected")
        return _FakeResponse(_chat_response("recovered"))

    with patch("urllib.request.urlopen", fake_urlopen):
        client = VllmModelClient(_config(connect_retries=1))
        out = client.generate("x", schema_constrained=False)

    assert out == "recovered"
    assert calls["n"] == 2
    assert client.stats.retries == 1


def test_transport_failure_raises_model_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    with patch("urllib.request.urlopen", fake_urlopen):
        client = VllmModelClient(_config(connect_retries=0))
        with pytest.raises(ModelUnavailable):
            client.generate("x", schema_constrained=False)


def test_one_bounded_retry_on_connection_reset_is_counted(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError("reset")
        return _FakeResponse(_chat_response("recovered"))

    with patch("urllib.request.urlopen", fake_urlopen):
        client = VllmModelClient(_config(connect_retries=1))
        out = client.generate("x", schema_constrained=False)

    assert out == "recovered"
    assert calls["n"] == 2
    assert client.stats.retries == 1
    assert client.stats.calls == 1


def test_retry_exhausted_still_raises_model_unavailable(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise ConnectionResetError("reset")

    with patch("urllib.request.urlopen", fake_urlopen):
        client = VllmModelClient(_config(connect_retries=1))
        with pytest.raises(ModelUnavailable):
            client.generate("x", schema_constrained=False)

    assert client.stats.retries == 1


def test_generate_accumulates_wall_seconds_in_stats(monkeypatch):
    # model_seconds is the instrument that bounds the network share of ttm_seconds per
    # measurement; it has to accumulate even when the call fails (a dead tunnel costs time too).
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(_chat_response("ok"))

    with patch("urllib.request.urlopen", fake_urlopen):
        client = VllmModelClient(_config())
        client.generate("x", schema_constrained=False)
        client.generate("y", schema_constrained=False)

    assert client.stats.calls == 2
    assert client.stats.seconds > 0.0

    def dead_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    with patch("urllib.request.urlopen", dead_urlopen):
        failing = VllmModelClient(_config(connect_retries=0))
        with pytest.raises(ModelUnavailable):
            failing.generate("x", schema_constrained=False)

    assert failing.stats.seconds > 0.0
    assert failing.stats.calls == 0


def _with_usage(content: str, prompt, completion) -> dict:
    body = _chat_response(content)
    body["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": 0}
    return body


def test_usage_is_counted_exactly_and_a_missing_or_bad_usage_is_counted_apart():
    bodies = iter([
        _with_usage("a", 120, 30),
        _with_usage("b", 80, 5),
        _chat_response("no usage block"),
        _with_usage("c", "80", 5),  # not an int: never coerced
        _with_usage("d", True, 5),  # bool is an int subclass in Python: still refused
    ])

    with patch("urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(next(bodies))):
        client = VllmModelClient(_config())
        for _ in range(5):
            client.generate("x", schema_constrained=False)

    assert (client.stats.prompt_tokens, client.stats.completion_tokens) == (200, 35)
    assert client.stats.calls == 5
    assert client.stats.calls_without_usage == 3


def test_thread_stats_are_per_thread_and_reset_on_take():
    import threading

    with patch("urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(_with_usage("x", 10, 1))):
        client = VllmModelClient(_config())
        client.generate("main", schema_constrained=False)

        seen = {}

        def episode(n: int) -> None:
            client.take_thread_stats()
            for _ in range(n):
                client.generate("x", schema_constrained=False)
            seen[n] = client.take_thread_stats()

        threads = [threading.Thread(target=episode, args=(n,)) for n in (2, 3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert (seen[2].prompt_tokens, seen[2].calls) == (20, 2)
    assert (seen[3].prompt_tokens, seen[3].calls) == (30, 3)
    assert client.stats.prompt_tokens == 60  # the client-wide total still sees every call
    main_thread = client.take_thread_stats()
    assert (main_thread.prompt_tokens, main_thread.calls) == (10, 1)
    assert client.take_thread_stats().calls == 0


# A newer vLLM (Qwen3.5) drops guided_json from the chat
# request and decodes unconstrained without an error, so the request shape is a declared dialect.


def _constrained_body(**config_overrides) -> dict:
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(_chat_response("{}"))

    with patch("urllib.request.urlopen", fake_urlopen):
        VllmModelClient(_config(**config_overrides)).generate("x", schema_constrained=True)
    return captured["body"]


def test_default_dialect_keeps_the_guided_json_request_of_every_earlier_window():
    body = _constrained_body()
    assert set(body) == {"model", "messages", "temperature", "max_tokens", "guided_json", "guided_decoding_backend"}


def test_structured_outputs_dialect_sends_the_same_schema_and_no_legacy_field():
    body = _constrained_body(decoding_dialect="structured_outputs")
    assert body["structured_outputs"] == {"json": AGENT_ACTION_DECODING_SCHEMA}
    assert "guided_json" not in body and "guided_decoding_backend" not in body


def test_structured_outputs_dialect_sends_the_imported_object_not_a_copy():
    captured = {}
    real_dumps = json.dumps

    def fake_dumps(obj, *args, **kwargs):
        if isinstance(obj, dict) and "structured_outputs" in obj:
            captured["schema"] = obj["structured_outputs"]["json"]
        return real_dumps(obj, *args, **kwargs)

    with patch("adapters.vllm_model_client.json.dumps", fake_dumps), patch(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(_chat_response("{}"))
    ):
        VllmModelClient(_config(decoding_dialect="structured_outputs")).generate("x", schema_constrained=True)

    assert captured["schema"] is AGENT_ACTION_DECODING_SCHEMA


def test_unconstrained_structured_outputs_dialect_sends_no_schema():
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(_chat_response("x"))

    with patch("urllib.request.urlopen", fake_urlopen):
        VllmModelClient(_config(decoding_dialect="structured_outputs")).generate("x", schema_constrained=False)

    assert "structured_outputs" not in captured["body"]


def test_unknown_dialect_is_refused_at_config_time():
    with pytest.raises(ValueError, match="decoding_dialect"):
        _config(decoding_dialect="structured-outputs")


@pytest.mark.parametrize(("raw", "expected"), [("", "guided_json"), ("structured_outputs", "structured_outputs")])
def test_parse_decoding_dialect(raw, expected):
    from adapters.vllm_model_client import parse_decoding_dialect

    assert parse_decoding_dialect(raw) == expected


def test_parse_decoding_dialect_refuses_a_typo():
    from adapters.vllm_model_client import parse_decoding_dialect

    with pytest.raises(ValueError, match="decoding dialect"):
        parse_decoding_dialect("outlines")
