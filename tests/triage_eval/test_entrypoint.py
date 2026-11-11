"""Tests for entrypoint.py's WebhookReceiver: the testable core behind the HTTP handler. Most of this file exercises WebhookReceiver directly, with
no socket opened; do_POST/do_GET wiring itself is thin enough (stdlib BaseHTTPRequestHandler
plumbing) that exercising it would mostly test the standard library, not this code.

The exception is TestHttpLayer at the bottom: two behaviors live IN that thin wiring, not in
WebhookReceiver (an invalid-encoding body decoded by json.loads, and the connection timeout),
and both were bugs found in review -- worth a real socket to
actually prove them, not worth generalizing the rest of this suite over.
"""

from __future__ import annotations

import http.client
import sys
import threading
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import entrypoint  # noqa: E402
from entrypoint import (  # noqa: E402
    WebhookReceiver,
    _build_episode_runner,
    _default_dispatch,
    _make_handler,
    _REQUEST_TIMEOUT_SECONDS,
    log_only_episode_runner,
)
from http.server import ThreadingHTTPServer  # noqa: E402

# Every test in this file except the "dispatch" ones below injects this: it makes
# `episode_runner` run inline, on the calling thread, so `seen`/counters assertions right after
# `handle_alert_payload(...)` stay deterministic. Production default is `_default_dispatch` (a
# background thread) -- see that function's docstring in entrypoint.py for why, and the
# "async dispatch" tests near the bottom of this file for coverage of the real default.
def _sync_dispatch(task):
    task()


def _payload(**overrides) -> dict:
    base = {
        "version": "4",
        "groupKey": '{}:{alertname="KubePodCrashLooping"}',
        "status": "firing",
        "receiver": "sslm-agent",
        "alerts": [{"status": "firing", "fingerprint": "fp-1"}],
    }
    base.update(overrides)
    return base


def test_valid_new_episode_returns_202_and_calls_the_runner():
    seen = []
    receiver = WebhookReceiver(
        episode_runner=lambda episode_id, raw: seen.append((episode_id, raw)), dispatch=_sync_dispatch
    )

    status, body = receiver.handle_alert_payload(_payload())

    assert status == 202
    assert body["new_episode"] is True
    assert body["episode_id"]
    assert len(seen) == 1
    assert seen[0][0] == body["episode_id"]
    assert receiver.counters.episodes_started_total == 1
    assert receiver.counters.received_total == 1


def test_folded_webhook_does_not_call_the_runner_again():
    seen = []
    receiver = WebhookReceiver(
        episode_runner=lambda episode_id, raw: seen.append(episode_id), dispatch=_sync_dispatch
    )

    receiver.handle_alert_payload(_payload())
    status, body = receiver.handle_alert_payload(_payload(alerts=[{"status": "firing", "fingerprint": "fp-2"}]))

    assert status == 202
    assert body["new_episode"] is False
    assert len(seen) == 1  # runner only fired for the first (new) episode
    assert receiver.counters.episodes_started_total == 1
    assert receiver.counters.episodes_folded_total == 1
    assert receiver.counters.received_total == 2


def test_malformed_payload_returns_400_and_never_calls_the_runner():
    seen = []
    receiver = WebhookReceiver(episode_runner=lambda episode_id, raw: seen.append(episode_id))

    status, body = receiver.handle_alert_payload({"not": "a webhook"})

    assert status == 400
    assert "error" in body
    assert seen == []
    assert receiver.counters.parse_errors_total == 1
    assert receiver.counters.received_total == 1
    assert receiver.counters.episodes_started_total == 0


def test_resolved_with_no_open_window_returns_200_and_no_episode():
    receiver = WebhookReceiver()
    status, body = receiver.handle_alert_payload(
        _payload(status="resolved", alerts=[{"status": "resolved", "fingerprint": "fp-1"}])
    )
    assert status == 200
    assert body["episode_id"] is None
    assert receiver.counters.episodes_started_total == 0


def test_default_episode_runner_is_log_only_and_does_not_raise():
    receiver = WebhookReceiver()  # uses log_only_episode_runner
    assert receiver.episode_runner is log_only_episode_runner
    status, _body = receiver.handle_alert_payload(_payload())
    assert status == 202


def test_counters_as_dict_shape():
    receiver = WebhookReceiver()
    receiver.handle_alert_payload(_payload())
    receiver.handle_alert_payload({"bad": "payload"})
    counters = receiver.counters.as_dict()
    assert counters == {
        "received_total": 2,
        "parse_errors_total": 1,
        "episodes_started_total": 1,
        "episodes_folded_total": 0,
    }


# ---------------------------------------------------------------------------
# _build_episode_runner: this function decides, in
# production, whether a deployment executes real cluster mutations at all; it had no regression
# test of its own.
# ---------------------------------------------------------------------------


def test_build_episode_runner_falls_back_to_log_only_when_env_missing(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("APP_NAMESPACE", raising=False)
    monkeypatch.delenv("PROMETHEUS_URL", raising=False)

    assert _build_episode_runner() is log_only_episode_runner


def test_build_episode_runner_falls_back_when_only_some_env_set(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.example:8000/v1")
    monkeypatch.delenv("APP_NAMESPACE", raising=False)
    monkeypatch.delenv("PROMETHEUS_URL", raising=False)

    assert _build_episode_runner() is log_only_episode_runner


def test_build_episode_runner_uses_log_only_mode_for_r1(monkeypatch):
    monkeypatch.setenv("SSLM_LOG_ONLY", "1")
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.example:8000/v1")
    monkeypatch.setenv("APP_NAMESPACE", "hotelReservation")
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus.example:9090")

    assert _build_episode_runner() is log_only_episode_runner


def test_build_episode_runner_falls_back_on_malformed_numeric_env(monkeypatch):
    """A malformed VLLM_MAX_OUTPUT_TOKENS/VLLM_TEMPERATURE
    (operator typo, not missing) must degrade to log-only, not crash the process at startup --
    same guarantee this function's own docstring already promises for a missing var."""
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.example:8000/v1")
    monkeypatch.setenv("APP_NAMESPACE", "hotelReservation")
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus.example:9090")
    monkeypatch.setenv("VLLM_MAX_OUTPUT_TOKENS", "not-a-number")

    assert _build_episode_runner() is log_only_episode_runner


def test_build_episode_runner_wires_real_orchestrator_when_env_set(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.example:8000/v1")
    monkeypatch.setenv("APP_NAMESPACE", "hotelReservation")
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus.example:9090")

    import harness.episode_runner as episode_runner_module

    sentinel = object()
    seen_kwargs = {}

    def fake_make_episode_runner(**kwargs):
        seen_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(episode_runner_module, "make_episode_runner", fake_make_episode_runner)

    import adapters.vllm_model_client as vllm_model_client_module

    class FakeVllmModelClient:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(vllm_model_client_module, "VllmModelClient", FakeVllmModelClient)

    result = _build_episode_runner()

    assert result is sentinel
    assert seen_kwargs["namespace"] == "hotelReservation"
    assert seen_kwargs["prometheus_url"] == "http://prometheus.example:9090"
    assert isinstance(seen_kwargs["model"], FakeVllmModelClient)
    assert seen_kwargs["model"].config.base_url == "http://vllm.example:8000/v1"
    # Default is dry-run ON (SSLM_C2_DRY_RUN unset here).
    assert seen_kwargs["c2_dry_run"] is True


def _capture_make_episode_runner_kwargs(monkeypatch):
    import adapters.vllm_model_client as vllm_model_client_module
    import harness.episode_runner as episode_runner_module

    seen_kwargs = {}

    def fake_make_episode_runner(**kwargs):
        seen_kwargs.update(kwargs)
        return object()

    class FakeVllmModelClient:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(episode_runner_module, "make_episode_runner", fake_make_episode_runner)
    monkeypatch.setattr(vllm_model_client_module, "VllmModelClient", FakeVllmModelClient)
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.example:8000/v1")
    monkeypatch.setenv("APP_NAMESPACE", "hotelReservation")
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus.example:9090")
    return seen_kwargs


def test_build_episode_runner_wires_jsonl_sink_when_triage_log_is_set(monkeypatch, tmp_path):
    from harness.episode_runner import JsonlTriageSink

    seen_kwargs = _capture_make_episode_runner_kwargs(monkeypatch)
    monkeypatch.setenv("SSLM_TRIAGE_LOG", str(tmp_path / "triage.jsonl"))

    _build_episode_runner()

    sink = seen_kwargs["triage_sink"]
    assert isinstance(sink, JsonlTriageSink)
    assert sink.path == str(tmp_path / "triage.jsonl")


def test_build_episode_runner_keeps_log_only_sink_when_triage_log_unset(monkeypatch):
    seen_kwargs = _capture_make_episode_runner_kwargs(monkeypatch)
    monkeypatch.delenv("SSLM_TRIAGE_LOG", raising=False)

    _build_episode_runner()

    assert seen_kwargs["triage_sink"] is None


def test_build_episode_runner_passes_c2_dry_run_false_when_env_disables_it(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.example:8000/v1")
    monkeypatch.setenv("APP_NAMESPACE", "hotelReservation")
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus.example:9090")
    monkeypatch.setenv("SSLM_C2_DRY_RUN", "0")

    import harness.episode_runner as episode_runner_module

    sentinel = object()
    seen_kwargs = {}

    def fake_make_episode_runner(**kwargs):
        seen_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(episode_runner_module, "make_episode_runner", fake_make_episode_runner)

    import adapters.vllm_model_client as vllm_model_client_module

    class FakeVllmModelClient:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(vllm_model_client_module, "VllmModelClient", FakeVllmModelClient)

    result = _build_episode_runner()

    assert result is sentinel
    assert seen_kwargs["c2_dry_run"] is False


# ---------------------------------------------------------------------------
# Async dispatch: episode_runner used to
# run synchronously inside handle_alert_payload, before the 202 was written. See
# entrypoint._default_dispatch's own docstring for the full reasoning (including why a longer
# webhook_configs timeout was evaluated and rejected as an alternative fix).
# ---------------------------------------------------------------------------


def test_default_dispatch_runs_task_on_a_background_thread_and_returns_immediately():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_task():
        started.set()
        release.wait(timeout=5)
        finished.set()

    _default_dispatch(slow_task)

    assert started.wait(timeout=5), "dispatched task never started"
    assert not finished.is_set(), "_default_dispatch must return before the task finishes"
    release.set()
    assert finished.wait(timeout=5), "dispatched task never finished after being released"


def test_default_dispatch_catches_and_logs_task_exceptions(monkeypatch):
    # Deterministic without a sleep: the fake logger.exception itself sets the event, so waiting
    # on it also waits for the background thread's except branch to actually run.
    logged = threading.Event()
    monkeypatch.setattr(entrypoint.logger, "exception", lambda *a, **k: logged.set())

    def raising_task():
        raise RuntimeError("boom")

    _default_dispatch(raising_task)  # must not raise on the calling thread

    assert logged.wait(timeout=5), "exception from the dispatched task was never logged"


def test_handle_alert_payload_returns_202_before_a_slow_runner_finishes():
    """Integration-level proof that WebhookReceiver's default (production) dispatch is really
    async, not just _default_dispatch in isolation: a slow episode_runner must not delay the 202
    response -- the exact bug this module's Resolved honest limit describes."""
    started = threading.Event()
    release = threading.Event()

    def slow_runner(episode_id, raw):
        started.set()
        release.wait(timeout=5)

    receiver = WebhookReceiver(episode_runner=slow_runner)  # default dispatch: _default_dispatch

    t0 = time.monotonic()
    status, body = receiver.handle_alert_payload(_payload())
    elapsed = time.monotonic() - t0

    assert status == 202
    assert body["new_episode"] is True
    assert elapsed < 1.0, "handle_alert_payload blocked on a slow episode_runner"
    assert started.wait(timeout=5), "episode_runner was never dispatched"
    release.set()


class TestHttpLayer:
    """Real socket, only for the two behaviors that live in do_POST itself rather than in
    WebhookReceiver (see module docstring)."""

    def setup_method(self):
        receiver = WebhookReceiver()
        handler_cls = _make_handler(receiver)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def teardown_method(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_invalid_utf8_body_returns_400_not_a_dropped_connection(self):
        # json.loads on bytes raises UnicodeDecodeError (not JSONDecodeError) for invalid
        # encoding. The earlier version only caught
        # json.JSONDecodeError, so this body crashed the request instead of returning 400.
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            invalid_utf8 = b"\xff\xfe\x00invalid"
            conn.request(
                "POST",
                "/alerts",
                body=invalid_utf8,
                headers={"Content-Type": "application/json", "Content-Length": str(len(invalid_utf8))},
            )
            response = conn.getresponse()
            assert response.status == 400
            response.read()
        finally:
            conn.close()

    def test_handler_declares_a_connection_timeout(self):
        # With no timeout, a slow/stalled client body pins a
        # ThreadingHTTPServer worker thread indefinitely. Asserting the class attribute (rather
        # than actually stalling a connection for the timeout duration) keeps this test fast;
        # cpython's BaseHTTPRequestHandler.handle_one_request is what turns the attribute into
        # actual socket.settimeout() behavior, not this codebase.
        handler_cls = _make_handler(WebhookReceiver())
        assert handler_cls.timeout == _REQUEST_TIMEOUT_SECONDS
        assert _REQUEST_TIMEOUT_SECONDS > 0


# SSLM_PROMPT_BUDGET_CHARS reaches the agent config once,
# at wiring time, and a malformed value keeps the pod from starting.


def test_build_episode_runner_leaves_the_prompt_budget_off_by_default(monkeypatch):
    seen_kwargs = _capture_make_episode_runner_kwargs(monkeypatch)
    monkeypatch.delenv("SSLM_PROMPT_BUDGET_CHARS", raising=False)
    monkeypatch.delenv("SSLM_HISTORY_BUDGET", raising=False)

    _build_episode_runner()

    config = seen_kwargs["agent_config"]
    assert config.prompt_budget is None
    assert config.history_budget is True
    assert config.schema_constrained_decoding is True
    assert config.memory_enabled is False


def test_build_episode_runner_threads_the_prompt_budget_into_the_agent_config(monkeypatch):
    from harness.agent import PromptBudget

    seen_kwargs = _capture_make_episode_runner_kwargs(monkeypatch)
    monkeypatch.setenv("SSLM_PROMPT_BUDGET_CHARS", "18000")

    _build_episode_runner()

    assert seen_kwargs["agent_config"].prompt_budget == PromptBudget(18_000)


def test_a_malformed_prompt_budget_refuses_to_start(monkeypatch):
    import pytest

    seen_kwargs = _capture_make_episode_runner_kwargs(monkeypatch)
    monkeypatch.setenv("SSLM_PROMPT_BUDGET_CHARS", "6k tokens")
    monkeypatch.setattr(entrypoint, "serve", lambda **_: pytest.fail("served with a broken budget"))

    with pytest.raises(SystemExit) as excinfo:
        entrypoint.main([])

    assert excinfo.value.code == 2
    assert "agent_config" not in seen_kwargs


# SSLM_TOOL_CONTRACT, set by setup_cluster.sh on the
# Deployment, reaches the agent config once, at wiring time, and a typo keeps the pod from starting.


def test_build_episode_runner_leaves_the_tool_contract_off_by_default(monkeypatch):
    seen_kwargs = _capture_make_episode_runner_kwargs(monkeypatch)
    monkeypatch.delenv("SSLM_TOOL_CONTRACT", raising=False)

    _build_episode_runner()

    assert seen_kwargs["agent_config"].tool_contract is False


def test_build_episode_runner_turns_the_tool_contract_on_from_the_deployment_value(monkeypatch):
    seen_kwargs = _capture_make_episode_runner_kwargs(monkeypatch)
    monkeypatch.setenv("SSLM_TOOL_CONTRACT", "on")

    _build_episode_runner()

    assert seen_kwargs["agent_config"].tool_contract is True


def test_a_malformed_tool_contract_refuses_to_start(monkeypatch):
    import pytest

    seen_kwargs = _capture_make_episode_runner_kwargs(monkeypatch)
    monkeypatch.setenv("SSLM_TOOL_CONTRACT", "onn")
    monkeypatch.setattr(entrypoint, "serve", lambda **_: pytest.fail("served with a broken contract switch"))

    with pytest.raises(SystemExit) as excinfo:
        entrypoint.main([])

    assert excinfo.value.code == 2
    assert "agent_config" not in seen_kwargs


# VLLM_MODEL and VLLM_DECODING_DIALECT reach the vLLM client
# once, at wiring time; empty keeps the 7B with guided_json, an unknown dialect refuses to start.


def test_build_episode_runner_keeps_the_7b_and_guided_json_when_both_are_empty(monkeypatch):
    seen_kwargs = _capture_make_episode_runner_kwargs(monkeypatch)
    monkeypatch.setenv("VLLM_MODEL", "")
    monkeypatch.setenv("VLLM_DECODING_DIALECT", "")

    _build_episode_runner()

    assert seen_kwargs["model"].config.model == "Qwen2.5-7B-Instruct"
    assert seen_kwargs["model"].config.decoding_dialect == "guided_json"


def test_build_episode_runner_threads_model_and_dialect_into_the_client(monkeypatch):
    seen_kwargs = _capture_make_episode_runner_kwargs(monkeypatch)
    monkeypatch.setenv("VLLM_MODEL", "Qwen3.5-9B")
    monkeypatch.setenv("VLLM_DECODING_DIALECT", "structured_outputs")

    _build_episode_runner()

    assert seen_kwargs["model"].config.model == "Qwen3.5-9B"
    assert seen_kwargs["model"].config.decoding_dialect == "structured_outputs"


def test_an_unknown_decoding_dialect_refuses_to_start(monkeypatch):
    import pytest

    seen_kwargs = _capture_make_episode_runner_kwargs(monkeypatch)
    monkeypatch.setenv("VLLM_DECODING_DIALECT", "guided-json")
    monkeypatch.setattr(entrypoint, "serve", lambda **_: pytest.fail("served with an unknown dialect"))

    with pytest.raises(SystemExit) as excinfo:
        entrypoint.main([])

    assert excinfo.value.code == 2
    assert "model" not in seen_kwargs
