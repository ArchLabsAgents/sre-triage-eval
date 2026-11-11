"""entrypoint.py picks the model client by COMPARATOR_ARM (hosted-model comparator arm).

"1" wires the Anthropic client into the same orchestrator and sinks as the vLLM arm; unset or
"0" keeps today's path. The comparator path fails closed: every missing or wrong piece raises
ComparatorWiringError, and main() turns that into a non-zero exit before serve() runs, so the
pod never becomes Ready and the window stops at `rollout status`. No test here opens a socket or
calls the API: the Anthropic client is only constructed, never asked to generate.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import entrypoint  # noqa: E402
from entrypoint import ComparatorWiringError, _build_episode_runner, log_only_episode_runner  # noqa: E402

FAKE_KEY = "sk-ant-test-not-a-real-key"


@pytest.fixture
def captured(monkeypatch):
    """Stubs make_episode_runner (records kwargs) and makes any VllmModelClient construction fail
    the test, so a comparator path that drifted back to vLLM cannot pass silently."""
    import adapters.vllm_model_client as vllm_model_client_module
    import harness.episode_runner as episode_runner_module

    seen: dict = {}
    sentinel = object()

    def fake_make_episode_runner(**kwargs):
        seen.update(kwargs)
        return sentinel

    class NoVllm:
        def __init__(self, config):
            seen["vllm_config"] = config

    monkeypatch.setattr(episode_runner_module, "make_episode_runner", fake_make_episode_runner)
    monkeypatch.setattr(vllm_model_client_module, "VllmModelClient", NoVllm)
    seen["sentinel"] = sentinel
    return seen


@pytest.fixture
def comparator_env(monkeypatch, tmp_path):
    for name in ("SSLM_LOG_ONLY", "VLLM_BASE_URL", "VLLM_MAX_OUTPUT_TOKENS", "VLLM_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)
    env = {
        "COMPARATOR_ARM": "1",
        "ANTHROPIC_API_KEY": FAKE_KEY,
        "SSLM_COMPARATOR_MAX_COST_USD": "5",
        "SSLM_COMPARATOR_USAGE_LOG": str(tmp_path / "comparator-usage.jsonl"),
        "SSLM_TRIAGE_LOG": str(tmp_path / "triage.jsonl"),
        "SSLM_HISTORY_BUDGET": "off",
        "APP_NAMESPACE": "test-hotel-reservation",
        "PROMETHEUS_URL": "http://prometheus.example:9090",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


def test_comparator_arm_wires_the_anthropic_client_into_the_shared_orchestrator(captured, comparator_env, tmp_path):
    from adapters.anthropic_model_client import PINNED_MODEL, AnthropicModelClient
    from harness.episode_runner import JsonlTriageSink, JsonlUsageSink

    result = _build_episode_runner()

    assert result is captured["sentinel"]
    assert "vllm_config" not in captured
    model = captured["model"]
    assert isinstance(model, AnthropicModelClient)
    assert model.config.model == PINNED_MODEL
    assert model.config.api_key == FAKE_KEY
    assert model.config.max_cost_usd == 5.0
    assert model.config.usage_log_path == tmp_path / "comparator-usage.jsonl"
    # Same output ceiling as the vLLM arm in this process (VLLM_MAX_OUTPUT_TOKENS default), not
    # configs/base.yaml's run_matrix value: max tokens is not a variable the comparison lets differ.
    assert model.config.max_output_tokens == 512
    # Same sinks, namespace, Prometheus and C2 gate as the vLLM arm.
    assert captured["namespace"] == "test-hotel-reservation"
    assert captured["prometheus_url"] == "http://prometheus.example:9090"
    assert isinstance(captured["triage_sink"], JsonlTriageSink)
    assert captured["triage_sink"].path == str(tmp_path / "triage.jsonl")
    assert isinstance(captured["usage_sink"], JsonlUsageSink)
    assert captured["c2_dry_run"] is True


def test_comparator_arm_ignores_a_vllm_endpoint_in_the_env(captured, comparator_env, monkeypatch):
    # setup_cluster.sh substitutes VLLM_BASE_URL="" on this arm, but even a stray value must not
    # bring the vLLM client back behind a NetworkPolicy that has no route to it.
    monkeypatch.setenv("VLLM_BASE_URL", "http://198.51.100.1:8000/v1")

    _build_episode_runner()

    assert "vllm_config" not in captured
    assert type(captured["model"]).__name__ == "AnthropicModelClient"


def test_comparator_arm_reuses_the_vllm_arm_output_ceiling_override(captured, comparator_env, monkeypatch):
    monkeypatch.setenv("VLLM_MAX_OUTPUT_TOKENS", "700")

    _build_episode_runner()

    assert captured["model"].config.max_output_tokens == 700


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        ({"ANTHROPIC_API_KEY": None}, "ANTHROPIC_API_KEY"),
        ({"ANTHROPIC_API_KEY": ""}, "ANTHROPIC_API_KEY"),
        ({"SSLM_HISTORY_BUDGET": None}, "SSLM_HISTORY_BUDGET"),
        ({"SSLM_HISTORY_BUDGET": "on"}, "SSLM_HISTORY_BUDGET"),
        ({"SSLM_COMPARATOR_MAX_COST_USD": None}, "SSLM_COMPARATOR_MAX_COST_USD"),
        ({"SSLM_COMPARATOR_MAX_COST_USD": "0"}, "SSLM_COMPARATOR_MAX_COST_USD"),
        ({"SSLM_COMPARATOR_MAX_COST_USD": "lots"}, "SSLM_COMPARATOR_MAX_COST_USD"),
        ({"SSLM_COMPARATOR_USAGE_LOG": None}, "SSLM_COMPARATOR_USAGE_LOG"),
        ({"SSLM_TRIAGE_LOG": None}, "SSLM_TRIAGE_LOG"),
        ({"APP_NAMESPACE": None}, "APP_NAMESPACE"),
        ({"PROMETHEUS_URL": ""}, "PROMETHEUS_URL"),
        ({"SSLM_LOG_ONLY": "1"}, "SSLM_LOG_ONLY"),
        ({"VLLM_MAX_OUTPUT_TOKENS": "many"}, "VLLM_MAX_OUTPUT_TOKENS"),
        # A set budget is valid here since the short Sonnet window; a malformed one still stops it.
        ({"SSLM_PROMPT_BUDGET_CHARS": "18k"}, "SSLM_PROMPT_BUDGET_CHARS"),
    ],
)
def test_comparator_arm_fails_closed_never_falls_back(captured, comparator_env, monkeypatch, change, needle):
    for name, value in change.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    with pytest.raises(ComparatorWiringError) as excinfo:
        _build_episode_runner()

    assert needle in str(excinfo.value)
    # Neither fallback happened: no vLLM client, no orchestrator, no log-only runner returned.
    assert "vllm_config" not in captured
    assert "model" not in captured
    assert FAKE_KEY not in str(excinfo.value)


@pytest.mark.parametrize("raw", ["true", "yes", "2", " 1"])
def test_an_unrecognized_comparator_arm_value_is_refused(captured, monkeypatch, raw):
    monkeypatch.setenv("COMPARATOR_ARM", raw)
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.example:8000/v1")
    monkeypatch.setenv("APP_NAMESPACE", "hotelReservation")
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus.example:9090")

    with pytest.raises(ComparatorWiringError, match="COMPARATOR_ARM"):
        _build_episode_runner()
    assert "model" not in captured


@pytest.mark.parametrize("raw", [None, "", "0"])
def test_default_and_zero_keep_the_vllm_path(captured, monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("COMPARATOR_ARM", raising=False)
    else:
        monkeypatch.setenv("COMPARATOR_ARM", raw)
    # A key in the env (e.g. a stale Secret) must not change the vLLM arm.
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.delenv("SSLM_LOG_ONLY", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.example:8000/v1")
    monkeypatch.setenv("APP_NAMESPACE", "hotelReservation")
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus.example:9090")

    result = _build_episode_runner()

    assert result is captured["sentinel"]
    assert captured["vllm_config"].base_url == "http://vllm.example:8000/v1"
    assert type(captured["model"]).__name__ == "NoVllm"


@pytest.mark.parametrize("raw", [None, "0"])
def test_default_path_still_degrades_to_log_only_when_under_configured(monkeypatch, raw):
    # The vLLM arm's own degrade-not-crash contract is untouched by the comparator switch.
    if raw is None:
        monkeypatch.delenv("COMPARATOR_ARM", raising=False)
    else:
        monkeypatch.setenv("COMPARATOR_ARM", raw)
    for name in ("VLLM_BASE_URL", "APP_NAMESPACE", "PROMETHEUS_URL", "SSLM_LOG_ONLY"):
        monkeypatch.delenv(name, raising=False)

    assert _build_episode_runner() is log_only_episode_runner


def test_main_exits_non_zero_before_serving_when_the_comparator_is_not_wired(comparator_env, monkeypatch, caplog):
    monkeypatch.delenv("SSLM_COMPARATOR_MAX_COST_USD")
    served = []
    monkeypatch.setattr(entrypoint, "serve", lambda **kwargs: served.append(kwargs))

    with caplog.at_level(logging.CRITICAL, logger="sslm.entrypoint"):
        with pytest.raises(SystemExit) as excinfo:
            entrypoint.main([])

    assert excinfo.value.code == 2
    assert served == []
    assert "SSLM_COMPARATOR_MAX_COST_USD" in caplog.text
    assert FAKE_KEY not in caplog.text


def test_main_serves_the_comparator_runner_when_wired(captured, comparator_env, monkeypatch, caplog):
    served = []
    monkeypatch.setattr(entrypoint, "serve", lambda **kwargs: served.append(kwargs))

    with caplog.at_level(logging.INFO, logger="sslm.entrypoint"):
        entrypoint.main([])

    assert served and served[0]["episode_runner"] is captured["sentinel"]
    assert "comparator arm wired" in caplog.text
    assert FAKE_KEY not in caplog.text


def test_dockerfile_ships_cloud_adapter_for_the_comparator_path():
    # entrypoint.py imports cloud_adapter on this arm; an image without it would crash every
    # comparator pod at startup with ImportError instead of the intended wiring check.
    text = (TOOLS_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY tools/triage-eval/cloud_adapter.py ./cloud_adapter.py" in text
    assert "COPY tools/triage-eval/configs ./configs" in text


# Short Sonnet window: the API arm takes the local arm's budget, nothing else moves.


def test_comparator_arm_without_a_budget_keeps_the_earlier_api_wiring(captured, comparator_env, monkeypatch):
    monkeypatch.delenv("SSLM_PROMPT_BUDGET_CHARS", raising=False)

    _build_episode_runner()

    assert captured["agent_config"] is None


def test_comparator_arm_threads_the_prompt_budget_into_the_agent_config(captured, comparator_env, monkeypatch):
    from harness.agent import PromptBudget

    monkeypatch.setenv("SSLM_PROMPT_BUDGET_CHARS", "14000")

    _build_episode_runner()

    config = captured["agent_config"]
    assert config.prompt_budget == PromptBudget(14_000)
    assert config.history_budget is False
    assert config.memory_enabled is False
    assert config.schema_constrained_decoding is True
    assert not config.tool_contract
