"""Unit tests for cloud_adapter, the vLLM-side counterpart of test_mlx_factories.py.

make_model_client() constructs a real VllmModelClient, which only needs the stdlib (no live
server) -- unlike mlx_factories's MLX path, this one IS exercised end-to-end for construction,
just never for an actual generate() call.
"""

from __future__ import annotations

import pytest

import cloud_adapter
from adapters.aiopslab_env import AiopsLabEnvironment
from adapters.vllm_model_client import VllmModelClient
from harness.envs import EnvConfigError


def test_resolve_vllm_env_defaults_to_the_declared_vllm_env(monkeypatch):
    monkeypatch.delenv("SSLM_ENV", raising=False)

    env_block = cloud_adapter._resolve_vllm_env()

    assert env_block["backend"] == "vllm"
    assert env_block["served_model_name"] == "Qwen2.5-7B-Instruct"


def test_make_model_client_refuses_a_non_vllm_env(monkeypatch):
    monkeypatch.setenv("SSLM_ENV", "mlx-local")

    with pytest.raises(EnvConfigError, match="vLLM factory"):
        cloud_adapter.make_model_client()


def test_make_model_client_refuses_without_vllm_base_url(monkeypatch):
    monkeypatch.delenv("SSLM_ENV", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    with pytest.raises(EnvConfigError, match="VLLM_BASE_URL"):
        cloud_adapter.make_model_client()


def test_make_model_client_wires_base_url_model_and_decoding_params(monkeypatch):
    monkeypatch.delenv("SSLM_ENV", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    client = cloud_adapter.make_model_client()

    assert isinstance(client, VllmModelClient)
    assert client.config.base_url == "http://127.0.0.1:8000/v1"
    assert client.config.model == "Qwen2.5-7B-Instruct"
    assert client.config.max_output_tokens == 2048
    assert client.config.temperature == 0.2
    assert client.config.api_key is None


def test_make_model_client_threads_the_api_key_when_set(monkeypatch):
    monkeypatch.delenv("SSLM_ENV", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "s3cr3t")

    client = cloud_adapter.make_model_client()

    assert client.config.api_key == "s3cr3t"


def test_make_anthropic_model_client_refuses_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(EnvConfigError, match="ANTHROPIC_API_KEY"):
        cloud_adapter.make_anthropic_model_client()


def test_make_anthropic_model_client_refuses_with_the_history_budget_on(monkeypatch):
    # The API arm sends the prompt from before the history budget existed, so the budget must be off.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SSLM_COMPARATOR_MAX_COST_USD", "15")
    for value in (None, "on"):
        if value is None:
            monkeypatch.delenv("SSLM_HISTORY_BUDGET", raising=False)
        else:
            monkeypatch.setenv("SSLM_HISTORY_BUDGET", value)
        with pytest.raises(EnvConfigError, match="SSLM_HISTORY_BUDGET"):
            cloud_adapter.make_anthropic_model_client()


def test_make_anthropic_model_client_refuses_without_a_spend_cap(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SSLM_HISTORY_BUDGET", "off")
    monkeypatch.delenv("SSLM_COMPARATOR_MAX_COST_USD", raising=False)

    with pytest.raises(EnvConfigError, match="SSLM_COMPARATOR_MAX_COST_USD"):
        cloud_adapter.make_anthropic_model_client()


@pytest.mark.parametrize("raw", ["fifteen", "0", "-3"])
def test_make_anthropic_model_client_refuses_a_bad_spend_cap(monkeypatch, raw):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SSLM_HISTORY_BUDGET", "off")
    monkeypatch.setenv("SSLM_COMPARATOR_MAX_COST_USD", raw)

    with pytest.raises(EnvConfigError, match="SSLM_COMPARATOR_MAX_COST_USD"):
        cloud_adapter.make_anthropic_model_client()


def test_make_anthropic_model_client_wires_key_budget_and_usage_log(monkeypatch, tmp_path):
    from adapters.anthropic_model_client import AnthropicModelClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SSLM_HISTORY_BUDGET", "off")
    monkeypatch.setenv("SSLM_COMPARATOR_MAX_COST_USD", "12.5")
    monkeypatch.setenv("SSLM_COMPARATOR_USAGE_LOG", str(tmp_path / "u.jsonl"))

    client = cloud_adapter.make_anthropic_model_client()

    assert isinstance(client, AnthropicModelClient)
    assert client.config.api_key == "sk-test"
    assert client.config.max_cost_usd == 12.5
    assert client.config.usage_log_path == tmp_path / "u.jsonl"
    assert client.config.max_output_tokens == 2048


def test_make_environment_passes_scenario_through_as_problem_id(fake_aiopslab_home):
    env = cloud_adapter.make_environment("misconfig_app-mitigation-1")

    assert isinstance(env, AiopsLabEnvironment)
    assert env.problem_id == "misconfig_app-mitigation-1"
    assert env.control_mode is False


def test_make_environment_control_flag_sets_control_mode(fake_aiopslab_home):
    env = cloud_adapter.make_environment("misconfig_app-mitigation-1", control=True)

    assert env.control_mode is True
