"""Unit tests for mlx_factories.

make_model_client() is not exercised end-to-end here: it constructs a real MlxModelClient,
which needs outlines/mlx-lm installed (see README's venv step), out of scope for the unit
suite. What's testable without those deps is the module's own logic: it reads the right
fields out of configs/base.yaml, and make_environment() wires the CLI --scenario value
straight through as the AIOpsLab problem_id, using the same fake `aiopslab` package as
test_aiopslab_env.py (fixture in conftest.py) so no real cluster is needed.
"""

from __future__ import annotations

import pytest

import mlx_factories
from adapters.aiopslab_env import AiopsLabEnvironment
from harness.envs import EnvConfigError


def test_resolve_mlx_env_defaults_to_the_declared_mlx_env(monkeypatch):
    monkeypatch.delenv("SSLM_ENV", raising=False)

    env_block = mlx_factories._resolve_mlx_env()

    assert env_block["backend"] == "mlx"
    assert env_block["weights_ref"] == "mlx-community/Qwen2.5-7B-Instruct-8bit"


def test_make_model_client_loads_env_weights_and_model_decoding_params(monkeypatch):
    monkeypatch.delenv("SSLM_ENV", raising=False)

    client = mlx_factories.make_model_client()

    assert client.config.hf_repo == "mlx-community/Qwen2.5-7B-Instruct-8bit"
    assert client.config.max_output_tokens == 2048
    assert client.config.temperature == 0.2


def test_make_model_client_refuses_a_non_mlx_env(monkeypatch):
    # This factory is the MLX path: with SSLM_ENV pointing at the vllm env, refusing beats
    # loading MLX weights and stamping the rows with the wrong env.
    monkeypatch.setenv("SSLM_ENV", "hyperstack-a6000")

    with pytest.raises(EnvConfigError, match="MLX factory"):
        mlx_factories.make_model_client()


def test_make_environment_passes_scenario_through_as_problem_id(fake_aiopslab_home):
    env = mlx_factories.make_environment("misconfig_app-mitigation-1")

    assert isinstance(env, AiopsLabEnvironment)
    assert env.problem_id == "misconfig_app-mitigation-1"
    assert env.control_mode is False


def test_make_environment_control_flag_sets_control_mode(fake_aiopslab_home):
    env = mlx_factories.make_environment("misconfig_app-mitigation-1", control=True)

    assert env.control_mode is True
