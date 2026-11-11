"""Unit tests for harness.envs: env declaration, legacy derivation and selection rules.
These rules are what keep two execution environments from ever
being read as one table, so every refusal here is load-bearing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harness.envs import EnvConfigError, load_envs, select_env

BASE_YAML = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "configs" / "base.yaml"


def _two_envs():
    return {
        "envs": {
            "mlx-local": {"backend": "mlx", "quantization": "8bit"},
            "hyperstack-a6000": {"backend": "vllm", "quantization": "fp16"},
        }
    }


def test_load_envs_returns_declared_map():
    envs = load_envs(_two_envs())

    assert set(envs) == {"mlx-local", "hyperstack-a6000"}
    assert envs["hyperstack-a6000"]["quantization"] == "fp16"


def test_load_envs_rejects_invalid_slug():
    with pytest.raises(EnvConfigError, match="slug"):
        load_envs({"envs": {"MLX_Local": {"quantization": "8bit"}}})


def test_load_envs_derives_mlx_local_from_legacy_model_keys():
    # Config written before the env column existed: derives a single mlx-local from the legacy
    # fields, without editing the file.
    config = {
        "model": {
            "backend": "mlx",
            "quantization": "8bit",
            "hf_repo": "mlx-community/Qwen2.5-7B-Instruct-8bit",
        }
    }
    envs = load_envs(config)

    assert list(envs) == ["mlx-local"]
    assert envs["mlx-local"]["quantization"] == "8bit"
    assert envs["mlx-local"]["weights_ref"] == "mlx-community/Qwen2.5-7B-Instruct-8bit"


def test_load_envs_without_envs_or_legacy_keys_declares_nothing():
    # The minimal unit-test configs (name + schema_constrained_decoding) do not
    # declare any env: the pre-env behavior is preserved.
    assert load_envs({"name": "config_a"}) == {}
    assert select_env({}, None) == (None, None)


def test_select_env_requires_flag_when_multiple_declared():
    with pytest.raises(EnvConfigError, match="required"):
        select_env(load_envs(_two_envs()), None)


def test_select_env_defaults_when_exactly_one_declared():
    envs = {"mlx-local": {"backend": "mlx", "quantization": "8bit"}}
    name, block = select_env(envs, None)

    assert name == "mlx-local"
    assert block["backend"] == "mlx"


def test_select_env_rejects_undeclared_name():
    with pytest.raises(EnvConfigError, match="not declared"):
        select_env(load_envs(_two_envs()), "scaleway-l4")


def test_select_env_refuses_missing_or_null_quantization():
    envs = {"mlx-local": {"backend": "mlx", "quantization": None}}
    with pytest.raises(EnvConfigError, match="quantization"):
        select_env(envs, "mlx-local")

    envs = {"mlx-local": {"backend": "mlx"}}
    with pytest.raises(EnvConfigError, match="quantization"):
        select_env(envs, "mlx-local")


def test_select_env_rejects_request_when_nothing_declared():
    with pytest.raises(EnvConfigError, match="declares no envs"):
        select_env({}, "mlx-local")


def test_base_yaml_declares_both_envs_with_quantization():
    # Ties the real base.yaml to the contract: the three current envs declared (mlx-local,
    # hyperstack-a6000 for the episodic matrix, hyperstack-a6000-continuous for the continuous
    # scenario), each with quantization (the start refusal never fires on a valid base.yaml).
    config = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    envs = load_envs(config)

    assert set(envs) == {"mlx-local", "hyperstack-a6000", "hyperstack-a6000-continuous"}
    for name in envs:
        selected_name, block = select_env(envs, name)
        assert selected_name == name
        assert block["quantization"]
        assert block["weights_ref"]
    # Invariants across envs live under model:, never per env (same family, same params
    # for decoding).
    assert "backend" not in config["model"]
    assert "quantization" not in config["model"]
    assert "hf_repo" not in config["model"]
