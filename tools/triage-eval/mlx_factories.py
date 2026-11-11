"""Wires the real MLX + AIOpsLab adapters into run_matrix.py's factory contract.

adapters/mlx_model_client.py and adapters/aiopslab_env.py already implement the
ModelClient/Environment protocols; this module only reads
configs/base.yaml once and constructs them, so `run_matrix.py --model-client
mlx_factories:make_model_client --environment-factory mlx_factories:make_environment`
matches the module:attr pair its own --model-client/--environment-factory help text
already names, instead of every caller reconstructing MlxModelConfig/AiopsLabEnvironment
by hand.

Imports of adapters.mlx_model_client and adapters.aiopslab_env are deferred into each
factory function, not top-level, for the same reason aiopslab_env.py itself defers its
`aiopslab` import: outlines/mlx-lm (base.yaml's decoding backend) and AIOPSLAB_HOME are
both real-cluster-run dependencies, not needed to unit-test this module's own wiring
logic.

--scenario at the CLI is passed straight through as the AIOpsLab problem_id (e.g. a
mitigation-family problem ID from AIOPSLAB_HOME/aiopslab/orchestrator/problem_registry.py).
This module does not hardcode a problem ID: the problem registry lives in whichever AIOpsLab
checkout AIOPSLAB_HOME points at, so hardcoding one here would be a guess, not a fact.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from harness.envs import EnvConfigError, load_envs, select_env

BASE_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "base.yaml"


def _load_base_config() -> dict:
    return yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))


def _resolve_mlx_env() -> dict:
    """Which env's weights this MLX factory loads. Resolution goes through harness.envs (the
    same loader run_matrix.py uses; run_matrix exports SSLM_ENV from its --env flag before
    resolving factories, so the two never disagree). This factory is the MLX path: it refuses
    an env whose backend is not mlx, instead of silently loading MLX weights and stamping the
    rows with a vLLM env."""
    config = _load_base_config()
    envs = load_envs(config)
    requested = os.environ.get("SSLM_ENV") or None
    if requested is None:
        mlx_envs = [name for name, block in envs.items() if block.get("backend") == "mlx"]
        if len(mlx_envs) == 1:
            requested = mlx_envs[0]
    env_name, env_block = select_env(envs, requested)
    if env_block is None or env_block.get("backend") != "mlx":
        raise EnvConfigError(
            f"mlx_factories is the MLX factory, but the selected env {env_name!r} declares "
            f"backend={env_block.get('backend') if env_block else None!r}"
        )
    return env_block


def make_model_client():
    from adapters.mlx_model_client import MlxModelClient, MlxModelConfig

    model_config = _load_base_config()["model"]
    env_block = _resolve_mlx_env()
    return MlxModelClient(
        MlxModelConfig(
            hf_repo=env_block["weights_ref"],
            max_output_tokens=model_config["max_output_tokens"],
            temperature=model_config["temperature"],
        )
    )


def make_environment(scenario_id: str, *, control: bool = False):
    from adapters.aiopslab_env import AiopsLabEnvironment

    return AiopsLabEnvironment(problem_id=scenario_id, control_mode=control)
