"""Wires the vLLM + AIOpsLab adapters into run_matrix.py's factory contract, the cloud-lab
counterpart of mlx_factories.py.

adapters/vllm_model_client.py and adapters/aiopslab_env.py already implement the
ModelClient/Environment protocols; this module only reads configs/base.yaml plus
VLLM_BASE_URL/VLLM_API_KEY once and constructs them, so `run_matrix.py --model-client
cloud_adapter:make_model_client --environment-factory cloud_adapter:make_environment` matches
the module:attr pair the CLI's own help text names.

AiopsLabEnvironment is reused unchanged: AIOpsLab itself is provider-agnostic (it only talks to
whatever kubeconfig is current), so the same factory that drives the Mac's kind cluster drives
the instance's, once setup_cluster.sh has created it there.

Import of adapters.vllm_model_client is deferred into make_model_client, not top-level, same
reason mlx_factories defers adapters.mlx_model_client: nothing here needs a live server to be
unit-tested.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from harness.envs import EnvConfigError, load_envs, select_env

BASE_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "base.yaml"


def _load_base_config() -> dict:
    return yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))


def _resolve_vllm_env() -> dict:
    """Which env this vLLM factory serves. Resolution goes through harness.envs, the same
    loader run_matrix.py uses (it exports SSLM_ENV from --env before resolving factories, so
    this never disagrees with the row the run stamps). Refuses an env whose backend is not
    vllm, instead of silently posting to a vLLM server while the row claims a different env."""
    config = _load_base_config()
    envs = load_envs(config)
    requested = os.environ.get("SSLM_ENV") or None
    if requested is None:
        # Auto-default only among the EPISODIC vllm envs: this adapter backs run_matrix.py (the
        # 7x3 matrix), never the continuous scenario's own tooling
        # (scenarios/aggregate_footprint.py), so a "-continuous" env never wins the default even
        # though it also declares backend: vllm. Explicit SSLM_ENV still selects it if asked.
        vllm_envs = [
            name
            for name, block in envs.items()
            if block.get("backend") == "vllm" and not name.endswith("-continuous")
        ]
        if len(vllm_envs) == 1:
            requested = vllm_envs[0]
    env_name, env_block = select_env(envs, requested)
    if env_block is None or env_block.get("backend") != "vllm":
        raise EnvConfigError(
            f"cloud_adapter is the vLLM factory, but the selected env {env_name!r} declares "
            f"backend={env_block.get('backend') if env_block else None!r}"
        )
    return env_block


def make_model_client():
    from adapters.vllm_model_client import VllmModelClient, VllmModelConfig

    model_config = _load_base_config()["model"]
    env_block = _resolve_vllm_env()
    base_url = os.environ.get("VLLM_BASE_URL")
    if not base_url:
        raise EnvConfigError(
            "VLLM_BASE_URL is not set; export the base URL of the vLLM server for this env"
        )
    served_model_name = env_block.get("served_model_name")
    if not served_model_name:
        raise EnvConfigError(
            f"env {env_block!r} declares no served_model_name (configs/base.yaml); it must "
            "match the --served-model-name vLLM was started with"
        )
    return VllmModelClient(
        VllmModelConfig(
            base_url=base_url,
            model=served_model_name,
            max_output_tokens=model_config["max_output_tokens"],
            temperature=model_config["temperature"],
            api_key=os.environ.get("VLLM_API_KEY") or None,
        )
    )


def make_anthropic_model_client(*, max_output_tokens: int | None = None):
    """Comparator arm (hosted API model): same ModelClient protocol, Anthropic backend. The key
    comes from ANTHROPIC_API_KEY, resolved by the operator from their own secret store.
    SSLM_COMPARATOR_MAX_COST_USD caps the arm's spend
    and is required (this arm bills real money per request, so there is no uncapped default);
    SSLM_COMPARATOR_USAGE_LOG names the per-request JSONL.

    `max_output_tokens` defaults to configs/base.yaml's model.max_output_tokens (what run_matrix.py
    gives the vLLM arm). The in-cluster agent (entrypoint.py) passes the value its own vLLM arm
    uses instead, so the output ceiling stays matched to the arm it is compared against."""
    from adapters.anthropic_model_client import AnthropicModelClient, AnthropicModelConfig

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvConfigError(
            "ANTHROPIC_API_KEY is not set; export it from your secret store before starting "
            "the comparator arm"
        )
    # Pre-registered: the API arm sends the exact prompt the earlier 9B rounds sent (before the
    # history budget existed), so the history budget must be off for the agent this client is paired with.
    # run_matrix.py reads the same variable when it builds AgentConfig.
    from harness.agent import HISTORY_BUDGET_ENV_VAR, history_budget_enabled

    if history_budget_enabled():
        raise EnvConfigError(
            f"{HISTORY_BUDGET_ENV_VAR} must be 'off' for the comparator arm (the API arm runs the "
            "prompt from before the history budget existed); export SSLM_HISTORY_BUDGET=off"
        )
    raw_cost = os.environ.get("SSLM_COMPARATOR_MAX_COST_USD")
    if not raw_cost:
        raise EnvConfigError(
            "SSLM_COMPARATOR_MAX_COST_USD is not set; the comparator arm bills per request and "
            "refuses to run without a spend cap (e.g. 15 for the pre-registered credit)"
        )
    try:
        max_cost = float(raw_cost)
    except ValueError as exc:
        raise EnvConfigError(f"SSLM_COMPARATOR_MAX_COST_USD={raw_cost!r} is not a number") from exc
    if not max_cost > 0:
        raise EnvConfigError(f"SSLM_COMPARATOR_MAX_COST_USD must be > 0, got {raw_cost!r}")
    usage_log = os.environ.get("SSLM_COMPARATOR_USAGE_LOG")
    return AnthropicModelClient(
        AnthropicModelConfig(
            api_key=api_key,
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else _load_base_config()["model"]["max_output_tokens"]
            ),
            max_cost_usd=max_cost,
            usage_log_path=Path(usage_log) if usage_log else None,
        )
    )


def make_environment(scenario_id: str, *, control: bool = False):
    from adapters.aiopslab_env import AiopsLabEnvironment

    return AiopsLabEnvironment(problem_id=scenario_id, control_mode=control)
