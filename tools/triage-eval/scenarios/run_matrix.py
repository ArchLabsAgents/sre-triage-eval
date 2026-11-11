#!/usr/bin/env python3
"""Drive the 7-scenario x 3-config matrix and write scenarios/results/matrix.csv.

Needs a real benchmark adapter (ModelClient + Environment implementations for ITBench or
AIOpsLab) to run for real; this CLI wires the harness end to end but has no built-in adapter.
Point --model-client and --environment-factory at your adapter module
(`mlx_factories:ModelClient`, `mlx_factories:make_environment`), or use --dry-run against the
bundled EchoModelClient/ScriptedEnvironment to sanity-check the harness plumbing itself.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from adapters.vllm_model_client import ModelUnavailable  # noqa: E402
from harness.agent import (  # noqa: E402
    Agent,
    AgentConfig,
    Environment,
    ModelClient,
    history_budget_enabled,
)
from harness.envs import EnvConfigError, load_envs, select_env  # noqa: E402
from harness.memory import EpisodicMemory  # noqa: E402
from harness.metrics import RunRecord, aggregate, collect_records, write_matrix_csv  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


class EchoModelClient:
    """Dry-run stand-in: always returns a valid, trivially-solving action.

    Only exists so `run_matrix.py --dry-run` can prove the harness plumbing (config loading,
    schema validation, memory, CSV output) works before a real benchmark adapter exists.
    """

    def generate(self, prompt: str, *, schema_constrained: bool) -> str:
        return json.dumps(
            {
                "diagnosis": "dry-run stub",
                "cause_class": "unknown",
                "action_type": "submit",
                "target": "n/a",
                "command": None,
                "confidence": 1.0,
            }
        )


class ScriptedEnvironment:
    """Dry-run stand-in: solves on the first action, for harness plumbing checks only."""

    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id

    def task_description(self) -> str:
        return f"dry-run scenario {self.scenario_id}"

    def apply_action(self, action: dict) -> tuple[bool, str]:
        return True, "dry-run: accepted"


def _load_config(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    extends = config.pop("extends", None)
    if extends:
        base = yaml.safe_load((config_path.parent / extends).read_text(encoding="utf-8"))
        base.update(config)
        config = base
    return config


def _resolve(dotted_path: str):
    module_name, _, attr = dotted_path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--env",
        help=(
            "execution environment this run happens in (a key of `envs:` in the config). "
            "Required when more than one env is declared, defaulted when exactly one is. "
            "Results land in results/<env>/ so two envs never overwrite each other."
        ),
    )
    parser.add_argument(
        "--model-client",
        help="module:attr resolving to a ModelClient instance factory, e.g. mlx_factories:make_model_client",
    )
    parser.add_argument(
        "--environment-factory",
        help="module:attr resolving to a callable(scenario_id) -> Environment",
    )
    parser.add_argument("--dry-run", action="store_true", help="use the bundled stub adapter")
    parser.add_argument(
        "--control",
        action="store_true",
        help=(
            "no-fault control run: deploys the same app but skips the environment's fault "
            "injection, to calibrate how often the eval passes with a cluster never touched "
            "(the benchmark's mitigation eval only checks pod state). Only meaningful "
            "with a real --environment-factory; ignored under --dry-run."
        ),
    )
    args = parser.parse_args()

    config = _load_config(args.config)

    # Resolve the env BEFORE building the model/environment: a refusal (ambiguous env, env
    # without quantization) refuses the START, never after a deploy has begun.
    try:
        env_name, env_block = select_env(load_envs(config), args.env)
    except EnvConfigError as exc:
        parser.error(str(exc))
        return
    if env_name is not None:
        # Factories (mlx_factories etc.) resolve the env from SSLM_ENV; exporting it here makes
        # this invocation's --env the single source, instead of two values disagreeing.
        os.environ["SSLM_ENV"] = env_name

    results_dir = RESULTS_DIR / env_name if env_name else RESULTS_DIR

    if env_name is not None:
        # env-descriptor.json: the env block copied verbatim from the declaration, written once
        # per env. Diverging from the current declaration means base.yaml changed mid-matrix
        # (the confound the column exists to prevent): refuse, never overwrite.
        descriptor_path = results_dir / "env-descriptor.json"
        if descriptor_path.exists():
            existing_descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            if existing_descriptor != env_block:
                parser.error(
                    f"{descriptor_path} differs from the current declaration of '{env_name}' in the "
                    "config; the declaration changed mid-matrix. Resolve it before running (a new "
                    "env or a new results/), never by overwriting the descriptor."
                )
                return
        else:
            descriptor_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor_path.write_text(
                json.dumps(env_block, indent=2, sort_keys=True), encoding="utf-8"
            )

    if args.dry_run:
        model: ModelClient = EchoModelClient()
        environment: Environment = ScriptedEnvironment(args.scenario)
    elif args.model_client and args.environment_factory:
        model = _resolve(args.model_client)()
        env_kwargs = {"control": True} if args.control else {}
        environment = _resolve(args.environment_factory)(args.scenario, **env_kwargs)
    else:
        parser.error("pass --dry-run or both --model-client and --environment-factory")
        return

    memory = None
    if config.get("memory", {}).get("enabled"):
        store_path = Path(config["memory"]["store_path"])
        if env_name is not None:
            # Episodic memory is per env: episodes accumulated on a local run leaking into a
            # cloud round (or the reverse) would mix the envs through a side channel the CSV's
            # env column does not show. The file name comes from the config; the directory is
            # the env's.
            store_path = results_dir / store_path.name
        memory = EpisodicMemory.load(store_path)

    agent_config = AgentConfig(
        schema_constrained_decoding=bool(config.get("schema_constrained_decoding", False)),
        memory_enabled=bool(config.get("memory", {}).get("enabled", False)),
        memory_top_k=int(config.get("memory", {}).get("top_k", 3)),
        # SSLM_HISTORY_BUDGET=off only on the API comparison arm;
        # cloud_adapter.make_anthropic_model_client refuses to build without it.
        history_budget=history_budget_enabled(),
    )

    agent = Agent(model=model, config=agent_config, memory=memory)
    finalize_episode = getattr(environment, "finalize_episode", None)
    # finalize's return value is the benchmark's verdict (results["success"], TTM). It used to
    # be discarded and the matrix reported `solved` (= called submit()) as if it were a
    # mitigation.
    verdict: dict = {}
    result = None
    model_error: str | None = None
    try:
        result = agent.run_episode(environment, task_id=args.scenario)
    except ModelUnavailable as exc:
        # Transport failure (dead tunnel, provider down): never a malformed turn nor a failure.
        # Verdict not measured (success=None) + the reason in model_error, same convention as
        # eval_error.
        model_error = str(exc)
    finally:
        if finalize_episode is not None:
            verdict = finalize_episode() or {}

    env_quantization = env_block.get("quantization") if env_block else None
    env_network = env_block.get("network") if env_block else None
    stats = getattr(model, "stats", None)
    if result is not None:
        record = RunRecord.from_episode(
            args.scenario,
            config["name"],
            result,
            verdict,
            env=env_name,
            env_quantization=env_quantization,
            env_network=env_network,
            model_stats=stats,
        )
    else:
        record = RunRecord(
            scenario_id=args.scenario,
            config_name=config["name"],
            submitted=False,
            output_tokens=0,
            elapsed_seconds=0.0,
            attempt_count=0,
            success=None,
            teardown_error=verdict.get("teardown_error"),
            eval_error=verdict.get("eval_error"),
            model_error=model_error,
            env=env_name,
            env_quantization=env_quantization,
            env_network=env_network,
            model_seconds=stats.seconds if stats else None,
            model_calls=stats.calls if stats else None,
            model_retries=stats.retries if stats else None,
        )
    record.write(RESULTS_DIR)

    if memory is not None and result is not None:
        # The verdict is the only success/failure label memory carries; per turn it keeps only
        # the environment's response (harness.memory.Episode).
        success = record.success
        memory.set_verdict(
            args.scenario,
            "unknown" if success is None else ("success" if success else "failure"),
        )

    if result is not None:
        # Per-turn transcript (raw model output included): the evidence that was missing when
        # 10 malformed turns in an early run left nothing to diagnose.
        transcript_path = results_dir / f"{args.scenario}_{config['name']}_transcript.jsonl"
        with transcript_path.open("w", encoding="utf-8") as handle:
            for turn in result.turns:
                handle.write(json.dumps(asdict(turn)) + "\n")

    # matrix.csv at the root covers ALL envs (env column first); records come from the root
    # (legacy) and from each results/<env>/.
    existing = collect_records(RESULTS_DIR)
    write_matrix_csv(existing, RESULTS_DIR / "matrix.csv")

    if record.teardown_error:
        print(f"[warn] teardown failed after eval: {record.teardown_error}", file=sys.stderr)
    if record.model_error:
        print(f"[warn] model unavailable, run not measured: {record.model_error}", file=sys.stderr)
    print(
        f"{args.scenario} / {config['name']} [env={record.env or 'unknown'}]: "
        f"success={record.success} "
        f"submitted={record.submitted} ttm={record.ttm_seconds} "
        f"root_cause_fixed={record.root_cause_fixed}"
    )
    for summary in aggregate(existing):
        print(
            f"[{summary.env or 'unknown'}] {summary.config_name}: "
            f"{summary.scenarios_succeeded}/{summary.scenarios_attempted} success "
            f"({summary.success_rate:.0%}, {summary.scenarios_unmeasured} unmeasured), "
            f"{summary.scenarios_root_cause_fixed}/{summary.scenarios_attempted} root_cause_fixed, "
            f"{summary.scenarios_submitted}/{summary.scenarios_attempted} submitted, "
            f"mean tokens (success)={summary.mean_output_tokens_succeeded}, "
            f"mean seconds (success)={summary.mean_elapsed_seconds_succeeded}"
        )


if __name__ == "__main__":
    main()
