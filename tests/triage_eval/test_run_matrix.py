from __future__ import annotations

import sys

import pytest

from scenarios import run_matrix


def test_main_finalizes_environment_when_agent_raises(tmp_path, monkeypatch):
    class FailingModel:
        def generate(self, prompt: str, *, schema_constrained: bool) -> str:
            raise RuntimeError("model failed")

    class TrackingEnvironment:
        def __init__(self) -> None:
            self.finalized = False

        def task_description(self) -> str:
            return "diagnose the service"

        def apply_action(self, action: dict) -> tuple[bool, str]:
            return False, "not reached"

        def finalize_episode(self) -> dict:
            self.finalized = True
            return {}

    environment = TrackingEnvironment()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: config_a\nschema_constrained_decoding: false\n",
        encoding="utf-8",
    )

    factories = {
        "test:model": lambda: FailingModel(),
        "test:environment": lambda scenario_id: environment,
    }
    monkeypatch.setattr(run_matrix, "_resolve", factories.__getitem__)
    monkeypatch.setattr(run_matrix, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix.py",
            "--config",
            str(config_path),
            "--scenario",
            "p1",
            "--model-client",
            "test:model",
            "--environment-factory",
            "test:environment",
        ],
    )

    with pytest.raises(RuntimeError, match="model failed"):
        run_matrix.main()

    assert environment.finalized is True


def test_control_flag_is_passed_to_environment_factory(tmp_path, monkeypatch):
    class TrackingEnvironment:
        def task_description(self) -> str:
            return "diagnose the service"

        def apply_action(self, action: dict) -> tuple[bool, str]:
            return True, "Submission accepted."

        def finalize_episode(self) -> dict:
            return {"success": True}

    calls = []

    def factory(scenario_id, **kwargs):
        calls.append((scenario_id, kwargs))
        return TrackingEnvironment()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: control\nschema_constrained_decoding: false\n", encoding="utf-8"
    )

    factories = {
        "test:model": run_matrix.EchoModelClient,
        "test:environment": factory,
    }
    monkeypatch.setattr(run_matrix, "_resolve", factories.__getitem__)
    monkeypatch.setattr(run_matrix, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix.py",
            "--config",
            str(config_path),
            "--scenario",
            "p1",
            "--model-client",
            "test:model",
            "--environment-factory",
            "test:environment",
            "--control",
        ],
    )

    run_matrix.main()

    assert calls == [("p1", {"control": True})]


def test_without_control_flag_environment_factory_gets_no_extra_kwargs(tmp_path, monkeypatch):
    class TrackingEnvironment:
        def task_description(self) -> str:
            return "diagnose the service"

        def apply_action(self, action: dict) -> tuple[bool, str]:
            return True, "Submission accepted."

        def finalize_episode(self) -> dict:
            return {"success": True}

    calls = []

    def factory(scenario_id, **kwargs):
        calls.append((scenario_id, kwargs))
        return TrackingEnvironment()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: config_a\nschema_constrained_decoding: false\n", encoding="utf-8"
    )

    factories = {
        "test:model": run_matrix.EchoModelClient,
        "test:environment": factory,
    }
    monkeypatch.setattr(run_matrix, "_resolve", factories.__getitem__)
    monkeypatch.setattr(run_matrix, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix.py",
            "--config",
            str(config_path),
            "--scenario",
            "p1",
            "--model-client",
            "test:model",
            "--environment-factory",
            "test:environment",
        ],
    )

    run_matrix.main()

    assert calls == [("p1", {})]


def test_dry_run_writes_per_turn_transcript(tmp_path, monkeypatch):
    import json as json_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: config_a\nschema_constrained_decoding: false\n",
        encoding="utf-8",
    )

    results_dir = tmp_path / "results"
    monkeypatch.setattr(run_matrix, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_matrix.py", "--config", str(config_path), "--scenario", "p1", "--dry-run"],
    )

    run_matrix.main()

    transcript_path = results_dir / "p1_config_a_transcript.jsonl"
    turns = [
        json_module.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(turns) == 1
    assert turns[0]["solved"] is True
    # The raw model output travels in the transcript; it is the evidence an early run did not
    # leave behind.
    assert "dry-run stub" in turns[0]["raw_output"]


def _submit_model():
    class SubmitModel:
        def generate(self, prompt: str, *, schema_constrained: bool) -> str:
            return (
                '{"diagnosis": "d", "cause_class": "unknown", "action_type": "submit", "target": "t", '
                '"command": null, "confidence": 0.9}'
            )

    return SubmitModel()


def _env_with_verdict(verdict):
    class VerdictEnvironment:
        def task_description(self) -> str:
            return "hotel reservation rate service degraded"

        def apply_action(self, action: dict) -> tuple[bool, str]:
            return True, "Submission accepted."

        def finalize_episode(self) -> dict:
            return verdict

    return VerdictEnvironment()


def _run(tmp_path, monkeypatch, environment, config_text):
    import json as json_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    results_dir = tmp_path / "results"
    factories = {
        "test:model": _submit_model,
        "test:environment": lambda scenario_id: environment,
    }
    monkeypatch.setattr(run_matrix, "_resolve", factories.__getitem__)
    monkeypatch.setattr(run_matrix, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix.py", "--config", str(config_path), "--scenario", "p1",
            "--model-client", "test:model", "--environment-factory", "test:environment",
        ],
    )
    run_matrix.main()
    return json_module.loads((results_dir / "p1_config_x_transcript.jsonl").read_text()), json_module.loads(
        (results_dir / "p1_config_x.json").read_text(encoding="utf-8")
    ), results_dir


def test_main_persists_eval_verdict_not_only_submission(tmp_path, monkeypatch, capsys):
    _, record, results_dir = _run(
        tmp_path,
        monkeypatch,
        _env_with_verdict({"success": False, "TTM": 12.0, "steps": 1}),
        "name: config_x\nschema_constrained_decoding: false\n",
    )

    # The early-matrix case end to end: submit() accepted, eval failed. Before, the record said
    # solved=True and the verdict was thrown away.
    assert record["submitted"] is True
    assert record["success"] is False
    assert record["ttm_seconds"] == 12.0
    csv_text = (results_dir / "matrix.csv").read_text(encoding="utf-8")
    # env first, the declared confounds right behind it.
    assert csv_text.splitlines()[0].startswith("env,env_quantization,env_network,scenario_id,config_name")
    out = capsys.readouterr().out
    assert "success=False submitted=True" in out
    assert "0/1 success" in out and "1/1 submitted" in out


def test_main_stamps_memory_with_eval_verdict(tmp_path, monkeypatch):
    from harness.memory import EpisodicMemory

    store = tmp_path / "mem.jsonl"
    _run(
        tmp_path,
        monkeypatch,
        _env_with_verdict({"success": True}),
        f"name: config_x\nschema_constrained_decoding: false\nmemory:\n  enabled: true\n  store_path: {store}\n  top_k: 3\n",
    )

    episodes = EpisodicMemory.load(store).episodes
    assert len(episodes) == 1
    assert episodes[0].outcome == "Submission accepted."
    assert episodes[0].verdict == "success"


def test_main_without_verdict_records_unmeasured(tmp_path, monkeypatch):
    _, record, _ = _run(
        tmp_path, monkeypatch, _env_with_verdict({}),
        "name: config_x\nschema_constrained_decoding: false\n",
    )

    assert record["submitted"] is True
    assert record["success"] is None


TWO_ENVS_YAML = """\
name: config_x
schema_constrained_decoding: false
envs:
  mlx-local:
    backend: mlx
    quantization: 8bit
    network: local
  hyperstack-a6000:
    backend: vllm
    quantization: fp16
    network: local
"""


def _run_env(tmp_path, monkeypatch, *, env_flag=None, config_text=TWO_ENVS_YAML, scenario="p1",
             model_factory=None, results_dir=None):
    import json as json_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    results_dir = results_dir or tmp_path / "results"
    factories = {
        "test:model": model_factory or _submit_model,
        "test:environment": lambda scenario_id: _env_with_verdict({"success": True}),
    }
    monkeypatch.setattr(run_matrix, "_resolve", factories.__getitem__)
    monkeypatch.setattr(run_matrix, "RESULTS_DIR", results_dir)
    argv = [
        "run_matrix.py", "--config", str(config_path), "--scenario", scenario,
        "--model-client", "test:model", "--environment-factory", "test:environment",
    ]
    if env_flag is not None:
        argv += ["--env", env_flag]
    monkeypatch.setattr(sys, "argv", argv)
    run_matrix.main()
    record_path = results_dir / (env_flag or "") / f"{scenario}_config_x.json"
    return json_module.loads(record_path.read_text(encoding="utf-8")), results_dir


def test_env_flag_required_when_config_declares_multiple_envs(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as excinfo:
        _run_env(tmp_path, monkeypatch, env_flag=None)

    assert excinfo.value.code == 2


def test_refuses_to_start_when_selected_env_has_no_quantization(tmp_path, monkeypatch):
    config_text = (
        "name: config_x\nschema_constrained_decoding: false\n"
        "envs:\n  mlx-local:\n    backend: mlx\n    quantization: null\n"
    )
    with pytest.raises(SystemExit) as excinfo:
        _run_env(tmp_path, monkeypatch, config_text=config_text)

    assert excinfo.value.code == 2


def test_two_envs_same_cell_write_two_files_and_two_csv_rows(tmp_path, monkeypatch):
    import csv as csv_module

    _, results_dir = _run_env(tmp_path, monkeypatch, env_flag="mlx-local")
    _, results_dir = _run_env(tmp_path, monkeypatch, env_flag="hyperstack-a6000",
                              results_dir=results_dir)

    # Same (scenario, config), two envs: two files in two directories, two lines.
    assert (results_dir / "mlx-local" / "p1_config_x.json").exists()
    assert (results_dir / "hyperstack-a6000" / "p1_config_x.json").exists()
    with (results_dir / "matrix.csv").open(encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    assert [(row["env"], row["scenario_id"]) for row in rows] == [
        ("hyperstack-a6000", "p1"),
        ("mlx-local", "p1"),
    ]


def test_env_descriptor_written_once_and_matching_declaration(tmp_path, monkeypatch):
    import json as json_module

    _, results_dir = _run_env(tmp_path, monkeypatch, env_flag="mlx-local")
    descriptor_path = results_dir / "mlx-local" / "env-descriptor.json"
    descriptor = json_module.loads(descriptor_path.read_text(encoding="utf-8"))

    # Verbatim copy of the declared block, in every field.
    assert descriptor == {"backend": "mlx", "quantization": "8bit", "network": "local"}

    # Second run of the same env: an identical descriptor still holds (written once).
    before = descriptor_path.read_text(encoding="utf-8")
    _, _ = _run_env(tmp_path, monkeypatch, env_flag="mlx-local", scenario="p2",
                    results_dir=results_dir)
    assert descriptor_path.read_text(encoding="utf-8") == before


def test_env_descriptor_mismatch_refuses_the_run(tmp_path, monkeypatch):
    _, results_dir = _run_env(tmp_path, monkeypatch, env_flag="mlx-local")

    changed = TWO_ENVS_YAML.replace("quantization: 8bit", "quantization: 4bit")
    with pytest.raises(SystemExit) as excinfo:
        _run_env(tmp_path, monkeypatch, env_flag="mlx-local", config_text=changed,
                 scenario="p2", results_dir=results_dir)

    # The declaration changed mid-matrix: refuse, never overwrite the descriptor.
    assert excinfo.value.code == 2
    assert not (results_dir / "mlx-local" / "p2_config_x.json").exists()


def test_record_carries_env_confounds_and_model_stats(tmp_path, monkeypatch):
    from harness.metrics import GenerateStats

    def model_with_stats():
        model = _submit_model()
        model.stats = GenerateStats(calls=3, retries=1, seconds=2.5)
        return model

    record, _ = _run_env(tmp_path, monkeypatch, env_flag="hyperstack-a6000",
                         model_factory=model_with_stats)

    assert record["env"] == "hyperstack-a6000"
    assert record["env_quantization"] == "fp16"
    assert record["env_network"] == "local"
    assert record["model_seconds"] == 2.5
    assert record["model_calls"] == 3
    assert record["model_retries"] == 1


def test_model_unavailable_writes_unmeasured_with_model_error(tmp_path, monkeypatch):
    from adapters.vllm_model_client import ModelUnavailable

    def dead_tunnel_model():
        class DeadModel:
            def generate(self, prompt: str, *, schema_constrained: bool) -> str:
                raise ModelUnavailable("vLLM request failed after 2 attempt(s): refused")

        return DeadModel()

    record, results_dir = _run_env(tmp_path, monkeypatch, env_flag="hyperstack-a6000",
                                   model_factory=dead_tunnel_model)

    # A dead tunnel never becomes a model failure: not measured + reason, never a malformed turn.
    assert record["success"] is None
    assert record["submitted"] is False
    assert "refused" in record["model_error"]
    assert not (results_dir / "hyperstack-a6000" / "p1_config_x_transcript.jsonl").exists()


def test_memory_store_lands_inside_the_env_directory(tmp_path, monkeypatch):
    config_text = TWO_ENVS_YAML + "memory:\n  enabled: true\n  store_path: scenarios/results/episodic_memory.jsonl\n  top_k: 3\n"
    _, results_dir = _run_env(tmp_path, monkeypatch, env_flag="mlx-local",
                              config_text=config_text)

    # Episodic memory is per env: episodes from a local run do not leak into a cloud round.
    assert (results_dir / "mlx-local" / "episodic_memory.jsonl").exists()


def test_legacy_root_record_stays_in_unknown_bucket_next_to_env_run(tmp_path, monkeypatch):
    import csv as csv_module
    import json as json_module

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    legacy = {
        "scenario_id": "p0", "config_name": "config_x", "submitted": True,
        "output_tokens": 5, "elapsed_seconds": 1.0, "attempt_count": 1, "success": True,
    }
    (results_dir / "p0_config_x.json").write_text(json_module.dumps(legacy), encoding="utf-8")

    _, results_dir = _run_env(tmp_path, monkeypatch, env_flag="mlx-local",
                              results_dir=results_dir)

    with (results_dir / "matrix.csv").open(encoding="utf-8") as handle:
        rows = {row["scenario_id"]: row["env"] for row in csv_module.DictReader(handle)}
    # The legacy record renders unknown and is never folded into the named env.
    assert rows == {"p0": "unknown", "p1": "mlx-local"}


@pytest.mark.parametrize("env_value, expected", [(None, True), ("on", True), ("off", False)])
def test_agent_config_history_budget_follows_sslm_history_budget(
    tmp_path, monkeypatch, env_value, expected
):
    # Only the API comparison arm exports SSLM_HISTORY_BUDGET=off; the
    # default (unset) keeps the history budget.
    seen = []

    class CapturingAgent:
        def __init__(self, *, model, config, memory):
            seen.append(config)
            raise RuntimeError("stop after wiring")

    if env_value is None:
        monkeypatch.delenv("SSLM_HISTORY_BUDGET", raising=False)
    else:
        monkeypatch.setenv("SSLM_HISTORY_BUDGET", env_value)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: config_a\nschema_constrained_decoding: false\n", encoding="utf-8")
    factories = {"test:model": lambda: object(), "test:environment": lambda scenario_id: object()}
    monkeypatch.setattr(run_matrix, "_resolve", factories.__getitem__)
    monkeypatch.setattr(run_matrix, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_matrix, "Agent", CapturingAgent)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix.py",
            "--config",
            str(config_path),
            "--scenario",
            "p1",
            "--model-client",
            "test:model",
            "--environment-factory",
            "test:environment",
        ],
    )

    with pytest.raises(RuntimeError, match="stop after wiring"):
        run_matrix.main()

    assert seen[0].history_budget is expected
