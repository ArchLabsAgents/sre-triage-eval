"""Tests for scenarios/backfill_env.py: the explicit stamping of legacy (pre-env) records with
env=mlx-local. The loader never does this implicitly; this
script is the decision made runnable, so its guardrails (dry-run default, never restamp, all
companions move together) are what these tests pin down."""

from __future__ import annotations

import csv
import json
import sys

import pytest

from scenarios import backfill_env

ENV_BLOCK = {
    "backend": "mlx",
    "quantization": "8bit",
    "weights_ref": "mlx-community/Qwen2.5-7B-Instruct-8bit",
    "network": "local",
}


@pytest.fixture
def legacy_results(tmp_path):
    """A results/ dir in the legacy layout: records at the root (one in the oldest format, with
    `solved`), a transcript and the episodic memory next to them."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "s1_config_a.json").write_text(
        json.dumps(
            {
                "scenario_id": "s1", "config_name": "config_a", "submitted": True,
                "output_tokens": 10, "elapsed_seconds": 1.0, "attempt_count": 1,
                "success": True,
            }
        ),
        encoding="utf-8",
    )
    (results / "s2_config_b.json").write_text(
        json.dumps(
            {
                # Oldest record format: `solved` instead of `submitted`.
                "scenario_id": "s2", "config_name": "config_b", "solved": True,
                "output_tokens": 5, "elapsed_seconds": 2.0, "attempt_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (results / "s1_config_a_transcript.jsonl").write_text("{}\n", encoding="utf-8")
    (results / "episodic_memory.jsonl").write_text("{}\n", encoding="utf-8")

    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        "model:\n  name: Qwen2.5-7B-Instruct\n"
        "envs:\n  mlx-local:\n"
        "    backend: mlx\n    quantization: 8bit\n"
        "    weights_ref: mlx-community/Qwen2.5-7B-Instruct-8bit\n"
        "    network: local\n",
        encoding="utf-8",
    )
    return results, base_config


def _main(monkeypatch, results, base_config, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_env.py", "--env", "mlx-local",
            "--results-dir", str(results), "--base-config", str(base_config), *extra,
        ],
    )
    backfill_env.main()


def test_dry_run_is_the_default_and_changes_nothing(legacy_results, monkeypatch, capsys):
    results, base_config = legacy_results

    _main(monkeypatch, results, base_config)

    out = capsys.readouterr().out
    assert "[dry-run]" in out and "nothing changed" in out
    assert (results / "s1_config_a.json").exists()
    assert not (results / "mlx-local").exists()


def test_apply_stamps_moves_and_rebuilds_the_matrix(legacy_results, monkeypatch):
    results, base_config = legacy_results

    _main(monkeypatch, results, base_config, "--apply")

    # Records stamped and moved to the env directory; nothing is left at the root.
    stamped = json.loads((results / "mlx-local" / "s1_config_a.json").read_text(encoding="utf-8"))
    assert stamped["env"] == "mlx-local"
    assert stamped["env_quantization"] == "8bit"
    assert stamped["env_network"] == "local"
    oldest_format = json.loads((results / "mlx-local" / "s2_config_b.json").read_text(encoding="utf-8"))
    assert oldest_format["env"] == "mlx-local"
    assert not (results / "s1_config_a.json").exists()

    # Companion files move together.
    assert (results / "mlx-local" / "s1_config_a_transcript.jsonl").exists()
    assert (results / "mlx-local" / "episodic_memory.jsonl").exists()

    # Descriptor written from the declaration.
    descriptor = json.loads(
        (results / "mlx-local" / "env-descriptor.json").read_text(encoding="utf-8")
    )
    assert descriptor == ENV_BLOCK

    # matrix.csv rebuilt with the env column already filled.
    with (results / "matrix.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["env"] for row in rows} == {"mlx-local"}
    assert len(rows) == 2


def test_refuses_to_restamp_a_record_from_another_env(legacy_results, monkeypatch, capsys):
    results, base_config = legacy_results
    (results / "s3_config_a.json").write_text(
        json.dumps(
            {
                "scenario_id": "s3", "config_name": "config_a", "submitted": True,
                "output_tokens": 1, "elapsed_seconds": 1.0, "attempt_count": 1,
                "env": "hyperstack-a6000",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        _main(monkeypatch, results, base_config, "--apply")

    # Aborts BEFORE any change: not even the records without env were moved.
    assert excinfo.value.code == 1
    assert "never restamps" in capsys.readouterr().err
    assert (results / "s1_config_a.json").exists()
    assert not (results / "mlx-local").exists()


def test_refuses_an_env_the_config_does_not_declare(legacy_results, monkeypatch):
    results, base_config = legacy_results
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_env.py", "--env", "scaleway-l4",
            "--results-dir", str(results), "--base-config", str(base_config), "--apply",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        backfill_env.main()

    assert excinfo.value.code == 2
    assert (results / "s1_config_a.json").exists()


def test_records_already_inside_env_dirs_are_left_alone(legacy_results, monkeypatch):
    results, base_config = legacy_results
    other = results / "hyperstack-a6000"
    other.mkdir()
    cloud_record = {
        "scenario_id": "s1", "config_name": "config_a", "submitted": True,
        "output_tokens": 1, "elapsed_seconds": 1.0, "attempt_count": 1,
        "env": "hyperstack-a6000",
    }
    (other / "s1_config_a.json").write_text(json.dumps(cloud_record), encoding="utf-8")

    _main(monkeypatch, results, base_config, "--apply")

    # The other env's record stays where it is and stays in the CSV with its own env.
    assert json.loads((other / "s1_config_a.json").read_text(encoding="utf-8")) == cloud_record
    with (results / "matrix.csv").open(encoding="utf-8") as handle:
        envs = sorted(row["env"] for row in csv.DictReader(handle))
    assert envs == ["hyperstack-a6000", "mlx-local", "mlx-local"]
