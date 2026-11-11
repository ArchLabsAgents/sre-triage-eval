import os
import shutil
import subprocess
from pathlib import Path

import pytest

PREFLIGHT_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"
RUN_FULL_MATRIX = PREFLIGHT_DIR / "scenarios" / "run_full_matrix.sh"


@pytest.fixture
def sandbox(tmp_path):
    """Copies tools/triage-eval to a tmp dir, with setup_cluster.sh replaced by a
    controllable stub. Avoids depending on real kind/docker and avoids writing
    scenarios/results/ into the real tree."""
    dest = tmp_path / "triage-eval"
    shutil.copytree(PREFLIGHT_DIR, dest)

    aiopslab_home = tmp_path / "AIOpsLab"
    (aiopslab_home / "aiopslab").mkdir(parents=True)

    return dest, aiopslab_home


def _write_stub(dest: Path, exit_code: int, marker: Path) -> None:
    stub = dest / "setup_cluster.sh"
    stub.write_text(
        f"#!/usr/bin/env bash\ntouch {marker}\nexit {exit_code}\n"
    )
    stub.chmod(0o755)


def test_aborts_matrix_when_setup_cluster_fails(sandbox):
    dest, aiopslab_home = sandbox
    marker = dest / "setup-was-called"
    _write_stub(dest, exit_code=1, marker=marker)

    result = subprocess.run(
        ["bash", str(dest / "scenarios" / "run_full_matrix.sh")],
        env={**os.environ, "AIOPSLAB_HOME": str(aiopslab_home), "SSLM_ENV": "mlx-local"},
        capture_output=True,
        text=True,
    )

    assert marker.exists(), "setup_cluster.sh deveria ter sido chamado"
    assert result.returncode == 1
    assert "setup_cluster.sh failed" in result.stdout + result.stderr
    assert "[run ]" not in result.stdout + result.stderr


def test_runs_setup_cluster_before_the_scenario_loop(sandbox):
    dest, aiopslab_home = sandbox
    marker = dest / "setup-was-called"
    _write_stub(dest, exit_code=0, marker=marker)

    result = subprocess.run(
        ["bash", str(dest / "scenarios" / "run_full_matrix.sh")],
        env={**os.environ, "AIOPSLAB_HOME": str(aiopslab_home), "SSLM_ENV": "mlx-local"},
        capture_output=True,
        text=True,
    )

    assert marker.exists()
    output = result.stdout + result.stderr
    setup_idx = output.index("ensuring cluster")
    run_idx = output.index("[run ]")
    assert setup_idx < run_idx
    # The 7th scenario (wrong_bin_usage) is really in the loop, not only in the comment.
    assert "wrong_bin_usage-mitigation-1" in output


def test_refuses_to_start_without_an_env(sandbox):
    dest, aiopslab_home = sandbox
    marker = dest / "setup-was-called"
    _write_stub(dest, exit_code=0, marker=marker)

    env = {**os.environ, "AIOPSLAB_HOME": str(aiopslab_home)}
    env.pop("SSLM_ENV", None)
    result = subprocess.run(
        ["bash", str(dest / "scenarios" / "run_full_matrix.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    # Without a declared env there is no safe resume: refuse BEFORE bringing the cluster up.
    assert result.returncode == 1
    assert "set the env" in result.stdout + result.stderr
    assert not marker.exists()


def test_refuses_an_env_with_no_adapter_mapping(sandbox):
    dest, aiopslab_home = sandbox
    marker = dest / "setup-was-called"
    _write_stub(dest, exit_code=0, marker=marker)

    result = subprocess.run(
        ["bash", str(dest / "scenarios" / "run_full_matrix.sh"), "--env", "some-unmapped-env"],
        env={**os.environ, "AIOPSLAB_HOME": str(aiopslab_home)},
        capture_output=True,
        text=True,
    )

    # Without a mapped adapter there is no way to know which ModelClient/Environment to build:
    # refuse BEFORE bringing the cluster up, same stance as "no declared env".
    assert result.returncode == 1
    assert "no adapter mapped" in result.stdout + result.stderr
    assert not marker.exists()


def test_selects_cloud_adapter_for_the_hyperstack_env(sandbox):
    dest, aiopslab_home = sandbox
    marker = dest / "setup-was-called"
    _write_stub(dest, exit_code=1, marker=marker)  # fails right after setup, before the loop

    result = subprocess.run(
        ["bash", str(dest / "scenarios" / "run_full_matrix.sh"), "--env", "hyperstack-a6000"],
        env={**os.environ, "AIOPSLAB_HOME": str(aiopslab_home)},
        capture_output=True,
        text=True,
    )

    assert marker.exists(), "setup_cluster.sh was called, so the ADAPTER case resolved"
    assert result.returncode == 1


def test_resume_skip_is_keyed_per_env(sandbox):
    dest, aiopslab_home = sandbox
    marker = dest / "setup-was-called"
    _write_stub(dest, exit_code=0, marker=marker)

    # An existing result for the SAME (scenario, config) in ANOTHER env (legacy root + another
    # env dir): neither may cause a skip of the mlx-local run.
    results = dest / "scenarios" / "results"
    (results / "hyperstack-a6000").mkdir(parents=True)
    (results / "misconfig_app_hotel_res-mitigation-1_config_a.json").write_text("{}")
    (results / "hyperstack-a6000" / "revoke_auth_mongodb-mitigation-1_config_a.json").write_text("{}")
    # An existing result IN the run's env: this one is a resume.
    (results / "mlx-local").mkdir(parents=True)
    (results / "mlx-local" / "user_unregistered_mongodb-mitigation-1_config_a.json").write_text("{}")

    result = subprocess.run(
        ["bash", str(dest / "scenarios" / "run_full_matrix.sh"), "--env", "mlx-local"],
        env={**os.environ, "AIOPSLAB_HOME": str(aiopslab_home)},
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert "[skip] user_unregistered_mongodb-mitigation-1 / config_a (result already exists in mlx-local/)" in output
    # The legacy root result and the other env's result do not skip this env's run.
    assert "[skip] misconfig_app_hotel_res-mitigation-1 / config_a" not in output
    assert "[skip] revoke_auth_mongodb-mitigation-1 / config_a" not in output
