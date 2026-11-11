"""seed_cluster.sh (round 2, N=3 seeds per round): each seed gets its own cluster, kubeconfig, ports and outputs.

The script runs for real in a sandbox copy of tools/triage-eval where setup_cluster.sh, the
driver (.venv/bin/python) and kubectl are stubs that record what reached them. The property under
test is isolation: three detached drivers on one VM must never share a kubeconfig, a
port-forward or an output file, and the agent must come up with the model on.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

PREFLIGHT_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"
SCRIPT = PREFLIGHT_DIR / "scenarios" / "seed_cluster.sh"


def _exe(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def sandbox(tmp_path: Path) -> dict:
    tree = tmp_path / "triage-eval"
    (tree / "scenarios").mkdir(parents=True)
    shutil.copy2(SCRIPT, tree / "scenarios" / "seed_cluster.sh")
    record = tmp_path / "record"
    record.mkdir()
    _exe(
        tree / "setup_cluster.sh",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            env | sort > "$RECORD/setup-$CLUSTER_NAME.env"
            """
        ),
    )
    # The driver and the harvest both go through .venv/bin/python: record argv and the env that
    # decides which cluster they talk to.
    _exe(
        tree / ".venv" / "bin" / "python",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            name="$(basename "$1")"
            {
              printf 'KUBECONFIG=%s\\n' "$KUBECONFIG"
              printf 'AIOPSLAB_CLUSTER=%s\\n' "$AIOPSLAB_CLUSTER"
              printf 'ARG=%s\\n' "$@"
            } > "$RECORD/$name.$AIOPSLAB_CLUSTER.args"
            """
        ),
    )
    bin_dir = tmp_path / "bin"
    _exe(
        bin_dir / "kubectl",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            printf '%s|%s\\n' "$KUBECONFIG" "$*" >> "$RECORD/kubectl.calls"
            # Port-forwards block like the real ones; the agent log pull returns at once.
            [[ "$*" == *" logs "* ]] && exit 0
            sleep 30
            """
        ),
    )
    kube_dir = tmp_path / "kube"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "RECORD": str(record),
        "R2_KUBECONFIG_DIR": str(kube_dir),
        "R2_RESULTS_DIR": "results/r2",
        "AIOPSLAB_HOME": str(tmp_path / "AIOpsLab"),
        "APP_NAMESPACE": "test-hotel-reservation",
        "VLLM_BASE_URL": "http://172.18.0.1:8000/v1",
        "VLLM_EGRESS_CIDR": "172.18.0.1/32",
        "VLLM_EGRESS_PORT": "8000",
        "VLLM_METRICS_URL": "http://127.0.0.1:8000/metrics",
    }
    return {"tree": tree, "record": record, "env": env, "kube_dir": kube_dir}


def _run(sb: dict, *args: str, **extra: str) -> subprocess.CompletedProcess:
    env = {**sb["env"], **extra}
    return subprocess.run(
        ["bash", str(sb["tree"] / "scenarios" / "seed_cluster.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _parse(text: str) -> dict:
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def _wait_for(path: Path, timeout: float = 10.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return path.read_text(encoding="utf-8")
        time.sleep(0.1)
    raise AssertionError(f"{path} never appeared")


def test_three_seeds_never_share_a_cluster_kubeconfig_or_port(sandbox):
    envs = [_parse(_run(sandbox, "env", str(i)).stdout) for i in (1, 2, 3)]
    for key in ("CLUSTER_NAME", "KUBECONFIG", "AIOPSLAB_CLUSTER", "ALERTMANAGER_PORT", "PROMETHEUS_PORT", "RECEIVER_PORT"):
        values = [e[key] for e in envs]
        assert len(set(values)) == 3, (key, values)
    ports = [int(e[k]) for e in envs for k in ("ALERTMANAGER_PORT", "PROMETHEUS_PORT", "RECEIVER_PORT")]
    assert len(set(ports)) == 9
    # The admission-round launcher's forwards stay free, so a leftover from it never collides
    # with a seed.
    assert not {19093, 19090, 18080} & set(ports)
    # AIOpsLab's KubeCtl reads AIOPSLAB_CLUSTER as the kind cluster name (context kind-<name>).
    for e in envs:
        assert e["AIOPSLAB_CLUSTER"] == e["CLUSTER_NAME"]
        assert Path(e["KUBECONFIG"]).name == f"{e['CLUSTER_NAME']}.yaml"


@pytest.mark.parametrize("index", ["0", "4", "a", ""])
def test_index_outside_one_to_three_is_refused(sandbox, index):
    result = _run(sandbox, "env", index)
    assert result.returncode != 0


def test_up_brings_the_agent_with_the_model_on_into_its_own_kubeconfig(sandbox):
    result = _run(sandbox, "up", "2", VLLM_API_KEY="k-secret")
    assert result.returncode == 0, result.stderr
    env = _parse((sandbox["record"] / "setup-sslm-r2-s2.env").read_text(encoding="utf-8"))
    assert env["SSLM_LOG_ONLY"] == "0"
    assert env["AGENT_INSTALL"] == "1" and env["KPS_INSTALL"] == "1" and env["CALICO_INSTALL"] == "1"
    assert env["CLUSTER_NAME"] == "sslm-r2-s2"
    assert env["KUBECONFIG"] == str(sandbox["kube_dir"] / "sslm-r2-s2.yaml")
    assert env["VLLM_BASE_URL"] == "http://172.18.0.1:8000/v1"
    assert env["VLLM_API_KEY"] == "k-secret"
    # Real C2 stays setup_cluster's own default: seed_cluster.sh does not force dry-run.
    assert "SSLM_C2_DRY_RUN" not in env


@pytest.mark.parametrize("missing", ["VLLM_BASE_URL", "VLLM_EGRESS_CIDR", "VLLM_EGRESS_PORT", "APP_NAMESPACE"])
def test_up_refuses_without_the_model_endpoint_or_namespace(sandbox, missing):
    env = dict(sandbox["env"])
    env.pop(missing)
    result = subprocess.run(
        ["bash", str(sandbox["tree"] / "scenarios" / "seed_cluster.sh"), "up", "1"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert missing in result.stderr
    assert not (sandbox["record"] / "setup-sslm-r2-s1.env").exists()


def test_launch_refuses_before_the_seed_cluster_exists(sandbox):
    result = _run(sandbox, "launch", "1")
    assert result.returncode != 0
    assert "seed_cluster.sh up 1" in result.stderr


def _kubeconfig(sb: dict, index: int) -> Path:
    path = sb["kube_dir"] / f"sslm-r2-s{index}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("apiVersion: v1\n", encoding="utf-8")
    return path


def test_r2_launch_runs_the_model_round_against_its_own_cluster_and_harvests(sandbox):
    kubeconfig = _kubeconfig(sandbox, 3)
    schedule = sandbox["tree"] / "scenarios" / "schedules" / "r2-seed-x.json"
    schedule.parent.mkdir(parents=True)
    schedule.write_text("{}", encoding="utf-8")
    result = _run(sandbox, "launch", "3", R2_MODE="r2", R2_SCHEDULE="scenarios/schedules/r2-seed-x.json")
    assert result.returncode == 0, result.stderr

    driver = _parse_args(_wait_for(sandbox["record"] / "run_continuous_admission.py.sslm-r2-s3.args"))
    assert driver["KUBECONFIG"] == str(kubeconfig)
    assert "--post-action-probe" in driver["ARGS"]
    assert _after(driver["ARGS"], "--schedule") == "scenarios/schedules/r2-seed-x.json"
    assert _after(driver["ARGS"], "--out") == "results/r2/s3-r2.json"
    assert _after(driver["ARGS"], "--alertmanager-url") == "http://127.0.0.1:20393"
    assert _after(driver["ARGS"], "--prometheus-url") == "http://127.0.0.1:20390"
    assert _after(driver["ARGS"], "--receiver-url") == "http://127.0.0.1:20380"

    harvest = _parse_args(_wait_for(sandbox["record"] / "harvest_round.py.sslm-r2-s3.args"))
    assert harvest["KUBECONFIG"] == str(kubeconfig)
    assert _after(harvest["ARGS"], "--out") == "results/r2/s3-r2.json"

    calls = _wait_for(sandbox["record"] / "kubectl.calls")
    for line in calls.splitlines():
        assert line.startswith(f"{kubeconfig}|"), line
    assert "20393:9093" in calls and "20390:9090" in calls and "20380:8080" in calls


def test_smoke_launch_has_no_harvest_and_no_schedule(sandbox):
    _kubeconfig(sandbox, 1)
    result = _run(sandbox, "launch", "1")
    assert result.returncode == 0, result.stderr
    driver = _parse_args(_wait_for(sandbox["record"] / "run_continuous_admission.py.sslm-r2-s1.args"))
    assert "--smoke" in driver["ARGS"]
    assert _after(driver["ARGS"], "--app-namespace") == "test-hotel-reservation"
    assert _after(driver["ARGS"], "--out") == "results/r2/s1-smoke.json"
    # The smoke proves the model path too: it reads vLLM's prompt-token counter.
    assert _after(driver["ARGS"], "--vllm-metrics-url") == "http://127.0.0.1:8000/metrics"
    assert "--schedule" not in driver["ARGS"]
    time.sleep(0.5)
    assert not (sandbox["record"] / "harvest_round.py.sslm-r2-s1.args").exists()


def test_smoke_launch_refuses_without_the_vllm_metrics_url(sandbox):
    # 2026-09-30: the smoke passed while every model call from the agent was refused. A smoke
    # that cannot read the token counter must not start at all.
    _kubeconfig(sandbox, 1)
    env = dict(sandbox["env"])
    env.pop("VLLM_METRICS_URL")
    result = subprocess.run(
        ["bash", str(sandbox["tree"] / "scenarios" / "seed_cluster.sh"), "launch", "1"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "VLLM_METRICS_URL" in result.stderr
    time.sleep(0.5)
    assert not (sandbox["record"] / "run_continuous_admission.py.sslm-r2-s1.args").exists()


def test_r2_launch_refuses_without_a_schedule_file(sandbox):
    _kubeconfig(sandbox, 1)
    assert _run(sandbox, "launch", "1", R2_MODE="r2").returncode != 0
    missing = _run(sandbox, "launch", "1", R2_MODE="r2", R2_SCHEDULE="scenarios/schedules/nope.json")
    assert missing.returncode != 0
    assert "does not exist" in missing.stderr


def _parse_args(text: str) -> dict:
    out: dict = {"ARGS": []}
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key == "ARG":
            out["ARGS"].append(value)
        else:
            out[key] = value
    return out


def _after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


# COMPARATOR_ARM=1: same seed, the agent calls the Anthropic API. `up` needs the key and
# the cap instead of the VLLM_* endpoint; the smoke proves the agent -> API path by the pod's usage
# log and must not need vLLM at all.


def _comparator_env(sb: dict) -> dict:
    env = {k: v for k, v in sb["env"].items() if not k.startswith("VLLM_")}
    env.update({"COMPARATOR_ARM": "1", "ANTHROPIC_API_KEY": "sk-ant-test", "SSLM_COMPARATOR_MAX_COST_USD": "5"})
    return env


def _run_env(sb: dict, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(sb["tree"] / "scenarios" / "seed_cluster.sh"), *args],
        env=env, capture_output=True, text=True, timeout=30,
    )


def test_comparator_up_needs_no_vllm_and_hands_the_key_and_cap_to_setup(sandbox):
    result = _run_env(sandbox, _comparator_env(sandbox), "up", "1")
    assert result.returncode == 0, result.stderr
    env = _parse((sandbox["record"] / "setup-sslm-r2-s1.env").read_text(encoding="utf-8"))
    assert env["COMPARATOR_ARM"] == "1"
    assert env["SSLM_LOG_ONLY"] == "0"
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert env["SSLM_COMPARATOR_MAX_COST_USD"] == "5"
    assert env["AGENT_INSTALL"] == "1" and env["KPS_INSTALL"] == "1" and env["CALICO_INSTALL"] == "1"
    assert not any(k.startswith("VLLM_") for k in env)
    assert "sk-ant-test" not in result.stdout + result.stderr


def test_comparator_up_hands_the_prompt_budget_to_setup(sandbox):
    result = _run_env(sandbox, {**_comparator_env(sandbox), "SSLM_PROMPT_BUDGET_CHARS": "14000"}, "up", "1")
    assert result.returncode == 0, result.stderr
    env = _parse((sandbox["record"] / "setup-sslm-r2-s1.env").read_text(encoding="utf-8"))
    assert env["SSLM_PROMPT_BUDGET_CHARS"] == "14000"


@pytest.mark.parametrize("missing", ["ANTHROPIC_API_KEY", "SSLM_COMPARATOR_MAX_COST_USD", "APP_NAMESPACE"])
def test_comparator_up_refuses_without_the_key_cap_or_namespace(sandbox, missing):
    env = _comparator_env(sandbox)
    env.pop(missing)
    result = _run_env(sandbox, env, "up", "1")
    assert result.returncode != 0
    assert missing in result.stderr
    assert not (sandbox["record"] / "setup-sslm-r2-s1.env").exists()


def test_comparator_smoke_reads_the_usage_log_and_needs_no_vllm_metrics(sandbox):
    _kubeconfig(sandbox, 2)
    result = _run_env(sandbox, _comparator_env(sandbox), "launch", "2")
    assert result.returncode == 0, result.stderr
    driver = _parse_args(_wait_for(sandbox["record"] / "run_continuous_admission.py.sslm-r2-s2.args"))
    assert "--smoke" in driver["ARGS"]
    assert "--comparator-usage" in driver["ARGS"]
    assert "--vllm-metrics-url" not in driver["ARGS"]
    assert _after(driver["ARGS"], "--out") == "results/r2/s2-smoke.json"


def test_comparator_r2_launch_is_the_same_round_as_the_vllm_arm(sandbox):
    _kubeconfig(sandbox, 1)
    schedule = sandbox["tree"] / "scenarios" / "schedules" / "r2-seed-x.json"
    schedule.parent.mkdir(parents=True)
    schedule.write_text("{}", encoding="utf-8")
    env = {**_comparator_env(sandbox), "R2_MODE": "r2", "R2_SCHEDULE": "scenarios/schedules/r2-seed-x.json"}
    result = _run_env(sandbox, env, "launch", "1")
    assert result.returncode == 0, result.stderr
    driver = _parse_args(_wait_for(sandbox["record"] / "run_continuous_admission.py.sslm-r2-s1.args"))
    assert "--post-action-probe" in driver["ARGS"]
    assert _after(driver["ARGS"], "--schedule") == "scenarios/schedules/r2-seed-x.json"
    assert "--comparator-usage" not in driver["ARGS"]


def test_an_unrecognized_comparator_arm_value_is_refused(sandbox):
    result = _run(sandbox, "up", "1", COMPARATOR_ARM="true")
    assert result.returncode != 0
    assert "COMPARATOR_ARM" in result.stderr


# Added 2026-10-19: the short 7B round. `up` hands the prompt budget to
# setup_cluster.sh, `launch` with R2_DRY_RUN=1 drives the committed short schedule in --dry-run.


def test_up_hands_the_prompt_budget_to_setup_cluster(sandbox):
    result = _run(sandbox, "up", "1", SSLM_PROMPT_BUDGET_CHARS="18000")
    assert result.returncode == 0, result.stderr
    env = _parse((sandbox["record"] / "setup-sslm-r2-s1.env").read_text(encoding="utf-8"))
    assert env["SSLM_PROMPT_BUDGET_CHARS"] == "18000"


def test_up_hands_the_tool_contract_to_setup_cluster(sandbox):
    result = _run(sandbox, "up", "1", SSLM_TOOL_CONTRACT="on")
    assert result.returncode == 0, result.stderr
    env = _parse((sandbox["record"] / "setup-sslm-r2-s1.env").read_text(encoding="utf-8"))
    assert env["SSLM_TOOL_CONTRACT"] == "on"


def test_up_without_a_tool_contract_passes_it_empty(sandbox):
    result = _run(sandbox, "up", "1")
    assert result.returncode == 0, result.stderr
    env = _parse((sandbox["record"] / "setup-sslm-r2-s1.env").read_text(encoding="utf-8"))
    assert env.get("SSLM_TOOL_CONTRACT", "") == ""


def test_up_without_a_budget_passes_it_empty(sandbox):
    result = _run(sandbox, "up", "1")
    assert result.returncode == 0, result.stderr
    env = _parse((sandbox["record"] / "setup-sslm-r2-s1.env").read_text(encoding="utf-8"))
    assert env.get("SSLM_PROMPT_BUDGET_CHARS", "") == ""


@pytest.mark.parametrize("dry_run, expected", [("1", True), ("0", False)])
def test_r2_dry_run_adds_the_driver_flag_and_keeps_the_r2_files(sandbox, dry_run, expected):
    _kubeconfig(sandbox, 1)
    schedule = sandbox["tree"] / "scenarios" / "schedules" / "r2short-seed-x.json"
    schedule.parent.mkdir(parents=True)
    schedule.write_text("{}", encoding="utf-8")
    result = _run(
        sandbox, "launch", "1", R2_MODE="r2", R2_DRY_RUN=dry_run, R2_SCHEDULE="scenarios/schedules/r2short-seed-x.json"
    )
    assert result.returncode == 0, result.stderr

    driver = _parse_args(_wait_for(sandbox["record"] / "run_continuous_admission.py.sslm-r2-s1.args"))
    assert ("--dry-run" in driver["ARGS"]) is expected
    assert "--post-action-probe" in driver["ARGS"]
    assert _after(driver["ARGS"], "--out") == "results/r2/s1-r2.json"
    harvest = _parse_args(_wait_for(sandbox["record"] / "harvest_round.py.sslm-r2-s1.args"))
    assert _after(harvest["ARGS"], "--schedule") == "scenarios/schedules/r2short-seed-x.json"


def test_an_unrecognized_r2_dry_run_value_is_refused(sandbox):
    result = _run(sandbox, "launch", "1", R2_MODE="r2", R2_DRY_RUN="yes")
    assert result.returncode != 0
    assert "R2_DRY_RUN" in result.stderr


# Added 2026-10-20: the 9B's model name and request dialect reach setup, and
# the agent pod's log (the only record of envelope denials) is pulled after every driver run.


def test_up_passes_the_model_and_decoding_dialect_to_setup(sandbox):
    result = _run(sandbox, "up", "1", VLLM_MODEL="Qwen3.5-9B", VLLM_DECODING_DIALECT="structured_outputs")
    assert result.returncode == 0, result.stderr
    env = _parse((sandbox["record"] / "setup-sslm-r2-s1.env").read_text(encoding="utf-8"))
    assert env["VLLM_MODEL"] == "Qwen3.5-9B"
    assert env["VLLM_DECODING_DIALECT"] == "structured_outputs"


def test_launch_pulls_the_agent_log_into_the_results_after_the_driver(sandbox):
    _kubeconfig(sandbox, 1)
    result = _run(sandbox, "launch", "1")
    assert result.returncode == 0, result.stderr
    _wait_for(sandbox["record"] / "run_continuous_admission.py.sslm-r2-s1.args")
    deadline = time.time() + 10
    calls = ""
    while time.time() < deadline and "logs deploy/sslm-agent" not in calls:
        time.sleep(0.1)
        path = sandbox["record"] / "kubectl.calls"
        calls = path.read_text(encoding="utf-8") if path.exists() else ""
    assert "-n sslm-agent logs deploy/sslm-agent --timestamps" in calls
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'AGENT_LOG="$R2_RESULTS_DIR/s${INDEX}-${R2_MODE}.agent.log"' in text


def test_smokegate_launch_is_a_dry_run_round_under_its_own_file_names(sandbox):
    """The gate round must never overwrite or join the long round."""
    schedule = sandbox["tree"] / "scenarios" / "schedules" / "smokegate-x.json"
    schedule.parent.mkdir(parents=True)
    schedule.write_text("{}", encoding="utf-8")
    _kubeconfig(sandbox, 1)
    result = _run(sandbox, "launch", "1", R2_MODE="smokegate", R2_SCHEDULE="scenarios/schedules/smokegate-x.json")
    assert result.returncode == 0, result.stderr

    driver = _parse_args(_wait_for(sandbox["record"] / "run_continuous_admission.py.sslm-r2-s1.args"))
    assert "--dry-run" in driver["ARGS"] and "--post-action-probe" in driver["ARGS"]
    assert _after(driver["ARGS"], "--schedule") == "scenarios/schedules/smokegate-x.json"
    assert _after(driver["ARGS"], "--out") == "results/r2/s1-smokegate.json"
    harvest = _parse_args(_wait_for(sandbox["record"] / "harvest_round.py.sslm-r2-s1.args"))
    assert _after(harvest["ARGS"], "--out") == "results/r2/s1-smokegate.json"


def test_smokegate_launch_refuses_without_a_schedule_file(sandbox):
    _kubeconfig(sandbox, 1)
    result = _run(sandbox, "launch", "1", R2_MODE="smokegate")
    assert result.returncode != 0 and "R2_SCHEDULE" in result.stderr
