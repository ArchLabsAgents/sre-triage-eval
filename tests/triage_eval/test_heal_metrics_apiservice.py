"""Regression for an observed failure: the AIOpsLab harness redeploys metrics-server every
episode and on this cluster it sometimes never becomes ready (MissingEndpoints). A broken
v1beta1.metrics.k8s.io APIService stalls the namespace controller's discovery step, which blocks
DELETION of any namespace while it stays broken. That breaks the current episode's finalize()
and cascades into the next episode, which tries to recreate a namespace that was never deleted.
Manual workaround, measured live: `kubectl delete apiservice v1beta1.metrics.k8s.io` unblocks
the deletion in ~15-20s.

run_full_matrix.sh now runs `heal_broken_metrics_apiservice` before EVERY episode (not just
once in setup_cluster.sh), to heal the state the previous episode left broken before it blocks
the next one. The test runs the real script in a sandbox with `kubectl` stubbed on PATH and
every matrix result already present except one target (scenario, config) combo, forcing
exactly ONE pass through the loop body: enough to see whether the heal called
`delete apiservice` or not.
"""

import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

PREFLIGHT_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"

SCENARIOS = (
    "misconfig_app_hotel_res-mitigation-1",
    "revoke_auth_mongodb-mitigation-1",
    "revoke_auth_mongodb-mitigation-2",
    "user_unregistered_mongodb-mitigation-1",
    "user_unregistered_mongodb-mitigation-2",
    "redeploy_without_PV-mitigation-1",
    "wrong_bin_usage-mitigation-1",
)
CONFIGS = ("config_a", "config_b", "config_c")

TARGET_SCENARIO = SCENARIOS[0]
TARGET_CONFIG = CONFIGS[0]

# The env is required since results are resumed per env (results/<env>/); mlx-local is the env
# with a mapped adapter that does not require VLLM_BASE_URL.
TEST_ENV = "mlx-local"

# Only the subset run_full_matrix.sh actually uses: `get apiservice ... -o jsonpath` and
# `delete apiservice ... --ignore-not-found`. State controlled by $STUB_APISERVICE_STATUS
# ("" = absent/get fails, "True" = healthy, any other string = broken).
KUBECTL_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    set -eu
    echo "$*" >> "${STUB_KUBECTL_LOG:-/dev/null}"
    args="$*"
    case "$args" in
      *"get apiservice v1beta1.metrics.k8s.io -o jsonpath"*)
        if [[ -n "${STUB_GET_FAILS:-}" ]]; then
          echo "Unable to connect to the server: timeout" >&2
          exit 1
        fi
        if [[ -z "${STUB_APISERVICE_STATUS:-}" ]]; then
          echo 'Error from server (NotFound): apiservices.apiregistration.k8s.io "v1beta1.metrics.k8s.io" not found' >&2
          exit 1
        fi
        printf '%s' "${STUB_APISERVICE_STATUS}" ;;
      *"delete apiservice v1beta1.metrics.k8s.io"*)
        if [[ -n "${STUB_DELETE_FAILS:-}" ]]; then
          exit 1
        fi ;;
      *) : ;;
    esac
    """
)


@pytest.fixture
def sandbox(tmp_path):
    dest = tmp_path / "triage-eval"
    shutil.copytree(PREFLIGHT_DIR, dest)

    setup_stub = dest / "setup_cluster.sh"
    setup_stub.write_text("#!/usr/bin/env bash\nexit 0\n")
    setup_stub.chmod(0o755)

    # Pre-populate EVERY result except the target combo, forcing exactly one pass through the
    # loop body (where the heal runs), with no kind/docker/vLLM needed.
    results_dir = dest / "scenarios" / "results" / TEST_ENV
    results_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        for config in CONFIGS:
            if scenario == TARGET_SCENARIO and config == TARGET_CONFIG:
                continue
            (results_dir / f"{scenario}_{config}.json").write_text("{}")

    aiopslab_home = tmp_path / "AIOpsLab"
    (aiopslab_home / "aiopslab").mkdir(parents=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl_stub = bin_dir / "kubectl"
    kubectl_stub.write_text(KUBECTL_STUB)
    kubectl_stub.chmod(kubectl_stub.stat().st_mode | stat.S_IEXEC)

    return dest, aiopslab_home, bin_dir


def _run(dest, aiopslab_home, bin_dir, extra_env):
    env = {**os.environ, "AIOPSLAB_HOME": str(aiopslab_home), "SSLM_ENV": TEST_ENV}
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(dest / "scenarios" / "run_full_matrix.sh")],
        env=env,
        capture_output=True,
        text=True,
    )


def test_heals_apiservice_left_broken_by_previous_episode(sandbox, tmp_path):
    dest, aiopslab_home, bin_dir = sandbox
    log = tmp_path / "kubectl.log"
    result = _run(
        dest,
        aiopslab_home,
        bin_dir,
        {"STUB_KUBECTL_LOG": str(log), "STUB_APISERVICE_STATUS": "False"},
    )

    output = result.stdout + result.stderr
    assert "[heal]" in output
    assert "delete apiservice v1beta1.metrics.k8s.io --ignore-not-found" in log.read_text()
    # the heal runs BEFORE the episode, not after
    heal_idx = output.index("[heal]")
    run_idx = output.index(f"[run ] {TARGET_SCENARIO} / {TARGET_CONFIG}")
    assert heal_idx < run_idx


def test_logs_when_get_fails_for_reason_other_than_not_found(sandbox, tmp_path):
    # A get that swallowed a real error (API server silent, RBAC) looked identical to
    # "APIService does not exist yet". A non-NotFound error must show up in the log.
    dest, aiopslab_home, bin_dir = sandbox
    log = tmp_path / "kubectl.log"
    result = _run(
        dest,
        aiopslab_home,
        bin_dir,
        {"STUB_KUBECTL_LOG": str(log), "STUB_GET_FAILS": "1"},
    )

    output = result.stdout + result.stderr
    assert "[heal] kubectl get apiservice failed" in output
    assert "delete apiservice" not in log.read_text()


def test_logs_when_delete_itself_fails(sandbox, tmp_path):
    # `|| true` swallowed a failed delete (RBAC, API server down), so a heal that did not heal
    # looked identical to a successful one in matrix-run.log.
    dest, aiopslab_home, bin_dir = sandbox
    log = tmp_path / "kubectl.log"
    result = _run(
        dest,
        aiopslab_home,
        bin_dir,
        {
            "STUB_KUBECTL_LOG": str(log),
            "STUB_APISERVICE_STATUS": "False",
            "STUB_DELETE_FAILS": "1",
        },
    )

    output = result.stdout + result.stderr
    assert "[heal] failed to remove" in output


def test_no_op_when_apiservice_is_healthy(sandbox, tmp_path):
    dest, aiopslab_home, bin_dir = sandbox
    log = tmp_path / "kubectl.log"
    result = _run(
        dest,
        aiopslab_home,
        bin_dir,
        {"STUB_KUBECTL_LOG": str(log), "STUB_APISERVICE_STATUS": "True"},
    )

    output = result.stdout + result.stderr
    assert "[heal]" not in output
    assert "delete apiservice" not in log.read_text()


def test_no_op_when_apiservice_does_not_exist(sandbox, tmp_path):
    dest, aiopslab_home, bin_dir = sandbox
    log = tmp_path / "kubectl.log"
    result = _run(
        dest,
        aiopslab_home,
        bin_dir,
        {"STUB_KUBECTL_LOG": str(log), "STUB_APISERVICE_STATUS": ""},
    )

    output = result.stdout + result.stderr
    assert "[heal]" not in output
    assert "delete apiservice" not in log.read_text()
