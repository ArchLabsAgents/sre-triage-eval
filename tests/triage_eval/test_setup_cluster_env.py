"""setup_cluster.sh regressions, starting with: the lease defaults reach the python subshell.

The bug (first standalone run of the script): DURATIONS_JSON was built by reading os.environ
BEFORE `export LEASE_DURATION RENEW_DEADLINE RETRY_PERIOD`, so the defaults set in the script
itself (not exported by the caller) never reached python and the script died with KeyError
under `set -e`. Until then the script had only been called with the vars exported, so that
path had never run.

The test runs the real script in a sandbox with `kind` and `docker` stubbed on PATH (python3 is
the real one), WITHOUT exporting the lease variables, and checks that the manifest written back
carries the defaults.
"""

import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

PREFLIGHT_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"

MANIFEST_FIXTURE = textwrap.dedent(
    """\
    apiVersion: v1
    kind: Pod
    spec:
      containers:
      - command:
        - kube-controller-manager
        - --leader-elect=true
    """
)

KIND_STUB = "#!/usr/bin/env bash\necho kind\n"

# The four uses setup_cluster.sh makes: `docker ps --filter ... --format`,
# `docker exec <c> cat <manifest>`, `docker exec -i <c> sh -c "cat > <manifest>"` and
# `docker network inspect kind -f '{{range .IPAM.Config}}{{println .Gateway}}{{end}}'`
# (SSLM_LOG_ONLY=1's VLLM_EGRESS_CIDR default). The state (manifests) lives in $STUB_STATE,
# seeded by the test.
DOCKER_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    set -eu
    cmd="$1"; shift
    case "$cmd" in
      ps) echo "kind-control-plane" ;;
      exec)
        if [[ "$1" == "-i" ]]; then
          shift 2
          target="${3##*> }"
          cat > "$STUB_STATE/$(basename "$target")"
        else
          shift 1
          cat "$STUB_STATE/$(basename "$2")"
        fi
        ;;
      network)
        # A real dual-stack "kind" docker network (IPv4+IPv6, seen on Apple Silicon hosts) can
        # order its IPAM.Config entries with IPv6 first. This stub defaults to exactly that
        # shape (IPv6 gateway printed before the IPv4 one), so every
        # test in this file that hits the default path exercises setup_cluster.sh's own
        # order-independent pick, never a lucky single-entry response.
        # STUB_DOCKER_GATEWAY="" simulates `docker network inspect kind` itself failing/returning
        # nothing (no network, permission issue, etc).
        # STUB_DOCKER_GATEWAY_IPV6_ONLY=1 simulates a network with an IPv6 entry but no IPv4
        # entry at all: the "no IPv4 anywhere" fail-loud case.
        # STUB_DOCKER_GATEWAY_NO_IPV6=1 simulates the common case: a single-stack,
        # IPv4-only "kind" network with no IPv6 entry to skip over at all.
        if [[ "${STUB_DOCKER_GATEWAY-172.19.0.1}" == "" ]]; then
          printf ''
        elif [[ "${STUB_DOCKER_GATEWAY_IPV6_ONLY:-0}" == "1" ]]; then
          printf 'fc00::1\n'
        elif [[ "${STUB_DOCKER_GATEWAY_NO_IPV6:-0}" == "1" ]]; then
          printf '%s\n' "${STUB_DOCKER_GATEWAY-172.19.0.1}"
        else
          printf 'fc00::1\n%s\n' "${STUB_DOCKER_GATEWAY-172.19.0.1}"
        fi
        ;;
      *) echo "docker stub: unexpected command: $cmd" >&2; exit 1 ;;
    esac
    """
)


# kubectl stub for the openebs provisioner block. Logs every call to $STUB_KUBECTL_LOG;
# `get deploy` answers per $STUB_LE_PRESENT (does it exist?) and $STUB_LE_VALUE (current
# LEADER_ELECTION_ENABLED value via jsonpath); `get pods` answers $STUB_INITPVC (raw lines).
# Default: deploy absent (the branch is a no-op).
KUBECTL_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    set -eu
    echo "$*" >> "${STUB_KUBECTL_LOG:-/dev/null}"
    args="$*"
    case "$args" in
      *"get deploy openebs-localpv-provisioner -o jsonpath"*)
        [[ "${STUB_LE_PRESENT:-0}" == "1" ]] || exit 1
        printf '%s' "${STUB_LE_VALUE:-}" ;;
      *"get deploy openebs-localpv-provisioner"*)
        [[ "${STUB_LE_PRESENT:-0}" == "1" ]] || exit 1 ;;
      *"get pods"*) printf '%s' "${STUB_INITPVC:-}" ;;
      *"get nodes -l role=sslm-agent --no-headers"*)
        if [[ "${STUB_AGENT_NODE_PRESENT:-0}" == "1" ]]; then
          printf 'kind-worker2   Ready   <none>   1m   v1.29.0\\n'
        fi ;;
      *"kube-system get daemonset calico-node"*)
        [[ "${STUB_CALICO_PRESENT:-0}" == "1" ]] || exit 1 ;;
      *"kube-system rollout status daemonset/calico-node"*)
        : ;;
      *"get nodes -o jsonpath="*"containerRuntimeVersion"*)
        printf '%s' "${STUB_CONTAINER_RUNTIME-containerd://1.7.18}" ;;
      *"get endpoints kubernetes -o jsonpath="*"addresses[0].ip"*)
        printf '%s' "${STUB_APISERVER_IP-198.51.100.9}" ;;
      *"get endpoints kubernetes -o jsonpath="*"ports[0].port"*)
        printf '%s' "${STUB_APISERVER_PORT-6443}" ;;
      *"create namespace "*"--dry-run=client -o yaml"*)
        namespace="$(printf '%s' "$args" | awk '{print $3}')"
        printf 'apiVersion: v1\nkind: Namespace\nmetadata:\n  name: %s\n' "$namespace" ;;
      *"create secret generic sslm-agent-anthropic"*)
        # COMPARATOR_ARM=1: the key arrives on stdin (never argv, which this stub logs above).
        cat > "${STUB_SECRET_STDIN_LOG:-/dev/null}" ;;
      *"apply -f -"*)
        # Deployment/Role/RoleBinding/NetworkPolicy are applied via `envsubst ... | kubectl apply
        # -f -`: consume stdin always (an unread pipe from a real envsubst would otherwise hang
        # the test), and additionally log it verbatim when a test wants to assert on the
        # substituted content (e.g. that PROMETHEUS_URL actually reached deployment.yaml).
        if [[ -n "${STUB_KUBECTL_APPLY_STDIN_LOG:-}" ]]; then
          cat >> "$STUB_KUBECTL_APPLY_STDIN_LOG"
          printf '\\n---stdin-boundary---\\n' >> "$STUB_KUBECTL_APPLY_STDIN_LOG"
        else
          cat >/dev/null
        fi ;;
      *) : ;;
    esac
    """
)


def _write_stub(path, body):
    """Write an executable PATH stub whose shebang is guaranteed to be on line 1, column 0.

    The textwrap.dedent stubs above embed `\\n` inside non-raw strings, which puts real
    newlines at column 0 and makes dedent a no-op, leaving an indented `#!`. The kernel then
    refuses the exec (ENOEXEC) and whether bash's fallback saves it varies run to run, so
    `docker ps` came back empty and the suite flaked. The stub never reaches the real docker;
    it just has to be a valid executable.
    """
    path.write_text(body.lstrip())
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def sandbox(tmp_path):
    dest = tmp_path / "triage-eval"
    dest.mkdir()
    for name in ("setup_cluster.sh", "patch_leader_election.py"):
        shutil.copy(PREFLIGHT_DIR / name, dest / name)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("kind", KIND_STUB), ("docker", DOCKER_STUB), ("kubectl", KUBECTL_STUB)):
        stub = bin_dir / name
        _write_stub(stub, body)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    state = tmp_path / "manifests"
    state.mkdir()
    for manifest in ("kube-controller-manager.yaml", "kube-scheduler.yaml"):
        (state / manifest).write_text(MANIFEST_FIXTURE)

    return dest, bin_dir, state


def test_defaults_reach_the_python_subshell_without_caller_export(sandbox):
    dest, bin_dir, state = sandbox
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LEASE_DURATION", "RENEW_DEADLINE", "RETRY_PERIOD")
    }
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_STATE"] = str(state)

    result = subprocess.run(
        ["bash", str(dest / "setup_cluster.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    written = (state / "kube-controller-manager.yaml").read_text()
    assert "--leader-elect-lease-duration=90s" in written
    assert "--leader-elect-renew-deadline=60s" in written
    assert "--leader-elect-retry-period=10s" in written


def _run(dest, bin_dir, state, extra_env):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LEASE_DURATION", "RENEW_DEADLINE", "RETRY_PERIOD")
    }
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_STATE"] = str(state)
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(dest / "setup_cluster.sh")],
        env=env,
        capture_output=True,
        text=True,
    )


def test_openebs_no_op_when_provisioner_absent(sandbox, tmp_path):
    dest, bin_dir, state = sandbox
    log = tmp_path / "kubectl.log"
    result = _run(dest, bin_dir, state, {"STUB_KUBECTL_LOG": str(log), "STUB_LE_PRESENT": "0"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "does not exist yet" in result.stdout
    # does not touch a provisioner that does not exist
    assert "set env" not in log.read_text()


def test_openebs_disables_leader_election_when_present(sandbox, tmp_path):
    dest, bin_dir, state = sandbox
    log = tmp_path / "kubectl.log"
    result = _run(
        dest,
        bin_dir,
        state,
        {
            "STUB_KUBECTL_LOG": str(log),
            "STUB_LE_PRESENT": "1",
            "STUB_LE_VALUE": "",  # leader election still on (env absent)
            "STUB_INITPVC": "init-pvc-aaa 0/1 Completed 0 5m\n",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = log.read_text()
    assert "set env deploy/openebs-localpv-provisioner LEADER_ELECTION_ENABLED=false" in calls
    # orfaos Completed sao varridos
    assert "delete pod init-pvc-aaa" in calls


def test_openebs_idempotent_when_already_disabled(sandbox, tmp_path):
    dest, bin_dir, state = sandbox
    log = tmp_path / "kubectl.log"
    result = _run(
        dest,
        bin_dir,
        state,
        {"STUB_KUBECTL_LOG": str(log), "STUB_LE_PRESENT": "1", "STUB_LE_VALUE": "false"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "already disabled" in result.stdout
    assert "set env" not in log.read_text()


# kube-prometheus-stack section. Opt-in by design: `helm upgrade --install`
# talks to the apiserver via client-go straight from the active kubeconfig, NOT through the
# `kubectl` binary, so stubbing `kubectl` in PATH (as every test above already does for the
# openebs block) does not shield this section from a real cluster. These tests exercise the
# KPS_INSTALL gate itself, never letting the real `helm` on this machine's PATH get invoked.

HELM_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    echo "$*" >> "${STUB_HELM_LOG:-/dev/null}"
    exit 0
    """
)


def test_kps_install_default_off_never_touches_helm(sandbox, tmp_path):
    """KPS_INSTALL unset must be a no-op: no helm invocation, regardless of what's on PATH."""
    dest, bin_dir, state = sandbox
    result = _run(dest, bin_dir, state, {})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "KPS_INSTALL != 1" in result.stdout
    assert "kube-prometheus-stack: upgrade" not in result.stdout


def test_kps_install_1_without_helm_on_path_fails_loud(sandbox, tmp_path):
    """KPS_INSTALL=1 with no helm anywhere reachable must exit non-zero with a clear message,
    never silently skip (that would look like a successful install that never happened)."""
    dest, bin_dir, state = sandbox

    # bash and helm both live in /opt/homebrew/bin on this machine (symlinks into the Cellar),
    # so simply dropping that dir from PATH would also drop the modern bash the script needs
    # (mapfile, used earlier in the script, is not in macOS's system /bin/bash 3.2). Instead,
    # symlink ONLY the real bash binary into its own dir -- a PATH built from that dir plus the
    # stub dir plus /usr/bin:/bin has a working bash and no way to reach a real helm.
    bash_only_dir = tmp_path / "bash-only"
    bash_only_dir.mkdir()
    real_bash = shutil.which("bash")
    (bash_only_dir / "bash").symlink_to(real_bash)

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LEASE_DURATION", "RENEW_DEADLINE", "RETRY_PERIOD")
    }
    env["PATH"] = f"{bin_dir}:{bash_only_dir}:/usr/bin:/bin"
    env["STUB_STATE"] = str(state)
    env["KPS_INSTALL"] = "1"

    result = subprocess.run(
        ["bash", str(dest / "setup_cluster.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "helm" in (result.stdout + result.stderr).lower()


def test_kps_install_1_with_helm_stub_calls_upgrade_install_pinned_and_applies_rules(sandbox, tmp_path):
    dest, bin_dir, state = sandbox
    (dest / "kube-prometheus-stack" / "rules").mkdir(parents=True)
    (dest / "kube-prometheus-stack" / "values.yaml").write_text("grafana:\n  enabled: false\n")
    (dest / "kube-prometheus-stack" / "rules" / "symptom-alerts.yaml").write_text(
        "apiVersion: monitoring.coreos.com/v1\nkind: PrometheusRule\n"
    )
    helm_stub = bin_dir / "helm"
    _write_stub(helm_stub, HELM_STUB)
    helm_stub.chmod(helm_stub.stat().st_mode | stat.S_IEXEC)

    helm_log = tmp_path / "helm.log"
    kubectl_log = tmp_path / "kubectl.log"
    result = _run(
        dest,
        bin_dir,
        state,
        {
            "KPS_INSTALL": "1",
            "STUB_HELM_LOG": str(helm_log),
            "STUB_KUBECTL_LOG": str(kubectl_log),
            "STUB_LE_PRESENT": "0",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    helm_calls = helm_log.read_text()
    assert "upgrade --install sslm-kube-prometheus prometheus-community/kube-prometheus-stack" in helm_calls
    assert "--version 91.4.1" in helm_calls, "chart version must stay pinned, never --version latest"
    assert "--namespace monitoring" in helm_calls
    kubectl_calls = kubectl_log.read_text()
    assert "apply -n monitoring -f" in kubectl_calls
    assert "symptom-alerts.yaml" in kubectl_calls


# Chaos Mesh section (needed by the container_kill candidate family). Same shielding as the KPS tests
# above: helm is always a stub, so no real chart is ever installed by the suite.


def _chaos_mesh_run(sandbox, tmp_path, extra_env):
    dest, bin_dir, state = sandbox
    helm_stub = bin_dir / "helm"
    _write_stub(helm_stub, HELM_STUB)
    helm_stub.chmod(helm_stub.stat().st_mode | stat.S_IEXEC)
    helm_log = tmp_path / "helm.log"
    kubectl_log = tmp_path / "kubectl.log"
    result = _run(
        dest,
        bin_dir,
        state,
        {"STUB_HELM_LOG": str(helm_log), "STUB_KUBECTL_LOG": str(kubectl_log), "STUB_LE_PRESENT": "0", **extra_env},
    )
    helm_calls = helm_log.read_text() if helm_log.exists() else ""
    kubectl_calls = kubectl_log.read_text() if kubectl_log.exists() else ""
    return result, helm_calls, kubectl_calls


def test_chaos_mesh_install_default_off_never_touches_helm(sandbox, tmp_path):
    result, helm_calls, _kubectl = _chaos_mesh_run(sandbox, tmp_path, {})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHAOS_MESH_INSTALL != 1" in result.stdout
    assert "chaos-mesh" not in helm_calls


def test_chaos_mesh_install_1_pins_aiopslab_release_and_waits_for_the_controller(sandbox, tmp_path):
    """Release, namespace and chart version must be AIOpsLab's own (inject_symp.py): anything else
    lets AIOpsLab's per-problem `helm install` add a second Chaos Mesh or measure another version."""
    result, helm_calls, kubectl_calls = _chaos_mesh_run(sandbox, tmp_path, {"CHAOS_MESH_INSTALL": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "upgrade --install chaos-mesh chaos-mesh/chaos-mesh" in helm_calls
    assert "--version 2.6.2" in helm_calls
    assert "--namespace chaos-mesh" in helm_calls
    assert "--wait" in helm_calls
    assert "--set chaosDaemon.runtime=containerd" in helm_calls
    assert "--set chaosDaemon.socketPath=/run/containerd/containerd.sock" in helm_calls
    assert "-n chaos-mesh rollout status deploy/chaos-controller-manager" in kubectl_calls
    assert "-n chaos-mesh rollout status daemonset/chaos-daemon" in kubectl_calls


def test_chaos_mesh_install_1_version_is_not_overridable(sandbox, tmp_path):
    _result, helm_calls, _kubectl = _chaos_mesh_run(
        sandbox, tmp_path, {"CHAOS_MESH_INSTALL": "1", "CHAOS_MESH_CHART_VERSION": "2.7.0"}
    )
    assert "--version 2.6.2" in helm_calls and "2.7.0" not in helm_calls


def test_chaos_mesh_install_1_on_docker_runtime_passes_no_containerd_flags(sandbox, tmp_path):
    result, helm_calls, _kubectl = _chaos_mesh_run(
        sandbox, tmp_path, {"CHAOS_MESH_INSTALL": "1", "STUB_CONTAINER_RUNTIME": "docker://24.0.7"}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "upgrade --install chaos-mesh" in helm_calls
    assert "chaosDaemon" not in helm_calls


@pytest.mark.parametrize("runtime", ["", "cri-o://1.29.1"])
def test_chaos_mesh_install_1_refuses_an_unknown_runtime_before_helm(sandbox, tmp_path, runtime):
    result, helm_calls, _kubectl = _chaos_mesh_run(
        sandbox, tmp_path, {"CHAOS_MESH_INSTALL": "1", "STUB_CONTAINER_RUNTIME": runtime}
    )
    assert result.returncode != 0
    assert "runtime" in result.stderr
    assert "upgrade --install chaos-mesh" not in helm_calls


def test_chaos_mesh_install_1_without_helm_fails_loud(sandbox, tmp_path):
    dest, bin_dir, state = sandbox
    bash_only_dir = tmp_path / "bash-only"
    bash_only_dir.mkdir()
    (bash_only_dir / "bash").symlink_to(shutil.which("bash"))
    env = {k: v for k, v in os.environ.items() if k not in ("LEASE_DURATION", "RENEW_DEADLINE", "RETRY_PERIOD")}
    env.update({"PATH": f"{bin_dir}:{bash_only_dir}:/usr/bin:/bin", "STUB_STATE": str(state), "CHAOS_MESH_INSTALL": "1"})
    result = subprocess.run(["bash", str(dest / "setup_cluster.sh")], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "CHAOS_MESH_INSTALL=1 but 'helm'" in result.stderr


def test_chaos_mesh_is_installed_after_calico_converges():
    text = PREFLIGHT_DIR.joinpath("setup_cluster.sh").read_text()
    assert text.index("rollout status daemonset/calico-node") < text.index(
        'helm upgrade --install "$CHAOS_MESH_RELEASE"'
    )


def test_calico_converges_before_kps_helm_wait():
    text = PREFLIGHT_DIR.joinpath("setup_cluster.sh").read_text()
    assert text.index("rollout status daemonset/calico-node") < text.index(
        "helm upgrade --install \"$KPS_RELEASE\""
    )


# Agent manifests section. AGENT_INSTALL=1's own preflight: the dedicated node (role=sslm-agent) only exists on a cluster that was
# CREATED with AGENT_INSTALL=1 (the kind-config patch block, tested separately in
# tests/triage_eval/test_patch_kind_config_agent_node.py). Running the manifest-apply block
# against a cluster that already existed without that node must fail loud and early, never with
# the opaque 180s rollout-status timeout an unguarded Pending Deployment would otherwise produce.
ENVSUBST_STUB = "#!/usr/bin/env bash\ncat\n"  # never actually reached by the failing-preflight
# tests below (the node check exits first); present only so `command -v envsubst` succeeds
# hermetically regardless of what's installed on the machine running the suite.

REQUIRED_AGENT_INSTALL_ENV = {
    "AGENT_INSTALL": "1",
    "APP_NAMESPACE": "test-hotel-reservation",
    "VLLM_BASE_URL": "http://198.51.100.1:8000/v1",
    "VLLM_EGRESS_CIDR": "198.51.100.1/32",
    "VLLM_EGRESS_PORT": "8000",
}


def test_agent_install_default_off_never_calls_kubectl_apply_for_the_agent_manifests(sandbox, tmp_path):
    dest, bin_dir, state = sandbox
    kubectl_log = tmp_path / "kubectl.log"

    result = _run(dest, bin_dir, state, {"STUB_KUBECTL_LOG": str(kubectl_log)})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "AGENT_INSTALL != 1" in result.stdout
    assert "k8s" not in kubectl_log.read_text()


def test_agent_install_1_without_dedicated_node_fails_loud_before_applying_anything(sandbox, tmp_path):
    """The kubectl stub's default case (unmatched invocation) already returns success with empty
    stdout -- exactly what a real `kubectl get nodes -l role=sslm-agent --no-headers` would print
    against a cluster with no such node. No special stubbing needed to hit that branch; this test
    proves the script itself, not the stub, is what turns that into a clear failure."""
    dest, bin_dir, state = sandbox
    envsubst_stub = bin_dir / "envsubst"
    _write_stub(envsubst_stub, ENVSUBST_STUB)
    envsubst_stub.chmod(envsubst_stub.stat().st_mode | stat.S_IEXEC)
    kubectl_log = tmp_path / "kubectl.log"

    result = _run(
        dest,
        bin_dir,
        state,
        {**REQUIRED_AGENT_INSTALL_ENV, "STUB_KUBECTL_LOG": str(kubectl_log)},
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "role=sslm-agent" in combined
    assert "recreate the cluster" in combined
    # Never got far enough to apply the RBAC/Deployment manifests.
    assert "namespace.yaml" not in kubectl_log.read_text()
    assert "role-app-mutate" not in kubectl_log.read_text()


def test_agent_install_1_missing_app_namespace_fails_loud_before_the_node_check(sandbox):
    dest, bin_dir, state = sandbox
    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    del env["APP_NAMESPACE"]

    result = _run(dest, bin_dir, state, env)

    assert result.returncode != 0
    assert "APP_NAMESPACE" in (result.stdout + result.stderr)


# PROMETHEUS_URL wiring: a real end-to-end run of the
# AGENT_INSTALL=1 path, node check passing this time, proving the value setup_cluster.sh derives
# (or the caller overrides) actually reaches deployment.yaml's env var through the real envsubst
# on this machine's PATH, the same "run the script for real, not just call the Python
# function in isolation" discipline as the kind-config patch test below.
def _agent_install_sandbox(tmp_path):
    dest = tmp_path / "triage-eval"
    dest.mkdir()
    for name in ("setup_cluster.sh", "patch_leader_election.py"):
        shutil.copy(PREFLIGHT_DIR / name, dest / name)
    shutil.copytree(PREFLIGHT_DIR / "k8s", dest / "k8s")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("kind", KIND_STUB), ("docker", DOCKER_STUB), ("kubectl", KUBECTL_STUB)):
        stub = bin_dir / name
        _write_stub(stub, body)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    state = tmp_path / "manifests"
    state.mkdir()
    for manifest in ("kube-controller-manager.yaml", "kube-scheduler.yaml"):
        (state / manifest).write_text(MANIFEST_FIXTURE)

    return dest, bin_dir, state


def test_prometheus_url_default_derives_from_kps_namespace_and_reaches_deployment_apply(tmp_path):
    dest, bin_dir, state = _agent_install_sandbox(tmp_path)
    apply_log = tmp_path / "kubectl-apply-stdin.log"
    kubectl_log = tmp_path / "kubectl.log"

    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    env["KPS_NAMESPACE"] = "custom-monitoring"
    result = _run(
        dest,
        bin_dir,
        state,
        {
            **env,
            "STUB_AGENT_NODE_PRESENT": "1",
            "STUB_KUBECTL_LOG": str(kubectl_log),
            "STUB_KUBECTL_APPLY_STDIN_LOG": str(apply_log),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    applied = apply_log.read_text()
    kubectl_calls = kubectl_log.read_text()
    assert "create namespace test-hotel-reservation --dry-run=client -o yaml" in kubectl_calls
    assert "kind: Namespace" in applied
    assert "name: test-hotel-reservation" in applied
    # PROMETHEUS_URL was never set by the caller here -- setup_cluster.sh must derive it from
    # KPS_NAMESPACE (the custom override above), never fall back to the "monitoring" default.
    assert 'value: "http://prometheus-operated.custom-monitoring.svc.cluster.local:9090"' in applied
    assert "name: PROMETHEUS_URL" in applied
    assert "name: APP_NAMESPACE" in applied
    assert 'value: "test-hotel-reservation"' in applied


# SSLM_LOG_ONLY: in a round without a model, VLLM_BASE_URL/VLLM_EGRESS_CIDR/VLLM_EGRESS_PORT stop
# being required with no default and get one, instead of the gate being simplified to drop them
# (the SSLM_LOG_ONLY block's own comment in setup_cluster.sh explains why).


def test_agent_install_1_log_only_defaults_vllm_egress_and_reaches_the_networkpolicy_apply(tmp_path):
    dest, bin_dir, state = _agent_install_sandbox(tmp_path)
    apply_log = tmp_path / "kubectl-apply-stdin.log"

    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    del env["VLLM_BASE_URL"]
    del env["VLLM_EGRESS_CIDR"]
    del env["VLLM_EGRESS_PORT"]
    env["SSLM_LOG_ONLY"] = "1"
    result = _run(
        dest,
        bin_dir,
        state,
        {
            **env,
            "STUB_AGENT_NODE_PRESENT": "1",
            "STUB_KUBECTL_APPLY_STDIN_LOG": str(apply_log),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SSLM_LOG_ONLY=1" in result.stdout
    applied = apply_log.read_text()
    # VLLM_EGRESS_CIDR derived from the docker stub's canned gateway (172.19.0.1); the stub
    # prints the IPv6 entry (fc00::1) BEFORE the IPv4 one by default (the dual-stack shape of the
    # bug), and the script must still pick the IPv4. VLLM_EGRESS_PORT falls back to vLLM's own
    # default port, 8000.
    assert "cidr: 172.19.0.1/32" in applied
    assert "port: 8000" in applied
    assert 'value: "http://172.19.0.1:8000/v1"' in applied
    assert "fc00::1" not in applied


def test_agent_install_1_log_only_dual_stack_ipv6_first_still_derives_the_ipv4_gateway(tmp_path):
    """Regression for an observed bug: `docker network inspect kind` on a dual-stack
    (IPv4+IPv6) "kind" network can list the IPv6 IPAM.Config entry before the IPv4 one. The old
    `index .IPAM.Config 0` picked whatever was first regardless of family and derived an invalid
    IPv6 VLLM_EGRESS_CIDR (e.g. "fc00::1/32"); the fix must keep picking the IPv4 gateway no
    matter which position it lands in."""
    dest, bin_dir, state = _agent_install_sandbox(tmp_path)
    apply_log = tmp_path / "kubectl-apply-stdin.log"

    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    del env["VLLM_BASE_URL"]
    del env["VLLM_EGRESS_CIDR"]
    del env["VLLM_EGRESS_PORT"]
    env["SSLM_LOG_ONLY"] = "1"
    # A gateway different from the DOCKER_STUB's own default (172.19.0.1), so this assertion can't
    # pass by coincidence if the picking logic were broken in some other way.
    env["STUB_DOCKER_GATEWAY"] = "192.0.2.1"
    result = _run(
        dest,
        bin_dir,
        state,
        {
            **env,
            "STUB_AGENT_NODE_PRESENT": "1",
            "STUB_KUBECTL_APPLY_STDIN_LOG": str(apply_log),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    applied = apply_log.read_text()
    assert "cidr: 192.0.2.1/32" in applied
    assert 'value: "http://192.0.2.1:8000/v1"' in applied
    assert "fc00::1" not in applied
    assert "fc00" not in applied


def test_agent_install_1_log_only_caller_override_wins_over_the_derived_defaults(tmp_path):
    dest, bin_dir, state = _agent_install_sandbox(tmp_path)
    apply_log = tmp_path / "kubectl-apply-stdin.log"

    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    env["SSLM_LOG_ONLY"] = "1"
    # Already in REQUIRED_AGENT_INSTALL_ENV (198.51.100.1:8000); an operator who knows a real vLLM
    # will come up later at the same address can pass these even in log-only mode.
    result = _run(
        dest,
        bin_dir,
        state,
        {
            **env,
            "STUB_AGENT_NODE_PRESENT": "1",
            "STUB_KUBECTL_APPLY_STDIN_LOG": str(apply_log),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    applied = apply_log.read_text()
    assert "cidr: 198.51.100.1/32" in applied
    assert 'value: "http://198.51.100.1:8000/v1"' in applied
    assert "172.19.0.1" not in applied


def test_agent_install_1_log_only_still_requires_app_namespace(sandbox):
    dest, bin_dir, state = sandbox
    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    del env["APP_NAMESPACE"]
    del env["VLLM_BASE_URL"]
    del env["VLLM_EGRESS_CIDR"]
    del env["VLLM_EGRESS_PORT"]
    env["SSLM_LOG_ONLY"] = "1"

    result = _run(dest, bin_dir, state, env)

    assert result.returncode != 0
    assert "APP_NAMESPACE" in (result.stdout + result.stderr)


def test_agent_install_1_without_log_only_still_requires_the_vllm_vars(sandbox):
    """Regression of the original behavior: without SSLM_LOG_ONLY=1 the three VLLM_* stay required
    with no default, so a round with a real model never inherits the log-only default."""
    dest, bin_dir, state = sandbox
    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    del env["VLLM_BASE_URL"]

    result = _run(dest, bin_dir, state, env)

    assert result.returncode != 0
    assert "VLLM_BASE_URL" in (result.stdout + result.stderr)


def test_agent_install_1_log_only_fails_loud_when_docker_network_inspect_is_empty(sandbox):
    dest, bin_dir, state = sandbox
    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    del env["VLLM_BASE_URL"]
    del env["VLLM_EGRESS_CIDR"]
    del env["VLLM_EGRESS_PORT"]
    env["SSLM_LOG_ONLY"] = "1"
    env["STUB_DOCKER_GATEWAY"] = ""

    result = _run(dest, bin_dir, state, env)

    assert result.returncode != 0
    assert "VLLM_EGRESS_CIDR" in (result.stdout + result.stderr)


def test_agent_install_1_log_only_single_stack_ipv4_only_still_derives_the_gateway(tmp_path):
    """The extraction mechanism itself changed (Go template `index 0` -> `range` piped through a
    bash loop that filters by content); this proves the common case (a single-stack "kind"
    network with only one IPAM.Config entry, no IPv6 to skip over) still works after
    that change, not just the dual-stack scenario the bug report was about."""
    dest, bin_dir, state = _agent_install_sandbox(tmp_path)
    apply_log = tmp_path / "kubectl-apply-stdin.log"

    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    del env["VLLM_BASE_URL"]
    del env["VLLM_EGRESS_CIDR"]
    del env["VLLM_EGRESS_PORT"]
    env["SSLM_LOG_ONLY"] = "1"
    env["STUB_DOCKER_GATEWAY"] = "172.19.0.1"
    env["STUB_DOCKER_GATEWAY_NO_IPV6"] = "1"
    result = _run(
        dest,
        bin_dir,
        state,
        {
            **env,
            "STUB_AGENT_NODE_PRESENT": "1",
            "STUB_KUBECTL_APPLY_STDIN_LOG": str(apply_log),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    applied = apply_log.read_text()
    assert "cidr: 172.19.0.1/32" in applied
    assert 'value: "http://172.19.0.1:8000/v1"' in applied


def test_agent_install_1_log_only_fails_loud_when_docker_network_has_only_ipv6(sandbox):
    """A "kind" network with an IPAM.Config entry but no IPv4 among them (only IPv6, e.g.
    fc00::1) must still fail loud, exactly like the empty-inspect case above, never fall back to
    silently emitting the IPv6 gateway as VLLM_EGRESS_CIDR."""
    dest, bin_dir, state = sandbox
    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    del env["VLLM_BASE_URL"]
    del env["VLLM_EGRESS_CIDR"]
    del env["VLLM_EGRESS_PORT"]
    env["SSLM_LOG_ONLY"] = "1"
    env["STUB_DOCKER_GATEWAY_IPV6_ONLY"] = "1"

    result = _run(dest, bin_dir, state, env)

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "VLLM_EGRESS_CIDR" in combined
    assert "fc00::1" not in combined


def test_prometheus_url_caller_override_wins_over_the_derived_default(tmp_path):
    dest, bin_dir, state = _agent_install_sandbox(tmp_path)
    apply_log = tmp_path / "kubectl-apply-stdin.log"

    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    env["PROMETHEUS_URL"] = "http://prom.override.example:9090"
    result = _run(
        dest,
        bin_dir,
        state,
        {
            **env,
            "STUB_AGENT_NODE_PRESENT": "1",
            "STUB_KUBECTL_APPLY_STDIN_LOG": str(apply_log),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    applied = apply_log.read_text()
    assert 'value: "http://prom.override.example:9090"' in applied
    assert "prometheus-operated" not in applied


# APISERVER_EGRESS_CIDR / APISERVER_EGRESS_PORT: a real
# end-to-end run proving setup_cluster.sh derives these from the live cluster's own `kubernetes`
# Endpoints object (never the Service's ClusterIP; see networkpolicy.yaml's own comment for why
# that used to be wrong under Calico) and that the derived value actually reaches
# networkpolicy.yaml's substituted egress rule, same "run the script for real" discipline as the
# PROMETHEUS_URL tests above.


def test_apiserver_egress_cidr_and_port_default_derive_from_kubernetes_endpoints(tmp_path):
    dest, bin_dir, state = _agent_install_sandbox(tmp_path)
    apply_log = tmp_path / "kubectl-apply-stdin.log"

    result = _run(
        dest,
        bin_dir,
        state,
        {
            **REQUIRED_AGENT_INSTALL_ENV,
            "STUB_AGENT_NODE_PRESENT": "1",
            "STUB_KUBECTL_APPLY_STDIN_LOG": str(apply_log),
            "STUB_APISERVER_IP": "203.0.113.55",
            "STUB_APISERVER_PORT": "6443",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    applied = apply_log.read_text()
    # Neither var was set by the caller -- setup_cluster.sh must derive both from
    # `kubectl get endpoints kubernetes`, never fall back to the Service ClusterIP.
    assert "cidr: 203.0.113.55/32" in applied
    assert "port: 6443" in applied
    assert "cidr: 10.96.0.1/32" not in applied


def test_apiserver_egress_cidr_and_port_caller_override_wins_over_the_derived_default(tmp_path):
    dest, bin_dir, state = _agent_install_sandbox(tmp_path)
    apply_log = tmp_path / "kubectl-apply-stdin.log"

    env = dict(REQUIRED_AGENT_INSTALL_ENV)
    env["APISERVER_EGRESS_CIDR"] = "198.51.100.77/32"
    env["APISERVER_EGRESS_PORT"] = "8443"
    result = _run(
        dest,
        bin_dir,
        state,
        {
            **env,
            "STUB_AGENT_NODE_PRESENT": "1",
            "STUB_KUBECTL_APPLY_STDIN_LOG": str(apply_log),
            # Deliberately different from the override, to prove it never gets consulted when the
            # caller already set the value.
            "STUB_APISERVER_IP": "203.0.113.55",
            "STUB_APISERVER_PORT": "6443",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    applied = apply_log.read_text()
    assert "cidr: 198.51.100.77/32" in applied
    assert "port: 8443" in applied
    assert "203.0.113.55" not in applied


def test_apiserver_egress_cidr_fails_loud_when_endpoints_lookup_is_empty(tmp_path):
    """A real `kubectl get endpoints kubernetes` returning nothing (RBAC-restricted kubeconfig,
    or a genuinely broken cluster) must not silently apply a NetworkPolicy with an empty ipBlock
    cidr -- that would parse as invalid YAML/API object, but failing loud here is a much clearer
    signal than whatever kubectl's own rejection message would say."""
    dest, bin_dir, state = _agent_install_sandbox(tmp_path)

    result = _run(
        dest,
        bin_dir,
        state,
        {
            **REQUIRED_AGENT_INSTALL_ENV,
            "STUB_AGENT_NODE_PRESENT": "1",
            "STUB_APISERVER_IP": "",
        },
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "APISERVER_EGRESS_CIDR" in combined


# The kind-config patch wiring at cluster CREATION time (this was previously exercised only at the Python patch() function level, never through
# setup_cluster.sh's own mktemp/trap/`kind create cluster --config` plumbing). A `kind` stub that
# reports no cluster exists yet forces the creation branch; it copies whatever config file it was
# given to a log path so the test can assert the AGENT worker node actually made it into what
# `kind create cluster` was called with, not just what patch() returns in isolation.
NEW_CLUSTER_KIND_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    set -eu
    if [[ "$1" == "get" && "$2" == "clusters" ]]; then
      exit 0  # empty stdout: no cluster named "$CLUSTER_NAME" exists yet
    fi
    if [[ "$1" == "create" && "$2" == "cluster" ]]; then
      # Args are: create cluster --name <name> --config <path> ($1..$6)
      config_path="${6}"
      cp "$config_path" "${STUB_KIND_CREATE_CONFIG_LOG:?}"
      exit 0
    fi
    echo "unexpected kind invocation: $*" >&2
    exit 1
    """
)

BASE_KIND_CONFIG_FIXTURE = textwrap.dedent(
    """\
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    nodes:
      - role: control-plane
      - role: worker
    """
)


def test_agent_install_1_patches_the_kind_config_before_creating_a_new_cluster(tmp_path):
    dest = tmp_path / "triage-eval"
    dest.mkdir()
    for name in ("setup_cluster.sh", "patch_leader_election.py"):
        shutil.copy(PREFLIGHT_DIR / name, dest / name)
    kind_dir = dest / "kind"
    kind_dir.mkdir()
    shutil.copy(PREFLIGHT_DIR / "kind" / "patch_kind_config_agent_node.py", kind_dir / "patch_kind_config_agent_node.py")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("kind", NEW_CLUSTER_KIND_STUB), ("docker", DOCKER_STUB), ("kubectl", KUBECTL_STUB)):
        stub = bin_dir / name
        _write_stub(stub, body)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    # setup_cluster.sh invokes the patch script as `python3 ...`; the plain `python3` first on
    # PATH on this machine may not be the interpreter that has pyyaml installed (this repo's dev
    # setup has several pythons around). Point it at whatever interpreter is running THIS test
    # (guaranteed to have pyyaml, since this module imports it), not whatever `python3` happens
    # to resolve to ambiently.
    python3_stub = bin_dir / "python3"
    python3_stub.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
    python3_stub.chmod(python3_stub.stat().st_mode | stat.S_IEXEC)

    state = tmp_path / "manifests"
    state.mkdir()
    for manifest in ("kube-controller-manager.yaml", "kube-scheduler.yaml"):
        (state / manifest).write_text(MANIFEST_FIXTURE)

    kind_config_fixture = tmp_path / "kind-config-fixture.yaml"
    kind_config_fixture.write_text(BASE_KIND_CONFIG_FIXTURE)
    created_config_log = tmp_path / "kind-create-config.yaml"

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LEASE_DURATION", "RENEW_DEADLINE", "RETRY_PERIOD")
    }
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_STATE"] = str(state)
    env["AIOPSLAB_HOME"] = str(tmp_path)  # required by the script's own check; unused here since
    # KIND_CONFIG is set explicitly below, which short-circuits the arch-default lookup.
    env["KIND_CONFIG"] = str(kind_config_fixture)
    env["STUB_KIND_CREATE_CONFIG_LOG"] = str(created_config_log)
    env["AGENT_INSTALL"] = "1"
    env.update(REQUIRED_AGENT_INSTALL_ENV)

    result = subprocess.run(
        ["bash", str(dest / "setup_cluster.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    # The script goes on to the AGENT_INSTALL manifest-apply section next, which needs a real
    # kubectl (not this test's stub) to succeed end-to-end -- out of scope for this test, which
    # only proves the kind-config patch actually reached `kind create cluster`. Assert on the log
    # directly rather than the script's overall exit code.
    assert created_config_log.exists(), result.stdout + result.stderr
    created_doc = yaml.safe_load(created_config_log.read_text())
    assert len(created_doc["nodes"]) == 3
    assert "role=sslm-agent" in "\n".join(created_doc["nodes"][2]["kubeadmConfigPatches"])
    # The original fixture on disk must be untouched (patched in memory only, per
    # kind/patch_kind_config_agent_node.py's own docstring: never fork a copy into this repo).
    assert yaml.safe_load(kind_config_fixture.read_text()) == yaml.safe_load(BASE_KIND_CONFIG_FIXTURE)


# SSLM_C2_DRY_RUN wiring (an operational decision made once the episode runner had a real
# rollback_state/read-back path): a deployment applied through
# AGENT_INSTALL=1 must default to real C2 execution (SSLM_C2_DRY_RUN=0), not dry-run, and an
# explicit override must reach the envsubst call that renders deployment.yaml unchanged. Same
# KUBECTL_STUB as above, plus a "node present" case so the AGENT_INSTALL=1 block's own preflight
# (tested failing above) gets past the node check and reaches the envsubst/apply calls this test
# actually cares about.
AGENT_KUBECTL_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    set -eu
    echo "$*" >> "${STUB_KUBECTL_LOG:-/dev/null}"
    args="$*"
    case "$args" in
      *"get deploy openebs-localpv-provisioner -o jsonpath"*)
        [[ "${STUB_LE_PRESENT:-0}" == "1" ]] || exit 1
        printf '%s' "${STUB_LE_VALUE:-}" ;;
      *"get deploy openebs-localpv-provisioner"*)
        [[ "${STUB_LE_PRESENT:-0}" == "1" ]] || exit 1 ;;
      *"get pods"*) printf '%s' "${STUB_INITPVC:-}" ;;
      *"get nodes -l role=sslm-agent --no-headers"*)
        echo "kind-worker2   Ready   <none>   1m   v1.29.0" ;;
      *"get endpoints kubernetes -o jsonpath="*"addresses[0].ip"*)
        printf '198.51.100.9' ;;
      *"get endpoints kubernetes -o jsonpath="*"ports[0].port"*)
        printf '6443' ;;
      # `apply -f -` (and the final `rollout status`) reach this stub; the real kubectl always
      # reads stdin to EOF, and this stub must too -- otherwise the envsubst|kubectl pipe's
      # writer gets SIGPIPE the moment this process exits without draining it (exit 141, not a
      # real script bug, just an unconsumed pipe in the stub).
      *) cat >/dev/null ;;
    esac
    """
)

# Logs the SSLM_C2_DRY_RUN value visible to each envsubst invocation (the script exports it before
# any envsubst call, same as AGENT_IMAGE/VLLM_BASE_URL), then passes the manifest through
# unmodified -- this test cares about what the script EXPORTS, not about real placeholder
# substitution.
ENVSUBST_LOG_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    set -eu
    echo "SSLM_C2_DRY_RUN=${SSLM_C2_DRY_RUN:-<unset>}" >> "${STUB_ENVSUBST_LOG:?}"
    cat
    """
)


def _run_agent_install_with_dry_run_env(sandbox, tmp_path, extra_env):
    dest, bin_dir, state = sandbox
    # Unlike every AGENT_INSTALL=1 test above (which stops at the node-check preflight, before
    # any manifest is read), this one reaches the actual envsubst/apply calls -- the sandbox
    # fixture never copies k8s/ (no earlier test needed it), so it has to be added here.
    shutil.copytree(PREFLIGHT_DIR / "k8s", dest / "k8s")
    kubectl_stub = bin_dir / "kubectl"
    _write_stub(kubectl_stub, AGENT_KUBECTL_STUB)
    kubectl_stub.chmod(kubectl_stub.stat().st_mode | stat.S_IEXEC)
    envsubst_stub = bin_dir / "envsubst"
    _write_stub(envsubst_stub, ENVSUBST_LOG_STUB)
    envsubst_stub.chmod(envsubst_stub.stat().st_mode | stat.S_IEXEC)
    kubectl_log = tmp_path / "kubectl.log"
    envsubst_log = tmp_path / "envsubst.log"

    result = _run(
        dest,
        bin_dir,
        state,
        {
            **REQUIRED_AGENT_INSTALL_ENV,
            "STUB_KUBECTL_LOG": str(kubectl_log),
            "STUB_ENVSUBST_LOG": str(envsubst_log),
            **extra_env,
        },
    )
    return result, envsubst_log


def test_agent_install_1_defaults_ssl_c2_dry_run_off(sandbox, tmp_path):
    result, envsubst_log = _run_agent_install_with_dry_run_env(sandbox, tmp_path, {})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SSLM_C2_DRY_RUN=0" in envsubst_log.read_text()


def test_agent_install_1_honors_an_explicit_ssl_c2_dry_run_override(sandbox, tmp_path):
    result, envsubst_log = _run_agent_install_with_dry_run_env(
        sandbox, tmp_path, {"SSLM_C2_DRY_RUN": "1"}
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SSLM_C2_DRY_RUN=1" in envsubst_log.read_text()


# Calico CNI section (closes the "NetworkPolicy declared but not enforced" limit; see
# k8s/networkpolicy.yaml's own comment and README.md). Opt-in via its own
# CALICO_INSTALL flag, deliberately independent from AGENT_INSTALL (setup_cluster.sh's own
# comment on the CALICO_INSTALL block explains why): a plain run with no env vars must stay fully
# untouched, and CALICO_INSTALL=1 must refuse to run against a cluster whose default CNI was
# never disabled at creation time.


def _copy_kind_dir(dest: Path) -> None:
    kind_dir = dest / "kind"
    kind_dir.mkdir(exist_ok=True)
    for name in ("patch_kind_config_agent_node.py", "patch_kind_config_disable_default_cni.py"):
        shutil.copy(PREFLIGHT_DIR / "kind" / name, kind_dir / name)
    vendor_dir = kind_dir / "vendor"
    vendor_dir.mkdir(exist_ok=True)
    shutil.copy(
        PREFLIGHT_DIR / "kind" / "vendor" / "calico-v3.28.2.yaml",
        vendor_dir / "calico-v3.28.2.yaml",
    )


def test_calico_install_default_off_never_touches_calico(sandbox, tmp_path):
    dest, bin_dir, state = sandbox
    kubectl_log = tmp_path / "kubectl.log"

    result = _run(dest, bin_dir, state, {"STUB_KUBECTL_LOG": str(kubectl_log)})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "CALICO_INSTALL != 1" in result.stdout
    assert "calico" not in kubectl_log.read_text().lower()


def test_calico_install_1_without_a_vendored_manifest_for_the_version_fails_loud(sandbox, tmp_path):
    """No kind/vendor/ directory at all in this sandbox (basic `sandbox` fixture never copies
    kind/): the missing-manifest check must fire before anything else, regardless of the pinned
    version asked for."""
    dest, bin_dir, state = sandbox

    result = _run(dest, bin_dir, state, {"CALICO_INSTALL": "1"})

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "CALICO_VERSION=v3.28.2" in combined
    assert "vendor" in combined


def test_calico_install_1_on_a_cluster_that_already_existed_without_calico_fails_loud(sandbox, tmp_path):
    """The basic `sandbox`/`_run` fixtures always hit the 'cluster already exists' branch (KIND_STUB
    unconditionally reports a cluster named 'kind', the CLUSTER_NAME default). CALICO_INSTALL=1
    against that must refuse rather than silently install Calico alongside a kindnet that's
    already running."""
    dest, bin_dir, state = sandbox
    _copy_kind_dir(dest)

    result = _run(dest, bin_dir, state, {"CALICO_INSTALL": "1"})

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "already existed without Calico" in combined
    assert "recreate the cluster" in combined


def test_calico_install_1_on_a_cluster_that_already_has_calico_skips_apply_and_waits(sandbox, tmp_path):
    """An operator re-running the script against a cluster CALICO_INSTALL=1 already set up: no
    re-apply of the manifest, but the readiness wait still runs (idempotent, same shape as the
    openebs/KPS sections above)."""
    dest, bin_dir, state = sandbox
    _copy_kind_dir(dest)
    kubectl_log = tmp_path / "kubectl.log"

    result = _run(
        dest,
        bin_dir,
        state,
        {
            "CALICO_INSTALL": "1",
            "STUB_CALICO_PRESENT": "1",
            "STUB_KUBECTL_LOG": str(kubectl_log),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "calico-node already exists" in result.stdout
    calls = kubectl_log.read_text()
    assert "calico-v3.28.2.yaml" not in calls
    assert "rollout status daemonset/calico-node" in calls


# The kind-config patch wiring at cluster CREATION time, same discipline as
# test_agent_install_1_patches_the_kind_config_before_creating_a_new_cluster above: run the real
# script against a `kind` stub that reports no cluster exists yet, and assert on what actually got
# passed to `kind create cluster --config`, not just what the Python patch() returns in isolation.
def _new_cluster_calico_sandbox(tmp_path):
    dest = tmp_path / "triage-eval"
    dest.mkdir()
    for name in ("setup_cluster.sh", "patch_leader_election.py"):
        shutil.copy(PREFLIGHT_DIR / name, dest / name)
    _copy_kind_dir(dest)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("kind", NEW_CLUSTER_KIND_STUB), ("docker", DOCKER_STUB), ("kubectl", KUBECTL_STUB)):
        stub = bin_dir / name
        _write_stub(stub, body)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    python3_stub = bin_dir / "python3"
    python3_stub.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
    python3_stub.chmod(python3_stub.stat().st_mode | stat.S_IEXEC)

    state = tmp_path / "manifests"
    state.mkdir()
    for manifest in ("kube-controller-manager.yaml", "kube-scheduler.yaml"):
        (state / manifest).write_text(MANIFEST_FIXTURE)

    return dest, bin_dir, state


def test_calico_install_1_disables_default_cni_before_creating_a_new_cluster(tmp_path):
    dest, bin_dir, state = _new_cluster_calico_sandbox(tmp_path)

    kind_config_fixture = tmp_path / "kind-config-fixture.yaml"
    kind_config_fixture.write_text(BASE_KIND_CONFIG_FIXTURE)
    created_config_log = tmp_path / "kind-create-config.yaml"
    kubectl_log = tmp_path / "kubectl.log"

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LEASE_DURATION", "RENEW_DEADLINE", "RETRY_PERIOD")
    }
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_STATE"] = str(state)
    env["AIOPSLAB_HOME"] = str(tmp_path)
    env["KIND_CONFIG"] = str(kind_config_fixture)
    env["STUB_KIND_CREATE_CONFIG_LOG"] = str(created_config_log)
    env["STUB_KUBECTL_LOG"] = str(kubectl_log)
    env["CALICO_INSTALL"] = "1"

    result = subprocess.run(
        ["bash", str(dest / "setup_cluster.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    assert created_config_log.exists(), result.stdout + result.stderr
    created_doc = yaml.safe_load(created_config_log.read_text())
    assert created_doc["networking"]["disableDefaultCNI"] is True
    assert created_doc["networking"]["podSubnet"] == "192.168.0.0/16"
    # The original fixture on disk must be untouched (patched in memory only).
    assert yaml.safe_load(kind_config_fixture.read_text()) == yaml.safe_load(BASE_KIND_CONFIG_FIXTURE)

    # The script goes on to apply the vendored manifest against this test's kubectl stub next.
    assert result.returncode == 0, result.stdout + result.stderr
    calls = kubectl_log.read_text()
    assert "calico-v3.28.2.yaml" in calls
    assert "rollout status daemonset/calico-node" in calls


def test_calico_install_1_and_agent_install_1_compose_on_the_same_created_config(tmp_path):
    """Both flags set together is a real, supported combination (the exact scenario the
    NetworkPolicy protects, per the README): the created kind config must carry BOTH the agent's
    dedicated worker node and disableDefaultCNI, proving the patches chain rather than one
    clobbering the other."""
    dest, bin_dir, state = _new_cluster_calico_sandbox(tmp_path)

    kind_config_fixture = tmp_path / "kind-config-fixture.yaml"
    kind_config_fixture.write_text(BASE_KIND_CONFIG_FIXTURE)
    created_config_log = tmp_path / "kind-create-config.yaml"

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LEASE_DURATION", "RENEW_DEADLINE", "RETRY_PERIOD")
    }
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_STATE"] = str(state)
    env["AIOPSLAB_HOME"] = str(tmp_path)
    env["KIND_CONFIG"] = str(kind_config_fixture)
    env["STUB_KIND_CREATE_CONFIG_LOG"] = str(created_config_log)
    env["CALICO_INSTALL"] = "1"
    env["AGENT_INSTALL"] = "1"
    env.update(REQUIRED_AGENT_INSTALL_ENV)
    env["STUB_AGENT_NODE_PRESENT"] = "1"  # past the post-creation node-check preflight too

    result = subprocess.run(
        ["bash", str(dest / "setup_cluster.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    assert created_config_log.exists(), result.stdout + result.stderr
    created_doc = yaml.safe_load(created_config_log.read_text())
    assert len(created_doc["nodes"]) == 3
    assert "role=sslm-agent" in "\n".join(created_doc["nodes"][2]["kubeadmConfigPatches"])
    assert created_doc["networking"]["disableDefaultCNI"] is True


# --- found on the first real run: namespace contract and local image load ------------------------

KIND_LOG_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    echo "$*" >> "${STUB_KIND_LOG:-/dev/null}"
    echo kind
    """
)


def _agent_install_with_kind_log(sandbox, tmp_path, extra_env):
    dest, bin_dir, _state = sandbox
    kind_stub = bin_dir / "kind"
    _write_stub(kind_stub, KIND_LOG_STUB)
    kind_stub.chmod(kind_stub.stat().st_mode | stat.S_IEXEC)
    kind_log = tmp_path / "kind.log"
    result, _envsubst_log = _run_agent_install_with_dry_run_env(
        sandbox, tmp_path, {"STUB_KIND_LOG": str(kind_log), **extra_env}
    )
    return result, (kind_log.read_text() if kind_log.exists() else ""), tmp_path / "kubectl.log"


@pytest.mark.parametrize("namespace", ["hotelReservation", "test_hotel", "-bad", ""])
def test_agent_install_rejects_a_non_rfc1123_app_namespace_before_any_apply(sandbox, tmp_path, namespace):
    result, _kind, kubectl_log = _agent_install_with_kind_log(sandbox, tmp_path, {"APP_NAMESPACE": namespace})

    assert result.returncode != 0
    applied = kubectl_log.read_text() if kubectl_log.exists() else ""
    assert "apply" not in applied


def test_agent_install_loads_the_local_image_into_kind_before_the_deployment(sandbox, tmp_path):
    result, kind_calls, kubectl_log = _agent_install_with_kind_log(sandbox, tmp_path, {})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "load docker-image sslm-agent:local --name kind" in kind_calls
    stdout = result.stdout
    assert stdout.index("kind load docker-image") < stdout.index("Deployment (image")


def test_agent_install_does_not_kind_load_a_registry_image(sandbox, tmp_path):
    result, kind_calls, _ = _agent_install_with_kind_log(
        sandbox, tmp_path, {"AGENT_IMAGE": "ghcr.io/example/sslm-agent@sha256:" + "a" * 64}
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "load docker-image" not in kind_calls


def test_kps_node_exporter_does_not_take_aiopslab_observer_host_port():
    # AIOpsLab's observer node-exporter binds hostNetwork 9100 on every node.
    values = yaml.safe_load((PREFLIGHT_DIR / "kube-prometheus-stack" / "values.yaml").read_text())
    service = values["prometheus-node-exporter"]["service"]
    assert service["port"] != 9100 and service["targetPort"] == service["port"]


# COMPARATOR_ARM=1 (hosted-API comparator arm): the API arm needs no VLLM_*, needs the key and the spend cap, and the
# key reaches the cluster only as a Secret created from stdin. Nothing it prints, logs or applies
# carries the key's value.

COMPARATOR_KEY = "sk-ant-test-not-a-real-key"
COMPARATOR_ENV = {
    "AGENT_INSTALL": "1",
    "APP_NAMESPACE": "test-hotel-reservation",
    "COMPARATOR_ARM": "1",
    "ANTHROPIC_API_KEY": COMPARATOR_KEY,
    "SSLM_COMPARATOR_MAX_COST_USD": "5",
}


def _comparator_run(tmp_path, env):
    dest, bin_dir, state = _agent_install_sandbox(tmp_path)
    logs = {name: tmp_path / f"{name}.log" for name in ("apply", "kubectl", "secret")}
    clean = {k: v for k, v in env.items() if v is not None}
    result = _run(
        dest,
        bin_dir,
        state,
        {
            **clean,
            "STUB_AGENT_NODE_PRESENT": "1",
            "STUB_KUBECTL_LOG": str(logs["kubectl"]),
            "STUB_KUBECTL_APPLY_STDIN_LOG": str(logs["apply"]),
            "STUB_SECRET_STDIN_LOG": str(logs["secret"]),
        },
    )
    read = {name: (path.read_text() if path.exists() else "") for name, path in logs.items()}
    return result, read


def test_comparator_arm_needs_no_vllm_and_puts_the_key_only_in_a_secret(tmp_path):
    result, logs = _comparator_run(tmp_path, COMPARATOR_ENV)

    assert result.returncode == 0, result.stdout + result.stderr
    assert logs["secret"] == COMPARATOR_KEY
    calls = logs["kubectl"].splitlines()
    delete = calls.index("-n sslm-agent delete secret sslm-agent-anthropic --ignore-not-found")
    create = calls.index(
        "-n sslm-agent create secret generic sslm-agent-anthropic --from-file=ANTHROPIC_API_KEY=/dev/stdin"
    )
    rollout = next(i for i, c in enumerate(calls) if "rollout status deploy/sslm-agent" in c)
    assert delete < create < rollout
    # The value is never in argv, output, or any applied manifest.
    for text in (logs["kubectl"], logs["apply"], result.stdout, result.stderr):
        assert COMPARATOR_KEY not in text
    applied = logs["apply"]
    deployment = next(d for d in applied.split("---stdin-boundary---") if "kind: Deployment" in d)
    assert 'name: COMPARATOR_ARM\n              value: "1"' in deployment
    assert 'name: SSLM_COMPARATOR_MAX_COST_USD\n              value: "5"' in deployment
    assert 'name: SSLM_HISTORY_BUDGET\n              value: "off"' in deployment
    assert 'name: VLLM_BASE_URL\n              value: ""' in deployment
    assert "${" not in deployment.split("env:", 1)[1].split("volumeMounts:", 1)[0].replace("# ", "")
    policy = next(d for d in applied.split("---stdin-boundary---") if "kind: NetworkPolicy" in d)
    assert "cidr: 160.79.104.0/23" in policy and "port: 443" in policy


@pytest.mark.parametrize(
    "change, needle",
    [
        ({"ANTHROPIC_API_KEY": None}, "ANTHROPIC_API_KEY"),
        ({"ANTHROPIC_API_KEY": ""}, "ANTHROPIC_API_KEY"),
        ({"SSLM_COMPARATOR_MAX_COST_USD": None}, "SSLM_COMPARATOR_MAX_COST_USD"),
        ({"SSLM_COMPARATOR_MAX_COST_USD": "0"}, "SSLM_COMPARATOR_MAX_COST_USD"),
        ({"SSLM_COMPARATOR_MAX_COST_USD": "nan"}, "SSLM_COMPARATOR_MAX_COST_USD"),
        ({"SSLM_LOG_ONLY": "1"}, "SSLM_LOG_ONLY"),
        ({"COMPARATOR_ARM": "true"}, "COMPARATOR_ARM"),
        # A budget itself is valid on this arm (short-context hosted-model run); malformed still stops.
        ({"SSLM_PROMPT_BUDGET_CHARS": "18k"}, "SSLM_PROMPT_BUDGET_CHARS"),
        # The comparator arm never talks to vLLM.
        ({"VLLM_MODEL": "Qwen3.5-9B"}, "VLLM_MODEL"),
        ({"VLLM_DECODING_DIALECT": "structured_outputs"}, "VLLM_DECODING_DIALECT"),
        # The API runs ran without the tool contract.
        ({"SSLM_TOOL_CONTRACT": "on"}, "SSLM_TOOL_CONTRACT"),
    ],
)
def test_comparator_arm_refuses_before_any_apply(tmp_path, change, needle):
    result, logs = _comparator_run(tmp_path, {**COMPARATOR_ENV, **change})

    assert result.returncode != 0
    assert needle in result.stderr
    assert logs["apply"] == "" and logs["secret"] == ""
    assert "create secret" not in logs["kubectl"]
    assert COMPARATOR_KEY not in result.stdout + result.stderr


def test_vllm_arm_deletes_a_stale_comparator_secret_and_never_creates_one(tmp_path):
    env = {**REQUIRED_AGENT_INSTALL_ENV, "ANTHROPIC_API_KEY": COMPARATOR_KEY}
    result, logs = _comparator_run(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "-n sslm-agent delete secret sslm-agent-anthropic --ignore-not-found" in logs["kubectl"]
    assert "create secret" not in logs["kubectl"] and logs["secret"] == ""
    deployment = next(d for d in logs["apply"].split("---stdin-boundary---") if "kind: Deployment" in d)
    assert 'name: COMPARATOR_ARM\n              value: "0"' in deployment
    assert 'name: SSLM_COMPARATOR_MAX_COST_USD\n              value: ""' in deployment
    assert 'name: VLLM_BASE_URL\n              value: "http://198.51.100.1:8000/v1"' in deployment
    assert COMPARATOR_KEY not in logs["apply"]


# SSLM_PROMPT_BUDGET_CHARS comes from the caller, empty by
# default, and reaches the Deployment as is.


def _vllm_deployment(logs):
    return next(d for d in logs["apply"].split("---stdin-boundary---") if "kind: Deployment" in d)


def test_vllm_arm_leaves_the_prompt_budget_empty_by_default(tmp_path):
    result, logs = _comparator_run(tmp_path, {**REQUIRED_AGENT_INSTALL_ENV, "SSLM_PROMPT_BUDGET_CHARS": None})

    assert result.returncode == 0, result.stdout + result.stderr
    assert 'name: SSLM_PROMPT_BUDGET_CHARS\n              value: ""' in _vllm_deployment(logs)
    assert "SSLM_PROMPT_BUDGET_CHARS=off" in result.stdout


def test_vllm_arm_passes_the_prompt_budget_to_the_deployment(tmp_path):
    result, logs = _comparator_run(tmp_path, {**REQUIRED_AGENT_INSTALL_ENV, "SSLM_PROMPT_BUDGET_CHARS": "18000"})

    assert result.returncode == 0, result.stdout + result.stderr
    deployment = _vllm_deployment(logs)
    assert 'name: SSLM_PROMPT_BUDGET_CHARS\n              value: "18000"' in deployment
    assert 'name: SSLM_HISTORY_BUDGET\n              value: "on"' in deployment


@pytest.mark.parametrize("raw", ["18k", "6000 tokens", "3999"])
def test_vllm_arm_refuses_a_malformed_prompt_budget_before_any_apply(tmp_path, raw):
    result, logs = _comparator_run(tmp_path, {**REQUIRED_AGENT_INSTALL_ENV, "SSLM_PROMPT_BUDGET_CHARS": raw})

    assert result.returncode != 0
    assert "SSLM_PROMPT_BUDGET_CHARS" in result.stderr
    assert logs["apply"] == ""


# SSLM_TOOL_CONTRACT comes from the caller, empty by default, reaches the Deployment as is, and
# a value that is neither on nor off stops before any apply.


def test_vllm_arm_leaves_the_tool_contract_empty_by_default(tmp_path):
    result, logs = _comparator_run(tmp_path, {**REQUIRED_AGENT_INSTALL_ENV, "SSLM_TOOL_CONTRACT": None})

    assert result.returncode == 0, result.stdout + result.stderr
    assert 'name: SSLM_TOOL_CONTRACT\n              value: ""' in _vllm_deployment(logs)
    assert "SSLM_TOOL_CONTRACT=off" in result.stdout


def test_vllm_arm_passes_the_tool_contract_to_the_deployment(tmp_path):
    result, logs = _comparator_run(tmp_path, {**REQUIRED_AGENT_INSTALL_ENV, "SSLM_TOOL_CONTRACT": "on"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert 'name: SSLM_TOOL_CONTRACT\n              value: "on"' in _vllm_deployment(logs)
    assert "SSLM_TOOL_CONTRACT=on" in result.stdout


@pytest.mark.parametrize("raw", ["yes", "true", "ON ", "1"])
def test_vllm_arm_refuses_a_malformed_tool_contract_before_any_apply(tmp_path, raw):
    result, logs = _comparator_run(tmp_path, {**REQUIRED_AGENT_INSTALL_ENV, "SSLM_TOOL_CONTRACT": raw})

    assert result.returncode != 0
    assert "SSLM_TOOL_CONTRACT" in result.stderr
    assert logs["apply"] == ""


# VLLM_MODEL and VLLM_DECODING_DIALECT come from the caller, empty by default, reach the
# Deployment as is, and an unknown dialect stops before any apply.


def test_vllm_arm_leaves_model_and_dialect_empty_by_default(tmp_path):
    result, logs = _comparator_run(
        tmp_path, {**REQUIRED_AGENT_INSTALL_ENV, "VLLM_MODEL": None, "VLLM_DECODING_DIALECT": None}
    )

    assert result.returncode == 0, result.stdout + result.stderr
    deployment = _vllm_deployment(logs)
    assert 'name: VLLM_MODEL\n              value: ""' in deployment
    assert 'name: VLLM_DECODING_DIALECT\n              value: ""' in deployment
    assert "VLLM_DECODING_DIALECT=guided_json" in result.stdout


def test_vllm_arm_passes_model_and_dialect_to_the_deployment(tmp_path):
    result, logs = _comparator_run(
        tmp_path,
        {**REQUIRED_AGENT_INSTALL_ENV, "VLLM_MODEL": "Qwen3.5-9B", "VLLM_DECODING_DIALECT": "structured_outputs"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    deployment = _vllm_deployment(logs)
    assert 'name: VLLM_MODEL\n              value: "Qwen3.5-9B"' in deployment
    assert 'name: VLLM_DECODING_DIALECT\n              value: "structured_outputs"' in deployment


def test_vllm_arm_refuses_an_unknown_dialect_before_any_apply(tmp_path):
    result, logs = _comparator_run(tmp_path, {**REQUIRED_AGENT_INSTALL_ENV, "VLLM_DECODING_DIALECT": "guided-json"})

    assert result.returncode != 0
    assert "VLLM_DECODING_DIALECT" in result.stderr
    assert logs["apply"] == ""


def test_comparator_arm_passes_the_prompt_budget_to_the_deployment(tmp_path):
    result, logs = _comparator_run(tmp_path, {**COMPARATOR_ENV, "SSLM_PROMPT_BUDGET_CHARS": "14000"})

    assert result.returncode == 0, result.stderr
    assert 'name: SSLM_PROMPT_BUDGET_CHARS\n              value: "14000"' in logs["apply"]
    assert 'name: SSLM_HISTORY_BUDGET\n              value: "off"' in logs["apply"]
