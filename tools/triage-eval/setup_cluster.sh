#!/usr/bin/env bash
# Brings up the AIOpsLab kind cluster and widens leader-election lease timings on the
# control-plane's kube-controller-manager and kube-scheduler static pods.
#
# Why this exists: lease-duration=90s/renew-deadline=60s/retry-period=10s were first applied
# by hand (docker exec into kind-control-plane, edited the manifests in
# /etc/kubernetes/manifests/ directly) to stop leader-election restarts caused by slow
# Docker Desktop I/O on macOS. That fix lives on the control-plane container's
# filesystem, which is thrown away on the next `kind delete cluster` / `kind create
# cluster`, so it had to be re-applied by hand every time. This script folds the fix into
# cluster setup itself: idempotent, safe to run against a cluster that's already up (it
# only creates what's missing and only patches a manifest whose flags don't already match).
#
# Usage:
#   export AIOPSLAB_HOME=/path/to/AIOpsLab   # the clone root, containing kind/kind-config-*.yaml
#   bash tools/triage-eval/setup_cluster.sh
#
# Env overrides (defaults match the values that fixed the restarts):
#   CLUSTER_NAME              default: kind (matches the AIOpsLab kind-config, which sets no name)
#   KIND_CONFIG                default: $AIOPSLAB_HOME/kind/kind-config-<arch>.yaml
#   LEASE_DURATION              default: 90s
#   RENEW_DEADLINE               default: 60s
#   RETRY_PERIOD                  default: 10s
#   CALICO_INSTALL                 default: 0 (opt-in; see the CALICO_INSTALL block below)
#   CALICO_VERSION                  default: v3.28.2 (pinned; only takes effect with CALICO_INSTALL=1)
#   CHAOS_MESH_INSTALL               default: 0 (opt-in; see the CHAOS_MESH_INSTALL block below)
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-kind}"
LEASE_DURATION="${LEASE_DURATION:-90s}"
RENEW_DEADLINE="${RENEW_DEADLINE:-60s}"
RETRY_PERIOD="${RETRY_PERIOD:-10s}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_SCRIPT="$SCRIPT_DIR/patch_leader_election.py"

for bin in kind docker python3; do
  command -v "$bin" >/dev/null 2>&1 || { echo "error: '$bin' is not on PATH" >&2; exit 1; }
done

CLUSTER_JUST_CREATED=0
if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  : "${AIOPSLAB_HOME:?error: AIOPSLAB_HOME must point at the AIOpsLab clone to create the cluster}"
  if [[ -z "${KIND_CONFIG:-}" ]]; then
    case "$(uname -m)" in
      arm64|aarch64) KIND_CONFIG="$AIOPSLAB_HOME/kind/kind-config-arm.yaml" ;;
      x86_64|amd64)  KIND_CONFIG="$AIOPSLAB_HOME/kind/kind-config-x86.yaml" ;;
      *) echo "error: no default kind-config for architecture $(uname -m), pass KIND_CONFIG" >&2; exit 1 ;;
    esac
  fi
  [[ -f "$KIND_CONFIG" ]] || { echo "error: KIND_CONFIG does not exist: $KIND_CONFIG" >&2; exit 1; }
  CLUSTER_CONFIG="$KIND_CONFIG"
  # Each of the two blocks below extends whatever CLUSTER_CONFIG currently points at (the raw
  # AIOpsLab config, or the previous block's own patched temp file) into a NEW temp file and
  # reassigns CLUSTER_CONFIG: a chain, not a fork, so both patches can compose on the same
  # cluster (AGENT_INSTALL=1 CALICO_INSTALL=1 together is a real, tested combination: see
  # tests/triage_eval/test_setup_cluster_env.py). All temp files collected in
  # TMP_KIND_CONFIGS and cleaned up by a single EXIT trap: setting `trap ... EXIT` a second time
  # would silently replace the first trap instead of running both, which is why this isn't two
  # separate `trap 'rm -f "$X"' EXIT` calls like earlier revisions of this file used.
  TMP_KIND_CONFIGS=()
  _cleanup_tmp_kind_configs() { rm -f "${TMP_KIND_CONFIGS[@]:-}"; }
  trap _cleanup_tmp_kind_configs EXIT
  # Only when someone asked for the agent's dedicated node (AGENT_INSTALL=1) do we extend
  # AIOpsLab's own kind config in memory with the extra worker, never on a plain `setup_cluster.sh` run, and never by writing a forked copy of that config
  # into this repo (see kind/patch_kind_config_agent_node.py's docstring for why: that file
  # lives in $AIOPSLAB_HOME, an external clone this repo doesn't own, and a hand-duplicated copy
  # here would drift the moment AIOpsLab changes its own config upstream).
  if [[ "${AGENT_INSTALL:-0}" == "1" ]]; then
    echo "[setup] AGENT_INSTALL=1: extending $KIND_CONFIG with the agent's dedicated worker (role=sslm-agent + taint)"
    NEXT_CONFIG="$(mktemp)"
    TMP_KIND_CONFIGS+=("$NEXT_CONFIG")
    python3 "$SCRIPT_DIR/kind/patch_kind_config_agent_node.py" < "$CLUSTER_CONFIG" > "$NEXT_CONFIG"
    CLUSTER_CONFIG="$NEXT_CONFIG"
  fi
  # Close the "NetworkPolicy declared but not enforced" limit (k8s/networkpolicy.yaml's own
  # comment, README.md). kind's default CNI (kindnet) never
  # enforces NetworkPolicy; disabling it only takes effect AT CLUSTER CREATION
  # (networking.disableDefaultCNI, kind's own contract), so this has to live in this same
  # creation branch, extended in memory (never a forked copy) the same way as the agent-node
  # patch above (see kind/patch_kind_config_disable_default_cni.py's docstring). Separate flag
  # from AGENT_INSTALL on purpose: see the CALICO_INSTALL block further down for why.
  if [[ "${CALICO_INSTALL:-0}" == "1" ]]; then
    echo "[setup] CALICO_INSTALL=1: disabling kind's default CNI (networking.disableDefaultCNI=true) to install Calico"
    NEXT_CONFIG="$(mktemp)"
    TMP_KIND_CONFIGS+=("$NEXT_CONFIG")
    python3 "$SCRIPT_DIR/kind/patch_kind_config_disable_default_cni.py" < "$CLUSTER_CONFIG" > "$NEXT_CONFIG"
    CLUSTER_CONFIG="$NEXT_CONFIG"
  fi
  echo "[setup] creating cluster '$CLUSTER_NAME' ($CLUSTER_CONFIG)"
  kind create cluster --name "$CLUSTER_NAME" --config "$CLUSTER_CONFIG"
  CLUSTER_JUST_CREATED=1
else
  echo "[setup] cluster '$CLUSTER_NAME' already exists, skipping kind create"
fi

MANIFESTS=(kube-controller-manager.yaml kube-scheduler.yaml)
# Export BEFORE the python subshell: without it the defaults set above (not exported by the
# caller) never reach os.environ and the script dies with KeyError. Measured on the first
# standalone run; until then this script had only ever been called with the vars exported.
export LEASE_DURATION RENEW_DEADLINE RETRY_PERIOD
DURATIONS_JSON=$(python3 -c "
import json, os
print(json.dumps({
    'leader-elect-lease-duration': os.environ['LEASE_DURATION'],
    'leader-elect-renew-deadline': os.environ['RENEW_DEADLINE'],
    'leader-elect-retry-period': os.environ['RETRY_PERIOD'],
}))
")

mapfile -t CONTROL_PLANE_CONTAINERS < <(docker ps --filter "name=${CLUSTER_NAME}-control-plane" --format '{{.Names}}')
if [[ ${#CONTROL_PLANE_CONTAINERS[@]} -eq 0 ]]; then
  echo "error: no '${CLUSTER_NAME}-control-plane*' container running" >&2
  exit 1
fi

for container in "${CONTROL_PLANE_CONTAINERS[@]}"; do
  for manifest in "${MANIFESTS[@]}"; do
    path="/etc/kubernetes/manifests/${manifest}"
    echo "[setup] $container:$path -> lease=${LEASE_DURATION} renew-deadline=${RENEW_DEADLINE} retry-period=${RETRY_PERIOD}"
    before="$(docker exec "$container" cat "$path")"
    after="$(printf '%s' "$before" | python3 "$PATCH_SCRIPT" --durations "$DURATIONS_JSON")"
    if [[ "$before" == "$after" ]]; then
      echo "[setup]   already applied, no change"
      continue
    fi
    printf '%s' "$after" | docker exec -i "$container" sh -c "cat > $path"
  done
done

# openebs-localpv-provisioner: turn off leader election on the single replica and clean up
# orphaned helpers. Why: the provisioner runs 1 replica with leader election on by default;
# with this kind cluster's slow etcd (the same I/O pressure as the lease patch above) it loses
# the lease, exits 255 and leaves an orphaned Completed init-pvc-* pod. The next provisioning
# of the same PVC hits `pods "init-pvc-..." already exists` and hangs, and the whole deploy
# stays Pending until the wait times out. Measured: 104 provisioner restarts in ~3 days, the
# root cause of almost every stuck deploy during probe admission. With no second replica there
# is nobody to elect: LEADER_ELECTION_ENABLED=false removes the failure mode.
# Ephemeral (lost on `kind delete`), so it lives here next to the lease patch. Idempotent, and a
# no-op when openebs is not up yet (AIOpsLab deploys it per episode).
if kubectl -n openebs get deploy openebs-localpv-provisioner >/dev/null 2>&1; then
  current="$(kubectl -n openebs get deploy openebs-localpv-provisioner \
    -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="LEADER_ELECTION_ENABLED")].value}' 2>/dev/null || true)"
  if [[ "$current" == "false" ]]; then
    echo "[setup] openebs provisioner: leader election already disabled, no change"
  else
    echo "[setup] openebs provisioner: turning off leader election (single replica)"
    kubectl -n openebs set env deploy/openebs-localpv-provisioner LEADER_ELECTION_ENABLED=false >/dev/null
    kubectl -n openebs rollout status deploy/openebs-localpv-provisioner --timeout=180s >/dev/null 2>&1 || true
  fi
  orphans="$(kubectl -n openebs get pods --no-headers 2>/dev/null \
    | awk '$3=="Completed" && $1 ~ /^init-pvc-/ {print $1}')"
  if [[ -n "$orphans" ]]; then
    echo "[setup] openebs provisioner: removing $(echo "$orphans" | wc -l | tr -d ' ') orphaned init-pvc helper(s)"
    echo "$orphans" | xargs -n1 -r kubectl -n openebs delete pod >/dev/null 2>&1 || true
  fi
else
  echo "[setup] openebs provisioner does not exist yet (deployed per episode), skipping the leader election patch"
fi


# When Calico replaces kindnet, it must make the nodes Ready before Helm waits for chart pods.
# Keep the monitoring installation below the Calico block for that creation-time combination.
KPS_INSTALL="${KPS_INSTALL:-0}"

# Calico CNI: closes the gap tracked in k8s/networkpolicy.yaml's own comment and README.md.
# kind's default CNI (kindnet) never enforces NetworkPolicy, so without this the agent's
# zero-external-egress policy is a correct declaration, not a tested control.
#
# Opt-in via its OWN flag (CALICO_INSTALL=1), deliberately NOT tied to AGENT_INSTALL=1, for two
# reasons: (1) disabling kind's default CNI only takes effect AT CLUSTER CREATION, but
# AGENT_INSTALL=1 is explicitly designed to also run against an already-existing cluster (see its
# own node-check preflight below); tying the two together would force every quick AGENT_INSTALL=1
# iteration against an existing dev cluster to also carry a from-creation-only requirement it
# doesn't share; (2) this same kind cluster is also used for plain AIOpsLab development/testing
# that never touches the sslm-agent at all, and disabling the default CNI without immediately
# installing a replacement breaks ALL pod-to-pod networking on the cluster, exactly the failure
# mode KPS_INSTALL's own comment above already warns about for a different default-on risk. A
# separate, explicit flag keeps a plain `setup_cluster.sh` run (no env vars at all) fully
# untouched, matching every other opt-in section in this script.
#
# CALICO_VERSION is pinned (default v3.28.2), never "latest": same discipline as
# KPS_CHART_VERSION above. The manifest itself is vendored under kind/vendor/ rather than fetched
# from raw.githubusercontent.com at apply time: what a given CALICO_VERSION actually installs is
# then exactly what's committed and reviewed in this repo, not whatever GitHub happens to serve
# that day (the upstream file at a given tag doesn't change, but relying on network reachability
# to a third party for a security-relevant control being installed is a limit this repo doesn't
# need to carry).
CALICO_INSTALL="${CALICO_INSTALL:-0}"
if [[ "$CALICO_INSTALL" == "1" ]]; then
  CALICO_VERSION="${CALICO_VERSION:-v3.28.2}"
  CALICO_MANIFEST="$SCRIPT_DIR/kind/vendor/calico-${CALICO_VERSION}.yaml"
  [[ -f "$CALICO_MANIFEST" ]] || {
    echo "error: CALICO_INSTALL=1 but there is no vendored manifest for CALICO_VERSION=$CALICO_VERSION (expected at $CALICO_MANIFEST)" >&2
    exit 1
  }

  if kubectl -n kube-system get daemonset calico-node >/dev/null 2>&1; then
    echo "[setup] calico-node already exists, skipping the manifest apply"
  else
    # Preflight (same shape as AGENT_INSTALL's own node-check below): disableDefaultCNI only took
    # effect if THIS run is the one that created the cluster. A cluster that already existed
    # before CALICO_INSTALL=1 was set still has kindnet running; installing Calico on top would
    # leave two CNIs fighting over the same pods, never a clean enforced NetworkPolicy.
    if [[ "$CLUSTER_JUST_CREATED" != "1" ]]; then
      echo "error: CALICO_INSTALL=1 but cluster '$CLUSTER_NAME' already existed without Calico. kind's default CNI can only be disabled at cluster CREATION (CALICO_INSTALL=1 must be set from the first 'kind create'); recreate the cluster with 'kind delete cluster --name $CLUSTER_NAME' and run this script again with CALICO_INSTALL=1." >&2
      exit 1
    fi
    echo "[setup] CALICO_INSTALL=1: applying Calico $CALICO_VERSION (vendored manifest, kind/vendor/)"
    kubectl apply -f "$CALICO_MANIFEST"
  fi

  echo "[setup] waiting for calico-node to be ready on every node"
  kubectl -n kube-system rollout status daemonset/calico-node --timeout=180s
else
  echo "[setup] CALICO_INSTALL != 1, skipping Calico (export CALICO_INSTALL=1 to enforce NetworkPolicy, see README)"
fi

# kube-prometheus-stack: install/update is idempotent and chart-pinned.
# The frozen symptom rules apply only after the operator is up, otherwise its CRD is absent.
# This remains opt-in because Helm talks to the kubeconfig directly, not through kubectl stubs.
if [[ "$KPS_INSTALL" == "1" ]]; then
  KPS_CHART_VERSION="${KPS_CHART_VERSION:-91.4.1}"
  KPS_NAMESPACE="${KPS_NAMESPACE:-monitoring}"
  # The release name is what the Service names derive from (<release>-alertmanager and
  # <release>-prometheus, confirmed live while label selectors came back empty).
  # scenarios/run_continuous_admission.py reads the same default, so keep the two in sync.
  KPS_RELEASE="${KPS_RELEASE:-sslm-kube-prometheus}"
  KPS_VALUES="$SCRIPT_DIR/kube-prometheus-stack/values.yaml"
  KPS_RULES="$SCRIPT_DIR/kube-prometheus-stack/rules/symptom-alerts.yaml"
  command -v helm >/dev/null 2>&1 || { echo "error: KPS_INSTALL=1 but 'helm' is not on PATH" >&2; exit 1; }
  # timeout guards repo add/update: helm upgrade itself has --timeout, but a slow chart repo
  # would otherwise hang before reaching it. GNU timeout is available on Linux CI/VPS, while
  # a bare BSD userland safely falls back to no guard rather than failing this setup outright.
  TIMEOUT_CMD=()
  command -v timeout >/dev/null 2>&1 && TIMEOUT_CMD=(timeout 60)
  echo "[setup] kube-prometheus-stack: helm repo add/update prometheus-community"
  "${TIMEOUT_CMD[@]}" helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
  "${TIMEOUT_CMD[@]}" helm repo update prometheus-community >/dev/null
  echo "[setup] kube-prometheus-stack: upgrade --install (chart ${KPS_CHART_VERSION}, ns ${KPS_NAMESPACE})"
  kubectl create namespace "$KPS_NAMESPACE" >/dev/null 2>&1 || true
  helm upgrade --install "$KPS_RELEASE" prometheus-community/kube-prometheus-stack \
    --version "$KPS_CHART_VERSION" --namespace "$KPS_NAMESPACE" --values "$KPS_VALUES" --wait --timeout 5m
  echo "[setup] kube-prometheus-stack: kubectl apply of the extra PrometheusRule (symptom-only)"
  kubectl apply -n "$KPS_NAMESPACE" -f "$KPS_RULES"
else
  echo "[setup] KPS_INSTALL != 1, skipping kube-prometheus-stack (export KPS_INSTALL=1 to install)"
fi

# Chaos Mesh (needed by the candidate family container_kill): AIOpsLab's container_kill problem
# drives a Chaos Mesh PodChaos, and its own SymptomFaultInjector
# `helm install`s chaos-mesh 2.6.2 from the problem's constructor, every time, with no --wait
# (aiopslab/generators/fault/inject_symp.py). On a fresh cluster the first PodChaos can then hit
# a controller/webhook that is not Ready; on a warm one the repeated install just fails without
# raising. Installing it here, once, idempotently and waited for, removes that race and makes the
# upstream install a no-op.
#
# Release name, namespace, chart version and the containerd flags are AIOpsLab's own values, NOT
# overridable: a different release name or namespace would let AIOpsLab's install create a second
# Chaos Mesh next to this one, and a different chart version would measure a different fault
# implementation than the benchmark's. Runtime detection mirrors KubeCtl.get_container_runtime
# (first node's containerRuntimeVersion): docker needs no extra flags, containerd (kind) needs the
# daemon runtime and socket path, anything else is refused, same as AIOpsLab.
#
# Opt-in for the same reason as KPS_INSTALL: helm talks to the kubeconfig directly.
CHAOS_MESH_INSTALL="${CHAOS_MESH_INSTALL:-0}"
if [[ "$CHAOS_MESH_INSTALL" == "1" ]]; then
  CHAOS_MESH_RELEASE="chaos-mesh"
  CHAOS_MESH_NAMESPACE="chaos-mesh"
  CHAOS_MESH_CHART_VERSION="2.6.2"
  command -v helm >/dev/null 2>&1 || { echo "error: CHAOS_MESH_INSTALL=1 but 'helm' is not on PATH" >&2; exit 1; }
  _runtime="$(kubectl get nodes -o jsonpath='{.items[0].status.nodeInfo.containerRuntimeVersion}' 2>/dev/null || true)"
  CHAOS_MESH_RUNTIME_ARGS=()
  case "$_runtime" in
    docker*) ;;
    containerd*)
      CHAOS_MESH_RUNTIME_ARGS=(--set chaosDaemon.runtime=containerd --set chaosDaemon.socketPath=/run/containerd/containerd.sock) ;;
    *)
      echo "error: CHAOS_MESH_INSTALL=1 but the node runtime ('${_runtime:-empty}') is neither docker nor containerd (AIOpsLab refuses it too)" >&2
      exit 1 ;;
  esac
  TIMEOUT_CMD=()
  command -v timeout >/dev/null 2>&1 && TIMEOUT_CMD=(timeout 60)
  echo "[setup] chaos-mesh: helm repo add/update chaos-mesh"
  "${TIMEOUT_CMD[@]}" helm repo add chaos-mesh https://charts.chaos-mesh.org >/dev/null 2>&1 || true
  "${TIMEOUT_CMD[@]}" helm repo update chaos-mesh >/dev/null
  echo "[setup] chaos-mesh: upgrade --install (chart ${CHAOS_MESH_CHART_VERSION}, ns ${CHAOS_MESH_NAMESPACE}, runtime ${_runtime})"
  helm upgrade --install "$CHAOS_MESH_RELEASE" chaos-mesh/chaos-mesh \
    --version "$CHAOS_MESH_CHART_VERSION" --namespace "$CHAOS_MESH_NAMESPACE" --create-namespace \
    "${CHAOS_MESH_RUNTIME_ARGS[@]}" --wait --timeout 5m
  # helm --wait already covers the chart's workloads; the explicit rollouts name the two that the
  # container-kill fault needs (the controller reconciles the PodChaos, the per-node daemon does
  # the kill) so a failure says which one never got Ready.
  echo "[setup] chaos-mesh: waiting for chaos-controller-manager and chaos-daemon"
  kubectl -n "$CHAOS_MESH_NAMESPACE" rollout status deploy/chaos-controller-manager --timeout=180s
  kubectl -n "$CHAOS_MESH_NAMESPACE" rollout status daemonset/chaos-daemon --timeout=180s
else
  echo "[setup] CHAOS_MESH_INSTALL != 1, skipping Chaos Mesh (export CHAOS_MESH_INSTALL=1 for the container_kill candidate family)"
fi

# Agent manifests: opt-in by design (explicit AGENT_INSTALL=1), same reasoning as KPS_INSTALL
# above: `kubectl apply` talks to the active kubeconfig, so default-on here could silently apply
# mutation RBAC against the wrong cluster.
#
# APP_NAMESPACE, VLLM_BASE_URL, VLLM_EGRESS_CIDR and VLLM_EGRESS_PORT are required with no
# default: they are facts of the lab and the round (which namespace hotelReservation landed in,
# the VM's Docker gateway, the vLLM port), never something this script should guess (see the
# manifests' own comments in k8s/). Exception (the SSLM_LOG_ONLY block just below): in a round
# without a model the three VLLM_* stop being facts of the round (no vLLM is running) and get a
# default. AGENT_IMAGE and VLLM_API_KEY have safe defaults (local build, lab without --api-key).
AGENT_INSTALL="${AGENT_INSTALL:-0}"
if [[ "$AGENT_INSTALL" == "1" ]]; then
  : "${APP_NAMESPACE:?error: AGENT_INSTALL=1 requires APP_NAMESPACE (the target app namespace, e.g. test-hotel-reservation, read from AIOpsLab metadata by scenarios/resolve_app_namespace.py, never guessed)}"
  # Observed: `hotelReservation` (the app name, not the namespace) got through until kubectl
  # refused it. Fail here, before any apply, when it is not an RFC 1123 label.
  if [[ ! "$APP_NAMESPACE" =~ ^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$ ]]; then
    echo "error: APP_NAMESPACE='$APP_NAMESPACE' is not a valid Kubernetes namespace (RFC 1123)" >&2
    exit 1
  fi

  # SSLM_LOG_ONLY: read HERE, before the three VLLM_* checks below, because it changes whether
  # they are required or get a default. entrypoint.py's _build_episode_runner already argues
  # that VLLM_BASE_URL should not be OMITTED in a round without a model ("preferable to
  # omitting VLLM_BASE_URL: setup_cluster deliberately requires that endpoint to keep the
  # Deployment and NetworkPolicy contract identical between rounds"), and this keeps exactly
  # that: the Deployment/NetworkPolicy still carry the three fields, always with a value, never a
  # simplified gate that drops them. The simplification is that the OPERATOR does not have to
  # invent an address for a server that by design is never called (SSLM_LOG_ONLY=1 makes
  # entrypoint.py fall back to log_only_episode_runner whatever VLLM_BASE_URL points at). A round
  # with a real model (SSLM_LOG_ONLY unset/0) keeps the three required with no default.
  SSLM_LOG_ONLY="${SSLM_LOG_ONLY:-0}"
  # COMPARATOR_ARM (the hosted-API comparator arm): read here for the same reason as
  # SSLM_LOG_ONLY, it changes what is required. Only 0 or 1: the pod reads the same value
  # (entrypoint.py) and refuses anything else, so a stray "true" fails here, before any apply,
  # not after a 180s rollout.
  COMPARATOR_ARM="${COMPARATOR_ARM:-0}"
  if [[ "$COMPARATOR_ARM" != "0" && "$COMPARATOR_ARM" != "1" ]]; then
    echo "error: COMPARATOR_ARM='$COMPARATOR_ARM' must be 0 or 1" >&2
    exit 1
  fi
  # Whole-call prompt budget: comes from the caller, empty by default (the prompt earlier rounds
  # used). The comparator arm only takes it for the short-context hosted-model run; without it
  # the API runs keep the 30k prompt. Checked here, before any apply: the pod refuses to start
  # with a bad value too, but only after the rollout wait.
  SSLM_PROMPT_BUDGET_CHARS="${SSLM_PROMPT_BUDGET_CHARS:-}"
  if [[ -n "$SSLM_PROMPT_BUDGET_CHARS" ]]; then
    if [[ ! "$SSLM_PROMPT_BUDGET_CHARS" =~ ^[0-9]+$ ]] || (( SSLM_PROMPT_BUDGET_CHARS < 4000 )); then
      echo "error: SSLM_PROMPT_BUDGET_CHARS must be an integer >= 4000 (chars), got '$SSLM_PROMPT_BUDGET_CHARS'" >&2
      exit 1
    fi
  fi
  # Tool contract: comes from the caller, empty by default (the prompt earlier rounds used). Only
  # "on" turns it on; "off" and empty leave the prompt as it was. The comparator arm never takes
  # it, since the API runs ran without it. Checked here, before any apply, because the pod only
  # refuses a bad value after the rollout wait.
  SSLM_TOOL_CONTRACT="${SSLM_TOOL_CONTRACT:-}"
  if [[ -n "$SSLM_TOOL_CONTRACT" ]]; then
    if [[ "$SSLM_TOOL_CONTRACT" != "on" && "$SSLM_TOOL_CONTRACT" != "off" ]]; then
      echo "error: SSLM_TOOL_CONTRACT must be on or off, got '$SSLM_TOOL_CONTRACT'" >&2
      exit 1
    fi
    if [[ "$COMPARATOR_ARM" == "1" && "$SSLM_TOOL_CONTRACT" == "on" ]]; then
      echo "error: SSLM_TOOL_CONTRACT=on is not valid on the comparator arm (the API runs ran without it)" >&2
      exit 1
    fi
  fi
  # Served model and request dialect for the vLLM arm: empty by default, so the pod stays on
  # Qwen2.5-7B-Instruct with guided_json, the request every earlier round used. An unknown dialect
  # stops here: a newer vLLM ignores guided_json and decodes without the schema, with no error at
  # all. The comparator arm does not talk to vLLM, so it takes neither.
  VLLM_MODEL="${VLLM_MODEL:-}"
  VLLM_DECODING_DIALECT="${VLLM_DECODING_DIALECT:-}"
  if [[ -n "$VLLM_DECODING_DIALECT" && "$VLLM_DECODING_DIALECT" != "guided_json" && "$VLLM_DECODING_DIALECT" != "structured_outputs" ]]; then
    echo "error: VLLM_DECODING_DIALECT='$VLLM_DECODING_DIALECT' must be guided_json or structured_outputs" >&2
    exit 1
  fi
  if [[ "$COMPARATOR_ARM" == "1" && ( -n "$VLLM_MODEL" || -n "$VLLM_DECODING_DIALECT" ) ]]; then
    echo "error: VLLM_MODEL/VLLM_DECODING_DIALECT are not valid on the comparator arm (it does not talk to vLLM)" >&2
    exit 1
  fi
  SSLM_COMPARATOR_MAX_COST_USD="${SSLM_COMPARATOR_MAX_COST_USD:-}"
  if [[ "$COMPARATOR_ARM" == "1" ]]; then
    # The API arm talks to no vLLM: its NetworkPolicy has no route to vLLM and the entrypoint
    # builds the Anthropic client. The VLLM_* are no longer required and go empty into the
    # Deployment. Instead the key and the spend cap are required here, before any apply: the pod
    # refuses to start without them too, but only after the rollout waits up to 180s.
    if [[ "$SSLM_LOG_ONLY" == "1" ]]; then
      echo "error: COMPARATOR_ARM=1 cannot be combined with SSLM_LOG_ONLY=1 (the API arm exists to call the API)" >&2
      exit 1
    fi
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] || {
      echo "error: COMPARATOR_ARM=1 requires ANTHROPIC_API_KEY in the environment" >&2
      exit 1
    }
    python3 -c 'import math, sys; v = float(sys.argv[1]); sys.exit(0 if math.isfinite(v) and v > 0 else 1)' \
      "$SSLM_COMPARATOR_MAX_COST_USD" 2>/dev/null || {
      echo "error: COMPARATOR_ARM=1 requires SSLM_COMPARATOR_MAX_COST_USD > 0 (spend cap per pod, one pod per seed), got '$SSLM_COMPARATOR_MAX_COST_USD'" >&2
      exit 1
    }
    VLLM_BASE_URL=""
    VLLM_API_KEY=""
    VLLM_EGRESS_CIDR=""
    VLLM_EGRESS_PORT=""
    echo "[setup] COMPARATOR_ARM=1: API arm, no vLLM (VLLM_* empty), spend cap per pod US\$${SSLM_COMPARATOR_MAX_COST_USD}"
  elif [[ "$SSLM_LOG_ONLY" == "1" ]]; then
    # VLLM_EGRESS_PORT: 8000 is vLLM's own default serving port, not a number invented by this
    # script.
    VLLM_EGRESS_PORT="${VLLM_EGRESS_PORT:-8000}"
    if [[ -z "${VLLM_EGRESS_CIDR:-}" ]]; then
      # Same gateway the README tells the operator to derive by hand with
      # `docker network inspect kind -> .Gateway`; kind uses a single docker network named "kind"
      # for every cluster, whatever CLUSTER_NAME is, so the name is literal.
      #
      # Bug (observed on Docker Desktop for Apple Silicon): `index .IPAM.Config 0` assumed Docker
      # lists IPv4 IPAM entries first, which is false on a dual-stack (IPv4+IPv6) "kind" network.
      # Index 0 can land on the IPv6 entry and derive VLLM_EGRESS_CIDR="fc00::1/32", invalid both
      # as a CIDR (an IPv6 host needs /128, not /32) and downstream (VLLM_BASE_URL and the
      # NetworkPolicy ipBlock assume IPv4).
      #
      # Fix: `range` over ALL entries, one per line (`{{println .Gateway}}`), then keep the first
      # line that looks like IPv4 (no ':'). A plain `{{range .IPAM.Config}}{{.Gateway}}{{end}}`
      # concatenates the entries WITHOUT a separator: on a dual-stack network that yields
      # "172.19.0.1fe80::1", not empty, so it would pass the "-n" check below as a corrupt CIDR
      # instead of failing loudly. `println` gives one line per entry, so we filter by content
      # instead of trusting position: correct however Docker orders the entries, and still correct
      # on a single-stack (IPv4-only, the common case) "kind" network, where the loop finds the
      # only line there is.
      _vllm_gateways="$(docker network inspect kind -f '{{range .IPAM.Config}}{{println .Gateway}}{{end}}' 2>/dev/null || true)"
      _vllm_gateway=""
      while IFS= read -r _candidate; do
        if [[ -n "$_candidate" && "$_candidate" != *:* ]]; then
          _vllm_gateway="$_candidate"
          break
        fi
      done <<<"$_vllm_gateways"
      [[ -n "$_vllm_gateway" ]] || {
        echo "error: SSLM_LOG_ONLY=1 but could not derive an IPv4 gateway via 'docker network inspect kind' (network empty/unreachable, or IPv6 entries only); export VLLM_EGRESS_CIDR by hand" >&2
        exit 1
      }
      VLLM_EGRESS_CIDR="${_vllm_gateway}/32"
    fi
    VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_EGRESS_CIDR%%/*}:${VLLM_EGRESS_PORT}/v1}"
    echo "[setup] SSLM_LOG_ONLY=1: no real model this round, using vLLM egress defaults (no real call happens, entrypoint.py falls back to log_only_episode_runner): VLLM_EGRESS_CIDR=$VLLM_EGRESS_CIDR VLLM_EGRESS_PORT=$VLLM_EGRESS_PORT VLLM_BASE_URL=$VLLM_BASE_URL"
  else
    : "${VLLM_BASE_URL:?error: AGENT_INSTALL=1 requires VLLM_BASE_URL (same contract as VllmModelClient; export SSLM_LOG_ONLY=1 for a round without a model, which has a default for it)}"
    : "${VLLM_EGRESS_CIDR:?error: AGENT_INSTALL=1 requires VLLM_EGRESS_CIDR (the VM's docker gateway; see 'docker network inspect kind'; export SSLM_LOG_ONLY=1 for a round without a model, which has a default for it)}"
    : "${VLLM_EGRESS_PORT:?error: AGENT_INSTALL=1 requires VLLM_EGRESS_PORT (the round's vLLM port, never hardcoded; export SSLM_LOG_ONLY=1 for a round without a model, which has a default for it)}"
  fi

  AGENT_IMAGE="${AGENT_IMAGE:-sslm-agent:local}"
  VLLM_API_KEY="${VLLM_API_KEY:-}"
  # Operational decision (made once rollback_state/read-back existed): a deployment applied through this script should run C2 for real by default, not dry-run,
  # once it is actually exercising the real orchestrator. Override with SSLM_C2_DRY_RUN=1 (or
  # any value harness/episode_runner.py's c2_dry_run_enabled() does NOT recognize as "off") to
  # force dry-run for a given AGENT_INSTALL=1 run.
  #
  # entrypoint.py's _build_episode_runner, the only code path that reads SSLM_C2_DRY_RUN, only
  # builds the real orchestrator when APP_NAMESPACE
  # and PROMETHEUS_URL are ALSO present in the deployed pod's env. Both are now exported and
  # substituted into deployment.yaml below, so this default is live for every AGENT_INSTALL=1 run.
  SSLM_C2_DRY_RUN="${SSLM_C2_DRY_RUN:-0}"
  # SSLM_LOG_ONLY already read further up, before the VLLM_* checks.
  # Same default as the KPS_INSTALL=1 block above (KPS_NAMESPACE), not re-declared there: the
  # NetworkPolicy's Prometheus/Alertmanager selectors must track whatever
  # namespace kube-prometheus-stack actually landed in, or the policy silently blocks its own
  # intended egress/ingress the moment someone overrides KPS_NAMESPACE.
  KPS_NAMESPACE="${KPS_NAMESPACE:-monitoring}"
  # PROMETHEUS_URL: derived from KPS_NAMESPACE, never a separate required var: the in-cluster
  # Prometheus this agent talks to is whichever one KPS_INSTALL=1 put up in
  # that namespace, so a value independent of KPS_NAMESPACE could silently point at the wrong
  # namespace the moment someone overrides it (same failure mode the KPS_NAMESPACE comment above
  # already guards against for the NetworkPolicy). "prometheus-operated" is not this chart's own
  # Service (templates/prometheus/service.yaml names that "<fullname>-prometheus"); it is the
  # separate headless Service the Prometheus Operator itself always creates for peer discovery,
  # fixed name "prometheus-operated" on port 9090 regardless of the Prometheus CR's name
  # (confirmed against the chart's own bundled CRD docs and values.yaml comment/example, both of
  # which cite this exact host:port form, when pulling the pinned chart version to verify the
  # alertmanager.config route wiring above). Overridable for a non-standard install, never
  # hardcoded past this default.
  PROMETHEUS_URL="${PROMETHEUS_URL:-http://prometheus-operated.${KPS_NAMESPACE}.svc.cluster.local:9090}"

  command -v envsubst >/dev/null 2>&1 || { echo "error: AGENT_INSTALL=1 but 'envsubst' is not on PATH" >&2; exit 1; }

  # Preflight: the dedicated worker (role=sslm-agent) only joins the cluster at CREATION (the
  # block above that calls patch_kind_config_agent_node.py, inside `if ! kind get clusters ...`).
  # Running AGENT_INSTALL=1 against a cluster that already existed without that node leaves the
  # Deployment Pending forever (nodeSelector with no match), and the only symptom would be the
  # opaque 180s timeout of the rollout status below. Fail early, with a clear message, instead.
  if ! kubectl get nodes -l role=sslm-agent --no-headers 2>/dev/null | grep -q .; then
    echo "error: no node labeled role=sslm-agent in this cluster. The dedicated worker is only added at cluster CREATION (AGENT_INSTALL=1 must be set from the first 'kind create'); recreate the cluster with 'kind delete cluster --name $CLUSTER_NAME' and run this script again." >&2
    exit 1
  fi

  # APISERVER_EGRESS_CIDR / APISERVER_EGRESS_PORT: the
  # NetworkPolicy's kube-apiserver egress rule cannot target the `kubernetes` Service's ClusterIP
  # via ipBlock, verified against a real kind+Calico cluster (see networkpolicy.yaml's own
  # comment and README.md for the full story). kube-proxy's Service DNAT rewrites the destination
  # to the apiserver's real backend address BEFORE Calico's per-workload egress policy evaluates
  # the packet, so the rule has to name that real backend instead. Unlike VLLM_EGRESS_CIDR (which
  # depends on the kind Docker network's gateway, something this script cannot see from inside the
  # cluster), the apiserver's real backend IS visible from inside the cluster (the `kubernetes`
  # Service's own Endpoints object always carries it), so this is derived live by default, no
  # manual export needed in the common case. Overridable (a non-standard cluster's apiserver
  # reachability might legitimately differ), same shape as every other placeholder here. Derived
  # AFTER the node-check preflight above (not before): a cluster that fails that check exits
  # before ever needing a live kubectl fact, same discipline as every other check in this block.
  if [[ -z "${APISERVER_EGRESS_CIDR:-}" ]]; then
    _apiserver_ip="$(kubectl get endpoints kubernetes -o jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null || true)"
    [[ -n "$_apiserver_ip" ]] || {
      echo "error: could not derive the kube-apiserver IP via 'kubectl get endpoints kubernetes' (empty or unreachable); export APISERVER_EGRESS_CIDR by hand" >&2
      exit 1
    }
    APISERVER_EGRESS_CIDR="${_apiserver_ip}/32"
  fi
  if [[ -z "${APISERVER_EGRESS_PORT:-}" ]]; then
    APISERVER_EGRESS_PORT="$(kubectl get endpoints kubernetes -o jsonpath='{.subsets[0].ports[0].port}' 2>/dev/null || true)"
    [[ -n "$APISERVER_EGRESS_PORT" ]] || {
      echo "error: could not derive the kube-apiserver port via 'kubectl get endpoints kubernetes' (empty or unreachable); export APISERVER_EGRESS_PORT by hand" >&2
      exit 1
    }
  fi

  K8S_DIR="$SCRIPT_DIR/k8s"
  export APP_NAMESPACE AGENT_IMAGE VLLM_BASE_URL VLLM_API_KEY VLLM_EGRESS_CIDR VLLM_EGRESS_PORT KPS_NAMESPACE SSLM_C2_DRY_RUN SSLM_LOG_ONLY PROMETHEUS_URL APISERVER_EGRESS_CIDR APISERVER_EGRESS_PORT COMPARATOR_ARM SSLM_COMPARATOR_MAX_COST_USD

  echo "[setup] sslm-agent: namespace + SA + cluster-wide RBAC (agent ns: sslm-agent)"
  kubectl apply -f "$K8S_DIR/namespace.yaml"
  kubectl apply -f "$K8S_DIR/serviceaccount.yaml"
  kubectl apply -f "$K8S_DIR/clusterrole-read-only.yaml"
  kubectl apply -f "$K8S_DIR/clusterrolebinding-read-only.yaml"

  # AIOpsLab creates its target namespace lazily when the first episode constructs the
  # application. The agent's namespaced Role and RoleBinding must exist before that first
  # fault, so materialize the otherwise empty namespace here. `apply` keeps this compatible
  # with an app that was already deployed by an earlier run.
  kubectl create namespace "$APP_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

  echo "[setup] sslm-agent: mutation RBAC (C1/C2) in the app namespace ($APP_NAMESPACE)"
  envsubst '${APP_NAMESPACE}' < "$K8S_DIR/role-app-mutate.yaml" | kubectl apply -f -
  envsubst '${APP_NAMESPACE}' < "$K8S_DIR/rolebinding-app-mutate.yaml" | kubectl apply -f -

  # Second, independent layer for the DB-pod exclusion: no
  # placeholder substitution needed, this policy matches by pod name pattern (mongo/memcached),
  # not by namespace, so it applies cluster-wide as-is. See the manifest's own comment for why
  # native ValidatingAdmissionPolicy over Kyverno/OPA Gatekeeper.
  echo "[setup] sslm-agent: ValidatingAdmissionPolicy (second layer, database pod exclusion)"
  kubectl apply -f "$K8S_DIR/validatingadmissionpolicy-deny-db-pod.yaml"

  # A locally built image only exists in the host's docker, never inside the kind nodes: the
  # Deployment would sit in ImagePullBackOff (IfNotPresent, no registry has it). `auto` loads it
  # when the tag is the local build's (`:local`); an image from a registry must be immutable
  # (digest or versioned tag) and is pulled normally. AGENT_KIND_LOAD=1/0 forces either way.
  AGENT_KIND_LOAD="${AGENT_KIND_LOAD:-auto}"
  if [[ "$AGENT_KIND_LOAD" == "1" || ( "$AGENT_KIND_LOAD" == "auto" && "$AGENT_IMAGE" == *:local ) ]]; then
    echo "[setup] sslm-agent: kind load docker-image $AGENT_IMAGE --name $CLUSTER_NAME"
    kind load docker-image "$AGENT_IMAGE" --name "$CLUSTER_NAME"
  fi

  # API arm key: only in a Secret, never in the Deployment's envsubst (it would become text in the
  # object and in the apply log). It goes to kubectl over stdin (printf is a builtin, it does not
  # show in ps) and no kubectl output here is printed. Delete + create, never apply: apply would
  # keep a second copy of the value in the last-applied-configuration annotation. On the vLLM arm
  # the Secret is deleted, so a key from an earlier run on this cluster never reaches a pod that
  # does not use it.
  if [[ "$COMPARATOR_ARM" == "1" ]]; then
    echo "[setup] COMPARATOR_ARM=1: Secret sslm-agent-anthropic (ANTHROPIC_API_KEY; value never printed)"
    kubectl -n sslm-agent delete secret sslm-agent-anthropic --ignore-not-found >/dev/null
    printf '%s' "$ANTHROPIC_API_KEY" \
      | kubectl -n sslm-agent create secret generic sslm-agent-anthropic --from-file=ANTHROPIC_API_KEY=/dev/stdin >/dev/null
  else
    kubectl -n sslm-agent delete secret sslm-agent-anthropic --ignore-not-found >/dev/null
  fi

  # History budget: on for the default agent, off only on the comparator arm (the API arm sends
  # the same prompt the earlier vLLM rounds sent). Never taken from the caller, so the default
  # agent cannot inherit a stray "off" left in the shell.
  if [[ "${COMPARATOR_ARM:-0}" == "1" ]]; then
    SSLM_HISTORY_BUDGET=off
  else
    SSLM_HISTORY_BUDGET=on
  fi
  export SSLM_HISTORY_BUDGET
  export SSLM_PROMPT_BUDGET_CHARS SSLM_TOOL_CONTRACT VLLM_MODEL VLLM_DECODING_DIALECT
  echo "[setup] sslm-agent: Deployment (image $AGENT_IMAGE, SSLM_HISTORY_BUDGET=$SSLM_HISTORY_BUDGET, SSLM_PROMPT_BUDGET_CHARS=${SSLM_PROMPT_BUDGET_CHARS:-off}, SSLM_TOOL_CONTRACT=${SSLM_TOOL_CONTRACT:-off}, VLLM_MODEL=${VLLM_MODEL:-default}, VLLM_DECODING_DIALECT=${VLLM_DECODING_DIALECT:-guided_json}) + Service + NetworkPolicy"
  envsubst '${AGENT_IMAGE} ${VLLM_BASE_URL} ${VLLM_API_KEY} ${SSLM_C2_DRY_RUN} ${SSLM_LOG_ONLY} ${APP_NAMESPACE} ${PROMETHEUS_URL} ${SSLM_HISTORY_BUDGET} ${SSLM_PROMPT_BUDGET_CHARS} ${SSLM_TOOL_CONTRACT} ${VLLM_MODEL} ${VLLM_DECODING_DIALECT} ${COMPARATOR_ARM} ${SSLM_COMPARATOR_MAX_COST_USD}' < "$K8S_DIR/deployment.yaml" | kubectl apply -f -
  kubectl apply -f "$K8S_DIR/service.yaml"
  if [[ "${COMPARATOR_ARM:-0}" == "1" ]]; then
    # Comparator arm: same NetworkPolicy, but the vLLM egress becomes egress to the Anthropic API,
    # and only that (the one difference between arms). The default range is the one Anthropic
    # documents for api.anthropic.com; NetworkPolicy matches IPs, not names, so check it before a
    # paid round and override ANTHROPIC_EGRESS_CIDR if it changed.
    ANTHROPIC_EGRESS_CIDR="${ANTHROPIC_EGRESS_CIDR:-160.79.104.0/23}"
    export ANTHROPIC_EGRESS_CIDR
    echo "[setup] COMPARATOR_ARM=1: comparator arm NetworkPolicy (HTTPS egress only to $ANTHROPIC_EGRESS_CIDR, no vLLM)"
    # Warns only (never fails): a stale range does not break setup, it only makes every call
    # from the pod time out.
    python3 "$SCRIPT_DIR/check_egress_cidr.py" api.anthropic.com "$ANTHROPIC_EGRESS_CIDR" || true
    envsubst '${ANTHROPIC_EGRESS_CIDR} ${KPS_NAMESPACE} ${APISERVER_EGRESS_CIDR} ${APISERVER_EGRESS_PORT}' < "$K8S_DIR/networkpolicy-comparator.yaml" | kubectl apply -f -
  else
    envsubst '${VLLM_EGRESS_CIDR} ${VLLM_EGRESS_PORT} ${KPS_NAMESPACE} ${APISERVER_EGRESS_CIDR} ${APISERVER_EGRESS_PORT}' < "$K8S_DIR/networkpolicy.yaml" | kubectl apply -f -
  fi

  kubectl -n sslm-agent rollout status deploy/sslm-agent --timeout=180s

  # The agent's tools are binaries it runs inside the pod. Without them every command exits -1
  # and the model triages from the alert text alone (found after three full rounds had run that
  # way). Fail here, before any paid round, never after the scorer.
  echo "[setup] sslm-agent: agent tools run inside the pod (kubectl get pods -n $APP_NAMESPACE, wget)"
  kubectl -n sslm-agent exec deploy/sslm-agent -- kubectl get pods -n "$APP_NAMESPACE" -o name >/dev/null \
    || { echo "error: the agent cannot run kubectl inside the pod (binary missing from the image, or RBAC); every action would exit -1" >&2; exit 1; }
  kubectl -n sslm-agent exec deploy/sslm-agent -- wget --version >/dev/null \
    || { echo "error: wget missing from the agent image" >&2; exit 1; }
else
  echo "[setup] AGENT_INSTALL != 1, skipping sslm-agent manifests (export AGENT_INSTALL=1 to install)"
fi

echo "[setup] done. kubelet re-applies the manifests on its own (file poll); check with:"
echo "  kubectl -n kube-system get pods -l component=kube-controller-manager -w"
