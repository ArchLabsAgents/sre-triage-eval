#!/usr/bin/env bash
# Runs the full matrix: 7 scenarios x 3 configs = 21 real episodes (the 7th scenario,
# wrong_bin_usage, was added later; earlier runs were 6x3).
#
# Resumable by design: a (scenario, config) whose JSON already exists under scenarios/results/
# is skipped, so interrupting and re-running continues where it stopped. One failed episode
# does not take the matrix down: it logs [FAIL] and moves on to the next.
#
# Loop order is not cosmetic: config in the outer loop keeps config_c's episodes (episodic
# memory) in a defined sequence, which is part of the experiment design (memory accumulates
# BETWEEN episodes).
#
# Scenarios: the 6 mitigation-family problems in the AIOpsLab registry whose app is the
# HotelReservation already deployed in this cluster (auth_miss/scale_pod/k8s_target_port use
# SocialNetwork and would need another deploy and other images). The geo/rate pairs of
# revoke_auth and user_unregistered are deliberate: the same fault class on a different target
# is exactly where config_c (memory) can differ from config_a/b. wrong_bin_usage-mitigation-1
# joined as the 7th scenario because it is the only hotelReservation problem whose native eval
# already checks the root cause, so it calibrates probe vs native eval where both measure the
# same thing.
#
# The real logic (parsing, results, aggregation) lives in run_matrix.py/metrics.py, which have
# tests; this file is only the convenience loop for a long run in a terminal.
#
# Before the matrix it calls setup_cluster.sh (brings the cluster up if needed and reapplies the
# leader-election lease patch). A failure there aborts the whole matrix before any episode.
#
# Usage (interactive terminal on an Apple Silicon Mac, with Metal; do not run alongside other
# memory-heavy builds):
#   SSLM_ENV=mlx-local bash scenarios/run_full_matrix.sh
#   bash scenarios/run_full_matrix.sh --env mlx-local
#
# The env is required: resume is per env (results/<env>/<scenario>_<config>.json), otherwise a
# cloud run would "resume" on top of the local results and the matrix would become a silent mix.
# The cloud VM provisioning prints the SSLM_ENV export line next to the VLLM_BASE_URL one.
set -u

cd "$(dirname "$0")/.."

SSLM_ENV="${SSLM_ENV:-}"
if [[ "${1:-}" == "--env" ]]; then
  SSLM_ENV="${2:-}"
fi
if [[ -z "$SSLM_ENV" ]]; then
  echo "error: set the env for this run (--env <name> or SSLM_ENV=<name>); resume is per env (results/<env>/)" >&2
  exit 1
fi
export SSLM_ENV

# Adapter module:attr pair follows the env's backend (harness.envs / configs/base.yaml), not a
# hardcoded default: mlx_factories constructs an MlxModelClient (refuses a non-mlx env),
# cloud_adapter constructs a VllmModelClient against VLLM_BASE_URL (refuses a non-vllm env).
case "$SSLM_ENV" in
  mlx-local) ADAPTER=mlx_factories ;;
  hyperstack-a6000) ADAPTER=cloud_adapter ;;
  *)
    echo "error: no adapter mapped for env '${SSLM_ENV}' (see the ADAPTER case in run_full_matrix.sh)" >&2
    exit 1
    ;;
esac

export AIOPSLAB_HOME="${AIOPSLAB_HOME:-$(cd ../../tmp/triage-eval/AIOpsLab 2>/dev/null && pwd)}"
if [[ -z "${AIOPSLAB_HOME}" || ! -d "${AIOPSLAB_HOME}/aiopslab" ]]; then
  echo "error: AIOPSLAB_HOME does not point at the AIOpsLab clone" >&2
  exit 1
fi

# setup_cluster.sh brings the cluster up (if needed) and reapplies the leader-election lease
# patch, which is lost on every `kind delete cluster`/`kind create cluster`.
echo "[setup] ensuring cluster + lease patch"
if ! bash setup_cluster.sh; then
  echo "error: setup_cluster.sh failed, aborting the matrix" >&2
  exit 1
fi

SCENARIOS=(
  misconfig_app_hotel_res-mitigation-1
  revoke_auth_mongodb-mitigation-1
  revoke_auth_mongodb-mitigation-2
  user_unregistered_mongodb-mitigation-1
  user_unregistered_mongodb-mitigation-2
  redeploy_without_PV-mitigation-1
  wrong_bin_usage-mitigation-1
)
CONFIGS=(config_a config_b config_c)

mkdir -p "scenarios/results/${SSLM_ENV}"
LOG=scenarios/results/matrix-run.log

# 2026-09-18: the AIOpsLab harness redeploys metrics-server on every episode, and on this
# cluster/instance (kind on a cloud GPU VM) it sometimes never becomes ready (MissingEndpoints).
# A broken v1beta1.metrics.k8s.io APIService stalls the *discovery* step of Kubernetes'
# namespace controller, which blocks DELETION of any namespace while it stays broken, including
# the namespace of the episode that broke it, so finalize() times out. The next episode then
# fails immediately trying to recreate a namespace that was never deleted ("already exists"),
# and it cascades. Measured live twice in the same run on 2026-09-18 (not a one-off); the manual
# workaround was `kubectl delete apiservice v1beta1.metrics.k8s.io` (the stuck namespace then
# finishes deleting within ~15-20s). Running this BEFORE every episode (not just once in
# setup_cluster.sh) heals the state the previous episode left broken before it can block the
# next one; idempotent and a no-op when the APIService is healthy or does not exist yet.
# --request-timeout on both calls: kubectl has no client-side timeout by default, and a silent
# API server (the same flakiness that motivated the heal) would hang the heal indefinitely
# BEFORE any episode, reintroducing the silent stall EPISODE_TIMEOUT_SEC exists to prevent.
heal_broken_metrics_apiservice() {
  local status rc err
  err="$(mktemp)"
  status="$(kubectl --request-timeout=15s get apiservice v1beta1.metrics.k8s.io \
    -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' 2>"$err")"
  rc=$?
  if (( rc != 0 )); then
    # NotFound is expected (fresh cluster, before the first metrics-server deploy); any other
    # get error means the cluster has a problem and must not stay silent.
    if ! grep -qi "notfound" "$err"; then
      echo "[heal] kubectl get apiservice failed (rc=$rc): $(head -c 200 "$err" | tr '\n' ' ')" | tee -a "$LOG"
    fi
    rm -f "$err"
    return 0
  fi
  rm -f "$err"
  if [[ -n "$status" && "$status" != "True" ]]; then
    echo "[heal] APIService v1beta1.metrics.k8s.io not Available (status=$status), removing it so namespace deletion does not stall" | tee -a "$LOG"
    if ! kubectl --request-timeout=15s delete apiservice v1beta1.metrics.k8s.io --ignore-not-found >/dev/null 2>&1; then
      echo "[heal] failed to remove the broken APIService; namespace deletion may stay stuck in this episode" | tee -a "$LOG"
    fi
  fi
}

echo "=== matrix $(date '+%Y-%m-%d %H:%M:%S') [env=${SSLM_ENV}]: ${#SCENARIOS[@]} scenarios x ${#CONFIGS[@]} configs ===" | tee -a "$LOG"
for config in "${CONFIGS[@]}"; do
  for scenario in "${SCENARIOS[@]}"; do
    # The resume skip is PER ENV: it only skips when the result exists for THIS env.
    out="scenarios/results/${SSLM_ENV}/${scenario}_${config}.json"
    if [[ -f "$out" ]]; then
      echo "[skip] $scenario / $config (result already exists in ${SSLM_ENV}/)" | tee -a "$LOG"
      continue
    fi
    heal_broken_metrics_apiservice
    echo "[run ] $scenario / $config ($(date '+%H:%M:%S'))" | tee -a "$LOG"
    # 2026-09-02: without a time cap, a hang in THIRD-PARTY code stalls the whole matrix
    # silently. Measured live: revoke_auth_mongodb-mitigation-1/config_a sat 6h27min in a
    # `while True` with no timeout inside AIOpsLab's own traffic generator
    # (aiopslab/generators/workload/wrk.py, polling a Job that never saw status.ready because the
    # geo pod was in CrashLoopBackOff), with no exception for run_matrix.py to catch.
    # EPISODE_TIMEOUT_SEC caps the WHOLE episode (deploy+traffic+turns+eval), not only the
    # model; --kill-after makes sure a process that ignores SIGTERM (the likeliest cause in the
    # measured case: an interpreter stuck in a syscall inside the k8s client) does not survive
    # the coreutils timeout.
    #
    # 2026-09-03: `timeout` (without --foreground) runs the episode in a BACKGROUND process
    # group. revoke_auth's fault injection runs `kubectl exec -it`, and `-t` calls tcsetattr on
    # the tmux TTY: from a background group that becomes SIGTTOU and kubectl STOPS (state T)
    # forever, with python waiting on it and the episode burning the full 1800s doing nothing
    # (measured live: 26min stopped after "All pods ready"). With stdin on /dev/null there is no
    # TTY to claim: kubectl warns "Unable to use a TTY" and goes on, and the injected script runs
    # the same. No episode reads stdin, so nothing is lost.
    if ! timeout --kill-after=60s "${EPISODE_TIMEOUT_SEC:-1800}" \
        .venv/bin/python scenarios/run_matrix.py \
        --config "configs/${config}.yaml" \
        --env "$SSLM_ENV" \
        --model-client "${ADAPTER}:make_model_client" \
        --environment-factory "${ADAPTER}:make_environment" \
        --scenario "$scenario" </dev/null >>"$LOG" 2>&1; then
      status=$?
      if [[ "$status" == 124 || "$status" == 137 ]]; then
        echo "[FAIL] $scenario / $config (timeout after ${EPISODE_TIMEOUT_SEC:-1800}s, moving on to the next)" | tee -a "$LOG"
      else
        echo "[FAIL] $scenario / $config (moving on to the next; details in the log)" | tee -a "$LOG"
      fi
    fi
  done
done

total=$(ls "scenarios/results/${SSLM_ENV}"/*_config_*.json 2>/dev/null | wc -l | tr -d ' ')
expected=$(( ${#SCENARIOS[@]} * ${#CONFIGS[@]} ))
echo "=== done $(date '+%H:%M:%S') [env=${SSLM_ENV}]: ${total}/${expected} results in scenarios/results/${SSLM_ENV}/ (matrix.csv aggregated) ===" | tee -a "$LOG"
