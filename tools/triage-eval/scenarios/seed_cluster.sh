#!/usr/bin/env bash
# One round-2 seed: its own kind cluster beside up to two others on the same VM (N=3 seeds per
# round), all sharing one vLLM. Runs on the cloud GPU VM (called by that VM's round launcher) and
# on a dev laptop for a local check of the same code path.
#
#   seed_cluster.sh up <index>       create/prepare cluster sslm-r2-s<index> (agent with the model on)
#   seed_cluster.sh launch <index>   start that seed's driver detached (smoke, smokegate or r2, see R2_MODE)
#   seed_cluster.sh env <index>      print the derived names/ports (tests, operators)
#
# COMPARATOR_ARM=1 runs the same seed with the Anthropic API instead of vLLM: `up` needs
# ANTHROPIC_API_KEY and SSLM_COMPARATOR_MAX_COST_USD instead of the VLLM_* endpoint, and the smoke
# proves the agent -> API path by the pod's usage log instead of vLLM's token counter.
#
# Isolation between seeds is by kubeconfig, never by switching a shared current-context: kind
# writes each cluster into its own KUBECONFIG file, so every kubectl/helm subprocess (setup_cluster,
# AIOpsLab's KubeCtl.exec_command, the harvest) talks to its own cluster, and AIOPSLAB_CLUSTER
# points AIOpsLab's python client at the same context. A shared `kubectl config use-context` would
# race between three detached drivers and inject into the wrong cluster without any error.
set -euo pipefail

ACTION="${1:?usage: seed_cluster.sh <up|launch|env> <index 1..3>}"
INDEX="${2:?usage: seed_cluster.sh <up|launch|env> <index 1..3>}"
[[ "$INDEX" =~ ^[1-3]$ ]] || { echo "error: seed index must be 1, 2 or 3 (N=3 seeds per round), got '$INDEX'" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFLIGHT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLUSTER_NAME="sslm-r2-s${INDEX}"
KUBECONFIG="${R2_KUBECONFIG_DIR:-$HOME/.kube}/${CLUSTER_NAME}.yaml"
AIOPSLAB_CLUSTER="${CLUSTER_NAME}"
# One port block per seed, all bound to 127.0.0.1 by kubectl port-forward: 20193/20190/20180 for
# seed 1, 20293/... for seed 2. The admission-round launcher's 19093/19090/18080 stay free.
PORT_BASE=$((20000 + INDEX * 100))
ALERTMANAGER_PORT=$((PORT_BASE + 93))
PROMETHEUS_PORT=$((PORT_BASE + 90))
RECEIVER_PORT=$((PORT_BASE + 80))
KPS_RELEASE="${KPS_RELEASE:-sslm-kube-prometheus}"
R2_RESULTS_DIR="${R2_RESULTS_DIR:-scenarios/results/r2}"
COMPARATOR_ARM="${COMPARATOR_ARM:-0}"
[[ "$COMPARATOR_ARM" == "0" || "$COMPARATOR_ARM" == "1" ]] || { echo "error: COMPARATOR_ARM must be 0 or 1, got '$COMPARATOR_ARM'" >&2; exit 1; }
# R2_DRY_RUN=1 (added 2026-10-19): the short round passes the committed schedule to the driver
# with --dry-run, which accepts a schedule without the admission-round minimums and never counts
# as a passed round (passed=False). Same s<N>-r2.* files, same harvest, same scorer.
R2_DRY_RUN="${R2_DRY_RUN:-0}"
[[ "$R2_DRY_RUN" == "0" || "$R2_DRY_RUN" == "1" ]] || { echo "error: R2_DRY_RUN must be 0 or 1, got '$R2_DRY_RUN'" >&2; exit 1; }
export CLUSTER_NAME KUBECONFIG AIOPSLAB_CLUSTER KPS_RELEASE

case "$ACTION" in
  env)
    printf 'CLUSTER_NAME=%s\nKUBECONFIG=%s\nAIOPSLAB_CLUSTER=%s\n' "$CLUSTER_NAME" "$KUBECONFIG" "$AIOPSLAB_CLUSTER"
    printf 'ALERTMANAGER_PORT=%s\nPROMETHEUS_PORT=%s\nRECEIVER_PORT=%s\n' "$ALERTMANAGER_PORT" "$PROMETHEUS_PORT" "$RECEIVER_PORT"
    ;;

  up)
    : "${APP_NAMESPACE:?error: APP_NAMESPACE comes from the AIOpsLab metadata (scenarios/resolve_app_namespace.py), never guessed}"
    if [[ "$COMPARATOR_ARM" == "1" ]]; then
      # API arm: no vLLM. The key and the spend cap reach setup_cluster.sh through the environment, never argv.
      [[ -n "${ANTHROPIC_API_KEY:-}" ]] || { echo "error: COMPARATOR_ARM=1 requires ANTHROPIC_API_KEY" >&2; exit 1; }
      : "${SSLM_COMPARATOR_MAX_COST_USD:?error: COMPARATOR_ARM=1 requires SSLM_COMPARATOR_MAX_COST_USD (spend cap per pod)}"
      mkdir -p "$(dirname "$KUBECONFIG")"
      cd "$PREFLIGHT_DIR"
      AGENT_INSTALL=1 KPS_INSTALL=1 CALICO_INSTALL=1 SSLM_LOG_ONLY=0 COMPARATOR_ARM=1 \
        APP_NAMESPACE="$APP_NAMESPACE" \
        ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" SSLM_COMPARATOR_MAX_COST_USD="$SSLM_COMPARATOR_MAX_COST_USD" \
        SSLM_PROMPT_BUDGET_CHARS="${SSLM_PROMPT_BUDGET_CHARS:-}" \
        bash setup_cluster.sh
      exit 0
    fi
    : "${VLLM_BASE_URL:?error: round 2 runs with the model; VLLM_BASE_URL is the shared vLLM of the round}"
    : "${VLLM_EGRESS_CIDR:?error: round 2 runs with the model; VLLM_EGRESS_CIDR is the kind network gateway}"
    : "${VLLM_EGRESS_PORT:?error: round 2 runs with the model; VLLM_EGRESS_PORT is the vllm_port of the round}"
    mkdir -p "$(dirname "$KUBECONFIG")"
    # SSLM_LOG_ONLY=0 is the point of round 2: the agent calls the model. setup_cluster.sh then
    # requires the three VLLM_* above with no default. SSLM_C2_DRY_RUN keeps setup_cluster's own
    # default (real C2) unless the caller overrides it.
    cd "$PREFLIGHT_DIR"
    AGENT_INSTALL=1 KPS_INSTALL=1 CALICO_INSTALL=1 SSLM_LOG_ONLY=0 \
      APP_NAMESPACE="$APP_NAMESPACE" \
      VLLM_BASE_URL="$VLLM_BASE_URL" VLLM_EGRESS_CIDR="$VLLM_EGRESS_CIDR" VLLM_EGRESS_PORT="$VLLM_EGRESS_PORT" \
      VLLM_API_KEY="${VLLM_API_KEY:-}" SSLM_PROMPT_BUDGET_CHARS="${SSLM_PROMPT_BUDGET_CHARS:-}" SSLM_TOOL_CONTRACT="${SSLM_TOOL_CONTRACT:-}" \
      VLLM_MODEL="${VLLM_MODEL:-}" VLLM_DECODING_DIALECT="${VLLM_DECODING_DIALECT:-}" \
      bash setup_cluster.sh
    ;;

  launch)
    R2_MODE="${R2_MODE:-smoke}"
    [[ "$R2_MODE" == "smoke" || "$R2_MODE" == "r2" || "$R2_MODE" == "smokegate" ]] || { echo "error: R2_MODE must be smoke, r2 or smokegate" >&2; exit 1; }
    : "${AIOPSLAB_HOME:?error: the driver imports AIOpsLab from AIOPSLAB_HOME}"
    export AIOPSLAB_HOME
    [[ -f "$KUBECONFIG" ]] || { echo "error: $KUBECONFIG does not exist, run 'seed_cluster.sh up $INDEX' first" >&2; exit 1; }
    cd "$PREFLIGHT_DIR"
    mkdir -p "$R2_RESULTS_DIR"
    HARVEST_OUT=""
    HARVEST_SCHEDULE=""
    if [[ "$R2_MODE" == "smoke" ]]; then
      : "${APP_NAMESPACE:?error: the smoke needs APP_NAMESPACE for the policy scope check}"
      OUT="$R2_RESULTS_DIR/s${INDEX}-smoke.json"
      if [[ "$COMPARATOR_ARM" == "1" ]]; then
        # API arm: the same proof as the vLLM branch below, from the other side. The smoke passes
        # only if the in-pod API usage log (SSLM_COMPARATOR_USAGE_LOG) gains a new line; no vLLM needed.
        DRIVER_ARGS=(--smoke --app-namespace "$APP_NAMESPACE" --comparator-usage --out "$OUT")
      else
        # Round 2 runs with the model: the smoke passes only if vLLM's prompt-token counter moves.
        # Without that check, the 2026-09-30 smoke passed while every agent call got connection refused.
        : "${VLLM_METRICS_URL:?error: the round-2 smoke needs VLLM_METRICS_URL (the vLLM /metrics endpoint) to prove the agent reaches the model}"
        DRIVER_ARGS=(--smoke --app-namespace "$APP_NAMESPACE" --vllm-metrics-url "$VLLM_METRICS_URL" --out "$OUT")
      fi
    else
      : "${R2_SCHEDULE:?error: round 2 needs a committed schedule (path relative to tools/triage-eval)}"
      [[ -f "$R2_SCHEDULE" ]] || { echo "error: round 2 schedule does not exist: $R2_SCHEDULE" >&2; exit 1; }
      # smokegate: the same short round on a one-fault schedule, under its own file names
      # (s<N>-smokegate.*) so it never overwrites or joins the long round, and always --dry-run.
      # scenarios/chars_window.py reads it as a gate before the long round starts.
      OUT="$R2_RESULTS_DIR/s${INDEX}-$([[ "$R2_MODE" == "smokegate" ]] && echo smokegate || echo r2).json"
      HARVEST_OUT="$OUT"
      HARVEST_SCHEDULE="$R2_SCHEDULE"
      # The model round: fault held to its planned end, every new alert counts, probe before recover.
      DRIVER_ARGS=(--schedule "$R2_SCHEDULE" --out "$OUT" --post-action-probe)
      if [[ "$R2_DRY_RUN" == "1" || "$R2_MODE" == "smokegate" ]]; then
        DRIVER_ARGS+=(--dry-run)
      fi
    fi
    # Per seed, by its own --out: the other two seeds' drivers must not block this one.
    if pgrep -f -- "run_continuous_admission.py.*--out ${OUT}" >/dev/null 2>&1; then
      echo "a driver is already running for ${OUT}, not launching another"
      exit 0
    fi
    LOG="$R2_RESULTS_DIR/s${INDEX}-${R2_MODE}.out"
    # Same shape as the admission-round launcher: harvest after the driver whatever its exit code, only the
    # port-forwards die in the exit trap, the driver's rc survives. KUBECONFIG and
    # AIOPSLAB_CLUSTER are exported above, so every child talks to this seed's cluster.
    # macOS has no setsid; nohup alone is enough for the local check, the VM gets its own session.
    DETACH=(nohup)
    command -v setsid >/dev/null 2>&1 && DETACH=(setsid nohup)
    # The agent pod's own log is the only record of envelope denials (`action DENIED`): the triage
    # record lists executed commands only. An earlier round showed "unrecognized tool 'mongo'" in
    # two diagnoses and could not prove where it came from, because this log was never pulled.
    AGENT_LOG="$R2_RESULTS_DIR/s${INDEX}-${R2_MODE}.agent.log"
    HARVEST_OUT="$HARVEST_OUT" HARVEST_SCHEDULE="$HARVEST_SCHEDULE" LOG_PREFIX="$R2_RESULTS_DIR/s${INDEX}-" AGENT_LOG="$AGENT_LOG" \
      AM_PORT="$ALERTMANAGER_PORT" PROM_PORT="$PROMETHEUS_PORT" RCV_PORT="$RECEIVER_PORT" \
      "${DETACH[@]}" bash -c '
      set -euo pipefail
      trap "kill \$(jobs -p) 2>/dev/null || true" EXIT
      kubectl -n monitoring port-forward "svc/$1-alertmanager" "${AM_PORT}:9093" > "${LOG_PREFIX}alertmanager-pf.out" 2>&1 &
      kubectl -n monitoring port-forward "svc/$1-prometheus" "${PROM_PORT}:9090" > "${LOG_PREFIX}prometheus-pf.out" 2>&1 &
      kubectl -n sslm-agent port-forward svc/sslm-agent "${RCV_PORT}:8080" > "${LOG_PREFIX}receiver-pf.out" 2>&1 &
      sleep 3
      shift 1
      rc=0
      .venv/bin/python scenarios/run_continuous_admission.py \
        --alertmanager-url "http://127.0.0.1:${AM_PORT}" \
        --prometheus-url "http://127.0.0.1:${PROM_PORT}" \
        --receiver-url "http://127.0.0.1:${RCV_PORT}" "$@" || rc=$?
      if [[ -n "$HARVEST_OUT" ]]; then
        .venv/bin/python scenarios/harvest_round.py --out "$HARVEST_OUT" --schedule "$HARVEST_SCHEDULE" \
          --prometheus-url "http://127.0.0.1:${PROM_PORT}" \
          || echo "harvest incomplete, reason in ${HARVEST_OUT%.json}.harvest.json" >&2
      fi
      kubectl -n sslm-agent logs deploy/sslm-agent --timestamps > "$AGENT_LOG" 2>&1 \
        || echo "agent log not collected, see $AGENT_LOG" >&2
      echo "driver rc=$rc"
      exit "$rc"
    ' bash "$KPS_RELEASE" "${DRIVER_ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
    disown
    echo "seed ${INDEX} (${CLUSTER_NAME}) launched in ${R2_MODE}, pid $! (log: ${PREFLIGHT_DIR}/${LOG})"
    ;;

  *)
    echo "error: unknown action '$ACTION' (up|launch|env)" >&2
    exit 1
    ;;
esac
