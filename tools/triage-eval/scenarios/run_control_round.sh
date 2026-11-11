#!/usr/bin/env bash
# Control round: the same 6 scenarios as the matrix, config_control (--control), with no fault
# injected. Measures how often the mitigation eval passes on its own, with the cluster never
# touched, to separate real mitigation from a permissive eval in the 4 success=True results of
# the real matrix (all in misconfig_app_hotel_res and redeploy_without_PV).
#
# Resumable and with the same per-episode cap as the real matrix (run_full_matrix.sh), for the
# same reason: shared cluster and third-party code.
#
# Usage (interactive terminal on an Apple Silicon Mac, with Metal; do not run alongside other
# memory-heavy builds):
#   SSLM_ENV=mlx-local bash scenarios/run_control_round.sh
#   bash scenarios/run_control_round.sh --env mlx-local
#
# The env is required for the same reason as in run_full_matrix.sh: resume is per env
# (results/<env>/), otherwise a run on other hardware would resume on top of this one.
set -u

cd "$(dirname "$0")/.."

SSLM_ENV="${SSLM_ENV:-}"
if [[ "${1:-}" == "--env" ]]; then
  SSLM_ENV="${2:-}"
fi
if [[ -z "$SSLM_ENV" ]]; then
  echo "error: set the env of the round (--env <name> or SSLM_ENV=<name>); resume is per env (results/<env>/)" >&2
  exit 1
fi
export SSLM_ENV

export AIOPSLAB_HOME="${AIOPSLAB_HOME:-$(cd ../../tmp/triage-eval/AIOpsLab 2>/dev/null && pwd)}"
if [[ -z "${AIOPSLAB_HOME}" || ! -d "${AIOPSLAB_HOME}/aiopslab" ]]; then
  echo "error: AIOPSLAB_HOME does not point at the AIOpsLab clone" >&2
  exit 1
fi

SCENARIOS=(
  misconfig_app_hotel_res-mitigation-1
  revoke_auth_mongodb-mitigation-1
  revoke_auth_mongodb-mitigation-2
  user_unregistered_mongodb-mitigation-1
  user_unregistered_mongodb-mitigation-2
  redeploy_without_PV-mitigation-1
)

mkdir -p "scenarios/results/${SSLM_ENV}"
LOG=scenarios/results/control-run.log

echo "=== control round $(date '+%Y-%m-%d %H:%M:%S') [env=${SSLM_ENV}]: ${#SCENARIOS[@]} scenarios, no fault injected ===" | tee -a "$LOG"
for scenario in "${SCENARIOS[@]}"; do
  out="scenarios/results/${SSLM_ENV}/${scenario}_control.json"
  if [[ -f "$out" ]]; then
    echo "[skip] $scenario / control (result already exists in ${SSLM_ENV}/)" | tee -a "$LOG"
    continue
  fi
  echo "[run ] $scenario / control ($(date '+%H:%M:%S'))" | tee -a "$LOG"
  if ! timeout --kill-after=60s "${EPISODE_TIMEOUT_SEC:-1800}" \
      .venv/bin/python scenarios/run_matrix.py \
      --config configs/config_control.yaml \
      --env "$SSLM_ENV" \
      --model-client mlx_factories:make_model_client \
      --environment-factory mlx_factories:make_environment \
      --scenario "$scenario" \
      --control </dev/null >>"$LOG" 2>&1; then
    status=$?
    if [[ "$status" == 124 || "$status" == 137 ]]; then
      echo "[FAIL] $scenario / control (timeout after ${EPISODE_TIMEOUT_SEC:-1800}s, moving on to the next)" | tee -a "$LOG"
    else
      echo "[FAIL] $scenario / control (moving on to the next; details in the log)" | tee -a "$LOG"
    fi
  fi
done

total=$(ls "scenarios/results/${SSLM_ENV}"/*_control.json 2>/dev/null | wc -l | tr -d ' ')
echo "=== done $(date '+%H:%M:%S') [env=${SSLM_ENV}]: ${total}/${#SCENARIOS[@]} control results in scenarios/results/${SSLM_ENV}/ ===" | tee -a "$LOG"
