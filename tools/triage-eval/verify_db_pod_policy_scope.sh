#!/usr/bin/env bash
# Live check of who the DB-pod ValidatingAdmissionPolicy stops (scoped after the admission
# round of 2026-09-23 found it also blocking AIOpsLab's own injector).
# Needs a running cluster with AGENT_INSTALL=1 applied and the app already deployed, so a
# mongodb-* pod exists. It never mutates anything: every probe is a server-side dry-run DELETE,
# which the apiserver still sends through validating admission.
#
#   (a) as the agent's ServiceAccount: RBAC allows `delete pods` in the app namespace, and the
#       policy must still deny the DB pod. RBAC yes plus admission no is the exact layering
#       role-app-mutate.yaml's HONEST LIMIT comment asked for.
#   (b) as the caller's own identity (the admin kubeconfig AIOpsLab's injector uses): the same
#       request must be admitted, otherwise the benchmark cannot inject or recover a Mongo fault.
#
# Exit 0 only when both hold. Anything it cannot measure (no DB pod, no RBAC grant) exits 2,
# never 0, so a smoke run cannot read "not measured" as "policy scoped correctly".
set -euo pipefail

: "${APP_NAMESPACE:?set APP_NAMESPACE to the AIOpsLab app namespace (e.g. test-hotel-reservation)}"
AGENT_USER="${AGENT_USER:-system:serviceaccount:sslm-agent:sslm-agent}"
POLICY_MARKER="${POLICY_MARKER:-excluded in any case}"

DB_POD="$(kubectl -n "$APP_NAMESPACE" get pods -o name | sed 's#^pod/##' | grep -E 'mongo|memcached' | head -n 1 || true)"
if [[ -z "$DB_POD" ]]; then
  echo "not measured: no mongo/memcached pod in namespace $APP_NAMESPACE" >&2
  exit 2
fi

if [[ "$(kubectl auth can-i delete pods -n "$APP_NAMESPACE" --as="$AGENT_USER")" != "yes" ]]; then
  echo "not measured: RBAC does not grant $AGENT_USER delete pods in $APP_NAMESPACE, so admission is never reached" >&2
  exit 2
fi

set +e
AGENT_OUT="$(kubectl -n "$APP_NAMESPACE" delete pod "$DB_POD" --dry-run=server --as="$AGENT_USER" 2>&1)"
AGENT_RC=$?
ADMIN_OUT="$(kubectl -n "$APP_NAMESPACE" delete pod "$DB_POD" --dry-run=server 2>&1)"
ADMIN_RC=$?
set -e

FAILED=0
if [[ $AGENT_RC -eq 0 || "$AGENT_OUT" != *"$POLICY_MARKER"* ]]; then
  echo "FAIL (a): agent identity was not denied by the DB-pod policy on $DB_POD: $AGENT_OUT" >&2
  FAILED=1
else
  echo "ok (a): agent identity denied on $DB_POD by the DB-pod policy"
fi
if [[ $ADMIN_RC -ne 0 ]]; then
  echo "FAIL (b): injector identity was denied on $DB_POD: $ADMIN_OUT" >&2
  FAILED=1
else
  echo "ok (b): injector identity admitted on $DB_POD (dry-run)"
fi
exit "$FAILED"
