"""Tests for k8s/validatingadmissionpolicy-deny-db-pod.yaml: the second, independent enforcement
layer for the DB-pod exclusion, closing the gap role-app-mutate.yaml's own HONEST LIMIT comment
used to document (vanilla RBAC cannot grant `pods/exec`/`pods delete` while excluding pods whose
name matches a pattern -- `resourceNames` is an exact-name allowlist, not a pattern-based
denylist).

Same discipline as tests/triage_eval/test_k8s_manifests.py: static, YAML-parsing assertions
that the manifest's shape and CEL expression actually mirror harness/envelope.py's
`_DB_POD_MARKERS`/`_mentions_db_pod`, not a re-derivation. This file holds two YAML documents
(the ValidatingAdmissionPolicy and its ValidatingAdmissionPolicyBinding), which is why it is not
in test_k8s_manifests.py's `test_every_manifest_is_a_single_valid_yaml_document` parametrize list
-- that helper's `_load()` calls `yaml.safe_load`, which raises on a multi-document stream.

Real enforcement against a live apiserver (mongo/memcached pod exec+delete denied, non-db pod
exec+delete still allowed) does not fit here -- it needs a running kind cluster, which this
repo's pytest suite does not stand up. That proof is documented instead: the manifest's own
top-of-file comment records the live kind cluster/K8s version this was confirmed against, and the
PR description for this change carries the raw command transcript.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

K8S_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "k8s"
PREFLIGHT_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"
if str(PREFLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(PREFLIGHT_DIR))

from harness.envelope import _DB_POD_MARKERS  # noqa: E402

MANIFEST_PATH = K8S_DIR / "validatingadmissionpolicy-deny-db-pod.yaml"


def _load_docs() -> list[dict]:
    docs = list(yaml.safe_load_all(MANIFEST_PATH.read_text(encoding="utf-8")))
    assert len(docs) == 2, f"expected exactly 2 YAML documents (policy + binding), got {len(docs)}"
    return docs


def _policy() -> dict:
    policy, _binding = _load_docs()
    assert policy["kind"] == "ValidatingAdmissionPolicy"
    return policy


def _binding() -> dict:
    _policy_doc, binding = _load_docs()
    assert binding["kind"] == "ValidatingAdmissionPolicyBinding"
    return binding


def test_manifest_is_two_valid_yaml_documents():
    docs = _load_docs()
    assert docs[0]["kind"] == "ValidatingAdmissionPolicy"
    assert docs[1]["kind"] == "ValidatingAdmissionPolicyBinding"


def test_policy_matches_pods_exec_connect_and_pods_delete_only():
    # CONNECT, not CREATE: kubectl exec's admission request against pods/exec is a CONNECT
    # operation (confirmed live -- an earlier CREATE-only version of this rule silently never
    # matched a real `kubectl exec`, letting it straight through unevaluated).
    policy = _policy()
    rules = policy["spec"]["matchConstraints"]["resourceRules"]
    shapes = {
        (tuple(rule.get("apiGroups", [])), tuple(rule.get("operations", [])), tuple(rule.get("resources", [])))
        for rule in rules
    }
    assert (("",), ("CONNECT",), ("pods/exec",)) in shapes
    assert (("",), ("DELETE",), ("pods",)) in shapes
    # Nothing broader: no CREATE (which would double-count and never match the real exec
    # request), no `pods/exec` DELETE, no verb on any resource beyond these two shapes.
    assert len(rules) == 2


def test_policy_failure_policy_is_fail_closed():
    # Fail (the default), not Ignore: an apiserver-side error evaluating this policy must never
    # silently let the request through -- fail-closed is this whole repo's default posture
    # (harness/envelope.py's own module docstring: "C3 is the default, not a detected case").
    policy = _policy()
    assert policy["spec"]["failurePolicy"] == "Fail"


def test_policy_cel_expression_mirrors_envelope_db_pod_markers():
    # Back-reference check, not a re-derivation: CEL has no way to import
    # harness/envelope.py's _DB_POD_MARKERS constant, so this asserts the literal substrings in
    # the committed CEL expression are exactly that tuple's contents -- a marker added to one but
    # not the other is the drift this test exists to catch.
    assert _DB_POD_MARKERS == ("mongo", "memcached")
    policy = _policy()
    expression = policy["spec"]["validations"][0]["expression"]
    for marker in _DB_POD_MARKERS:
        assert f"contains('{marker}')" in expression, f"CEL expression missing marker {marker!r}: {expression}"
    # Nothing beyond the two documented markers.
    assert expression.count("contains(") == len(_DB_POD_MARKERS)


def test_policy_cel_expression_matches_on_request_name_case_insensitively():
    # request.name (not object/oldObject.metadata.name): the exec CREATE/CONNECT admission
    # request's `object` is a PodExecOptions, which carries no pod name field, so request.name is
    # the only structural way to get the target pod's name for both matched operations. lowerAscii
    # mirrors _is_db_pod's `name.lower()` before the substring check.
    policy = _policy()
    expression = policy["spec"]["validations"][0]["expression"]
    assert "request.name.lowerAscii()" in expression


def test_policy_denies_by_default_when_the_expression_is_false():
    # The validation expression is the "allow" condition (CEL convention for
    # ValidatingAdmissionPolicy: true = admit); it must be a negation of the marker check, i.e.
    # deny (return false) exactly when a marker is present, never the inverse.
    policy = _policy()
    expression = policy["spec"]["validations"][0]["expression"].strip()
    assert expression.startswith("!("), (
        "expression must be a negated marker check (true admits, false denies): "
        f"{expression!r}"
    )


def test_binding_targets_this_policy_and_denies():
    binding = _binding()
    policy = _policy()
    assert binding["spec"]["policyName"] == policy["metadata"]["name"]
    assert binding["spec"]["validationActions"] == ["Deny"]


def test_binding_has_no_namespace_narrowing_so_enforcement_is_cluster_wide():
    # Deliberately broader than role-app-mutate.yaml's own namespace-scoped Role: this backstop
    # must not silently stop covering a namespace a future Role starts touching.
    binding = _binding()
    match_resources = binding["spec"].get("matchResources", {})
    assert "namespaceSelector" not in match_resources
    assert "objectSelector" not in match_resources


@pytest.mark.parametrize("name", ["role-app-mutate.yaml", "validatingadmissionpolicy-deny-db-pod.yaml"])
def test_setup_cluster_applies_this_manifest_in_the_agent_install_block(name: str):
    # setup_cluster.sh's AGENT_INSTALL=1 block must apply both this manifest and the Role it backs
    # up -- a regression here (e.g. the apply line silently dropped in a future edit) would leave
    # the RBAC in place with no second layer behind it, exactly the gap this manifest closes.
    script = (PREFLIGHT_DIR / "setup_cluster.sh").read_text(encoding="utf-8")
    assert name in script


# --- requester scope (admission round, 2026-09-23): the policy binds the agent, not the benchmark

import re  # noqa: E402
import subprocess  # noqa: E402
import textwrap  # noqa: E402


def _requester_prefix() -> str:
    conditions = _policy()["spec"].get("matchConditions") or []
    assert len(conditions) == 1, f"expected exactly one requester matchCondition, got {conditions}"
    expression = conditions[0]["expression"].strip()
    match = re.fullmatch(r"request\.userInfo\.username\.startsWith\('([^']+)'\)", expression)
    assert match, f"matchCondition must be a plain username prefix check, got {expression!r}"
    return match.group(1)


def _policy_applies_to(username: str) -> bool:
    return username.startswith(_requester_prefix())


def test_requester_scope_is_derived_from_the_agent_serviceaccount_namespace():
    sa = yaml.safe_load((K8S_DIR / "serviceaccount.yaml").read_text(encoding="utf-8"))
    namespace = sa["metadata"]["namespace"]
    assert _requester_prefix() == f"system:serviceaccount:{namespace}:"
    assert _policy_applies_to(f"system:serviceaccount:{namespace}:{sa['metadata']['name']}")


@pytest.mark.parametrize(
    "username",
    [
        "kubernetes-admin",  # kind's admin kubeconfig, what AIOpsLab's injector runs as
        "system:serviceaccount:default:default",
        "system:serviceaccount:sslm-agent-other:sslm-agent",  # prefix must end at the ':'
        "system:serviceaccount:kube-system:replicaset-controller",
    ],
)
def test_policy_does_not_bind_the_benchmark_injector_or_other_identities(username):
    assert not _policy_applies_to(username)


def test_policy_binds_any_serviceaccount_in_the_agent_namespace():
    assert _policy_applies_to("system:serviceaccount:sslm-agent:a-future-sa")


VERIFY_SCRIPT = PREFLIGHT_DIR / "verify_db_pod_policy_scope.sh"

# kubectl double for verify_db_pod_policy_scope.sh. STUB_PODS lists pod names, STUB_CAN_I answers
# `auth can-i`, STUB_AGENT_DENIED/STUB_ADMIN_DENIED decide the two dry-run deletes.
KUBECTL_VERIFY_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    args="$*"
    case "$args" in
      *"get pods -o name"*) for p in ${STUB_PODS:-}; do echo "pod/$p"; done ;;
      *"auth can-i"*) echo "${STUB_CAN_I:-yes}" ;;
      *"delete pod"*"--as="*)
        if [[ "${STUB_AGENT_DENIED:-1}" == "1" ]]; then
          echo "denied: exec into, or delete of, a database pod ... is excluded in any case" >&2; exit 1
        fi
        echo "pod deleted (server dry run)" ;;
      *"delete pod"*)
        if [[ "${STUB_ADMIN_DENIED:-0}" == "1" ]]; then
          echo "denied: ... is excluded in any case" >&2; exit 1
        fi
        echo "pod deleted (server dry run)" ;;
      *) echo "unexpected kubectl $args" >&2; exit 99 ;;
    esac
    """
)


def _run_verify(tmp_path, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "kubectl"
    stub.write_text(KUBECTL_VERIFY_STUB, encoding="utf-8")
    stub.chmod(0o755)
    full_env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "APP_NAMESPACE": "test-hotel-reservation",
        "STUB_PODS": "frontend-1 mongodb-geo-abc",
        **env,
    }
    return subprocess.run(["bash", str(VERIFY_SCRIPT)], env=full_env, capture_output=True, text=True)


def test_verify_script_passes_when_agent_denied_and_injector_admitted(tmp_path):
    result = _run_verify(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "ok (a)" in result.stdout and "ok (b)" in result.stdout


def test_verify_script_fails_when_the_policy_blocks_the_injector(tmp_path):
    # The exact failure of the 2026-09-23 real run: admin exec/delete on mongodb-geo denied.
    result = _run_verify(tmp_path, STUB_ADMIN_DENIED="1")
    assert result.returncode == 1
    assert "FAIL (b)" in result.stderr


def test_verify_script_fails_when_the_agent_is_not_denied(tmp_path):
    result = _run_verify(tmp_path, STUB_AGENT_DENIED="0")
    assert result.returncode == 1
    assert "FAIL (a)" in result.stderr


@pytest.mark.parametrize("env", [{"STUB_PODS": "frontend-1 geo-2"}, {"STUB_CAN_I": "no"}])
def test_verify_script_reports_not_measured_never_success(tmp_path, env):
    result = _run_verify(tmp_path, **env)
    assert result.returncode == 2
    assert "not measured" in result.stderr
