"""Tests for the sslm-agent K8s manifests: versioned YAML plus a test that asserts the exact verbs
of the envelope and denies the rest.

The hard rule this file enforces mechanically, not just by review: the RBAC objects must grant
*exactly* the C0/C1/C2 verbs the envelope classifier (harness/envelope.py) actually allows to
execute, and must grant none of the C3 shapes that classifier always denies. The expected verb
sets below are a deliberate mirror of harness/envelope.py's allowlists (imported directly where
the module exposes a comparable constant; mirrored with a back-reference comment where it
doesn't -- envelope.py classifies *commands*, not RBAC verb/resource pairs, so there is no single
constant to import for "the RBAC shape C2 needs"). See harness/envelope.py's own module
docstring: this module's job (classifying one command) is deliberately narrower than owning the
RBAC shape, so a mirrored-and-commented list here, not an import, is correct, not a shortcut.
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

from harness.envelope import (  # noqa: E402
    C0_CALL_CAP,
    C1_CALL_CAP,
    C2_CALL_CAP,
    CONFIDENCE_GATE,
)

# Placeholders every parametrized manifest carries (see each file's own comment for why each one
# is a placeholder, never a hardcoded value). Substituted with harmless test values before a
# manifest is checked for anything beyond "is this valid YAML with a placeholder namespace" --
# the substitution mirrors what setup_cluster.sh's envsubst calls do, not a real cluster fact.
_PLACEHOLDER_SUBSTITUTIONS = {
    "${APP_NAMESPACE}": "test-hotel-reservation",
    "${AGENT_IMAGE}": "sslm-agent:test",
    "${VLLM_BASE_URL}": "http://198.51.100.1:8000/v1",
    "${VLLM_API_KEY}": "",
    "${VLLM_EGRESS_CIDR}": "198.51.100.1/32",
    "${VLLM_EGRESS_PORT}": "8000",
    "${KPS_NAMESPACE}": "test-monitoring",
    "${SSLM_C2_DRY_RUN}": "0",
    "${SSLM_HISTORY_BUDGET}": "on",
    "${SSLM_PROMPT_BUDGET_CHARS}": "",
    "${SSLM_TOOL_CONTRACT}": "",
    "${VLLM_MODEL}": "",
    "${VLLM_DECODING_DIALECT}": "",
    "${COMPARATOR_ARM}": "0",
    "${SSLM_COMPARATOR_MAX_COST_USD}": "",
    "${PROMETHEUS_URL}": "http://prometheus-operated.test-monitoring.svc.cluster.local:9090",
    # The `kubernetes` Service's real backend address (found by the live NetworkPolicy verification:
    # setup_cluster.sh derives this live via `kubectl get endpoints kubernetes`, never the
    # Service's ClusterIP -- see networkpolicy.yaml's own comment for why the ClusterIP form
    # never actually matched under Calico).
    "${APISERVER_EGRESS_CIDR}": "203.0.113.9/32",
    "${APISERVER_EGRESS_PORT}": "6443",
}


def _load(name: str, *, substitute: bool = True) -> dict:
    text = (K8S_DIR / name).read_text(encoding="utf-8")
    if substitute:
        for placeholder, value in _PLACEHOLDER_SUBSTITUTIONS.items():
            text = text.replace(placeholder, value)
    return yaml.safe_load(text)


# The C3 shapes the envelope names explicitly as always-denied, and the C1/C2 shapes it names as
# always-allowed-with-gates -- mirrored here as (apiGroup, resource, verb) tuples so
# test_role_app_mutate_grants_no_c3_shape can assert every one of them is absent from every rule
# in role-app-mutate.yaml, regardless of how the rule groups resources/verbs.
_C3_FORBIDDEN_SHAPES = {
    # "apply/patch/edit/replace of spec" of the full Deployment resource: `create`/`delete` are
    # never legitimate C2 shapes (only `kubectl rollout restart|undo`'s Patch is, granted
    # separately and asserted by test_role_app_mutate_grants_exactly_the_c2_verb_shapes -- `patch`
    # on `deployments` is deliberately NOT in this forbidden set, since that IS the C2 rollout
    # shape, not a C3 one; the strict-allowlist test below is what catches any OTHER verb on
    # `deployments` sneaking in).
    ("apps", "deployments", "create"),
    ("apps", "deployments", "delete"),
    # "delete of deployment/service/PVC/namespace": only pods may be deleted (C2's "delete pod").
    ("", "persistentvolumeclaims", "delete"),
    ("", "services", "delete"),
    ("", "namespaces", "delete"),
    # "create/change RBAC/secret/configmap": no verb on any of these, ever.
    ("", "secrets", "get"),
    ("", "secrets", "create"),
    ("", "secrets", "update"),
    ("", "secrets", "patch"),
    ("", "configmaps", "create"),
    ("", "configmaps", "update"),
    ("", "configmaps", "patch"),
    ("rbac.authorization.k8s.io", "roles", "create"),
    ("rbac.authorization.k8s.io", "rolebindings", "create"),
    ("rbac.authorization.k8s.io", "clusterroles", "create"),
    ("rbac.authorization.k8s.io", "clusterrolebindings", "create"),
}

_WRITE_VERBS = {"create", "update", "patch", "delete", "deletecollection"}


def _rule_shapes(rules: list[dict]) -> set[tuple[str, str, str]]:
    shapes = set()
    for rule in rules:
        for api_group in rule.get("apiGroups", [""]):
            for resource in rule.get("resources", []):
                for verb in rule.get("verbs", []):
                    shapes.add((api_group, resource, verb))
    return shapes


# ---------------------------------------------------------------------------
# namespace.yaml / serviceaccount.yaml
# ---------------------------------------------------------------------------


def test_namespace_is_sslm_agent_not_parametrized():
    doc = _load("namespace.yaml")
    assert doc["kind"] == "Namespace"
    assert doc["metadata"]["name"] == "sslm-agent"


def test_serviceaccount_lives_in_agent_namespace():
    doc = _load("serviceaccount.yaml")
    assert doc["kind"] == "ServiceAccount"
    assert doc["metadata"]["name"] == "sslm-agent"
    assert doc["metadata"]["namespace"] == "sslm-agent"


# ---------------------------------------------------------------------------
# C0: cluster-wide read-only ClusterRole
# ---------------------------------------------------------------------------


def test_clusterrole_read_only_is_cluster_scoped_and_grants_only_read_verbs():
    doc = _load("clusterrole-read-only.yaml")
    assert doc["kind"] == "ClusterRole"
    rules = doc["rules"]
    all_verbs = {verb for rule in rules for verb in rule["verbs"]}
    assert all_verbs <= {"get", "list", "watch"}, (
        f"ClusterRole must be strictly read-only (C0), got verbs {all_verbs}"
    )
    assert not (all_verbs & _WRITE_VERBS)


def test_clusterrole_read_only_covers_the_resources_c0_coverage_needs():
    # Section 7.2's table: "get, list, watch (pods, deployments, replicasets, events, services,
    # pvc) + pods/log".
    doc = _load("clusterrole-read-only.yaml")
    shapes = _rule_shapes(doc["rules"])
    expected_resources = {
        ("", "pods"),
        ("", "events"),
        ("", "services"),
        ("", "persistentvolumeclaims"),
        ("apps", "deployments"),
        ("apps", "replicasets"),
        ("", "pods/log"),
    }
    covered = {(group, resource) for group, resource, _verb in shapes}
    assert expected_resources <= covered, expected_resources - covered


def test_clusterrole_read_only_never_grants_exec():
    # pods/exec is C1, and C1 is namespace-scoped to the app, never cluster-wide (a Role in the
    # app namespace, and NO exec into the database namespace/pods) -- granting it
    # cluster-wide here would also reach the database pods C1 must never touch).
    doc = _load("clusterrole-read-only.yaml")
    shapes = _rule_shapes(doc["rules"])
    assert not any(resource == "pods/exec" for _group, resource, _verb in shapes)


def test_clusterrolebinding_binds_the_agent_serviceaccount_to_the_read_only_role():
    doc = _load("clusterrolebinding-read-only.yaml")
    assert doc["kind"] == "ClusterRoleBinding"
    assert doc["roleRef"]["name"] == "sslm-agent-read-only"
    subjects = doc["subjects"]
    assert len(subjects) == 1
    assert subjects[0]["kind"] == "ServiceAccount"
    assert subjects[0]["name"] == "sslm-agent"
    assert subjects[0]["namespace"] == "sslm-agent"


# ---------------------------------------------------------------------------
# C1 + C2: the app-namespace Role, with the exact verbs the envelope classifier admits
# ---------------------------------------------------------------------------


def test_role_app_mutate_is_namespace_scoped_to_the_app_not_the_agent():
    doc = _load("role-app-mutate.yaml")
    assert doc["kind"] == "Role"
    # Substituted: proves the placeholder actually drives the namespace, and that it never
    # collapses to the agent's own namespace (sslm-agent) by accident.
    assert doc["metadata"]["namespace"] == "test-hotel-reservation"
    assert doc["metadata"]["namespace"] != "sslm-agent"


def test_role_app_mutate_namespace_is_a_placeholder_in_the_committed_file():
    # Read WITHOUT substitution: the committed file itself must never carry a real/guessed
    # namespace (e.g. "hotelReservation" hardcoded) -- only the placeholder token.
    raw = (K8S_DIR / "role-app-mutate.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    assert doc["metadata"]["namespace"] == "${APP_NAMESPACE}"


def test_role_app_mutate_grants_exactly_the_c1_exec_shape():
    doc = _load("role-app-mutate.yaml")
    shapes = _rule_shapes(doc["rules"])
    assert ("", "pods/exec", "create") in shapes


def test_role_app_mutate_grants_the_c1_top_shape_via_the_metrics_api():
    # `kubectl top` (C1's other admitted verb, harness/envelope.py's
    # _classify_c1) is served through the aggregation layer and IS RBAC-gated, on the
    # metrics.k8s.io API group, not left ungated as an earlier version of this suite/the
    # ClusterRole comment assumed.
    doc = _load("role-app-mutate.yaml")
    shapes = _rule_shapes(doc["rules"])
    assert ("metrics.k8s.io", "pods", "get") in shapes
    assert ("metrics.k8s.io", "pods", "list") in shapes


def test_role_app_mutate_grants_exactly_the_c2_verb_shapes():
    # Mirrors harness/envelope.py's C2 allowlist v1 (module docstring: "_C2_ROLLOUT_SUBCOMMANDS",
    # _classify_delete's "only 'delete pod' is C2", _classify_scale's deployments/scale gate),
    # with the verb each kubectl subcommand actually sends over the wire (an earlier
    # version of this file had update/patch swapped between the two C2 mutation
    # shapes) -- `kubectl scale` Updates the deployments/scale subresource (same verb HPA's own
    # ClusterRole is granted on */scale); `kubectl rollout restart|undo` both Patch the
    # Deployment resource itself.
    doc = _load("role-app-mutate.yaml")
    shapes = _rule_shapes(doc["rules"])
    expected_c2_shapes = {
        ("apps", "deployments/scale", "update"),
        ("", "pods", "delete"),
        ("apps", "deployments", "patch"),
    }
    assert expected_c2_shapes <= shapes, expected_c2_shapes - shapes


def test_role_app_mutate_grants_no_c3_shape():
    doc = _load("role-app-mutate.yaml")
    shapes = _rule_shapes(doc["rules"])
    granted_forbidden = shapes & _C3_FORBIDDEN_SHAPES
    assert not granted_forbidden, f"Role grants C3 shape(s), must never: {granted_forbidden}"


def test_role_app_mutate_grants_no_verb_beyond_the_c1_c2_allowlist():
    # The other half of "deny the rest": not just "no known-bad shape", but "nothing beyond the
    # exact allowlist" -- catches a future verb/resource this test's C3 exemplar list doesn't
    # happen to name.
    doc = _load("role-app-mutate.yaml")
    shapes = _rule_shapes(doc["rules"])
    allowlist = {
        ("", "pods/exec", "create"),
        ("metrics.k8s.io", "pods", "get"),
        ("metrics.k8s.io", "pods", "list"),
        ("apps", "deployments/scale", "update"),
        ("", "pods", "delete"),
        ("apps", "deployments", "patch"),
    }
    assert shapes == allowlist, f"unexpected shape(s): {shapes - allowlist}"


def test_rolebinding_app_mutate_subject_is_the_agent_serviceaccount_in_its_own_namespace():
    doc = _load("rolebinding-app-mutate.yaml")
    assert doc["kind"] == "RoleBinding"
    assert doc["metadata"]["namespace"] == "test-hotel-reservation"
    subjects = doc["subjects"]
    assert len(subjects) == 1
    assert subjects[0]["kind"] == "ServiceAccount"
    assert subjects[0]["name"] == "sslm-agent"
    # Cross-namespace subject, on purpose: the SA lives in sslm-agent, not in the app namespace
    # the RoleBinding itself is created in.
    assert subjects[0]["namespace"] == "sslm-agent"
    assert doc["roleRef"]["kind"] == "Role"
    assert doc["roleRef"]["name"] == "sslm-agent-mutate"


def test_envelope_caps_referenced_by_this_suite_are_the_live_constants():
    # Not an RBAC assertion -- proves the import above actually pulls live values (a stale copy
    # pasted into this file instead of importing would silently drift from harness/envelope.py).
    assert C0_CALL_CAP == 12
    assert C1_CALL_CAP == 4
    assert C2_CALL_CAP == 2
    assert CONFIDENCE_GATE == 0.7


# ---------------------------------------------------------------------------
# Deployment / Service
# ---------------------------------------------------------------------------


def test_deployment_runs_as_the_agent_serviceaccount_in_its_own_namespace():
    doc = _load("deployment.yaml")
    assert doc["kind"] == "Deployment"
    assert doc["metadata"]["namespace"] == "sslm-agent"
    pod_spec = doc["spec"]["template"]["spec"]
    assert pod_spec["serviceAccountName"] == "sslm-agent"


def test_deployment_pins_the_dedicated_node_via_selector_and_toleration():
    doc = _load("deployment.yaml")
    pod_spec = doc["spec"]["template"]["spec"]
    assert pod_spec["nodeSelector"] == {"role": "sslm-agent"}
    tolerations = pod_spec["tolerations"]
    assert any(
        t.get("key") == "sslm-agent" and t.get("effect") == "NoSchedule" for t in tolerations
    ), tolerations


def test_deployment_runs_non_root_matching_the_dockerfile_uid():
    doc = _load("deployment.yaml")
    pod_spec = doc["spec"]["template"]["spec"]
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    assert pod_spec["securityContext"]["runAsUser"] == 10001  # Dockerfile: useradd --uid 10001


def test_deployment_probes_hit_healthz_on_the_http_port():
    doc = _load("deployment.yaml")
    container = doc["spec"]["template"]["spec"]["containers"][0]
    assert container["ports"][0]["containerPort"] == 8080
    for probe_name in ("livenessProbe", "readinessProbe"):
        probe = container[probe_name]
        assert probe["httpGet"]["path"] == "/healthz"
        assert probe["httpGet"]["port"] == "http"


def test_deployment_image_and_vllm_base_url_are_placeholders_in_the_committed_file():
    raw = (K8S_DIR / "deployment.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    container = doc["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "${AGENT_IMAGE}"
    env_by_name = {e["name"]: e.get("value") for e in container["env"]}
    assert env_by_name["VLLM_BASE_URL"] == "${VLLM_BASE_URL}"


def test_deployment_carries_ssl_c2_dry_run_as_a_placeholder_never_a_hardcoded_value():
    # Same rule as VLLM_BASE_URL above: the committed manifest never bakes in an operational
    # decision (setup_cluster.sh's SSLM_C2_DRY_RUN default is what actually decides this at apply
    # time, tested in test_setup_cluster_env.py).
    raw = (K8S_DIR / "deployment.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    container = doc["spec"]["template"]["spec"]["containers"][0]
    env_by_name = {e["name"]: e.get("value") for e in container["env"]}
    assert env_by_name["SSLM_C2_DRY_RUN"] == "${SSLM_C2_DRY_RUN}"


def test_deployment_app_namespace_and_prometheus_url_are_placeholders_in_the_committed_file():
    # Build plan item 4's honest-limit fix: entrypoint.py's _build_episode_runner only wires the
    # real orchestrator (instead of falling back to log_only_episode_runner) when APP_NAMESPACE
    # and PROMETHEUS_URL both reach the container's environment -- neither may ever be a hardcoded
    # value in the committed manifest, same contract as VLLM_BASE_URL/AGENT_IMAGE above.
    raw = (K8S_DIR / "deployment.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    container = doc["spec"]["template"]["spec"]["containers"][0]
    env_by_name = {e["name"]: e.get("value") for e in container["env"]}
    assert env_by_name["APP_NAMESPACE"] == "${APP_NAMESPACE}"
    assert env_by_name["PROMETHEUS_URL"] == "${PROMETHEUS_URL}"


def test_deployment_app_namespace_and_prometheus_url_substitute_cleanly():
    # The substituted-load side of the assertion above: with real values in place, both env vars
    # actually carry them through (a regression here would mean the placeholder syntax itself is
    # malformed, e.g. missing braces, even though the raw-text check above would still pass).
    doc = _load("deployment.yaml")
    container = doc["spec"]["template"]["spec"]["containers"][0]
    env_by_name = {e["name"]: e.get("value") for e in container["env"]}
    assert env_by_name["APP_NAMESPACE"] == "test-hotel-reservation"
    assert env_by_name["PROMETHEUS_URL"] == "http://prometheus-operated.test-monitoring.svc.cluster.local:9090"


def test_deployment_triage_log_lives_on_a_writable_emptydir_mount():
    """readOnlyRootFilesystem is on, so SSLM_TRIAGE_LOG must point inside a mounted volume, and
    that volume must be an emptyDir (honors fsGroup; a hostPath dir is root-owned)."""
    doc = _load("deployment.yaml")
    pod_spec = doc["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    assert container["securityContext"]["readOnlyRootFilesystem"] is True

    env = {e["name"]: e.get("value") for e in container["env"]}
    triage_log = env["SSLM_TRIAGE_LOG"]
    mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
    volumes = {v["name"]: v for v in pod_spec["volumes"]}

    owning = [name for name, path in mounts.items() if triage_log.startswith(path.rstrip("/") + "/")]
    assert owning, f"{triage_log} is not under any volumeMount"
    assert "emptyDir" in volumes[owning[0]]
    assert volumes[owning[0]]["emptyDir"].get("sizeLimit"), "an append-only log needs a size cap"


def test_deployment_anthropic_key_comes_only_from_an_optional_secret():
    # The comparator key must reach the pod as a Secret, never as a substituted value: a
    # placeholder would put it in the applied manifest text. optional, so the vLLM arm (no
    # Secret) still starts; entrypoint.py refuses a comparator pod without it.
    raw = (K8S_DIR / "deployment.yaml").read_text(encoding="utf-8")
    assert "${ANTHROPIC_API_KEY}" not in raw
    container = yaml.safe_load(raw)["spec"]["template"]["spec"]["containers"][0]
    (key_env,) = [e for e in container["env"] if e["name"] == "ANTHROPIC_API_KEY"]
    assert "value" not in key_env
    ref = key_env["valueFrom"]["secretKeyRef"]
    assert ref == {"name": "sslm-agent-anthropic", "key": "ANTHROPIC_API_KEY", "optional": True}
    # Every other env entry is a plain value: the Secret is the only secret source.
    assert [e["name"] for e in container["env"] if "valueFrom" in e] == ["ANTHROPIC_API_KEY"]


def test_deployment_comparator_switch_and_cap_are_placeholders_and_usage_log_is_on_the_emptydir():
    raw = (K8S_DIR / "deployment.yaml").read_text(encoding="utf-8")
    container = yaml.safe_load(raw)["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert env["COMPARATOR_ARM"] == "${COMPARATOR_ARM}"
    assert env["SSLM_COMPARATOR_MAX_COST_USD"] == "${SSLM_COMPARATOR_MAX_COST_USD}"
    usage_log = env["SSLM_COMPARATOR_USAGE_LOG"]
    mount_paths = [m["mountPath"].rstrip("/") + "/" for m in container["volumeMounts"]]
    assert any(usage_log.startswith(path) for path in mount_paths), usage_log
    assert usage_log != env["SSLM_TRIAGE_LOG"]


def test_setup_cluster_substitutes_every_deployment_placeholder():
    # A placeholder missing from the envsubst list reaches the pod as the literal "${NAME}" (the
    # comparator would then read COMPARATOR_ARM="${COMPARATOR_ARM}" and refuse to start).
    import re

    placeholders = set(re.findall(r"\$\{[A-Z_]+\}", (K8S_DIR / "deployment.yaml").read_text(encoding="utf-8")))
    setup = (K8S_DIR.parent / "setup_cluster.sh").read_text(encoding="utf-8")
    (line,) = [l for l in setup.splitlines() if "envsubst" in l and "deployment.yaml" in l]
    listed = set(re.findall(r"\$\{[A-Z_]+\}", line.split("<", 1)[0]))
    assert placeholders == listed, (placeholders ^ listed)


def test_service_selects_the_deployment_pods_on_the_http_port():
    doc = _load("service.yaml")
    assert doc["kind"] == "Service"
    assert doc["metadata"]["namespace"] == "sslm-agent"
    assert doc["spec"]["selector"] == {"app.kubernetes.io/name": "sslm-agent"}
    assert doc["spec"]["ports"][0]["port"] == 8080


# ---------------------------------------------------------------------------
# NetworkPolicy
# ---------------------------------------------------------------------------


def test_networkpolicy_is_default_deny_egress_and_ingress_with_no_catch_all_rule():
    doc = _load("networkpolicy.yaml")
    assert doc["kind"] == "NetworkPolicy"
    assert set(doc["spec"]["policyTypes"]) == {"Ingress", "Egress"}
    for direction in ("ingress", "egress"):
        for rule in doc["spec"][direction]:
            selector_key = "from" if direction == "ingress" else "to"
            peers = rule.get(selector_key, [])
            assert peers, (
                f"{direction} rule with no {selector_key!r} peers is a catch-all "
                "(allows everything for that protocol/port), not zero-egress-externo"
            )


def test_networkpolicy_ingress_is_alertmanager_only_on_the_agent_http_port():
    doc = _load("networkpolicy.yaml")
    ingress_rules = doc["spec"]["ingress"]
    assert len(ingress_rules) == 1
    ports = ingress_rules[0]["ports"]
    assert ports == [{"protocol": "TCP", "port": 8080}]


def test_networkpolicy_kps_namespace_is_a_placeholder_in_the_committed_file():
    # A hardcoded "monitoring" here would silently desync from
    # setup_cluster.sh's KPS_INSTALL=1 block, which lets that namespace be overridden via
    # KPS_NAMESPACE. Must never be hardcoded, same as APP_NAMESPACE/VLLM_*.
    #
    # Checked against the PARSED, unsubstituted doc's actual selector values, not a raw text
    # search: "monitoring" legitimately appears in this file's own prose comments explaining the
    # default, which a plain `"monitoring" not in raw_text` assertion would also catch as a false
    # positive.
    raw = (K8S_DIR / "networkpolicy.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    namespaces = set()
    for rule in doc["spec"]["ingress"]:
        for peer in rule.get("from", []):
            ns = peer.get("namespaceSelector", {}).get("matchLabels", {}).get("kubernetes.io/metadata.name")
            if ns:
                namespaces.add(ns)
    for rule in doc["spec"]["egress"]:
        for peer in rule.get("to", []):
            ns = peer.get("namespaceSelector", {}).get("matchLabels", {}).get("kubernetes.io/metadata.name")
            if ns and ns != "kube-system":
                namespaces.add(ns)
    assert namespaces == {"${KPS_NAMESPACE}"}


def test_networkpolicy_kps_namespace_substitution_drives_every_monitoring_selector():
    doc = _load("networkpolicy.yaml")
    namespaces = set()
    for rule in doc["spec"]["ingress"]:
        for peer in rule.get("from", []):
            ns = peer.get("namespaceSelector", {}).get("matchLabels", {}).get("kubernetes.io/metadata.name")
            if ns:
                namespaces.add(ns)
    for rule in doc["spec"]["egress"]:
        for peer in rule.get("to", []):
            ns = peer.get("namespaceSelector", {}).get("matchLabels", {}).get("kubernetes.io/metadata.name")
            if ns and ns != "kube-system":
                namespaces.add(ns)
    assert namespaces == {"test-monitoring"}


def test_networkpolicy_egress_never_carries_an_unrestricted_ipblock():
    # 0.0.0.0/0 (or an ipBlock with no cidr at all) would be the "zero external egress" claim
    # failing by construction, not by a live cluster test -- catch it in the YAML itself.
    doc = _load("networkpolicy.yaml")
    for rule in doc["spec"]["egress"]:
        for peer in rule.get("to", []):
            ip_block = peer.get("ipBlock")
            if ip_block is not None:
                assert ip_block["cidr"] != "0.0.0.0/0"
                assert ip_block["cidr"] not in ("", None)


def test_networkpolicy_apiserver_egress_cidr_and_port_are_placeholders_in_the_committed_file():
    # Build plan item 8's verification finding: an earlier version of this rule hardcoded the
    # `kubernetes` Service's own ClusterIP (10.96.0.1/32:443, "kube-apiserver always allocates the
    # first IP of the service CIDR to it"). That is correct as a declaration but was proven, live
    # against a real kind+Calico cluster, to never actually match -- kube-proxy's Service DNAT
    # rewrites the destination before Calico's egress policy evaluates the packet. The rule now
    # has to be substituted with the apiserver's real backend address at apply time, same
    # placeholder discipline as VLLM_EGRESS_CIDR/PORT: checked against the PARSED, unsubstituted
    # doc, not a raw text search (same rationale as the KPS_NAMESPACE test below).
    raw = (K8S_DIR / "networkpolicy.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    apiserver_rules = [
        rule
        for rule in doc["spec"]["egress"]
        for peer in rule.get("to", [])
        if peer.get("ipBlock", {}).get("cidr") == "${APISERVER_EGRESS_CIDR}"
    ]
    assert len(apiserver_rules) == 1
    assert apiserver_rules[0]["ports"] == [{"protocol": "TCP", "port": "${APISERVER_EGRESS_PORT}"}]
    # The old hardcoded ClusterIP form, and the old over-broad /12 it replaced, are both gone from every egress rule.
    for rule in doc["spec"]["egress"]:
        for peer in rule.get("to", []):
            ip_block = peer.get("ipBlock")
            if ip_block is not None:
                assert ip_block["cidr"] not in ("10.96.0.1/32", "10.96.0.0/12")


def test_networkpolicy_apiserver_egress_cidr_and_port_substitute_cleanly():
    # The substituted-load side of the assertion above: with a real value in place of the
    # placeholder, the rule is exactly what setup_cluster.sh's live-derived
    # APISERVER_EGRESS_CIDR/PORT (kubectl get endpoints kubernetes) produces.
    doc = _load("networkpolicy.yaml")
    apiserver_rules = [
        rule
        for rule in doc["spec"]["egress"]
        for peer in rule.get("to", [])
        if peer.get("ipBlock", {}).get("cidr") == "203.0.113.9/32"
    ]
    assert len(apiserver_rules) == 1
    assert apiserver_rules[0]["ports"] == [{"protocol": "TCP", "port": 6443}]


_COMPARATOR_PLACEHOLDERS = {**_PLACEHOLDER_SUBSTITUTIONS, "${ANTHROPIC_EGRESS_CIDR}": "192.0.2.0/24"}


def _load_comparator() -> dict:
    text = (K8S_DIR / "networkpolicy-comparator.yaml").read_text(encoding="utf-8")
    for placeholder, value in _COMPARATOR_PLACEHOLDERS.items():
        text = text.replace(placeholder, value)
    return yaml.safe_load(text)


def _is_vllm_rule(rule: dict) -> bool:
    return any(peer.get("ipBlock", {}).get("cidr") == "198.51.100.1/32" for peer in rule.get("to", []))


def test_comparator_policy_differs_from_the_base_only_by_swapping_the_vllm_egress_rule():
    # "Egress is the difference between arms" (as pre-registered for the API comparator arm): anything else drifting
    # between the two files (ingress, DNS, apiserver, Prometheus, Alertmanager) would be a
    # second variable.
    base = _load("networkpolicy.yaml")
    variant = _load_comparator()

    assert variant["metadata"] == base["metadata"]
    assert variant["spec"]["podSelector"] == base["spec"]["podSelector"]
    assert variant["spec"]["policyTypes"] == base["spec"]["policyTypes"]
    assert variant["spec"]["ingress"] == base["spec"]["ingress"]
    shared = [rule for rule in base["spec"]["egress"] if not _is_vllm_rule(rule)]
    assert len(shared) == len(base["spec"]["egress"]) - 1
    assert variant["spec"]["egress"][:-1] == shared


def test_comparator_policy_has_no_vllm_rule_and_one_https_rule_to_the_api_cidr():
    variant = _load_comparator()

    assert not any(_is_vllm_rule(rule) for rule in variant["spec"]["egress"])
    assert variant["spec"]["egress"][-1] == {
        "to": [{"ipBlock": {"cidr": "192.0.2.0/24"}}],
        "ports": [{"protocol": "TCP", "port": 443}],
    }


def test_comparator_policy_api_cidr_is_a_placeholder_and_never_unrestricted():
    raw = yaml.safe_load((K8S_DIR / "networkpolicy-comparator.yaml").read_text(encoding="utf-8"))
    cidrs = [
        peer["ipBlock"]["cidr"]
        for rule in raw["spec"]["egress"]
        for peer in rule.get("to", [])
        if "ipBlock" in peer
    ]

    assert "${ANTHROPIC_EGRESS_CIDR}" in cidrs
    assert "0.0.0.0/0" not in cidrs
    for rule in raw["spec"]["egress"]:
        assert rule.get("to"), "egress rule with no peers is a catch-all"


def test_networkpolicy_prometheus_and_alertmanager_egress_are_separate_rules_not_a_shared_port_list():
    # K8s NetworkPolicy grants every peer in a rule's `to` list access to
    # every port in that same rule's `ports` list -- a combined Prometheus+Alertmanager rule
    # would let the agent reach Prometheus on Alertmanager's port (9093) and vice versa.
    doc = _load("networkpolicy.yaml")
    egress = doc["spec"]["egress"]

    def _single_peer_rules_for(target_name: str) -> list[dict]:
        return [
            rule
            for rule in egress
            if len(rule.get("to", [])) == 1
            and rule["to"][0].get("podSelector", {}).get("matchLabels", {}).get("app.kubernetes.io/name")
            == target_name
        ]

    prometheus_rules = _single_peer_rules_for("prometheus")
    alertmanager_rules = _single_peer_rules_for("alertmanager")
    assert len(prometheus_rules) == 1
    assert len(alertmanager_rules) == 1
    assert prometheus_rules[0]["ports"] == [{"protocol": "TCP", "port": 9090}]
    assert alertmanager_rules[0]["ports"] == [{"protocol": "TCP", "port": 9093}]


def test_networkpolicy_vllm_egress_cidr_and_port_are_placeholders_in_the_committed_file():
    raw = (K8S_DIR / "networkpolicy.yaml").read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)
    egress_rules = doc["spec"]["egress"]
    vllm_rule = next(
        r for r in egress_rules for peer in r.get("to", []) if "ipBlock" in peer and "${VLLM_EGRESS_CIDR}" in peer["ipBlock"]["cidr"]
    )
    assert vllm_rule["to"][0]["ipBlock"]["cidr"] == "${VLLM_EGRESS_CIDR}"
    assert vllm_rule["ports"][0]["port"] == "${VLLM_EGRESS_PORT}"


@pytest.mark.parametrize(
    "name",
    [
        "namespace.yaml",
        "serviceaccount.yaml",
        "clusterrole-read-only.yaml",
        "clusterrolebinding-read-only.yaml",
        "role-app-mutate.yaml",
        "rolebinding-app-mutate.yaml",
        "deployment.yaml",
        "service.yaml",
        "networkpolicy.yaml",
    ],
)
def test_every_manifest_is_a_single_valid_yaml_document(name: str):
    assert _load(name) is not None
