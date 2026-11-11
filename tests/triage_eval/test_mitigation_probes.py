"""Root-cause probes (adapters/mitigation_probes.py) against a fake kubectl.

What is under test is each probe's rule (which symptom fails it, which output becomes "not
measured"), never the cluster. Admission against a real cluster is scenarios/validate_probes.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

from adapters.mitigation_probes import (
    BUGGY_GEO_IMAGE,
    probe_for,
    probe_misconfig_app,
    probe_mongo_admin_readwrite,
    probe_redeploy_without_pv,
    probe_wrong_bin_usage,
    run_probe,
)
from scenarios.validate_probes import admitted


class FakeKube:
    def __init__(self, deployments=None, pods=None, exec_outputs=None):
        self.deployments = deployments or {}
        self.pods = pods or []
        self.exec_outputs = exec_outputs or {}
        self.commands = []

    def get_deployment(self, name, namespace):
        return self.deployments.get(name)

    def list_pods(self, namespace):
        return NS(items=self.pods)

    def exec_command(self, command, input_data=None):
        self.commands.append(command)
        for needle, output in self.exec_outputs.items():
            if needle in command:
                return output
        return ""


def _deployment(image="yinfangchen/hotelreservation:latest", command=("geo",), ready=1):
    container = NS(name="c", image=image, command=list(command))
    return NS(spec=NS(template=NS(spec=NS(containers=[container]))), status=NS(ready_replicas=ready))


def _pod(name, phase="Running"):
    return NS(metadata=NS(name=name), status=NS(phase=phase))


# misconfig_app


def test_misconfig_buggy_image_is_not_fixed():
    kube = FakeKube(deployments={"geo": _deployment(image=BUGGY_GEO_IMAGE)})
    assert probe_misconfig_app(kube, "ns", "geo").root_cause_fixed is False


def test_misconfig_right_image_but_no_ready_replica_is_not_fixed():
    kube = FakeKube(deployments={"geo": _deployment(ready=0)})
    assert probe_misconfig_app(kube, "ns", "geo").root_cause_fixed is False


def test_misconfig_right_image_and_ready_is_fixed():
    kube = FakeKube(deployments={"geo": _deployment()})
    assert probe_misconfig_app(kube, "ns", None).root_cause_fixed is True


# redeploy_without_PV


def _pvc_json(*phases):
    return json.dumps({"items": [{"metadata": {"name": f"pvc{i}"}, "status": {"phase": p}} for i, p in enumerate(phases)]})


def test_redeploy_unbound_pvc_is_not_fixed():
    kube = FakeKube(exec_outputs={"get pvc": _pvc_json("Bound", "Pending")})
    result = probe_redeploy_without_pv(kube, "ns", None)
    assert result.root_cause_fixed is False
    assert "pvc1" in result.detail


def test_redeploy_bound_pvc_but_pending_pod_is_not_fixed():
    kube = FakeKube(exec_outputs={"get pvc": _pvc_json("Bound")}, pods=[_pod("geo-1", "Pending")])
    assert probe_redeploy_without_pv(kube, "ns", None).root_cause_fixed is False


def test_redeploy_all_bound_and_running_is_fixed():
    kube = FakeKube(exec_outputs={"get pvc": _pvc_json("Bound", "Bound")}, pods=[_pod("geo-1")])
    assert probe_redeploy_without_pv(kube, "ns", None).root_cause_fixed is True


def test_redeploy_non_json_or_no_pvc_is_unmeasured_never_a_verdict():
    assert probe_redeploy_without_pv(FakeKube(exec_outputs={"get pvc": "error: boom"}), "ns", None).root_cause_fixed is None
    assert probe_redeploy_without_pv(FakeKube(exec_outputs={"get pvc": '{"items": []}'}), "ns", None).root_cause_fixed is None


# revoke_auth / user_unregistered (same probe)


def _user_json(*roles):
    return json.dumps({"user": "admin", "roles": [{"role": r, "db": d} for r, d in roles]})


def test_mongo_dropped_user_is_not_fixed():
    kube = FakeKube(pods=[_pod("mongodb-geo-abc")], exec_outputs={"getUser": "null\n"})
    result = probe_mongo_admin_readwrite(kube, "ns", "mongodb-geo")
    assert result.root_cause_fixed is False
    assert "does not exist" in result.detail


def test_mongo_user_without_readwrite_on_target_db_is_not_fixed():
    kube = FakeKube(
        pods=[_pod("mongodb-rate-abc")],
        exec_outputs={"getUser": _user_json(("userAdminAnyDatabase", "admin"), ("readWrite", "geo-db"))},
    )
    assert probe_mongo_admin_readwrite(kube, "ns", "mongodb-rate").root_cause_fixed is False


def test_mongo_user_with_readwrite_on_target_db_is_fixed_and_queries_the_right_pod():
    kube = FakeKube(
        pods=[_pod("geo-xyz"), _pod("mongodb-geo-abc")],
        exec_outputs={"getUser": "connecting to: mongodb://...\n" + _user_json(("readWrite", "geo-db"))},
    )
    assert probe_mongo_admin_readwrite(kube, "ns", "mongodb-geo").root_cause_fixed is True
    assert "kubectl exec mongodb-geo-abc -n ns" in kube.commands[0]


def test_mongo_unparseable_shell_output_or_unknown_service_is_unmeasured():
    kube = FakeKube(pods=[_pod("mongodb-geo-abc")], exec_outputs={"getUser": "command terminated with exit code 1"})
    assert probe_mongo_admin_readwrite(kube, "ns", "mongodb-geo").root_cause_fixed is None
    assert probe_mongo_admin_readwrite(kube, "ns", "mongodb-other").root_cause_fixed is None


def test_mongo_missing_pod_is_not_fixed():
    assert probe_mongo_admin_readwrite(FakeKube(), "ns", "mongodb-geo").root_cause_fixed is False


# wrong_bin_usage


def test_wrong_bin_geo_command_is_not_fixed_and_profile_is_fixed():
    assert probe_wrong_bin_usage(FakeKube(deployments={"profile": _deployment(command=("geo",))}), "ns", "profile").root_cause_fixed is False
    assert probe_wrong_bin_usage(FakeKube(deployments={"profile": _deployment(command=("profile",))}), "ns", "profile").root_cause_fixed is True


# registry + run_probe


def test_probe_for_strips_mitigation_suffix_and_unknown_is_none():
    assert probe_for("revoke_auth_mongodb-mitigation-2") is probe_mongo_admin_readwrite
    assert probe_for("user_unregistered_mongodb-mitigation-1") is probe_mongo_admin_readwrite
    assert probe_for("k8s_target_port-misconfig-mitigation-1") is None


def test_run_probe_unknown_scenario_is_unmeasured_with_reason():
    result = run_probe("nope-mitigation-1", FakeKube(), "ns", None)
    assert result["root_cause_fixed"] is None
    assert "no probe" in result["probe_detail"]


def test_run_probe_exception_becomes_probe_error_never_raises():
    class Boom(FakeKube):
        def get_deployment(self, name, namespace):
            raise RuntimeError("api down")

    result = run_probe("misconfig_app_hotel_res-mitigation-1", Boom(), "ns", None)
    assert result["root_cause_fixed"] is None
    assert result["probe_error"].startswith("RuntimeError")


def test_admission_requires_false_after_inject_and_true_after_recover():
    assert admitted({"root_cause_fixed": False}, {"root_cause_fixed": True}) is True
    assert admitted({"root_cause_fixed": True}, {"root_cause_fixed": True}) is False
    assert admitted({"root_cause_fixed": None}, {"root_cause_fixed": True}) is False


# --- candidate families: three promoted 2026-09-25, container_kill still a candidate ----------

from adapters.mitigation_probes import (  # noqa: E402
    CANDIDATE_PROBES,
    NON_EXISTENT_NODE,
    PROBES,
    candidate_probe_for,
    probe_assign_to_non_existent_node,
    probe_container_kill,
    run_candidate_probe,
)


def _kpod(name, service, *, phase="Running", ready=True, restarts=0, waiting=None, deleting=False):
    """A pod shaped like the kubernetes client's V1Pod, only the fields the candidate probes read."""
    state = NS(waiting=NS(reason=waiting) if waiting else None)
    return NS(
        metadata=NS(
            name=name,
            labels={"io.kompose.service": service},
            deletion_timestamp="2026-09-24T00:00:00Z" if deleting else None,
        ),
        status=NS(
            phase=phase,
            conditions=[NS(type="Ready", status="True" if ready else "False")],
            container_statuses=[NS(restart_count=restarts, state=state)],
        ),
    )


def _pod_template(node_selector=None, required_hostnames=None):
    affinity = None
    if required_hostnames is not None:
        expr = NS(key="kubernetes.io/hostname", operator="In", values=list(required_hostnames))
        affinity = NS(
            node_affinity=NS(
                required_during_scheduling_ignored_during_execution=NS(
                    node_selector_terms=[NS(match_expressions=[expr])]
                )
            )
        )
    return NS(spec=NS(template=NS(spec=NS(node_selector=node_selector, affinity=affinity))))


# assign_to_non_existent_node_hotel_res


def test_assign_node_selector_pinned_to_the_injector_node_is_not_fixed():
    kube = FakeKube(
        deployments={"geo": _pod_template({"kubernetes.io/hostname": NON_EXISTENT_NODE})},
        pods=[_kpod("geo-new", "geo", phase="Pending", ready=False)],
    )
    result = probe_assign_to_non_existent_node(kube, "ns", "geo")
    assert result.root_cause_fixed is False
    assert NON_EXISTENT_NODE in result.detail


def test_assign_pin_via_required_node_affinity_is_not_fixed_either():
    kube = FakeKube(deployments={"geo": _pod_template(required_hostnames=[NON_EXISTENT_NODE])})
    assert probe_assign_to_non_existent_node(kube, "ns", "geo").root_cause_fixed is False


def test_assign_no_pin_and_a_ready_geo_pod_is_fixed_and_mongodb_geo_is_not_mistaken_for_geo():
    # mongodb-geo is Ready too, but it is not the geo service: only the label decides.
    kube = FakeKube(
        deployments={"geo": _pod_template()},
        pods=[_kpod("mongodb-geo-1", "mongodb-geo"), _kpod("geo-1", "geo")],
    )
    assert probe_assign_to_non_existent_node(kube, "ns", None).root_cause_fixed is True


def test_assign_no_pin_but_no_ready_geo_pod_is_unmeasured_never_true():
    """Recover deletes (background GC) and re-applies without waiting: the old unschedulable pod
    and the new starting one coexist for a moment. That is None (settle retry), never True, and a
    Ready mongodb-geo must not rescue it."""
    kube = FakeKube(
        deployments={"geo": _pod_template()},
        pods=[
            _kpod("geo-old", "geo", phase="Pending", ready=False, deleting=True),
            _kpod("geo-new", "geo", phase="Pending", ready=False),
            _kpod("mongodb-geo-1", "mongodb-geo"),
        ],
    )
    result = probe_assign_to_non_existent_node(kube, "ns", "geo")
    assert result.root_cause_fixed is None
    assert "geo-old" not in result.detail  # terminating pods say nothing about the current state


class _ApiException(Exception):
    """Shape of kubernetes.client.ApiException that the probe reads (status)."""

    def __init__(self, status):
        super().__init__(f"({status})")
        self.status = status


class _RaisingKube(FakeKube):
    def __init__(self, status):
        super().__init__()
        self.status = status

    def get_deployment(self, name, namespace):
        raise _ApiException(self.status)


def test_assign_deployment_404_is_unmeasured_never_a_verdict():
    """Live, KubeCtl.get_deployment raises ApiException 404 (never returns None): the injector and
    recover both pass through delete-then-apply, so absence is a moment to wait out."""
    result = probe_assign_to_non_existent_node(_RaisingKube(404), "ns", "geo")
    assert result.root_cause_fixed is None
    assert "404" in result.detail


def test_assign_other_api_errors_still_surface_as_probe_error():
    # Promoted 2026-09-25: assign_to_non_existent_node_hotel_res is in PROBES now, so it goes
    # through run_probe/probe_for like any other admitted family, not run_candidate_probe.
    result = run_probe("assign_to_non_existent_node_hotel_res-mitigation-1", _RaisingKube(500), "ns", "geo")
    assert result["root_cause_fixed"] is None
    assert result["probe_error"].startswith("_ApiException")


# container_kill


def _podchaos(*, service="geo", action="container-kill", injected=1, phase="Injected"):
    records = [] if injected is None else [{"id": "ns/geo-1/hotel-reserv-geo", "phase": phase, "injectedCount": injected}]
    return {
        "metadata": {"name": "container-kill"},
        "spec": {"action": action, "selector": {"labelSelectors": {"io.kompose.service": service}}},
        "status": {"experiment": {"containerRecords": records}},
    }


def _chaos_json(*experiments):
    return json.dumps({"items": list(experiments)})


def test_container_kill_active_experiment_with_an_injection_record_is_not_fixed():
    kube = FakeKube(
        exec_outputs={"get podchaos": _chaos_json(_podchaos(injected=1, phase="Not Injected"))},
        pods=[_kpod("geo-1", "geo", restarts=1)],
    )
    result = probe_container_kill(kube, "ns", "geo")
    assert result.root_cause_fixed is False
    assert "restarts=1" in result.detail
    assert "kubectl get podchaos -n ns -o json" in kube.commands


def test_container_kill_experiment_that_never_killed_is_unmeasured_so_a_no_op_fault_is_never_admitted():
    kube = FakeKube(exec_outputs={"get podchaos": _chaos_json(_podchaos(injected=None))}, pods=[_kpod("geo-1", "geo")])
    after_inject = probe_container_kill(kube, "ns", "geo")
    assert after_inject.root_cause_fixed is None
    recovered = FakeKube(exec_outputs={"get podchaos": _chaos_json()}, pods=[_kpod("geo-1", "geo")])
    after_recover = probe_container_kill(recovered, "ns", "geo")
    assert after_recover.root_cause_fixed is True
    assert admitted({"root_cause_fixed": after_inject.root_cause_fixed}, {"root_cause_fixed": True}) is False


def test_container_kill_experiment_on_another_service_or_action_is_ignored():
    kube = FakeKube(
        exec_outputs={
            "get podchaos": _chaos_json(_podchaos(service="rate"), _podchaos(action="pod-failure")),
        },
        pods=[_kpod("geo-1", "geo")],
    )
    assert probe_container_kill(kube, "ns", "geo").root_cause_fixed is True


def test_container_kill_no_experiment_but_crashlooping_is_not_fixed():
    kube = FakeKube(
        exec_outputs={"get podchaos": _chaos_json()},
        pods=[_kpod("geo-1", "geo", ready=False, restarts=4, waiting="CrashLoopBackOff")],
    )
    assert probe_container_kill(kube, "ns", "geo").root_cause_fixed is False


def test_container_kill_no_experiment_and_no_ready_pod_is_unmeasured():
    kube = FakeKube(exec_outputs={"get podchaos": _chaos_json()}, pods=[_kpod("geo-1", "geo", ready=False)])
    assert probe_container_kill(kube, "ns", "geo").root_cause_fixed is None


def test_container_kill_without_the_podchaos_crd_is_unmeasured_never_a_verdict():
    kube = FakeKube(exec_outputs={"get podchaos": 'error: the server doesn\'t have a resource type "podchaos"'})
    result = probe_container_kill(kube, "ns", "geo")
    assert result.root_cause_fixed is None
    assert "Chaos Mesh" in result.detail


# candidate registry


def test_candidate_registry_is_disjoint_from_the_admitted_one_and_probe_for_never_sees_it():
    assert not set(CANDIDATE_PROBES) & set(PROBES)
    for family in CANDIDATE_PROBES:
        # probe_for is the admitted lookup; a candidate family name must resolve to nothing there
        assert probe_for(f"{family}-mitigation-1") is None
    assert candidate_probe_for("container_kill") is probe_container_kill
    # Promoted 2026-09-25: these three moved to PROBES, so they no longer resolve as candidates.
    assert candidate_probe_for("revoke_auth_mongodb_rate") is None
    assert candidate_probe_for("user_unregistered_mongodb_rate") is None
    assert candidate_probe_for("assign_to_non_existent_node_hotel_res") is None
    assert candidate_probe_for("misconfig_app_hotel_res") is None


def test_run_candidate_probe_still_covers_container_kill():
    kube = FakeKube(exec_outputs={"get podchaos": _chaos_json()}, pods=[_kpod("geo-1", "geo")])
    result = run_candidate_probe("container_kill", kube, "ns", "geo")
    assert result["root_cause_fixed"] is True


def test_mongo_rate_reads_rate_db_through_the_problem_faulty_service():
    # Promoted 2026-09-25: revoke_auth_mongodb_rate is in PROBES now, same probe_for/run_probe
    # path as any other admitted family (still faulty_service-driven, so still reads rate-db).
    kube = FakeKube(
        pods=[_pod("mongodb-rate-abc")],
        exec_outputs={"getUser": _user_json(("readWrite", "rate-db"))},
    )
    result = run_probe("revoke_auth_mongodb_rate-mitigation-1", kube, "ns", "mongodb-rate")
    assert result["root_cause_fixed"] is True
    assert "rate-db" in result["probe_detail"]


def test_run_probe_exceptions_are_unmeasured_never_raise_for_a_promoted_family():
    class Boom(FakeKube):
        def get_deployment(self, name, namespace):
            raise RuntimeError("api down")

    result = run_probe("assign_to_non_existent_node_hotel_res-mitigation-1", Boom(), "ns", "geo")
    assert result["root_cause_fixed"] is None
    assert result["probe_error"].startswith("RuntimeError")
