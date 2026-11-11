"""Candidate fault families and their promotion (2026-09-25).

Three of the four candidates (assign_to_non_existent_node_hotel_res, revoke_auth_mongodb_rate,
user_unregistered_mongodb_rate) were promoted into the admitted catalog (PROBES) after a live
admission run showed 2/2 detection on each; container_kill stayed a candidate (0/2). The load-bearing
property below is that the promotion changed exactly these registries and nothing else silently:
PROBES/MECHANISM_BY_FAMILY now have eight keys, CANDIDATE_PROBES/CANDIDATE_MECHANISM_BY_FAMILY
have exactly one (container_kill), and the pre-registered scenarios/schedules/
r1-admission-seed-600.json covers all eight admitted families with the >= 2 injections/family quota. The rest covers
the remaining candidate path: registries agree, validate_probes --candidates measures with the
candidate probe, and the local HotelReservation wrapper is registered into an orchestrator's
registry instance, never into the AIOpsLab clone.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from adapters import aiopslab_problems
from adapters.aiopslab_problems import (
    ASSIGN_HOTEL_RES_PROBLEM_ID,
    LOCAL_PROBLEM_IDS,
    WRONG_BIN_PROBLEM_ID,
    PROBLEM_ID_BY_FAMILY,
    LocalProblemConflict,
    register_local_problems,
)
from adapters.mitigation_probes import CANDIDATE_PROBES, PROBES
from fake_deployment_kube import FakeDeploymentKube
from scenarios import validate_probes
from scenarios.fault_schedule import DEFAULT_FAULT_FAMILIES, EntryKind, FaultSchedule, ScheduleEntry, generate_schedule
from scenarios.run_continuous_admission import (
    CANDIDATE_MECHANISM_BY_FAMILY,
    MECHANISM_BY_FAMILY,
    R1Error,
    generate_r1_schedule,
    validate_schedule_for_r1,
)

ADMITTED = {
    "misconfig_app_hotel_res",
    "redeploy_without_PV",
    "revoke_auth_mongodb",
    "user_unregistered_mongodb",
    "wrong_bin_usage",
    "assign_to_non_existent_node_hotel_res",
    "revoke_auth_mongodb_rate",
    "user_unregistered_mongodb_rate",
}
SCHEDULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "triage-eval"
    / "scenarios"
    / "schedules"
    / "r1-admission-seed-600.json"
)
# sha256 of the committed, pre-registered admission schedule as of the promotion (2026-09-25).
# Changing that file is a new pre-registration, never a side effect of adding a candidate family.
SCHEDULE_SHA256 = "42de4c426b6e3b438e5abc34c898e685ba4b15ab161e43f519d24756aefcd3b1"


# --- nothing beyond the promoted three moved --------------------------------------------------


def test_admitted_catalog_is_now_the_eight_promoted_families():
    assert set(PROBES) == ADMITTED
    assert set(MECHANISM_BY_FAMILY) == ADMITTED
    assert DEFAULT_FAULT_FAMILIES == tuple(sorted(ADMITTED))


def test_committed_seed_600_schedule_is_byte_identical_and_admitted_only():
    raw = SCHEDULE_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SCHEDULE_SHA256
    payload = json.loads(raw)
    schedule = FaultSchedule.from_json(payload)
    assert {entry.fault_family for entry in schedule.fault_episodes} == ADMITTED
    # Unlike the seed-128 file it replaces (generated before the quota), this schedule is
    # pre-registered to satisfy the admission run's quota (generate_r1_schedule), so it must actually validate, not just avoid candidates.
    validate_schedule_for_r1(schedule)
    assert not any(family in ADMITTED for family in CANDIDATE_PROBES)


def test_r1_schedule_generation_still_demands_exactly_the_admitted_families():
    schedule = generate_r1_schedule(1)
    validate_schedule_for_r1(schedule)
    assert {entry.fault_family for entry in schedule.fault_episodes} == ADMITTED


@pytest.mark.parametrize("family", sorted(CANDIDATE_PROBES))
def test_r1_validation_still_rejects_a_candidate_family(family):
    payload = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    schedule = FaultSchedule.from_json(payload)
    extra = ScheduleEntry(EntryKind.FAULT, schedule.entries[-1].end_minute + 30, 20, family)
    with pytest.raises(R1Error, match="without an admitted probe"):
        validate_schedule_for_r1(FaultSchedule(schedule.seed, schedule.window_hours, schedule.entries + (extra,)))


def test_default_probe_admission_run_is_unchanged():
    assert validate_probes.DEFAULT_SCENARIOS == [
        "misconfig_app_hotel_res-mitigation-1",
        "redeploy_without_PV-mitigation-1",
        "revoke_auth_mongodb-mitigation-1",
        "revoke_auth_mongodb-mitigation-2",
        "user_unregistered_mongodb-mitigation-1",
        "user_unregistered_mongodb-mitigation-2",
        "wrong_bin_usage-mitigation-1",
    ]


# --- the candidate registries agree with each other and stay apart from the admitted ones --------


def test_candidate_registries_share_one_key_set_disjoint_from_the_admitted_one():
    keys = set(CANDIDATE_PROBES)
    assert keys == {"container_kill"}
    assert set(CANDIDATE_MECHANISM_BY_FAMILY) == keys
    assert not keys & set(PROBES)
    assert not keys & set(MECHANISM_BY_FAMILY)


def test_problem_id_map_resolves_every_family_admitted_or_candidate():
    """One permanent map, independent of candidate status: every REGULAR family (problem id is
    its own name plus -mitigation-1) resolves that way whether admitted or candidate, and the
    irregular ones (promoted or not) keep their own irregular id -- promotion moves a key between
    PROBES/CANDIDATE_PROBES, never changes what it resolves to (see the README's promotion
    recipe)."""
    irregular = {
        "container_kill",
        "revoke_auth_mongodb_rate",
        "user_unregistered_mongodb_rate",
        "assign_to_non_existent_node_hotel_res",
        "wrong_bin_usage",  # revision-preserving wrapper, its own local id
    }
    assert set(PROBLEM_ID_BY_FAMILY) == set(PROBES) | set(CANDIDATE_PROBES)
    for family in set(PROBES) - irregular:
        assert PROBLEM_ID_BY_FAMILY[family] == f"{family}-mitigation-1"
    assert PROBLEM_ID_BY_FAMILY["container_kill"] == "container_kill-detection"  # no numeric suffix
    assert PROBLEM_ID_BY_FAMILY["revoke_auth_mongodb_rate"] == "revoke_auth_mongodb-mitigation-2"
    assert PROBLEM_ID_BY_FAMILY["user_unregistered_mongodb_rate"] == "user_unregistered_mongodb-mitigation-2"
    assert PROBLEM_ID_BY_FAMILY["assign_to_non_existent_node_hotel_res"] == ASSIGN_HOTEL_RES_PROBLEM_ID
    assert PROBLEM_ID_BY_FAMILY["wrong_bin_usage"] == WRONG_BIN_PROBLEM_ID
    assert LOCAL_PROBLEM_IDS <= set(PROBLEM_ID_BY_FAMILY.values())


@pytest.mark.parametrize("family", sorted(CANDIDATE_PROBES))
def test_promoting_a_candidate_per_the_readme_keeps_its_problem_id(monkeypatch, family):
    """README recipe: move the key from CANDIDATE_PROBES to PROBES (and the mechanism map). The
    injector must then resolve the same problem, never f"{family}-mitigation-1"."""
    from adapters import mitigation_probes
    from scenarios import run_continuous_admission as rca

    expected = PROBLEM_ID_BY_FAMILY[family]
    monkeypatch.setitem(mitigation_probes.PROBES, family, mitigation_probes.CANDIDATE_PROBES[family])
    monkeypatch.delitem(mitigation_probes.CANDIDATE_PROBES, family)
    monkeypatch.setitem(rca.MECHANISM_BY_FAMILY, family, rca.CANDIDATE_MECHANISM_BY_FAMILY[family])
    monkeypatch.delitem(rca.CANDIDATE_MECHANISM_BY_FAMILY, family)

    injector = rca.AIOpsLabFaultInjector()
    assert family in injector._families
    assert injector._problem_id(family) == expected


# --- the local HotelReservation wrapper -------------------------------------------------------------


def test_register_local_problems_adds_only_to_the_given_registry_instance_idempotently():
    factory = object()
    registry = types.SimpleNamespace(PROBLEM_REGISTRY={"x-detection-1": "upstream"})
    assert register_local_problems(registry, {ASSIGN_HOTEL_RES_PROBLEM_ID: factory}) == [ASSIGN_HOTEL_RES_PROBLEM_ID]
    assert registry.PROBLEM_REGISTRY == {"x-detection-1": "upstream", ASSIGN_HOTEL_RES_PROBLEM_ID: factory}
    assert register_local_problems(registry, {ASSIGN_HOTEL_RES_PROBLEM_ID: factory}) == []  # same factory
    assert registry.PROBLEM_REGISTRY[ASSIGN_HOTEL_RES_PROBLEM_ID] is factory


def test_register_local_problems_raises_when_upstream_already_defines_the_id():
    """If AIOpsLab ever ships this id, silently yielding would run upstream's problem while the
    family's probe still assumes geo + extra-node."""
    registry = types.SimpleNamespace(PROBLEM_REGISTRY={ASSIGN_HOTEL_RES_PROBLEM_ID: "upstream-factory"})
    with pytest.raises(LocalProblemConflict, match="already in the AIOpsLab registry"):
        register_local_problems(registry, {ASSIGN_HOTEL_RES_PROBLEM_ID: object()})
    assert registry.PROBLEM_REGISTRY[ASSIGN_HOTEL_RES_PROBLEM_ID] == "upstream-factory"


def _install_fake_modules(monkeypatch, modules):
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)


def test_assign_wrapper_swaps_in_hotel_res_geo_and_injects_by_revision(monkeypatch, tmp_path):
    """Against a fake aiopslab package: the wrapper subclasses AIOpsLab's own SocialNetwork
    problem (eval stays upstream), but inject/recover are the revision-preserving pair
    (adapters/revision_faults.py), never upstream's delete + apply."""
    calls = []
    kube = FakeDeploymentKube("geo", "test-hotel-reservation", [{"name": "hotel-reserv-geo", "command": ["geo"]}])

    class HotelReservation:
        namespace = "test-hotel-reservation"

    class MitigationTask:
        def __init__(self, app):
            calls.append(("MitigationTask", app.namespace))

    class SocialNetMitigation:
        def __init__(self):
            raise AssertionError("the SocialNetwork __init__ would deploy the wrong app")

        def inject_fault(self):
            raise AssertionError("upstream inject deletes and re-applies the deployment")

        def recover_fault(self):
            raise AssertionError("upstream recover deletes and re-applies the deployment")

    _install_fake_modules(
        monkeypatch,
        {
            "aiopslab": {},
            "aiopslab.generators": {},
            "aiopslab.generators.workload": {},
            "aiopslab.generators.workload.wrk": {"Wrk": object},
            "aiopslab.orchestrator": {},
            "aiopslab.orchestrator.problems": {},
            "aiopslab.orchestrator.problems.assign_non_existent_node": {},
            "aiopslab.orchestrator.problems.assign_non_existent_node.assign_non_existent_node_social_net": {
                "AssignNonExistentNodeSocialNetMitigation": SocialNetMitigation
            },
            "aiopslab.orchestrator.problems.assign_non_existent_node.helpers": {"get_frontend_url": lambda app: ""},
            "aiopslab.orchestrator.tasks": {"MitigationTask": MitigationTask},
            "aiopslab.paths": {"TARGET_MICROSERVICES": tmp_path},
            "aiopslab.service": {},
            "aiopslab.service.apps": {},
            "aiopslab.service.apps.hotelres": {"HotelReservation": HotelReservation},
            "aiopslab.service.kubectl": {"KubeCtl": lambda: kube},
        },
    )
    aiopslab_problems._assign_hotel_res_class.cache_clear()
    try:
        problem = aiopslab_problems._assign_hotel_res_class()()
        assert isinstance(problem, SocialNetMitigation)
        assert problem.payload_script.name == "mixed-workload_type_1.lua"
        assert calls == [("MitigationTask", "test-hotel-reservation")]

        problem.inject_fault()
        assert kube.deployment["spec"]["strategy"]["type"] == "Recreate"
        assert kube.deployment["spec"]["template"]["spec"]["nodeSelector"] == {"kubernetes.io/hostname": "extra-node"}
        recovered = problem.recover_fault()
        assert "nodeSelector" not in kube.deployment["spec"]["template"]["spec"]
        assert kube.deployment["spec"]["strategy"]["type"] == "RollingUpdate"
        assert not any("kubectl delete" in c or "kubectl apply" in c for c in kube.commands)
        # The revision-injector live revalidation reads this to confirm "recovered by undo", not just "did not raise".
        assert recovered == {"undo_ran": True, "to_revision": "3", "strategy_restored": "RollingUpdate"}
    finally:
        aiopslab_problems._assign_hotel_res_class.cache_clear()


def test_wrong_bin_wrapper_keeps_upstream_problem_and_injects_by_revision(monkeypatch):
    kube = FakeDeploymentKube("profile", "test-hotel-reservation", [{"name": "hotel-reserv-profile", "command": ["profile"]}])

    class WrongBinUsageMitigation:
        def __init__(self, faulty_service="profile"):
            self.kubectl = kube
            self.namespace = "test-hotel-reservation"
            self.faulty_service = faulty_service

        def inject_fault(self):
            raise AssertionError("upstream inject deletes and re-applies the deployment")

        def recover_fault(self):
            raise AssertionError("upstream recover deletes and re-applies the deployment")

    _install_fake_modules(
        monkeypatch,
        {
            "aiopslab": {},
            "aiopslab.orchestrator": {},
            "aiopslab.orchestrator.problems": {},
            "aiopslab.orchestrator.problems.wrong_bin_usage": {},
            "aiopslab.orchestrator.problems.wrong_bin_usage.wrong_bin_usage": {
                "WrongBinUsageMitigation": WrongBinUsageMitigation
            },
        },
    )
    aiopslab_problems._wrong_bin_class.cache_clear()
    try:
        problem = aiopslab_problems._wrong_bin_class()()
        assert isinstance(problem, WrongBinUsageMitigation) and problem.faulty_service == "profile"

        problem.inject_fault()
        assert kube.deployment["spec"]["template"]["spec"]["containers"][0]["command"] == ["geo"]
        recovered = problem.recover_fault()
        assert kube.deployment["spec"]["template"]["spec"]["containers"][0]["command"] == ["profile"]
        assert not any("kubectl delete" in c or "kubectl apply" in c for c in kube.commands)
        assert recovered == {"undo_ran": True, "to_revision": "3", "strategy_restored": "RollingUpdate"}
    finally:
        aiopslab_problems._wrong_bin_class.cache_clear()


def test_revision_recover_before_inject_refuses():
    problem = types.SimpleNamespace(faulty_service="geo")
    with pytest.raises(aiopslab_problems.RevisionFaultError, match="before inject_fault"):
        aiopslab_problems._revision_recover(problem)


# --- validate_probes --candidates -------------------------------------------------------------------


class _ChaosKube:
    """PodChaos present with one recorded kill until recover_fault() deletes it."""

    def __init__(self):
        self.experiment_alive = True

    def exec_command(self, command, input_data=None):
        items = []
        if self.experiment_alive:
            items.append(
                {
                    "metadata": {"name": "container-kill"},
                    "spec": {"action": "container-kill", "selector": {"labelSelectors": {"io.kompose.service": "geo"}}},
                    "status": {"experiment": {"containerRecords": [{"phase": "Injected", "injectedCount": 1}]}},
                }
            )
        return json.dumps({"items": items})

    def list_pods(self, namespace):
        pod = types.SimpleNamespace(
            metadata=types.SimpleNamespace(name="geo-1", labels={"io.kompose.service": "geo"}, deletion_timestamp=None),
            status=types.SimpleNamespace(
                phase="Running",
                conditions=[types.SimpleNamespace(type="Ready", status="True")],
                container_statuses=[types.SimpleNamespace(restart_count=1, state=types.SimpleNamespace(waiting=None))],
            ),
        )
        return types.SimpleNamespace(items=[pod])


def _chaos_env_factory(seen):
    kube = _ChaosKube()

    class Problem:
        kubectl = kube
        namespace = "test-hotel-reservation"
        faulty_service = "geo"

        def recover_fault(self):
            kube.experiment_alive = False

    class Env:
        def __init__(self, scenario_id):
            seen.append(scenario_id)
            self._orchestrator = types.SimpleNamespace(session=types.SimpleNamespace(problem=Problem()))

        def task_description(self):
            return "deployed"

        def finalize(self, solution, history, duration):
            return {"success": None}

    return Env


def test_validate_candidate_measures_with_the_candidate_probe_on_the_resolved_problem_id():
    seen = []
    row = validate_probes.validate_candidate("container_kill", _chaos_env_factory(seen), timeout_sec=0)
    assert seen == ["container_kill-detection"]
    assert row["candidate_family"] == "container_kill"
    assert row["after_inject"]["root_cause_fixed"] is False
    assert row["after_recover"]["root_cause_fixed"] is True
    assert row["admitted"] is True


def test_validate_candidate_refuses_an_admitted_or_unknown_family():
    with pytest.raises(ValueError, match="is not a candidate family"):
        validate_probes.validate_candidate("misconfig_app_hotel_res", lambda _id: None)


def test_make_candidate_environment_registers_the_local_wrapper_before_deploying(monkeypatch):
    import mlx_factories

    registry = types.SimpleNamespace(PROBLEM_REGISTRY={})
    env = types.SimpleNamespace(_orchestrator=types.SimpleNamespace(probs=registry))
    monkeypatch.setattr(mlx_factories, "make_environment", lambda problem_id: env)
    monkeypatch.setattr(aiopslab_problems, "local_problem_factories", lambda: {ASSIGN_HOTEL_RES_PROBLEM_ID: object})

    assert validate_probes.make_candidate_environment(ASSIGN_HOTEL_RES_PROBLEM_ID) is env
    assert ASSIGN_HOTEL_RES_PROBLEM_ID in registry.PROBLEM_REGISTRY


def test_main_candidates_runs_every_candidate_family_into_its_own_output(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(
        validate_probes,
        "validate_candidate",
        lambda family: ran.append(family) or {"scenario_id": PROBLEM_ID_BY_FAMILY[family], "admitted": False},
    )
    out = tmp_path / "candidates.json"
    validate_probes.main(["--candidates", "--out", str(out)])
    assert ran == sorted(CANDIDATE_PROBES)
    assert [row["scenario_id"] for row in json.loads(out.read_text())] == [PROBLEM_ID_BY_FAMILY[f] for f in ran]


def test_main_candidates_records_a_failing_family_and_keeps_going(monkeypatch, tmp_path):
    def boom(family):
        if family == "container_kill":
            raise RuntimeError("chaos-mesh webhook not ready")
        return {"scenario_id": family, "admitted": True}

    monkeypatch.setattr(validate_probes, "validate_candidate", boom)
    out = tmp_path / "candidates.json"
    validate_probes.main(["--candidates", "--scenario", "container_kill", "--scenario", "revoke_auth_mongodb_rate", "--out", str(out)])
    rows = json.loads(out.read_text())
    assert rows[0] == {
        "scenario_id": "container_kill",
        "error": "RuntimeError: chaos-mesh webhook not ready",
        "admitted": False,
        "candidate_family": "container_kill",
    }
    assert rows[1]["admitted"] is True
