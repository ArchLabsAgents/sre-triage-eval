"""Fault family -> AIOpsLab problem id, for admitted and candidate families alike, plus the
local AIOpsLab problem wrappers two families need.

Why a map and not f"{family}-mitigation-1": two families have AIOpsLab ids of another shape
(container_kill-detection has no numeric suffix; the mongo rate variants are fixed -mitigation-2
ids), and two resolve to local wrappers (HotelReservation assign, revisioned wrong_bin). The map is
PERMANENT and says nothing about admission: whether a family is admitted is decided only by
mitigation_probes.PROBES (vs CANDIDATE_PROBES) and run_continuous_admission.MECHANISM_BY_FAMILY
(vs CANDIDATE_MECHANISM_BY_FAMILY). Promoting a candidate moves its key between those registries
and never touches this map, so the injector keeps resolving it to the same problem id after
promotion. The admitted entries are the ids the injector built before candidates existed
(f"{family}-mitigation-1"), pinned by tests, except wrong_bin_usage, whose injection itself
changed to the revision-preserving one (a new id, on purpose, see below).

Representation of the candidate families, one family key each:

- assign_to_non_existent_node_hotel_res -> local wrapper (below). AIOpsLab's registry only has
  the SocialNetwork variant (assign_to_non_existent_node_social_net-*, target user-service). The
  fault itself is app-agnostic (inject_virtual.VirtualizationFaultInjector, nodeSelector
  kubernetes.io/hostname=extra-node plus delete/apply), so the wrapper reuses AIOpsLab's own
  problem class and only swaps the app (HotelReservation), the target (geo, where the measured
  KubePodNotReady already lands) and the workload. The AIOpsLab clone is never patched: the
  wrapper is registered into an Orchestrator's ProblemRegistry INSTANCE at runtime. The wrapper
  also replaces inject/recover with the revision-preserving injection in
  adapters/revision_faults.py (Recreate + template patch instead of delete + apply), so
  `rollout undo` recovers it and the family's disposition is "agir" (act) in harness/fault_table.py. Unlike
  upstream's recover_fault (no return value), this wrapper's recover_fault returns
  RevisionFault.recover()'s detail dict (undo_ran/to_revision/strategy_restored) so a caller can
  confirm live that the undo actually ran, not just that recovery did not raise.
- wrong_bin_usage -> local wrapper (WRONG_BIN_PROBLEM_ID) for the same reason (so `rollout
  undo` can recover it): AIOpsLab's WrongBinUsageMitigation with inject/recover swapped for the
  revision-preserving pair. A new id, not an override of wrong_bin_usage-mitigation-1:
  register_local_problems refuses to shadow an upstream id, and the id says the injection
  differs from what the earlier detectability runs measured.
- container_kill -> container_kill-detection (AIOpsLab ships only detection/localization tasks).
- revoke_auth_mongodb_rate / user_unregistered_mongodb_rate -> the -mitigation-2 variants
  (faulty_service mongodb-rate). A distinct family key, not a suffix knob on the existing
  family, because the family key is the unit a schedule entry, a FamilyResult and a detection
  verdict are keyed by: a "revoke_auth_mongodb" whose target silently depends on an injector
  option would merge geo and rate episodes into one verdict and make the committed schedule
  ambiguous about which service it faulted. Same mechanism as the geo variants (mongo_auth).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from adapters.revision_faults import RevisionFault, RevisionFaultError, assign_fault, wrong_bin_fault

ASSIGN_HOTEL_RES_PROBLEM_ID = "assign_to_non_existent_node_hotel_res-mitigation-1"
# The service whose pod already fires KubePodNotReady under the three geo faults measured locally
# (misconfig_app_hotel_res, revoke_auth_mongodb, user_unregistered_mongodb): pinning the same
# service makes the alert collide on purpose, so the alert alone cannot tell the faults apart.
ASSIGN_HOTEL_RES_TARGET_SERVICE = "geo"
# Same workload script and rate as MisconfigAppHotelResBaseTask (the geo misconfig problem).
HOTEL_RES_PAYLOAD_SCRIPT = "hotelReservation/wrk2/scripts/hotel-reservation/mixed-workload_type_1.lua"

# AIOpsLab's wrong_bin_usage-mitigation-1 with the revision-preserving injection.
WRONG_BIN_PROBLEM_ID = "wrong_bin_usage_revisioned-mitigation-1"
WRONG_BIN_TARGET_SERVICE = "profile"

PROBLEM_ID_BY_FAMILY: dict[str, str] = {
    # admitted (PROBES): the ids the injector always used
    "misconfig_app_hotel_res": "misconfig_app_hotel_res-mitigation-1",
    "redeploy_without_PV": "redeploy_without_PV-mitigation-1",
    "revoke_auth_mongodb": "revoke_auth_mongodb-mitigation-1",
    "user_unregistered_mongodb": "user_unregistered_mongodb-mitigation-1",
    # was wrong_bin_usage-mitigation-1 before the revision-preserving injection
    "wrong_bin_usage": WRONG_BIN_PROBLEM_ID,
    # candidates (CANDIDATE_PROBES today; the entry stays put when promoted)
    "assign_to_non_existent_node_hotel_res": ASSIGN_HOTEL_RES_PROBLEM_ID,
    "container_kill": "container_kill-detection",
    "revoke_auth_mongodb_rate": "revoke_auth_mongodb-mitigation-2",
    "user_unregistered_mongodb_rate": "user_unregistered_mongodb-mitigation-2",
}

# Problem ids that do not exist in AIOpsLab's registry and must be registered first.
LOCAL_PROBLEM_IDS = frozenset({ASSIGN_HOTEL_RES_PROBLEM_ID, WRONG_BIN_PROBLEM_ID})


class LocalProblemConflict(RuntimeError):
    """AIOpsLab already defines a problem under one of our local ids."""


def problem_id_for(family: str) -> str | None:
    return PROBLEM_ID_BY_FAMILY.get(family)


@lru_cache(maxsize=None)
def _assign_hotel_res_class() -> type:
    """Built lazily (and once, so type(instance) is stable for control_mode's mock.patch.object and
    re-registration is idempotent): importing aiopslab needs AIOPSLAB_HOME on sys.path, which unit
    tests and CI do not have."""
    from aiopslab.generators.workload.wrk import Wrk
    from aiopslab.orchestrator.problems.assign_non_existent_node.assign_non_existent_node_social_net import (
        AssignNonExistentNodeSocialNetMitigation,
    )
    from aiopslab.orchestrator.problems.assign_non_existent_node.helpers import get_frontend_url
    from aiopslab.orchestrator.tasks import MitigationTask
    from aiopslab.paths import TARGET_MICROSERVICES
    from aiopslab.service.apps.hotelres import HotelReservation
    from aiopslab.service.kubectl import KubeCtl

    class AssignNonExistentNodeHotelResMitigation(AssignNonExistentNodeSocialNetMitigation):
        """AIOpsLab's SocialNetwork problem with HotelReservation/geo swapped in.

        eval is inherited unchanged. __init__ replaces the base task's (which would construct a
        SocialNetwork app) and sets the same attributes it sets; start_workload replaces the
        SocialNetwork compose-post endpoint; inject_fault/recover_fault are the revision-preserving
        pair (adapters/revision_faults.py), same nodeSelector artifact as upstream.
        """

        def __init__(self) -> None:
            self.app = HotelReservation()
            self.kubectl = KubeCtl()
            self.namespace = self.app.namespace
            self.faulty_service = ASSIGN_HOTEL_RES_TARGET_SERVICE
            self.payload_script = TARGET_MICROSERVICES / HOTEL_RES_PAYLOAD_SCRIPT
            MitigationTask.__init__(self, self.app)

        def start_workload(self) -> None:
            print("== Start Workload ==")
            wrk = Wrk(rate=10, dist="exp", connections=2, duration=10, threads=2)
            wrk.start_workload(payload_script=self.payload_script, url=get_frontend_url(self.app))

        inject_fault = _revision_inject(assign_fault)
        recover_fault = _revision_recover

    return AssignNonExistentNodeHotelResMitigation


def _revision_inject(make_fault: Callable[..., RevisionFault]) -> Callable[[Any], None]:
    def inject_fault(self) -> None:
        print("== Fault Injection (revision-preserving) ==")
        self.revision_fault = make_fault(self.kubectl, self.namespace, self.faulty_service)
        self.revision_fault.inject()
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    return inject_fault


def _revision_recover(self) -> dict:
    print("== Fault Recovery (revision-preserving) ==")
    fault = getattr(self, "revision_fault", None)
    if fault is None:
        raise RevisionFaultError(f"recover_fault on {self.faulty_service!r} before inject_fault")
    detail = fault.recover()
    print(f"Recovered: {detail}")
    return detail


@lru_cache(maxsize=None)
def _wrong_bin_class() -> type:
    """Lazy and cached for the same reasons as _assign_hotel_res_class."""
    from aiopslab.orchestrator.problems.wrong_bin_usage.wrong_bin_usage import WrongBinUsageMitigation

    class WrongBinUsageRevisionedMitigation(WrongBinUsageMitigation):
        """AIOpsLab's own problem (HotelReservation, profile, same workload and eval) with the
        revision-preserving inject/recover."""

        def __init__(self) -> None:
            WrongBinUsageMitigation.__init__(self, faulty_service=WRONG_BIN_TARGET_SERVICE)

        inject_fault = _revision_inject(wrong_bin_fault)
        recover_fault = _revision_recover

    return WrongBinUsageRevisionedMitigation


def local_problem_factories() -> dict[str, Callable[[], Any]]:
    return {ASSIGN_HOTEL_RES_PROBLEM_ID: _assign_hotel_res_class(), WRONG_BIN_PROBLEM_ID: _wrong_bin_class()}


def register_local_problems(registry: Any, factories: dict[str, Callable[[], Any]] | None = None) -> list[str]:
    """Add the local problems to one ProblemRegistry instance (its PROBLEM_REGISTRY dict), never
    to the AIOpsLab clone. Re-registering the SAME factory is a no-op. An id AIOpsLab already
    defines with another factory raises LocalProblemConflict: the probes for that family assume
    this wrapper's geo target and extra-node pin, so silently running upstream's problem under
    the family name would measure a different fault. Returns the ids actually added."""
    table = registry.PROBLEM_REGISTRY
    added = []
    for problem_id, factory in (local_problem_factories() if factories is None else factories).items():
        existing = table.get(problem_id)
        if existing is None:
            table[problem_id] = factory
            added.append(problem_id)
        elif existing is not factory:
            raise LocalProblemConflict(
                f"{problem_id!r} already in the AIOpsLab registry ({existing!r}); the local wrapper "
                "and its probe would disagree with it. Rename the local id or retire the wrapper."
            )
    return added
