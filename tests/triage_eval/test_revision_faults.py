"""Revision-preserving injection: assign_to_non_existent_node_hotel_res and wrong_bin_usage land as a new revision.

The load-bearing property is the one harness/fault_table.py now relies on: after inject(), the
table's canonical C2 action (`rollout undo --to-revision=<good revision>`) recovers the fault, and
the old pod does not keep serving meanwhile (Recreate). Checked against an in-memory deployment
that bumps revisions the way the controller does (fake_deployment_kube.py).
"""

from __future__ import annotations

import pytest

from adapters.revision_faults import (
    REVISION_ANNOTATION,
    RevisionFaultError,
    assign_fault,
    assign_pinned,
    wrong_bin_fault,
    wrong_bin_swapped,
)
from fake_deployment_kube import FakeDeploymentKube
from harness.fault_table import FAULT_TABLE, REVISION_PLACEHOLDER

NS = "test-hotel-reservation"


def _geo():
    return FakeDeploymentKube("geo", NS, [{"name": "hotel-reserv-geo", "command": ["geo"]}])


def _profile():
    return FakeDeploymentKube("profile", NS, [{"name": "hotel-reserv-profile", "command": ["profile"]}])


@pytest.mark.parametrize(
    "make_kube, make_fault, faulty",
    [(_geo, assign_fault, assign_pinned), (_profile, wrong_bin_fault, wrong_bin_swapped)],
    ids=["assign", "wrong_bin"],
)
def test_inject_is_a_new_revision_under_recreate(make_kube, make_fault, faulty):
    kube = make_kube()
    fault = make_fault(kube, NS, kube.name)
    fault.inject()

    assert faulty(kube.deployment)
    assert kube.deployment["spec"]["strategy"]["type"] == "Recreate"
    assert "rollingUpdate" not in kube.deployment["spec"]["strategy"]
    assert kube.revision == 4 and fault.good_revision == "3"
    assert not faulty({"spec": {"template": kube.history[3]}})  # the previous revision is the good spec


@pytest.mark.parametrize(
    "make_kube, make_fault, family",
    [(_geo, assign_fault, "assign_to_non_existent_node_hotel_res"), (_profile, wrong_bin_fault, "wrong_bin_usage")],
    ids=["assign", "wrong_bin"],
)
def test_the_table_canonical_action_recovers_the_injected_fault(make_kube, make_fault, family):
    """What makes these families "agir": the row's own command, with the placeholder filled by the
    good revision, is exactly what fixes it."""
    kube = make_kube()
    fault = make_fault(kube, NS, kube.name)
    fault.inject()

    (canonical,) = FAULT_TABLE[family].expected_action
    assert canonical.replace(REVISION_PLACEHOLDER, fault.good_revision) == fault.undo_command()
    kube.exec_command(fault.undo_command())
    assert kube.deployment["spec"]["template"] == kube.history[3]


def test_recover_undoes_and_restores_the_original_strategy():
    kube = _geo()
    original = dict(kube.deployment["spec"]["strategy"])
    fault = assign_fault(kube, NS, "geo")
    fault.inject()

    result = fault.recover()

    assert result == {"undo_ran": True, "to_revision": "3", "strategy_restored": "RollingUpdate"}
    assert not assign_pinned(kube.deployment)
    assert kube.deployment["spec"]["strategy"] == original
    assert any("rollout undo deployment/geo" in c and "--to-revision=3" in c for c in kube.commands)


def test_recover_after_someone_already_undid_does_not_roll_back_to_the_bad_revision():
    kube = _profile()
    fault = wrong_bin_fault(kube, NS, "profile")
    fault.inject()
    kube.exec_command(fault.undo_command())  # the agent (or a human) fixed it first
    undos = sum("rollout undo" in c for c in kube.commands)

    result = fault.recover()

    assert result["undo_ran"] is False
    assert sum("rollout undo" in c for c in kube.commands) == undos
    assert not wrong_bin_swapped(kube.deployment)
    assert kube.deployment["spec"]["strategy"]["type"] == "RollingUpdate"


def test_inject_that_did_not_land_raises_instead_of_scoring_a_miss():
    """KubeCtl.exec_command returns stderr on failure; a silent no-op must not become an episode."""
    kube = FakeDeploymentKube("geo", NS, [{"name": "g", "command": ["geo"]}], inert=True)
    with pytest.raises(RevisionFaultError, match="did not land"):
        assign_fault(kube, NS, "geo").inject()


def test_half_landed_inject_restores_the_strategy_before_raising():
    """Recreate switched on but the template patch lost: the next family must not run under
    Recreate."""
    kube = FakeDeploymentKube("geo", NS, [{"name": "g", "command": ["geo"]}], template_patches_fail=True)
    original = dict(kube.deployment["spec"]["strategy"])
    with pytest.raises(RevisionFaultError, match="did not land"):
        assign_fault(kube, NS, "geo").inject()
    assert kube.deployment["spec"]["strategy"] == original
    assert not assign_pinned(kube.deployment)


def test_inject_refuses_a_deployment_that_already_carries_the_fault():
    kube = _geo()
    assign_fault(kube, NS, "geo").inject()
    with pytest.raises(RevisionFaultError, match="already carries the fault"):
        assign_fault(kube, NS, "geo").inject()


def test_recover_without_inject_refuses_instead_of_guessing_a_revision():
    kube = _geo()
    fault = assign_fault(kube, NS, "geo")
    kube.exec_command(
        "kubectl patch deployment geo -n test-hotel-reservation --type=merge "
        "-p '{\"spec\": {\"template\": {\"spec\": {\"nodeSelector\": {\"kubernetes.io/hostname\": \"extra-node\"}}}}}'"
    )
    with pytest.raises(RevisionFaultError, match="without a recorded good revision"):
        fault.recover()


def test_unreadable_deployment_raises():
    class Broken:
        def exec_command(self, command, input_data=None):
            return 'Error from server (NotFound): deployments.apps "geo" not found'

    with pytest.raises(RevisionFaultError, match="not readable"):
        assign_fault(Broken(), NS, "geo").inject()


def test_wrong_bin_without_a_profile_container_raises():
    kube = FakeDeploymentKube("profile", NS, [{"name": "x", "command": ["other"]}])
    with pytest.raises(RevisionFaultError, match="nothing to swap"):
        wrong_bin_fault(kube, NS, "profile").inject()


def test_missing_revision_annotation_raises():
    kube = _geo()
    del kube.deployment["metadata"]["annotations"][REVISION_ANNOTATION]
    with pytest.raises(RevisionFaultError, match="numeric"):
        assign_fault(kube, NS, "geo").inject()
