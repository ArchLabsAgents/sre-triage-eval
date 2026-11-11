"""Tests for harness/fault_table.py: the versioned fault -> cause/action/disposition table, the
closed cause taxonomy with distractors, the fine->coarse mapping, and the O1 oracle key."""

from __future__ import annotations

import pytest

from adapters.mitigation_probes import CANDIDATE_PROBES, PROBES
from harness.envelope import ActionClass, EpisodeEnvelopeState, classify
from harness.fault_table import (
    ADMITTED_COARSE_BY_FINE,
    APP_NAMESPACE,
    CAUSE_CLASSES,
    COARSE_BY_FINE,
    DISTRACTOR_CLASSES,
    FALSE_POSITIVE_CLASS,
    FAULT_TABLE,
    TRIAGE_CLASS_ENUM,
    UNCLASSIFIED_CLASS,
    UNKNOWN_CLASS,
    OracleKey,
    ambiguous_oracle_keys,
    cause_match,
    causes_by_oracle_key,
    oracle_key,
    oracle_service,
)
from scenarios.run_continuous_admission import CANDIDATE_MECHANISM_BY_FAMILY, MECHANISM_BY_FAMILY


# ---------------------------------------------------------------------------
# Single source with the admitted universe (promotion pattern of PROBES / MECHANISM_BY_FAMILY)
# ---------------------------------------------------------------------------


def test_table_covers_exactly_the_admitted_families():
    assert set(FAULT_TABLE) == set(MECHANISM_BY_FAMILY) == set(PROBES)


def test_candidates_never_enter_the_table():
    for family in set(CANDIDATE_MECHANISM_BY_FAMILY) | set(CANDIDATE_PROBES):
        assert family not in FAULT_TABLE
    assert "container_kill" not in FAULT_TABLE


def test_row_mechanism_is_the_admission_mechanism():
    for family, row in FAULT_TABLE.items():
        assert row.mechanism == MECHANISM_BY_FAMILY[family], family


def test_every_admitted_mechanism_has_a_fine_class():
    assert set(ADMITTED_COARSE_BY_FINE.values()) == set(MECHANISM_BY_FAMILY.values())


# ---------------------------------------------------------------------------
# Closed taxonomy: distractors, special classes, fine -> coarse
# ---------------------------------------------------------------------------


def test_enum_holds_admitted_distractors_and_special_classes():
    assert set(CAUSE_CLASSES) == (
        set(ADMITTED_COARSE_BY_FINE) | set(DISTRACTOR_CLASSES) | {UNKNOWN_CLASS, FALSE_POSITIVE_CLASS}
    )
    assert len(CAUSE_CLASSES) == len(set(CAUSE_CLASSES))


def test_unclassified_is_a_record_value_but_never_a_model_answer():
    assert UNCLASSIFIED_CLASS in TRIAGE_CLASS_ENUM
    assert UNCLASSIFIED_CLASS not in CAUSE_CLASSES


def test_distractors_are_disjoint_from_real_causes_and_never_accepted():
    accepted = {row.accepted_cause for row in FAULT_TABLE.values()}
    partial = {p for row in FAULT_TABLE.values() for p in row.partial_pairs}
    assert not set(DISTRACTOR_CLASSES) & (accepted | partial)
    assert not set(DISTRACTOR_CLASSES) & set(MECHANISM_BY_FAMILY.values())


def test_distractors_outnumber_admitted_fine_classes():
    # Elimination must not be a strategy: more wrong options than right ones in the enum.
    assert len(DISTRACTOR_CLASSES) > len(ADMITTED_COARSE_BY_FINE)


def test_each_distractor_is_its_own_coarse_class():
    for d in DISTRACTOR_CLASSES:
        assert COARSE_BY_FINE[d] == d


def test_fine_to_coarse_merges_mongo_variants_into_one_mechanism():
    assert COARSE_BY_FINE["mongo_role_revoked"] == COARSE_BY_FINE["mongo_user_dropped"] == "mongo_auth"


# ---------------------------------------------------------------------------
# Row contents
# ---------------------------------------------------------------------------


def test_partial_pairs_are_symmetric_and_stay_inside_the_mechanism():
    by_fine = {row.accepted_cause: row for row in FAULT_TABLE.values()}
    for row in FAULT_TABLE.values():
        for pair in row.partial_pairs:
            assert row.accepted_cause in by_fine[pair].partial_pairs
            assert COARSE_BY_FINE[pair] == row.mechanism


def test_disposition_follows_the_envelope_class():
    for family, row in FAULT_TABLE.items():
        assert row.action_class in {"C2", "C3"}, family
        assert row.expected_disposition == ("agir" if row.action_class == "C2" else "escalar"), family
        assert row.expected_urgency in {"P1", "P2", "P3", "P4"}, family


def test_expected_actions_classify_as_declared_by_the_real_envelope():
    """The table's C2/C3 label is not asserted by hand: every expected command goes through the
    envelope parser, with the read-before-mutate state satisfied so only the verb decides."""
    for family, row in FAULT_TABLE.items():
        target = f"deployment/{row.cause_service}"
        for command in row.expected_action:
            state = EpisodeEnvelopeState(c0_calls=1, confidence=0.9, targets_seen={target})
            decision = classify(command, state, target=target)
            assert decision.action_class == ActionClass(row.action_class), (family, command, decision)
            if row.action_class == "C2":
                assert decision.allowed, (family, command, decision)


def test_c2_rows_are_executable_and_c3_rows_carry_a_proposal_when_one_exists():
    for family, row in FAULT_TABLE.items():
        if row.action_class == "C2":
            assert row.expected_action, family
    # redeploy_without_PV is the only fault with no single recovering command.
    assert [f for f, r in FAULT_TABLE.items() if not r.expected_action] == ["redeploy_without_PV"]


def test_no_credentials_in_canonical_actions():
    for row in FAULT_TABLE.values():
        for command in row.expected_action:
            assert "pwd" not in command
            assert " -u " not in command
            assert "--password" not in command


def test_undetected_families_have_no_oracle_keys():
    assert FAULT_TABLE["wrong_bin_usage"].oracle_keys == ()
    assert FAULT_TABLE["redeploy_without_PV"].oracle_keys == ()


# ---------------------------------------------------------------------------
# cause_match (cause axis of the rubric)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "cause_class", "expected"),
    [
        ("revoke_auth_mongodb", "mongo_role_revoked", "exata"),
        ("revoke_auth_mongodb", "mongo_user_dropped", "parcial"),
        ("user_unregistered_mongodb_rate", "mongo_role_revoked", "parcial"),
        ("revoke_auth_mongodb", "mongo_unreachable", "errada"),
        ("misconfig_app_hotel_res", "image_pull_failure", "errada"),
        ("misconfig_app_hotel_res", "app_image_misconfig", "exata"),
        ("assign_to_non_existent_node_hotel_res", "node_not_ready", "errada"),
        ("misconfig_app_hotel_res", UNCLASSIFIED_CLASS, "errada"),
        ("misconfig_app_hotel_res", UNKNOWN_CLASS, "errada"),
    ],
)
def test_cause_match(family, cause_class, expected):
    assert cause_match(family, cause_class) == expected


def test_cause_match_rejects_class_outside_the_enum():
    with pytest.raises(ValueError):
        cause_match("misconfig_app_hotel_res", "wrong_image")


def test_cause_match_rejects_unknown_family():
    with pytest.raises(KeyError):
        cause_match("container_kill", "oom_killed")


# ---------------------------------------------------------------------------
# O1 oracle key, against label sets copied from a live detection run (2026-09-25)
# ---------------------------------------------------------------------------

LIVE_NOT_READY_MISCONFIG = {
    "alertname": "KubePodNotReady",
    "job": "kube-state-metrics",
    "namespace": APP_NAMESPACE,
    "pod": "geo-c5d8d5f75-xxhgt",
    "prometheus": "monitoring/sslm-kube-prometheus-prometheus",
    "severity": "warning",
}
LIVE_NOT_READY_MONGO = {**LIVE_NOT_READY_MISCONFIG, "pod": "geo-84fbc958c7-6jlqw"}
LIVE_CRASHLOOP_MONGO = {
    "alertname": "KubePodCrashLooping",
    "container": "hotel-reserv-geo",
    "endpoint": "http",
    "instance": "192.168.162.135:8080",
    "job": "kube-state-metrics",
    "namespace": APP_NAMESPACE,
    "pod": "geo-84fbc958c7-6jlqw",
    "reason": "CrashLoopBackOff",
    "service": "sslm-kube-prometheus-kube-state-metrics",
    "severity": "warning",
}


def test_o1_strips_the_replicaset_hash_that_leaks_the_mechanism():
    assert oracle_key(LIVE_NOT_READY_MISCONFIG) == oracle_key(LIVE_NOT_READY_MONGO)
    assert oracle_key(LIVE_NOT_READY_MISCONFIG) == OracleKey("KubePodNotReady", APP_NAMESPACE, "geo")


def test_o1_ignores_the_scrape_target_service_and_the_container_label():
    assert oracle_service(LIVE_CRASHLOOP_MONGO) == "geo"
    assert oracle_key(LIVE_CRASHLOOP_MONGO) == OracleKey("KubePodCrashLooping", APP_NAMESPACE, "geo")


def test_deployment_label_wins_when_the_rule_keeps_it():
    assert oracle_service({"deployment": "rate", "pod": "rate-7bdc978679-wm7dd"}) == "rate"


def test_statefulset_ordinal_is_stripped():
    assert oracle_service({"pod": "mongodb-geo-0"}) == "mongodb-geo"


def test_alert_without_a_workload_label_has_no_key():
    assert oracle_key({"alertname": "Watchdog", "namespace": APP_NAMESPACE}) is None
    assert oracle_key({"alertname": "KubePodNotReady", "pod": "geo-84fbc958c7-6jlqw"}) is None


def test_table_oracle_keys_match_what_the_normalizer_produces():
    observed = {oracle_key(LIVE_NOT_READY_MISCONFIG), oracle_key(LIVE_CRASHLOOP_MONGO)}
    table_keys = {k for row in FAULT_TABLE.values() for k in row.oracle_keys}
    assert observed <= table_keys


def test_ambiguous_subset_is_the_geo_collision_the_plan_describes():
    """Under O1, geo has 3 causes colliding on KubePodNotReady (misconfig, mongo_auth,
    assign_to_non_existent_node); rate is not ambiguous."""
    ambiguous = ambiguous_oracle_keys()
    not_ready_geo = OracleKey("KubePodNotReady", APP_NAMESPACE, "geo")
    assert ambiguous[not_ready_geo] == {
        "misconfig_app_hotel_res",
        "mongo_auth",
        "assign_to_non_existent_node",
    }
    assert ambiguous[OracleKey("KubePodCrashLooping", APP_NAMESPACE, "geo")] == {
        "misconfig_app_hotel_res",
        "mongo_auth",
    }
    assert all(key.service != "rate" for key in ambiguous)
    assert causes_by_oracle_key()[OracleKey("KubePodNotReady", APP_NAMESPACE, "rate")] == {"mongo_auth"}
