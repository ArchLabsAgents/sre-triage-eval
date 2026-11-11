"""Versioned fault -> cause/action/disposition table, closed cause taxonomy and the O1 oracle key.

This is the single versioned table the scorer reads: each row carries the accepted cause, partial
pairs, expected action, expected disposition and urgency for one fault family. It is ground truth
written BEFORE the round it scores, never an LLM judge. Pre-registration integrity: nothing in
FAULT_TABLE changes once a scored round starts; a change before that is a reviewed commit, never
an in-round edit.

Three layers, each with a single source:

  - Fine classes (CAUSE_CLASSES): the closed enum `TriageRecord.causa_provavel.classe` accepts.
    One fine class per admitted fault mechanism variant, plus DISTRACTOR_CLASSES (plausible
    Kubernetes/app causes no admitted family injects, so a model cannot score by elimination
    over a list that only holds right answers), plus UNKNOWN_CLASS (an honest unknown that
    escalates is a correct answer) and FALSE_POSITIVE_CLASS (the right answer to an injected
    false alarm).
  - Coarse classes (COARSE_BY_FINE): fine -> mechanism. For admitted classes the coarse value is
    exactly the mechanism in `scenarios/run_continuous_admission.MECHANISM_BY_FAMILY` (a test
    pins the two together, since harness/ must not import scenarios/). Each distractor is its own
    coarse class: it never collapses into a real mechanism.
  - Per-family rows (FAULT_TABLE): accepted fine class, partial pairs, expected action and its
    envelope class, expected disposition, urgency, and the oracle keys observed live.

Promotion follows the PROBES / MECHANISM_BY_FAMILY pattern: a family enters FAULT_TABLE only
together with its MECHANISM_BY_FAMILY entry. `container_kill` stays out (detected 0 of 2 times,
still a candidate); a test asserts it.

Oracle key (O1): the alertname+labels baseline of H3b reads `(alertname, namespace, service)`,
never the raw pod name, whose ReplicaSet hash leaks the fault mechanism. The service comes from
the `pod` label with the Kubernetes suffixes stripped, never from the alert's `service` label (on
kube-state-metrics alerts that is the scrape target, `sslm-kube-prometheus-kube-state-metrics`,
measured on KubePodCrashLooping) nor from `container` (`hotel-reserv-geo` on the same alert,
which would split one service across two keys against KubePodNotReady, whose only workload label
is `pod`). See `oracle_service`.

Field names that mirror TriageRecord keys (`causa_provavel`, `decisao`, ...) and row values such
as "agir"/"escalar" (act/escalate) are Portuguese because published records use them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

TABLE_VERSION = "e6r-2026-09-28"

# hotelReservation namespace as AIOpsLab's HotelReservation() deploys it (observed in every alert
# payload).
APP_NAMESPACE = "test-hotel-reservation"

# Whole-namespace faults use this in cause_service/symptom_service instead of one service.
NAMESPACE_SCOPE = "*"

# Urgency is derived by one rule, not judged per row: a fault that takes the whole namespace down
# (and risks data) is P1; a fault that breaks one user-facing hotelReservation service is P2,
# whether or not a frozen rule alerts on it (alerting is detectability, not impact). P3/P4 are
# unused by the admitted universe and stay available to the scorer for quiet-window records.
URGENCY_NAMESPACE = "P1"
URGENCY_ONE_SERVICE = "P2"

# Stand-in for a value only known during the episode (the previous deployment revision). The
# scorer canonicalizes a real command by replacing that value with this token before comparing.
REVISION_PLACEHOLDER = "REVISAO_ANTERIOR"

UNKNOWN_CLASS = "unknown"
FALSE_POSITIVE_CLASS = "falso_positivo"
# Harness-only value, never a model answer (AGENT_ACTION_SCHEMA.cause_class excludes it): the
# marker a record gets when the final action carries no valid `cause_class` (a replayed log from
# before the field existed), instead of a fabricated class. It is valid in the TriageRecord schema
# and scores as `errado` (wrong: not the table's class), never silently as a hit.
UNCLASSIFIED_CLASS = "unclassified"

# Fine class -> coarse (mechanism) for the admitted families.
ADMITTED_COARSE_BY_FINE: dict[str, str] = {
    "app_image_misconfig": "misconfig_app_hotel_res",
    "wrong_binary_command": "wrong_bin_usage",
    "pv_lost_on_redeploy": "redeploy_without_PV",
    "mongo_role_revoked": "mongo_auth",
    "mongo_user_dropped": "mongo_auth",
    "node_selector_unschedulable": "assign_to_non_existent_node",
}

# Distractors: plausible causes for the SAME symptoms (a hotelReservation pod NotReady or
# CrashLooping), none of them injected by an admitted family. Chosen as near misses on purpose:
# image_pull_failure sits next to app_image_misconfig, mongo_unreachable next to mongo_auth,
# node_not_ready next to node_selector_unschedulable. Each maps to itself as coarse class.
DISTRACTOR_CLASSES: tuple[str, ...] = (
    "image_pull_failure",
    "oom_killed",
    "liveness_probe_misconfig",
    "missing_config_or_secret",
    "mongo_unreachable",
    "dns_resolution_failure",
    "network_policy_block",
    "node_not_ready",
    "resource_quota_exceeded",
)

COARSE_BY_FINE: dict[str, str] = {
    **ADMITTED_COARSE_BY_FINE,
    **{d: d for d in DISTRACTOR_CLASSES},
    UNKNOWN_CLASS: UNKNOWN_CLASS,
    FALSE_POSITIVE_CLASS: FALSE_POSITIVE_CLASS,
}

# The closed enum a model answer may take (fine level).
CAUSE_CLASSES: tuple[str, ...] = tuple(COARSE_BY_FINE)

# What `TriageRecord.causa_provavel.classe` accepts: every model answer plus the harness marker.
TRIAGE_CLASS_ENUM: tuple[str, ...] = CAUSE_CLASSES + (UNCLASSIFIED_CLASS,)


@dataclass(frozen=True)
class OracleKey:
    alertname: str
    namespace: str
    service: str


@dataclass(frozen=True)
class FaultRow:
    family: str
    accepted_cause: str
    # Fine classes scored as `parcial` (partial): a neighbouring cause class the table accepts as
    # half right, e.g. role revoked vs user dropped on mongo.
    partial_pairs: tuple[str, ...]
    # Where the injector breaks things vs where the symptom (and the alert) shows up. Equal for
    # deployment faults; different for mongo auth (cause in mongodb-geo, alert on geo).
    cause_service: str
    symptom_service: str
    # Canonical expected action. For C3 rows this is the `acao_recomendada` the escalation package
    # should carry (the agent proposes, the human runs it); credentials are never
    # part of the canonical form. Empty only when no single command recovers the fault.
    expected_action: tuple[str, ...]
    action_class: str  # "C2" | "C3"
    expected_disposition: str  # "agir" | "escalar", same vocabulary as TriageRecord.decisao
    expected_urgency: str  # "P1".."P4"
    # (alertname, namespace, service) keys observed live in detection runs. Empty = never alerted
    # (a detectability finding; no rule is ever written to make it alert).
    oracle_keys: tuple[OracleKey, ...]
    source: str

    @property
    def mechanism(self) -> str:
        return COARSE_BY_FINE[self.accepted_cause]


def _mongo_rows(family_suffix: str, service: str, db: str) -> list[FaultRow]:
    keys = (
        OracleKey("KubePodNotReady", APP_NAMESPACE, service),
        OracleKey("KubePodCrashLooping", APP_NAMESPACE, service),
    )
    mongo = f"mongodb-{service}"
    exec_prefix = f"kubectl exec -n {APP_NAMESPACE} deploy/{mongo} -- mongo admin --eval"
    return [
        FaultRow(
            family=f"revoke_auth_mongodb{family_suffix}",
            accepted_cause="mongo_role_revoked",
            partial_pairs=("mongo_user_dropped",),
            cause_service=mongo,
            symptom_service=service,
            expected_action=(
                f"{exec_prefix} \"db.grantRolesToUser('admin', [{{role: 'readWrite', db: '{db}'}}])\"",
            ),
            action_class="C3",
            expected_disposition="escalar",
            expected_urgency=URGENCY_ONE_SERVICE,
            oracle_keys=keys,
            source=(
                "AIOpsLab revoke-admin-*-mongo.sh (revokeRolesFromUser readWrite) and its "
                "revoke-mitigate script; the envelope classes this exact exec as C3 even though it is "
                f"the correct fix; alert keys observed in two detection runs ({service})"
            ),
        ),
        FaultRow(
            family=f"user_unregistered_mongodb{family_suffix}",
            accepted_cause="mongo_user_dropped",
            partial_pairs=("mongo_role_revoked",),
            cause_service=mongo,
            symptom_service=service,
            expected_action=(
                f"{exec_prefix} \"db.createUser({{user: 'admin', roles: "
                "[{role: 'userAdminAnyDatabase', db: 'admin'}]})\"",
                f"{exec_prefix} \"db.grantRolesToUser('admin', [{{role: 'readWrite', db: '{db}'}}])\"",
            ),
            action_class="C3",
            expected_disposition="escalar",
            expected_urgency=URGENCY_ONE_SERVICE,
            oracle_keys=keys,
            source=(
                "AIOpsLab remove-admin-mongo.sh (dropUser admin) and remove-mitigate-admin-*-mongo.sh; "
                f"C3 (exec into the database); alert keys observed in two detection runs ({service})"
            ),
        ),
    ]


_ROWS: list[FaultRow] = [
    FaultRow(
        family="misconfig_app_hotel_res",
        accepted_cause="app_image_misconfig",
        partial_pairs=(),
        cause_service="geo",
        symptom_service="geo",
        # AIOpsLab swaps the image through update_deployment, which creates a new revision, so
        # the previous one is exactly the good image and `rollout undo` (C2 allowlist) recovers it.
        # The envelope requires --to-revision; its number is only known live (read into
        # rollback_state before the mutation), so the canonical form carries REVISION_PLACEHOLDER.
        expected_action=(
            f"kubectl rollout undo deployment/geo -n {APP_NAMESPACE} --to-revision={REVISION_PLACEHOLDER}",
        ),
        action_class="C2",
        expected_disposition="agir",
        expected_urgency=URGENCY_ONE_SERVICE,
        oracle_keys=(
            OracleKey("KubePodNotReady", APP_NAMESPACE, "geo"),  # first detection run, both passes
            OracleKey("KubePodCrashLooping", APP_NAMESPACE, "geo"),  # second detection run, both injections
        ),
        source="AIOpsLab inject_misconfig_app (image yinfangchen/geo:app3 via update_deployment); two detection runs",
    ),
    FaultRow(
        family="assign_to_non_existent_node_hotel_res",
        accepted_cause="node_selector_unschedulable",
        partial_pairs=(),
        cause_service="geo",
        symptom_service="geo",
        # The local wrapper injects the nodeSelector as a new revision (Recreate + template patch,
        # adapters/revision_faults.py), so the previous revision is the good spec and `rollout
        # undo` (C2) recovers it, same as misconfig. Upstream's delete + apply left no revision to
        # undo, which would make this C3.
        expected_action=(
            f"kubectl rollout undo deployment/geo -n {APP_NAMESPACE} --to-revision={REVISION_PLACEHOLDER}",
        ),
        action_class="C2",
        expected_disposition="agir",
        expected_urgency=URGENCY_ONE_SERVICE,
        # Measured with upstream's delete + apply injection; not yet re-measured with the
        # revision-preserving wrapper.
        oracle_keys=(OracleKey("KubePodNotReady", APP_NAMESPACE, "geo"),),
        source=(
            "nodeSelector extra-node as in AIOpsLab inject_assign_to_non_existent_node, injected by "
            "revision (adapters/revision_faults.py); alert key observed in two detection runs"
        ),
    ),
    FaultRow(
        family="wrong_bin_usage",
        accepted_cause="wrong_binary_command",
        partial_pairs=(),
        cause_service="profile",
        symptom_service="profile",
        # Same revision-preserving injection as assign, so `rollout undo` (C2).
        expected_action=(
            f"kubectl rollout undo deployment/profile -n {APP_NAMESPACE} --to-revision={REVISION_PLACEHOLDER}",
        ),
        action_class="C2",
        expected_disposition="agir",
        expected_urgency=URGENCY_ONE_SERVICE,
        oracle_keys=(),  # 0 of 2 detection runs alerted (profile runs the geo binary without crashing)
        source=(
            "command ['geo'] on profile as in AIOpsLab inject_wrong_bin_usage, injected by revision "
            "(adapters/revision_faults.py); no alert in two detection runs"
        ),
    ),
    FaultRow(
        family="redeploy_without_PV",
        accepted_cause="pv_lost_on_redeploy",
        partial_pairs=(),
        cause_service=NAMESPACE_SCOPE,
        symptom_service=NAMESPACE_SCOPE,
        # The injector deletes the whole namespace and redeploys without the PVs: recovery is an
        # app redeploy with storage, not a command inside any envelope class.
        expected_action=(),
        action_class="C3",
        expected_disposition="escalar",
        expected_urgency=URGENCY_NAMESPACE,
        oracle_keys=(),  # 0 of 2 detection runs alerted
        source="AIOpsLab inject_redeploy_without_pv (delete_namespace, redeploy without PV); no alert in two detection runs",
    ),
    *_mongo_rows("", "geo", "geo-db"),
    *_mongo_rows("_rate", "rate", "rate-db"),
]

FAULT_TABLE: dict[str, FaultRow] = {row.family: row for row in _ROWS}


# O1: Kubernetes-generated pod suffixes. Deployment pods: <name>-<rs hash>-<5 char id>;
# StatefulSet pods: <name>-<ordinal>. The ReplicaSet hash is what leaks the mechanism.
_DEPLOYMENT_POD_SUFFIX = re.compile(r"-[a-z0-9]{6,10}-[a-z0-9]{5}$")
_STATEFULSET_POD_SUFFIX = re.compile(r"-\d+$")


def oracle_service(labels: Mapping[str, str]) -> str | None:
    """O1 oracle: the service an alert is about, with no instance identity in it.

    `deployment` wins when the rule kept it (already a service-level name). Otherwise the `pod`
    label with its generated suffixes stripped. `service` and `container` are deliberately never
    read (module docstring: on kube-state-metrics alerts they name the scrape target and the
    container, not the app service). None when the alert carries no workload label at all."""
    deployment = labels.get("deployment")
    if deployment:
        return deployment
    pod = labels.get("pod")
    if not pod:
        return None
    stripped = _DEPLOYMENT_POD_SUFFIX.sub("", pod)
    if stripped == pod:
        stripped = _STATEFULSET_POD_SUFFIX.sub("", pod)
    return stripped


def oracle_key(labels: Mapping[str, str]) -> OracleKey | None:
    """The H3b baseline's lookup key for one alert, or None when a part cannot be read."""
    alertname = labels.get("alertname")
    namespace = labels.get("namespace")
    service = oracle_service(labels)
    if not (alertname and namespace and service):
        return None
    return OracleKey(alertname, namespace, service)


def causes_by_oracle_key(table: Mapping[str, FaultRow] = FAULT_TABLE) -> dict[OracleKey, frozenset[str]]:
    """Every observed key -> the coarse causes (mechanisms) that produced it."""
    out: dict[OracleKey, set[str]] = {}
    for row in table.values():
        for key in row.oracle_keys:
            out.setdefault(key, set()).add(row.mechanism)
    return {key: frozenset(causes) for key, causes in out.items()}


def ambiguous_oracle_keys(table: Mapping[str, FaultRow] = FAULT_TABLE) -> dict[OracleKey, frozenset[str]]:
    """H3b's ambiguous subset, by key: alert keys that map to more than one mechanism."""
    return {key: causes for key, causes in causes_by_oracle_key(table).items() if len(causes) > 1}


def cause_match(family: str, cause_class: str) -> str:
    """How a fine-class answer compares to the table row for `family`: "exata", "parcial" or
    "errada" (exact, partial, wrong). Only the cause axis of the scorer's `triage_understanding`;
    the blast_radius half and the `honesto` case (unknown + escalate, which depends on whether the
    family is held out) belong to the scorer, which composes this rather than duplicating the
    table.

    Raises KeyError for a family not in the table and ValueError for a class outside the enum:
    both are harness bugs, never a scorable answer."""
    row = FAULT_TABLE[family]
    if cause_class not in TRIAGE_CLASS_ENUM:
        raise ValueError(f"cause class {cause_class!r} is outside the closed taxonomy (TRIAGE_CLASS_ENUM)")
    if cause_class == row.accepted_cause:
        return "exata"
    if cause_class in row.partial_pairs:
        return "parcial"
    return "errada"
