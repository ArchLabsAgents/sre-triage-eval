"""Per-scenario root-cause probes.

Why they exist: a control round with no model and no injected fault showed that the `eval()` of
the 6 AIOpsLab mitigation-family problems used in the matrix only checks "pods healthy" (no
CrashLoopBackOff / wait_for_ready), and it passed 6/6 episodes on a cluster that never received a
fault. `success=True` cannot tell "the model mitigated" from "Kubernetes rescheduled on its own".
Each probe here checks the SPECIFIC symptom that AIOpsLab's own injector writes (read from
aiopslab/generators/fault/inject_app.py and inject_virtual.py), never generic pod health:

- misconfig_app_hotel_res: the injector swaps the image of container `hotel-reserv-geo` for
  `yinfangchen/geo:app3` (a buggy image). Fixed = buggy image gone from the deployment AND >= 1
  ready replica.
- redeploy_without_PV: the injector deletes the namespace and redeploys without waiting, leaving
  the old PV Released and the new PVC unbound (pod Pending). Fixed = every PVC in the namespace
  Bound AND no pod Pending.
- revoke_auth_mongodb: the injector runs `db.revokeRolesFromUser('admin', readWrite@<db>)` in the
  mongo pod. Fixed = `admin` exists and has readWrite on the target db.
- user_unregistered_mongodb: the injector runs `db.dropUser('admin')`. Fixed = the same criterion
  as revoke (the symptom converges: without the user or without the role, the service cannot
  write).
- wrong_bin_usage (spare, same app): the injector swaps `command: [profile]` for `[geo]`.
  Fixed = deployment `profile` with `profile` in its command. The native eval already checks
  this; the probe exists so both axes are measured with the same ruler.
- assign_to_non_existent_node_hotel_res (promoted after a live admission run detected 2/2): the
  injector pins deployment `geo` to a nonexistent node (nodeSelector/affinity). Fixed = no pin
  AND >= 1 pod Ready.
- revoke_auth_mongodb_rate / user_unregistered_mongodb_rate (promoted the same way): the same
  fault as revoke/user_unregistered above, on the `-mitigation-2` variant (db `rate-db`). Same
  probe, same fixed criterion.

What a probe is NOT: it does not replace the benchmark's `success` (still recorded as
`success`); it is a second, independent and stricter column (`root_cause_fixed`). Anything not
measured comes out as `None` with the reason in `probe_detail`/`probe_error`, never as False or
True.

Probe admission (scenarios/validate_probes.py, no model): a probe enters the matrix only if it
reads False with the fault injected and nobody touching the cluster, and True after AIOpsLab's
own `recover_fault()`. A probe that passes both is as blind as the eval it replaces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

BUGGY_GEO_IMAGE = "yinfangchen/geo:app3"

# Target db of each faulty mongo, the same map the injector scripts use
# (revoke-admin-geo-mongo.sh -> geo-db, revoke-admin-rate-mongo.sh -> rate-db).
MONGO_TARGET_DB = {"mongodb-geo": "geo-db", "mongodb-rate": "rate-db"}
MONGO_APP_USER = "admin"
# Fixture credential of AIOpsLab's hotelReservation chart (the same one the benchmark's
# injection/recovery scripts hard-code), not anyone's secret.
MONGO_ROOT_USER = "root"
MONGO_ROOT_PASSWORD = "root"


@dataclass(frozen=True)
class ProbeResult:
    # True = fault symptom absent; False = symptom present; None = could not measure.
    root_cause_fixed: bool | None
    detail: str


class KubeLike(Protocol):
    """The subset of aiopslab.service.kubectl.KubeCtl the probes use."""

    def list_pods(self, namespace: str) -> Any: ...

    def get_deployment(self, name: str, namespace: str) -> Any: ...

    def exec_command(self, command: str, input_data: Any = None) -> str: ...


Probe = Callable[[KubeLike, str, "str | None"], ProbeResult]


def probe_misconfig_app(kubectl: KubeLike, namespace: str, faulty_service: str | None) -> ProbeResult:
    service = faulty_service or "geo"
    deployment = kubectl.get_deployment(service, namespace)
    if deployment is None:
        return ProbeResult(False, f"deployment {service!r} missing in {namespace}")
    images = [c.image for c in deployment.spec.template.spec.containers]
    if BUGGY_GEO_IMAGE in images:
        return ProbeResult(False, f"buggy image still in deployment {service!r}: {images}")
    ready = getattr(deployment.status, "ready_replicas", None) or 0
    if ready < 1:
        return ProbeResult(False, f"image fixed ({images}) but 0 ready replicas")
    return ProbeResult(True, f"image {images}, {ready} ready replica(s)")


def probe_redeploy_without_pv(kubectl: KubeLike, namespace: str, faulty_service: str | None) -> ProbeResult:
    raw = kubectl.exec_command(f"kubectl get pvc -n {namespace} -o json")
    try:
        pvcs = json.loads(raw).get("items", [])
    except (ValueError, AttributeError):
        return ProbeResult(None, f"kubectl get pvc returned no JSON: {raw[:200]!r}")
    if not pvcs:
        return ProbeResult(None, f"no PVC in {namespace}, probe does not apply")
    unbound = [
        (pvc["metadata"]["name"], pvc.get("status", {}).get("phase"))
        for pvc in pvcs
        if pvc.get("status", {}).get("phase") != "Bound"
    ]
    if unbound:
        return ProbeResult(False, f"PVC not Bound: {unbound}")
    pending = [
        pod.metadata.name for pod in kubectl.list_pods(namespace).items if pod.status.phase == "Pending"
    ]
    if pending:
        return ProbeResult(False, f"every PVC Bound but pods Pending: {pending}")
    return ProbeResult(True, f"{len(pvcs)} PVC(s) Bound, no pod Pending")


def _mongo_pod_name(kubectl: KubeLike, namespace: str, faulty_service: str) -> str | None:
    names = [pod.metadata.name for pod in kubectl.list_pods(namespace).items if faulty_service in pod.metadata.name]
    return names[0] if names else None


def _mongo_get_user(kubectl: KubeLike, namespace: str, pod: str, user: str) -> tuple[Any, str]:
    """Return (user document | None when it does not exist, raw output). Raises ValueError
    when the output does not parse (the probe becomes None, never False)."""
    command = (
        f"kubectl exec {pod} -n {namespace} -- mongo admin "
        f"-u {MONGO_ROOT_USER} -p {MONGO_ROOT_PASSWORD} --authenticationDatabase admin --quiet "
        f"--eval \"JSON.stringify(db.getUser('{user}'))\""
    )
    raw = kubectl.exec_command(command) or ""
    for line in reversed(raw.strip().splitlines()):
        line = line.strip()
        if line == "null":
            return None, raw
        if line.startswith("{"):
            try:
                return json.loads(line), raw
            except ValueError:
                continue
    raise ValueError(f"mongo shell output did not parse: {raw[:200]!r}")


def probe_mongo_admin_readwrite(kubectl: KubeLike, namespace: str, faulty_service: str | None) -> ProbeResult:
    target_db = MONGO_TARGET_DB.get(faulty_service or "")
    if target_db is None:
        return ProbeResult(None, f"faulty_service {faulty_service!r} has no known target db")
    pod = _mongo_pod_name(kubectl, namespace, faulty_service)
    if pod is None:
        return ProbeResult(False, f"no {faulty_service!r} pod in {namespace}")
    try:
        user, _raw = _mongo_get_user(kubectl, namespace, pod, MONGO_APP_USER)
    except ValueError as exc:
        return ProbeResult(None, str(exc))
    if user is None:
        return ProbeResult(False, f"user {MONGO_APP_USER!r} does not exist in {pod}")
    roles = {(role.get("role"), role.get("db")) for role in user.get("roles", [])}
    if ("readWrite", target_db) not in roles:
        return ProbeResult(False, f"{MONGO_APP_USER!r} lacks readWrite on {target_db}: roles={sorted(roles)}")
    return ProbeResult(True, f"{MONGO_APP_USER!r} has readWrite on {target_db}")


def probe_wrong_bin_usage(kubectl: KubeLike, namespace: str, faulty_service: str | None) -> ProbeResult:
    service = faulty_service or "profile"
    deployment = kubectl.get_deployment(service, namespace)
    if deployment is None:
        return ProbeResult(False, f"deployment {service!r} missing in {namespace}")
    commands = [list(c.command or []) for c in deployment.spec.template.spec.containers]
    if not any("profile" in command for command in commands):
        return ProbeResult(False, f"deployment {service!r} still runs the wrong binary: {commands}")
    return ProbeResult(True, f"deployment {service!r} runs the profile binary: {commands}")


# --- candidate family kept out: built, NOT admitted ---------------------------------------------
#
# Why container_kill is not a PROBES key: a live admission run scored it 0/2 detected (a
# kill restarts the container too fast for the frozen symptom rules to catch, see
# probe_container_kill's docstring), so it stays a candidate here, reported as a detectability
# finding, until a future run demonstrates detection. The other three candidates (assign_to_non_existent_node_
# hotel_res, revoke_auth_mongodb_rate, user_unregistered_mongodb_rate) scored 2/2 and were
# promoted into PROBES below.
#
# Keys are candidate FAMILY names (the unit a schedule, a FamilyResult and a verdict are keyed by),
# not problem ids: adapters/aiopslab_problems.PROBLEM_ID_BY_FAMILY maps every family, admitted or
# candidate, to the AIOpsLab (or local wrapper) problem id the injector instantiates. Promotion
# moves a key from here to PROBES (and its mechanism to MECHANISM_BY_FAMILY); that map stays put.

# The node name AIOpsLab's own injector pins (aiopslab/generators/fault/inject_virtual.py,
# inject_assign_to_non_existent_node). Same stance as BUGGY_GEO_IMAGE: the probe reads the exact
# artifact the injector writes, never "some node selector looks odd".
NON_EXISTENT_NODE = "extra-node"
HOSTNAME_LABEL = "kubernetes.io/hostname"
# Label every hotelReservation workload carries (kompose output, also the PodChaos selector key in
# aiopslab/generators/fault/inject_symp.py). Selecting by label, never by name substring: "geo" is
# a substring of "mongodb-geo", and the mongo probe's name match would pick the wrong pod here.
SERVICE_LABEL = "io.kompose.service"
CHAOS_CONTAINER_KILL_ACTION = "container-kill"


def _service_pods(kubectl: KubeLike, namespace: str, service: str) -> list:
    """Pods of `service` that are not being deleted. A pod with a deletion timestamp belongs to a
    ReplicaSet the injector or recover already threw away (both of the faults below delete and
    recreate), so it says nothing about the current state."""
    pods = []
    for pod in kubectl.list_pods(namespace).items:
        metadata = pod.metadata
        labels = getattr(metadata, "labels", None) or {}
        if labels.get(SERVICE_LABEL) != service:
            continue
        if getattr(metadata, "deletion_timestamp", None):
            continue
        pods.append(pod)
    return pods


def _pod_ready(pod: Any) -> bool:
    conditions = getattr(pod.status, "conditions", None) or []
    return any(c.type == "Ready" and c.status == "True" for c in conditions)


def _pod_summary(pod: Any) -> str:
    statuses = getattr(pod.status, "container_statuses", None) or []
    restarts = sum(getattr(s, "restart_count", 0) or 0 for s in statuses)
    waiting = [
        s.state.waiting.reason
        for s in statuses
        if getattr(getattr(s, "state", None), "waiting", None) is not None
    ]
    ready = "Ready" if _pod_ready(pod) else "not Ready"
    extra = f", waiting={waiting}" if waiting else ""
    return f"{pod.metadata.name}({pod.status.phase}, {ready}, restarts={restarts}{extra})"


def _non_existent_node_pins(pod_spec: Any) -> list[str]:
    """Every place the pod template pins NON_EXISTENT_NODE: nodeSelector (what AIOpsLab writes) or
    a required node affinity term (same effect, so a hand-made variant of the fault is caught
    too). Preferred affinity does not make a pod unschedulable, so it is not a pin."""
    pins = []
    selector = getattr(pod_spec, "node_selector", None) or {}
    if selector.get(HOSTNAME_LABEL) == NON_EXISTENT_NODE:
        pins.append(f"nodeSelector {HOSTNAME_LABEL}={NON_EXISTENT_NODE}")
    affinity = getattr(pod_spec, "affinity", None)
    node_affinity = getattr(affinity, "node_affinity", None)
    required = getattr(node_affinity, "required_during_scheduling_ignored_during_execution", None)
    for term in getattr(required, "node_selector_terms", None) or []:
        for expr in getattr(term, "match_expressions", None) or []:
            if expr.key == HOSTNAME_LABEL and expr.operator == "In" and NON_EXISTENT_NODE in (expr.values or []):
                pins.append(f"nodeAffinity {HOSTNAME_LABEL} In [{NON_EXISTENT_NODE}]")
    return pins


def probe_assign_to_non_existent_node(
    kubectl: KubeLike, namespace: str, faulty_service: str | None
) -> ProbeResult:
    """assign_to_non_existent_node on HotelReservation's `geo` (promoted after 2/2 detection).

    The injector reads the live deployment, adds nodeSelector kubernetes.io/hostname=extra-node,
    deletes the deployment and re-applies it, so the new pod can never be scheduled (Pending,
    PodScheduled=False/Unschedulable). AIOpsLab's recover does the same delete/apply with the
    nodeSelector removed, which is the baseline: geo-deployment.yaml ships no nodeSelector.

    False = the deployment's pod template still pins the nonexistent node (every pod it makes is
    unschedulable, including one not created yet). True = no pin AND >= 1 geo pod Ready.
    None = the deployment is absent (live, KubeCtl.get_deployment raises ApiException 404: the
    injector and recover both sit between a delete and an apply for a moment), or no pin but no
    Ready geo pod yet. That middle state is deliberate, not a gap: recover
    deletes the deployment with background propagation and re-applies without waiting, so for
    a few seconds the old unschedulable pod and the new, still-starting one coexist. Scoring that
    False would fail admission on a race; None makes validate_probes' settle retry wait for it,
    and a geo that never comes up stays None forever, which is never admitted.
    """
    service = faulty_service or "geo"
    try:
        deployment = kubectl.get_deployment(service, namespace)
    except Exception as exc:  # noqa: BLE001 - only a 404 is handled, the rest re-raises
        if getattr(exc, "status", None) != 404:
            raise
        deployment = None
    if deployment is None:
        return ProbeResult(None, f"deployment {service!r} missing in {namespace} (404), not measured")
    pods = _service_pods(kubectl, namespace, service)
    summary = [_pod_summary(pod) for pod in pods]
    pins = _non_existent_node_pins(deployment.spec.template.spec)
    if pins:
        return ProbeResult(False, f"deployment {service!r} pinned to a nonexistent node ({pins}); pods: {summary}")
    if any(_pod_ready(pod) for pod in pods):
        return ProbeResult(True, f"deployment {service!r} has no nonexistent-node pin; pods: {summary}")
    return ProbeResult(None, f"deployment {service!r} has no pin, but no pod Ready yet: {summary}")


def _container_kill_injections(experiment: dict) -> int:
    """How many kills Chaos Mesh recorded for this experiment. Chaos Mesh 2.x keeps one record per
    selected container under status.experiment.containerRecords, with injectedCount incremented
    on every injection; it stays after the experiment's duration ends (container-kill has no
    recovery action), so it is evidence the kill happened, not only that it was requested."""
    records = (experiment.get("status") or {}).get("experiment", {}).get("containerRecords") or []
    total = 0
    for record in records:
        count = record.get("injectedCount") or 0
        if not count and record.get("phase") == "Injected":
            count = 1
        total += count
    return total


def container_kill_experiments(experiments: list[dict], service: str) -> list[dict]:
    """PodChaos items (from `kubectl get podchaos -o json`) that are a container-kill on `service`.
    Shared with the injector's post-inject check, so "the fault exists" means the same thing to
    both."""
    return [
        item
        for item in experiments
        if (item.get("spec") or {}).get("action") == CHAOS_CONTAINER_KILL_ACTION
        and ((item.get("spec") or {}).get("selector") or {}).get("labelSelectors", {}).get(SERVICE_LABEL) == service
    ]


def probe_container_kill(kubectl: KubeLike, namespace: str, faulty_service: str | None) -> ProbeResult:
    """container_kill on HotelReservation's `geo` (candidate; AIOpsLab id
    container_kill-detection).

    The injector applies a Chaos Mesh PodChaos (action container-kill, mode one, duration 200s,
    selector io.kompose.service=geo, container hotel-reserv-geo); AIOpsLab's recover deletes it.
    The kill itself is one-shot: kubelet restarts the container immediately, so the lasting
    artifact of the fault is the experiment object plus its injection record.

    False = a container-kill PodChaos targeting the service exists AND Chaos Mesh recorded >= 1
    injection, or (no experiment) a service container sits in CrashLoopBackOff. True = no such
    experiment AND >= 1 service pod Ready. None = experiment present with no injection record
    (chaos-daemon has not killed yet, or cannot: wrong runtime socket), the PodChaos CRD is
    absent (Chaos Mesh not installed), or no Ready pod yet. Requiring the injection record is
    what keeps a no-op fault from being admitted: an experiment that never killed anything would
    otherwise read False then True and pass admission while measuring nothing.

    Honest limit on detection (not on this probe): one kill restarts the container once, which
    usually does not produce CrashLoopBackOff at all (the kubelet back-off only starts on repeated
    failures). The frozen KubePodCrashLooping rule looks at max_over_time of the waiting reason
    over 5m, so whether it fires depends on a restart being slow enough to be scraped in that
    state, and KubePodNotReady needs the pod not Ready for its `for` window. Detection of this
    family is timing-dependent by construction; the live detection run is what answers it.
    """
    service = faulty_service or "geo"
    raw = kubectl.exec_command(f"kubectl get podchaos -n {namespace} -o json")
    try:
        experiments = json.loads(raw).get("items", [])
    except (ValueError, AttributeError):
        return ProbeResult(None, f"kubectl get podchaos returned no JSON (is Chaos Mesh installed?): {str(raw)[:200]!r}")
    kills = container_kill_experiments(experiments, service)
    pods = _service_pods(kubectl, namespace, service)
    summary = [_pod_summary(pod) for pod in pods]
    if kills:
        names = [(item.get("metadata") or {}).get("name") for item in kills]
        injected = sum(_container_kill_injections(item) for item in kills)
        if injected:
            return ProbeResult(False, f"active PodChaos container-kill {names}, {injected} injection(s); pods: {summary}")
        return ProbeResult(None, f"PodChaos container-kill {names} has no injection record yet; pods: {summary}")
    crashlooping = [
        pod.metadata.name
        for pod in pods
        for status in (getattr(pod.status, "container_statuses", None) or [])
        if getattr(getattr(getattr(status, "state", None), "waiting", None), "reason", None) == "CrashLoopBackOff"
    ]
    if crashlooping:
        return ProbeResult(False, f"no PodChaos, but {service!r} in CrashLoopBackOff: {crashlooping}")
    if any(_pod_ready(pod) for pod in pods):
        return ProbeResult(True, f"no PodChaos container-kill in {namespace}; pods: {summary}")
    return ProbeResult(None, f"no PodChaos container-kill, but no {service!r} pod Ready yet: {summary}")


# Key = problem id without the "-mitigation-N" suffix (AIOpsLab's registry only varies
# faulty_service between -1 and -2, and the probe reads faulty_service from the problem instance).
#
# Promotion: assign_to_non_existent_node_hotel_res and the
# two mongo rate variants (revoke_auth_mongodb_rate, user_unregistered_mongodb_rate) moved here
# from CANDIDATE_PROBES above after a live admission run showed 2/2 detection on each
# (the earlier admission schedule had no evidence either way for these families; the
# pre-registered admission schedule covers all eight). container_kill stayed a candidate:
# 0/2 detected, reported as a detectability finding.
PROBES: dict[str, Probe] = {
    "misconfig_app_hotel_res": probe_misconfig_app,
    "redeploy_without_PV": probe_redeploy_without_pv,
    "revoke_auth_mongodb": probe_mongo_admin_readwrite,
    "user_unregistered_mongodb": probe_mongo_admin_readwrite,
    "wrong_bin_usage": probe_wrong_bin_usage,
    "assign_to_non_existent_node_hotel_res": probe_assign_to_non_existent_node,
    # Same fault as revoke_auth_mongodb / user_unregistered_mongodb, AIOpsLab's -mitigation-2
    # variant (faulty_service mongodb-rate, db rate-db via MONGO_TARGET_DB). Same probe: it reads
    # faulty_service from the problem instance, so the target db follows the variant.
    "revoke_auth_mongodb_rate": probe_mongo_admin_readwrite,
    "user_unregistered_mongodb_rate": probe_mongo_admin_readwrite,
}


def probe_for(problem_id: str) -> Probe | None:
    return PROBES.get(problem_id.rsplit("-mitigation-", 1)[0])


CANDIDATE_PROBES: dict[str, Probe] = {
    "container_kill": probe_container_kill,
}


def candidate_probe_for(family: str) -> Probe | None:
    return CANDIDATE_PROBES.get(family)


def _run_safely(probe: Probe | None, label: str, kubectl: KubeLike, namespace: str, faulty_service: str | None) -> dict:
    if probe is None:
        return {"root_cause_fixed": None, "probe_detail": f"no probe registered for {label!r}"}
    try:
        result = probe(kubectl, namespace, faulty_service)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return {"root_cause_fixed": None, "probe_error": f"{type(exc).__name__}: {exc}"}
    return {"root_cause_fixed": result.root_cause_fixed, "probe_detail": result.detail}


def run_probe(problem_id: str, kubectl: KubeLike, namespace: str, faulty_service: str | None) -> dict:
    """Run the scenario's probe and return the keys the RunRecord carries. Never raises: an
    exception becomes `probe_error` with a None verdict (not measured), in the same spirit as the
    adapter's `eval_error`."""
    return _run_safely(probe_for(problem_id), problem_id, kubectl, namespace, faulty_service)


def run_candidate_probe(family: str, kubectl: KubeLike, namespace: str, faulty_service: str | None) -> dict:
    """run_probe for a candidate family (CANDIDATE_PROBES), same never-raises contract. Keyed
    by family, not problem id: container_kill-detection has no "-mitigation-N" suffix to strip,
    and the mongo rate variants share their problem-id prefix with the admitted geo families."""
    return _run_safely(candidate_probe_for(family), family, kubectl, namespace, faulty_service)
