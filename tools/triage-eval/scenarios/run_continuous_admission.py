#!/usr/bin/env python3
"""Admission run ("R1" in identifiers): admits the continuous scenario instruments, no model.

The same driver also runs the model rounds (--post-action-probe) and short dry runs (--dry-run).

This is intentionally a real-time driver.  It replays a committed FaultSchedule against
AIOpsLab, waits for Alertmanager symptom notifications, verifies one six-hour quiet window,
and compares Alertmanager's webhook-notification counter with WebhookReceiver's counter.

Lifecycle: prepare once (AIOpsLab bootstrap, before the round clock) -> inject/recover per
schedule entry, never tearing the app down between them -> cleanup once at the end, also on
abort. Every phase is journaled to an fsynced JSONL and the result JSON is rewritten after
each entry. `--smoke` runs the cheap pre-round check (run_smoke) on the same lifecycle.  It
does not create cloud resources; run_continuous_admission_on_instance.sh is the remote wrapper
used only after the lab already exists.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Protocol

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from adapters.aiopslab_problems import LOCAL_PROBLEM_IDS, PROBLEM_ID_BY_FAMILY  # noqa: E402
from adapters.mitigation_probes import (  # noqa: E402
    PROBES,
    container_kill_experiments,
    run_candidate_probe,
    run_probe,
)
from scenarios.fault_schedule import EntryKind, FaultSchedule, ScheduleEntry, generate_schedule  # noqa: E402

UTC = timezone.utc

R1_QUIET_WINDOW_MINUTES = 6 * 60
MAX_WEBHOOK_LOSS = 0.01
DEFAULT_ALERT_TIMEOUT_SECONDS = 10 * 60
DEFAULT_POLL_SECONDS = 15.0
# The journal is also the scorer bridge's ground truth (the scheduler log of timestamped
# inject/recover events, harness/episode_bridge.py). Bumped when an event gains or
# loses a field the joiner reads, so a joiner never guesses which shape an old journal has.
JOURNAL_VERSION = 3  # v3: probe_post_action, and alert_observed covers the whole entry with post_action_probe
# The alertname send_false_alarm posts; journaled so a false-alarm episode can be matched to the
# record it produced without re-reading this module.
FALSE_ALARM_ALERTNAME = "SslmSyntheticControl"

# A single injection is weak evidence for a detectability verdict, so the admission run's
# committed schedule must give every admitted family at least
# this many independent fault episodes.
MIN_INJECTIONS_PER_FAMILY = 2

# revoke_auth_mongodb and user_unregistered_mongodb are the same underlying mechanism
# (Mongo auth), so they count as one family for the ">= 2 distinct mechanisms" gate. The mongo
# rate variants are the same mechanism on another db (MONGO_TARGET_DB's rate-db), so they never
# add a mechanism of their own.
#
# Promotion (2026-09-25): assign_to_non_existent_node_hotel_res, revoke_auth_mongodb_rate and
# user_unregistered_mongodb_rate moved here from CANDIDATE_MECHANISM_BY_FAMILY below after a live
# admission run showed 2/2 detection on each. container_kill stayed a candidate (0/2).
MECHANISM_BY_FAMILY: dict[str, str] = {
    "revoke_auth_mongodb": "mongo_auth",
    "user_unregistered_mongodb": "mongo_auth",
    "wrong_bin_usage": "wrong_bin_usage",
    "redeploy_without_PV": "redeploy_without_PV",
    "misconfig_app_hotel_res": "misconfig_app_hotel_res",
    "assign_to_non_existent_node_hotel_res": "assign_to_non_existent_node",
    "revoke_auth_mongodb_rate": "mongo_auth",
    "user_unregistered_mongodb_rate": "mongo_auth",
}

# Candidate kept out of MECHANISM_BY_FAMILY for the same reason it is kept out of PROBES:
# nothing admitted may change until a live run admits it. On admission a family moves from here
# to MECHANISM_BY_FAMILY, together with its PROBES entry.
CANDIDATE_MECHANISM_BY_FAMILY: dict[str, str] = {
    "container_kill": "container_kill",
}

# The admission run passes when >= 3 families are detected AND those detected families span >= 2 distinct
# mechanisms (a single mechanism, however many families it touches, is not enough).
MIN_FAMILIES_DETECTED = 3
MIN_DISTINCT_MECHANISMS = 2

# Settle gate: before every fault injection and
# before the quiet window, the app namespace must be settled -- no alert firing OR pending in
# Prometheus and every pod Ready -- continuously for PRE_ENTRY_STABLE_MINUTES up to the entry's
# start. Watching starts PRE_ENTRY_SETTLE_LEAD_MINUTES early (inside the schedule's 30-minute gap
# between entries); if the app is still dirty at the planned start, the entry waits up to
# PRE_ENTRY_MAX_OVERRUN_MINUTES more. A fault entry that never settles is not injected and
# counts as not measured, never as detected; a quiet window that never settles is not admissible.
# Why: the first scoped detectability pass scored misconfig_app_hotel_res "detected" off a geo pod left
# crashlooping by an earlier run (alert pending before the injection, invisible to a
# firing-only baseline), and user_unregistered_mongodb was injected while the previous family's
# symptom was still firing. A diff of firing fingerprints cannot tell those apart from a real
# symptom: every replaced pod gets a new fingerprint.
PRE_ENTRY_SETTLE_LEAD_MINUTES = 10
PRE_ENTRY_STABLE_MINUTES = 1
PRE_ENTRY_MAX_OVERRUN_MINUTES = 10
# Measured in the admission run: the quiet window's start was read AFTER _sleep_until() returned,
# so it always landed a few hundred ms past the planned start. With the committed schedule's window
# at exactly R1_QUIET_WINDOW_MINUTES, effective_minutes came out 359.997 and a window with zero
# unexpected fingerprints was scored truncated. Lateness up to this bound is sleep/poll jitter and
# is measured from the planned start; anything later is a real overrun and still eats the window.
QUIET_START_JITTER_SECONDS = 5.0


class R1Error(RuntimeError):
    """The round ("R1" = the admission run, the name this driver started with) could not produce
    an admissible observation."""


class Clock(Protocol):
    def time(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class AlertGateway(Protocol):
    def firing_alerts(self) -> dict[str, dict]: ...

    def firing_fingerprints(self) -> set[str]: ...

    def webhook_notifications(self, window_minutes: int) -> float: ...

    def send_false_alarm(self, duration_minutes: int) -> None: ...

    def unsettled_signals(self) -> list[str]:
        """What keeps the app from counting as settled right now (settle gate): alerts firing or pending
        and pods not Ready, in the scoped namespace. Empty means settled."""
        ...


class ReceiverGateway(Protocol):
    def received_total(self) -> int: ...


@dataclass
class FamilyResult:
    family: str
    start_minute: int
    symptom_alert_admitted: bool
    # One entry per newly-firing fingerprint at alert time: {"fingerprint", "alertname",
    # "labels"}. Earlier the result only kept the fingerprint, so nothing on disk could say
    # which rule fired or audit the claim that the rules alert on symptoms, never on mechanisms.
    observed_alerts: list[dict] = field(default_factory=list)
    error: str | None = None
    planned_at: float | None = None
    actual_at: float | None = None
    injected: bool = False
    alert_at: float | None = None
    recovered: bool = False
    recovery: dict = field(default_factory=dict)
    webhook_received_so_far: int | None = None
    # Settle gate: False when the app never settled before this entry, so it was not injected at all.
    # Such an episode is excluded from the family verdict and keeps the round from passing.
    measured: bool = True
    settle: dict = field(default_factory=dict)
    # Only with post_action_probe: the root-cause probe read at the planned end of the
    # entry, after the agent had the whole window to act and before recover() undid the fault.
    probe: dict | None = None


@dataclass
class QuietResult:
    start_minute: int
    duration_minutes: int
    alert_free: bool
    unexpected_fingerprints: list[str] = field(default_factory=list)
    actual_start: float | None = None
    actual_end: float | None = None
    effective_minutes: float | None = None
    truncated: bool = False
    settle: dict = field(default_factory=dict)


@dataclass
class WebhookAccounting:
    """Denominator is Alertmanager's own webhook-notification counter over the whole round."""

    query_window_minutes: int
    alertmanager_sent: float
    receiver_start: int
    receiver_end: int
    received: int
    denominator: float
    loss: float | None


@dataclass
class R1Result:
    schedule: dict
    families: list[FamilyResult]
    quiet_window: QuietResult | None
    webhook_sent: float | None
    webhook_received: int | None
    webhook_loss: float | None
    passed: bool
    started_at: float | None = None
    aborted: str | None = None
    webhook: WebhookAccounting | None = None
    cleanup: dict = field(default_factory=dict)
    # Per family, across every injection episode of that family in this round: "detected" (every
    # episode alerted), "not_detected" (none did) or "intermittent" (mixed). This
    # replaces the old 5/5-or-abort rule: a family that never alerts is a detectability finding
    # to report, not a reason to throw the whole round away.
    family_verdicts: dict[str, str] = field(default_factory=dict)
    # A pipeline check on a short schedule, never a round. passed is always False.
    dry_run: bool = False

    def to_json(self) -> dict:
        return asdict(self)


def validate_schedule_for_r1(schedule: FaultSchedule) -> None:
    """Reject an undeclared or insufficient schedule before a live fault is injected."""
    unknown = {entry.fault_family for entry in schedule.fault_episodes} - set(PROBES)
    if unknown:
        raise R1Error(f"schedule has fault family without an admitted probe: {sorted(unknown)}")
    counts = Counter(entry.fault_family for entry in schedule.fault_episodes)
    missing = set(PROBES) - set(counts)
    if missing:
        raise R1Error(f"the admission run needs all {len(PROBES)} fault families, missing: {sorted(missing)}")
    under_quota = {family: count for family, count in counts.items() if count < MIN_INJECTIONS_PER_FAMILY}
    if under_quota:
        raise R1Error(
            f"the admission run needs >= {MIN_INJECTIONS_PER_FAMILY} injections per family for a tri-state "
            f"verdict, under quota: {under_quota}"
        )
    if not any(entry.duration_minutes >= R1_QUIET_WINDOW_MINUTES for entry in schedule.quiet_windows):
        raise R1Error("the admission run needs one contiguous quiet window of at least 360 minutes")


def validate_quiet_window_placement(schedule: FaultSchedule) -> None:
    """Lesson from round 3 (two seeds lost their quiet window to an artifact, not to the model): the scored
    6h quiet window must not open at minute 0 (the smoke's alerts and agent load are still
    settling) and must not touch a synthetic false alarm (its alert lingers into the window).

    Separate from validate_schedule_for_r1 on purpose: the round-2 and round-3 schedules are
    committed and stay as they are, so this only gates seeds generated with generate_r1_schedule(clean_quiet_window=True).
    """
    false_alarms = schedule.false_alarms
    for window in schedule.quiet_windows:
        if window.duration_minutes < R1_QUIET_WINDOW_MINUTES or window.start_minute == 0:
            continue
        if any(
            alarm.end_minute == window.start_minute or alarm.start_minute == window.end_minute
            for alarm in false_alarms
        ):
            continue
        return
    raise R1Error(
        "the round needs a 6h quiet window that does not start at minute 0 and is not adjacent to a false alarm"
    )


def validate_schedule_for_dry_run(schedule: FaultSchedule) -> None:
    """End-to-end pipeline check on a short schedule. Drops only the admission run's minimums (every
    family, per-family quota, 360-minute quiet window); a family without an admitted probe is
    still refused, and a dry run never counts as a round (run_r1 forces passed=False)."""
    unknown = {entry.fault_family for entry in schedule.fault_episodes} - set(PROBES)
    if unknown:
        raise R1Error(f"schedule has fault family without an admitted probe: {sorted(unknown)}")
    if not schedule.fault_episodes:
        raise R1Error("a dry run needs at least one fault episode to exercise the pipeline")


def quiet_rerun_schedule() -> FaultSchedule:
    """Re-measure only the admission run's quiet-window criterion, the one the admission run could
    not judge because of the start-jitter bug (see QUIET_START_JITTER_SECONDS). The window starts
    after the settle lead so the settle gate applies exactly as it does inside a full round."""
    start = PRE_ENTRY_SETTLE_LEAD_MINUTES
    return FaultSchedule(
        seed=0,
        window_hours=(start + R1_QUIET_WINDOW_MINUTES) / 60,
        entries=(ScheduleEntry(EntryKind.QUIET, start, R1_QUIET_WINDOW_MINUTES),),
    )


def validate_schedule_for_quiet_rerun(schedule: FaultSchedule) -> None:
    if schedule.fault_episodes or schedule.false_alarms:
        raise R1Error("a quiet-only rerun injects nothing: faults and false alarms are not allowed")
    if [entry.duration_minutes >= R1_QUIET_WINDOW_MINUTES for entry in schedule.quiet_windows] != [True]:
        raise R1Error("a quiet-only rerun needs exactly one quiet window of at least 360 minutes")


def compute_family_verdicts(family_results: list[FamilyResult]) -> dict[str, str]:
    """Unit of analysis is the episode (one schedule entry), never the fingerprint: a family with
    3 injections and 2 alerts is "intermittent" regardless of how many fingerprints each alert
    produced (fingerprints of one injection are pseudo-replicates, not independent evidence)."""
    by_family: dict[str, list[bool]] = {}
    for item in family_results:
        admissions = by_family.setdefault(item.family, [])
        # Settle gate: an episode that never got a settled app was not injected; it is no evidence either
        # way, so it never counts toward "detected" or "not_detected".
        if item.measured:
            admissions.append(item.symptom_alert_admitted)
    verdicts: dict[str, str] = {}
    for family, admissions in by_family.items():
        if not admissions:
            verdicts[family] = "not_measured"
        elif all(admissions):
            verdicts[family] = "detected"
        elif not any(admissions):
            verdicts[family] = "not_detected"
        else:
            verdicts[family] = "intermittent"
    return verdicts


def r1_admission_passed(family_verdicts: dict[str, str]) -> bool:
    """>= 3 families detected AND those detected families span >= 2 distinct
    mechanisms (MECHANISM_BY_FAMILY), replacing the pre-registered 5/5 that reproves by design.

    Only the admitted, pre-registered families (MECHANISM_BY_FAMILY's keys) count, toward both
    totals. A verdict for any other family (a candidate measured in a detection run, or a
    typo) is data for its own admission decision, never evidence that the admission run passed: counting it would
    let a candidates-only run clear a gate the pre-registration never declared it could."""
    detected = {
        family
        for family, verdict in family_verdicts.items()
        if verdict == "detected" and family in MECHANISM_BY_FAMILY
    }
    mechanisms = {MECHANISM_BY_FAMILY[family] for family in detected}
    return len(detected) >= MIN_FAMILIES_DETECTED and len(mechanisms) >= MIN_DISTINCT_MECHANISMS


def generate_r1_schedule(
    seed: int, *, max_attempts: int = 10_000, clean_quiet_window: bool = False
) -> FaultSchedule:
    """Generate a schedule that satisfies the admission run's stricter, predeclared sample requirements.

    `fault_schedule.generate_schedule` is the general continuous scheduler and intentionally
    allows 2-6 hour quiet windows plus repeated fault families. The admission run needs all admitted families
    with >= MIN_INJECTIONS_PER_FAMILY episodes each and one six-hour negative-control window, so
    it chooses the first deterministic candidate from a documented seed sequence. The resulting
    JSON, including its actual seed, must be committed before it is replayed against a lab.

    `clean_quiet_window` (opt-in, new seeds only) also requires validate_quiet_window_placement.
    The default keeps the committed admission, round-2 and round-3 seeds regenerating byte for byte.

    A candidate seed can fail two different ways, both skipped to try the next seed: R1Error
    (validate_schedule_for_r1 rejects it, e.g. missing a family) or ValueError (generate_schedule
    itself: the quiet-window carving loop's random overshoot left too little room for the fault
    episodes on this particular seed -- generate_schedule's own docstring calls this out as a
    late, seed-dependent failure, never evidence the family set does not fit the window at all).
    Found empirically after the 2026-09-25 promotion (8 admitted families): about 60% of
    seeds are infeasible under window_hours=24 this way, so a search that stopped on the first
    ValueError would practically never find a schedule.
    """
    for candidate_seed in range(seed, seed + max_attempts):
        try:
            candidate = generate_schedule(candidate_seed, min_injections_per_family=MIN_INJECTIONS_PER_FAMILY)
            validate_schedule_for_r1(candidate)
            if clean_quiet_window:
                validate_quiet_window_placement(candidate)
        except (ValueError, R1Error):
            continue
        return candidate
    raise R1Error(f"no admissible schedule in {max_attempts} deterministic seeds from {seed}")


def _request_json(url: str, *, method: str = "GET", payload: object | None = None) -> object:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - URLs are explicit CLI args
            body = response.read()
            return json.loads(body) if body else None
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise R1Error(f"HTTP observation failed for {url}: {exc}") from exc


@dataclass
class HttpAlertGateway:
    alertmanager_url: str
    prometheus_url: str
    # Scopes firing_alerts()/firing_fingerprints() to this namespace once set. Found in the first
    # local detectability pass (2026-09-24): an unscoped diff against Alertmanager's full
    # alert list picked up NodeClockNotSynchronising (namespace=monitoring) and TargetDown
    # (namespace=kube-system) -- cluster-infra alerts with no relation to the injected fault -- as
    # a false "detected" verdict for misconfig_app_hotel_res. run_r1()/run_smoke() set this from
    # the injector's own prepare() result as soon as the app namespace is known, before the first
    # baseline is captured. None (the default) means unscoped, for callers inspecting the cluster
    # outside a round.
    namespace: str | None = None

    def firing_alerts(self) -> dict[str, dict]:
        payload = _request_json(f"{self.alertmanager_url.rstrip('/')}/api/v2/alerts?active=true&silenced=false")
        if not isinstance(payload, list):
            raise R1Error("Alertmanager /api/v2/alerts did not return a list")
        alerts: dict[str, dict] = {}
        for alert in payload:
            if not isinstance(alert, dict):
                continue
            if alert.get("status", {}).get("state") != "active":
                continue
            labels = dict(alert.get("labels") or {})
            if labels.get("alertname") == "Watchdog":
                continue
            if self.namespace is not None and labels.get("namespace") != self.namespace:
                continue
            fingerprint = alert.get("fingerprint")
            if not fingerprint:
                continue
            alerts[str(fingerprint)] = {"alertname": labels.get("alertname"), "labels": labels}
        return alerts

    def firing_fingerprints(self) -> set[str]:
        return set(self.firing_alerts().keys())

    def _prometheus_vector(self, query: str) -> list[dict]:
        encoded = urllib.parse.urlencode({"query": query})
        payload = _request_json(f"{self.prometheus_url.rstrip('/')}/api/v1/query?{encoded}")
        try:
            result = payload["data"]["result"]
            if not isinstance(result, list):
                raise TypeError("result is not a list")
            return result
        except (KeyError, TypeError) as exc:
            raise R1Error(f"Prometheus returned an invalid response for {query!r}") from exc

    def unsettled_signals(self) -> list[str]:
        # Prometheus' ALERTS series, not Alertmanager: Alertmanager never sees a pending alert,
        # and pending-before-injection is exactly what the firing-only baseline missed (settle gate).
        selector = f'namespace="{self.namespace}"' if self.namespace is not None else ""
        signals: list[str] = []
        for series in self._prometheus_vector(f"ALERTS{{{selector}}}"):
            metric = series.get("metric") or {}
            if metric.get("alertname") == "Watchdog":
                continue
            signals.append(f"alert {metric.get('alertname')} {metric.get('alertstate')} pod={metric.get('pod')}")
        pod_selector = f"{selector}," if selector else ""
        not_ready = (
            f'kube_pod_status_ready{{{pod_selector}condition="false"}} == 1 '
            f'unless on(namespace, pod) kube_pod_status_phase{{{pod_selector}phase="Succeeded"}} == 1'
        )
        for series in self._prometheus_vector(not_ready):
            signals.append(f"pod_not_ready {(series.get('metric') or {}).get('pod')}")
        return sorted(signals)

    def webhook_notifications(self, window_minutes: int) -> float:
        query = (
            'sum(increase(alertmanager_notifications_total{integration="webhook"}'
            f"[{window_minutes}m]))"
        )
        result = self._prometheus_vector(query)
        try:
            return float(result[0]["value"][1]) if result else 0.0
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise R1Error("Prometheus returned an invalid webhook notification query") from exc

    def send_false_alarm(self, duration_minutes: int) -> None:
        # Without a namespace label this control is invisible to every namespace-scoped read
        # (firing_alerts/firing_fingerprints, same filter the oracle alertname+namespace+service key
        # relies on -- see test_firing_alerts_scoped_to_namespace_excludes_infra_noise): it would
        # be silently dropped exactly like the cluster-infra noise the scoping was built to
        # exclude, so the round could never confirm the control fired nor let the model's episode
        # correlate to this namespace. Stamp the namespace whenever the caller has scoped one.
        now = datetime.now(UTC)
        labels = {"alertname": FALSE_ALARM_ALERTNAME, "source": "r1"}
        if self.namespace is not None:
            labels["namespace"] = self.namespace
        alert = {
            "labels": labels,
            "annotations": {"summary": "R1 scheduled false-alarm control"},
            "startsAt": now.isoformat(),
            "endsAt": (now + timedelta(minutes=duration_minutes)).isoformat(),
        }
        _request_json(f"{self.alertmanager_url.rstrip('/')}/api/v2/alerts", method="POST", payload=[alert])


@dataclass
class HttpReceiverGateway:
    receiver_url: str

    def received_total(self) -> int:
        payload = _request_json(f"{self.receiver_url.rstrip('/')}/metrics")
        try:
            return int(payload["received_total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise R1Error("WebhookReceiver /metrics lacks received_total") from exc


@dataclass
class HttpVllmTokenGateway:
    """vLLM's prompt-token counter, read from /metrics on the host. Seen in round 2 (2026-09-30): the receiver
    took every webhook while the agent's every model call was refused, and the smoke passed. This
    counter only moves when a prompt actually reaches the model, so it is the smoke's proof that
    the agent -> vLLM path answers, not just the Alertmanager -> receiver one."""

    metrics_url: str

    def prompt_tokens_total(self) -> float:
        from harness.metrics import TOKEN_COUNTERS, parse_prometheus_text

        try:
            with urllib.request.urlopen(self.metrics_url, timeout=20) as response:  # noqa: S310 - explicit CLI arg
                text = response.read().decode("utf-8")
            series = parse_prometheus_text(text).get(TOKEN_COUNTERS[0], [])
        except (urllib.error.URLError, ValueError) as exc:
            raise R1Error(f"vLLM /metrics unreadable at {self.metrics_url}: {exc}") from exc
        if len(series) != 1:
            raise R1Error(f"{TOKEN_COUNTERS[0]}: expected 1 series at {self.metrics_url}, found {len(series)}")
        return series[0][1]


COMPARATOR_USAGE_LINES_SIGNAL = "comparator usage log request lines"

# Runs inside the agent pod (python:3.12-slim, stdlib only). Reads the path from the pod's own env
# so k8s/deployment.yaml stays the only place it is written. A file that does not exist yet means
# no paid response so far (0); a pod without the variable is an error, never 0.
_COUNT_USAGE_LINES = (
    "import os, sys\n"
    "p = os.environ.get('SSLM_COMPARATOR_USAGE_LOG')\n"
    "if not p:\n"
    "    sys.exit('SSLM_COMPARATOR_USAGE_LOG is not set in the agent pod')\n"
    "try:\n"
    "    with open(p, 'rb') as fh:\n"
    "        print(sum(1 for line in fh if line.strip()))\n"
    "except FileNotFoundError:\n"
    "    print(0)\n"
)


@dataclass
class KubectlComparatorUsageGateway:
    """The comparator arm's equivalent of HttpVllmTokenGateway: how many request lines the agent's
    Anthropic client has written to its usage JSONL (adapters/anthropic_model_client.py appends one
    per API response, usage block or not). A line only appears after the API answered, so a
    refused egress, a missing key, a schema the API rejects or a pod that fell back to anything
    else all leave it flat. Read with `kubectl exec` into the agent Deployment, under whatever
    KUBECONFIG the driver runs with (seed_cluster.sh exports the seed's own)."""

    namespace: str = "sslm-agent"
    deployment: str = "sslm-agent"
    timeout_seconds: float = 60.0
    run: Callable[..., object] | None = None

    def request_lines_total(self) -> float:
        import subprocess

        run = self.run or subprocess.run
        argv = [
            "kubectl", "-n", self.namespace, "exec", f"deploy/{self.deployment}", "--",
            "python3", "-c", _COUNT_USAGE_LINES,
        ]
        try:
            completed = run(argv, capture_output=True, text=True, timeout=self.timeout_seconds)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise R1Error(f"comparator usage log unreadable via kubectl exec: {exc}") from exc
        if completed.returncode != 0:
            raise R1Error(
                f"comparator usage log unreadable via kubectl exec (rc={completed.returncode}): "
                f"{(completed.stderr or '').strip()[-400:]}"
            )
        try:
            return float(int(completed.stdout.strip()))
        except ValueError as exc:
            raise R1Error(f"comparator usage log count is not an integer: {completed.stdout!r}") from exc


class FaultInjector(Protocol):
    """prepare once -> inject/recover per schedule entry -> cleanup once at the end."""

    def prepare(self) -> dict: ...

    def inject(self, family: str) -> None: ...

    def recover(self) -> dict: ...

    def cleanup(self) -> dict: ...

    def probe(self) -> dict:
        """Root-cause probe of the ACTIVE fault: {"root_cause_fixed": bool | None, ...}. Only called
        by run_r1(post_action_probe=True). Never raises by contract (None + reason instead)."""
        ...


# AIOpsLab's own recover_fault for this family is `app.cleanup()` (namespace and PVs deleted,
# read in aiopslab/generators/fault/inject_virtual.py: recover_redepoly_without_pv). A
# continuous round cannot keep going on a deleted app, so after this recovery the injector
# restores the same baseline prepare() built, and says so in the family record.
BASELINE_DESTROYING_RECOVERY = frozenset({"redeploy_without_PV"})

# Families whose AIOpsLab injector is a Chaos Mesh experiment (the candidate container_kill).
# AIOpsLab's KubeCtl.exec_command returns stderr instead of raising, so `kubectl apply` of a
# PodChaos with Chaos Mesh absent (no CRD) or its webhook not ready "succeeds" with nothing
# applied, and a detection run would score that silent no-op as not_detected: a false negative
# in the results. The injector checks the preconditions before, and the experiment after, and
# raises R1Error otherwise, which every caller records as an error (not measured), never a verdict.
CHAOS_MESH_FAMILIES = frozenset({"container_kill"})
CHAOS_MESH_NAMESPACE = "chaos-mesh"  # AIOpsLab's own (inject_symp.py), same as setup_cluster.sh
PODCHAOS_CRD = "podchaos.chaos-mesh.org"

AGENT_RBAC_MANIFESTS = ("role-app-mutate.yaml", "rolebinding-app-mutate.yaml")
KUBECTL_APPLY_TIMEOUT_SECONDS = 120


class AIOpsLabFaultInjector:
    """Drive AIOpsLab's own problem classes without its per-episode deploy/teardown.

    `Orchestrator.init_problem` deploys OpenEBS, the observer Prometheus and the app, then
    injects, all inside one call; `AiopsLabEnvironment.finalize` recovers and deletes the app
    namespace. The 2026-09-23 real run used that pair per schedule entry: the first ~20 minutes
    of bootstrap ran after the round clock had started, and every family ended by deleting the app.
    Here the bootstrap half of init_problem runs once in prepare(), before the clock, and each
    entry only calls the problem's inject_fault/recover_fault, the same methods init_problem and
    finalize call, so the fault semantics stay AIOpsLab's.

    `families` defaults to the admitted catalog (PROBES). A candidate family
    (CANDIDATE_PROBES) is injected and recovered by name like any other, e.g.
    AIOpsLabFaultInjector(families=("container_kill",)) or `--smoke --smoke-family
    container_kill`; it only never reaches an admission schedule until it is admitted.

    The former `problem_suffix` knob is gone: nothing live passed it, and with explicit
    ids per family it could only produce ids that do not exist upstream (wrong_bin_usage-
    mitigation-2) or silently run the rate variant under the admitted geo family's name.
    """

    def __init__(self, families: tuple[str, ...] | None = None) -> None:
        self._families = tuple(sorted(families or PROBES))
        self._orchestrator = None
        self._namespace: str | None = None
        self._active = None
        self._active_family: str | None = None

    def _problem_id(self, family: str) -> str:
        """Every family, admitted or candidate, through the one permanent map
        (adapters/aiopslab_problems.PROBLEM_ID_BY_FAMILY): promoting a candidate moves it between
        registries and must never change the problem it resolves to."""
        problem_id = PROBLEM_ID_BY_FAMILY.get(family)
        if problem_id is None:
            raise R1Error(f"no AIOpsLab problem id for fault family {family!r} (PROBLEM_ID_BY_FAMILY)")
        return problem_id

    def _require_chaos_mesh(self) -> None:
        """Chaos Mesh must be installed and its controller Ready before a PodChaos is applied
        (see CHAOS_MESH_FAMILIES for why a silent no-op is the failure mode)."""
        assert self._orchestrator is not None
        kubectl = self._orchestrator.kubectl
        crd = (kubectl.exec_command(f"kubectl get crd {PODCHAOS_CRD} -o name") or "").strip()
        # `-o name` prints exactly "customresourcedefinition.../<name>"; a NotFound error also
        # names the CRD, so a substring check would pass on the error text itself.
        if not (crd.startswith("customresourcedefinition") and crd.endswith(f"/{PODCHAOS_CRD}")):
            raise R1Error(
                f"Chaos Mesh absent: CRD {PODCHAOS_CRD} not found ({crd[:200]!r}); "
                "run setup_cluster.sh with CHAOS_MESH_INSTALL=1"
            )
        raw = kubectl.exec_command(
            f"kubectl -n {CHAOS_MESH_NAMESPACE} get deploy chaos-controller-manager -o json"
        ) or ""
        try:
            status = json.loads(raw).get("status") or {}
        except (ValueError, AttributeError):
            raise R1Error(f"chaos-controller-manager not readable: {raw.strip()[:200]!r}") from None
        if (status.get("readyReplicas") or 0) < 1:
            raise R1Error(f"chaos-controller-manager not Ready: status={status}")

    def _require_chaos_experiment(self, problem) -> None:
        """After inject_fault: the container-kill PodChaos must exist on the target service."""
        raw = problem.kubectl.exec_command(f"kubectl get podchaos -n {self._namespace} -o json") or ""
        try:
            items = json.loads(raw).get("items", [])
        except (ValueError, AttributeError):
            raise R1Error(f"PodChaos not readable after inject: {raw.strip()[:200]!r}") from None
        service = getattr(problem, "faulty_service", None) or ""
        if not container_kill_experiments(items, service):
            raise R1Error(
                f"inject_fault applied no container-kill PodChaos on {service!r} in {self._namespace} "
                "(kubectl apply failed without raising?); episode not measured"
            )

    def _problem(self, family: str):
        assert self._orchestrator is not None
        problem_id = self._problem_id(family)
        if problem_id in LOCAL_PROBLEM_IDS:
            # The HotelReservation assign_to_non_existent_node wrapper is not in AIOpsLab's
            # registry; add it to this orchestrator's registry instance (idempotent). Lazy, so an
            # admitted-only round never imports the wrapper's AIOpsLab modules.
            from adapters.aiopslab_problems import register_local_problems

            register_local_problems(self._orchestrator.probs)
        # container_kill's problem constructor runs `helm install chaos-mesh` on every
        # instantiation (aiopslab/generators/fault/inject_symp.py, SymptomFaultInjector.__init__,
        # no --wait). With the release already up (setup_cluster.sh CHAOS_MESH_INSTALL=1) that
        # install fails without raising and changes nothing; without it, the first PodChaos can
        # race a controller that is not Ready yet. Hence the setup step, not a change here.
        return self._orchestrator.probs.get_problem_instance(problem_id)

    def prepare(self) -> dict:
        if self._orchestrator is not None:
            raise R1Error("prepare() already ran for this injector")
        from adapters.aiopslab_env import _ensure_aiopslab_importable, _widen_infra_wait_defaults

        _ensure_aiopslab_importable()
        _widen_infra_wait_defaults()
        from aiopslab.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        orchestrator.agent_name = "sslm-r1-admission"
        self._orchestrator = orchestrator

        if CHAOS_MESH_FAMILIES & set(self._families):
            # Before instantiating: container_kill's constructor helm-installs Chaos Mesh itself.
            self._require_chaos_mesh()
        problems = {family: self._problem(family) for family in self._families}
        namespaces = {family: getattr(problem, "namespace", None) for family, problem in problems.items()}
        if len(set(namespaces.values())) != 1 or None in namespaces.values():
            raise R1Error(
                f"families do not share one app namespace, so this is not one continuous round: {namespaces}"
            )
        deployments = {
            family: orchestrator.probs.get_problem_deployment(self._problem_id(family))
            for family in self._families
        }
        if any(value == "docker" for value in deployments.values()):
            raise R1Error(f"the round needs Kubernetes-deployed problems, got {deployments}")
        self._namespace = next(iter(namespaces.values()))

        # Same bootstrap as Orchestrator.init_problem, minus the fault.
        from aiopslab.service.telemetry.prometheus import Prometheus

        orchestrator.kubectl.exec_command(
            "kubectl apply -f https://openebs.github.io/charts/openebs-operator.yaml"
        )
        orchestrator.kubectl.exec_command(
            "kubectl patch storageclass openebs-hostpath -p "
            "'{\"metadata\": {\"annotations\":{\"storageclass.kubernetes.io/is-default-class\":\"true\"}}}'"
        )
        orchestrator.kubectl.wait_for_ready("openebs")
        orchestrator.prometheus = Prometheus()
        orchestrator.prometheus.deploy()
        self._deploy_app(problems[self._families[0]].app)
        # AIOpsLab's app.delete() removes the manifests' resources, not the namespace, so the
        # Role/RoleBinding setup_cluster.sh applied should survive. Reapplying is idempotent and
        # makes that independent of the upstream implementation.
        self._reapply_agent_rbac()
        return {"app_namespace": self._namespace, "families": list(self._families), "agent_rbac_reapplied": True}

    def _deploy_app(self, app) -> None:
        app.delete()
        app.deploy()  # waits for the namespace to be Ready (widened max_wait)

    def _reapply_agent_rbac(self) -> None:
        """A recreated app namespace loses the agent's namespaced Role/RoleBinding."""
        import subprocess

        k8s_dir = TOOLS_DIR / "k8s"
        for name in AGENT_RBAC_MANIFESTS:
            text = (k8s_dir / name).read_text(encoding="utf-8").replace("${APP_NAMESPACE}", str(self._namespace))
            # Bounded like the smoke's policy check: a hung apiserver must raise into the
            # journaled recovery error, not block the detached round forever.
            subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=text,
                text=True,
                check=True,
                capture_output=True,
                timeout=KUBECTL_APPLY_TIMEOUT_SECONDS,
            )

    def inject(self, family: str) -> None:
        if self._orchestrator is None:
            raise R1Error("inject() before prepare(): the app is not deployed")
        if self._active is not None:
            raise R1Error("attempted to inject a fault while another fault is active")
        import asyncio
        import atexit
        import inspect

        from aiopslab.orchestrator.orchestrator import exit_cleanup_fault

        if family in CHAOS_MESH_FAMILIES:
            self._require_chaos_mesh()
        problem = self._problem(family)
        # Set before inject_fault so a half-applied injection is still recovered.
        self._active, self._active_family = problem, family
        problem.inject_fault()
        atexit.register(exit_cleanup_fault, prob=problem)
        if family in CHAOS_MESH_FAMILIES:
            # Raised with the fault marked active, so the caller's recover() still runs
            # AIOpsLab's recover_fault (a no-op delete if nothing was applied).
            self._require_chaos_experiment(problem)
        if inspect.iscoroutinefunction(problem.start_workload):
            asyncio.run(problem.start_workload())
        else:
            problem.start_workload()

    def recover(self) -> dict:
        if self._active is None:
            raise R1Error("attempted to recover with no active AIOpsLab fault")
        import atexit

        from aiopslab.orchestrator.orchestrator import exit_cleanup_fault

        problem, family = self._active, self._active_family
        # Cleared even when recovery raises: the caller records that error, and cleanup() must
        # not retry recover_fault() on a fault that already failed to recover once.
        self._active, self._active_family = None, None
        try:
            recovery_detail = problem.recover_fault()
        finally:
            # A failed recovery is already recorded and aborts the round; leaving the hook armed
            # would retry it at interpreter exit, on a cluster cleanup() has since torn down.
            atexit.unregister(exit_cleanup_fault)
        if family in BASELINE_DESTROYING_RECOVERY:
            self._deploy_app(type(problem.app)())
            self._reapply_agent_rbac()
            return {"baseline_restored": True, "agent_rbac_reapplied": True}
        problem.kubectl.wait_for_ready(self._namespace)
        result: dict = {"baseline_restored": False}
        if isinstance(recovery_detail, dict):
            # Only the two revision-preserving local wrappers return anything (RevisionFault.recover()'s
            # undo_ran/to_revision/strategy_restored); upstream's own recover_fault returns None.
            # Surfacing it here, instead of leaving it print()-only, is what lets a live
            # revalidation prove "recovered by undo" from the journaled result, not just that
            # recover_fault() did not raise.
            result["revision_recovery"] = recovery_detail
        return result

    def probe(self) -> dict:
        if self._active is None:
            return {"root_cause_fixed": None, "probe_error": "probe() with no active fault"}
        problem, family = self._active, self._active_family
        service = getattr(problem, "faulty_service", None)
        runner = run_probe if family in PROBES else run_candidate_probe
        return runner(family, problem.kubectl, self._namespace, service)

    def cleanup(self) -> dict:
        if self._namespace is None:
            # prepare() never got past its guards, so no app was deployed.
            return {"cleanup": "skipped, prepare never deployed the app"}
        if self._active is not None:
            self.recover()
        self._problem(self._families[0]).app.cleanup()
        return {"cleanup": "app namespace deleted"}


class Journal:
    """Incremental evidence: one fsynced JSONL line per phase/event plus an atomic state file.

    The 2026-09-23 run kept its result only in memory and died without proof of anything.
    Every line lands on disk before the next phase starts, so a killed run still shows how far
    it got and why.
    """

    def __init__(self, events_path: Path | None, state_path: Path | None, clock: Clock) -> None:
        self.events_path = events_path
        self.state_path = state_path
        self.clock = clock
        for path in (events_path, state_path):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)

    def _stamp(self) -> dict:
        now = self.clock.time()
        return {"t": now, "at": datetime.fromtimestamp(now, UTC).isoformat()}

    def event(self, phase: str, **fields: object) -> None:
        record = {**self._stamp(), "phase": phase, **fields}
        print(f"[r1] {record['at']} {phase} {json.dumps(fields, sort_keys=True, default=str)}", file=sys.stderr, flush=True)
        if self.events_path is None:
            return
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def state(self, payload: dict) -> None:
        if self.state_path is None:
            return
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, self.state_path)


def _sleep_until(clock: Clock, started_at: float, minute: int, seconds_per_minute: float) -> None:
    remaining = started_at + minute * seconds_per_minute - clock.time()
    if remaining > 0:
        clock.sleep(remaining)


def _scope_alerts_to_namespace(alerts: AlertGateway, namespace: str | None) -> None:
    """Best-effort: only gateways that carry a settable `namespace` attribute (HttpAlertGateway)
    are scoped. A double without one (most test fakes) is left untouched, same as before this
    scoping existed."""
    if namespace is not None and hasattr(alerts, "namespace"):
        alerts.namespace = namespace


@dataclass
class SettleResult:
    settled: bool
    watch_started_at: float
    decided_at: float
    # The reading at decided_at: empty once settled, and also empty on a timeout where the app
    # had already gone clean but not for stable_seconds yet (clean_for_seconds says how long).
    signals: list[str] = field(default_factory=list)
    # The last non-empty reading seen during the watch, whatever happened after it.
    last_dirty: list[str] = field(default_factory=list)
    # How long the app had read clean at decided_at; None when it was dirty at decided_at.
    clean_for_seconds: float | None = None


def _wait_for_app_settled(
    alerts: AlertGateway,
    clock: Clock,
    *,
    not_before: float,
    deadline: float,
    stable_seconds: float,
    poll_seconds: float,
) -> SettleResult:
    """Settle gate: returns once the app has read settled (unsettled_signals() empty) continuously for
    `stable_seconds`, and never before `not_before` (the entry's planned start), so the last
    reading is taken at the moment the caller acts. Gives up at `deadline`."""
    watch_started_at = clock.time()
    clean_since: float | None = None
    last_dirty: list[str] = []
    while True:
        now = clock.time()
        signals = alerts.unsettled_signals()
        if signals:
            clean_since, last_dirty = None, list(signals)
        elif clean_since is None:
            clean_since = now
        ready_at = None if clean_since is None else max(not_before, clean_since + stable_seconds)
        clean_for = None if clean_since is None else now - clean_since
        if ready_at is not None and now >= ready_at:
            return SettleResult(True, watch_started_at, now, [], last_dirty, clean_for)
        if now >= deadline:
            return SettleResult(False, watch_started_at, now, list(signals), last_dirty, clean_for)
        wake = min(deadline, now + max(poll_seconds, 1e-3))
        if ready_at is not None:
            wake = min(wake, ready_at)
        clock.sleep(max(0.0, wake - now))


def _settle_before_entry(
    alerts: AlertGateway,
    entry: ScheduleEntry,
    started_at: float,
    clock: Clock,
    seconds_per_minute: float,
    poll_seconds: float,
) -> SettleResult:
    _sleep_until(clock, started_at, entry.start_minute - PRE_ENTRY_SETTLE_LEAD_MINUTES, seconds_per_minute)
    planned_at = started_at + entry.start_minute * seconds_per_minute
    return _wait_for_app_settled(
        alerts,
        clock,
        not_before=planned_at,
        deadline=planned_at + PRE_ENTRY_MAX_OVERRUN_MINUTES * seconds_per_minute,
        stable_seconds=PRE_ENTRY_STABLE_MINUTES * seconds_per_minute,
        poll_seconds=poll_seconds,
    )


def _wait_for_new_alert(
    alerts: AlertGateway,
    baseline: set[str],
    deadline: float,
    clock: Clock,
    poll_seconds: float,
) -> dict[str, dict]:
    """Returns the newly-firing alerts (fingerprints not in `baseline`) as
    {fingerprint: {"alertname", "labels"}}, so the caller can audit which rule fired."""

    def _new() -> dict[str, dict]:
        current = alerts.firing_alerts()
        return {fingerprint: detail for fingerprint, detail in current.items() if fingerprint not in baseline}

    while clock.time() < deadline:
        observed = _new()
        if observed:
            return observed
        clock.sleep(min(poll_seconds, max(0.0, deadline - clock.time())))
    return _new()


def _collect_new_alerts(
    alerts: AlertGateway,
    baseline: set[str],
    deadline: float,
    clock: Clock,
    poll_seconds: float,
) -> tuple[dict[str, dict], float | None]:
    """Like _wait_for_new_alert, but keeps watching until the deadline: every new fingerprint
    seen at any poll, plus when the first one showed. The scorer's alert_fired is "any
    in-namespace symptom alert during the episode", and a record opened by a later alert of the same fault
    needs that alert's labels too."""
    seen: dict[str, dict] = {}
    first_at: float | None = None

    def _poll() -> None:
        nonlocal first_at
        for fingerprint, detail in alerts.firing_alerts().items():
            if fingerprint not in baseline and fingerprint not in seen:
                seen[fingerprint] = detail
                if first_at is None:
                    first_at = clock.time()

    while clock.time() < deadline:
        _poll()
        clock.sleep(min(poll_seconds, max(0.0, deadline - clock.time())))
    _poll()
    return seen, first_at


def _observe_quiet_window(
    alerts: AlertGateway,
    entry: ScheduleEntry,
    started_at: float,
    clock: Clock,
    seconds_per_minute: float,
    poll_seconds: float,
) -> QuietResult:
    """Observe from max(planned start, now): a recovery that overran eats into the window.

    The previous entry's recovery (including a baseline redeploy) has already returned when
    this runs, so nothing of ours is bootstrapping or tearing down during the observation.
    A window left shorter than the required 360 minutes is recorded as not admissible, never padded.
    Settle gate: the app must be settled first (same gate as a fault injection); a settle that overruns
    the planned start eats into the window like an overrunning recovery does, and a window that
    never settled is not admissible at all.
    """
    settle = _settle_before_entry(alerts, entry, started_at, clock, seconds_per_minute, poll_seconds)
    _sleep_until(clock, started_at, entry.start_minute, seconds_per_minute)
    begin = clock.time()
    planned_start = started_at + entry.start_minute * seconds_per_minute
    end = started_at + entry.end_minute * seconds_per_minute
    window_start = planned_start if begin - planned_start <= QUIET_START_JITTER_SECONDS else begin
    effective_minutes = max(0.0, (end - window_start) / seconds_per_minute)
    unexpected: set[str] = set()
    while settle.settled and clock.time() < end:
        unexpected.update(alerts.firing_fingerprints())
        if unexpected:
            break
        clock.sleep(min(poll_seconds, max(0.0, end - clock.time())))
    long_enough = effective_minutes >= R1_QUIET_WINDOW_MINUTES
    return QuietResult(
        start_minute=entry.start_minute,
        duration_minutes=entry.duration_minutes,
        alert_free=settle.settled and not unexpected and long_enough,
        unexpected_fingerprints=sorted(unexpected),
        actual_start=begin,
        actual_end=clock.time(),
        effective_minutes=effective_minutes,
        truncated=not long_enough,
        settle=asdict(settle),
    )


class _Abort(Exception):
    pass


def run_r1(
    schedule: FaultSchedule,
    *,
    injector: FaultInjector,
    alerts: AlertGateway,
    receiver: ReceiverGateway,
    clock: Clock = time,
    seconds_per_minute: float = 60.0,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    alert_timeout_seconds: float = DEFAULT_ALERT_TIMEOUT_SECONDS,
    journal: Journal | None = None,
    quiet_only: bool = False,
    post_action_probe: bool = False,
    dry_run: bool = False,
) -> R1Result:
    # quiet_only: nothing is injected, so family admission and webhook loss (0 sent) are not part
    # of the verdict; passed means the one quiet window was admissible and alert-free.
    if quiet_only and dry_run:
        raise R1Error("--dry-run and --quiet-only are separate modes")
    if quiet_only:
        validate_schedule_for_quiet_rerun(schedule)
    elif dry_run:
        validate_schedule_for_dry_run(schedule)
    else:
        validate_schedule_for_r1(schedule)
    if seconds_per_minute <= 0:
        raise R1Error("seconds_per_minute must be positive for a real-time run")
    journal = journal or Journal(None, None, clock)

    family_results: list[FamilyResult] = []
    quiet_result: QuietResult | None = None
    aborted: str | None = None
    started_at: float | None = None
    receiver_start: int | None = None
    cleanup_detail: dict = {}

    def snapshot(complete: bool) -> dict:
        return {
            "complete": complete,
            "started_at": started_at,
            "aborted": aborted,
            "families": [asdict(item) for item in family_results],
            "quiet_window": None if quiet_result is None else asdict(quiet_result),
            "receiver_start": receiver_start,
            "dry_run": dry_run,
        }

    # Bootstrap happens BEFORE the clock. The schedule's minute 0 is "environment and receiver
    # ready", never "started deploying".
    journal.event("prepare_start")
    try:
        prepare_detail = injector.prepare()
    except Exception as exc:  # noqa: BLE001 - recorded, then the round is not started at all
        journal.event("prepare_failed", error=f"{type(exc).__name__}: {exc}")
        # A half-done prepare can leave the app deployed; same cleanup contract as run_smoke.
        try:
            cleanup_detail = injector.cleanup() or {}
            journal.event("cleanup_done", **cleanup_detail)
        except Exception as cleanup_exc:  # noqa: BLE001 - recorded, the prepare error wins
            cleanup_detail = {"cleanup_error": f"{type(cleanup_exc).__name__}: {cleanup_exc}"}
            journal.event("cleanup_failed", **cleanup_detail)
        journal.state({**snapshot(False), "aborted": "prepare failed", "cleanup": cleanup_detail})
        raise R1Error(f"prepare failed before the admission clock started: {exc}") from exc
    _scope_alerts_to_namespace(alerts, (prepare_detail or {}).get("app_namespace"))
    receiver_start = receiver.received_total()
    started_at = clock.time()
    # round_id is half of the scorer's bootstrap cluster key (scorer.Episode.round_id): seed alone would
    # merge two runs of the same schedule into one cluster, so the clock start goes in with it.
    round_id = f"seed{schedule.seed}-{datetime.fromtimestamp(started_at, UTC):%Y%m%dT%H%M%SZ}"
    journal.event(
        "clock_started",
        receiver_start=receiver_start,
        journal_version=JOURNAL_VERSION,
        rodada_id=round_id,  # journal key kept for compatibility with published run data ("rodada" = round)
        seed=schedule.seed,
        seconds_per_minute=seconds_per_minute,
        **(prepare_detail or {}),
    )
    journal.state(snapshot(False))

    try:
        for entry in sorted(schedule.entries, key=lambda item: item.start_minute):
            planned_at = started_at + entry.start_minute * seconds_per_minute
            if entry.kind is EntryKind.FAULT:
                assert entry.fault_family is not None
                settle = _settle_before_entry(alerts, entry, started_at, clock, seconds_per_minute, poll_seconds)
                result = FamilyResult(entry.fault_family, entry.start_minute, False)
                result.planned_at, result.actual_at = planned_at, clock.time()
                result.settle = asdict(settle)
                if not settle.settled:
                    # Settle gate: injecting into a dirty app would make any new alert unattributable.
                    result.measured = False
                    family_results.append(result)
                    journal.event(
                        "entry_not_measured",
                        family=entry.fault_family,
                        start_minute=entry.start_minute,
                        end_minute=entry.end_minute,
                        **result.settle,
                    )
                    journal.state(snapshot(False))
                    continue
                baseline = alerts.firing_fingerprints()
                journal.event(
                    "inject_start",
                    family=entry.fault_family,
                    start_minute=entry.start_minute,
                    end_minute=entry.end_minute,
                )
                try:
                    injector.inject(entry.fault_family)
                    result.injected = True
                    journal.event("inject_done", family=entry.fault_family)
                    entry_end = started_at + entry.end_minute * seconds_per_minute
                    if post_action_probe:
                        # The agent gets the whole window, so the fault stays in until the
                        # planned end and every new alert in it counts toward alert_fired.
                        new_alerts, first_alert_at = _collect_new_alerts(
                            alerts, baseline, entry_end, clock, poll_seconds
                        )
                    else:
                        deadline = min(entry_end, clock.time() + alert_timeout_seconds)
                        new_alerts = _wait_for_new_alert(alerts, baseline, deadline, clock, poll_seconds)
                        first_alert_at = clock.time() if new_alerts else None
                    result.observed_alerts = [
                        {"fingerprint": fingerprint, "alertname": detail.get("alertname"), "labels": detail.get("labels", {})}
                        for fingerprint, detail in sorted(new_alerts.items())
                    ]
                    result.symptom_alert_admitted = bool(result.observed_alerts)
                    result.alert_at = first_alert_at
                    journal.event(
                        "alert_observed" if result.symptom_alert_admitted else "alert_missing",
                        family=entry.fault_family,
                        alerts=result.observed_alerts,
                    )
                    if post_action_probe:
                        # Decision 2026-09-28: probe at the planned end, right before recover, so
                        # the schedule stays deterministic. A record that lands after this read
                        # gets no mitigation verdict (episode_bridge), never a borrowed one.
                        try:
                            result.probe = injector.probe()
                        except Exception as exc:  # noqa: BLE001 - contract says never raises
                            result.probe = {"root_cause_fixed": None, "probe_error": f"{type(exc).__name__}: {exc}"}
                        journal.event("probe_post_action", family=entry.fault_family, **result.probe)
                except Exception as exc:  # noqa: BLE001 - recorded, then fail fast below
                    result.error = f"inject {type(exc).__name__}: {exc}"
                finally:
                    try:
                        result.recovery = injector.recover()
                        result.recovered = True
                        journal.event("recover_done", family=entry.fault_family, **result.recovery)
                    except Exception as exc:  # noqa: BLE001
                        recovery_error = f"recovery {type(exc).__name__}: {exc}"
                        result.error = f"{result.error}; {recovery_error}" if result.error else recovery_error
                    result.webhook_received_so_far = receiver.received_total() - receiver_start
                    family_results.append(result)
                    journal.event("family_done", **asdict(result))
                    journal.state(snapshot(False))
                if result.error:
                    # Environment state is unknown after a failed inject or recovery; nothing
                    # later in this round would be evidence of anything.
                    raise _Abort(f"{entry.fault_family} at minute {entry.start_minute}: {result.error}")
                # A family that misses its alert is a detectability finding, not
                # an abort. The round keeps going; compute_family_verdicts() judges it afterward.
                continue
            if entry.kind is EntryKind.QUIET and entry.duration_minutes >= R1_QUIET_WINDOW_MINUTES:
                # _observe_quiet_window() does its own sleeping: the settle watch starts
                # PRE_ENTRY_SETTLE_LEAD_MINUTES before the window, not at it.
                if quiet_result is None:
                    journal.event("quiet_start", start_minute=entry.start_minute)
                    quiet_result = _observe_quiet_window(
                        alerts, entry, started_at, clock, seconds_per_minute, poll_seconds
                    )
                    journal.event("quiet_done", **asdict(quiet_result))
                    journal.state(snapshot(False))
            elif entry.kind is EntryKind.FALSE_ALARM:
                _sleep_until(clock, started_at, entry.start_minute, seconds_per_minute)
                alerts.send_false_alarm(entry.duration_minutes)
                journal.event(
                    "false_alarm_sent",
                    start_minute=entry.start_minute,
                    end_minute=entry.end_minute,
                    duration_minutes=entry.duration_minutes,
                    alertname=FALSE_ALARM_ALERTNAME,
                )
    except _Abort as exc:
        aborted = str(exc)
        journal.event("aborted", reason=aborted)
    finally:
        try:
            cleanup_detail = injector.cleanup() or {}
            journal.event("cleanup_done", **cleanup_detail)
        except Exception as exc:  # noqa: BLE001 - recorded; the verdict below does not need it
            cleanup_detail = {"cleanup_error": f"{type(exc).__name__}: {exc}"}
            journal.event("cleanup_failed", **cleanup_detail)

    receiver_end = receiver.received_total()
    elapsed_minutes = max(1, math.ceil((clock.time() - started_at) / seconds_per_minute))
    sent = alerts.webhook_notifications(elapsed_minutes)
    received = receiver_end - receiver_start
    loss = None if sent <= 0 else max(0.0, 1 - received / sent)
    family_verdicts = compute_family_verdicts(family_results)
    passed = (
        False
        if dry_run
        else aborted is None and quiet_result is not None and quiet_result.alert_free
        if quiet_only
        else aborted is None
        and all(item.measured for item in family_results)
        and quiet_result is not None
        and quiet_result.alert_free
        and loss is not None
        and loss <= MAX_WEBHOOK_LOSS
        and r1_admission_passed(family_verdicts)
    )
    webhook = WebhookAccounting(
        query_window_minutes=elapsed_minutes,
        alertmanager_sent=sent,
        receiver_start=receiver_start,
        receiver_end=receiver_end,
        received=received,
        denominator=sent,
        loss=loss,
    )
    journal.event("webhook_accounted", **asdict(webhook))
    journal.event("family_verdicts", **family_verdicts)
    result = R1Result(
        schedule.to_json(),
        family_results,
        quiet_result,
        sent,
        received,
        loss,
        passed,
        started_at=started_at,
        aborted=aborted,
        webhook=webhook,
        cleanup=cleanup_detail,
        family_verdicts=family_verdicts,
        dry_run=dry_run,
    )
    journal.event("finished", passed=passed, aborted=aborted, dry_run=dry_run)
    journal.state({"complete": True, **result.to_json()})
    return result


DEFAULT_SMOKE_FAMILY = "user_unregistered_mongodb"  # the family the policy blocked on 2026-09-23


def run_smoke(
    *,
    injector: FaultInjector,
    alerts: AlertGateway,
    receiver: ReceiverGateway,
    policy_check: Callable[[], tuple[int, str]],
    family: str = DEFAULT_SMOKE_FAMILY,
    clock: Clock = time,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    alert_timeout_seconds: float = DEFAULT_ALERT_TIMEOUT_SECONDS,
    journal: Journal | None = None,
    model_tokens: Callable[[], float] | None = None,
    model_signal: str = "vLLM prompt_tokens_total",
) -> dict:
    """Cheap pre-round check on a fresh lab: every seam a round depends on, exercised once.

    prepare once -> policy scope (agent denied, injector admitted) -> one inject/alert/recover
    of a Mongo family -> one synthetic alert through Alertmanager to the receiver -> cleanup.
    Each step is journaled; the first failing step stops the smoke. A passing smoke is the
    precondition for starting a full round, never evidence for the round itself.

    With `model_tokens` (a round with the model on), the smoke also requires vLLM's prompt
    token counter to move between before the inject and after the receiver step: the agent's
    episodes for those alerts must have reached the model. A counter that cannot be read fails
    the smoke too, never passes it. Several seeds share one vLLM, so this proves the path from
    the VM's kind network, not from each seed on its own; the failure it exists for (vLLM not
    reachable from kind at all) hits every seed alike.

    The comparator arm (API) passes its own counter as `model_tokens`
    (KubectlComparatorUsageGateway.request_lines_total, per seed, since each pod has its own log)
    and names it in `model_signal`; the rule is the same: it must move, or the smoke fails.
    """
    journal = journal or Journal(None, None, clock)
    steps: dict[str, object] = {}
    failure: str | None = None
    try:
        journal.event("smoke_prepare_start")
        steps["prepare"] = injector.prepare()
        _scope_alerts_to_namespace(alerts, (steps["prepare"] or {}).get("app_namespace"))
        code, output = policy_check()
        steps["policy_scope"] = {"exit_code": code, "output": output}
        journal.event("smoke_policy_scope", exit_code=code)
        if code != 0:
            raise _Abort(f"policy scope check exited {code}: {output.strip()[-400:]}")
        tokens_before = model_tokens() if model_tokens is not None else None
        now = clock.time()
        settle = _wait_for_app_settled(
            alerts,
            clock,
            not_before=now,
            deadline=now + PRE_ENTRY_MAX_OVERRUN_MINUTES * 60.0,
            stable_seconds=PRE_ENTRY_STABLE_MINUTES * 60.0,
            poll_seconds=poll_seconds,
        )
        steps["settle"] = asdict(settle)
        journal.event("smoke_settle", **steps["settle"])
        if not settle.settled:
            raise _Abort(
                f"app never settled before inject (settle gate): at the deadline {settle.signals[:10] or 'clean'}"
                f" for {settle.clean_for_seconds}s, last dirty reading {settle.last_dirty[:10]}"
            )
        baseline = alerts.firing_fingerprints()
        observed: dict[str, dict] = {}
        try:
            injector.inject(family)
            observed = _wait_for_new_alert(
                alerts, baseline, clock.time() + alert_timeout_seconds, clock, poll_seconds
            )
        finally:
            # Same contract as run_r1: every inject attempt is followed by its own recover.
            observed_alerts = [
                {"fingerprint": fingerprint, "alertname": detail.get("alertname"), "labels": detail.get("labels", {})}
                for fingerprint, detail in sorted(observed.items())
            ]
            steps["inject"] = {"family": family, "alerts": observed_alerts}
            try:
                steps["recover"] = injector.recover()
                journal.event("smoke_family", family=family, alerts=observed_alerts, **steps["recover"])
            except Exception as exc:  # noqa: BLE001 - journaled; the inject error, if any, wins
                steps["recover"] = {"recovery_error": f"{type(exc).__name__}: {exc}"}
                journal.event("smoke_recover_failed", family=family, **steps["recover"])
        if "recovery_error" in steps["recover"]:
            raise _Abort(f"{family} recovery failed: {steps['recover']['recovery_error']}")
        if not observed:
            raise _Abort(f"{family} injected and recovered but produced no symptom alert")
        before = receiver.received_total()
        alerts.send_false_alarm(1)
        deadline = clock.time() + alert_timeout_seconds
        after = receiver.received_total()
        while after <= before and clock.time() < deadline:
            clock.sleep(poll_seconds)
            after = receiver.received_total()
        steps["receiver"] = {"before": before, "after": after}
        journal.event("smoke_receiver", before=before, after=after)
        if after <= before:
            raise _Abort("synthetic alert never reached the WebhookReceiver")
        if model_tokens is not None:
            deadline = clock.time() + alert_timeout_seconds
            tokens_after = model_tokens()
            while tokens_after <= tokens_before and clock.time() < deadline:
                clock.sleep(poll_seconds)
                tokens_after = model_tokens()
            steps["model"] = {"prompt_tokens_before": tokens_before, "prompt_tokens_after": tokens_after}
            if model_signal != "vLLM prompt_tokens_total":
                steps["model"]["signal"] = model_signal
            journal.event("smoke_model", **steps["model"])
            if tokens_after <= tokens_before:
                path = "agent -> vLLM" if model_signal == "vLLM prompt_tokens_total" else "agent -> API"
                raise _Abort(
                    f"the agent's episodes never reached the model: {model_signal} stayed at "
                    f"{tokens_after} ({path} path refused or unrouted, see the agent log)"
                )
    except _Abort as exc:
        failure = str(exc)
    except Exception as exc:  # noqa: BLE001 - any seam failing is the answer the smoke exists for
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            steps["cleanup"] = injector.cleanup() or {}
            journal.event("cleanup_done", **steps["cleanup"])
        except Exception as exc:  # noqa: BLE001
            steps["cleanup"] = {"cleanup_error": f"{type(exc).__name__}: {exc}"}
            journal.event("cleanup_failed", **steps["cleanup"])
    result = {"smoke": True, "passed": failure is None, "failure": failure, "steps": steps}
    journal.event("smoke_finished", passed=failure is None, failure=failure)
    journal.state({"complete": True, **result})
    return result


POLICY_CHECK_TIMEOUT_SECONDS = 300


def _policy_check_script(
    app_namespace: str, timeout_seconds: float = POLICY_CHECK_TIMEOUT_SECONDS
) -> Callable[[], tuple[int, str]]:
    def check() -> tuple[int, str]:
        import subprocess

        try:
            completed = subprocess.run(
                ["bash", str(TOOLS_DIR / "verify_db_pod_policy_scope.sh")],
                env={**os.environ, "APP_NAMESPACE": app_namespace},
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            # A hung apiserver must fail the smoke, not hang the detached launcher.
            return 124, f"policy scope check timed out after {timeout_seconds}s"
        return completed.returncode, completed.stdout + completed.stderr

    return check


def _load_schedule(path: Path | None, seed: int | None) -> FaultSchedule:
    if path is not None:
        return FaultSchedule.from_json(json.loads(path.read_text(encoding="utf-8")))
    if seed is None:
        raise R1Error("pass --schedule (the committed schedule) or --seed to generate one for review")
    return generate_r1_schedule(seed)


def _smoke_model_signal(args: argparse.Namespace) -> dict:
    if args.comparator_usage:
        return {
            "model_tokens": KubectlComparatorUsageGateway().request_lines_total,
            "model_signal": COMPARATOR_USAGE_LINES_SIGNAL,
        }
    return {
        "model_tokens": (
            HttpVllmTokenGateway(args.vllm_metrics_url).prompt_tokens_total if args.vllm_metrics_url else None
        )
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, help="committed FaultSchedule JSON")
    parser.add_argument("--seed", type=int, help="generate a schedule for review before committing it")
    parser.add_argument("--alertmanager-url", required=True)
    parser.add_argument("--prometheus-url", required=True)
    parser.add_argument("--receiver-url", required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="result JSON, rewritten atomically after every entry (complete=false until the end)",
    )
    parser.add_argument("--journal", type=Path, help="phase/event JSONL, default: <out>.events.jsonl")
    parser.add_argument("--seconds-per-minute", type=float, default=60.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the cheap pre-round smoke (one Mongo family, policy scope, receiver) instead of a round",
    )
    parser.add_argument(
        "--quiet-only",
        action="store_true",
        help="re-measure only the admission run's 360-minute quiet window (no injections); incompatible with --schedule/--seed",
    )
    parser.add_argument(
        "--post-action-probe",
        action="store_true",
        help="model round: hold each fault to its planned end, collect every alert, probe before recover",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run a short committed schedule end to end; drops the admission minimums, never passes",
    )
    parser.add_argument("--smoke-family", default=DEFAULT_SMOKE_FAMILY)
    parser.add_argument("--app-namespace", help="required with --smoke, for the policy scope check")
    parser.add_argument(
        "--vllm-metrics-url",
        help="with --smoke on a model round: vLLM /metrics; the smoke then fails unless prompt tokens move",
    )
    parser.add_argument(
        "--comparator-usage",
        action="store_true",
        help=(
            "with --smoke on the API comparator arm: count request lines in the agent pod's "
            "SSLM_COMPARATOR_USAGE_LOG; the smoke then fails unless that count moves (no vLLM needed)"
        ),
    )
    args = parser.parse_args(argv)
    if args.comparator_usage and args.vllm_metrics_url:
        raise R1Error("--comparator-usage and --vllm-metrics-url are two arms; pass one")
    if args.comparator_usage and not args.smoke:
        raise R1Error("--comparator-usage only applies to --smoke")

    journal_path = args.journal or args.out.with_suffix(".events.jsonl")
    if args.smoke:
        if not args.app_namespace:
            raise R1Error("--smoke needs --app-namespace for the policy scope check")
        smoke = run_smoke(
            injector=AIOpsLabFaultInjector(),
            alerts=HttpAlertGateway(args.alertmanager_url, args.prometheus_url),
            receiver=HttpReceiverGateway(args.receiver_url),
            policy_check=_policy_check_script(args.app_namespace),
            family=args.smoke_family,
            journal=Journal(journal_path, args.out, time),
            **_smoke_model_signal(args),
        )
        print(json.dumps(smoke, indent=2, sort_keys=True, default=str))
        if not smoke["passed"]:
            raise SystemExit(1)
        return

    if args.dry_run and (args.quiet_only or args.schedule is None):
        raise R1Error("--dry-run needs --schedule (a committed short schedule) and no --quiet-only")
    if args.quiet_only and (args.schedule or args.seed is not None):
        raise R1Error("--quiet-only builds its own schedule; drop --schedule/--seed")
    schedule = quiet_rerun_schedule() if args.quiet_only else _load_schedule(args.schedule, args.seed)
    result = run_r1(
        schedule,
        injector=AIOpsLabFaultInjector(),
        alerts=HttpAlertGateway(args.alertmanager_url, args.prometheus_url),
        receiver=HttpReceiverGateway(args.receiver_url),
        seconds_per_minute=args.seconds_per_minute,
        journal=Journal(journal_path, args.out, time),
        quiet_only=args.quiet_only,
        post_action_probe=args.post_action_probe,
        dry_run=args.dry_run,
    )
    print(json.dumps(result.to_json(), indent=2, sort_keys=True))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
