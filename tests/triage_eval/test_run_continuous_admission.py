"""Pure-logic coverage for the continuous-instrument admission driver ("R1" in identifiers)."""

from __future__ import annotations

import pytest

from adapters.mitigation_probes import PROBES
from scenarios.fault_schedule import EntryKind, FaultSchedule, ScheduleEntry
import json
import sys
import threading
import types
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scenarios.run_continuous_admission import (
    AIOpsLabFaultInjector,
    HttpAlertGateway,
    HttpReceiverGateway,
    Journal,
    MIN_INJECTIONS_PER_FAMILY,
    R1Error,
    _request_json,
    _scope_alerts_to_namespace,
    _wait_for_app_settled,
    QUIET_START_JITTER_SECONDS,
    compute_family_verdicts,
    generate_r1_schedule,
    main,
    quiet_rerun_schedule,
    r1_admission_passed,
    run_r1,
    validate_schedule_for_dry_run,
    validate_schedule_for_r1,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeAlerts:
    def __init__(self, *, sent=5, quiet_alert=False):
        self.firing: dict[str, dict] = {}
        self.sent = sent
        self.quiet_alert = quiet_alert
        self.false_alarms = 0
        # Settle gate: one reading consumed per unsettled_signals() poll, then `always_unsettled` for good.
        self.unsettled: list[list[str]] = []
        self.always_unsettled: list[str] = []
        self.settle_polls = 0

    def firing_alerts(self):
        alerts = dict(self.firing)
        if self.quiet_alert:
            alerts["unexpected-quiet-alert"] = {"alertname": "UnexpectedQuietAlert", "labels": {}}
        return alerts

    def firing_fingerprints(self):
        return set(self.firing_alerts().keys())

    def webhook_notifications(self, window_minutes):
        return self.sent

    def send_false_alarm(self, duration_minutes):
        self.false_alarms += 1

    def unsettled_signals(self):
        self.settle_polls += 1
        if self.unsettled:
            return list(self.unsettled.pop(0))
        return list(self.always_unsettled)


class FakeReceiver:
    """First read is the baseline, every later read is `end` (the receiver kept up)."""

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.reads = 0

    def received_total(self):
        self.reads += 1
        return self.start if self.reads == 1 else self.end


class FakeInjector:
    def __init__(self, alerts, *, clock=None, prepare_seconds=0.0, recover_seconds=0.0,
                 fail_inject=(), silent=(), fail_prepare=False):
        self.alerts = alerts
        self.clock = clock
        self.prepare_seconds = prepare_seconds
        self.recover_seconds = recover_seconds
        self.fail_inject = set(fail_inject)
        self.silent = set(silent)
        self.fail_prepare = fail_prepare
        self.calls = []
        self.injected = []

    def prepare(self):
        self.calls.append("prepare")
        if self.fail_prepare:
            raise RuntimeError("openebs never became ready")
        if self.clock is not None:
            self.clock.sleep(self.prepare_seconds)  # a slow bootstrap, like the real ~20 minutes
        return {"app_namespace": "test-hotel-reservation"}

    def inject(self, family):
        self.calls.append(f"inject:{family}")
        self.injected.append(family)
        if family in self.fail_inject:
            raise RuntimeError(f"admission webhook denied exec for {family}")
        if family not in self.silent:
            self.alerts.firing = {
                f"symptom-{family}": {"alertname": f"Symptom{family}", "labels": {"family": family}}
            }

    def recover(self):
        self.calls.append("recover")
        self.alerts.firing = {}
        if self.clock is not None:
            self.clock.sleep(self.recover_seconds)
        return {"baseline_restored": False}

    def cleanup(self):
        self.calls.append("cleanup")
        return {"cleanup": "fake"}


# Promotion (2026-09-25): PROBES grew from 5 to 8 admitted families, and validate_schedule_for_r1
# (run at the top of run_r1) demands every PROBES key with >= MIN_INJECTIONS_PER_FAMILY episodes --
# so this fixture must track the real admitted catalog, not a fixed list, or every _r1_schedule()
# below silently falls short and every test using it starts failing with "missing" families.
FAMILIES = tuple(sorted(PROBES))

# The admission run needs >= MIN_INJECTIONS_PER_FAMILY episodes per family for a tri-state
# verdict, so every fixture schedule below injects each family that many times, never just once.
TOTAL_FAULT_EPISODES = len(FAMILIES) * MIN_INJECTIONS_PER_FAMILY


# The quiet window starts at minute 12, not 0: the settle watch needs its lead
# (PRE_ENTRY_SETTLE_LEAD_MINUTES) inside the round, like the real schedule's 30-minute gaps.
QUIET_START = 12


def _r1_schedule():
    entries = [ScheduleEntry(EntryKind.QUIET, QUIET_START, 360)]
    minute = QUIET_START + 360 + 2  # same 2-minute gap the faults keep between each other
    for family in FAMILIES:
        for _ in range(MIN_INJECTIONS_PER_FAMILY):
            entries.append(ScheduleEntry(EntryKind.FAULT, minute, 1, family))
            minute += 2
    return FaultSchedule(seed=1, window_hours=24, entries=tuple(entries))


def _run(schedule=None, *, alerts=None, injector=None, receiver=None, clock=None, journal=None):
    alerts = alerts or FakeAlerts()
    clock = clock or FakeClock()
    return run_r1(
        schedule or _r1_schedule(),
        injector=injector or FakeInjector(alerts),
        alerts=alerts,
        receiver=receiver or FakeReceiver(100, 105),
        clock=clock,
        seconds_per_minute=1,
        poll_seconds=1,
        alert_timeout_seconds=2,
        journal=journal,
    )


def test_r1_passes_only_when_every_family_quiet_window_and_webhook_loss_pass():
    alerts = FakeAlerts()
    result = run_r1(
        _r1_schedule(),
        injector=FakeInjector(alerts),
        alerts=alerts,
        receiver=FakeReceiver(100, 105),
        clock=FakeClock(),
        seconds_per_minute=1,
        poll_seconds=1,
        alert_timeout_seconds=2,
    )

    assert result.passed is True
    assert {item.family for item in result.families if item.symptom_alert_admitted} == set(FAMILIES)
    assert result.quiet_window and result.quiet_window.alert_free is True
    assert result.webhook_loss == 0


def test_r1_fails_when_webhook_loss_exceeds_one_percent():
    alerts = FakeAlerts(sent=5)
    result = run_r1(
        _r1_schedule(),
        injector=FakeInjector(alerts),
        alerts=alerts,
        receiver=FakeReceiver(100, 104),
        clock=FakeClock(),
        seconds_per_minute=1,
        poll_seconds=1,
        alert_timeout_seconds=2,
    )

    assert result.webhook_loss == pytest.approx(0.2)
    assert result.passed is False


def test_r1_fails_when_the_six_hour_quiet_window_has_an_alert():
    alerts = FakeAlerts(quiet_alert=True)
    result = run_r1(
        _r1_schedule(),
        injector=FakeInjector(alerts),
        alerts=alerts,
        receiver=FakeReceiver(100, 105),
        clock=FakeClock(),
        seconds_per_minute=1,
        poll_seconds=1,
        alert_timeout_seconds=2,
    )

    assert result.quiet_window and result.quiet_window.alert_free is False
    assert result.passed is False


def test_r1_rejects_schedule_missing_an_admitted_fault_family():
    schedule = FaultSchedule(
        seed=1,
        window_hours=24,
        entries=(
            ScheduleEntry(EntryKind.QUIET, 0, 360),
            ScheduleEntry(EntryKind.FAULT, 360, 1, "wrong_bin_usage"),
        ),
    )

    try:
        validate_schedule_for_r1(schedule)
    except R1Error as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("the admission validator accepted a schedule without all probe families")


def test_generated_r1_schedule_has_every_family_and_six_hour_quiet_window():
    schedule = generate_r1_schedule(1)

    validate_schedule_for_r1(schedule)


# --- lifecycle: bootstrap before the clock, one prepare, one cleanup ------------------------------


def test_bootstrap_runs_before_the_clock_starts():
    # 2026-09-23: AIOpsLab's lazy bootstrap (~20 min) ran inside the first inject, after the round
    # clock had started. The schedule's minute 0 must be "ready", not "started deploying".
    clock = FakeClock()
    alerts = FakeAlerts()
    injector = FakeInjector(alerts, clock=clock, prepare_seconds=1200)
    result = _run(alerts=alerts, injector=injector, clock=clock)

    assert result.started_at == 1200
    first = result.families[0]
    assert first.planned_at == first.actual_at == 1200 + first.start_minute
    assert result.passed is True


def test_prepare_runs_once_and_cleanup_once_with_no_teardown_between_families():
    alerts = FakeAlerts()
    injector = FakeInjector(alerts)
    _run(alerts=alerts, injector=injector)

    assert injector.calls[0] == "prepare"
    assert injector.calls[-1] == "cleanup"
    assert injector.calls.count("prepare") == 1
    assert injector.calls.count("cleanup") == 1
    assert injector.calls.count("recover") == TOTAL_FAULT_EPISODES


def test_prepare_failure_never_starts_the_clock_or_injects(tmp_path):
    alerts = FakeAlerts()
    injector = FakeInjector(alerts, fail_prepare=True)
    events = tmp_path / "r1.events.jsonl"
    with pytest.raises(R1Error, match="before the admission clock started"):
        _run(alerts=alerts, injector=injector, journal=Journal(events, tmp_path / "r1.json", FakeClock()))

    assert injector.injected == []
    assert injector.calls == ["prepare", "cleanup"]  # a half-done prepare is still cleaned up
    phases = [json.loads(line)["phase"] for line in events.read_text().splitlines()]
    assert phases == ["prepare_start", "prepare_failed", "cleanup_done"]


# --- incremental evidence -------------------------------------------------------------------------


def test_journal_and_state_are_on_disk_before_the_next_family_starts(tmp_path):
    events = tmp_path / "r1.events.jsonl"
    state = tmp_path / "r1.json"
    clock = FakeClock()
    alerts = FakeAlerts()
    seen_mid_run = []

    class PeekingInjector(FakeInjector):
        def inject(self, family):
            if self.injected:  # second family onward: the previous one must already be persisted
                snapshot = json.loads(state.read_text())
                seen_mid_run.append((snapshot["complete"], [f["family"] for f in snapshot["families"]]))
            super().inject(family)

    _run(alerts=alerts, injector=PeekingInjector(alerts), clock=clock, journal=Journal(events, state, clock))

    assert seen_mid_run[0] == (False, [FAMILIES[0]])
    assert len(seen_mid_run[-1][1]) == TOTAL_FAULT_EPISODES - 1
    phases = [json.loads(line)["phase"] for line in events.read_text().splitlines()]
    assert phases[:2] == ["prepare_start", "clock_started"]
    assert phases.count("family_done") == TOTAL_FAULT_EPISODES
    assert phases[-4:] == ["cleanup_done", "webhook_accounted", "family_verdicts", "finished"]
    final = json.loads(state.read_text())
    assert final["complete"] is True and final["passed"] is True


def test_journal_carries_what_the_scorer_bridge_joins_on(tmp_path):
    """Round id, planned window per entry and the false-alarm alertname are in
    the journal, so the joiner never re-derives them from the schedule or this module."""
    from scenarios.run_continuous_admission import FALSE_ALARM_ALERTNAME, JOURNAL_VERSION

    events = tmp_path / "r1.events.jsonl"
    clock = FakeClock()
    alerts = FakeAlerts()
    schedule = FaultSchedule(
        seed=1,
        window_hours=24,
        entries=(ScheduleEntry(EntryKind.FALSE_ALARM, 0, 5),) + _r1_schedule().entries,
    )
    _run(schedule, alerts=alerts, injector=FakeInjector(alerts), clock=clock, journal=Journal(events, None, clock))
    lines = [json.loads(line) for line in events.read_text().splitlines()]
    by_phase = {}
    for line in lines:
        by_phase.setdefault(line["phase"], []).append(line)

    (started,) = by_phase["clock_started"]
    assert started["journal_version"] == JOURNAL_VERSION
    assert started["rodada_id"].startswith(f"seed{started['seed']}-")
    assert started["seconds_per_minute"] > 0

    for inject in by_phase["inject_start"]:
        assert inject["end_minute"] > inject["start_minute"]
    assert by_phase["false_alarm_sent"], "schedule above has one false alarm"
    for false_alarm in by_phase["false_alarm_sent"]:
        assert false_alarm["alertname"] == FALSE_ALARM_ALERTNAME
        assert false_alarm["end_minute"] == false_alarm["start_minute"] + false_alarm["duration_minutes"]


# --- fail fast ------------------------------------------------------------------------------------


def test_injector_failure_aborts_the_round_and_still_cleans_up():
    # The exact 2026-09-23 failure: the injector was denied on the Mongo pod.
    alerts = FakeAlerts()
    injector = FakeInjector(alerts, fail_inject={"revoke_auth_mongodb"})
    result = _run(alerts=alerts, injector=injector)

    assert result.passed is False
    assert "revoke_auth_mongodb" in result.aborted and "denied exec" in result.aborted
    assert injector.injected[-1] == "revoke_auth_mongodb"
    assert "user_unregistered_mongodb" not in injector.injected  # nothing after the failure
    assert injector.calls[-1] == "cleanup"


# --- a family that misses its alert is a finding, never an abort ---------------------------------


def test_family_that_never_alerts_gets_a_not_detected_verdict_and_the_round_keeps_going():
    alerts = FakeAlerts()
    injector = FakeInjector(alerts, silent={"redeploy_without_PV"})
    result = _run(alerts=alerts, injector=injector)

    assert result.aborted is None
    assert result.family_verdicts["redeploy_without_PV"] == "not_detected"
    # Every other family (7 of 8) still detected: misconfig, mongo_auth (4 families across the
    # geo/rate variants), wrong_bin_usage, assign_to_non_existent_node -- 4 distinct mechanisms.
    assert result.passed is True
    assert injector.injected[-1] == FAMILIES[-1]  # the round ran through every scheduled episode


def test_family_that_alerts_on_only_one_of_two_injections_is_intermittent():
    alerts = FakeAlerts()

    class FirstMissInjector(FakeInjector):
        def inject(self, family):
            if family == "wrong_bin_usage":
                self.silent = set() if "wrong_bin_usage" in self.injected else {"wrong_bin_usage"}
            super().inject(family)

    result = _run(alerts=alerts, injector=FirstMissInjector(alerts))

    assert result.aborted is None
    assert [f.symptom_alert_admitted for f in result.families if f.family == "wrong_bin_usage"] == [False, True]
    assert result.family_verdicts["wrong_bin_usage"] == "intermittent"


def test_compute_family_verdicts_is_tri_state_per_family_unit_episode():
    from scenarios.run_continuous_admission import FamilyResult

    results = [
        FamilyResult("a", 0, True),
        FamilyResult("a", 10, True),
        FamilyResult("b", 20, False),
        FamilyResult("b", 30, False),
        FamilyResult("c", 40, True),
        FamilyResult("c", 50, False),
    ]

    assert compute_family_verdicts(results) == {"a": "detected", "b": "not_detected", "c": "intermittent"}


def test_r1_admission_passed_needs_three_families_and_two_distinct_mechanisms():
    # revoke_auth_mongodb + user_unregistered_mongodb are the same mongo_auth mechanism.
    only_mongo_auth = {
        "revoke_auth_mongodb": "detected",
        "user_unregistered_mongodb": "detected",
        "misconfig_app_hotel_res": "detected",
    }
    assert r1_admission_passed(only_mongo_auth) is True  # 3 families, 2 mechanisms
    two_families_only = {"revoke_auth_mongodb": "detected", "wrong_bin_usage": "detected"}
    assert r1_admission_passed(two_families_only) is False  # only 2 families detected
    one_mechanism_only = {
        "revoke_auth_mongodb": "detected",
        "user_unregistered_mongodb": "detected",
        "redeploy_without_PV": "intermittent",
    }
    assert r1_admission_passed(one_mechanism_only) is False  # 3rd family isn't "detected"


def test_r1_needs_at_least_two_injections_per_family():
    schedule = FaultSchedule(
        seed=1,
        window_hours=24,
        entries=(
            ScheduleEntry(EntryKind.QUIET, 0, 360),
            *(ScheduleEntry(EntryKind.FAULT, 360 + i, 1, family) for i, family in enumerate(FAMILIES)),
        ),
    )

    try:
        validate_schedule_for_r1(schedule)
    except R1Error as exc:
        assert "under quota" in str(exc)
    else:
        raise AssertionError("the admission validator accepted a schedule with only 1 injection for some families")


# --- quiet window and webhook accounting ----------------------------------------------------------


def test_quiet_window_truncated_by_an_overrunning_recovery_is_not_admissible():
    # FAMILIES[0]'s quota (MIN_INJECTIONS_PER_FAMILY) is supplied by the two explicit entries
    # below; FAMILIES[1:] * MIN_INJECTIONS_PER_FAMILY supplies every other family's quota.
    schedule = FaultSchedule(
        seed=1,
        window_hours=24,
        entries=(
            ScheduleEntry(EntryKind.FAULT, 0, 1, FAMILIES[0]),
            ScheduleEntry(EntryKind.QUIET, 1, 360),
            ScheduleEntry(EntryKind.FAULT, 400, 1, FAMILIES[0]),
            *(
                ScheduleEntry(EntryKind.FAULT, 410 + 2 * i, 1, family)
                for i, family in enumerate(FAMILIES[1:] * MIN_INJECTIONS_PER_FAMILY)
            ),
        ),
    )
    clock = FakeClock()
    alerts = FakeAlerts()
    # recovery of the first family takes 30 "minutes", eating into the 360-minute window
    result = _run(schedule, alerts=alerts, injector=FakeInjector(alerts, clock=clock, recover_seconds=30), clock=clock)

    quiet = result.quiet_window
    assert quiet.truncated is True and quiet.alert_free is False
    assert quiet.effective_minutes < 360
    assert result.passed is False


class OversleepingClock(FakeClock):
    """Real time.sleep() wakes a little late; the admission run's quiet window started ~0.2s
    past plan."""

    def __init__(self, oversleep):
        super().__init__()
        self.oversleep = oversleep

    def sleep(self, seconds):
        self.now += seconds + (self.oversleep if seconds > 0 else 0.0)


def _quiet_only(*, alerts=None, clock=None):
    alerts = alerts or FakeAlerts(sent=0)
    clock = clock or FakeClock()
    return run_r1(
        quiet_rerun_schedule(),
        injector=FakeInjector(alerts),
        alerts=alerts,
        receiver=FakeReceiver(7, 7),
        clock=clock,
        seconds_per_minute=60,
        poll_seconds=15,
        quiet_only=True,
    )


def test_quiet_window_start_jitter_is_not_a_truncation():
    # Admission run (2026-09-26): 0 unexpected fingerprints, settled, and still scored truncated at 359.997.
    result = _quiet_only(clock=OversleepingClock(0.2))

    quiet = result.quiet_window
    assert quiet.actual_start > 60 * quiet.start_minute  # it did start late
    assert quiet.truncated is False and quiet.effective_minutes == 360
    assert quiet.alert_free is True and result.passed is True


def test_quiet_window_start_later_than_jitter_bound_still_truncates():
    result = _quiet_only(clock=OversleepingClock(QUIET_START_JITTER_SECONDS + 1))

    assert result.quiet_window.truncated is True
    assert result.quiet_window.alert_free is False and result.passed is False


def test_quiet_only_rerun_fails_on_an_unexpected_alert():
    result = _quiet_only(alerts=FakeAlerts(sent=0, quiet_alert=True))

    assert result.quiet_window.unexpected_fingerprints == ["unexpected-quiet-alert"]
    assert result.passed is False


def test_quiet_only_rerun_injects_nothing_and_rejects_fault_schedules():
    alerts = FakeAlerts(sent=0)
    injector = FakeInjector(alerts)
    run_r1(quiet_rerun_schedule(), injector=injector, alerts=alerts, receiver=FakeReceiver(0, 0),
           clock=FakeClock(), seconds_per_minute=60, poll_seconds=15, quiet_only=True)
    assert injector.injected == []

    with pytest.raises(R1Error, match="injects nothing"):
        run_r1(_r1_schedule(), injector=FakeInjector(alerts), alerts=alerts, receiver=FakeReceiver(0, 0),
               clock=FakeClock(), quiet_only=True)


def test_quiet_only_cli_refuses_a_schedule(tmp_path):
    with pytest.raises(R1Error, match="--quiet-only"):
        main(["--quiet-only", "--seed", "600", "--alertmanager-url", "x", "--prometheus-url", "x",
              "--receiver-url", "x", "--out", str(tmp_path / "r.json")])


def test_webhook_accounting_declares_denominator_and_counters():
    result = _run(receiver=FakeReceiver(100, 105))
    webhook = result.webhook
    assert (webhook.receiver_start, webhook.receiver_end, webhook.received) == (100, 105, 5)
    assert webhook.denominator == webhook.alertmanager_sent == 5
    assert webhook.loss == 0
    assert webhook.query_window_minutes >= 366


# --- HTTP gateways against local doubles (dry integration, no cluster, no GPU) --------------------


class _Doubles(BaseHTTPRequestHandler):
    firing: list = []
    posted: list = []
    received = 0
    # Settle gate: Prometheus instant-query results by query prefix, plus every query seen.
    prom_alerts: list = []
    not_ready: list = []
    queries: list = []

    def log_message(self, *_args):
        pass

    def _send(self, code, body=b""):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/v2/alerts"):
            self._send(200, json.dumps(type(self).firing).encode())
        elif self.path.startswith("/api/v1/query"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)["query"][0]
            type(self).queries.append(query)
            if query.startswith("ALERTS"):
                result = type(self).prom_alerts
            elif query.startswith("kube_pod_status_ready"):
                result = type(self).not_ready
            else:
                result = [{"value": [0, str(float(len(type(self).posted)))]}]
            self._send(200, json.dumps({"data": {"result": result}}).encode())
        elif self.path == "/metrics":
            self._send(200, json.dumps({"received_total": type(self).received}).encode())
        else:
            self._send(404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        type(self).posted.append(json.loads(self.rfile.read(length)))
        type(self).received += 1
        self._send(200)  # Alertmanager answers POST /api/v2/alerts with 200 and an empty body


@pytest.fixture()
def doubles():
    _Doubles.firing, _Doubles.posted, _Doubles.received = [], [], 0
    _Doubles.prom_alerts, _Doubles.not_ready, _Doubles.queries = [], [], []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Doubles)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_request_json_accepts_an_empty_2xx_body(doubles):
    assert _request_json(f"{doubles}/api/v2/alerts", method="POST", payload=[{"labels": {}}]) is None


# --- namespace scoping (2026-09-24 finding: infra-noise alerts counted as detections) -------------


def test_firing_alerts_unscoped_by_default_includes_every_namespace(doubles):
    _Doubles.firing = [
        {"fingerprint": "fp-a", "status": {"state": "active"}, "labels": {"alertname": "A", "namespace": "app"}},
        {"fingerprint": "fp-b", "status": {"state": "active"}, "labels": {"alertname": "B", "namespace": "monitoring"}},
    ]
    alerts = HttpAlertGateway(doubles, doubles)
    assert set(alerts.firing_alerts()) == {"fp-a", "fp-b"}


def test_firing_alerts_scoped_to_namespace_excludes_infra_noise(doubles):
    _Doubles.firing = [
        {"fingerprint": "fp-a", "status": {"state": "active"}, "labels": {"alertname": "A", "namespace": "app"}},
        {
            "fingerprint": "fp-b",
            "status": {"state": "active"},
            "labels": {"alertname": "NodeClockNotSynchronising", "namespace": "monitoring"},
        },
        # No namespace label at all (e.g. TargetDown in the real incident): excluded too.
        {"fingerprint": "fp-c", "status": {"state": "active"}, "labels": {"alertname": "TargetDown"}},
    ]
    alerts = HttpAlertGateway(doubles, doubles, namespace="app")
    assert set(alerts.firing_alerts()) == {"fp-a"}


def test_send_false_alarm_stamps_the_scoped_namespace(doubles):
    # Regression: the SslmSyntheticControl false alarm used to post with no namespace label, so a
    # namespace-scoped firing_alerts() dropped it exactly like the infra noise above -- the oracle
    # alertname+namespace+service key never matched it.
    alerts = HttpAlertGateway(doubles, doubles, namespace="app")
    alerts.send_false_alarm(5)
    assert _Doubles.posted[0][0]["labels"] == {
        "alertname": "SslmSyntheticControl",
        "source": "r1",
        "namespace": "app",
    }


def test_send_false_alarm_omits_namespace_when_unscoped(doubles):
    alerts = HttpAlertGateway(doubles, doubles)
    alerts.send_false_alarm(5)
    assert "namespace" not in _Doubles.posted[0][0]["labels"]


def test_scope_alerts_to_namespace_sets_it_on_gateways_that_have_the_attribute():
    alerts = HttpAlertGateway("http://am", "http://prom")
    _scope_alerts_to_namespace(alerts, "test-hotel-reservation")
    assert alerts.namespace == "test-hotel-reservation"


def test_scope_alerts_to_namespace_is_a_noop_for_doubles_without_the_attribute():
    alerts = FakeAlerts()
    _scope_alerts_to_namespace(alerts, "test-hotel-reservation")  # must not raise
    assert not hasattr(alerts, "namespace")


def test_dry_integration_over_http_doubles(doubles, tmp_path):
    alerts = HttpAlertGateway(doubles, doubles)

    class HttpDrivenInjector(FakeInjector):
        def inject(self, family):
            self.calls.append(f"inject:{family}")
            self.injected.append(family)
            _Doubles.firing = [
                {
                    "fingerprint": f"fp-{family}",
                    "status": {"state": "active"},
                    "labels": {"alertname": family, "namespace": "test-hotel-reservation"},
                },
                {"fingerprint": "wd", "status": {"state": "active"}, "labels": {"alertname": "Watchdog"}},
                # A cluster-infra alert with no relation to the fault, from a different namespace
                # (the exact shape of the 2026-09-24 finding: NodeClockNotSynchronising in
                # namespace=monitoring). Namespace scoping must exclude it, never count it.
                {
                    "fingerprint": "fp-infra-noise",
                    "status": {"state": "active"},
                    "labels": {"alertname": "NodeClockNotSynchronising", "namespace": "monitoring"},
                },
            ]
            _Doubles.received += 1  # the receiver saw the notification Alertmanager sent
            _Doubles.posted.append({"family": family})

        def recover(self):
            self.calls.append("recover")
            _Doubles.firing = []
            return {"baseline_restored": False}

    schedule = FaultSchedule(
        seed=1,
        window_hours=24,
        entries=(ScheduleEntry(EntryKind.FALSE_ALARM, 0, 5),) + _r1_schedule().entries,
    )
    clock = FakeClock()
    result = run_r1(
        schedule,
        injector=HttpDrivenInjector(alerts),
        alerts=alerts,
        receiver=HttpReceiverGateway(doubles),
        clock=clock,
        seconds_per_minute=0.001,
        poll_seconds=0.001,
        alert_timeout_seconds=1,
        journal=Journal(tmp_path / "e.jsonl", tmp_path / "r.json", clock),
    )

    assert result.passed is True, result
    # The control must carry the app namespace, or namespace-scoped firing_alerts() drops it
    # exactly like the infra-noise alert below (the oracle's alertname+namespace+service key never sees
    # an alert with no namespace).
    assert _Doubles.posted[0][0]["labels"] == {
        "alertname": "SslmSyntheticControl",
        "source": "r1",
        "namespace": "test-hotel-reservation",
    }
    assert result.webhook.received == result.webhook.alertmanager_sent == TOTAL_FAULT_EPISODES + 1
    first_alert = next(f for f in result.families if f.family == FAMILIES[0]).observed_alerts[0]
    assert first_alert == {
        "fingerprint": f"fp-{FAMILIES[0]}",
        "alertname": FAMILIES[0],
        "labels": {"alertname": FAMILIES[0], "namespace": "test-hotel-reservation"},
    }
    # The infra-noise alert (different namespace) never shows up anywhere in the result.
    all_fingerprints = {a["fingerprint"] for f in result.families for a in f.observed_alerts}
    assert "fp-infra-noise" not in all_fingerprints


# --- the real AIOpsLab injector, against a fake aiopslab package ----------------------------------


class _FakeApp:
    deploys = 0
    cleanups = 0
    namespace = "test-hotel-reservation"

    def delete(self):
        pass

    def deploy(self):
        type(self).deploys += 1

    def cleanup(self):
        type(self).cleanups += 1


class _FakeKubectl:
    def __init__(self):
        self.commands = []
        self.waited = []
        # needle -> output, first match wins; like AIOpsLab's KubeCtl, errors come back as text.
        self.responses = {}

    def exec_command(self, command):
        self.commands.append(command)
        for needle, output in self.responses.items():
            if needle in command:
                return output
        return ""

    def wait_for_ready(self, namespace):
        self.waited.append(namespace)


def _fake_aiopslab(monkeypatch, namespaces=None):
    import adapters.aiopslab_env as aiopslab_env

    monkeypatch.setattr(aiopslab_env, "_ensure_aiopslab_importable", lambda: None)
    monkeypatch.setattr(aiopslab_env, "_widen_infra_wait_defaults", lambda: None)
    _FakeApp.deploys = _FakeApp.cleanups = 0
    kubectl = _FakeKubectl()
    log = []

    class Problem:
        def __init__(self, family):
            self.family = family
            self.app = _FakeApp()
            self.kubectl = kubectl
            self.namespace = (namespaces or {}).get(family, "test-hotel-reservation")
            self.faulty_service = "geo"

        def inject_fault(self):
            log.append(f"inject:{self.family}")

        def recover_fault(self):
            log.append(f"recover:{self.family}")
            if self.family == "redeploy_without_PV":
                self.app.cleanup()  # what AIOpsLab's own recovery does

        def start_workload(self):
            log.append(f"workload:{self.family}")

    class Registry:
        def __init__(self):
            # Same shape as AIOpsLab's ProblemRegistry (a PROBLEM_REGISTRY dict of factories), so
            # register_local_problems can add the local candidate wrapper to this instance.
            self.PROBLEM_REGISTRY = {}
            self.requested = []

        def get_problem_instance(self, pid):
            self.requested.append(pid)
            if pid in self.PROBLEM_REGISTRY:
                return self.PROBLEM_REGISTRY[pid]()
            return Problem(pid.rsplit("-", 2)[0])

        def get_problem_deployment(self, _pid):
            return "k8s"

    class Orchestrator:
        def __init__(self):
            self.probs = Registry()
            self.kubectl = kubectl

    class Prometheus:
        def deploy(self):
            log.append("prometheus")

    modules = {
        "aiopslab": types.ModuleType("aiopslab"),
        "aiopslab.orchestrator": types.ModuleType("aiopslab.orchestrator"),
        "aiopslab.orchestrator.orchestrator": types.ModuleType("aiopslab.orchestrator.orchestrator"),
        "aiopslab.service": types.ModuleType("aiopslab.service"),
        "aiopslab.service.telemetry": types.ModuleType("aiopslab.service.telemetry"),
        "aiopslab.service.telemetry.prometheus": types.ModuleType("aiopslab.service.telemetry.prometheus"),
    }
    modules["aiopslab.orchestrator"].Orchestrator = Orchestrator
    modules["aiopslab.orchestrator.orchestrator"].exit_cleanup_fault = lambda prob: None
    modules["aiopslab.service.telemetry.prometheus"].Prometheus = Prometheus
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    applied = []
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **kw: applied.append((cmd, kw["input"])) or types.SimpleNamespace(returncode=0)
    )
    # The local HotelReservation assign wrapper would import real AIOpsLab modules; stand
    # in a Problem the log can tell apart from the registry's own ones.
    import adapters.aiopslab_problems as aiopslab_problems

    # One stable factory object, like the real lru_cached wrapper class, so re-registration on
    # every inject stays idempotent instead of reading as an upstream conflict.
    def local_assign_factory():
        return Problem("local-assign-hotel-res")

    # wrong_bin_usage also resolves to a local (revision-preserving) wrapper. It keeps its
    # family name here so the namespace guard below still keys on "wrong_bin_usage".
    def local_wrong_bin_factory():
        return Problem("wrong_bin_usage")

    monkeypatch.setattr(
        aiopslab_problems,
        "local_problem_factories",
        lambda: {
            aiopslab_problems.ASSIGN_HOTEL_RES_PROBLEM_ID: local_assign_factory,
            aiopslab_problems.WRONG_BIN_PROBLEM_ID: local_wrong_bin_factory,
        },
    )
    return log, kubectl, applied


def test_real_injector_prepares_once_and_never_tears_down_between_families(monkeypatch):
    log, kubectl, _applied = _fake_aiopslab(monkeypatch)
    injector = AIOpsLabFaultInjector()

    prepared = injector.prepare()
    assert prepared["app_namespace"] == "test-hotel-reservation" and prepared["agent_rbac_reapplied"]
    assert len(_applied) == 2
    assert _FakeApp.deploys == 1 and log == ["prometheus"]
    assert any("openebs-operator" in c for c in kubectl.commands)

    injector.inject("wrong_bin_usage")
    assert injector.recover() == {"baseline_restored": False}
    injector.inject("misconfig_app_hotel_res")
    injector.recover()

    assert _FakeApp.cleanups == 0 and _FakeApp.deploys == 1
    assert kubectl.waited == ["openebs", "test-hotel-reservation", "test-hotel-reservation"]
    injector.cleanup()
    assert _FakeApp.cleanups == 1


def test_real_injector_forwards_the_revision_wrapper_s_undo_detail(monkeypatch):
    """The two revision-preserving local wrappers' recover_fault returns RevisionFault.recover()'s detail
    (undo_ran/to_revision/strategy_restored, see adapters/aiopslab_problems._revision_recover).
    A live revalidation needs that forwarded, not swallowed, to tell "recovered by undo" apart
    from "recover_fault() merely did not raise"."""
    log, _kubectl, _applied = _fake_aiopslab(monkeypatch)
    injector = AIOpsLabFaultInjector()
    injector.prepare()

    class _WithRevisionDetail:
        def __init__(self, problem):
            self.__dict__.update(problem.__dict__)
            self._problem = problem

        def inject_fault(self):
            self._problem.inject_fault()

        def start_workload(self):
            self._problem.start_workload()

        def recover_fault(self):
            self._problem.recover_fault()
            return {"undo_ran": True, "to_revision": "3", "strategy_restored": "RollingUpdate"}

    monkeypatch.setattr(
        injector, "_problem", lambda family, _orig=injector._problem: _WithRevisionDetail(_orig(family))
    )

    injector.inject("wrong_bin_usage")
    detail = injector.recover()

    assert detail == {
        "baseline_restored": False,
        "revision_recovery": {"undo_ran": True, "to_revision": "3", "strategy_restored": "RollingUpdate"},
    }
    assert "recover:wrong_bin_usage" in log


def test_real_injector_restores_baseline_after_redeploy_without_pv(monkeypatch):
    _log, _kubectl, applied = _fake_aiopslab(monkeypatch)
    injector = AIOpsLabFaultInjector()
    injector.prepare()
    injector.inject("redeploy_without_PV")

    detail = injector.recover()

    assert detail == {"baseline_restored": True, "agent_rbac_reapplied": True}
    assert _FakeApp.cleanups == 1 and _FakeApp.deploys == 2  # AIOpsLab deleted it, we redeployed
    assert len(applied) == 4  # Role + RoleBinding after prepare, and again after the redeploy
    assert all("${APP_NAMESPACE}" not in text and "test-hotel-reservation" in text for _cmd, text in applied)


def test_real_injector_refuses_inject_before_prepare_and_double_inject(monkeypatch):
    _fake_aiopslab(monkeypatch)
    injector = AIOpsLabFaultInjector()
    with pytest.raises(R1Error, match="before prepare"):
        injector.inject("wrong_bin_usage")
    injector.prepare()
    injector.inject("wrong_bin_usage")
    with pytest.raises(R1Error, match="another fault is active"):
        injector.inject("revoke_auth_mongodb")


def test_real_injector_recovers_a_half_applied_injection(monkeypatch):
    log, _kubectl, _applied = _fake_aiopslab(monkeypatch)
    injector = AIOpsLabFaultInjector()
    injector.prepare()
    monkeypatch.setattr(injector, "_problem", lambda family, _orig=injector._problem: _Exploding(_orig(family)))

    with pytest.raises(RuntimeError, match="half applied"):
        injector.inject("revoke_auth_mongodb")
    injector.recover()
    assert "recover:revoke_auth_mongodb" in log


class _Exploding:
    def __init__(self, problem):
        self.__dict__.update(problem.__dict__)
        self._problem = problem

    def inject_fault(self):
        raise RuntimeError("half applied")

    def recover_fault(self):
        self._problem.recover_fault()


def test_real_injector_fails_before_the_clock_when_families_do_not_share_a_namespace(monkeypatch):
    _fake_aiopslab(monkeypatch, namespaces={"wrong_bin_usage": "other-app"})
    with pytest.raises(R1Error, match="not one continuous round"):
        AIOpsLabFaultInjector().prepare()


# --- candidate and promoted families through the real injector: irregular problem ids ------------
#
# revoke_auth_mongodb_rate and user_unregistered_mongodb_rate were promoted into PROBES (2026-09-25);
# container_kill is still the one remaining candidate. All three keep the same irregular
# PROBLEM_ID_BY_FAMILY entries either way (promotion never touches that map, see the README), so
# these tests exercise the injector mechanics regardless of admission status.

CHAOS_READY = {
    "get crd podchaos.chaos-mesh.org": "customresourcedefinition.apiextensions.k8s.io/podchaos.chaos-mesh.org\n",
    "get deploy chaos-controller-manager": json.dumps({"status": {"replicas": 3, "readyReplicas": 3}}),
    "get podchaos -n test-hotel-reservation": json.dumps(
        {
            "items": [
                {
                    "metadata": {"name": "container-kill"},
                    "spec": {"action": "container-kill", "selector": {"labelSelectors": {"io.kompose.service": "geo"}}},
                }
            ]
        }
    ),
}


@pytest.mark.parametrize(
    "family, problem_id",
    [
        ("container_kill", "container_kill-detection"),
        ("revoke_auth_mongodb_rate", "revoke_auth_mongodb-mitigation-2"),
        ("user_unregistered_mongodb_rate", "user_unregistered_mongodb-mitigation-2"),
    ],
)
def test_real_injector_injects_and_recovers_an_irregular_family_by_name(monkeypatch, family, problem_id):
    log, kubectl, _applied = _fake_aiopslab(monkeypatch)
    kubectl.responses.update(CHAOS_READY)
    injector = AIOpsLabFaultInjector()
    injector.prepare()

    injector.inject(family)
    assert injector.recover() == {"baseline_restored": False}

    assert injector._orchestrator.probs.requested[-1] == problem_id
    upstream = problem_id.rsplit("-", 2)[0]  # what the fake registry names the Problem it built
    assert log[-3:] == [f"inject:{upstream}", f"workload:{upstream}", f"recover:{upstream}"]
    assert kubectl.waited[-1] == "test-hotel-reservation"


def test_real_injector_resolves_the_local_assign_wrapper(monkeypatch):
    log, _kubectl, _applied = _fake_aiopslab(monkeypatch)
    # Promotion (2026-09-25): assign_to_non_existent_node_hotel_res is now a default family, so
    # a bare AIOpsLabFaultInjector() would register the wrapper during prepare() itself. Scoping to
    # one unrelated family with an upstream id (not wrong_bin_usage, a local revision wrapper too)
    # is what keeps the registration this test checks lazy.
    injector = AIOpsLabFaultInjector(families=("misconfig_app_hotel_res",))
    injector.prepare()
    registry = injector._orchestrator.probs
    assert "assign_to_non_existent_node_hotel_res-mitigation-1" not in registry.PROBLEM_REGISTRY

    injector.inject("assign_to_non_existent_node_hotel_res")
    injector.recover()
    injector.inject("assign_to_non_existent_node_hotel_res")  # re-registration is idempotent
    injector.recover()

    assert registry.requested[-1] == "assign_to_non_existent_node_hotel_res-mitigation-1"
    assert "assign_to_non_existent_node_hotel_res-mitigation-1" in registry.PROBLEM_REGISTRY
    assert "inject:local-assign-hotel-res" in log and "recover:local-assign-hotel-res" in log


def test_real_injector_has_no_suffix_knob_and_refuses_an_unmapped_family(monkeypatch):
    """The suffix knob could only build ids that do not exist upstream (wrong_bin_usage-
    mitigation-2) or run the rate variant under the admitted geo family's name."""
    with pytest.raises(TypeError):
        AIOpsLabFaultInjector(problem_suffix="mitigation-2")
    _fake_aiopslab(monkeypatch)
    injector = AIOpsLabFaultInjector()
    injector.prepare()
    with pytest.raises(R1Error, match="no AIOpsLab problem id"):
        injector.inject("wrong_bin_usage_typo")
    assert injector._problem_id("revoke_auth_mongodb") == "revoke_auth_mongodb-mitigation-1"


def test_real_injector_can_prepare_a_round_of_an_explicit_family_subset(monkeypatch):
    """`families=` accepts any mix of admitted and candidate families explicitly -- it is the
    injector, not admission status, that decides what a round can drive."""
    _log, kubectl, _applied = _fake_aiopslab(monkeypatch)
    kubectl.responses.update(CHAOS_READY)
    injector = AIOpsLabFaultInjector(families=("container_kill", "assign_to_non_existent_node_hotel_res"))
    prepared = injector.prepare()
    assert prepared["families"] == ["assign_to_non_existent_node_hotel_res", "container_kill"]
    assert prepared["app_namespace"] == "test-hotel-reservation"


def test_real_injector_default_families_are_still_exactly_the_admitted_catalog(monkeypatch):
    _fake_aiopslab(monkeypatch)
    injector = AIOpsLabFaultInjector()
    assert injector.prepare()["families"] == sorted(PROBES)
    requested = injector._orchestrator.probs.requested
    # Every family resolves to a real AIOpsLab (or local wrapper) problem id: regular families to
    # their own name + -mitigation-1, and the two promoted mongo rate variants to the geo family's
    # -mitigation-2 (PROBLEM_ID_BY_FAMILY is untouched by promotion, see the README).
    rate_variants = {"revoke_auth_mongodb_rate", "user_unregistered_mongodb_rate"}
    assert requested and all(
        pid.endswith("-mitigation-2") if family in rate_variants else pid.endswith("-mitigation-1")
        for family, pid in zip(sorted(PROBES), requested)
    )


# Chaos Mesh guard: AIOpsLab's exec_command returns kubectl's stderr instead of raising, so each
# of these would otherwise be a silent no-op injection scored as not_detected.


@pytest.mark.parametrize(
    "broken, match",
    [
        ({"get crd podchaos.chaos-mesh.org": 'Error from server (NotFound): "podchaos.chaos-mesh.org" not found'}, "Chaos Mesh absent"),
        ({"get deploy chaos-controller-manager": json.dumps({"status": {"replicas": 3}})}, "not Ready"),
        ({"get deploy chaos-controller-manager": "Error from server (NotFound)"}, "not readable"),
    ],
)
def test_container_kill_inject_refuses_before_touching_the_fault_when_chaos_mesh_is_not_ready(monkeypatch, broken, match):
    log, kubectl, _applied = _fake_aiopslab(monkeypatch)
    kubectl.responses.update({**CHAOS_READY, **broken})
    injector = AIOpsLabFaultInjector()
    injector.prepare()

    with pytest.raises(R1Error, match=match):
        injector.inject("container_kill")

    assert "inject:container_kill" not in log
    assert "container_kill-detection" not in injector._orchestrator.probs.requested  # never instantiated
    with pytest.raises(R1Error, match="no active"):
        injector.recover()  # nothing was marked active


def test_container_kill_inject_that_applied_no_podchaos_raises_and_is_still_recovered(monkeypatch):
    log, kubectl, _applied = _fake_aiopslab(monkeypatch)
    kubectl.responses.update(
        {**CHAOS_READY, "get podchaos -n test-hotel-reservation": 'Error from server (InternalError): failed calling webhook'}
    )
    injector = AIOpsLabFaultInjector()
    injector.prepare()

    with pytest.raises(R1Error, match="not readable after inject"):
        injector.inject("container_kill")
    kubectl.responses["get podchaos -n test-hotel-reservation"] = json.dumps({"items": []})
    injector.recover()
    assert log[-2:] == ["inject:container_kill", "recover:container_kill"]

    with pytest.raises(R1Error, match="applied no container-kill PodChaos"):
        injector.inject("container_kill")
    injector.recover()


def test_prepare_with_a_chaos_family_fails_before_the_clock_when_chaos_mesh_is_absent(monkeypatch):
    _log, kubectl, _applied = _fake_aiopslab(monkeypatch)
    injector = AIOpsLabFaultInjector(families=("container_kill",))
    with pytest.raises(R1Error, match="Chaos Mesh absent"):
        injector.prepare()
    assert injector._orchestrator.probs.requested == []  # AIOpsLab's own helm install never ran
    assert injector.cleanup() == {"cleanup": "skipped, prepare never deployed the app"}


def test_candidate_verdicts_never_count_toward_r1_totals():
    """The pass rule counts only the admitted, pre-registered families (MECHANISM_BY_FAMILY): a
    candidate's detection is data for its own admission, never a family or a mechanism toward the
    admission totals."""
    from scenarios.run_continuous_admission import CANDIDATE_MECHANISM_BY_FAMILY, MECHANISM_BY_FAMILY

    candidates_only = {family: "detected" for family in CANDIDATE_MECHANISM_BY_FAMILY}
    assert r1_admission_passed(candidates_only) is False
    # two admitted families on two mechanisms: one short of 3 families, and candidates do not fill it
    two_admitted = {"wrong_bin_usage": "detected", "redeploy_without_PV": "detected"}
    assert r1_admission_passed(two_admitted) is False
    assert r1_admission_passed({**two_admitted, **candidates_only}) is False
    # the two admitted mongo families are one mechanism; the remaining candidate (container_kill)
    # must not supply the missing family or the missing mechanism
    mongo = {"revoke_auth_mongodb": "detected", "user_unregistered_mongodb": "detected"}
    assert r1_admission_passed({**mongo, "container_kill": "detected"}) is False
    # control: the same gate passes on admitted families alone
    assert r1_admission_passed({**two_admitted, "misconfig_app_hotel_res": "detected"}) is True
    assert not set(CANDIDATE_MECHANISM_BY_FAMILY) & set(MECHANISM_BY_FAMILY)


# --- pre-round smoke --------------------------------------------------------------------------------

from scenarios.run_continuous_admission import run_smoke  # noqa: E402


class _CountingReceiver:
    def __init__(self, alerts):
        self.alerts = alerts
        self.value = 10

    def received_total(self):
        return self.value + self.alerts.false_alarms


def _smoke(
    tmp_path,
    *,
    policy=(0, "ok (a)\nok (b)"),
    silent=(),
    fail_inject=(),
    receiver_dead=False,
    model_tokens=None,
    **extra,
):
    clock = FakeClock()
    alerts = FakeAlerts()
    injector = FakeInjector(alerts, silent=silent, fail_inject=fail_inject)
    receiver = FakeReceiver(10, 10) if receiver_dead else _CountingReceiver(alerts)
    result = run_smoke(
        injector=injector,
        alerts=alerts,
        receiver=receiver,
        policy_check=lambda: policy,
        clock=clock,
        poll_seconds=1,
        alert_timeout_seconds=3,
        journal=Journal(tmp_path / "smoke.events.jsonl", tmp_path / "smoke.json", clock),
        model_tokens=model_tokens,
        **extra,
    )
    return result, injector


def test_smoke_passes_when_every_seam_works(tmp_path):
    result, injector = _smoke(tmp_path)
    assert result["passed"] is True, result
    assert injector.calls == ["prepare", "inject:user_unregistered_mongodb", "recover", "cleanup"]
    phases = [json.loads(line)["phase"] for line in (tmp_path / "smoke.events.jsonl").read_text().splitlines()]
    assert phases[-2:] == ["cleanup_done", "smoke_finished"]
    assert json.loads((tmp_path / "smoke.json").read_text())["passed"] is True


def test_smoke_stops_before_injecting_when_the_policy_scope_is_wrong(tmp_path):
    result, injector = _smoke(tmp_path, policy=(1, "FAIL (b): injector identity was denied"))
    assert result["passed"] is False and "policy scope" in result["failure"]
    assert injector.injected == [] and injector.calls[-1] == "cleanup"


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"silent": {"user_unregistered_mongodb"}}, "no symptom alert"),
        ({"fail_inject": {"user_unregistered_mongodb"}}, "denied exec"),
        ({"receiver_dead": True}, "never reached the WebhookReceiver"),
    ],
)
def test_smoke_fails_on_each_broken_seam_and_still_cleans_up(tmp_path, kwargs, reason):
    result, injector = _smoke(tmp_path, **kwargs)
    assert result["passed"] is False and reason in result["failure"]
    assert injector.calls[-1] == "cleanup"
    assert "recover" in injector.calls  # every inject attempt gets its own recover


def _counter(*values):
    """A token counter that returns `values` in order, then repeats the last one."""
    readings = list(values)

    def read():
        return readings.pop(0) if len(readings) > 1 else readings[0]

    return read


def test_smoke_with_the_model_on_passes_when_prompt_tokens_move(tmp_path):
    result, _ = _smoke(tmp_path, model_tokens=_counter(0.0, 0.0, 812.0))
    assert result["passed"] is True, result
    assert result["steps"]["model"] == {"prompt_tokens_before": 0.0, "prompt_tokens_after": 812.0}


def test_smoke_with_the_model_on_fails_when_the_agent_never_reaches_vllm(tmp_path):
    # Round 2 (2026-09-30): receiver took every webhook, every model call was connection refused, the
    # counter stayed at 0 and the old smoke passed anyway.
    result, injector = _smoke(tmp_path, model_tokens=_counter(0.0))
    assert result["passed"] is False
    assert "never reached the model" in result["failure"]
    assert result["steps"]["model"]["prompt_tokens_after"] == 0.0
    assert injector.calls[-1] == "cleanup"


def test_smoke_with_the_model_on_fails_when_the_counter_cannot_be_read(tmp_path):
    def unreadable():
        raise R1Error("vLLM /metrics unreadable")

    result, injector = _smoke(tmp_path, model_tokens=unreadable)
    assert result["passed"] is False and "unreadable" in result["failure"]
    assert injector.calls[-1] == "cleanup"


def test_vllm_token_gateway_reads_the_single_prompt_counter(monkeypatch):
    import io
    import urllib.request

    from scenarios.run_continuous_admission import HttpVllmTokenGateway

    body = (
        "# HELP vllm:prompt_tokens_total Number of prefill tokens processed.\n"
        'vllm:prompt_tokens_total{model_name="Qwen2.5-7B-Instruct"} 1234.0\n'
        'vllm:generation_tokens_total{model_name="Qwen2.5-7B-Instruct"} 56.0\n'
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(body.encode()))
    assert HttpVllmTokenGateway("http://127.0.0.1:8000/metrics").prompt_tokens_total() == 1234.0

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(b"other_metric 1\n"))
    with pytest.raises(R1Error, match="expected 1 series"):
        HttpVllmTokenGateway("http://127.0.0.1:8000/metrics").prompt_tokens_total()


def test_policy_check_times_out_instead_of_hanging(monkeypatch):
    import subprocess

    from scenarios.run_continuous_admission import _policy_check_script

    def hang(*_args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="verify", timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", hang)
    code, output = _policy_check_script("test-hotel-reservation", timeout_seconds=5)()
    assert code == 124 and "timed out" in output


def test_real_injector_does_not_retry_a_failed_recovery_from_cleanup(monkeypatch):
    log, _kubectl, _applied = _fake_aiopslab(monkeypatch)
    injector = AIOpsLabFaultInjector()
    injector.prepare()
    injector.inject("wrong_bin_usage")
    monkeypatch.setattr(
        injector._active, "recover_fault", lambda: (log.append("recover-attempt"), 1 / 0)
    )

    with pytest.raises(ZeroDivisionError):
        injector.recover()
    injector.cleanup()

    assert log.count("recover-attempt") == 1
    assert _FakeApp.cleanups == 1


def test_real_injector_cleanup_skips_the_app_when_prepare_failed_its_guards(monkeypatch):
    _fake_aiopslab(monkeypatch, namespaces={"wrong_bin_usage": "other-app"})
    injector = AIOpsLabFaultInjector()
    with pytest.raises(R1Error):
        injector.prepare()

    assert "skipped" in injector.cleanup()["cleanup"]
    assert _FakeApp.cleanups == 0


def test_smoke_fails_when_recovery_fails(tmp_path):
    clock = FakeClock()
    alerts = FakeAlerts()

    class BrokenRecovery(FakeInjector):
        def recover(self):
            self.calls.append("recover")
            raise RuntimeError("kubectl exec timed out")

    result = run_smoke(
        injector=BrokenRecovery(alerts),
        alerts=alerts,
        receiver=_CountingReceiver(alerts),
        policy_check=lambda: (0, "ok"),
        clock=clock,
        poll_seconds=1,
        alert_timeout_seconds=3,
    )
    assert result["passed"] is False and "recovery failed" in result["failure"]


def test_agent_rbac_reapply_is_bounded(monkeypatch):
    _log, _kubectl, applied = _fake_aiopslab(monkeypatch)
    seen = {}
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: seen.update(kw) or types.SimpleNamespace(returncode=0))
    injector = AIOpsLabFaultInjector()
    injector.prepare()
    injector._reapply_agent_rbac()
    assert seen["timeout"] > 0


# --- settle gate (2026-09-24 finding): the app must be settled before every injection --------------
#
# The scoped detectability pass scored misconfig_app_hotel_res "detected" off a geo pod an earlier run had left
# crashlooping: KubePodCrashLooping was already *pending* when the fault went in, so a
# firing-only baseline could not see it, and it fired 71s later as a "new" alert.

DIRTY = ["alert KubePodCrashLooping pending pod=geo-c47ff745-crs4h"]


def test_wait_for_app_settled_needs_a_stable_clean_stretch_and_never_returns_early():
    clock = FakeClock()
    alerts = FakeAlerts()
    alerts.unsettled = [DIRTY, DIRTY, []]  # dirty on the first two polls, clean from the third

    result = _wait_for_app_settled(alerts, clock, not_before=10, deadline=100, stable_seconds=5, poll_seconds=1)

    assert result.settled is True and result.signals == []
    # clean from t=2, stable for 5s -> t=7, but never before not_before=10
    assert result.decided_at == 10


def test_wait_for_app_settled_gives_up_at_the_deadline_with_the_last_dirty_reading():
    clock = FakeClock()
    alerts = FakeAlerts()
    alerts.always_unsettled = DIRTY

    result = _wait_for_app_settled(alerts, clock, not_before=0, deadline=30, stable_seconds=5, poll_seconds=1)

    assert result.settled is False
    assert result.decided_at == 30
    assert result.signals == result.last_dirty == DIRTY
    assert result.clean_for_seconds is None


def test_a_timeout_after_the_app_went_clean_does_not_report_a_stale_reading():
    # Clean at t=28, deadline t=30, stable needs 5s. The app is not dirty
    # at the deadline, it just has not been clean long enough; reporting DIRTY as the current
    # reading would send whoever debugs the round after an alert that already resolved.
    clock = FakeClock()
    alerts = FakeAlerts()
    alerts.unsettled = [DIRTY] * 28

    result = _wait_for_app_settled(alerts, clock, not_before=0, deadline=30, stable_seconds=5, poll_seconds=1)

    assert result.settled is False and result.decided_at == 30
    assert result.signals == []
    assert result.clean_for_seconds == 2
    assert result.last_dirty == DIRTY


def test_a_clean_stretch_interrupted_by_a_new_signal_starts_over():
    clock = FakeClock()
    alerts = FakeAlerts()
    alerts.unsettled = [[], [], DIRTY, []]  # clean at t=0,1, dirty at t=2, clean again from t=3

    result = _wait_for_app_settled(alerts, clock, not_before=0, deadline=100, stable_seconds=3, poll_seconds=1)

    assert result.settled is True and result.decided_at == 6  # 3 + 3, not 0 + 3


def test_injection_waits_for_the_app_to_settle_and_is_then_measured():
    clock = FakeClock()
    alerts = FakeAlerts()
    # Watch starts at t=2 (planned 12 minus the 10-minute lead, 1 s/min here). 15 dirty polls ->
    # clean from t=17, stable at t=18: past the planned start, inside the deadline (12 + 10).
    alerts.unsettled = [DIRTY] * 15
    injector = FakeInjector(alerts)
    schedule = FaultSchedule(
        seed=1,
        window_hours=24,
        entries=(
            ScheduleEntry(EntryKind.FAULT, 12, 1, FAMILIES[0]),
            ScheduleEntry(EntryKind.QUIET, 40, 360),
            *(
                ScheduleEntry(EntryKind.FAULT, 402 + 2 * i, 1, family)
                for i, family in enumerate(FAMILIES[1:] + FAMILIES)
            ),
        ),
    )
    result = _run(schedule, alerts=alerts, injector=injector, clock=clock)

    first = result.families[0]
    assert first.measured is True and first.injected is True
    assert first.actual_at == first.planned_at + 6  # the settle overran the planned start
    assert first.settle["settled"] is True
    assert result.passed is True


def test_injection_into_an_app_that_never_settles_is_skipped_and_not_measured():
    # The exact shape seen in the detectability pass: a symptom that outlives the settle budget. Here the app is dirty from
    # t=300 to t=390 (1 s per schedule minute): after the quiet window settled, before the faults.
    clock = FakeClock()

    class DirtyWindowAlerts(FakeAlerts):
        def unsettled_signals(self):
            return list(DIRTY) if 300 <= clock.time() < 390 else []

    alerts = DirtyWindowAlerts()
    injector = FakeInjector(alerts)
    result = _run(alerts=alerts, injector=injector, clock=clock)

    skipped = [f for f in result.families if not f.measured]
    # Faults planned at 374, 376, 378, 380: each one's deadline (planned + 10) runs out before the
    # app has been clean for a full minute (clean from 390, stable at 391).
    assert [f.start_minute for f in skipped] == [374, 376, 378, 380]
    assert all(not f.injected for f in skipped)
    assert all(f.settle["settled"] is False and f.settle["last_dirty"] == DIRTY for f in skipped)
    # 374/376/378 hit their deadline while still dirty; 380 hits it at t=390, already clean.
    assert [f.settle["signals"] for f in skipped] == [DIRTY, DIRTY, DIRTY, []]
    assert skipped[-1].settle["clean_for_seconds"] == 0
    assert len(injector.injected) == TOTAL_FAULT_EPISODES - len(skipped)
    assert result.quiet_window.alert_free is True  # the quiet window had settled before t=300
    assert result.passed is False  # an unmeasured episode keeps the round from passing
    assert result.aborted is None  # ... but it is not an abort: the round keeps going
    assert injector.calls[-1] == "cleanup"


def test_quiet_window_that_never_settles_is_not_admissible():
    clock = FakeClock()
    alerts = FakeAlerts()
    alerts.always_unsettled = DIRTY
    result = _run(alerts=alerts, clock=clock)

    quiet = result.quiet_window
    assert quiet.alert_free is False
    assert quiet.settle["settled"] is False and quiet.settle["signals"] == DIRTY
    assert quiet.settle["clean_for_seconds"] is None
    assert result.passed is False


def test_compute_family_verdicts_ignores_unmeasured_episodes():
    from scenarios.run_continuous_admission import FamilyResult

    unmeasured = FamilyResult("a", 10, False)
    unmeasured.measured = False
    only_unmeasured = FamilyResult("b", 20, False)
    only_unmeasured.measured = False

    verdicts = compute_family_verdicts([FamilyResult("a", 0, True), unmeasured, only_unmeasured])

    # "a" is judged on its one measured episode only; "b" has no evidence at all.
    assert verdicts == {"a": "detected", "b": "not_measured"}


def test_smoke_aborts_before_injecting_when_the_app_never_settles(tmp_path):
    clock = FakeClock()
    alerts = FakeAlerts()
    alerts.always_unsettled = DIRTY
    injector = FakeInjector(alerts)
    result = run_smoke(
        injector=injector,
        alerts=alerts,
        receiver=_CountingReceiver(alerts),
        policy_check=lambda: (0, "ok"),
        clock=clock,
        poll_seconds=1,
        alert_timeout_seconds=3,
        journal=Journal(tmp_path / "smoke.events.jsonl", tmp_path / "smoke.json", clock),
    )

    assert result["passed"] is False and "never settled" in result["failure"]
    assert injector.injected == [] and injector.calls[-1] == "cleanup"


def test_http_unsettled_signals_reads_pending_alerts_and_not_ready_pods_in_the_namespace(doubles):
    _Doubles.prom_alerts = [
        {"metric": {"alertname": "KubePodCrashLooping", "alertstate": "pending", "pod": "geo-1", "namespace": "app"}},
        {"metric": {"alertname": "Watchdog", "alertstate": "firing"}},
    ]
    _Doubles.not_ready = [{"metric": {"pod": "geo-2", "namespace": "app"}}]
    alerts = HttpAlertGateway(doubles, doubles, namespace="app")

    assert alerts.unsettled_signals() == [
        "alert KubePodCrashLooping pending pod=geo-1",
        "pod_not_ready geo-2",
    ]
    assert _Doubles.queries[0] == 'ALERTS{namespace="app"}'
    assert 'namespace="app",condition="false"' in _Doubles.queries[1]
    assert 'phase="Succeeded"' in _Doubles.queries[1]  # a completed job pod is not "not ready"


def test_http_unsettled_signals_is_empty_for_a_clean_namespace(doubles):
    alerts = HttpAlertGateway(doubles, doubles, namespace="app")
    assert alerts.unsettled_signals() == []


# --- post-action probe ------------------------------------------------------------------------------


class ProbingInjector(FakeInjector):
    def __init__(self, alerts, clock, *, fixed=False, probe_raises=False):
        super().__init__(alerts)
        self.clock = clock
        self.fixed = fixed
        self.probe_raises = probe_raises
        self.probe_times = []

    def probe(self):
        self.calls.append("probe")
        self.probe_times.append(self.clock.time())
        if self.probe_raises:
            raise RuntimeError("kubectl went away")
        return {"root_cause_fixed": self.fixed, "probe_detail": "fake"}


def _run_with_probe(injector, alerts, clock, journal=None):
    return run_r1(
        _r1_schedule(),
        injector=injector,
        alerts=alerts,
        receiver=FakeReceiver(100, 105),
        clock=clock,
        seconds_per_minute=1,
        poll_seconds=1,
        alert_timeout_seconds=2,
        journal=journal,
        post_action_probe=True,
    )


def test_post_action_probe_runs_at_the_planned_end_right_before_recover(tmp_path):
    clock, alerts = FakeClock(), FakeAlerts()
    injector = ProbingInjector(alerts, clock, fixed=True)
    events = tmp_path / "r.events.jsonl"

    result = _run_with_probe(injector, alerts, clock, Journal(events, None, clock))

    faults = [e for e in _r1_schedule().entries if e.kind is EntryKind.FAULT]
    assert len(injector.probe_times) == len(faults)
    # started_at is the clock after prepare (0 here), so the planned end is end_minute seconds.
    assert injector.probe_times == [float(e.end_minute) for e in faults]
    for i, call in enumerate(injector.calls):
        if call == "probe":
            assert injector.calls[i + 1] == "recover"
    assert all(item.probe == {"root_cause_fixed": True, "probe_detail": "fake"} for item in result.families)
    phases = [json.loads(line)["phase"] for line in events.read_text().splitlines()]
    assert phases.count("probe_post_action") == len(faults)


def test_post_action_probe_off_is_the_r1_behavior():
    clock, alerts = FakeClock(), FakeAlerts()
    injector = ProbingInjector(alerts, clock)

    _run(alerts=alerts, injector=injector, clock=clock)

    assert "probe" not in injector.calls


def test_a_raising_probe_is_recorded_as_not_measured_and_the_round_goes_on():
    clock, alerts = FakeClock(), FakeAlerts()
    injector = ProbingInjector(alerts, clock, probe_raises=True)

    result = _run_with_probe(injector, alerts, clock)

    assert result.aborted is None
    first = result.families[0]
    assert first.probe["root_cause_fixed"] is None
    assert "kubectl went away" in first.probe["probe_error"]
    assert first.recovered is True


def test_collect_new_alerts_keeps_watching_after_the_first_alert():
    from scenarios.run_continuous_admission import _collect_new_alerts

    clock = FakeClock()

    class ScriptedAlerts:
        def firing_alerts(self):
            if clock.time() >= 5:
                return {"old": {}, "a": {"alertname": "A"}, "b": {"alertname": "B"}}
            if clock.time() >= 2:
                return {"old": {}, "a": {"alertname": "A"}}
            return {"old": {}}

    seen, first_at = _collect_new_alerts(ScriptedAlerts(), {"old"}, deadline=10, clock=clock, poll_seconds=1)

    assert set(seen) == {"a", "b"}
    assert first_at == 2


def test_aiopslab_injector_probe_routes_the_active_fault_to_its_probe(monkeypatch):
    import scenarios.run_continuous_admission as rca

    seen = {}

    def fake_run_probe(family, kubectl, namespace, service):
        seen.update(family=family, kubectl=kubectl, namespace=namespace, service=service)
        return {"root_cause_fixed": False, "probe_detail": "still broken"}

    monkeypatch.setattr(rca, "run_probe", fake_run_probe)
    injector = rca.AIOpsLabFaultInjector.__new__(rca.AIOpsLabFaultInjector)
    injector._active = types.SimpleNamespace(kubectl="kube", faulty_service="frontend")
    injector._active_family = FAMILIES[0]
    injector._namespace = "hotel"

    assert injector.probe() == {"root_cause_fixed": False, "probe_detail": "still broken"}
    assert seen == {"family": FAMILIES[0], "kubectl": "kube", "namespace": "hotel", "service": "frontend"}


def test_aiopslab_injector_probe_without_active_fault_is_not_measured():
    import scenarios.run_continuous_admission as rca

    injector = rca.AIOpsLabFaultInjector.__new__(rca.AIOpsLabFaultInjector)
    injector._active = None

    assert injector.probe()["root_cause_fixed"] is None



def _dry_run_schedule():
    return FaultSchedule(
        seed=623,
        window_hours=4,
        entries=(
            ScheduleEntry(EntryKind.FAULT, 0, 1, "assign_to_non_existent_node_hotel_res"),
            ScheduleEntry(EntryKind.QUIET, 2, 3),
            ScheduleEntry(EntryKind.FAULT, 6, 1, "wrong_bin_usage"),
        ),
    )


def test_dry_run_accepts_a_short_schedule_that_r1_refuses():
    with pytest.raises(R1Error, match="missing"):
        validate_schedule_for_r1(_dry_run_schedule())
    validate_schedule_for_dry_run(_dry_run_schedule())


def test_dry_run_still_refuses_a_family_without_probe_and_an_empty_schedule():
    unknown = FaultSchedule(seed=1, window_hours=1, entries=(ScheduleEntry(EntryKind.FAULT, 0, 1, "nope"),))
    with pytest.raises(R1Error, match="without an admitted probe"):
        validate_schedule_for_dry_run(unknown)
    quiet = FaultSchedule(seed=1, window_hours=1, entries=(ScheduleEntry(EntryKind.QUIET, 0, 3),))
    with pytest.raises(R1Error, match="at least one fault"):
        validate_schedule_for_dry_run(quiet)


def test_dry_run_runs_end_to_end_but_never_passes(tmp_path):
    alerts = FakeAlerts()
    out = tmp_path / "r.json"
    result = run_r1(
        _dry_run_schedule(),
        injector=FakeInjector(alerts),
        alerts=alerts,
        receiver=FakeReceiver(100, 102),
        clock=FakeClock(),
        seconds_per_minute=1,
        poll_seconds=1,
        alert_timeout_seconds=2,
        journal=Journal(tmp_path / "r.events.jsonl", out, FakeClock()),
        dry_run=True,
    )

    assert result.aborted is None
    assert [item.family for item in result.families] == [
        "assign_to_non_existent_node_hotel_res",
        "wrong_bin_usage",
    ]
    assert result.dry_run is True
    assert result.passed is False
    state = json.loads(out.read_text(encoding="utf-8"))
    assert state["dry_run"] is True and state["passed"] is False


def test_real_r1_result_is_not_marked_dry_run():
    assert _run().dry_run is False


def test_dry_run_cli_needs_a_committed_schedule_and_no_quiet_only(tmp_path):
    base = ["--alertmanager-url", "x", "--prometheus-url", "x", "--receiver-url", "x",
            "--out", str(tmp_path / "r.json"), "--dry-run"]
    with pytest.raises(R1Error, match="--dry-run"):
        main([*base, "--seed", "623"])
    with pytest.raises(R1Error, match="--dry-run"):
        main([*base, "--quiet-only"])


def test_dry_run_and_quiet_only_are_exclusive_in_run_r1():
    alerts = FakeAlerts()
    with pytest.raises(R1Error, match="separate modes"):
        run_r1(_dry_run_schedule(), injector=FakeInjector(alerts), alerts=alerts, receiver=FakeReceiver(0, 0),
               clock=FakeClock(), quiet_only=True, dry_run=True)


# --- comparator arm smoke guard ------------------------------------------------------------------
# The API arm has no vLLM to read. Its proof that episodes reached the model is the agent pod's
# usage JSONL gaining request lines, read with kubectl exec; same rule as the vLLM counter: it
# must move, and an unreadable count fails the smoke.


def test_comparator_smoke_passes_when_usage_lines_move(tmp_path):
    from scenarios.run_continuous_admission import COMPARATOR_USAGE_LINES_SIGNAL

    result, _ = _smoke(tmp_path, model_tokens=_counter(0.0, 0.0, 2.0), model_signal=COMPARATOR_USAGE_LINES_SIGNAL)
    assert result["passed"] is True, result
    assert result["steps"]["model"] == {
        "prompt_tokens_before": 0.0,
        "prompt_tokens_after": 2.0,
        "signal": COMPARATOR_USAGE_LINES_SIGNAL,
    }


def test_comparator_smoke_fails_when_the_api_arm_never_calls_the_api(tmp_path):
    from scenarios.run_continuous_admission import COMPARATOR_USAGE_LINES_SIGNAL

    result, injector = _smoke(tmp_path, model_tokens=_counter(4.0), model_signal=COMPARATOR_USAGE_LINES_SIGNAL)
    assert result["passed"] is False
    assert "never reached the model" in result["failure"]
    assert COMPARATOR_USAGE_LINES_SIGNAL in result["failure"]
    assert "agent -> API" in result["failure"] and "vLLM" not in result["failure"]
    assert injector.calls[-1] == "cleanup"


def test_vllm_smoke_failure_text_is_unchanged(tmp_path):
    result, _ = _smoke(tmp_path, model_tokens=_counter(0.0))
    assert "vLLM prompt_tokens_total stayed at 0.0 (agent -> vLLM path refused or unrouted" in result["failure"]
    assert "signal" not in result["steps"]["model"]


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_comparator_usage_gateway_counts_lines_through_kubectl_exec():
    from scenarios.run_continuous_admission import KubectlComparatorUsageGateway

    seen = {}

    def run(argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs
        return _Completed(stdout="3\n")

    assert KubectlComparatorUsageGateway(run=run).request_lines_total() == 3.0
    argv = seen["argv"]
    assert argv[:6] == ["kubectl", "-n", "sslm-agent", "exec", "deploy/sslm-agent", "--"]
    assert argv[6:8] == ["python3", "-c"]
    # The path is read from the pod's own env, never duplicated in the driver.
    assert "SSLM_COMPARATOR_USAGE_LOG" in argv[8] and "/var/lib/sslm" not in argv[8]
    assert seen["kwargs"]["timeout"] > 0


@pytest.mark.parametrize(
    "outcome, needle",
    [
        (_Completed(returncode=1, stderr="error: pod not running"), "rc=1"),
        (_Completed(stdout="not-a-number"), "not an integer"),
    ],
)
def test_comparator_usage_gateway_raises_on_an_unreadable_count(outcome, needle):
    from scenarios.run_continuous_admission import KubectlComparatorUsageGateway

    with pytest.raises(R1Error, match=needle):
        KubectlComparatorUsageGateway(run=lambda *_a, **_k: outcome).request_lines_total()


def test_comparator_usage_gateway_raises_on_a_hung_exec():
    import subprocess

    from scenarios.run_continuous_admission import KubectlComparatorUsageGateway

    def hung(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    with pytest.raises(R1Error, match="unreadable"):
        KubectlComparatorUsageGateway(run=hung).request_lines_total()


def test_the_in_pod_counter_script_counts_request_lines(tmp_path):
    import os
    import subprocess

    from scenarios.run_continuous_admission import _COUNT_USAGE_LINES

    log = tmp_path / "comparator-usage.jsonl"

    def count(env_extra):
        env = {k: v for k, v in os.environ.items() if k != "SSLM_COMPARATOR_USAGE_LOG"}
        env.update(env_extra)
        return subprocess.run([sys.executable, "-c", _COUNT_USAGE_LINES], env=env, capture_output=True, text=True)

    # No paid response yet: the file does not exist, which is 0, not an error.
    missing = count({"SSLM_COMPARATOR_USAGE_LOG": str(log)})
    assert missing.returncode == 0 and missing.stdout.strip() == "0"
    log.write_text('{"request_id": "a"}\n\n{"request_id": "b"}\n', encoding="utf-8")
    two = count({"SSLM_COMPARATOR_USAGE_LOG": str(log)})
    assert two.returncode == 0 and two.stdout.strip() == "2"
    # A pod without the variable is an error, never a quiet 0.
    unset = count({})
    assert unset.returncode != 0 and "SSLM_COMPARATOR_USAGE_LOG" in unset.stderr


def test_comparator_usage_cli_wires_the_usage_gateway():
    import argparse

    from scenarios.run_continuous_admission import (
        COMPARATOR_USAGE_LINES_SIGNAL,
        HttpVllmTokenGateway,
        KubectlComparatorUsageGateway,
        _smoke_model_signal,
    )

    comparator = _smoke_model_signal(argparse.Namespace(comparator_usage=True, vllm_metrics_url=None))
    assert comparator["model_signal"] == COMPARATOR_USAGE_LINES_SIGNAL
    assert isinstance(comparator["model_tokens"].__self__, KubectlComparatorUsageGateway)

    vllm = _smoke_model_signal(argparse.Namespace(comparator_usage=False, vllm_metrics_url="http://h/metrics"))
    assert set(vllm) == {"model_tokens"}
    assert isinstance(vllm["model_tokens"].__self__, HttpVllmTokenGateway)
    assert _smoke_model_signal(argparse.Namespace(comparator_usage=False, vllm_metrics_url=None)) == {
        "model_tokens": None
    }


def test_comparator_usage_cli_refuses_both_arms_or_a_non_smoke_run(tmp_path):
    base = ["--alertmanager-url", "x", "--prometheus-url", "x", "--receiver-url", "x", "--out", str(tmp_path / "r.json")]
    with pytest.raises(R1Error, match="two arms"):
        main([*base, "--smoke", "--app-namespace", "ns", "--comparator-usage", "--vllm-metrics-url", "http://h/m"])
    with pytest.raises(R1Error, match="only applies to --smoke"):
        main([*base, "--seed", "623", "--comparator-usage"])
