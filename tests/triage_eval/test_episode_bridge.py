"""harness/episode_bridge.py joins a round journal with its TriageRecords."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from harness.episode_bridge import (
    FALSE_ALARM_FAMILY,
    HALF_WRITTEN,
    NOT_REACHED,
    NOT_SETTLED,
    BridgeError,
    build_episodes,
    main,
)
from harness.scorer import KIND_FALSE_ALARM, KIND_FAULT, score

T0 = 1_790_000_000.0  # clock start
SPM = 60.0
ROUND_ID = "seed7-20260928T000000Z"
FP = "fp-checkout"
LABELS = {"alertname": "PodNotReady", "namespace": "hotel", "pod": "frontend-1"}


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


def _at(minute: float) -> float:
    return T0 + minute * SPM


def _clock(**overrides):
    return {"phase": "clock_started", "t": T0, "journal_version": 2, "rodada_id": ROUND_ID, "seed": 7,
            "seconds_per_minute": SPM, "app_namespace": "hotel", **overrides}


def _fault(family, start, end, *, alert=True, error=None, done=True):
    events = [
        {"phase": "inject_start", "t": _at(start), "family": family, "start_minute": start, "end_minute": end},
        {"phase": "inject_done", "t": _at(start) + 5, "family": family},
    ]
    if alert:
        events.append({"phase": "alert_observed", "t": _at(start) + 120, "family": family,
                       "alerts": [{"fingerprint": FP, "alertname": LABELS["alertname"], "labels": LABELS}]})
    else:
        events.append({"phase": "alert_missing", "t": _at(start) + 600, "family": family, "alerts": []})
    if done:
        events.append({"phase": "family_done", "t": _at(start) + 700, "family": family, "injected": error is None,
                       "error": error, "start_minute": start})
    return events


def _record(episode_id, t_alarme, *, delay=60.0, fingerprint=FP, cause_class="app_image_misconfig"):
    return {"episode_id": episode_id, "alert_fingerprint": fingerprint, "t_alarme": _iso(t_alarme),
            "t_record": _iso(t_alarme + delay), "causa_provavel": {"classe": cause_class, "detalhe": "d"},
            "decisao": "escalar"}


def test_record_inside_the_fault_window_becomes_that_episode():
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]
    record = _record("ep-a", _at(33))

    result = build_episodes(journal, [record])

    (ep,) = result.episodes
    assert ep.episode_id == f"{ROUND_ID}-f30"
    assert (ep.family, ep.round_id, ep.kind) == ("misconfig_app_hotel_res", ROUND_ID, KIND_FAULT)
    assert ep.record is record
    assert ep.alert_labels == LABELS  # the journal saw this exact fingerprint
    assert ep.alert_fired is True
    assert ep.root_cause_fixed_post_action is None  # this journal has no post-action probe
    assert ep.not_measured is None


def test_window_runs_to_the_planned_end_not_the_recover():
    """recover() runs as soon as the first alert shows, well before end_minute; a record whose
    alarm fires after that is still this fault's (nothing else is injected in the window)."""
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]

    result = build_episodes(journal, [_record("ep-a", _at(55))])

    assert result.episodes[0].record is not None
    assert result.orphan_records == []


def test_record_from_a_different_alert_keeps_record_but_not_labels():
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]

    result = build_episodes(journal, [_record("ep-a", _at(33), fingerprint="fp-other")])

    assert result.episodes[0].record is not None
    assert result.episodes[0].alert_labels is None


def test_fault_without_alert_and_without_record_is_a_scored_miss():
    journal = [_clock(), *_fault("wrong_bin_usage", 30, 60, alert=False)]

    (ep,) = build_episodes(journal, []).episodes

    assert ep.record is None and ep.alert_fired is False and ep.not_measured is None


def test_record_outside_every_window_is_reported_as_orphan():
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]
    stray = _record("ep-stray", _at(120))

    result = build_episodes(journal, [stray])

    assert result.episodes[0].record is None
    assert result.orphan_records == [stray]


def test_second_record_in_a_window_is_extra_and_the_first_to_land_wins():
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]
    first, second = _record("ep-1", _at(33)), _record("ep-2", _at(40))

    result = build_episodes(journal, [second, first])

    assert result.episodes[0].record is first
    assert result.extra_records == [second]


def test_late_record_for_the_first_alarm_does_not_hide_one_that_landed_in_time():
    """A false negative is "no record within 15 min of the FIRST alarm": if a second alert's record
    landed inside that, the fault had a timely record and is not a miss."""
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]
    late = _record("ep-late", _at(31), delay=20 * 60)  # alarm first, record lands at minute 51
    timely = _record("ep-timely", _at(33), delay=3 * 60)  # lands at minute 36

    result = build_episodes(journal, [late, timely])

    assert result.episodes[0].record is timely
    assert result.late_records == [late]
    assert result.extra_records == []


def test_timely_means_within_15_min_of_the_first_alarm_not_of_its_own():
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]
    first = _record("ep-first", _at(31), delay=20 * 60)  # first alarm at 31, lands at 51
    second = _record("ep-second", _at(46), delay=60)  # its own delay is short, but lands at 47 > 31+15

    result = build_episodes(journal, [first, second])

    assert result.episodes[0].record is None
    assert result.late_records == [second, first]  # in arrival order


@pytest.mark.parametrize(
    "bad",
    [{"episode_id": "ep-no-times"}, {"episode_id": "ep-bad", "t_alarme": "not-a-date", "t_record": "x"}, "{truncated"],
)
def test_malformed_record_is_reported_and_does_not_abort_the_join(bad):
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]
    good = _record("ep-a", _at(33))

    result = build_episodes(journal, [bad, good])

    assert result.episodes[0].record is good
    assert result.malformed_records == [bad]


def test_record_after_the_15_minute_deadline_counts_as_no_record():
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]
    late = _record("ep-late", _at(33), delay=16 * 60)

    result = build_episodes(journal, [late])

    assert result.episodes[0].record is None
    assert result.late_records == [late]


def test_unsettled_entry_is_not_measured():
    journal = [_clock(), {"phase": "entry_not_measured", "t": _at(30), "family": "redeploy_without_PV",
                          "start_minute": 30, "end_minute": 60, "settled": False}]

    (ep,) = build_episodes(journal, []).episodes

    assert ep.not_measured == NOT_SETTLED


def test_harness_error_makes_the_entry_not_measured_and_it_claims_no_record():
    journal = [_clock(), *_fault("revoke_auth_mongodb", 30, 60, error="inject RuntimeError: denied exec")]
    record = _record("ep-a", _at(33))

    result = build_episodes(journal, [record])

    assert result.episodes[0].not_measured == "harness: inject RuntimeError: denied exec"
    assert result.episodes[0].record is None
    assert result.orphan_records == [record]


def test_half_written_entry_is_not_measured():
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60, done=False)]

    (ep,) = build_episodes(journal, []).episodes

    assert ep.not_measured == HALF_WRITTEN


def test_schedule_entries_the_journal_never_reached_are_not_measured_not_missing():
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]
    schedule = {"seed": 7, "window_hours": 24, "entries": [
        {"kind": "fault", "start_minute": 30, "duration_minutes": 30, "fault_family": "misconfig_app_hotel_res"},
        {"kind": "quiet", "start_minute": 90, "duration_minutes": 360, "fault_family": None},
        {"kind": "fault", "start_minute": 500, "duration_minutes": 30, "fault_family": "wrong_bin_usage"},
        {"kind": "false_alarm", "start_minute": 600, "duration_minutes": 5, "fault_family": "wrong_bin_usage"},
    ]}

    episodes = build_episodes(journal, [], schedule=schedule).episodes

    assert [(e.kind, e.family, e.not_measured) for e in episodes] == [
        (KIND_FAULT, "misconfig_app_hotel_res", None),
        (KIND_FAULT, "wrong_bin_usage", NOT_REACHED),
        (KIND_FALSE_ALARM, FALSE_ALARM_FAMILY, NOT_REACHED),
    ]


def test_false_alarm_window_claims_its_record():
    journal = [_clock(), {"phase": "false_alarm_sent", "t": _at(10), "start_minute": 10, "end_minute": 15,
                          "duration_minutes": 5, "alertname": "SslmSyntheticControl"}]
    record = _record("ep-fa", _at(11), cause_class="falso_positivo")

    (ep,) = build_episodes(journal, [record]).episodes

    assert (ep.kind, ep.family) == (KIND_FALSE_ALARM, FALSE_ALARM_FAMILY)
    assert ep.record is record
    assert ep.alert_fired is None


def test_false_alarm_record_stamped_before_the_journal_event_is_still_paired():
    """Regression (seen live): send_false_alarm stamps startsAt (the record's t_alarme) before the POST and
    the driver journals false_alarm_sent after it, so t_alarme precedes the event's t. The record
    must not fall out of the window (it used to come back an orphan, episode record null)."""
    journal = [_clock(), {"phase": "false_alarm_sent", "t": _at(10) + 0.4, "start_minute": 10, "end_minute": 15,
                          "duration_minutes": 5, "alertname": "SslmSyntheticControl"}]
    record = _record("ep-fa", _at(10) + 0.1, cause_class="falso_positivo")

    result = build_episodes(journal, [record])

    (ep,) = result.episodes
    assert ep.record is record
    assert result.orphan_records == []


@pytest.mark.parametrize("journal", [[], [_clock(journal_version=1)], [_clock(journal_version=None)]])
def test_journal_without_a_v2_clock_start_is_refused(journal):
    with pytest.raises(BridgeError):
        build_episodes(journal, [])


def test_real_driver_journal_joins_and_scores(tmp_path):
    """Reads the journal run_r1 actually writes, not a hand copy of its shape, so a field rename
    on the driver side breaks here instead of in a live round."""
    from test_run_continuous_admission import FakeAlerts, FakeClock, FakeInjector, _run

    from scenarios.run_continuous_admission import Journal

    events_path = tmp_path / "r1.events.jsonl"
    clock = FakeClock()
    alerts = FakeAlerts()
    _run(alerts=alerts, injector=FakeInjector(alerts), clock=clock, journal=Journal(events_path, None, clock))
    journal = [json.loads(line) for line in events_path.read_text().splitlines()]
    first_inject = next(e for e in journal if e["phase"] == "inject_start")
    started = next(e for e in journal if e["phase"] == "clock_started")
    planned_end = started["t"] + first_inject["end_minute"] * started["seconds_per_minute"]
    mid_window = (first_inject["t"] + planned_end) / 2  # _run compresses minutes, so no fixed offset

    result = build_episodes(journal, [_record("ep-real", mid_window, delay=0.0)])

    faults = [e for e in result.episodes if e.kind == KIND_FAULT]
    assert len(faults) == sum(1 for e in journal if e["phase"] == "inject_start")
    assert faults[0].record is not None and faults[0].family == first_inject["family"]
    assert all(e.round_id == result.round_id for e in result.episodes)
    report = score(result.episodes, resamples=50)
    assert report  # the scorer accepts what the bridge builds


def test_cli_writes_the_join(tmp_path):
    journal_path, records_path, out = tmp_path / "j.jsonl", tmp_path / "r.jsonl", tmp_path / "out" / "e.json"
    journal_path.write_text("\n".join(json.dumps(e) for e in [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]))
    records_path.write_text(json.dumps(_record("ep-a", _at(33))) + "\n" + '{"episode_id": "ep-trunc')  # killed mid-write

    assert main(["--journal", str(journal_path), "--records", str(records_path), "--out", str(out)]) == 0

    written = json.loads(out.read_text())
    assert written["rodada_id"] == ROUND_ID
    assert written["episodes"][0]["record"]["episode_id"] == "ep-a"
    assert len(written["malformed_records"]) == 1


# --- post-action probe ---------------------------------------------------------------------------


def _probe(start, *, fixed, at_minute):
    return {"phase": "probe_post_action", "t": _at(at_minute), "family": "misconfig_app_hotel_res",
            "root_cause_fixed": fixed, "probe_detail": "d"}


def _fault_with_probe(fixed, *, probe_minute=60):
    events = _fault("misconfig_app_hotel_res", 30, 60)
    done = events.pop()  # family_done goes after the probe, like run_r1 writes it
    return [*events, _probe(30, fixed=fixed, at_minute=probe_minute), done]


@pytest.mark.parametrize("fixed", [True, False])
def test_probe_reading_becomes_root_cause_fixed_post_action(fixed):
    journal = [_clock(journal_version=3), *_fault_with_probe(fixed)]

    (ep,) = build_episodes(journal, [_record("ep-a", _at(33))]).episodes

    assert ep.root_cause_fixed_post_action is fixed


def test_record_that_landed_after_the_probe_gets_no_mitigation_verdict():
    journal = [_clock(journal_version=3), *_fault_with_probe(True, probe_minute=40)]
    record = _record("ep-a", _at(38), delay=5 * 60)  # lands at minute 43, probe read at 40

    (ep,) = build_episodes(journal, [record]).episodes

    assert ep.record is record
    assert ep.root_cause_fixed_post_action is None


def test_probe_reading_without_any_record_is_kept():
    """No action, symptom still there: that False is exactly the false-negative evidence."""
    journal = [_clock(journal_version=3), *_fault_with_probe(False)]

    (ep,) = build_episodes(journal, []).episodes

    assert ep.record is None and ep.root_cause_fixed_post_action is False


def test_real_driver_journal_with_probe_joins(tmp_path):
    from test_run_continuous_admission import FakeAlerts, FakeClock, ProbingInjector, _run_with_probe

    from scenarios.run_continuous_admission import Journal

    events_path = tmp_path / "r.events.jsonl"
    clock, alerts = FakeClock(), FakeAlerts()
    _run_with_probe(ProbingInjector(alerts, clock, fixed=True), alerts, clock, Journal(events_path, None, clock))
    journal = [json.loads(line) for line in events_path.read_text().splitlines()]

    episodes = build_episodes(journal, []).episodes

    assert episodes and all(e.root_cause_fixed_post_action is True for e in episodes if e.kind == KIND_FAULT)


def test_false_alarm_and_fault_at_the_same_minute_get_distinct_ids():
    journal = [_clock(), {"phase": "false_alarm_sent", "t": _at(30), "start_minute": 30, "end_minute": 35,
                          "duration_minutes": 5, "alertname": "SslmSyntheticControl"},
               *_fault("misconfig_app_hotel_res", 30, 60)]

    ids = [e.episode_id for e in build_episodes(journal, []).episodes]

    assert len(set(ids)) == 2


def test_duplicated_record_line_is_two_records_not_one():
    journal = [_clock(), *_fault("misconfig_app_hotel_res", 30, 60)]
    record = _record("ep-a", _at(33))
    duplicate = dict(record)  # equal by ==, a separate line in the JSONL

    result = build_episodes(journal, [record, duplicate])

    assert result.episodes[0].record is record
    assert result.extra_records == [duplicate]
    assert result.late_records == []


def test_cli_refuses_a_corrupt_journal_with_a_bridge_error(tmp_path):
    journal_path, records_path = tmp_path / "j.jsonl", tmp_path / "r.jsonl"
    journal_path.write_text(json.dumps(_clock()) + "\n{broken")
    records_path.write_text("")

    with pytest.raises(BridgeError, match="unparsable"):
        main(["--journal", str(journal_path), "--records", str(records_path), "--out", str(tmp_path / "o.json")])

