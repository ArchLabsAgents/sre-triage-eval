"""harness/availability.py: 1 - downtime/window, downtime = heartbeat missing for > 2 intervals."""

from __future__ import annotations

from harness.availability import compute_availability

START = 1_790_000_000.0
STEP = 30.0
END = START + 100 * STEP  # 101 grid points


def _series(*, missing: range = range(0), zeros: range = range(0)):
    return [[(START + i * STEP, 0.0 if i in zeros else 1.0) for i in range(101) if i not in missing]]


def test_full_heartbeat_is_fully_available():
    result = compute_availability(_series(), START, END)

    assert result.measured and result.availability == 1.0 and result.outages == 0


def test_gap_of_exactly_two_intervals_is_jitter_not_downtime():
    result = compute_availability(_series(missing=range(10, 12)), START, END)

    assert result.availability == 1.0


def test_gap_longer_than_two_intervals_is_downtime_for_its_whole_length():
    result = compute_availability(_series(missing=range(10, 20)), START, END)

    assert (result.outages, result.downtime_seconds) == (1, 10 * STEP)
    assert result.availability == 1.0 - (10 * STEP) / (101 * STEP)


def test_not_ready_zero_counts_like_a_missing_sample():
    result = compute_availability(_series(zeros=range(50, 56)), START, END)

    assert result.outages == 1 and result.downtime_seconds == 6 * STEP


def test_outage_running_to_the_end_of_the_window_is_counted():
    result = compute_availability(_series(missing=range(90, 101)), START, END)

    assert result.outages == 1 and result.downtime_seconds == 11 * STEP


def test_a_second_pod_covering_the_gap_keeps_the_agent_up():
    first = _series(missing=range(10, 20))[0]
    second = [(START + i * STEP, 1.0) for i in range(10, 20)]

    result = compute_availability([first, second], START, END)

    assert result.availability == 1.0


def test_no_series_is_not_measured_never_a_pass():
    result = compute_availability([], START, END)

    assert not result.measured and result.availability is None and "no ready sample" in result.reason


def test_empty_window_is_not_measured():
    assert not compute_availability(_series(), START, START).measured
