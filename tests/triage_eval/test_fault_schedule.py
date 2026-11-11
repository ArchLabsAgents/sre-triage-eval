"""Tests for scenarios/fault_schedule.py: determinism, one fault at a time, quiet windows and
false alarms, and the committed schedules regenerating from their declared seeds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scenarios.fault_schedule import (
    MIN_FAULT_EPISODES_PER_ROUND,
    MIN_GAP_MINUTES,
    MIN_QUIET_HOURS_PER_ROUND,
    EntryKind,
    FaultSchedule,
    generate_schedule,
    short_schedule_from,
)


def test_same_seed_is_byte_for_byte_deterministic():
    """The API comparator arm reuses the seeds of base rounds, episode by episode, for a paired
    test; that only works if regenerating from the same seed reproduces the exact same
    schedule."""
    a = generate_schedule(seed=42)
    b = generate_schedule(seed=42)
    assert a.to_json() == b.to_json()


def test_committed_r1_admission_schedule_matches_its_declared_seed():
    """The admission ground-truth artifact remains reproducible from its pre-registered seed.

    When three candidate families were promoted (2026-09-25), the committed schedule moved from
    seed 128 (generated before the quota existed, invalid under the >= 2 injections/family quota) to seed 600, the first admissible seed generate_r1_schedule
    found for the eight admitted families -- so reproducing it needs the same
    min_injections_per_family=2 the search used, not plain generate_schedule.
    """
    schedule_path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "triage-eval"
        / "scenarios"
        / "schedules"
        / "r1-admission-seed-600.json"
    )

    assert json.loads(schedule_path.read_text(encoding="utf-8")) == generate_schedule(
        seed=600, min_injections_per_family=2
    ).to_json()


def test_different_seeds_produce_different_schedules():
    a = generate_schedule(seed=1)
    b = generate_schedule(seed=2)
    assert a.to_json() != b.to_json()


def test_default_24h_round_meets_minimums():
    schedule = generate_schedule(seed=7)
    assert len(schedule.fault_episodes) >= MIN_FAULT_EPISODES_PER_ROUND
    assert schedule.quiet_minutes_total >= MIN_QUIET_HOURS_PER_ROUND * 60
    assert 1 <= len(schedule.false_alarms) <= 2


def test_one_fault_at_a_time_no_overlaps():
    """One fault at a time."""
    schedule = generate_schedule(seed=13)
    ordered = sorted(schedule.entries, key=lambda e: e.start_minute)
    for prev, nxt in zip(ordered, ordered[1:]):
        assert prev.end_minute <= nxt.start_minute, f"{prev} overlaps {nxt}"


def test_minimum_gap_after_a_fault_before_the_next_entry():
    schedule = generate_schedule(seed=99)
    ordered = sorted(schedule.entries, key=lambda e: e.start_minute)
    for prev, nxt in zip(ordered, ordered[1:]):
        if prev.kind is EntryKind.FAULT:
            gap = nxt.start_minute - prev.end_minute
            assert gap >= MIN_GAP_MINUTES, f"gap {gap} after fault {prev} before {nxt} is under {MIN_GAP_MINUTES}"


def test_quiet_windows_within_declared_range():
    schedule = generate_schedule(seed=21)
    for entry in schedule.quiet_windows:
        assert 120 <= entry.duration_minutes <= 360


def test_fault_family_assigned_from_the_admitted_catalog():
    from scenarios.fault_schedule import DEFAULT_FAULT_FAMILIES

    schedule = generate_schedule(seed=5)
    families = {e.fault_family for e in schedule.fault_episodes}
    assert families
    assert families.issubset(set(DEFAULT_FAULT_FAMILIES))


def test_entries_fit_inside_the_declared_window():
    window_hours = 24.0
    schedule = generate_schedule(seed=3, window_hours=window_hours)
    assert max(e.end_minute for e in schedule.entries) <= window_hours * 60


def test_raises_when_minimums_cannot_fit_the_window():
    with pytest.raises(ValueError):
        generate_schedule(seed=1, window_hours=1.0)


def test_round_trip_json_serialization():
    schedule = generate_schedule(seed=8)
    restored = FaultSchedule.from_json(schedule.to_json())
    assert restored == schedule


def test_custom_fault_families_are_respected():
    schedule = generate_schedule(seed=8, fault_families=("wrong_bin_usage",))
    assert all(e.fault_family == "wrong_bin_usage" for e in schedule.fault_episodes)


# --- every family gets >= min_injections_per_family episodes --------------------------------------


def test_min_injections_per_family_is_opt_in_and_leaves_existing_seeds_untouched():
    """Not passing the parameter must reproduce the exact same schedule as before -- every
    unquota'd caller (e.g. generate_schedule's non-admission uses) depends on this."""
    assert generate_schedule(seed=42).to_json() == generate_schedule(seed=42, min_injections_per_family=None).to_json()


def test_min_injections_per_family_guarantees_the_quota_for_every_family():
    from scenarios.fault_schedule import DEFAULT_FAULT_FAMILIES
    from collections import Counter

    schedule = generate_schedule(seed=42, min_injections_per_family=2)
    counts = Counter(e.fault_family for e in schedule.fault_episodes)
    for family in DEFAULT_FAULT_FAMILIES:
        assert counts[family] >= 2, (family, counts)


def test_min_injections_per_family_fails_loud_when_it_cannot_fit_the_window():
    with pytest.raises(ValueError, match="min_injections_per_family"):
        generate_schedule(seed=1, window_hours=1.0, min_injections_per_family=2)


def test_min_injections_per_family_rejects_zero_or_negative():
    with pytest.raises(ValueError, match="min_injections_per_family must be >= 1"):
        generate_schedule(seed=1, min_injections_per_family=0)


def test_min_injections_per_family_still_deterministic_and_fits_the_window():
    """After the 2026-09-25 promotion, 8 families at quota 2 need 16 episodes, and a majority of seeds'
    quiet-window overshoot now leaves too little room in the default 24h window for that (the
    late, seed-dependent ValueError generate_schedule's own docstring documents) -- this is
    exactly why generate_r1_schedule searches across seeds instead of trusting the first one.
    Determinism and window-fit only need checking on the seeds that ARE feasible; skipping an
    infeasible seed here still asserts at least one of the 50 is."""
    feasible = 0
    for seed in range(50):
        try:
            a = generate_schedule(seed=seed, min_injections_per_family=2)
        except ValueError:
            continue
        feasible += 1
        b = generate_schedule(seed=seed, min_injections_per_family=2)
        assert a.to_json() == b.to_json(), seed
        assert max(e.end_minute for e in a.entries) <= 24 * 60, seed
    assert feasible > 0


def test_invariants_hold_across_many_seeds():
    """Regression: seed 168 used to overflow past the 24h window (a false alarm landed at
    minute 1439-1444) because the fault-sizing loop could shrink-then-still-overflow instead of
    reserving room for the remaining required episodes up front. Every seed must fit the window
    and respect the minimum gap after a fault, not just the ones covered by name above."""
    for seed in range(300):
        schedule = generate_schedule(seed=seed)
        assert len(schedule.fault_episodes) >= MIN_FAULT_EPISODES_PER_ROUND, seed
        assert schedule.quiet_minutes_total >= MIN_QUIET_HOURS_PER_ROUND * 60, seed
        assert max(e.end_minute for e in schedule.entries) <= 24 * 60, seed
        ordered = sorted(schedule.entries, key=lambda e: e.start_minute)
        for prev, nxt in zip(ordered, ordered[1:]):
            assert prev.end_minute <= nxt.start_minute, (seed, prev, nxt)
            if prev.kind is EntryKind.FAULT:
                assert nxt.start_minute - prev.end_minute >= MIN_GAP_MINUTES, (seed, prev, nxt)


R2_SEEDS = (1052, 1140, 1283)
# Admission-run verdicts: the six families with an alert in 2 of 2 injections.
DETECTED_FAMILIES = {
    "assign_to_non_existent_node_hotel_res",
    "misconfig_app_hotel_res",
    "revoke_auth_mongodb",
    "revoke_auth_mongodb_rate",
    "user_unregistered_mongodb",
    "user_unregistered_mongodb_rate",
}


def test_r2_schedules_are_the_first_three_admissible_seeds_from_700():
    """Round 2 pre-registration: the seeds are not chosen, they fall out of a declared rule.

    generate_r1_schedule's deterministic search from 700 (the block after the admission run's 600
    and the dry run's 623), then from each found seed + 1. Same generator and admission validator
    as the admission run, so the three round-2 schedules replay the universe it measured. A schedule edited by hand after commit
    stops matching and fails here.
    """
    from scenarios.run_continuous_admission import generate_r1_schedule

    schedules_dir = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "scenarios" / "schedules"
    start = 700
    for expected_seed in R2_SEEDS:
        generated = generate_r1_schedule(start)
        assert generated.seed == expected_seed
        committed = json.loads((schedules_dir / f"r2-seed-{expected_seed}.json").read_text(encoding="utf-8"))
        assert committed == generated.to_json()
        start = generated.seed + 1


def test_every_r2_schedule_leaves_room_for_the_yield_criterion():
    # Round 2 criterion: yield >= 8 detected-family episodes with alert, out of >= 9 scheduled.
    schedules_dir = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "scenarios" / "schedules"
    for seed in R2_SEEDS:
        payload = json.loads((schedules_dir / f"r2-seed-{seed}.json").read_text(encoding="utf-8"))
        detected = [
            e for e in payload["entries"] if e["kind"] == "fault" and e["fault_family"] in DETECTED_FAMILIES
        ]
        assert len(detected) >= 9, (seed, len(detected))


R3_SEEDS = (1627, 1695, 2425)


def test_r3_schedules_are_the_next_three_admissible_seeds_after_round_2():
    """Round 3 pre-registration: same rule as round 2, continued from the seed after round 2's
    last (1283 + 1).

    The seeds are not chosen, they fall out of generate_r1_schedule's deterministic search. A
    schedule edited by hand after commit stops matching and fails here.
    """
    from scenarios.run_continuous_admission import generate_r1_schedule

    schedules_dir = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "scenarios" / "schedules"
    start = R2_SEEDS[-1] + 1
    for expected_seed in R3_SEEDS:
        generated = generate_r1_schedule(start)
        assert generated.seed == expected_seed
        committed = json.loads((schedules_dir / f"r3-seed-{expected_seed}.json").read_text(encoding="utf-8"))
        assert committed == generated.to_json()
        start = generated.seed + 1


def test_every_r3_schedule_leaves_room_for_the_yield_criterion():
    schedules_dir = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "scenarios" / "schedules"
    for seed in R3_SEEDS:
        payload = json.loads((schedules_dir / f"r3-seed-{seed}.json").read_text(encoding="utf-8"))
        detected = [
            e for e in payload["entries"] if e["kind"] == "fault" and e["fault_family"] in DETECTED_FAMILIES
        ]
        assert len(detected) >= 9, (seed, len(detected))


def test_r3_seeds_do_not_reuse_any_earlier_seed():
    assert not set(R3_SEEDS) & set(R2_SEEDS)


def _entry(kind, start, duration, family=None):
    from scenarios.fault_schedule import ScheduleEntry

    return ScheduleEntry(kind=kind, start_minute=start, duration_minutes=duration, fault_family=family)


def _schedule(*entries):
    return FaultSchedule(seed=0, window_hours=24.0, entries=tuple(entries))


def test_quiet_placement_rejects_a_6h_window_at_minute_0():
    from scenarios.run_continuous_admission import R1Error, validate_quiet_window_placement

    schedule = _schedule(_entry(EntryKind.QUIET, 0, 360), _entry(EntryKind.FAULT, 360, 30, "x"))
    with pytest.raises(R1Error):
        validate_quiet_window_placement(schedule)


def test_quiet_placement_rejects_a_6h_window_glued_to_a_false_alarm_on_either_side():
    from scenarios.run_continuous_admission import R1Error, validate_quiet_window_placement

    after_alarm = _schedule(_entry(EntryKind.FALSE_ALARM, 0, 5, "x"), _entry(EntryKind.QUIET, 5, 360))
    before_alarm = _schedule(
        _entry(EntryKind.FAULT, 0, 30, "x"),
        _entry(EntryKind.QUIET, 60, 360),
        _entry(EntryKind.FALSE_ALARM, 420, 5, "x"),
    )
    for schedule in (after_alarm, before_alarm):
        with pytest.raises(R1Error):
            validate_quiet_window_placement(schedule)


def test_quiet_placement_accepts_a_6h_window_clear_of_minute_0_and_false_alarms():
    from scenarios.run_continuous_admission import validate_quiet_window_placement

    schedule = _schedule(
        _entry(EntryKind.FAULT, 0, 30, "x"),
        _entry(EntryKind.QUIET, 60, 360),
        _entry(EntryKind.FAULT, 420, 30, "x"),
    )
    validate_quiet_window_placement(schedule)


def test_clean_quiet_window_is_opt_in_and_leaves_committed_seeds_untouched():
    from scenarios.run_continuous_admission import generate_r1_schedule, validate_quiet_window_placement

    # Round 3's first seed regenerates identically by default (the committed file stays valid).
    assert generate_r1_schedule(R2_SEEDS[-1] + 1).seed == R3_SEEDS[0]
    validate_quiet_window_placement(generate_r1_schedule(R3_SEEDS[-1] + 1, clean_quiet_window=True))


# The short diagnostic round's schedule (used for the 7B model) is cut from r2-seed-1052 by rule,
# and the committed file is exactly that cut.

_SCHEDULES = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "scenarios" / "schedules"
# Families with no alert in any of the four measured windows on this seed.
_NO_ALERT_FAMILIES = frozenset({"redeploy_without_PV", "wrong_bin_usage"})


def _load(name: str) -> FaultSchedule:
    return FaultSchedule.from_json(json.loads((_SCHEDULES / name).read_text(encoding="utf-8")))


def test_committed_short_schedule_is_the_rule_applied_to_seed_1052():
    base = _load("r2-seed-1052.json")
    assert _load("r2short-seed-1052.json") == short_schedule_from(base, exclude_families=_NO_ALERT_FAMILIES)


def test_short_schedule_keeps_one_fault_per_scored_family_and_the_false_alarm():
    short = _load("r2short-seed-1052.json")
    base = _load("r2-seed-1052.json")
    scored = {e.fault_family for e in base.fault_episodes} - _NO_ALERT_FAMILIES
    assert sorted(e.fault_family for e in short.fault_episodes) == sorted(scored)
    assert len(short.false_alarms) == len(base.false_alarms) == 1
    assert short.quiet_windows == ()
    assert short.seed == base.seed
    assert short.window_hours < 6


def test_short_schedule_layout_follows_the_generator_gap_rule():
    short = _load("r2short-seed-1052.json")
    cursor = 0
    for entry in short.entries:
        assert entry.start_minute == cursor
        cursor = entry.end_minute + (MIN_GAP_MINUTES if entry.kind is EntryKind.FAULT else 0)
    assert short.window_hours * 60 == cursor


def test_short_schedule_runs_only_as_a_dry_run():
    from scenarios.run_continuous_admission import (
        R1Error,
        validate_schedule_for_dry_run,
        validate_schedule_for_r1,
    )

    short = _load("r2short-seed-1052.json")
    validate_schedule_for_dry_run(short)
    with pytest.raises(R1Error):
        validate_schedule_for_r1(short)
