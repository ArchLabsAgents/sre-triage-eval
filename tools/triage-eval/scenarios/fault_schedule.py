"""Pseudo-random fault schedule for the continuous 24/7 scenario.

The schedule (and the fault table with expected cause, urgency and disposition per family) is
committed BEFORE the round it drives and is immutable once the round starts: a threshold or table
written before the data is never revised after. One fault at a time, never overlapping; minimum
30-minute gap between a fault's recovery and the next injection; quiet windows of 2-6h with zero
fault active (control 1: everything a TriageRecord reports inside a quiet window is a false
positive by definition); 1-2 injected false alarms per round (control 2, honesty check: an alert
fires via the Alertmanager API with no fault behind it).

Everything here is generation logic only: it produces the ground-truth timeline a round will
follow. Actually firing alerts and injecting faults against a live cluster is the driver's job
(scenarios/run_continuous_admission.py), which needs a live lab.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from adapters.mitigation_probes import PROBES

# The admitted fault families, sourced from the probe registry so this
# module can never drift from the one the harness actually knows how to verify.
DEFAULT_FAULT_FAMILIES: tuple[str, ...] = tuple(sorted(PROBES))

MIN_GAP_MINUTES = 30
QUIET_WINDOW_MINUTES_RANGE = (120, 360)  # 2-6h, control 1
MIN_QUIET_HOURS_PER_ROUND = 8  # a 24h round carries >= 8h of quiet windows
MIN_FAULT_EPISODES_PER_ROUND = 8  # a round must yield >= 8 fault episodes
DEFAULT_FALSE_ALARM_COUNT_RANGE = (1, 2)  # control 2
FALSE_ALARM_SETTLE_MINUTES = 5

# Fault "active" duration (inject -> scheduled recover) is not a number the doc fixes; this is a
# generation parameter, not a pre-registered value. Callers driving a real round should review/override it
# before committing a schedule, per the same "written before the data, then immutable" rule.
DEFAULT_FAULT_DURATION_MINUTES_RANGE = (20, 45)


class EntryKind(str, Enum):
    FAULT = "fault"
    QUIET = "quiet"
    FALSE_ALARM = "false_alarm"


@dataclass(frozen=True)
class ScheduleEntry:
    kind: EntryKind
    start_minute: int
    duration_minutes: int
    # Set for FAULT and FALSE_ALARM; None for QUIET.
    fault_family: str | None = None

    @property
    def end_minute(self) -> int:
        return self.start_minute + self.duration_minutes


@dataclass(frozen=True)
class FaultSchedule:
    seed: int
    window_hours: float
    entries: tuple[ScheduleEntry, ...]

    @property
    def fault_episodes(self) -> tuple[ScheduleEntry, ...]:
        return tuple(e for e in self.entries if e.kind is EntryKind.FAULT)

    @property
    def false_alarms(self) -> tuple[ScheduleEntry, ...]:
        return tuple(e for e in self.entries if e.kind is EntryKind.FALSE_ALARM)

    @property
    def quiet_windows(self) -> tuple[ScheduleEntry, ...]:
        return tuple(e for e in self.entries if e.kind is EntryKind.QUIET)

    @property
    def quiet_minutes_total(self) -> int:
        return sum(e.duration_minutes for e in self.quiet_windows)

    def to_json(self) -> dict:
        """Serialization committed to disk before a round (seed and tables are committed together,
        before the round is launched)."""
        return {
            "seed": self.seed,
            "window_hours": self.window_hours,
            "entries": [
                {
                    "kind": e.kind.value,
                    "start_minute": e.start_minute,
                    "duration_minutes": e.duration_minutes,
                    "fault_family": e.fault_family,
                }
                for e in self.entries
            ],
        }

    @classmethod
    def from_json(cls, payload: dict) -> "FaultSchedule":
        return cls(
            seed=payload["seed"],
            window_hours=payload["window_hours"],
            entries=tuple(
                ScheduleEntry(
                    kind=EntryKind(e["kind"]),
                    start_minute=e["start_minute"],
                    duration_minutes=e["duration_minutes"],
                    fault_family=e.get("fault_family"),
                )
                for e in payload["entries"]
            ),
        )


def generate_schedule(
    seed: int,
    *,
    window_hours: float = 24.0,
    fault_families: tuple[str, ...] = DEFAULT_FAULT_FAMILIES,
    min_fault_episodes: int = MIN_FAULT_EPISODES_PER_ROUND,
    min_quiet_hours: float = MIN_QUIET_HOURS_PER_ROUND,
    false_alarm_count_range: tuple[int, int] = DEFAULT_FALSE_ALARM_COUNT_RANGE,
    fault_duration_minutes_range: tuple[int, int] = DEFAULT_FAULT_DURATION_MINUTES_RANGE,
    min_injections_per_family: int | None = None,
) -> FaultSchedule:
    """Deterministic given (seed, every other parameter): same inputs always produce the same
    entries, byte for byte. This is what lets the API comparator arm reuse a base round's seed and
    replay the identical episode-by-episode schedule for a paired test.

    `min_injections_per_family` is opt-in (default None): when set, every family in
    `fault_families` is guaranteed at least that many fault episodes (a family needs >= 2
    injections for a tri-state detected/not_detected/intermittent verdict, not a single coin
    flip). It only raises `min_fault_episodes` when the quota needs
    more episodes than the default minimum, and never changes family assignment when left None,
    so every existing seed (including the committed r1-admission-seed-600.json) stays byte for
    byte identical.

    Raises ValueError if the requested minimums cannot fit inside window_hours -- fail loud at
    generation time, never silently produce a schedule that fails the round's own criteria.

    Note for a caller narrowing window_hours/min_fault_episodes/min_quiet_hours close to the
    limit: the feasibility check below uses each quiet window's minimum width
    (fault_duration_minutes_range[0]), but the actual quiet-window carving loop can overshoot
    min_quiet_hours by up to one window's width (QUIET_WINDOW_MINUTES_RANGE[1] - [0]) before the
    fault-episode budget is computed. That overshoot is accounted for later, so an over-tight
    call can still raise ValueError even though this check passed -- fails loud either way, never
    silently, but a seed-dependent late failure is a worse debugging experience than an early one
    when the caller is tuning parameters near the edge of feasibility.
    """
    if not fault_families:
        raise ValueError("fault_families must not be empty")

    quota_note = ""
    if min_injections_per_family is not None:
        if min_injections_per_family < 1:
            raise ValueError("min_injections_per_family must be >= 1")
        required_family_slots = min_injections_per_family * len(fault_families)
        if required_family_slots > min_fault_episodes:
            quota_note = (
                f" (min_fault_episodes raised from {min_fault_episodes} to {required_family_slots} "
                f"to fit min_injections_per_family={min_injections_per_family} "
                f"across {len(fault_families)} families)"
            )
            min_fault_episodes = required_family_slots

    total_minutes = int(round(window_hours * 60))
    min_quiet_minutes = int(round(min_quiet_hours * 60))
    min_fault_minutes = min_fault_episodes * (fault_duration_minutes_range[0] + MIN_GAP_MINUTES)
    if min_quiet_minutes + min_fault_minutes > total_minutes:
        raise ValueError(
            f"window_hours={window_hours} cannot fit >= {min_fault_episodes} fault episodes "
            f"(>= {min_fault_minutes} min with gaps) plus >= {min_quiet_hours}h quiet "
            f"(>= {min_quiet_minutes} min) -- widen the window or relax the minimums{quota_note}"
        )

    rng = random.Random(seed)

    # 1. Carve out quiet windows (2-6h each) until the minimum quiet budget is met. May overshoot
    # min_quiet_minutes by less than one window's width (each window must stay inside the
    # declared 2-6h range, so an exact-fit last window is not guaranteed); the overshoot is
    # accounted for below before fault episodes are sized, so the total never exceeds the window.
    quiet_windows_minutes: list[int] = []
    quiet_total = 0
    while quiet_total < min_quiet_minutes:
        lo, hi = QUIET_WINDOW_MINUTES_RANGE
        window = rng.randint(lo, hi)
        quiet_windows_minutes.append(window)
        quiet_total += window

    # 2. False alarms: brief, standalone entries.
    false_alarm_count = rng.randint(*false_alarm_count_range)
    false_alarm_total = false_alarm_count * FALSE_ALARM_SETTLE_MINUTES

    # 3. Fault episodes: exactly min_fault_episodes required ones, sized so the running total
    # (duration + gap) never exceeds what's left after quiet windows and false alarms, then zero
    # or more optional extra episodes while room remains. Reserving room for the *remaining*
    # required episodes up front (max_allowed below) is what keeps this exact -- no shrink-after-
    # the-fact pass that can silently overflow.
    active_budget = total_minutes - quiet_total - false_alarm_total
    lo, hi = fault_duration_minutes_range
    if active_budget < min_fault_episodes * (lo + MIN_GAP_MINUTES):
        raise ValueError(
            f"window_hours={window_hours} cannot fit {min_fault_episodes} fault episodes "
            f"(>= {min_fault_episodes * (lo + MIN_GAP_MINUTES)} min) after reserving "
            f"{quiet_total} min quiet + {false_alarm_total} min false alarms "
            f"(only {active_budget} min left) -- widen the window or relax the minimums{quota_note}"
        )

    fault_durations: list[int] = []
    remaining_slots = active_budget
    for i in range(min_fault_episodes):
        remaining_required_after_this = min_fault_episodes - i - 1
        max_allowed = remaining_slots - MIN_GAP_MINUTES - remaining_required_after_this * (lo + MIN_GAP_MINUTES)
        duration = rng.randint(lo, max(lo, min(hi, max_allowed)))
        fault_durations.append(duration)
        remaining_slots -= duration + MIN_GAP_MINUTES

    while remaining_slots >= lo + MIN_GAP_MINUTES and rng.random() >= 0.15:
        duration = rng.randint(lo, min(hi, remaining_slots - MIN_GAP_MINUTES))
        fault_durations.append(duration)
        remaining_slots -= duration + MIN_GAP_MINUTES

    # 3b. Family assignment for faults. Left None (default), family is drawn per fault during the
    # interleave loop below, exactly as before -- byte for byte identical to every seed already
    # committed. With a quota, every family gets its guaranteed count up front, any remaining slots
    # (required extras or the optional tail above) get a random family, then the whole list is
    # shuffled once so quota families are not front-loaded in schedule order.
    fault_family_assignment: list[str] | None = None
    if min_injections_per_family is not None:
        fault_family_assignment = list(fault_families) * min_injections_per_family
        extra = len(fault_durations) - len(fault_family_assignment)
        fault_family_assignment.extend(rng.choice(fault_families) for _ in range(extra))
        rng.shuffle(fault_family_assignment)

    # 4. Interleave: shuffle the placement order of {fault block, quiet block, false alarm}
    # segments, then lay them out back to back with the mandatory gap after each fault.
    if fault_family_assignment is not None:
        fault_segments = [(EntryKind.FAULT, d, f) for d, f in zip(fault_durations, fault_family_assignment)]
    else:
        fault_segments = [(EntryKind.FAULT, d, None) for d in fault_durations]
    segments: list[tuple[EntryKind, int, str | None]] = (
        fault_segments
        + [(EntryKind.QUIET, d, None) for d in quiet_windows_minutes]
        + [(EntryKind.FALSE_ALARM, FALSE_ALARM_SETTLE_MINUTES, None) for _ in range(false_alarm_count)]
    )
    rng.shuffle(segments)

    entries: list[ScheduleEntry] = []
    cursor = 0
    for kind, duration, family in segments:
        if family is None and kind in (EntryKind.FAULT, EntryKind.FALSE_ALARM):
            family = rng.choice(fault_families)
        entries.append(ScheduleEntry(kind=kind, start_minute=cursor, duration_minutes=duration, fault_family=family))
        cursor += duration
        if kind is EntryKind.FAULT:
            cursor += MIN_GAP_MINUTES

    return FaultSchedule(seed=seed, window_hours=window_hours, entries=tuple(entries))


def short_schedule_from(base: FaultSchedule, *, exclude_families: frozenset[str]) -> FaultSchedule:
    """Short diagnostic round cut from a committed schedule, by rule and never by hand: walking
    `base` in order, keep every false alarm and the FIRST fault of each family outside `exclude_families`, drop every quiet window,
    keep each kept entry's own duration, and lay them out back to back with the same gap rule as
    generate_schedule (MIN_GAP_MINUTES after each fault, none after a false alarm). Same seed as
    `base`, so the file says which schedule it was cut from; window_hours is the laid-out length.

    Only for runs driven with --dry-run: it has no quiet window and one injection per family, so
    validate_schedule_for_r1 refuses it on purpose."""
    seen: set[str] = set()
    kept: list[ScheduleEntry] = []
    for entry in base.entries:
        if entry.kind is EntryKind.FALSE_ALARM:
            kept.append(entry)
        elif entry.kind is EntryKind.FAULT and entry.fault_family not in exclude_families:
            if entry.fault_family not in seen:
                seen.add(entry.fault_family)
                kept.append(entry)
    entries: list[ScheduleEntry] = []
    cursor = 0
    for entry in kept:
        entries.append(
            ScheduleEntry(
                kind=entry.kind,
                start_minute=cursor,
                duration_minutes=entry.duration_minutes,
                fault_family=entry.fault_family,
            )
        )
        cursor += entry.duration_minutes
        if entry.kind is EntryKind.FAULT:
            cursor += MIN_GAP_MINUTES
    return FaultSchedule(seed=base.seed, window_hours=cursor / 60, entries=tuple(entries))
