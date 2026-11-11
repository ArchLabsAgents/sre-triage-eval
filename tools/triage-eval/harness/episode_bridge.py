"""Joins a continuous round's journal with its TriageRecords into scorer Episodes.

Ground truth is the scheduler log (inject/recover with timestamps) plus the post-action probes. The
journal is the one written by scenarios/run_continuous_admission.py (JOURNAL_VERSION 2 or later,
which carries the round id and each entry's planned window); the records are the JSONL
harness/episode_runner.py's JsonlTriageSink appends. This module is offline and pure: it reads
two lists of dicts and never talks to a cluster, so the same round always joins to the same
episodes.

Matching rule: time window, not a label carried end to end. A record belongs to the entry whose
window contains its `t_alarme`. A FAULT entry's window runs from its `inject_start` to its planned
end (clock start + end_minute): nothing else is injected inside it, since the next entry starts
later and its own settle lead comes before that. Namespace is not re-checked here because the
record does not carry it; it is enforced upstream (the agent is wired to a single APP_NAMESPACE
and the driver scopes alerts to it).

Honest limits, stated once:
  - `alert_labels` is the journal's label set for the record's `alert_fingerprint` when the
    driver saw that exact alert (`alert_observed`), else None. Only a driver run with
    `post_action_probe` watches the whole entry; an admission-style journal stops at the first
    new alert, so there a record opened by a later alert of the same fault gets None.
  - `alert_fired` is True on `alert_observed`, False on `alert_missing`. It means "any
    in-namespace symptom alert during the episode" only for a `post_action_probe` journal; an
    admission-style journal only watched until the first alert or its timeout.
  - `root_cause_fixed_post_action` is the `probe_post_action` reading (journal v3, driver run with
    `post_action_probe`), taken at the planned end right before recover. It is None when no probe
    ran, and also when the matched record landed AFTER the probe: the agent acted on a fault the
    probe had already measured, so the reading says nothing about that action.
  - An episode never disappears: an entry the journal never reached (aborted round, when the
    schedule is given) or left half-written comes out not_measured with the reason, and a record
    no window claims comes back as an orphan, not dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from harness.scorer import KIND_FALSE_ALARM, KIND_FAULT, Episode, to_json_with_round_key

MIN_JOURNAL_VERSION = 2
# False-negative rule: a window with no record landing within this long of its FIRST alarm (the
# earliest t_alarme among its records) counts as having no record.
RECORD_DEADLINE_SECONDS = 15 * 60
FALSE_ALARM_FAMILY = "falso_alarme"  # same family name scorer.synthetic_episodes uses

# not_measured reasons written into the episodes JSON.
NOT_SETTLED = "app did not settle before injection"
NOT_REACHED = "round aborted before this entry"
HALF_WRITTEN = "journal ends in the middle of this entry"


class BridgeError(ValueError):
    """The journal cannot be joined at all (no clock start, or a shape older than v2)."""


@dataclass
class BridgeResult:
    round_id: str
    episodes: list[Episode]
    # Records no window claimed, extra records in an already-claimed window, records that matched
    # but missed the deadline, and lines that are not a joinable record at all (a truncated last
    # line of a harvested JSONL is realistic): every one is reported, none silently dropped.
    orphan_records: list[dict] = field(default_factory=list)
    extra_records: list[dict] = field(default_factory=list)
    late_records: list[dict] = field(default_factory=list)
    malformed_records: list[Any] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            # Key kept for compatibility with published run data ("rodada" = round).
            "rodada_id": self.round_id,
            "episodes": [to_json_with_round_key(ep) for ep in self.episodes],
            "orphan_records": [r.get("episode_id") for r in self.orphan_records],
            "extra_records": [r.get("episode_id") for r in self.extra_records],
            "late_records": [r.get("episode_id") for r in self.late_records],
            "malformed_records": [
                r.get("episode_id") if isinstance(r, dict) else str(r)[:200] for r in self.malformed_records
            ],
        }


@dataclass
class _Window:
    kind: str
    family: str
    start_minute: int
    opens_at: float | None = None
    closes_at: float | None = None
    not_measured: str | None = None
    alert_fired: bool | None = None
    observed: dict[str, Mapping[str, str]] = field(default_factory=dict)  # fingerprint -> labels
    done: bool = False
    records: list[dict] = field(default_factory=list)
    probe_fixed: bool | None = None
    probe_at: float | None = None


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _record_times(record: Any) -> tuple[float, float] | None:
    """(t_alarme, t_record) as epoch seconds, or None when the record cannot be placed in time."""
    if not isinstance(record, dict):
        return None
    try:
        return _epoch(record["t_alarme"]), _epoch(record["t_record"])
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _windows_from_journal(events: Sequence[Mapping[str, Any]]) -> tuple[str, list[_Window]]:
    started = next((e for e in events if e.get("phase") == "clock_started"), None)
    if started is None:
        raise BridgeError("journal has no clock_started event; the round never started")
    version = started.get("journal_version")
    if not isinstance(version, int) or version < MIN_JOURNAL_VERSION:
        raise BridgeError(
            f"journal_version {version!r} is older than {MIN_JOURNAL_VERSION}: no rodada_id (round id) or "
            "planned windows to join on"
        )
    clock_start = float(started["t"])
    per_minute = float(started["seconds_per_minute"])

    def planned(minute: int) -> float:
        return clock_start + minute * per_minute

    windows: list[_Window] = []
    current: _Window | None = None
    for event in events:
        phase = event.get("phase")
        if phase == "entry_not_measured":
            windows.append(
                _Window(KIND_FAULT, event["family"], event["start_minute"], not_measured=NOT_SETTLED, done=True)
            )
        elif phase == "inject_start":
            current = _Window(
                KIND_FAULT,
                event["family"],
                event["start_minute"],
                opens_at=float(event["t"]),
                closes_at=planned(event["end_minute"]),
            )
            windows.append(current)
        elif phase in ("alert_observed", "alert_missing") and current is not None:
            current.alert_fired = phase == "alert_observed"
            for alert in event.get("alerts") or []:
                current.observed[alert["fingerprint"]] = alert.get("labels") or {}
        elif phase == "probe_post_action" and current is not None:
            current.probe_fixed = event.get("root_cause_fixed")
            current.probe_at = float(event["t"])
        elif phase == "family_done" and current is not None:
            if event.get("error"):
                current.not_measured = f"harness: {event['error']}"
            elif not event.get("injected"):
                current.not_measured = "injection did not land"
            current.done = True
            current = None
        elif phase == "false_alarm_sent":
            windows.append(
                _Window(
                    KIND_FALSE_ALARM,
                    FALSE_ALARM_FAMILY,
                    event["start_minute"],
                    # The driver journals this event AFTER the POST returns, but the alert's
                    # startsAt (the record's t_alarme) is stamped BEFORE it, so opening at the
                    # event time left the false alarm's own record an orphan (measured: record
                    # null in all 3 seeds of a live round). The driver sleeps until the planned start before sending, so
                    # startsAt is never earlier than that.
                    opens_at=min(float(event["t"]), planned(event["start_minute"])),
                    closes_at=planned(event["end_minute"]),
                    done=True,
                )
            )
    for window in windows:
        if not window.done:
            window.not_measured = HALF_WRITTEN
    return started["rodada_id"], windows


def _add_unreached(windows: list[_Window], schedule: Mapping[str, Any]) -> None:
    seen = {(w.kind, w.start_minute) for w in windows}
    for entry in schedule.get("entries") or []:
        kind = entry.get("kind")
        if kind not in (KIND_FAULT, KIND_FALSE_ALARM):
            continue
        if (kind, entry["start_minute"]) in seen:
            continue
        family = entry["fault_family"] if kind == KIND_FAULT else FALSE_ALARM_FAMILY
        windows.append(_Window(kind, family, entry["start_minute"], not_measured=NOT_REACHED, done=True))


def build_episodes(
    journal_events: Sequence[Mapping[str, Any]],
    triage_records: Iterable[dict],
    *,
    schedule: Mapping[str, Any] | None = None,
    record_deadline_seconds: float = RECORD_DEADLINE_SECONDS,
) -> BridgeResult:
    """Pass the schedule (FaultSchedule.to_json()) whenever the round may have aborted: it is the
    only way an entry the driver never reached still shows up as a not_measured episode."""
    round_id, windows = _windows_from_journal(journal_events)
    if schedule is not None:
        _add_unreached(windows, schedule)
    windows.sort(key=lambda w: (w.start_minute, w.kind))

    result = BridgeResult(round_id, [])
    open_windows = [w for w in windows if w.opens_at is not None and w.not_measured is None]
    times: dict[int, tuple[float, float]] = {}
    for record in triage_records:
        placed = _record_times(record)
        if placed is None:
            result.malformed_records.append(record)
            continue
        times[id(record)] = placed
        owner = next((w for w in open_windows if w.opens_at <= placed[0] <= w.closes_at), None)
        if owner is None:
            result.orphan_records.append(record)
        else:
            owner.records.append(record)

    for window in windows:
        record = None
        if window.records:
            # The first record to LAND within the deadline of the window's first alarm wins, not
            # the one with the earliest alarm: a late record for the first alarm must not turn a
            # second record that did arrive in time into a false FN.
            first_alarm = min(times[id(r)][0] for r in window.records)
            by_arrival = sorted(window.records, key=lambda r: times[id(r)][1])
            on_time = [r for r in by_arrival if times[id(r)][1] - first_alarm <= record_deadline_seconds]
            on_time_ids = {id(r) for r in on_time}  # identity, not ==: a duplicated line is 2 records
            result.late_records.extend(r for r in by_arrival if id(r) not in on_time_ids)
            if on_time:
                record = on_time[0]
                result.extra_records.extend(on_time[1:])
        labels = None
        fixed = window.probe_fixed if window.probe_at is not None else None
        if record is not None:
            labels = window.observed.get(record.get("alert_fingerprint"))
            if fixed is not None and times[id(record)][1] > window.probe_at:
                fixed = None
        result.episodes.append(
            Episode(
                # kind in the id: a fault and a false alarm sharing a start_minute stay distinct.
                episode_id=f"{round_id}-{'fa' if window.kind == KIND_FALSE_ALARM else 'f'}{window.start_minute}",
                family=window.family,
                round_id=round_id,
                kind=window.kind,
                record=record,
                alert_labels=dict(labels) if labels is not None else None,
                alert_fired=window.alert_fired if window.kind == KIND_FAULT else None,
                root_cause_fixed_post_action=fixed,
                not_measured=window.not_measured,
            )
        )
    return result


def _read_jsonl(path: Path, *, tolerant: bool = False) -> list[Any]:
    """tolerant=True keeps an unparsable line as its raw string (build_episodes reports it as
    malformed) instead of losing the whole file to one truncated line. The journal is not read
    tolerantly: it is the ground truth, and a hole in it must stop the join."""
    out: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if not tolerant:
                raise
            out.append(line)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--journal", type=Path, required=True, help="<out>.events.jsonl of the round")
    parser.add_argument("--records", type=Path, required=True, help="JSONL harvested from SSLM_TRIAGE_LOG")
    parser.add_argument("--schedule", type=Path, help="schedule JSON; required to account for an aborted round")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    schedule = json.loads(args.schedule.read_text(encoding="utf-8")) if args.schedule else None
    try:
        journal = _read_jsonl(args.journal)
    except json.JSONDecodeError as exc:
        raise BridgeError(f"journal {args.journal} has an unparsable line ({exc}); refusing to join a hole in ground truth") from exc
    result = build_episodes(journal, _read_jsonl(args.records, tolerant=True), schedule=schedule)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"{result.round_id}: {len(result.episodes)} episodes, {len(result.orphan_records)} orphan, "
        f"{len(result.extra_records)} extra, {len(result.late_records)} late, "
        f"{len(result.malformed_records)} malformed records",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
