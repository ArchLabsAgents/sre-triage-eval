"""Schedules for the training corpus: seeds outside the evaluation, hold-out families never injected.

Pre-registered rule: corpus schedules are generated with seeds outside the evaluation set and with
the hold-out families (H) left out of the injection list (never injected, rather than injected and
filtered afterwards), and they are committed with their hash before any training.

The disjointness is checked on the schedule itself, never on a file name: `r2short-seed-1052.json`
does not say "r2" where a name filter would look for it and carries hold-out families, and
`revoke_auth_mongodb-mitigation-2` is a `rate` problem whose id has no "rate" in it. So seeds are
compared as integers, EVERY entry is read (a false alarm carries a family too, see
fault_schedule.generate_schedule), and each family is resolved through its problem id and its
table row.

Regenerate the committed files with `python -m scenarios.corpus_schedule` from tools/triage-eval.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

from adapters.aiopslab_problems import PROBLEM_ID_BY_FAMILY
from harness.fault_table import FAULT_TABLE, NAMESPACE_SCOPE, TABLE_VERSION
from scenarios.fault_schedule import MIN_GAP_MINUTES, EntryKind, FaultSchedule, generate_schedule

RULE_VERSION = "e9-corpus-2026-10-30"

# S0 is the only stratum a corpus schedule may inject; S1 + S2 form the hold-out H.
S0_FAMILIES: tuple[str, ...] = (
    "misconfig_app_hotel_res",
    "revoke_auth_mongodb",
    "user_unregistered_mongodb",
)
S1_FAMILIES: tuple[str, ...] = ("revoke_auth_mongodb_rate", "user_unregistered_mongodb_rate")
S2_FAMILIES: tuple[str, ...] = ("assign_to_non_existent_node_hotel_res",)
HOLDOUT_FAMILIES: tuple[str, ...] = S1_FAMILIES + S2_FAMILIES

# The hold-out covers rate and mongodb-rate; memcached-rate is the third pod of the same service.
# NAMESPACE_SCOPE is here because a whole-namespace fault (redeploy_without_PV) restarts rate too,
# and its row says "*", which a literal comparison against "rate" would let through.
FORBIDDEN_SERVICES: frozenset[str] = frozenset({"rate", "mongodb-rate", "memcached-rate", NAMESPACE_SCOPE})

# The six paired evaluation rounds.
EVAL_SEEDS: frozenset[int] = frozenset({1052, 1140, 1283, 1627, 1695, 2425})

# Every seed so far sits below 3000, so a floor keeps the corpus clear of all of them, including
# the admission (600) and dry-run (623) schedules, and of any seed the evaluation rule would pick next.
CORPUS_SEED_FLOOR = 30000

# Fault clock. The first alert lands at most 266 s after injection (misconfig, 8 windows) and a 9B
# triage takes at most 57 s, while the 20-45 min of the evaluation schedules only buy late alerts
# the evaluation never scores. MIN_GAP_MINUTES stays: no gap under 29 min was ever measured.
CORPUS_FAULT_DURATION_MINUTES_RANGE = (12, 18)
CORPUS_FALSE_ALARM_COUNT_RANGE = (1, 2)
CORPUS_MIN_QUIET_HOURS = 0

# Gate G0: 8 injections per S0 family, the same schedule for every source so sources pair up.
# 18h, not 24h: 16 injections per cluster is the most ever measured, and G0 is what measures more.
G0_SEED = CORPUS_SEED_FLOOR
G0_WINDOW_HOURS = 18.0
G0_INJECTIONS_PER_FAMILY = 8

# One seed in five is validation, by integer and before any data. Rollouts of one injection share
# pods, logs and events, and neighbouring injections share events, so the split unit is the whole
# schedule, never a trajectory. Choosing a seed would choose the split, so the two validation
# schedules are committed now and collection takes the train seeds in order, skipping none.
VALIDATION_SEED_MODULUS = 5
VALIDATION_SEED_REMAINDER = 4
VALIDATION_SEEDS: tuple[int, ...] = (CORPUS_SEED_FLOOR + 4, CORPUS_SEED_FLOOR + 9)

# Clause 7 of the acceptance rule: what must have reached the model for an answer to count as read
# and not guessed, per accepted class, lowercased substring. Frozen in the manifest before any
# corpus window. Sources: the injector and the 9B's own windows, never the commercial API arm.
EVIDENCE_TOKENS: dict[str, tuple[str, ...]] = {
    "app_image_misconfig": ("yinfangchen/geo:app3",),  # the image the injector sets (FAULT_TABLE source)
    "mongo_role_revoked": ("not authorized on",),  # 9B windows, revoke_auth_mongodb
    "mongo_user_dropped": ("sasl", "authentication failed"),  # 9B contract window, smoke family
}

SCHEDULES_DIR = Path(__file__).resolve().parent / "schedules" / "corpus"
MANIFEST_NAME = "MANIFEST.json"


class CorpusScheduleError(ValueError):
    """A schedule that must not drive a corpus window."""


def corpus_split(seed: int) -> str:
    """Which side of the corpus split a seed's episodes go to; nothing else may decide this."""
    return "validation" if seed % VALIDATION_SEED_MODULUS == VALIDATION_SEED_REMAINDER else "train"


def _balance_families(schedule: FaultSchedule, rng: random.Random) -> FaultSchedule:
    """Balance S0 over the fault entries, at most one apart, and shuffle once. Repeats stay in on
    purpose: forbidding them would let an event left by the previous injection rule out one of
    three answers, and 3 of the 10 neighbouring S0 pairs in the evaluation schedules repeat."""
    count = len(schedule.fault_episodes)
    families = list(S0_FAMILIES) * (count // len(S0_FAMILIES)) + rng.sample(S0_FAMILIES, count % len(S0_FAMILIES))
    rng.shuffle(families)
    assigned = iter(families)
    entries = tuple(
        replace(entry, fault_family=next(assigned)) if entry.kind is EntryKind.FAULT else entry
        for entry in schedule.entries
    )
    return replace(schedule, entries=entries)


def generate_corpus_schedule(seed: int, *, window_hours: float, injections_per_family: int) -> FaultSchedule:
    """Same seed and arguments, same schedule, byte for byte. Refuses what it could not commit."""
    base = generate_schedule(
        seed,
        window_hours=window_hours,
        fault_families=S0_FAMILIES,
        min_fault_episodes=injections_per_family * len(S0_FAMILIES),
        min_quiet_hours=CORPUS_MIN_QUIET_HOURS,
        false_alarm_count_range=CORPUS_FALSE_ALARM_COUNT_RANGE,
        fault_duration_minutes_range=CORPUS_FAULT_DURATION_MINUTES_RANGE,
    )
    # A stream of its own: reusing Random(seed) would replay the draws generate_schedule already
    # spent, and tie the first fault's family to the number of false alarms.
    schedule = _balance_families(base, random.Random(f"{seed}:families"))
    validate_corpus_schedule(schedule)
    return schedule


def validate_corpus_schedule(schedule: FaultSchedule) -> None:
    """The pre-registered disjointness, plus what a schedule must hold to come out of
    generate_corpus_schedule. Reads every entry, not only the fault episodes."""
    if schedule.seed in EVAL_SEEDS:
        raise CorpusScheduleError(f"seed {schedule.seed} is an evaluation seed")
    if schedule.seed < CORPUS_SEED_FLOOR:
        raise CorpusScheduleError(f"seed {schedule.seed} is below the corpus floor {CORPUS_SEED_FLOOR}")
    if not schedule.fault_episodes:
        raise CorpusScheduleError("a corpus schedule needs at least one fault episode")

    if schedule.quiet_windows:
        raise CorpusScheduleError("a corpus schedule has no quiet window")

    holdout_problems = {PROBLEM_ID_BY_FAMILY[family] for family in HOLDOUT_FAMILIES}
    for entry in schedule.entries:
        family = entry.fault_family
        if family is None:
            raise CorpusScheduleError(f"{entry.kind.value} at minute {entry.start_minute} has no family")
        if family not in S0_FAMILIES:
            raise CorpusScheduleError(f"family {family!r} at minute {entry.start_minute} is not in S0")
        if PROBLEM_ID_BY_FAMILY[family] in holdout_problems:
            raise CorpusScheduleError(f"family {family!r} resolves to a hold-out problem id")
        row = FAULT_TABLE[family]
        if {row.cause_service, row.symptom_service} & FORBIDDEN_SERVICES:
            raise CorpusScheduleError(f"family {family!r} touches a forbidden service")

    lo, hi = CORPUS_FAULT_DURATION_MINUTES_RANGE
    faults = schedule.fault_episodes
    if any(not lo <= entry.duration_minutes <= hi for entry in faults):
        raise CorpusScheduleError(f"a fault runs outside the {lo}-{hi} min clock")
    if any(b.start_minute < a.end_minute + MIN_GAP_MINUTES for a, b in zip(faults, faults[1:])):
        raise CorpusScheduleError(f"two faults sit closer than {MIN_GAP_MINUTES} min")
    counts = Counter(entry.fault_family for entry in faults)
    if set(counts) != set(S0_FAMILIES) or max(counts.values()) - min(counts.values()) > 1:
        raise CorpusScheduleError(f"S0 is not balanced: {dict(counts)}")


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def schedule_hash(schedule: FaultSchedule) -> str:
    return hashlib.sha256(canonical_bytes(schedule.to_json())).hexdigest()


def committed_schedules() -> tuple[FaultSchedule, ...]:
    """The committed set: every file in the directory, whatever the manifest says about it."""
    return tuple(
        FaultSchedule.from_json(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(SCHEDULES_DIR.glob("*.json"))
        if path.name != MANIFEST_NAME
    )


def build_manifest(schedules: tuple[FaultSchedule, ...]) -> dict:
    """What is frozen before training: the rule, the table it reads and each schedule's hash."""
    return {
        "rule_version": RULE_VERSION,
        "table_version": TABLE_VERSION,
        "s0_families": list(S0_FAMILIES),
        "holdout_families": list(HOLDOUT_FAMILIES),
        "eval_seeds": sorted(EVAL_SEEDS),
        "evidence_tokens": {cause_class: list(tokens) for cause_class, tokens in sorted(EVIDENCE_TOKENS.items())},
        "schedules": {
            f"corpus-seed-{schedule.seed}.json": {
                "sha256": schedule_hash(schedule),
                "split": corpus_split(schedule.seed),
            }
            for schedule in sorted(schedules, key=lambda s: s.seed)
        },
    }


def main() -> None:
    SCHEDULES_DIR.mkdir(parents=True, exist_ok=True)
    for seed in (G0_SEED, *VALIDATION_SEEDS):
        schedule = generate_corpus_schedule(
            seed, window_hours=G0_WINDOW_HOURS, injections_per_family=G0_INJECTIONS_PER_FAMILY
        )
        path = SCHEDULES_DIR / f"corpus-seed-{seed}.json"
        text = json.dumps(schedule.to_json(), indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise CorpusScheduleError(f"{path.name} is committed and frozen, this would change it")
        path.write_text(text, encoding="utf-8")
        print(f"{path.name}: {len(schedule.fault_episodes)} faults, sha256 {schedule_hash(schedule)}")
    manifest = build_manifest(committed_schedules())
    (SCHEDULES_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
