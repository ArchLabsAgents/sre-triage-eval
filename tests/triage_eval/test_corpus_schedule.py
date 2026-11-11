"""Disjointness of the training corpus schedules from the evaluation (seeds and hold-out families)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from adapters.aiopslab_problems import PROBLEM_ID_BY_FAMILY
from harness.fault_table import FAULT_TABLE
from scenarios.corpus_schedule import (
    CORPUS_SEED_FLOOR,
    EVAL_SEEDS,
    EVIDENCE_TOKENS,
    FORBIDDEN_SERVICES,
    G0_INJECTIONS_PER_FAMILY,
    G0_SEED,
    G0_WINDOW_HOURS,
    HOLDOUT_FAMILIES,
    MANIFEST_NAME,
    S0_FAMILIES,
    S1_FAMILIES,
    S2_FAMILIES,
    SCHEDULES_DIR,
    VALIDATION_SEEDS,
    CorpusScheduleError,
    build_manifest,
    committed_schedules,
    corpus_split,
    generate_corpus_schedule,
    schedule_hash,
    validate_corpus_schedule,
)
from scenarios.corpus_schedule import main as corpus_schedule_main
from scenarios.fault_schedule import EntryKind, FaultSchedule, ScheduleEntry

EVAL_SCHEDULES = SCHEDULES_DIR.parent
PREREGISTRATION = Path(__file__).resolve().parents[2] / "PREREGISTRATION.md"

# Pinned outside the generator: the manifest test alone compares the disk with itself, so a changed
# generator followed by a regeneration would pass. Changing a frozen schedule means changing these
# literals, and PREREGISTRATION.md in one reviewed commit.
FROZEN_SHA256 = {
    "corpus-seed-30000.json": "98b26a6a0136147c8d4246ca2946a23bfe01f913d0d6340570e4fb02d0defb1e",
    "corpus-seed-30004.json": "df730331a6f816933f6528fead975cabe914b060225df4f30b065abf9f3e34ab",
    "corpus-seed-30009.json": "aa6fcf815f9afa281f15792b9414f7f7210d45b852846c0fdb3e958b4e8ee11b",
}


def _schedule(seed: int, *entries: ScheduleEntry) -> FaultSchedule:
    return FaultSchedule(seed=seed, window_hours=1.0, entries=entries)


def _fault(family: str, start: int = 0) -> ScheduleEntry:
    return ScheduleEntry(EntryKind.FAULT, start, 15, family)


def _all_s0() -> tuple[ScheduleEntry, ...]:
    return tuple(_fault(family, index * 60) for index, family in enumerate(S0_FAMILIES))


def _g0_at(seed: int) -> FaultSchedule:
    return generate_corpus_schedule(
        seed, window_hours=G0_WINDOW_HOURS, injections_per_family=G0_INJECTIONS_PER_FAMILY
    )


def _g0() -> FaultSchedule:
    return _g0_at(G0_SEED)


def test_strata_cover_exactly_the_six_detected_families():
    detected = {family for family, row in FAULT_TABLE.items() if row.oracle_keys}
    assert set(S0_FAMILIES) | set(S1_FAMILIES) | set(S2_FAMILIES) == detected
    assert not set(S0_FAMILIES) & set(HOLDOUT_FAMILIES)


def test_s2_class_is_never_an_s0_answer():
    assert FAULT_TABLE["assign_to_non_existent_node_hotel_res"].accepted_cause not in {
        FAULT_TABLE[family].accepted_cause for family in S0_FAMILIES
    }


def test_eval_seeds_are_the_six_committed_evaluation_rounds():
    for seed in EVAL_SEEDS:
        prefix = "r2" if seed < 1500 else "r3"
        payload = json.loads((EVAL_SCHEDULES / f"{prefix}-seed-{seed}.json").read_text(encoding="utf-8"))
        assert payload["seed"] == seed


def test_floor_is_above_every_seed_ever_committed():
    seeds = {
        json.loads(path.read_text(encoding="utf-8"))["seed"]
        for path in EVAL_SCHEDULES.glob("*.json")
    }
    assert seeds and max(seeds) < CORPUS_SEED_FLOOR


def test_every_committed_evaluation_schedule_is_refused():
    """Includes r2short-seed-1052: a filter by file name would not see it as evaluation data."""
    paths = sorted(EVAL_SCHEDULES.glob("*.json"))
    assert any(path.name == "r2short-seed-1052.json" for path in paths)
    for path in paths:
        schedule = FaultSchedule.from_json(json.loads(path.read_text(encoding="utf-8")))
        with pytest.raises(CorpusScheduleError):
            validate_corpus_schedule(schedule)


@pytest.mark.parametrize("family", HOLDOUT_FAMILIES)
def test_holdout_family_is_refused_as_a_fault(family):
    with pytest.raises(CorpusScheduleError, match="not in S0"):
        validate_corpus_schedule(_schedule(CORPUS_SEED_FLOOR, _fault(family)))


@pytest.mark.parametrize("family", HOLDOUT_FAMILIES)
def test_holdout_family_is_refused_on_a_false_alarm(family):
    """A false alarm carries a family even though the alert never uses it: 4 of the 9 in the
    evaluation schedules carry a hold-out family, so reading only the faults would miss them."""
    schedule = _schedule(
        CORPUS_SEED_FLOOR,
        *_all_s0(),
        ScheduleEntry(EntryKind.FALSE_ALARM, 300, 5, family),
    )
    with pytest.raises(CorpusScheduleError, match="not in S0"):
        validate_corpus_schedule(schedule)


def test_family_outside_the_strata_is_refused():
    """redeploy_without_PV restarts rate and its row says "*", not "rate"."""
    for family in ("redeploy_without_PV", "wrong_bin_usage", "container_kill"):
        with pytest.raises(CorpusScheduleError, match="not in S0"):
            validate_corpus_schedule(_schedule(CORPUS_SEED_FLOOR, _fault(family)))


def test_entry_without_family_is_refused():
    """from_json turns a missing key into None; it would only blow up mid window otherwise."""
    for kind in (EntryKind.FAULT, EntryKind.FALSE_ALARM):
        schedule = _schedule(CORPUS_SEED_FLOOR, *_all_s0(), ScheduleEntry(kind, 300, 15, None))
        with pytest.raises(CorpusScheduleError, match="has no family"):
            validate_corpus_schedule(schedule)


def test_quiet_window_is_refused():
    schedule = _schedule(CORPUS_SEED_FLOOR, *_all_s0(), ScheduleEntry(EntryKind.QUIET, 300, 120, None))
    with pytest.raises(CorpusScheduleError, match="no quiet window"):
        validate_corpus_schedule(schedule)


def test_hand_written_schedule_outside_the_clock_is_refused():
    long_fault = ScheduleEntry(EntryKind.FAULT, 0, 45, S0_FAMILIES[0])
    with pytest.raises(CorpusScheduleError, match="clock"):
        validate_corpus_schedule(_schedule(CORPUS_SEED_FLOOR, long_fault, *_all_s0()[1:]))


def test_overlapping_faults_are_refused():
    entries = tuple(_fault(family, index * 20) for index, family in enumerate(S0_FAMILIES))
    with pytest.raises(CorpusScheduleError, match="closer than"):
        validate_corpus_schedule(_schedule(CORPUS_SEED_FLOOR, *entries))


def test_unbalanced_schedule_is_refused():
    entries = tuple(_fault(S0_FAMILIES[0], index * 60) for index in range(3))
    with pytest.raises(CorpusScheduleError, match="not balanced"):
        validate_corpus_schedule(_schedule(CORPUS_SEED_FLOOR, *entries))


@pytest.mark.parametrize("service", sorted(FORBIDDEN_SERVICES))
def test_table_row_drifting_into_a_forbidden_service_is_refused(monkeypatch, service):
    family = S0_FAMILIES[0]
    monkeypatch.setitem(FAULT_TABLE, family, replace(FAULT_TABLE[family], symptom_service=service))
    with pytest.raises(CorpusScheduleError, match="forbidden service"):
        validate_corpus_schedule(_schedule(CORPUS_SEED_FLOOR, *_all_s0()))


def test_problem_id_drifting_into_the_holdout_is_refused(monkeypatch):
    monkeypatch.setitem(PROBLEM_ID_BY_FAMILY, S0_FAMILIES[1], "revoke_auth_mongodb-mitigation-2")
    with pytest.raises(CorpusScheduleError, match="hold-out problem id"):
        validate_corpus_schedule(_schedule(CORPUS_SEED_FLOOR, *_all_s0()))


def test_seed_below_the_floor_is_refused_even_outside_the_evaluation():
    with pytest.raises(CorpusScheduleError, match="below the corpus floor"):
        validate_corpus_schedule(_schedule(600, *_all_s0()))


def test_evaluation_seed_is_refused_by_integer():
    with pytest.raises(CorpusScheduleError, match="evaluation seed"):
        validate_corpus_schedule(_schedule(1052, *_all_s0()))


def test_schedule_without_faults_is_refused():
    with pytest.raises(CorpusScheduleError, match="at least one fault"):
        validate_corpus_schedule(_schedule(CORPUS_SEED_FLOOR))


def test_s0_families_resolve_outside_the_holdout_problems_and_services():
    holdout_problems = {PROBLEM_ID_BY_FAMILY[family] for family in HOLDOUT_FAMILIES}
    assert any("rate" not in problem for problem in holdout_problems)  # why names are not enough
    for family in S0_FAMILIES:
        assert PROBLEM_ID_BY_FAMILY[family] not in holdout_problems
        row = FAULT_TABLE[family]
        assert not {row.cause_service, row.symptom_service} & FORBIDDEN_SERVICES


def test_generation_is_deterministic_and_balanced():
    schedule = _g0()
    assert schedule.to_json() == _g0().to_json()
    counts = Counter(entry.fault_family for entry in schedule.fault_episodes)
    assert counts == {family: G0_INJECTIONS_PER_FAMILY for family in S0_FAMILIES}


@pytest.mark.parametrize("seed", range(CORPUS_SEED_FLOOR, CORPUS_SEED_FLOOR + 40))
def test_any_seed_stays_balanced_and_valid(seed):
    schedule = generate_corpus_schedule(seed, window_hours=24.0, injections_per_family=10)
    counts = Counter(entry.fault_family for entry in schedule.fault_episodes)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_families_may_repeat_back_to_back():
    """Forbidding it would let a leftover event rule out one of three answers."""
    faults = _g0().fault_episodes
    assert any(a.fault_family == b.fault_family for a, b in zip(faults, faults[1:]))


def test_family_stream_is_independent_of_the_false_alarm_draw():
    firsts: dict[int, set[str]] = {1: set(), 2: set()}
    for seed in range(CORPUS_SEED_FLOOR, CORPUS_SEED_FLOOR + 200):
        schedule = _g0_at(seed)
        firsts[len(schedule.false_alarms)].add(schedule.fault_episodes[0].fault_family)
    assert firsts[1] == set(S0_FAMILIES) and firsts[2] == set(S0_FAMILIES)


def test_generator_refuses_a_seed_it_could_not_commit():
    with pytest.raises(CorpusScheduleError):
        generate_corpus_schedule(1052, window_hours=G0_WINDOW_HOURS, injections_per_family=1)


def test_split_is_a_rule_on_the_seed():
    assert corpus_split(30004) == "validation"
    assert corpus_split(30009) == "validation"
    assert {corpus_split(seed) for seed in (30000, 30001, 30002, 30003, 30005)} == {"train"}
    assert {corpus_split(seed) for seed in VALIDATION_SEEDS} == {"validation"}


def test_committed_schedules_pass_the_validator_and_regenerate():
    schedules = committed_schedules()
    assert schedules
    for schedule in schedules:
        validate_corpus_schedule(schedule)
    assert _g0().to_json() in [schedule.to_json() for schedule in schedules]


def test_manifest_covers_every_committed_file_and_its_hash():
    """A file added to the directory without its hash, or edited after, fails here."""
    manifest = json.loads((SCHEDULES_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest == build_manifest(committed_schedules())
    on_disk = {path.name for path in SCHEDULES_DIR.glob("*.json")} - {MANIFEST_NAME}
    assert on_disk == set(manifest["schedules"])


def test_committed_hashes_are_the_frozen_literals():
    manifest = json.loads((SCHEDULES_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {name: entry["sha256"] for name, entry in manifest["schedules"].items()} == FROZEN_SHA256


def test_preregistration_cites_every_frozen_hash():
    text = PREREGISTRATION.read_text(encoding="utf-8")
    assert all(digest in text for digest in FROZEN_SHA256.values())


def test_committed_seeds_follow_the_order_and_skip_none():
    seeds = sorted(schedule.seed for schedule in committed_schedules())
    train = [seed for seed in seeds if corpus_split(seed) == "train"]
    assert train == [seed for seed in range(CORPUS_SEED_FLOOR, max(train) + 1) if corpus_split(seed) == "train"]
    assert set(VALIDATION_SEEDS) <= set(seeds)


def test_evidence_tokens_cover_every_s0_answer_and_are_in_the_manifest():
    assert set(EVIDENCE_TOKENS) == {FAULT_TABLE[family].accepted_cause for family in S0_FAMILIES}
    assert all(token == token.lower() for tokens in EVIDENCE_TOKENS.values() for token in tokens)
    manifest = json.loads((SCHEDULES_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["evidence_tokens"] == {cause_class: list(tokens) for cause_class, tokens in EVIDENCE_TOKENS.items()}


def test_regeneration_refuses_to_touch_a_frozen_file(tmp_path, monkeypatch):
    monkeypatch.setattr("scenarios.corpus_schedule.SCHEDULES_DIR", tmp_path)
    (tmp_path / "corpus-seed-30000.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(CorpusScheduleError, match="frozen"):
        corpus_schedule_main()


def test_hash_moves_when_one_entry_moves():
    schedule = _g0()
    first, *rest = schedule.entries
    moved = FaultSchedule(
        seed=schedule.seed,
        window_hours=schedule.window_hours,
        entries=(ScheduleEntry(first.kind, first.start_minute + 1, first.duration_minutes, first.fault_family), *rest),
    )
    assert schedule_hash(moved) != schedule_hash(schedule)
