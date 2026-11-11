"""The corpus budget stays arithmetic: render() derives every number from the committed schedules."""

from __future__ import annotations

from pathlib import Path

import pytest

from scenarios.corpus_schedule import G0_SEED, VALIDATION_SEEDS, committed_schedules
from scenarios.corpus_budget import (
    CORPUS_VM_HOURS_CEILING,
    TOKENS_PER_TURN,
    EVAL_ROUNDS,
    EVAL_WINDOW_HOURS_ONE_SEED,
    G0_INJECTIONS,
    G0_SCHEDULE_HOURS,
    SCENARIOS,
    USD_PER_HOUR,
    VALIDATION_WINDOW_HOURS,
    accepted_injections_collected,
    budget,
    corpus_hours,
    eval_hours,
    lora_megatokens,
    render,
)

BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


def _end_hours(schedule) -> float:
    return max(entry.end_minute for entry in schedule.entries) / 60


def test_gate_and_validation_are_the_committed_schedules():
    by_seed = {schedule.seed: schedule for schedule in committed_schedules()}
    g0 = by_seed[G0_SEED]
    assert len(g0.fault_episodes) == G0_INJECTIONS
    assert _end_hours(g0) <= G0_SCHEDULE_HOURS + 0.05
    assert max(_end_hours(by_seed[seed]) for seed in VALIDATION_SEEDS) <= VALIDATION_WINDOW_HOURS + 0.05


def test_evaluation_one_seed_per_vm_is_two_arms_of_six_windows():
    """A new window goes back to one seed per VM, so it costs more than the US$27.36 three-seed
    estimate."""
    hours = eval_hours(BY_NAME["likely, 3 stacks"])
    assert hours == pytest.approx(2 * EVAL_ROUNDS * EVAL_WINDOW_HOURS_ONE_SEED)
    assert hours * USD_PER_HOUR["A6000"] > 2 * 27.36


def test_three_seed_evaluation_only_exists_in_the_best_case():
    assert [s.name for s in SCENARIOS if s.eval_seeds_per_window != 1] == ["best"]


def test_collection_stops_at_the_ceiling():
    hours, capped = corpus_hours(BY_NAME["worst"])
    assert capped and hours == CORPUS_VM_HOURS_CEILING
    assert not corpus_hours(BY_NAME["likely, 3 stacks"])[1]


def test_training_never_counts_injections_the_ceiling_cut():
    worst = BY_NAME["worst"]
    assert accepted_injections_collected(worst) == pytest.approx(43.2)
    expected = 43.2 * 2.0 * 8 * TOKENS_PER_TURN * 5 * 6 / 1e6
    assert lora_megatokens(worst) == pytest.approx(expected)


def test_total_is_ordered_and_contingents_stay_outside_it():
    totals = [budget(scenario)["total_usd"] for scenario in SCENARIOS]
    assert totals == sorted(totals)
    for scenario in SCENARIOS:
        result = budget(scenario)
        assert result["ceiling_usd"] > result["total_with_overrun_usd"]


def test_every_gpu_named_has_a_price():
    assert {scenario.source_gpu for scenario in SCENARIOS} <= set(USD_PER_HOUR)
