"""Smoke gate and window reading of a short comparator window under a chars budget."""

from __future__ import annotations

import pytest

from harness.fault_table import FAULT_TABLE
from harness.scorer import KIND_FAULT, Episode
from scenarios.chars_window import (
    BESIDE_PAIR_FAMILY,
    PAIR_FAMILIES,
    check_call_chars,
    count_model_unavailable,
    pair_cut,
    read_window,
    record_has_exit0,
    smoke_gate,
)

BUDGET = 14_000


def _call(covered=9_000, prompt=12_000, **extra):
    return {
        "request_id": "r",
        "prompt_chars": prompt,
        "covered_chars": covered,
        "context_chars": 5_000,
        "history_chars": 2_000,
        "observation_chars": None if covered is None else covered - 7_000,
        "input_tokens": 4,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 3_000,
        **extra,
    }


def _record(cause_class, exit_codes=(0,)):
    return {
        "causa_provavel": {"classe": cause_class, "detalhe": "x"},
        "decisao": "escalar",
        "blast_radius": [],
        "acoes_executadas": [
            {"classe": "C1", "comando": f"kubectl get pods {i}", "resultado": f"executado, exit={code}"}
            for i, code in enumerate(exit_codes)
        ],
    }


def _episode(family, cause_class=None, exit_codes=(0,), **kw):
    exact = FAULT_TABLE[family].accepted_cause
    return Episode(
        episode_id=f"ep-{family}",
        family=family,
        round_id="r",
        kind=KIND_FAULT,
        record=_record(cause_class or exact, exit_codes),
        alert_fired=True,
        **kw,
    )


def _pair_episodes(exits=None, wrong=()):
    exits = exits or {}
    eps = []
    for family in PAIR_FAMILIES:
        wrong_class = "node_selector_unschedulable" if FAULT_TABLE[family].accepted_cause != "node_selector_unschedulable" else "mongo_user_dropped"
        eps.append(_episode(family, cause_class=wrong_class if family in wrong else None, exit_codes=exits.get(family, (0,))))
    return eps


def test_check_call_chars_passes_when_every_call_has_chars_inside_the_budget():
    result = check_call_chars([_call(), _call(covered=14_000)], BUDGET)
    assert result["passed"] and result["max_covered_chars"] == 14_000
    assert result["max_prompt_tokens_descriptive"] == 3_104


def test_check_call_chars_fails_on_a_call_without_chars_or_with_null_covered():
    result = check_call_chars([_call(), {"input_tokens": 5}, _call(covered=None)], BUDGET)
    assert not result["passed"] and result["calls_without_chars"] == [1, 2]


def test_check_call_chars_fails_over_budget_and_names_the_breakdown():
    result = check_call_chars([_call(covered=14_001)], BUDGET)
    assert not result["passed"]
    assert result["calls_over_budget"][0]["covered_chars"] == 14_001
    assert "context_chars" in result["calls_over_budget"][0]


def test_an_empty_usage_log_is_not_a_pass():
    assert not check_call_chars([], BUDGET)["passed"]


def test_record_has_exit0_needs_an_executed_command_that_exited_zero():
    assert record_has_exit0(_record("x", (1, 0)))
    assert not record_has_exit0(_record("x", (1, 2)))
    assert not record_has_exit0(_record("x", ()))
    assert not record_has_exit0(None)
    dry = {"acoes_executadas": [{"comando": "kubectl get pods", "resultado": "DRY-RUN: nao executado"}]}
    assert not record_has_exit0(dry)


def test_count_model_unavailable():
    assert count_model_unavailable("ok\nERROR ModelUnavailable: x\ncomparator budget reached: 2\nfine") == 2


def test_smoke_gate_passes_on_a_closed_chain_whatever_the_diagnosis():
    wrong = _episode("revoke_auth_mongodb", cause_class="node_selector_unschedulable")
    result = smoke_gate([_call(), _call()], [wrong], BUDGET)
    assert result["passed"], result["reasons"]
    assert result["chain"] == {
        "injection": True,
        "triage": True,
        "command_exit0": True,
        "scorer": True,
        "family": "revoke_auth_mongodb",
    }


@pytest.mark.parametrize(
    "episode_kw,step",
    [
        ({"alert_fired": False}, "injection"),
        ({"record": None}, "triage"),
        ({"record": _record("mongo_user_dropped", (1,))}, "command_exit0"),
        ({"not_measured": "harness crash"}, "injection"),
    ],
)
def test_smoke_gate_fails_where_the_chain_opens(episode_kw, step):
    base = _episode("revoke_auth_mongodb")
    ep = Episode(**{**base.__dict__, **episode_kw})
    result = smoke_gate([_call()], [ep], BUDGET)
    assert not result["passed"] and any(step in r for r in result["reasons"])


def test_smoke_gate_fails_on_chars_even_with_a_closed_chain():
    ep = _episode("revoke_auth_mongodb")
    assert not smoke_gate([_call(covered=None)], [ep], BUDGET)["passed"]
    assert not smoke_gate([_call(covered=14_500)], [ep], BUDGET)["passed"]


def test_smoke_gate_with_no_episode_does_not_close():
    result = smoke_gate([_call()], [], BUDGET)
    assert not result["passed"] and result["chain"]["family"] is None


@pytest.mark.parametrize("n,cut", [(5, 3), (4, 2), (3, 2)])
def test_pair_cut_is_3_of_5_then_half_rounded_up(n, cut):
    assert pair_cut(n) == cut


def test_reading_capacity_when_measure1_holds_and_class_reaches_the_cut():
    reading = read_window([_call()], _pair_episodes(wrong=PAIR_FAMILIES[:2]), BUDGET, 0)
    assert reading["reading"] == "capacidade" and reading["pair"]["exact"] == 3 and reading["pair"]["cut"] == 3


def test_reading_budget_explains_when_class_is_below_the_cut():
    reading = read_window([_call()], _pair_episodes(wrong=PAIR_FAMILIES[:3]), BUDGET, 0)
    assert reading["reading"] == "orcamento_explica" and reading["pair"]["exact"] == 2


def test_a_family_without_exit0_leaves_both_columns_and_the_cut_halves():
    exits = {PAIR_FAMILIES[2]: (1,)}  # scored record without a command that exited 0
    reading = read_window([_call()], _pair_episodes(exits=exits, wrong=(PAIR_FAMILIES[0],)), BUDGET, 0)
    assert reading["pair"]["faltas"] == 4 and reading["pair"]["cut"] == 2 and reading["pair"]["exact"] == 3
    assert reading["reading"] == "capacidade"
    left = [r for r in reading["pair"]["rows"] if not r["in_pair"]]
    assert left == [{"family": PAIR_FAMILIES[2], "in_pair": False, "why": "record pontuado sem comando exit=0"}]


def test_a_pair_left_with_fewer_than_3_families_is_null():
    exits = {f: (1,) for f in PAIR_FAMILIES[:3]}
    reading = read_window([_call()], _pair_episodes(exits=exits), BUDGET, 0)
    assert reading["reading"] == "nula" and reading["pair"]["cut"] is None


def test_measure1_failure_is_null_whatever_the_class():
    over = read_window([_call(covered=14_100)], _pair_episodes(), BUDGET, 0)
    assert over["reading"] == "nula" and "medida 1" in over["why"]
    unavailable = read_window([_call()], _pair_episodes(), BUDGET, 1)
    assert unavailable["reading"] == "nula"


def test_a_family_not_measured_shrinks_the_pair():
    eps = [e for e in _pair_episodes() if e.family != PAIR_FAMILIES[4]]
    reading = read_window([_call()], eps, BUDGET, 0)
    assert reading["pair"]["faltas"] == 4 and reading["pair"]["cut"] == 2


def test_the_family_beside_the_pair_is_reported_but_never_counted():
    eps = _pair_episodes() + [_episode(BESIDE_PAIR_FAMILY)]
    reading = read_window([_call()], eps, BUDGET, 0)
    assert reading["beside_pair"] == {"family": BESIDE_PAIR_FAMILY, "exact": True}
    assert reading["pair"]["faltas"] == 5


def test_merge_usage_counts_a_call_once_and_keeps_lines_without_request_id():
    from scenarios.chars_window import merge_usage

    a = [{"request_id": "r1", "n": 1}, {"request_id": "r2", "n": 2}, {"n": 3}]
    b = [{"request_id": "r2", "n": 2}, {"request_id": "r3", "n": 4}, {"n": 3}]
    assert [r["n"] for r in merge_usage([a, b])] == [1, 2, 3, 4, 3]


def test_the_committed_smoke_schedule_is_one_pair_family_and_loads_like_every_schedule():
    import json
    from pathlib import Path

    from harness.fault_table import FAULT_TABLE
    from scenarios.fault_schedule import FaultSchedule

    path = Path(__file__).resolve().parents[2] / "tools/triage-eval/scenarios/schedules/smokegate-seed-1052.json"
    schedule = FaultSchedule.from_json(json.loads(path.read_text()))
    entries = json.loads(path.read_text())["entries"]
    assert [e["kind"] for e in entries] == ["fault"]
    assert entries[0]["fault_family"] in PAIR_FAMILIES and entries[0]["fault_family"] in FAULT_TABLE
    assert schedule is not None
