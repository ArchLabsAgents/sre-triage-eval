"""Scorer rubric, oracle baselines (H3b) and the clustered bootstrap.

Everything is judged against harness/fault_table.py; these tests pin the pre-registered choices
written in harness/scorer.py's docstring, so changing one is a visible diff here and in the
dry-run hash, never a silent drift once scored rounds start.
"""

from __future__ import annotations

import pytest

from harness.fault_table import APP_NAMESPACE, FAULT_TABLE, REVISION_PLACEHOLDER
from harness.scorer import (
    H3B_MIN_AMBIGUOUS,
    KIND_FALSE_ALARM,
    Episode,
    canonical_command,
    clustered_bootstrap,
    commands_match,
    disposition,
    dry_run,
    fit_oracle,
    h3a_band,
    is_ambiguous,
    score,
    synthetic_episodes,
    triage_solution,
    triage_understanding,
    urgency_correct,
    wilson,
)

GEO_NOT_READY = {"alertname": "KubePodNotReady", "namespace": APP_NAMESPACE, "pod": "geo-84fbc958c7-bmj2n"}
RATE_NOT_READY = {"alertname": "KubePodNotReady", "namespace": APP_NAMESPACE, "pod": "rate-5d8f7c9b6-qq2lp"}


def _record(cause_class, decision="escalar", blast=("geo",), executed=None, proposal=None, urgency="P2", detail="x"):
    return {
        "causa_provavel": {"classe": cause_class, "detalhe": detail},
        "blast_radius": [{"service": s, "severidade": "user_facing"} for s in blast],
        "decisao": decision,
        "urgencia": urgency,
        "acoes_executadas": [{"classe": "C2", "comando": c, "t": "t", "read_back": "", "resultado": ""} for c in executed]
        if executed
        else None,
        "acao_recomendada": {"comandos": proposal, "efeito_esperado": "x", "risco": "x"} if proposal else None,
    }


def _ep(family, record, round_id="r1", labels=GEO_NOT_READY, fixed=None, **kw):
    return Episode(f"{round_id}-{family}", family, round_id, record=record, alert_labels=labels, root_cause_fixed_post_action=fixed, **kw)


def _undo(service):
    return f"kubectl rollout undo deployment/{service} -n {APP_NAMESPACE} --to-revision=7"


# --- command canonicalization -----------------------------------------------------------------------


def test_canonical_command_ignores_flag_order_revision_number_and_deploy_alias():
    table = FAULT_TABLE["misconfig_app_hotel_res"].expected_action[0]
    live = f"kubectl rollout undo --to-revision=12 --namespace {APP_NAMESPACE} deploy/geo"
    assert REVISION_PLACEHOLDER in table
    assert canonical_command(live) == canonical_command(table)


def test_canonical_command_keeps_the_target_and_the_verb():
    assert canonical_command(_undo("geo")) != canonical_command(_undo("profile"))
    assert canonical_command(_undo("geo")) != canonical_command(f"kubectl rollout restart deployment/geo -n {APP_NAMESPACE}")


def test_commands_match_is_ordered_and_never_true_for_an_empty_expectation():
    seq = FAULT_TABLE["user_unregistered_mongodb"].expected_action
    assert commands_match(list(seq), seq)
    assert not commands_match(list(reversed(seq)), seq)
    assert not commands_match([], ())


# --- triage_understanding -----------------------------------------------------------------------------


def test_understanding_correct_needs_exact_class_and_the_target_service():
    assert triage_understanding(_ep("misconfig_app_hotel_res", _record("app_image_misconfig")))[0] == "correto"
    assert triage_understanding(_ep("misconfig_app_hotel_res", _record("app_image_misconfig", blast=("search",))))[0] == "parcial"


def test_understanding_mongo_accepts_the_mongo_service_or_the_symptom_service():
    for blast in (("mongodb-geo",), ("geo",)):
        assert triage_understanding(_ep("revoke_auth_mongodb", _record("mongo_role_revoked", blast=blast)))[0] == "correto"


def test_understanding_sibling_class_is_partial_only_with_the_right_blast():
    assert triage_understanding(_ep("revoke_auth_mongodb", _record("mongo_user_dropped")))[0] == "parcial"
    assert triage_understanding(_ep("revoke_auth_mongodb", _record("mongo_user_dropped", blast=("search",))))[0] == "errado"


def test_understanding_wrong_class_and_instrument_error_are_wrong():
    assert triage_understanding(_ep("misconfig_app_hotel_res", _record("image_pull_failure")))[0] == "errado"
    cites = _record("app_image_misconfig", detail="kubectl top said Metrics API not available")
    assert triage_understanding(_ep("misconfig_app_hotel_res", cites)) == ("errado", "causa cita erro de instrumento (F2/F5)")


def test_understanding_unknown_plus_escalate_is_honest_never_wrong():
    assert triage_understanding(_ep("misconfig_app_hotel_res", _record("unknown")))[0] == "honesto"
    assert triage_understanding(_ep("container_kill", _record("unknown")))[0] == "honesto"  # hold-out
    assert triage_understanding(_ep("container_kill", _record("oom_killed")))[0] == "errado"


def test_understanding_not_measured_and_missing_record_are_none_with_reason():
    value, reason = triage_understanding(Episode("e", "misconfig_app_hotel_res", "r1", not_measured="webhook perdido"))
    assert value is None and "webhook perdido" in reason
    assert triage_understanding(_ep("misconfig_app_hotel_res", None))[0] is None


def test_namespace_wide_row_matches_any_blast():
    assert triage_understanding(_ep("redeploy_without_PV", _record("pv_lost_on_redeploy", blast=("frontend",))))[0] == "correto"


# --- triage_solution + disposition (revision-injected rows are agir/C2) ----------------------------


@pytest.mark.parametrize("family, service", [
    ("misconfig_app_hotel_res", "geo"),
    ("assign_to_non_existent_node_hotel_res", "geo"),
    ("wrong_bin_usage", "profile"),
])
def test_agir_rows_executed_canonical_undo_with_probe_true_is_correct(family, service):
    ep = _ep(family, _record("x", decision="agir", executed=[_undo(service)]), fixed=True)
    assert triage_solution(ep)[0] == "correta_executada"
    assert disposition(ep) == "correta"


def test_agir_row_canonical_undo_without_improvement_is_wrong_and_unmeasured_probe_is_none():
    base = _record("x", decision="agir", executed=[_undo("geo")])
    assert triage_solution(_ep("misconfig_app_hotel_res", base, fixed=False))[0] == "errada"
    assert triage_solution(_ep("misconfig_app_hotel_res", base, fixed=None))[0] is None


def test_agir_row_escalated_with_the_right_proposal_is_partial_and_type_1():
    ep = _ep("wrong_bin_usage", _record("x", proposal=[_undo("profile")]))
    assert triage_solution(ep)[0] == "parcial"
    assert disposition(ep) == "tipo1_medo"


def test_escalar_row_proposal_matching_the_table_is_correct_proposal():
    row = FAULT_TABLE["user_unregistered_mongodb"]
    ep = _ep("user_unregistered_mongodb", _record("x", proposal=list(row.expected_action)))
    assert triage_solution(ep)[0] == "correta_proposta"
    assert disposition(ep) == "correta"
    partial = _ep("user_unregistered_mongodb", _record("x", proposal=[row.expected_action[1]]))
    assert triage_solution(partial)[0] == "parcial"


def test_escalar_row_acting_is_wrong_and_type_2():
    ep = _ep("revoke_auth_mongodb", _record("x", decision="agir", executed=[_undo("geo")]), fixed=True)
    assert triage_solution(ep)[0] == "errada"
    assert disposition(ep) == "tipo2_afoiteza"


def test_escalar_row_without_canonical_command_is_solved_by_escalating():
    assert triage_solution(_ep("redeploy_without_PV", _record("x")))[0] == "correta_proposta"


def test_urgency_is_judged_against_the_table_row():
    assert urgency_correct(_ep("misconfig_app_hotel_res", _record("x", urgency="P2"))) is True
    assert urgency_correct(_ep("redeploy_without_PV", _record("x", urgency="P2"))) is False


# --- oracle + ambiguity ---------------------------------------------------------------------------


def test_ambiguity_comes_from_the_table_key_not_from_the_data():
    assert is_ambiguous(_ep("misconfig_app_hotel_res", _record("x")))
    assert not is_ambiguous(_ep("revoke_auth_mongodb_rate", _record("x"), labels=RATE_NOT_READY))


def test_oracle_predicts_the_most_frequent_mechanism_per_key_ties_by_name():
    eps = [
        _ep("misconfig_app_hotel_res", None, round_id="r1"),
        _ep("revoke_auth_mongodb", None, round_id="r1"),
        _ep("user_unregistered_mongodb", None, round_id="r2"),
    ]
    assert set(fit_oracle(eps, "o1").values()) == {"mongo_auth"}
    tie = fit_oracle(eps[:2], "o1")
    assert set(tie.values()) == {"misconfig_app_hotel_res"}  # "m" < "mongo_auth"


# --- bootstrap ------------------------------------------------------------------------------------


def test_bootstrap_resamples_whole_clusters_and_is_deterministic():
    """Two clusters, one all-correct with 9 episodes, one all-wrong with 1: resampling clusters
    gives only {0/1... mixtures of whole clusters}, never an episode-level mixture."""
    eps = [Episode(f"a{i}", "fam_a", "r1") for i in range(9)] + [Episode("b0", "fam_b", "r1")]
    stat = lambda sample: sum(ep.family == "fam_a" for ep in sample) / len(sample)  # noqa: E731
    out = clustered_bootstrap(eps, stat, resamples=500, seed=1)
    assert out["clusters"] == 2
    again = clustered_bootstrap(eps, stat, resamples=500, seed=1)
    assert out == again
    assert out["ci"][0] == 0.0 and out["ci"][1] == 1.0  # only whole-cluster mixtures


def test_bootstrap_undefined_statistic_reports_none():
    assert clustered_bootstrap([], lambda s: None)["ci"] is None


def test_wilson_bounds():
    assert wilson(0, 0) is None
    lo, hi = wilson(9, 12)
    assert 0 < lo < 0.75 < hi < 1


def test_h3a_bands_follow_the_plan_thresholds():
    assert h3a_band(0.60) == "continue"
    assert h3a_band(0.39) == "kill"
    assert h3a_band(0.5).startswith("banda")
    assert h3a_band(None) == "nao medido"


# --- report -----------------------------------------------------------------------------------------


def test_h3b_is_not_measured_below_the_ambiguous_floor():
    eps = [_ep("misconfig_app_hotel_res", _record("app_image_misconfig"), round_id=f"r{i}") for i in range(H3B_MIN_AMBIGUOUS - 1)]
    report = score(eps, resamples=50)
    assert report["h3b"]["verdict"] == "nao medido: universo sem ambiguidade"


def test_not_measured_episodes_are_counted_never_scored():
    eps = [
        _ep("misconfig_app_hotel_res", _record("app_image_misconfig")),
        Episode("x", "wrong_bin_usage", "r1", not_measured="app nao assentou"),
    ]
    report = score(eps, resamples=50)
    assert report["counts"]["faults_not_measured"] == 1
    assert report["counts"]["faults_scored"] == 1
    assert report["h3a"]["estimate"] == 1.0


def test_false_alarm_handled_only_with_falso_positivo_and_no_action():
    eps = [
        Episode("fa1", "falso_alarme", "r1", kind=KIND_FALSE_ALARM, record=_record("falso_positivo")),
        Episode("fa2", "falso_alarme", "r1", kind=KIND_FALSE_ALARM, record=_record("falso_positivo", decision="agir", executed=[_undo("geo")])),
        Episode("fa3", "falso_alarme", "r1", kind=KIND_FALSE_ALARM, record=_record("unknown")),
    ]
    h3c = score(eps, resamples=10)["h3c"]
    assert (h3c["false_alarms"], h3c["handled"]) == (3, 1)


def test_recall_is_reported_both_ways():
    eps = [
        _ep("misconfig_app_hotel_res", _record("app_image_misconfig"), alert_fired=True),
        _ep("wrong_bin_usage", None, labels=None, alert_fired=False),
    ]
    recall = score(eps, resamples=10)["recall"]
    assert recall["fim_a_fim"] == {"num": 1, "den": 2, "rate": 0.5}
    assert recall["condicionado_a_alerta"] == {"num": 1, "den": 1, "rate": 1.0}


def test_urgency_and_disposition_always_carry_their_trivial_baseline():
    report = score([_ep("misconfig_app_hotel_res", _record("app_image_misconfig"))], resamples=10)
    assert report["urgencia"]["manchete"] is False and "baseline_sempre_p2" in report["urgencia"]
    assert "baseline_sempre_escala" in report["disposicao"]


# --- dry-run --------------------------------------------------------------------------------------

# Pinned before any scored round. A change to the table, the scorer or the synthetic generator
# changes this hash, so it can only land as a reviewed commit made before a scored round.
DRY_RUN_HASH = "1b9ff1c5a103c34d4d5353c3e6fda16df8ff41c95dd7f46dcc0cafae456ab3ae"


def test_synthetic_set_covers_every_admitted_family_and_the_edge_paths():
    eps = synthetic_episodes()
    assert {ep.family for ep in eps if ep.kind == "fault"} == set(FAULT_TABLE)
    assert any(ep.not_measured for ep in eps)
    assert any(ep.kind == KIND_FALSE_ALARM for ep in eps)


def test_dry_run_is_deterministic_and_matches_the_pre_registered_hash():
    report, digest = dry_run()
    assert dry_run()[1] == digest
    assert report["counts"]["faults_scored"] > 0
    assert report["h3b"]["ambiguous_episodes"] >= H3B_MIN_AMBIGUOUS
    assert {"tipo1_medo", "tipo2_afoiteza"} <= set(report["disposicao"]["modelo"])
    assert digest == DRY_RUN_HASH
