"""Tests for harness/triage.py: TriageRecord schema, the honesty (null+motivo) rule, and the
instrument-error-as-cause rejection."""

from __future__ import annotations

import copy

import pytest

from harness.triage import (
    INSTRUMENT_ERROR_SIGNATURES,
    TriageValidationError,
    cause_cites_instrument_error,
    validate_triage_record,
)

VALID_RECORD = {
    "episode_id": "ep-001",
    "alert_fingerprint": "abc123",
    "sintese": "recommendation deployment CrashLoopBackOff after image rollout",
    "causa_provavel": {"classe": "app_image_misconfig", "detalhe": "new image build crashes on start"},
    "blast_radius": [{"service": "recommendation", "severidade": "user_facing"}],
    "evidencia": [
        {
            "fonte": "k8s_api",
            "consulta": "get pods -n hotel",
            "timestamp": "2026-09-20T12:00:00Z",
            "trecho": "ImagePullBackOff",
            "fonte_suspeita": False,
        }
    ],
    "acao_proposta": {
        "tipo": "envelope",
        "comandos": ["kubectl rollout undo deployment/recommendation --to-revision=3 -n hotel"],
        "reversivel": True,
    },
    "acao_executada": None,
    "acao_executada_motivo": "read-back/rollback_state not implemented when this record was captured",
    "confianca": 0.85,
    "urgencia": "P2",
    "decisao": "agir",
    "t_alarme": "2026-09-20T11:58:00Z",
    "t_record": "2026-09-20T12:00:30Z",
    # motivo_escalacao is one of the honesty-rule fields, but a hand-written
    # fixture representing a clean episode (confianca=0.85 >= the 0.7 gate, no denials/loop/
    # timeout/evidence-gap modeled here) legitimately has zero triggers -- an empty list, not
    # null+motivo (see harness/triage.py's module docstring on why those two are different).
    "motivo_escalacao": [],
    # rollback_state describes the real C2 execution this fixture's
    # decisao="agir" implies -- a rollout undo, so comando_rollback is populated (unlike restart/
    # delete_pod, which are always null there).
    "rollback_state": {
        "tipo": "rollout_undo",
        "estado_pre_mutacao": "revisao vigente antes do undo: 2",
        "comando_rollback": "kubectl rollout undo deployment/recommendation -n hotel --to-revision=2",
        "motivo": "reversao e reaplicar a revisao 2, que estava vigente antes desta acao",
    },
    # acoes_executadas is the structured audit trail for the same
    # rollout undo decisao="agir" already implies above -- one entry, C2, matching the command in
    # rollback_state/acao_proposta.
    "acoes_executadas": [
        {
            "classe": "C2",
            "comando": "kubectl rollout undo deployment/recommendation --to-revision=3 -n hotel",
            "t": "2026-09-20T12:00:30Z",
            "read_back": "melhora confirmada: rollout complete, symptom target ready",
            "resultado": "executado, exit=0",
        }
    ],
    # acao_recomendada is always null+motivo -- the model does not emit it yet; harness/fault_table.py only scores it (see triage_assembly).
    "acao_recomendada": None,
    "acao_recomendada_motivo": (
        "the model does not emit a structured recommendation yet; harness/fault_table.py is the "
        "scoring key, never copied into the record"
    ),
}


def test_valid_record_passes():
    assert validate_triage_record(copy.deepcopy(VALID_RECORD)) == VALID_RECORD


def test_missing_required_field_rejected():
    record = copy.deepcopy(VALID_RECORD)
    del record["decisao"]
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


def test_unknown_cause_with_escalation_is_a_valid_honest_answer():
    """'causa_provavel=unknown with escalation is the CORRECT answer for a fault outside the
    trained families' -- the schema must accept it, not just tolerate it."""
    record = copy.deepcopy(VALID_RECORD)
    record["causa_provavel"] = {"classe": "unknown", "detalhe": "no matching signature in the fault table"}
    record["decisao"] = "escalar"
    record["acao_proposta"] = {"tipo": "escalate", "comandos": [], "reversivel": True}
    assert validate_triage_record(record)["decisao"] == "escalar"


def test_sintese_over_280_chars_rejected():
    record = copy.deepcopy(VALID_RECORD)
    record["sintese"] = "x" * 281
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


def test_null_field_without_motivo_rejected():
    """Honesty rule: 'a field not measurable in an episode comes out null with a reason, never a
    guess'."""
    record = copy.deepcopy(VALID_RECORD)
    record["evidencia"] = None
    with pytest.raises(TriageValidationError, match="evidencia_motivo"):
        validate_triage_record(record)


def test_null_field_with_empty_motivo_rejected():
    record = copy.deepcopy(VALID_RECORD)
    record["evidencia"] = None
    record["evidencia_motivo"] = "   "
    with pytest.raises(TriageValidationError, match="evidencia_motivo"):
        validate_triage_record(record)


def test_null_field_with_stated_motivo_accepted():
    record = copy.deepcopy(VALID_RECORD)
    record["evidencia"] = None
    record["evidencia_motivo"] = "prometheus range query timed out twice"
    assert validate_triage_record(record)["evidencia"] is None


def test_acao_executada_null_without_motivo_rejected():
    """acao_executada is an honesty-rule field (C2 dry-run gate: a C2 without rollback_state
    does not execute)."""
    record = copy.deepcopy(VALID_RECORD)
    del record["acao_executada_motivo"]
    with pytest.raises(TriageValidationError, match="acao_executada_motivo"):
        validate_triage_record(record)


def test_acao_executada_null_with_dry_run_motivo_accepted():
    record = copy.deepcopy(VALID_RECORD)
    record["acao_executada_motivo"] = "dry-run: rollback_state ainda nao implementado, ver item 2b-iii"
    assert validate_triage_record(record)["acao_executada"] is None


def test_quiet_hour_false_positive_shape_is_valid():
    """A quiet-hour episode is a false positive BY DEFINITION (control 1): no fault
    signal, decisao=silenciar_proposto, confianca/urgencia legitimately null with a stated
    reason."""
    record = copy.deepcopy(VALID_RECORD)
    record["decisao"] = "silenciar_proposto"
    record["confianca"] = None
    record["confianca_motivo"] = "no fault active, quiet-hour control window"
    record["urgencia"] = None
    record["urgencia_motivo"] = "no fault active, quiet-hour control window"
    assert validate_triage_record(record)["decisao"] == "silenciar_proposto"


@pytest.mark.parametrize("signature", INSTRUMENT_ERROR_SIGNATURES)
def test_instrument_error_signature_detected(signature):
    record = copy.deepcopy(VALID_RECORD)
    record["causa_provavel"] = {"classe": "unknown", "detalhe": f"tool reported: {signature}"}
    assert cause_cites_instrument_error(record) == signature


def test_instrument_error_as_cause_rejected_by_default():
    """Named test: 'simulated instrument error, a TriageRecord that cites it as the cause is
    rejected; assertion instrument_error_as_cause = 0'."""
    record = copy.deepcopy(VALID_RECORD)
    record["causa_provavel"] = {"classe": "unknown", "detalhe": "root cause: Metrics API not available"}
    with pytest.raises(TriageValidationError, match="instrument-error"):
        validate_triage_record(record)


def test_instrument_error_as_cause_allowed_once_for_the_documented_retry():
    record = copy.deepcopy(VALID_RECORD)
    record["causa_provavel"] = {"classe": "unknown", "detalhe": "root cause: Metrics API not available"}
    assert validate_triage_record(record, allow_instrument_error_cause=True)


def test_real_cluster_cause_not_mistaken_for_instrument_error():
    record = copy.deepcopy(VALID_RECORD)
    record["causa_provavel"] = {"classe": "app_image_misconfig", "detalhe": "geo image swapped to a build that crashes on start"}
    assert cause_cites_instrument_error(record) is None
    assert validate_triage_record(record)


# ---------------------------------------------------------------------------
# motivo_escalacao (escalation triggers).
# ---------------------------------------------------------------------------


def test_motivo_escalacao_accepts_multiple_codes():
    """One episode can fire more than one trigger at the same time."""
    record = copy.deepcopy(VALID_RECORD)
    record["motivo_escalacao"] = ["E1", "E6"]
    assert validate_triage_record(record)["motivo_escalacao"] == ["E1", "E6"]


def test_motivo_escalacao_null_without_motivo_rejected():
    record = copy.deepcopy(VALID_RECORD)
    record["motivo_escalacao"] = None
    with pytest.raises(TriageValidationError, match="motivo_escalacao_motivo"):
        validate_triage_record(record)


def test_motivo_escalacao_null_with_motivo_accepted():
    """The genuinely-unmeasured case: harness/triage_assembly.py's build_fallback_triage_record
    falls back to this shape only when there is no episode data at all to compute triggers from
    (the orchestrator crashed before any turns ran) -- see that function's own docstring."""
    record = copy.deepcopy(VALID_RECORD)
    record["motivo_escalacao"] = None
    record["motivo_escalacao_motivo"] = "orchestrator raised before the episode could finish"
    assert validate_triage_record(record)["motivo_escalacao"] is None


def test_motivo_escalacao_rejects_unknown_code():
    record = copy.deepcopy(VALID_RECORD)
    record["motivo_escalacao"] = ["E99"]
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


def test_motivo_escalacao_accepts_e3():
    """E3 is in the accepted enum (harness/triage.py's module docstring): the read-back machinery
    that grounds it exists (LiveClusterEnvironment.rollback_events),
    so the schema accepts it like any other trigger code."""
    record = copy.deepcopy(VALID_RECORD)
    record["motivo_escalacao"] = ["E3"]
    assert validate_triage_record(record)["motivo_escalacao"] == ["E3"]


def test_motivo_escalacao_rejects_duplicate_codes():
    record = copy.deepcopy(VALID_RECORD)
    record["motivo_escalacao"] = ["E1", "E1"]
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


def test_motivo_escalacao_missing_entirely_rejected():
    """motivo_escalacao is in NULLABLE_WITH_REASON_FIELDS, which the schema's `required` list
    always includes -- a record that omits the key entirely (not even null) must fail, same as
    every other honesty-rule field."""
    record = copy.deepcopy(VALID_RECORD)
    del record["motivo_escalacao"]
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


# ---------------------------------------------------------------------------
# rollback_state (the hard rollback rule).
# ---------------------------------------------------------------------------


def test_rollback_state_null_without_motivo_rejected():
    record = copy.deepcopy(VALID_RECORD)
    record["rollback_state"] = None
    with pytest.raises(TriageValidationError, match="rollback_state_motivo"):
        validate_triage_record(record)


def test_rollback_state_null_with_motivo_accepted():
    """The common case for an episode that escalated instead of acting, or whose C2 dry-run gate
    is still on (episode_runner.py's SSLM_C2_DRY_RUN default)."""
    record = copy.deepcopy(VALID_RECORD)
    record["rollback_state"] = None
    record["rollback_state_motivo"] = "dry-run: rollback_state ainda nao executado de verdade (SSLM_C2_DRY_RUN ligado), ver item 2b-iii"
    assert validate_triage_record(record)["rollback_state"] is None


def test_rollback_state_missing_entirely_rejected():
    record = copy.deepcopy(VALID_RECORD)
    del record["rollback_state"]
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


def test_rollback_state_restart_shape_with_null_comando_rollback_accepted():
    """restart is idempotent: comando_rollback is legitimately null even though
    rollback_state itself is populated -- distinct from the whole-field null+motivo case above."""
    record = copy.deepcopy(VALID_RECORD)
    record["rollback_state"] = {
        "tipo": "rollout_restart",
        "estado_pre_mutacao": "n/a: restart nao muda o spec do deployment",
        "comando_rollback": None,
        "motivo": "nenhuma acao de rollback necessaria: restart e idempotente",
    }
    assert validate_triage_record(record)["rollback_state"]["comando_rollback"] is None


def test_rollback_state_unknown_tipo_rejected():
    record = copy.deepcopy(VALID_RECORD)
    record["rollback_state"]["tipo"] = "not-a-real-type"
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


def test_rollback_state_missing_required_key_rejected():
    record = copy.deepcopy(VALID_RECORD)
    del record["rollback_state"]["motivo"]
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


# ---------------------------------------------------------------------------
# acoes_executadas (executed actions).
# ---------------------------------------------------------------------------


def test_acoes_executadas_accepts_empty_list():
    """An escalated episode that ran no C0/C1/C2 command at all: [] is the honest fact (harness/
    triage_assembly.py's module docstring), never null+motivo."""
    record = copy.deepcopy(VALID_RECORD)
    record["acoes_executadas"] = []
    assert validate_triage_record(record)["acoes_executadas"] == []


def test_acoes_executadas_null_without_motivo_rejected():
    record = copy.deepcopy(VALID_RECORD)
    record["acoes_executadas"] = None
    with pytest.raises(TriageValidationError, match="acoes_executadas_motivo"):
        validate_triage_record(record)


def test_acoes_executadas_null_with_motivo_accepted():
    """The genuinely-unmeasured case: harness/triage_assembly.py's build_fallback_triage_record
    falls back to this shape when there is no episode data at all to read (the orchestrator
    crashed before any turns ran)."""
    record = copy.deepcopy(VALID_RECORD)
    record["acoes_executadas"] = None
    record["acoes_executadas_motivo"] = "orchestrator raised before the episode could finish"
    assert validate_triage_record(record)["acoes_executadas"] is None


def test_acoes_executadas_missing_entirely_rejected():
    record = copy.deepcopy(VALID_RECORD)
    del record["acoes_executadas"]
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


def test_acoes_executadas_accepts_c0_and_c1_entries():
    record = copy.deepcopy(VALID_RECORD)
    record["acoes_executadas"] = [
        {
            "classe": "C0",
            "comando": "kubectl get pods -n hotel",
            "t": "2026-09-20T11:59:00Z",
            "read_back": "nao aplicavel: C0/C1 nao envolve mutacao a reconferir",
            "resultado": "executado, exit=0",
        },
        {
            "classe": "C1",
            "comando": "kubectl exec recommendation-2 -- cat /health",
            "t": "2026-09-20T11:59:30Z",
            "read_back": "nao aplicavel: C0/C1 nao envolve mutacao a reconferir",
            "resultado": "executado, exit=0",
        },
    ]
    assert len(validate_triage_record(record)["acoes_executadas"]) == 2


def test_acoes_executadas_rejects_unknown_classe():
    record = copy.deepcopy(VALID_RECORD)
    record["acoes_executadas"][0]["classe"] = "C3"
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


def test_acoes_executadas_rejects_entry_missing_required_key():
    record = copy.deepcopy(VALID_RECORD)
    del record["acoes_executadas"][0]["resultado"]
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


# ---------------------------------------------------------------------------
# acao_recomendada (recommended action).
# ---------------------------------------------------------------------------


def test_acao_recomendada_null_without_motivo_rejected():
    record = copy.deepcopy(VALID_RECORD)
    del record["acao_recomendada_motivo"]
    with pytest.raises(TriageValidationError, match="acao_recomendada_motivo"):
        validate_triage_record(record)


def test_acao_recomendada_null_with_motivo_accepted():
    """The only shape the assembly produces: the expected-action table is the scoring key and is
    never copied into the record (see harness/triage_assembly.py's honest-limits paragraph)."""
    record = copy.deepcopy(VALID_RECORD)
    assert validate_triage_record(record)["acao_recomendada"] is None


def test_acao_recomendada_missing_entirely_rejected():
    record = copy.deepcopy(VALID_RECORD)
    del record["acao_recomendada"]
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)


def test_acao_recomendada_populated_shape_accepted():
    """The schema shape the field takes once the model emits a structured recommendation --
    exercised here even though the assembly never produces it, so the shape itself is pinned by
    a test."""
    record = copy.deepcopy(VALID_RECORD)
    record["acao_recomendada"] = {
        "comandos": ["kubectl exec mongo-0 -- mongosh --eval \"db.grantRolesToUser(...)\""],
        "efeito_esperado": "restores the revoked role of the application user in mongo",
        "risco": "low: read-write command limited to the target user, no effect on other clients",
    }
    del record["acao_recomendada_motivo"]
    assert validate_triage_record(record)["acao_recomendada"]["comandos"]


def test_acao_recomendada_missing_required_key_rejected():
    record = copy.deepcopy(VALID_RECORD)
    record["acao_recomendada"] = {
        "comandos": ["kubectl ..."],
        "efeito_esperado": "restaura o acesso",
    }
    del record["acao_recomendada_motivo"]
    with pytest.raises(TriageValidationError):
        validate_triage_record(record)
