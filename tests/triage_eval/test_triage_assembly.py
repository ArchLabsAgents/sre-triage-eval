"""Tests for harness/triage_assembly.py: TriageRecord assembly from a finished episode, and the
honesty fallbacks that keep "emit a TriageRecord, always" true even when assembly itself cannot
honestly build a full record."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from harness import triage_assembly as ta
from harness.agent import EpisodeResult, TurnResult
from harness.envelope import ActionClass, EnvelopeDecision
from harness.fault_table import UNCLASSIFIED_CLASS
from harness.triage import validate_triage_record


@dataclass
class FakeSource:
    name: str
    collected: bool
    reason: str | None = None
    text: str = ""


@dataclass
class FakeContext:
    sources: list = field(default_factory=list)


@dataclass
class FakeEnvironment:
    executions: list = field(default_factory=list)
    denials: list = field(default_factory=list)
    # E4: `LiveClusterEnvironment.loop_detections` real attribute, mirrored
    # here so a test can set it directly instead of driving a full apply_action loop just to
    # produce one. `_build_motivo_escalacao` reads this via `getattr(..., default=[])` precisely
    # so a fake that predates this field (or a future Environment implementation) degrades to "no
    # E4 signal" instead of raising -- this dataclass field just makes the fake explicit too.
    loop_detections: list = field(default_factory=list)
    # E3: `LiveClusterEnvironment.rollback_events` real attribute,
    # mirrored here for the same reason `loop_detections` is -- `_e3_triggered` reads it via
    # `getattr(..., default=[])`.
    rollback_events: list = field(default_factory=list)


def _webhook_raw(*, fingerprint: str = "fp-1", starts_at: str | None = "2026-09-21T10:00:00Z") -> dict:
    alert = {"status": "firing", "fingerprint": fingerprint, "labels": {}, "annotations": {}}
    if starts_at is not None:
        alert["startsAt"] = starts_at
    return {"groupKey": "grp", "status": "firing", "receiver": "sslm-agent", "alerts": [alert]}


def _turn(action: dict | None) -> TurnResult:
    return TurnResult(output_tokens=1, schema_valid=action is not None, action=action, solved=False, observation="")


def _result(turns: list[TurnResult], *, e5_reason: str | None = None) -> EpisodeResult:
    return EpisodeResult(task_id="ep-1", solved=False, turns=turns, elapsed_seconds=1.0, e5_reason=e5_reason)


# A context with all 3 evidencia categories (E6) collected, so tests that are not
# specifically about E6 do not accidentally trip it.
def _full_evidence_context() -> FakeContext:
    return FakeContext(
        sources=[
            FakeSource("prometheus_range[0]", True, text='query: up{job="checkout"}\n[]'),
            FakeSource("k8s_pods", True, text="{}"),
            FakeSource("k8s_logs[checkout-2]", True, text="ok"),
        ]
    )


ESCALATE_ACTION = {
    "diagnosis": "unclear root cause, deferring to a human",
    "cause_class": "unknown",
    "action_type": "escalate",
    "target": "checkout",
    "command": None,
    "confidence": 0.2,
}

REMEDIATE_ACTION = {
    "diagnosis": "checkout deployment crashlooping after bad rollout",
    "cause_class": "app_image_misconfig",
    "action_type": "remediate",
    "target": "deployment/checkout",
    "command": "kubectl rollout undo deployment/checkout --to-revision=3",
    "confidence": 0.9,
}


# ---------------------------------------------------------------------------
# extract_t_alarme / alert_fingerprint
# ---------------------------------------------------------------------------


def test_extract_t_alarme_reads_starts_at():
    assert ta.extract_t_alarme(_webhook_raw(starts_at="2026-09-21T09:00:00Z"), fallback="fallback") == "2026-09-21T09:00:00Z"


def test_extract_t_alarme_falls_back_when_missing():
    assert ta.extract_t_alarme(_webhook_raw(starts_at=None), fallback="fallback-time") == "fallback-time"


# ---------------------------------------------------------------------------
# assemble_triage_record -- escalation path (no C2 executed)
# ---------------------------------------------------------------------------


def test_assemble_escalation_record_is_schema_valid_and_honest():
    context = FakeContext(sources=[])
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(ESCALATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-1",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
        t_record="2026-09-21T09:01:00Z",
    )

    validate_triage_record(dict(record))  # does not raise
    assert record["episode_id"] == "ep-1"
    assert record["alert_fingerprint"] == "fp-1"
    assert record["decisao"] == "escalar"
    assert record["causa_provavel"]["classe"] == "unknown"
    assert "unclear root cause" in record["causa_provavel"]["detalhe"]
    assert record["blast_radius"] is None
    assert record["blast_radius_motivo"]
    assert record["urgencia"] is None
    assert record["urgencia_motivo"]
    assert record["evidencia"] is None
    assert record["evidencia_motivo"]
    assert record["acao_proposta"] == {"tipo": "escalate", "comandos": [], "reversivel": True}
    assert record["acao_executada"] is None
    assert record["acao_executada_motivo"]
    assert record["confianca"] == pytest.approx(0.2)
    assert record["t_alarme"] == "2026-09-21T09:00:00Z"
    assert record["t_record"] == "2026-09-21T09:01:00Z"


def test_assemble_record_with_no_valid_action_is_honest_about_it():
    """Every turn malformed (schema rejection): the record still emits, decisao=escalar, and
    confianca/sintese/causa_provavel all say plainly that no action was ever parsed."""
    context = FakeContext(sources=[])
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(None), _turn(None)])

    record = ta.assemble_triage_record(
        episode_id="ep-2",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["decisao"] == "escalar"
    assert record["confianca"] is None
    assert record["confianca_motivo"]
    assert record["causa_provavel"]["classe"] == "unknown"
    assert "no valid model action" in record["causa_provavel"]["detalhe"]


# ---------------------------------------------------------------------------
# causa_provavel.classe comes from the model's cause_class
# ---------------------------------------------------------------------------


def _assemble_with(action):
    return ta.assemble_triage_record(
        episode_id="ep-classe",
        webhook_raw=_webhook_raw(),
        context=FakeContext(sources=[]),
        environment=FakeEnvironment(executions=[]),
        result=_result([_turn(action)]),
        t_alarme="2026-09-21T09:00:00Z",
    )


def test_classe_is_the_model_cause_class_verbatim():
    record = _assemble_with(REMEDIATE_ACTION)

    validate_triage_record(dict(record))
    assert record["causa_provavel"]["classe"] == "app_image_misconfig"


def test_escalating_with_a_specific_class_keeps_that_class():
    """Escalating is not the same answer as "unknown": before cause_class existed the assembly
    forced unknown on every escalation, so a model that escalated while naming the right class was
    scored as if it had no hypothesis at all."""
    record = _assemble_with({**ESCALATE_ACTION, "cause_class": "mongo_role_revoked"})

    validate_triage_record(dict(record))
    assert record["decisao"] == "escalar"
    assert record["causa_provavel"]["classe"] == "mongo_role_revoked"


@pytest.mark.parametrize("cause_class", [None, "not_a_class", UNCLASSIFIED_CLASS])
def test_action_without_a_valid_cause_class_gets_the_unclassified_marker(cause_class):
    """Only a replayed pre-field log can reach here (validation requires the field); it must
    score as errado (wrong) via the harness marker, never pass an invented class through."""
    action = {k: v for k, v in REMEDIATE_ACTION.items() if k != "cause_class"}
    if cause_class is not None:
        action["cause_class"] = cause_class

    record = _assemble_with(action)

    validate_triage_record(dict(record))
    assert record["causa_provavel"]["classe"] == UNCLASSIFIED_CLASS


# ---------------------------------------------------------------------------
# assemble_triage_record -- acted path (C2 executed, real or dry-run)
# ---------------------------------------------------------------------------


def test_assemble_record_for_real_c2_execution():
    rollback_state = {
        "tipo": "rollout_undo",
        "estado_pre_mutacao": "revisao vigente antes do undo: 2",
        "comando_rollback": "kubectl rollout undo deployment/checkout -n hotel --to-revision=2",
        "motivo": "reversao e reaplicar a revisao 2, que estava vigente antes desta acao",
    }
    context = FakeContext(sources=[])
    environment = FakeEnvironment(
        executions=[
            {
                "command": "kubectl rollout undo deployment/checkout --to-revision=3",
                "action_class": ActionClass.C2.value,
                "returncode": 0,
                "dry_run": False,
                "rollback_state": rollback_state,
                "read_back": {"ready": True, "detail": "rollout complete, symptom target ready"},
                "rollback_executed": False,
            }
        ],
        rollback_events=[
            {
                "command": "kubectl rollout undo deployment/checkout --to-revision=3",
                "improved": True,
                "rollback_executed": False,
            }
        ],
    )
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-3",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["decisao"] == "agir"
    assert record["acao_proposta"]["tipo"] == "envelope"
    assert record["acao_proposta"]["comandos"] == ["kubectl rollout undo deployment/checkout --to-revision=3"]
    assert record["acao_executada"] == {
        "comandos": ["kubectl rollout undo deployment/checkout --to-revision=3"],
        "read_back": "melhora confirmada: rollout complete, symptom target ready",
    }
    assert "acao_executada_motivo" not in record
    # rollback_state travels on the record, describing the same real C2.
    assert record["rollback_state"] == rollback_state
    assert "rollback_state_motivo" not in record
    # E3 does not fire here: only 1 rollback_event, and it improved -- the empty `context.sources`
    # above legitimately trips E6 instead (no evidence collected), out of scope for this test.
    assert "E3" not in record["motivo_escalacao"]


def test_assemble_record_after_remediate_then_escalate_describes_the_remediate_turn():
    """The agent loop keeps running after a C2 action (real or
    dry-run) until the model escalates or hits max_turns, so a turn sequence of
    remediate -> escalate must still have causa_provavel/sintese/confianca describe the
    remediate turn (the same one decisao="agir"/acao_executada already describe), not the
    later escalate turn that has nothing to do with the mutation that actually ran."""
    context = FakeContext(sources=[])
    environment = FakeEnvironment(
        executions=[
            {
                "command": "kubectl rollout undo deployment/checkout --to-revision=3",
                "action_class": ActionClass.C2.value,
                "returncode": 0,
                "dry_run": False,
            }
        ]
    )
    result = _result([_turn(REMEDIATE_ACTION), _turn(ESCALATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-5",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["decisao"] == "agir"
    assert record["acao_executada"]["comandos"] == ["kubectl rollout undo deployment/checkout --to-revision=3"]
    assert "checkout deployment crashlooping" in record["causa_provavel"]["detalhe"]
    assert "checkout deployment crashlooping" in record["sintese"]
    assert record["confianca"] == pytest.approx(0.9)


def test_assemble_record_for_c2_dry_run_marks_acao_executada_null_with_dry_run_motivo():
    """The C2 dry-run gate: acao_proposta is filled normally, but
    acao_executada must be null with the exact dry-run motivo -- this is the assertion the C2
    dry-run gate exists to make true."""
    context = FakeContext(sources=[])
    environment = FakeEnvironment(
        executions=[
            {
                "command": "kubectl rollout restart deployment/checkout",
                "action_class": ActionClass.C2.value,
                "returncode": None,
                "dry_run": True,
            }
        ]
    )
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-4",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["decisao"] == "agir"  # the harness DID decide to act; the gate only blocked execution
    assert record["acao_proposta"]["tipo"] == "envelope"
    assert record["acao_proposta"]["comandos"] == ["kubectl rollout restart deployment/checkout"]
    assert record["acao_executada"] is None
    assert record["acao_executada_motivo"] == "dry-run: rollback_state ainda nao implementado, ver item 2b-iii"
    assert record["rollback_state"] is None
    assert record["rollback_state_motivo"] == (
        "dry-run: rollback_state ainda nao executado de verdade (SSLM_C2_DRY_RUN ligado), ver item 2b-iii"
    )


# ---------------------------------------------------------------------------
# assemble_triage_record -- evidencia from context.sources
# ---------------------------------------------------------------------------


def test_assemble_record_builds_evidencia_from_collected_sources():
    context = FakeContext(
        sources=[
            FakeSource("alertmanager_payload", True, text="{}"),  # never becomes evidencia
            FakeSource("prometheus_range[0]", True, text='query: up{job="checkout"}\n[{"values": []}]'),
            FakeSource("k8s_pods", True, text='{"items": []}'),
            FakeSource("k8s_logs[checkout-2]", True, text="crashloop traceback"),
            FakeSource("k8s_events", False, reason="transport failed"),
        ]
    )
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(ESCALATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-5",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
        t_record="2026-09-21T09:01:00Z",
    )

    validate_triage_record(dict(record))
    evidencia = record["evidencia"]
    assert evidencia is not None
    fontes = {e["fonte"] for e in evidencia}
    assert fontes == {"prometheus", "k8s_api", "logs"}
    prom_entry = next(e for e in evidencia if e["fonte"] == "prometheus")
    assert prom_entry["consulta"] == 'up{job="checkout"}'
    assert prom_entry["timestamp"] == "2026-09-21T09:01:00Z"
    for entry in evidencia:
        assert entry["fonte_suspeita"] is False
        assert len(entry["trecho"]) <= ta.EVIDENCE_EXCERPT_MAX_CHARS


def test_assemble_record_evidencia_null_when_nothing_collected():
    context = FakeContext(
        sources=[
            FakeSource("k8s_pods", False, reason="NotFound"),
            FakeSource("prometheus_range[0]", False, reason="timeout"),
        ]
    )
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(ESCALATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-6",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["evidencia"] is None
    assert "NotFound" in record["evidencia_motivo"] and "timeout" in record["evidencia_motivo"]


def test_evidencia_trecho_is_capped():
    context = FakeContext(sources=[FakeSource("k8s_pods", True, text="x" * 10_000)])
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(ESCALATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-7",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    assert len(record["evidencia"][0]["trecho"]) == ta.EVIDENCE_EXCERPT_MAX_CHARS


# ---------------------------------------------------------------------------
# build_fallback_triage_record
# ---------------------------------------------------------------------------


def test_build_fallback_triage_record_is_schema_valid():
    record = ta.build_fallback_triage_record(
        episode_id="ep-crash",
        alert_fingerprint="fp-9",
        t_alarme="2026-09-21T09:00:00Z",
        t_record="2026-09-21T09:02:00Z",
        reason="orchestrator raised before the episode could finish: RuntimeError('boom')",
    )

    validate_triage_record(dict(record))
    assert record["decisao"] == "escalar"
    assert record["episode_id"] == "ep-crash"
    assert record["alert_fingerprint"] == "fp-9"
    for motivo_field in (
        "blast_radius_motivo",
        "evidencia_motivo",
        "confianca_motivo",
        "urgencia_motivo",
        "acao_executada_motivo",
        "rollback_state_motivo",
        "acoes_executadas_motivo",
        "acao_recomendada_motivo",
    ):
        assert record[motivo_field] == "orchestrator raised before the episode could finish: RuntimeError('boom')"


def test_assemble_triage_record_never_raises_even_on_pathological_input():
    """A best-effort guarantee, not a formal proof: feed assemble_triage_record something that
    would make a naive implementation choke (a turn whose action is present but has no
    diagnosis/confidence fields at all) and confirm it still returns a valid record."""
    context = FakeContext(sources=[])
    environment = FakeEnvironment(executions=[])
    result = _result([_turn({"action_type": "escalate"})])

    record = ta.assemble_triage_record(
        episode_id="ep-8",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))


# ---------------------------------------------------------------------------
# motivo_escalacao (escalation triggers). Each E-trigger is exercised in isolation (a synthetic
# episode that produces exactly that one signal and no other) plus one test combining several at
# once ("one episode can fire more than one trigger at the same time").
# ---------------------------------------------------------------------------


def test_motivo_escalacao_empty_when_no_trigger_fires():
    """The clean-episode baseline every other test in this section diffs against: high
    confidence, no denials, no loop, no time/turn cap, full evidence, no instrument-error
    citations. An empty list is a real, honest fact here (harness/triage.py's module docstring:
    "checked, zero triggers fired"), not an unmeasured value."""
    context = _full_evidence_context()
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-clean",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == []
    assert record["causa_provavel"]["detalhe"] == REMEDIATE_ACTION["diagnosis"]  # not a fallback


def test_motivo_escalacao_e1_low_confidence():
    context = _full_evidence_context()
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(ESCALATE_ACTION)])  # confidence 0.2 < CONFIDENCE_GATE (0.7)

    record = ta.assemble_triage_record(
        episode_id="ep-e1",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == ["E1"]


def test_motivo_escalacao_e2_from_c3_denial():
    context = _full_evidence_context()
    environment = FakeEnvironment(
        executions=[],
        denials=[EnvelopeDecision(action_class=ActionClass.C3, allowed=False, reason="fail-closed default")],
    )
    result = _result([_turn(REMEDIATE_ACTION)])  # confidence 0.9, keeps E1 out

    record = ta.assemble_triage_record(
        episode_id="ep-e2",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == ["E2"]


def test_motivo_escalacao_c1_denial_does_not_trigger_e2():
    """E2's own wording: 'the needed action is C3, or a C2 denied by RBAC/gate' -- literally
    those two, not 'any denial'. A denied C1 (e.g. no prior C0 this episode) is real
    containment but is not what the table names for E2."""
    context = _full_evidence_context()
    environment = FakeEnvironment(
        executions=[],
        denials=[EnvelopeDecision(action_class=ActionClass.C1, allowed=False, reason="C1 requires a prior C0 call")],
    )
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-c1-denial",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == []


def test_motivo_escalacao_e4_from_loop_detections():
    context = _full_evidence_context()
    environment = FakeEnvironment(
        executions=[],
        loop_detections=[{"normalized_command": "kubectl get pod/checkout-<pod> -n hotel", "error_signature": "..."}],
    )
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-e4",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == ["E4"]


def test_motivo_escalacao_e5_from_harness_owned_cap():
    context = _full_evidence_context()
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(REMEDIATE_ACTION)], e5_reason="turn cap reached (8 turns) without a final decision (E5)")

    record = ta.assemble_triage_record(
        episode_id="ep-e5",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == ["E5"]


def test_motivo_escalacao_e6_two_of_three_sources_failed():
    """E6: '>= 2 of the 3 sources ... failed to collect'. Only k8s_api collected here;
    prometheus and logs never even produced a source (the trigger does not distinguish "never
    attempted" from "attempted and failed" for this trigger -- see `_e6_triggered`'s docstring)."""
    context = FakeContext(sources=[FakeSource("k8s_pods", True, text="{}")])
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-e6",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == ["E6"]


def test_motivo_escalacao_e6_not_triggered_when_only_one_source_failed():
    context = FakeContext(
        sources=[
            FakeSource("prometheus_range[0]", True, text="query: up\n[]"),
            FakeSource("k8s_pods", True, text="{}"),
            FakeSource("k8s_events", False, reason="transport failed"),  # not one of the 3 categories
            FakeSource("k8s_logs[checkout-2]", False, reason="pod recovered before logs were read"),
        ]
    )
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-e6-partial",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == []


INSTRUMENT_ERROR_ACTION_1 = {
    "diagnosis": "tool reported: Metrics API not available, investigating further",
    "action_type": "inspect",
    "target": "checkout",
    "command": "kubectl top pod",
    "confidence": 0.5,
}

INSTRUMENT_ERROR_ACTION_2 = {
    "diagnosis": "root cause: Metrics API not available",
    "action_type": "escalate",
    "target": "checkout",
    "command": None,
    "confidence": 0.3,
}


def test_motivo_escalacao_e7_not_triggered_on_first_instrument_error_citation():
    """The 'warned' occurrence ('retry once with a warning'): a SINGLE turn citing an
    instrument-error signature this episode does not raise E7, and -- because
    `assemble_triage_record` allows the one-shot exception through `validate_triage_record`'s
    `allow_instrument_error_cause` -- the record is NOT replaced by the generic fallback either;
    the real diagnosis text survives."""
    context = _full_evidence_context()
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(INSTRUMENT_ERROR_ACTION_2)])

    record = ta.assemble_triage_record(
        episode_id="ep-e7-warn",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    assert "E7" not in record["motivo_escalacao"]
    assert record["causa_provavel"]["detalhe"] == INSTRUMENT_ERROR_ACTION_2["diagnosis"]  # not a fallback
    validate_triage_record(dict(record), allow_instrument_error_cause=True)


def test_motivo_escalacao_e7_triggers_and_forces_fallback_on_second_citation():
    """'A repeat escalates with E7'. A SECOND turn (anywhere in the episode) citing
    an instrument-error signature both raises E7 and -- because the FINAL turn is one of the
    offenders -- forces `assemble_triage_record`'s generic validation-failure fallback, same as
    any other schema violation would. `motivo_escalacao` on the fallback record still carries the
    real, already-computed list (E7 included), never null+motivo: see
    `build_fallback_triage_record`'s docstring on why those two null cases differ."""
    context = _full_evidence_context()
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(INSTRUMENT_ERROR_ACTION_1), _turn(INSTRUMENT_ERROR_ACTION_2)])

    record = ta.assemble_triage_record(
        episode_id="ep-e7-escalate",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert "E7" in record["motivo_escalacao"]
    assert record["decisao"] == "escalar"
    # the fallback replaced the real diagnosis text -- proof the generic fallback path was taken,
    # not the normal assembly path.
    assert record["causa_provavel"]["detalhe"] != INSTRUMENT_ERROR_ACTION_2["diagnosis"]


def test_motivo_escalacao_e7_second_citation_on_an_earlier_non_final_turn_still_counts():
    """The count is over ALL turns' diagnoses, not just the final one: here the final turn's own
    diagnosis does not cite an instrument-error signature, but two EARLIER turns did, so E7 must
    still fire (the model warned itself twice and only escalated cleanly on the third turn)."""
    context = _full_evidence_context()
    environment = FakeEnvironment(executions=[])
    result = _result(
        [
            _turn(INSTRUMENT_ERROR_ACTION_1),
            _turn(INSTRUMENT_ERROR_ACTION_1),
            _turn(REMEDIATE_ACTION),
        ]
    )

    record = ta.assemble_triage_record(
        episode_id="ep-e7-earlier-turns",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert "E7" in record["motivo_escalacao"]
    # the final action itself never cited an instrument-error signature, so validation never had
    # a reason to replace it with the generic fallback.
    assert record["causa_provavel"]["detalhe"] == REMEDIATE_ACTION["diagnosis"]


def test_motivo_escalacao_e3_from_two_no_improvement_rollback_events():
    """E3: 'no progress: 2 C2 actions executed and read-back without improvement'."""
    context = _full_evidence_context()
    environment = FakeEnvironment(
        executions=[],
        rollback_events=[
            {"command": "kubectl rollout undo deployment/checkout --to-revision=3", "improved": False, "rollback_executed": True},
            {"command": "kubectl scale deployment/checkout --replicas=4", "improved": False, "rollback_executed": True},
        ],
    )
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-e3",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == ["E3"]


def test_motivo_escalacao_e3_not_triggered_by_a_single_no_improvement_event():
    """The cap is 2 C2 actions per episode (harness.envelope.C2_CALL_CAP): a SINGLE no-improvement
    execution is not yet 'no progress' by the trigger's own number."""
    context = _full_evidence_context()
    environment = FakeEnvironment(
        executions=[],
        rollback_events=[
            {"command": "kubectl rollout undo deployment/checkout --to-revision=3", "improved": False, "rollback_executed": True}
        ],
    )
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-e3-single",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == []


def test_motivo_escalacao_e3_counts_inconclusive_read_back_as_no_improvement():
    """`improved=None` (delete pod's honest limit, episode_runner.py) counts the same as `False`
    for E3 -- the conservative reading documented on `_e3_triggered`."""
    context = _full_evidence_context()
    environment = FakeEnvironment(
        executions=[],
        rollback_events=[
            {"command": "kubectl delete pod checkout-2", "improved": None, "rollback_executed": False},
            {"command": "kubectl delete pod checkout-3", "improved": None, "rollback_executed": False},
        ],
    )
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-e3-inconclusive",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == ["E3"]


def test_motivo_escalacao_multiple_triggers_simultaneously():
    """One episode can fire more than one trigger at the same time."""
    context = FakeContext(sources=[FakeSource("k8s_pods", True, text="{}")])  # E6: only 1 of 3
    environment = FakeEnvironment(
        executions=[],
        denials=[EnvelopeDecision(action_class=ActionClass.C3, allowed=False, reason="fail-closed default")],  # E2
    )
    result = _result([_turn(ESCALATE_ACTION)])  # confidence 0.2 < 0.7: E1

    record = ta.assemble_triage_record(
        episode_id="ep-multi",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert set(record["motivo_escalacao"]) == {"E1", "E2", "E6"}


def test_build_fallback_triage_record_defaults_motivo_escalacao_to_null_with_reason():
    """The one path with no episode data to compute triggers from at all (episode_runner.py's
    orchestrator-raised-before-any-turns except branch): `motivo_escalacao` is null+motivo, not a
    fabricated empty list -- see the function's own docstring on why this differs from the
    "measured zero triggers" case."""
    record = ta.build_fallback_triage_record(
        episode_id="ep-crash",
        alert_fingerprint="fp-9",
        t_alarme="2026-09-21T09:00:00Z",
        t_record="2026-09-21T09:02:00Z",
        reason="orchestrator raised before the episode could finish",
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] is None
    assert record["motivo_escalacao_motivo"] == "orchestrator raised before the episode could finish"


# ---------------------------------------------------------------------------
# _build_rollback_state (the hard rollback rule).
# ---------------------------------------------------------------------------


def test_build_rollback_state_null_when_no_c2_executed():
    environment = FakeEnvironment(executions=[])
    rollback_state, motivo = ta._build_rollback_state(environment)
    assert rollback_state is None
    assert "escalated" in motivo or "read-only" in motivo


def test_build_rollback_state_picks_the_last_real_c2():
    """Two real C2 executions this episode: rollback_state reports the LAST one, same convention
    as `_final_action`/`_build_acao_executada`."""
    first_rollback_state = {
        "tipo": "rollout_undo",
        "estado_pre_mutacao": "revisao vigente antes do undo: 2",
        "comando_rollback": "kubectl rollout undo deployment/checkout -n hotel --to-revision=2",
        "motivo": "reversao e reaplicar a revisao 2",
    }
    second_rollback_state = {
        "tipo": "scale",
        "estado_pre_mutacao": "replicas antes da mutacao: 3",
        "comando_rollback": "kubectl scale deployment/checkout -n hotel --replicas=3",
        "motivo": "reversao e escalar de volta para a contagem observada antes desta acao (3)",
    }
    environment = FakeEnvironment(
        executions=[
            {
                "command": "kubectl rollout undo deployment/checkout --to-revision=3",
                "action_class": ActionClass.C2.value,
                "returncode": 0,
                "dry_run": False,
                "rollback_state": first_rollback_state,
                "read_back": {"ready": False, "detail": "not ready"},
                "rollback_executed": True,
            },
            {
                "command": "kubectl scale deployment/checkout --replicas=4",
                "action_class": ActionClass.C2.value,
                "returncode": 0,
                "dry_run": False,
                "rollback_state": second_rollback_state,
                "read_back": {"ready": True, "detail": "rollout complete"},
                "rollback_executed": False,
            },
        ],
    )
    rollback_state, motivo = ta._build_rollback_state(environment)
    assert rollback_state == second_rollback_state
    assert motivo is None


# ---------------------------------------------------------------------------
# _build_acoes_executadas / acoes_executadas on the assembled record.
# ---------------------------------------------------------------------------


def test_acoes_executadas_empty_when_no_command_ran():
    """An escalated episode with no C0/C1/C2 command at all: [] is the honest fact, not null+
    motivo (module docstring's "checked, found zero" convention)."""
    context = FakeContext(sources=[])
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(ESCALATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-ae-1",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
        t_record="2026-09-21T09:01:00Z",
    )

    validate_triage_record(dict(record))
    assert record["acoes_executadas"] == []
    assert "acoes_executadas_motivo" not in record


def test_acoes_executadas_lists_c0_c1_and_c2_entries_in_order():
    context = FakeContext(sources=[])
    environment = FakeEnvironment(
        executions=[
            {
                "command": "kubectl get pods -n hotel",
                "action_class": ActionClass.C0.value,
                "returncode": 0,
                "dry_run": False,
            },
            {
                "command": "kubectl exec checkout-2 -- cat /health",
                "action_class": ActionClass.C1.value,
                "returncode": 0,
                "dry_run": False,
            },
            {
                "command": "kubectl rollout undo deployment/checkout --to-revision=3",
                "action_class": ActionClass.C2.value,
                "returncode": 0,
                "dry_run": False,
                "rollback_state": {
                    "tipo": "rollout_undo",
                    "estado_pre_mutacao": "revisao vigente antes do undo: 2",
                    "comando_rollback": "kubectl rollout undo deployment/checkout -n hotel --to-revision=2",
                    "motivo": "reversao e reaplicar a revisao 2",
                },
                "read_back": {"ready": True, "detail": "rollout complete, symptom target ready"},
                "rollback_executed": False,
            },
        ]
    )
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-ae-2",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
        t_record="2026-09-21T09:01:00Z",
    )

    validate_triage_record(dict(record))
    acoes = record["acoes_executadas"]
    assert [a["classe"] for a in acoes] == ["C0", "C1", "C2"]
    assert acoes[0]["comando"] == "kubectl get pods -n hotel"
    assert acoes[0]["t"] == "2026-09-21T09:01:00Z"  # honest limit: t_record, not a real per-command time
    assert acoes[0]["read_back"] == "nao aplicavel: C0/C1 nao envolve mutacao a reconferir"
    assert acoes[0]["resultado"] == "executado, exit=0"
    assert acoes[1]["classe"] == "C1"
    assert acoes[1]["read_back"] == "nao aplicavel: C0/C1 nao envolve mutacao a reconferir"
    assert acoes[2]["classe"] == "C2"
    assert acoes[2]["read_back"] == "melhora confirmada: rollout complete, symptom target ready"
    assert acoes[2]["resultado"] == "executado, exit=0"


def test_acoes_executadas_dry_run_c2_entry_says_no_apiserver_correlate():
    context = FakeContext(sources=[])
    environment = FakeEnvironment(
        executions=[
            {
                "command": "kubectl rollout restart deployment/checkout",
                "action_class": ActionClass.C2.value,
                "returncode": None,
                "dry_run": True,
            }
        ]
    )
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-ae-3",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
        t_record="2026-09-21T09:01:00Z",
    )

    validate_triage_record(dict(record))
    acao = record["acoes_executadas"][0]
    assert "DRY-RUN" in acao["resultado"]
    assert "sem correlato no audit log" in acao["resultado"]
    assert acao["read_back"] == "nao aplicavel: dry-run, C2 classificado mas nao executado (SSLM_C2_DRY_RUN ligado)"


def test_acoes_executadas_failed_c2_mutation_has_no_read_back():
    context = FakeContext(sources=[])
    environment = FakeEnvironment(
        executions=[
            {
                "command": "kubectl rollout undo deployment/checkout --to-revision=3",
                "action_class": ActionClass.C2.value,
                "returncode": 1,
                "dry_run": False,
                "rollback_state": {
                    "tipo": "rollout_undo",
                    "estado_pre_mutacao": "revisao vigente antes do undo: 2",
                    "comando_rollback": "kubectl rollout undo deployment/checkout -n hotel --to-revision=2",
                    "motivo": "reversao e reaplicar a revisao 2",
                },
                "read_back": None,
            }
        ]
    )
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-ae-4",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
        t_record="2026-09-21T09:01:00Z",
    )

    validate_triage_record(dict(record))
    acao = record["acoes_executadas"][0]
    assert acao["resultado"] == "executado, exit=1"
    assert acao["read_back"].startswith("read-back nao executado")


def test_build_fallback_triage_record_sets_acoes_executadas_and_acao_recomendada_null_with_reason():
    record = ta.build_fallback_triage_record(
        episode_id="ep-crash",
        alert_fingerprint="fp-9",
        t_alarme="2026-09-21T09:00:00Z",
        t_record="2026-09-21T09:02:00Z",
        reason="orchestrator raised before the episode could finish: RuntimeError('boom')",
    )

    validate_triage_record(dict(record))
    assert record["acoes_executadas"] is None
    assert record["acoes_executadas_motivo"] == "orchestrator raised before the episode could finish: RuntimeError('boom')"
    assert record["acao_recomendada"] is None
    assert record["acao_recomendada_motivo"] == "orchestrator raised before the episode could finish: RuntimeError('boom')"


# ---------------------------------------------------------------------------
# acao_recomendada: always null+motivo.
# ---------------------------------------------------------------------------


def test_acao_recomendada_always_null_with_motivo_in_assembled_record():
    context = _full_evidence_context()
    environment = FakeEnvironment(executions=[])
    result = _result([_turn(REMEDIATE_ACTION)])

    record = ta.assemble_triage_record(
        episode_id="ep-ar-1",
        webhook_raw=_webhook_raw(),
        context=context,
        environment=environment,
        result=result,
        t_alarme="2026-09-21T09:00:00Z",
    )

    validate_triage_record(dict(record))
    assert record["acao_recomendada"] is None
    assert "harness/fault_table.py" in record["acao_recomendada_motivo"]


def test_build_fallback_triage_record_preserves_a_real_motivo_escalacao_when_given():
    record = ta.build_fallback_triage_record(
        episode_id="ep-invalid",
        alert_fingerprint="fp-9",
        t_alarme="2026-09-21T09:00:00Z",
        t_record="2026-09-21T09:02:00Z",
        reason="assembly produced an invalid TriageRecord",
        motivo_escalacao=["E1", "E7"],
    )

    validate_triage_record(dict(record))
    assert record["motivo_escalacao"] == ["E1", "E7"]
    assert "motivo_escalacao_motivo" not in record
