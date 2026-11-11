"""TriageRecord: the schema and the metric for the 24/7 continuous scenario.

Emitted once per triage episode (fault-driven or a quiet-hour false positive), always, including
when the episode escalates instead of acting. Decoded under the same schema-constrained-decoding
mechanism already built for config B/C (harness/schema.py): this is a new schema, not new infra.

Two rules are enforced here in code rather than left as prose:
  - Honesty rule: a field that could not be measured in an episode is `null` with a sibling
    `<field>_motivo` explaining why -- never a guess, never silently coerced to a default. See
    NULLABLE_WITH_REASON_FIELDS and `validate_triage_record`.
  - Instrument-error-as-cause rejection (C1 gate, failure modes F2/F5): a TriageRecord whose
    `causa_provavel.detalhe` cites a known tool-error signature is rejected, not scored as a
    diagnosis, so the harness can retry once with a warning instead of letting the model diagnose
    its own tooling as the cluster's root cause.

`motivo_escalacao` is the list of E1-E7 escalation trigger codes this episode fired,
deterministically, never by text heuristic -- each code's own ground truth lives in harness data
(`harness.envelope.CONFIDENCE_GATE` for E1, `EnvelopeDecision` denials for E2,
`LiveClusterEnvironment.rollback_events` for E3, `LiveClusterEnvironment.loop_detections` for E4,
`EpisodeResult.e5_triggered` for E5, `EpisodeContext.sources` coverage for E6, and turn-by-turn
`cause_cites_instrument_error` recurrence for E7 -- see `harness/triage_assembly.py`'s
`_build_motivo_escalacao` for how each is computed). It is in NULLABLE_WITH_REASON_FIELDS, but its
null case means something different from blast_radius/urgencia's "not wired into this codebase":
`harness/triage_assembly.py`'s normal path can ALWAYS compute this field (the seven signals above
are all things the harness already observed), so in that path it is a list, possibly empty (a
real fact: "checked, zero triggers fired", not an unmeasured value). It is only null+motivo in the
one path that has no episode data to check at all -- `build_fallback_triage_record` called from
`harness/episode_runner.py`'s `except` branch when the orchestrator raised before an episode ever
produced turns.

**E3** ("no progress: 2 C2 actions executed and read-back without improvement") is accepted only
because the read-back machinery that grounds it exists (`harness/episode_runner.py`'s
`LiveClusterEnvironment._execute_c2_with_rollback`). Accepting "E3" without any real measurement
backing it would have been exactly the kind of ungrounded value the honesty rule forbids.

`rollback_state` (the hard rollback rule): the pre-mutation state and reversal command a real C2
execution registered BEFORE it ran, or `null` with a reason when no real mutation happened this
episode (escalated, read-only diagnosis only, or the C2 dry-run gate is still on). It is in
NULLABLE_WITH_REASON_FIELDS for the same reason `acao_executada` is: a field this schema can always
either populate or explain, never silently omit.

The escalation package's other two fields:
  - `acoes_executadas`: the structured {classe, comando, t, read_back, resultado} list, one entry
    per C0/C1/C2 command this episode ran (`LiveClusterEnvironment.executions`). Unlike
    `motivo_escalacao`'s empty-list case, this one is genuinely ALWAYS computable in the normal
    assembly path (the harness already recorded every command it ran, or it ran none) -- see
    `harness/triage_assembly.py`'s `_build_acoes_executadas` for why an empty list here is a real
    fact ("checked, zero commands ran"), never a stand-in for "could not measure". It is
    null+motivo only in the one path with no episode data to read at all (`build_fallback_
    triage_record`'s orchestrator-raised-before-any-turns branch), the same distinction drawn
    above for `motivo_escalacao`.
  - `acao_recomendada`: always null+motivo. Its ground truth exists
    (`harness/fault_table.py`'s `FaultRow.expected_action`), but that is what a proposal is
    SCORED against, not a value to copy into the record: the model's action schema does not emit
    a structured recommendation yet, and filling it from the table would put the answer key in
    the answer. It stays in NULLABLE_WITH_REASON_FIELDS until the model produces it.

`causa_provavel.classe` is closed to the fault table's taxonomy: the schema enum is
`harness.fault_table.TRIAGE_CLASS_ENUM` (fine classes of the admitted families, distractors,
`unknown`, `falso_positivo`, and the harness-only `unclassified`), imported from there so the two
cannot drift apart.

Record keys are Portuguese and kept for compatibility with published run data. Glossary:
sintese = summary, causa_provavel = probable cause (classe = class, detalhe = detail),
blast_radius.severidade = severity, evidencia = evidence (fonte = source, consulta = query,
trecho = excerpt, fonte_suspeita = suspect source), acao_proposta = proposed action (tipo = type,
comandos = commands, reversivel = reversible), acao_executada = executed action,
confianca = confidence, urgencia = urgency, decisao = decision (agir = act, escalar = escalate,
silenciar_proposto = propose silencing), t_alarme = alarm time, motivo_escalacao = escalation
triggers, rollback_state.estado_pre_mutacao = pre-mutation state, comando_rollback = rollback
command, acoes_executadas = executed actions (resultado = result), acao_recomendada = recommended
action (efeito_esperado = expected effect, risco = risk), `<field>_motivo` = reason the field is
null. Class values: falso_positivo = false positive.
"""

from __future__ import annotations

from typing import Any

import jsonschema

from harness.fault_table import TRIAGE_CLASS_ENUM

# Fields where `null` is a legitimate value, but only paired with a stated reason -- the
# honesty rule ("not measured comes out None with a reason, never zero or a guess").
# `causa_provavel.classe` is a closed enum owned by harness/fault_table.py, versioned and
# committed before the round it scores, never edited inside one.
#
# `acao_executada` is here because of the C2 dry-run gate (rollback rule: "a C2 without
# rollback_state does not execute"): while the gate is on, every C2 the envelope allows is
# dry-run only, so `acao_executada` is null on every record that acted; the honesty rule applies
# to it the same way it applies to the other fields, instead of silently allowing a bare null.
NULLABLE_WITH_REASON_FIELDS = (
    "blast_radius",
    "evidencia",
    "confianca",
    "urgencia",
    "acao_executada",
    "motivo_escalacao",
    "rollback_state",
    "acoes_executadas",
    "acao_recomendada",
)

# Escalation trigger codes this schema accepts. E3 is accepted because
# `LiveClusterEnvironment.rollback_events` gives it real ground truth to be measured against
# (module docstring).
ESCALATION_TRIGGER_CODES = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")

# Versioned list of tool-error signatures (C1 gate): output that matches one of
# these is tool failure, not cluster signal. Extend this list only with a reviewed change, never
# ad hoc inside a round -- it is what makes "instrument_error_as_cause = 0" a real assertion.
INSTRUMENT_ERROR_SIGNATURES = (
    "Metrics API not available",
    "prometheus query timeout",
    "context deadline exceeded",
    "404 health endpoint not found",
)

TRIAGE_RECORD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "episode_id",
        "alert_fingerprint",
        "sintese",
        "causa_provavel",
        "acao_proposta",
        "decisao",
        "t_alarme",
        "t_record",
    ]
    + list(NULLABLE_WITH_REASON_FIELDS),
    "properties": {
        "episode_id": {"type": "string", "minLength": 1},
        "alert_fingerprint": {"type": "string", "minLength": 1},
        "sintese": {"type": "string", "maxLength": 280},
        "causa_provavel": {
            "type": "object",
            "required": ["classe", "detalhe"],
            "properties": {
                "classe": {"type": "string", "enum": list(TRIAGE_CLASS_ENUM)},
                "detalhe": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "blast_radius": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "required": ["service", "severidade"],
                "properties": {
                    "service": {"type": "string", "minLength": 1},
                    "severidade": {"type": "string", "enum": ["user_facing", "degraded", "internal"]},
                },
                "additionalProperties": False,
            },
        },
        "blast_radius_motivo": {"type": "string"},
        "evidencia": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "required": ["fonte", "consulta", "timestamp", "trecho", "fonte_suspeita"],
                "properties": {
                    "fonte": {"type": "string", "enum": ["prometheus", "k8s_api", "logs"]},
                    "consulta": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "trecho": {"type": "string", "maxLength": 500},
                    "fonte_suspeita": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
        "evidencia_motivo": {"type": "string"},
        "acao_proposta": {
            "type": "object",
            "required": ["tipo", "comandos", "reversivel"],
            "properties": {
                "tipo": {"type": "string", "enum": ["envelope", "escalate"]},
                "comandos": {"type": "array", "items": {"type": "string"}},
                "reversivel": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "acao_executada": {
            "type": ["object", "null"],
            "properties": {
                "comandos": {"type": "array", "items": {"type": "string"}},
                "read_back": {"type": "string"},
            },
            "required": ["comandos", "read_back"],
            "additionalProperties": False,
        },
        "acao_executada_motivo": {"type": "string"},
        "confianca": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "confianca_motivo": {"type": "string"},
        "urgencia": {"type": ["string", "null"], "enum": ["P1", "P2", "P3", "P4", None]},
        "urgencia_motivo": {"type": "string"},
        "decisao": {"type": "string", "enum": ["agir", "escalar", "silenciar_proposto"]},
        "t_alarme": {"type": "string", "minLength": 1},
        "t_record": {"type": "string", "minLength": 1},
        "motivo_escalacao": {
            "type": ["array", "null"],
            "items": {"type": "string", "enum": list(ESCALATION_TRIGGER_CODES)},
            "uniqueItems": True,
        },
        "motivo_escalacao_motivo": {"type": "string"},
        "rollback_state": {
            "type": ["object", "null"],
            "properties": {
                "tipo": {
                    "type": "string",
                    "enum": ["rollout_restart", "rollout_undo", "delete_pod", "scale"],
                },
                "estado_pre_mutacao": {"type": "string", "minLength": 1},
                # null for rollout_restart (idempotent) and delete_pod (no executable reversal:
                # the recorded ReplicaSet name is informative, not a command) -- see
                # harness/episode_runner.py's `_compute_rollback_state*` for exactly which types
                # populate this and which leave it null.
                "comando_rollback": {"type": ["string", "null"]},
                "motivo": {"type": "string", "minLength": 1},
            },
            "required": ["tipo", "estado_pre_mutacao", "comando_rollback", "motivo"],
            "additionalProperties": False,
        },
        "rollback_state_motivo": {"type": "string"},
        "acoes_executadas": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "required": ["classe", "comando", "t", "read_back", "resultado"],
                "properties": {
                    "classe": {"type": "string", "enum": ["C0", "C1", "C2"]},
                    "comando": {"type": "string"},
                    "t": {"type": "string"},
                    # "nao aplicavel: ..." ("not applicable") for C0/C1 (no mutation to confirm) and for a dry-run C2
                    # -- never null, see harness/triage_assembly.py's `_read_back_for_execution`.
                    "read_back": {"type": "string"},
                    "resultado": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "acoes_executadas_motivo": {"type": "string"},
        "acao_recomendada": {
            "type": ["object", "null"],
            "properties": {
                "comandos": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "efeito_esperado": {"type": "string", "minLength": 1},
                "risco": {"type": "string", "minLength": 1},
            },
            "required": ["comandos", "efeito_esperado", "risco"],
            "additionalProperties": False,
        },
        "acao_recomendada_motivo": {"type": "string"},
    },
    "additionalProperties": False,
}


class TriageValidationError(ValueError):
    """Raised when a TriageRecord fails schema validation or the honesty/instrument-error rules."""


def _check_nullable_with_reason(payload: dict[str, Any]) -> None:
    for field in NULLABLE_WITH_REASON_FIELDS:
        if payload.get(field) is None:
            reason = payload.get(f"{field}_motivo")
            if not reason or not str(reason).strip():
                raise TriageValidationError(
                    f"{field!r} is null but {field}_motivo is missing or empty: "
                    "the honesty rule requires a stated reason, never a silent null"
                )


def cause_cites_instrument_error(payload: dict[str, Any]) -> str | None:
    """Return the matched signature if `causa_provavel.detalhe` cites a known tool-error
    signature, else None. This is grounds for rejecting the record (retry once with
    a warning; a second occurrence escalates as E7), not for scoring it as a diagnosis."""
    detail = (payload.get("causa_provavel") or {}).get("detalhe", "")
    for signature in INSTRUMENT_ERROR_SIGNATURES:
        if signature.lower() in detail.lower():
            return signature
    return None


def validate_triage_record(payload: dict[str, Any], *, allow_instrument_error_cause: bool = False) -> dict[str, Any]:
    """Validate a TriageRecord: JSON schema, the honesty (null+motivo) rule, and the
    instrument-error-as-cause rejection. Returns the payload unchanged on success.

    `allow_instrument_error_cause=True` is only for the one deliberate retry-with-warning path;
    the harness sets it back to False on the retry, so a second occurrence
    raises (feeding E7)."""
    try:
        jsonschema.validate(instance=payload, schema=TRIAGE_RECORD_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise TriageValidationError(str(exc)) from exc

    _check_nullable_with_reason(payload)

    if not allow_instrument_error_cause:
        signature = cause_cites_instrument_error(payload)
        if signature is not None:
            raise TriageValidationError(
                f"causa_provavel.detalhe cites instrument-error signature {signature!r}: "
                "tool failure is not cluster signal (failure modes F2/F5)"
            )

    return payload
