"""Assembles a `harness.triage.TriageRecord` from a finished triage episode.

The triage flow emits a TriageRecord always, including when the episode escalates. This
module's two entry points (`assemble_triage_record`, `build_fallback_triage_record`) are written
so their caller (`harness/episode_runner.py`'s `make_episode_runner`) always has a schema-valid
record to hand to its `triage_sink`, never a raised exception standing in for one.
`assemble_triage_record` itself never raises: a record it built that fails
`harness.triage.validate_triage_record` is replaced with an honest escalation fallback, not left
unemitted.

Reuses `harness/triage.py`'s schema and validation rather than a second one -- this module's job
is turning episode data (EpisodeResult, LiveClusterEnvironment, EpisodeContext, the raw webhook)
into the dict shape that schema expects.

Honest limits this module does NOT paper over (documented once here, not repeated per field):
  - `causa_provavel.classe`: the model's `AGENT_ACTION_SCHEMA` output (harness/schema.py) carries
    a structured `cause_class` from the closed taxonomy (`fault_table.CAUSE_CLASSES`), and
    `classe` is copied from the final action verbatim, never inferred from `action_type` or
    mapped from free text. Escalating with `unknown` (the correct answer for a fault outside the
    trained families) is the model's own answer, not a harness default. An action without a valid
    `cause_class` (only possible for a replayed log from before the field existed, since
    validation requires it) gets `fault_table.UNCLASSIFIED_CLASS`, which scores as `errado`
    (wrong); `detalhe` always carries the model's raw diagnosis text verbatim.
  - `blast_radius`: always null+motivo. No service-topology graph is wired into this codebase
    (hotelReservation's 19-deployment graph is an example of what a real deployment would
    declare, not something this module can read).
  - `urgencia`: always null+motivo. Its ground truth is `FaultRow.expected_urgency` in
    `harness/fault_table.py`, which is what the scorer compares a model's urgency to; copying it
    into the record would be the answer key, and the model does not emit urgency yet.
  - `evidencia` timestamps: `episode_runner.ScriptedSource` does not record a per-source
    wall-clock timestamp, only the whole 90s collection window's elapsed duration -- every
    evidencia entry sourced from the scripted collection is timestamped with `t_record` (when the
    harness finished the episode), not the instant that particular source was actually queried.
  - `evidencia[].fonte_suspeita`: always False. No adversarial-payload detector is wired into
    collection yet.
  - `acao_executada`: null with the dry-run motivo whenever the C2 dry-run gate (episode_runner's
    `c2_dry_run`) blocked a real mutation -- see that module's docstring for why the gate exists.
    When it is off and a C2 actually ran, `read_back` reports the REAL read-back verdict
    (`LiveClusterEnvironment._perform_read_back`).
  - `rollback_state`: describes the LAST real (non-dry-run) C2 execution's pre-computed rollback
    state -- see `_build_rollback_state` below. Null with a reason when no real C2 executed this
    episode (escalated, read-only, or the dry-run gate is still on).
  - `acoes_executadas`: every C0/C1/C2 command this episode ran (`environment.executions`), never
    null in the normal assembly path -- an EMPTY list is the honest fact for an episode that
    escalated without acting (same "checked, found zero" convention `motivo_escalacao`'s
    empty-list case uses in this module), not a stand-in for "could not measure". `t` (per-entry
    timestamp) inherits the same honest limit `evidencia[].timestamp` documents above:
    `environment.executions` entries do not carry a per-command wall-clock timestamp, only
    `t_record` (when the harness finished the whole episode) is available, so every entry is
    stamped with that, not the instant the command actually ran. See `_build_acoes_executadas`
    below for `read_back`/`resultado` per action class.
  - `acao_recomendada`: always null+motivo. Its ground truth is the expected-action column of
    the fault table (the same one the rubric uses), and copying it into the record would put the
    answer key in the answer. Same honest-limit shape as `blast_radius`/`urgencia` above, not a
    heuristic guess at what the agent "would" run.

`proxima_evidencia` (the escalation package's "next evidence to look at" field) is deliberately
NOT built: unlike the other fields, it needs a real new design decision (which query the harness
would suggest the human run first) rather than reuse of the existing null+motivo/last-real-action
conventions.

`_build_motivo_escalacao` below is the single place that turns all seven independent,
deterministic signals into the list this record carries. Each trigger's own docstring on that
function names its exact source; none of them is a text heuristic over `causa_provavel`/`sintese`
-- codes come from harness bookkeeping (envelope denials, rollback outcomes, loop detections,
elapsed time, evidence coverage), never from guessing at what the model's prose means.

E7 (instrument-error recidivism) needed its own design decision, documented here because it does
not fit the "one signal, one trigger" shape of the others: the rule is "retry once with a
warning; a repeat escalates with E7", which reads as a live retry loop inside a SINGLE model
call/validation cycle. `harness/triage.py`'s `validate_triage_record(allow_instrument_error_
cause=...)` parameter exists as that retry's API surface, but nothing in the live episode loop
(`harness.agent.Agent.run_episode`) ever calls back into the model for a live "you cited a tool
error, try again" turn -- that would mean either injecting a mid-episode intervention into a loop
shared with configs A/B/C, or a new schema-validation retry that only lives on the live-cluster
path.

**Simplification adopted instead**: this module treats the EPISODE's own turn history as the
retry sequence it already has for free, since a multi-turn episode's turns are themselves
sequential model responses to the same underlying incident. `_instrument_error_turn_count` scans
every turn's parsed `diagnosis` (not just the final action's) for an instrument-error signature.
The FIRST turn in the episode that cites one is the "warned" occurrence (logged, `causa_provavel`
still allowed through `validate_triage_record` if the FINAL turn is the only offender); a SECOND
(or later) turn citing one -- anywhere in the same episode -- is the repeat that raises E7 and, if
the final turn is among the offenders, forces the generic validation-failure fallback
(`build_fallback_triage_record`, decisao=escalar) exactly like any other schema violation would.
This is a coarser unit of "retry" than the per-call wording implies (per-episode, not
per-model-call), stated explicitly rather than silently narrowed: wiring a true per-call retry
into `Agent.run_episode` is future work if the coarser version proves too lenient (e.g. a model
that cites the instrument error once, is "warned", then never revisits it again this episode
never accumulates a second occurrence at all).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .envelope import CONFIDENCE_GATE, ActionClass
from .fault_table import CAUSE_CLASSES, UNCLASSIFIED_CLASS, UNKNOWN_CLASS
from .triage import TriageValidationError, cause_cites_instrument_error, validate_triage_record

logger = logging.getLogger("sslm.triage_assembly")

# E6: the 3 evidence categories the trigger names ("Prometheus, apiserver, logs"), not the many
# individual `ScriptedSource`s that can exist per category (e.g.
# several `k8s_logs[pod]` sources). A category "failed" when NONE of its sources collected
# successfully this episode.
_EVIDENCE_CATEGORIES = ("prometheus", "k8s_api", "logs")

# evidencia.trecho (evidence excerpt): "<= 500 chars".
EVIDENCE_EXCERPT_MAX_CHARS = 500

# ScriptedSource.name prefixes -> the evidencia.fonte enum (harness/triage.py). Order matters:
# "k8s_logs[...]" must match before the generic "k8s_" prefix. "alertmanager_payload" matches
# nothing here on purpose -- it is the alert itself, not one of the three evidencia sources the
# schema recognizes (prometheus/k8s_api/logs), so it never becomes an evidencia entry.
_EVIDENCE_SOURCE_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("prometheus_range", "prometheus"),
    ("k8s_logs", "logs"),
    ("k8s_", "k8s_api"),
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_t_alarme(webhook_raw: dict, *, fallback: str) -> str:
    """`t_alarme` (alarm time): when the alert itself fired, per Alertmanager's own `startsAt` on
    the first alert in the group. `fallback` (the caller passes the webhook-received timestamp) is
    used only when `startsAt` is missing or not a string -- `harness.webhook.parse_webhook_payload`
    does not require `startsAt` (Alertmanager always sends it in practice, but this module does not
    assume that), and `t_alarme` is a required, non-null schema field, so an honest fallback beats
    leaving it empty."""
    alerts = webhook_raw.get("alerts") or []
    if alerts and isinstance(alerts[0], dict):
        starts_at = alerts[0].get("startsAt")
        if isinstance(starts_at, str) and starts_at:
            return starts_at
    return fallback


def _alert_fingerprint(webhook_raw: dict) -> str:
    alerts = webhook_raw.get("alerts") or []
    if alerts and isinstance(alerts[0], dict):
        fingerprint = alerts[0].get("fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            return fingerprint
    # Defensive only: harness.webhook.parse_webhook_payload already rejects a payload missing a
    # fingerprint before an episode is ever opened (EpisodeDeduper.ingest), so this branch should
    # be unreachable for any webhook_raw that made it this far -- never a fabricated fingerprint.
    return webhook_raw.get("groupKey") or "unknown"


def _evidence_source_for_source_name(name: str) -> str | None:
    for prefix, evidence_source in _EVIDENCE_SOURCE_BY_PREFIX:
        if name.startswith(prefix):
            return evidence_source
    return None


def _query_for_source(source: Any) -> str:
    # collect_episode_context's prometheus sources embed the actual PromQL as the text's first
    # line ("query: <promql>\n..."); recovering it here gives a real consulta instead of just the
    # source's internal name. k8s/logs sources do not carry a separately stored command string, so
    # for them consulta is honestly just the source name (which identifies which scripted step).
    if source.collected and source.text.startswith("query: "):
        return source.text.split("\n", 1)[0][len("query: ") :]
    return source.name


def _build_evidencia(context: Any, t_record: str) -> tuple[list[dict] | None, str | None]:
    entries: list[dict] = []
    failure_reasons: list[str] = []
    for source in context.sources:
        evidence_source = _evidence_source_for_source_name(source.name)
        if evidence_source is None:
            continue
        if not source.collected:
            failure_reasons.append(f"{source.name}: {source.reason}")
            continue
        entries.append(
            {
                "fonte": evidence_source,
                "consulta": _query_for_source(source),
                # Honest limit (module docstring): per-source wall-clock timestamps do not exist,
                # only the whole collection window's elapsed duration. t_record is the closest
                # honest approximation, not a claim about when THIS source specifically ran.
                "timestamp": t_record,
                "trecho": source.text[:EVIDENCE_EXCERPT_MAX_CHARS],
                "fonte_suspeita": False,
            }
        )
    if entries:
        return entries, None
    reason = "no evidence source was successfully collected this episode"
    if failure_reasons:
        reason += ": " + "; ".join(failure_reasons[:5])
    return None, reason


def _c2_executions(environment: Any) -> list[dict]:
    return [e for e in environment.executions if e.get("action_class") == ActionClass.C2.value]


def _final_action(result: Any, environment: Any) -> dict | None:
    """The turn whose parsed action `causa_provavel`/`sintese`/`confianca` must describe.

    When this episode executed a C2 mutation (`decisao="agir"`), that must be the SAME turn
    `acao_executada`/`acao_proposta` already describe, not whichever turn happened to run last --
    the agent loop keeps going after a C2 (dry-run or real) until the model escalates or hits
    `max_turns`, so a plain "scan backward for any parsed action" picks up that later, unrelated
    escalate turn instead (measured: an inspect -> remediate -> escalate episode produced a record where decisao/acao_executada described the remediation but
    causa_provavel/sintese/confianca described the escalate turn that followed it). Matched by
    command text against `environment.executions` (the ground truth of what actually ran) rather
    than by action_type, since the model's own action_type label is not the authoritative
    classification -- `harness.envelope.classify()` is.

    Falls back to the plain last-parsed-action turn (scanning backward) when no C2 executed this
    episode (`decisao="escalar"`, `_c2_executions` empty) -- a malformed final turn (retry
    feedback, no action) must not blank out an earlier turn's real diagnosis/confidence.
    """
    c2 = _c2_executions(environment)
    if c2:
        last_c2_command = c2[-1]["command"]
        for turn in reversed(result.turns):
            if turn.action is not None and turn.action.get("command") == last_c2_command:
                return turn.action
        # Defensive only: every entry in environment.executions was produced by some turn's
        # parsed action reaching LiveClusterEnvironment.apply_action, so this should be
        # unreachable. Never fabricate a match if it somehow is -- fall through to the plain scan.
    for turn in reversed(result.turns):
        if turn.action is not None:
            return turn.action
    return None


def _build_acao_proposta(environment: Any) -> dict:
    c2 = _c2_executions(environment)
    if c2:
        return {
            "tipo": "envelope",
            "comandos": [e["command"] for e in c2],
            # Only C0-C2 ever reach environment.executions (harness.envelope.classify's C3
            # fail-closed default never executes); C2's allowlist is
            # reversible-mutation-only by construction, so an executed C2 is always reversivel.
            "reversivel": True,
        }
    return {"tipo": "escalate", "comandos": [], "reversivel": True}


def _build_acao_executada(environment: Any) -> tuple[dict | None, str | None]:
    c2 = _c2_executions(environment)
    if not c2:
        return None, "no C2 mutation was attempted this episode (escalated, or read-only diagnosis only)"
    if any(e.get("dry_run") for e in c2):
        # Wording kept verbatim: published run data and tests match on it ("dry-run:
        # rollback_state not implemented yet"). See episode_runner.py's module docstring and
        # LiveClusterEnvironment's c2_dry_run for why the gate exists.
        return None, "dry-run: rollback_state ainda nao implementado, ver item 2b-iii"
    # `read_back` reports the LAST real C2's actual read-back verdict
    # (`LiveClusterEnvironment._perform_read_back`), not a fixed placeholder. `comandos` lists
    # every real C2 command this episode executed; `read_back` describes only the LAST one, same "last real action wins" convention `_final_action` above
    # already documents for causa_provavel/sintese/confianca.
    last = c2[-1]
    read_back_info = last.get("read_back")
    if read_back_info is None:
        # The mutation command itself failed (non-zero exit): `_execute_c2_with_rollback` never
        # attempts read-back against a mutation that never took effect (episode_runner.py).
        read_back_text = "read-back nao executado: o comando de mutacao falhou (ver returncode), nada para reconferir"
    else:
        ready = read_back_info.get("ready")
        verdict = {True: "melhora confirmada", False: "sem melhora"}.get(ready, "inconclusivo")
        read_back_text = f"{verdict}: {read_back_info.get('detail', '(sem detalhe registrado)')}"
    return {
        "comandos": [e["command"] for e in c2],
        "read_back": read_back_text,
    }, None


def _build_causa_provavel(final_action: dict | None) -> dict:
    if final_action is None:
        return {
            "classe": UNKNOWN_CLASS,
            "detalhe": "no valid model action was produced this episode (every turn malformed)",
        }
    diagnosis = str(final_action.get("diagnosis") or "").strip() or "(model gave no diagnosis text)"
    # The class is the model's answer (AGENT_ACTION_SCHEMA.cause_class), never inferred from
    # action_type: escalating with a specific class is a different answer from escalating with
    # unknown, and the scorer scores the two differently. A validated action always carries the
    # field; the UNCLASSIFIED_CLASS fallback only covers an action that arrived here without it
    # (a replayed log from before the field existed), and scores as wrong, never as a silent hit.
    classe = final_action.get("cause_class")
    if classe not in CAUSE_CLASSES:
        classe = UNCLASSIFIED_CLASS
    return {"classe": classe, "detalhe": diagnosis}


def _build_sintese(final_action: dict | None) -> str:
    if final_action is None:
        return "no valid model action produced this episode; harness could not extract a diagnosis"[:280]
    diagnosis = str(final_action.get("diagnosis") or "").strip() or "(no diagnosis text)"
    return diagnosis[:280]


def _build_confianca(final_action: dict | None) -> tuple[float | None, str | None]:
    if final_action is None:
        return None, "no valid model action was produced this episode to read a confidence from"
    try:
        return float(final_action.get("confidence", 0.0)), None
    except (TypeError, ValueError):
        return None, f"model confidence field was not a number: {final_action.get('confidence')!r}"


# ---------------------------------------------------------------------------
# motivo_escalacao (escalation triggers). See the module docstring's E7
# paragraph for the one trigger whose computation is not a straight one-signal read.
# ---------------------------------------------------------------------------


def _e2_triggered(environment: Any) -> bool:
    """E2: "the needed action is C3, or a C2 denied by RBAC/gate" -- literally, not "any
    denial". A denied C1 (e.g. no prior C0 this episode, or the C1 call cap) is real containment
    but is not what the trigger table names for E2; only a C3 classification (any command the parser
    fail-closed into C3) or a C2 the gate denied count. `environment.denials` only ever holds
    entries where `allowed` is already False (`LiveClusterEnvironment.apply_action` appends there
    exactly on denial), so `not d.allowed` below is a belt-and-suspenders check, not load-bearing
    on its own."""
    return any(
        (d.action_class == ActionClass.C3 or d.action_class == ActionClass.C2) and not d.allowed
        for d in environment.denials
    )


def _e3_triggered(environment: Any) -> bool:
    """E3: "no progress: 2 C2 actions executed and read-back without improvement". Reads
    `LiveClusterEnvironment.rollback_events` -- one entry per REAL (non-dry-run) C2 execution
    whose mutation command itself succeeded (episode_runner.py's `_execute_c2_with_rollback` never
    appends an entry for a mutation that failed to even apply: nothing to read back there, so it
    is not a "no progress from read-back" case, it is a plain execution failure, which other
    trigger/audit paths already surface).

    "Without improvement" counts `improved` values of `False` OR `None`: `None` means read-back could not
    determine readiness at all (episode_runner.py's delete-pod honest limit) -- the conservative
    reading for a trigger whose entire point is catching no-progress is to treat "could not
    confirm improvement" the same as "confirmed no improvement", never as a pass. Read via
    `getattr` with an empty-list default, same convention as `loop_detections`/E4 above, so a test
    double or a future Environment implementation without this attribute degrades to "no E3
    signal" rather than raising."""
    events = getattr(environment, "rollback_events", None) or []
    no_improvement = [e for e in events if e.get("improved") is not True]
    return len(no_improvement) >= 2


def _build_rollback_state(environment: Any) -> tuple[dict | None, str | None]:
    """The LAST real (non-dry-run) C2 execution's pre-computed rollback_state --
    same "last real action wins" convention `_final_action`/`_build_acao_executada` already use.
    `rollback_state` itself is computed and attached to `environment.executions` entries by
    `LiveClusterEnvironment._execute_c2_with_rollback` (episode_runner.py); this function only
    picks which one the record reports, it never computes one itself (never fabricated here)."""
    c2 = _c2_executions(environment)
    if not c2:
        return None, "no C2 mutation was attempted this episode (escalated, or read-only diagnosis only)"
    last = c2[-1]
    if last.get("dry_run"):
        return None, "dry-run: rollback_state ainda nao executado de verdade (SSLM_C2_DRY_RUN ligado), ver item 2b-iii"
    rollback_state = last.get("rollback_state")
    if rollback_state is None:
        # Defensive only: every real C2 path in LiveClusterEnvironment always attaches
        # rollback_state before executing (the rule "a C2 without rollback_state does not
        # execute" is enforced there, not here) -- if this is somehow missing anyway, report it honestly
        # rather than fabricate a value.
        return None, "real C2 executed but no rollback_state was recorded (should be unreachable, see LiveClusterEnvironment._execute_c2_with_rollback)"
    return rollback_state, None


# ---------------------------------------------------------------------------
# acoes_executadas (executed actions). See the module docstring's honest-
# limits paragraph for why an empty list here is a real fact, and why every entry's `t` reuses
# `t_record` rather than a (nonexistent) per-command timestamp.
# ---------------------------------------------------------------------------


def _result_for_execution(execution: dict) -> str:
    """`resultado` (result): the outcome the audit log join checks the record against. A
    dry-run C2 never reaches `command_runner` (episode_runner.py's dry-run gate), so it never hits
    the apiserver's audit log at all -- said explicitly here rather than reported as a normal
    "executed" outcome that a 1:1 join would then fail to find."""
    if execution.get("dry_run"):
        # Kept verbatim: tests and published run data match on it ("classified and allowed by
        # the envelope, but NOT executed; no matching entry in the apiserver audit log").
        return (
            "DRY-RUN: classificado e permitido pelo envelope, mas NAO executado "
            "(SSLM_C2_DRY_RUN ligado) -- sem correlato no audit log do apiserver"
        )
    returncode = execution.get("returncode")
    if returncode is None:
        # Defensive only: every non-dry-run entry in environment.executions was produced after a
        # real command_runner call, which always sets returncode (episode_runner.py). Report
        # honestly rather than fabricate a code if this is somehow missing anyway.
        return "executed, returncode not recorded (should be unreachable)"
    # "executado" (executed) is kept: scenarios/chars_window.py and published run data match on
    # the "executado, exit=<n>" prefix.
    return f"executado, exit={returncode}"


def _read_back_for_execution(execution: dict) -> str:
    """`read_back`: only C2 mutations have a symptom to reconfirm -- C0/C1 never change
    cluster state, so the envelope never asks them for one. Never null (schema requires a
    string): "nao aplicavel" ("not applicable") is itself the honest value for the classes it
    does not apply to, same
    convention `_build_acao_executada`'s own `read_back` text already uses for its "not executed"
    cases.

    The read-back strings stay in Portuguese because tests and published run data match on them:
    "nao aplicavel" = not applicable, "read-back nao executado" = read-back not run, "melhora
    confirmada" = improvement confirmed, "sem melhora" = no improvement, "inconclusivo" =
    inconclusive, "sem detalhe registrado" = no detail recorded."""
    if execution.get("action_class") != ActionClass.C2.value:
        return "nao aplicavel: C0/C1 nao envolve mutacao a reconferir"
    if execution.get("dry_run"):
        return "nao aplicavel: dry-run, C2 classificado mas nao executado (SSLM_C2_DRY_RUN ligado)"
    read_back_info = execution.get("read_back")
    if read_back_info is None:
        # Same case `_build_acao_executada` already documents: the mutation command itself failed,
        # so `_execute_c2_with_rollback` never attempted a read-back against it.
        return "read-back nao executado: o comando de mutacao falhou (ver resultado), nada para reconferir"
    ready = read_back_info.get("ready")
    verdict = {True: "melhora confirmada", False: "sem melhora"}.get(ready, "inconclusivo")
    return f"{verdict}: {read_back_info.get('detail', '(sem detalhe registrado)')}"


def _build_acoes_executadas(environment: Any, t_record: str) -> list[dict]:
    """A list of {classe, comando, t, read_back, resultado}, evaluable 1:1 against
    the apiserver's audit log. Reads `environment.executions` directly -- every C0/C1/C2 command
    the episode ran, in the order it ran (`LiveClusterEnvironment.apply_action` only ever appends
    there, never reorders) -- and never `environment.denials` (a denied command never reached
    `command_runner`, so it never produced an apiserver audit entry to join against either;
    the audit-log join is exactly why denials do not belong in this list).

    Always a list, never None, in this function itself -- an episode that escalated without
    acting legitimately produces `[]` here (module docstring: "checked, found zero", the same
    honest-empty convention `_build_motivo_escalacao` already uses). The null+motivo case lives
    one level up, in `assemble_triage_record`/`build_fallback_triage_record`, for the one path
    that has no `environment` to read at all."""
    return [
        {
            "classe": execution.get("action_class"),
            "comando": execution.get("command") or "",
            "t": t_record,
            "read_back": _read_back_for_execution(execution),
            "resultado": _result_for_execution(execution),
        }
        for execution in environment.executions
    ]


# ---------------------------------------------------------------------------
# acao_recomendada (recommended action). Always null+motivo -- see the module docstring's
# honest-limits paragraph: the table this field is evaluated against is `harness/fault_table.py`
# (`FaultRow.expected_action`), which is the answer
# key, not a source for the field; the model does not emit a structured recommendation yet.
# ---------------------------------------------------------------------------


def _build_acao_recomendada() -> tuple[None, str]:
    return None, (
        "the model's action schema does not emit a structured recommendation yet; the expected "
        "action lives in harness/fault_table.py as the scoring key and is never copied into the "
        "record"
    )


def _e6_failed_categories(context: Any) -> list[str]:
    collected_by_category = {category: False for category in _EVIDENCE_CATEGORIES}
    for source in context.sources:
        evidence_source = _evidence_source_for_source_name(source.name)
        if evidence_source is not None and source.collected:
            collected_by_category[evidence_source] = True
    return [category for category in _EVIDENCE_CATEGORIES if not collected_by_category[category]]


def _e6_triggered(context: Any) -> bool:
    """E6: ">= 2 of the 3 sources ... failed to collect". A category with ZERO sources at all (e.g.
    no not-Ready pod ever existed, so `collect_episode_context` never even attempted a
    "k8s_logs[...]" source) counts as failed here, same as a category every source of which was
    attempted and failed -- both mean this episode ends up with no usable evidence from that
    category, which is the thing E6 exists to catch ("do not decide blind"). A documented
    simplification, not an oversight: the trigger definition does not distinguish "never
    attempted" from "attempted and failed" for this trigger, and treating "never attempted" as
    success would make E6 easier to satisfy by collecting less, the wrong direction for a
    trigger whose entire point is penalizing insufficient evidence."""
    return len(_e6_failed_categories(context)) >= 2


def _instrument_error_turn_count(result: Any) -> int:
    """How many turns THIS episode had a parsed action whose `diagnosis` cites a known
    instrument-error signature (harness.triage.INSTRUMENT_ERROR_SIGNATURES via
    `cause_cites_instrument_error`). See the module docstring's E7 paragraph for why this counts
    turns, not model-call retries: >= 2 is E7's "repeat after 1 warning"."""
    count = 0
    for turn in result.turns:
        if turn.action is None:
            continue
        diagnosis = str(turn.action.get("diagnosis") or "")
        if cause_cites_instrument_error({"causa_provavel": {"detalhe": diagnosis}}):
            count += 1
    return count


def _build_motivo_escalacao(
    *,
    context: Any,
    environment: Any,
    result: Any,
    confidence: float | None,
    instrument_error_turn_count: int,
) -> list[str]:
    """The single place all seven computed triggers (E1, E2, E3, E4, E5, E6, E7) come together.
    Each is read from a distinct, already-existing piece of harness state -- never derived from
    `causa_provavel`/`sintese` text: classify by rule, never by textual judgment. `environment.
    loop_detections`/`rollback_events` are read via `getattr` with an empty-list default so a test
    double (or a future `Environment` implementation) that has not been updated to carry the
    attribute yet degrades to "no signal" rather than raising -- `assemble_triage_record` must
    never raise, module docstring."""
    triggers: list[str] = []
    if confidence is not None and confidence < CONFIDENCE_GATE:
        triggers.append("E1")
    if _e2_triggered(environment):
        triggers.append("E2")
    if _e3_triggered(environment):
        triggers.append("E3")
    if getattr(environment, "loop_detections", None):
        triggers.append("E4")
    if getattr(result, "e5_triggered", False):
        triggers.append("E5")
    if _e6_triggered(context):
        triggers.append("E6")
    if instrument_error_turn_count >= 2:
        triggers.append("E7")
    return triggers


def assemble_triage_record(
    *,
    episode_id: str,
    webhook_raw: dict,
    context: Any,
    environment: Any,
    result: Any,
    t_alarme: str,
    t_record: str | None = None,
) -> dict[str, Any]:
    """Build one TriageRecord for a finished episode. Never raises: a record that
    fails `harness.triage.validate_triage_record` is replaced with an honest escalation fallback
    (`build_fallback_triage_record`), logged as a warning, so the caller can always emit
    something -- "emit a TriageRecord, always" is a promise about every episode, not only the ones this module could describe well."""
    t_record = t_record or now_iso()
    final_action = _final_action(result, environment)

    evidence, evidence_reason = _build_evidencia(context, t_record)
    confidence, confidence_reason = _build_confianca(final_action)
    executed_action, executed_action_reason = _build_acao_executada(environment)
    rollback_state, rollback_state_reason = _build_rollback_state(environment)
    executed_actions = _build_acoes_executadas(environment, t_record)
    recommended_action, recommended_action_reason = _build_acao_recomendada()
    decision = "agir" if _c2_executions(environment) else "escalar"

    instrument_error_turn_count = _instrument_error_turn_count(result)
    motivo_escalacao = _build_motivo_escalacao(
        context=context,
        environment=environment,
        result=result,
        confidence=confidence,
        instrument_error_turn_count=instrument_error_turn_count,
    )
    # E7 (module docstring): a SECOND (or later) turn citing an instrument-error signature this
    # episode is the repeat that must reject the record outright -- allow the
    # final action's own citation through only when this is the episode's single (first, "warned")
    # occurrence. `validate_triage_record` only ever inspects the FINAL `causa_provavel.detalhe`
    # (harness/triage.py's `cause_cites_instrument_error`), so a count of 0 or 1 here is always
    # safe to allow even if the final turn is not itself the offending one.
    allow_instrument_error_cause = instrument_error_turn_count <= 1

    record: dict[str, Any] = {
        "episode_id": episode_id,
        "alert_fingerprint": _alert_fingerprint(webhook_raw),
        "sintese": _build_sintese(final_action),
        "causa_provavel": _build_causa_provavel(final_action),
        "blast_radius": None,
        "blast_radius_motivo": (
            "service topology is not wired into this harness yet (the app's service graph is "
            "not data this module can read)"
        ),
        "evidencia": evidence,
        "acao_proposta": _build_acao_proposta(environment),
        "acao_executada": executed_action,
        "confianca": confidence,
        "urgencia": None,
        "urgencia_motivo": (
            "the model's action schema does not emit urgency yet; the expected urgency lives in "
            "harness/fault_table.py as the scoring key and is never copied into the record"
        ),
        "decisao": decision,
        "t_alarme": t_alarme,
        "t_record": t_record,
        "motivo_escalacao": motivo_escalacao,
        "rollback_state": rollback_state,
        "acoes_executadas": executed_actions,
        "acao_recomendada": recommended_action,
    }
    if evidence_reason is not None:
        record["evidencia_motivo"] = evidence_reason
    if confidence_reason is not None:
        record["confianca_motivo"] = confidence_reason
    if executed_action_reason is not None:
        record["acao_executada_motivo"] = executed_action_reason
    if rollback_state_reason is not None:
        record["rollback_state_motivo"] = rollback_state_reason
    if recommended_action_reason is not None:
        record["acao_recomendada_motivo"] = recommended_action_reason

    return _validated_or_fallback(
        record, allow_instrument_error_cause=allow_instrument_error_cause, motivo_escalacao=motivo_escalacao
    )


def _validated_or_fallback(
    record: dict[str, Any],
    *,
    allow_instrument_error_cause: bool = False,
    motivo_escalacao: list[str] | None = None,
) -> dict[str, Any]:
    try:
        return validate_triage_record(record, allow_instrument_error_cause=allow_instrument_error_cause)
    except TriageValidationError as exc:
        logger.warning(
            "episode %s: assembled TriageRecord failed validation (%s) -- emitting an honest "
            "escalation fallback instead of a record that would fail the TriageRecord contract",
            record.get("episode_id"),
            exc,
        )
        fallback_t_record = record.get("t_record")
        return build_fallback_triage_record(
            episode_id=record.get("episode_id", "unknown"),
            alert_fingerprint=record.get("alert_fingerprint", "unknown"),
            t_alarme=record.get("t_alarme", now_iso()),
            t_record=fallback_t_record if isinstance(fallback_t_record, str) else now_iso(),
            reason=f"assembly produced an invalid TriageRecord: {exc}",
            # The real, already-computed trigger list travels into the fallback record instead of
            # being thrown away: `motivo_escalacao` is independent of whether `causa_provavel`/
            # the rest of the record happened to be schema-valid. A validation failure caused BY
            # an E7 repeat is
            # exactly the case this preserves -- the fallback still reports "E7" honestly, instead
            # of falling back to null+motivo as if no triggers had ever been computed.
            motivo_escalacao=motivo_escalacao,
        )


def build_fallback_triage_record(
    *,
    episode_id: str,
    alert_fingerprint: str,
    t_alarme: str,
    t_record: str,
    reason: str,
    motivo_escalacao: list[str] | None = None,
) -> dict[str, Any]:
    """The record this module falls back to whenever it cannot honestly build a real one --
    validation failure above, or `make_episode_runner`'s own `except` branch when the orchestrator
    raised before an episode ever produced turns. Always schema-valid, always `decisao=escalar`:
    "emit a TriageRecord, always" is a promise about EMISSION, not about how
    much the record could measure -- an episode the harness cannot describe still gets an honest,
    all-null-with-motivo record instead of no record at all.

    `motivo_escalacao`: pass the already-computed trigger list when one exists (the
    `_validated_or_fallback` call site above always does -- see its own comment). Left as `None`
    (the default) only by `harness/episode_runner.py`'s `except` branch for the orchestrator-
    raised-before-any-turns case, where there is no episode data left to compute triggers from at
    all -- that is a genuinely different kind of "could not measure" than "measured zero
    triggers", so it gets null+motivo rather than a fabricated empty list.

    `causa_provavel.detalhe` is `reason` UNLESS `reason` itself happens to cite an
    instrument-error signature (a measured case: the fallback triggered by E7's OWN
    rejection quotes the original exception message, which quotes the model's diagnosis, which
    is exactly the instrument-error text that got it rejected in the first place). Reusing
    `reason` verbatim there would make this function emit a record that fails its own
    `harness.triage.validate_triage_record` on a plain re-check -- this function's whole job is
    to always be schema- and rule-valid, so `causa_provavel.detalhe` falls back to a generic,
    signature-free sentence in that one case; the full `reason` text is not lost, it is still in
    every `*_motivo` field (`cause_cites_instrument_error` only ever inspects
    `causa_provavel.detalhe`)."""
    detail = reason
    if cause_cites_instrument_error({"causa_provavel": {"detalhe": reason}}) is not None:
        detail = (
            "fallback record: TriageRecord assembly could not produce a compliant record for "
            "this episode (see the *_motivo fields for the underlying reason)"
        )
    record = {
        "episode_id": episode_id,
        "alert_fingerprint": alert_fingerprint,
        "sintese": reason[:280],
        "causa_provavel": {"classe": UNKNOWN_CLASS, "detalhe": detail},
        "blast_radius": None,
        "blast_radius_motivo": reason,
        "evidencia": None,
        "evidencia_motivo": reason,
        "acao_proposta": {"tipo": "escalate", "comandos": [], "reversivel": True},
        "acao_executada": None,
        "acao_executada_motivo": reason,
        "confianca": None,
        "confianca_motivo": reason,
        "urgencia": None,
        "urgencia_motivo": reason,
        "decisao": "escalar",
        "t_alarme": t_alarme,
        "t_record": t_record,
        "rollback_state": None,
        "rollback_state_motivo": reason,
        "acoes_executadas": None,
        "acoes_executadas_motivo": reason,
        "acao_recomendada": None,
        "acao_recomendada_motivo": reason,
    }
    if motivo_escalacao is not None:
        record["motivo_escalacao"] = motivo_escalacao
    else:
        record["motivo_escalacao"] = None
        record["motivo_escalacao_motivo"] = reason
    return record
