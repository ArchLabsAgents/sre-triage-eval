"""Per-episode rubric, oracle baselines and clustered bootstrap for H3a/H3b/H3c.

Everything the scorer judges against comes from harness/fault_table.py (the one versioned table);
nothing here re-states a cause, action, disposition or urgency. No LLM judge, no benchmark
`success`.

Unit and clustering: one Episode per injected fault (or injected false alarm). Every interval is a
percentile bootstrap that resamples whole clusters, cluster = (family, round). Never family alone:
with 3-8 families that would be 3-8 clusters. The seed and resample count are fixed so the same
episodes always give the same numbers, and the dry-run hash (dry_run()) pins that.

Report vocabulary. The rubric values and report keys are Portuguese because published reports and
the pre-registered dry-run hash use them; they are data, not prose: correto/parcial/errado/honesto
(right/partial/wrong/honest unknown), exata/parcial/errada (exact/partial/wrong cause match),
agir/escalar (act/escalate), tipo1_medo (escalated when it should have acted), tipo2_afoiteza
(acted when it should have escalated), nao medido (not measured), familia/rodada (family/round).

Scoring choices that are pre-registration decisions (not tuned):

- H3a = exact fine-class accuracy (cause_match == "exata") over measured episodes of known
  families. `honesto` counts in the denominator and not in the numerator: it is never an error,
  and it is also not a correct cause.
- H3b compares at the MECHANISM (coarse) level, on episodes whose O1 oracle key is ambiguous in the
  table (ambiguous_oracle_keys), model vs the alertname+namespace+service oracle. The oracle is
  refit inside every bootstrap resample on the resampled episodes themselves (in-sample, the most
  favourable version of the baseline), predicting the most frequent mechanism for its key, ties
  broken by name. < 12 ambiguous episodes -> "nao medido: universo sem ambiguidade" (not measured:
  no ambiguity in the universe).
- blast_radius matches when it names the row's cause_service OR symptom_service (mongo faults are
  injected on mongodb-<svc> and felt on <svc>); a namespace-wide row matches any non-empty list.
- Urgency is descriptive only, always reported next to the "always P2" baseline. Disposition is
  always reported next to the "always escalate" baseline.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from math import sqrt
from typing import Any, Callable, Iterable, Mapping, Sequence

from harness.fault_table import (
    COARSE_BY_FINE,
    FALSE_POSITIVE_CLASS,
    FAULT_TABLE,
    NAMESPACE_SCOPE,
    REVISION_PLACEHOLDER,
    TABLE_VERSION,
    UNKNOWN_CLASS,
    FaultRow,
    ambiguous_oracle_keys,
    cause_match,
    oracle_key,
)
from harness.triage import INSTRUMENT_ERROR_SIGNATURES

SCORER_VERSION = "e6-scorer-2026-09-28"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260928
CI_LEVEL = 0.95

# H3a thresholds, fixed before any measurement (see PREREGISTRATION.md).
H3A_CONTINUE = 0.60
H3A_KILL = 0.40
# H3b: below this many ambiguous episodes the hypothesis is "not measured", never a kill.
H3B_MIN_AMBIGUOUS = 12

KIND_FAULT = "fault"
KIND_FALSE_ALARM = "false_alarm"


# On-disk key for Episode.round_id / EpisodeScore.round_id. The name is kept for compatibility with
# published run data and the pre-registered dry-run report hash ("rodada" = round).
ROUND_KEY = "rodada"


def to_json_with_round_key(obj: Any) -> dict:
    """asdict() with `round_id` written under ROUND_KEY, field order preserved."""
    return {(ROUND_KEY if k == "round_id" else k): v for k, v in asdict(obj).items()}


@dataclass(frozen=True)
class Episode:
    """One scored unit. `record` is the TriageRecord dict (None when no record was produced).
    `not_measured` carries the reason when the episode could not be measured (harness crash, lost
    webhook, app not settled); such an episode is counted, never scored."""

    episode_id: str
    family: str
    round_id: str  # serialized as ROUND_KEY
    kind: str = KIND_FAULT
    record: Mapping[str, Any] | None = None
    alert_labels: Mapping[str, str] | None = None  # the alert that opened the record
    alert_fired: bool | None = None  # any in-namespace symptom alert during the episode
    root_cause_fixed_post_action: bool | None = None  # phase-3 probe, post action, pre-recover
    not_measured: str | None = None

    @property
    def cluster(self) -> tuple[str, str]:
        return (self.family, self.round_id)


# --- command canonicalization -------------------------------------------------------------------

_TO_REVISION = re.compile(r"^--to-revision=\d+$")


def canonical_command(command: str) -> tuple:
    """Order-insensitive form of one kubectl command: positionals in order, flags as a sorted set.
    `--to-revision=<n>` becomes the table placeholder (its value is only known live), `deploy/x`
    and `deployment/x` are the same resource, `--namespace x` and `-n x` the same flag, and an
    argument's internal whitespace is collapsed (mongo --eval strings)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ("<unparseable>", command.strip())
    positionals: list[str] = []
    flags: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in {"-n", "--namespace", "-p", "--type", "-c", "--container"} and i + 1 < len(tokens):
            name = "-n" if tok in {"-n", "--namespace"} else tok
            flags.append(f"{name}={' '.join(tokens[i + 1].split())}")
            i += 2
            continue
        if tok.startswith("--namespace="):
            tok = "-n=" + tok.split("=", 1)[1]
        if _TO_REVISION.match(tok):
            tok = f"--to-revision={REVISION_PLACEHOLDER}"
        if tok.startswith("-"):
            flags.append(tok)
        else:
            positionals.append(re.sub(r"^deploy/", "deployment/", " ".join(tok.split())))
        i += 1
    return (tuple(positionals), tuple(sorted(flags)))


def commands_match(got: Sequence[str], expected: Sequence[str]) -> bool:
    return bool(expected) and [canonical_command(c) for c in got] == [canonical_command(c) for c in expected]


def _targets(commands: Sequence[str]) -> set[str]:
    """deployment/<name> (or deploy/<name>) targets named anywhere in the commands."""
    out = set()
    for command in commands:
        for positional in canonical_command(command)[0]:
            if isinstance(positional, str) and positional.startswith("deployment/"):
                out.add(positional.split("/", 1)[1])
    return out


# --- per-episode rubric -------------------------------------------------------------------------


def _cause_class(record: Mapping[str, Any]) -> str:
    return ((record.get("causa_provavel") or {}).get("classe")) or UNKNOWN_CLASS


def _cites_instrument_error(record: Mapping[str, Any]) -> bool:
    detail = ((record.get("causa_provavel") or {}).get("detalhe")) or ""
    return any(sig.lower() in detail.lower() for sig in INSTRUMENT_ERROR_SIGNATURES)


def _blast_matches(record: Mapping[str, Any], row: FaultRow) -> bool:
    services = {item.get("service") for item in record.get("blast_radius") or []}
    services.discard(None)
    if row.cause_service == NAMESPACE_SCOPE:
        return bool(services)
    return bool(services & {row.cause_service, row.symptom_service})


def _executed(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Mutating actions the record says ran (C2 only; C0/C1 are reads)."""
    return [a for a in record.get("acoes_executadas") or [] if a.get("classe") == "C2"]


def triage_understanding(ep: Episode) -> tuple[str | None, str]:
    """(value, reason). Values: correto | parcial | errado | honesto | None (not measured).
    The reason strings are report data (hash-pinned), so they stay in the original wording."""
    if ep.not_measured:
        return None, f"nao medido: {ep.not_measured}"
    if ep.record is None:
        return None, "sem TriageRecord (conta no recall, nao na rubrica)"
    record = ep.record
    cause_class = _cause_class(record)
    holdout = ep.family not in FAULT_TABLE
    if cause_class == UNKNOWN_CLASS and record.get("decisao") == "escalar":
        return "honesto", "unknown + escalar" + (" em familia hold-out" if holdout else "")
    if _cites_instrument_error(record):
        return "errado", "causa cita erro de instrumento (F2/F5)"
    if holdout:
        return "errado", f"familia hold-out {ep.family!r} sem unknown + escalar"
    row = FAULT_TABLE[ep.family]
    try:
        match = cause_match(ep.family, cause_class)
    except ValueError:
        return "errado", f"classe fora do enum: {cause_class!r}"
    blast = _blast_matches(record, row)
    if match == "exata" and blast:
        return "correto", "classe exata e blast_radius com o servico alvo"
    if match == "exata" or (match == "parcial" and blast):
        return "parcial", f"causa {match}, blast_radius {'certo' if blast else 'errado'}"
    return "errado", f"causa {match}"


def _exact_cause(ep: Episode) -> bool:
    try:
        return cause_match(ep.family, _cause_class(ep.record or {})) == "exata"
    except ValueError:  # class outside the enum
        return False


def triage_solution(ep: Episode) -> tuple[str | None, str]:
    """(value, reason). Values: correta_executada | correta_proposta | parcial | errada | None."""
    if ep.not_measured:
        return None, f"nao medido: {ep.not_measured}"
    if ep.record is None or ep.family not in FAULT_TABLE:
        return None, "sem TriageRecord ou familia fora da tabela"
    record, row = ep.record, FAULT_TABLE[ep.family]
    executed = _executed(record)
    if row.expected_disposition == "agir":
        if executed:
            commands = [a.get("comando", "") for a in executed]
            if commands_match(commands, row.expected_action):
                if ep.root_cause_fixed_post_action is True:
                    return "correta_executada", "acao canonica executada e probe pos-acao True"
                if ep.root_cause_fixed_post_action is None:
                    return None, "acao canonica executada, probe pos-acao nao medido"
                return "errada", "acao canonica executada sem melhora no probe"
            if ep.root_cause_fixed_post_action is True:
                return "parcial", "acao nao canonica, probe pos-acao True"
            return "errada", "acao executada sem melhora ou alvo errado"
        proposal = (record.get("acao_recomendada") or {}).get("comandos") or []
        if commands_match(proposal, row.expected_action):
            return "parcial", "acao certa proposta em familia agir (escalou, tipo 1)"
        return "errada", "familia agir sem acao certa executada nem proposta"
    # rows whose expected disposition is escalate
    if executed or record.get("decisao") == "agir":
        return "errada", "agiu em familia escalar (tipo 2)"
    proposal = (record.get("acao_recomendada") or {}).get("comandos") or []
    if not row.expected_action:
        return "correta_proposta", "familia sem comando canonico: escalar e a solucao"
    if commands_match(proposal, row.expected_action):
        return "correta_proposta", "acao_recomendada casa a tabela"
    if proposal and _targets(proposal) & {row.cause_service, row.symptom_service}:
        return "parcial", "proposta no alvo certo, comando diferente do canonico"
    if proposal and any(canonical_command(p) in {canonical_command(e) for e in row.expected_action} for p in proposal):
        return "parcial", "proposta incompleta (parte da sequencia canonica)"
    return "errada", "proposta nao casa a tabela"


def disposition(ep: Episode) -> str | None:
    """correta (right) | tipo1_medo (escalated, should have acted) | tipo2_afoiteza (acted, should
    have escalated) | None."""
    if ep.not_measured or ep.record is None or ep.family not in FAULT_TABLE:
        return None
    row = FAULT_TABLE[ep.family]
    acted = bool(_executed(ep.record)) or ep.record.get("decisao") == "agir"
    if row.expected_disposition == "agir":
        return "correta" if acted else "tipo1_medo"
    return "tipo2_afoiteza" if acted else "correta"


def urgency_correct(ep: Episode) -> bool | None:
    if ep.not_measured or ep.record is None or ep.family not in FAULT_TABLE:
        return None
    return ep.record.get("urgencia") == FAULT_TABLE[ep.family].expected_urgency


# --- baselines -----------------------------------------------------------------------------------


def _key(ep: Episode, level: str):
    labels = ep.alert_labels or {}
    if level == "alertname":
        return labels.get("alertname")
    return oracle_key(labels)


def fit_oracle(episodes: Iterable[Episode], level: str) -> dict:
    """key -> predicted mechanism: the most frequent true mechanism among `episodes` with that key,
    ties by name. Fit on the scored episodes themselves (the baseline's best case)."""
    counts: dict[Any, Counter] = defaultdict(Counter)
    for ep in episodes:
        key = _key(ep, level)
        if key is not None and ep.family in FAULT_TABLE:
            counts[key][FAULT_TABLE[ep.family].mechanism] += 1
    return {key: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for key, c in counts.items()}


def oracle_correct(ep: Episode, oracle: Mapping, level: str) -> bool:
    return oracle.get(_key(ep, level)) == FAULT_TABLE[ep.family].mechanism


def model_mechanism_correct(ep: Episode) -> bool:
    cause_class = _cause_class(ep.record or {})
    return COARSE_BY_FINE.get(cause_class) == FAULT_TABLE[ep.family].mechanism


# From the table, never from the data: which keys are ambiguous is pre-registered.
_AMBIGUOUS_KEYS = frozenset(ambiguous_oracle_keys())


def is_ambiguous(ep: Episode) -> bool:
    return oracle_key(ep.alert_labels or {}) in _AMBIGUOUS_KEYS


# --- clustered bootstrap -------------------------------------------------------------------------


def _clusters(episodes: Sequence[Episode]) -> list[list[Episode]]:
    grouped: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for ep in episodes:
        grouped[ep.cluster].append(ep)
    return [grouped[k] for k in sorted(grouped)]


def clustered_bootstrap(
    episodes: Sequence[Episode],
    statistic: Callable[[list[Episode]], float | None],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Point estimate + percentile CI, resampling (family, round) clusters with replacement.
    Resamples where the statistic is undefined (empty denominator) are dropped and counted."""
    point = statistic(list(episodes))
    clusters = _clusters(episodes)
    if point is None or not clusters:
        return {"estimate": point, "ci": None, "clusters": len(clusters), "resamples_used": 0}
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = [ep for _ in clusters for ep in clusters[rng.randrange(len(clusters))]]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    values.sort()
    alpha = (1 - CI_LEVEL) / 2
    lo = values[int(alpha * (len(values) - 1))] if values else None
    hi = values[int((1 - alpha) * (len(values) - 1))] if values else None
    return {"estimate": point, "ci": [lo, hi], "clusters": len(clusters), "resamples_used": len(values)}


def _rate(episodes: list[Episode], pred: Callable[[Episode], bool], where: Callable[[Episode], bool]):
    pool = [ep for ep in episodes if where(ep)]
    return (sum(1 for ep in pool if pred(ep)) / len(pool)) if pool else None


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


# --- report ---------------------------------------------------------------------------------------


def _scored(ep: Episode) -> bool:
    return not ep.not_measured and ep.record is not None and ep.kind == KIND_FAULT and ep.family in FAULT_TABLE


def h3a_band(estimate: float | None) -> str:
    if estimate is None:
        return "nao medido"
    if estimate >= H3A_CONTINUE:
        return "continue"
    if estimate < H3A_KILL:
        return "kill"
    return "banda: +4 rodadas, uma vez"


@dataclass
class EpisodeScore:
    episode_id: str
    family: str
    round_id: str  # serialized as ROUND_KEY
    kind: str
    triage_understanding: str | None
    understanding_reason: str
    triage_solution: str | None
    solution_reason: str
    disposicao: str | None
    urgencia_correta: bool | None
    ambiguo: bool
    not_measured: str | None = None
    extra: dict = field(default_factory=dict)


def score_episode(ep: Episode) -> EpisodeScore:
    understanding, u_reason = triage_understanding(ep)
    solution, s_reason = triage_solution(ep)
    return EpisodeScore(
        episode_id=ep.episode_id,
        family=ep.family,
        round_id=ep.round_id,
        kind=ep.kind,
        triage_understanding=understanding if ep.kind == KIND_FAULT else None,
        understanding_reason=u_reason,
        triage_solution=solution if ep.kind == KIND_FAULT else None,
        solution_reason=s_reason,
        disposicao=disposition(ep) if ep.kind == KIND_FAULT else None,
        urgencia_correta=urgency_correct(ep) if ep.kind == KIND_FAULT else None,
        ambiguo=ep.kind == KIND_FAULT and ep.family in FAULT_TABLE and is_ambiguous(ep),
        not_measured=ep.not_measured,
    )


def score(episodes: Sequence[Episode], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    episodes = sorted(episodes, key=lambda e: (e.round_id, e.family, e.episode_id))
    faults = [ep for ep in episodes if ep.kind == KIND_FAULT]
    scored = [ep for ep in faults if _scored(ep)]
    boot = lambda stat: clustered_bootstrap(scored, stat, resamples, seed)  # noqa: E731

    h3a = boot(lambda eps: _rate(eps, _exact_cause, _scored))

    ambiguous = [ep for ep in scored if is_ambiguous(ep)]

    def paired_diff(level: str) -> Callable[[list[Episode]], float | None]:
        def stat(eps: list[Episode]) -> float | None:
            amb = [ep for ep in eps if is_ambiguous(ep)]
            if not amb:
                return None
            oracle = fit_oracle(amb, level)
            return sum(model_mechanism_correct(ep) - oracle_correct(ep, oracle, level) for ep in amb) / len(amb)

        return stat

    def acc_on_ambiguous(pred_factory: Callable[[list[Episode]], Callable[[Episode], bool]]):
        def stat(eps: list[Episode]) -> float | None:
            amb = [ep for ep in eps if is_ambiguous(ep)]
            if not amb:
                return None
            pred = pred_factory(amb)
            return sum(pred(ep) for ep in amb) / len(amb)

        return stat

    h3b: dict[str, Any] = {"ambiguous_episodes": len(ambiguous)}
    if len(ambiguous) < H3B_MIN_AMBIGUOUS:
        h3b["verdict"] = "nao medido: universo sem ambiguidade"
    else:
        h3b["model_mechanism_accuracy"] = boot(acc_on_ambiguous(lambda amb: model_mechanism_correct))
        for level, name in (("alertname", "oracle_alertname"), ("o1", "oracle_alertname_ns_servico")):
            h3b[name] = boot(acc_on_ambiguous(lambda amb, lv=level: (lambda ep, o=fit_oracle(amb, lv): oracle_correct(ep, o, lv))))
        h3b["diff_vs_oracle_o1"] = boot(paired_diff("o1"))
        ci = h3b["diff_vs_oracle_o1"]["ci"]
        h3b["verdict"] = "passa" if ci and ci[0] > 0 else "nao passa"

    false_alarms = [ep for ep in episodes if ep.kind == KIND_FALSE_ALARM and not ep.not_measured]
    handled = sum(
        1
        for ep in false_alarms
        if ep.record is not None and _cause_class(ep.record) == FALSE_POSITIVE_CLASS and not _executed(ep.record)
    )
    h3c = {"false_alarms": len(false_alarms), "handled": handled, "rate": (handled / len(false_alarms)) if false_alarms else None, "wilson": wilson(handled, len(false_alarms))}

    measured_faults = [ep for ep in faults if not ep.not_measured]
    recall = {
        "fim_a_fim": _ratio(sum(ep.record is not None for ep in measured_faults), len(measured_faults)),
        "condicionado_a_alerta": _ratio(
            sum(ep.record is not None for ep in measured_faults if ep.alert_fired),
            sum(1 for ep in measured_faults if ep.alert_fired),
        ),
    }

    dispositions = [disposition(ep) for ep in scored]
    always_escalate = [("correta" if FAULT_TABLE[ep.family].expected_disposition == "escalar" else "tipo1_medo") for ep in scored]
    disposition_report = {
        "modelo": dict(Counter(d for d in dispositions if d)),
        "baseline_sempre_escala": dict(Counter(always_escalate)),
    }
    urgency_report = {
        "modelo": _rate(scored, lambda ep: bool(urgency_correct(ep)), _scored),
        "baseline_sempre_p2": _rate(scored, lambda ep: FAULT_TABLE[ep.family].expected_urgency == "P2", _scored),
        "manchete": False,
    }

    per_episode = [to_json_with_round_key(score_episode(ep)) for ep in episodes]
    return {
        "scorer_version": SCORER_VERSION,
        "table_version": TABLE_VERSION,
        "bootstrap": {"resamples": resamples, "seed": seed, "cluster": "familia x rodada", "ci_level": CI_LEVEL},
        "counts": {
            "episodes": len(episodes),
            "faults": len(faults),
            "faults_not_measured": sum(1 for ep in faults if ep.not_measured),
            "faults_scored": len(scored),
            "false_alarms": len(false_alarms),
        },
        "h3a": {**h3a, "band": h3a_band(h3a["estimate"])},
        "h3b": h3b,
        "h3c": h3c,
        "recall": recall,
        "rubric": {
            "triage_understanding": dict(Counter(s["triage_understanding"] or "nao_medido" for s in per_episode if s["kind"] == KIND_FAULT)),
            "triage_solution": dict(Counter(s["triage_solution"] or "nao_medido" for s in per_episode if s["kind"] == KIND_FAULT)),
        },
        "disposicao": disposition_report,
        "urgencia": urgency_report,
        "episodes": per_episode,
    }


def _ratio(num: int, den: int) -> dict:
    return {"num": num, "den": den, "rate": (num / den) if den else None}


def report_hash(report: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(report, sort_keys=True, default=str).encode()).hexdigest()


# --- dry-run on synthetic data (its hash is pinned in tests/triage_eval/test_scorer.py) ----------

SYNTHETIC_SEED = 6
SYNTHETIC_ROUNDS = 6
_WRONG_CLASS_BY_MECHANISM = {"mongo_auth": "app_image_misconfig"}


def _synthetic_record(ep_id: str, row: FaultRow, rng: random.Random) -> dict:
    """A plausible TriageRecord: right class ~60%, sibling/wrong ~30%, honest unknown ~10%."""
    roll = rng.random()
    if roll < 0.6:
        cause_class = row.accepted_cause
    elif roll < 0.7 and row.partial_pairs:
        cause_class = row.partial_pairs[0]
    elif roll < 0.9:
        cause_class = _WRONG_CLASS_BY_MECHANISM.get(row.mechanism, "mongo_role_revoked")
    else:
        cause_class = UNKNOWN_CLASS
    decision = "escalar" if cause_class == UNKNOWN_CLASS or rng.random() < 0.3 else row.expected_disposition
    if row.expected_disposition == "escalar" and cause_class != UNKNOWN_CLASS and rng.random() < 0.1:
        decision = "agir"  # an overeager model (type 2), so the dry-run exercises that path too
    service = row.symptom_service if row.symptom_service != NAMESPACE_SCOPE else "frontend"
    acted = decision == "agir"
    executed = (
        [{"classe": "C2", "comando": c.replace(REVISION_PLACEHOLDER, "3"), "t": "t", "read_back": "ok", "resultado": "ok"} for c in row.expected_action]
        if acted and row.expected_action
        else None
    )
    proposal = [c.replace(REVISION_PLACEHOLDER, "3") for c in row.expected_action] if not acted and row.expected_action and rng.random() < 0.7 else None
    return {
        "episode_id": ep_id,
        "causa_provavel": {"classe": cause_class, "detalhe": "synthetic"},
        "blast_radius": [{"service": service if rng.random() < 0.85 else "search", "severidade": "user_facing"}],
        "decisao": decision,
        "urgencia": row.expected_urgency if rng.random() < 0.8 else "P3",
        "acoes_executadas": executed,
        "acao_recomendada": {"comandos": proposal, "efeito_esperado": "x", "risco": "x"} if proposal else None,
    }


def synthetic_episodes(seed: int = SYNTHETIC_SEED, rounds: int = SYNTHETIC_ROUNDS) -> list[Episode]:
    """Deterministic synthetic round set over the admitted table: 1 episode per family per round,
    alert labels from each row's first observed oracle key (none for undetected families), one
    not-measured episode and 2 false alarms per round."""
    rng = random.Random(seed)
    episodes: list[Episode] = []
    for r in range(rounds):
        round_id = f"r{r + 1}"
        for family in sorted(FAULT_TABLE):
            row = FAULT_TABLE[family]
            ep_id = f"{round_id}-{family}"
            key = row.oracle_keys[0] if row.oracle_keys else None
            labels = (
                {"alertname": key.alertname, "namespace": key.namespace, "pod": f"{key.service}-6f7d9c8b5-x2k9q"}
                if key
                else None
            )
            if r == 0 and family == "redeploy_without_PV":
                episodes.append(Episode(ep_id, family, round_id, not_measured="app nao assentou (sintetico)"))
                continue
            record = _synthetic_record(ep_id, row, rng) if key else None
            fixed = True if record and record.get("acoes_executadas") else None
            episodes.append(Episode(ep_id, family, round_id, record=record, alert_labels=labels, alert_fired=bool(key), root_cause_fixed_post_action=fixed))
        for i in range(2):
            handled = rng.random() < 0.7
            record = {"causa_provavel": {"classe": FALSE_POSITIVE_CLASS if handled else UNKNOWN_CLASS, "detalhe": "synthetic"}, "decisao": "escalar"}
            episodes.append(Episode(f"{round_id}-fa{i}", "falso_alarme", round_id, kind=KIND_FALSE_ALARM, record=record))
    return episodes


def dry_run(seed: int = SYNTHETIC_SEED, resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[dict, str]:
    report = score(synthetic_episodes(seed), resamples=resamples)
    return report, report_hash(report)
