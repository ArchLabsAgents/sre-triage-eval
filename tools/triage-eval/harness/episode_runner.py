"""The real orchestrator behind `entrypoint.py`'s `episode_runner` seam.

What this module does: collect -> decide -> classify -> execute.

1. `collect_episode_context` runs the 90s scripted, read-only collection: the alert payload, a
   Prometheus range query per alert (recovered from the alert's own `generatorURL`, see
   `_extract_promql_from_generator_url`), `kubectl get pods/events/deployments -o json` for the
   target namespace, and `kubectl logs --tail=200` for every not-Ready pod found. Each source is
   collected independently and honestly: a source that fails or that the 90s budget ran out
   before reaching is recorded as `collected=False` with a `reason`, never silently dropped or
   replaced with a guess (same honesty convention as `harness/triage.py` and
   `scenarios/collect_pod_footprint.py`).
2. That context becomes the prompt via `harness.agent.Agent`/`harness.schema` (reused, not
   reimplemented): `LiveClusterEnvironment.task_description()` feeds it in, and the model's
   response is parsed and schema-validated by `Agent` exactly as it is for the benchmark
   adapters.
3. Every action the model proposes is classified by `harness.envelope.classify()` BEFORE
   anything runs, never by the model itself.
4. `classify()` allows (C0/C1/C2) -> the command executes for real (`CommandRunner`, default
   `subprocess_command_runner`) and the real stdout/stderr feeds back as the next observation.
   `classify()` denies (always C3, or a gated C0-C2 denial) -> nothing executes; the denial and
   its reason are logged and fed back as the observation instead.

TriageRecord assembly lives in `harness/triage_assembly.py`, a separate module, so this file's
collect/decide/classify/execute loop stays the single thing it is; assembly reads this module's
data, it does not live inside it. Wired at `make_episode_runner`'s level: every call to the
returned runner, success or exception, ends in exactly one `triage_sink(record)` call (default:
logged as JSON), so a TriageRecord is emitted for every episode, including one that escalates.

Escalation triggers (the E1..E7 codes of `motivo_escalacao`): this module's own runtime
contributions are E4 (loop detection blocks a repeated failing command, in
`LiveClusterEnvironment.apply_action`/`_check_loop`), E3's data (`rollback_state`
pre-computation, read-back and rollback-on-no-improvement in
`LiveClusterEnvironment._execute_c2_with_rollback`, recorded in `environment.rollback_events`),
and threading `max_elapsed_seconds` into `harness.agent.Agent.run_episode` for the elapsed-time
half of E5 (the turn-count half is `max_turns`). E1 (confidence < 0.7), E2 (C3/C2-gate denial),
E6 (evidence coverage) and E7 (instrument-error recidivism) are computed from data this module
already exposes (`environment.denials`, `context.sources`, turn diagnoses); their assembly into
`motivo_escalacao` lives in `harness/triage_assembly.py`. That module also builds
`acoes_executadas` (from `environment.executions`) and `acao_recomendada` (always null with a
reason: no fault->action table is visible to the agent); `proxima_evidencia` is out of scope.

C2 dry-run gate: **every C2 mutation the envelope allows runs dry-run (logged, never actually
executed against the cluster) unless `SSLM_C2_DRY_RUN` is explicitly turned off.** The rollback
rule says a C2 without a rollback_state does not run; the rollback_state/read-back machinery now
exists, so there is no technical reason left for the gate, but its *default* stays `True`:
running real mutations against a real cluster is an explicit operational decision, not a code
default. See `c2_dry_run_enabled()` and `LiveClusterEnvironment`'s docstring for the toggle and
the exact mechanics; `c2_dry_run=False` (or `SSLM_C2_DRY_RUN=0`) is how the real path below is
exercised, and is what the tests do to cover it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

# cap_observation is a generic, dependency-free truncation helper (head+tail, stdlib only); it
# happens to live in adapters/aiopslab_env.py but importing it here triggers no AIOPSLAB_HOME
# requirement -- that module only imports `aiopslab` lazily, inside AiopsLabEnvironment methods,
# never at module import time (see that file). Reused rather than duplicated so the two prompt
# paths (benchmark adapter, live cluster) cap observations the same way.
from adapters.aiopslab_env import cap_observation, cap_within_share

from .agent import (
    DEFAULT_MAX_ELAPSED_SECONDS,
    Agent,
    AgentConfig,
    EpisodeResult,
    ModelClient,
    history_budget_enabled,
)
from .envelope import ActionClass, EnvelopeDecision, EpisodeEnvelopeState, classify

logger = logging.getLogger("sslm.episode_runner")

# Ceiling for the scripted, read-only collection.
SCRIPTED_COLLECTION_BUDGET_SECONDS = 90.0
# Range of the Prometheus series the alert cites.
PROMETHEUS_RANGE_MINUTES = 30.0
# Not specified by the design; a scoped choice (60s buckets over a 30min window = 30 points, small
# enough for the prompt, coarse enough that this is not the confound the aggregate cost pipeline
# (scenarios/collect_pod_footprint.py) cares about). Revisit if evidence shows it is too coarse.
PROMETHEUS_RANGE_STEP_SECONDS = 60.0
# `kubectl logs` tail for each not-Ready pod.
LOGS_TAIL_LINES = 200
# cap_observation() below bounds each individual source, but not how many capped sources get
# concatenated into one EpisodeContext.as_prompt_text(). A cascading
# failure (e.g. a bad rollout across a large ReplicaSet) can produce dozens of not-Ready pods;
# uncapped, that is exactly the failure mode adapters/aiopslab_env.py's OBSERVATION_MAX_CHARS
# comment documents (a single ~4MB observation drove the host to 13GB swap / load 181). Not a
# guess: same class of unbounded-concatenation bug as that measured incident. First N not-Ready
# pods, alphabetically first (deterministic given the same cluster state, not a claim about pod age -- Deployment-managed pod names carry a ReplicaSet
# hash + random suffix, so lexicographic order has no relationship to creationTimestamp); the
# rest are recorded as a single skipped ScriptedSource so the model still knows collection was
# incomplete rather than silently missing pods.
MAX_NOT_READY_PODS_LOGGED = 10
# Same failure class, same fix, for the OTHER unbounded-count axis in this module:
# harness/webhook.py's parse_webhook_payload only requires `alerts` to be a
# non-empty list, it does not cap its length, so an Alertmanager group with many alerts (the same
# bad-rollout-across-a-large-ReplicaSet scenario invoked to justify the cap above, just via a
# per-pod symptom-alert rule instead of the pod list) can still grow
# EpisodeContext.as_prompt_text() unboundedly through prometheus_range[i] sources.
MAX_ALERTS_QUERIED = 10
# Per-command timeout for a MODEL-proposed C0/C1/C2 action once classify() allows it (separate
# from the 90s scripted-collection budget above, which spans many commands).
DEFAULT_COMMAND_TIMEOUT_SECONDS = 30.0
_PROMETHEUS_QUERY_TIMEOUT_SECONDS = 10.0
# E5's turn cap ("8 turns or 15 min without a final decision"). The elapsed-time half of
# the same cap (`DEFAULT_MAX_ELAPSED_SECONDS`) lives in `harness.agent` -- see that module's
# docstring for why both caps live in `Agent.run_episode` rather than here: it is the one place
# shared by every caller that actually holds the turn loop and the clock. This module only
# threads `max_elapsed_seconds` through to it (before that, `Agent.run_episode`'s own default
# applied silently).
DEFAULT_MAX_TURNS = 8

# Rollback rule: "read back within 120s after the action". A single window shared by
# every C2 type (module docstring's "unified rollout target" choice below): rather than trying to
# reconsult whatever Prometheus metric the alert cited (there is no structured "the metric for
# this symptom" anywhere in this codebase -- see `_perform_read_back`'s own docstring for why that
# is a real gap, not an oversight), read-back re-checks the mutated resource's own K8s readiness
# via `kubectl rollout status`, which is deterministic, requires no metric-selection heuristic, and
# is the most direct operational proxy for "did the mutation help" a harness can check without a
# human in the loop.
READ_BACK_WINDOW_SECONDS = 120.0
# Not specified by the design (same "documented interpretation" convention as
# PROMETHEUS_RANGE_STEP_SECONDS above): how often to re-poll `kubectl rollout status` within the
# 120s window rather than blocking for the whole window on one call. Chosen small enough that a
# fast rollout is detected well before the window closes, coarse enough not to hammer the
# apiserver with a request every second.
READ_BACK_POLL_INTERVAL_SECONDS = 10.0

# The temporary C2 dry-run gate's env var (module docstring). Same "never a committed value"
# contract as VLLM_BASE_URL/APP_NAMESPACE/PROMETHEUS_URL in entrypoint.py: read once, at wiring
# time, never hardcoded. Default is ON (dry-run active) by construction -- see
# `c2_dry_run_enabled()`: an unset or unrecognized value degrades to the SAFE state (dry-run on),
# never to the permissive one, matching this repo's other degrade-safe env readers (e.g.
# entrypoint.py's malformed-numeric-env handling falls back to log-only, not to "assume it's fine").
C2_DRY_RUN_ENV_VAR = "SSLM_C2_DRY_RUN"
_C2_DRY_RUN_FALSE_VALUES = {"0", "false", "no", "off"}


def c2_dry_run_enabled(env: dict[str, str] | None = None) -> bool:
    """True unless C2_DRY_RUN_ENV_VAR is explicitly set to a recognized "off" value. `env`
    defaults to `os.environ`; a caller passes an explicit mapping only in tests, the same pattern
    `harness/envs.py` already uses elsewhere in this tool for env-driven config."""
    raw = (env if env is not None else os.environ).get(C2_DRY_RUN_ENV_VAR)
    if raw is None:
        return True
    return raw.strip().lower() not in _C2_DRY_RUN_FALSE_VALUES


# ---------------------------------------------------------------------------
# Command execution: the real thing that runs once classify() allows an action.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def __call__(self, command: str, *, timeout: float) -> CommandResult: ...


def subprocess_command_runner(command: str, *, timeout: float) -> CommandResult:
    """Default CommandRunner: argv-exec via subprocess, never `shell=True`. `harness.envelope.
    classify()` already refuses any command carrying a shell metacharacter before this runner
    ever sees it, but this function does not rely on that -- it parses with `shlex.split` and
    execs the resulting argv directly regardless of caller, so there is no second path in here a
    reviewer needs to trust separately."""
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return CommandResult(returncode=-1, stdout="", stderr=f"unparseable command: {exc}")
    if not tokens:
        return CommandResult(returncode=-1, stdout="", stderr="empty command")
    try:
        proc = subprocess.run(tokens, capture_output=True, text=True, timeout=timeout)  # noqa: S603 - argv exec, no shell
    except subprocess.TimeoutExpired as exc:
        return CommandResult(returncode=-1, stdout=exc.stdout or "", stderr=f"timed out after {timeout}s")
    except FileNotFoundError as exc:
        return CommandResult(returncode=-1, stdout="", stderr=str(exc))
    return CommandResult(returncode=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")


_SCALE_REPLICAS_RE = re.compile(r"--replicas[=\s](\d+)")


def _extract_scale_replicas(command: str) -> int | None:
    """`envelope.classify()` needs `scale_replicas` as a typed int for the scale gate; the model
    only emits a free-text `command` string. Recognizes both `--replicas=N` and `--replicas N`
    (matching how `harness.envelope._has_forbidden_flag` already handles both shapes for -f)."""
    match = _SCALE_REPLICAS_RE.search(command)
    if not match:
        return None
    return int(match.group(1))


def _command_references_target(command: str, target: str) -> bool:
    """Best-effort, token-based check (same style as `_extract_scale_replicas` above, not a
    kubectl API call or output parse) that `command` actually names `target` (`"<kind>/<name>"`,
    e.g. "pod/checkout-1"), instead of trusting the model's self-reported `target` field on its
    own. AGENT_ACTION_SCHEMA does not tie `target` to `command`, so without this a model could
    pair a cheap unrelated C0 read with an arbitrary `target` and have it marked "seen".

    Recognizes the shapes kubectl's own C0 verbs actually accept: `kubectl get pod/checkout-1`
    (kind/name as one token), `kubectl get pod checkout-1` (kind and name as separate tokens),
    and `kubectl logs checkout-1` (bare pod name, no kind prefix -- `logs` only ever targets
    pods). A command this does not recognize returns False (never a false "seen"): a false
    negative here only costs the model a wasted turn (envelope.classify still denies the
    eventual mutate until a real C0 call is recognized), a false positive would recreate the
    exact bypass this function exists to close, so this errs toward strict.

    Fixed after review: the bare-pod-name fallback used to accept `name` as ANY token
    in the command, on ANY C0 verb -- so `kubectl get pods -l checkout-1 -n ns` (a label
    selector, always exits 0 even matching zero pods) forged "pod/checkout-1 seen" without ever
    reading that pod. Scoped to exactly `kubectl logs <name> ...`, positional argument right
    after the verb, not a flag value: `tokens[2] == name` with `logs` as the verb, nothing looser.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    kind, sep, name = target.partition("/")
    if not sep or not name:
        return False
    if f"{kind}/{name}" in tokens:
        return True
    for i, tok in enumerate(tokens[:-1]):
        if tok == kind and tokens[i + 1] == name:
            return True
    if (
        kind == "pod"
        and len(tokens) >= 3
        and tokens[0] == "kubectl"
        and tokens[1] == "logs"
        and tokens[2] == name
    ):
        return True
    return False


def _format_command_result(result: CommandResult) -> str:
    parts = [f"exit={result.returncode}"]
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout}")
    if result.stderr:
        parts.append(f"stderr:\n{result.stderr}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# E4 loop detection: "the same normalized command (whitespace and pod name canonicalized)
# emitted twice with the same error" -> "blocked on the 3rd emission". Runtime effect, not
# just a TriageRecord statistic: `LiveClusterEnvironment.apply_action` calls `_check_loop` BEFORE
# `classify()`, so a detected loop's 3rd (and every later) emission never reaches the parser or
# the command runner at all -- it is blocked here, the same "never reclassified/re-executed"
# guarantee C3 gets from `harness.envelope`, just for a different reason (the repeat-a-failed-
# command failure mode, not the allowlist).
# ---------------------------------------------------------------------------

# Two normalization axes, both explicit interpretations of an underspecified doc line (same
# convention as `_extract_promql_from_generator_url` below, which flags its own reading of
# "the annotation's queries" the same way):
#   - whitespace: shlex-split then rejoin with single spaces, so "kubectl  get  pod x" and
#     "kubectl get pod x" compare equal.
#   - pod name canonicalization: a Deployment-managed pod name carries a ReplicaSet hash plus a
#     random per-pod suffix (e.g. "checkout-7d9f6c8b47-x8z2p"), which changes across
#     reschedules/restarts even when the model is retrying "the same" logical command against
#     "the same" logical pod. Without canonicalizing that suffix, a genuine loop would present as
#     a brand-new normalized command every time the pod happened to get rescheduled mid-episode,
#     defeating E4 silently (the design calls for semantic canonicalization if 3+ loops go
#     undetected in one round). The regex only matches the
#     two-hash-segment shape kubectl/K8s actually generates (ReplicaSet hash: hex, 6-10 chars;
#     pod suffix: base36, 5 chars) -- a token that merely LOOKS hash-like but has a different
#     shape (e.g. a bare deployment or namespace name) is left alone, erring toward
#     under-canonicalizing (a missed loop, still caught by later repeats) rather than
#     over-canonicalizing (collapsing two actually-different resources into "the same").
_POD_HASH_SUFFIX_RE = re.compile(r"^(?P<base>.+)-[0-9a-f]{6,10}-[a-z0-9]{5}$")


def _canonicalize_pod_token(token: str) -> str:
    match = _POD_HASH_SUFFIX_RE.match(token)
    if not match:
        return token
    return f"{match.group('base')}-<pod>"


def normalize_command_for_loop_detection(command: str) -> str:
    """E4's "normalized command". Returns `command.strip()` unchanged when it does not even
    parse as shell tokens -- an unparseable command is already C3 by construction
    (`harness.envelope.classify`), a distinct problem from loop detection, and this function must
    not raise on input `classify()` will reject anyway."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command.strip()
    return " ".join(_canonicalize_pod_token(tok) for tok in tokens)


def _failure_signature_for_denial(decision: EnvelopeDecision) -> str:
    return f"DENIED:{decision.action_class.value}:{decision.reason}"


def _failure_signature_for_result(result: CommandResult) -> str | None:
    """None means "not a failure, do not count this attempt toward a loop" -- a command that
    succeeded (exit 0) is not the loop pattern (repeating something that already FAILED), so
    it must not poison the history with a "matching" pair against a later, unrelated success."""
    if result.returncode == 0:
        return None
    return f"exit={result.returncode}:{(result.stderr or result.stdout).strip()[:200]}"


# ---------------------------------------------------------------------------
# Prometheus range query over the series the alert cites.
# ---------------------------------------------------------------------------


class PrometheusQueryError(Exception):
    """Transport or response-shape failure. Same contract as scenarios/collect_pod_footprint.py's
    exception of the same name: never turned into a fabricated result -- the caller records
    `collected=False` with this exception's text as the reason, never a guessed value."""


def _http_get_json(url: str, *, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - trusted lab endpoint, same convention as collect_pod_footprint.py
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PrometheusQueryError(f"transport failed querying {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PrometheusQueryError(f"non-JSON response from {url}: {exc}") from exc
    except ValueError as exc:
        # urlopen raises a bare ValueError, not URLError, for a
        # malformed/schemeless URL (e.g. PROMETHEUS_URL="monitoring-prometheus:9090" with no
        # "http://" prefix -- a plausible operator typo). Uncaught, this propagated past
        # collect_episode_context's per-source `except PrometheusQueryError` and aborted the
        # WHOLE episode via make_episode_runner's blanket except, instead of degrading to one
        # collected=False Prometheus source like every other per-source failure in this module.
        raise PrometheusQueryError(f"malformed URL querying {url}: {exc}") from exc


def query_range(
    prometheus_url: str,
    promql: str,
    *,
    start: float,
    end: float,
    step_seconds: float = PROMETHEUS_RANGE_STEP_SECONDS,
    timeout: float = _PROMETHEUS_QUERY_TIMEOUT_SECONDS,
) -> list:
    """Query Prometheus's `/api/v1/query_range`. Returns the raw `data.result` list (possibly
    empty: absence of series is not an error, same distinction `collect_pod_footprint._query_
    instant` makes)."""
    params = {"query": promql, "start": f"{start:.3f}", "end": f"{end:.3f}", "step": str(step_seconds)}
    url = f"{prometheus_url.rstrip('/')}/api/v1/query_range?{urllib.parse.urlencode(params)}"
    body = _http_get_json(url, timeout=timeout)
    if body.get("status") != "success":
        raise PrometheusQueryError(f"Prometheus returned status={body.get('status')!r}: {body}")
    return body.get("data", {}).get("result", [])


def _extract_promql_from_generator_url(generator_url: str) -> str | None:
    """The design says "Prometheus: the annotation's queries". Read literally, that would be
    the alert's `annotations` map -- but kube-prometheus-stack's default rules, and this repo's
    own (`kube-prometheus-stack/rules/symptom-alerts.yaml`), only put human-readable text there
    (`summary`, `description`), never a runnable PromQL string. The one place Alertmanager's
    webhook payload actually carries the fired rule's *query* is `generatorURL`: Prometheus's own
    graph link for the alert, of the form `<url>/graph?g0.expr=<expr>&g0.tab=1`. This function
    recovers `g0.expr` from that link, which is what this module treats as "the annotation's
    query" in practice. Flagged here explicitly as an interpretation of an underspecified
    doc line, not a hardcoded guess: if a future alert-rule convention adds a real
    query-carrying annotation, prefer that over this."""
    if not generator_url:
        return None
    parsed = urllib.parse.urlparse(generator_url)
    params = urllib.parse.parse_qs(parsed.query)
    values = params.get("g0.expr")
    if not values:
        return None
    return values[0]


# ---------------------------------------------------------------------------
# kube-apiserver collection: pods/events/deployments of the target namespace and
# `kubectl logs --tail=200` of the not-Ready pods.
# ---------------------------------------------------------------------------


def _kubectl_get_json(
    command_runner: CommandRunner, resource: str, namespace: str, *, timeout: float
) -> tuple[dict | None, str | None]:
    """Runs `kubectl get <resource> -n <namespace> -o json`. Returns (parsed, None) on success,
    (None, reason) otherwise -- never raises: this is scripted collection, and one source failing
    must not abort the others or the episode (same honesty rule as harness/triage.py: unmeasured
    is `None` with a reason)."""
    command = f"kubectl get {resource} -n {namespace} -o json"
    result = command_runner(command, timeout=timeout)
    if result.returncode != 0:
        return None, f"{command!r} exited {result.returncode}: {(result.stderr or result.stdout).strip()}"
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"{command!r} returned non-JSON stdout: {exc}"


# Section 3.1's rollback rule for `rollout undo`: "revisao vigente lida antes" -- the current
# revision has to come from a real read, never an assumption. `kubectl rollout history <target>`
# prints a table (header line, then one "<revision>  <change-cause>" line per revision this
# Deployment's revisionHistoryLimit retained); the CURRENT revision is the highest number listed
# (Kubernetes assigns revisions monotonically increasing, and prunes old ones off the bottom, so
# the max seen is always the one actually running). Matches only lines that start with a bare
# integer, so the header line ("REVISION  CHANGE-CAUSE") and the resource-name line above it never
# parse as a revision.
_ROLLOUT_HISTORY_REVISION_RE = re.compile(r"^(\d+)\s")


def _parse_current_revision_from_rollout_history(output: str) -> int | None:
    revisions = [
        int(match.group(1))
        for match in (_ROLLOUT_HISTORY_REVISION_RE.match(line.strip()) for line in output.splitlines())
        if match is not None
    ]
    return max(revisions) if revisions else None


def _controller_owner_reference(resource_json: dict) -> tuple[str, str] | None:
    """Rollback rule for `delete pod`: "annotate the ReplicaSet name". Read from the
    resource's own `metadata.ownerReferences` (real API data, never guessed from the pod's name --
    a Deployment-managed pod name embeds its ReplicaSet's hash, but parsing that back out of the
    name is exactly the kind of "assumed, not observed" shortcut the read-before-mutate rule
    exists to forbid). Prefers the entry with `controller: true` (the one K8s itself
    considers authoritative when a resource has multiple ownerReferences); falls back to the first
    entry when none is marked controller (should not happen for a pod/ReplicaSet in practice, but
    this function does not assume it never will). Returns None, never a fabricated owner, when
    there is nothing to read."""
    owners = (resource_json.get("metadata") or {}).get("ownerReferences") or []
    for owner in owners:
        if owner.get("controller") is True and owner.get("kind") and owner.get("name"):
            return owner["kind"], owner["name"]
    if owners and owners[0].get("kind") and owners[0].get("name"):
        return owners[0]["kind"], owners[0]["name"]
    return None


def _non_ready_pod_names(pods_json: dict) -> list[str]:
    names = []
    for item in pods_json.get("items", []):
        name = (item.get("metadata") or {}).get("name")
        if not name:
            continue
        conditions = (item.get("status") or {}).get("conditions") or []
        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
        if not ready:
            names.append(name)
    # Sorted, not apiserver list order: apiserver order is not a stable contract, and
    # MAX_NOT_READY_PODS_LOGGED below needs "which pods get logged" to be deterministic given the
    # same cluster state, not an artifact of whatever order `kubectl get pods -o json` happened
    # to return this call.
    return sorted(names)


def _targets_from_list_json(kind_prefix: str, list_json: dict) -> set[str]:
    """`target` strings this module treats as "seen via C0 this episode" for the read-before-
    mutate gate (harness.envelope's C2 common gate, `state.targets_seen`). Shape
    `"<kind>/<name>"`, matching the examples in envelope.py's own docstrings (e.g.
    "deployment/recommendation")."""
    seen = set()
    for item in list_json.get("items", []):
        name = (item.get("metadata") or {}).get("name")
        if name:
            seen.add(f"{kind_prefix}/{name}")
    return seen


# ---------------------------------------------------------------------------
# The scripted collection itself.
# ---------------------------------------------------------------------------


@dataclass
class ScriptedSource:
    name: str
    collected: bool
    reason: str | None = None
    text: str = ""


@dataclass
class EpisodeContext:
    episode_id: str
    namespace: str
    sources: list[ScriptedSource] = field(default_factory=list)
    budget_seconds: float = SCRIPTED_COLLECTION_BUDGET_SECONDS
    elapsed_seconds: float = 0.0

    def covered_chars(self, max_chars: int | None = None) -> int:
        """Chars of collected-source text that as_prompt_text(max_chars) renders: the part
        PromptBudget.context_chars bounds. Headers and NOT-collected lines are outside it."""
        texts = {i: s.text for i, s in enumerate(self.sources) if s.collected}
        if max_chars is not None:
            texts = _split_char_budget(texts, max_chars)
        return sum(len(t) for t in texts.values())

    def as_prompt_text(self, max_chars: int | None = None) -> str:
        """`max_chars` (harness.agent.PromptBudget.context_chars) bounds the sum of the
        collected sources' text, split so that every source keeps a share: the ones that fit in
        an even share stay whole and give back what they leave over, the rest are cut head+tail
        by cap_within_share. A single big dump (the pods JSON) can then no longer push events or
        logs out of the call. Headers and NOT-collected lines sit outside the bound. None keeps
        the text byte for byte as it was before the budget existed."""
        texts = {i: s.text for i, s in enumerate(self.sources) if s.collected}
        if max_chars is not None:
            texts = _split_char_budget(texts, max_chars)
        parts = [
            f"Episode {self.episode_id}, namespace {self.namespace}.",
            f"Scripted read-only collection (budget {self.budget_seconds:.0f}s, "
            f"took {self.elapsed_seconds:.1f}s):",
        ]
        for i, source in enumerate(self.sources):
            if source.collected:
                parts.append(f"\n--- {source.name} ---\n{texts[i]}")
            else:
                parts.append(f"\n--- {source.name} (NOT collected: {source.reason}) ---")
        return "\n".join(parts)


def _split_char_budget(texts: dict[int, str], max_chars: int) -> dict[int, str]:
    """Water-filling over `texts`: smallest first, each takes min(its length, an even share of
    what is left), so a short source never wastes budget a long one could use. Deterministic for
    the same input (ties keep source order)."""
    out: dict[int, str] = {}
    remaining = max_chars
    order = sorted(texts, key=lambda i: (len(texts[i]), i))
    for n, i in enumerate(order):
        share = remaining // (len(order) - n)
        text = texts[i]
        out[i] = cap_within_share(text, share)
        remaining -= min(len(text), share)
    return out


def collect_episode_context(
    episode_id: str,
    webhook_raw: dict,
    *,
    namespace: str,
    prometheus_url: str,
    command_runner: CommandRunner,
    budget_seconds: float = SCRIPTED_COLLECTION_BUDGET_SECONDS,
    per_source_timeout: float = 20.0,
    range_minutes: float = PROMETHEUS_RANGE_MINUTES,
    range_step_seconds: float = PROMETHEUS_RANGE_STEP_SECONDS,
    logs_tail_lines: int = LOGS_TAIL_LINES,
    clock=time.monotonic,
    wall_clock=time.time,
) -> tuple[EpisodeContext, set[str]]:
    """Section 2, step 2: the 90s scripted read-only collection. Returns the context (for the
    prompt) and the set of `targets_seen` this collection already confirmed exist (deployments
    and pods returned by `kubectl get ... -o json`), which the caller seeds
    `EpisodeEnvelopeState.targets_seen` with -- read-before-mutate is satisfied by the scripted
    collection alone when it already listed the target, not only by a model-issued C0 call.

    Budget discipline: every source checks remaining budget before it runs; a source reached
    after the budget is already spent is recorded `collected=False` with that reason, never
    skipped silently. A per-source failure (transport error, non-zero kubectl exit, bad JSON)
    never aborts the remaining sources.
    """
    start = clock()
    sources: list[ScriptedSource] = []
    targets_seen: set[str] = set()

    def remaining() -> float:
        return budget_seconds - (clock() - start)

    def has_budget() -> bool:
        return remaining() > 0

    def step_timeout() -> float:
        return max(min(per_source_timeout, remaining()), 0.1)

    # 1. Alert payload: no I/O, always collected.
    sources.append(
        ScriptedSource(
            name="alertmanager_payload",
            collected=True,
            text=cap_observation(json.dumps(webhook_raw, indent=2, sort_keys=True)),
        )
    )

    # 2. Prometheus: a 30-minute range query per alert, over the expr recovered from its
    #    generatorURL (see _extract_promql_from_generator_url). Capped at MAX_ALERTS_QUERIED
    #    (see its own comment); the rest are recorded as a single skipped ScriptedSource, same
    #    pattern as the not-ready-pod-logs cap below.
    all_alerts = webhook_raw.get("alerts") or []
    alerts, skipped_alerts = all_alerts[:MAX_ALERTS_QUERIED], all_alerts[MAX_ALERTS_QUERIED:]
    for index, alert in enumerate(alerts):
        source_name = f"prometheus_range[{index}]"
        if not has_budget():
            sources.append(ScriptedSource(source_name, False, "90s scripted-collection budget exhausted"))
            continue
        promql = _extract_promql_from_generator_url(alert.get("generatorURL", ""))
        if not promql:
            sources.append(
                ScriptedSource(
                    source_name, False, "alert carries no generatorURL/g0.expr to recover a query from"
                )
            )
            continue
        end = wall_clock()
        try:
            result = query_range(
                prometheus_url,
                promql,
                start=end - range_minutes * 60.0,
                end=end,
                step_seconds=range_step_seconds,
                timeout=step_timeout(),
            )
        except PrometheusQueryError as exc:
            sources.append(ScriptedSource(source_name, False, str(exc)))
            continue
        sources.append(
            ScriptedSource(
                source_name,
                True,
                text=cap_observation(f"query: {promql}\n{json.dumps(result, indent=2)}"),
            )
        )
    if skipped_alerts:
        sources.append(
            ScriptedSource(
                "prometheus_range[...skipped]",
                False,
                f"MAX_ALERTS_QUERIED={MAX_ALERTS_QUERIED} reached, {len(skipped_alerts)} more "
                "alert(s) in this group not queried",
            )
        )

    # 3. kube-apiserver: pods/events/deployments of the target namespace.
    pods_json: dict | None = None
    for resource, kind_prefix in (("pods", "pod"), ("events", None), ("deployments", "deployment")):
        source_name = f"k8s_{resource}"
        if not has_budget():
            sources.append(ScriptedSource(source_name, False, "90s scripted-collection budget exhausted"))
            continue
        parsed, reason = _kubectl_get_json(command_runner, resource, namespace, timeout=step_timeout())
        if parsed is None:
            sources.append(ScriptedSource(source_name, False, reason))
            continue
        sources.append(ScriptedSource(source_name, True, text=cap_observation(json.dumps(parsed, indent=2))))
        if resource == "pods":
            pods_json = parsed
        if kind_prefix is not None:
            targets_seen |= _targets_from_list_json(kind_prefix, parsed)

    # 4. kubectl logs --tail=200 of every not-Ready pod, only when the pods collection above
    #    actually succeeded (no pod list, no way to know which pods are not-Ready). Capped at
    #    MAX_NOT_READY_PODS_LOGGED (see its own comment): the 90s budget bounds wall-clock time,
    #    not the number of already-capped 12k-char sources concatenated into one prompt.
    if pods_json is not None:
        not_ready = _non_ready_pod_names(pods_json)
        logged, skipped = not_ready[:MAX_NOT_READY_PODS_LOGGED], not_ready[MAX_NOT_READY_PODS_LOGGED:]
        for pod_name in logged:
            source_name = f"k8s_logs[{pod_name}]"
            if not has_budget():
                sources.append(ScriptedSource(source_name, False, "90s scripted-collection budget exhausted"))
                continue
            command = f"kubectl logs {pod_name} -n {namespace} --tail={logs_tail_lines}"
            result = command_runner(command, timeout=step_timeout())
            if result.returncode != 0:
                sources.append(
                    ScriptedSource(source_name, False, f"{command!r} exited {result.returncode}: {result.stderr.strip()}")
                )
                continue
            sources.append(ScriptedSource(source_name, True, text=cap_observation(result.stdout)))
        if skipped:
            sources.append(
                ScriptedSource(
                    "k8s_logs[...skipped]",
                    False,
                    f"MAX_NOT_READY_PODS_LOGGED={MAX_NOT_READY_PODS_LOGGED} reached, "
                    f"{len(skipped)} more not-Ready pod(s) not logged: {', '.join(skipped)}",
                )
            )
    else:
        sources.append(
            ScriptedSource("k8s_logs", False, "skipped: pods collection did not succeed, no not-Ready pod list available")
        )

    elapsed = clock() - start
    context = EpisodeContext(
        episode_id=episode_id,
        namespace=namespace,
        sources=sources,
        budget_seconds=budget_seconds,
        elapsed_seconds=elapsed,
    )
    return context, targets_seen


# ---------------------------------------------------------------------------
# The live Environment: classify() gates every model-proposed action before it executes.
# ---------------------------------------------------------------------------


class LiveClusterEnvironment:
    """`harness.agent.Environment` over a real cluster.

    `task_description()` returns the scripted collection's text (built once, before the episode
    starts). `apply_action(action)` is where the safety envelope lives: every `inspect`/
    `remediate` action is classified by `harness.envelope.classify()` BEFORE it runs; only an
    allowed C0/C1/C2 executes for real, a denial (always C3, or a gated C0-C2 refusal) never
    reaches `command_runner` at all and is fed back to the model as the observation instead, so
    the loop can adapt. `escalate`/`submit` end the episode immediately by the model's own choice
    -- no command, nothing to classify.

    **A C2 action that clears classify()'s gates gets a pre-computed rollback_state and a
    read-back re-measurement of the symptom, when `c2_dry_run` is off.** The hard rollback rule
    requires both before a C2 is allowed to run; `_execute_c2_with_rollback` (below) is that
    machinery: it pre-computes `rollback_state` from a real read BEFORE the mutation runs (never
    executing a C2 whose rollback_state read failed: a C2 without a rollback_state does not run),
    then re-checks the mutated
    resource's readiness within a 120s window after it runs, and executes the registered rollback
    command (when one exists) if that check shows no improvement. `environment.rollback_events`
    records every real C2 execution's outcome; `harness/triage_assembly.py` reads it to fold "E3"
    into `motivo_escalacao` when 2 or more real C2 executions this episode showed no improvement.

    `c2_dry_run` (default True, see `c2_dry_run_enabled()`) still keeps this class from actually
    running a C2 mutation by default: the command is classified, counted against the episode's C2
    cap, and logged as what WOULD have run, but `command_runner` is never called for it, and the
    observation fed back to the model says so explicitly (so the model does not mistake a dry-run
    for a real fix). The original technical reason for the gate (no rollback_state/read-back
    machinery to ground a real mutation) is gone, but the gate's *default* does not change: see
    the module docstring's dry-run paragraph for why flipping it is an explicit operator decision.

    **Honest limit: `kubectl scale` is unreachable through this class,
    not merely gated.** `EpisodeEnvelopeState` here is created with no `scale_bounds` (nothing in
    this module or `entrypoint.py` populates it), and `envelope._classify_scale` denies any
    scale action outright when `state.scale_bounds.get(target) is None` -- fail-closed, so not a
    safety bug, but silently permanent rather than an explicit deferral like the two limits
    above. Declaring real `[min, max]` bounds per deployment is out of scope (it needs a source
    of truth for what bounds are safe per app, not yet designed).

    **Honest limit: no cross-episode coordination.** `entrypoint.py`'s
    `WebhookReceiver` runs on a `ThreadingHTTPServer`, one worker thread per connection, and each
    call to `run_real_episode` builds its own independent `EpisodeEnvelopeState`. Two different
    alerts in the same namespace (different fingerprints, so `EpisodeDeduper` opens separate
    episodes) can run concurrently against the same cluster resources, each with its own
    C0/C1/C2 caps and its own `targets_seen`, unaware of the other's in-flight mutation (e.g. one
    episode's `rollout restart` racing another's `scale`/`delete pod` on the same deployment).
    Acceptable for an evaluation harness (proving the loop, not a production path), but a real gap
    for any deployment that makes this the default handler.

    **E4 loop detection is a runtime effect here, not just a TriageRecord statistic.** Every
    `inspect`/`remediate` command is checked against this episode's own history (`_check_loop`,
    keyed by `normalize_command_for_loop_detection`) BEFORE `classify()` ever sees it: the 3rd emission of a normalized command that already
    failed identically twice in a row is blocked outright -- it is never reclassified, never
    reaches `command_runner`, and does not count against any C0/C1/C2 cap (the block happens
    before `_register_allowed`). `self.loop_detections` records every block, which is what
    `harness/triage_assembly.py` reads to fold "E4" into `motivo_escalacao`. E1, E2, E6 and E7
    are computed entirely from data this class (and `EpisodeContext`/`EpisodeResult`) exposes;
    E4 is the one trigger that needs a runtime effect here, because E4 is specifically about NOT
    re-running a command, not about describing one after the fact.

    **E3 (no-progress after 2 executed C2s with a failed read-back)** is raised via
    `rollback_events` above. `harness/triage_assembly.py` builds `acoes_executadas`/
    `acao_recomendada` directly from `self.executions`, with no changes needed in this class;
    `proxima_evidencia` remains out of scope.

    **Honest limit: read-back re-checks K8s readiness of the mutated resource, never the
    Prometheus metric that triggered the alert.** The design says the harness "re-measures the
    symptom"; there is no structured mapping anywhere in this codebase from an alert to "the one
    metric that defines its symptom" (the alert's `annotations` are human prose, and
    `_extract_promql_from_generator_url` already documents that the query recoverable from
    `generatorURL` is the alert RULE's query, evaluated over the alert's own label set, not
    parameterized per-resource in a way this module could safely re-run against an arbitrary
    target). `kubectl rollout status`/`kubectl get <target> -o json` readiness is the
    deterministic, always-available proxy this module uses instead -- documented here as a chosen
    interpretation, not a silent narrowing, same convention as `_extract_promql_from_generator_url`
    itself.

    **Honest limit: `delete pod`'s read-back can be inconclusive.** Unlike
    restart/undo/scale (whose `target` is already the workload resource `kubectl rollout status`
    understands), a deleted pod's own name stops existing the moment the delete succeeds -- the
    controller creates a *new* pod with a different name. Read-back for `delete pod` instead
    resolves the pod's owning Deployment (via ownerReferences: pod -> ReplicaSet -> Deployment) at
    rollback_state pre-compute time and rolls out-status-checks THAT. When the ReplicaSet's own
    owner cannot be resolved (e.g. a bare ReplicaSet with no Deployment, or the second read fails),
    read-back is honestly `None` (inconclusive) rather than fabricating a verdict -- `_e3_triggered`
    (harness/triage_assembly.py) treats an inconclusive read-back the same as a failed one (no
    confirmed improvement), the conservative direction for a trigger whose job is catching
    no-progress, not the lenient one.
    """

    def __init__(
        self,
        *,
        context: EpisodeContext,
        state: EpisodeEnvelopeState,
        command_runner: CommandRunner,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        c2_dry_run: bool = True,
        context_max_chars: int | None = None,
        read_back_window_seconds: float = READ_BACK_WINDOW_SECONDS,
        read_back_poll_interval_seconds: float = READ_BACK_POLL_INTERVAL_SECONDS,
        clock=time.monotonic,
        sleep_fn=time.sleep,
    ) -> None:
        self._context = context
        self._state = state
        self._command_runner = command_runner
        self._command_timeout = command_timeout
        self._c2_dry_run = c2_dry_run
        # PromptBudget.context_chars; None renders the context whole.
        self._context_max_chars = context_max_chars
        # Read-back timing. `clock`/`sleep_fn` are injectable the
        # same way `collect_episode_context`'s `clock`/`wall_clock` are -- a test drives the 120s
        # window deterministically instead of actually sleeping.
        self._read_back_window_seconds = read_back_window_seconds
        self._read_back_poll_interval_seconds = read_back_poll_interval_seconds
        self._clock = clock
        self._sleep_fn = sleep_fn
        # Evidence trail for this episode. Not a harness.triage.TriageRecord itself: a plain list
        # the caller (or a test) can read/log, and exactly the data harness/triage_assembly.py's
        # TriageRecord assembly reads (acao_proposta/acao_executada) -- this
        # class is where the decision is actually made. Every entry carries "dry_run": bool so the
        # assembler can tell a real C2 execution from a logged-only one without re-deriving it.
        self.denials: list[EnvelopeDecision] = []
        self.executions: list[dict] = []
        # E4 loop detection (class docstring above). Keyed by
        # `normalize_command_for_loop_detection(command)`, valued by the ordered list of failure
        # signatures every past attempt of that normalized command produced (denial reason, or
        # `exit=N:...` for a real failed execution) -- a successful attempt (exit 0) is never
        # appended (see `_failure_signature_for_result`), so a prior success never "matches" a
        # later failure and never poisons the loop check.
        self._command_failure_history: dict[str, list[str]] = {}
        # Every blocked (E4) emission, in order. `harness/triage_assembly.py` only checks
        # truthiness of this list today, but it carries the normalized command and the repeated
        # error so a transcript reviewer (or a future metric) does not have to re-derive them from
        # `denials`/`executions`.
        self.loop_detections: list[dict] = []
        # E3 signal ("2 actions, a 120s read-back window each"). One entry per REAL (non-dry-run)
        # C2 execution whose mutation command itself succeeded (a command that failed to even
        # apply never reaches read-back, see `_execute_c2_with_rollback`): {"command", "improved": bool | None, "rollback_executed"}.
        # `improved` is None when read-back could not determine readiness at all (delete pod's
        # honest limit, class docstring) -- `harness/triage_assembly.py`'s `_e3_triggered` treats
        # that the same as False (no CONFIRMED improvement), never as a pass.
        self.rollback_events: list[dict] = []

    def task_description(self) -> str:
        return self._context.as_prompt_text(self._context_max_chars)

    def covered_context_chars(self) -> int:
        """The context share as rendered, for harness.prompt_accounting."""
        return self._context.covered_chars(self._context_max_chars)

    def apply_action(self, action: dict) -> tuple[bool, str]:
        action_type = action["action_type"]

        if action_type in ("escalate", "submit"):
            logger.info(
                "episode %s: model ended the episode via %s: %s",
                self._context.episode_id,
                action_type,
                action.get("diagnosis", ""),
            )
            # Raised in review, investigated and NOT changed: `escalate` and `submit` both
            # return True here, so EpisodeResult.solved cannot distinguish them (an on-call
            # skimming "solved=True" would read an escalation as "handled"). Returning False for
            # `escalate` instead would be wrong in a different way: harness/agent.py's
            # `run_episode` loop only stops on `if solved: break`, so a False here would NOT end
            # the episode -- the agent would keep getting prompted for more turns after it already
            # asked to hand off to a human. Fixing this for real needs either a third return state
            # from apply_action or a schema change to EpisodeResult (harness/agent.py, shared with
            # aiopslab_env.py), which belongs with TriageRecord assembly (the actual consumer
            # that needs the distinction), not this module's scope.
            return True, f"episode ended via {action_type}"

        command = action.get("command") or ""
        target = action.get("target") or None
        try:
            self._state.confidence = float(action.get("confidence", 0.0))
        except (TypeError, ValueError):
            self._state.confidence = 0.0

        normalized_command = normalize_command_for_loop_detection(command)
        loop_observation = self._check_loop(normalized_command)
        if loop_observation is not None:
            return False, loop_observation

        decision = classify(command, self._state, target=target, scale_replicas=_extract_scale_replicas(command))

        if not decision.allowed:
            self.denials.append(decision)
            self._record_attempt(normalized_command, _failure_signature_for_denial(decision))
            logger.warning(
                "episode %s: action DENIED (%s): %s -- command=%r",
                self._context.episode_id,
                decision.action_class.value,
                decision.reason,
                command,
            )
            return False, f"DENIED ({decision.action_class.value}): {decision.reason}"

        self._register_allowed(decision)

        if decision.action_class == ActionClass.C2 and self._c2_dry_run:
            # The temporary dry-run gate (module docstring): classified, counted against the C2
            # cap, logged -- never handed to command_runner. The observation told back to the
            # model says explicitly that nothing ran, so it does not treat this as a completed
            # fix and, e.g., `submit` on the strength of a mutation that never happened.
            #
            # Deliberately NOT fed into E4's failure history: a dry-run is not a failure (the loop
            # pattern is retrying something that FAILED), it is a gate that will say the exact
            # same "DRY-RUN" thing every time by design. Counting repeats of an intentional gate
            # as a loop would put E4 in tension with the rollback rule this gate exists to
            # enforce, which expects a model that keeps proposing the same reversible fix to keep
            # hitting dry-run until the gate lifts.
            self.executions.append(
                {
                    "command": command,
                    "action_class": decision.action_class.value,
                    "returncode": None,
                    "dry_run": True,
                }
            )
            logger.warning(
                "episode %s: C2 action DRY-RUN (SSLM_C2_DRY_RUN still on -- see episode_runner.py "
                "module docstring): would execute command=%r",
                self._context.episode_id,
                command,
            )
            return False, (
                f"DRY-RUN (C2 gate active): classified {decision.action_class.value} and allowed "
                f"by the envelope, but NOT executed -- C2 mutations are dry-run only while "
                f"SSLM_C2_DRY_RUN is on (an operational decision, not a technical gap: "
                f"rollback_state/read-back are built, see build plan item 2b-iii). Command that "
                f"would have run: {command!r}. Nothing changed in the cluster; escalate or continue "
                f"read-only diagnosis if you need to confirm this."
            )

        if decision.action_class == ActionClass.C2:
            return self._execute_c2_with_rollback(command, target, normalized_command)

        result = self._command_runner(command, timeout=self._command_timeout)
        self._maybe_learn_target(decision, target, result, command)
        self._record_attempt(normalized_command, _failure_signature_for_result(result))
        self.executions.append(
            {
                "command": command,
                "action_class": decision.action_class.value,
                "returncode": result.returncode,
                "dry_run": False,
            }
        )
        logger.info(
            "episode %s: executed %s command=%r exit=%d",
            self._context.episode_id,
            decision.action_class.value,
            command,
            result.returncode,
        )
        return False, cap_observation(_format_command_result(result))

    # -----------------------------------------------------------------------
    # rollback_state pre-computation + read-back + rollback-on-no-improvement for a REAL
    # (non-dry-run) C2 mutation: the hard rollback rule.
    # -----------------------------------------------------------------------

    def _execute_c2_with_rollback(
        self, command: str, target: str | None, normalized_command: str
    ) -> tuple[bool, str]:
        """`classify()` already allowed this C2 and `_register_allowed` already counted it against
        the episode's C2 cap (caller, `apply_action`) -- this method owns everything the rollback
        rule adds on top: pre-compute `rollback_state` from a real read BEFORE the mutation runs,
        refuse to run it at all if that read failed (a C2 without a rollback_state does not run),
        then read back the mutated resource's readiness within the 120s window and
        execute the registered rollback if it shows no improvement.
        """
        rollback_state, rollout_target, precheck_failure = self._compute_rollback_state(command, target)

        if precheck_failure is not None:
            # Hard rule: a C2 whose rollback_state could not be grounded in a real read must not
            # execute at all. Treated as a gate denial (feeds E2, same class as an RBAC/gate
            # denial), not a silent skip.
            denial = EnvelopeDecision(
                action_class=ActionClass.C2,
                allowed=False,
                reason=f"rollback_state precheck failed: {precheck_failure}",
            )
            self.denials.append(denial)
            self._record_attempt(normalized_command, _failure_signature_for_denial(denial))
            logger.warning(
                "episode %s: C2 rollback_state precheck failed, command NOT executed (hard rule: "
                "a C2 without a rollback_state does not run): %s -- command=%r",
                self._context.episode_id,
                precheck_failure,
                command,
            )
            return False, (
                f"DENIED (C2): rollback_state precheck failed ({precheck_failure}). Section 3.1's "
                "hard rule: C2 sem rollback_state nao executa. Nothing changed in the cluster."
            )

        result = self._command_runner(command, timeout=self._command_timeout)
        # No `_maybe_learn_target` call here: that method only ever acts on a C0 decision (it
        # feeds read-before-mutate's targets_seen), so it is a guaranteed no-op for a C2 execution
        # -- calling it would be dead code, not a missed learning opportunity.
        self._record_attempt(normalized_command, _failure_signature_for_result(result))

        if result.returncode != 0:
            # The mutation itself never took effect: nothing to read back, nothing to roll back.
            # rollback_state still travels with this execution (it WAS computed, honestly, before
            # the attempt) so a transcript reviewer can see what would have backed a retry.
            self.executions.append(
                {
                    "command": command,
                    "action_class": ActionClass.C2.value,
                    "returncode": result.returncode,
                    "dry_run": False,
                    "rollback_state": rollback_state,
                    "read_back": None,
                }
            )
            logger.warning(
                "episode %s: C2 command itself failed (exit %d), read-back/rollback skipped: "
                "command=%r",
                self._context.episode_id,
                result.returncode,
                command,
            )
            return False, cap_observation(_format_command_result(result))

        read_back_ready, read_back_detail = self._perform_read_back(rollout_target)

        rollback_executed = False
        rollback_detail = ""
        comando_rollback = rollback_state.get("comando_rollback") if rollback_state else None
        if read_back_ready is not True and comando_rollback:
            rollback_result = self._command_runner(comando_rollback, timeout=self._command_timeout)
            rollback_executed = True
            rollback_detail = f"\nrollback executed ({comando_rollback!r}): exit={rollback_result.returncode}"
            logger.warning(
                "episode %s: C2 read-back showed no improvement, executing registered rollback: "
                "%r (exit=%d)",
                self._context.episode_id,
                comando_rollback,
                rollback_result.returncode,
            )

        self.rollback_events.append(
            {"command": command, "improved": read_back_ready, "rollback_executed": rollback_executed}
        )
        self.executions.append(
            {
                "command": command,
                "action_class": ActionClass.C2.value,
                "returncode": result.returncode,
                "dry_run": False,
                "rollback_state": rollback_state,
                "read_back": {"ready": read_back_ready, "detail": read_back_detail},
                "rollback_executed": rollback_executed,
            }
        )
        logger.info(
            "episode %s: executed C2 command=%r exit=%d read_back_ready=%s rollback_executed=%s",
            self._context.episode_id,
            command,
            result.returncode,
            read_back_ready,
            rollback_executed,
        )
        observation = cap_observation(
            _format_command_result(result)
            + f"\nread-back ({self._read_back_window_seconds:.0f}s window): {read_back_detail}"
            + rollback_detail
        )
        return False, observation

    def _compute_rollback_state(
        self, command: str, target: str | None
    ) -> tuple[dict | None, str | None, str | None]:
        """Pre-compute `rollback_state` for a real C2 mutation, BEFORE it executes (hard rollback
        rule). Every read here is itself a read-only command run through
        `self._command_runner` -- never fabricated, never assumed.

        Returns `(rollback_state, rollout_target, failure_reason)`. `rollout_target` is the
        `<kind>/<name>` `_perform_read_back` polls afterward -- for restart/undo/scale it is
        `target` itself (already the workload resource `kubectl rollout status` understands); for
        `delete pod` it is the pod's owning Deployment, resolved via ownerReferences, since
        `rollout status` does not understand a bare pod (class docstring's honest limit on this).

        `failure_reason` not None means the read that would ground `rollback_state` failed -- the
        caller (`_execute_c2_with_rollback`) is what actually blocks execution on this; this
        method only reports the failure, it never blocks anything itself.
        """
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            return None, None, f"could not parse command to determine its C2 type: {exc}"

        verb = tokens[1] if len(tokens) > 1 else ""

        if verb == "rollout" and len(tokens) > 2 and tokens[2] == "restart":
            return (
                {
                    "tipo": "rollout_restart",
                    "estado_pre_mutacao": "n/a: restart does not change the deployment spec",
                    "comando_rollback": None,
                    "motivo": (
                        "no rollback action needed: restart is idempotent, it only recycles the "
                        "pods without changing the spec, so there is no prior state to restore"
                    ),
                },
                target,
                None,
            )

        if verb == "rollout" and len(tokens) > 2 and tokens[2] == "undo":
            return self._compute_rollback_state_for_undo(target)

        if verb == "delete":
            return self._compute_rollback_state_for_delete_pod(tokens)

        if verb == "scale":
            return self._compute_rollback_state_for_scale(target)

        # Unreachable in practice: classify() only ever allows these four C2 shapes
        # to get here -- never fabricate a rollback_state for a shape this method does not
        # recognize, report it as a failure instead.
        return None, None, f"unrecognized C2 command shape for rollback_state precompute: {command!r}"

    def _compute_rollback_state_for_undo(self, target: str | None) -> tuple[dict | None, str | None, str | None]:
        if not target:
            return None, None, "no target named for rollout undo, cannot read the current revision"
        command = f"kubectl rollout history {target} -n {self._context.namespace}"
        result = self._command_runner(command, timeout=self._command_timeout)
        if result.returncode != 0:
            return None, None, f"{command!r} exited {result.returncode}: {(result.stderr or result.stdout).strip()[:200]}"
        revision = _parse_current_revision_from_rollout_history(result.stdout)
        if revision is None:
            return None, None, f"{command!r} succeeded but no revision number could be parsed from its output"
        return (
            {
                "tipo": "rollout_undo",
                # Pinned by tests, kept as published: "current revision before the undo: N".
                "estado_pre_mutacao": f"revisao vigente antes do undo: {revision}",
                "comando_rollback": f"kubectl rollout undo {target} -n {self._context.namespace} --to-revision={revision}",
                "motivo": (
                    f"rollback is re-applying revision {revision}, which was current before this "
                    f"action (read via kubectl rollout history before the undo)"
                ),
            },
            target,
            None,
        )

    def _compute_rollback_state_for_delete_pod(
        self, tokens: list[str]
    ) -> tuple[dict | None, str | None, str | None]:
        try:
            pod_index = tokens.index("pod")
        except ValueError:
            return None, None, "delete command does not name 'pod' positionally, cannot resolve pod name"
        if pod_index + 1 >= len(tokens):
            return None, None, "delete pod command missing the pod name"
        pod_name = tokens[pod_index + 1]

        pods_json, reason = _kubectl_get_json(
            self._command_runner, f"pod {pod_name}", self._context.namespace, timeout=self._command_timeout
        )
        if pods_json is None:
            return None, None, reason

        owner = _controller_owner_reference(pods_json)
        if owner is None:
            return (
                None,
                None,
                f"pod {pod_name!r} has no controller ownerReference: cannot determine which "
                "resource recreates it, nothing to annotate as rollback_state's owning resource",
            )
        owner_kind, owner_name = owner

        rollback_state = {
            "tipo": "delete_pod",
            "estado_pre_mutacao": f"owning resource (via ownerReferences): {owner_kind}/{owner_name}",
            "comando_rollback": None,
            "motivo": (
                "delete pod has no executable rollback: the controller already recreates the pod. "
                "rollback_state here is informational (which resource owned it), not a rollback "
                "command (the rule asks for the ReplicaSet name to be annotated, not a command)"
            ),
        }

        rollout_target = None
        if owner_kind.lower() == "replicaset":
            deployment_name = self._deployment_owning_replicaset(owner_name)
            if deployment_name is not None:
                rollout_target = f"deployment/{deployment_name}"
        return rollback_state, rollout_target, None

    def _deployment_owning_replicaset(self, replicaset_name: str) -> str | None:
        """Second read, only for read-back's benefit (class docstring's delete-pod honest limit):
        resolves the Deployment that owns `replicaset_name`, so `_perform_read_back` has a
        workload resource `kubectl rollout status` can actually check. A failure here does NOT
        block the C2 (rollback_state is already complete per the rollback rule, which only
        requires the ReplicaSet name) -- it only degrades read-back to inconclusive."""
        json_result, reason = _kubectl_get_json(
            self._command_runner,
            f"replicaset {replicaset_name}",
            self._context.namespace,
            timeout=self._command_timeout,
        )
        if json_result is None:
            logger.warning(
                "episode %s: could not resolve the Deployment owning ReplicaSet %r for read-back "
                "(%s): read-back will be inconclusive for this delete-pod action",
                self._context.episode_id,
                replicaset_name,
                reason,
            )
            return None
        owner = _controller_owner_reference(json_result)
        if owner is None or owner[0].lower() != "deployment":
            return None
        return owner[1]

    def _compute_rollback_state_for_scale(self, target: str | None) -> tuple[dict | None, str | None, str | None]:
        if not target:
            return None, None, "no target named for scale, cannot read current replicas"
        json_result, reason = _kubectl_get_json(
            self._command_runner, target, self._context.namespace, timeout=self._command_timeout
        )
        if json_result is None:
            return None, None, reason
        replicas = (json_result.get("spec") or {}).get("replicas")
        if not isinstance(replicas, int):
            return None, None, f"spec.replicas for {target!r} was not an int: {replicas!r}"
        return (
            {
                "tipo": "scale",
                # Pinned by tests, kept as published: "replicas before the mutation: N".
                "estado_pre_mutacao": f"replicas antes da mutacao: {replicas}",
                "comando_rollback": f"kubectl scale {target} -n {self._context.namespace} --replicas={replicas}",
                "motivo": f"rollback is scaling back to the replica count observed before this action ({replicas})",
            },
            target,
            None,
        )

    def _perform_read_back(self, rollout_target: str | None) -> tuple[bool | None, str]:
        """Rollback rule: "read back within 120s after the action". Polls `kubectl rollout status
        <rollout_target>` (class docstring: the chosen, documented interpretation of "re-measure
        the symptom") every `READ_BACK_POLL_INTERVAL_SECONDS` until it reports the rollout complete
        (exit 0 -> improved) or the window closes (not confirmed -> not improved). `rollout_target`
        is `None` when `_compute_rollback_state` could not resolve one (delete pod's honest limit)
        -- read-back is then honestly inconclusive (`None`), never a fabricated verdict either way.
        """
        if not rollout_target:
            return None, "no rollout target could be resolved for read-back (see rollback_state precompute)"

        deadline = self._clock() + self._read_back_window_seconds
        last_detail = "read-back not attempted"
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                return False, (
                    f"read-back window ({self._read_back_window_seconds:.0f}s) exhausted without "
                    f"confirming readiness: {last_detail}"
                )
            attempt_timeout = min(remaining, self._command_timeout)
            command = (
                f"kubectl rollout status {rollout_target} -n {self._context.namespace} "
                f"--timeout={attempt_timeout:.0f}s"
            )
            result = self._command_runner(command, timeout=attempt_timeout)
            if result.returncode == 0:
                return True, f"{command!r}: rollout complete, symptom target ready"
            last_detail = f"{command!r} exited {result.returncode}: {(result.stderr or result.stdout).strip()[:200]}"
            remaining = deadline - self._clock()
            if remaining <= 0:
                return False, (
                    f"read-back window ({self._read_back_window_seconds:.0f}s) exhausted without "
                    f"confirming readiness: {last_detail}"
                )
            self._sleep_fn(min(self._read_back_poll_interval_seconds, remaining))

    def _check_loop(self, normalized_command: str) -> str | None:
        """E4: if `normalized_command` already failed with the SAME error the last
        TWO times it was attempted, this (3rd) emission is blocked -- returns the observation to
        feed back to the model instead of letting the caller proceed to `classify()`. Returns
        None when this emission may proceed normally (fewer than 2 prior attempts, or the last
        two did not match).

        Runs before `classify()` and before `_register_allowed`, so a blocked emission never
        counts against any C0/C1/C2 cap and never appears in `denials`/`executions` -- it is a
        third, separate outcome (`loop_detections`), not a denial classify() itself produced.
        """
        history = self._command_failure_history.get(normalized_command)
        if history and len(history) >= 2 and history[-1] == history[-2]:
            repeated_error = history[-1]
            self.loop_detections.append(
                {
                    "normalized_command": normalized_command,
                    "error_signature": repeated_error,
                    "prior_attempts": len(history),
                }
            )
            logger.warning(
                "episode %s: E4 loop detected -- %r failed identically twice (%r); blocking this "
                "(3rd+) emission instead of re-running it",
                self._context.episode_id,
                normalized_command,
                repeated_error,
            )
            return (
                f"BLOCKED (E4 loop detected): command {normalized_command!r} already failed "
                f"twice in a row with the same error ({repeated_error!r}). Not re-running it a "
                "third time -- propose a different command, or escalate."
            )
        return None

    def _record_attempt(self, normalized_command: str, failure_signature: str | None) -> None:
        """Appends `failure_signature` to this normalized command's history for `_check_loop`'s
        next call -- unless it is None (`_failure_signature_for_denial` always returns a string,
        `_failure_signature_for_result` returns None for a successful execution), in which case
        nothing is recorded: a success must not sit in the history where a later, unrelated
        failure could spuriously "match" it (there is nothing to match; None never equals None
        in this history because it is never stored)."""
        if failure_signature is None:
            return
        self._command_failure_history.setdefault(normalized_command, []).append(failure_signature)

    def _register_allowed(self, decision: EnvelopeDecision) -> None:
        # classify() itself never mutates state (its own docstring), by design: a denied call
        # never counts against a cap or a target it did not actually establish. Executing here,
        # right after `decision.allowed` is confirmed True, is what keeps that invariant intact.
        if decision.action_class == ActionClass.C0:
            self._state.c0_calls += 1
        elif decision.action_class == ActionClass.C1:
            self._state.c1_calls += 1
        elif decision.action_class == ActionClass.C2:
            self._state.c2_calls += 1

    def _maybe_learn_target(
        self, decision: EnvelopeDecision, target: str | None, result: CommandResult, command: str
    ) -> None:
        # Read-before-mutate proxy, not literal output-content parsing (a scoped simplification):
        # a C0 command that exits 0 is treated as having confirmed `target` exists.
        # kubectl get/describe against a resource that does not exist exits non-zero (NotFound),
        # so requiring returncode==0 is what keeps this from widening the gate for a target the
        # command never actually reached -- it just does not parse the JSON/text output to
        # confirm `target` is literally named in it the way the design's wording implies.
        #
        # The model's self-reported `target` field is not tied to
        # `command` by AGENT_ACTION_SCHEMA, so without _command_references_target below, a model
        # could pair a cheap unrelated C0 read (e.g. `kubectl get pods`) with an arbitrary
        # `target` string and get it marked "seen": exactly the bypass (mutate from a prior, not
        # from an observation) the read-before-mutate rule exists to block.
        if (
            decision.action_class == ActionClass.C0
            and target
            and result.returncode == 0
            and _command_references_target(command, target)
        ):
            self._state.targets_seen.add(target)


# ---------------------------------------------------------------------------
# Entry points: what entrypoint.py's episode_runner seam actually calls.
# ---------------------------------------------------------------------------


def _run_episode_with_environment(
    episode_id: str,
    webhook_raw: dict,
    *,
    model: ModelClient,
    namespace: str,
    prometheus_url: str,
    command_runner: CommandRunner = subprocess_command_runner,
    agent_config: AgentConfig | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_elapsed_seconds: float = DEFAULT_MAX_ELAPSED_SECONDS,
    c2_dry_run: bool = True,
) -> tuple[EpisodeResult, EpisodeContext, LiveClusterEnvironment]:
    """Collect -> build the Environment -> run the (reused, not reimplemented) `harness.agent.
    Agent` loop against it. One call per triage episode. Returns the `EpisodeContext` and
    `LiveClusterEnvironment` alongside the `EpisodeResult` -- `run_real_episode` below throws
    those two away to keep its own original return shape; `make_episode_runner` calls this
    function directly because TriageRecord assembly (harness/triage_assembly.py) needs both.

    Default `AgentConfig` has `schema_constrained_decoding=True`: this is the highest-stakes
    path of the three (config A/B/C benchmark runs vs. a live cluster), and `harness/agent.py`'s own docstring documents constrained decoding as what cut
    malformed-output retries in the benchmark runs -- there is no reason to run the live path
    with the weaker of the two decoding modes by default. `VllmModelClient` (the model client
    `entrypoint.py` wires in) already supports it. Still overridable via `agent_config` for a
    `ModelClient` that cannot honor it.

    `EpisodeEnvelopeState.targets_seen` starts EMPTY: an earlier
    version seeded it from `collect_episode_context`'s returned target set, which lists every
    pod/deployment `kubectl get -o json` returned for the whole namespace, not just resources
    related to this alert. That satisfied read-before-mutate for anything in the namespace from
    turn 1, defeating the read-before-mutate rule (the failure mode it guards against: "mutates
    without reading... acts from a prior, not from observation"). The model must now issue its
    own C0 get/describe/logs against a target during THIS episode
    (`LiveClusterEnvironment._maybe_learn_target`, which also checks the C0 command actually names
    that target) before a C2 mutate against it is
    allowed -- collection's own `targets_seen` return value is still meaningful as "confirmed to
    exist in the namespace" for other callers/tests, just not as a seed for the mutate gate.

    `c2_dry_run` (default True) is threaded straight into
    `LiveClusterEnvironment` -- see that class's docstring and `c2_dry_run_enabled()`.

    `max_elapsed_seconds` (default `harness.agent.
    DEFAULT_MAX_ELAPSED_SECONDS` = 900s) is threaded straight into `Agent.run_episode` -- see
    that module's docstring for why the elapsed-time half of E5's cap lives there rather than
    here."""
    context, _ = collect_episode_context(
        episode_id,
        webhook_raw,
        namespace=namespace,
        prometheus_url=prometheus_url,
        command_runner=command_runner,
    )
    state = EpisodeEnvelopeState()
    agent_config = agent_config or AgentConfig(
        schema_constrained_decoding=True,
        memory_enabled=False,
        history_budget=history_budget_enabled(),
    )
    budget = agent_config.prompt_budget
    environment = LiveClusterEnvironment(
        context=context,
        state=state,
        command_runner=command_runner,
        c2_dry_run=c2_dry_run,
        context_max_chars=budget.context_chars if budget is not None else None,
    )
    agent = Agent(model, agent_config)
    result = agent.run_episode(
        environment, task_id=episode_id, max_turns=max_turns, max_elapsed_seconds=max_elapsed_seconds
    )
    return result, context, environment


def run_real_episode(
    episode_id: str,
    webhook_raw: dict,
    *,
    model: ModelClient,
    namespace: str,
    prometheus_url: str,
    command_runner: CommandRunner = subprocess_command_runner,
    agent_config: AgentConfig | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_elapsed_seconds: float = DEFAULT_MAX_ELAPSED_SECONDS,
    c2_dry_run: bool = True,
) -> EpisodeResult:
    """Same as `_run_episode_with_environment`, returning only the `EpisodeResult` -- the public
    shape this function had before TriageRecord assembly existed, kept unchanged for callers (and
    their tests) that only need the episode's outcome, not TriageRecord assembly."""
    result, _context, _environment = _run_episode_with_environment(
        episode_id,
        webhook_raw,
        model=model,
        namespace=namespace,
        prometheus_url=prometheus_url,
        command_runner=command_runner,
        agent_config=agent_config,
        max_turns=max_turns,
        max_elapsed_seconds=max_elapsed_seconds,
        c2_dry_run=c2_dry_run,
    )
    return result


def _default_triage_sink(record: dict) -> None:
    """Default sink: log the full record as JSON, at INFO. This is what makes "always emit a
    TriageRecord" true without any storage infrastructure; a real sink (a file, as
    `JsonlTriageSink` below, or anything else) is a `triage_sink` override, not a change to this
    function."""
    logger.info("triage_record %s", json.dumps(record, sort_keys=True))


# The path entrypoint.py hands to JsonlTriageSink. Unset keeps the log-only
# default above, so any caller outside the Deployment behaves exactly as before.
TRIAGE_LOG_ENV = "SSLM_TRIAGE_LOG"


class JsonlTriageSink:
    """Appends each TriageRecord as one JSON line to `path`, fsynced, so the scorer bridge can
    read a round's records back after the fact instead of scraping pod stdout.

    The record is ALSO logged through `_default_triage_sink` first, every time: that line is the
    second copy the harvest can fall back on if the file is lost (the Deployment mounts an
    emptyDir, which survives a container restart but not a pod deletion). A failed append is
    logged and swallowed, never raised: `make_episode_runner` promises exactly one sink call per
    episode, and an OSError here must not turn into a missing record AND a crashed handler.

    One lock per instance: concurrent `/alerts` episodes share the single sink the entrypoint
    builds, and interleaved partial writes would corrupt the JSONL for the whole round."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = os.fspath(path)
        self._lock = threading.Lock()

    def __call__(self, record: dict) -> None:
        _default_triage_sink(record)
        line = json.dumps(record, sort_keys=True) + "\n"
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            logger.exception(
                "triage sink could not append to %s; the record survives only in the triage_record log line above",
                self.path,
            )


def usage_log_path(triage_log: str | os.PathLike) -> str:
    """Per-episode token usage lands next to the TriageRecords, never inside
    them (TRIAGE_RECORD_SCHEMA is closed and is the paper's artifact). Derived from
    SSLM_TRIAGE_LOG so the Deployment needs no new variable; the harvest derives it the same way."""
    return os.fspath(triage_log) + ".usage.jsonl"


class JsonlUsageSink:
    """One line per episode: `harness.metrics.EpisodeTokenUsage`. Same append discipline as
    JsonlTriageSink (lock, fsync, a failed append logged and swallowed), without the log-line
    copy: a lost usage line leaves the episode unattributed in the cost slice, not missing."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = os.fspath(path)
        self._lock = threading.Lock()

    def __call__(self, usage: dict) -> None:
        line = json.dumps(usage, sort_keys=True) + "\n"
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            logger.exception("usage sink could not append to %s; episode %s stays unattributed",
                             self.path, usage.get("episode_id"))


def make_episode_runner(
    *,
    model: ModelClient,
    namespace: str,
    prometheus_url: str,
    command_runner: CommandRunner = subprocess_command_runner,
    agent_config: AgentConfig | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_elapsed_seconds: float = DEFAULT_MAX_ELAPSED_SECONDS,
    c2_dry_run: bool | None = None,
    triage_sink=None,
    usage_sink=None,
):
    """Builds an `entrypoint.EpisodeRunner` (`episode_id, webhook_raw -> None`) closing over the
    dependencies above -- the exact shape `entrypoint.WebhookReceiver` already calls. Exceptions
    from the orchestrator are caught and logged here, never left to propagate into the HTTP
    handler thread: an episode raising must not take the webhook receiver down.

    `c2_dry_run` defaults to `c2_dry_run_enabled()` (the `SSLM_C2_DRY_RUN` env var, read once
    here at wiring time -- not per episode) when left as `None`; pass an explicit bool in tests.
    `triage_sink` defaults to `_default_triage_sink` (logged as JSON); pass a callable to send
    records somewhere else.

    **Every call to the returned runner ends in exactly one `triage_sink(record)` call, success
    or exception.** "Always emit a TriageRecord, including when escalating" is a promise about
    every episode, including one the orchestrator itself could not finish:
    `harness.triage_assembly.build_fallback_triage_record` is what the `except` branch below
    emits instead of silently losing the episode.

    **Resolved (was an honest limit):** `entrypoint.py`'s
    `WebhookReceiver` used to call `self.episode_runner(...)` SYNCHRONOUSLY, before writing the
    `202` response -- harmless with `log_only_episode_runner`, but with `run_real_episode` wired
    in, a single `POST /alerts` could block for minutes (up to the 90s scripted-collection budget
    plus E5's own elapsed-time cap), tying up a `ThreadingHTTPServer` worker thread and risking an
    Alertmanager-side timeout/retry landing after the 120s group window had already closed and
    opening a second episode for the same fault. `WebhookReceiver` now dispatches
    `episode_runner` to a background thread (`entrypoint._default_dispatch`, see its docstring)
    and returns `202` immediately (fixed in `entrypoint.py`, not this module)."""
    resolved_c2_dry_run = c2_dry_run_enabled() if c2_dry_run is None else c2_dry_run
    sink = triage_sink or _default_triage_sink

    # Imported here, not at module level: harness/triage_assembly.py imports harness/triage.py
    # and harness/envelope.py, neither of which import this module, so there is no real cycle --
    # kept local anyway so a future edit to that module can freely import episode_runner types
    # (EpisodeContext, LiveClusterEnvironment) for typing without creating one by accident.
    from .triage_assembly import (
        _alert_fingerprint,
        assemble_triage_record,
        build_fallback_triage_record,
        extract_t_alarme,
        now_iso,
    )

    take_stats = getattr(model, "take_thread_stats", None)

    def _runner(episode_id: str, webhook_raw: dict) -> None:
        if usage_sink is None or take_stats is None:
            _triage(episode_id, webhook_raw)
            return
        take_stats()  # drop anything this thread counted before the episode
        t_start = time.time()
        try:
            _triage(episode_id, webhook_raw)
        finally:
            stats = take_stats()
            usage_sink({
                "episode_id": episode_id,
                "t_start": t_start,
                "t_end": time.time(),
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "calls": stats.calls,
                "calls_without_usage": stats.calls_without_usage,
            })

    def _triage(episode_id: str, webhook_raw: dict) -> None:
        received_at = now_iso()
        t_alarme = extract_t_alarme(webhook_raw, fallback=received_at)
        try:
            result, context, environment = _run_episode_with_environment(
                episode_id,
                webhook_raw,
                model=model,
                namespace=namespace,
                prometheus_url=prometheus_url,
                command_runner=command_runner,
                agent_config=agent_config,
                max_turns=max_turns,
                max_elapsed_seconds=max_elapsed_seconds,
                c2_dry_run=resolved_c2_dry_run,
            )
        except Exception as exc:  # noqa: BLE001 - logged, not propagated (see docstring above)
            logger.exception("episode %s: orchestrator raised, episode aborted without a decision", episode_id)
            record = build_fallback_triage_record(
                episode_id=episode_id,
                alert_fingerprint=_alert_fingerprint(webhook_raw),
                t_alarme=t_alarme,
                t_record=now_iso(),
                reason=f"orchestrator raised before the episode could finish: {exc!r}",
            )
            sink(record)
            return

        logger.info(
            "episode %s finished: solved=%s turns=%d elapsed=%.1fs",
            episode_id,
            result.solved,
            result.attempt_count,
            result.elapsed_seconds,
        )
        try:
            record = assemble_triage_record(
                episode_id=episode_id,
                webhook_raw=webhook_raw,
                context=context,
                environment=environment,
                result=result,
                t_alarme=t_alarme,
            )
        except Exception as exc:  # noqa: BLE001 - logged, not propagated (see docstring above)
            logger.exception("episode %s: triage record assembly raised after the episode finished", episode_id)
            record = build_fallback_triage_record(
                episode_id=episode_id,
                alert_fingerprint=_alert_fingerprint(webhook_raw),
                t_alarme=t_alarme,
                t_record=now_iso(),
                reason=f"episode finished but triage assembly raised: {exc!r}",
            )
        sink(record)

    return _runner
