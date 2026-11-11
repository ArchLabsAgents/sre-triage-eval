"""Container entrypoint: the Alertmanager -> agent webhook receiver, stdlib-only on purpose (same
reasoning as adapters/vllm_model_client.py: this image runs in the cluster's dedicated GPU node,
not the Mac dev environment, so it only depends on what tools/triage-eval/requirements-agent.txt
actually needs -- mlx-lm and outlines, both Mac/MLX-only per conftest.py, are never installed
here).

Routes:
  GET  /healthz  -> liveness/readiness probe target for the Deployment.
                     Also doubles as the one self-health signal allowed outside the
                     push-only contract (a real heartbeat poller to Prometheus/apiserver, on a
                     timer, is separate future work -- this endpoint only proves the process is
                     up and answering HTTP).
  GET  /metrics  -> plain counters (received/parsed/rejected/episodes_started), so the
                     webhooks-sent-vs-received check has an agent-side number to diff against
                     Alertmanager's own sent count.
  POST /alerts   -> Alertmanager's configured webhook URL. Validates the payload
                     (harness.webhook.parse_webhook_payload), dedups it into an episode
                     (harness.webhook.EpisodeDeduper), and -- only for a genuinely new episode --
                     hands `episode_runner` to `dispatch` (default: a background daemon thread,
                     see `_default_dispatch`) and returns `202` without waiting for it to finish.
                     `episode_runner` used to run SYNCHRONOUSLY here, before the `202` was
                     written -- see `_default_dispatch`'s own docstring for why that risked a
                     duplicate episode once the real orchestrator was wired in.

Scope boundary: `episode_runner` is a seam. `log_only_episode_runner` (below) is a placeholder
that only logs; `harness.episode_runner.make_episode_runner` is the real orchestrator -- collect -> decide -> classify -> execute -> assemble a
TriageRecord (see that module's docstring for what it does: it emits a TriageRecord for every
episode, including on escalation or an orchestrator crash, and gates every C2 mutation to dry-run
by default). `main()` wires the real runner in when VLLM_BASE_URL/APP_NAMESPACE/PROMETHEUS_URL are all
set in the environment, and falls back to `log_only_episode_runner` (with a logged warning
explaining why) otherwise -- so an under-configured deployment degrades to "accept, validate,
dedup, log" instead of crashing at startup.

Comparator arm (hosted API model): `COMPARATOR_ARM=1` (the same switch setup_cluster.sh reads, passed into
the pod by k8s/deployment.yaml) makes `main()` wire the Anthropic client
(cloud_adapter.make_anthropic_model_client) instead of the vLLM one, into the same
`make_episode_runner` with the same sinks. Unlike the vLLM path this one fails closed: any missing
or wrong piece (key, spend cap, usage log, history budget on, log-only mode) stops the process at
startup, see `_build_comparator_episode_runner`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Protocol

from harness.webhook import EpisodeDeduper, WebhookPayloadError, parse_webhook_payload

logger = logging.getLogger("sslm.entrypoint")

DEFAULT_PORT = 8080
_MAX_BODY_BYTES = 1_000_000  # an Alertmanager group webhook is a handful of KB; refuse anything
# wildly larger up front rather than buffering an unbounded body into memory.


class EpisodeRunner(Protocol):
    def __call__(self, episode_id: str, webhook_raw: dict) -> None: ...


class EpisodeDispatch(Protocol):
    def __call__(self, task: Callable[[], None]) -> None: ...


def _default_dispatch(task: Callable[[], None]) -> None:
    """The default `EpisodeDispatch`: runs `task` (a zero-arg closure over one
    `episode_runner(episode_id, webhook_raw)` call) on its own daemon thread and returns
    immediately, without waiting for it to finish.

    `WebhookReceiver.handle_alert_payload` used to call `self.episode_runner(...)` inline, and
    `entrypoint.py`'s do_POST wrote the `202` response only after that call returned. With
    `log_only_episode_runner` that was instant and harmless; with the real orchestrator
    (`harness.episode_runner.make_episode_runner`) wired in, a single episode can now run for
    minutes -- up to `harness.episode_runner.SCRIPTED_COLLECTION_BUDGET_SECONDS` (90s) plus the
    agent's own elapsed-time cap (`harness.agent.DEFAULT_MAX_ELAPSED_SECONDS`, 900s: an episode
    ends after 8 turns or 15 minutes without a final decision), roughly 990s worst case. Blocking the HTTP response for that long risked
    an Alertmanager-side webhook timeout/retry landing after `harness.webhook.
    GROUP_WINDOW_SECONDS` (120s) had already closed, which opens a SECOND episode for the same
    fault -- see `harness/webhook.py`'s `EpisodeDeduper` docstring: a window is 120s from the
    FIRST webhook, never extended by how long the prior episode's processing actually takes, so a
    second, legitimate webhook landing after that window closes is indistinguishable from a
    genuinely new fault.

    A longer `webhook_configs[].timeout` (kube-prometheus-stack/values.yaml) was evaluated and
    rejected as the fix, for two reasons found while researching it, not assumed: (1) Alertmanager
    documents that a webhook `timeout` "will have no effect if set higher than the
    group_interval", and `group_interval` is deliberately pinned to `GROUP_WINDOW_SECONDS` (120s,
    see that key's own comment in values.yaml) for a still-wanted, unrelated reason -- an
    in-progress episode's update lands close to its own window's close. Raising `group_interval`
    to cover the ~990s worst case would undo that alignment and still leave an unenforced margin
    between the configured ceiling and the real worst case: any slowdown (a slow kubectl call, a
    model latency spike) silently blows through it, since nothing measures or caps that margin at
    runtime. (2) it does nothing for the concurrency cost of pinning a `ThreadingHTTPServer`
    worker thread for up to 15 minutes per episode. Dispatching asynchronously removes the
    coupling to Alertmanager's timing entirely: the `202` is written, and the worker thread freed,
    as soon as the episode is handed off, never after it finishes.

    Exceptions from `task` are caught and logged here rather than left to crash silently on a
    background thread (Python would otherwise only report them to stderr via
    `threading.excepthook`, easy to miss in a container's logs). `make_episode_runner`'s own
    returned callable already catches its own exceptions (see that function's docstring: "never
    left to propagate into the HTTP handler thread"), so this is a second, cheap safety net for
    `log_only_episode_runner` or any other `EpisodeRunner` that does not.

    In-memory and per-process, same as `EpisodeDeduper`: not meant to survive a pod restart, and
    not bounded (unlike a fixed-size thread pool) -- the lab pilot's own scope (a handful of
    faults, never truly concurrent by design) never approaches a thread count where that matters;
    revisit if a later scenario needs backpressure.
    """

    def _run() -> None:
        try:
            task()
        except Exception:  # noqa: BLE001 - last-resort safety net, see docstring above
            logger.exception("episode dispatch raised on background thread")

    threading.Thread(target=_run, daemon=True).start()


def log_only_episode_runner(episode_id: str, webhook_raw: dict) -> None:
    """Placeholder `EpisodeRunner`: records that an episode WOULD start and why, does not
    triage. See this module's docstring -- replace this with the real orchestrator, do not
    extend it in place."""
    group_key = webhook_raw.get("groupKey", "<unknown>")
    logger.warning(
        "episode %s would start for group %r (orchestrator not wired yet, see entrypoint.py docstring)",
        episode_id,
        group_key,
    )


@dataclass
class _Counters:
    received_total: int = 0
    parse_errors_total: int = 0
    episodes_started_total: int = 0
    episodes_folded_total: int = 0

    def as_dict(self) -> dict:
        return {
            "received_total": self.received_total,
            "parse_errors_total": self.parse_errors_total,
            "episodes_started_total": self.episodes_started_total,
            "episodes_folded_total": self.episodes_folded_total,
        }


@dataclass
class WebhookReceiver:
    """The testable core: takes a decoded JSON body, returns (status_code, response_body).
    Kept separate from BaseHTTPRequestHandler so tests never need a real socket."""

    episode_runner: EpisodeRunner = field(default=log_only_episode_runner)
    deduper: EpisodeDeduper = field(default_factory=EpisodeDeduper)
    counters: _Counters = field(default_factory=_Counters)
    # See `_default_dispatch`'s docstring: this is what keeps `episode_runner` off the request
    # thread. Tests override it with a synchronous `lambda task: task()` where they need to
    # assert on the runner's effects deterministically, right after the call.
    dispatch: EpisodeDispatch = field(default=_default_dispatch)

    def handle_alert_payload(self, payload: object) -> tuple[int, dict]:
        self.counters.received_total += 1
        try:
            webhook = parse_webhook_payload(payload)
        except WebhookPayloadError as exc:
            self.counters.parse_errors_total += 1
            return 400, {"error": str(exc)}

        result = self.deduper.ingest(webhook)
        if not result.episode_id:
            # Resolved status with no open window: valid payload, nothing to triage.
            return 200, {"episode_id": None, "new_episode": False}

        if result.is_new_episode:
            self.counters.episodes_started_total += 1
            episode_id = result.episode_id
            raw = webhook.raw
            # Captured by value in the closure (episode_id/raw are this call's locals, not the
            # loop-variable-in-closure trap) -- dispatched, never run inline, so the 202 below is
            # written before the episode (potentially minutes long) finishes. See `dispatch`'s
            # field comment and `_default_dispatch`'s docstring.
            self.dispatch(lambda: self.episode_runner(episode_id, raw))
        else:
            self.counters.episodes_folded_total += 1

        return 202, {"episode_id": result.episode_id, "new_episode": result.is_new_episode}


_REQUEST_TIMEOUT_SECONDS = 30.0  # with no timeout, a slow or stalled client body pins a ThreadingHTTPServer worker thread indefinitely. BaseHTTPRequestHandler
# reads this class attribute, applies it as the connection's socket timeout, and already handles
# the resulting socket.timeout by logging and closing the connection (cpython's
# handle_one_request), so setting the attribute is the whole fix.


def _make_handler(receiver: WebhookReceiver) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "sslm-agent/1"
        timeout = _REQUEST_TIMEOUT_SECONDS

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib signature
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _write_json(self, status: int, body: dict) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 - stdlib method name
            if self.path == "/healthz":
                self._write_json(200, {"status": "ok"})
                return
            if self.path == "/metrics":
                self._write_json(200, receiver.counters.as_dict())
                return
            self._write_json(404, {"error": f"no GET route for {self.path!r}"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib method name
            if self.path != "/alerts":
                self._write_json(404, {"error": f"no POST route for {self.path!r}"})
                return

            length_header = self.headers.get("Content-Length")
            try:
                length = int(length_header) if length_header is not None else 0
            except ValueError:
                self._write_json(400, {"error": f"invalid Content-Length header {length_header!r}"})
                return
            if length <= 0:
                self._write_json(400, {"error": "empty request body"})
                return
            if length > _MAX_BODY_BYTES:
                self._write_json(400, {"error": f"body exceeds {_MAX_BODY_BYTES} bytes"})
                return

            raw_body = self.rfile.read(length)
            try:
                payload = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                # json.loads on bytes/bytearray decodes as UTF-8/16/32 internally (RFC 8259);
                # invalid encoding raises UnicodeDecodeError, not JSONDecodeError (review
                # finding on this PR: the earlier version only caught JSONDecodeError, so a
                # non-UTF-8 body degraded to an unhandled exception -- a dropped connection --
                # instead of the intended 400).
                self._write_json(400, {"error": f"invalid JSON body: {exc}"})
                return

            status, body = receiver.handle_alert_payload(payload)
            self._write_json(status, body)

    return Handler


def serve(*, port: int = DEFAULT_PORT, episode_runner: EpisodeRunner = log_only_episode_runner) -> None:
    receiver = WebhookReceiver(episode_runner=episode_runner)
    handler_cls = _make_handler(receiver)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler_cls)  # noqa: S104 - container-internal
    # listener, exposure is controlled by the K8s Service/NetworkPolicy.
    logger.info("sslm-agent webhook receiver listening on :%d", port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


COMPARATOR_ARM_ENV = "COMPARATOR_ARM"


class ComparatorWiringError(RuntimeError):
    """The pod was deployed as the comparator arm but cannot run as one. Raised at wiring time and
    turned into a non-zero exit by `main()`, never caught into a fallback."""


class PromptBudgetWiringError(RuntimeError):
    """SSLM_PROMPT_BUDGET_CHARS is set but unusable. Same exit path as ComparatorWiringError: a
    budget typo must keep the pod from going Ready, not degrade to log-only or to the unbounded
    prompt the budget exists to replace."""


class DecodingDialectWiringError(RuntimeError):
    """VLLM_DECODING_DIALECT is set but unknown. Same exit path as PromptBudgetWiringError: a
    newer vLLM ignores the old request fields and decodes unconstrained without an error, so a
    typo must keep the pod from going Ready rather than pick a shape."""


def _agent_config_from_env():
    """The vLLM arm's AgentConfig, read once at wiring time. Same defaults the episode runner
    builds on its own (constrained decoding, no memory, history budget from the env), plus
    the whole-call prompt budget when SSLM_PROMPT_BUDGET_CHARS carries one."""
    from harness.agent import AgentConfig, history_budget_enabled, prompt_budget_from_env, tool_contract_from_env

    try:
        budget = prompt_budget_from_env()
    except ValueError as exc:
        raise PromptBudgetWiringError(str(exc)) from exc
    try:
        tool_contract = tool_contract_from_env()
    except ValueError as exc:
        raise PromptBudgetWiringError(str(exc)) from exc
    if tool_contract:
        logger.info("tool contract on: command rules appended to the system prompt")
    if budget is not None:
        logger.info(
            "prompt budget on: %d chars (context %d, history %d, last observation %d)",
            budget.total_chars,
            budget.context_chars,
            budget.history_chars,
            budget.observation_chars,
        )
    return AgentConfig(
        schema_constrained_decoding=True,
        memory_enabled=False,
        history_budget=history_budget_enabled(),
        prompt_budget=budget,
        tool_contract=tool_contract,
    )


def _agent_config_from_env_comparator():
    """The comparator arm's AgentConfig: None (the episode runner's own default, history budget
    from the env, which cloud_adapter requires off) unless SSLM_PROMPT_BUDGET_CHARS carries a
    budget, then that budget on top of the same defaults. No tool contract, no memory."""
    from harness.agent import AgentConfig, history_budget_enabled, prompt_budget_from_env

    budget = prompt_budget_from_env()
    if budget is None:
        return None
    logger.info(
        "comparator prompt budget on: %d chars (context %d, history %d, last observation %d)",
        budget.total_chars,
        budget.context_chars,
        budget.history_chars,
        budget.observation_chars,
    )
    return AgentConfig(
        schema_constrained_decoding=True,
        memory_enabled=False,
        history_budget=history_budget_enabled(),
        prompt_budget=budget,
    )


def _comparator_arm_selected() -> bool:
    """Reads COMPARATOR_ARM once, at wiring time. Unset or "0" is the vLLM arm (today's path);
    "1" is the comparator arm. Anything else is refused: a typo like "true" must not quietly run
    the vLLM client behind the comparator NetworkPolicy, where every call would time out."""
    raw = os.environ.get(COMPARATOR_ARM_ENV, "")
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ComparatorWiringError(f"{COMPARATOR_ARM_ENV}={raw!r} is not '0' or '1'")


def _build_comparator_episode_runner() -> EpisodeRunner:
    """Comparator arm: the Anthropic client in the same orchestrator, sinks and prompt as the vLLM
    arm. Only the model client differs here; egress differs in k8s/networkpolicy-comparator.yaml.

    Fail closed, never a fallback. The vLLM path degrades to log_only_episode_runner when it is
    under-configured; doing that here would leave a pod that answers /healthz, takes every webhook
    and triages nothing, and the window would only find out from an empty usage log after the VM
    has been billing for a while. Falling back to vLLM is worse: the comparator NetworkPolicy has no
    route to it, so every episode would be ModelUnavailable. Raising instead makes the container
    exit before it is ever Ready, so setup_cluster.sh's `rollout status` fails, seed_cluster.sh `up`
    fails, and the launcher stops before any smoke or round starts.

    Refuses: SSLM_LOG_ONLY=1 (a comparator deployment that never calls the model); a missing
    APP_NAMESPACE/PROMETHEUS_URL/SSLM_COMPARATOR_USAGE_LOG (the usage log is the signal the smoke
    guard reads); a malformed VLLM_MAX_OUTPUT_TOKENS; and everything
    cloud_adapter.make_anthropic_model_client refuses (no ANTHROPIC_API_KEY, no or bad
    SSLM_COMPARATOR_MAX_COST_USD, SSLM_HISTORY_BUDGET not off).

    max_output_tokens comes from VLLM_MAX_OUTPUT_TOKENS with the vLLM path's own default, on
    purpose: the output ceiling is not one of the variables the comparison lets differ between arms, so the
    comparator reuses the exact knob and default its vLLM counterpart reads in this same process.
    """
    if os.environ.get("SSLM_LOG_ONLY") == "1":
        raise ComparatorWiringError(
            f"{COMPARATOR_ARM_ENV}=1 with SSLM_LOG_ONLY=1: the comparator arm exists to call the API"
        )
    # The API windows ran without a prompt budget (30k history cap only). The short Sonnet window
    # sets it so the API arm sees the same chars the 9B arm saw; a malformed
    # value still keeps the pod from going Ready. Unset keeps the earlier API prompt byte for byte.
    try:
        comparator_agent_config = _agent_config_from_env_comparator()
    except ValueError as exc:
        raise ComparatorWiringError(str(exc)) from exc
    from harness.agent import TOOL_CONTRACT_ENV_VAR, tool_contract_from_env

    try:
        comparator_contract = tool_contract_from_env()
    except ValueError as exc:
        raise ComparatorWiringError(str(exc)) from exc
    if comparator_contract:
        raise ComparatorWiringError(
            f"{TOOL_CONTRACT_ENV_VAR} is on for the comparator arm; the API windows ran without it"
        )
    from harness.episode_runner import TRIAGE_LOG_ENV

    namespace = os.environ.get("APP_NAMESPACE")
    prometheus_url = os.environ.get("PROMETHEUS_URL")
    missing = [
        name
        for name, value in (
            ("APP_NAMESPACE", namespace),
            ("PROMETHEUS_URL", prometheus_url),
            ("SSLM_COMPARATOR_USAGE_LOG", os.environ.get("SSLM_COMPARATOR_USAGE_LOG")),
            (TRIAGE_LOG_ENV, os.environ.get(TRIAGE_LOG_ENV)),
        )
        if not value
    ]
    if missing:
        raise ComparatorWiringError(f"comparator arm missing env var(s): {', '.join(missing)}")
    try:
        max_output_tokens = int(os.environ.get("VLLM_MAX_OUTPUT_TOKENS", "512"))
    except ValueError as exc:
        raise ComparatorWiringError(f"malformed VLLM_MAX_OUTPUT_TOKENS: {exc}") from exc

    from cloud_adapter import make_anthropic_model_client
    from harness.envs import EnvConfigError

    try:
        model = make_anthropic_model_client(max_output_tokens=max_output_tokens)
    except (EnvConfigError, ValueError) as exc:
        # str(exc) names the variable, never its value: cloud_adapter's messages carry no key.
        raise ComparatorWiringError(str(exc)) from exc
    logger.info(
        "%s=1: comparator arm wired (model %s, max_output_tokens %d, spend cap $%.2f, usage log %s)",
        COMPARATOR_ARM_ENV,
        model.config.model,
        model.config.max_output_tokens,
        model.config.max_cost_usd,
        model.config.usage_log_path,
    )
    return _make_real_runner(
        model, namespace=namespace, prometheus_url=prometheus_url, agent_config=comparator_agent_config
    )


def _make_real_runner(model, *, namespace: str, prometheus_url: str, agent_config=None) -> EpisodeRunner:
    """The orchestrator both arms share: same gates, same TriageRecord and usage sinks."""
    from harness.episode_runner import (
        TRIAGE_LOG_ENV,
        JsonlTriageSink,
        JsonlUsageSink,
        c2_dry_run_enabled,
        make_episode_runner,
        usage_log_path,
    )

    # c2_dry_run_enabled() reads SSLM_C2_DRY_RUN once here, at wiring
    # time (same "read env once, never per episode" contract as VLLM_BASE_URL above). Default is
    # dry-run ON -- see harness/episode_runner.py's module docstring for why the gate exists.
    # Records land in a JSONL the scorer bridge harvests after the round. Unset
    # leaves make_episode_runner's log-only default in place (triage_sink=None).
    triage_log = os.environ.get(TRIAGE_LOG_ENV)
    return make_episode_runner(
        model=model,
        namespace=namespace,
        prometheus_url=prometheus_url,
        c2_dry_run=c2_dry_run_enabled(),
        agent_config=agent_config,
        triage_sink=JsonlTriageSink(triage_log) if triage_log else None,
        usage_sink=JsonlUsageSink(usage_log_path(triage_log)) if triage_log else None,
    )


def _build_episode_runner() -> EpisodeRunner:
    """Wires harness.episode_runner's real orchestrator when the environment carries what it
    needs, else falls back to log_only_episode_runner with a logged reason. Same "never a
    committed value" contract as adapters/vllm_model_client.py's VLLM_BASE_URL/VLLM_API_KEY and
    k8s/role-app-mutate.yaml's APP_NAMESPACE: all three, plus PROMETHEUS_URL, come from the
    environment only.

    k8s/deployment.yaml passes APP_NAMESPACE and PROMETHEUS_URL
    to the container (alongside VLLM_BASE_URL/VLLM_API_KEY), and setup_cluster.sh's envsubst call
    for that file lists both, so a deployment applied via setup_cluster.sh's AGENT_INSTALL=1 path
    wires the real orchestrator this function builds below.

    Known limit: the `VllmModelClient` built here is a single instance,
    closed over by the returned `episode_runner` and installed once in `WebhookReceiver` for the
    life of the process. `harness/episode_runner.py`'s own concurrency note already covers the
    safety-relevant path (each episode's `EpisodeEnvelopeState`/gates are correctly scoped per
    call, not shared); this is the non-safety-relevant corollary -- concurrently handled `/alerts`
    requests share this one client's unsynchronized `stats` counters (`calls`/`retries`/
    `seconds`), so per-episode `model_seconds`/`model_calls` telemetry can be slightly off under
    concurrent episodes. Not the execution/classify path, just its bookkeeping.
    """
    # The admission round validates the continuous instrumentation, not the model.  The deployment still has
    # the same in-cluster topology and Alertmanager route as later rounds, but must not issue
    # inference requests while its only verdict is receiver delivery.  This explicit switch is
    # preferable to omitting VLLM_BASE_URL: setup_cluster deliberately requires that endpoint
    # to keep the Deployment and NetworkPolicy contract identical between rounds.
    if _comparator_arm_selected():
        return _build_comparator_episode_runner()
    if os.environ.get("SSLM_LOG_ONLY") == "1":
        logger.info("SSLM_LOG_ONLY=1: webhook receiver is recording delivery without model inference")
        return log_only_episode_runner

    base_url = os.environ.get("VLLM_BASE_URL")
    namespace = os.environ.get("APP_NAMESPACE")
    prometheus_url = os.environ.get("PROMETHEUS_URL")
    missing = [
        name
        for name, value in (
            ("VLLM_BASE_URL", base_url),
            ("APP_NAMESPACE", namespace),
            ("PROMETHEUS_URL", prometheus_url),
        )
        if not value
    ]
    if missing:
        logger.warning(
            "real episode orchestrator not wired: missing env var(s) %s -- falling back to "
            "log_only_episode_runner (see this module's docstring and harness/episode_runner.py)",
            ", ".join(missing),
        )
        return log_only_episode_runner

    # This function's own docstring promises "degrades... instead of crashing at startup", but
    # that guarantee only covered a MISSING var (the `missing` list above), not a malformed one,
    # the same "plausible operator typo" class the PROMETHEUS_URL scheme-less-URL guard handles,
    # just at startup instead of mid-episode. int()/float() raise ValueError on a bad VLLM_MAX_OUTPUT_TOKENS/VLLM_TEMPERATURE,
    # and _build_episode_runner runs from main() before serve() starts, so an unguarded raise here
    # crashes the process rather than degrading to log-only.
    try:
        max_output_tokens = int(os.environ.get("VLLM_MAX_OUTPUT_TOKENS", "512"))
        temperature = float(os.environ.get("VLLM_TEMPERATURE", "0.0"))
    except ValueError as exc:
        logger.warning(
            "real episode orchestrator not wired: malformed VLLM_MAX_OUTPUT_TOKENS/VLLM_TEMPERATURE "
            "(%s) -- falling back to log_only_episode_runner",
            exc,
        )
        return log_only_episode_runner

    agent_config = _agent_config_from_env()

    from adapters.vllm_model_client import VllmModelClient, VllmModelConfig, parse_decoding_dialect

    try:
        decoding_dialect = parse_decoding_dialect(os.environ.get("VLLM_DECODING_DIALECT", ""))
    except ValueError as exc:
        raise DecodingDialectWiringError(f"VLLM_DECODING_DIALECT: {exc}") from exc
    model_name = os.environ.get("VLLM_MODEL") or "Qwen2.5-7B-Instruct"
    logger.info("vLLM client: model %s, decoding dialect %s", model_name, decoding_dialect)

    model = VllmModelClient(
        VllmModelConfig(
            base_url=base_url,
            # configs/base.yaml's fixed model (README's "Base fixed 2026-08-25"); overridable,
            # never assumed to be the only option a deployment could run. Empty counts as unset:
            # the Deployment always carries VLLM_MODEL, empty when the caller gave none.
            model=model_name,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            api_key=os.environ.get("VLLM_API_KEY") or None,
            decoding_dialect=decoding_dialect,
        )
    )
    return _make_real_runner(
        model, namespace=namespace, prometheus_url=prometheus_url, agent_config=agent_config
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    try:
        episode_runner = _build_episode_runner()
    except ComparatorWiringError as exc:
        # See _build_comparator_episode_runner: exit before the pod is ever Ready, so the
        # rollout fails and the window stops before it spends anything.
        logger.critical("comparator arm not wired, refusing to start: %s", exc)
        raise SystemExit(2) from exc
    except PromptBudgetWiringError as exc:
        logger.critical("prompt budget unusable, refusing to start: %s", exc)
        raise SystemExit(2) from exc
    except DecodingDialectWiringError as exc:
        logger.critical("decoding dialect unusable, refusing to start: %s", exc)
        raise SystemExit(2) from exc
    serve(port=args.port, episode_runner=episode_runner)


if __name__ == "__main__":
    main(sys.argv[1:])
