"""ModelClient implementation backed by the Anthropic Messages API (harness.agent.ModelClient
protocol): the hosted-model comparator arm. Same protocol as adapters.vllm_model_client.VllmModelClient, so the agent loop cannot
tell which backend it is talking to; the only things that differ between arms are the model and
the network egress the pod is allowed (k8s/networkpolicy-comparator.yaml).

Pinned on purpose: `PINNED_MODEL` is the only model this client accepts. A comparison arm whose
model silently drifts (an alias moving, someone passing another id) is not a comparison.

Structured output: when the agent asks for schema_constrained, the request carries
`output_config.format = json_schema` derived from harness.schema.AGENT_ACTION_SCHEMA (never a
hand-kept copy). Forced tool_choice is a 400 on this model, so tool use is not an option here.
Keywords the API does not enforce (minLength/minimum/maximum) are stripped from the wire schema;
the harness still validates every reply against the full schema (agent._parse_action), so a reply
the API let through but the schema rejects is counted as malformed, same as on the vLLM arm.

No `temperature`: a non-default value is a 400 on this model. Sampling is therefore not matched
to the vLLM arm (configs/base.yaml model.temperature); the pre-registration states it.

Prompt caching: the agent hands over one prompt string (system prompt, task, optional memory,
history window, last observation). Everything before the first volatile section is identical
across the turns of an episode, so it is sent as its own block with `cache_control`; the volatile
tail is a second block. Whether the prefix is long enough to be cached is the API's call (a
too-short prefix silently is not), which is why cache_read/cache_write tokens are logged per
request instead of assumed.

Per-request usage and cost are logged (logger `sslm.anthropic`, plus a JSONL file when
`usage_log_path` is set) and accumulated in `.totals`; `max_cost_usd` stops the arm with
ModelUnavailable (not a malformed turn: a spent budget is not a model failure) before the prepaid
credit runs out mid-episode. It is a soft brake: cost is only known after a response, so with N
episode threads in flight the arm can overshoot by up to N requests. A response with no usage
block is charged a worst-case estimate (see `_estimated_cost_usd`) so it still counts against the
cap.

Provider-agnostic w.r.t. the harness rule: this module reads no environment variable. The key is
a plain config field the caller supplies (cloud_adapter.make_anthropic_model_client reads
ANTHROPIC_API_KEY, which the operator exports from their own secret store). Stdlib only, same reason as the vLLM client
(requirements-agent.txt carries no HTTP client and the container image should not grow one).
"""

from __future__ import annotations

import copy
import http.client
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from adapters.vllm_model_client import ModelUnavailable, _read_http_error_body
from harness.metrics import GenerateStats
from harness.prompt_accounting import log_fields as _chars_log_fields
from harness.schema import AGENT_ACTION_SCHEMA

logger = logging.getLogger("sslm.anthropic")

PINNED_MODEL = "claude-sonnet-5-5"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"

# USD per million tokens for PINNED_MODEL (first-party list price, 2026-09-25): $2 in, $10 out,
# cache reads $0.20. The 5-minute cache write is billed at 1.25x input; that multiplier is the
# documented rule, not a number read from a response, so it is the one figure here to re-check
# against the invoice after the first paid run.
PRICE_INPUT_PER_MTOK = 2.00
PRICE_OUTPUT_PER_MTOK = 10.00
PRICE_CACHE_READ_PER_MTOK = 0.20
PRICE_CACHE_WRITE_PER_MTOK = 2.50

# Where agent.Agent._build_prompt starts the sections that change turn to turn. Everything before
# the first of these is the cacheable prefix. Duplicated here (not imported) so this client does not
# reach into agent.py internals; tests/triage_eval/test_anthropic_model_client.py builds a real
# prompt with Agent._build_prompt and fails if any marker drifts.
VOLATILE_MARKERS = (
    "\nRelevant past attempts:",
    "\nEarlier turns this episode",
    "\nLast observation:",
)

_UNSUPPORTED_SCHEMA_KEYWORDS = ("minLength", "maxLength", "minimum", "maximum")
_RETRYABLE_STATUS = (429, 500, 502, 503, 504, 529)
_MAX_BACKOFF_SECONDS = 30.0
# Worst-case token estimate for a response that carries no usage block: every 2 bytes of request
# body counted as one input token (real text runs ~3.5 chars/token, so this overcounts) plus the
# full max_tokens as output.
_ESTIMATE_BYTES_PER_TOKEN = 2


def _strip_unsupported(node):
    if isinstance(node, dict):
        return {
            k: _strip_unsupported(v) for k, v in node.items() if k not in _UNSUPPORTED_SCHEMA_KEYWORDS
        }
    if isinstance(node, list):
        return [_strip_unsupported(v) for v in node]
    return node


def build_output_schema() -> dict:
    """Wire schema for output_config.format, derived from the harness validation schema.

    Root stays a plain object (no anyOf): the conditional "command required for inspect/remediate"
    is dropped like in harness.schema.build_decoding_schema, and command is widened to
    string-or-null so escalate/submit can say null. The harness validator is what enforces the
    conditional afterwards."""
    schema = copy.deepcopy(AGENT_ACTION_SCHEMA)
    for key in ("$schema", "if", "then"):
        schema.pop(key, None)
    schema = _strip_unsupported(schema)
    schema["additionalProperties"] = False
    schema["properties"]["command"] = {"type": ["string", "null"]}
    if "command" not in schema["required"]:
        schema["required"] = [*schema["required"], "command"]
    return schema


def split_cacheable_prefix(prompt: str) -> tuple[str, str]:
    """(stable prefix, volatile tail). Tail is "" when the prompt has no volatile section."""
    cut = min((i for i in (prompt.find(m) for m in VOLATILE_MARKERS) if i > 0), default=-1)
    if cut == -1:
        return prompt, ""
    return prompt[:cut], prompt[cut:]


def request_cost_usd(
    *, input_tokens: int, output_tokens: int, cache_read_tokens: int, cache_write_tokens: int
) -> float:
    return (
        input_tokens * PRICE_INPUT_PER_MTOK
        + output_tokens * PRICE_OUTPUT_PER_MTOK
        + cache_read_tokens * PRICE_CACHE_READ_PER_MTOK
        + cache_write_tokens * PRICE_CACHE_WRITE_PER_MTOK
    ) / 1_000_000


@dataclass
class AnthropicUsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class AnthropicModelConfig:
    api_key: str  # From the ANTHROPIC_API_KEY env var / Secrets Manager and nowhere else.
    max_output_tokens: int
    model: str = PINNED_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 120.0
    connect_retries: int = 2
    # "between_tools" is how this model turns thinking off (`disabled` is a 400). Off by default so
    # max_output_tokens bounds the visible JSON and the arm is not billed for reasoning the vLLM
    # arm does not do; pass "adaptive" to let the model think. Anything else is refused.
    thinking: str = "between_tools"
    max_cost_usd: float | None = None
    usage_log_path: Path | None = None

    def __post_init__(self) -> None:
        if self.model != PINNED_MODEL:
            raise ValueError(f"comparator arm is pinned to {PINNED_MODEL!r}, got {self.model!r}")
        if not self.api_key:
            raise ValueError("AnthropicModelConfig.api_key is empty")
        if self.thinking not in ("between_tools", "adaptive"):
            raise ValueError(f"thinking must be 'between_tools' or 'adaptive', got {self.thinking!r}")


class AnthropicModelClient:
    """Posts to {base_url}/messages. Same stats/thread-stats contract as VllmModelClient, so
    harness/episode_runner.py can read model_seconds/calls/tokens off it unchanged."""

    def __init__(self, config: AnthropicModelConfig) -> None:
        self.config = config
        self.stats = GenerateStats()
        self.totals = AnthropicUsageTotals()
        self._lock = threading.Lock()
        self._local = threading.local()
        self._sleep = time.sleep

    def _thread_stats(self) -> GenerateStats:
        stats = getattr(self._local, "stats", None)
        if stats is None:
            stats = self._local.stats = GenerateStats()
        return stats

    def take_thread_stats(self) -> GenerateStats:
        stats = self._thread_stats()
        self._local.stats = GenerateStats()
        return stats

    def _add(self, **deltas: float) -> None:
        for stats in (self.stats, self._thread_stats()):
            for name, delta in deltas.items():
                setattr(stats, name, getattr(stats, name) + delta)

    def generate(self, prompt: str, *, schema_constrained: bool) -> str:
        start = time.monotonic()
        try:
            return self._generate(prompt, schema_constrained=schema_constrained)
        finally:
            self._add(seconds=time.monotonic() - start)

    def _build_body(self, prompt: str, schema_constrained: bool) -> dict:
        stable, volatile = split_cacheable_prefix(prompt)
        content = [{"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}}]
        if volatile:
            content.append({"type": "text", "text": volatile})
        body: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": self.config.thinking},
        }
        if schema_constrained:
            body["output_config"] = {"format": {"type": "json_schema", "schema": build_output_schema()}}
        return body

    def _generate(self, prompt: str, *, schema_constrained: bool) -> str:
        budget = self.config.max_cost_usd
        with self._lock:
            spent = self.totals.cost_usd
        if budget is not None and spent >= budget:
            raise ModelUnavailable(
                f"comparator budget reached: ${spent:.4f} >= ${budget:.2f} "
                "(max_cost_usd); episode left unmeasured, not scored as a model failure"
            )

        payload = json.dumps(self._build_body(prompt, schema_constrained)).encode("utf-8")
        url = f"{self.config.base_url.rstrip('/')}/messages"

        attempt = 0
        while True:
            call_start = time.monotonic()
            try:
                response_json, request_id = self._post(url, payload)
            except urllib.error.HTTPError as exc:
                detail = _read_http_error_body(exc)
                attempt += 1
                if exc.code in _RETRYABLE_STATUS and attempt <= self.config.connect_retries:
                    self._add(retries=1)
                    self._sleep(self._backoff(exc, attempt))
                    continue
                raise ModelUnavailable(
                    f"Anthropic request failed after {attempt} attempt(s): HTTP {exc.code} {exc.reason}: {detail}"
                ) from exc
            except (
                urllib.error.URLError,
                ConnectionError,
                TimeoutError,
                OSError,
                http.client.HTTPException,
                ValueError,
            ) as exc:
                # ValueError covers a truncated or non-JSON body (json.JSONDecodeError);
                # HTTPException covers IncompleteRead. Same path as a dropped connection.
                # A body arrived (ValueError, HTTPException), or a bare TimeoutError hit the read
                # after the request was sent (urllib wraps connect-phase timeouts in URLError, which
                # stays free): the request was likely billed, so charge the worst-case estimate and
                # a run of broken replies or read timeouts still hits the budget cap.
                post_send = isinstance(exc, (ValueError, http.client.HTTPException)) or (
                    isinstance(exc, TimeoutError) and not isinstance(exc, urllib.error.URLError)
                )
                if post_send:
                    with self._lock:
                        self.totals.cost_usd += self._estimated_cost_usd(len(payload))
                attempt += 1
                if attempt <= self.config.connect_retries:
                    self._add(retries=1)
                    self._sleep(self._backoff(None, attempt))
                    continue
                raise ModelUnavailable(f"Anthropic request failed after {attempt} attempt(s): {exc}") from exc
            break

        self._record_usage(
            response_json, request_id, time.monotonic() - call_start, len(payload), prompt_chars=len(prompt)
        )
        # Verbatim text, no fence stripping or repair, same contract as the vLLM arm. A refusal or
        # max_tokens stop yields whatever text exists (possibly ""), which the harness scores as a
        # malformed turn; the stop_reason is in the usage log for the post-mortem.
        return "".join(
            block.get("text", "") for block in response_json.get("content", []) if block.get("type") == "text"
        )

    @staticmethod
    def _backoff(exc: urllib.error.HTTPError | None, attempt: int) -> float:
        retry_after = exc.headers.get("retry-after") if exc is not None and exc.headers else None
        try:
            if retry_after is not None:
                return min(float(retry_after), _MAX_BACKOFF_SECONDS)
        except ValueError:
            pass
        return min(2.0**attempt, _MAX_BACKOFF_SECONDS)

    def _post(self, url: str, payload: bytes) -> tuple[dict, str | None]:
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            headers = getattr(response, "headers", None)
            request_id = headers.get("request-id") if headers else None
            return json.loads(response.read().decode("utf-8")), request_id

    def _estimated_cost_usd(self, payload_bytes: int) -> float:
        return request_cost_usd(
            input_tokens=payload_bytes // _ESTIMATE_BYTES_PER_TOKEN + 1,
            output_tokens=self.config.max_output_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )

    def _record_usage(
        self,
        response_json: dict,
        request_id: str | None,
        seconds: float,
        payload_bytes: int = 0,
        prompt_chars: int | None = None,
    ) -> None:
        usage = response_json.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        counts = {
            name: usage.get(name) if isinstance(usage.get(name), int) and usage.get(name) >= 0 else 0
            for name in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        }
        # A usage block whose counts are all missing, malformed or zero is as good as none: a
        # real reply always reads at least one input token, so charge the estimate either way.
        if not usage or not any(counts.values()):
            self._add(calls=1, calls_without_usage=1)
            estimate = self._estimated_cost_usd(payload_bytes)
            with self._lock:
                self.totals.cost_usd += estimate
            self._append_usage_log(
                {
                    "ts": time.time(),
                    "request_id": request_id,
                    "model": response_json.get("model", self.config.model),
                    "stop_reason": response_json.get("stop_reason"),
                    "seconds": round(seconds, 3),
                    "cost_usd": estimate,
                    "usage_estimated": True,
                    **_chars_log_fields(prompt_chars),
                }
            )
            logger.warning(
                "anthropic response %s carried no usage block; charged worst-case estimate $%.4f",
                request_id,
                estimate,
            )
            return
        cost = request_cost_usd(
            input_tokens=counts["input_tokens"],
            output_tokens=counts["output_tokens"],
            cache_read_tokens=counts["cache_read_input_tokens"],
            cache_write_tokens=counts["cache_creation_input_tokens"],
        )
        # GenerateStats.prompt_tokens means "everything the model read", cached or not, so the
        # per-episode token accounting stays comparable with the vLLM arm; the split lives in the
        # usage log.
        self._add(
            calls=1,
            prompt_tokens=counts["input_tokens"]
            + counts["cache_read_input_tokens"]
            + counts["cache_creation_input_tokens"],
            completion_tokens=counts["output_tokens"],
        )
        record = {
            "ts": time.time(),
            "request_id": request_id,
            "model": response_json.get("model", self.config.model),
            "stop_reason": response_json.get("stop_reason"),
            "seconds": round(seconds, 3),
            "cost_usd": cost,
            **counts,
            **_chars_log_fields(prompt_chars),
        }
        with self._lock:
            totals = self.totals
            totals.requests += 1
            totals.input_tokens += counts["input_tokens"]
            totals.output_tokens += counts["output_tokens"]
            totals.cache_read_tokens += counts["cache_read_input_tokens"]
            totals.cache_write_tokens += counts["cache_creation_input_tokens"]
            totals.cost_usd += cost
        self._append_usage_log(record)
        logger.info("anthropic request %s", json.dumps(record))

    def _append_usage_log(self, record: dict) -> None:
        # Runs after a paid response: a failing log write must not lose the reply or escape as a
        # non-ModelUnavailable error. The record still goes to the logger and to .totals.
        if self.config.usage_log_path is None:
            return
        try:
            with self._lock, open(self.config.usage_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.error("anthropic usage log write failed (%s): %s", exc, json.dumps(record))
