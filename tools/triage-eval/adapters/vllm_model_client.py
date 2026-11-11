"""ModelClient implementation backed by a vLLM OpenAI-compatible server (harness.agent.ModelClient
protocol), for the cloud GPU environments. Same protocol as
adapters.mlx_model_client.MlxModelClient; the agent loop cannot tell which backend it is talking
to, which is the whole point of the env column (per-env result schema, see
tools/triage-eval/configs/base.yaml).

Deliberately provider-agnostic: this module knows nothing about any specific cloud provider. It
only knows VLLM_BASE_URL (an HTTP address) and VLLM_API_KEY (a bearer token, optional), both
supplied by the caller. See
tests/triage_eval/test_harness_no_provider_reference.py, which greps this file (and the rest
of harness/ and adapters/) for provider names to keep that true.

Uses only the standard library (urllib), not `requests`: tools/triage-eval/requirements.txt
does not carry an HTTP client dependency today, and this is the harness's only HTTP call.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from harness.metrics import GenerateStats
from harness.schema import AGENT_ACTION_DECODING_SCHEMA


_HTTP_ERROR_BODY_MAX_CHARS = 2000  # enough for vLLM's validation-error JSON, short of echoing
# an entire long prompt back into model_error / matrix.csv.


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read()
    except Exception:  # noqa: BLE001 - best-effort; a body we can't read must not mask the error
        return "<no response body>"
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return "<empty response body>"
    if len(text) > _HTTP_ERROR_BODY_MAX_CHARS:
        text = text[:_HTTP_ERROR_BODY_MAX_CHARS] + "...<truncated>"
    return text


# How a constrained request names its schema. "guided_json" is the vLLM v0.6.3 request shape every
# window through 2026-10-20 ran with (`guided_json` + `guided_decoding_backend`). vLLM dropped
# those fields from the chat request later; a newer server ignores them and decodes UNCONSTRAINED,
# with no error, so the dialect is declared by the caller and never guessed. "structured_outputs"
# is the newer shape (`structured_outputs.json`), with the backend fixed server side
# (`--structured-outputs-config`, cloud-init) instead of per request. Qwen3.5 only loads on the
# newer server.
DECODING_DIALECTS = ("guided_json", "structured_outputs")


def parse_decoding_dialect(raw: str) -> str:
    """Empty keeps "guided_json", the request of every earlier window. Anything outside
    DECODING_DIALECTS raises: a typo must not fall back to a shape the server would ignore. The
    caller reads the raw value (the pod from VLLM_DECODING_DIALECT); this module knows no env."""
    if raw == "":
        return "guided_json"
    if raw not in DECODING_DIALECTS:
        raise ValueError(f"decoding dialect {raw!r} must be one of {DECODING_DIALECTS}")
    return raw


def _usage_deltas(response_json: dict) -> dict[str, int]:
    usage = response_json.get("usage")
    if isinstance(usage, dict):
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (prompt, completion)):
            return {"prompt_tokens": prompt, "completion_tokens": completion}
    return {"calls_without_usage": 1}


class ModelUnavailable(Exception):
    """Raised on a transport failure (connection refused, timeout, 5xx).

    Must never be turned into a malformed-output turn: that would score a dead tunnel or a
    provider outage as a model failure. The caller (run_matrix.py, not this module) is
    responsible for writing success=None plus model_error, the existing "not measured, never
    failed" convention already used by eval_error (harness/metrics.py).
    """


@dataclass
class VllmModelConfig:
    base_url: str  # From the VLLM_BASE_URL env var and nowhere else. Never a YAML file: the
    # address changes on every `make up`, and a committed YAML with an IP is stale state in git.
    model: str
    max_output_tokens: int
    temperature: float
    timeout_seconds: float = 60.0
    connect_retries: int = 1
    # From the VLLM_API_KEY env var and nowhere else, same rule as base_url. Optional: the
    # loopback-only default lab config runs vLLM with no --api-key, so an unset/empty key is the
    # expected common case, not a misconfiguration.
    api_key: str | None = None
    # See DECODING_DIALECTS. The default is the request shape every earlier window used.
    decoding_dialect: str = "guided_json"

    def __post_init__(self) -> None:
        if self.decoding_dialect not in DECODING_DIALECTS:
            raise ValueError(f"decoding_dialect={self.decoding_dialect!r} must be one of {DECODING_DIALECTS}")


class VllmModelClient:
    """Posts to {base_url}/chat/completions. Uses the chat endpoint (not /completions) so the
    model's chat template is applied server side, matching what
    adapters.mlx_model_client.MlxModelClient._chat_prompt does locally -- the whole point of
    comparing envs is that only the backend changes, never the prompt shape."""

    def __init__(self, config: VllmModelConfig) -> None:
        self.config = config
        # Counters per instance so a flaky tunnel is visible in the data rather than smoothed
        # away (model_seconds, model_calls, model_retries).
        # run_matrix.py reads these off the client after each episode; this module does not
        # write to RunRecord itself, to keep this file's only job being ModelClient.
        self.stats = GenerateStats()
        # The in-pod driver shares one client across concurrent episodes, each on its own
        # thread (entrypoint._default_dispatch), so per-episode counts live per thread.
        self._local = threading.local()

    def _thread_stats(self) -> GenerateStats:
        stats = getattr(self._local, "stats", None)
        if stats is None:
            stats = self._local.stats = GenerateStats()
        return stats

    def take_thread_stats(self) -> GenerateStats:
        """This thread's counters since the last take, then reset. harness/episode_runner.py
        calls it once before and once after each episode."""
        stats = self._thread_stats()
        self._local.stats = GenerateStats()
        return stats

    def _add(self, **deltas: float) -> None:
        for stats in (self.stats, self._thread_stats()):
            for name, delta in deltas.items():
                setattr(stats, name, getattr(stats, name) + delta)

    def generate(self, prompt: str, *, schema_constrained: bool) -> str:
        # Wall time inside generate, retries and failures included: model_seconds is what
        # bounds the network share of ttm_seconds by measurement instead of by opinion.
        start = time.monotonic()
        try:
            return self._generate(prompt, schema_constrained=schema_constrained)
        finally:
            self._add(seconds=time.monotonic() - start)

    def _generate(self, prompt: str, *, schema_constrained: bool) -> str:
        body: dict = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        if schema_constrained:
            # Same schema OBJECT the MLX client uses, not a copy: an early run lost six turns to
            # command=null because the validator and the decoder disagreed, and two
            # hand-maintained copies would reopen exactly that hole.
            if self.config.decoding_dialect == "structured_outputs":
                body["structured_outputs"] = {"json": AGENT_ACTION_DECODING_SCHEMA}
            else:
                body["guided_json"] = AGENT_ACTION_DECODING_SCHEMA
                body["guided_decoding_backend"] = "outlines"

        payload = json.dumps(body).encode("utf-8")
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        attempt = 0
        last_exc: Exception | None = None
        while attempt <= self.config.connect_retries:
            try:
                response_json = self._post(url, payload)
                self._add(calls=1, **_usage_deltas(response_json))
                # Verbatim, no fence stripping or JSON repair: the harness's malformed-output
                # retry loop is part of what is being measured.
                return response_json["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                # HTTPError is a URLError subclass, so without this branch it fell into the one
                # below and its response body (vLLM's actual rejection reason, e.g. a guided-json
                # schema-compile error) was discarded -- only "HTTP Error 400: Bad Request"
                # survived into model_error. A round on 2026-09-17 hit this twice on config_b and
                # the real cause was unrecoverable after the fact. Read the body here or it is
                # gone for good.
                last_exc = exc
                detail = _read_http_error_body(exc)
                attempt += 1
                if attempt <= self.config.connect_retries:
                    self._add(retries=1)
                    continue
                raise ModelUnavailable(
                    f"vLLM request failed after {attempt} attempt(s): "
                    f"HTTP {exc.code} {exc.reason}: {detail}"
                ) from exc
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                last_exc = exc
                attempt += 1
                if attempt <= self.config.connect_retries:
                    self._add(retries=1)
                    continue
                raise ModelUnavailable(f"vLLM request failed after {attempt} attempt(s): {exc}") from exc

        # Unreachable in practice (the loop above always returns or raises), kept for clarity.
        raise ModelUnavailable(f"vLLM request failed: {last_exc}")

    def _post(self, url: str, payload: bytes) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            if response.status >= 500:
                raise ModelUnavailable(f"vLLM server returned {response.status}")
            return json.loads(response.read().decode("utf-8"))
