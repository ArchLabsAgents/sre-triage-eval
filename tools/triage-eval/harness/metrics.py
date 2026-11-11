"""Per-task result records and the 18-run matrix aggregator.

Records, per (scenario, config) cell: success, output tokens, elapsed time, attempt count.
`aggregate` produces the headline number of the matrix: success rate, mean tokens, mean time
per config, averaged only over successful tasks for tokens/time (an unsuccessful task's token
count is cost without a result, and mixing it into the mean muddies "cost per solved task" with
"cost of failing").

Two verdicts per run, kept apart on purpose: `submitted` is what the agent loop sees (the
environment accepted a submit(), which for AIOpsLab mitigation problems means only that submit()
was called), `success` is the benchmark's own eval of the cluster state after the episode. An
early matrix reported the first as if it were the second; the primary number is `success`, and
`None` there means the eval did not return one (not measured), never "failed".
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agent import EpisodeResult

# `env` first and the two declared confounds right behind it, so a human reading one row can
# never read two environments as one table. Everything else about the env stays in
# <env>/env-descriptor.json rather than widening the CSV.
CSV_FIELDS = [
    "env",
    "env_quantization",
    "env_network",
    "scenario_id",
    "config_name",
    "success",
    "submitted",
    "ttm_seconds",
    "output_tokens",
    "elapsed_seconds",
    "model_seconds",
    "model_calls",
    "attempt_count",
    "teardown_error",
    "eval_error",
    "model_error",
    "model_retries",
    "root_cause_fixed",
    "probe_detail",
    "probe_error",
]


@dataclass
class GenerateStats:
    """Per-client counters for time spent inside `generate`:
    with model_seconds/model_calls measured, the network share of ttm_seconds is bounded by
    measurement instead of by opinion, and a flaky tunnel shows up as retries in the data
    rather than being smoothed away. Model clients expose this as `.stats`; run_matrix.py
    reads it off the client after each episode."""

    calls: int = 0
    retries: int = 0
    seconds: float = 0.0
    # Exact counts from the server's `usage` block, never estimated. A successful call whose
    # response carried no usable `usage` is counted in `calls_without_usage` instead, so an
    # incomplete token count is visible rather than read as a cheap episode.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls_without_usage: int = 0


@dataclass
class RunRecord:
    scenario_id: str
    config_name: str
    submitted: bool
    output_tokens: int
    elapsed_seconds: float
    attempt_count: int
    # The benchmark eval's verdict; None = the eval returned no `success` (not measured).
    success: bool | None = None
    # Time-to-mitigate measured by AIOpsLab itself (results["TTM"]), when returned.
    ttm_seconds: float | None = None
    # Teardown (recover_fault/cleanup) that failed AFTER the eval: the verdict stands, the
    # cluster may have been left dirty for the next episode.
    teardown_error: str | None = None
    # Benchmark eval that raised (observed: a TypeError inside AIOpsLab with a pod Pending).
    # The verdict stays not measured (success=None), the reason goes here.
    eval_error: str | None = None
    # The fault's symptom, measured by adapters/mitigation_probes.py. An axis independent of
    # `success` (a no-op control round passed the benchmark eval 6/6, so the eval alone cannot
    # tell a fix from no fix). None = not measured.
    root_cause_fixed: bool | None = None
    probe_detail: str | None = None
    probe_error: str | None = None
    # env column: required on a new record, None only on a legacy record (written before the
    # column existed) or a test config with no declared env. Legacy loads as env=None, renders
    # "unknown" in the CSV and gets its own bucket in the aggregate: never folded into
    # mlx-local by default (the explicit stamp is scenarios/backfill_env.py).
    env: str | None = None
    env_quantization: str | None = None
    env_network: str | None = None
    # Network latency instrument: model_seconds is the wall time INSIDE generate(); with it the
    # network share of ttm_seconds is bounded by measurement. None = not measured (client
    # without .stats), never zero.
    model_seconds: float | None = None
    model_calls: int | None = None
    model_retries: int | None = None
    # Transport failure (ModelUnavailable): a dead tunnel never becomes a malformed turn; the
    # verdict stays not measured (success=None) and the reason goes here, same convention as
    # eval_error.
    model_error: str | None = None

    @classmethod
    def from_episode(
        cls,
        scenario_id: str,
        config_name: str,
        result: EpisodeResult,
        verdict: dict[str, Any] | None = None,
        *,
        env: str | None = None,
        env_quantization: str | None = None,
        env_network: str | None = None,
        model_stats: GenerateStats | None = None,
    ) -> "RunRecord":
        verdict = verdict or {}
        success = verdict.get("success")
        ttm = verdict.get("TTM")
        return cls(
            scenario_id=scenario_id,
            config_name=config_name,
            submitted=result.solved,
            output_tokens=result.total_output_tokens,
            elapsed_seconds=result.elapsed_seconds,
            attempt_count=result.attempt_count,
            success=bool(success) if success is not None else None,
            ttm_seconds=float(ttm) if ttm is not None else None,
            teardown_error=verdict.get("teardown_error"),
            eval_error=verdict.get("eval_error"),
            root_cause_fixed=verdict.get("root_cause_fixed"),
            probe_detail=verdict.get("probe_detail"),
            probe_error=verdict.get("probe_error"),
            env=env,
            env_quantization=env_quantization,
            env_network=env_network,
            model_seconds=model_stats.seconds if model_stats else None,
            model_calls=model_stats.calls if model_stats else None,
            model_retries=model_stats.retries if model_stats else None,
        )

    @classmethod
    def from_json(cls, text: str) -> "RunRecord":
        data = json.loads(text)
        # The earliest record format called `submitted` `solved` and had no verdict: it loads as
        # submitted + success=None, never as success=solved.
        if "solved" in data and "submitted" not in data:
            data["submitted"] = data.pop("solved")
        return cls(**data)

    def write(self, results_dir: str | Path) -> Path:
        # A record with an env goes to results/<env>/: the per-env directory is what prevents a
        # silent overwrite when two envs write the same (scenario, config). Without env, the
        # legacy layout (root).
        base = Path(results_dir)
        if self.env is not None:
            base = base / self.env
        path = base / f"{self.scenario_id}_{self.config_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self)), encoding="utf-8")
        return path


@dataclass
class ConfigSummary:
    config_name: str
    # The bucket's env. None = legacy records without env ("unknown"): their own bucket, never
    # folded into a named env.
    env: str | None
    scenarios_attempted: int
    scenarios_succeeded: int
    scenarios_submitted: int
    # Runs without an eval verdict (success=None). Counted in attempted, never in succeeded.
    scenarios_unmeasured: int
    # Episodes where the root-cause probe returned True (a stricter axis than success).
    scenarios_root_cause_fixed: int
    mean_output_tokens_succeeded: float | None
    mean_elapsed_seconds_succeeded: float | None

    @property
    def success_rate(self) -> float:
        if self.scenarios_attempted == 0:
            return 0.0
        return self.scenarios_succeeded / self.scenarios_attempted

    @property
    def submit_rate(self) -> float:
        if self.scenarios_attempted == 0:
            return 0.0
        return self.scenarios_submitted / self.scenarios_attempted


def aggregate(records: list[RunRecord]) -> list[ConfigSummary]:
    """Fold raw RunRecords into one ConfigSummary per (env, config_name) pair.

    This is the function that produces the matrix's headline number: success rate plus mean
    cost/time per successful task, per configuration. Submit rate rides along as a separate
    column because it measures something else (confidence to stop).

    Grouping is by (env, config_name), never by config alone, and there is deliberately no
    function that returns a single number across envs: a cross-env delta measures quantization
    plus hardware plus inference engine together, and offering it would be the same class of
    error as reporting `submitted` as if it were `success`. A cross-env view is two tables side
    by side. Legacy records (env=None) get their own bucket
    and never appear inside a named env's summary.
    """
    by_key: dict[tuple[str | None, str], list[RunRecord]] = {}
    for record in records:
        by_key.setdefault((record.env, record.config_name), []).append(record)

    summaries: list[ConfigSummary] = []
    # Named envs in alphabetical order; the legacy bucket (None) last.
    for (env, config_name), config_records in sorted(
        by_key.items(), key=lambda item: (item[0][0] is None, item[0][0] or "", item[0][1])
    ):
        succeeded = [r for r in config_records if r.success is True]
        summaries.append(
            ConfigSummary(
                config_name=config_name,
                env=env,
                scenarios_attempted=len(config_records),
                scenarios_succeeded=len(succeeded),
                scenarios_submitted=sum(1 for r in config_records if r.submitted),
                scenarios_unmeasured=sum(1 for r in config_records if r.success is None),
                scenarios_root_cause_fixed=sum(1 for r in config_records if r.root_cause_fixed is True),
                mean_output_tokens_succeeded=(
                    sum(r.output_tokens for r in succeeded) / len(succeeded) if succeeded else None
                ),
                mean_elapsed_seconds_succeeded=(
                    sum(r.elapsed_seconds for r in succeeded) / len(succeeded)
                    if succeeded
                    else None
                ),
            )
        )
    return summaries


def write_matrix_csv(records: list[RunRecord], out_path: str | Path) -> Path:
    """Write the raw 18-row (6 scenarios x 3 configs) table, the artifact the final number is
    computed from."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            # A legacy record without env renders "unknown", never an empty cell: whoever reads
            # the table sees the origin was not declared, instead of assuming one.
            if row["env"] is None:
                row["env"] = "unknown"
            writer.writerow(row)
    return path


# ---------------------------------------------------------------------------------------------
# The cost/latency/footprint collector for the 24/7 continuous scenario. A record NEW and
# separate from RunRecord above, never layered on top of it: the continuous round has its own
# env (the "-continuous" env declared in configs/base.yaml) and its own result files under
# scenarios/results/<env>/, so the two schemas can never be read as one table (same rule that
# keeps envs apart in RunRecord).
#
# List price, never invoice: the on-demand USD/h of the continuous env's GPU VM. Kept here as a
# plain number, never a provider name -- harness/ stays provider-agnostic on purpose (see
# tests/triage_eval/test_harness_no_provider_reference.py); which vendor that number prices
# is a fact configs/base.yaml carries, not this module. Any USD figure this module produces is a projection against this constant, not a bill.
PRICE_CONTINUOUS_ENV_USD_PER_HOUR = 0.507


@dataclass
class PodFootprintSample:
    """One scrape tick of the agent pod's cgroup, via cAdvisor/kubelet, scraped by the
    kube-prometheus-stack the cluster setup installs (agent CPU/memory via the kubelet's
    cAdvisor, the container cgroup on the dedicated node).

    Honesty rule, same as the mitigation probes: a tick where the Prometheus query
    returned no series (scrape gap, query timeout, apiserver hiccup) is `collected=False` with
    `reason` set -- never coerced to 0.0. `cpu_cores`/`memory_bytes` are only ever populated
    when `collected` is True.
    """

    timestamp: float  # epoch seconds, Prometheus vector `value[0]`
    collected: bool
    cpu_cores: float | None = None  # rate(container_cpu_usage_seconds_total[..]) summed for the pod
    memory_bytes: float | None = None  # container_memory_working_set_bytes summed for the pod
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.collected and self.reason is None:
            raise ValueError("PodFootprintSample(collected=False) requires a reason (honesty rule)")


@dataclass
class VllmFootprintSample:
    """One tick of the vLLM host, logged OUTSIDE the cluster (nvidia-smi --query-gpu plus a
    cgroup read, by a simple logger on the host). Deliberately a separate dataclass from
    PodFootprintSample: the POD footprint never mixes with the MODEL footprint (separate series,
    separate columns).

    The original design's "cgroup of the systemd unit" undersold what the logger actually
    reads: vllm.service wraps `docker run` in the foreground, so the unit's own cgroup is the docker CLI client, not
    the container (see cgroup_cpu_seconds/cgroup_memory_bytes below and
    log_vllm_footprint.py's _resolve_container_cgroup_path).

    Same honesty rule as PodFootprintSample: `collected=False` always carries a `reason`.
    """

    timestamp: float  # epoch seconds
    collected: bool
    gpu_vram_used_mb: float | None = None  # nvidia-smi --query-gpu=memory.used
    gpu_utilization_pct: float | None = None  # nvidia-smi --query-gpu=utilization.gpu
    gpu_power_watts: float | None = None  # nvidia-smi --query-gpu=power.draw
    # The sslm-vllm CONTAINER's own cgroup (resolved via docker inspect), never vllm.service's
    # cgroup: that unit's ExecStart is `docker run` in the foreground, so its own cgroup holds
    # the docker CLI client, not the GPU-bound container (found live on the GPU host).
    # See log_vllm_footprint.py's _resolve_container_cgroup_path.
    cgroup_cpu_seconds: float | None = None  # container cgroup cpu.stat usage_usec / 1e6
    cgroup_memory_bytes: float | None = None  # container cgroup memory.current
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.collected and self.reason is None:
            raise ValueError("VllmFootprintSample(collected=False) requires a reason (honesty rule)")


def write_footprint_sample_jsonl(
    path: str | Path, sample: PodFootprintSample | VllmFootprintSample | VllmQueueSample | HostSample
) -> None:
    """Append one sample as a JSON line. The collector/logger scripts call this once per tick,
    so a killed process loses at most the in-flight tick, never the whole run."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(sample)) + "\n")


def load_pod_footprint_samples(path: str | Path) -> list[PodFootprintSample]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        PodFootprintSample(**json.loads(line))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_vllm_footprint_samples(path: str | Path) -> list[VllmFootprintSample]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        VllmFootprintSample(**json.loads(line))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# Queue gate of the contention check. vLLM v0.6.3, the lab's pin, exposes no per-request
# queue time on /metrics, only the TTFT histogram, and queue time <= TTFT. So the gate reads the
# TTFT p95 as a CEILING on the queue p95: a ceiling under the limit proves the gate, a ceiling
# over it (or past the last bucket) is "not measured", never a queue number.
TTFT_METRIC = "vllm:time_to_first_token_seconds"
QUEUE_GAUGES = ("vllm:num_requests_waiting", "vllm:num_requests_running")
TOKEN_COUNTERS = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total")


@dataclass
class VllmQueueSample:
    """One scrape of vLLM's /metrics. A series separate from VllmFootprintSample on purpose: a
    footprint tick is all-or-nothing on nvidia-smi + cgroup, and a GPU driver hiccup must not
    blank the queue series (or the reverse). `ttft_buckets` maps the Prometheus `le` string
    ("0.5", "+Inf") to the CUMULATIVE count, exactly as scraped."""

    timestamp: float
    collected: bool
    ttft_buckets: dict[str, float] | None = None
    ttft_count: float | None = None
    ttft_sum: float | None = None
    num_requests_waiting: float | None = None
    num_requests_running: float | None = None
    prompt_tokens_total: float | None = None
    generation_tokens_total: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.collected and self.reason is None:
            raise ValueError("VllmQueueSample(collected=False) requires a reason (honesty rule)")


def load_vllm_queue_samples(path: str | Path) -> list[VllmQueueSample]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        VllmQueueSample(**json.loads(line))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_prometheus_text(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Minimal parser for the Prometheus text exposition format: name -> [(labels, value)].
    Comments and blank lines are skipped; a line that does not parse raises ValueError, so a
    garbled scrape becomes a not-collected sample instead of a partial one."""
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        labels: dict[str, str] = {}
        if "{" in line:
            name, _, rest = line.partition("{")
            body, sep, tail = rest.rpartition("}")
            if not sep:
                raise ValueError(f"unterminated label set: {raw!r}")
            for part in _split_labels(body):
                key, eq, value = part.partition("=")
                if not eq or len(value) < 2 or value[0] != '"' or value[-1] != '"':
                    raise ValueError(f"bad label {part!r} in {raw!r}")
                labels[key.strip()] = value[1:-1]
            value_text = tail.split()[0] if tail.split() else ""
        else:
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"no value in {raw!r}")
            name, value_text = fields[0], fields[1]
        out.setdefault(name.strip(), []).append((labels, float(value_text)))
    return out


def _split_labels(body: str) -> list[str]:
    parts, current, quoted, escaped = [], [], False, False
    for ch in body:
        if escaped:
            current.append(ch)
            escaped = False
        elif ch == "\\":
            current.append(ch)
            escaped = True
        elif ch == '"':
            current.append(ch)
            quoted = not quoted
        elif ch == "," and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if "".join(current).strip():
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def queue_sample_from_metrics(text: str, *, timestamp: float) -> VllmQueueSample:
    """Build a sample from one /metrics body. Each metric must have exactly one series (one
    served model): summing several models' series would hide which one queued."""
    try:
        metrics = parse_prometheus_text(text)
    except ValueError as exc:
        return VllmQueueSample(timestamp=timestamp, collected=False, reason=f"unparsable /metrics: {exc}")

    def single(name: str) -> float:
        series = metrics.get(name, [])
        if len(series) != 1:
            raise KeyError(f"{name}: expected 1 series, found {len(series)}")
        return series[0][1]

    try:
        buckets: dict[str, float] = {}
        for labels, value in metrics.get(f"{TTFT_METRIC}_bucket", []):
            le = labels.get("le")
            if le is None or le in buckets:
                raise KeyError(f"{TTFT_METRIC}_bucket: missing or repeated le={le!r} (more than one series?)")
            buckets[le] = value
        if "+Inf" not in buckets:
            raise KeyError(f"{TTFT_METRIC}_bucket: no +Inf bucket")
        return VllmQueueSample(
            timestamp=timestamp,
            collected=True,
            ttft_buckets=buckets,
            ttft_count=single(f"{TTFT_METRIC}_count"),
            ttft_sum=single(f"{TTFT_METRIC}_sum"),
            num_requests_waiting=single(QUEUE_GAUGES[0]),
            num_requests_running=single(QUEUE_GAUGES[1]),
            prompt_tokens_total=single(TOKEN_COUNTERS[0]),
            generation_tokens_total=single(TOKEN_COUNTERS[1]),
        )
    except KeyError as exc:
        return VllmQueueSample(timestamp=timestamp, collected=False, reason=str(exc.args[0]))


@dataclass
class QueueCeiling:
    """TTFT p95 over a window, read as a ceiling on the queue p95. `p95_ceiling_seconds` is the
    upper edge of the bucket holding the 95th request, never interpolated down: this number is
    only allowed to overstate the queue. None always comes with a `reason`."""

    start: float
    end: float
    requests: int
    p95_ceiling_seconds: float | None
    max_waiting: float | None
    reason: str | None = None


def _window_edges(
    samples: list[Any], start: float, end: float, max_gap_seconds: float, *, series: str = "/metrics"
) -> tuple[Any, Any] | str:
    """The (baseline, end) pair every counter delta over a window is taken between: the last
    collected sample at or before `start` and the last collected sample at or before `end`. A
    reason string instead when the pair would make the delta a guess, including an edge sample
    more than `max_gap_seconds` from its edge: a logger that died mid-window would otherwise
    shrink the measured span without saying so."""
    collected = sorted((s for s in samples if s.collected), key=lambda s: s.timestamp)
    before = [s for s in collected if s.timestamp <= start]
    upto = [s for s in collected if s.timestamp <= end]
    if not before:
        return f"no collected {series} sample at or before the window start"
    first, last = before[-1], upto[-1]
    if last is first:
        return f"no collected {series} sample inside the window"
    if start - first.timestamp > max_gap_seconds:
        return f"baseline sample is {start - first.timestamp:.0f}s before the window start"
    if end - last.timestamp > max_gap_seconds:
        return f"last sample is {end - last.timestamp:.0f}s before the window end"
    return first, last


def ttft_p95_ceiling(
    samples: list[VllmQueueSample], start: float, end: float, *, max_gap_seconds: float = 60.0
) -> QueueCeiling:
    """Cumulative bucket deltas between the edges `_window_edges` picks. Anything that would make
    the delta a guess (no baseline, a counter going backwards after a vLLM restart, no request,
    an edge sample too far from its edge) is a reason."""
    inside = [s for s in samples if s.collected and start <= s.timestamp <= end]
    max_waiting = max((s.num_requests_waiting for s in inside if s.num_requests_waiting is not None), default=None)

    def unmeasured(reason: str, requests: int = 0) -> QueueCeiling:
        return QueueCeiling(start, end, requests, None, max_waiting, reason)

    edges = _window_edges(samples, start, end, max_gap_seconds)
    if isinstance(edges, str):
        return unmeasured(edges)
    first, last = edges
    if set(first.ttft_buckets) != set(last.ttft_buckets):
        return unmeasured("TTFT bucket layout changed inside the window")
    deltas = {le: last.ttft_buckets[le] - first.ttft_buckets[le] for le in last.ttft_buckets}
    if any(d < 0 for d in deltas.values()):
        return unmeasured("TTFT counters went backwards (vLLM restarted inside the window)")
    total = deltas["+Inf"]
    if total <= 0:
        return unmeasured("no request reached vLLM inside the window")
    requests = int(total)
    target = 0.95 * total
    for le in sorted((k for k in deltas if k != "+Inf"), key=float):
        if deltas[le] >= target:
            return QueueCeiling(start, end, requests, float(le), max_waiting)
    return unmeasured("TTFT p95 is past the largest finite bucket", requests)


# Cost attribution by token slice. N seeds share one vLLM inside one VM window and vLLM counts
# tokens per server, never per seed. The window's list price is spread over every token the
# server counted in it (prompt + generation weighted 1:1), and an episode is charged its own
# exact tokens (the server's `usage`, written per episode by harness/episode_runner.py) at that
# price. Server tokens no fully-measured
# episode claims (retries of a dead call, probes, an episode straddling the edge or with an
# incomplete count) stay in `unattributed_tokens`, never smeared over the episodes.
@dataclass
class EpisodeTokenUsage:
    episode_id: str
    t_start: float
    t_end: float
    prompt_tokens: int
    completion_tokens: int
    calls: int
    calls_without_usage: int


def load_episode_usage(path: str | Path) -> list[EpisodeTokenUsage]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        EpisodeTokenUsage(**json.loads(line))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass
class EpisodeTokenCost:
    episode_id: str
    prompt_tokens: int
    completion_tokens: int
    usd: float | None
    reason: str | None = None


@dataclass
class TokenCostSlice:
    """`usd_window` is always known (hours x list price); everything priced per token is None
    with `reason` when the server-side token count over the window is not a measurement. By
    construction, sum(episode usd) + usd_unattributed == usd_window."""

    start: float
    end: float
    price_usd_per_hour: float
    usd_window: float
    server_prompt_tokens: float | None
    server_generation_tokens: float | None
    usd_per_token: float | None
    episodes: list[EpisodeTokenCost]
    unattributed_tokens: float | None
    usd_unattributed: float | None
    reason: str | None = None


def token_slice_cost(
    samples: list[VllmQueueSample],
    usage: list[EpisodeTokenUsage],
    start: float,
    end: float,
    *,
    price_usd_per_hour: float = PRICE_CONTINUOUS_ENV_USD_PER_HOUR,
    max_gap_seconds: float = 60.0,
) -> TokenCostSlice:
    """Price the [start, end] window per token and charge each episode inside it. The server
    delta runs between the same edges `ttft_p95_ceiling` uses, so it can start up to
    `max_gap_seconds` early and stop up to that much short of `end`: an episode count above the
    server's is refused as a whole, never clipped."""
    if end <= start:
        raise ValueError(f"window must have end > start, got [{start!r}, {end!r}]")
    usd_window = (end - start) / 3600.0 * price_usd_per_hour
    overlapping = [u for u in usage if u.t_end >= start and u.t_start <= end]

    def unmeasured(reason: str, prompt: float | None = None, generation: float | None = None) -> TokenCostSlice:
        return TokenCostSlice(
            start, end, price_usd_per_hour, usd_window, prompt, generation, None,
            [EpisodeTokenCost(u.episode_id, u.prompt_tokens, u.completion_tokens, None, reason) for u in overlapping],
            None, None, reason,
        )

    ids = [u.episode_id for u in overlapping]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        return unmeasured(f"episode_id repeated in the usage log: {', '.join(repeated)}")
    edges = _window_edges(samples, start, end, max_gap_seconds)
    if isinstance(edges, str):
        return unmeasured(edges)
    first, last = edges
    counters = (first.prompt_tokens_total, first.generation_tokens_total,
                last.prompt_tokens_total, last.generation_tokens_total)
    if any(c is None for c in counters):
        return unmeasured("token counters missing from an edge sample")
    prompt = last.prompt_tokens_total - first.prompt_tokens_total
    generation = last.generation_tokens_total - first.generation_tokens_total
    if prompt < 0 or generation < 0:
        return unmeasured("token counters went backwards (vLLM restarted inside the window)")
    server_tokens = prompt + generation
    if server_tokens <= 0:
        return unmeasured("no token reached vLLM inside the window", prompt, generation)

    usd_per_token = usd_window / server_tokens
    costs: list[EpisodeTokenCost] = []
    attributed = 0
    for u in overlapping:
        tokens = u.prompt_tokens + u.completion_tokens
        if u.t_start < start or u.t_end > end:
            reason = "episode straddles the window edge; its tokens stay unattributed"
        elif u.calls_without_usage:
            reason = f"{u.calls_without_usage} call(s) returned no usage; the token count is incomplete"
        else:
            attributed += tokens
            costs.append(EpisodeTokenCost(u.episode_id, u.prompt_tokens, u.completion_tokens, tokens * usd_per_token))
            continue
        costs.append(EpisodeTokenCost(u.episode_id, u.prompt_tokens, u.completion_tokens, None, reason))
    if attributed > server_tokens:
        return unmeasured(
            f"episodes claim {attributed} tokens, more than the {server_tokens:.0f} the server counted in the window",
            prompt,
            generation,
        )
    unattributed = server_tokens - attributed
    return TokenCostSlice(
        start, end, price_usd_per_hour, usd_window, prompt, generation, usd_per_token,
        costs, unattributed, unattributed * usd_per_token,
    )


# Contention gate for running several seeds on one VM. Three signals over one VM window, and the
# window passes only when all three were measured and are under their limit. A signal that is not
# a measurement counts as over (a failed gate means going back to 1 seed per VM).
STEAL_MAX_PCT = 5.0
QUEUE_MAX_FRACTION_OF_MTTA = 0.10
VM_MEMORY_MAX_PCT = 70.0
# /proc/stat's aggregate cpu line: user nice system idle iowait irq softirq steal guest guest_nice.
# guest and guest_nice are already inside user and nice, so the total stops at steal.
_PROC_STAT_TOTAL_FIELDS = 8
_PROC_STAT_STEAL_INDEX = 7


@dataclass
class HostSample:
    """One tick of the lab VM itself, read on the host outside kind: the aggregate cpu line of
    /proc/stat and /proc/meminfo. The kind nodes are containers on this same kernel, so the agent
    node's steal IS this counter; there is no per-node steal to read. A series separate from the
    vLLM footprint and the queue, so one unreadable file never blanks the other two."""

    timestamp: float
    collected: bool
    cpu_total_jiffies: float | None = None
    cpu_steal_jiffies: float | None = None
    mem_total_bytes: float | None = None
    mem_available_bytes: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.collected and self.reason is None:
            raise ValueError("HostSample(collected=False) requires a reason (honesty rule)")


def host_sample_from_proc(stat_text: str, meminfo_text: str, *, timestamp: float) -> HostSample:
    """Build a sample from the text of /proc/stat and /proc/meminfo. A kernel whose cpu line has
    no steal column, or a meminfo without MemAvailable, is not collected: reading steal as 0 or
    free memory as MemFree would be a number the kernel never gave."""

    def missing(reason: str) -> HostSample:
        return HostSample(timestamp=timestamp, collected=False, reason=reason)

    cpu = next((line.split() for line in stat_text.splitlines() if line.split()[:1] == ["cpu"]), None)
    if cpu is None:
        return missing("no aggregate cpu line in /proc/stat")
    fields = cpu[1:]
    if len(fields) < _PROC_STAT_TOTAL_FIELDS:
        return missing(f"/proc/stat cpu line has {len(fields)} fields, no steal column")
    mem: dict[str, float] = {}
    try:
        jiffies = [float(v) for v in fields[:_PROC_STAT_TOTAL_FIELDS]]
        for line in meminfo_text.splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if key in ("MemTotal", "MemAvailable") and parts:
                if parts[1:] != ["kB"]:
                    return missing(f"/proc/meminfo {key} is not in kB: {line!r}")
                mem[key] = float(parts[0]) * 1024
    except ValueError as exc:
        return missing(f"non-numeric value in /proc/stat or /proc/meminfo: {exc}")
    for key in ("MemTotal", "MemAvailable"):
        if key not in mem:
            return missing(f"{key} missing from /proc/meminfo")
    if mem["MemTotal"] <= 0:
        return missing("MemTotal is 0 in /proc/meminfo")
    return HostSample(
        timestamp=timestamp,
        collected=True,
        cpu_total_jiffies=sum(jiffies),
        cpu_steal_jiffies=jiffies[_PROC_STAT_STEAL_INDEX],
        mem_total_bytes=mem["MemTotal"],
        mem_available_bytes=mem["MemAvailable"],
    )


def load_host_samples(path: str | Path) -> list[HostSample]:
    p = Path(path)
    if not p.exists():
        return []
    return [HostSample(**json.loads(line)) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class GateSignal:
    """`status` is "pass", "over" or "not_measured"; `value` is None only when not measured."""

    name: str
    value: float | None
    limit: float | None
    status: str
    reason: str | None = None


def _judge(name: str, value: float, limit: float) -> GateSignal:
    return GateSignal(name, value, limit, "pass" if value <= limit else "over")


STEAL_UNPROVEN_REASON = (
    "steal accounting unproven: the /proc/stat steal counter is 0 since boot, so a hypervisor "
    "that does not account steal and one that stole nothing look the same"
)


def steal_signal(
    samples: list[HostSample],
    start: float,
    end: float,
    *,
    max_gap_seconds: float = 60.0,
    accept_unproven_steal: bool = False,
) -> GateSignal:
    """Steal share of all CPU time over the window, from the counter delta between the same edges
    the queue ceiling uses. A counter going backwards (VM rebooted) is not measured.

    The steal counter is cumulative since boot, so a nonzero value at the closing edge proves the
    hypervisor accounts steal and a zero delta is a real zero. A counter still at 0 since boot
    proves nothing: that is not measured, unless `accept_unproven_steal` signs the limit, and
    then the pass carries the reason so it never reads as a measured zero."""
    name = "cpu_steal_pct"
    edges = _window_edges(samples, start, end, max_gap_seconds, series="host")
    if isinstance(edges, str):
        return GateSignal(name, None, STEAL_MAX_PCT, "not_measured", edges)
    first, last = edges
    total = last.cpu_total_jiffies - first.cpu_total_jiffies
    steal = last.cpu_steal_jiffies - first.cpu_steal_jiffies
    if total < 0 or steal < 0:
        return GateSignal(name, None, STEAL_MAX_PCT, "not_measured", "/proc/stat counters went backwards (VM rebooted inside the window)")
    if total == 0:
        return GateSignal(name, None, STEAL_MAX_PCT, "not_measured", "no CPU time elapsed between the edge samples")
    if last.cpu_steal_jiffies == 0:
        if not accept_unproven_steal:
            return GateSignal(name, None, STEAL_MAX_PCT, "not_measured", STEAL_UNPROVEN_REASON)
        return GateSignal(name, 0.0, STEAL_MAX_PCT, "pass", f"{STEAL_UNPROVEN_REASON}; accepted by --accept-unproven-steal")
    return _judge(name, 100.0 * steal / total, STEAL_MAX_PCT)


def memory_signal(
    samples: list[HostSample], start: float, end: float, *, max_gap_seconds: float = 60.0
) -> GateSignal:
    """PEAK share of VM memory in use (MemTotal - MemAvailable) over the window, never a mean: an
    OOM happens at the peak. A peak is only a measurement if nothing could hide between ticks, so
    every gap between consecutive collected samples, edges included, must be within
    `max_gap_seconds`."""
    name = "vm_memory_peak_pct"
    edges = _window_edges(samples, start, end, max_gap_seconds, series="host")
    if isinstance(edges, str):
        return GateSignal(name, None, VM_MEMORY_MAX_PCT, "not_measured", edges)
    first, last = edges
    span = sorted(
        (s for s in samples if s.collected and first.timestamp <= s.timestamp <= last.timestamp),
        key=lambda s: s.timestamp,
    )
    widest = max(b.timestamp - a.timestamp for a, b in zip(span, span[1:]))
    if widest > max_gap_seconds:
        return GateSignal(
            name, None, VM_MEMORY_MAX_PCT, "not_measured",
            f"a {widest:.0f}s gap between host samples inside the window could hide the peak",
        )
    peak = max(100.0 * (s.mem_total_bytes - s.mem_available_bytes) / s.mem_total_bytes for s in span)
    return _judge(name, peak, VM_MEMORY_MAX_PCT)


def mtta_seconds(usage: list[EpisodeTokenUsage], counted_episode_ids: set[str]) -> list[float] | str:
    """Per-episode MTTA, defined as t_record minus the time the webhook was received: the usage
    line's t_start is taken when the webhook hands the episode to its thread and t_end after the
    TriageRecord is emitted. `t_alarme` is NOT used: it is Alertmanager's startsAt, and the
    group_wait inside t_record - t_alarme would inflate MTTA and loosen the gate. Only the
    records the harvest counted as episodes enter; a counted record without its usage line makes
    the whole list a reason, never a median over whoever happened to be logged."""
    ids = [u.episode_id for u in usage]
    repeated = sorted({i for i in ids if ids.count(i) > 1 and i in counted_episode_ids})
    if repeated:
        return f"episode_id repeated in the usage log: {', '.join(repeated)}"
    by_id = {u.episode_id: u for u in usage}
    lacking = sorted(counted_episode_ids - by_id.keys())
    if lacking:
        return f"{len(lacking)} counted episode(s) have no usage line: {', '.join(lacking[:5])}"
    return [by_id[i].t_end - by_id[i].t_start for i in sorted(counted_episode_ids)]


def queue_signal(
    samples: list[VllmQueueSample],
    mtta: list[float] | str,
    start: float,
    end: float,
    *,
    max_gap_seconds: float = 60.0,
) -> tuple[GateSignal, float | None]:
    """TTFT p95 ceiling over the window against 10% of the median MTTA. Returns the signal and
    the median it was judged against (None when there is no median)."""
    name = "queue_p95_ceiling_seconds"
    if isinstance(mtta, str):
        return GateSignal(name, None, None, "not_measured", f"MTTA: {mtta}"), None
    if not mtta:
        return GateSignal(name, None, None, "not_measured", "MTTA: no counted episode in the window"), None
    median = statistics.median(mtta)
    limit = QUEUE_MAX_FRACTION_OF_MTTA * median
    ceiling = ttft_p95_ceiling(samples, start, end, max_gap_seconds=max_gap_seconds)
    if ceiling.p95_ceiling_seconds is None:
        return GateSignal(name, None, limit, "not_measured", ceiling.reason), median
    return _judge(name, ceiling.p95_ceiling_seconds, limit), median


@dataclass
class ContentionGate:
    """`passed` is True only when every signal is "pass"; "not_measured" fails the gate exactly
    like "over" does, and the two stay distinguishable in `signals`."""

    start: float
    end: float
    passed: bool
    signals: list[GateSignal]
    mtta_median_seconds: float | None
    mtta_episodes: int
    # True only when the flag was USED: the steal counter was unproven and passed on the flag.
    # A run that passed the flag on a VM that proves accounting reads False.
    accepted_unproven_steal: bool = False


def contention_gate(
    queue_samples: list[VllmQueueSample],
    host_samples: list[HostSample],
    mtta: list[float] | str,
    start: float,
    end: float,
    *,
    max_gap_seconds: float = 60.0,
    accept_unproven_steal: bool = False,
) -> ContentionGate:
    if end <= start:
        raise ValueError(f"window must have end > start, got [{start!r}, {end!r}]")
    queue, median = queue_signal(queue_samples, mtta, start, end, max_gap_seconds=max_gap_seconds)
    steal = steal_signal(host_samples, start, end, max_gap_seconds=max_gap_seconds,
                         accept_unproven_steal=accept_unproven_steal)
    signals = [
        steal,
        queue,
        memory_signal(host_samples, start, end, max_gap_seconds=max_gap_seconds),
    ]
    return ContentionGate(
        start, end, all(s.status == "pass" for s in signals), signals, median,
        0 if isinstance(mtta, str) else len(mtta),
        steal.status == "pass" and (steal.reason or "").startswith(STEAL_UNPROVEN_REASON),
    )


# TriageRecord interface this aggregator joins against: `episode_id` (harness/triage.py),
# `t_alarme`/`t_record` (ISO-8601 strings, the episode's window). The orchestrator that emits
# TriageRecords to disk lives elsewhere; this module consumes a directory of TriageRecord-shaped
# JSON files if one is given, and degrades honestly (empty join, stated reason) when the
# directory is absent or empty, the same boundary discipline entrypoint.py uses for
# `episode_runner`.
FOOTPRINT_CSV_FIELDS = [
    "episode_id",
    "env",
    "window_start",
    "window_end",
    "pod_cpu_cores_mean",
    "pod_memory_bytes_mean",
    "pod_samples_in_window",
    "pod_coverage_reason",
    "vllm_gpu_vram_used_mb_mean",
    "vllm_gpu_utilization_pct_mean",
    "vllm_gpu_power_watts_mean",
    "vllm_cgroup_cpu_seconds_delta",
    "vllm_cgroup_memory_bytes_mean",
    "vllm_samples_in_window",
    "vllm_coverage_reason",
]


@dataclass
class EpisodeFootprint:
    """One row per TriageRecord episode: pod + vLLM footprint, meaned over samples whose
    timestamp falls inside the episode's [window_start, window_end]. Joined by episode_id
    (`harness/triage.py` TriageRecord field), never by config_name/scenario_id -- those are
    RunRecord's axis, not this one's.

    An episode with zero collected samples in its window (gap in the scrape/log cadence, or a
    window shorter than the collector's interval) gets `*_mean=None` and `*_coverage_reason`
    set -- never a mean over zero samples silently reading as 0.0.

    `vllm_cgroup_cpu_seconds_delta` is deliberately NOT a mean like its siblings:
    `cgroup_cpu_seconds` (VllmFootprintSample) is `cpu.stat`'s `usage_usec`, a counter
    monotonically increasing since the vLLM systemd unit started, not an instantaneous rate.
    Averaging it would read as "total CPU-seconds vLLM has ever consumed" and land at nearly
    the same number for every episode in a round -- the CPU cost attributable to THIS episode
    is the counter's delta across the window (last collected sample minus first), the same
    quantity `pod_cpu_cores_mean` gets for free from `rate()` on the Prometheus side.
    """

    episode_id: str
    env: str
    window_start: float
    window_end: float
    pod_cpu_cores_mean: float | None
    pod_memory_bytes_mean: float | None
    pod_samples_in_window: int
    pod_coverage_reason: str | None
    vllm_gpu_vram_used_mb_mean: float | None
    vllm_gpu_utilization_pct_mean: float | None
    vllm_gpu_power_watts_mean: float | None
    vllm_cgroup_cpu_seconds_delta: float | None
    vllm_cgroup_memory_bytes_mean: float | None
    vllm_samples_in_window: int
    vllm_coverage_reason: str | None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _counter_delta(samples: list[VllmFootprintSample]) -> float | None:
    """Last minus first `cgroup_cpu_seconds` across chronologically-in-window samples, for a
    monotonic counter (see EpisodeFootprint's docstring). None (not 0.0) with fewer than 2
    values with the field populated: a single sample cannot measure a delta, and a negative
    delta (counter reset, e.g. the vLLM unit restarted mid-window) is likewise unmeasurable, not
    a real negative cost -- both report None rather than a fabricated number.
    """
    ordered = sorted(
        (s for s in samples if s.cgroup_cpu_seconds is not None), key=lambda s: s.timestamp
    )
    if len(ordered) < 2:
        return None
    delta = ordered[-1].cgroup_cpu_seconds - ordered[0].cgroup_cpu_seconds
    return delta if delta >= 0 else None


def join_episode_footprint(
    *,
    episode_id: str,
    env: str,
    window_start: float,
    window_end: float,
    pod_samples: list[PodFootprintSample],
    vllm_samples: list[VllmFootprintSample],
) -> EpisodeFootprint:
    """Fold pod + vLLM samples into one EpisodeFootprint row for the [window_start, window_end]
    window of a single episode (contract: window_start <= window_end, both epoch seconds).
    Only `collected=True` samples inside the window feed the means; an uncollected tick is
    coverage the honesty rule already refused to fabricate, and averaging it in would just
    re-fabricate it one level up.
    """
    pod_in_window = [
        s for s in pod_samples if s.collected and window_start <= s.timestamp <= window_end
    ]
    vllm_in_window = [
        s for s in vllm_samples if s.collected and window_start <= s.timestamp <= window_end
    ]

    pod_reason = None if pod_in_window else "no collected pod sample in [window_start, window_end]"
    vllm_reason = None if vllm_in_window else "no collected vllm sample in [window_start, window_end]"

    return EpisodeFootprint(
        episode_id=episode_id,
        env=env,
        window_start=window_start,
        window_end=window_end,
        pod_cpu_cores_mean=_mean([s.cpu_cores for s in pod_in_window if s.cpu_cores is not None]),
        pod_memory_bytes_mean=_mean([s.memory_bytes for s in pod_in_window if s.memory_bytes is not None]),
        pod_samples_in_window=len(pod_in_window),
        pod_coverage_reason=pod_reason,
        vllm_gpu_vram_used_mb_mean=_mean(
            [s.gpu_vram_used_mb for s in vllm_in_window if s.gpu_vram_used_mb is not None]
        ),
        vllm_gpu_utilization_pct_mean=_mean(
            [s.gpu_utilization_pct for s in vllm_in_window if s.gpu_utilization_pct is not None]
        ),
        vllm_gpu_power_watts_mean=_mean(
            [s.gpu_power_watts for s in vllm_in_window if s.gpu_power_watts is not None]
        ),
        vllm_cgroup_cpu_seconds_delta=_counter_delta(vllm_in_window),
        vllm_cgroup_memory_bytes_mean=_mean(
            [s.cgroup_memory_bytes for s in vllm_in_window if s.cgroup_memory_bytes is not None]
        ),
        vllm_samples_in_window=len(vllm_in_window),
        vllm_coverage_reason=vllm_reason,
    )


def join_all_episode_footprints(
    *,
    env: str,
    triage_records: list[dict[str, Any]],
    pod_samples: list[PodFootprintSample],
    vllm_samples: list[VllmFootprintSample],
) -> list[EpisodeFootprint]:
    """join_episode_footprint for every TriageRecord in `triage_records` (dicts shaped per
    harness/triage.py's TRIAGE_RECORD_SCHEMA: `episode_id`, `t_alarme`, `t_record`, both
    ISO-8601). Records missing `episode_id`, whose timestamps fail to parse, or whose
    `t_record` precedes `t_alarme` are skipped, not fabricated into a window -- same honesty
    rule as the other two cases, applied consistently rather than silently swapping the two
    timestamps to make an inverted window "work". `load_triage_records` never validates
    against TRIAGE_RECORD_SCHEMA, so a malformed file here is realistic input, not a
    hypothetical.
    """
    from datetime import datetime

    rows: list[EpisodeFootprint] = []
    for record in triage_records:
        try:
            episode_id = record["episode_id"]
            start = datetime.fromisoformat(record["t_alarme"]).timestamp()
            end = datetime.fromisoformat(record["t_record"]).timestamp()
        except (KeyError, ValueError):
            continue
        if end < start:
            continue
        rows.append(
            join_episode_footprint(
                episode_id=episode_id,
                env=env,
                window_start=start,
                window_end=end,
                pod_samples=pod_samples,
                vllm_samples=vllm_samples,
            )
        )
    return rows


def write_episode_footprint_csv(records: list[EpisodeFootprint], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FOOTPRINT_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return path


@dataclass
class ContinuousCostSummary:
    """USD per triage = VM hours x list price / number of episodes.
    A run-level number, not per-episode: the VM is billed by the hour regardless of how many
    episodes landed in it. `episode_count=0` (no TriageRecords joined, e.g. the orchestrator
    boundary above) yields `usd_per_triage=None` with a reason -- dividing by zero is exactly
    the kind of fabricated number the honesty rule exists to block.
    """

    env: str
    vm_hours: float
    price_usd_per_hour: float
    episode_count: int
    usd_total: float
    usd_per_triage: float | None
    usd_per_triage_reason: str | None


def compute_continuous_cost_summary(
    *,
    env: str,
    vm_hours: float,
    episode_count: int,
    price_usd_per_hour: float = PRICE_CONTINUOUS_ENV_USD_PER_HOUR,
) -> ContinuousCostSummary:
    if vm_hours < 0:
        raise ValueError(f"vm_hours must be >= 0, got {vm_hours!r}")
    if episode_count < 0:
        raise ValueError(f"episode_count must be >= 0, got {episode_count!r}")

    usd_total = vm_hours * price_usd_per_hour
    if episode_count == 0:
        return ContinuousCostSummary(
            env=env,
            vm_hours=vm_hours,
            price_usd_per_hour=price_usd_per_hour,
            episode_count=0,
            usd_total=usd_total,
            usd_per_triage=None,
            usd_per_triage_reason="episode_count is 0 (no TriageRecords joined for this window)",
        )
    return ContinuousCostSummary(
        env=env,
        vm_hours=vm_hours,
        price_usd_per_hour=price_usd_per_hour,
        episode_count=episode_count,
        usd_total=usd_total,
        usd_per_triage=usd_total / episode_count,
        usd_per_triage_reason=None,
    )


def write_continuous_cost_summary_json(summary: ContinuousCostSummary, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return path


def collect_records(results_root: str | Path) -> list[RunRecord]:
    """Load every RunRecord under a results root: legacy records at the root plus one
    directory per env (env-descriptor.json is the env declaration, not a record). Single
    source for run_matrix.py's matrix.csv rebuild and scenarios/backfill_env.py."""
    root = Path(results_root)
    paths = sorted(root.glob("*.json")) + sorted(
        path for path in root.glob("*/*.json") if path.name != "env-descriptor.json"
    )
    return [RunRecord.from_json(path.read_text(encoding="utf-8")) for path in paths]
