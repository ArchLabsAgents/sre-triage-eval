"""Pod-side footprint collector for the continuous scenario. Queries the kube-prometheus-stack
installed by setup_cluster.sh (KPS_INSTALL=1) for the sslm-agent pod's cAdvisor/kubelet cgroup
metrics (CPU and memory) and appends one PodFootprintSample per tick to a JSONL file.

Deliberately talks to Prometheus's HTTP query API (`/api/v1/query`), not `kubectl top`: `kubectl
top` has no history, so it cannot produce the time series this scenario needs to join against
TriageRecords. Uses only the standard library (urllib), same reasoning and same convention as
adapters/vllm_model_client.py: this script runs on a dev machine or the lab VM, never
inside the cluster, and tools/triage-eval/requirements.txt does not carry an HTTP client dep.

Honesty rule: a tick where the query errors, times out, or returns no series is written as
`collected=False` with a `reason`, never as 0.0. See harness/metrics.py's PodFootprintSample.

Usage:
    python scenarios/collect_pod_footprint.py \\
        --prometheus-url http://localhost:9090 --namespace sslm-agent \\
        --out scenarios/results/hyperstack-a6000-continuous/pod-footprint.jsonl \\
        --interval-seconds 30
    # a single tick, for scripts/tests (does not enter the loop):
    python scenarios/collect_pod_footprint.py ... --once
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from harness.metrics import PodFootprintSample, write_footprint_sample_jsonl  # noqa: E402

DEFAULT_PROMETHEUS_URL = "http://localhost:9090"
DEFAULT_NAMESPACE = "sslm-agent"
DEFAULT_INTERVAL_SECONDS = 30.0
_QUERY_TIMEOUT_SECONDS = 10.0

# container!="" and container!="POD" exclude the aggregate series of the sandbox/pause container
# that cAdvisor also exposes per pod: without the filter the pod's CPU would silently double.
_CPU_QUERY_TEMPLATE = (
    'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}",'
    'container!="",container!="POD"}}[2m]))'
)
_MEMORY_QUERY_TEMPLATE = (
    'sum(container_memory_working_set_bytes{{namespace="{namespace}",'
    'container!="",container!="POD"}})'
)


class PrometheusQueryError(Exception):
    """Transport failure or invalid response from the Prometheus API. Never becomes a sample with
    an invented value: the caller writes collected=False with this exception's text as reason."""


def _query_instant(prometheus_url: str, promql: str, *, timeout: float = _QUERY_TIMEOUT_SECONDS) -> float | None:
    """Runs an instant query against the Prometheus API and returns the scalar value (or None when
    the query returned no series: missing data, not a transport error)."""
    url = f"{prometheus_url.rstrip('/')}/api/v1/query?{urllib.parse.urlencode({'query': promql})}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - trusted lab endpoint
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PrometheusQueryError(f"transport failed querying {prometheus_url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PrometheusQueryError(f"non-JSON response from {prometheus_url}: {exc}") from exc

    if body.get("status") != "success":
        raise PrometheusQueryError(f"Prometheus returned status={body.get('status')!r}: {body}")

    result = body.get("data", {}).get("result", [])
    if not result:
        return None  # no series: the pod does not exist yet, or no scrape ran yet. Not an error.

    try:
        return float(result[0]["value"][1])
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise PrometheusQueryError(f"unexpected result format: {result!r} ({exc})") from exc


def collect_one_tick(prometheus_url: str, namespace: str, *, now: float | None = None) -> PodFootprintSample:
    """One tick: queries CPU and memory, returns a PodFootprintSample. A failure of EITHER query
    marks the whole tick as not collected: a sample with only half its fields filled would be as
    misleading as an invented zero."""
    timestamp = now if now is not None else time.time()
    try:
        cpu_cores = _query_instant(prometheus_url, _CPU_QUERY_TEMPLATE.format(namespace=namespace))
        memory_bytes = _query_instant(prometheus_url, _MEMORY_QUERY_TEMPLATE.format(namespace=namespace))
    except PrometheusQueryError as exc:
        return PodFootprintSample(timestamp=timestamp, collected=False, reason=str(exc))

    if cpu_cores is None or memory_bytes is None:
        # `or`, not `and`: the two queries fail independently (the CPU query is a rate() over a
        # 2m window and needs two minutes of scrape history, while the memory gauge is available
        # as soon as the pod is scheduled; realistic to have one without the other right after
        # startup). A tick with only one of the two fields is exactly as misleading as a
        # fabricated zero, so either query returning no series marks the whole tick uncollected.
        missing = "cpu" if cpu_cores is None and memory_bytes is not None else (
            "memory" if memory_bytes is None and cpu_cores is not None else "cpu and memory"
        )
        return PodFootprintSample(
            timestamp=timestamp,
            collected=False,
            reason=f"no series for namespace={namespace!r} ({missing}; pod not scheduled yet, or scrape has not run)",
        )

    return PodFootprintSample(
        timestamp=timestamp,
        collected=True,
        cpu_cores=cpu_cores,
        memory_bytes=memory_bytes,
    )


def run_loop(
    prometheus_url: str,
    namespace: str,
    out_path: str | Path,
    *,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    once: bool = False,
    _sleep=time.sleep,
) -> None:
    while True:
        sample = collect_one_tick(prometheus_url, namespace)
        write_footprint_sample_jsonl(out_path, sample)
        status = "ok" if sample.collected else f"MISS ({sample.reason})"
        print(f"[pod-footprint] {sample.timestamp:.0f} {status}", flush=True)
        if once:
            return
        _sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prometheus-url", default=DEFAULT_PROMETHEUS_URL)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--out", required=True, help="output JSONL path (append)")
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true", help="a single tick, without entering the loop (debug/test)")
    args = parser.parse_args()

    run_loop(
        args.prometheus_url,
        args.namespace,
        args.out,
        interval_seconds=args.interval_seconds,
        once=args.once,
    )


if __name__ == "__main__":
    main()
