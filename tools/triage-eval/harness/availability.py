"""Agent availability: 1 - downtime / window, from a heartbeat series.

Definition: downtime is a
heartbeat missing for more than 2 intervals. The agent has no heartbeat poller of its own (the
entrypoint's `/healthz` is only probed by the kubelet), so the heartbeat read here is the one
Prometheus already holds: `kube_pod_status_ready{condition="true"}` of the agent pod, which is the
kubelet's readiness probe of `/healthz` every 10s, scraped by kube-state-metrics. Pure: it takes
the samples, never talks to a cluster, and an unreadable series is "not measured", never a pass.

Honest limits: Prometheus' 5 min staleness lookback can hide a gap shorter than that when the pod
disappears without a stale marker, and a pod replaced inside one step is not seen. A pod that was
down for the whole window has no `1` sample and comes out not measured, not 0%.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

STEP_SECONDS = 30.0
# "heartbeat ausente > 2 intervalos": a run of missing/not-ready steps counts as downtime only
# when it is longer than this many steps. Shorter runs are scrape jitter, not an outage.
TOLERATED_MISSING_STEPS = 2
READY_QUERY = 'kube_pod_status_ready{namespace="sslm-agent",condition="true"}'


@dataclass(frozen=True)
class Availability:
    measured: bool
    availability: float | None = None
    downtime_seconds: float | None = None
    window_seconds: float | None = None
    outages: int | None = None
    reason: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def not_measured(reason: str) -> Availability:
    return Availability(measured=False, reason=reason)


def compute_availability(
    series: Iterable[Sequence[tuple[float, float]]],
    start: float,
    end: float,
    *,
    step: float = STEP_SECONDS,
    tolerated_missing_steps: int = TOLERATED_MISSING_STEPS,
) -> Availability:
    """`series`: one list of (epoch, value) per Prometheus series, evaluated on the grid
    start + k*step (what query_range returns). A step is up when any series reads 1 there; a
    missing sample or a 0 is not up."""
    if end <= start:
        return not_measured(f"empty window ({start} .. {end})")
    steps = int((end - start) // step) + 1
    up = [False] * steps
    seen_up = False
    for samples in series:
        for ts, value in samples:
            idx = round((ts - start) / step)
            if 0 <= idx < steps and value == 1:
                up[idx] = True
                seen_up = True
    if not seen_up:
        return not_measured("heartbeat series has no ready sample in the window")

    downtime_steps = 0
    outages = 0
    run = 0
    for is_up in up + [True]:  # sentinel closes a run that reaches the end of the window
        if is_up:
            if run > tolerated_missing_steps:
                downtime_steps += run
                outages += 1
            run = 0
        else:
            run += 1
    window = steps * step
    downtime = downtime_steps * step
    return Availability(
        measured=True,
        availability=1.0 - downtime / window,
        downtime_seconds=downtime,
        window_seconds=window,
        outages=outages,
    )
