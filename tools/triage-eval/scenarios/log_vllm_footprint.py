"""vLLM host-side footprint logger for the continuous scenario. Runs on the lab VM, OUTSIDE the
kind cluster (vLLM runs as a systemd unit on the host): loops `nvidia-smi --query-gpu` plus a read
of the vLLM container's own cgroup, and appends one VllmFootprintSample per tick to a JSONL file.

vllm.service's ExecStart is `docker run --name sslm-vllm ...` in the foreground: systemd's own
cgroup for that unit (system.slice/vllm.service) holds the `docker` CLI client, which the
daemon detaches from as soon as the container starts. Reading cpu.stat/memory.current there
measures the idle client, never the GPU-bound container it launched (found live on 2026-09-29).
So the logger resolves the container's own cgroup via `docker inspect` (the container's PID, then
/proc/<pid>/cgroup) instead of trusting the unit's own cgroup path; see
`_resolve_container_cgroup_path` below.

Deliberately a separate script and a separate sample type from collect_pod_footprint.py: the pod
footprint never mixes with the model footprint (separate series, separate columns). Stdlib-only
(subprocess for nvidia-smi and docker inspect, plain file reads for the cgroup), same convention
as the rest of this repo's HTTP/process calls.

Honesty rule: a tick where nvidia-smi fails, docker inspect fails, or the cgroup files are
unreadable is written as `collected=False` with a `reason`, never as 0.0. See harness/metrics.py's
VllmFootprintSample.

Usage (on the VM host, outside the cluster):
    python scenarios/log_vllm_footprint.py \\
        --container-name sslm-vllm --cgroup-root /sys/fs/cgroup \\
        --out scenarios/results/hyperstack-a6000-continuous/vllm-footprint.jsonl \\
        --interval-seconds 30
    # a single tick, for scripts/tests (does not enter the loop):
    python scenarios/log_vllm_footprint.py ... --once
    # vLLM queue (contention gate), a separate series on the same tick:
    python scenarios/log_vllm_footprint.py ... \\
        --metrics-url http://127.0.0.1:8000/metrics --metrics-out .../vllm-queue.jsonl
    # VM steal and memory (contention gate), another separate series on the same tick:
    python scenarios/log_vllm_footprint.py ... --host-out .../host.jsonl
"""

from __future__ import annotations

import argparse
import http.client
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from harness.metrics import (  # noqa: E402
    HostSample,
    VllmFootprintSample,
    VllmQueueSample,
    host_sample_from_proc,
    queue_sample_from_metrics,
    write_footprint_sample_jsonl,
)

DEFAULT_CONTAINER_NAME = "sslm-vllm"  # docker run --name in vllm.service's ExecStart
DEFAULT_CGROUP_ROOT = "/sys/fs/cgroup"
DEFAULT_INTERVAL_SECONDS = 30.0
_NVIDIA_SMI_TIMEOUT_SECONDS = 10.0
_NVIDIA_SMI_QUERY = "memory.used,utilization.gpu,power.draw"
_METRICS_TIMEOUT_SECONDS = 5.0
_DOCKER_INSPECT_TIMEOUT_SECONDS = 10.0


class NvidiaSmiError(Exception):
    """nvidia-smi missing, no visible GPU, or output in an unexpected format. Never becomes a sample
    with an invented value: the caller writes collected=False with this exception's text."""


class DockerInspectError(Exception):
    """docker missing, the sslm-vllm container does not exist or is not running, or
    /proc/<pid>/cgroup is unreadable or has no unified v2 line. Same rule: not collected, never
    zero."""


class CgroupReadError(Exception):
    """A file of the resolved cgroup (the container's, not the systemd unit's) is unreadable
    (wrong path, or the host is not cgroup v2). Same rule: not collected, never zero."""


def _run_nvidia_smi(*, timeout: float = _NVIDIA_SMI_TIMEOUT_SECONDS, _runner=subprocess.run) -> tuple[float, float, float]:
    """Returns (vram_used_mb, utilization_pct, power_watts). Raises NvidiaSmiError on any
    failure (missing binary, exit != 0, timeout, empty or malformed output)."""
    cmd = [
        "nvidia-smi",
        f"--query-gpu={_NVIDIA_SMI_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = _runner(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise NvidiaSmiError("nvidia-smi not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise NvidiaSmiError(f"nvidia-smi did not answer within {timeout}s") from exc

    if proc.returncode != 0:
        raise NvidiaSmiError(f"nvidia-smi exited with code {proc.returncode}: {proc.stderr.strip()}")

    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        raise NvidiaSmiError(f"unexpected nvidia-smi output: {proc.stdout!r}")

    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise NvidiaSmiError(f"could not parse numbers from the output: {parts!r} ({exc})") from exc


def _read_cgroup_footprint(cgroup_path: str | Path) -> tuple[float, float]:
    """Returns (cpu_seconds, memory_bytes) of the cgroup at `cgroup_path`, reading cpu.stat and
    memory.current (cgroup v2). Raises CgroupReadError if either file is missing or unparseable."""
    base = Path(cgroup_path)
    cpu_stat_path = base / "cpu.stat"
    memory_current_path = base / "memory.current"

    try:
        cpu_stat_text = cpu_stat_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CgroupReadError(f"could not read {cpu_stat_path}: {exc}") from exc

    usage_usec = None
    for line in cpu_stat_text.splitlines():
        key, _, value = line.partition(" ")
        if key == "usage_usec":
            usage_usec = value.strip()
            break
    if usage_usec is None:
        raise CgroupReadError(f"'usage_usec' missing in {cpu_stat_path}")

    try:
        memory_bytes_text = memory_current_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CgroupReadError(f"could not read {memory_current_path}: {exc}") from exc

    try:
        cpu_seconds = float(usage_usec) / 1_000_000
        memory_bytes = float(memory_bytes_text)
    except ValueError as exc:
        raise CgroupReadError(f"non-numeric value in cpu.stat/memory.current: {exc}") from exc

    return cpu_seconds, memory_bytes


def _resolve_container_cgroup_path(
    container_name: str,
    *,
    cgroup_root: str | Path = DEFAULT_CGROUP_ROOT,
    _runner=subprocess.run,
    _proc_root: str | Path = "/proc",
) -> Path:
    """Returns the cgroup path of the CONTAINER (not the systemd unit): `docker inspect` for the
    PID of the container's root process, then the unified line ("0::<path>") of
    /proc/<pid>/cgroup joined to cgroup_root. Works with either docker cgroup driver (systemd or
    cgroupfs) because it reads the resolved path straight from the kernel instead of guessing the
    docker-<id>.scope naming convention. Raises DockerInspectError on any failure."""
    cmd = ["docker", "inspect", "--format", "{{.State.Pid}}", container_name]
    try:
        proc = _runner(
            cmd, capture_output=True, text=True, timeout=_DOCKER_INSPECT_TIMEOUT_SECONDS, check=False
        )
    except FileNotFoundError as exc:
        raise DockerInspectError("docker not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerInspectError(
            f"docker inspect did not answer within {_DOCKER_INSPECT_TIMEOUT_SECONDS}s"
        ) from exc

    if proc.returncode != 0:
        raise DockerInspectError(
            f"docker inspect {container_name} exited with code {proc.returncode}: {proc.stderr.strip()}"
        )

    pid_text = proc.stdout.strip()
    try:
        pid = int(pid_text)
    except ValueError as exc:
        raise DockerInspectError(f"non-numeric PID from docker inspect: {pid_text!r}") from exc
    if pid <= 0:
        raise DockerInspectError(f"container {container_name!r} has no PID (not running?): {pid_text!r}")

    cgroup_file = Path(_proc_root) / str(pid) / "cgroup"
    try:
        cgroup_text = cgroup_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise DockerInspectError(f"could not read {cgroup_file}: {exc}") from exc

    for line in cgroup_text.splitlines():
        hierarchy_id, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy_id == "0" and controllers == "":
            return Path(cgroup_root) / path.lstrip("/")
    raise DockerInspectError(f"no cgroup v2 line ('0::') in {cgroup_file}: {cgroup_text!r}")


def collect_one_tick(
    container_name: str,
    *,
    cgroup_root: str | Path = DEFAULT_CGROUP_ROOT,
    now: float | None = None,
    _nvidia_smi=_run_nvidia_smi,
    _resolve_cgroup=_resolve_container_cgroup_path,
    _cgroup=_read_cgroup_footprint,
) -> VllmFootprintSample:
    """One tick: nvidia-smi + the container's cgroup, returns a VllmFootprintSample. The two
    sources fail independently (GPU without a driver but a readable cgroup, or the reverse): the
    tick only counts as collected=True when BOTH answer, for the same reason as the pod footprint:
    a partial sample would be as misleading as an invented zero."""
    timestamp = now if now is not None else time.time()
    reasons: list[str] = []

    gpu_values = None
    try:
        gpu_values = _nvidia_smi()
    except NvidiaSmiError as exc:
        reasons.append(str(exc))

    cgroup_values = None
    try:
        cgroup_path = _resolve_cgroup(container_name, cgroup_root=cgroup_root)
        cgroup_values = _cgroup(cgroup_path)
    except (DockerInspectError, CgroupReadError) as exc:
        reasons.append(str(exc))

    if gpu_values is None or cgroup_values is None:
        return VllmFootprintSample(timestamp=timestamp, collected=False, reason="; ".join(reasons))

    vram_used_mb, utilization_pct, power_watts = gpu_values
    cpu_seconds, memory_bytes = cgroup_values
    return VllmFootprintSample(
        timestamp=timestamp,
        collected=True,
        gpu_vram_used_mb=vram_used_mb,
        gpu_utilization_pct=utilization_pct,
        gpu_power_watts=power_watts,
        cgroup_cpu_seconds=cpu_seconds,
        cgroup_memory_bytes=memory_bytes,
    )


def _fetch_metrics(url: str, *, timeout: float = _METRICS_TIMEOUT_SECONDS) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def scrape_vllm_metrics(url: str, *, now: float | None = None, _fetch=_fetch_metrics) -> VllmQueueSample:
    """One scrape of vLLM's /metrics. A network failure or unreadable body becomes
    collected=False with a reason, never an empty queue."""
    timestamp = now if now is not None else time.time()
    try:
        text = _fetch(url)
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
        return VllmQueueSample(timestamp=timestamp, collected=False, reason=f"GET {url} failed: {exc}")
    return queue_sample_from_metrics(text, timestamp=timestamp)


def read_host_sample(
    *, now: float | None = None, proc_root: str | Path = "/proc"
) -> HostSample:
    """One tick of the host's /proc/stat + /proc/meminfo. An unreadable file becomes
    collected=False with a reason, never
    zero steal or free memory."""
    timestamp = now if now is not None else time.time()
    root = Path(proc_root)
    try:
        stat_text = (root / "stat").read_text(encoding="utf-8")
        meminfo_text = (root / "meminfo").read_text(encoding="utf-8")
    except OSError as exc:
        return HostSample(timestamp=timestamp, collected=False, reason=f"could not read {root}: {exc}")
    return host_sample_from_proc(stat_text, meminfo_text, timestamp=timestamp)


def run_loop(
    container_name: str,
    out_path: str | Path,
    *,
    cgroup_root: str | Path = DEFAULT_CGROUP_ROOT,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    once: bool = False,
    metrics_url: str | None = None,
    metrics_out: str | Path | None = None,
    host_out: str | Path | None = None,
    _sleep=time.sleep,
    _nvidia_smi=_run_nvidia_smi,
    _resolve_cgroup=_resolve_container_cgroup_path,
    _cgroup=_read_cgroup_footprint,
    _fetch=_fetch_metrics,
    _host=read_host_sample,
) -> None:
    if (metrics_url is None) != (metrics_out is None):
        raise ValueError("--metrics-url and --metrics-out go together")
    while True:
        sample = collect_one_tick(
            container_name,
            cgroup_root=cgroup_root,
            _nvidia_smi=_nvidia_smi,
            _resolve_cgroup=_resolve_cgroup,
            _cgroup=_cgroup,
        )
        write_footprint_sample_jsonl(out_path, sample)
        status = "ok" if sample.collected else f"MISS ({sample.reason})"
        print(f"[vllm-footprint] {sample.timestamp:.0f} {status}", flush=True)
        if metrics_url is not None:
            queue = scrape_vllm_metrics(metrics_url, _fetch=_fetch)
            write_footprint_sample_jsonl(metrics_out, queue)
            status = "ok" if queue.collected else f"MISS ({queue.reason})"
            print(f"[vllm-queue] {queue.timestamp:.0f} {status}", flush=True)
        if host_out is not None:
            host = _host()
            write_footprint_sample_jsonl(host_out, host)
            status = "ok" if host.collected else f"MISS ({host.reason})"
            print(f"[host] {host.timestamp:.0f} {status}", flush=True)
        if once:
            return
        _sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--container-name",
        default=DEFAULT_CONTAINER_NAME,
        help="docker container name of vLLM (the --name in vllm.service's ExecStart)",
    )
    parser.add_argument(
        "--cgroup-root",
        default=DEFAULT_CGROUP_ROOT,
        help="cgroupfs v2 root on the host, where the container's resolved path is anchored",
    )
    parser.add_argument("--out", required=True, help="output JSONL path (append)")
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true", help="a single tick, without entering the loop (debug/test)")
    parser.add_argument("--metrics-url", help="vLLM /metrics on the host (queue series, contention gate)")
    parser.add_argument("--metrics-out", help="JSONL of the queue series (append); requires --metrics-url")
    parser.add_argument("--host-out", help="JSONL of VM steal + memory (append), contention gate")
    args = parser.parse_args()
    if (args.metrics_url is None) != (args.metrics_out is None):
        parser.error("--metrics-url and --metrics-out go together")

    run_loop(
        args.container_name,
        args.out,
        cgroup_root=args.cgroup_root,
        interval_seconds=args.interval_seconds,
        once=args.once,
        metrics_url=args.metrics_url,
        metrics_out=args.metrics_out,
        host_out=args.host_out,
    )


if __name__ == "__main__":
    main()
