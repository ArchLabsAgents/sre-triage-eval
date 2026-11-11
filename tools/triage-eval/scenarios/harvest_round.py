"""Harvest a round's TriageRecords from the agent pod and join them with the round journal.

Runs on the lab VM right after scenarios/run_continuous_admission.py exits (the launcher chains
it), while the kind cluster still exists. Writes, next to the driver's --out:

  <out>.triage.jsonl   the pod's SSLM_TRIAGE_LOG, byte for byte
  <out>.usage.jsonl    per-episode token usage (SSLM_TRIAGE_LOG + ".usage.jsonl"), byte for byte
  <out>.episodes.json  harness.episode_bridge output (only when the harvest is complete)
  <out>.harvest.json   what was measured, and why anything was not (agent availability included)

The emptyDir behind SSLM_TRIAGE_LOG survives a container restart but not a pod replacement. A
pod created after the round started holds only part of the records, and joining that would turn
every lost record into a false negative, so the join is refused and the reason written instead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from harness import availability as availability_mod  # noqa: E402
from harness import episode_bridge  # noqa: E402

NAMESPACE = "sslm-agent"
POD_SELECTOR = "app.kubernetes.io/name=sslm-agent"
# Exit codes of the in-pod read, so an empty round and an unconfigured sink never look alike.
NO_RECORD_YET = 3
SINK_NOT_CONFIGURED = 4
READ_SCRIPT = (
    'if [ -z "${SSLM_TRIAGE_LOG:-}" ]; then exit %d; fi; '
    'if [ -e "$SSLM_TRIAGE_LOG" ]; then cat "$SSLM_TRIAGE_LOG"; else exit %d; fi'
) % (SINK_NOT_CONFIGURED, NO_RECORD_YET)
# Same derivation as harness.episode_runner.usage_log_path.
USAGE_READ_SCRIPT = READ_SCRIPT.replace('"$SSLM_TRIAGE_LOG"', '"$SSLM_TRIAGE_LOG.usage.jsonl"')

Runner = Callable[[list[str]], subprocess.CompletedProcess]
# (query, start, end, step) -> Prometheus query_range "result" list; raises on any failure.
RangeQuery = Callable[[str, float, float, float], list]


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    # A hung or missing kubectl becomes a failed call with the reason in stderr, so it reaches
    # refuse() and the report still gets written instead of a traceback.
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return subprocess.CompletedProcess(argv, 1, "", f"{type(exc).__name__}: {exc}")


def _round_start(journal: Path) -> float | None:
    for line in journal.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return float(json.loads(line)["t"])
    return None


def _round_end(journal: Path) -> float | None:
    """`t` of the last journal line: the window availability is read over, measured not chosen."""
    last = None
    for line in journal.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = float(json.loads(line)["t"])
    return last


def _prometheus_range(base_url: str) -> RangeQuery:
    def query(promql: str, start: float, end: float, step: float) -> list:
        qs = urllib.parse.urlencode({"query": promql, "start": start, "end": end, "step": step})
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/v1/query_range?{qs}", timeout=60) as resp:
            payload = json.load(resp)
        if payload.get("status") != "success":
            raise ValueError(f"prometheus answered {payload.get('status')!r}")
        return payload["data"]["result"]

    return query


def measure_availability(journal: Path, query_range: RangeQuery | None) -> availability_mod.Availability:
    """The agent's heartbeat series over the journal's own window. Never raises: any failure is a
    not-measured Availability with the reason, because availability must not block the join."""
    if query_range is None:
        return availability_mod.not_measured("no --prometheus-url given; heartbeat not read")
    try:
        start, end = _round_start(journal), _round_end(journal)
        if start is None or end is None:
            return availability_mod.not_measured(f"journal {journal} is empty")
        step = availability_mod.STEP_SECONDS
        result = query_range(availability_mod.READY_QUERY, start, end, step)
        series = [[(float(ts), float(v)) for ts, v in r["values"]] for r in result]
    except Exception as exc:  # noqa: BLE001 - recorded as the reason, never a pass
        return availability_mod.not_measured(f"heartbeat query failed ({type(exc).__name__}: {exc})")
    return availability_mod.compute_availability(series, start, end, step=step)


def harvest(
    out: Path,
    *,
    schedule: Path | None = None,
    namespace: str = NAMESPACE,
    run: Runner | None = None,
    query_range: RangeQuery | None = None,
) -> dict:
    run = run or _run
    journal = out.with_suffix(".events.jsonl")
    records = out.with_suffix(".triage.jsonl")
    episodes = out.with_suffix(".episodes.json")
    usage = out.with_suffix(".usage.jsonl")
    report: dict = {
        "journal": str(journal), "records": None, "usage": None, "episodes": None,
        "availability": None, "not_measured": [],
    }

    def refuse(reason: str) -> dict:
        report["not_measured"].append(reason)
        return report

    if not journal.exists():
        return refuse(f"journal {journal} does not exist; the driver never started the round")
    try:
        started = _round_start(journal)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return refuse(f"journal {journal} first line is unreadable ({type(exc).__name__}: {exc})")
    if started is None:
        return refuse(f"journal {journal} is empty")

    # Before the pod checks on purpose: the heartbeat lives in Prometheus, so it is readable even
    # when the pod is gone or the record copy is refused. Reported, never a reason to refuse.
    report["availability"] = measure_availability(journal, query_range).to_json()

    pods = run(["kubectl", "-n", namespace, "get", "pods", "-l", POD_SELECTOR, "-o", "json"])
    if pods.returncode != 0:
        return refuse(f"kubectl get pods failed: {pods.stderr.strip()}")
    try:
        items = json.loads(pods.stdout).get("items", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        return refuse(f"kubectl get pods returned unparsable output ({exc})")
    if len(items) != 1:
        return refuse(f"expected exactly one agent pod, found {len(items)}")
    try:
        pod = items[0]["metadata"]
        created = datetime.fromisoformat(pod["creationTimestamp"].replace("Z", "+00:00")).timestamp()
    except (KeyError, TypeError, ValueError) as exc:
        return refuse(f"agent pod metadata is incomplete ({type(exc).__name__}: {exc})")
    report["pod"] = {"name": pod["name"], "created": pod["creationTimestamp"]}

    read = run(["kubectl", "-n", namespace, "exec", pod["name"], "--", "sh", "-c", READ_SCRIPT])
    if read.returncode == SINK_NOT_CONFIGURED:
        return refuse("SSLM_TRIAGE_LOG is unset in the agent pod; no record was ever written to a file")
    if read.returncode not in (0, NO_RECORD_YET):
        return refuse(f"reading SSLM_TRIAGE_LOG from {pod['name']} failed: {read.stderr.strip()}")
    records.parent.mkdir(parents=True, exist_ok=True)
    records.write_text(read.stdout if read.returncode == 0 else "", encoding="utf-8")
    report["records"] = str(records)

    # Token usage for the cost slice. Never blocks the join: a missing usage line leaves an
    # episode unattributed in the cost, it does not change what the agent decided.
    read_usage = run(["kubectl", "-n", namespace, "exec", pod["name"], "--", "sh", "-c", USAGE_READ_SCRIPT])
    if read_usage.returncode == 0 or (read_usage.returncode == NO_RECORD_YET and not read.stdout.strip()):
        usage.write_text(read_usage.stdout if read_usage.returncode == 0 else "", encoding="utf-8")
        report["usage"] = str(usage)
    elif read_usage.returncode == NO_RECORD_YET:
        report["not_measured"].append(
            "the pod holds TriageRecords but no usage log; its image predates per-episode token usage"
        )
    else:
        report["not_measured"].append(f"reading the usage log from {pod['name']} failed: {read_usage.stderr.strip()}")

    if created > started:
        # Checked after the copy on purpose: the partial file is still evidence worth keeping.
        return refuse(
            f"agent pod {pod['name']} was created at {pod['creationTimestamp']}, after the round "
            "started; records written before that were lost with the old pod's emptyDir"
        )

    argv = ["--journal", str(journal), "--records", str(records), "--out", str(episodes)]
    if schedule is not None:
        argv += ["--schedule", str(schedule)]
    try:
        episode_bridge.main(argv)
    except (episode_bridge.BridgeError, ValueError) as exc:
        return refuse(f"episode_bridge refused the join: {exc}")
    report["episodes"] = str(episodes)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the driver's --out path")
    parser.add_argument("--schedule", type=Path, help="schedule JSON of the round, passed to the bridge")
    parser.add_argument("--namespace", default=NAMESPACE)
    parser.add_argument("--prometheus-url", help="Prometheus of the round's cluster; reads the agent heartbeat")
    args = parser.parse_args(argv)

    query_range = _prometheus_range(args.prometheus_url) if args.prometheus_url else None
    report = harvest(args.out, schedule=args.schedule, namespace=args.namespace, query_range=query_range)
    report_path = args.out.with_suffix(".harvest.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for reason in report["not_measured"]:
        print(f"harvest: not measured: {reason}", file=sys.stderr)
    avail = report["availability"]
    if avail is not None and not avail["measured"]:
        print(f"harvest: availability not measured: {avail['reason']}", file=sys.stderr)
    return 0 if not report["not_measured"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
