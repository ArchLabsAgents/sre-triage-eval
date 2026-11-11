"""Aggregator for the continuous-scenario footprint. Reads the pod-footprint and vllm-footprint
JSONL files (collect_pod_footprint.py / log_vllm_footprint.py), joins them per TriageRecord
episode, and writes scenarios/results/<env>/continuous/continuous-footprint.csv (one row per
episode) plus continuous-footprint-cost.json (the run-level USD per triage number) in that same
continuous/ subdirectory: nested, never dropped straight into results/<env>/, so
harness.metrics.collect_records() (which run_matrix.py and scenarios/backfill_env.py call to
rebuild the episodic matrix.csv, and which globs results/<env>/*.json one level deep) never
tries to parse this output as a RunRecord.

Scope boundary, same discipline as entrypoint.py's `episode_runner` seam: the real orchestrator
that emits TriageRecords to disk lives elsewhere, so `--triage-dir` is optional. Given a directory
of TriageRecord-shaped JSON files, this aggregator does the real per-episode join. Without one (or with an empty one), it still writes the run-level cost summary
against `--episode-count` (0 by default, honestly reported as "not measured", never guessed) and
skips the per-episode CSV rather than fabricating episodes that were never recorded. Do not
extend this script to emit TriageRecords itself: that is the orchestrator's job, elsewhere.

Usage:
    python scenarios/aggregate_footprint.py \\
        --env hyperstack-a6000-continuous \\
        --pod-samples scenarios/results/hyperstack-a6000-continuous/pod-footprint.jsonl \\
        --vllm-samples scenarios/results/hyperstack-a6000-continuous/vllm-footprint.jsonl \\
        --triage-dir scenarios/results/hyperstack-a6000-continuous/triage-records \\
        --vm-hours 24.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from harness.metrics import (  # noqa: E402
    PRICE_CONTINUOUS_ENV_USD_PER_HOUR,
    compute_continuous_cost_summary,
    join_all_episode_footprints,
    load_pod_footprint_samples,
    load_vllm_footprint_samples,
    write_continuous_cost_summary_json,
    write_episode_footprint_csv,
)

DEFAULT_RESULTS_ROOT = TOOLS_DIR / "scenarios" / "results"


def load_triage_records(triage_dir: str | Path | None) -> list[dict]:
    """Load every *.json file in `triage_dir` as a TriageRecord dict. Missing or empty
    directory returns [] -- the documented boundary (see module docstring), never an error:
    the aggregator's job is to be honest about what it could join, not to insist the
    orchestrator already exists.

    A file that fails to parse (truncated write, corrupted contents -- plausible since nothing
    upstream validates against harness.triage.TRIAGE_RECORD_SCHEMA before this reads it) is
    skipped with a printed warning, never allowed to crash the whole run: one bad TriageRecord
    must not cost the run the footprint of every other episode it could still join.
    """
    if triage_dir is None:
        return []
    path = Path(triage_dir)
    if not path.is_dir():
        return []
    records = []
    for file in sorted(path.glob("*.json")):
        try:
            records.append(json.loads(file.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[aggregate-footprint] skipping unreadable TriageRecord {file}: {exc}", flush=True)
    return records


def run(
    *,
    env: str,
    pod_samples_path: str | Path,
    vllm_samples_path: str | Path,
    triage_dir: str | Path | None,
    vm_hours: float,
    price_usd_per_hour: float,
    out_dir: str | Path,
) -> dict:
    pod_samples = load_pod_footprint_samples(pod_samples_path)
    vllm_samples = load_vllm_footprint_samples(vllm_samples_path)
    triage_records = load_triage_records(triage_dir)

    out_dir = Path(out_dir)
    # A nested "continuous/" subdirectory, never files dropped straight into out_dir (which is
    # scenarios/results/<env>/ by default): harness.metrics.collect_records() -- the function
    # run_matrix.py and scenarios/backfill_env.py both call to rebuild the episodic matrix.csv
    # -- globs results/<env>/*.json one level deep and would try to parse a continuous-scenario
    # JSON file sitting there as a RunRecord (crashing with a TypeError on the first field
    # mismatch). One extra directory level puts these files two levels deep, outside that glob,
    # which is the real enforcement of "the two schemas can never be read as one table" -- a
    # by-name exclusion list would need updating for every future continuous output file.
    continuous_dir = out_dir / "continuous"
    result: dict = {
        "env": env,
        "pod_samples_loaded": len(pod_samples),
        "vllm_samples_loaded": len(vllm_samples),
        "triage_records_loaded": len(triage_records),
    }

    if triage_records:
        episodes = join_all_episode_footprints(
            env=env,
            triage_records=triage_records,
            pod_samples=pod_samples,
            vllm_samples=vllm_samples,
        )
        csv_path = write_episode_footprint_csv(episodes, continuous_dir / "continuous-footprint.csv")
        result["episode_footprint_csv"] = str(csv_path)
        result["episode_count"] = len(episodes)
    else:
        result["episode_footprint_csv"] = None
        result["episode_count"] = 0
        result["episode_footprint_skipped_reason"] = (
            "no TriageRecords found under --triage-dir (orchestrator boundary, see module docstring)"
        )

    summary = compute_continuous_cost_summary(
        env=env,
        vm_hours=vm_hours,
        episode_count=result["episode_count"],
        price_usd_per_hour=price_usd_per_hour,
    )
    cost_path = write_continuous_cost_summary_json(summary, continuous_dir / "continuous-footprint-cost.json")
    result["cost_summary_json"] = str(cost_path)
    result["usd_per_triage"] = summary.usd_per_triage
    result["usd_per_triage_reason"] = summary.usd_per_triage_reason
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, help="ex.: hyperstack-a6000-continuous (chave de envs: em base.yaml)")
    parser.add_argument("--pod-samples", required=True, help="JSONL from collect_pod_footprint.py")
    parser.add_argument("--vllm-samples", required=True, help="JSONL from log_vllm_footprint.py")
    parser.add_argument(
        "--triage-dir",
        default=None,
        help="diretorio de TriageRecord JSON (opcional; ausente/vazio = join pulado, ver docstring)",
    )
    parser.add_argument("--vm-hours", type=float, required=True, help="VM lifetime hours in the measured window")
    parser.add_argument("--price-usd-per-hour", type=float, default=PRICE_CONTINUOUS_ENV_USD_PER_HOUR)
    parser.add_argument("--out-dir", default=None, help="default: scenarios/results/<env>/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_RESULTS_ROOT / args.env

    result = run(
        env=args.env,
        pod_samples_path=args.pod_samples,
        vllm_samples_path=args.vllm_samples,
        triage_dir=args.triage_dir,
        vm_hours=args.vm_hours,
        price_usd_per_hour=args.price_usd_per_hour,
        out_dir=out_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
