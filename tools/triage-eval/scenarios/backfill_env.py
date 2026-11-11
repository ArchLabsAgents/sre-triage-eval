#!/usr/bin/env python3
"""Stamp legacy (pre-env) result records with an explicit env and move them into results/<env>/.

The early matrices ran on a dev machine before RunRecord carried `env`. The loader never stamps
those records implicitly: they load as env=None and render as "unknown", because assuming an
origin is exactly the ambiguity the env column exists to prevent. Stamping them is an explicit
decision, and this script is that decision made runnable, once per results directory, on the machine that holds the results:

    .venv/bin/python scenarios/backfill_env.py --env mlx-local                # dry-run
    .venv/bin/python scenarios/backfill_env.py --env mlx-local --apply
    .venv/bin/python scenarios/backfill_env.py --env mlx-local \\
        --results-dir scenarios/results-archive --apply                       # archived dirs

Dry-run is the default: it prints the full plan and touches nothing. Rules:

- Only records at the results-dir ROOT (the legacy layout) are touched; records already inside
  an env directory are left alone.
- A root record that already carries a DIFFERENT env aborts the whole run before any change:
  this script stamps the unstamped, it never restamps.
- Companion files move together: per-record transcripts and episodic_memory.jsonl.
- <env>/env-descriptor.json is written from the env's declaration in the base config; if one
  already exists and differs, the run aborts (the declaration changed since that env's runs).
- matrix.csv is rebuilt from every record found afterwards, so the env column shows up
  immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from harness.envs import EnvConfigError, load_envs, select_env  # noqa: E402
from harness.metrics import RunRecord, collect_records, write_matrix_csv  # noqa: E402

TOOL_ROOT = Path(__file__).resolve().parent.parent


def _load_env_block(base_config_path: Path, env_name: str) -> dict:
    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    # select_env applies the same rules as a run's start: declared slug and quantization
    # present. A backfill is no less strict than a new run.
    _, block = select_env(load_envs(config), env_name)
    assert block is not None  # with an explicit env, select_env returns the block or raises
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="env to stamp (a key of `envs:` in the base config)")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=TOOL_ROOT / "scenarios" / "results",
        help="results directory holding legacy root-level records (default: scenarios/results)",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=TOOL_ROOT / "configs" / "base.yaml",
        help="config declaring `envs:` (default: configs/base.yaml)",
    )
    parser.add_argument("--apply", action="store_true", help="execute the plan (default: dry-run)")
    args = parser.parse_args()

    try:
        env_block = _load_env_block(args.base_config, args.env)
    except EnvConfigError as exc:
        parser.error(str(exc))
        return

    results_dir: Path = args.results_dir
    if not results_dir.is_dir():
        parser.error(f"{results_dir} is not a directory")
        return
    env_dir = results_dir / args.env

    record_paths = sorted(results_dir.glob("*.json"))
    records: list[tuple[Path, RunRecord]] = []
    for path in record_paths:
        record = RunRecord.from_json(path.read_text(encoding="utf-8"))
        if record.env is not None and record.env != args.env:
            print(
                f"abort: {path.name} already carries env={record.env!r} (this script stamps "
                "the unstamped, it never restamps); nothing changed",
                file=sys.stderr,
            )
            raise SystemExit(1)
        records.append((path, record))

    descriptor_path = env_dir / "env-descriptor.json"
    if descriptor_path.exists():
        existing = json.loads(descriptor_path.read_text(encoding="utf-8"))
        if existing != env_block:
            print(
                f"abort: {descriptor_path} differs from the current declaration of "
                f"{args.env!r} in {args.base_config}; nothing changed",
                file=sys.stderr,
            )
            raise SystemExit(1)

    companions = sorted(results_dir.glob("*_transcript.jsonl"))
    memory_store = results_dir / "episodic_memory.jsonl"
    if memory_store.exists():
        companions.append(memory_store)

    mode = "apply" if args.apply else "dry-run"
    print(f"[{mode}] {len(records)} record(s), {len(companions)} companion file(s) -> {env_dir}/")
    for path, record in records:
        stamp = "stamp+" if record.env is None else ""
        print(f"[{mode}] {stamp}move {path.name}")
        if args.apply:
            record.env = args.env
            record.env_quantization = record.env_quantization or env_block.get("quantization")
            record.env_network = record.env_network or env_block.get("network")
            env_dir.mkdir(parents=True, exist_ok=True)
            (env_dir / path.name).write_text(json.dumps(asdict(record)), encoding="utf-8")
            path.unlink()
    for path in companions:
        print(f"[{mode}] move {path.name}")
        if args.apply:
            env_dir.mkdir(parents=True, exist_ok=True)
            path.rename(env_dir / path.name)

    if args.apply:
        if records and not descriptor_path.exists():
            descriptor_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor_path.write_text(
                json.dumps(env_block, indent=2, sort_keys=True), encoding="utf-8"
            )
        write_matrix_csv(collect_records(results_dir), results_dir / "matrix.csv")
        print(f"[apply] matrix.csv rebuilt at {results_dir / 'matrix.csv'}")
    else:
        print("[dry-run] nothing changed; re-run with --apply to execute")


if __name__ == "__main__":
    main()
