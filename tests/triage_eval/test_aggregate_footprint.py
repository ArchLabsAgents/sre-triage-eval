"""Unit coverage for scenarios/aggregate_footprint.py, the CLI that ties the pod/vllm sample files
together with TriageRecords and writes scenarios/results/<env>/. Covers
both the real-join path and the documented boundary (no TriageRecords yet -> cost-only output,
never a fabricated per-episode row)."""

from __future__ import annotations

import json

from harness.metrics import (
    PodFootprintSample,
    VllmFootprintSample,
    collect_records,
    write_footprint_sample_jsonl,
)
from scenarios import aggregate_footprint as mod


def test_load_triage_records_missing_dir_returns_empty_list(tmp_path):
    assert mod.load_triage_records(tmp_path / "does-not-exist") == []


def test_load_triage_records_none_returns_empty_list():
    assert mod.load_triage_records(None) == []


def test_load_triage_records_reads_every_json_file(tmp_path):
    triage_dir = tmp_path / "triage-records"
    triage_dir.mkdir()
    (triage_dir / "ep-1.json").write_text(json.dumps({"episode_id": "ep-1"}))
    (triage_dir / "ep-2.json").write_text(json.dumps({"episode_id": "ep-2"}))

    records = mod.load_triage_records(triage_dir)

    assert {r["episode_id"] for r in records} == {"ep-1", "ep-2"}


def test_load_triage_records_skips_malformed_file_instead_of_crashing(tmp_path, capsys):
    # Regression: json.loads had no error handling, so one corrupted
    # file (plausible mid-write) crashed the entire run before any output was written.
    triage_dir = tmp_path / "triage-records"
    triage_dir.mkdir()
    (triage_dir / "ep-1.json").write_text(json.dumps({"episode_id": "ep-1"}))
    (triage_dir / "corrupted.json").write_text("{not valid json")

    records = mod.load_triage_records(triage_dir)

    assert {r["episode_id"] for r in records} == {"ep-1"}
    assert "corrupted.json" in capsys.readouterr().out


def test_run_without_triage_dir_skips_per_episode_csv_but_still_writes_cost_summary(tmp_path):
    pod_path = tmp_path / "pod.jsonl"
    vllm_path = tmp_path / "vllm.jsonl"
    write_footprint_sample_jsonl(pod_path, PodFootprintSample(timestamp=1.0, collected=True, cpu_cores=0.1, memory_bytes=100.0))
    write_footprint_sample_jsonl(vllm_path, VllmFootprintSample(timestamp=1.0, collected=True, gpu_vram_used_mb=1.0))

    out_dir = tmp_path / "results" / "hyperstack-a6000-continuous"
    result = mod.run(
        env="hyperstack-a6000-continuous",
        pod_samples_path=pod_path,
        vllm_samples_path=vllm_path,
        triage_dir=None,
        vm_hours=2.0,
        price_usd_per_hour=1.0,
        out_dir=out_dir,
    )

    assert result["episode_count"] == 0
    assert result["episode_footprint_csv"] is None
    assert "episode_footprint_skipped_reason" in result
    assert result["usd_per_triage"] is None
    assert result["usd_per_triage_reason"]

    cost_data = json.loads((out_dir / "continuous" / "continuous-footprint-cost.json").read_text(encoding="utf-8"))
    assert cost_data["usd_total"] == 2.0
    assert not (out_dir / "continuous" / "continuous-footprint.csv").exists()
    # nada solto direto em out_dir (results/<env>/): so o subdiretorio continuous/
    assert not (out_dir / "continuous-footprint-cost.json").exists()


def test_run_with_triage_dir_writes_per_episode_csv_and_cost_summary(tmp_path):
    pod_path = tmp_path / "pod.jsonl"
    vllm_path = tmp_path / "vllm.jsonl"
    from datetime import datetime

    t_alarme = "2026-09-21T10:00:00+00:00"
    t_record = "2026-09-21T10:05:00+00:00"
    mid = datetime.fromisoformat("2026-09-21T10:02:00+00:00").timestamp()

    write_footprint_sample_jsonl(pod_path, PodFootprintSample(timestamp=mid, collected=True, cpu_cores=0.3, memory_bytes=300.0))
    write_footprint_sample_jsonl(vllm_path, VllmFootprintSample(timestamp=mid, collected=True, gpu_vram_used_mb=9000.0))

    triage_dir = tmp_path / "triage-records"
    triage_dir.mkdir()
    (triage_dir / "ep-1.json").write_text(
        json.dumps({"episode_id": "ep-1", "t_alarme": t_alarme, "t_record": t_record})
    )

    out_dir = tmp_path / "results" / "hyperstack-a6000-continuous"
    result = mod.run(
        env="hyperstack-a6000-continuous",
        pod_samples_path=pod_path,
        vllm_samples_path=vllm_path,
        triage_dir=triage_dir,
        vm_hours=1.0,
        price_usd_per_hour=1.0,
        out_dir=out_dir,
    )

    assert result["episode_count"] == 1
    assert result["usd_per_triage"] == 1.0

    csv_text = (out_dir / "continuous" / "continuous-footprint.csv").read_text(encoding="utf-8")
    assert "ep-1" in csv_text


def test_run_output_never_confuses_collect_records_episodic_matrix_rebuild(tmp_path):
    # Regression: harness.metrics.collect_records() (the function run_matrix.py and
    # scenarios/backfill_env.py call to rebuild the episodic matrix.csv) globs results/<env>/*.json one level deep. Before the "continuous/" subdirectory, the
    # continuous aggregator's own cost-summary JSON sat right there and collect_records() tried
    # to parse it as a RunRecord, crashing with a TypeError. This proves the two now coexist.
    pod_path = tmp_path / "pod.jsonl"
    vllm_path = tmp_path / "vllm.jsonl"
    write_footprint_sample_jsonl(pod_path, PodFootprintSample(timestamp=1.0, collected=True, cpu_cores=0.1, memory_bytes=100.0))
    write_footprint_sample_jsonl(vllm_path, VllmFootprintSample(timestamp=1.0, collected=True, gpu_vram_used_mb=1.0))

    results_root = tmp_path / "results"
    out_dir = results_root / "hyperstack-a6000-continuous"
    mod.run(
        env="hyperstack-a6000-continuous",
        pod_samples_path=pod_path,
        vllm_samples_path=vllm_path,
        triage_dir=None,
        vm_hours=2.0,
        price_usd_per_hour=1.0,
        out_dir=out_dir,
    )

    # A real episodic RunRecord living in the same results/ tree.
    (out_dir / "s1_config_a.json").write_text(
        json.dumps(
            {
                "scenario_id": "s1",
                "config_name": "config_a",
                "submitted": True,
                "output_tokens": 10,
                "elapsed_seconds": 1.0,
                "attempt_count": 1,
                "env": "hyperstack-a6000-continuous",
            }
        )
    )

    records = collect_records(results_root)  # must not raise TypeError

    assert len(records) == 1
    assert records[0].scenario_id == "s1"
