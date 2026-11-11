"""Unit coverage for the continuous-scenario footprint dataclasses/aggregator in
harness/metrics.py. Pure functions, no cluster/GPU I/O: the join and USD-per-triage math are
testable with synthetic time series.
"""

from __future__ import annotations

import json

import pytest

from harness.metrics import (
    EpisodeFootprint,
    PodFootprintSample,
    PRICE_CONTINUOUS_ENV_USD_PER_HOUR,
    VllmFootprintSample,
    compute_continuous_cost_summary,
    join_all_episode_footprints,
    join_episode_footprint,
    load_pod_footprint_samples,
    load_vllm_footprint_samples,
    write_continuous_cost_summary_json,
    write_episode_footprint_csv,
    write_footprint_sample_jsonl,
)


# --------------------------------------------------------------------------------------------
# Honesty rule: collected=False always needs a reason, never a silent/implicit zero.
# --------------------------------------------------------------------------------------------


def test_pod_sample_uncollected_without_reason_is_rejected():
    with pytest.raises(ValueError, match="reason"):
        PodFootprintSample(timestamp=1.0, collected=False)


def test_vllm_sample_uncollected_without_reason_is_rejected():
    with pytest.raises(ValueError, match="reason"):
        VllmFootprintSample(timestamp=1.0, collected=False)


def test_pod_sample_collected_true_carries_values():
    sample = PodFootprintSample(timestamp=1.0, collected=True, cpu_cores=0.5, memory_bytes=1024.0)
    assert sample.reason is None


# --------------------------------------------------------------------------------------------
# JSONL round trip (the collector/logger scripts append; the aggregator reads back).
# --------------------------------------------------------------------------------------------


def test_write_and_load_pod_footprint_samples_round_trip(tmp_path):
    path = tmp_path / "pod.jsonl"
    s1 = PodFootprintSample(timestamp=100.0, collected=True, cpu_cores=0.3, memory_bytes=2048.0)
    s2 = PodFootprintSample(timestamp=130.0, collected=False, reason="prometheus unreachable")
    write_footprint_sample_jsonl(path, s1)
    write_footprint_sample_jsonl(path, s2)

    loaded = load_pod_footprint_samples(path)

    assert loaded == [s1, s2]
    # append, not overwrite: one line per call
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_write_and_load_vllm_footprint_samples_round_trip(tmp_path):
    path = tmp_path / "vllm.jsonl"
    sample = VllmFootprintSample(
        timestamp=100.0,
        collected=True,
        gpu_vram_used_mb=12000.0,
        gpu_utilization_pct=87.0,
        gpu_power_watts=210.5,
        cgroup_cpu_seconds=42.1,
        cgroup_memory_bytes=4_000_000.0,
    )
    write_footprint_sample_jsonl(path, sample)

    loaded = load_vllm_footprint_samples(path)

    assert loaded == [sample]


def test_load_pod_footprint_samples_missing_file_returns_empty_list(tmp_path):
    assert load_pod_footprint_samples(tmp_path / "does-not-exist.jsonl") == []


def test_load_vllm_footprint_samples_missing_file_returns_empty_list(tmp_path):
    assert load_vllm_footprint_samples(tmp_path / "does-not-exist.jsonl") == []


# --------------------------------------------------------------------------------------------
# join_episode_footprint: the core aggregation logic.
# --------------------------------------------------------------------------------------------


def _pod(ts, cpu=None, mem=None, collected=True, reason=None):
    if not collected and reason is None:
        reason = "synthetic gap"
    return PodFootprintSample(timestamp=ts, collected=collected, cpu_cores=cpu, memory_bytes=mem, reason=reason)


def _vllm(ts, vram=None, util=None, power=None, cpu=None, mem=None, collected=True, reason=None):
    if not collected and reason is None:
        reason = "synthetic gap"
    return VllmFootprintSample(
        timestamp=ts,
        collected=collected,
        gpu_vram_used_mb=vram,
        gpu_utilization_pct=util,
        gpu_power_watts=power,
        cgroup_cpu_seconds=cpu,
        cgroup_memory_bytes=mem,
        reason=reason,
    )


def test_join_episode_footprint_means_only_samples_inside_window():
    pod_samples = [
        _pod(50.0, cpu=0.1, mem=100.0),  # before the window
        _pod(110.0, cpu=0.4, mem=200.0),  # inside
        _pod(120.0, cpu=0.6, mem=300.0),  # inside
        _pod(200.0, cpu=0.9, mem=900.0),  # after the window
    ]
    vllm_samples = [
        _vllm(110.0, vram=10000.0, util=80.0, power=200.0, cpu=1.0, mem=5000.0),
        _vllm(120.0, vram=12000.0, util=90.0, power=220.0, cpu=1.5, mem=5500.0),
    ]

    row = join_episode_footprint(
        episode_id="ep-1",
        env="hyperstack-a6000-continuous",
        window_start=100.0,
        window_end=150.0,
        pod_samples=pod_samples,
        vllm_samples=vllm_samples,
    )

    assert row.pod_samples_in_window == 2
    assert row.pod_cpu_cores_mean == pytest.approx(0.5)
    assert row.pod_memory_bytes_mean == pytest.approx(250.0)
    assert row.pod_coverage_reason is None

    assert row.vllm_samples_in_window == 2
    assert row.vllm_gpu_vram_used_mb_mean == pytest.approx(11000.0)
    assert row.vllm_gpu_utilization_pct_mean == pytest.approx(85.0)
    assert row.vllm_gpu_power_watts_mean == pytest.approx(210.0)
    # delta (last - first in window), never a mean over a monotonic counter: 1.5 - 1.0
    assert row.vllm_cgroup_cpu_seconds_delta == pytest.approx(0.5)
    assert row.vllm_cgroup_memory_bytes_mean == pytest.approx(5250.0)
    assert row.vllm_coverage_reason is None


def test_join_episode_footprint_excludes_uncollected_samples_even_inside_window():
    pod_samples = [_pod(110.0, collected=False), _pod(120.0, cpu=0.5, mem=500.0)]

    row = join_episode_footprint(
        episode_id="ep-1",
        env="e",
        window_start=100.0,
        window_end=150.0,
        pod_samples=pod_samples,
        vllm_samples=[],
    )

    assert row.pod_samples_in_window == 1
    assert row.pod_cpu_cores_mean == pytest.approx(0.5)


def test_join_episode_footprint_no_samples_in_window_is_none_with_reason_never_zero():
    row = join_episode_footprint(
        episode_id="ep-1",
        env="e",
        window_start=100.0,
        window_end=150.0,
        pod_samples=[_pod(50.0, cpu=0.1, mem=100.0)],  # outside the window
        vllm_samples=[],
    )

    assert row.pod_samples_in_window == 0
    assert row.pod_cpu_cores_mean is None
    assert row.pod_memory_bytes_mean is None
    assert row.pod_coverage_reason is not None
    assert row.pod_coverage_reason != 0  # never zero, not even by a type accident

    assert row.vllm_samples_in_window == 0
    assert row.vllm_gpu_vram_used_mb_mean is None
    assert row.vllm_coverage_reason is not None


def test_join_episode_footprint_cgroup_cpu_delta_needs_at_least_two_samples():
    # Regression: a single sample cannot measure a delta. One
    # collected vllm sample in the window still reports the other vllm means, but the cpu delta
    # is None -- not the single value, and never 0.0.
    row = join_episode_footprint(
        episode_id="ep-1",
        env="e",
        window_start=100.0,
        window_end=150.0,
        pod_samples=[],
        vllm_samples=[_vllm(110.0, vram=10000.0, cpu=5.0)],
    )

    assert row.vllm_samples_in_window == 1
    assert row.vllm_gpu_vram_used_mb_mean == pytest.approx(10000.0)
    assert row.vllm_cgroup_cpu_seconds_delta is None


def test_join_episode_footprint_cgroup_cpu_delta_uses_chronological_order_not_list_order():
    row = join_episode_footprint(
        episode_id="ep-1",
        env="e",
        window_start=100.0,
        window_end=150.0,
        pod_samples=[],
        # out of order in the list: the delta must sort by timestamp, not by input order
        vllm_samples=[_vllm(140.0, cpu=9.0), _vllm(110.0, cpu=4.0)],
    )

    assert row.vllm_cgroup_cpu_seconds_delta == pytest.approx(5.0)


def test_join_episode_footprint_cgroup_cpu_counter_reset_is_none_not_negative():
    # A restart of vLLM's systemd unit mid-window resets the cgroup counter: a negative delta is
    # not a real negative cost, it is not measured, never a fabricated number.
    row = join_episode_footprint(
        episode_id="ep-1",
        env="e",
        window_start=100.0,
        window_end=150.0,
        pod_samples=[],
        vllm_samples=[_vllm(110.0, cpu=50.0), _vllm(120.0, cpu=2.0)],
    )

    assert row.vllm_cgroup_cpu_seconds_delta is None


def test_join_all_episode_footprints_parses_triage_windows():
    triage_records = [
        {
            "episode_id": "ep-1",
            "t_alarme": "2026-09-21T10:00:00+00:00",
            "t_record": "2026-09-21T10:05:00+00:00",
        },
        {
            "episode_id": "ep-2",
            "t_alarme": "2026-09-21T11:00:00+00:00",
            "t_record": "2026-09-21T11:02:00+00:00",
        },
    ]
    # ep-1 timestamp in epoch: 2026-09-21T10:02:30Z falls inside ep-1's window
    from datetime import datetime

    mid_ep1 = datetime.fromisoformat("2026-09-21T10:02:30+00:00").timestamp()
    pod_samples = [_pod(mid_ep1, cpu=0.2, mem=200.0)]

    rows = join_all_episode_footprints(
        env="hyperstack-a6000-continuous",
        triage_records=triage_records,
        pod_samples=pod_samples,
        vllm_samples=[],
    )

    assert [r.episode_id for r in rows] == ["ep-1", "ep-2"]
    assert rows[0].pod_samples_in_window == 1
    assert rows[1].pod_samples_in_window == 0  # no sample, not zero: coverage_reason set
    assert rows[1].pod_coverage_reason is not None


def test_join_all_episode_footprints_skips_records_with_inverted_window():
    # Regression: t_record before t_alarme was silently swapped into a valid-looking window
    # instead of being skipped like every other malformed-input case in this function,
    # inconsistent with the honesty rule the rest of the module follows.
    triage_records = [
        {
            "episode_id": "ep-inverted",
            "t_alarme": "2026-09-21T10:05:00+00:00",
            "t_record": "2026-09-21T10:00:00+00:00",
        },
        {
            "episode_id": "ep-ok",
            "t_alarme": "2026-09-21T10:00:00+00:00",
            "t_record": "2026-09-21T10:05:00+00:00",
        },
    ]

    rows = join_all_episode_footprints(
        env="e", triage_records=triage_records, pod_samples=[], vllm_samples=[]
    )

    assert [r.episode_id for r in rows] == ["ep-ok"]


def test_join_all_episode_footprints_skips_records_with_unparseable_timestamps():
    triage_records = [
        {"episode_id": "ep-bad", "t_alarme": "not-a-timestamp", "t_record": "also-not"},
        {"episode_id": "ep-missing-field"},
    ]

    rows = join_all_episode_footprints(
        env="e", triage_records=triage_records, pod_samples=[], vllm_samples=[]
    )

    assert rows == []


def test_join_all_episode_footprints_skips_records_missing_episode_id():
    # Regression: episode_id was read outside the try/except that
    # guards the timestamp fields, so a record with valid timestamps but no episode_id raised
    # an uncaught KeyError and crashed the whole run instead of being skipped.
    triage_records = [
        {"t_alarme": "2026-09-21T10:00:00+00:00", "t_record": "2026-09-21T10:05:00+00:00"},
        {
            "episode_id": "ep-ok",
            "t_alarme": "2026-09-21T10:00:00+00:00",
            "t_record": "2026-09-21T10:05:00+00:00",
        },
    ]

    rows = join_all_episode_footprints(
        env="e", triage_records=triage_records, pod_samples=[], vllm_samples=[]
    )

    assert [r.episode_id for r in rows] == ["ep-ok"]


# --------------------------------------------------------------------------------------------
# USD per triage: hours x price / episodes, never a division by zero fabricating a number.
# --------------------------------------------------------------------------------------------


def test_compute_continuous_cost_summary_divides_evenly():
    summary = compute_continuous_cost_summary(
        env="hyperstack-a6000-continuous",
        vm_hours=24.0,
        episode_count=12,
        price_usd_per_hour=0.5,
    )

    assert summary.usd_total == pytest.approx(12.0)
    assert summary.usd_per_triage == pytest.approx(1.0)
    assert summary.usd_per_triage_reason is None


def test_compute_continuous_cost_summary_uses_the_declared_default_price():
    summary = compute_continuous_cost_summary(env="e", vm_hours=1.0, episode_count=1)
    assert summary.price_usd_per_hour == PRICE_CONTINUOUS_ENV_USD_PER_HOUR
    assert summary.usd_total == pytest.approx(PRICE_CONTINUOUS_ENV_USD_PER_HOUR)


def test_compute_continuous_cost_summary_zero_episodes_is_none_with_reason_never_zero_division():
    summary = compute_continuous_cost_summary(env="e", vm_hours=24.0, episode_count=0, price_usd_per_hour=0.5)

    assert summary.usd_per_triage is None
    assert summary.usd_per_triage_reason
    # the total is still reported (the VM cost something even with no episode triaged)
    assert summary.usd_total == pytest.approx(12.0)


def test_compute_continuous_cost_summary_rejects_negative_inputs():
    with pytest.raises(ValueError):
        compute_continuous_cost_summary(env="e", vm_hours=-1.0, episode_count=1)
    with pytest.raises(ValueError):
        compute_continuous_cost_summary(env="e", vm_hours=1.0, episode_count=-1)


# --------------------------------------------------------------------------------------------
# Writers: CSV/JSON output under scenarios/results/<env>/ convention.
# --------------------------------------------------------------------------------------------


def test_write_episode_footprint_csv(tmp_path):
    rows = [
        EpisodeFootprint(
            episode_id="ep-1",
            env="hyperstack-a6000-continuous",
            window_start=100.0,
            window_end=150.0,
            pod_cpu_cores_mean=0.5,
            pod_memory_bytes_mean=250.0,
            pod_samples_in_window=2,
            pod_coverage_reason=None,
            vllm_gpu_vram_used_mb_mean=11000.0,
            vllm_gpu_utilization_pct_mean=85.0,
            vllm_gpu_power_watts_mean=210.0,
            vllm_cgroup_cpu_seconds_delta=1.25,
            vllm_cgroup_memory_bytes_mean=5250.0,
            vllm_samples_in_window=2,
            vllm_coverage_reason=None,
        )
    ]
    out = write_episode_footprint_csv(rows, tmp_path / "continuous" / "continuous-footprint.csv")

    text = out.read_text(encoding="utf-8")
    assert "ep-1" in text
    assert "episode_id" in text.splitlines()[0]


def test_write_continuous_cost_summary_json(tmp_path):
    summary = compute_continuous_cost_summary(env="e", vm_hours=1.0, episode_count=0)
    out = write_continuous_cost_summary_json(summary, tmp_path / "continuous" / "cost.json")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["env"] == "e"
    assert data["usd_per_triage"] is None
    assert data["usd_per_triage_reason"]
