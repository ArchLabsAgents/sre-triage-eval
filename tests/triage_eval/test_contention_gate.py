"""The contention gate (CPU steal, queue ceiling vs MTTA, VM memory peak)."""

from __future__ import annotations

import json

import pytest

from harness.metrics import (
    EpisodeTokenUsage,
    HostSample,
    VllmQueueSample,
    contention_gate,
    host_sample_from_proc,
    load_host_samples,
    memory_signal,
    mtta_seconds,
    queue_signal,
    steal_signal,
    write_footprint_sample_jsonl,
)
from scenarios import contention_gate as cli
from scenarios import log_vllm_footprint as logger

MEMINFO = "MemTotal:       10000 kB\nMemFree:         1000 kB\nMemAvailable:    6000 kB\n"


def _stat(user=100, steal=0, guest=50) -> str:
    # user nice system idle iowait irq softirq steal guest guest_nice
    return f"cpu  {user} 0 0 900 0 0 0 {steal} {guest} 0\ncpu0 {user} 0 0 900 0 0 0 {steal} {guest} 0\n"


def _host(t: float, *, total: float, steal: float, used_pct: float = 40.0) -> HostSample:
    return HostSample(
        timestamp=t, collected=True, cpu_total_jiffies=total, cpu_steal_jiffies=steal,
        mem_total_bytes=1000.0, mem_available_bytes=1000.0 * (1 - used_pct / 100),
    )


def _queue(t: float, *, under_half_second: float, total: float) -> VllmQueueSample:
    """Cumulative TTFT: `under_half_second` requests at or below 0.5s, the rest at or below 5s."""
    buckets = {"0.5": under_half_second, "5.0": total, "+Inf": total}
    return VllmQueueSample(timestamp=t, collected=True, ttft_buckets=buckets, ttft_count=total, ttft_sum=0.0,
                           num_requests_waiting=0.0, num_requests_running=0.0,
                           prompt_tokens_total=0.0, generation_tokens_total=0.0)


def _usage(episode_id: str, t_start: float, seconds: float) -> EpisodeTokenUsage:
    return EpisodeTokenUsage(episode_id, t_start, t_start + seconds, 10, 5, 1, 0)


# --- host series -------------------------------------------------------------------------------


def test_host_sample_totals_stop_at_steal_because_guest_is_already_inside_user():
    sample = host_sample_from_proc(_stat(user=100, steal=30, guest=50), MEMINFO, timestamp=1.0)
    assert sample.collected
    assert sample.cpu_total_jiffies == 100 + 900 + 30
    assert sample.cpu_steal_jiffies == 30
    assert sample.mem_total_bytes == 10000 * 1024
    assert sample.mem_available_bytes == 6000 * 1024  # MemAvailable, never MemFree


@pytest.mark.parametrize(
    "stat, meminfo, reason",
    [
        ("cpu0 1 2 3 4 5 6 7 8\n", MEMINFO, "no aggregate cpu line"),
        ("cpu  1 2 3 4 5 6 7\n", MEMINFO, "no steal column"),
        (_stat(), "MemTotal: 10000 kB\nMemFree: 1000 kB\n", "MemAvailable missing"),
        (_stat(), "MemTotal: 10000 MB\nMemAvailable: 1 kB\n", "not in kB"),
        (_stat(), "MemTotal: 0 kB\nMemAvailable: 0 kB\n", "MemTotal is 0"),
        ("cpu  1 2 x 4 5 6 7 8\n", MEMINFO, "non-numeric"),
    ],
)
def test_host_sample_refuses_what_the_kernel_did_not_give(stat, meminfo, reason):
    sample = host_sample_from_proc(stat, meminfo, timestamp=1.0)
    assert not sample.collected
    assert reason in sample.reason
    assert sample.cpu_steal_jiffies is None


def test_uncollected_host_sample_requires_a_reason():
    with pytest.raises(ValueError):
        HostSample(timestamp=1.0, collected=False)


def test_logger_reads_proc_and_writes_the_host_series_beside_the_others(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "stat").write_text(_stat(steal=7))
    (proc / "meminfo").write_text(MEMINFO)
    assert logger.read_host_sample(now=3.0, proc_root=proc).cpu_steal_jiffies == 7
    missing = logger.read_host_sample(now=3.0, proc_root=tmp_path / "nope")
    assert not missing.collected and "could not read" in missing.reason

    out, host_out = tmp_path / "fp.jsonl", tmp_path / "host.jsonl"
    logger.run_loop(
        "sslm-vllm", out, once=True, host_out=host_out,
        _nvidia_smi=lambda: (1.0, 2.0, 3.0),
        _resolve_cgroup=lambda name, **kwargs: tmp_path / "no-cgroup",
        _host=lambda: host_sample_from_proc(_stat(), MEMINFO, timestamp=9.0),
    )
    (sample,) = load_host_samples(host_out)
    assert sample.collected and sample.timestamp == 9.0
    # the footprint tick failed on the cgroup; the host series did not care
    assert json.loads(out.read_text())["collected"] is False


# --- steal -------------------------------------------------------------------------------------


def test_steal_is_the_counter_delta_share_between_the_window_edges():
    samples = [_host(0, total=1000, steal=10), _host(50, total=1500, steal=12), _host(100, total=3000, steal=120)]
    signal = steal_signal(samples, 0, 100)
    assert signal.value == pytest.approx(100 * 110 / 2000)
    assert signal.status == "over"
    at_limit = [_host(0, total=1000, steal=10), _host(100, total=3000, steal=110)]
    assert steal_signal(at_limit, 0, 100).status == "pass"
    assert steal_signal([_host(0, total=1000, steal=0), _host(100, total=3000, steal=100)], 0, 100).status == "pass"


@pytest.mark.parametrize(
    "samples, reason",
    [
        ([], "no collected host sample at or before the window start"),
        ([_host(0, total=1000, steal=5), _host(100, total=500, steal=1)], "went backwards"),
        ([_host(0, total=1000, steal=5), _host(100, total=1000, steal=5)], "no CPU time elapsed"),
        ([_host(0, total=1, steal=0), _host(30, total=9, steal=0)], "70s before the window end"),
    ],
)
def test_steal_not_measured_is_never_zero(samples, reason):
    signal = steal_signal(samples, 0, 100)
    assert signal.status == "not_measured" and signal.value is None
    assert reason in signal.reason


def test_steal_zero_delta_is_a_real_zero_when_the_counter_proves_accounting():
    signal = steal_signal([_host(0, total=1000, steal=40), _host(100, total=3000, steal=40)], 0, 100)
    assert signal.status == "pass" and signal.value == 0.0 and signal.reason is None


def test_steal_counter_at_zero_since_boot_is_not_measured():
    samples = [_host(0, total=1000, steal=0), _host(100, total=3000, steal=0)]
    signal = steal_signal(samples, 0, 100)
    assert signal.status == "not_measured" and signal.value is None
    assert "steal accounting unproven" in signal.reason


def test_accepting_unproven_steal_passes_but_says_so():
    samples = [_host(0, total=1000, steal=0), _host(100, total=3000, steal=0)]
    signal = steal_signal(samples, 0, 100, accept_unproven_steal=True)
    assert signal.status == "pass" and signal.value == 0.0
    assert "unproven" in signal.reason and "--accept-unproven-steal" in signal.reason


def test_accepting_unproven_steal_never_hides_a_measured_value():
    # Accounting proven: the flag changes nothing, steal is judged as measured.
    samples = [_host(0, total=1000, steal=0), _host(100, total=3000, steal=300)]
    signal = steal_signal(samples, 0, 100, accept_unproven_steal=True)
    assert signal.status == "over" and signal.reason is None


# --- memory ------------------------------------------------------------------------------------


def test_memory_is_the_peak_not_the_mean():
    samples = [_host(t, total=t + 1, steal=0, used_pct=p) for t, p in [(0, 30), (30, 71), (60, 30), (90, 30)]]
    signal = memory_signal(samples, 0, 90)
    assert signal.value == pytest.approx(71.0)
    assert signal.status == "over"


def test_memory_at_the_limit_passes():
    samples = [_host(0, total=1, steal=0, used_pct=70), _host(30, total=2, steal=0, used_pct=70)]
    assert memory_signal(samples, 0, 30).status == "pass"


def test_memory_gap_inside_the_window_could_hide_the_peak():
    samples = [_host(0, total=1, steal=0), _host(20, total=2, steal=0), _host(100, total=3, steal=0)]
    signal = memory_signal(samples, 0, 100)
    assert signal.status == "not_measured"
    assert "80s gap" in signal.reason


def test_memory_skips_uncollected_ticks_when_measuring_gaps():
    gap = HostSample(timestamp=50, collected=False, reason="read failed")
    samples = [_host(0, total=1, steal=0), gap, _host(90, total=2, steal=0)]
    assert memory_signal(samples, 0, 90).status == "not_measured"


# --- MTTA and queue ----------------------------------------------------------------------------


def test_mtta_uses_the_usage_line_span_only_for_counted_episodes():
    usage = [_usage("a", 0, 40), _usage("b", 100, 60), _usage("probe", 5, 1)]
    assert mtta_seconds(usage, {"a", "b"}) == [40, 60]


def test_mtta_refuses_a_counted_episode_without_usage():
    result = mtta_seconds([_usage("a", 0, 40)], {"a", "b"})
    assert isinstance(result, str) and "1 counted episode(s) have no usage line: b" in result


def test_mtta_refuses_a_repeated_counted_episode():
    result = mtta_seconds([_usage("a", 0, 40), _usage("a", 50, 40)], {"a"})
    assert isinstance(result, str) and "repeated" in result


def test_queue_limit_is_ten_percent_of_the_median_mtta():
    samples = [_queue(0, under_half_second=0, total=0), _queue(100, under_half_second=20, total=20)]
    signal, median = queue_signal(samples, [4.0, 5.0, 6.0], 0, 100)
    assert median == 5.0
    assert signal.limit == pytest.approx(0.5)
    assert signal.value == 0.5 and signal.status == "pass"  # ceiling == limit passes

    over, _ = queue_signal(samples, [4.0, 4.5, 4.9], 0, 100)
    assert over.status == "over"


def test_queue_ceiling_only_overstates_so_a_slow_tail_fails():
    samples = [_queue(0, under_half_second=0, total=0), _queue(100, under_half_second=18, total=20)]
    signal, _ = queue_signal(samples, [60.0], 0, 100)
    assert signal.value == 5.0 and signal.status == "pass"
    signal, _ = queue_signal(samples, [40.0], 0, 100)
    assert signal.status == "over"


@pytest.mark.parametrize("mtta, reason", [("2 counted episode(s) have no usage line", "MTTA: 2 counted"), ([], "MTTA: no counted")])
def test_queue_without_mtta_is_not_measured(mtta, reason):
    signal, median = queue_signal([_queue(0, under_half_second=0, total=0)], mtta, 0, 100)
    assert signal.status == "not_measured" and median is None
    assert reason in signal.reason


def test_queue_carries_the_ceiling_reason_when_ttft_is_not_measured():
    samples = [_queue(0, under_half_second=5, total=5), _queue(100, under_half_second=5, total=5)]
    signal, median = queue_signal(samples, [10.0], 0, 100)
    assert signal.status == "not_measured" and median == 10.0
    assert "no request" in signal.reason


# --- gate --------------------------------------------------------------------------------------


def _clean_window():
    # Steal accounted since boot (counter at 5) but none inside the window: a proven zero.
    host = [_host(t, total=1000 * (t + 1), steal=5) for t in (0, 30, 60, 90, 100)]
    queue = [_queue(0, under_half_second=0, total=0), _queue(100, under_half_second=20, total=20)]
    return host, queue


def test_gate_passes_only_when_all_three_signals_pass():
    host, queue = _clean_window()
    gate = contention_gate(queue, host, [30.0, 40.0], 0, 100)
    assert gate.passed
    assert [s.name for s in gate.signals] == ["cpu_steal_pct", "queue_p95_ceiling_seconds", "vm_memory_peak_pct"]
    assert gate.mtta_median_seconds == 35.0 and gate.mtta_episodes == 2


def test_not_measured_fails_the_gate_like_over_does():
    host, queue = _clean_window()
    gate = contention_gate(queue, host, "harvest incomplete", 0, 100)
    assert not gate.passed
    assert [s.status for s in gate.signals] == ["pass", "not_measured", "pass"]


def test_gate_with_unproven_steal_fails_unless_accepted():
    _, queue = _clean_window()
    host = [_host(t, total=1000 * (t + 1), steal=0) for t in (0, 30, 60, 90, 100)]
    refused = contention_gate(queue, host, [30.0, 40.0], 0, 100)
    assert not refused.passed and refused.signals[0].status == "not_measured"
    assert not refused.accepted_unproven_steal
    accepted = contention_gate(queue, host, [30.0, 40.0], 0, 100, accept_unproven_steal=True)
    assert accepted.passed and accepted.accepted_unproven_steal


def test_gate_does_not_report_an_acceptance_the_flag_was_not_needed_for():
    host, queue = _clean_window()
    gate = contention_gate(queue, host, [30.0, 40.0], 0, 100, accept_unproven_steal=True)
    assert gate.passed and not gate.accepted_unproven_steal


def test_gate_refuses_an_empty_window():
    with pytest.raises(ValueError):
        contention_gate([], [], [], 5, 5)


# --- CLI ---------------------------------------------------------------------------------------


def _seed(tmp_path, name, *, t0, t1, episodes):
    out = tmp_path / name / "r2.json"
    out.parent.mkdir()
    out.with_suffix(".events.jsonl").write_text(
        "\n".join(json.dumps({"t": t, "phase": "x"}) for t in (t0, t1)) + "\n"
    )
    out.with_suffix(".episodes.json").write_text(json.dumps({"episodes": [
        {"record": {"episode_id": e}} for e, _ in episodes
    ] + [{"record": None}]}))
    out.with_suffix(".usage.jsonl").write_text(
        "".join(json.dumps(_usage(e, t0 + 1, s).__dict__) + "\n" for e, s in episodes)
    )
    return out


def _series(tmp_path):
    host, queue = _clean_window()
    host_path, queue_path = tmp_path / "host.jsonl", tmp_path / "queue.jsonl"
    for sample in host:
        write_footprint_sample_jsonl(host_path, sample)
    for sample in queue:
        write_footprint_sample_jsonl(queue_path, sample)
    return host_path, queue_path


def test_cli_pools_seeds_over_the_window_the_journals_span(tmp_path):
    host, queue = _series(tmp_path)
    a = _seed(tmp_path, "a", t0=0, t1=60, episodes=[("a1", 20.0)])
    b = _seed(tmp_path, "b", t0=10, t1=100, episodes=[("b1", 40.0), ("b2", 60.0)])
    verdict_path = tmp_path / "gate.json"
    rc = cli.main(["--seed-out", str(a), "--seed-out", str(b), "--queue", str(queue), "--host", str(host),
                   "--out", str(verdict_path)])
    verdict = json.loads(verdict_path.read_text())
    assert rc == 0 and verdict["passed"]
    assert verdict["window"] == {"start": 0.0, "end": 100.0}
    assert verdict["mtta_episodes"] == 3 and verdict["mtta_median_seconds"] == 40.0


def test_cli_one_seed_without_harvest_refuses_the_whole_mtta(tmp_path):
    host, queue = _series(tmp_path)
    a = _seed(tmp_path, "a", t0=0, t1=100, episodes=[("a1", 20.0)])
    b = _seed(tmp_path, "b", t0=0, t1=100, episodes=[("b1", 40.0)])
    b.with_suffix(".episodes.json").unlink()
    verdict = cli.evaluate([a, b], queue, host)
    assert not verdict["passed"]
    queue_signal_out = verdict["signals"][1]
    assert queue_signal_out["status"] == "not_measured"
    assert "harvest incomplete" in queue_signal_out["reason"]


def test_cli_without_a_journal_has_no_window_and_fails(tmp_path, capsys):
    host, queue = _series(tmp_path)
    a = _seed(tmp_path, "a", t0=0, t1=100, episodes=[("a1", 20.0)])
    a.with_suffix(".events.jsonl").unlink()
    rc = cli.main(["--seed-out", str(a), "--queue", str(queue), "--host", str(host)])
    assert rc == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["window"] is None and "does not exist" in verdict["reasons"][0]


def test_cli_records_the_unproven_steal_acceptance_in_the_verdict(tmp_path):
    _, queue_path = _series(tmp_path)
    host_path = tmp_path / "host-blind.jsonl"
    for t in (0, 30, 60, 90, 100):
        write_footprint_sample_jsonl(host_path, _host(t, total=1000 * (t + 1), steal=0))
    a = _seed(tmp_path, "a", t0=0, t1=100, episodes=[("a1", 20.0)])
    base = ["--seed-out", str(a), "--queue", str(queue_path), "--host", str(host_path)]
    refused, accepted = tmp_path / "refused.json", tmp_path / "accepted.json"
    assert cli.main(base + ["--out", str(refused)]) == 1
    assert cli.main(base + ["--out", str(accepted), "--accept-unproven-steal"]) == 0
    refused_v, accepted_v = json.loads(refused.read_text()), json.loads(accepted.read_text())
    assert refused_v["accepted_unproven_steal"] is False
    assert refused_v["signals"][0]["status"] == "not_measured"
    assert accepted_v["accepted_unproven_steal"] is True and accepted_v["passed"]
    assert "unproven" in accepted_v["signals"][0]["reason"]
