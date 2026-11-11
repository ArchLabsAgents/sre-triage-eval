"""vLLM /metrics scrape and the TTFT p95 ceiling the contention gate reads."""

from __future__ import annotations

import http.client
import json
import urllib.error

import pytest

from harness.metrics import (
    EpisodeTokenUsage,
    VllmQueueSample,
    load_episode_usage,
    load_vllm_queue_samples,
    parse_prometheus_text,
    queue_sample_from_metrics,
    token_slice_cost,
    ttft_p95_ceiling,
)
from scenarios import log_vllm_footprint as mod

# vLLM v0.6.3 TTFT buckets (vllm/engine/metrics.py at that tag).
LES = ["0.001", "0.005", "0.01", "0.02", "0.04", "0.06", "0.08", "0.1", "0.25", "0.5",
       "0.75", "1.0", "2.5", "5.0", "7.5", "10.0", "+Inf"]
MODEL = 'model_name="Qwen/Qwen2.5-7B-Instruct"'


def _metrics(cumulative: dict[str, float] | None = None, *, waiting=0.0, running=0.0, prompt=100.0, gen=50.0,
             extra: str = "") -> str:
    cumulative = cumulative or {}
    lines = [
        "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{{{MODEL}}} {waiting}",
        f"vllm:num_requests_running{{{MODEL}}} {running}",
        f"vllm:prompt_tokens_total{{{MODEL}}} {prompt}",
        f"vllm:generation_tokens_total{{{MODEL}}} {gen}",
        "# TYPE vllm:time_to_first_token_seconds histogram",
    ]
    running_total = 0.0
    for le in LES:
        running_total = cumulative.get(le, running_total)
        lines.append(f'vllm:time_to_first_token_seconds_bucket{{le="{le}",{MODEL}}} {running_total}')
    lines.append(f"vllm:time_to_first_token_seconds_count{{{MODEL}}} {running_total}")
    lines.append(f"vllm:time_to_first_token_seconds_sum{{{MODEL}}} {running_total * 0.4}")
    return "\n".join(lines) + "\n" + extra


def _sample(t: float, cumulative: dict[str, float], **kw) -> VllmQueueSample:
    return queue_sample_from_metrics(_metrics(cumulative, **kw), timestamp=t)


def test_parser_reads_labels_including_quoted_commas_and_skips_comments():
    text = '# HELP x y\nfoo{a="1,2",b="x\\"y"} 3.5\nbar 7\n\n'
    parsed = parse_prometheus_text(text)
    assert parsed["foo"] == [({"a": "1,2", "b": 'x\\"y'}, 3.5)]
    assert parsed["bar"] == [({}, 7.0)]


def test_sample_from_a_real_shaped_scrape_keeps_cumulative_buckets_verbatim():
    sample = _sample(10.0, {"0.5": 3, "1.0": 8, "+Inf": 10}, waiting=2, running=1)
    assert sample.collected
    assert sample.ttft_buckets["0.25"] == 0 and sample.ttft_buckets["0.5"] == 3
    assert sample.ttft_buckets["10.0"] == 8 and sample.ttft_buckets["+Inf"] == 10
    assert sample.ttft_count == 10
    assert (sample.num_requests_waiting, sample.num_requests_running) == (2, 1)
    assert (sample.prompt_tokens_total, sample.generation_tokens_total) == (100, 50)


@pytest.mark.parametrize(
    "body, reason",
    [
        ("vllm:num_requests_waiting{broken 1\n", "unparsable"),
        (_metrics().replace("vllm:num_requests_running", "vllm:other"), "vllm:num_requests_running"),
        (_metrics(extra=f'vllm:num_requests_waiting{{model_name="other"}} 1\n'), "expected 1 series, found 2"),
        (_metrics(extra='vllm:time_to_first_token_seconds_bucket{le="0.5",model_name="other"} 1\n'), "repeated le"),
        ("\n".join(l for l in _metrics().splitlines() if 'le="+Inf"' not in l), "no +Inf bucket"),
    ],
)
def test_scrape_that_is_garbled_partial_or_multi_model_is_not_collected(body, reason):
    sample = queue_sample_from_metrics(body, timestamp=1.0)
    assert not sample.collected and reason in sample.reason


def test_ceiling_is_the_upper_edge_of_the_bucket_holding_the_95th_request():
    base = _sample(0.0, {"0.5": 5, "+Inf": 5})
    # 20 new requests: 18 under 1.0s, 19 under 2.5s, all 20 under 5.0s. 0.95 * 20 = 19.
    after = _sample(100.0, {"0.5": 10, "1.0": 23, "2.5": 24, "5.0": 25, "+Inf": 25}, waiting=3)
    ceiling = ttft_p95_ceiling([base, after], start=0.0, end=100.0)
    assert ceiling.requests == 20
    assert ceiling.p95_ceiling_seconds == 2.5
    assert ceiling.max_waiting == 3
    assert ceiling.reason is None


def test_baseline_is_the_last_sample_before_the_window_and_the_end_the_last_inside():
    samples = [
        _sample(-60.0, {"0.5": 1, "+Inf": 1}),
        _sample(-5.0, {"0.5": 100, "+Inf": 100}),  # baseline
        _sample(50.0, {"0.5": 110, "+Inf": 110}),  # end
        _sample(500.0, {"0.5": 110, "10.0": 900, "+Inf": 900}),  # after the window, ignored
    ]
    ceiling = ttft_p95_ceiling(samples, start=0.0, end=100.0)
    assert ceiling.requests == 10 and ceiling.p95_ceiling_seconds == 0.5


def test_p95_past_the_last_finite_bucket_is_not_measured_never_ten_seconds():
    base = _sample(0.0, {"+Inf": 0})
    after = _sample(60.0, {"10.0": 90, "+Inf": 100})
    ceiling = ttft_p95_ceiling([base, after], start=0.0, end=60.0)
    assert ceiling.p95_ceiling_seconds is None and ceiling.requests == 100
    assert "past the largest finite bucket" in ceiling.reason


@pytest.mark.parametrize(
    "samples, reason",
    [
        ([], "at or before the window start"),
        ([VllmQueueSample(timestamp=0.0, collected=False, reason="down")], "at or before the window start"),
        ([_sample(0.0, {"+Inf": 5})], "inside the window"),
        ([_sample(0.0, {"+Inf": 5}), _sample(60.0, {"+Inf": 5})], "no request"),
        ([_sample(0.0, {"0.5": 50, "+Inf": 50}), _sample(60.0, {"0.5": 2, "+Inf": 2})], "went backwards"),
        ([_sample(-120.0, {"+Inf": 0}), _sample(60.0, {"0.5": 2, "+Inf": 2})], "before the window start"),
        ([_sample(0.0, {"+Inf": 0}), _sample(10.0, {"0.5": 2, "+Inf": 2})], "300s before the window end"),
    ],
)
def test_ceiling_refuses_to_guess(samples, reason):
    end = 310.0 if "window end" in reason else 60.0
    ceiling = ttft_p95_ceiling(samples, start=0.0, end=end)
    assert ceiling.p95_ceiling_seconds is None and reason in ceiling.reason


def test_not_collected_samples_are_skipped_when_picking_baseline_and_end():
    samples = [
        _sample(-10.0, {"0.5": 0, "+Inf": 0}),
        VllmQueueSample(timestamp=30.0, collected=False, reason="timeout"),
        _sample(40.0, {"0.5": 4, "+Inf": 4}),
        VllmQueueSample(timestamp=59.0, collected=False, reason="timeout"),
    ]
    assert ttft_p95_ceiling(samples, start=0.0, end=60.0).p95_ceiling_seconds == 0.5


def test_not_collected_sample_requires_a_reason():
    with pytest.raises(ValueError):
        VllmQueueSample(timestamp=1.0, collected=False)


def test_logger_scrape_turns_network_failure_into_a_reason():
    def boom(url):
        raise urllib.error.URLError("connection refused")

    sample = mod.scrape_vllm_metrics("http://127.0.0.1:8000/metrics", now=5.0, _fetch=boom)
    assert not sample.collected and "connection refused" in sample.reason


@pytest.mark.parametrize("exc", [http.client.IncompleteRead(b"partial"), ValueError("unknown url type")])
def test_logger_scrape_survives_a_dropped_body_or_a_bad_url(exc):
    def boom(url):
        raise exc

    sample = mod.scrape_vllm_metrics("http://127.0.0.1:8000/metrics", now=5.0, _fetch=boom)
    assert not sample.collected and sample.reason


def test_loop_writes_the_queue_series_to_its_own_file_even_when_the_gpu_tick_fails(tmp_path):
    def no_gpu():
        raise mod.NvidiaSmiError("nvidia-smi not found on PATH")

    footprint, queue = tmp_path / "fp.jsonl", tmp_path / "queue.jsonl"
    mod.run_loop(
        "sslm-vllm",
        footprint,
        once=True,
        metrics_url="http://x/metrics",
        metrics_out=queue,
        _nvidia_smi=no_gpu,
        _resolve_cgroup=lambda name, **kwargs: tmp_path / "no-cgroup",
        _fetch=lambda url: _metrics({"0.5": 1, "+Inf": 1}),
    )
    assert json.loads(footprint.read_text())["collected"] is False
    (sample,) = load_vllm_queue_samples(queue)
    assert sample.collected and sample.ttft_buckets["+Inf"] == 1


def test_loop_without_metrics_flags_writes_no_queue_series(tmp_path):
    mod.run_loop(tmp_path / "cg", tmp_path / "fp.jsonl", once=True,
                 _nvidia_smi=lambda: (1.0, 2.0, 3.0),
                 _resolve_cgroup=lambda name, **kwargs: tmp_path / "cg",
                 _cgroup=lambda p: (1.0, 2.0),
                 _fetch=lambda url: pytest.fail("scraped without --metrics-url"))
    assert not (tmp_path / "queue.jsonl").exists()


def test_metrics_flags_go_together():
    with pytest.raises(ValueError):
        mod.run_loop("cg", "fp.jsonl", once=True, metrics_url="http://x/metrics")


# ---------------------------------------------------------------------------------------------
# Cost attribution by token slice.


def _tok(t: float, prompt: float, gen: float) -> VllmQueueSample:
    return _sample(t, {"+Inf": 0}, prompt=prompt, gen=gen)


def _use(eid: str, t0: float, t1: float, prompt: int, completion: int, *, missing: int = 0) -> EpisodeTokenUsage:
    return EpisodeTokenUsage(eid, t0, t1, prompt, completion, calls=3, calls_without_usage=missing)


def test_window_price_is_spread_over_server_tokens_and_each_episode_pays_its_own():
    # One hour at US$1/h, 1000 server tokens (800 prompt + 200 gen): US$0.001 per token.
    samples = [_tok(0.0, 0, 0), _tok(3600.0, 800, 200)]
    usage = [_use("seed1-ep", 100, 200, 300, 100), _use("seed2-ep", 150, 400, 200, 50)]

    cost = token_slice_cost(samples, usage, 0.0, 3600.0, price_usd_per_hour=1.0)

    assert cost.reason is None
    assert cost.usd_window == pytest.approx(1.0)
    assert (cost.server_prompt_tokens, cost.server_generation_tokens) == (800, 200)
    assert cost.usd_per_token == pytest.approx(0.001)
    assert [(e.episode_id, e.usd) for e in cost.episodes] == [
        ("seed1-ep", pytest.approx(0.4)), ("seed2-ep", pytest.approx(0.25)),
    ]
    assert cost.unattributed_tokens == 350
    assert cost.usd_unattributed == pytest.approx(0.35)
    assert sum(e.usd for e in cost.episodes) + cost.usd_unattributed == pytest.approx(cost.usd_window)


def test_straddling_or_incomplete_episodes_stay_unattributed_never_smeared():
    samples = [_tok(0.0, 0, 0), _tok(3600.0, 900, 100)]
    usage = [
        _use("inside", 10, 20, 100, 0),
        _use("straddles", -30, 20, 100, 0),
        _use("no-usage", 30, 40, 100, 0, missing=1),
        _use("outside", 4000, 4100, 999, 999),
    ]

    cost = token_slice_cost(samples, usage, 0.0, 3600.0, price_usd_per_hour=1.0)

    by_id = {e.episode_id: e for e in cost.episodes}
    assert set(by_id) == {"inside", "straddles", "no-usage"}
    assert by_id["inside"].usd == pytest.approx(0.1)
    assert by_id["straddles"].usd is None and "straddles" in by_id["straddles"].reason
    assert by_id["no-usage"].usd is None and "incomplete" in by_id["no-usage"].reason
    assert cost.unattributed_tokens == 900


@pytest.mark.parametrize(
    "samples, usage, reason",
    [
        ([], [], "at or before the window start"),
        ([_tok(0.0, 500, 50), _tok(3600.0, 10, 5)], [], "went backwards"),
        ([_tok(0.0, 10, 5), _tok(3600.0, 10, 5)], [], "no token reached"),
        ([_tok(0.0, 0, 0), _tok(3600.0, 90, 10)], [_use("greedy", 10, 20, 150, 0)], "more than the 100"),
        ([_tok(0.0, 0, 0), _tok(3600.0, 90, 10)], [_use("dup", 1, 2, 1, 0), _use("dup", 3, 4, 1, 0)], "repeated"),
        ([VllmQueueSample(0.0, True, {"+Inf": 0}, 0, 0, 0, 0), _tok(3600.0, 90, 10)], [], "missing from an edge"),
    ],
)
def test_token_slice_refuses_to_price_what_it_did_not_measure(samples, usage, reason):
    cost = token_slice_cost(samples, usage, 0.0, 3600.0, price_usd_per_hour=1.0)
    assert cost.usd_per_token is None and reason in cost.reason
    assert cost.usd_window == pytest.approx(1.0)  # hours x list price is known either way
    assert all(e.usd is None for e in cost.episodes)


def test_token_slice_rejects_an_empty_window_and_loads_the_sidecar(tmp_path):
    with pytest.raises(ValueError):
        token_slice_cost([], [], 10.0, 10.0)
    path = tmp_path / "triage.jsonl.usage.jsonl"
    path.write_text(json.dumps({"episode_id": "e", "t_start": 1.0, "t_end": 2.0, "prompt_tokens": 3,
                                "completion_tokens": 4, "calls": 1, "calls_without_usage": 0}) + "\n")
    assert load_episode_usage(path) == [EpisodeTokenUsage("e", 1.0, 2.0, 3, 4, 1, 0)]
    assert load_episode_usage(tmp_path / "absent") == []
