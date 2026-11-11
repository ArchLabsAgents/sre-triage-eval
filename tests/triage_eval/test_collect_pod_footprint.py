"""Unit coverage for scenarios/collect_pod_footprint.py, the Prometheus/cAdvisor query side.
Mocks urllib.request.urlopen: no live Prometheus needed. Each test drives
collect_one_tick() directly rather than the sleep loop (run_loop is exercised separately for its
own wiring, with the loop's `_sleep` stubbed out)."""

from __future__ import annotations

import json
import urllib.error

import pytest

from harness.metrics import load_pod_footprint_samples
from scenarios import collect_pod_footprint as mod


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _success_payload(value: str) -> dict:
    return {
        "status": "success",
        "data": {"resultType": "vector", "result": [{"metric": {}, "value": [1700000000, value]}]},
    }


def _empty_payload() -> dict:
    return {"status": "success", "data": {"resultType": "vector", "result": []}}


def test_collect_one_tick_success_marks_collected_true(monkeypatch):
    responses = [_success_payload("0.42"), _success_payload("123456789")]

    def fake_urlopen(url, timeout=None):
        return _FakeResponse(responses.pop(0))

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    sample = mod.collect_one_tick("http://localhost:9090", "sslm-agent", now=42.0)

    assert sample.collected is True
    assert sample.timestamp == 42.0
    assert sample.cpu_cores == pytest.approx(0.42)
    assert sample.memory_bytes == pytest.approx(123456789)
    assert sample.reason is None


def test_collect_one_tick_transport_failure_is_not_collected_never_zero(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    sample = mod.collect_one_tick("http://localhost:9090", "sslm-agent", now=1.0)

    assert sample.collected is False
    assert sample.cpu_cores is None
    assert sample.memory_bytes is None
    assert "connection refused" in sample.reason


def test_collect_one_tick_no_series_is_not_collected_with_reason(monkeypatch):
    def fake_urlopen(url, timeout=None):
        return _FakeResponse(_empty_payload())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    sample = mod.collect_one_tick("http://localhost:9090", "sslm-agent", now=1.0)

    assert sample.collected is False
    assert "no series" in sample.reason


def test_collect_one_tick_partial_series_is_not_collected_never_half_filled(monkeypatch):
    # Regression: only one of the two queries (cpu, memory) returning a series must still mark
    # the whole tick uncollected, not a sample with the other field silently None. Realistic on
    # pod startup: the CPU query is a rate() over a 2m window and needs scrape history, while the
    # memory gauge is available immediately.
    responses = [_success_payload("0.5"), _empty_payload()]

    def fake_urlopen(url, timeout=None):
        return _FakeResponse(responses.pop(0))

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    sample = mod.collect_one_tick("http://localhost:9090", "sslm-agent", now=1.0)

    assert sample.collected is False
    assert sample.cpu_cores is None
    assert sample.memory_bytes is None
    assert "memory" in sample.reason


def test_collect_one_tick_prometheus_error_status_is_not_collected(monkeypatch):
    def fake_urlopen(url, timeout=None):
        return _FakeResponse({"status": "error", "error": "bad query"})

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    sample = mod.collect_one_tick("http://localhost:9090", "sslm-agent", now=1.0)

    assert sample.collected is False
    assert "status='error'" in sample.reason or "status=\"error\"" in sample.reason or "error" in sample.reason


def test_collect_one_tick_malformed_result_is_not_collected(monkeypatch):
    def fake_urlopen(url, timeout=None):
        return _FakeResponse(
            {"status": "success", "data": {"result": [{"metric": {}, "value": ["not-a-timestamp"]}]}}
        )

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    sample = mod.collect_one_tick("http://localhost:9090", "sslm-agent", now=1.0)

    assert sample.collected is False


def test_run_loop_once_writes_a_single_line_and_returns(tmp_path, monkeypatch):
    def fake_urlopen(url, timeout=None):
        return _FakeResponse(_success_payload("0.1"))

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    out = tmp_path / "pod.jsonl"

    sleeps = []
    mod.run_loop(
        "http://localhost:9090",
        "sslm-agent",
        out,
        once=True,
        _sleep=lambda s: sleeps.append(s),
    )

    assert sleeps == []  # --once nunca dorme
    samples = load_pod_footprint_samples(out)
    assert len(samples) == 1
    assert samples[0].collected is True


def test_run_loop_without_once_sleeps_between_ticks(tmp_path, monkeypatch):
    def fake_urlopen(url, timeout=None):
        return _FakeResponse(_success_payload("0.1"))

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    out = tmp_path / "pod.jsonl"

    calls = {"n": 0}

    def fake_sleep(seconds):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise SystemExit("stop the loop for the test")

    with pytest.raises(SystemExit):
        mod.run_loop("http://localhost:9090", "sslm-agent", out, interval_seconds=5.0, _sleep=fake_sleep)

    assert calls["n"] == 2
    samples = load_pod_footprint_samples(out)
    assert len(samples) == 2
