"""Tests for harness/webhook.py: payload validation and the 120s-group-window dedup contract."""

from __future__ import annotations

import threading

import pytest

from harness.webhook import (
    EpisodeDeduper,
    WebhookPayloadError,
    parse_webhook_payload,
)


def _payload(**overrides) -> dict:
    base = {
        "version": "4",
        "groupKey": '{}:{alertname="KubePodCrashLooping"}',
        "status": "firing",
        "receiver": "sslm-agent",
        "alerts": [
            {
                "status": "firing",
                "labels": {"alertname": "KubePodCrashLooping", "namespace": "hotel"},
                "annotations": {},
                "startsAt": "2026-09-21T12:00:00Z",
                "fingerprint": "fp-1",
            }
        ],
    }
    base.update(overrides)
    return base


# --- parse_webhook_payload ---


def test_parses_a_valid_payload():
    webhook = parse_webhook_payload(_payload())
    assert webhook.group_key == '{}:{alertname="KubePodCrashLooping"}'
    assert webhook.status == "firing"
    assert webhook.receiver == "sslm-agent"
    assert webhook.alert_fingerprints == ("fp-1",)


def test_rejects_non_dict_body():
    with pytest.raises(WebhookPayloadError, match="must be a JSON object"):
        parse_webhook_payload(["not", "a", "dict"])


@pytest.mark.parametrize("missing_key", ["groupKey", "status", "receiver", "alerts"])
def test_rejects_missing_top_level_key(missing_key):
    payload = _payload()
    del payload[missing_key]
    with pytest.raises(WebhookPayloadError):
        parse_webhook_payload(payload)


def test_rejects_empty_group_key():
    with pytest.raises(WebhookPayloadError, match="groupKey"):
        parse_webhook_payload(_payload(groupKey=""))


def test_rejects_unknown_status():
    with pytest.raises(WebhookPayloadError, match="status"):
        parse_webhook_payload(_payload(status="pending"))


def test_rejects_empty_alerts_list():
    with pytest.raises(WebhookPayloadError, match="alerts"):
        parse_webhook_payload(_payload(alerts=[]))


def test_rejects_alert_missing_fingerprint():
    payload = _payload()
    del payload["alerts"][0]["fingerprint"]
    with pytest.raises(WebhookPayloadError, match="fingerprint"):
        parse_webhook_payload(payload)


def test_rejects_alert_with_bad_status():
    payload = _payload()
    payload["alerts"][0]["status"] = "unknown"
    with pytest.raises(WebhookPayloadError, match="status"):
        parse_webhook_payload(payload)


def test_multiple_alerts_all_fingerprints_collected():
    payload = _payload(
        alerts=[
            {"status": "firing", "fingerprint": "fp-1"},
            {"status": "firing", "fingerprint": "fp-2"},
        ]
    )
    webhook = parse_webhook_payload(payload)
    assert webhook.alert_fingerprints == ("fp-1", "fp-2")


# --- EpisodeDeduper ---


class _FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_first_firing_webhook_for_a_group_opens_a_new_episode():
    deduper = EpisodeDeduper(clock=_FakeClock())
    webhook = parse_webhook_payload(_payload())
    result = deduper.ingest(webhook)
    assert result.is_new_episode is True
    assert result.episode_id
    assert result.new_fingerprints == ("fp-1",)


def test_second_webhook_within_120s_for_same_group_folds_into_same_episode():
    clock = _FakeClock()
    deduper = EpisodeDeduper(clock=clock)
    webhook = parse_webhook_payload(_payload())

    first = deduper.ingest(webhook)
    clock.advance(60.0)
    second = deduper.ingest(parse_webhook_payload(_payload(alerts=[
        {"status": "firing", "fingerprint": "fp-2"},
    ])))

    assert second.is_new_episode is False
    assert second.episode_id == first.episode_id
    assert second.new_fingerprints == ("fp-2",)


def test_repeated_identical_fingerprint_within_window_reports_no_new_fingerprints():
    clock = _FakeClock()
    deduper = EpisodeDeduper(clock=clock)
    webhook = parse_webhook_payload(_payload())

    deduper.ingest(webhook)
    clock.advance(10.0)
    result = deduper.ingest(webhook)

    assert result.is_new_episode is False
    assert result.new_fingerprints == ()


def test_webhook_after_window_closes_opens_a_new_episode():
    clock = _FakeClock()
    deduper = EpisodeDeduper(clock=clock)
    webhook = parse_webhook_payload(_payload())

    first = deduper.ingest(webhook)
    clock.advance(120.0)  # exactly at the boundary: no longer "< 120s", so it does not fold
    second = deduper.ingest(webhook)

    assert second.is_new_episode is True
    assert second.episode_id != first.episode_id


def test_different_groups_never_share_an_episode():
    clock = _FakeClock()
    deduper = EpisodeDeduper(clock=clock)
    first = deduper.ingest(parse_webhook_payload(_payload(groupKey="group-a")))
    second = deduper.ingest(parse_webhook_payload(_payload(groupKey="group-b")))
    assert first.episode_id != second.episode_id
    assert first.is_new_episode is True
    assert second.is_new_episode is True


def test_resolved_status_with_no_open_window_starts_nothing():
    deduper = EpisodeDeduper(clock=_FakeClock())
    webhook = parse_webhook_payload(_payload(status="resolved", alerts=[
        {"status": "resolved", "fingerprint": "fp-1"},
    ]))
    result = deduper.ingest(webhook)
    assert result.episode_id == ""
    assert result.is_new_episode is False


def test_expired_windows_are_evicted_not_kept_forever():
    # Review finding on the PR that introduced EpisodeDeduper: without pruning, _windows grows
    # one entry per distinct groupKey ever seen, for the life of a process meant to run 24/7.
    clock = _FakeClock()
    deduper = EpisodeDeduper(clock=clock)

    for i in range(50):
        deduper.ingest(parse_webhook_payload(_payload(groupKey=f"group-{i}")))
        clock.advance(200.0)  # each group's window is long expired by the time the next arrives

    assert len(deduper._windows) == 1  # only the most recent group's window is still live


def test_concurrent_ingest_for_a_brand_new_group_opens_exactly_one_episode():
    # Review finding on the PR that introduced EpisodeDeduper: entrypoint.py serves requests
    # via ThreadingHTTPServer, so ingest()'s check-then-act on self._windows is a real race,
    # not just a sequential call site. N threads racing to open the SAME brand-new group's
    # window must still converge on exactly one episode.
    deduper = EpisodeDeduper()  # real clock: every call lands inside the same live window
    thread_count = 32
    barrier = threading.Barrier(thread_count)
    results = [None] * thread_count

    def worker(index: int) -> None:
        webhook = parse_webhook_payload(_payload(alerts=[
            {"status": "firing", "fingerprint": f"fp-{index}"},
        ]))
        barrier.wait()  # maximize contention at the check-then-act point
        results[index] = deduper.ingest(webhook)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    episode_ids = {r.episode_id for r in results}
    new_episode_count = sum(1 for r in results if r.is_new_episode)
    assert len(episode_ids) == 1, f"expected exactly one episode, got {episode_ids}"
    assert new_episode_count == 1, f"expected exactly one is_new_episode=True, got {new_episode_count}"


def test_resolved_status_folds_into_an_already_open_window():
    clock = _FakeClock()
    deduper = EpisodeDeduper(clock=clock)
    firing = parse_webhook_payload(_payload())
    opened = deduper.ingest(firing)

    clock.advance(5.0)
    resolved = parse_webhook_payload(_payload(status="resolved", alerts=[
        {"status": "resolved", "fingerprint": "fp-1"},
    ]))
    result = deduper.ingest(resolved)

    assert result.episode_id == opened.episode_id
    assert result.is_new_episode is False
