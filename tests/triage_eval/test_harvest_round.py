"""scenarios/harvest_round.py pulls SSLM_TRIAGE_LOG off the agent pod and joins it."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime

from scenarios.harvest_round import NO_RECORD_YET, SINK_NOT_CONFIGURED, harvest, main

T0 = 1_790_000_000.0
SPM = 60.0
ROUND_ID = "seed7-20260928T000000Z"
FP = "fp-checkout"
LABELS = {"alertname": "PodNotReady", "namespace": "hotel", "pod": "frontend-1"}
POD = "sslm-agent-7d9f-abcde"


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


def _at(minute: float) -> float:
    return T0 + minute * SPM


def _journal() -> list[dict]:
    family = "misconfig_app_hotel_res"
    return [
        {"phase": "prepare_start", "t": T0 - 30},
        {"phase": "clock_started", "t": T0, "journal_version": 3, "rodada_id": ROUND_ID, "seed": 7,
         "seconds_per_minute": SPM, "app_namespace": "hotel"},
        {"phase": "inject_start", "t": _at(30), "family": family, "start_minute": 30, "end_minute": 60},
        {"phase": "inject_done", "t": _at(30) + 5, "family": family},
        {"phase": "alert_observed", "t": _at(30) + 120, "family": family,
         "alerts": [{"fingerprint": FP, "alertname": LABELS["alertname"], "labels": LABELS}]},
        {"phase": "family_done", "t": _at(30) + 700, "family": family, "injected": True, "error": None,
         "start_minute": 30},
    ]


def _record() -> dict:
    t = _at(33)
    return {"episode_id": "ep-a", "alert_fingerprint": FP, "t_alarme": _iso(t), "t_record": _iso(t + 60),
            "causa_provavel": {"classe": "app_image_misconfig", "detalhe": "d"}, "decisao": "escalar"}


def _write_journal(out, events):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


class FakeKube:
    def __init__(self, *, created: float = T0 - 3600, pods: int = 1, read_rc: int = 0, read_out: str = "",
                 get_rc: int = 0, usage_rc: int | None = None, usage_out: str = ""):
        self.created, self.pods, self.read_rc, self.read_out, self.get_rc = created, pods, read_rc, read_out, get_rc
        # Default: the usage sidecar is present exactly when the triage log is (same image).
        self.usage_rc = read_rc if usage_rc is None else usage_rc
        self.usage_out = usage_out
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(argv)
        if argv[3] == "get":
            items = [{"metadata": {"name": POD, "creationTimestamp": _iso(self.created)}}] * self.pods
            return subprocess.CompletedProcess(argv, self.get_rc, json.dumps({"items": items}), "boom")
        assert argv[3] == "exec" and argv[4] == POD
        if ".usage.jsonl" in argv[-1]:
            return subprocess.CompletedProcess(argv, self.usage_rc, self.usage_out, "usage exec failed")
        return subprocess.CompletedProcess(argv, self.read_rc, self.read_out, "exec failed")


def test_records_are_copied_verbatim_and_joined_into_episodes(tmp_path):
    out = tmp_path / "continuous" / "r1-admission.json"
    _write_journal(out, _journal())
    raw = json.dumps(_record()) + "\n"

    report = harvest(out, run=FakeKube(read_out=raw))

    assert report["not_measured"] == []
    assert out.with_suffix(".triage.jsonl").read_text(encoding="utf-8") == raw
    episodes = json.loads(out.with_suffix(".episodes.json").read_text(encoding="utf-8"))
    assert episodes["rodada_id"] == ROUND_ID
    assert len(episodes["episodes"]) == 1
    assert report["episodes"] == str(out.with_suffix(".episodes.json"))


def _ready_series(last_minute: float, *, skip: tuple[float, float] | None = None):
    """A query_range result: ready=1 every 30s from the clock start, optionally with a hole."""
    values = []
    t = T0
    while t <= _at(last_minute):
        if skip is None or not (_at(skip[0]) <= t < _at(skip[1])):
            values.append([t, "1"])
        t += 30
    return [{"metric": {"pod": POD}, "values": values}]


def _journal_end_minute() -> float:
    return (_journal()[-1]["t"] - T0) / SPM


def test_availability_is_read_from_the_heartbeat_over_the_journal_window(tmp_path):
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())
    calls = []

    def query_range(promql, start, end, step):
        calls.append((promql, start, end, step))
        return _ready_series(_journal_end_minute())

    report = harvest(out, run=FakeKube(), query_range=query_range)

    assert report["availability"]["measured"] is True
    assert report["availability"]["availability"] == 1.0
    assert calls[0][0] == 'kube_pod_status_ready{namespace="sslm-agent",condition="true"}'
    assert (calls[0][1], calls[0][2]) == (T0 - 30, _journal()[-1]["t"])  # first and last journal line


def test_availability_counts_a_gap_longer_than_two_intervals_as_downtime(tmp_path):
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())

    report = harvest(out, run=FakeKube(),
                     query_range=lambda *_: _ready_series(_journal_end_minute(), skip=(20, 25)))

    avail = report["availability"]
    assert avail["outages"] == 1
    assert avail["downtime_seconds"] == 300.0
    assert avail["availability"] < 1.0


def test_availability_without_prometheus_is_not_measured_and_never_blocks_the_join(tmp_path):
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())

    report = harvest(out, run=FakeKube(read_out=json.dumps(_record()) + "\n"))

    assert report["availability"]["measured"] is False
    assert report["not_measured"] == []
    assert out.with_suffix(".episodes.json").exists()


def test_prometheus_failure_is_not_measured_with_the_reason(tmp_path):
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())

    def boom(*_):
        raise OSError("connection refused")

    report = harvest(out, run=FakeKube(), query_range=boom)

    assert report["availability"]["measured"] is False
    assert "connection refused" in report["availability"]["reason"]


def test_the_pod_reads_the_path_from_its_own_env(tmp_path):
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())
    kube = FakeKube(read_out="")

    harvest(out, run=kube)

    exec_call = kube.calls[1]
    assert exec_call[:5] == ["kubectl", "-n", "sslm-agent", "exec", POD]
    assert "$SSLM_TRIAGE_LOG" in exec_call[-1]
    assert "/var/lib/sslm" not in exec_call[-1]


def test_no_record_written_yet_is_an_empty_round_not_a_failure(tmp_path):
    """The sink creates the file on the first record; a round where the agent never triaged
    anything is a measured result (every fault a miss), not a harvest error."""
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())

    report = harvest(out, run=FakeKube(read_rc=NO_RECORD_YET))

    assert report["not_measured"] == []
    assert out.with_suffix(".triage.jsonl").read_text(encoding="utf-8") == ""
    assert out.with_suffix(".episodes.json").exists()


def test_unset_sink_env_is_not_measured_never_an_empty_round(tmp_path):
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())

    report = harvest(out, run=FakeKube(read_rc=SINK_NOT_CONFIGURED))

    assert "SSLM_TRIAGE_LOG is unset" in report["not_measured"][0]
    assert not out.with_suffix(".triage.jsonl").exists()
    assert not out.with_suffix(".episodes.json").exists()


def test_pod_replaced_mid_round_keeps_the_partial_copy_but_refuses_the_join(tmp_path):
    """A fresh emptyDir lost the earlier records; joining would score each of them as a miss."""
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())
    raw = json.dumps(_record()) + "\n"

    report = harvest(out, run=FakeKube(created=_at(10), read_out=raw))

    assert "after the round started" in report["not_measured"][0]
    assert out.with_suffix(".triage.jsonl").read_text(encoding="utf-8") == raw
    assert report["episodes"] is None
    assert not out.with_suffix(".episodes.json").exists()


def test_missing_journal_stops_before_touching_the_cluster(tmp_path):
    kube = FakeKube()

    report = harvest(tmp_path / "r1-admission.json", run=kube)

    assert "does not exist" in report["not_measured"][0]
    assert kube.calls == []


def test_zero_or_many_agent_pods_is_not_measured(tmp_path):
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())

    for pods in (0, 2):
        report = harvest(out, run=FakeKube(pods=pods))
        assert f"found {pods}" in report["not_measured"][0]


def test_failed_exec_is_not_measured(tmp_path):
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())

    report = harvest(out, run=FakeKube(read_rc=1))

    assert "exec failed" in report["not_measured"][0]
    assert not out.with_suffix(".episodes.json").exists()


def test_bridge_refusal_becomes_a_reason_not_a_crash(tmp_path):
    out = tmp_path / "r1-admission.json"
    _write_journal(out, [{"phase": "prepare_start", "t": T0 - 30}])  # no clock_started: nothing to join

    report = harvest(out, run=FakeKube())

    assert report["not_measured"] and "episode_bridge refused" in report["not_measured"][0]


def test_main_writes_the_report_and_exits_nonzero_when_anything_was_not_measured(tmp_path, monkeypatch):
    import scenarios.harvest_round as mod

    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())
    monkeypatch.setattr(mod, "_run", FakeKube(read_rc=SINK_NOT_CONFIGURED))

    assert main(["--out", str(out)]) == 1
    report = json.loads(out.with_suffix(".harvest.json").read_text(encoding="utf-8"))
    assert report["not_measured"]

    monkeypatch.setattr(mod, "_run", FakeKube(read_out=json.dumps(_record()) + "\n"))
    assert main(["--out", str(out)]) == 0


def test_hung_or_missing_kubectl_is_not_measured_and_the_report_is_still_written(tmp_path, monkeypatch):
    import scenarios.harvest_round as mod

    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())
    for exc in (subprocess.TimeoutExpired(["kubectl"], 120), FileNotFoundError("kubectl")):
        def raising(argv, *a, _exc=exc, **k):
            raise _exc

        monkeypatch.setattr(mod.subprocess, "run", raising)
        assert main(["--out", str(out)]) == 1
        report = json.loads(out.with_suffix(".harvest.json").read_text(encoding="utf-8"))
        assert type(exc).__name__ in report["not_measured"][0]


def test_truncated_journal_or_garbled_kubectl_output_is_not_measured(tmp_path):
    out = tmp_path / "r1-admission.json"
    out.with_suffix(".events.jsonl").write_text('{"phase": "prepare_st\n', encoding="utf-8")
    assert "unreadable" in harvest(out, run=FakeKube())["not_measured"][0]

    _write_journal(out, _journal())

    def garbled(argv):
        return subprocess.CompletedProcess(argv, 0, "not json", "")

    assert "unparsable" in harvest(out, run=garbled)["not_measured"][0]


def test_usage_sidecar_is_copied_verbatim_next_to_the_records(tmp_path):
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())
    usage = json.dumps({"episode_id": "ep-a", "prompt_tokens": 10}) + "\n"
    kube = FakeKube(read_out=json.dumps(_record()) + "\n", usage_out=usage)

    report = harvest(out, run=kube)

    assert report["not_measured"] == []
    assert out.with_suffix(".usage.jsonl").read_text(encoding="utf-8") == usage
    assert report["usage"] == str(out.with_suffix(".usage.jsonl"))
    usage_call = kube.calls[2]
    assert '"$SSLM_TRIAGE_LOG.usage.jsonl"' in usage_call[-1]


def test_records_without_a_usage_log_are_not_measured_but_still_joined(tmp_path):
    """An image older than per-episode usage: the cost slice cannot charge these episodes, the
    triage verdicts are still good."""
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())

    report = harvest(out, run=FakeKube(read_out=json.dumps(_record()) + "\n", usage_rc=NO_RECORD_YET))

    assert len(report["not_measured"]) == 1 and "no usage log" in report["not_measured"][0]
    assert report["usage"] is None
    assert out.with_suffix(".episodes.json").exists()


def test_failed_usage_read_is_not_measured_and_an_empty_round_has_an_empty_usage_file(tmp_path):
    out = tmp_path / "r1-admission.json"
    _write_journal(out, _journal())

    failed = harvest(out, run=FakeKube(read_out=json.dumps(_record()) + "\n", usage_rc=1))
    assert "usage exec failed" in failed["not_measured"][0]

    empty = harvest(out, run=FakeKube(read_rc=NO_RECORD_YET))
    assert empty["not_measured"] == []
    assert out.with_suffix(".usage.jsonl").read_text(encoding="utf-8") == ""
