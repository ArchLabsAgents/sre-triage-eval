"""Detectability pass: same observation rules as the driver (namespace-scoped alerts, settle before inject)."""

from __future__ import annotations

import json

import pytest

from scenarios.run_continuous_admission import R1Error
from scenarios.run_detectability import run_r1a

DIRTY = ["alert KubePodCrashLooping pending pod=geo-c47ff745-crs4h"]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeAlerts:
    def __init__(self):
        self.firing: dict[str, dict] = {}
        self.namespace = None
        self.unsettled: list[list[str]] = []
        self.always_unsettled: list[str] = []

    def firing_alerts(self):
        return dict(self.firing)

    def firing_fingerprints(self):
        return set(self.firing)

    def unsettled_signals(self):
        if self.unsettled:
            return list(self.unsettled.pop(0))
        return list(self.always_unsettled)


class FakeInjector:
    def __init__(self, alerts, *, silent=(), namespace="test-hotel-reservation", fail_prepare=False):
        self.alerts = alerts
        self.silent = set(silent)
        self.namespace = namespace
        self.fail_prepare = fail_prepare
        self.calls: list[str] = []

    def prepare(self):
        self.calls.append("prepare")
        if self.fail_prepare:
            raise RuntimeError("openebs never became ready")
        return {"app_namespace": self.namespace} if self.namespace else {}

    def inject(self, family):
        self.calls.append(f"inject:{family}")
        if family not in self.silent:
            self.alerts.firing = {f"fp-{family}": {"alertname": "KubePodNotReady", "labels": {"pod": "geo-1"}}}

    def recover(self):
        self.calls.append("recover")
        self.alerts.firing = {}
        return {"baseline_restored": False}

    def cleanup(self):
        self.calls.append("cleanup")


def _run(alerts, injector, tmp_path, families=("a", "b")):
    return run_r1a(
        families,
        injector=injector,
        alerts=alerts,
        clock=FakeClock(),
        alert_timeout_seconds=30,
        poll_seconds=5,
        settle_seconds=10,
        out_path=tmp_path / "r1a.json",
    )


def test_detected_and_not_detected_scoped_to_the_app_namespace(tmp_path):
    alerts = FakeAlerts()
    injector = FakeInjector(alerts, silent={"b"})

    results = _run(alerts, injector, tmp_path)

    assert alerts.namespace == "test-hotel-reservation"  # alerts scoped to the app namespace
    assert [(r.family, r.verdict) for r in results] == [("a", "detected"), ("b", "not_detected")]
    assert results[0].observed_alerts[0]["alertname"] == "KubePodNotReady"
    assert all(r.settle["settled"] for r in results)  # the settle check ran before each injection
    assert injector.calls == ["prepare", "inject:a", "recover", "inject:b", "recover", "cleanup"]
    # The revision-injector live revalidation reads recover_detail to confirm "recovered by undo", not just that
    # injector.recover() did not raise.
    assert all(r.recover_detail == {"baseline_restored": False} for r in results)
    written = json.loads((tmp_path / "r1a.json").read_text())
    assert written["complete"] is True and len(written["families"]) == 2


def test_a_family_whose_app_never_settles_is_not_injected_and_not_measured(tmp_path):
    alerts = FakeAlerts()
    alerts.always_unsettled = DIRTY
    injector = FakeInjector(alerts)

    results = _run(alerts, injector, tmp_path, families=("a",))

    assert results[0].verdict == "not_measured" and results[0].injected is False
    assert results[0].settle["settled"] is False and results[0].settle["last_dirty"] == DIRTY
    assert injector.calls == ["prepare", "cleanup"]


def test_settle_waits_out_a_dirty_app_then_injects(tmp_path):
    alerts = FakeAlerts()
    alerts.unsettled = [DIRTY] * 3  # dirty for the first 3 polls, clean afterwards
    injector = FakeInjector(alerts)

    results = _run(alerts, injector, tmp_path, families=("a",))

    assert results[0].verdict == "detected"
    assert results[0].settle["settled"] is True and results[0].settle["last_dirty"] == DIRTY


@pytest.mark.parametrize("kwargs", [{"namespace": None}, {"fail_prepare": True}])
def test_prepare_problems_refuse_the_pass_and_still_clean_up(tmp_path, kwargs):
    alerts = FakeAlerts()
    injector = FakeInjector(alerts, **kwargs)

    with pytest.raises((R1Error, RuntimeError)):
        _run(alerts, injector, tmp_path)

    assert injector.calls == ["prepare", "cleanup"]  # never injected, never left deployed
    written = json.loads((tmp_path / "r1a.json").read_text())
    assert written["complete"] is False  # an aborted pass is never marked complete


# --- main(): --families scopes both the pass and the injector, for a short revalidation run -----


def _main_args(tmp_path, *extra):
    return [
        "--alertmanager-url", "http://am.example",
        "--prometheus-url", "http://prom.example",
        "--out", str(tmp_path / "out.json"),
        *extra,
    ]


def test_main_families_flag_scopes_the_pass_and_the_injector_to_the_revision_families(monkeypatch, tmp_path):
    import scenarios.run_detectability as r1a_module

    captured = {}

    def fake_run_r1a(families, *, injector, **kwargs):
        captured["families"] = families
        captured["injector_families"] = injector._families
        return []

    monkeypatch.setattr(r1a_module, "run_r1a", fake_run_r1a)
    monkeypatch.setattr(r1a_module, "HttpAlertGateway", lambda *a, **k: None)

    r1a_module.main(
        _main_args(tmp_path, "--families", "wrong_bin_usage", "assign_to_non_existent_node_hotel_res")
    )

    expected = ("assign_to_non_existent_node_hotel_res", "wrong_bin_usage")  # sorted
    assert captured["families"] == expected
    assert captured["injector_families"] == expected


def test_main_without_families_flag_still_runs_every_admitted_family(monkeypatch, tmp_path):
    import scenarios.run_detectability as r1a_module
    from adapters.mitigation_probes import PROBES

    captured = {}

    def fake_run_r1a(families, *, injector, **kwargs):
        captured["families"] = families
        return []

    monkeypatch.setattr(r1a_module, "run_r1a", fake_run_r1a)
    monkeypatch.setattr(r1a_module, "HttpAlertGateway", lambda *a, **k: None)

    r1a_module.main(_main_args(tmp_path))

    assert captured["families"] == tuple(sorted(PROBES))


def test_main_families_flag_rejects_a_candidate_or_unknown_family(tmp_path):
    import scenarios.run_detectability as r1a_module

    with pytest.raises(SystemExit):
        r1a_module.main(_main_args(tmp_path, "--families", "container_kill"))
