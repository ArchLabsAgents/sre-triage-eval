"""Tests for the Alertmanager -> agent route/receiver wiring.

Two layers, same as test_symptom_alert_rules.py's own pattern:

1. Static assertions against `kube-prometheus-stack/values.yaml`'s `alertmanager.config` map
   (always runs, no external dependency).
2. A best-effort `helm template` render against the pinned chart, decoding the actual
   `alertmanager.yaml` Secret payload Alertmanager would load -- this is the only way to catch a
   config that is valid YAML but semantically broken (e.g. a sub-route naming a receiver that
   doesn't exist, which is exactly the bug an earlier draft of this file's own values.yaml change
   had: overriding `receivers` replaces the chart's default list wholesale, since Helm does not
   merge YAML lists, which silently deleted the `null` receiver the chart's own default Watchdog
   sub-route still pointed at). Skipped, not failed, when helm/network are unavailable, mirroring
   test_symptom_alert_rules.py's own skip discipline.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

KPS_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "kube-prometheus-stack"
VALUES_YAML = KPS_DIR / "values.yaml"
PREFLIGHT_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"
if str(PREFLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(PREFLIGHT_DIR))

# Imported, not copied: a hardcoded literal here would drift silently the moment
# harness/webhook.py's own contract changes (an earlier version of
# this file asserted a local literal against itself, which can never fail). Same sys.path setup
# as test_k8s_manifests.py's own harness.envelope import.
from harness.webhook import GROUP_WINDOW_SECONDS  # noqa: E402

AGENT_WEBHOOK_URL = "http://sslm-agent.sslm-agent.svc.cluster.local:8080/alerts"


def _load_values() -> dict:
    return yaml.safe_load(VALUES_YAML.read_text())


def _alertmanager_config() -> dict:
    values = _load_values()
    assert "config" in values["alertmanager"], "alertmanager.config missing: no route/receiver is wired"
    return values["alertmanager"]["config"]


# ---------------------------------------------------------------------------
# Static assertions against values.yaml (no external dependency)
# ---------------------------------------------------------------------------


def test_route_receiver_is_the_sslm_agent_receiver():
    config = _alertmanager_config()
    assert config["route"]["receiver"] == "sslm-agent"


def test_sslm_agent_receiver_is_a_webhook_config_pointing_at_the_agent_service():
    config = _alertmanager_config()
    receivers = {r["name"]: r for r in config["receivers"]}
    assert "sslm-agent" in receivers
    webhook_configs = receivers["sslm-agent"].get("webhook_configs")
    assert webhook_configs, "sslm-agent receiver must be a webhook_config, not any other integration type"
    assert len(webhook_configs) == 1
    assert webhook_configs[0]["url"] == AGENT_WEBHOOK_URL


def test_group_interval_matches_the_agent_dedup_window():
    # Doesn't have to be exactly 120s (the deduper already absorbs multiple calls inside its own
    # window -- see harness/webhook.py's module docstring), but values.yaml's own comment commits
    # to matching it exactly, so pin that here as a regression guard against the LIVE constant
    # (an earlier version asserted a local literal against itself,
    # which could never catch harness/webhook.py's own contract changing).
    assert GROUP_WINDOW_SECONDS == 120.0
    config = _alertmanager_config()
    group_interval = config["route"]["group_interval"]
    assert group_interval == "2m"


def test_repeat_interval_is_not_shorter_than_group_interval():
    # A repeat_interval shorter than group_interval would make Alertmanager re-send an unchanged
    # group faster than it re-sends a CHANGED one -- backwards from the intent (see values.yaml's
    # comment: repeats should be rare, changes should be prompt).
    config = _alertmanager_config()
    route = config["route"]
    assert route["repeat_interval"] == "4h"
    assert route["group_interval"] == "2m"


def test_watchdog_subroute_points_at_a_receiver_that_actually_exists():
    # Regression for the exact bug this test file's own docstring describes: a route stanza is
    # syntactically valid YAML even when it names a receiver absent from `receivers`, and
    # Alertmanager only catches that at load time, not at `helm lint` time.
    config = _alertmanager_config()
    receiver_names = {r["name"] for r in config["receivers"]}
    sub_routes = config["route"].get("routes", [])
    watchdog_routes = [r for r in sub_routes if any("Watchdog" in m for m in r.get("matchers", []))]
    assert watchdog_routes, "no sub-route found for the chart's always-firing Watchdog alert"
    for route in watchdog_routes:
        assert route["receiver"] in receiver_names, (
            f"Watchdog sub-route names receiver {route['receiver']!r}, not present in {receiver_names}"
        )


def test_no_route_or_subroute_names_an_undefined_receiver():
    # The general form of the check above: every receiver any route/sub-route names (including
    # the root route) must exist in `receivers`, full stop.
    config = _alertmanager_config()
    receiver_names = {r["name"] for r in config["receivers"]}
    named = {config["route"]["receiver"]}
    for route in config["route"].get("routes", []):
        named.add(route["receiver"])
    assert named <= receiver_names, f"route(s) name undefined receiver(s): {named - receiver_names}"


# ---------------------------------------------------------------------------
# helm template render against the pinned chart (best-effort, skipped without helm/network)
# ---------------------------------------------------------------------------


def _pinned_chart_version() -> str:
    text = (PREFLIGHT_DIR / "setup_cluster.sh").read_text()
    for line in text.splitlines():
        if line.strip().startswith('KPS_CHART_VERSION="${KPS_CHART_VERSION:-'):
            return line.split(":-", 1)[1].split("}", 1)[0]
    raise AssertionError("KPS_CHART_VERSION default not found in setup_cluster.sh")


def test_rendered_alertmanager_secret_has_no_dangling_receiver_reference(tmp_path):
    """Renders the real chart (pinned version) with this repo's values.yaml, decodes the
    alertmanager.yaml Secret payload it produces, and re-checks the dangling-receiver invariant
    against what Alertmanager would ACTUALLY load -- not just against values.yaml's own
    `alertmanager.config` map, which is only Helm's INPUT to that render (Helm deep-merges map
    keys with the chart's own defaults; a values.yaml override can look complete in isolation and
    still combine with the chart's defaults into a broken final config, which is exactly what
    happened here: overriding `receivers` without also keeping a `null` entry silently dropped
    the receiver the chart's own default Watchdog sub-route still references).

    Skipped, not failed, when helm isn't on PATH or the pull fails (no network) -- mirrors
    test_symptom_alert_rules.py's own `test_helm_lint_passes_against_the_pinned_chart_...`.
    """
    import shutil as _shutil

    if _shutil.which("helm") is None:
        pytest.skip("helm not on PATH: cannot verify chart rendering in this environment")

    version = _pinned_chart_version()
    subprocess.run(
        ["helm", "repo", "add", "prometheus-community", "https://prometheus-community.github.io/helm-charts"],
        capture_output=True,
        text=True,
    )
    pull = subprocess.run(
        [
            "helm", "pull", "prometheus-community/kube-prometheus-stack",
            "--version", version, "--untar", "--untardir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    if pull.returncode != 0:
        pytest.skip(f"could not pull pinned chart {version} (no network / repo unreachable): {pull.stderr[:200]}")

    render = subprocess.run(
        [
            "helm", "template", "sslm-kube-prometheus", str(tmp_path / "kube-prometheus-stack"),
            "--values", str(VALUES_YAML),
            "--show-only", "templates/alertmanager/secret.yaml",
        ],
        capture_output=True,
        text=True,
    )
    assert render.returncode == 0, render.stdout + render.stderr

    secret_doc = yaml.safe_load(render.stdout)
    rendered_config = yaml.safe_load(base64.b64decode(secret_doc["data"]["alertmanager.yaml"]).decode())

    receiver_names = {r["name"] for r in rendered_config["receivers"]}
    assert "sslm-agent" in receiver_names

    named = {rendered_config["route"]["receiver"]}
    for route in rendered_config["route"].get("routes", []):
        named.add(route["receiver"])
    assert named <= receiver_names, f"rendered config names undefined receiver(s): {named - receiver_names}"

    sslm_agent_receiver = next(r for r in rendered_config["receivers"] if r["name"] == "sslm-agent")
    assert sslm_agent_receiver["webhook_configs"][0]["url"] == AGENT_WEBHOOK_URL
