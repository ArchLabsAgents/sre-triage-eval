"""Tests for the kube-prometheus-stack config and symptom-only alert rules.

The hard rule these tests enforce mechanically (not just by review): a symptom alert rule may
describe a generic K8s/RED/SLO condition, but must NEVER name a fault mechanism. Doing this by
hand review only leaves a gap: a mechanism-keyed rule would license tuning the instrument until
the known fault fires.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

KPS_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "kube-prometheus-stack"
VALUES_YAML = KPS_DIR / "values.yaml"
RULES_YAML = KPS_DIR / "rules" / "symptom-alerts.yaml"
PREFLIGHT_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"

import sys  # noqa: E402

if str(PREFLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(PREFLIGHT_DIR))

from adapters.mitigation_probes import CANDIDATE_PROBES, PROBES  # noqa: E402

# Mechanism-specific strings a symptom rule must never cite, beyond the fault-family names
# themselves (checked dynamically against PROBES below). Sourced from mitigation_probes.py's own
# docstring of what each injector actually does.
_MECHANISM_STRINGS = (
    "revokeRolesFromUser",
    "dropUser",
    "yinfangchen/geo:app3",  # BUGGY_GEO_IMAGE
    "MongoAuthRevoked",
    # Candidate families: the injector's pinned node name and the Chaos Mesh artifact. A detector
    # keyed to them is forbidden exactly as one keyed to a revoked role is.
    "extra-node",  # mitigation_probes.NON_EXISTENT_NODE
    "PodChaos",
    "podchaos",
    "container-kill",
    "chaos_mesh",
    "chaos-mesh",
)


def test_rules_file_is_valid_yaml_and_a_prometheusrule():
    doc = yaml.safe_load(RULES_YAML.read_text())
    assert doc["kind"] == "PrometheusRule"
    assert doc["spec"]["groups"], "PrometheusRule must declare at least one rule group"


def _rule_alert_and_expr_text() -> str:
    """Only the `alert:`/`expr:` fields of every parsed rule, never the surrounding YAML
    comments or docstrings -- those legitimately name fault families (e.g. FROZEN_RULES.md's own
    fault-family -> alert mapping table is the point of that file). The prohibition
    ('no rule may name the fault mechanism') is about the rule's own identity/expr, not
    about documentation that explains why a generic rule was chosen."""
    doc = yaml.safe_load(RULES_YAML.read_text())
    chunks = []
    for group in doc["spec"]["groups"]:
        for rule in group["rules"]:
            chunks.append(rule.get("alert", ""))
            chunks.append(rule.get("expr", ""))
    return "\n".join(chunks)


def test_no_rule_expr_names_a_fault_family():
    """Every admitted fault family's registry key (adapters/mitigation_probes.PROBES) must never
    appear in an alert's own name or expr -- checked against the live registry, not a copy, so a
    new fault family added later is covered automatically."""
    text = _rule_alert_and_expr_text()
    for family in (*PROBES, *CANDIDATE_PROBES):
        assert family not in text, f"rule expr/name cites fault family {family!r}: not symptom-only"


def test_no_rule_expr_names_a_known_mechanism_string():
    text = _rule_alert_and_expr_text()
    for mechanism in _MECHANISM_STRINGS:
        assert mechanism not in text, f"rule expr/name cites fault mechanism {mechanism!r}: not symptom-only"


def test_custom_rules_for_duration_is_tighter_than_the_chart_default():
    """The chart's own default (15m) would eat most of a scheduled fault's active window
    (20-45min, scenarios/fault_schedule.py) and most of the 15-minute episode timeout before
    ever firing (values.yaml's own comment). Guards the override, not the doc's prose about it."""
    values = yaml.safe_load(VALUES_YAML.read_text())
    custom_rules = values.get("customRules", {})
    for alert in ("KubePodCrashLooping", "KubePodNotReady"):
        assert alert in custom_rules, f"{alert} override missing from top-level customRules"
        for_duration = custom_rules[alert]["for"]
        assert for_duration.endswith("m")
        assert int(for_duration[:-1]) < 15, f"{alert} for={for_duration} is not tighter than the chart's 15m default"


def test_custom_rules_key_is_top_level_not_nested_under_default_rules():
    """Regression: a first version of this file nested `customRules` under `defaultRules`,
    which the chart silently ignores (the template reads `.Values.customRules` directly) --
    caught only by rendering the chart and inspecting the output, not by lint or lookig at the
    values file alone. See values.yaml's own comment on this."""
    values = yaml.safe_load(VALUES_YAML.read_text())
    assert "customRules" in values, "customRules must be a top-level key"
    assert "customRules" not in values.get("defaultRules", {}), (
        "customRules must not be nested under defaultRules -- the chart reads .Values.customRules directly"
    )


def test_grafana_disabled():
    """Grafana is optional: the agent never reads a dashboard."""
    values = yaml.safe_load(VALUES_YAML.read_text())
    assert values["grafana"]["enabled"] is False


def test_prometheus_avoids_the_dedicated_agent_node():
    """Test apparatus (Prometheus included) must not share the node whose
    footprint the scenario is measuring (the sslm-agent-tainted node)."""
    values = yaml.safe_load(VALUES_YAML.read_text())
    terms = values["prometheus"]["prometheusSpec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    exprs = [expr for term in terms for expr in term["matchExpressions"]]
    assert any(
        e["key"] == "role" and e["operator"] == "NotIn" and "sslm-agent" in e["values"] for e in exprs
    ), "prometheus must declare an anti-affinity away from role=sslm-agent"


@pytest.mark.parametrize("family", sorted(PROBES))
def test_every_admitted_fault_family_has_a_documented_mapping_in_frozen_rules(family):
    """The requirement 'every injector fires >= 1 symptom alert' as a documentation-completeness
    check: a unit test has no live cluster access, so it cannot prove a live fire -- what it CAN and must prove is
    that every admitted fault family is at least accounted for in FROZEN_RULES.md, confirmed or
    not, rather than silently absent."""
    frozen_rules = (KPS_DIR / "FROZEN_RULES.md").read_text()
    assert family in frozen_rules, f"{family!r} has no entry in FROZEN_RULES.md's fault-family mapping table"


@pytest.mark.parametrize("family", sorted(CANDIDATE_PROBES))
def test_every_candidate_family_is_documented_as_not_admitted_in_frozen_rules(family):
    """Candidate families are listed in their own section (expected alert, unconfirmed), never in the
    admitted mapping table.
    """
    markers = {
        "FROZEN_RULES.md": ("NOT admitted",),
    }
    for name, candidates in markers.items():
        text = (KPS_DIR / name).read_text()
        marker = next((m for m in candidates if m in text), None)
        assert marker is not None, f"{name} lacks any of {candidates}"
        section = text.split(marker, 1)
        assert len(section) == 2, f"{name} lacks the candidate section"
        assert family in section[1], f"{family!r} missing from {name}'s candidate section"
        assert f"| `{family}` |" not in section[0], f"{family!r} listed as admitted in {name}"


def _pinned_chart_version() -> str:
    text = (PREFLIGHT_DIR / "setup_cluster.sh").read_text()
    for line in text.splitlines():
        if line.strip().startswith('KPS_CHART_VERSION="${KPS_CHART_VERSION:-'):
            return line.split(":-", 1)[1].split("}", 1)[0]
    raise AssertionError("KPS_CHART_VERSION default not found in setup_cluster.sh")


def test_helm_lint_passes_against_the_pinned_chart_if_helm_and_network_are_available(tmp_path):
    """Best-effort: pulls the exact pinned chart version and renders it with these values
    through `helm lint`. Skipped, not failed, when helm isn't on PATH or the pull fails (no
    network, repo unreachable) -- this repo's CI does not install helm (.github/workflows/tests.yml), so this
    is a local/manual verification gate, not a CI-enforced one; skipping must say so, never
    silently pass as if it checked something."""
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

    result = subprocess.run(
        ["helm", "lint", str(tmp_path / "kube-prometheus-stack"), "--values", str(VALUES_YAML)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
