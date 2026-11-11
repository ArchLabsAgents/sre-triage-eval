"""check_egress_cidr: the COMPARATOR_ARM=1 preflight that warns when the API host does not
resolve into ANTHROPIC_EGRESS_CIDR (a stale range would silently block the paid arm)."""

import subprocess
import sys
from pathlib import Path

import check_egress_cidr as cec

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "check_egress_cidr.py"


def test_address_inside_the_cidr_is_ok():
    ok, message = cec.check("api.example", "192.0.2.0/23", resolve=lambda _h: ["192.0.3.10"])

    assert ok
    assert "192.0.3.10" in message


def test_no_address_inside_the_cidr_warns():
    ok, message = cec.check("api.example", "192.0.2.0/24", resolve=lambda _h: ["198.51.100.7"])

    assert not ok
    assert "WARNING" in message and "198.51.100.7" in message and "ANTHROPIC_EGRESS_CIDR" in message


def test_one_address_inside_is_enough_when_the_host_answers_from_several_ranges():
    ok, _ = cec.check("h", "192.0.2.0/24", resolve=lambda _h: ["198.51.100.7", "192.0.2.5"])

    assert ok


def test_resolution_failure_warns_instead_of_raising():
    def boom(_host):
        raise OSError("no dns")

    ok, message = cec.check("h", "192.0.2.0/24", resolve=boom)

    assert not ok
    assert "could not resolve" in message


def test_invalid_cidr_warns():
    ok, message = cec.check("h", "not-a-cidr", resolve=lambda _h: ["192.0.2.5"])

    assert not ok
    assert "CIDR" in message


def test_cli_always_exits_zero_even_on_a_mismatch():
    # setup_cluster.sh must never abort on this check; it only warns. 'invalid' hits the
    # no-network path (bad CIDR) so the test does not depend on DNS.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "api.example", "invalid"], capture_output=True, text=True
    )

    assert result.returncode == 0
    assert "WARNING" in result.stderr


def test_setup_cluster_runs_the_check_only_in_the_comparator_branch():
    text = (SCRIPT.parent / "setup_cluster.sh").read_text(encoding="utf-8")
    branches = text.split('if [[ "${COMPARATOR_ARM:-0}" == "1" ]]; then')[1:]
    (comparator_branch,) = [
        b.split("else", 1)[0] for b in branches if "networkpolicy-comparator.yaml" in b.split("else", 1)[0]
    ]

    assert 'check_egress_cidr.py" api.anthropic.com "$ANTHROPIC_EGRESS_CIDR" || true' in comparator_branch
    assert text.count("check_egress_cidr.py") == 1


def test_setup_cluster_sets_history_budget_off_only_for_the_comparator_arm():
    # The default agent never gets SSLM_HISTORY_BUDGET=off (only the API comparator arm keeps
    # the earlier prompt), and the value is set by the script itself, not inherited from the
    # caller's shell.
    text = (SCRIPT.parent / "setup_cluster.sh").read_text(encoding="utf-8")
    block = (
        '  if [[ "${COMPARATOR_ARM:-0}" == "1" ]]; then\n'
        "    SSLM_HISTORY_BUDGET=off\n"
        "  else\n"
        "    SSLM_HISTORY_BUDGET=on\n"
        "  fi\n"
        "  export SSLM_HISTORY_BUDGET\n"
    )
    assert block in text
    assert text.count("SSLM_HISTORY_BUDGET=off") == 1
    deployment_line = next(line for line in text.splitlines() if "deployment.yaml" in line and "envsubst" in line)
    assert "${SSLM_HISTORY_BUDGET}" in deployment_line
    # Assignment happens before the Deployment is rendered.
    assert text.index(block) < text.index(deployment_line)


def test_deployment_carries_the_history_budget_placeholder():
    deployment = (SCRIPT.parent / "k8s" / "deployment.yaml").read_text(encoding="utf-8")

    assert '- name: SSLM_HISTORY_BUDGET\n              value: "${SSLM_HISTORY_BUDGET}"' in deployment
