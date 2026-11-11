import subprocess
import sys
from pathlib import Path

import pytest

import patch_leader_election as ple

MANIFEST = """\
apiVersion: v1
kind: Pod
metadata:
  labels:
    component: kube-controller-manager
  name: kube-controller-manager
  namespace: kube-system
spec:
  containers:
  - command:
    - kube-controller-manager
    - --allocate-node-cidrs=true
    - --authentication-kubeconfig=/etc/kubernetes/controller-manager.conf
    - --leader-elect=true
    - --root-ca-file=/etc/kubernetes/pki/ca.crt
    - --use-service-account-credentials=true
    image: registry.k8s.io/kube-controller-manager:v1.31.0
    name: kube-controller-manager
"""

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "patch_leader_election.py"


def test_patch_inserts_all_three_flags_with_matching_indent():
    out = ple.patch(MANIFEST)

    assert "    - --leader-elect-lease-duration=90s\n" in out
    assert "    - --leader-elect-renew-deadline=60s\n" in out
    assert "    - --leader-elect-retry-period=10s\n" in out
    # inserted right after the anchor line, not scattered
    lines = out.splitlines()
    anchor_idx = lines.index("    - --leader-elect=true")
    assert lines[anchor_idx + 1] == "    - --leader-elect-lease-duration=90s"


def test_patch_preserves_rest_of_manifest():
    out = ple.patch(MANIFEST)

    assert "- --root-ca-file=/etc/kubernetes/pki/ca.crt" in out
    assert "name: kube-controller-manager" in out
    assert out.count("- kube-controller-manager") == 1


def test_patch_is_idempotent():
    once = ple.patch(MANIFEST)
    twice = ple.patch(once)

    assert once == twice


def test_patch_replaces_existing_value_instead_of_duplicating():
    already_patched = ple.patch(MANIFEST)

    changed = ple.patch(already_patched, {"leader-elect-lease-duration": "120s"})

    assert changed.count("--leader-elect-lease-duration=") == 1
    assert "--leader-elect-lease-duration=120s" in changed
    assert "--leader-elect-lease-duration=90s" not in changed


def test_patch_honors_custom_durations():
    # durations is the full set to apply, not a partial override merged onto the
    # defaults (the CLI does that merging itself before calling patch()).
    out = ple.patch(MANIFEST, {"leader-elect-lease-duration": "45s"})

    assert "- --leader-elect-lease-duration=45s" in out
    assert "--leader-elect-renew-deadline=" not in out


def test_patch_raises_without_leader_elect_anchor():
    manifest_without_leader_elect = MANIFEST.replace("- --leader-elect=true\n", "")

    with pytest.raises(ValueError):
        ple.patch(manifest_without_leader_elect)


def test_cli_reads_stdin_writes_stdout():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        input=MANIFEST,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "- --leader-elect-lease-duration=90s" in result.stdout


def test_cli_durations_override():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--durations", '{"leader-elect-lease-duration": "30s"}'],
        input=MANIFEST,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "- --leader-elect-lease-duration=30s" in result.stdout
    assert "- --leader-elect-renew-deadline=60s" in result.stdout
