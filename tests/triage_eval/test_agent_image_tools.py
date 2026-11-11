"""The agent's tools are binaries it execs inside its own pod (harness/envelope.py). Until
2026-10-15 the image shipped neither kubectl nor wget, so every command in the earlier model
rounds and the API arm round ended exit=-1 (FileNotFoundError) and the model triaged from the alert text alone."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
SETUP = (ROOT / "setup_cluster.sh").read_text(encoding="utf-8")


def test_image_installs_every_top_level_tool_the_envelope_allows():
    assert "apt-get install -y --no-install-recommends wget" in DOCKERFILE
    assert "/usr/local/bin/kubectl" in DOCKERFILE


def test_kubectl_is_pinned_and_hash_checked_for_both_arches():
    assert "ARG KUBECTL_VERSION=v1.36.1" in DOCKERFILE
    assert "sha256sum -c -" in DOCKERFILE
    sums = re.findall(r"(amd64|arm64)\) sum=([0-9a-f]{64})", DOCKERFILE)
    assert {arch for arch, _ in sums} == {"amd64", "arm64"}


def test_tools_install_before_the_user_drops_root():
    assert DOCKERFILE.index("/usr/local/bin/kubectl") < DOCKERFILE.index("USER sslm-agent")


def test_kubectl_matches_the_kind_node_version():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    node = re.search(r"kindest/node:(v[0-9.]+)", readme).group(1)
    assert f"ARG KUBECTL_VERSION={node}" in DOCKERFILE


def test_setup_fails_closed_when_the_agent_cannot_run_its_tools():
    rollout = SETUP.index("rollout status deploy/sslm-agent")
    probe = SETUP.index("exec deploy/sslm-agent -- kubectl get pods")
    assert rollout < probe
    assert "exec deploy/sslm-agent -- wget --version" in SETUP
    assert "every action would exit -1" in SETUP and "exit 1; }" in SETUP[probe:probe + 400]
