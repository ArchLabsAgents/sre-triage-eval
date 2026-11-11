import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "kind"))
import patch_kind_config_disable_default_cni as pkc  # noqa: E402

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "triage-eval"
    / "kind"
    / "patch_kind_config_disable_default_cni.py"
)

# Shaped like a real AIOpsLab kind-config-<arch>.yaml (a control-plane + one worker, nothing
# fancy) -- not a copy of that file (this repo doesn't have one), just a fixture with the same
# `kind: Cluster` / `nodes:` shape the patch function actually depends on. No `networking:` key,
# matching the honest-limit gap this patch closes: a plain AIOpsLab kind config never sets one.
BASE_CONFIG = textwrap.dedent(
    """\
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    nodes:
      - role: control-plane
      - role: worker
    """
)


def test_patch_sets_disable_default_cni_true():
    out = pkc.patch(BASE_CONFIG)
    doc = yaml.safe_load(out)

    assert doc["networking"]["disableDefaultCNI"] is True


def test_patch_defaults_pod_subnet_to_calicos_own_default_cidr():
    out = pkc.patch(BASE_CONFIG)
    doc = yaml.safe_load(out)

    assert doc["networking"]["podSubnet"] == "192.168.0.0/16"


def test_patch_preserves_an_explicit_pod_subnet_already_in_the_config():
    config_with_subnet = textwrap.dedent(
        """\
        kind: Cluster
        apiVersion: kind.x-k8s.io/v1alpha4
        networking:
          podSubnet: 10.244.0.0/16
        nodes:
          - role: control-plane
        """
    )

    out = pkc.patch(config_with_subnet)
    doc = yaml.safe_load(out)

    # disableDefaultCNI is still forced on, but a deliberate pre-existing podSubnet choice is
    # never silently overridden.
    assert doc["networking"]["disableDefaultCNI"] is True
    assert doc["networking"]["podSubnet"] == "10.244.0.0/16"


def test_patch_preserves_other_networking_keys():
    config_with_extra = textwrap.dedent(
        """\
        kind: Cluster
        apiVersion: kind.x-k8s.io/v1alpha4
        networking:
          apiServerPort: 6443
        nodes:
          - role: control-plane
        """
    )

    out = pkc.patch(config_with_extra)
    doc = yaml.safe_load(out)

    assert doc["networking"]["apiServerPort"] == 6443
    assert doc["networking"]["disableDefaultCNI"] is True


def test_patch_preserves_nodes_and_other_top_level_keys():
    out = pkc.patch(BASE_CONFIG)
    doc = yaml.safe_load(out)

    assert doc["kind"] == "Cluster"
    assert doc["apiVersion"] == "kind.x-k8s.io/v1alpha4"
    assert len(doc["nodes"]) == 2


def test_patch_is_idempotent():
    once = pkc.patch(BASE_CONFIG)
    twice = pkc.patch(once)

    doc_once = yaml.safe_load(once)
    doc_twice = yaml.safe_load(twice)
    assert doc_once == doc_twice


def test_patch_raises_on_non_cluster_kind():
    not_a_kind_config = "kind: Pod\nnodes: []\n"

    with pytest.raises(ValueError):
        pkc.patch(not_a_kind_config)


def test_cli_reads_stdin_writes_stdout():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        input=BASE_CONFIG,
        capture_output=True,
        text=True,
        check=True,
    )

    doc = yaml.safe_load(result.stdout)
    assert doc["networking"]["disableDefaultCNI"] is True


def test_cli_rejects_extra_arguments():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--bogus"],
        input=BASE_CONFIG,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
