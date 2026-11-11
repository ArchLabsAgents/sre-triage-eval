import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "kind"))
import patch_kind_config_agent_node as pkc  # noqa: E402

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "triage-eval"
    / "kind"
    / "patch_kind_config_agent_node.py"
)

# Shaped like a real AIOpsLab kind-config-<arch>.yaml (a control-plane + one worker, nothing
# fancy) -- not a copy of that file (this repo doesn't have one, see the module's own docstring
# for why), just a fixture with the same `kind: Cluster` / `nodes:` shape the patch function
# actually depends on.
BASE_CONFIG = textwrap.dedent(
    """\
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    nodes:
      - role: control-plane
      - role: worker
    """
)


def test_patch_appends_exactly_one_worker_node():
    out = pkc.patch(BASE_CONFIG)
    doc = yaml.safe_load(out)

    assert len(doc["nodes"]) == 3
    assert doc["nodes"][0]["role"] == "control-plane"
    assert doc["nodes"][1]["role"] == "worker"
    assert doc["nodes"][2]["role"] == "worker"


def test_patch_sets_the_role_label_and_taint_via_kubeadm_config_patches():
    out = pkc.patch(BASE_CONFIG)
    doc = yaml.safe_load(out)

    agent_node = doc["nodes"][2]
    patches_text = "\n".join(agent_node["kubeadmConfigPatches"])
    assert "role=sslm-agent" in patches_text
    assert "sslm-agent=true:NoSchedule" in patches_text
    assert "node-labels" in patches_text
    assert "register-with-taints" in patches_text


def test_patch_preserves_the_original_nodes():
    out = pkc.patch(BASE_CONFIG)
    doc = yaml.safe_load(out)

    # Neither pre-existing node gained the agent's label/taint -- only the appended one carries
    # them (a bug that patched every node instead of appending one would still pass a looser
    # "role=sslm-agent is present somewhere" assertion; this one would catch it).
    for node in doc["nodes"][:2]:
        assert "kubeadmConfigPatches" not in node or "role=sslm-agent" not in "\n".join(
            node.get("kubeadmConfigPatches", [])
        )


# Same shape as AIOpsLab's kind-config-arm.yaml: every node pinned to a custom image older than
# kind's default. The agent node has to join that control-plane, so it must run the same image.
PINNED_CONFIG = textwrap.dedent(
    """\
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    nodes:
      - role: control-plane
        image: example/aiopslab-kind-arm:latest
        extraMounts:
          - hostPath: /run/udev
            containerPath: /run/udev
      - role: worker
        image: example/aiopslab-kind-arm:latest
    """
)


def test_patch_agent_node_inherits_the_control_plane_image():
    doc = yaml.safe_load(pkc.patch(PINNED_CONFIG))

    assert doc["nodes"][2]["image"] == "example/aiopslab-kind-arm:latest"


def test_patch_leaves_image_unset_when_the_control_plane_pins_none():
    doc = yaml.safe_load(pkc.patch(BASE_CONFIG))

    assert "image" not in doc["nodes"][2]


def test_patch_follows_the_control_plane_not_a_pinned_worker():
    # The join constraint is against the control-plane's version: a worker-only pin would put
    # the agent node on a different version than the control-plane it joins.
    worker_only_pin = textwrap.dedent(
        """\
        kind: Cluster
        apiVersion: kind.x-k8s.io/v1alpha4
        nodes:
          - role: control-plane
          - role: worker
            image: example/other:v1
        """
    )

    doc = yaml.safe_load(pkc.patch(worker_only_pin))

    assert "image" not in doc["nodes"][2]


def test_patch_does_not_mutate_the_module_level_node_template():
    pkc.patch(PINNED_CONFIG)

    assert "image" not in pkc.AGENT_NODE


def test_patch_is_idempotent():
    once = pkc.patch(BASE_CONFIG)
    twice = pkc.patch(once)

    doc_twice = yaml.safe_load(twice)
    assert len(doc_twice["nodes"]) == 3, "re-patching must not append a second agent node"


def test_patch_preserves_apiversion_and_other_top_level_keys():
    out = pkc.patch(BASE_CONFIG)
    doc = yaml.safe_load(out)

    assert doc["kind"] == "Cluster"
    assert doc["apiVersion"] == "kind.x-k8s.io/v1alpha4"


def test_patch_raises_on_non_cluster_kind():
    not_a_kind_config = "kind: Pod\nnodes: []\n"

    with pytest.raises(ValueError):
        pkc.patch(not_a_kind_config)


def test_patch_raises_when_nodes_list_is_absent():
    no_nodes = "kind: Cluster\napiVersion: kind.x-k8s.io/v1alpha4\n"

    with pytest.raises(ValueError):
        pkc.patch(no_nodes)


def test_cli_reads_stdin_writes_stdout():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        input=BASE_CONFIG,
        capture_output=True,
        text=True,
        check=True,
    )

    doc = yaml.safe_load(result.stdout)
    assert len(doc["nodes"]) == 3


def test_cli_rejects_extra_arguments():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--bogus"],
        input=BASE_CONFIG,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
