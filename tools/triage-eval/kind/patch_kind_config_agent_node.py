#!/usr/bin/env python3
"""Add the agent's dedicated worker node to a kind Cluster config: kind comes up with one extra
worker dedicated to the agent (label role=sslm-agent + taint).

Why this exists, and why it is a patch rather than a config file committed here: the kind config
kind actually creates the cluster from (kind-config-arm.yaml / kind-config-x86.yaml) lives in
$AIOPSLAB_HOME, an external clone this repo does not own and cannot version (setup_cluster.sh's
own comment: "the kind config ... containing kind/kind-config-*.yaml"). Committing a hand-copied
duplicate of that file here would drift the moment AIOpsLab's own config changes upstream, and a
silent divergence between the config measured and the config declared is a confound. Extending
the config kind actually gets, at cluster-creation time, in memory, is the "extend, don't fork"
answer setup_cluster.sh already uses for the leader-election patch
(patch_leader_election.py) and the kube-prometheus-stack install: read the real file, add exactly
the one node the agent needs, write it back out. Same shape as that module: a pure
`patch(text) -> text` function importable and unit-testable without a real kind/docker
environment, plus a stdin/stdout CLI filter setup_cluster.sh pipes through before
`kind create cluster --config`.

The node's kubelet gets its role label and taint via the node's own `kubeadmConfigPatches`
(a real, documented kind config field: a list of raw YAML strings merged into that node's
kubeadm JoinConfiguration) setting the standard kubelet flags `--node-labels` and
`--register-with-taints` -- not a separate `kubectl label`/`kubectl taint` step, so the
label/taint exist from the moment the node joins, with nothing that a first scheduling decision
could race against.
"""

from __future__ import annotations

import sys

import yaml

NODE_ROLE_LABEL = "role=sslm-agent"
NODE_TAINT = "sslm-agent=true:NoSchedule"

AGENT_NODE = {
    "role": "worker",
    "kubeadmConfigPatches": [
        "kind: JoinConfiguration\n"
        "nodeRegistration:\n"
        "  kubeletExtraArgs:\n"
        f'    node-labels: "{NODE_ROLE_LABEL}"\n'
        f'    register-with-taints: "{NODE_TAINT}"\n'
    ],
}

# What marks a node as "already the agent node" on a re-patch (idempotency check): the presence
# of NODE_ROLE_LABEL inside its kubeadmConfigPatches text, not object identity -- a config
# hand-edited or re-serialized by a different YAML dumper would still round-trip through this
# check correctly, where a dict `==` comparison against AGENT_NODE's exact key order/formatting
# would not.
def _is_agent_node(node: dict) -> bool:
    if not isinstance(node, dict):
        return False
    patches = node.get("kubeadmConfigPatches") or []
    return any(NODE_ROLE_LABEL in patch for patch in patches if isinstance(patch, str))


def patch(kind_config_text: str) -> str:
    """Return kind_config_text with the sslm-agent worker node appended to `nodes:` (idempotent:
    a config that already has the node, by the _is_agent_node check above, is returned
    unchanged). The appended node carries the control-plane's `image:` when the control-plane
    pins one, so the join runs the same Kubernetes version. Raises ValueError if the document is not a kind Cluster config (wrong `kind:`,
    or no `nodes:` list to append to -- kind defaults to a single control-plane node when
    `nodes:` is entirely absent, and this module never invents that implicit default just to
    append to it: a config with no explicit nodes: list is out of scope, not silently handled)."""
    doc = yaml.safe_load(kind_config_text)
    if not isinstance(doc, dict) or doc.get("kind") != "Cluster":
        raise ValueError(f"not a kind Cluster config (kind={doc.get('kind') if isinstance(doc, dict) else doc!r})")

    nodes = doc.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("kind Cluster config has no explicit 'nodes:' list to append the agent node to")

    if any(_is_agent_node(node) for node in nodes):
        return kind_config_text

    # Without an explicit image the new node gets kind's own default node image, and a newer
    # kubeadm refuses to join an older control-plane (measured 2026-09-25: AIOpsLab pins
    # v1.32.1 via a custom image, kind v0.32 defaults to v1.36.1, join fails with ">= 1.35.0").
    # The control-plane is what the join has to match, so its image is the one copied; when it
    # pins none, both fall back to the same kind default and the agent node pins none either.
    agent_node = dict(AGENT_NODE)
    control_plane_image = next(
        (
            node.get("image")
            for node in nodes
            if isinstance(node, dict) and node.get("role") == "control-plane"
        ),
        None,
    )
    if control_plane_image:
        agent_node["image"] = control_plane_image

    nodes.append(agent_node)
    doc["nodes"] = nodes
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def _main(argv: list[str]) -> int:
    """Filter mode, same convention as patch_leader_election.py: read a kind config on stdin,
    write the patched config to stdout."""
    if argv:
        sys.stderr.write("usage: patch_kind_config_agent_node.py < kind-config.yaml > patched.yaml\n")
        return 2
    text = sys.stdin.read()
    sys.stdout.write(patch(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
