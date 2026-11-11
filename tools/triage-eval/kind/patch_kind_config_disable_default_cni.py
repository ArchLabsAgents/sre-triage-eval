#!/usr/bin/env python3
"""Disable kind's default CNI (kindnet) in a kind Cluster config, in memory (closes the
"NetworkPolicy is declared but not enforced" honest limit tracked in
k8s/networkpolicy.yaml and README.md -- kindnet does not enforce NetworkPolicy at all).

Why this exists, and why it is a patch rather than a config file committed here: same rationale as
kind/patch_kind_config_agent_node.py (read its own docstring first). The kind config
setup_cluster.sh actually creates the cluster from lives in $AIOPSLAB_HOME, an external clone this
repo does not own and cannot version. Extending the config kind actually gets, at
cluster-creation time, in memory, is the same "extend, don't fork" shape already used for the
agent worker node and the leader-election patch: a pure `patch(text) -> text` function, importable
and unit-testable without a real kind/docker environment, plus a stdin/stdout CLI filter
setup_cluster.sh pipes through before `kind create cluster --config`.

`networking.disableDefaultCNI: true` only takes effect at cluster CREATION -- there is no way to
retroactively disable kindnet on a cluster that already has it running (setup_cluster.sh's
CALICO_INSTALL=1 block enforces this at runtime: it refuses to install Calico against a
pre-existing cluster that was not created with this patch applied). `networking.podSubnet` is also
set here, to `192.168.0.0/16` -- Calico's own default IPPool CIDR (calico.yaml as vendored in
kind/vendor/ does not set CALICO_IPV4POOL_CIDR, so it relies on the cluster's actual pod CIDR
lining up with that default; see the Calico "Installing on Kind" guide, which recommends this
exact CIDR for the same reason). Only set when the caller's kind config does not already declare
its own `networking.podSubnet` -- an explicit value in the source config is a deliberate choice
this patch must not silently override.
"""

from __future__ import annotations

import sys

import yaml

CALICO_DEFAULT_POD_SUBNET = "192.168.0.0/16"


def patch(kind_config_text: str) -> str:
    """Return kind_config_text with `networking.disableDefaultCNI: true` set (idempotent: a
    config that already has it set to true is returned with the same value, never toggled off)
    and `networking.podSubnet` defaulted to Calico's own default IPPool CIDR when the config does
    not already declare one. Raises ValueError if the document is not a kind Cluster config (same
    contract as patch_kind_config_agent_node.py: wrong `kind:`, or the doc does not parse to a
    dict)."""
    doc = yaml.safe_load(kind_config_text)
    if not isinstance(doc, dict) or doc.get("kind") != "Cluster":
        raise ValueError(f"not a kind Cluster config (kind={doc.get('kind') if isinstance(doc, dict) else doc!r})")

    networking = doc.get("networking")
    if not isinstance(networking, dict):
        networking = {}

    networking["disableDefaultCNI"] = True
    networking.setdefault("podSubnet", CALICO_DEFAULT_POD_SUBNET)

    doc["networking"] = networking
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def _main(argv: list[str]) -> int:
    """Filter mode, same convention as patch_kind_config_agent_node.py and
    patch_leader_election.py: read a kind config on stdin, write the patched config to stdout."""
    if argv:
        sys.stderr.write("usage: patch_kind_config_disable_default_cni.py < kind-config.yaml > patched.yaml\n")
        return 2
    text = sys.stdin.read()
    sys.stdout.write(patch(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
