#!/usr/bin/env python3
"""Widen leader-election lease timings in a kube-controller-manager/kube-scheduler
static pod manifest.

kind's control-plane manifests run leader election at the upstream k8s defaults
(lease 15s / renew-deadline 10s / retry-period 2s). On a macOS dev host, Docker Desktop's I/O is
slow enough that kubelet's own health checks miss those deadlines and the static pods
restart on their own, tearing down leader election mid-run. On 2026-09-03 the fix
(lease-duration=90s, renew-deadline=60s, retry-period=10s) was applied by hand inside the
running kind-control-plane container and was lost on the next `kind delete`/`kind create`,
since /etc/kubernetes/manifests/ lives on that container's ephemeral filesystem. This module
is the reusable half of making the fix survive a cluster recreate; setup_cluster.sh is the
half that actually reaches into the container.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

DEFAULT_DURATIONS = {
    "leader-elect-lease-duration": "90s",
    "leader-elect-renew-deadline": "60s",
    "leader-elect-retry-period": "10s",
}

_ANCHOR = re.compile(r"^([ \t]*)-\s*--leader-elect=true[ \t]*$", re.MULTILINE)


def patch(manifest_text: str, durations: dict[str, str] | None = None) -> str:
    """Return manifest_text with each leader-elect duration flag set (idempotent).

    A flag already present gets its value replaced; a flag absent gets inserted right
    after the `- --leader-elect=true` line, matching that line's indentation. Raises if the
    manifest has no `--leader-elect=true` line at all (wrong manifest, nothing to anchor on).
    """
    durations = dict(DEFAULT_DURATIONS if durations is None else durations)

    insert_at = None  # offset right after the last line we anchored/inserted on
    for flag, value in durations.items():
        existing = re.compile(rf"^([ \t]*-\s*--{re.escape(flag)}=)\S*([ \t]*)$", re.MULTILINE)
        replaced, count = existing.subn(rf"\g<1>{value}\g<2>", manifest_text, count=1)
        if count:
            manifest_text = replaced
            continue

        if insert_at is None:
            anchor = _ANCHOR.search(manifest_text)
            if not anchor:
                raise ValueError("manifest has no '--leader-elect=true' line to anchor insertion on")
            indent = anchor.group(1)
            insert_at = (
                manifest_text.index("\n", anchor.end()) + 1
                if "\n" in manifest_text[anchor.end():]
                else len(manifest_text)
            )
        insertion = f"{indent}- --{flag}={value}\n"
        manifest_text = manifest_text[:insert_at] + insertion + manifest_text[insert_at:]
        insert_at += len(insertion)

    return manifest_text


def _main(argv: list[str]) -> int:
    """Filter mode: read a manifest on stdin, write the patched manifest to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--durations",
        help="JSON object overriding one or more of the default duration flags",
    )
    args = parser.parse_args(argv)

    durations = dict(DEFAULT_DURATIONS)
    if args.durations:
        durations.update(json.loads(args.durations))

    text = sys.stdin.read()
    sys.stdout.write(patch(text, durations))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
