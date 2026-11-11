"""Warns when a host does not resolve into the CIDR a NetworkPolicy egress rule allows.

NetworkPolicy matches IPs, not names, so the comparator arm's egress rule
(k8s/networkpolicy-comparator.yaml, ANTHROPIC_EGRESS_CIDR) is only as good as the range it was
given. A stale range would not fail anything at setup time: every model call would just time out
inside the pod and the paid window would end as ModelUnavailable. setup_cluster.sh runs this when
COMPARATOR_ARM=1 so a mismatch shows up before the run starts.

Warn, never fail: DNS from the setup host can differ from what the pod sees, and a host may
legitimately answer from more than one range, so a mismatch is a signal for the operator to check,
not proof. Exit code is always 0; the verdict is the printed line.

Usage: python3 check_egress_cidr.py <host> <cidr>
"""

from __future__ import annotations

import ipaddress
import socket
import sys
from collections.abc import Callable, Iterable


def _resolve(host: str) -> list[str]:
    return sorted({info[4][0] for info in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)})


def check(host: str, cidr: str, resolve: Callable[[str], Iterable[str]] = _resolve) -> tuple[bool, str]:
    """(ok, message). ok is False when the CIDR is invalid, the host does not resolve, or none of
    its addresses fall inside the CIDR."""
    try:
        network = ipaddress.ip_network(cidr, strict=False)
    except ValueError as exc:
        return False, f"WARNING: ANTHROPIC_EGRESS_CIDR={cidr!r} is not a valid CIDR ({exc})"
    try:
        addresses = list(resolve(host))
    except OSError as exc:
        return False, f"WARNING: could not resolve {host} ({exc}); check {cidr} by hand before a paid round"
    inside = [a for a in addresses if ipaddress.ip_address(a) in network]
    if inside:
        return True, f"[setup] {host} resolves inside {cidr} ({', '.join(inside)})"
    return False, (
        f"WARNING: {host} resolves to {', '.join(addresses) or 'nothing'}, outside {cidr}: the NetworkPolicy "
        "will block the comparator arm. Check the published range and override ANTHROPIC_EGRESS_CIDR"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 0
    ok, message = check(argv[1], argv[2])
    print(message, file=sys.stdout if ok else sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
