#!/usr/bin/env python3
"""Print the Kubernetes namespace AIOpsLab deploys HotelReservation into.

The 2026-09-23 real run passed `hotelReservation` (the app's name, not its namespace, and not a
valid Kubernetes name) as APP_NAMESPACE, so the agent's Role landed nowhere AIOpsLab uses. The
source of truth is AIOpsLab's own metadata (`aiopslab/service/metadata/hotel-reservation.json`,
key `Namespace`), read here instead of typed by hand. An explicit `--expected` value is accepted
only when it is a valid RFC 1123 label AND equals the metadata, so a typo fails before any
cluster work instead of after an hour of it.

stdlib only: it runs on the lab VM before the venv exists.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

METADATA_RELATIVE = Path("aiopslab") / "service" / "metadata" / "hotel-reservation.json"
RFC1123_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")


class NamespaceError(ValueError):
    pass


def validate_label(value: str) -> str:
    if not RFC1123_LABEL.fullmatch(value or ""):
        raise NamespaceError(f"{value!r} is not a valid Kubernetes namespace (RFC 1123 label)")
    return value


def resolve(aiopslab_home: Path, expected: str | None = None) -> str:
    path = aiopslab_home / METADATA_RELATIVE
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        namespace = metadata["Namespace"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise NamespaceError(f"cannot read the HotelReservation namespace from {path}: {exc}") from exc
    validate_label(namespace)
    if expected is not None:
        validate_label(expected)
        if expected != namespace:
            raise NamespaceError(
                f"expected namespace {expected!r} differs from AIOpsLab metadata {namespace!r} ({path})"
            )
    return namespace


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("aiopslab_home", type=Path)
    parser.add_argument("--expected", help="optional explicit namespace, must match the metadata")
    args = parser.parse_args(argv)
    try:
        print(resolve(args.aiopslab_home, args.expected or None))
    except NamespaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
