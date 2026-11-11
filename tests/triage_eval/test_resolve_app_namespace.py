"""scenarios/resolve_app_namespace.py: the app namespace comes from AIOpsLab's metadata."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scenarios.resolve_app_namespace import METADATA_RELATIVE, NamespaceError, resolve

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "triage-eval" / "scenarios" / "resolve_app_namespace.py"


def _home(tmp_path, namespace="test-hotel-reservation"):
    path = tmp_path / METADATA_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"Name": "Hotel Reservation", "Namespace": namespace}))
    return tmp_path


def test_resolves_the_namespace_from_aiopslab_metadata(tmp_path):
    assert resolve(_home(tmp_path)) == "test-hotel-reservation"


def test_an_explicit_value_must_match_the_metadata(tmp_path):
    home = _home(tmp_path)
    assert resolve(home, "test-hotel-reservation") == "test-hotel-reservation"
    with pytest.raises(NamespaceError, match="differs"):
        resolve(home, "other-ns")


def test_the_2026_09_23_value_is_rejected_as_not_rfc1123(tmp_path):
    with pytest.raises(NamespaceError, match="RFC 1123"):
        resolve(_home(tmp_path), "hotelReservation")


def test_invalid_metadata_namespace_is_rejected(tmp_path):
    with pytest.raises(NamespaceError, match="RFC 1123"):
        resolve(_home(tmp_path, "hotelReservation"))


def test_missing_metadata_fails_loud(tmp_path):
    with pytest.raises(NamespaceError, match="cannot read"):
        resolve(tmp_path)


def test_cli_prints_the_namespace_and_exits_nonzero_on_mismatch(tmp_path):
    home = _home(tmp_path)
    ok = subprocess.run([sys.executable, str(SCRIPT), str(home), "--expected", ""], capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.strip() == "test-hotel-reservation"
    bad = subprocess.run(
        [sys.executable, str(SCRIPT), str(home), "--expected", "hotelReservation"], capture_output=True, text=True
    )
    assert bad.returncode == 1 and "RFC 1123" in bad.stderr
