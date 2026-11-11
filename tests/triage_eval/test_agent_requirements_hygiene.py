"""Proves the container's dependency set (requirements-agent.txt) stays Linux-installable: no
MLX/Apple-Silicon-only package (mlx-lm, outlines -- see that file's own comment for why) or
dev-only package (pytest) ever sneaks back in. A regression here means the next `docker build`
fails, or ships a dependency the container never imports."""

from __future__ import annotations

from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"

_FORBIDDEN_PACKAGES = ("mlx-lm", "mlx_lm", "outlines", "pytest")


def _requirement_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for sep in ("==", ">=", "<=", "~=", ">", "<"):
            if sep in stripped:
                stripped = stripped.split(sep, 1)[0]
                break
        names.append(stripped.strip().lower())
    return names


def test_requirements_agent_file_exists():
    assert (TOOLS_DIR / "requirements-agent.txt").is_file()


def test_requirements_agent_has_no_mac_only_or_dev_only_packages():
    names = _requirement_names(TOOLS_DIR / "requirements-agent.txt")
    for forbidden in _FORBIDDEN_PACKAGES:
        assert forbidden.lower() not in names, (
            f"{forbidden!r} must not be in requirements-agent.txt: it is Mac/MLX-only or "
            "dev-only, and would break (or bloat) the container build. See that file's comment."
        )


def test_requirements_agent_declares_what_entrypoint_actually_imports():
    # A loose floor, not a tight allowlist: entrypoint.py + harness/webhook.py are stdlib-only,
    # but harness/schema.py and harness/triage.py (which the future orchestrator will import)
    # need jsonschema, and cloud_adapter.py-style config loading needs pyyaml.
    names = _requirement_names(TOOLS_DIR / "requirements-agent.txt")
    assert "pyyaml" in names
    assert "jsonschema" in names
