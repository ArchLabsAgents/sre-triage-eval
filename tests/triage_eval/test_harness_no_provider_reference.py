"""Proves the harness stays provider-agnostic: nothing under harness/ or adapters/ (including
the new VllmModelClient) names a cloud provider. The harness only ever knows an HTTP address
(VLLM_BASE_URL) or, for MLX, a local weights path -- it never knows whether that address is a
Hyperstack VM, a DigitalOcean droplet, or a laptop. That separation is what makes the env column
(tools/triage-eval/configs/base.yaml) a fair comparison: if the harness special-cased a
provider, "which env" and "which code path" would be the same variable.
"""

from __future__ import annotations

import re
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"

# GPU provider names surveyed when the lab was designed, plus the generic clouds. Matched case-insensitively as a whole word so "runpod-style" would still be
# caught but a word like "hyperstacked" (unlikely, but the point is precision) would not
# accidentally clear an actual mention -- word-boundary regex, not substring search either way,
# to avoid both false negatives and noisy false positives on unrelated substrings.
PROVIDER_NAMES = [
    "hyperstack",
    "digitalocean",
    "digital_ocean",
    "scaleway",
    "runpod",
    "vast.ai",
    "vastai",
    "tensordock",
    "nexgencloud",
    "fly.io",
]

# Only source, never bytecode/cache.
_PY_FILES = [
    p
    for p in (TOOLS_DIR / "harness").rglob("*.py")
] + [
    p
    for p in (TOOLS_DIR / "adapters").rglob("*.py")
]


def test_harness_and_adapters_have_python_files_to_check():
    # Guards the test itself against a silently-empty glob (e.g. a future directory rename)
    # passing by producing zero assertions.
    assert len(_PY_FILES) >= 5


def test_no_provider_name_anywhere_in_harness_or_adapters():
    pattern = re.compile(
        r"(?<![a-z0-9])(" + "|".join(re.escape(name) for name in PROVIDER_NAMES) + r")(?![a-z0-9])",
        re.IGNORECASE,
    )
    offenders = []
    for path in _PY_FILES:
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(TOOLS_DIR)}:{line_no}: {match.group(0)!r}")

    assert not offenders, (
        "harness/adapters must stay provider-agnostic (VLLM_BASE_URL is an address, not a "
        "cloud vendor), found:\n" + "\n".join(offenders)
    )


def test_vllm_client_only_reads_base_url_from_env_var_not_hardcoded():
    source = (TOOLS_DIR / "adapters" / "vllm_model_client.py").read_text(encoding="utf-8")
    # base_url must be a plain dataclass field the caller supplies, never a call to
    # os.environ / os.getenv baked into this module -- run_matrix.py owns reading
    # VLLM_BASE_URL, so this module has zero knowledge of env var names at all, only of the
    # ModelClient protocol.
    assert "os.environ" not in source
    assert "os.getenv" not in source
