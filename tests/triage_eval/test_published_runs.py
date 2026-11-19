"""The published runs in runs/ recompute to the published scores with the code in this repository.

Two checks:
- runs/MANIFEST.sha256 lists exactly the files under runs/, with matching hashes.
- every runs/*/score_*.py and runs/t5-cost-*/compute_t5.py, run from tools/triage-eval as a reader
  would, writes JSON identical to the file the preprint reads.
- the character reading of the budget-matched Sonnet window recomputes from its usage logs.

Two known differences are allowed, documented in runs/README.md: the not-measured reason string of
one episode in the two 9B scores, and the `why` string of the character reading, were translated to
English after those files were written.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
TOOLS = REPO / "tools" / "triage-eval"
MANIFEST = RUNS / "MANIFEST.sha256"

TRANSLATED_REASON = (
    "app did not settle before injection",
    "app nao assentou antes da injecao (I2)",
)


def _manifest() -> dict[str, str]:
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        digest, path = line.split("  ", 1)
        entries[path] = digest
    return entries


def test_manifest_matches_files():
    on_disk = {
        p.relative_to(RUNS).as_posix()
        for p in RUNS.rglob("*")
        if p.is_file() and p != MANIFEST and "__pycache__" not in p.parts
    }
    entries = _manifest()
    assert set(entries) == on_disk
    for path, digest in entries.items():
        assert hashlib.sha256((RUNS / path).read_bytes()).hexdigest() == digest, path


def _published_for(script: Path, name: str) -> Path:
    flat = script.parent / name
    return flat if flat.exists() else script.parent / "score" / name


def _normalize_translated_reason(value):
    if isinstance(value, dict):
        return {k: _normalize_translated_reason(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_translated_reason(v) for v in value]
    if isinstance(value, str):
        return value.replace(TRANSLATED_REASON[1], TRANSLATED_REASON[0])
    return value


def _run(script: Path, arg: Path) -> None:
    env = {**os.environ, "PYTHONPATH": "."}
    rel = os.path.relpath(script, TOOLS)
    result = subprocess.run([sys.executable, rel, str(arg)], cwd=TOOLS, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]


SCORE_SCRIPTS = sorted(RUNS.glob("*/score_*.py"))


@pytest.mark.parametrize("script", SCORE_SCRIPTS, ids=lambda p: p.parent.name)
def test_score_script_reproduces_published(script, tmp_path):
    _run(script, tmp_path)
    produced = sorted(tmp_path.glob("*.json"))
    assert produced, "script wrote nothing"
    for out in produced:
        published = _published_for(script, out.name)
        assert published.exists(), f"{out.name} has no published counterpart"
        if out.read_bytes() == published.read_bytes():
            continue
        new, old = json.loads(out.read_text()), json.loads(published.read_text())
        assert new != old  # bytes differ, so the JSON must too
        assert _normalize_translated_reason(old) == new, f"{script.parent.name}/{out.name} differs"
        assert script.parent.name.startswith("9b-"), "only the 9B scores carry the translated reason"


def test_t5_reproduces_published(tmp_path):
    (script,) = RUNS.glob("t5-cost-*/compute_t5.py")
    out = tmp_path / "t5.json"
    _run(script, out)
    assert out.read_bytes() == (script.parent / "t5.json").read_bytes()


def test_chars_reading_reproduces_published(tmp_path):
    run = RUNS / "sonnet-marker-r2short-20261102"
    out = tmp_path / "reading.json"
    env = {**os.environ, "PYTHONPATH": "."}
    result = subprocess.run(
        [
            sys.executable, "-m", "scenarios.chars_window", "read",
            "--results", str(run / "results"),
            "--prefix", "s1-r2",
            "--schedule", "scenarios/schedules/r2short-seed-1052.json",
            "--usage", str(run / "results" / "s1-smokegate-comparator-usage.jsonl"),
            str(run / "results" / "s1-comparator-usage.jsonl"),
            "--out", str(out),
        ],
        cwd=TOOLS, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    new = json.loads(out.read_text())
    old = json.loads((run / "score" / "chars-window-reading.json").read_text())
    assert old.pop("why") == "classe fina exata 5/5 >= corte 3"
    assert new.pop("why") == "exact fine class 5/5 >= cut 3"
    assert new == old
    assert new["reading"] == "capacidade" and new["measure1"]["passed"]


def test_score_scripts_found():
    # A layout change that hides the scripts would make the parametrized test vacuous.
    assert len(SCORE_SCRIPTS) == 9
