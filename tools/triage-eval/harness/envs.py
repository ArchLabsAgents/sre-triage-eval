"""Execution-environment declaration and selection.

An "env" is everything invariant about where a matrix run executes: backend, quantization,
weights, accelerator, provider, network. It is declared in configs/base.yaml under `envs:` and
becomes the first column of every result row, so two environments can never be read as one
table. Quantization lives here on purpose: it is the confound the whole per-env schema exists
to keep visible.

This module is the single source for loading and selecting envs. run_matrix.py (the --env
flag), mlx_factories.py (which env's weights to load) and scenarios/backfill_env.py (stamping
legacy records) all resolve through here, so the rules cannot drift apart.
"""

from __future__ import annotations

import re

# <backend-or-provider>-<accelerator>, lowercase, stable across runs (it is a column key).
ENV_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")

# Derived-env block for a config written before `envs:` existed (see load_envs).
_LEGACY_MODEL_KEYS = ("backend", "quantization", "hf_repo")


class EnvConfigError(ValueError):
    """A config whose env declaration cannot be resolved. Callers turn this into a refusal
    to start, never into a default: an undeclared or ambiguous env is the exact silent-mixing
    hazard the env column exists to prevent."""


def load_envs(config: dict) -> dict[str, dict]:
    """Return the declared envs as {name: block}.

    With `envs:` absent, derives a single `mlx-local` entry from the legacy top-level
    `model.backend` / `model.quantization` / `model.hf_repo` keys, so configs written before
    the env column keep working unchanged. A config with neither `envs:`
    nor any legacy model key (the bare configs unit tests use) declares no env at all and
    returns {}: those runs keep the pre-env behavior (records at the results root, `unknown`
    in the CSV).
    """
    declared = config.get("envs")
    if declared:
        if not isinstance(declared, dict):
            raise EnvConfigError("`envs:` must be a map of <env-name>: {block}")
        for name, block in declared.items():
            if not isinstance(name, str) or not ENV_SLUG_RE.match(name):
                raise EnvConfigError(
                    f"env name {name!r} is not a valid slug (expected {ENV_SLUG_RE.pattern})"
                )
            if not isinstance(block, dict):
                raise EnvConfigError(f"env {name!r} must be a map, got {type(block).__name__}")
        return dict(declared)

    model = config.get("model") or {}
    if not any(model.get(key) is not None for key in _LEGACY_MODEL_KEYS):
        return {}
    return {
        "mlx-local": {
            "backend": model.get("backend"),
            "quantization": model.get("quantization"),
            "weights_ref": model.get("hf_repo"),
            "provider": "local",
            "harness_location": "on-instance",
            "network": "local",
        }
    }


def select_env(envs: dict[str, dict], requested: str | None) -> tuple[str | None, dict | None]:
    """Resolve which env this run executes in.

    Required when more than one env is declared, defaulted when exactly one is. Returns
    (None, None) only for configs that declare no env at all (see load_envs). Refuses a
    selected env with a null or missing quantization: an undeclared quant is the ambiguity
    this schema exists to prevent.
    """
    if not envs:
        if requested is not None:
            raise EnvConfigError(
                f"--env {requested!r} given, but the config declares no envs (no `envs:` map "
                "and no legacy model.backend/quantization/hf_repo keys)"
            )
        return None, None

    if requested is None:
        if len(envs) > 1:
            names = ", ".join(sorted(envs))
            raise EnvConfigError(f"--env is required: {len(envs)} envs declared ({names})")
        requested = next(iter(envs))

    if requested not in envs:
        names = ", ".join(sorted(envs))
        raise EnvConfigError(f"env {requested!r} is not declared (declared: {names})")

    block = envs[requested]
    if block.get("quantization") in (None, ""):
        raise EnvConfigError(
            f"env {requested!r} declares no quantization; refusing to start (quantization is "
            "the declared confound every result row must carry)"
        )
    return requested, block
