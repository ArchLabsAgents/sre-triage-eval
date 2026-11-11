"""ModelClient implementation backed by MLX (harness.agent.ModelClient protocol).

Config A calls this with schema_constrained=False (plain sampling); configs B and C call it
with schema_constrained=True, which routes through Outlines so the raw output already parses
as JSON against harness.schema.AGENT_ACTION_SCHEMA instead of relying on the model to emit
well-formed JSON on its own. Same model, same weights, same decoding params across all three
configs (base.yaml), only the decoding strategy changes: the base model is held fixed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import outlines
from mlx_lm import generate as mlx_generate
from mlx_lm import load as mlx_load
from mlx_lm.sample_utils import make_sampler
from outlines.types import JsonSchema

from harness.metrics import GenerateStats
from harness.schema import AGENT_ACTION_DECODING_SCHEMA


@dataclass
class MlxModelConfig:
    hf_repo: str
    max_output_tokens: int
    temperature: float


class MlxModelClient:
    """Loads one MLX model once and serves both unconstrained and schema-constrained generation.

    Loaded lazily on first `generate` call so importing this module (e.g. from a unit test that
    never calls `generate`) never pays the weight-load cost.
    """

    def __init__(self, config: MlxModelConfig) -> None:
        self.config = config
        # Same contract as VllmModelClient.stats: model_seconds/model_calls per episode, so the
        # mlx-local column is comparable to the cloud column in the same unit.
        self.stats = GenerateStats()
        self._mlx_model = None
        self._tokenizer = None
        self._outlines_model = None
        self._constrained_generator = None

    def _ensure_loaded(self) -> None:
        if self._mlx_model is not None:
            return
        self._mlx_model, self._tokenizer = mlx_load(self.config.hf_repo)
        self._outlines_model = outlines.from_mlxlm(self._mlx_model, self._tokenizer)

    def _chat_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        # tokenize=False is required: the default (True) returns a BatchEncoding of ids, not the
        # template string, and mlx_generate treating that as the prompt produced episodes with ~2
        # output tokens per turn (measured on a real run: 10/10 turns malformed).
        return self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def generate(self, prompt: str, *, schema_constrained: bool) -> str:
        start = time.monotonic()
        try:
            output = self._generate(prompt, schema_constrained=schema_constrained)
            self.stats.calls += 1
            return output
        finally:
            self.stats.seconds += time.monotonic() - start

    def _generate(self, prompt: str, *, schema_constrained: bool) -> str:
        self._ensure_loaded()
        chat_prompt = self._chat_prompt(prompt)

        if schema_constrained:
            if self._constrained_generator is None:
                # Outlines 1.x does not accept a raw dict as output_type (TypeError measured on
                # the config_b smoke), so the schema goes in through JsonSchema. Built once and
                # cached because the Generator compiles the schema's FSM, and one episode calls
                # generate up to 10 times. This is the DECODING schema, not the validation one:
                # the if/then that requires command on inspect/remediate is not part of the
                # compiled grammar (harness.schema.build_decoding_schema).
                self._constrained_generator = outlines.Generator(
                    self._outlines_model, JsonSchema(AGENT_ACTION_DECODING_SCHEMA)
                )
            # Outlines' MLXLM.generate passes kwargs straight to mlx_lm.generate, which takes
            # sampler, not temperature (same contract as the unconstrained path).
            return self._constrained_generator(
                chat_prompt,
                max_tokens=self.config.max_output_tokens,
                sampler=make_sampler(temp=self.config.temperature),
            )

        # Without a sampler mlx_generate runs pure greedy, ignoring the temperature base.yaml
        # declares; measured: the whole episode cycled between 2 identical deterministic
        # answers because each prompt state had only one continuation.
        return mlx_generate(
            self._mlx_model,
            self._tokenizer,
            prompt=chat_prompt,
            max_tokens=self.config.max_output_tokens,
            sampler=make_sampler(temp=self.config.temperature),
            verbose=False,
        )
