"""Unit tests for adapters.mlx_model_client, with mlx-lm/outlines mocked out.

These tests exercise the adapter's own logic (lazy load, chat templating, routing constrained
vs unconstrained generation to the right library call) without downloading or running real
model weights, which belongs in a manual smoke test (see tools/triage-eval/README.md), not
the unit suite.
"""

from __future__ import annotations

import adapters.mlx_model_client as mlx_client_module
from adapters.mlx_model_client import MlxModelClient, MlxModelConfig


class _FakeTokenizer:
    def __init__(self):
        self.chat_template_calls: list[list[dict]] = []

    def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool = True):
        assert add_generation_prompt is True
        # Without tokenize=False the HF tokenizer returns ids (BatchEncoding), not the template
        # string; the adapter has to ask for the string explicitly.
        assert tokenize is False
        self.chat_template_calls.append(messages)
        return f"<chat>{messages[0]['content']}</chat>"


def _config() -> MlxModelConfig:
    return MlxModelConfig(hf_repo="fake/repo", max_output_tokens=64, temperature=0.2)


def test_model_loaded_lazily_and_only_once(monkeypatch):
    load_calls = []

    def fake_load(repo):
        load_calls.append(repo)
        return object(), _FakeTokenizer()

    monkeypatch.setattr(mlx_client_module, "mlx_load", fake_load)
    monkeypatch.setattr(mlx_client_module, "mlx_generate", lambda *a, **k: "out")
    monkeypatch.setattr(mlx_client_module.outlines, "from_mlxlm", lambda model, tok: object())

    client = MlxModelClient(_config())
    assert load_calls == []  # not loaded at construction time

    client.generate("prompt one", schema_constrained=False)
    client.generate("prompt two", schema_constrained=False)

    assert load_calls == ["fake/repo"]  # loaded exactly once across two calls


def test_unconstrained_generate_uses_mlx_generate_with_chat_prompt(monkeypatch):
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(mlx_client_module, "mlx_load", lambda repo: (object(), tokenizer))
    monkeypatch.setattr(mlx_client_module.outlines, "from_mlxlm", lambda model, tok: object())

    captured = {}

    def fake_generate(model, tok, *, prompt, max_tokens, sampler, verbose):
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        captured["sampler"] = sampler
        captured["verbose"] = verbose
        return "plain output"

    monkeypatch.setattr(mlx_client_module, "mlx_generate", fake_generate)
    # Without a sampler mlx_generate runs greedy and ignores the temperature base.yaml declares
    # (measured: an episode cycling between 2 deterministic answers).
    monkeypatch.setattr(
        mlx_client_module, "make_sampler", lambda temp: ("sampler-sentinel", temp)
    )

    client = MlxModelClient(MlxModelConfig(hf_repo="r", max_output_tokens=99, temperature=0.2))
    result = client.generate("diagnose this", schema_constrained=False)

    assert result == "plain output"
    assert captured["prompt"] == "<chat>diagnose this</chat>"
    assert captured["max_tokens"] == 99
    assert captured["sampler"] == ("sampler-sentinel", 0.2)
    assert captured["verbose"] is False
    assert tokenizer.chat_template_calls == [[{"role": "user", "content": "diagnose this"}]]


def test_constrained_generate_routes_through_outlines_generator(monkeypatch):
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(mlx_client_module, "mlx_load", lambda repo: (object(), tokenizer))

    outlines_model_sentinel = object()
    monkeypatch.setattr(
        mlx_client_module.outlines, "from_mlxlm", lambda model, tok: outlines_model_sentinel
    )

    generator_calls = []
    generators_created = []

    class _FakeGenerator:
        def __init__(self, model, output_type):
            assert model is outlines_model_sentinel
            # Outlines 1.x rejects a raw dict as output_type (TypeError on a real config_b
            # smoke): the schema has to arrive wrapped in JsonSchema.
            assert isinstance(output_type, mlx_client_module.JsonSchema)
            generators_created.append(output_type)

        def __call__(self, prompt, *, max_tokens, sampler):
            generator_calls.append((prompt, max_tokens, sampler))
            return '{"diagnosis": "x", "action_type": "submit", "target": "y", "confidence": 0.9}'

    monkeypatch.setattr(mlx_client_module.outlines, "Generator", _FakeGenerator)
    # MLXLM.generate passes kwargs to mlx_lm.generate, which takes sampler, not temperature
    # (the declared temperature travels inside the sampler).
    monkeypatch.setattr(
        mlx_client_module, "make_sampler", lambda temp: ("sampler-sentinel", temp)
    )
    monkeypatch.setattr(
        mlx_client_module, "mlx_generate", lambda *a, **k: pytest_fail_if_called()
    )

    client = MlxModelClient(MlxModelConfig(hf_repo="r", max_output_tokens=64, temperature=0.7))
    result = client.generate("diagnose this", schema_constrained=True)
    client.generate("diagnose this again", schema_constrained=True)

    assert "action_type" in result
    assert generator_calls[0] == ("<chat>diagnose this</chat>", 64, ("sampler-sentinel", 0.7))
    # The Generator compiles the schema's FSM: built once and reused across turns.
    assert len(generator_calls) == 2
    assert len(generators_created) == 1
    # It is the DECODING schema (anyOf per action_type), not the validation one with if/then,
    # which Outlines ignores (an early run let command=null on remediate through the grammar).
    import json as json_module

    from harness.schema import AGENT_ACTION_DECODING_SCHEMA

    compiled = generators_created[0].schema
    compiled = json_module.loads(compiled) if isinstance(compiled, str) else compiled
    assert compiled == AGENT_ACTION_DECODING_SCHEMA


def pytest_fail_if_called(*_a, **_k):
    raise AssertionError("mlx_generate must not be called when schema_constrained=True")
