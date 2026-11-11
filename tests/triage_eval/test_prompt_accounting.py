"""Chars per call: what PromptBudget covers, measured where the prompt is built and written to
the comparator usage log. Includes the worst case: a full history window and a huge command output, on every turn."""

from __future__ import annotations

import json
from unittest.mock import patch

from adapters.aiopslab_env import OBSERVATION_MAX_CHARS, _omission_marker, cap_observation, cap_within_share
from adapters.anthropic_model_client import PINNED_MODEL, AnthropicModelClient, AnthropicModelConfig
from harness.agent import HISTORY_WINDOW, Agent, AgentConfig, PromptBudget, TurnResult
from harness.episode_runner import EpisodeContext, ScriptedSource
from harness.prompt_accounting import CallChars, current_call_chars, log_fields, set_call_chars

BUDGET = PromptBudget(14_000)
HUGE = 500_000


class _FakeResponse:
    status = 200
    headers = {"request-id": "req_1"}

    def __init__(self, text: str):
        self._body = json.dumps(
            {
                "model": PINNED_MODEL,
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _context(n_sources: int, size: int = 200_000) -> EpisodeContext:
    return EpisodeContext(
        "ep-1", "ns", sources=[ScriptedSource(f"src{i}", True, text="x" * size) for i in range(n_sources)]
    )


def _history(n: int, size: int = HUGE) -> list[TurnResult]:
    return [
        TurnResult(
            output_tokens=1,
            schema_valid=True,
            action={"action_type": "inspect", "command": "kubectl get pods " + "a" * 40},
            solved=False,
            observation="y" * size,
            raw_output="",
        )
        for n_ in range(n)
    ]


def _agent(budget: PromptBudget | None = BUDGET, model=None) -> Agent:
    return Agent(
        model or object(),
        AgentConfig(
            schema_constrained_decoding=True, memory_enabled=False, history_budget=False, prompt_budget=budget
        ),
    )


def _measure(n_sources: int, observation_size: int = HUGE, history_entry_size: int = HUGE) -> CallChars:
    ctx = _context(n_sources)
    _, chars = _agent()._build_prompt_measured(
        ctx.as_prompt_text(BUDGET.context_chars),
        "z" * observation_size,
        _history(HISTORY_WINDOW + 3, history_entry_size),
        ctx.covered_chars(BUDGET.context_chars),
    )
    return chars


def test_without_a_budget_only_the_prompt_length_is_known():
    prompt, chars = _agent(budget=None)._build_prompt_measured("task text", "obs", None)
    assert chars.prompt_chars == len(prompt)
    assert chars.covered_chars is None and chars.budget_chars is None


def test_covered_chars_exclude_system_prompt_task_label_and_headers():
    ctx = EpisodeContext("ep-1", "ns", sources=[ScriptedSource("a", True, text="A" * 100)])
    prompt, chars = _agent()._build_prompt_measured(
        ctx.as_prompt_text(BUDGET.context_chars), "obs" * 10, None, ctx.covered_chars(BUDGET.context_chars)
    )
    assert chars.context_chars == 100
    assert chars.history_chars == 0
    assert chars.observation_chars == 30
    assert chars.covered_chars == 130
    assert chars.prompt_chars == len(prompt) and chars.prompt_chars > chars.covered_chars


def test_worst_case_full_history_and_huge_output_stays_inside_the_budget():
    # An earlier version of this test declared 14_056 as the worst case, but its history never
    # filled the share, and the smoke measured 14_489. Every cut marker now sits inside the
    # share, so the three parts are bounded one by one and the sum cannot pass the budget.
    for n_sources in (1, 3, 4, 5, 6, 8):
        for entry_size in (200, 700, 1_000, 1_100, 1_200, 3_000, HUGE):
            chars = _measure(n_sources, history_entry_size=entry_size)
            assert chars.context_chars <= BUDGET.context_chars, (n_sources, chars)
            assert chars.history_chars <= BUDGET.history_chars, (n_sources, chars)
            assert chars.observation_chars <= BUDGET.observation_chars, (n_sources, chars)
            assert chars.covered_chars <= BUDGET.total_chars, (n_sources, chars)


def test_the_share_is_used_up_to_the_marker_not_left_empty():
    chars = _measure(5)
    assert chars.observation_chars >= BUDGET.observation_chars - 10
    assert chars.context_chars >= BUDGET.context_chars - 50


def test_cap_within_share_never_passes_the_limit_and_keeps_head_and_tail():
    text = "H" * 50 + "m" * 10_000 + "T" * 50
    for limit in (1, 50, 150, 151, 200, 1_400, 3_500, 10_099, 10_100):
        capped = cap_within_share(text, limit)
        assert len(capped) <= limit, limit
    assert cap_within_share(text, 1_400).startswith("H") and cap_within_share(text, 1_400).endswith("T")
    assert "omitted" in cap_within_share(text, 1_400)
    assert cap_within_share("short", 100) == "short"


def test_without_a_budget_the_old_cap_is_untouched():
    text = "a" * 20_000
    capped = cap_observation(text)
    assert len(capped) > OBSERVATION_MAX_CHARS  # marker on top, every earlier window byte for byte
    assert capped == text[:8_000] + _omission_marker(8_000) + text[-4_000:]


def test_a_published_measure_is_visible_only_around_one_call():
    seen = []

    class Model:
        def generate(self, prompt, *, schema_constrained):
            seen.append(current_call_chars())
            return json.dumps(
                {
                    "diagnosis": "d",
                    "cause_class": "unknown",
                    "action_type": "submit",
                    "target": "t",
                    "command": None,
                    "confidence": 0.5,
                }
            )

    ctx = _context(2, 1000)

    class Env:
        def task_description(self):
            return ctx.as_prompt_text(BUDGET.context_chars)

        def covered_context_chars(self):
            return ctx.covered_chars(BUDGET.context_chars)

        def apply_action(self, action):
            return True, "done"

    _agent(model=Model()).run_episode(Env(), "t", max_turns=1)
    assert seen[0] is not None and seen[0].context_chars == 2000
    assert current_call_chars() is None


def test_a_measure_for_another_prompt_is_dropped_to_nulls():
    set_call_chars(CallChars(100, 90, 50, 20, 20, 14_000))
    try:
        assert log_fields(100)["covered_chars"] == 90
        dropped = log_fields(101)
        assert dropped["prompt_chars"] == 101 and dropped["covered_chars"] is None
    finally:
        set_call_chars(None)


def test_episode_through_the_client_logs_chars_on_every_call_and_all_inside_the_budget(tmp_path):
    usage = tmp_path / "usage.jsonl"
    client = AnthropicModelClient(
        AnthropicModelConfig(api_key="sk-test", max_output_tokens=512, usage_log_path=usage)
    )
    ctx = _context(4)
    turns = 8

    class Env:
        n = 0

        def task_description(self):
            return ctx.as_prompt_text(BUDGET.context_chars)

        def covered_context_chars(self):
            return ctx.covered_chars(BUDGET.context_chars)

        def apply_action(self, action):
            Env.n += 1
            return False, "o" * HUGE

    inspect = json.dumps(
        {
            "diagnosis": "d",
            "cause_class": "unknown",
            "action_type": "inspect",
            "target": "t",
            "command": "kubectl get pods",
            "confidence": 0.5,
        }
    )
    with patch("urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(inspect)):
        _agent(model=client).run_episode(Env(), "t", max_turns=turns)
    rows = [json.loads(line) for line in usage.read_text().splitlines()]
    assert len(rows) == turns
    for row in rows:
        assert isinstance(row["prompt_chars"], int) and isinstance(row["covered_chars"], int)
        assert row["budget_chars"] == 14_000
        assert row["covered_chars"] <= 14_000
        assert row["covered_chars"] == row["context_chars"] + row["history_chars"] + row["observation_chars"]
    assert rows[-1]["history_chars"] > 0 and rows[0]["history_chars"] == 0


def test_a_call_made_outside_the_agent_logs_null_covered_chars(tmp_path):
    usage = tmp_path / "usage.jsonl"
    client = AnthropicModelClient(
        AnthropicModelConfig(api_key="sk-test", max_output_tokens=64, usage_log_path=usage)
    )
    with patch("urllib.request.urlopen", lambda request, timeout=None: _FakeResponse("{}")):
        client.generate("hello", schema_constrained=False)
    row = json.loads(usage.read_text().splitlines()[0])
    assert row["prompt_chars"] == 5 and row["covered_chars"] is None
