import json

from harness.agent import (
    HISTORY_WINDOW,
    PROMPT_BUDGET_MIN_CHARS,
    PROMPT_HISTORY_MAX_CHARS,
    SYSTEM_PROMPT,
    Agent,
    AgentConfig,
    PromptBudget,
    TurnResult,
    _trim_history_to_budget,
    history_budget_enabled,
    prompt_budget_from_env,
    TOOL_CONTRACT_PROMPT,
    tool_contract_from_env,
)
import pytest
from harness.fault_table import CAUSE_CLASSES, UNCLASSIFIED_CLASS
from harness.memory import EpisodicMemory


class StubModel:
    def __init__(self, outputs):
        self._outputs = iter(outputs)

    def generate(self, prompt: str, *, schema_constrained: bool) -> str:
        return next(self._outputs)


class StubEnvironment:
    def __init__(self, solved_on_turn: int):
        self.solved_on_turn = solved_on_turn
        self.turn = 0

    def task_description(self) -> str:
        return "checkout service returning 500s"

    def apply_action(self, action: dict):
        self.turn += 1
        solved = self.turn >= self.solved_on_turn
        return solved, "ok" if solved else "not yet"


VALID_ACTION = json.dumps(
    {
        "diagnosis": "checkout overloaded",
        "cause_class": "unknown",
        "action_type": "remediate",
        "target": "checkout",
        "command": "kubectl rollout restart deployment/checkout",
        "confidence": 0.9,
    }
)


def test_solved_on_first_turn_stops_loop():
    agent = Agent(
        model=StubModel([VALID_ACTION]),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )
    result = agent.run_episode(StubEnvironment(solved_on_turn=1), task_id="t1")

    assert result.solved is True
    assert result.attempt_count == 1


def test_malformed_output_retries_without_crashing():
    agent = Agent(
        model=StubModel(["not json", VALID_ACTION]),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )
    result = agent.run_episode(StubEnvironment(solved_on_turn=1), task_id="t1")

    assert result.solved is True
    assert result.attempt_count == 2
    assert result.turns[0].schema_valid is False
    assert result.turns[0].raw_output == "not json"


def test_malformed_output_feeds_parse_error_back_into_next_prompt():
    class PromptRecordingModel:
        def __init__(self, outputs):
            self._outputs = iter(outputs)
            self.prompts: list[str] = []

        def generate(self, prompt: str, *, schema_constrained: bool) -> str:
            self.prompts.append(prompt)
            return next(self._outputs)

    model = PromptRecordingModel(["not json", VALID_ACTION])
    agent = Agent(
        model=model,
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )
    agent.run_episode(StubEnvironment(solved_on_turn=1), task_id="t1")

    # Without feeding the parse error back, the 2nd prompt is identical to the 1st and a greedy
    # model repeats the same malformed output until max_turns (the pattern of a real run).
    assert model.prompts[0] != model.prompts[1]
    assert "was rejected" in model.prompts[1]
    # The exact reason travels in the feedback (here: it was not JSON).
    assert "not valid JSON" in model.prompts[1]


def test_schema_violation_feedback_names_the_exact_violation():
    class PromptRecordingModel:
        def __init__(self, outputs):
            self._outputs = iter(outputs)
            self.prompts: list[str] = []

        def generate(self, prompt: str, *, schema_constrained: bool) -> str:
            self.prompts.append(prompt)
            return next(self._outputs)

    # confidence as a percentage: the real error of a measured run, repeated on every odd turn
    # because the generic feedback never said the ceiling was 1.0.
    percent_confidence = json.dumps(
        {
            "diagnosis": "x",
            "cause_class": "unknown",
            "action_type": "inspect",
            "target": "ns",
            "command": "kubectl get pods",
            "confidence": 80,
        }
    )
    model = PromptRecordingModel([percent_confidence, VALID_ACTION])
    agent = Agent(
        model=model,
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )
    agent.run_episode(StubEnvironment(solved_on_turn=1), task_id="t1")

    assert "maximum" in model.prompts[1]


def test_prompt_includes_earlier_turns_of_the_episode():
    class PromptRecordingModel:
        def __init__(self, outputs):
            self._outputs = iter(outputs)
            self.prompts: list[str] = []

        def generate(self, prompt: str, *, schema_constrained: bool) -> str:
            self.prompts.append(prompt)
            return next(self._outputs)

    model = PromptRecordingModel([VALID_ACTION] * 3)
    agent = Agent(
        model=model,
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )
    agent.run_episode(StubEnvironment(solved_on_turn=3), task_id="t1", max_turns=3)

    # Without the episode's history the model only sees the last observation and cycles between 2
    # states (measured on a real run). The 1st prompt has no history; the 3rd does.
    assert "Earlier turns this episode" not in model.prompts[0]
    assert "Earlier turns this episode" in model.prompts[2]
    assert "kubectl rollout restart deployment/checkout" in model.prompts[2]


def test_trim_history_to_budget_drops_oldest_first():
    entries = ["a" * 100, "b" * 100, "c" * 100]

    trimmed = _trim_history_to_budget(entries, max_chars=250)

    # "a" (oldest) is dropped first; "b" and "c" together (200 chars) fit the budget.
    assert trimmed == ["b" * 100, "c" * 100]


def test_trim_history_to_budget_counts_join_separators():
    entries = ["a" * 100, "b" * 100]

    # 200 chars of text, but 202 once each entry's joining "\n" is counted: over a 200 budget.
    assert _trim_history_to_budget(entries, max_chars=200) == ["b" * 100]
    assert _trim_history_to_budget(entries, max_chars=202) == entries


def test_trim_history_to_budget_keeps_single_oversized_last_entry():
    entries = ["a" * 100, "b" * 100, "c" * 9000]

    # Even the single most recent entry alone blows the budget; it must not be dropped, or the
    # harness would be left with no history and nothing to tell the model not to repeat.
    trimmed = _trim_history_to_budget(entries, max_chars=500)

    assert trimmed == ["c" * 9000]


def test_trim_history_to_budget_noop_when_already_under_budget():
    entries = ["a" * 10, "b" * 10]

    assert _trim_history_to_budget(entries, max_chars=1000) == entries


def test_history_section_stays_within_budget_and_keeps_most_recent_turns():
    # revoke_auth_mongodb, measured on a round-2 run: several turns of kubectl dumps, each already
    # capped by adapters.aiopslab_env.cap_observation at 12_000 chars, still overflowed
    # max_model_len once HISTORY_WINDOW stacked enough of them into one prompt.
    large_observation = "x" * 12_000

    class PromptRecordingModel:
        def __init__(self, outputs):
            self._outputs = iter(outputs)
            self.prompts: list[str] = []

        def generate(self, prompt: str, *, schema_constrained: bool) -> str:
            self.prompts.append(prompt)
            return next(self._outputs)

    class LargeObservationEnvironment:
        def task_description(self) -> str:
            return "mongodb auth revoked for geo service"

        def apply_action(self, action: dict):
            return False, large_observation

    model = PromptRecordingModel([VALID_ACTION] * 6)
    agent = Agent(
        model=model,
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )
    agent.run_episode(LargeObservationEnvironment(), task_id="t1", max_turns=6)

    last_prompt = model.prompts[-1]
    history_section = last_prompt.split("Earlier turns this episode")[1].split("Last observation")[0]

    assert len(history_section) <= PROMPT_HISTORY_MAX_CHARS
    # The most recent prior turn (closest to the current decision) survives the trim; only the
    # oldest ones are dropped to make the budget.
    assert large_observation in history_section


def _pre_history_budget_build_prompt(agent, task_description, observation, history=None):
    """Verbatim copy of Agent._build_prompt from before the history budget (the agent the first
    model rounds ran). Frozen here on purpose: the pre-registered API comparison arm must send
    exactly this, and CI has no git history to read it from."""
    parts = [SYSTEM_PROMPT, f"\nTask:\n{task_description}"]

    if agent.config.memory_enabled and agent.memory is not None:
        episodes = agent.memory.retrieve(task_description, top_k=agent.config.memory_top_k)
        if episodes:
            parts.append("\nRelevant past attempts:")
            for episode in episodes:
                parts.append(
                    f"- task: {episode.task_description}\n"
                    f"  action: {episode.action}\n"
                    f"  environment response: {episode.outcome}\n"
                    f"  episode verdict: {episode.verdict}"
                )

    if history:
        parts.append("\nEarlier turns this episode (oldest first), do not repeat them:")
        for turn in history[-HISTORY_WINDOW:]:
            action_repr = json.dumps(turn.action) if turn.action is not None else "(invalid output)"
            parts.append(f"- action: {action_repr}\n  result: {turn.observation}")

    if observation:
        parts.append(f"\nLast observation:\n{observation}")

    return "\n".join(parts)


def _oversized_history(n: int = 6) -> list[TurnResult]:
    # Same shape as the measured round-2 revoke_auth_mongodb overflow: several turns at the 12_000 char
    # observation cap, far over PROMPT_HISTORY_MAX_CHARS once windowed together. One invalid turn
    # so the "(invalid output)" branch is covered too.
    turns = []
    for i in range(n):
        action = None if i == 2 else {"action_type": "inspect", "command": f"kubectl get pods -n x{i}"}
        turns.append(
            TurnResult(
                output_tokens=1,
                schema_valid=action is not None,
                action=action,
                solved=False,
                observation=chr(ord("a") + i) * 12_000,
            )
        )
    return turns


def test_history_budget_off_builds_the_exact_unbudgeted_prompt_on_a_history_that_gets_trimmed():
    history = _oversized_history()
    off = Agent(
        model=StubModel([]),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False, history_budget=False),
    )
    on = Agent(
        model=StubModel([]),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )

    expected = _pre_history_budget_build_prompt(off, "mongodb auth revoked", "last obs", history)
    # Precondition: this history really is one the budget trims, so the comparison means something.
    assert on._build_prompt("mongodb auth revoked", "last obs", history) != expected
    assert off._build_prompt("mongodb auth revoked", "last obs", history) == expected


def test_history_budget_off_matches_the_unbudgeted_prompt_on_every_turn_of_a_real_episode():
    class RecordingModel:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, *, schema_constrained):
            self.prompts.append(prompt)
            return VALID_ACTION

    class BigObservationEnv:
        def task_description(self):
            return "mongodb auth revoked for geo service"

        def apply_action(self, action):
            return False, "x" * 12_000

    model = RecordingModel()
    agent = Agent(
        model=model,
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False, history_budget=False),
    )
    result = agent.run_episode(BigObservationEnv(), task_id="t1", max_turns=7)

    for i, prompt in enumerate(model.prompts):
        prior = result.turns[:i]
        observation = prior[-1].observation if prior else None
        # Same slicing as run_episode: the last turn feeds "Last observation", the rest history.
        assert prompt == _pre_history_budget_build_prompt(
            agent, "mongodb auth revoked for geo service", observation, prior[:-1]
        )
    assert len(model.prompts[-1]) > PROMPT_HISTORY_MAX_CHARS  # untrimmed, like the first model rounds


def test_default_agent_config_keeps_the_history_budget():
    config = AgentConfig(schema_constrained_decoding=False, memory_enabled=False)
    agent = Agent(model=StubModel([]), config=config)

    assert config.history_budget is True
    prompt = agent._build_prompt("t", "obs", _oversized_history())
    section = prompt.split("Earlier turns this episode")[1].split("Last observation")[0]
    assert len(section) <= PROMPT_HISTORY_MAX_CHARS


def test_history_budget_env_switch_defaults_on_and_only_explicit_off_disables_it():
    assert history_budget_enabled({}) is True
    assert history_budget_enabled({"SSLM_HISTORY_BUDGET": "on"}) is True
    assert history_budget_enabled({"SSLM_HISTORY_BUDGET": "typo"}) is True
    for off in ("off", "OFF", "0", "false", "no", " off "):
        assert history_budget_enabled({"SSLM_HISTORY_BUDGET": off}) is False


def test_max_turns_gives_up_unsolved():
    agent = Agent(
        model=StubModel([VALID_ACTION] * 3),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )
    result = agent.run_episode(StubEnvironment(solved_on_turn=99), task_id="t1", max_turns=3)

    assert result.solved is False
    assert result.attempt_count == 3


def test_memory_enabled_records_episode_after_solve(tmp_path):
    memory = EpisodicMemory.load(tmp_path / "episodes.jsonl")
    agent = Agent(
        model=StubModel([VALID_ACTION]),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=True),
        memory=memory,
    )
    agent.run_episode(StubEnvironment(solved_on_turn=1), task_id="t1")

    assert len(memory.episodes) == 1
    # Per turn, memory stores the ENVIRONMENT's answer; the success label belongs to the eval,
    # written after finalize (set_verdict), and stays unknown until then.
    assert memory.episodes[0].outcome == "ok"
    assert memory.episodes[0].verdict == "unknown"


def test_memory_skips_turn_that_reacts_to_schema_retry_feedback(tmp_path):
    memory = EpisodicMemory.load(tmp_path / "episodes.jsonl")
    agent = Agent(
        model=StubModel(["not json", VALID_ACTION, VALID_ACTION]),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=True),
        memory=memory,
    )
    agent.run_episode(StubEnvironment(solved_on_turn=2), task_id="t1")

    # Turn 2 came right after a schema retry: it reacts to the harness, not to the system (the
    # measured "YAML with unterminated quotes" contamination). Turn 3 goes in.
    assert len(memory.episodes) == 1
    assert memory.episodes[0].outcome == "ok"


def test_prompt_renders_memory_with_environment_response_and_verdict(tmp_path):
    from harness.memory import Episode

    memory = EpisodicMemory.load(tmp_path / "episodes.jsonl")
    memory.add(
        Episode(
            task_id="old",
            task_description="checkout service returning 500s",
            action={"action_type": "inspect", "command": "kubectl get pods"},
            outcome="pod checkout-1 CrashLoopBackOff",
            verdict="failure",
        )
    )

    class PromptRecordingModel:
        def __init__(self, outputs):
            self._outputs = iter(outputs)
            self.prompts: list[str] = []

        def generate(self, prompt: str, *, schema_constrained: bool) -> str:
            self.prompts.append(prompt)
            return next(self._outputs)

    model = PromptRecordingModel([VALID_ACTION])
    agent = Agent(
        model=model,
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=True),
        memory=memory,
    )
    agent.run_episode(StubEnvironment(solved_on_turn=1), task_id="t1")

    assert "environment response: pod checkout-1 CrashLoopBackOff" in model.prompts[0]
    assert "episode verdict: failure" in model.prompts[0]


def test_memory_enabled_without_memory_instance_raises():
    try:
        Agent(
            model=StubModel([VALID_ACTION]),
            config=AgentConfig(schema_constrained_decoding=False, memory_enabled=True),
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError when memory_enabled but no memory given")


# ---------------------------------------------------------------------------
# E5: "episode cap: 8 turns or 15 min without a final decision ... fired by the HARNESS". Both
# halves of the cap live in run_episode/EpisodeResult.
# ---------------------------------------------------------------------------


def test_turn_cap_sets_e5_reason_and_e5_triggered():
    agent = Agent(
        model=StubModel([VALID_ACTION] * 3),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )
    result = agent.run_episode(StubEnvironment(solved_on_turn=99), task_id="t1", max_turns=3)

    assert result.solved is False
    assert result.attempt_count == 3
    assert result.e5_triggered is True
    assert "turn cap" in result.e5_reason
    assert "3 turns" in result.e5_reason


def test_time_cap_ends_the_episode_before_any_turn_and_sets_e5_reason():
    """max_elapsed_seconds=0.0 forces the very first iteration's elapsed check to trip
    immediately, before the model is ever asked for a turn -- proves the time cap is a real
    pre-turn gate, not just a post-hoc label on however many turns happened to run."""
    model = StubModel([VALID_ACTION] * 5)  # never consumed if the time cap works
    agent = Agent(
        model=model,
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )
    result = agent.run_episode(
        StubEnvironment(solved_on_turn=99), task_id="t1", max_turns=8, max_elapsed_seconds=0.0
    )

    assert result.solved is False
    assert result.attempt_count == 0
    assert result.e5_triggered is True
    assert "time cap" in result.e5_reason


def test_solved_on_the_last_allowed_turn_does_not_trigger_e5():
    """An episode that solves exactly on its final allowed turn must not be mislabeled as an E5
    escalation just because it also happened to exhaust max_turns -- `solved=True` wins."""
    agent = Agent(
        model=StubModel([VALID_ACTION] * 3),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )
    result = agent.run_episode(StubEnvironment(solved_on_turn=3), task_id="t1", max_turns=3)

    assert result.solved is True
    assert result.attempt_count == 3
    assert result.e5_triggered is False
    assert result.e5_reason is None


def test_escalate_or_submit_before_any_cap_does_not_trigger_e5():
    """`LiveClusterEnvironment.apply_action` returns solved=True for both escalate and submit
    (its own honest-limit docstring); either way, a model-initiated end must never read as a
    harness-owned E5 escalation."""
    agent = Agent(
        model=StubModel([VALID_ACTION]),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
    )
    result = agent.run_episode(StubEnvironment(solved_on_turn=1), task_id="t1", max_turns=8)

    assert result.solved is True
    assert result.e5_triggered is False


def test_system_prompt_lists_the_whole_cause_taxonomy():
    """config_a has no constrained decoding: the prompt is the only place the model learns the
    closed enum, so every class must be in it and the harness-only marker must not."""
    for cause_class in CAUSE_CLASSES:
        assert f'"{cause_class}"' in SYSTEM_PROMPT
    assert f'"{UNCLASSIFIED_CLASS}"' not in SYSTEM_PROMPT


# Whole-call prompt budget, pre-registered for the short 7B round.


def test_prompt_budget_env_is_off_unless_set_to_an_integer():
    assert prompt_budget_from_env({}) is None
    for off in ("", "  ", "off", "OFF", "0", "false", "no"):
        assert prompt_budget_from_env({"SSLM_PROMPT_BUDGET_CHARS": off}) is None
    assert prompt_budget_from_env({"SSLM_PROMPT_BUDGET_CHARS": "18000"}) == PromptBudget(18_000)
    assert prompt_budget_from_env({"SSLM_PROMPT_BUDGET_CHARS": " 18_000 "}) == PromptBudget(18_000)


@pytest.mark.parametrize("raw", ["on", "18k", "6000 tokens", "-5", str(PROMPT_BUDGET_MIN_CHARS - 1)])
def test_prompt_budget_env_refuses_what_it_cannot_read(raw):
    # A typo must stop the wiring, never fall back to the unbounded prompt.
    with pytest.raises(ValueError, match="SSLM_PROMPT_BUDGET_CHARS"):
        prompt_budget_from_env({"SSLM_PROMPT_BUDGET_CHARS": raw})


@pytest.mark.parametrize("total", [PROMPT_BUDGET_MIN_CHARS, 18_000, 18_003])
def test_prompt_budget_shares_add_up_to_the_total(total):
    budget = PromptBudget(total)
    assert budget.context_chars + budget.history_chars + budget.observation_chars == total
    assert budget.context_chars == total // 2


def test_prompt_budget_caps_history_and_last_observation():
    budget = PromptBudget(18_000)
    config = AgentConfig(schema_constrained_decoding=False, memory_enabled=False, prompt_budget=budget)
    agent = Agent(model=StubModel([]), config=config)

    prompt = agent._build_prompt("t", "z" * 12_000, _oversized_history())
    history = prompt.split("Earlier turns this episode")[1].split("\nLast observation:")[0]
    last = prompt.split("\nLast observation:\n")[1]

    assert len(history) <= budget.history_chars + len(":, do not repeat them:")
    # The two newest entries survive, each cut head+tail; the older ones are trimmed away.
    assert "f" * 100 in history and "e" * 100 in history and "d" * 100 not in history
    # cap_observation adds its marker on top of the limit; the bound is the limit plus that line.
    assert len(last) <= budget.observation_chars + 120
    assert last.startswith("z" * 100) and last.endswith("z" * 100)


def test_prompt_budget_none_keeps_the_default_prompt_byte_for_byte():
    plain = Agent(model=StubModel([]), config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False))
    explicit = Agent(
        model=StubModel([]),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False, prompt_budget=None),
    )
    history = _oversized_history()
    assert plain._build_prompt("t", "obs", history) == explicit._build_prompt("t", "obs", history)


def test_tool_contract_env_is_off_unless_set_to_on():
    assert tool_contract_from_env({}) is False
    for off in ("", " ", "0", "off", "false", "no"):
        assert tool_contract_from_env({"SSLM_TOOL_CONTRACT": off}) is False
    for on in ("1", "on", " True ", "yes"):
        assert tool_contract_from_env({"SSLM_TOOL_CONTRACT": on}) is True


@pytest.mark.parametrize("raw", ["maybe", "2", "onn"])
def test_tool_contract_env_refuses_what_it_cannot_read(raw):
    with pytest.raises(ValueError):
        tool_contract_from_env({"SSLM_TOOL_CONTRACT": raw})


def test_tool_contract_off_keeps_the_default_prompt_byte_for_byte():
    plain = Agent(model=StubModel([]), config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False))
    off = Agent(
        model=StubModel([]),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False, tool_contract=False),
    )
    assert plain._build_prompt("t", "obs") == off._build_prompt("t", "obs")
    assert "Command contract" not in plain._build_prompt("t", "obs")


def test_tool_contract_on_follows_the_system_prompt_and_precedes_the_task():
    agent = Agent(
        model=StubModel([]),
        config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False, tool_contract=True),
    )
    prompt = agent._build_prompt("the task", "obs")
    assert prompt.startswith(SYSTEM_PROMPT)
    assert prompt.index(SYSTEM_PROMPT) < prompt.index(TOOL_CONTRACT_PROMPT) < prompt.index("the task")


def test_tool_contract_text_comes_from_the_envelope_constants():
    from harness.envelope import _C1_EXEC_ALLOWED_BINARIES, _SHELL_METACHARACTERS

    for meta in _SHELL_METACHARACTERS:
        assert f"`{meta}`" in TOOL_CONTRACT_PROMPT
    for binary in _C1_EXEC_ALLOWED_BINARIES:
        assert binary in TOOL_CONTRACT_PROMPT


@pytest.mark.parametrize(
    "command",
    [
        "kubectl get pods -n ns | grep Running",
        "kubectl describe pod x -n ns 2>&1 | grep State",
        "kubectl get pods && kubectl get svc",
    ],
)
def test_the_commands_the_contract_forbids_are_the_ones_the_envelope_denies(command):
    from harness.envelope import ActionClass, EpisodeEnvelopeState, classify

    assert classify(command, EpisodeEnvelopeState()).action_class is ActionClass.C3
