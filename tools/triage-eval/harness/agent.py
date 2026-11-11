"""Config-driven agent loop shared by configs A, B and C.

Only this file's behavior is allowed to change across the three configurations (prompt vs
prompt+schema vs prompt+schema+memory); the model underneath never changes. A benchmark adapter
(adapters/) supplies the `ModelClient` and `Environment` implementations; this module only
orchestrates them.

`run_episode` owns the E5 escalation trigger: the harness-owned turn cap (`max_turns`) and
elapsed-time cap (`max_elapsed_seconds`) both live here, not in `harness/episode_runner.py`, because this
is the one place shared by every caller (live-cluster episodes AND the config A/B/C benchmark
adapters) that actually holds `time.monotonic()` and the turn-by-turn state. `EpisodeResult.
e5_triggered` is what `harness/triage_assembly.py` reads to fold E5 into `motivo_escalacao` --
this module does not know about TriageRecord or escalation codes, it only reports the fact.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

# Same head+tail truncation the observations already get (dependency-free, see episode_runner.py).
from adapters.aiopslab_env import cap_observation, cap_within_share
from harness.prompt_accounting import CallChars, set_call_chars

from .envelope import _C1_EXEC_ALLOWED_BINARIES, _SHELL_METACHARACTERS
from .fault_table import CAUSE_CLASSES
from .memory import Episode, EpisodicMemory, truncate_outcome
from .schema import SchemaValidationError, validate_action

logger = logging.getLogger("sslm.agent")

# The format is spelled out in full because the vague version was expensive: without the explicit
# range the model returned confidence as a percentage (70, 80) and the schema (maximum 1) rejected
# half the turns of a real run.
#
# The cause_class list comes from fault_table.CAUSE_CLASSES, never written by hand: without the list
# in the prompt, the path without constrained decoding (config_a) would only learn the enum from
# the schema error.
_CAUSE_CLASS_CHOICES = ", ".join(f'"{c}"' for c in CAUSE_CLASSES)

SYSTEM_PROMPT = (
    "You are an SRE agent. You are given a task description of a degraded system. "
    "Diagnose the root cause and propose one action at a time. Respond with ONLY a single "
    "JSON object (no prose, no code fences) with these fields: "
    "diagnosis (string), "
    f"cause_class (exactly one of {_CAUSE_CLASS_CHOICES}; use \"unknown\" when the evidence "
    "does not support any of them), "
    'action_type (exactly one of "inspect", "remediate", "escalate", '
    '"submit"), target (string), command (shell command string when action_type is inspect '
    "or remediate, else null), confidence (number between 0.0 and 1.0)."
)

# Tool contract. A measured 9B run logged 40 envelope denials, 27 of them shell metacharacters (`| grep`,
# `2>&1`, `&&`), and every wrong triage came from an episode with a denial. The envelope never
# lets those commands through and SYSTEM_PROMPT never said so, so each denial spent one of the 8
# turns (E5) on a command that could not run. This text says it once, up front. The lists come
# from the envelope's own constants so the prompt cannot drift from what classify() enforces.
#
# Off by default: unset keeps every earlier window's prompt byte for byte. A value that is neither
# an off value nor "on" raises, same as the budget: a typo must stop the wiring, not run a paid
# window without the text it was meant to carry.
TOOL_CONTRACT_ENV_VAR = "SSLM_TOOL_CONTRACT"
_TOOL_CONTRACT_ON_VALUES = {"1", "true", "yes", "on"}

TOOL_CONTRACT_PROMPT = (
    "\nCommand contract (commands that break it are refused and cost you a turn): "
    "one kubectl command per action, with the verb first (kubectl get pods -n <ns>, never "
    "kubectl --namespace <ns> get). It runs directly, not through a shell, so these are refused: "
    + " ".join(f"`{m}`" for m in _SHELL_METACHARACTERS)
    + ". Do not pipe to grep, redirect (2>&1) or chain; select with kubectl's own flags "
    "(-l, --field-selector, -o name, -o jsonpath=... without a pipe, --tail). "
    "kubectl exec only runs "
    + ", ".join(sorted(_C1_EXEC_ALLOWED_BINARIES))
    + " and never in a database pod (mongo, memcached). No other tool exists "
    "(no mongo, mongosh, etcdctl, shell): read the database state through the app's logs and "
    "its pods instead."
)


def tool_contract_from_env(env: dict[str, str] | None = None) -> bool:
    """False when TOOL_CONTRACT_ENV_VAR is unset, empty or an "off" value; True for "1", "true",
    "yes", "on". Raises ValueError on anything else. `env` defaults to `os.environ`."""
    raw = (env if env is not None else os.environ).get(TOOL_CONTRACT_ENV_VAR)
    if raw is None or not raw.strip():
        return False
    value = raw.strip().lower()
    if value in _HISTORY_BUDGET_OFF_VALUES:
        return False
    if value in _TOOL_CONTRACT_ON_VALUES:
        return True
    raise ValueError(f"{TOOL_CONTRACT_ENV_VAR}={raw!r} is neither on nor off")


# How many earlier turns of the SAME episode go into the prompt. Without history the model only
# saw the last observation and cycled between 2 deterministic states (measured on a real run);
# with it the loop can accumulate a diagnosis. Shared by the 3 configs, so it does not confound
# the A/B/C comparison (config C is memory ACROSS episodes, a different thing).
HISTORY_WINDOW = 5

# E5: "episode cap: 8 turns or 15 min without a final decision ... fired by the HARNESS, not by
# the model". This module is the harness half of that sentence: failure mode F4 (measured on the
# first config matrix) showed the MODEL never notices its own loop, so the wall-clock cap has to
# live where the caller cannot lie about it -- here, next to `time.monotonic()`, not in a prompt
# instruction the model could ignore. `max_turns` is the turn half; this constant is the other half (elapsed-time cap), 15 min = 900s.
DEFAULT_MAX_ELAPSED_SECONDS = 900.0

# adapters/aiopslab_env.py.OBSERVATION_MAX_CHARS caps ONE observation (12_000 chars) so a single
# giant command output cannot blow the prompt by itself. It does not cap the SUM: this module's
# own `_build_prompt` stacks up to HISTORY_WINDOW (5) prior turns plus the current observation,
# and 6 observations sitting at the per-observation cap is enough on its own to overflow the
# model's context window even though none of them individually violates it. A round-2 run hit
# exactly this on revoke_auth_mongodb (kubectl dumps across several turns): prompt measured
# at 17_084 tokens against vLLM's max_model_len of 16_384, 3 episodes per seed, HTTP 400 surfaced
# as ModelUnavailable. This budget bounds the history section's TOTAL size, trimming the oldest
# turns first (least relevant to the next decision) until it fits, regardless of what
# max_model_len happens to be configured to -- raising max_model_len only buys headroom until the
# next oversized dump, it does not bound the agent's own prompt growth.
#
# Chars, not tokens, matching cap_observation's own reasoning: no tokenizer dependency here.
# Budgeted off max_output_tokens=2048 (configs/base.yaml) reserved in the same context window,
# leaving ~14_336 input tokens at max_model_len=16_384; ~3.5 chars/token is the ratio
# aiopslab_env.py's own docstring reports from a live incident (~4MB dump -> 1_174_469 tokens).
# 30_000 chars leaves comfortable margin below that for the system prompt, task description,
# memory section and the always-kept last observation (up to OBSERVATION_MAX_CHARS on its own).
PROMPT_HISTORY_MAX_CHARS = 30_000

_HISTORY_HEADER = "\nEarlier turns this episode (oldest first), do not repeat them:"

# Switch for the budget above (API comparison arm). The first model rounds ran before the history
# budget existed, so the comparator arm can turn it off to send the exact earlier prompt and keep
# the model as the only intended difference. Default ON: later rounds keep the budget, and an unset
# or unrecognized value degrades to ON (the budgeted behavior), never to the unbounded prompt. Read at wiring time (run_matrix.py, episode_runner.py), never inside the loop.
HISTORY_BUDGET_ENV_VAR = "SSLM_HISTORY_BUDGET"
_HISTORY_BUDGET_OFF_VALUES = {"0", "false", "no", "off"}


def history_budget_enabled(env: dict[str, str] | None = None) -> bool:
    """True unless HISTORY_BUDGET_ENV_VAR is explicitly set to a recognized "off" value. `env`
    defaults to `os.environ`; tests pass an explicit mapping."""
    raw = (env if env is not None else os.environ).get(HISTORY_BUDGET_ENV_VAR)
    if raw is None:
        return True
    return raw.strip().lower() not in _HISTORY_BUDGET_OFF_VALUES


# Whole-call budget. A measured 7B run degenerated on every call between 8.6k and 14.5k tokens of a
# 16_384 window, and 17 of its episodes did so on their FIRST call, before any history existed:
# the scripted collection alone (webhook, Prometheus ranges, pods, events, logs, each source up to
# OBSERVATION_MAX_CHARS) already filled the call. PROMPT_HISTORY_MAX_CHARS cannot reach that, so
# this budget bounds the three variable sections together: the scripted context (split across its
# sources by EpisodeContext.as_prompt_text), the history and the last observation. SYSTEM_PROMPT
# (~930 chars, 232 tokens) and the section headers sit outside it.
#
# Chars, like PROMPT_HISTORY_MAX_CHARS, but sized off a measured ratio: Qwen2.5-7B's own tokenizer
# over that run's evidence excerpts gave 2.58 chars/token overall (logs 1.75, events 2.55,
# pods JSON 2.30, Prometheus 2.99), not the ~3.5 the history budget above assumed. The pre-registered
# setting is 14_000 (about 5.4k tokens plus the system prompt at that ratio).
#
# Unset (or "off") keeps every earlier window's prompt byte for byte. A value that is neither off
# nor an integer >= PROMPT_BUDGET_MIN_CHARS raises: a typo must stop the wiring, not run a paid
# window on the unbounded prompt it was meant to replace.
PROMPT_BUDGET_ENV_VAR = "SSLM_PROMPT_BUDGET_CHARS"
PROMPT_BUDGET_MIN_CHARS = 4_000


@dataclass(frozen=True)
class PromptBudget:
    """Split of `total_chars`: half to the scripted context, a quarter each to the history and
    to the last observation. Fixed shares, not tuned to the data: the context is what the model
    sees with no command run, the other two are what tool use adds per turn."""

    total_chars: int

    @property
    def context_chars(self) -> int:
        return self.total_chars // 2

    @property
    def history_chars(self) -> int:
        return self.total_chars // 4

    @property
    def observation_chars(self) -> int:
        return self.total_chars - self.context_chars - self.history_chars


def prompt_budget_from_env(env: dict[str, str] | None = None) -> PromptBudget | None:
    """None when PROMPT_BUDGET_ENV_VAR is unset, empty or an "off" value; else the budget.
    Raises ValueError on anything else. `env` defaults to `os.environ`."""
    raw = (env if env is not None else os.environ).get(PROMPT_BUDGET_ENV_VAR)
    if raw is None or not raw.strip() or raw.strip().lower() in _HISTORY_BUDGET_OFF_VALUES:
        return None
    try:
        total = int(raw.strip().replace("_", ""))
    except ValueError as exc:
        raise ValueError(f"{PROMPT_BUDGET_ENV_VAR}={raw!r} is not an integer char count") from exc
    if total < PROMPT_BUDGET_MIN_CHARS:
        raise ValueError(
            f"{PROMPT_BUDGET_ENV_VAR}={total} is below the {PROMPT_BUDGET_MIN_CHARS} chars floor"
        )
    return PromptBudget(total)


class ModelClient(Protocol):
    """Implemented by the benchmark adapter's LLM backend (e.g. a vLLM server)."""

    def generate(self, prompt: str, *, schema_constrained: bool) -> str:
        """Return raw model output. When schema_constrained is True, the implementation is
        expected to use constrained decoding (e.g. Outlines/guidance) so the output already
        parses as JSON matching AGENT_ACTION_SCHEMA."""
        ...


class Environment(Protocol):
    """Implemented by the benchmark adapter (ITBench or AIOpsLab)."""

    def task_description(self) -> str: ...

    def apply_action(self, action: dict) -> tuple[bool, str]:
        """Apply the action in the live/simulated environment.

        Returns (task_solved, observation) where observation is fed back to the agent on the
        next turn when task_solved is False.
        """
        ...


@dataclass
class TurnResult:
    output_tokens: int
    schema_valid: bool
    action: dict | None
    solved: bool
    observation: str
    # The model's raw output this turn. Without it a failing episode leaves no evidence at all of
    # what the model said (a real run had 10 malformed turns and zero transcript to diagnose), and
    # per-task logging needs that evidence.
    raw_output: str = ""


@dataclass
class AgentConfig:
    schema_constrained_decoding: bool
    memory_enabled: bool
    memory_top_k: int = 3
    # False reproduces the prompt from before the history budget byte for byte (no
    # PROMPT_HISTORY_MAX_CHARS trim). Only
    # the API comparison arm sets it, via SSLM_HISTORY_BUDGET=off; see history_budget_enabled().
    history_budget: bool = True
    # None keeps the prompt as above. Set, it also caps each history entry and the last
    # observation (the context share is applied by the environment, see PromptBudget).
    prompt_budget: PromptBudget | None = None
    # True appends TOOL_CONTRACT_PROMPT right after SYSTEM_PROMPT. False keeps the prompt as above.
    tool_contract: bool = False


class Agent:
    """One episode of observe -> decide -> act, config-driven per AgentConfig."""

    def __init__(
        self,
        model: ModelClient,
        config: AgentConfig,
        memory: EpisodicMemory | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.memory = memory
        if config.memory_enabled and memory is None:
            raise ValueError("memory_enabled requires an EpisodicMemory instance")

    def _build_prompt(
        self,
        task_description: str,
        observation: str | None,
        history: list["TurnResult"] | None = None,
    ) -> str:
        return self._build_prompt_measured(task_description, observation, history)[0]

    def _build_prompt_measured(
        self,
        task_description: str,
        observation: str | None,
        history: list["TurnResult"] | None = None,
        covered_context_chars: int | None = None,
    ) -> tuple[str, CallChars]:
        """The prompt plus its char measure (harness.prompt_accounting). `covered_context_chars`
        is what the environment put in the context share; None means the environment cannot say,
        and the whole task description is counted, which can only overstate the covered part."""
        history_section_chars = 0
        observation_section_chars = 0
        parts = [SYSTEM_PROMPT]
        if self.config.tool_contract:
            parts.append(TOOL_CONTRACT_PROMPT)
        parts.append(f"\nTask:\n{task_description}")

        if self.config.memory_enabled and self.memory is not None:
            episodes = self.memory.retrieve(task_description, top_k=self.config.memory_top_k)
            if episodes:
                parts.append("\nRelevant past attempts:")
                for episode in episodes:
                    parts.append(
                        f"- task: {episode.task_description}\n"
                        f"  action: {episode.action}\n"
                        f"  environment response: {episode.outcome}\n"
                        f"  episode verdict: {episode.verdict}"
                    )

        budget = self.config.prompt_budget
        if history:
            entries = []
            for turn in history[-HISTORY_WINDOW:]:
                action_repr = (
                    json.dumps(turn.action) if turn.action is not None else "(invalid output)"
                )
                result = turn.observation
                if budget is not None:
                    # Per entry first, a third of the share: with the action line and the cut
                    # marker on top, the two newest entries still fit together.
                    result = cap_observation(result, budget.history_chars // 3)
                entries.append(f"- action: {action_repr}\n  result: {result}")
            # The header and the "\n" joins count against the same budget, so the whole section
            # as rendered (not just the entry text) stays within PROMPT_HISTORY_MAX_CHARS.
            if budget is not None:
                entries = _trim_history_to_budget(entries, budget.history_chars)
            elif self.config.history_budget:
                entries = _trim_history_to_budget(
                    entries, PROMPT_HISTORY_MAX_CHARS - len(_HISTORY_HEADER)
                )
            if entries:
                parts.append(_HISTORY_HEADER)
                parts.extend(entries)
                history_section_chars = len("\n".join([_HISTORY_HEADER, *entries]))

        if observation:
            if budget is not None:
                observation = cap_within_share(observation, budget.observation_chars)
            observation_section_chars = len(observation)
            parts.append(f"\nLast observation:\n{observation}")

        prompt = "\n".join(parts)
        context_chars = covered_context_chars if covered_context_chars is not None else len(task_description)
        chars = CallChars(
            prompt_chars=len(prompt),
            covered_chars=(
                context_chars + history_section_chars + observation_section_chars if budget is not None else None
            ),
            context_chars=context_chars if budget is not None else None,
            history_chars=history_section_chars if budget is not None else None,
            observation_chars=observation_section_chars if budget is not None else None,
            budget_chars=budget.total_chars if budget is not None else None,
        )
        return prompt, chars

    def run_episode(
        self,
        environment: Environment,
        task_id: str,
        *,
        max_turns: int = 10,
        max_elapsed_seconds: float = DEFAULT_MAX_ELAPSED_SECONDS,
    ) -> "EpisodeResult":
        start = time.monotonic()
        task_description = environment.task_description()
        observation: str | None = None
        turns: list[TurnResult] = []
        solved = False
        # E5: set to a non-None reason string the moment either half of the cap
        # trips (turn count via the `for...else` below, elapsed time via the explicit check
        # inside the loop), cleared to None implicitly by never being set when `solved` breaks
        # the loop first. `EpisodeResult.e5_triggered` reads this alongside `solved` so a episode
        # that happens to solve on its LAST allowed turn is never mislabeled as an E5 escalation.
        e5_reason: str | None = None

        for _ in range(max_turns):
            elapsed_before_turn = time.monotonic() - start
            if elapsed_before_turn >= max_elapsed_seconds:
                e5_reason = (
                    f"time cap reached ({elapsed_before_turn:.1f}s >= {max_elapsed_seconds:.0f}s) "
                    "without a final decision (E5)"
                )
                logger.warning(
                    "episode %s: E5 time cap reached after %d turn(s) (%.1fs >= %.0fs)",
                    task_id,
                    len(turns),
                    elapsed_before_turn,
                    max_elapsed_seconds,
                )
                break
            covered_context = getattr(environment, "covered_context_chars", None)
            prompt, call_chars = self._build_prompt_measured(
                task_description,
                observation,
                history=turns[:-1],
                covered_context_chars=covered_context() if callable(covered_context) else None,
            )
            set_call_chars(call_chars)
            try:
                raw_output = self.model.generate(
                    prompt, schema_constrained=self.config.schema_constrained_decoding
                )
            finally:
                set_call_chars(None)

            action, schema_valid, parse_error = self._parse_action(raw_output)
            if action is None:
                # Feed the parse error into the next turn, including the exact REASON for the
                # rejection: the generic feedback did not say why, and on a real run the model
                # repeated confidence as a percentage (80) on every odd turn without ever finding
                # out the ceiling was 1.0.
                observation = (
                    f"Your previous output was rejected: {parse_error}. "
                    "Respond with ONLY one JSON object with fields diagnosis, cause_class "
                    "(one of the classes listed in the instructions), action_type "
                    '(one of "inspect", "remediate", "escalate", "submit"), target, command, '
                    "confidence (a number between 0.0 and 1.0). No prose, no code fences."
                )
                turns.append(
                    TurnResult(
                        output_tokens=_approx_tokens(raw_output),
                        schema_valid=False,
                        action=None,
                        solved=False,
                        observation=f"malformed output, retry ({parse_error})",
                        raw_output=raw_output,
                    )
                )
                continue

            solved, observation = environment.apply_action(action)
            turns.append(
                TurnResult(
                    output_tokens=_approx_tokens(raw_output),
                    schema_valid=schema_valid,
                    action=action,
                    solved=solved,
                    observation=observation,
                    raw_output=raw_output,
                )
            )

            if self.config.memory_enabled and self.memory is not None and self._worth_remembering(turns):
                self.memory.add(
                    Episode(
                        task_id=task_id,
                        task_description=task_description,
                        action=action,
                        outcome=truncate_outcome(observation),
                    )
                )

            if solved:
                break
        else:
            # `for...else`: reached only when the loop ran out of `max_turns` iterations without
            # ever hitting a `break` above (neither the time-cap break nor `if solved: break`).
            # That is exactly E5's other half -- the turn cap -- and `continue` (the malformed-
            # output retry path) does not suppress this `else`, so a run that spent every turn on
            # malformed output still reports the turn cap correctly instead of silently reporting
            # nothing. `solved` is guaranteed False here (the only way to reach `else` with
            # `solved=True` would require `break` to have fired), but the check is kept explicit
            # rather than assumed, matching this repo's "state by rule, not by inference" habit.
            if not solved:
                e5_reason = f"turn cap reached ({max_turns} turns) without a final decision (E5)"
                logger.warning("episode %s: E5 turn cap reached (%d turns)", task_id, max_turns)

        return EpisodeResult(
            task_id=task_id,
            solved=solved,
            turns=turns,
            elapsed_seconds=time.monotonic() - start,
            e5_reason=e5_reason if not solved else None,
        )

    @staticmethod
    def _worth_remembering(turns: list["TurnResult"]) -> bool:
        """A turn taken right after a schema rejection reacts to the harness, not to the system.

        Week 1 matrix (2026-09-01): config_c diagnosed "YAML with unterminated quotes" as the
        incident's root cause in several turns. That text was the schema-retry feedback of an
        earlier turn, stored in memory inside the next action's `diagnosis`, and retrieved
        into later episodes as if it were operational experience. Memory is for
        (task, action, environment response); a reaction to retry feedback is none of those.
        """
        return len(turns) < 2 or turns[-2].schema_valid

    @staticmethod
    def _parse_action(raw_output: str) -> tuple[dict | None, bool, str | None]:
        """Returns (action, schema_valid, parse_error).

        parse_error carries WHY the output was rejected (first line of the jsonschema
        message), so the retry feedback can name the exact violation instead of a generic
        "malformed" that the model cannot act on.
        """
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            return None, False, f"not valid JSON ({exc})"

        try:
            validate_action(payload)
        except SchemaValidationError as exc:
            first_line = str(exc).splitlines()[0]
            return None, False, f"JSON does not match the required schema: {first_line}"

        return payload, True, None


@dataclass
class EpisodeResult:
    task_id: str
    solved: bool
    turns: list[TurnResult]
    elapsed_seconds: float
    # E5: non-None exactly when `run_episode` gave up unsolved because it hit the
    # turn cap or the elapsed-time cap (never both messages at once -- whichever tripped first
    # wins). None whenever `solved` is True, by construction (`run_episode` never sets both).
    # `harness/triage_assembly.py` reads `e5_triggered` (not this field directly) to decide
    # whether "E5" belongs in `motivo_escalacao` -- see that module's own docstring.
    e5_reason: str | None = None

    @property
    def total_output_tokens(self) -> int:
        return sum(turn.output_tokens for turn in self.turns)

    @property
    def attempt_count(self) -> int:
        return len(self.turns)

    @property
    def e5_triggered(self) -> bool:
        """True iff this episode ended unsolved via the harness-owned cap (turns or elapsed
        time), never via the model's own `escalate`/`submit`. `not self.solved` is required
        explicitly (not just `self.e5_reason is not None`) because `LiveClusterEnvironment.
        apply_action` returns `solved=True` for BOTH `escalate` and `submit` (see that module's
        own honest-limit docstring on that ambiguity) -- an episode cannot be solved-by-model AND
        E5-escalated-by-harness at the same time, and `run_episode` already guarantees
        `e5_reason` is None whenever `solved` is True, but this property does not trust that
        invariant blindly either."""
        return not self.solved and self.e5_reason is not None


def _approx_tokens(text: str) -> int:
    """Cheap token proxy (whitespace split) so the harness has no tokenizer dependency.

    Good enough as a relative cost signal across configs A/B/C on the same base model; not a
    substitute for the real tokenizer count when the final number is reported (see metrics.py).
    """
    return len(text.split())


def _trim_history_to_budget(entries: list[str], max_chars: int) -> list[str]:
    """Drops the oldest entries (the front of the list) until the total fits `max_chars`.

    `entries` is already oldest-first and already windowed to HISTORY_WINDOW; this only handles
    the case where even that window is too big (PROMPT_HISTORY_MAX_CHARS, see its own docstring
    for the incident that motivated it). Always keeps the single most recent entry even if it
    alone exceeds the budget: that turn's observation is already bounded by
    adapters.aiopslab_env.OBSERVATION_MAX_CHARS, so an entry that size is a true outlier, and
    dropping the only context left would leave "do not repeat them" with nothing to point at.

    Each entry is counted with the "\n" that joins it into the prompt, so the budget matches the
    rendered section and not just the entry text.
    """
    start = 0
    total = sum(len(entry) + 1 for entry in entries)
    while start < len(entries) - 1 and total > max_chars:
        total -= len(entries[start]) + 1
        start += 1
    return entries[start:]
