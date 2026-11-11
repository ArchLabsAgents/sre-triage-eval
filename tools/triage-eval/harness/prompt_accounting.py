"""Chars of one model call, measured where the prompt is built and logged where the call is paid.

Why it exists: the target was "no call above 7_200 prompt_tokens", and a measured call of 8_141
tokens could not be attributed to the evidence or to something outside the char budget, because
the usage log only carried tokens. PromptBudget bounds chars, so
chars are what the window measures: per call, the part PromptBudget covers (scripted context,
history section, last observation) against the budget, plus the whole prompt for context.

What PromptBudget covers, and what it does not (stated once, tested in test_prompt_accounting.py):
  covered     the collected sources' text of the scripted context (context share), the history
              section as rendered, header and joins included (history share), the last observation
              as capped (observation share).
  not covered the system prompt, the "Task:" line, source headers and NOT-collected lines, the
              memory section, the "Last observation:" label. Those are fixed text or outside the
              budget by construction; `prompt_chars - covered_chars` is their sum.

Side channel, not a signature change: ModelClient.generate(prompt, *, schema_constrained) is the
protocol every arm implements, so the Agent publishes the measure of the call it is about to make in
a thread-local and the client that logs the call reads it. Cleared right after the call so a later
caller that never measured cannot inherit a stale figure; a call without a measure is logged with
`covered_chars: null`, which the smoke gate treats as missing chars.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

_local = threading.local()


@dataclass(frozen=True)
class CallChars:
    prompt_chars: int
    covered_chars: int | None  # None when no PromptBudget is on: nothing to compare against
    context_chars: int | None
    history_chars: int | None
    observation_chars: int | None
    budget_chars: int | None

    def as_log_fields(self) -> dict:
        return asdict(self)


def set_call_chars(chars: CallChars | None) -> None:
    _local.chars = chars


def current_call_chars() -> CallChars | None:
    return getattr(_local, "chars", None)


def log_fields(prompt_chars: int | None) -> dict:
    """Fields the usage record carries for one call. `prompt_chars` is the length of the prompt
    handed to the client (always known there); the rest comes from the Agent's measure and is
    null when the caller did not publish one. A published measure whose prompt length disagrees
    with the client's is dropped to nulls: it belongs to another prompt."""
    chars = current_call_chars()
    empty = {
        "prompt_chars": prompt_chars,
        "covered_chars": None,
        "context_chars": None,
        "history_chars": None,
        "observation_chars": None,
        "budget_chars": None,
    }
    if chars is None or (prompt_chars is not None and chars.prompt_chars != prompt_chars):
        return empty
    return chars.as_log_fields()
