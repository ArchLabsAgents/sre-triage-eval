"""Training-free episodic memory for config C.

Store past (task, action, outcome) episodes
and retrieve the most relevant ones into the next prompt, without touching any model weights.
Retrieval here is deliberately naive (token-overlap scoring over the task description) so it
has zero external dependencies and is easy to audit; swapping in a real embedding index later
is a config change, not a rewrite, since EpisodicMemory's public surface (`add`, `retrieve`)
does not change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


# How much of the environment's observation goes into memory. The retrieve prompt carries top_k
# of these, and a whole `kubectl describe` per entry would drown the task in its own context.
OUTCOME_MAX_CHARS = 400

VERDICT_UNKNOWN = "unknown"


@dataclass
class Episode:
    task_id: str
    task_description: str
    action: dict
    # What the ENVIRONMENT answered to this action (command output, error, "Submission
    # accepted"), truncated. This used to be "success"/"failure" derived from `solved`, which for
    # AIOpsLab only meant "called submit()": memory handed back a success label for actions that
    # never mitigated anything.
    outcome: str
    # The benchmark eval's verdict for the whole episode, written by set_verdict() AFTER
    # finalize; "unknown" until then. It is the only success/failure label memory carries.
    verdict: str = VERDICT_UNKNOWN

    def to_json(self) -> str:
        return json.dumps(
            {
                "task_id": self.task_id,
                "task_description": self.task_description,
                "action": self.action,
                "outcome": self.outcome,
                "verdict": self.verdict,
            }
        )

    @staticmethod
    def from_json(line: str) -> "Episode":
        data = json.loads(line)
        return Episode(**data)


def truncate_outcome(observation: str, limit: int = OUTCOME_MAX_CHARS) -> str:
    text = observation.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + " [...]"


@dataclass
class EpisodicMemory:
    store_path: Path
    episodes: list[Episode] = field(default_factory=list)

    @classmethod
    def load(cls, store_path: str | Path) -> "EpisodicMemory":
        path = Path(store_path)
        episodes: list[Episode] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                episodes = [Episode.from_json(line) for line in handle if line.strip()]
        return cls(store_path=path, episodes=episodes)

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("a", encoding="utf-8") as handle:
            handle.write(episode.to_json() + "\n")

    def set_verdict(self, task_id: str, verdict: str) -> int:
        """Stamp the benchmark's verdict on every stored turn of task_id; returns how many.

        Called once per episode after finalize, because the verdict only exists then. The
        store is rewritten in place (it is small: one line per turn) so the label survives
        the next `load`. Only turns still marked unknown are touched: a re-run of the same
        task_id after a crash must not relabel the earlier attempt's turns.
        """
        touched = 0
        for episode in self.episodes:
            if episode.task_id == task_id and episode.verdict == VERDICT_UNKNOWN:
                episode.verdict = verdict
                touched += 1
        if touched:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with self.store_path.open("w", encoding="utf-8") as handle:
                for episode in self.episodes:
                    handle.write(episode.to_json() + "\n")
        return touched

    def retrieve(self, task_description: str, top_k: int = 3) -> list[Episode]:
        """Return up to top_k past episodes ranked by word overlap with task_description.

        Ties broken by recency (later episodes rank higher), so a fresh failure on a similar
        task outranks a stale one when overlap is equal.
        """
        query_tokens = _tokenize(task_description)
        if not query_tokens:
            return []

        scored: list[tuple[float, int, Episode]] = []
        for index, episode in enumerate(self.episodes):
            overlap = query_tokens & _tokenize(episode.task_description)
            score = len(overlap) / len(query_tokens)
            if score > 0:
                scored.append((score, index, episode))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [episode for _, _, episode in scored[:top_k]]
