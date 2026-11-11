"""Alertmanager webhook ingest: validates the POST payload and folds it into at most one
triage episode per alert group. Triage flow, step 1:

    ingest: dedup by alert fingerprint + group window (alarms of the same group within 120s
    become ONE triage episode)

Scope boundary, read before extending this file: this module only decides WHICH episode a
webhook belongs to. It does not run the episode (harness/agent.py's loop, envelope.py's C0-C3
gating, TriageRecord assembly, the E1-E7 escalation triggers) -- that orchestrator is
harness/episode_runner.py. Keeping ingest/dedup decoupled from episode execution means the HTTP layer
(entrypoint.py) and its tests never need a live model, cluster or Prometheus to exercise.

Alertmanager's own grouping (route tree, group_wait/group_interval) already batches alerts that
fire together into one webhook call with one `groupKey`. What this module adds on top: a
groupKey can legitimately receive MULTIPLE webhook calls over time (Alertmanager re-notifies as
the group's alert set changes, per repeat_interval), and the contract above says calls for the
same group within 120s of the group's first sighting are still one episode, not one each.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

GROUP_WINDOW_SECONDS = 120.0

# Only these alert-level statuses exist in an Alertmanager webhook payload (the enum the schema
# below validates against). A firing status forces a decision; a fully-resolved
# group ("status": "resolved" at the group level) is fold-only bookkeeping, it never opens a new
# episode -- see ingest().
_ALERT_STATUSES = ("firing", "resolved")
_GROUP_STATUSES = ("firing", "resolved")


class WebhookPayloadError(ValueError):
    """The POST body does not match Alertmanager's webhook shape closely enough to trust.
    Callers turn this into a 400, never into a best-effort partial episode: an episode built
    from a payload we half-understood is exactly the "deciding blind" failure mode the
    evidence honesty rule exists to reject."""


@dataclass(frozen=True)
class AlertmanagerWebhook:
    """The subset of Alertmanager's webhook payload this ingest step actually reads. Fields not
    listed here (externalURL, commonAnnotations, ...) are not this module's concern and are left
    in the raw payload the episode orchestrator receives unchanged."""

    group_key: str
    status: str
    receiver: str
    alert_fingerprints: tuple[str, ...]
    raw: dict


def parse_webhook_payload(payload: object) -> AlertmanagerWebhook:
    """Validate and narrow a decoded JSON body to an AlertmanagerWebhook.

    Deliberately hand-rolled rather than jsonschema here: the checks needed are a handful of
    "is this key present and the right shape", and the honest error message (which key, what it
    got instead) matters more than schema-generality for an HTTP 400 body an on-call human or a
    replaying test will read.
    """
    if not isinstance(payload, dict):
        raise WebhookPayloadError(f"webhook body must be a JSON object, got {type(payload).__name__}")

    group_key = payload.get("groupKey")
    if not isinstance(group_key, str) or not group_key:
        raise WebhookPayloadError("missing or empty 'groupKey'")

    status = payload.get("status")
    if status not in _GROUP_STATUSES:
        raise WebhookPayloadError(f"'status' must be one of {_GROUP_STATUSES}, got {status!r}")

    receiver = payload.get("receiver")
    if not isinstance(receiver, str) or not receiver:
        raise WebhookPayloadError("missing or empty 'receiver'")

    alerts = payload.get("alerts")
    if not isinstance(alerts, list) or not alerts:
        raise WebhookPayloadError("'alerts' must be a non-empty list")

    fingerprints: list[str] = []
    for index, alert in enumerate(alerts):
        if not isinstance(alert, dict):
            raise WebhookPayloadError(f"alerts[{index}] must be a JSON object")
        fingerprint = alert.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise WebhookPayloadError(f"alerts[{index}] missing or empty 'fingerprint'")
        alert_status = alert.get("status")
        if alert_status not in _ALERT_STATUSES:
            raise WebhookPayloadError(
                f"alerts[{index}]['status'] must be one of {_ALERT_STATUSES}, got {alert_status!r}"
            )
        fingerprints.append(fingerprint)

    return AlertmanagerWebhook(
        group_key=group_key,
        status=status,
        receiver=receiver,
        alert_fingerprints=tuple(fingerprints),
        raw=payload,
    )


@dataclass
class _OpenWindow:
    episode_id: str
    window_start: float
    fingerprints: set[str] = field(default_factory=set)


@dataclass
class DedupResult:
    episode_id: str
    is_new_episode: bool
    # Fingerprints folded into this call that this window had not already seen (for the
    # received-vs-episodes counters entrypoint.py exposes; not itself part of the contract).
    new_fingerprints: tuple[str, ...]


class EpisodeDeduper:
    """One open window per groupKey, in-process, in-memory.

    In-memory and single-process by design for v1: the agent is one Deployment replica (section
    1.2), and a window's only job is collapsing Alertmanager's re-notifications for the SAME
    fault within a 2-minute span into one episode -- it is not meant to survive a pod restart,
    and a restart losing an in-flight window just means the next webhook for that group opens a
    fresh episode, which is a correct (if not maximally deduped) outcome, never a wrong one.

    Thread-safe: entrypoint.py serves requests via ThreadingHTTPServer (one thread per
    connection), so ingest() is a real concurrent check-then-act, not just a sequential call
    site. Two webhooks for the same brand-new groupKey arriving close
    together (HA Alertmanager, or a client retry racing the original) could otherwise both read
    "no live window" and each open its own episode, breaking the one-episode-per-window
    guarantee this class exists to provide. A single coarse lock around the whole method is
    enough -- ingest() is dict lookups and a handful of comparisons, never I/O.
    """

    def __init__(self, *, window_seconds: float = GROUP_WINDOW_SECONDS, clock=time.monotonic):
        self._window_seconds = window_seconds
        self._clock = clock
        self._windows: dict[str, _OpenWindow] = {}
        self._lock = threading.Lock()

    def _prune_expired(self, now: float) -> None:
        """Drop every window whose group has aged out. Run once per ingest() call: without this, `_windows` grows for the life of the process, one
        entry per groupKey ever seen, which matters here specifically because the agent this
        deduper runs inside is designed to run 24/7, never
        restarted between episodes the way a batch job would be. Cheap by construction: alert
        rule cardinality (the source of distinct groupKeys) is bounded and small,
        so this is never a large dict to sweep."""
        expired = [key for key, window in self._windows.items() if (now - window.window_start) >= self._window_seconds]
        for key in expired:
            del self._windows[key]

    def ingest(self, webhook: AlertmanagerWebhook) -> DedupResult:
        """Fold `webhook` into the open window for its group, or start a new one.

        A group-level `status == "resolved"` never opens a new window: every episode
        starts from an alarm (firing), and a lone resolved notification (no prior open
        window) has nothing to triage. It still folds into an ALREADY-open window when one
        exists (Alertmanager legitimately sends a resolved notification for a group it already
        told us about), so the episode orchestrator can see the fault cleared mid-episode.
        """
        with self._lock:
            now = self._clock()
            self._prune_expired(now)
            open_window = self._windows.get(webhook.group_key)
            window_is_live = open_window is not None and (now - open_window.window_start) < self._window_seconds

            if not window_is_live:
                if webhook.status == "resolved":
                    return DedupResult(episode_id="", is_new_episode=False, new_fingerprints=())
                episode_id = f"ep-{uuid.uuid4().hex[:16]}"
                open_window = _OpenWindow(episode_id=episode_id, window_start=now)
                self._windows[webhook.group_key] = open_window
                new_fingerprints = tuple(webhook.alert_fingerprints)
                open_window.fingerprints.update(new_fingerprints)
                return DedupResult(episode_id=episode_id, is_new_episode=True, new_fingerprints=new_fingerprints)

            new_fingerprints = tuple(fp for fp in webhook.alert_fingerprints if fp not in open_window.fingerprints)
            open_window.fingerprints.update(new_fingerprints)
            return DedupResult(
                episode_id=open_window.episode_id, is_new_episode=False, new_fingerprints=new_fingerprints
            )
