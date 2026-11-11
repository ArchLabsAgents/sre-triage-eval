"""Revision-preserving injection for assign_to_non_existent_node_hotel_res and wrong_bin_usage.

Why: AIOpsLab injects both faults by deleting the
deployment and re-applying it with the bad spec (aiopslab/generators/fault/inject_virtual.py,
inject_assign_to_non_existent_node / inject_wrong_bin_usage). That restarts the revision history
at 1, so there is no previous revision and `kubectl rollout undo` (the only C2 recovery in the
envelope) cannot fix them: the fix becomes a spec patch (C3) and the family can only be escalated.
With that, only one of the eight admitted families (misconfig) exercised the "agir" (act) path.

Here the bad spec lands as a new revision of the same deployment instead:

1. the rollout strategy is switched to Recreate (strategy is outside the pod template, so this
   step creates no revision). Without it a RollingUpdate would keep the old, healthy pod serving
   while the new one sits Pending (assign), turning an outage into a stuck rollout and changing
   what the family measures; Recreate kills the old pod first, which is the symptom the
   delete+apply produced in the earlier detection and admission runs.
2. the pod template is patched (new revision N+1). Revision N is the good spec, recorded before
   the patch, so `rollout undo --to-revision=N` recovers.

recover() runs that exact undo when the fault is still present (it is the table's canonical
action, so every recovered episode exercises it live), then restores the original strategy. When
the fault is already gone (someone undid it) it only restores the strategy: undoing again would
roll back to the bad revision.

AIOpsLab's KubeCtl.exec_command returns stderr instead of raising, so every step is read back
and a mismatch raises RevisionFaultError: an episode whose fault did not land is not measured,
never scored as a miss.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Callable, Protocol

from adapters.mitigation_probes import HOSTNAME_LABEL, NON_EXISTENT_NODE

REVISION_ANNOTATION = "deployment.kubernetes.io/revision"
# The binary AIOpsLab swaps in (inject_wrong_bin_usage) and the one the service really runs.
WRONG_BINARY = "geo"
RIGHT_BINARY = "profile"


class RevisionFaultError(RuntimeError):
    """A revision-preserving inject/recover step did not take effect."""


class ExecKube(Protocol):
    def exec_command(self, command: str, input_data: Any = None) -> str: ...


def read_deployment(kubectl: ExecKube, namespace: str, name: str) -> dict:
    raw = kubectl.exec_command(f"kubectl get deployment {name} -n {namespace} -o json") or ""
    try:
        deployment = json.loads(raw)
    except ValueError:
        raise RevisionFaultError(f"deployment {name!r} not readable: {raw.strip()[:200]!r}") from None
    if not isinstance(deployment, dict) or deployment.get("kind") != "Deployment":
        raise RevisionFaultError(f"deployment {name!r} not readable: {raw.strip()[:200]!r}")
    return deployment


def _patch(kubectl: ExecKube, namespace: str, name: str, patch_type: str, body: Any) -> None:
    kubectl.exec_command(
        f"kubectl patch deployment {name} -n {namespace} --type={patch_type} -p {shlex.quote(json.dumps(body))}"
    )


def _revision(deployment: dict) -> str:
    revision = ((deployment.get("metadata") or {}).get("annotations") or {}).get(REVISION_ANNOTATION)
    if not revision or not str(revision).isdigit():
        raise RevisionFaultError(f"deployment has no numeric {REVISION_ANNOTATION} annotation: {revision!r}")
    return str(revision)


def _pod_spec(deployment: dict) -> dict:
    return ((deployment.get("spec") or {}).get("template") or {}).get("spec") or {}


# --- per-fault spec checks and template patches -------------------------------------------------


def assign_pinned(deployment: dict) -> bool:
    return (_pod_spec(deployment).get("nodeSelector") or {}).get(HOSTNAME_LABEL) == NON_EXISTENT_NODE


def assign_patch(deployment: dict) -> tuple[str, Any]:
    return "merge", {"spec": {"template": {"spec": {"nodeSelector": {HOSTNAME_LABEL: NON_EXISTENT_NODE}}}}}


def wrong_bin_swapped(deployment: dict) -> bool:
    return any(WRONG_BINARY in (c.get("command") or []) for c in _pod_spec(deployment).get("containers") or [])


def wrong_bin_patch(deployment: dict) -> tuple[str, Any]:
    """Same selection as upstream: every container whose command runs the `profile` binary."""
    ops = [
        {"op": "replace", "path": f"/spec/template/spec/containers/{i}/command", "value": [WRONG_BINARY]}
        for i, c in enumerate(_pod_spec(deployment).get("containers") or [])
        if RIGHT_BINARY in (c.get("command") or [])
    ]
    if not ops:
        raise RevisionFaultError(f"no container runs {RIGHT_BINARY!r}; nothing to swap")
    return "json", ops


class RevisionFault:
    """One injected fault on one deployment: remembers the good revision and the original strategy
    between inject() and recover(), the same problem instance the injector keeps active."""

    def __init__(
        self,
        kubectl: ExecKube,
        namespace: str,
        service: str,
        faulty: Callable[[dict], bool],
        template_patch: Callable[[dict], tuple[str, Any]],
    ) -> None:
        self.kubectl = kubectl
        self.namespace = namespace
        self.service = service
        self._faulty = faulty
        self._template_patch = template_patch
        self.good_revision: str | None = None
        self.original_strategy: dict | None = None

    def _read(self) -> dict:
        return read_deployment(self.kubectl, self.namespace, self.service)

    def inject(self) -> None:
        before = self._read()
        if self._faulty(before):
            raise RevisionFaultError(f"{self.service!r} already carries the fault before inject")
        self.good_revision = _revision(before)
        self.original_strategy = (before.get("spec") or {}).get("strategy") or {"type": "RollingUpdate"}

        _patch(self.kubectl, self.namespace, self.service, "merge", {"spec": {"strategy": {"type": "Recreate", "rollingUpdate": None}}})
        patch_type, body = self._template_patch(before)
        _patch(self.kubectl, self.namespace, self.service, patch_type, body)

        after = self._read()
        strategy = ((after.get("spec") or {}).get("strategy") or {}).get("type")
        if strategy != "Recreate" or not self._faulty(after):
            # A half-landed inject (Recreate on, template patch lost) must not leak Recreate into
            # the next family's episode: put the original strategy back before raising.
            if not self._faulty(after):
                self._restore_strategy()
            raise RevisionFaultError(
                f"inject on {self.service!r} did not land (strategy={strategy!r}, fault present={self._faulty(after)})"
            )

    def _restore_strategy(self) -> None:
        if self.original_strategy is None:
            return
        strategy = dict(self.original_strategy)
        if strategy.get("type") == "RollingUpdate":
            strategy.setdefault("rollingUpdate", {"maxSurge": "25%", "maxUnavailable": "25%"})
        _patch(self.kubectl, self.namespace, self.service, "merge", {"spec": {"strategy": strategy}})

    def undo_command(self) -> str:
        if self.good_revision is None:
            raise RevisionFaultError("recover() without a recorded good revision (inject() never ran here)")
        return (
            f"kubectl rollout undo deployment/{self.service} -n {self.namespace} "
            f"--to-revision={self.good_revision}"
        )

    def recover(self) -> dict:
        """Undo to the good revision if the fault is still there, then restore the strategy.
        Returns what it did, for the family record."""
        current = self._read()
        undone = self._faulty(current)
        if undone:
            self.kubectl.exec_command(self.undo_command())
        self._restore_strategy()

        after = self._read()
        restored = ((after.get("spec") or {}).get("strategy") or {}).get("type")
        expected = (self.original_strategy or {}).get("type", restored)
        if self._faulty(after) or restored != expected:
            raise RevisionFaultError(
                f"recover on {self.service!r} did not land (fault present={self._faulty(after)}, "
                f"strategy={restored!r}, expected {expected!r})"
            )
        return {"undo_ran": undone, "to_revision": self.good_revision, "strategy_restored": restored}


def assign_fault(kubectl: ExecKube, namespace: str, service: str) -> RevisionFault:
    return RevisionFault(kubectl, namespace, service, assign_pinned, assign_patch)


def wrong_bin_fault(kubectl: ExecKube, namespace: str, service: str) -> RevisionFault:
    return RevisionFault(kubectl, namespace, service, wrong_bin_swapped, wrong_bin_patch)
