"""Deterministic C0-C3 command classifier for the 24/7 triage safety envelope.

Every action the model
emits is classified by this parser BEFORE execution, never by the model itself. A command this
parser cannot map to C0/C1/C2 is C3 -- fail-closed, never executes -- by construction: C3 is
the default, not a detected case.

Scope of this module: classifying ONE command, given the episode facts it needs (call counts,
resources already observed via C0 this episode, declared scale bounds, confidence). It does not
own episode lifecycle, webhook ingestion or escalation-code bookkeeping (E1-E7) -- those live
above this module, in the episode loop (harness/episode_runner.py). Denials from here
feed E2 (envelope/gate denial); loop and timeout detection (E4/E5) are the caller's job since
only the caller sees the full command history and clock.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from enum import Enum

# Per-episode caps (the envelope's summary table).
C0_CALL_CAP = 12
C1_CALL_CAP = 4
C2_CALL_CAP = 2
CONFIDENCE_GATE = 0.7

# Pods of these kinds are never touched by C1 exec or C2 mutation, in any case (C1 rule and
# the note under C3's mongo examples).
_DB_POD_MARKERS = ("mongo", "memcached")

# C1's exec allowlist: subcommand must be exactly one of these binaries, no pipe, no redirect,
# no shell. Anything else (sh, bash, sh -c, a pipe/redirect character) reclassifies C3.
_C1_EXEC_ALLOWED_BINARIES = {"cat", "ls", "wget", "env"}
_SHELL_METACHARACTERS = ("|", ">", "<", ";", "&&", "||", "`", "$(")

# C2's allowlist v1: exactly these three kubectl verb shapes, nothing else mutates.
_C2_ROLLOUT_SUBCOMMANDS = {"restart", "undo"}

# The -f/--filename flag is forbidden entirely in v1 (kills the file-hallucination failure mode
# F1 by construction, never by detection): any command carrying it is never C0-C2.
_FORBIDDEN_FLAGS = ("-f", "--filename")

# wget, unlike cat/ls/env, can write files (-O <path>) or exfiltrate data (--post-data,
# --post-file) on its own, with no shell needed -- being on the C1 exec-binary allowlist is not
# enough to make it read-only. Restrict it to exactly the health-probe shape: GET, dumped to
# stdout, one URL, nothing else. Only the combined short forms are accepted (-O-, -qO-); the
# space-separated equivalent (-O -) is deliberately NOT recognized, so a bare "-O" token is
# always rejected rather than requiring this parser to track flag/value pairing.
#
# One of _WGET_STDOUT_FLAGS MUST be present: wget's own default with no -O at all is to write
# the response to a local file (named from the URL), not stdout, so "no flags" is not a safe
# default here the way it would be for cat/ls -- a bare "wget <url>" with nothing else looked
# like the safe case in an earlier version of this check, but it is not.
_WGET_STDOUT_FLAGS = {"-O-", "-qO-"}
_WGET_OTHER_SAFE_FLAGS = {"-q", "--quiet"}
_WGET_SAFE_FLAGS = _WGET_STDOUT_FLAGS | _WGET_OTHER_SAFE_FLAGS


class ActionClass(str, Enum):
    C0 = "C0"  # read/observe: free
    C1 = "C1"  # active diagnosis: read-only exec, contained
    C2 = "C2"  # reversible mutation: gated, rollback pre-computed
    C3 = "C3"  # irreversible or out of scope: never executes, always a proposal


@dataclass
class EpisodeEnvelopeState:
    """Per-episode facts the gates need. One instance per triage episode, mutated by the caller
    as calls execute -- classify() itself never mutates state, so a denied call never counts
    against a cap or a "target seen" fact it did not actually establish."""

    c0_calls: int = 0
    c1_calls: int = 0
    c2_calls: int = 0
    confidence: float = 0.0
    # Resources named in a C0 get/describe OUTPUT this episode ("the referenced resource appeared
    # in C0 output of this same episode"), e.g. "deployment/recommendation".
    # Populated by the caller after reading C0 output, never by classify().
    targets_seen: set[str] = field(default_factory=set)
    # deployment target -> (min, max) declared replica bounds for the scale gate.
    scale_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvelopeDecision:
    action_class: ActionClass
    allowed: bool
    reason: str


def _is_db_pod(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _DB_POD_MARKERS)


def _mentions_db_pod(tokens: list[str]) -> str | None:
    """Return the first token that looks like a database pod reference, scanning every token
    rather than trusting a fixed position.

    Real bypass caught in review: kubectl accepts flags before the target name too (e.g.
    "kubectl exec -n hotel mongo-0 -- ..." or "kubectl delete pod -n hotel mongo-0"), so
    tokens[2]/tokens[3]-style fixed-position lookups miss the pod name whenever a flag precedes
    it. Scanning every token trades a theoretical false positive (some other token happening to
    contain a marker substring) for eliminating the false negative entirely -- the safe
    direction here, since this module's default is fail-closed (C3) already.
    """
    for tok in tokens:
        if _is_db_pod(tok):
            return tok
    return None


def _has_forbidden_flag(tokens: list[str]) -> bool:
    # kubectl accepts -f in three shapes: a standalone token ("-f", "x.yaml"), the concatenated
    # short form ("-fx.yaml"), and the long form ("--filename" / "--filename=x.yaml"). Missing
    # the concatenated short form was a real bypass of the F1 (path-hallucination) control this
    # flag exists to kill by construction: a model output using -fmalicious.yaml sailed straight
    # through the standalone-token check.
    for tok in tokens:
        if tok in _FORBIDDEN_FLAGS:
            return True
        if tok.startswith("--filename"):
            return True
        if tok.startswith("-f") and tok != "-f":
            return True
    return False


def _wget_args_are_safe(args: list[str]) -> tuple[bool, str]:
    """`args` is everything after the `wget` token itself. Enforce the health-probe shape only:
    any flag not in `_WGET_SAFE_FLAGS` is rejected (this is what blocks a bare "-O", "--post-data"
    or "--post-file"), exactly one non-flag argument (the URL) is required, and one of
    `_WGET_STDOUT_FLAGS` must be present -- without it wget writes a local file by default, so a
    flag-free "wget <url>" is not the safe case it might look like."""
    urls = []
    saw_stdout_flag = False
    for arg in args:
        if arg in _WGET_STDOUT_FLAGS:
            saw_stdout_flag = True
            continue
        if arg in _WGET_OTHER_SAFE_FLAGS:
            continue
        if arg.startswith("-"):
            return False, f"wget flag {arg!r} is not in the read-only allowlist {sorted(_WGET_SAFE_FLAGS)}"
        urls.append(arg)
    if not saw_stdout_flag:
        return False, "wget must explicitly dump to stdout (-O- or -qO-): without it, wget writes a local file by default"
    if len(urls) != 1:
        return False, f"wget must take exactly one URL argument, got {urls}"
    return True, "wget restricted to a single GET dumped to stdout"


def _deny(action_class: ActionClass, reason: str) -> EnvelopeDecision:
    return EnvelopeDecision(action_class=action_class, allowed=False, reason=reason)


def _allow(action_class: ActionClass, reason: str) -> EnvelopeDecision:
    return EnvelopeDecision(action_class=action_class, allowed=True, reason=reason)


def classify(
    command: str,
    state: EpisodeEnvelopeState,
    *,
    target: str | None = None,
    scale_replicas: int | None = None,
) -> EnvelopeDecision:
    """Classify one command and decide whether it may execute now.

    `target` is the resource the command names (e.g. "deployment/recommendation"), required for
    C2 gating (read-before-mutate / target-exists collapse into the same check here: the target
    must already be in `state.targets_seen`, which only a C0 get/describe output this episode
    can have populated). `scale_replicas` is the requested replica count for a `scale` command.

    A command this function cannot parse, or that carries a forbidden flag anywhere, is C3:
    fail-closed is the default outcome, not a special case.
    """
    if not command or not command.strip():
        return _deny(ActionClass.C3, "empty command")

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return _deny(ActionClass.C3, f"unparseable command ({exc})")

    if not tokens:
        return _deny(ActionClass.C3, "empty command")

    if _has_forbidden_flag(tokens):
        return _deny(ActionClass.C3, "-f/--filename is forbidden entirely in v1 (F1: path hallucination)")

    if any(meta in command for meta in _SHELL_METACHARACTERS):
        return _deny(ActionClass.C3, "shell metacharacter present: only a bare allowlisted binary is C1, never a shell")

    tool = tokens[0]

    if tool == "wget":
        safe, reason = _wget_args_are_safe(tokens[1:])
        if not safe:
            return _deny(ActionClass.C3, reason)
        return _classify_c1(tokens, state, exec_binary="wget")

    if tool != "kubectl":
        return _deny(ActionClass.C3, f"unrecognized tool {tool!r}: fail-closed")

    if len(tokens) < 2:
        return _deny(ActionClass.C3, "kubectl invocation with no verb: fail-closed")

    verb = tokens[1]

    if verb in ("get", "list", "watch", "describe", "logs"):
        return _classify_c0(state)

    if verb == "top":
        return _classify_c1(tokens, state, exec_binary=None)

    if verb == "exec":
        return _classify_exec(tokens, state)

    if verb == "rollout":
        return _classify_rollout(tokens, state, target)

    if verb == "delete":
        return _classify_delete(tokens, state, target)

    if verb == "scale":
        return _classify_scale(tokens, state, target, scale_replicas)

    # apply/patch/edit/replace, delete of anything other than a pod, rbac/secret/configmap
    # mutation, and every other verb: not in any allowlist, so C3 by the same fail-closed
    # default ("no gate releases C3 in v1").
    return _deny(ActionClass.C3, f"kubectl {verb} is not in the C0-C2 allowlist: fail-closed")


def _classify_c0(state: EpisodeEnvelopeState) -> EnvelopeDecision:
    if state.c0_calls >= C0_CALL_CAP:
        return _deny(ActionClass.C0, f"C0 call cap reached ({C0_CALL_CAP}/episode): harness forces a decision")
    return _allow(ActionClass.C0, "read/observe, free")


def _classify_c1(tokens: list[str], state: EpisodeEnvelopeState, *, exec_binary: str | None) -> EnvelopeDecision:
    # Checked first, ahead of the C0/cap gates: a disallowed binary is a C3 (fail-closed) fact
    # about the command itself, true regardless of episode state. Checking it after the C0/cap
    # gates would label a disallowed-binary command as a C1 denial (feeding E2, "envelope/gate
    # denial") whenever c0_calls happened to be 0 or the cap was already hit, instead of the C3
    # fail-closed path it actually is.
    if exec_binary is not None and exec_binary not in _C1_EXEC_ALLOWED_BINARIES:
        return _deny(ActionClass.C3, f"{exec_binary!r} is not in the C1 exec binary allowlist {sorted(_C1_EXEC_ALLOWED_BINARIES)}")
    if state.c0_calls < 1:
        return _deny(ActionClass.C1, "C1 requires at least one C0 call first in this episode")
    if state.c1_calls >= C1_CALL_CAP:
        return _deny(ActionClass.C1, f"C1 call cap reached ({C1_CALL_CAP}/episode)")
    return _allow(ActionClass.C1, "active diagnosis, contained")


def _classify_exec(tokens: list[str], state: EpisodeEnvelopeState) -> EnvelopeDecision:
    # kubectl exec <pod> [-n ns] -- <binary> [args...] (or with -n/other flags before the pod).
    if "--" not in tokens:
        return _deny(ActionClass.C3, "kubectl exec without -- <command> separator: fail-closed")
    sep = tokens.index("--")
    db_pod = _mentions_db_pod(tokens[:sep])
    if db_pod is not None:
        return _deny(ActionClass.C3, f"exec into a database pod ({db_pod!r}) is excluded in any case")
    remote_command = tokens[sep + 1 :]
    if not remote_command:
        return _deny(ActionClass.C3, "kubectl exec -- with no remote command: fail-closed")
    binary = remote_command[0]
    if binary in ("sh", "bash", "ash", "dash"):
        return _deny(ActionClass.C3, f"exec shell {binary!r} reclassifies C3: only a bare allowlisted binary is C1")
    if binary == "wget":
        safe, reason = _wget_args_are_safe(remote_command[1:])
        if not safe:
            return _deny(ActionClass.C3, reason)
    return _classify_c1(tokens, state, exec_binary=binary)


def _classify_c2_common(
    tokens: list[str], state: EpisodeEnvelopeState, target: str | None
) -> EnvelopeDecision | None:
    """Shared C2 gates ("all conditions, in order"). Returns a denial
    EnvelopeDecision if any gate fails, or None if every shared gate passes.

    The db-resource exclusion is checked here, once, for every C2 verb (rollout/delete/scale)
    rather than per-verb: a per-verb check already missed this once for delete (see that
    function's history) and a second review round found it missing again for rollout and scale.
    Centralizing it is what stops the same gap from recurring for a future C2 verb."""
    db_resource = _mentions_db_pod(tokens)
    if db_resource is not None:
        return _deny(ActionClass.C3, f"a database resource ({db_resource!r}) is excluded from C2 mutation in any case")
    if state.confidence < CONFIDENCE_GATE:
        return _deny(ActionClass.C2, f"confidence {state.confidence} < {CONFIDENCE_GATE}")
    if not target:
        return _deny(ActionClass.C2, "no target named for a C2 mutation")
    if target not in state.targets_seen:
        return _deny(
            ActionClass.C2,
            f"read-before-mutate: {target!r} was not observed via a C0 get/describe this episode",
        )
    if state.c2_calls >= C2_CALL_CAP:
        return _deny(ActionClass.C2, f"C2 call cap reached ({C2_CALL_CAP}/episode): 3rd need is signal E3")
    return None


def _classify_rollout(tokens: list[str], state: EpisodeEnvelopeState, target: str | None) -> EnvelopeDecision:
    if len(tokens) < 3 or tokens[2] not in _C2_ROLLOUT_SUBCOMMANDS:
        return _deny(ActionClass.C3, "kubectl rollout <subcommand> not in {restart, undo}: fail-closed")
    # The db-resource exclusion is checked before every other rollout-specific rule (including
    # the --to-revision requirement below), never after: this exact shape (a db target failing
    # the --to-revision check) was the third occurrence of the same C2-instead-of-C3 mislabel
    # class this module has had, each time because a verb-specific denial ran before the shared
    # _classify_c2_common gate that carries the check.
    db_resource = _mentions_db_pod(tokens)
    if db_resource is not None:
        return _deny(ActionClass.C3, f"a database resource ({db_resource!r}) is excluded from C2 mutation in any case")
    if tokens[2] == "undo" and not any(tok.startswith("--to-revision") for tok in tokens):
        return _deny(ActionClass.C2, "rollout undo must name a known previous revision (--to-revision)")
    denial = _classify_c2_common(tokens, state, target)
    if denial is not None:
        return denial
    return _allow(ActionClass.C2, f"rollout {tokens[2]}, reversible")


def _classify_delete(tokens: list[str], state: EpisodeEnvelopeState, target: str | None) -> EnvelopeDecision:
    # Only "delete pod" is C2 (forces reschedule, reversible via the owning controller).
    # Anything else (pvc, deployment, service, namespace, ...) is C3.
    if len(tokens) < 3 or tokens[2] != "pod":
        return _deny(ActionClass.C3, "kubectl delete of anything other than a pod is irreversible here: fail-closed")
    denial = _classify_c2_common(tokens, state, target)
    if denial is not None:
        return denial
    return _allow(ActionClass.C2, "delete pod forces reschedule, reversible")


def _classify_scale(
    tokens: list[str],
    state: EpisodeEnvelopeState,
    target: str | None,
    scale_replicas: int | None,
) -> EnvelopeDecision:
    denial = _classify_c2_common(tokens, state, target)
    if denial is not None:
        return denial
    if scale_replicas is None:
        return _deny(ActionClass.C3, "scale with no explicit replica count: fail-closed")
    bounds = state.scale_bounds.get(target) if target else None
    if bounds is None:
        return _deny(ActionClass.C2, f"no declared [min,max] scale bounds for {target!r}")
    lo, hi = bounds
    if not (lo <= scale_replicas <= hi):
        return _deny(ActionClass.C2, f"scale to {scale_replicas} outside declared bounds [{lo},{hi}] for {target!r}")
    return _allow(ActionClass.C2, f"scale within declared bounds [{lo},{hi}]")
