"""Environment implementation backed by AIOpsLab (harness.agent.Environment protocol).

AIOpsLab is not pip-installed; it is cloned (see README) into a location this process locates
via the AIOPSLAB_HOME env var. The clone is a third-party repo and is not committed here, so
importing it is deferred to first use, not module import time, and fails with a clear message rather than an ImportError stack trace if AIOPSLAB_HOME is
unset or wrong.

Scope: mitigation-family AIOpsLab problems only (action space is exec_shell + submit, which maps
cleanly onto AGENT_ACTION_SCHEMA's action_type: inspect/remediate -> exec_shell(command),
escalate/submit -> submit()). Detection/localization/analysis problems have a different action
surface (bool / list / dict submissions) and are out of scope for this adapter.
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path
from unittest import mock

# Ceiling on what ONE environment observation can carry into the prompt. Measured live
# (redeploy_without_PV-mitigation-1/config_a): an agent command dumped the apiserver's /metrics
# (~4MB), the observation went into the prompt whole and the tokenizer counted 1,174,469 tokens
# against the model's 131,072-token context. MLX tried to allocate the KV cache for that and the
# host went to 13GB of swap and load 181 before the episode was killed by hand. The prompt
# carries up to HISTORY_WINDOW earlier observations plus the latest, so the per-observation
# ceiling is what bounds the whole prompt. Head + tail because the start says what the command
# returned and the end usually carries the error; the middle of a big dump informs the least.
OBSERVATION_MAX_CHARS = 12_000
_OBSERVATION_HEAD_CHARS = 8_000


def cap_observation(text: str, limit: int = OBSERVATION_MAX_CHARS) -> str:
    """Truncate a large observation keeping its start and end, with a marker of what was cut."""
    if len(text) <= limit:
        return text
    head = _OBSERVATION_HEAD_CHARS if limit > _OBSERVATION_HEAD_CHARS else limit // 2
    tail = limit - head
    dropped = len(text) - head - tail
    return text[:head] + _omission_marker(dropped) + text[-tail:]


def _omission_marker(dropped: int) -> str:
    return (
        f"\n[... {dropped} chars omitted: output too large for the prompt, "
        "narrow the command (grep, head, -o jsonpath) ...]\n"
    )


def cap_within_share(text: str, limit: int) -> str:
    """cap_observation with the marker INSIDE the limit: the result never exceeds `limit`. Exists
    for PromptBudget: measured, a 14,489-char prompt overflowed its budget because every cut added
    the marker on top of the share. cap_observation is unchanged, so paths without a budget keep
    the prompt byte for byte."""
    if len(text) <= limit:
        return text
    # The real marker is never longer than this one: dropped < len(text).
    room = limit - len(_omission_marker(len(text)))
    return cap_observation(text, room) if room > 0 else text[:limit]


def _ensure_aiopslab_importable() -> None:
    if "aiopslab" in sys.modules:
        return
    home = os.environ.get("AIOPSLAB_HOME")
    if not home:
        raise RuntimeError(
            "AIOPSLAB_HOME is not set. Point it at the AIOpsLab clone directory "
            "(see tools/triage-eval/README.md for the clone command); "
            "the clone is a third-party repo and is intentionally not committed here."
        )
    path = Path(home)
    if not (path / "aiopslab").is_dir():
        raise RuntimeError(f"AIOPSLAB_HOME={home!r} does not contain an 'aiopslab' package.")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    _pin_observer_kubeconfig(path)


# aiopslab/observer/__init__.py runs load_kube_config(config_file="~/.kube/config") at IMPORT
# (kubernetes_path fixed in monitor_config.yaml), and an explicit config_file ignores KUBECONFIG.
# With one cluster per seed, each seed has its own file (seed_cluster.sh) and ~/.kube/config does
# not even exist: a smoke run died on all 3 seeds with "Invalid kube-config file. No
# configuration found." before deploying the app. With KUBECONFIG set, the observer import reads
# the seed's file instead.
_UPSTREAM_OBSERVER_KUBECONFIG = "~/.kube/config"


def _pin_observer_kubeconfig(home: Path) -> None:
    kubeconfig = os.environ.get("KUBECONFIG")
    if not kubeconfig or "aiopslab.observer" in sys.modules:
        return
    if not (home / "aiopslab" / "observer" / "__init__.py").is_file():
        return
    from kubernetes.config import kube_config

    original = kube_config.load_kube_config

    def load_seed_kubeconfig(config_file=None, *args, **kwargs):
        if config_file is None or os.path.expanduser(config_file) == os.path.expanduser(
            _UPSTREAM_OBSERVER_KUBECONFIG
        ):
            config_file = kubeconfig
        return original(config_file, *args, **kwargs)

    kube_config.load_kube_config = load_seed_kubeconfig
    try:
        import aiopslab.observer  # noqa: F401
    finally:
        kube_config.load_kube_config = original


# Infra wait ceiling for a local kind cluster. The upstream default (max_wait=300) is too tight
# there: during probe admission the hotelReservation deploy took >5min repeatedly (slow
# openebs-hostpath PVC + mongodb after a Docker restart), and the DEPLOY's wait_for_ready timed
# out before the app finished coming up. Same rationale as the leader-election lease patch
# (setup_cluster.sh): tolerance for this infrastructure, never benchmark semantics.
INFRA_WAIT_MAX_SEC = 900


def _widen_infra_wait_defaults(max_wait: int = INFRA_WAIT_MAX_SEC) -> None:
    """Widen the DEFAULT max_wait of KubeCtl.wait_for_ready/wait_for_namespace_deletion.

    Only the default: a caller passing an explicit max_wait still decides, and the only eval
    that uses wait_for_ready (misconfig_app) passes max_wait=60 explicitly, so the benchmark's
    evaluation semantics do not change. What changes is the deploy/teardown path (hotelres.py,
    helm.py, the openebs orchestrator), which uses the default. Idempotent.
    """
    from aiopslab.service.kubectl import KubeCtl

    for name in ("wait_for_ready", "wait_for_namespace_deletion"):
        method = getattr(KubeCtl, name, None)
        if method is None or getattr(method, "_sslm_widened", False):
            continue

        def widened(self, namespace, sleep=2, max_wait=max_wait, _orig=method):
            return _orig(self, namespace, sleep=sleep, max_wait=max_wait)

        widened._sslm_widened = True
        setattr(KubeCtl, name, widened)


class AiopsLabEnvironment:
    """Drives one AIOpsLab mitigation-family problem through the harness's turn loop.

    Deploys the problem (and injects its fault) lazily on the first `task_description()` call,
    not in `__init__`, so constructing this object never triggers cluster/Helm work on its own,
    matching how `harness.agent.Agent.run_episode` only touches the environment once it actually
    starts an episode.
    """

    def __init__(
        self, problem_id: str, results_dir: str | None = None, control_mode: bool = False
    ) -> None:
        _ensure_aiopslab_importable()
        _widen_infra_wait_defaults()
        from aiopslab.orchestrator import Orchestrator

        self.problem_id = problem_id
        # Control round: calibrates how often the mitigation eval passes on its own, with no
        # fault injected. The 4 success=True of the real matrix all fall on
        # misconfig_app_hotel_res and redeploy_without_PV, and the eval of both (read in the
        # AIOpsLab code) only checks pod state (wait_for_ready / no CrashLoopBackOff), never
        # whether the root cause was mitigated. control_mode=True measures that: with the
        # cluster never touched, does the same eval still report success=True?
        self.control_mode = control_mode
        self._orchestrator = Orchestrator(results_dir=results_dir)
        # AIOpsLab's init_problem records the agent name before any agent is
        # registered. The pre-flight harness owns the agent loop, so provide a
        # stable session label without registering an AIOpsLab-native agent.
        self._orchestrator.agent_name = "triage-eval"
        self._task_desc: str | None = None
        self._deployed = False
        self._closed = False
        self._fault_patchers: list = []

    def task_description(self) -> str:
        if not self._deployed:
            if self.control_mode:
                # AIOpsLab's Orchestrator.init_problem calls prob.inject_fault()
                # unconditionally, with no "skip the fault" flag of its own (it is a
                # third-party clone, not forked here). Each problem class defines its own
                # inject_fault/recover_fault (see e.g. MisconfigAppHotelResBaseTask), so the
                # only seam available without forking AIOpsLab is patching those two methods
                # to no-ops on the specific problem class for the life of this episode: the
                # app still gets deployed healthy, it just never receives the fault.
                prob_cls = type(self._orchestrator.probs.get_problem_instance(self.problem_id))
                self._fault_patchers = [
                    mock.patch.object(prob_cls, "inject_fault", lambda self: None),
                    mock.patch.object(prob_cls, "recover_fault", lambda self: None),
                ]
                for patcher in self._fault_patchers:
                    patcher.start()
            # init_problem's own `instructions` and `actions` describe AIOpsLab's NATIVE
            # response protocol (a fenced ```<API_NAME>(<PARAMS>)``` call against its full
            # TaskActions registry: get_logs, get_traces, get_metrics, exec_shell, submit...).
            # This adapter's harness.agent.SYSTEM_PROMPT asks for a different protocol (JSON
            # matching AGENT_ACTION_SCHEMA), and _format_action only bridges two of those
            # native APIs (exec_shell, submit), per this module's own documented scope.
            # Concatenating AIOpsLab's instructions/actions verbatim put both protocols in the
            # same prompt and the model followed the native one (e.g. action_type="get_traces",
            # which our schema's enum rejects): an early run's transcript showed 10/10
            # turns malformed, 9 of them the identical get_traces response. `task_desc` (app
            # description) is real context and is kept; the conflicting half is replaced with a
            # scope note matching what this adapter can actually execute.
            task_desc, _instructions, _actions = self._orchestrator.init_problem(self.problem_id)
            self._orchestrator.session.start()
            self._task_desc = (
                f"{task_desc}\n\n"
                "Available actions (respond using the JSON schema from the system prompt, "
                "not AIOpsLab's native API-call syntax):\n"
                "- action_type=inspect or remediate: `command` runs as a shell command "
                "(kubectl, etc.) inside the debugging environment.\n"
                "- action_type=escalate or submit: submits your current solution for "
                "evaluation; no `command` needed."
            )
            self._deployed = True
        assert self._task_desc is not None
        return self._task_desc

    def apply_action(self, action: dict) -> tuple[bool, str]:
        from aiopslab.utils.status import SubmissionStatus

        if not self._deployed:
            raise RuntimeError("apply_action called before task_description deployed the problem")

        response_text = _format_action(action)
        env_response = asyncio.run(self._orchestrator.ask_env(response_text))
        solved = env_response == SubmissionStatus.VALID_SUBMISSION
        observation = (
            "Submission accepted."
            if solved
            else (
                "Submission rejected, keep investigating."
                if env_response == SubmissionStatus.INVALID_SUBMISSION
                else cap_observation(str(env_response))
            )
        )
        return solved, observation

    def finalize(self, solution, history, duration: float) -> dict:
        """Score the episode and tear down the deployed problem.

        Mirrors the eval + fault-recovery + app-cleanup steps at the end of
        `Orchestrator.start_problem`, split out here because this harness's Agent (unlike
        AIOpsLab's own runner) does not own the environment lifecycle, the caller does. Must be
        called exactly once after the episode ends (solved or max_turns reached), win or lose,
        so the fault does not outlive the episode on a shared cluster.
        """
        if self._closed:
            raise RuntimeError("finalize() already called for this environment instance")
        self._closed = True

        import atexit

        # init_problem registered exit_cleanup_fault with atexit; recovering the fault here
        # without unregistering makes the interpreter try again at exit, with the namespace
        # already deleted (ApiException 404 observed in a real run). Same order as
        # Orchestrator.start_problem: recover_fault -> unregister -> cleanup.
        from aiopslab.orchestrator.orchestrator import exit_cleanup_fault

        session = self._orchestrator.session
        results: dict = {}
        try:
            try:
                results = dict(session.problem.eval(solution, history, duration) or {})
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                # Observed on redeploy_without_PV/config_b: AIOpsLab's OWN eval raised a
                # TypeError (`for ... in pod.status.container_statuses`, which is None on a
                # Pending pod, the exact symptom of the unmitigated fault). Without this the
                # whole episode vanished (no JSON) and the scenario had no record in any of the
                # 3 configs. The verdict stays not measured (missing success = None in the
                # RunRecord, counted as unmeasured), the error goes into the record, and
                # teardown runs anyway.
                results = {"eval_error": f"{type(exc).__name__}: {exc}"}
                traceback.print_exc(file=sys.stderr)
                print(f"[finalize] eval raised, verdict unmeasured: {results['eval_error']}", file=sys.stderr)
            # Second verdict column, the fault's symptom (adapters/mitigation_probes.py). It
            # has to run HERE, before the finally's recover_fault(): after that the benchmark
            # itself has repaired the cluster and every probe would return True. Read-only,
            # never raises (an exception becomes probe_error).
            results.update(self._run_root_cause_probe(session.problem))
        finally:
            # A teardown that fails AFTER the eval (observed: namespace not deleted within
            # 300s, twice) must not take the verdict with it: the verdict is the episode's data,
            # and the next init_problem already cleans a leftover namespace. Recorded in the
            # result itself (teardown_error) and on stderr, never swallowed; it only does not
            # raise once the eval has answered.
            try:
                session.problem.recover_fault()
                atexit.unregister(exit_cleanup_fault)
                session.problem.app.cleanup()
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                if "success" not in results:
                    raise
                results["teardown_error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc(file=sys.stderr)
                print(
                    f"[finalize] teardown failed after eval (success={results['success']}): "
                    f"{results['teardown_error']}",
                    file=sys.stderr,
                )
            finally:
                # The inject_fault/recover_fault patches (control_mode) only hold until here:
                # the problem class is shared across episodes, so leaving them active after
                # teardown would leak into the next real episode of the matrix.
                for patcher in self._fault_patchers:
                    patcher.stop()
                self._fault_patchers = []
        session.end()
        return results

    def _run_root_cause_probe(self, problem) -> dict:
        from adapters.mitigation_probes import run_probe

        kubectl = getattr(problem, "kubectl", None)
        namespace = getattr(problem, "namespace", None)
        if kubectl is None or namespace is None:
            return {
                "root_cause_fixed": None,
                "probe_detail": "problem has no kubectl/namespace, probe does not apply",
            }
        return run_probe(self.problem_id, kubectl, namespace, getattr(problem, "faulty_service", None))

    def finalize_episode(self) -> dict:
        """Evaluate and tear down using the state captured by AIOpsLab's session."""
        if not self._deployed:
            # init_problem failed mid-deploy (observed: etcd timeout with pods never ready):
            # there is no started session to evaluate, and evaluating here raises a TypeError
            # in get_duration (start_time=None), masking the real exception. An injected fault
            # (if it got that far) is recovered by AIOpsLab's own atexit, and the leftover
            # namespace is cleaned by the next init_problem's app.delete().
            self._closed = True
            return {}
        session = self._orchestrator.session
        # AIOpsLab's native Orchestrator ends the session before asking it for
        # the duration. This adapter owns that lifecycle, so mirror the same
        # ordering instead of calling get_duration() with end_time still unset.
        session.end()
        return self.finalize(session.solution, session.history, session.get_duration())


def _format_action(action: dict) -> str:
    """Render a harness action dict as the fenced function-call text AIOpsLab's parser expects.

    See aiopslab/orchestrator/parser.py: a single ``` fenced block containing exactly one
    `api_name(args)` call.
    """
    action_type = action["action_type"]

    if action_type in ("submit", "escalate"):
        return "```\nsubmit()\n```"

    command = action.get("command")
    if not command:
        raise ValueError(
            f"action_type={action_type!r} requires a non-empty 'command' "
            "(the pre-flight harness only wires exec_shell for inspect/remediate)"
        )
    escaped = command.replace("\\", "\\\\").replace('"', '\\"')
    return f'```\nexec_shell("{escaped}")\n```'
