"""Unit tests for adapters.aiopslab_env.

AIOpsLab is a third-party clone located via AIOPSLAB_HOME (not pip-installed, not committed
here, see README). These tests use the `fake_aiopslab_home` fixture (conftest.py) to build a
minimal fake `aiopslab` package on disk with the same shapes our adapter actually imports
(Orchestrator.init_problem/ask_env/session, SubmissionStatus) so what's under test is our
adapter's own logic (deploy-once caching, the async bridge, solved detection, the fenced-command
formatting, the finalize teardown/double-call guard), not AIOpsLab itself. A real end-to-end run
against a live AIOpsLab problem is a manual step (README), not part of the unit suite, since it
needs a real Kind cluster.
"""

from __future__ import annotations

import sys
import types

import pytest
from adapters.aiopslab_env import AiopsLabEnvironment, _format_action


def test_missing_aiopslab_home_raises_clear_error(monkeypatch):
    monkeypatch.delenv("AIOPSLAB_HOME", raising=False)
    with pytest.raises(RuntimeError, match="AIOPSLAB_HOME is not set"):
        AiopsLabEnvironment("some-problem-id")


def test_aiopslab_home_without_package_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("AIOPSLAB_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="does not contain an 'aiopslab' package"):
        AiopsLabEnvironment("some-problem-id")


def test_task_description_deploys_once_and_combines_fields(fake_aiopslab_home):
    env = AiopsLabEnvironment("misconfig_app-mitigation-1")

    assert env._orchestrator.agent_name == "triage-eval"

    first = env.task_description()
    second = env.task_description()

    assert first == second
    assert "task desc" in first
    # AIOpsLab's own instructions/actions (here "do the thing" / "exec_shell, submit") describe
    # its native response protocol, which conflicts with harness.agent.SYSTEM_PROMPT's JSON
    # schema; this adapter only bridges exec_shell/submit (see task_description's docstring
    # comment), so it must not surface AIOpsLab's raw instructions/actions text verbatim.
    assert "do the thing" not in first
    assert "exec_shell, submit" not in first
    assert "action_type=inspect or remediate" in first
    assert env._orchestrator.init_problem_calls == ["misconfig_app-mitigation-1"]
    assert env._orchestrator.session.started is True


def test_control_mode_patches_fault_methods_to_no_ops_for_the_episode(fake_aiopslab_home):
    env = AiopsLabEnvironment("p1", control_mode=True)
    env.task_description()

    problem = env._orchestrator.session.problem
    # While the episode is open, the problem CLASS (not just this instance) is patched: any
    # instance init_problem creates (including the real one, inside the real Orchestrator)
    # neither injects nor recovers a fault.
    problem.inject_fault()
    problem.recover_fault()
    assert problem.inject_fault_calls == 0
    assert problem.recover_fault_calls == 0

    env.finalize(solution={"x": 1}, history=[], duration=1.0)

    # finalize() calls recover_fault() as part of teardown; still patched, so it does not count.
    assert problem.recover_fault_calls == 0
    # After finalize the patches are undone: the class is back to normal so the next episode of
    # the real matrix does not inherit the no-op.
    problem.inject_fault()
    assert problem.inject_fault_calls == 1


def test_control_mode_off_leaves_fault_methods_untouched(fake_aiopslab_home):
    env = AiopsLabEnvironment("p1", control_mode=False)
    env.task_description()

    problem = env._orchestrator.session.problem
    problem.inject_fault()
    assert problem.inject_fault_calls == 1


def test_apply_action_before_deploy_raises(fake_aiopslab_home):
    env = AiopsLabEnvironment("p1")
    with pytest.raises(RuntimeError, match="before task_description deployed"):
        env.apply_action({"action_type": "submit"})


def test_apply_action_inspect_is_not_solved(fake_aiopslab_home):
    env = AiopsLabEnvironment("p1")
    env.task_description()

    solved, observation = env.apply_action(
        {"action_type": "inspect", "command": "kubectl get pods"}
    )

    assert solved is False
    assert "kubectl get pods" in observation
    assert env._orchestrator.ask_env_calls == ['```\nexec_shell("kubectl get pods")\n```']


def test_apply_action_submit_is_solved(fake_aiopslab_home):
    env = AiopsLabEnvironment("p1")
    env.task_description()

    solved, observation = env.apply_action({"action_type": "submit"})

    assert solved is True
    assert observation == "Submission accepted."


def test_finalize_evaluates_and_tears_down_once(fake_aiopslab_home):
    env = AiopsLabEnvironment("p1")
    env.task_description()

    results = env.finalize(solution={"x": 1}, history=[], duration=12.5)

    # The benchmark verdict stays intact; the root-cause probe is an extra axis and, on the fake
    # problem without kubectl, comes out as "not measured".
    assert results["score"] == 1.0
    assert results["root_cause_fixed"] is None
    problem = env._orchestrator.session.problem
    assert problem.eval_calls == [({"x": 1}, [], 12.5)]
    assert problem.recover_fault_calls == 1
    assert problem.app.cleanup_calls == 1
    assert env._orchestrator.session.ended is True

    with pytest.raises(RuntimeError, match="finalize\\(\\) already called"):
        env.finalize(solution={}, history=[], duration=0.0)


def test_finalize_unregisters_aiopslab_atexit_hook(fake_aiopslab_home, monkeypatch):
    import atexit

    unregistered = []
    monkeypatch.setattr(atexit, "unregister", lambda fn: unregistered.append(fn))

    env = AiopsLabEnvironment("p1")
    env.task_description()
    env.finalize(solution={"x": 1}, history=[], duration=1.0)

    from aiopslab.orchestrator.orchestrator import exit_cleanup_fault

    # Without this, atexit re-runs recover_fault after the namespace is deleted (404 in a real run).
    assert unregistered == [exit_cleanup_fault]


def test_finalize_episode_before_deploy_skips_eval(fake_aiopslab_home):
    env = AiopsLabEnvironment("p1")

    def boom(problem_id):
        raise RuntimeError("deploy failed mid-way")

    env._orchestrator.init_problem = boom
    with pytest.raises(RuntimeError, match="deploy failed mid-way"):
        env.task_description()

    # Without a deploy there is nothing to evaluate or tear down; before this the path raised
    # a TypeError in session.get_duration (start_time=None) and masked the real error.
    assert env.finalize_episode() == {}
    assert env._orchestrator.session is None
    with pytest.raises(RuntimeError, match="finalize\\(\\) already called"):
        env.finalize(solution={}, history=[], duration=0.0)


def test_finalize_episode_uses_aiopslab_session_state(fake_aiopslab_home):
    env = AiopsLabEnvironment("p1")
    env.task_description()

    env.finalize_episode()

    problem = env._orchestrator.session.problem
    assert problem.eval_calls == [({"x": 1}, [], 12.5)]
    assert env._orchestrator.session.ended is True


def test_format_action_submit_and_escalate_ignore_command():
    assert _format_action({"action_type": "submit"}) == "```\nsubmit()\n```"
    assert _format_action({"action_type": "escalate", "command": None}) == "```\nsubmit()\n```"


def test_format_action_remediate_requires_command():
    with pytest.raises(ValueError, match="requires a non-empty 'command'"):
        _format_action({"action_type": "remediate", "command": None})


def test_format_action_escapes_quotes():
    action = {"action_type": "inspect", "command": 'echo "hi"'}
    assert _format_action(action) == '```\nexec_shell("echo \\"hi\\"")\n```'


def test_finalize_keeps_verdict_when_teardown_fails_after_eval(fake_aiopslab_home, capsys):
    env = AiopsLabEnvironment("p1")
    env.task_description()
    problem = env._orchestrator.session.problem
    problem.eval = lambda solution, history, duration: {"success": True, "TTM": 9.0}

    def failing_cleanup():
        raise RuntimeError("Timeout: Namespace was not deleted within 300 seconds")

    problem.app.cleanup = failing_cleanup

    results = env.finalize(solution={}, history=[], duration=9.0)

    # Observed: 2 episodes lost their whole record to a namespace timeout AFTER the eval. The
    # verdict is the data; the teardown failure is recorded, not swallowed.
    assert results["success"] is True
    assert results["TTM"] == 9.0
    assert results["teardown_error"].startswith("RuntimeError: Timeout")
    assert "teardown failed after eval" in capsys.readouterr().err
    assert env._orchestrator.session.ended is True


def test_finalize_reraises_teardown_failure_when_eval_gave_no_verdict(fake_aiopslab_home):
    env = AiopsLabEnvironment("p1")
    env.task_description()
    problem = env._orchestrator.session.problem
    problem.eval = lambda solution, history, duration: {"TTM": 1.0}

    def failing_recover():
        raise RuntimeError("recover exploded")

    problem.recover_fault = failing_recover

    with pytest.raises(RuntimeError, match="recover exploded"):
        env.finalize(solution={}, history=[], duration=1.0)


def test_finalize_records_eval_exception_as_unmeasured_and_still_tears_down(
    fake_aiopslab_home, capsys
):
    env = AiopsLabEnvironment("p1")
    env.task_description()
    problem = env._orchestrator.session.problem

    def exploding_eval(solution, history, duration):
        # Observed on redeploy_without_PV/config_b: a TypeError inside AIOpsLab itself
        # (`for ... in pod.status.container_statuses`, None on a Pending pod).
        raise TypeError("'NoneType' object is not iterable")

    problem.eval = exploding_eval

    results = env.finalize(solution={}, history=[], duration=3.0)

    # The episode does not vanish: verdict not measured (no `success`), reason recorded.
    assert "success" not in results
    assert results["eval_error"] == "TypeError: 'NoneType' object is not iterable"
    assert "eval raised, verdict unmeasured" in capsys.readouterr().err
    # And the cluster is cleaned anyway, otherwise the next episode inherits the injected fault.
    assert problem.recover_fault_calls == 1
    assert problem.app.cleanup_calls == 1
    assert env._orchestrator.session.ended is True


def test_cap_observation_keeps_small_text_intact():
    from adapters.aiopslab_env import cap_observation

    assert cap_observation("pod/geo ready") == "pod/geo ready"


def test_cap_observation_keeps_head_and_tail_and_says_how_much_was_dropped():
    from adapters.aiopslab_env import OBSERVATION_MAX_CHARS, cap_observation

    head = "H" * 20_000
    tail = "T" * 20_000
    capped = cap_observation(head + "M" * 100_000 + tail)

    # Start and end survive (what the command returned and the error at the end); the middle goes.
    assert capped.startswith("H" * 8_000)
    assert capped.endswith("T" * 4_000)
    assert "M" not in capped
    assert "100000" not in capped and "128000 chars omitted" in capped
    assert "narrow the command" in capped
    assert len(capped) < OBSERVATION_MAX_CHARS + 200


def test_apply_action_caps_huge_environment_output(fake_aiopslab_home):
    from adapters.aiopslab_env import OBSERVATION_MAX_CHARS

    env = AiopsLabEnvironment("p1")
    env.task_description()

    # Measured: an agent command dumped ~4MB of the apiserver's /metrics, the whole observation
    # went into the prompt (1.17M tokens against a 131k context) and MLX drove the host to 13GB
    # of swap. The fake returns the command itself as the observation, so a huge command
    # reproduces the shape of the problem without a cluster.
    huge = "kubectl get --raw /metrics " + "x" * 200_000
    solved, observation = env.apply_action({"action_type": "inspect", "command": huge})

    assert solved is False
    assert len(observation) < OBSERVATION_MAX_CHARS + 200
    assert observation.startswith("observed: ")
    assert "chars omitted" in observation


def test_finalize_runs_root_cause_probe_before_recover_fault(fake_aiopslab_home):
    from types import SimpleNamespace as NS

    env = AiopsLabEnvironment("misconfig_app_hotel_res-mitigation-1")
    env.task_description()
    problem = env._orchestrator.session.problem

    class _Kube:
        def __init__(self):
            self.recover_calls_when_probed = None

        def get_deployment(self, name, namespace):
            # The probe has to see the cluster BEFORE teardown's recover_fault.
            self.recover_calls_when_probed = problem.recover_fault_calls
            container = NS(name="c", image="yinfangchen/geo:app3", command=["geo"])
            return NS(spec=NS(template=NS(spec=NS(containers=[container]))), status=NS(ready_replicas=1))

    kube = _Kube()
    problem.kubectl, problem.namespace, problem.faulty_service = kube, "test-hotel", "geo"

    results = env.finalize(solution={}, history=[], duration=1.0)

    assert results["score"] == 1.0  # the benchmark verdict stays intact
    assert results["root_cause_fixed"] is False
    assert "buggy image" in results["probe_detail"]
    assert kube.recover_calls_when_probed == 0
    assert problem.recover_fault_calls == 1


def test_finalize_without_kubectl_on_problem_records_probe_as_unmeasured(fake_aiopslab_home):
    env = AiopsLabEnvironment("p1")
    env.task_description()

    results = env.finalize(solution={}, history=[], duration=1.0)

    assert results["root_cause_fixed"] is None
    assert "does not apply" in results["probe_detail"]


def _home_with_upstream_observer(tmp_path, monkeypatch):
    # Same shape as upstream: load_kube_config with a fixed config_file, at package import.
    pkg = tmp_path / "aiopslab"
    (pkg / "observer").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "observer" / "__init__.py").write_text(
        "from kubernetes import config\n"
        "config.kube_config.load_kube_config(config_file='~/.kube/config')\n"
    )
    for name in [m for m in sys.modules if m == "aiopslab" or m.startswith("aiopslab.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setenv("AIOPSLAB_HOME", str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    # kubernetes is not in requirements (only the VM venv has it): a fake with the same shape.
    calls = []
    kube_config = types.ModuleType("kubernetes.config.kube_config")
    kube_config.load_kube_config = lambda config_file=None, *a, **k: calls.append(config_file)
    config = types.ModuleType("kubernetes.config")
    config.kube_config = kube_config
    kubernetes = types.ModuleType("kubernetes")
    kubernetes.config = config
    monkeypatch.setitem(sys.modules, "kubernetes", kubernetes)
    monkeypatch.setitem(sys.modules, "kubernetes.config", config)
    monkeypatch.setitem(sys.modules, "kubernetes.config.kube_config", kube_config)
    return kube_config, calls


def test_observer_import_reads_the_seed_kubeconfig_not_the_upstream_default(tmp_path, monkeypatch):
    from adapters.aiopslab_env import _ensure_aiopslab_importable

    kube_config, calls = _home_with_upstream_observer(tmp_path, monkeypatch)
    recorder = kube_config.load_kube_config
    seed = str(tmp_path / "sslm-r2-s2.yaml")
    monkeypatch.setenv("KUBECONFIG", seed)

    _ensure_aiopslab_importable()

    assert calls == [seed]
    assert kube_config.load_kube_config is recorder


def test_observer_import_keeps_upstream_path_without_kubeconfig(tmp_path, monkeypatch):
    from adapters.aiopslab_env import _ensure_aiopslab_importable

    _, calls = _home_with_upstream_observer(tmp_path, monkeypatch)
    monkeypatch.delenv("KUBECONFIG", raising=False)

    _ensure_aiopslab_importable()

    assert "aiopslab.observer" not in sys.modules
    assert calls == []
