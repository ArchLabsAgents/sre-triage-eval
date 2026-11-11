"""Tests for harness/episode_runner.py: the real
collect -> decide -> classify -> execute orchestrator behind entrypoint.py's `episode_runner`
seam.

Scope mirrors the module's own docstring: these tests prove the loop end to end with fakes (no
real cluster, no real Prometheus, no real model), and prove the safety-relevant invariant --
classify() gates every model-proposed action BEFORE anything executes, and a denial never reaches
the CommandRunner at all.
"""

from __future__ import annotations

import json

import pytest

from harness import episode_runner as er
from harness.agent import Agent, AgentConfig
from harness.envelope import ActionClass, EpisodeEnvelopeState

NAMESPACE = "testns"
PROMETHEUS_URL = "http://prom.example:9090"


class FakeCommandRunner:
    """Records every command it is asked to run and answers by longest matching prefix, so a
    test only has to declare the commands it cares about."""

    def __init__(self, responses: dict[str, er.CommandResult] | None = None):
        self.responses = responses or {}
        self.calls: list[str] = []

    def __call__(self, command: str, *, timeout: float) -> er.CommandResult:
        self.calls.append(command)
        best_prefix = None
        for prefix in self.responses:
            if command.startswith(prefix) and (best_prefix is None or len(prefix) > len(best_prefix)):
                best_prefix = prefix
        if best_prefix is not None:
            return self.responses[best_prefix]
        raise AssertionError(f"FakeCommandRunner: no response configured for {command!r}")


def _ok(stdout: str = "") -> er.CommandResult:
    return er.CommandResult(returncode=0, stdout=stdout, stderr="")


def _fail(stderr: str = "boom", returncode: int = 1) -> er.CommandResult:
    return er.CommandResult(returncode=returncode, stdout="", stderr=stderr)


PODS_JSON = {
    "items": [
        {
            "metadata": {"name": "checkout-1"},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]},
        },
        {
            "metadata": {"name": "checkout-2"},
            "status": {"conditions": [{"type": "Ready", "status": "False"}]},
        },
    ]
}
DEPLOYMENTS_JSON = {"items": [{"metadata": {"name": "checkout"}}]}
EVENTS_JSON = {"items": []}


def _collection_responses() -> dict[str, er.CommandResult]:
    return {
        f"kubectl get pods -n {NAMESPACE} -o json": _ok(json.dumps(PODS_JSON)),
        f"kubectl get events -n {NAMESPACE} -o json": _ok(json.dumps(EVENTS_JSON)),
        f"kubectl get deployments -n {NAMESPACE} -o json": _ok(json.dumps(DEPLOYMENTS_JSON)),
        f"kubectl logs checkout-2 -n {NAMESPACE} --tail=200": _ok("crashloop traceback here"),
    }


def _webhook_raw(*, generator_url: str = "") -> dict:
    return {
        "groupKey": '{}:{alertname="KubePodCrashLooping"}',
        "status": "firing",
        "receiver": "sslm-agent",
        "alerts": [
            {
                "status": "firing",
                "fingerprint": "fp-1",
                "labels": {"alertname": "KubePodCrashLooping", "namespace": NAMESPACE},
                "annotations": {"summary": "pod crashlooping"},
                "generatorURL": generator_url,
            }
        ],
    }


# ---------------------------------------------------------------------------
# subprocess_command_runner
# ---------------------------------------------------------------------------


def test_subprocess_command_runner_runs_a_real_command():
    result = er.subprocess_command_runner("echo hello", timeout=5.0)
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"


def test_subprocess_command_runner_unparseable_command_is_reported_not_raised():
    result = er.subprocess_command_runner('kubectl exec -- cat "unterminated', timeout=5.0)
    assert result.returncode == -1
    assert "unparseable" in result.stderr


def test_subprocess_command_runner_missing_binary_is_reported_not_raised():
    result = er.subprocess_command_runner("definitely-not-a-real-binary-xyz --version", timeout=5.0)
    assert result.returncode == -1
    assert result.stderr


# ---------------------------------------------------------------------------
# _extract_scale_replicas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command,expected",
    [
        ("kubectl scale deployment/checkout --replicas=3", 3),
        ("kubectl scale deployment/checkout --replicas 7", 7),
        ("kubectl rollout restart deployment/checkout", None),
        ("kubectl scale deployment/checkout", None),
    ],
)
def test_extract_scale_replicas(command, expected):
    assert er._extract_scale_replicas(command) == expected


# ---------------------------------------------------------------------------
# _extract_promql_from_generator_url
# ---------------------------------------------------------------------------


def test_extract_promql_from_generator_url_reads_g0_expr():
    url = "http://prometheus:9090/graph?g0.expr=up%7Bjob%3D%22checkout%22%7D&g0.tab=1"
    assert er._extract_promql_from_generator_url(url) == 'up{job="checkout"}'


def test_extract_promql_from_generator_url_missing_expr_returns_none():
    assert er._extract_promql_from_generator_url("http://prometheus:9090/graph?g0.tab=1") is None


def test_extract_promql_from_generator_url_empty_returns_none():
    assert er._extract_promql_from_generator_url("") is None


# ---------------------------------------------------------------------------
# query_range
# ---------------------------------------------------------------------------


def test_query_range_returns_result_list(monkeypatch):
    monkeypatch.setattr(
        er,
        "_http_get_json",
        lambda url, *, timeout: {"status": "success", "data": {"result": [{"metric": {}, "values": [[1, "1"]]}]}},
    )
    result = er.query_range(PROMETHEUS_URL, "up", start=0.0, end=60.0, step_seconds=30.0)
    assert result == [{"metric": {}, "values": [[1, "1"]]}]


def test_query_range_raises_on_non_success_status(monkeypatch):
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "error", "error": "bad query"})
    with pytest.raises(er.PrometheusQueryError):
        er.query_range(PROMETHEUS_URL, "up", start=0.0, end=60.0, step_seconds=30.0)


def test_query_range_transport_failure_raises_prometheus_query_error(monkeypatch):
    import urllib.error

    def _raise(url, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(er.urllib.request, "urlopen", lambda request, timeout: _raise(request, timeout))
    with pytest.raises(er.PrometheusQueryError):
        er.query_range(PROMETHEUS_URL, "up", start=0.0, end=60.0, step_seconds=30.0)


def test_query_range_malformed_url_raises_prometheus_query_error_not_value_error(monkeypatch):
    """urlopen raises a bare ValueError (not URLError) for a
    schemeless/malformed URL, e.g. a PROMETHEUS_URL missing its http:// prefix. Must degrade to
    PrometheusQueryError like every other transport failure, not propagate and abort the whole
    episode."""

    def _raise(url, timeout):
        raise ValueError("unknown url type: 'monitoring-prometheus:9090'")

    monkeypatch.setattr(er.urllib.request, "urlopen", lambda request, timeout: _raise(request, timeout))
    with pytest.raises(er.PrometheusQueryError):
        er.query_range(PROMETHEUS_URL, "up", start=0.0, end=60.0, step_seconds=30.0)


# ---------------------------------------------------------------------------
# collect_episode_context
# ---------------------------------------------------------------------------


def test_collect_episode_context_happy_path(monkeypatch):
    monkeypatch.setattr(
        er,
        "_http_get_json",
        lambda url, *, timeout: {"status": "success", "data": {"result": [{"values": [[1, "1"]]}]}},
    )
    runner = FakeCommandRunner(_collection_responses())

    context, targets_seen = er.collect_episode_context(
        "ep-1",
        _webhook_raw(generator_url="http://prom/graph?g0.expr=up&g0.tab=1"),
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=runner,
    )

    names = {source.name: source for source in context.sources}
    assert names["alertmanager_payload"].collected
    assert names["prometheus_range[0]"].collected
    assert names["k8s_pods"].collected
    assert names["k8s_events"].collected
    assert names["k8s_deployments"].collected
    # only checkout-2 is not-Ready
    assert names["k8s_logs[checkout-2]"].collected
    assert "k8s_logs[checkout-1]" not in names

    assert targets_seen == {"pod/checkout-1", "pod/checkout-2", "deployment/checkout"}
    assert "Episode ep-1" in context.as_prompt_text()


def test_collect_episode_context_caps_not_ready_pod_logs(monkeypatch):
    """A cascading failure with many not-Ready pods must not collect logs for all of them
    unbounded -- same failure mode as the measured OOM incident (adapters/aiopslab_env.py's
    OBSERVATION_MAX_CHARS), just at the "how many sources" axis instead of "how big one source
    is"."""
    monkeypatch.setattr(er, "MAX_NOT_READY_PODS_LOGGED", 2)
    monkeypatch.setattr(
        er,
        "_http_get_json",
        lambda url, *, timeout: {"status": "success", "data": {"result": [{"values": [[1, "1"]]}]}},
    )
    pods_json = {
        "items": [
            {"metadata": {"name": name}, "status": {"conditions": [{"type": "Ready", "status": "False"}]}}
            for name in ["pod-c", "pod-a", "pod-b", "pod-d"]
        ]
    }
    responses = {
        f"kubectl get pods -n {NAMESPACE} -o json": _ok(json.dumps(pods_json)),
        f"kubectl get events -n {NAMESPACE} -o json": _ok(json.dumps(EVENTS_JSON)),
        f"kubectl get deployments -n {NAMESPACE} -o json": _ok(json.dumps(DEPLOYMENTS_JSON)),
        f"kubectl logs pod-a -n {NAMESPACE} --tail=200": _ok("log a"),
        f"kubectl logs pod-b -n {NAMESPACE} --tail=200": _ok("log b"),
    }
    runner = FakeCommandRunner(responses)

    context, _ = er.collect_episode_context(
        "ep-cap",
        _webhook_raw(),
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=runner,
    )

    names = {source.name: source for source in context.sources}
    # sorted names, so the first 2 (the cap) are pod-a and pod-b, not pod-c/pod-d.
    assert names["k8s_logs[pod-a]"].collected
    assert names["k8s_logs[pod-b]"].collected
    assert "k8s_logs[pod-c]" not in names
    assert "k8s_logs[pod-d]" not in names
    skipped = names["k8s_logs[...skipped]"]
    assert not skipped.collected
    assert "pod-c" in skipped.reason and "pod-d" in skipped.reason
    # neither pod-c nor pod-d's logs were ever requested from the command runner.
    assert not any("pod-c" in call or "pod-d" in call for call in runner.calls if call.startswith("kubectl logs"))


def test_collect_episode_context_caps_alerts_queried(monkeypatch):
    """The OTHER unbounded-count axis -- an Alertmanager group
    with many alerts (e.g. a per-pod symptom-alert rule firing across a bad rollout) must not
    query Prometheus for all of them unbounded, same failure class as the not-ready-pod-logs
    cap above."""
    monkeypatch.setattr(er, "MAX_ALERTS_QUERIED", 2)
    monkeypatch.setattr(
        er,
        "_http_get_json",
        lambda url, *, timeout: {"status": "success", "data": {"result": [{"values": [[1, "1"]]}]}},
    )
    webhook_raw = {
        "groupKey": '{}:{alertname="KubePodCrashLooping"}',
        "status": "firing",
        "receiver": "sslm-agent",
        "alerts": [
            {
                "status": "firing",
                "fingerprint": f"fp-{i}",
                "labels": {"alertname": "KubePodCrashLooping", "namespace": NAMESPACE},
                "annotations": {},
                "generatorURL": f"http://prom/graph?g0.expr=up{{pod%3D%22p{i}%22}}&g0.tab=1",
            }
            for i in range(4)
        ],
    }
    runner = FakeCommandRunner(_collection_responses())

    context, _ = er.collect_episode_context(
        "ep-alert-cap",
        webhook_raw,
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=runner,
    )

    names = {source.name: source for source in context.sources}
    assert names["prometheus_range[0]"].collected
    assert names["prometheus_range[1]"].collected
    assert "prometheus_range[2]" not in names
    assert "prometheus_range[3]" not in names
    skipped = names["prometheus_range[...skipped]"]
    assert not skipped.collected
    assert "2 more" in skipped.reason


def test_collect_episode_context_no_generator_url_marks_prometheus_source_uncollected():
    runner = FakeCommandRunner(_collection_responses())
    context, _targets = er.collect_episode_context(
        "ep-1", _webhook_raw(generator_url=""), namespace=NAMESPACE, prometheus_url=PROMETHEUS_URL, command_runner=runner
    )
    source = next(s for s in context.sources if s.name == "prometheus_range[0]")
    assert source.collected is False
    assert "generatorURL" in source.reason


def test_collect_episode_context_kubectl_failure_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "success", "data": {"result": []}})
    responses = _collection_responses()
    responses[f"kubectl get pods -n {NAMESPACE} -o json"] = _fail("NotFound")
    runner = FakeCommandRunner(responses)

    context, targets_seen = er.collect_episode_context(
        "ep-1", _webhook_raw(), namespace=NAMESPACE, prometheus_url=PROMETHEUS_URL, command_runner=runner
    )

    pods_source = next(s for s in context.sources if s.name == "k8s_pods")
    assert pods_source.collected is False
    assert "NotFound" in pods_source.reason
    # no pod list -> logs step is skipped honestly, not silently omitted
    logs_source = next(s for s in context.sources if s.name == "k8s_logs")
    assert logs_source.collected is False
    assert "pod" in logs_source.reason
    assert "pod/checkout-1" not in targets_seen


def test_collect_episode_context_honors_budget_exhaustion(monkeypatch):
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "success", "data": {"result": []}})
    runner = FakeCommandRunner(_collection_responses())

    # A clock that reports the budget already spent from the very first check onward.
    ticks = iter([0.0] + [1000.0] * 50)
    context, _targets = er.collect_episode_context(
        "ep-1",
        _webhook_raw(generator_url="http://prom/graph?g0.expr=up&g0.tab=1"),
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=runner,
        budget_seconds=90.0,
        clock=lambda: next(ticks),
    )

    non_payload_sources = [s for s in context.sources if s.name != "alertmanager_payload"]
    assert non_payload_sources, "expected at least the prometheus/k8s sources to be attempted"
    for source in non_payload_sources:
        assert source.collected is False
    # every budget-gated source (everything except the final logs fallback, which reports why it
    # was skipped in terms of the pods collection rather than the budget directly) names the
    # budget explicitly
    budget_named = [s for s in non_payload_sources if s.name != "k8s_logs"]
    assert budget_named
    for source in budget_named:
        assert "budget" in source.reason
    # the scripted collection never called any command once the budget was already spent
    assert runner.calls == []


# ---------------------------------------------------------------------------
# LiveClusterEnvironment.apply_action -- the safety-relevant part.
# ---------------------------------------------------------------------------


def _environment(*, targets_seen: set[str] | None = None, command_runner: FakeCommandRunner | None = None) -> er.LiveClusterEnvironment:
    context = er.EpisodeContext(episode_id="ep-1", namespace=NAMESPACE)
    state = EpisodeEnvelopeState(targets_seen=set(targets_seen or set()))
    return er.LiveClusterEnvironment(
        context=context, state=state, command_runner=command_runner or FakeCommandRunner()
    )


def test_escalate_ends_the_episode_without_touching_the_command_runner():
    runner = FakeCommandRunner()
    env = _environment(command_runner=runner)

    solved, observation = env.apply_action({"action_type": "escalate", "diagnosis": "unknown fault", "cause_class": "unknown", "target": "", "confidence": 0.1})

    assert solved is True
    assert "escalate" in observation
    assert runner.calls == []
    assert env.denials == []
    assert env.executions == []


def test_submit_ends_the_episode():
    env = _environment()
    solved, _observation = env.apply_action({"action_type": "submit", "diagnosis": "fixed", "cause_class": "unknown", "target": "", "confidence": 0.9})
    assert solved is True


def test_c3_action_is_denied_and_never_executed():
    runner = FakeCommandRunner()
    env = _environment(command_runner=runner)

    solved, observation = env.apply_action(
        {
            "action_type": "remediate",
            "diagnosis": "bad yaml",
            "cause_class": "unknown",
            "target": "deployment/checkout",
            "command": "kubectl apply -f recommendation-pvc.yaml",
            "confidence": 0.9,
        }
    )

    assert solved is False
    assert observation.startswith("DENIED (C3)")
    assert runner.calls == []  # never reached the command runner
    assert len(env.denials) == 1
    assert env.denials[0].action_class == ActionClass.C3
    assert env.executions == []


def test_c0_action_executes_and_learns_the_target_on_success():
    runner = FakeCommandRunner({f"kubectl get deployment/checkout -n {NAMESPACE}": _ok("deployment details")})
    env = _environment(command_runner=runner)

    solved, observation = env.apply_action(
        {
            "action_type": "inspect",
            "diagnosis": "checking state",
            "cause_class": "unknown",
            "target": "deployment/checkout",
            "command": f"kubectl get deployment/checkout -n {NAMESPACE}",
            "confidence": 0.5,
        }
    )

    assert solved is False
    assert "exit=0" in observation
    assert runner.calls == [f"kubectl get deployment/checkout -n {NAMESPACE}"]
    assert env._state.c0_calls == 1
    assert "deployment/checkout" in env._state.targets_seen
    assert len(env.executions) == 1
    assert env.executions[0]["action_class"] == "C0"


def test_c0_action_that_fails_does_not_learn_the_target():
    runner = FakeCommandRunner({f"kubectl get deployment/ghost -n {NAMESPACE}": _fail("NotFound", returncode=1)})
    env = _environment(command_runner=runner)

    env.apply_action(
        {
            "action_type": "inspect",
            "diagnosis": "checking a resource that does not exist",
            "cause_class": "unknown",
            "target": "deployment/ghost",
            "command": f"kubectl get deployment/ghost -n {NAMESPACE}",
            "confidence": 0.5,
        }
    )

    assert "deployment/ghost" not in env._state.targets_seen
    # a call still counts against the C0 cap even though the target lookup failed -- the gate
    # only refuses to trust a failed lookup as "target confirmed", it does not refund the call.
    assert env._state.c0_calls == 1


def test_c0_action_does_not_learn_a_target_the_command_never_named():
    """The model's self-reported `target` is not tied to `command` by
    AGENT_ACTION_SCHEMA. A cheap unrelated C0 read (list every pod) paired with a fabricated
    `target` must not mark that target "seen" -- that would forge read-before-mutate."""
    runner = FakeCommandRunner({f"kubectl get pods -n {NAMESPACE}": _ok(json.dumps(PODS_JSON))})
    env = _environment(command_runner=runner)

    solved, observation = env.apply_action(
        {
            "action_type": "inspect",
            "diagnosis": "listing pods",
            "cause_class": "unknown",
            "target": "deployment/payments-db",  # not named anywhere in `command` below
            "command": f"kubectl get pods -n {NAMESPACE}",
            "confidence": 0.5,
        }
    )

    assert solved is False
    assert "exit=0" in observation  # the C0 call itself still succeeds and executes
    assert "deployment/payments-db" not in env._state.targets_seen
    assert env._state.c0_calls == 1  # still counts against the cap, same as any other C0 call


def test_c0_action_does_not_learn_target_via_label_selector_bypass():
    """Concrete repro: `-l <podname>`
    is a valid label selector that always exits 0 (even matching zero pods), not a read of that
    pod. Must not mark it "seen" via the bare-name fallback."""
    command = f"kubectl get pods -l checkout-1 -n {NAMESPACE}"
    runner = FakeCommandRunner({command: _ok(json.dumps({"items": []}))})
    env = _environment(command_runner=runner)

    env.apply_action(
        {
            "action_type": "inspect",
            "diagnosis": "checking by label",
            "cause_class": "unknown",
            "target": "pod/checkout-1",
            "command": command,
            "confidence": 0.5,
        }
    )

    assert "pod/checkout-1" not in env._state.targets_seen


@pytest.mark.parametrize(
    "command,target,expected",
    [
        (f"kubectl get pod/checkout-1 -n {NAMESPACE}", "pod/checkout-1", True),
        (f"kubectl get pod checkout-1 -n {NAMESPACE}", "pod/checkout-1", True),
        (f"kubectl logs checkout-1 -n {NAMESPACE} --tail=200", "pod/checkout-1", True),
        (f"kubectl get pods -n {NAMESPACE}", "pod/checkout-1", False),
        (f"kubectl get deployment/checkout -n {NAMESPACE}", "pod/checkout-1", False),
        ("not even a valid shell command'", "pod/checkout-1", False),
        # The bare-name fallback must be scoped to exactly
        # `kubectl logs <name>`, not any C0 verb with `name` appearing as any token anywhere --
        # `-l <name>` is a valid, always-exits-0 label selector, not a read of that pod.
        (f"kubectl get pods -l checkout-1 -n {NAMESPACE}", "pod/checkout-1", False),
        (f"kubectl get pods -n checkout-1", "pod/checkout-1", False),
        (f"kubectl describe pod checkout-1 -n {NAMESPACE}", "pod/checkout-1", True),
        (f"kubectl logs -n {NAMESPACE} checkout-1", "pod/checkout-1", False),  # not positional
    ],
)
def test_command_references_target(command, target, expected):
    assert er._command_references_target(command, target) is expected


def test_c1_action_requires_a_prior_c0_call_in_the_same_episode():
    runner = FakeCommandRunner(
        {
            f"kubectl get pod/checkout-1 -n {NAMESPACE}": _ok("pod details"),
            "kubectl exec checkout-1 -n testns -- cat /var/log/app.log": _ok("log tail"),
        }
    )
    env = _environment(command_runner=runner)

    # C1 before any C0 this episode: denied, the exec never runs.
    solved, observation = env.apply_action(
        {
            "action_type": "inspect",
            "diagnosis": "reading app log",
            "cause_class": "unknown",
            "target": "pod/checkout-1",
            "command": "kubectl exec checkout-1 -n testns -- cat /var/log/app.log",
            "confidence": 0.5,
        }
    )
    assert solved is False
    assert observation.startswith("DENIED (C1)")
    assert env._state.c1_calls == 0

    # A C0 call first ...
    env.apply_action(
        {
            "action_type": "inspect",
            "diagnosis": "checking pod state",
            "cause_class": "unknown",
            "target": "pod/checkout-1",
            "command": f"kubectl get pod/checkout-1 -n {NAMESPACE}",
            "confidence": 0.5,
        }
    )
    assert env._state.c0_calls == 1

    # ... unblocks the same C1 command.
    solved, observation = env.apply_action(
        {
            "action_type": "inspect",
            "diagnosis": "reading app log",
            "cause_class": "unknown",
            "target": "pod/checkout-1",
            "command": "kubectl exec checkout-1 -n testns -- cat /var/log/app.log",
            "confidence": 0.5,
        }
    )
    assert solved is False
    assert "exit=0" in observation
    assert env._state.c1_calls == 1
    assert env.executions[-1]["action_class"] == "C1"


def test_c2_action_requires_read_before_mutate_and_confidence():
    runner = FakeCommandRunner({"kubectl rollout restart deployment/checkout": _ok("restarted")})
    env = _environment(command_runner=runner)  # no targets_seen yet

    solved, observation = env.apply_action(
        {
            "action_type": "remediate",
            "diagnosis": "checkout is overloaded",
            "cause_class": "unknown",
            "target": "deployment/checkout",
            "command": "kubectl rollout restart deployment/checkout",
            "confidence": 0.9,
        }
    )

    assert solved is False
    assert observation.startswith("DENIED (C2)")
    assert runner.calls == []
    assert env._state.c2_calls == 0


def test_c2_action_executes_once_gates_are_satisfied():
    # c2_dry_run=False: this test is about the envelope's own gates (confidence/read-before-
    # mutate/cap), not the C2 dry-run gate -- that gate has its own dedicated
    # tests below (test_c2_dry_run_*). restart's rollback_state precompute needs no read (it is
    # idempotent), but real execution also runs a read-back, hence
    # the extra `kubectl rollout status` response.
    runner = FakeCommandRunner(
        {
            "kubectl rollout restart deployment/checkout": _ok("restarted"),
            "kubectl rollout status deployment/checkout": _ok('deployment "checkout" successfully rolled out'),
        }
    )
    env = _environment_with_dry_run(c2_dry_run=False, targets_seen={"deployment/checkout"}, command_runner=runner)

    solved, observation = env.apply_action(
        {
            "action_type": "remediate",
            "diagnosis": "checkout is overloaded",
            "cause_class": "unknown",
            "target": "deployment/checkout",
            "command": "kubectl rollout restart deployment/checkout",
            "confidence": 0.9,
        }
    )

    assert solved is False
    assert "exit=0" in observation
    assert runner.calls[0] == "kubectl rollout restart deployment/checkout"
    assert runner.calls[1].startswith("kubectl rollout status deployment/checkout")
    assert env._state.c2_calls == 1
    assert env.executions[0]["action_class"] == "C2"
    assert env.executions[0]["rollback_state"]["tipo"] == "rollout_restart"
    assert env.executions[0]["rollback_state"]["comando_rollback"] is None
    assert env.executions[0]["read_back"]["ready"] is True
    assert env.rollback_events == [
        {"command": "kubectl rollout restart deployment/checkout", "improved": True, "rollback_executed": False}
    ]


def test_scale_action_denied_with_no_declared_bounds():
    runner = FakeCommandRunner({"kubectl scale deployment/checkout --replicas=3": _ok("scaled")})
    env = _environment(targets_seen={"deployment/checkout"}, command_runner=runner)

    solved, observation = env.apply_action(
        {
            "action_type": "remediate",
            "diagnosis": "needs more replicas",
            "cause_class": "unknown",
            "target": "deployment/checkout",
            "command": "kubectl scale deployment/checkout --replicas=3",
            "confidence": 0.9,
        }
    )

    assert solved is False
    assert observation.startswith("DENIED (C2)")
    assert "bounds" in observation
    assert runner.calls == []


# ---------------------------------------------------------------------------
# run_real_episode / make_episode_runner -- the full loop, reusing harness.agent.Agent.
# ---------------------------------------------------------------------------


class StubModel:
    def __init__(self, outputs: list[str]):
        self._outputs = iter(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, schema_constrained: bool) -> str:
        self.prompts.append(prompt)
        return next(self._outputs)


ESCALATE_ACTION = json.dumps(
    {
        "diagnosis": "unclear root cause, deferring to a human",
        "cause_class": "unknown",
        "action_type": "escalate",
        "target": "checkout",
        "command": None,
        "confidence": 0.2,
    }
)


def test_run_real_episode_end_to_end_reuses_agent_and_schema(monkeypatch):
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "success", "data": {"result": []}})
    runner = FakeCommandRunner(_collection_responses())
    model = StubModel([ESCALATE_ACTION])

    result = er.run_real_episode(
        "ep-1",
        _webhook_raw(),
        model=model,
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=runner,
    )

    assert result.solved is True
    assert result.attempt_count == 1
    # the scripted collection's text made it into the prompt the model actually saw
    assert "Episode ep-1" in model.prompts[0]
    assert "k8s_pods" in model.prompts[0] or "checkout" in model.prompts[0]


def test_run_real_episode_does_not_seed_targets_seen_from_scripted_collection(monkeypatch):
    """An earlier version seeded EpisodeEnvelopeState.targets_seen from
    every pod/deployment the scripted collection listed, satisfying read-before-mutate for
    anything in the namespace from turn 1. `deployment/checkout` IS in DEPLOYMENTS_JSON (the
    scripted collection sees it), but this episode never issues a C0 call against it -- the
    mutate must still be denied."""
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "success", "data": {"result": []}})
    runner = FakeCommandRunner(_collection_responses())
    scale_action = json.dumps(
        {
            "diagnosis": "scaling up without reading it first this episode",
            "cause_class": "unknown",
            "action_type": "remediate",
            "target": "deployment/checkout",
            "command": "kubectl scale deployment/checkout --replicas=3",
            "confidence": 0.9,
        }
    )
    model = StubModel([scale_action, ESCALATE_ACTION])

    result = er.run_real_episode(
        "ep-2",
        _webhook_raw(),
        model=model,
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=runner,
    )

    assert result.attempt_count == 2
    assert result.turns[0].observation.startswith("DENIED (C2)")
    assert "read-before-mutate" in result.turns[0].observation
    assert not any(call.startswith("kubectl scale") for call in runner.calls)


def test_make_episode_runner_matches_the_entrypoint_episode_runner_shape(monkeypatch):
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "success", "data": {"result": []}})
    runner = FakeCommandRunner(_collection_responses())
    model = StubModel([ESCALATE_ACTION])

    episode_runner = er.make_episode_runner(
        model=model, namespace=NAMESPACE, prometheus_url=PROMETHEUS_URL, command_runner=runner
    )

    # Same call shape entrypoint.WebhookReceiver uses: (episode_id, webhook_raw) -> None.
    assert episode_runner("ep-1", _webhook_raw()) is None


def test_run_real_episode_accepts_a_custom_agent_config():
    runner = FakeCommandRunner(_collection_responses())
    model = StubModel([ESCALATE_ACTION])

    result = er.run_real_episode(
        "ep-1",
        _webhook_raw(),
        model=model,
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=runner,
        agent_config=AgentConfig(schema_constrained_decoding=False, memory_enabled=False),
        max_turns=3,
    )
    assert isinstance(result.solved, bool)


@pytest.mark.parametrize("env_value, expected", [(None, True), ("on", True), ("off", False)])
def test_run_real_episode_default_agent_config_follows_sslm_history_budget(
    monkeypatch, env_value, expected
):
    # The pod path (entrypoint -> make_episode_runner -> run_real_episode) builds the default
    # AgentConfig; setup_cluster.sh sets SSLM_HISTORY_BUDGET=off only when COMPARATOR_ARM=1.
    seen = []
    real_agent = er.Agent

    def capturing_agent(model, config, *args, **kwargs):
        seen.append(config)
        return real_agent(model, config, *args, **kwargs)

    if env_value is None:
        monkeypatch.delenv("SSLM_HISTORY_BUDGET", raising=False)
    else:
        monkeypatch.setenv("SSLM_HISTORY_BUDGET", env_value)
    monkeypatch.setattr(er, "Agent", capturing_agent)
    runner = FakeCommandRunner(_collection_responses())

    er.run_real_episode(
        "ep-1",
        _webhook_raw(),
        model=StubModel([ESCALATE_ACTION]),
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=runner,
        max_turns=3,
    )

    assert seen[0].history_budget is expected


# ---------------------------------------------------------------------------
# c2_dry_run_enabled -- the SSLM_C2_DRY_RUN env flag.
# ---------------------------------------------------------------------------


def test_c2_dry_run_enabled_defaults_true_when_unset():
    assert er.c2_dry_run_enabled(env={}) is True


@pytest.mark.parametrize("raw", ["0", "false", "False", "no", "NO", "off", " off "])
def test_c2_dry_run_enabled_false_for_recognized_off_values(raw):
    assert er.c2_dry_run_enabled(env={er.C2_DRY_RUN_ENV_VAR: raw}) is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "anything-unrecognized"])
def test_c2_dry_run_enabled_true_for_everything_else(raw):
    # Degrade-safe (module docstring): an unrecognized value stays in the SAFE state (dry-run
    # on), never silently interpreted as "off".
    assert er.c2_dry_run_enabled(env={er.C2_DRY_RUN_ENV_VAR: raw}) is True


# ---------------------------------------------------------------------------
# LiveClusterEnvironment.apply_action -- the C2 dry-run gate itself.
# ---------------------------------------------------------------------------


def _environment_with_dry_run(
    *, c2_dry_run: bool, targets_seen: set[str] | None = None, command_runner: FakeCommandRunner | None = None
) -> er.LiveClusterEnvironment:
    context = er.EpisodeContext(episode_id="ep-1", namespace=NAMESPACE)
    state = EpisodeEnvelopeState(targets_seen=set(targets_seen or set()))
    return er.LiveClusterEnvironment(
        context=context,
        state=state,
        command_runner=command_runner or FakeCommandRunner(),
        c2_dry_run=c2_dry_run,
    )


def test_c2_dry_run_default_is_true():
    """LiveClusterEnvironment's own default, independent of the env-var reader above: a caller
    that forgets to pass c2_dry_run explicitly still gets the safe behavior."""
    runner = FakeCommandRunner({"kubectl rollout restart deployment/checkout": _ok("restarted")})
    env = _environment(targets_seen={"deployment/checkout"}, command_runner=runner)

    solved, observation = env.apply_action(
        {
            "action_type": "remediate",
            "diagnosis": "checkout is overloaded",
            "cause_class": "unknown",
            "target": "deployment/checkout",
            "command": "kubectl rollout restart deployment/checkout",
            "confidence": 0.9,
        }
    )

    assert solved is False
    assert observation.startswith("DRY-RUN")
    assert runner.calls == []  # never reached the command runner


def test_c2_dry_run_blocks_real_execution_but_still_counts_against_the_cap():
    runner = FakeCommandRunner({"kubectl rollout restart deployment/checkout": _ok("restarted")})
    env = _environment_with_dry_run(c2_dry_run=True, targets_seen={"deployment/checkout"}, command_runner=runner)

    solved, observation = env.apply_action(
        {
            "action_type": "remediate",
            "diagnosis": "checkout is overloaded",
            "cause_class": "unknown",
            "target": "deployment/checkout",
            "command": "kubectl rollout restart deployment/checkout",
            "confidence": 0.9,
        }
    )

    assert solved is False
    assert observation.startswith("DRY-RUN (C2 gate active)")
    assert "NOT executed" in observation
    assert runner.calls == []  # command_runner never called for the dry-run C2
    assert env._state.c2_calls == 1  # the gate still counted the attempt
    assert len(env.executions) == 1
    assert env.executions[0]["action_class"] == "C2"
    assert env.executions[0]["dry_run"] is True
    assert env.executions[0]["returncode"] is None


def test_c2_dry_run_false_executes_for_real():
    runner = FakeCommandRunner(
        {
            "kubectl rollout restart deployment/checkout": _ok("restarted"),
            "kubectl rollout status deployment/checkout": _ok('deployment "checkout" successfully rolled out'),
        }
    )
    env = _environment_with_dry_run(c2_dry_run=False, targets_seen={"deployment/checkout"}, command_runner=runner)

    solved, observation = env.apply_action(
        {
            "action_type": "remediate",
            "diagnosis": "checkout is overloaded",
            "cause_class": "unknown",
            "target": "deployment/checkout",
            "command": "kubectl rollout restart deployment/checkout",
            "confidence": 0.9,
        }
    )

    assert solved is False
    assert "exit=0" in observation
    assert runner.calls[0] == "kubectl rollout restart deployment/checkout"
    assert env.executions[0]["dry_run"] is False


def test_c2_dry_run_does_not_affect_c0_or_c1_execution():
    runner = FakeCommandRunner({f"kubectl get deployment/checkout -n {NAMESPACE}": _ok("deployment details")})
    env = _environment_with_dry_run(c2_dry_run=True, command_runner=runner)

    solved, observation = env.apply_action(
        {
            "action_type": "inspect",
            "diagnosis": "checking state",
            "cause_class": "unknown",
            "target": "deployment/checkout",
            "command": f"kubectl get deployment/checkout -n {NAMESPACE}",
            "confidence": 0.5,
        }
    )

    assert "exit=0" in observation
    assert runner.calls == [f"kubectl get deployment/checkout -n {NAMESPACE}"]
    assert env.executions[0]["dry_run"] is False


def test_c2_denied_before_dry_run_gate_is_never_logged_as_dry_run():
    """A C2 the envelope itself denies (e.g. no read-before-mutate) must stay a plain denial --
    the dry-run gate only applies to a C2 that classify() already allowed."""
    runner = FakeCommandRunner({"kubectl rollout restart deployment/checkout": _ok("restarted")})
    env = _environment_with_dry_run(c2_dry_run=True, command_runner=runner)  # no targets_seen

    solved, observation = env.apply_action(
        {
            "action_type": "remediate",
            "diagnosis": "checkout is overloaded",
            "cause_class": "unknown",
            "target": "deployment/checkout",
            "command": "kubectl rollout restart deployment/checkout",
            "confidence": 0.9,
        }
    )

    assert observation.startswith("DENIED (C2)")
    assert env.executions == []
    assert len(env.denials) == 1


# ---------------------------------------------------------------------------
# make_episode_runner -- TriageRecord emission.
# ---------------------------------------------------------------------------


def test_make_episode_runner_emits_a_triage_record_on_the_happy_path(monkeypatch):
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "success", "data": {"result": []}})
    runner = FakeCommandRunner(_collection_responses())
    model = StubModel([ESCALATE_ACTION])
    emitted = []

    episode_runner = er.make_episode_runner(
        model=model,
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=runner,
        triage_sink=emitted.append,
    )

    episode_runner("ep-1", _webhook_raw())

    assert len(emitted) == 1
    record = emitted[0]
    assert record["episode_id"] == "ep-1"
    assert record["decisao"] == "escalar"
    assert record["alert_fingerprint"] == "fp-1"


def test_make_episode_runner_emits_a_triage_record_even_when_the_orchestrator_raises(caplog):
    class ExplodingModel:
        def generate(self, prompt: str, *, schema_constrained: bool) -> str:
            raise RuntimeError("model backend unreachable")

    emitted = []
    episode_runner = er.make_episode_runner(
        model=ExplodingModel(),
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=FakeCommandRunner(_collection_responses()),
        triage_sink=emitted.append,
    )

    with caplog.at_level("ERROR", logger="sslm.episode_runner"):
        result = episode_runner("ep-1", _webhook_raw())

    assert result is None
    assert len(emitted) == 1  # "emite TriageRecord, sempre" holds even when the orchestrator raised
    record = emitted[0]
    assert record["decisao"] == "escalar"
    assert record["episode_id"] == "ep-1"
    assert record["alert_fingerprint"] == "fp-1"
    assert "orchestrator raised" in record["acao_executada_motivo"]


def test_make_episode_runner_default_sink_logs_the_record(monkeypatch, caplog):
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "success", "data": {"result": []}})
    runner = FakeCommandRunner(_collection_responses())
    model = StubModel([ESCALATE_ACTION])

    episode_runner = er.make_episode_runner(
        model=model, namespace=NAMESPACE, prometheus_url=PROMETHEUS_URL, command_runner=runner
    )

    with caplog.at_level("INFO", logger="sslm.episode_runner"):
        episode_runner("ep-1", _webhook_raw())

    assert any("triage_record" in message and "ep-1" in message for message in caplog.messages)


def _inspect_then_remediate_actions() -> list[str]:
    # Section 3.1's read-before-mutate gate needs its own C0 get/describe against the target
    # THIS episode before a C2 mutate against it is allowed (run_real_episode's docstring: the
    # scripted collection's own targets_seen is deliberately NOT seeded into the gate).
    inspect_action = json.dumps(
        {
            "diagnosis": "checking deployment state before acting",
            "cause_class": "unknown",
            "action_type": "inspect",
            "target": "deployment/checkout",
            "command": f"kubectl get deployment/checkout -n {NAMESPACE}",
            "confidence": 0.5,
        }
    )
    remediate_action = json.dumps(
        {
            "diagnosis": "checkout crashlooping",
            "cause_class": "unknown",
            "action_type": "remediate",
            "target": "deployment/checkout",
            "command": "kubectl rollout restart deployment/checkout",
            "confidence": 0.9,
        }
    )
    return [inspect_action, remediate_action, ESCALATE_ACTION]


def test_make_episode_runner_respects_explicit_c2_dry_run_false(monkeypatch):
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "success", "data": {"result": []}})
    responses = _collection_responses()
    responses[f"kubectl get deployment/checkout -n {NAMESPACE}"] = _ok("deployment details")
    responses["kubectl rollout restart deployment/checkout"] = _ok("restarted")
    # A real (non-dry-run) C2 also reads back readiness before the episode can report "agir"
    # (act) cleanly.
    responses["kubectl rollout status deployment/checkout"] = _ok('deployment "checkout" successfully rolled out')
    runner = FakeCommandRunner(responses)
    model = StubModel(_inspect_then_remediate_actions())
    emitted = []

    episode_runner = er.make_episode_runner(
        model=model,
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=runner,
        c2_dry_run=False,
        triage_sink=emitted.append,
    )

    episode_runner("ep-1", _webhook_raw())

    assert any(call.startswith("kubectl rollout restart") for call in runner.calls)
    record = emitted[0]
    assert record["acao_executada"] is not None
    assert record["acao_executada"]["comandos"] == ["kubectl rollout restart deployment/checkout"]
    assert record["acao_executada"]["read_back"].startswith("melhora confirmada")
    # causa_provavel/sintese/confianca must describe the SAME turn as
    # decisao="agir"/acao_executada (the remediate turn), not the escalate turn that followed it
    # once the model kept going after the C2 mutation.
    assert record["decisao"] == "agir"
    assert record["causa_provavel"]["detalhe"] == "checkout crashlooping"
    assert record["sintese"] == "checkout crashlooping"
    assert record["confianca"] == 0.9
    # rollback_state travels on the record too, describing the same real
    # C2 execution (restart is idempotent, so no executable rollback command).
    assert record["rollback_state"]["tipo"] == "rollout_restart"
    assert record["rollback_state"]["comando_rollback"] is None
    assert "motivo_escalacao_motivo" not in record
    assert record["motivo_escalacao"] == []


def test_make_episode_runner_defaults_c2_dry_run_from_env(monkeypatch):
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "success", "data": {"result": []}})
    monkeypatch.setenv(er.C2_DRY_RUN_ENV_VAR, "0")
    responses = _collection_responses()
    responses[f"kubectl get deployment/checkout -n {NAMESPACE}"] = _ok("deployment details")
    responses["kubectl rollout restart deployment/checkout"] = _ok("restarted")
    responses["kubectl rollout status deployment/checkout"] = _ok('deployment "checkout" successfully rolled out')
    runner = FakeCommandRunner(responses)
    model = StubModel(_inspect_then_remediate_actions())
    emitted = []

    episode_runner = er.make_episode_runner(
        model=model, namespace=NAMESPACE, prometheus_url=PROMETHEUS_URL, command_runner=runner, triage_sink=emitted.append
    )
    episode_runner("ep-1", _webhook_raw())

    # SSLM_C2_DRY_RUN=0 read from the environment (no explicit c2_dry_run kwarg above) turned the
    # gate off, same as passing c2_dry_run=False explicitly.
    assert any(call.startswith("kubectl rollout restart") for call in runner.calls)
    assert emitted[0]["acao_executada"] is not None
    # Same fix as test_make_episode_runner_respects_explicit_c2_dry_run_false above.
    assert emitted[0]["decisao"] == "agir"
    assert emitted[0]["causa_provavel"]["detalhe"] == "checkout crashlooping"
    assert emitted[0]["confianca"] == 0.9
    assert emitted[0]["rollback_state"]["tipo"] == "rollout_restart"


def test_make_episode_runner_swallows_orchestrator_exceptions(caplog):
    class ExplodingModel:
        def generate(self, prompt: str, *, schema_constrained: bool) -> str:
            raise RuntimeError("model backend unreachable")

    def exploding_command_runner(command: str, *, timeout: float) -> er.CommandResult:
        raise AssertionError("should never be reached: collection itself should not blow up here")

    episode_runner = er.make_episode_runner(
        model=ExplodingModel(),
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=FakeCommandRunner(_collection_responses()),
    )

    with caplog.at_level("ERROR", logger="sslm.episode_runner"):
        result = episode_runner("ep-1", _webhook_raw())

    assert result is None  # never propagates
    assert any("orchestrator raised" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# normalize_command_for_loop_detection -- E4's "normalized command".
# ---------------------------------------------------------------------------


def test_normalize_command_collapses_whitespace():
    assert er.normalize_command_for_loop_detection("kubectl  get   pod/checkout-1  -n  testns") == (
        "kubectl get pod/checkout-1 -n testns"
    )


@pytest.mark.parametrize(
    "command,expected",
    [
        ("kubectl get pod/checkout-7d9f6c8b47-x8z2p -n testns", "kubectl get pod/checkout-<pod> -n testns"),
        ("kubectl logs checkout-7d9f6c8b47-x8z2p -n testns", "kubectl logs checkout-<pod> -n testns"),
        # a plain deployment/namespace name (no two-hash-segment suffix) is left alone.
        ("kubectl get deployment/checkout -n testns", "kubectl get deployment/checkout -n testns"),
        # only one hash segment (a ReplicaSet name, not a pod name): not canonicalized -- the
        # regex requires the two-segment shape a POD name carries, erring toward under-, not
        # over-, canonicalization (module docstring).
        ("kubectl get replicaset/checkout-7d9f6c8b47 -n testns", "kubectl get replicaset/checkout-7d9f6c8b47 -n testns"),
    ],
)
def test_normalize_command_canonicalizes_pod_hash_suffix(command, expected):
    assert er.normalize_command_for_loop_detection(command) == expected


def test_normalize_command_unparseable_returns_stripped():
    assert er.normalize_command_for_loop_detection('  kubectl exec -- cat "unterminated  ') == (
        'kubectl exec -- cat "unterminated'
    )


# ---------------------------------------------------------------------------
# LiveClusterEnvironment.apply_action -- E4 loop detection's runtime effect.
# ---------------------------------------------------------------------------


def test_e4_blocks_third_emission_of_a_repeated_c0_failure():
    command = f"kubectl get pod/ghost -n {NAMESPACE}"
    runner = FakeCommandRunner({command: _fail("NotFound", returncode=1)})
    env = _environment(command_runner=runner)
    action = {
        "action_type": "inspect",
        "diagnosis": "checking a resource that does not exist",
        "cause_class": "unknown",
        "target": "pod/ghost",
        "command": command,
        "confidence": 0.5,
    }

    solved1, obs1 = env.apply_action(action)
    solved2, obs2 = env.apply_action(action)
    solved3, obs3 = env.apply_action(action)

    assert (solved1, solved2, solved3) == (False, False, False)
    assert "exit=1" in obs1 and "exit=1" in obs2  # both really executed
    assert obs3.startswith("BLOCKED (E4 loop detected)")
    assert runner.calls == [command, command]  # the 3rd emission never reached the command runner
    assert len(env.loop_detections) == 1
    assert env.loop_detections[0]["normalized_command"] == command
    # the blocked emission never counts against the C0 cap or gets added to denials/executions --
    # it is a third, separate outcome (module docstring).
    assert env._state.c0_calls == 2
    assert env.denials == []
    assert len(env.executions) == 2


def test_e4_blocks_third_emission_of_a_repeated_c3_denial():
    command = "kubectl apply -f recommendation-pvc.yaml"
    runner = FakeCommandRunner()
    env = _environment(command_runner=runner)
    action = {
        "action_type": "remediate",
        "diagnosis": "bad yaml",
        "cause_class": "unknown",
        "target": "deployment/checkout",
        "command": command,
        "confidence": 0.9,
    }

    env.apply_action(action)
    env.apply_action(action)
    solved3, obs3 = env.apply_action(action)

    assert obs3.startswith("BLOCKED (E4 loop detected)")
    assert runner.calls == []  # C3 never reaches the runner regardless
    assert len(env.denials) == 2  # only the first two denials were classified at all
    assert len(env.loop_detections) == 1


def test_e4_uses_normalized_command_across_pod_reschedules():
    """The canonicalization half of E4: two LITERALLY different commands (different pod-suffix
    hash, simulating a reschedule) that normalize to the same thing and fail identically must
    still be recognized as the same loop."""
    command_a = f"kubectl get pod/checkout-7d9f6c8b47-x8z2p -n {NAMESPACE}"
    command_b = f"kubectl get pod/checkout-7d9f6c8b47-qwer7 -n {NAMESPACE}"
    command_c = f"kubectl get pod/checkout-7d9f6c8b47-zzzzz -n {NAMESPACE}"
    runner = FakeCommandRunner(
        {
            command_a: _fail("NotFound", returncode=1),
            command_b: _fail("NotFound", returncode=1),
            command_c: _fail("NotFound", returncode=1),
        }
    )
    env = _environment(command_runner=runner)

    def _inspect(command: str):
        return env.apply_action(
            {
                "action_type": "inspect",
                "diagnosis": "checking after reschedule",
                "cause_class": "unknown",
                "target": "pod/checkout",
                "command": command,
                "confidence": 0.5,
            }
        )

    _inspect(command_a)
    _inspect(command_b)
    solved3, obs3 = _inspect(command_c)

    assert obs3.startswith("BLOCKED (E4 loop detected)")
    assert runner.calls == [command_a, command_b]


def test_e4_does_not_block_when_the_error_changes_between_attempts():
    command = f"kubectl get pod/checkout-1 -n {NAMESPACE}"
    runner = FakeCommandRunner()
    env = _environment(command_runner=runner)
    action = {
        "action_type": "inspect",
        "diagnosis": "checking pod",
        "cause_class": "unknown",
        "target": "pod/checkout-1",
        "command": command,
        "confidence": 0.5,
    }

    # Same normalized command, but a DIFFERENT failure each time (a flaky apiserver, not a loop)
    # -- E4 requires the SAME error twice in a row, so the 3rd emission must still run for real.
    runner.responses = {command: _fail("NotFound", returncode=1)}
    env.apply_action(action)
    runner.responses = {command: _fail("connection refused", returncode=2)}
    env.apply_action(action)
    runner.responses = {command: _ok("pod details")}
    solved3, obs3 = env.apply_action(action)

    assert not obs3.startswith("BLOCKED")
    assert "exit=0" in obs3
    assert env.loop_detections == []
    assert runner.calls == [command, command, command]


def test_e4_does_not_track_c2_dry_run_repeats():
    """Design decision (LiveClusterEnvironment's own docstring, C2 dry-run gate paragraph): a
    dry-run is a deliberate, always-identical gate response, not a failure -- repeating it must
    never trip E4, or E4 would fight the rollback rule the dry-run gate exists to enforce. Only
    2 attempts here (not 3): the envelope's own C2_CALL_CAP is 2/episode, so a 3rd attempt would
    hit that cap (a real, separate denial, not a dry-run) before E4 ever gets a
    chance to matter -- this test is scoped to what E4 does within the cap, the cap itself is
    covered elsewhere (test_c2_action_requires_read_before_mutate_and_confidence and friends)."""
    command = "kubectl rollout restart deployment/checkout"
    runner = FakeCommandRunner({command: _ok("restarted")})
    env = _environment_with_dry_run(c2_dry_run=True, targets_seen={"deployment/checkout"}, command_runner=runner)
    action = {
        "action_type": "remediate",
        "diagnosis": "checkout is overloaded",
        "cause_class": "unknown",
        "target": "deployment/checkout",
        "command": command,
        "confidence": 0.9,
    }

    _, obs1 = env.apply_action(action)
    _, obs2 = env.apply_action(action)

    assert obs1.startswith("DRY-RUN") and obs2.startswith("DRY-RUN")
    assert env.loop_detections == []
    assert runner.calls == []  # dry-run never reaches the command runner regardless


# ---------------------------------------------------------------------------
# max_elapsed_seconds threading -- E5's elapsed-time half.
# ---------------------------------------------------------------------------


def test_make_episode_runner_threads_max_elapsed_seconds_into_e5(monkeypatch):
    """max_elapsed_seconds=0.0 trips the time cap on the very first turn, before the model is
    ever asked for output -- StubModel([]) would raise StopIteration if `generate` were somehow
    still called, so this also proves the cap is enforced BEFORE prompting, not after."""
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "success", "data": {"result": []}})
    runner = FakeCommandRunner(_collection_responses())
    model = StubModel([])
    emitted = []

    episode_runner = er.make_episode_runner(
        model=model,
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=runner,
        max_elapsed_seconds=0.0,
        triage_sink=emitted.append,
    )

    episode_runner("ep-1", _webhook_raw())

    assert len(emitted) == 1
    record = emitted[0]
    assert record["decisao"] == "escalar"
    assert record["motivo_escalacao"] == ["E5"]


# ---------------------------------------------------------------------------
# rollback_state pre-computation + read-back + rollback-on-no-improvement (the hard rollback
# rule). LiveClusterEnvironment level, c2_dry_run=False throughout: the dry-run gate's own
# behavior is already covered above.
# ---------------------------------------------------------------------------


def _environment_for_rollback(
    *,
    targets_seen: set[str] | None = None,
    command_runner: FakeCommandRunner | None = None,
    sleep_fn=None,
    read_back_window_seconds: float | None = None,
    scale_bounds: dict[str, tuple[int, int]] | None = None,
) -> er.LiveClusterEnvironment:
    context = er.EpisodeContext(episode_id="ep-1", namespace=NAMESPACE)
    state = EpisodeEnvelopeState(
        targets_seen=set(targets_seen or set()), confidence=0.9, scale_bounds=scale_bounds or {}
    )
    kwargs: dict = {}
    if sleep_fn is not None:
        kwargs["sleep_fn"] = sleep_fn
    if read_back_window_seconds is not None:
        kwargs["read_back_window_seconds"] = read_back_window_seconds
    return er.LiveClusterEnvironment(
        context=context,
        state=state,
        command_runner=command_runner or FakeCommandRunner(),
        c2_dry_run=False,
        **kwargs,
    )


def _remediate(command: str, target: str) -> dict:
    return {
        "action_type": "remediate",
        "diagnosis": "acting on it",
        "cause_class": "unknown",
        "target": target,
        "command": command,
        "confidence": 0.9,
    }


# --- _parse_current_revision_from_rollout_history ---------------------------------------------


def test_parse_current_revision_from_rollout_history_picks_the_highest():
    output = "deployment.apps/checkout\nREVISION  CHANGE-CAUSE\n1         <none>\n2         <none>\n3         <none>\n"
    assert er._parse_current_revision_from_rollout_history(output) == 3


def test_parse_current_revision_from_rollout_history_none_when_unparseable():
    assert er._parse_current_revision_from_rollout_history("deployment.apps/checkout\nno revisions here\n") is None


# --- _controller_owner_reference -----------------------------------------------------------


def test_controller_owner_reference_prefers_the_controller_true_entry():
    resource = {
        "metadata": {
            "ownerReferences": [
                {"kind": "OtherThing", "name": "not-the-controller", "controller": False},
                {"kind": "ReplicaSet", "name": "checkout-7d9f6c8b47", "controller": True},
            ]
        }
    }
    assert er._controller_owner_reference(resource) == ("ReplicaSet", "checkout-7d9f6c8b47")


def test_controller_owner_reference_none_without_owners():
    assert er._controller_owner_reference({"metadata": {}}) is None


# --- rollback_state precompute: rollout restart (idempotent, no read needed) ------------------


def test_rollback_state_for_restart_is_idempotent_with_no_read():
    runner = FakeCommandRunner(
        {
            "kubectl rollout restart deployment/checkout": _ok("restarted"),
            "kubectl rollout status deployment/checkout": _ok('deployment "checkout" successfully rolled out'),
        }
    )
    env = _environment_for_rollback(targets_seen={"deployment/checkout"}, command_runner=runner)

    solved, observation = env.apply_action(_remediate("kubectl rollout restart deployment/checkout", "deployment/checkout"))

    assert solved is False
    assert "DENIED" not in observation
    execution = env.executions[0]
    assert execution["rollback_state"]["tipo"] == "rollout_restart"
    assert execution["rollback_state"]["comando_rollback"] is None
    # restart's precompute makes no read at all: the FIRST call is the mutation itself, the
    # only OTHER call is the read-back (kubectl rollout status).
    assert runner.calls[0] == "kubectl rollout restart deployment/checkout"


# --- rollback_state precompute: rollout undo ---------------------------------------------------


def test_rollback_state_for_undo_reads_the_current_revision_first():
    responses = {
        f"kubectl rollout history deployment/checkout -n {NAMESPACE}": _ok(
            "deployment.apps/checkout\nREVISION  CHANGE-CAUSE\n1         <none>\n2         <none>\n"
        ),
        "kubectl rollout undo deployment/checkout --to-revision=3": _ok("rolled back"),
        "kubectl rollout status deployment/checkout": _ok("rollout complete"),
    }
    runner = FakeCommandRunner(responses)
    env = _environment_for_rollback(targets_seen={"deployment/checkout"}, command_runner=runner)

    solved, observation = env.apply_action(
        _remediate("kubectl rollout undo deployment/checkout --to-revision=3", "deployment/checkout")
    )

    assert solved is False
    assert "DENIED" not in observation
    rollback_state = env.executions[0]["rollback_state"]
    assert rollback_state["tipo"] == "rollout_undo"
    assert rollback_state["estado_pre_mutacao"] == "revisao vigente antes do undo: 2"
    assert rollback_state["comando_rollback"] == f"kubectl rollout undo deployment/checkout -n {NAMESPACE} --to-revision=2"
    # the history read happened BEFORE the mutation.
    assert runner.calls[0].startswith("kubectl rollout history")
    assert runner.calls[1] == "kubectl rollout undo deployment/checkout --to-revision=3"


def test_rollback_state_for_undo_blocks_execution_when_history_read_fails():
    """Hard rule: a C2 without a rollback_state does not run. A failed rollout history
    read must deny the C2 outright, not let it run without a registered rollback."""
    responses = {
        f"kubectl rollout history deployment/checkout -n {NAMESPACE}": _fail("connection refused"),
        "kubectl rollout undo deployment/checkout --to-revision=3": _ok("rolled back"),
    }
    runner = FakeCommandRunner(responses)
    env = _environment_for_rollback(targets_seen={"deployment/checkout"}, command_runner=runner)

    solved, observation = env.apply_action(
        _remediate("kubectl rollout undo deployment/checkout --to-revision=3", "deployment/checkout")
    )

    assert solved is False
    assert observation.startswith("DENIED (C2): rollback_state precheck failed")
    assert "kubectl rollout undo deployment/checkout --to-revision=3" not in runner.calls  # the mutation never ran
    assert len(env.denials) == 1
    assert env.denials[0].action_class == ActionClass.C2
    assert env.executions == []


# --- rollback_state precompute: delete pod ------------------------------------------------------


POD_JSON_WITH_RS_OWNER = {
    "metadata": {
        "name": "checkout-2",
        "ownerReferences": [{"kind": "ReplicaSet", "name": "checkout-7d9f6c8b47", "controller": True}],
    }
}
REPLICASET_JSON_WITH_DEPLOYMENT_OWNER = {
    "metadata": {"ownerReferences": [{"kind": "Deployment", "name": "checkout", "controller": True}]}
}


def test_rollback_state_for_delete_pod_annotates_the_owning_replicaset_and_resolves_deployment():
    responses = {
        f"kubectl get pod checkout-2 -n {NAMESPACE} -o json": _ok(json.dumps(POD_JSON_WITH_RS_OWNER)),
        f"kubectl get replicaset checkout-7d9f6c8b47 -n {NAMESPACE} -o json": _ok(
            json.dumps(REPLICASET_JSON_WITH_DEPLOYMENT_OWNER)
        ),
        "kubectl delete pod checkout-2": _ok("pod deleted"),
        "kubectl rollout status deployment/checkout": _ok("rollout complete"),
    }
    runner = FakeCommandRunner(responses)
    env = _environment_for_rollback(targets_seen={"pod/checkout-2"}, command_runner=runner)

    solved, observation = env.apply_action(_remediate("kubectl delete pod checkout-2", "pod/checkout-2"))

    assert solved is False
    assert "DENIED" not in observation
    rollback_state = env.executions[0]["rollback_state"]
    assert rollback_state["tipo"] == "delete_pod"
    assert "ReplicaSet/checkout-7d9f6c8b47" in rollback_state["estado_pre_mutacao"]
    # delete pod has no executable reversal: informative/diagnostic only.
    assert rollback_state["comando_rollback"] is None
    # the Deployment resolved from the ReplicaSet's own owner is what read-back polled.
    assert any(call.startswith("kubectl rollout status deployment/checkout") for call in runner.calls)


def test_rollback_state_for_delete_pod_blocks_execution_when_pod_read_fails():
    responses = {f"kubectl get pod checkout-2 -n {NAMESPACE} -o json": _fail("NotFound")}
    runner = FakeCommandRunner(responses)
    env = _environment_for_rollback(targets_seen={"pod/checkout-2"}, command_runner=runner)

    solved, observation = env.apply_action(_remediate("kubectl delete pod checkout-2", "pod/checkout-2"))

    assert observation.startswith("DENIED (C2): rollback_state precheck failed")
    assert "kubectl delete pod checkout-2" not in runner.calls
    assert env.executions == []


def test_rollback_state_for_delete_pod_read_back_is_inconclusive_without_a_resolvable_deployment():
    """Honest limit (class docstring): when the ReplicaSet's own owner cannot be resolved, delete
    pod's rollback_state is still valid (the rule only requires the ReplicaSet name), but
    read-back cannot poll anything -- it is None (inconclusive), never a fabricated verdict."""
    pod_json = {
        "metadata": {
            "name": "checkout-2",
            "ownerReferences": [{"kind": "ReplicaSet", "name": "orphan-rs", "controller": True}],
        }
    }
    responses = {
        f"kubectl get pod checkout-2 -n {NAMESPACE} -o json": _ok(json.dumps(pod_json)),
        f"kubectl get replicaset orphan-rs -n {NAMESPACE} -o json": _fail("NotFound"),
        "kubectl delete pod checkout-2": _ok("pod deleted"),
    }
    runner = FakeCommandRunner(responses)
    env = _environment_for_rollback(targets_seen={"pod/checkout-2"}, command_runner=runner)

    solved, observation = env.apply_action(_remediate("kubectl delete pod checkout-2", "pod/checkout-2"))

    assert "DENIED" not in observation  # the C2 still executes: rollback_state itself is complete
    execution = env.executions[0]
    assert execution["rollback_state"]["tipo"] == "delete_pod"
    assert execution["read_back"]["ready"] is None
    assert env.rollback_events == [{"command": "kubectl delete pod checkout-2", "improved": None, "rollback_executed": False}]


# --- rollback_state precompute: scale ------------------------------------------------------------


def test_rollback_state_for_scale_reads_current_replicas_first():
    responses = {
        f"kubectl get deployment/checkout -n {NAMESPACE} -o json": _ok(json.dumps({"spec": {"replicas": 2}})),
        "kubectl scale deployment/checkout --replicas=4": _ok("scaled"),
        "kubectl rollout status deployment/checkout": _ok("rollout complete"),
    }
    runner = FakeCommandRunner(responses)
    state = EpisodeEnvelopeState(targets_seen={"deployment/checkout"}, confidence=0.9, scale_bounds={"deployment/checkout": (1, 6)})
    env = er.LiveClusterEnvironment(
        context=er.EpisodeContext(episode_id="ep-1", namespace=NAMESPACE),
        state=state,
        command_runner=runner,
        c2_dry_run=False,
    )

    solved, observation = env.apply_action(_remediate("kubectl scale deployment/checkout --replicas=4", "deployment/checkout"))

    assert solved is False
    assert "DENIED" not in observation
    rollback_state = env.executions[0]["rollback_state"]
    assert rollback_state["tipo"] == "scale"
    assert rollback_state["estado_pre_mutacao"] == "replicas antes da mutacao: 2"
    assert rollback_state["comando_rollback"] == f"kubectl scale deployment/checkout -n {NAMESPACE} --replicas=2"
    assert runner.calls[0] == f"kubectl get deployment/checkout -n {NAMESPACE} -o json"


def test_rollback_state_for_scale_blocks_execution_when_replicas_read_fails():
    responses = {f"kubectl get deployment/checkout -n {NAMESPACE} -o json": _fail("NotFound")}
    runner = FakeCommandRunner(responses)
    state = EpisodeEnvelopeState(targets_seen={"deployment/checkout"}, confidence=0.9, scale_bounds={"deployment/checkout": (1, 6)})
    env = er.LiveClusterEnvironment(
        context=er.EpisodeContext(episode_id="ep-1", namespace=NAMESPACE),
        state=state,
        command_runner=runner,
        c2_dry_run=False,
    )

    solved, observation = env.apply_action(_remediate("kubectl scale deployment/checkout --replicas=4", "deployment/checkout"))

    assert observation.startswith("DENIED (C2): rollback_state precheck failed")
    assert "kubectl scale deployment/checkout --replicas=4" not in runner.calls


# --- read-back: no improvement executes the registered rollback --------------------------------


def test_read_back_no_improvement_executes_the_registered_rollback_for_undo():
    """`read_back_window_seconds=0.0` makes the window exhausted before any `kubectl rollout
    status` attempt (deterministic, no fake clock arithmetic needed: `deadline = clock() + 0`
    means `remaining <= 0` on the very first check) -> "not improved" -> the registered
    undo-of-the-undo runs for real, and the episode's rollback_events records no improvement."""
    responses = {
        f"kubectl rollout history deployment/checkout -n {NAMESPACE}": _ok(
            "deployment.apps/checkout\nREVISION  CHANGE-CAUSE\n1         <none>\n2         <none>\n"
        ),
        "kubectl rollout undo deployment/checkout --to-revision=3": _ok("rolled back"),
        f"kubectl rollout undo deployment/checkout -n {NAMESPACE} --to-revision=2": _ok("rolled back to 2"),
    }
    runner = FakeCommandRunner(responses)
    env = _environment_for_rollback(
        targets_seen={"deployment/checkout"}, command_runner=runner, read_back_window_seconds=0.0
    )

    solved, observation = env.apply_action(
        _remediate("kubectl rollout undo deployment/checkout --to-revision=3", "deployment/checkout")
    )

    assert solved is False
    assert "rollback executed" in observation
    rollback_command = f"kubectl rollout undo deployment/checkout -n {NAMESPACE} --to-revision=2"
    assert rollback_command in runner.calls
    assert not any(call.startswith("kubectl rollout status") for call in runner.calls)  # window exhausted before any attempt
    execution = env.executions[0]
    assert execution["read_back"]["ready"] is False
    assert execution["rollback_executed"] is True
    assert env.rollback_events == [
        {"command": "kubectl rollout undo deployment/checkout --to-revision=3", "improved": False, "rollback_executed": True}
    ]


def test_read_back_no_improvement_does_not_execute_a_rollback_for_restart():
    """restart has no executable rollback command (idempotent) -- no-improvement still records the
    rollback_event (feeds E3), but there is nothing to execute."""
    responses = {"kubectl rollout restart deployment/checkout": _ok("restarted")}
    runner = FakeCommandRunner(responses)
    env = _environment_for_rollback(
        targets_seen={"deployment/checkout"}, command_runner=runner, read_back_window_seconds=0.0
    )

    solved, observation = env.apply_action(_remediate("kubectl rollout restart deployment/checkout", "deployment/checkout"))

    assert "rollback executed" not in observation
    assert env.executions[0]["rollback_executed"] is False
    assert env.rollback_events == [
        {"command": "kubectl rollout restart deployment/checkout", "improved": False, "rollback_executed": False}
    ]


def test_read_back_improvement_within_the_window_does_not_wait_it_out():
    """The FakeCommandRunner is configured to succeed on the very first `kubectl rollout status`
    call -- `_perform_read_back` must return immediately (True), never polling/sleeping further."""
    responses = {
        "kubectl rollout restart deployment/checkout": _ok("restarted"),
        "kubectl rollout status deployment/checkout": _ok('deployment "checkout" successfully rolled out'),
    }
    runner = FakeCommandRunner(responses)

    def _never_sleep(seconds):
        raise AssertionError("read-back should not sleep when the first attempt already succeeds")

    env = _environment_for_rollback(
        targets_seen={"deployment/checkout"}, command_runner=runner, sleep_fn=_never_sleep
    )

    solved, observation = env.apply_action(_remediate("kubectl rollout restart deployment/checkout", "deployment/checkout"))

    assert "melhora" not in observation  # observation text lives in triage_assembly, not here
    assert env.executions[0]["read_back"]["ready"] is True
    assert env.rollback_events[0]["improved"] is True


# --- mutation command itself fails: read-back/rollback are skipped entirely --------------------


def test_mutation_command_failure_skips_read_back_and_rollback():
    runner = FakeCommandRunner({"kubectl rollout restart deployment/checkout": _fail("connection refused")})
    env = _environment_for_rollback(targets_seen={"deployment/checkout"}, command_runner=runner)

    solved, observation = env.apply_action(_remediate("kubectl rollout restart deployment/checkout", "deployment/checkout"))

    assert solved is False
    assert "exit=" in observation
    assert "read-back" not in observation
    execution = env.executions[0]
    assert execution["read_back"] is None
    assert env.rollback_events == []  # never even attempted: nothing to read back


# --- E3 signal across a whole episode (2 real C2 executions, neither improves) ------------------


def test_e3_signal_after_two_no_improvement_real_c2_executions_at_environment_level():
    """Two real C2 actions in one episode (the envelope's own cap, C2_CALL_CAP=2/episode), both
    executed for real, neither confirmed by read-back -> `rollback_events` carries two
    no-improvement entries, which is exactly what `harness/triage_assembly.py`'s `_e3_triggered`
    reads to fold "E3" into `motivo_escalacao` (also unit-tested directly there)."""
    responses = {
        "kubectl rollout restart deployment/checkout": _ok("restarted"),
        f"kubectl get deployment/checkout -n {NAMESPACE} -o json": _ok(json.dumps({"spec": {"replicas": 2}})),
        "kubectl scale deployment/checkout --replicas=4": _ok("scaled"),
        f"kubectl scale deployment/checkout -n {NAMESPACE} --replicas=2": _ok("scaled back"),
    }
    runner = FakeCommandRunner(responses)
    env = _environment_for_rollback(
        targets_seen={"deployment/checkout"},
        command_runner=runner,
        read_back_window_seconds=0.0,
        scale_bounds={"deployment/checkout": (1, 6)},
    )

    env.apply_action(_remediate("kubectl rollout restart deployment/checkout", "deployment/checkout"))
    env.apply_action(_remediate("kubectl scale deployment/checkout --replicas=4", "deployment/checkout"))

    assert len(env.rollback_events) == 2
    assert all(e["improved"] is False for e in env.rollback_events)
    # feeds harness/triage_assembly.py's E3 directly (unit-tested there too).
    from harness.triage_assembly import _e3_triggered

    assert _e3_triggered(env) is True


# ---------------------------------------------------------------------------
# JsonlTriageSink
# ---------------------------------------------------------------------------


def test_jsonl_triage_sink_appends_one_line_per_record(tmp_path):
    from harness.episode_runner import JsonlTriageSink

    path = tmp_path / "triage.jsonl"
    sink = JsonlTriageSink(path)
    sink({"episode_id": "ep-1", "decisao": "escalar"})
    sink({"episode_id": "ep-2", "decisao": "agir"})

    lines = path.read_text().splitlines()
    assert [json.loads(line)["episode_id"] for line in lines] == ["ep-1", "ep-2"]


def test_jsonl_triage_sink_keeps_lines_whole_under_concurrent_episodes(tmp_path):
    """One sink is shared by every concurrent /alerts episode; interleaved writes would corrupt
    the JSONL for the whole round."""
    import threading

    from harness.episode_runner import JsonlTriageSink

    path = tmp_path / "triage.jsonl"
    sink = JsonlTriageSink(path)
    padding = "x" * 20_000  # large enough that an unlocked write would not be atomic
    threads = [
        threading.Thread(target=sink, args=({"episode_id": f"ep-{i}", "detalhe": padding},))
        for i in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ids = sorted(json.loads(line)["episode_id"] for line in path.read_text().splitlines())
    assert ids == sorted(f"ep-{i}" for i in range(16))


def test_jsonl_triage_sink_also_logs_the_record_and_never_raises_on_io_error(tmp_path, caplog):
    """The log line is the fallback copy: when the append fails the record must still be there,
    and the failure must not escape into the episode handler."""
    from harness.episode_runner import JsonlTriageSink

    sink = JsonlTriageSink(tmp_path / "missing-dir" / "triage.jsonl")
    with caplog.at_level("INFO", logger="sslm.episode_runner"):
        sink({"episode_id": "ep-lost"})  # does not raise

    assert any("triage_record" in r.getMessage() and "ep-lost" in r.getMessage() for r in caplog.records)
    assert any("could not append" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Per-episode token usage
# ---------------------------------------------------------------------------


class CountingModel(StubModel):
    """StubModel plus the take_thread_stats contract VllmModelClient implements."""

    def __init__(self, outputs: list[str]):
        super().__init__(outputs)
        from harness.metrics import GenerateStats

        self._stats = GenerateStats()

    def generate(self, prompt: str, *, schema_constrained: bool) -> str:
        self._stats.calls += 1
        self._stats.prompt_tokens += 100
        self._stats.completion_tokens += 10
        return super().generate(prompt, schema_constrained=schema_constrained)

    def take_thread_stats(self):
        from harness.metrics import GenerateStats

        stats, self._stats = self._stats, GenerateStats()
        return stats


def _usage_runner(model, usage, monkeypatch):
    monkeypatch.setattr(er, "_http_get_json", lambda url, *, timeout: {"status": "success", "data": {"result": []}})
    return er.make_episode_runner(
        model=model,
        namespace=NAMESPACE,
        prometheus_url=PROMETHEUS_URL,
        command_runner=FakeCommandRunner(_collection_responses()),
        triage_sink=lambda record: None,
        usage_sink=usage.append,
    )


def test_every_episode_writes_its_own_token_usage_and_drops_counts_from_before_it(monkeypatch):
    model = CountingModel([ESCALATE_ACTION])
    model._stats.prompt_tokens = 9_999  # counted on this thread before the episode: not its cost
    usage = []

    _usage_runner(model, usage, monkeypatch)("ep-1", _webhook_raw())

    assert len(usage) == 1
    line = usage[0]
    assert line["episode_id"] == "ep-1"
    assert (line["prompt_tokens"], line["completion_tokens"], line["calls"]) == (100, 10, 1)
    assert line["calls_without_usage"] == 0
    assert line["t_start"] <= line["t_end"]
    from harness.metrics import EpisodeTokenUsage

    EpisodeTokenUsage(**line)  # the sidecar line is exactly what the cost slice loads


def test_token_usage_is_written_even_when_the_orchestrator_raises(monkeypatch):
    class ExplodingCountingModel(CountingModel):
        def generate(self, prompt: str, *, schema_constrained: bool) -> str:
            raise RuntimeError("model backend unreachable")

    usage = []
    _usage_runner(ExplodingCountingModel([]), usage, monkeypatch)("ep-1", _webhook_raw())

    assert [u["episode_id"] for u in usage] == ["ep-1"]
    assert usage[0]["calls"] == 0


def test_model_without_thread_stats_writes_no_usage(monkeypatch):
    usage = []
    _usage_runner(StubModel([ESCALATE_ACTION]), usage, monkeypatch)("ep-1", _webhook_raw())
    assert usage == []


def test_usage_log_sits_next_to_the_triage_log_and_the_sink_never_raises(tmp_path, caplog):
    from harness.episode_runner import JsonlUsageSink, usage_log_path

    assert usage_log_path(tmp_path / "triage.jsonl") == str(tmp_path / "triage.jsonl") + ".usage.jsonl"

    sink = JsonlUsageSink(tmp_path / "u.jsonl")
    sink({"episode_id": "ep-1"})
    sink({"episode_id": "ep-2"})
    assert [json.loads(l)["episode_id"] for l in (tmp_path / "u.jsonl").read_text().splitlines()] == ["ep-1", "ep-2"]

    with caplog.at_level("ERROR", logger="sslm.episode_runner"):
        JsonlUsageSink(tmp_path / "missing-dir" / "u.jsonl")({"episode_id": "ep-lost"})
    assert any("stays unattributed" in r.getMessage() for r in caplog.records)


# The scripted context under a char budget.


def _context_with(*texts, missing=False):
    context = er.EpisodeContext(episode_id="ep-1", namespace=NAMESPACE)
    for i, text in enumerate(texts):
        context.sources.append(er.ScriptedSource(f"src{i}", True, text=text))
    if missing:
        context.sources.append(er.ScriptedSource("k8s_logs", False, "skipped: test"))
    return context


def test_context_without_budget_renders_byte_for_byte_as_before():
    context = _context_with("a" * 12_000, "b" * 300, missing=True)
    assert context.as_prompt_text() == context.as_prompt_text(None)
    assert "a" * 12_000 in context.as_prompt_text()


def test_context_budget_keeps_every_source_and_small_ones_whole():
    # The measured shape: one big pods dump next to short sources (webhook, events).
    context = _context_with("p" * 12_000, "w" * 800, "e" * 12_000, "l" * 300, missing=True)
    text = context.as_prompt_text(9_000)

    for i in range(4):
        assert f"--- src{i} ---" in text
    assert "--- k8s_logs (NOT collected: skipped: test) ---" in text
    assert "w" * 800 in text and "l" * 300 in text
    # 9_000 - 800 - 300 = 7_900 left for the two dumps, 3_950 each, marker included, cut head+tail.
    body = text.split("--- src0 ---\n")[1].split("\n\n--- src1")[0]
    assert body[:1_900] == "p" * 1_900 and body[-1_900:] == "p" * 1_900
    assert "chars omitted" in body and len(body) <= 3_950
    assert ("e" * 1_900) in text and ("e" * 12_000) not in text


def test_context_budget_bounds_the_sum_of_the_source_text():
    texts = ["x" * n for n in (12_000, 12_000, 12_000, 5_000, 40, 0)]
    out = er._split_char_budget(dict(enumerate(texts)), 9_000)
    assert sum(len(t) for t in out.values()) <= 9_000  # cut markers count
    assert out[4] == "x" * 40 and out[5] == ""


def test_live_environment_passes_the_context_budget_to_the_task_description():
    context = _context_with("p" * 12_000)
    env = er.LiveClusterEnvironment(
        context=context,
        state=EpisodeEnvelopeState(),
        command_runner=lambda *a, **k: None,
        context_max_chars=4_000,
    )
    assert env.task_description() == context.as_prompt_text(4_000)
    assert "p" * 12_000 not in env.task_description()


def test_agent_config_prompt_budget_reaches_the_environment(monkeypatch):
    from harness.agent import PromptBudget

    seen = {}
    real = er.LiveClusterEnvironment

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(er, "LiveClusterEnvironment", spy)
    monkeypatch.setattr(
        er, "collect_episode_context", lambda *a, **k: (er.EpisodeContext(episode_id="ep-1", namespace=NAMESPACE), set())
    )

    class Escalates:
        def generate(self, prompt, *, schema_constrained):
            return json.dumps(
                {
                    "diagnosis": "d",
                    "cause_class": "unknown",
                    "action_type": "escalate",
                    "target": "t",
                    "command": None,
                    "confidence": 0.1,
                }
            )

    config = AgentConfig(schema_constrained_decoding=True, memory_enabled=False, prompt_budget=PromptBudget(18_000))
    er.run_real_episode(
        "ep-1", {}, model=Escalates(), namespace=NAMESPACE, prometheus_url="http://p", agent_config=config
    )
    assert seen["context_max_chars"] == 9_000
