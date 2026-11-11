import sys
import textwrap
import types
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "triage-eval"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


# mlx_lm has no Linux wheel (Apple Silicon only), so CI (ubuntu-latest) can't install it.
# adapters/mlx_model_client.py imports it at module level; test_mlx_model_client.py
# monkeypatches every real mlx_lm call it uses (mlx_load, mlx_generate, make_sampler), so a
# stub with no behavior is enough to make the import succeed on a machine without mlx_lm.
try:
    import mlx_lm  # noqa: F401
except ImportError:
    mlx_lm_stub = types.ModuleType("mlx_lm")
    mlx_lm_stub.generate = lambda *a, **k: None
    mlx_lm_stub.load = lambda *a, **k: (None, None)
    sample_utils_stub = types.ModuleType("mlx_lm.sample_utils")
    sample_utils_stub.make_sampler = lambda *a, **k: None
    mlx_lm_stub.sample_utils = sample_utils_stub
    sys.modules.setdefault("mlx_lm", mlx_lm_stub)
    sys.modules.setdefault("mlx_lm.sample_utils", sample_utils_stub)


# Shared with test_aiopslab_env.py and test_mlx_factories.py: a minimal fake `aiopslab`
# package on disk with the same shapes those modules actually import (see
# test_aiopslab_env.py's module docstring for why a fake package, not mocks, is used).
FAKE_ORCHESTRATOR_MODULE = textwrap.dedent(
    """
    from aiopslab.utils.status import SubmissionStatus


    class _FakeProblem:
        def __init__(self):
            self.eval_calls = []
            self.inject_fault_calls = 0
            self.recover_fault_calls = 0
            self.app = _FakeApp()

        def eval(self, solution, history, duration):
            self.eval_calls.append((solution, history, duration))
            return {"score": 1.0}

        def inject_fault(self):
            self.inject_fault_calls += 1

        def recover_fault(self):
            self.recover_fault_calls += 1


    class _FakeApp:
        def __init__(self):
            self.cleanup_calls = 0

        def cleanup(self):
            self.cleanup_calls += 1


    class _FakeSession:
        def __init__(self):
            self.problem = _FakeProblem()
            self.solution = {"x": 1}
            self.history = []
            self.started = False
            self.ended = False
            self.start_time = None
            self.end_time = None

        def get_duration(self):
            if self.start_time is None or self.end_time is None:
                raise TypeError("session must be ended before duration is calculated")
            return self.end_time - self.start_time

        def start(self):
            self.started = True
            self.start_time = 1.0

        def end(self):
            self.ended = True
            self.end_time = 13.5


    class _FakeProblemRegistry:
        def get_problem_instance(self, problem_id):
            return _FakeProblem()


    class Orchestrator:
        def __init__(self, results_dir=None):
            self.results_dir = results_dir
            self.session = None
            self.init_problem_calls = []
            self.ask_env_calls = []
            self.probs = _FakeProblemRegistry()

        def init_problem(self, problem_id):
            self.init_problem_calls.append(problem_id)
            self.session = _FakeSession()
            return ("task desc", "do the thing", "exec_shell, submit")

        async def ask_env(self, response_text):
            self.ask_env_calls.append(response_text)
            if "submit()" in response_text:
                return SubmissionStatus.VALID_SUBMISSION
            return f"observed: {response_text}"
    """
)

FAKE_KUBECTL_MODULE = textwrap.dedent(
    """
    class KubeCtl:
        def __init__(self):
            self.calls = []

        def wait_for_ready(self, namespace, sleep=2, max_wait=300):
            self.calls.append(("wait_for_ready", namespace, sleep, max_wait))

        def wait_for_namespace_deletion(self, namespace, sleep=2, max_wait=300):
            self.calls.append(("wait_for_namespace_deletion", namespace, sleep, max_wait))
    """
)

FAKE_STATUS_MODULE = textwrap.dedent(
    """
    from enum import Enum


    class SubmissionStatus(Enum):
        VALID_SUBMISSION = 1
        INVALID_SUBMISSION = 2
    """
)


@pytest.fixture
def fake_aiopslab_home(tmp_path, monkeypatch):
    # Mirrors the real layout: aiopslab/orchestrator/ is a PACKAGE with orchestrator.py inside
    # (and exit_cleanup_fault lives in aiopslab.orchestrator.orchestrator), because the adapter's
    # finalize imports exactly that path to unregister the atexit hook.
    pkg = tmp_path / "aiopslab"
    (pkg / "utils").mkdir(parents=True)
    (pkg / "orchestrator").mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "orchestrator" / "__init__.py").write_text(
        "from .orchestrator import Orchestrator, exit_cleanup_fault\n"
    )
    (pkg / "orchestrator" / "orchestrator.py").write_text(
        FAKE_ORCHESTRATOR_MODULE
        + textwrap.dedent(
            """

            def exit_cleanup_fault(prob):
                prob.recover_fault()
            """
        )
    )
    (pkg / "utils" / "__init__.py").write_text("")
    (pkg / "utils" / "status.py").write_text(FAKE_STATUS_MODULE)
    (pkg / "service").mkdir()
    (pkg / "service" / "__init__.py").write_text("")
    (pkg / "service" / "kubectl.py").write_text(FAKE_KUBECTL_MODULE)

    monkeypatch.setenv("AIOPSLAB_HOME", str(tmp_path))
    for mod in list(sys.modules):
        if mod == "aiopslab" or mod.startswith("aiopslab."):
            del sys.modules[mod]
    yield tmp_path
    for mod in list(sys.modules):
        if mod == "aiopslab" or mod.startswith("aiopslab."):
            del sys.modules[mod]
