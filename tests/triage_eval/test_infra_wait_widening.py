"""The adapter widens only the DEFAULT of AIOpsLab's infra waits, never an explicit max_wait.

Context: on a laptop kind cluster the hotelReservation deploy often takes longer than 300s
(slow openebs PVCs + mongodb), and the DEPLOY path's wait_for_ready uses the upstream default.
The misconfig_app eval uses the SAME method but passes an explicit max_wait=60, so the test
pins both ends: default widened, explicit value untouched.
"""

import adapters.aiopslab_env as aiopslab_env
from adapters.aiopslab_env import INFRA_WAIT_MAX_SEC, AiopsLabEnvironment


def _fresh_kubectl():
    from aiopslab.service.kubectl import KubeCtl

    return KubeCtl()


def test_env_init_widens_infra_wait_defaults(fake_aiopslab_home):
    AiopsLabEnvironment("misconfig_app_hotel_res-mitigation-1")
    kubectl = _fresh_kubectl()

    kubectl.wait_for_ready("test-ns")
    kubectl.wait_for_namespace_deletion("test-ns")

    assert kubectl.calls == [
        ("wait_for_ready", "test-ns", 2, INFRA_WAIT_MAX_SEC),
        ("wait_for_namespace_deletion", "test-ns", 2, INFRA_WAIT_MAX_SEC),
    ]


def test_explicit_max_wait_still_wins(fake_aiopslab_home):
    # the misconfig_app eval calls wait_for_ready(ns, sleep=5, max_wait=60): the benchmark's
    # evaluation semantics must not change with the widening
    AiopsLabEnvironment("misconfig_app_hotel_res-mitigation-1")
    kubectl = _fresh_kubectl()

    kubectl.wait_for_ready("test-ns", sleep=5, max_wait=60)

    assert kubectl.calls == [("wait_for_ready", "test-ns", 5, 60)]


def test_widening_is_idempotent(fake_aiopslab_home):
    AiopsLabEnvironment("misconfig_app_hotel_res-mitigation-1")
    aiopslab_env._widen_infra_wait_defaults()
    from aiopslab.service.kubectl import KubeCtl

    method = KubeCtl.wait_for_ready
    aiopslab_env._widen_infra_wait_defaults()
    assert KubeCtl.wait_for_ready is method

    kubectl = KubeCtl()
    kubectl.wait_for_ready("test-ns")
    assert kubectl.calls == [("wait_for_ready", "test-ns", 2, INFRA_WAIT_MAX_SEC)]
