"""episode_deadline / validate_one timeout: a third-party hang becomes a per-scenario error.

Reason: AIOpsLab's wrk.py has a `while True` that only exits on job_status.status.ready or
.failed; a Job that completes too fast to be observed ready loops forever (the same hang once
stalled a full matrix run for 6h27, and admission of revoke_auth-1 twice). run_full_matrix.sh
guards with an external `timeout`; admission had no ceiling. episode_deadline arms a SIGALRM
that turns the hang into an EpisodeTimeout, caught by main's except as one error line, without
blocking the run.
"""

import time

import pytest

from scenarios.validate_probes import EpisodeTimeout, episode_deadline, validate_one


def test_deadline_raises_on_hang():
    with pytest.raises(EpisodeTimeout):
        with episode_deadline(1):
            time.sleep(5)


def test_deadline_noop_for_fast_work():
    with episode_deadline(5):
        result = 1 + 1
    assert result == 2


def test_deadline_zero_disables_alarm():
    # seconds<=0 arms no alarm (used when run_full_matrix's external ceiling already applies)
    with episode_deadline(0):
        result = "ok"
    assert result == "ok"


def test_validate_one_times_out_a_hanging_environment(fake_aiopslab_home):
    class _HangingEnv:
        def __init__(self, scenario_id):
            self.scenario_id = scenario_id

        def task_description(self):
            time.sleep(30)  # simula o while True do wrk

    with pytest.raises(EpisodeTimeout):
        validate_one("revoke_auth_mongodb-mitigation-1", _HangingEnv, timeout_sec=1)
