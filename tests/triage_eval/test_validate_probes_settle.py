"""probe_until_measured: retry ONLY while the probe returns None (not measured), never over a verdict.

The race behind it (seen twice on revoke_auth-1): a probe right after recover_fault() catches the
mongo container being recreated ("container not found") and returns None. Re-measuring a None is
waiting for the infra to settle; re-measuring a True/False would loosen admission, and this test
locks that it never happens.
"""

from scenarios.validate_probes import probe_until_measured


def test_returns_first_measured_verdict_without_sleeping():
    sleeps = []
    result = probe_until_measured(
        lambda: {"root_cause_fixed": False, "probe_detail": "sintoma presente"},
        _sleep=sleeps.append,
    )
    assert result["root_cause_fixed"] is False
    assert sleeps == []


def test_retries_while_unmeasured_then_returns_verdict():
    sleeps = []
    outcomes = iter(
        [
            {"root_cause_fixed": None, "probe_error": "container not found"},
            {"root_cause_fixed": None, "probe_error": "container not found"},
            {"root_cause_fixed": True, "probe_detail": "role restaurado"},
        ]
    )
    result = probe_until_measured(lambda: next(outcomes), _sleep=sleeps.append)
    assert result["root_cause_fixed"] is True
    assert len(sleeps) == 2


def test_exhausted_returns_last_unmeasured_with_reason():
    sleeps = []
    result = probe_until_measured(
        lambda: {"root_cause_fixed": None, "probe_error": "container not found"},
        max_attempts=3,
        _sleep=sleeps.append,
    )
    assert result["root_cause_fixed"] is None
    assert result["probe_error"] == "container not found"
    # sleeps BETWEEN attempts, not after the last one
    assert len(sleeps) == 2
