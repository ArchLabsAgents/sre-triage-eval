"""Admission of the root-cause probes, with no model at all.

For each scenario: deploy + inject the fault (the same path as the matrix), run the probe with
nobody touching the cluster (expected: root_cause_fixed=False), call AIOpsLab's own
recover_fault() and run the probe again (expected: True), then do the normal teardown through
finalize(). A probe is admitted into the matrix only if it passes at both ends: False with the
fault and True after the benchmark's own fix. A probe that reads True in both is as blind as the
eval it complements; a probe that reads False in both is measuring something else.

Usage (live kind cluster, AIOPSLAB_HOME set, same venv as the matrix):
    python scenarios/validate_probes.py [--scenario ID ...] [--out scenarios/results/probe-validation.json]

Candidate families (outside PROBES until admitted):
    python scenarios/validate_probes.py --candidates [--scenario FAMILY ...]
    (--scenario then names candidate families; default: all of them; default out:
    scenarios/results/probe-validation-candidates.json)

Runs no inference, only the cluster: the cost is about one hotelReservation deploy per scenario.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# Per-scenario ceiling, in seconds. AIOpsLab's traffic generator (wrk.py) has a `while True`
# that only exits on `job_status.status.ready` (truthy) or `.failed`: a Job that completes too
# fast to ever be observed as ready loops forever. Measured live (revoke_auth-1 stuck twice,
# >50min each; the same hang once stalled a full matrix run for 6h27). run_full_matrix.sh
# already guards against it with coreutils `timeout` per episode; admission (this script) had
# no ceiling, and a hang here blocks the whole run. SIGALRM on the main thread turns the hang
# into one error line per scenario (the same path as main's except), never losing the run.
EPISODE_TIMEOUT_SEC = 1800


class EpisodeTimeout(Exception):
    """Raised when a scenario exceeds EPISODE_TIMEOUT_SEC (a hang in third-party code)."""


@contextlib.contextmanager
def episode_deadline(seconds: int):
    """Arm a SIGALRM that raises EpisodeTimeout after `seconds`. No-op when `seconds<=0` or
    when not on the main thread (signal.alarm only exists there); in that case the ceiling in
    run_full_matrix.sh (external `timeout`) still applies to a matrix run."""
    if seconds <= 0:
        yield
        return

    def _raise(signum, frame):
        raise EpisodeTimeout(f"scenario exceeded {seconds}s (likely a hang in third-party wrk/k8s code)")

    try:
        previous = signal.signal(signal.SIGALRM, _raise)
    except ValueError:
        # not the main thread: no alarm, but the run does not break
        yield
        return
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

# Race measured live twice during admission (revoke_auth-1): the probe runs right after
# recover_fault() while the mongo container is still being recreated ("container not found"),
# so the probe comes out None (not measured). The retry is ONLY against that None: a True/False
# verdict is never re-measured, so this does not loosen admission, it only waits for recover to
# settle before accepting "could not measure" as the final answer.
SETTLE_MAX_ATTEMPTS = 18
SETTLE_SLEEP_SEC = 10.0


def probe_until_measured(
    measure,
    max_attempts: int = SETTLE_MAX_ATTEMPTS,
    sleep_sec: float = SETTLE_SLEEP_SEC,
    _sleep=time.sleep,
) -> dict:
    """Call measure() until the probe returns a verdict (root_cause_fixed is not None).

    Returns the first measured result; once the ceiling is exhausted, returns the last None with
    the reason it carries (probe_detail/probe_error), which still means "not measured".
    """
    result: dict = {}
    for attempt in range(max_attempts):
        result = measure()
        if result.get("root_cause_fixed") is not None:
            return result
        if attempt < max_attempts - 1:
            _sleep(sleep_sec)
    return result

DEFAULT_SCENARIOS = [
    "misconfig_app_hotel_res-mitigation-1",
    "redeploy_without_PV-mitigation-1",
    "revoke_auth_mongodb-mitigation-1",
    "revoke_auth_mongodb-mitigation-2",
    "user_unregistered_mongodb-mitigation-1",
    "user_unregistered_mongodb-mitigation-2",
    # spare (same app), native eval already sensitive to the root cause: a control for the probe
    "wrong_bin_usage-mitigation-1",
]


# Candidate families, keyed by FAMILY: candidate status comes
# from adapters/mitigation_probes.CANDIDATE_PROBES, the problem id from the permanent
# adapters/aiopslab_problems.PROBLEM_ID_BY_FAMILY. Separate from DEFAULT_SCENARIOS on purpose: the
# admission criterion is the same, but a candidate row must never be read as one of the five
# admitted probes, and the default run must keep measuring exactly what it did.
def _candidate_scenarios() -> tuple[str, ...]:
    from adapters.mitigation_probes import CANDIDATE_PROBES

    return tuple(sorted(CANDIDATE_PROBES))


def make_candidate_environment(problem_id: str):
    """make_environment plus the local candidate problems (the HotelReservation
    assign_to_non_existent_node wrapper) registered into that environment's own orchestrator
    registry, before task_description() asks it for the problem. The AIOpsLab clone is untouched."""
    from adapters.aiopslab_problems import register_local_problems
    from mlx_factories import make_environment

    env = make_environment(problem_id)
    register_local_problems(env._orchestrator.probs)
    return env


def admitted(after_inject: dict, after_recover: dict) -> bool:
    return after_inject.get("root_cause_fixed") is False and after_recover.get("root_cause_fixed") is True


def validate_one(
    scenario_id: str,
    environment_factory,
    timeout_sec: int = EPISODE_TIMEOUT_SEC,
    candidate_family: str | None = None,
) -> dict:
    from adapters.mitigation_probes import run_candidate_probe, run_probe

    with episode_deadline(timeout_sec):
        env = environment_factory(scenario_id)
        env.task_description()  # deploy + inject_fault, same as the start of an episode
        problem = env._orchestrator.session.problem
        kubectl, namespace = problem.kubectl, problem.namespace
        faulty = getattr(problem, "faulty_service", None)

        # A candidate's probe is looked up by family (CANDIDATE_PROBES), never through
        # probe_for(): the mongo rate variants' problem ids resolve to the ADMITTED geo keys
        # there, and container_kill-detection resolves to nothing.
        def measure() -> dict:
            if candidate_family is not None:
                return run_candidate_probe(candidate_family, kubectl, namespace, faulty)
            return run_probe(scenario_id, kubectl, namespace, faulty)

        after_inject = probe_until_measured(measure)
        problem.recover_fault()
        after_recover = probe_until_measured(measure)
        # finalize() would call recover_fault again; we already recovered, so teardown only cleans up.
        problem.recover_fault = lambda: None
        verdict = env.finalize(solution=None, history=[], duration=0.0)

    row = {
        "scenario_id": scenario_id,
        "after_inject": after_inject,
        "after_recover": after_recover,
        "benchmark_success_after_recover": verdict.get("success"),
        "admitted": admitted(after_inject, after_recover),
    }
    if candidate_family is not None:
        row["candidate_family"] = candidate_family
    return row


def validate_candidate(family: str, environment_factory=None, timeout_sec: int = EPISODE_TIMEOUT_SEC) -> dict:
    """validate_one for a candidate family: resolves the problem id and uses the candidate
    probe. `admitted` here means only that the PROBE passed admission; the family still needs a
    live detection run before it may move into PROBES."""
    from adapters.aiopslab_problems import PROBLEM_ID_BY_FAMILY
    from adapters.mitigation_probes import CANDIDATE_PROBES

    if family not in CANDIDATE_PROBES:
        raise ValueError(f"{family!r} is not a candidate family: {sorted(CANDIDATE_PROBES)}")
    return validate_one(
        PROBLEM_ID_BY_FAMILY[family],
        environment_factory or make_candidate_environment,
        timeout_sec,
        candidate_family=family,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scenario",
        action="append",
        help="problem id (repeatable); default: the 7 above. With --candidates: a candidate family",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="run the candidate families, never the admitted ones",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    results_dir = TOOLS_DIR / "scenarios" / "results"
    if args.candidates:
        out_default = results_dir / "probe-validation-candidates.json"
        targets = args.scenario or list(_candidate_scenarios())
    else:
        out_default = results_dir / "probe-validation.json"
        targets = args.scenario or DEFAULT_SCENARIOS
    args.out = args.out or str(out_default)

    rows = []
    for target in targets:
        print(f"[probe] {target}", flush=True)
        try:
            if args.candidates:
                row = validate_candidate(target)
            else:
                from mlx_factories import make_environment

                row = validate_one(target, make_environment)
        except Exception as exc:  # noqa: BLE001 - recorded per scenario, moves on to the next
            row = {"scenario_id": target, "error": f"{type(exc).__name__}: {exc}", "admitted": False}
            if args.candidates:
                row["candidate_family"] = target
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    total = len(rows)
    ok = sum(1 for r in rows if r.get("admitted"))
    print(f"admitted: {ok}/{total} -> {args.out}")


if __name__ == "__main__":
    main()
