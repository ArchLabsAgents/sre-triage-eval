import csv
import json

from harness.agent import EpisodeResult, TurnResult
from harness.metrics import CSV_FIELDS, RunRecord, aggregate, write_matrix_csv


def _record(scenario_id, config_name, success, submitted=None, tokens=100, seconds=10.0):
    return RunRecord(
        scenario_id=scenario_id,
        config_name=config_name,
        submitted=bool(success) if submitted is None else submitted,
        output_tokens=tokens,
        elapsed_seconds=seconds,
        attempt_count=1,
        success=success,
    )


def _episode(solved):
    return EpisodeResult(
        task_id="s1",
        solved=solved,
        turns=[TurnResult(output_tokens=7, schema_valid=True, action={}, solved=solved, observation="")],
        elapsed_seconds=3.0,
    )


def test_aggregate_success_rate():
    records = [
        _record("1", "config_a", True),
        _record("2", "config_a", False),
        _record("1", "config_b", True),
        _record("2", "config_b", True),
    ]
    summaries = {s.config_name: s for s in aggregate(records)}

    assert summaries["config_a"].success_rate == 0.5
    assert summaries["config_b"].success_rate == 1.0


def test_aggregate_success_and_submit_are_separate_axes():
    # The early-matrix case: submit() called, eval failed. Counts as submitted, never
    # as success.
    records = [_record("1", "config_c", success=False, submitted=True)]
    summary = aggregate(records)[0]

    assert summary.scenarios_submitted == 1
    assert summary.scenarios_succeeded == 0
    assert summary.submit_rate == 1.0
    assert summary.success_rate == 0.0


def test_aggregate_unmeasured_counts_in_attempted_never_in_succeeded():
    records = [_record("1", "config_a", success=None, submitted=True), _record("2", "config_a", True)]
    summary = aggregate(records)[0]

    assert summary.scenarios_attempted == 2
    assert summary.scenarios_unmeasured == 1
    assert summary.scenarios_succeeded == 1
    assert summary.success_rate == 0.5


def test_aggregate_means_exclude_unsuccessful_tasks():
    records = [
        _record("1", "config_a", True, tokens=100, seconds=10.0),
        _record("2", "config_a", False, tokens=99999, seconds=99999.0),
    ]
    summary = aggregate(records)[0]

    assert summary.mean_output_tokens_succeeded == 100
    assert summary.mean_elapsed_seconds_succeeded == 10.0


def test_aggregate_no_successful_tasks_returns_none_means():
    records = [_record("1", "config_a", False)]
    summary = aggregate(records)[0]

    assert summary.mean_output_tokens_succeeded is None
    assert summary.mean_elapsed_seconds_succeeded is None


def test_from_episode_reads_verdict_from_eval_results():
    record = RunRecord.from_episode(
        "s1", "config_b", _episode(solved=True), {"success": False, "TTM": 42.5, "steps": 3}
    )

    assert record.submitted is True
    assert record.success is False
    assert record.ttm_seconds == 42.5
    assert record.teardown_error is None


def test_from_episode_without_verdict_is_unmeasured_not_failed():
    record = RunRecord.from_episode("s1", "config_b", _episode(solved=True), {})

    assert record.submitted is True
    assert record.success is None
    assert record.ttm_seconds is None


def test_from_episode_keeps_teardown_error():
    record = RunRecord.from_episode(
        "s1", "config_a", _episode(solved=False), {"success": True, "teardown_error": "Timeout: ns"}
    )

    assert record.success is True
    assert record.teardown_error == "Timeout: ns"


def test_from_episode_eval_error_is_unmeasured_with_reason_and_round_trips(tmp_path):
    # Observed: AIOpsLab's eval raised a TypeError on redeploy_without_PV; the record survives
    # with a not-measured verdict and the reason, and counts as unmeasured in the aggregate.
    record = RunRecord.from_episode(
        "redeploy_without_PV-mitigation-1",
        "config_b",
        _episode(solved=False),
        {"eval_error": "TypeError: 'NoneType' object is not iterable"},
    )

    assert record.success is None
    assert record.eval_error == "TypeError: 'NoneType' object is not iterable"
    assert "eval_error" in CSV_FIELDS
    path = record.write(tmp_path)
    assert RunRecord.from_json(path.read_text(encoding="utf-8")) == record
    assert aggregate([record])[0].scenarios_unmeasured == 1


def test_from_json_reads_legacy_solved_record_as_submitted_without_verdict():
    legacy = json.dumps(
        {
            "scenario_id": "s1",
            "config_name": "config_c",
            "solved": True,
            "output_tokens": 10,
            "elapsed_seconds": 1.0,
            "attempt_count": 1,
        }
    )
    record = RunRecord.from_json(legacy)

    # The earliest format's `solved` measured submit(); it never becomes success=True.
    assert record.submitted is True
    assert record.success is None


def test_from_json_round_trips_current_record(tmp_path):
    original = _record("1", "config_b", success=True, submitted=True)
    path = original.write(tmp_path)

    assert RunRecord.from_json(path.read_text(encoding="utf-8")) == original


def test_write_matrix_csv_has_one_row_per_record_and_success_first(tmp_path):
    records = [_record("1", "config_a", True), _record("1", "config_b", False)]
    path = write_matrix_csv(records, tmp_path / "matrix.csv")

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert list(rows[0].keys()) == CSV_FIELDS
    assert CSV_FIELDS.index("success") < CSV_FIELDS.index("submitted")
    assert rows[0]["success"] == "True"
    assert rows[1]["success"] == "False"


def test_from_episode_reads_root_cause_probe_and_csv_carries_it(tmp_path):
    verdict = {"success": True, "root_cause_fixed": False, "probe_detail": "buggy image"}
    record = RunRecord.from_episode("s1", "config_a", _episode(solved=False), verdict)

    assert record.success is True
    assert record.root_cause_fixed is False
    assert record.probe_detail == "buggy image"
    assert "root_cause_fixed" in CSV_FIELDS

    path = write_matrix_csv([record], tmp_path / "m.csv")
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["root_cause_fixed"] == "False"


def test_aggregate_counts_root_cause_fixed_separately_from_success():
    a = _record("s1", "config_a", success=True)
    a.root_cause_fixed = False
    b = _record("s2", "config_a", success=True)
    b.root_cause_fixed = True
    c = _record("s3", "config_a", success=None)

    summary = aggregate([a, b, c])[0]
    assert summary.scenarios_succeeded == 2
    assert summary.scenarios_root_cause_fixed == 1


def test_from_json_reads_record_written_before_probe_fields(tmp_path):
    old = {
        "scenario_id": "s1", "config_name": "config_a", "submitted": False, "output_tokens": 1,
        "elapsed_seconds": 1.0, "attempt_count": 1, "success": True,
    }
    record = RunRecord.from_json(json.dumps(old))
    assert record.root_cause_fixed is None


def test_from_json_reads_pre_env_record_as_env_none():
    # A record written before the env column (no env): loads as env=None, never as mlx-local.
    # Stamping is an explicit decision (scenarios/backfill_env.py), not a loader default.
    old = {
        "scenario_id": "s1", "config_name": "config_a", "submitted": True, "output_tokens": 1,
        "elapsed_seconds": 1.0, "attempt_count": 1, "success": True,
    }
    record = RunRecord.from_json(json.dumps(old))

    assert record.env is None
    assert record.env_quantization is None
    assert record.model_seconds is None


def test_write_routes_record_into_env_directory(tmp_path):
    # Two envs writing the same (scenario, config): two files in two directories,
    # neither overwrites the other (the most dangerous point of the per-env change).
    mac = _record("s1", "config_a", True)
    mac.env = "mlx-local"
    cloud = _record("s1", "config_a", False)
    cloud.env = "hyperstack-a6000"

    mac_path = mac.write(tmp_path)
    cloud_path = cloud.write(tmp_path)

    assert mac_path == tmp_path / "mlx-local" / "s1_config_a.json"
    assert cloud_path == tmp_path / "hyperstack-a6000" / "s1_config_a.json"
    assert mac_path.exists() and cloud_path.exists()
    assert RunRecord.from_json(mac_path.read_text(encoding="utf-8")).success is True
    assert RunRecord.from_json(cloud_path.read_text(encoding="utf-8")).success is False


def test_csv_renders_env_none_as_unknown_and_carries_both_envs(tmp_path):
    legacy = _record("s1", "config_a", True)
    cloud = _record("s1", "config_a", False)
    cloud.env = "hyperstack-a6000"
    cloud.env_quantization = "fp16"
    cloud.env_network = "local"

    path = write_matrix_csv([legacy, cloud], tmp_path / "matrix.csv")

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert CSV_FIELDS[:3] == ["env", "env_quantization", "env_network"]
    assert rows[0]["env"] == "unknown"
    assert rows[1]["env"] == "hyperstack-a6000"
    assert rows[1]["env_quantization"] == "fp16"


def test_aggregate_groups_by_env_and_config_never_config_alone():
    mac = _record("s1", "config_a", True)
    mac.env = "mlx-local"
    cloud = _record("s1", "config_a", False)
    cloud.env = "hyperstack-a6000"

    summaries = aggregate([mac, cloud])

    # One summary per (env, config); no mean crosses envs (100% and 0%, never 50%).
    assert [(s.env, s.config_name) for s in summaries] == [
        ("hyperstack-a6000", "config_a"),
        ("mlx-local", "config_a"),
    ]
    by_env = {s.env: s for s in summaries}
    assert by_env["mlx-local"].success_rate == 1.0
    assert by_env["hyperstack-a6000"].success_rate == 0.0


def test_aggregate_keeps_legacy_env_none_in_its_own_bucket():
    legacy = _record("s1", "config_a", True)
    named = _record("s2", "config_a", True)
    named.env = "mlx-local"

    summaries = aggregate([legacy, named])

    assert [(s.env, s.scenarios_attempted) for s in summaries] == [
        ("mlx-local", 1),
        (None, 1),
    ]


def test_collect_records_reads_root_and_env_dirs_but_never_the_descriptor(tmp_path):
    from harness.metrics import collect_records

    legacy = _record("s1", "config_a", True)
    legacy.write(tmp_path)
    named = _record("s1", "config_a", False)
    named.env = "mlx-local"
    named.write(tmp_path)
    (tmp_path / "mlx-local" / "env-descriptor.json").write_text(
        json.dumps({"backend": "mlx"}), encoding="utf-8"
    )

    records = collect_records(tmp_path)

    assert len(records) == 2
    assert {record.env for record in records} == {None, "mlx-local"}
