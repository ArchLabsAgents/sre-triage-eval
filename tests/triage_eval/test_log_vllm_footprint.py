"""Unit coverage for scenarios/log_vllm_footprint.py, the vLLM host-side logger. Mocks
subprocess.run (nvidia-smi) and uses tmp_path for the cgroup files: no real GPU or systemd unit
needed."""

from __future__ import annotations

import subprocess

import pytest

from scenarios import log_vllm_footprint as mod


def _write_cgroup(tmp_path, usage_usec="1500000", memory_bytes="4000000"):
    cgroup = tmp_path / "vllm.service"
    cgroup.mkdir()
    (cgroup / "cpu.stat").write_text(f"usage_usec {usage_usec}\nuser_usec 1000000\nsystem_usec 500000\n")
    (cgroup / "memory.current").write_text(f"{memory_bytes}\n")
    return cgroup


class _FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_collect_one_tick_success_marks_collected_true(tmp_path):
    cgroup = _write_cgroup(tmp_path)

    def fake_runner(cmd, **kwargs):
        return _FakeProc(stdout="12000, 87, 210.50\n")

    sample = mod.collect_one_tick(
        "sslm-vllm",
        now=42.0,
        _nvidia_smi=lambda: mod._run_nvidia_smi(_runner=fake_runner),
        _resolve_cgroup=lambda name, **kwargs: cgroup,
    )

    assert sample.collected is True
    assert sample.timestamp == 42.0
    assert sample.gpu_vram_used_mb == pytest.approx(12000.0)
    assert sample.gpu_utilization_pct == pytest.approx(87.0)
    assert sample.gpu_power_watts == pytest.approx(210.50)
    assert sample.cgroup_cpu_seconds == pytest.approx(1.5)
    assert sample.cgroup_memory_bytes == pytest.approx(4_000_000.0)
    assert sample.reason is None


def test_run_nvidia_smi_binary_missing_raises(monkeypatch):
    def fake_runner(cmd, **kwargs):
        raise FileNotFoundError("no such file")

    with pytest.raises(mod.NvidiaSmiError, match="PATH"):
        mod._run_nvidia_smi(_runner=fake_runner)


def test_run_nvidia_smi_timeout_raises():
    def fake_runner(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

    with pytest.raises(mod.NvidiaSmiError, match="10"):
        mod._run_nvidia_smi(_runner=fake_runner)


def test_run_nvidia_smi_nonzero_exit_raises():
    def fake_runner(cmd, **kwargs):
        return _FakeProc(returncode=1, stderr="NVIDIA-SMI has failed")

    with pytest.raises(mod.NvidiaSmiError, match="NVIDIA-SMI has failed"):
        mod._run_nvidia_smi(_runner=fake_runner)


def test_run_nvidia_smi_malformed_output_raises():
    def fake_runner(cmd, **kwargs):
        return _FakeProc(stdout="not,enough\n")

    with pytest.raises(mod.NvidiaSmiError):
        mod._run_nvidia_smi(_runner=fake_runner)


def test_resolve_container_cgroup_path_reads_pid_and_unified_line(tmp_path):
    proc_root = tmp_path / "proc"
    (proc_root / "4242").mkdir(parents=True)
    (proc_root / "4242" / "cgroup").write_text("0::/system.slice/docker-abc123.scope\n")

    def fake_runner(cmd, **kwargs):
        return _FakeProc(stdout="4242\n")

    path = mod._resolve_container_cgroup_path(
        "sslm-vllm", cgroup_root=tmp_path / "cgroupfs", _runner=fake_runner, _proc_root=proc_root
    )

    assert path == tmp_path / "cgroupfs" / "system.slice/docker-abc123.scope"


def test_resolve_container_cgroup_path_docker_missing_raises():
    def fake_runner(cmd, **kwargs):
        raise FileNotFoundError("no such file")

    with pytest.raises(mod.DockerInspectError, match="PATH"):
        mod._resolve_container_cgroup_path("sslm-vllm", _runner=fake_runner)


def test_resolve_container_cgroup_path_timeout_raises():
    def fake_runner(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=10)

    with pytest.raises(mod.DockerInspectError, match="10"):
        mod._resolve_container_cgroup_path("sslm-vllm", _runner=fake_runner)


def test_resolve_container_cgroup_path_container_not_found_raises():
    def fake_runner(cmd, **kwargs):
        return _FakeProc(returncode=1, stderr="Error: No such object: sslm-vllm")

    with pytest.raises(mod.DockerInspectError, match="No such object"):
        mod._resolve_container_cgroup_path("sslm-vllm", _runner=fake_runner)


def test_resolve_container_cgroup_path_non_numeric_pid_raises():
    def fake_runner(cmd, **kwargs):
        return _FakeProc(stdout="<no value>\n")

    with pytest.raises(mod.DockerInspectError, match="non-numeric"):
        mod._resolve_container_cgroup_path("sslm-vllm", _runner=fake_runner)


def test_resolve_container_cgroup_path_stopped_container_raises():
    def fake_runner(cmd, **kwargs):
        return _FakeProc(stdout="0\n")

    with pytest.raises(mod.DockerInspectError, match="not running"):
        mod._resolve_container_cgroup_path("sslm-vllm", _runner=fake_runner)


def test_resolve_container_cgroup_path_unreadable_proc_cgroup_raises(tmp_path):
    def fake_runner(cmd, **kwargs):
        return _FakeProc(stdout="999\n")

    with pytest.raises(mod.DockerInspectError, match="could not read"):
        mod._resolve_container_cgroup_path(
            "sslm-vllm", _runner=fake_runner, _proc_root=tmp_path / "no-proc"
        )


def test_resolve_container_cgroup_path_missing_unified_line_raises(tmp_path):
    proc_root = tmp_path / "proc"
    (proc_root / "4242").mkdir(parents=True)
    (proc_root / "4242" / "cgroup").write_text("4:cpu,cpuacct:/system.slice/docker-abc123.scope\n")

    def fake_runner(cmd, **kwargs):
        return _FakeProc(stdout="4242\n")

    with pytest.raises(mod.DockerInspectError, match="0::"):
        mod._resolve_container_cgroup_path("sslm-vllm", _runner=fake_runner, _proc_root=proc_root)


def test_read_cgroup_footprint_missing_dir_raises(tmp_path):
    with pytest.raises(mod.CgroupReadError):
        mod._read_cgroup_footprint(tmp_path / "does-not-exist")


def test_read_cgroup_footprint_missing_usage_usec_raises(tmp_path):
    cgroup = tmp_path / "vllm.service"
    cgroup.mkdir()
    (cgroup / "cpu.stat").write_text("user_usec 1000000\n")
    (cgroup / "memory.current").write_text("100\n")

    with pytest.raises(mod.CgroupReadError, match="usage_usec"):
        mod._read_cgroup_footprint(cgroup)


def test_collect_one_tick_gpu_failure_keeps_collected_false_never_partial(tmp_path):
    cgroup = _write_cgroup(tmp_path)

    def failing_nvidia_smi():
        raise mod.NvidiaSmiError("no GPU visible")

    sample = mod.collect_one_tick(
        "sslm-vllm",
        now=1.0,
        _nvidia_smi=failing_nvidia_smi,
        _resolve_cgroup=lambda name, **kwargs: cgroup,
    )

    assert sample.collected is False
    assert sample.gpu_vram_used_mb is None
    assert sample.cgroup_cpu_seconds is None  # the cgroup would have worked, but the tick is all-or-nothing
    assert "no GPU visible" in sample.reason


def test_collect_one_tick_cgroup_read_failure_keeps_collected_false_never_partial(tmp_path):
    def working_nvidia_smi():
        return (12000.0, 87.0, 210.0)

    sample = mod.collect_one_tick(
        "sslm-vllm",
        now=1.0,
        _nvidia_smi=working_nvidia_smi,
        _resolve_cgroup=lambda name, **kwargs: tmp_path / "missing-cgroup",
    )

    assert sample.collected is False
    assert sample.gpu_vram_used_mb is None  # the GPU would have worked, but the tick is all-or-nothing


def test_collect_one_tick_docker_inspect_failure_keeps_collected_false_never_partial(tmp_path):
    def working_nvidia_smi():
        return (12000.0, 87.0, 210.0)

    def failing_resolve(name, **kwargs):
        raise mod.DockerInspectError("container sslm-vllm has no PID (not running?): '0'")

    sample = mod.collect_one_tick(
        "sslm-vllm", now=1.0, _nvidia_smi=working_nvidia_smi, _resolve_cgroup=failing_resolve
    )

    assert sample.collected is False
    assert sample.gpu_vram_used_mb is None  # the GPU would have worked, but the tick is all-or-nothing
    assert "not running" in sample.reason


def test_run_loop_once_writes_a_single_line_and_returns(tmp_path):
    cgroup = _write_cgroup(tmp_path)
    out = tmp_path / "vllm-footprint.jsonl"

    def fake_nvidia_smi():
        return (1.0, 2.0, 3.0)

    sleeps = []
    mod.run_loop(
        "sslm-vllm",
        out,
        once=True,
        _sleep=lambda s: sleeps.append(s),
        _nvidia_smi=fake_nvidia_smi,
        _resolve_cgroup=lambda name, **kwargs: cgroup,
    )

    assert sleeps == []
    from harness.metrics import load_vllm_footprint_samples

    samples = load_vllm_footprint_samples(out)
    assert len(samples) == 1
    assert samples[0].collected is True
