"""Monitoring must flag a live process that no longer produces measurements."""
import importlib.util
import os
from datetime import UTC, datetime
from pathlib import Path


def module(name="monitor_leo_study"):
    path = Path(__file__).resolve().parents[1]/f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location("leo_monitor", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_live_process_with_old_records_is_stalled_but_validation_is_progress(tmp_path):
    monitor = module()
    folder = tmp_path/"combined"
    folder.mkdir()
    now = datetime.now(UTC)
    path = folder/"learning_steps.jsonl"
    path.write_text('{}\n')
    os.utime(path, (now.timestamp()-3600,)*2)
    arm = {"status": None, "process": {"verified": True, "creation_utc": now.isoformat()}}
    state = {"status": "running", "active_condition": "combined"}
    assert monitor.health(tmp_path, "combined", arm, state, now)["state"] == "stalled"
    (folder/"validation_cases.jsonl").write_text('{}\n')
    assert monitor.health(tmp_path, "combined", arm, state, now)["state"] == "progressing"


def test_finished_clean_is_not_mistaken_for_a_dead_worker(tmp_path):
    monitor = module()
    arm = {"status": {"qualified": True, "status": "qualified"}, "process": {"verified": False}}
    assert monitor.health(tmp_path, "clean", arm, {}, datetime.now(UTC))["state"] == "qualified"


def test_controller_failure_is_actionable_even_when_worker_was_alive(tmp_path):
    monitor = module()
    arm = {"status": None, "process": {"verified": True}}
    result = monitor.health(tmp_path, "combined", arm,
                           {"status": "error", "error": "Watchdog found a stall"}, datetime.now(UTC))
    assert result["state"] == "failed" and "Watchdog" in result["reason"]


def test_watchdog_terminates_stuck_worker_and_preserves_checkpoint(tmp_path, monkeypatch):
    import subprocess
    import time
    import psutil
    import pytest
    runner = module("run_leo_study")
    folder = tmp_path/"combined"
    folder.mkdir()
    checkpoint = folder/"checkpoint.pt"
    checkpoint.write_bytes(b"last complete checkpoint")
    baseline = time.time()
    # Mock the timestamp rather than filesystem clock translation on Windows.
    from types import SimpleNamespace
    original_stat = Path.stat
    monkeypatch.setattr(Path, "stat", lambda self, *a, **kw:
        SimpleNamespace(st_mtime=baseline) if self == checkpoint else original_stat(self, *a, **kw))
    clock = [baseline]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    stopped = []
    class Process:
        pid = 42
        def wait(self, timeout):
            if not stopped:
                clock[0] = baseline+1201.
                raise subprocess.TimeoutExpired("stuck-worker", timeout)
            return -15
        def children(self, recursive):
            return []
        def terminate(self):
            stopped.append(self.pid)
    process = Process()
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    with pytest.raises(RuntimeError, match="stalled"):
        runner.supervised_wait(process, tmp_path, "combined")
    assert stopped == [42]
    assert checkpoint.read_bytes() == b"last complete checkpoint"
    assert (tmp_path/"watchdog_failure.json").exists()
