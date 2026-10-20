"""Run the three fresh learning-delay arms against a preserved clean reference.

Uses the reference's frozen package and exact configuration. This launcher never
resumes training, changes qualification rules, or launches mission inference.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CONDITIONS = ("uplink_delayed", "downlink_delayed", "combined")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_creation(pid):
    """Actual Windows process creation time, rather than launcher wall time."""
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    values = [wintypes.FILETIME() for _ in range(4)]
    try:
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in values)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)
    ticks = (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime
    seconds = ticks / 10_000_000 - 11_644_473_600
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def preflight(root, check_all=True):
    manifest = read_json(root / "clean/manifest.json")
    status = read_json(root / "clean/status.json")
    if manifest["condition"] != "clean" or manifest["seed"] != 42:
        raise ValueError("The reference must be the selected seed-42 clean pilot")
    if status["status"] not in ("budget_exhausted", "qualified"):
        raise ValueError("The clean reference is not terminal")
    actual = {p.name: digest(p) for p in (root / "code/panama_uav").glob("*.py")}
    if actual != manifest["code_sha256"]:
        raise ValueError("Frozen package differs from the clean reference")
    config = manifest["config"]
    if config["communication"]["profile"] != "controlled":
        raise ValueError("This comparison requires the controlled profile")
    if config["communication"]["stale_information"]:
        raise ValueError("Mission information must remain fresh")
    if check_all:
        for condition in CONDITIONS:
            if (root / condition).exists() or (root / f"launch-{condition}.json").exists():
                raise ValueError(f"{condition} already exists; refusing a duplicate")
    return manifest


def run_worker(root, condition):
    manifest = preflight(root, check_all=False)
    destination = root / condition
    if destination.exists():
        raise ValueError("Worker destination exists; fresh initialization required")
    sys.path.insert(0, str(root / "code"))
    from panama_uav.config import (
        CurriculumStage, EvaluationConfig, MapConfig, ProjectConfig, RewardConfig,
        TrainingConfig, TransportConfig, UAVConfig,
    )
    from panama_uav.training import train_policy

    # Construct from the persisted full configuration, so later edits to TOML
    # files cannot change an overnight queued arm.
    values = manifest["config"]
    communication = dict(values["communication"])
    communication["stale_information"] = tuple(communication["stale_information"])
    evaluation = dict(values["evaluation"])
    for key in ("agent_counts", "delay_steps"):
        evaluation[key] = tuple(evaluation[key])
    config = ProjectConfig(
        project_root=Path(values["project_root"]),
        map=MapConfig(**values["map"]),
        reward=RewardConfig(**values["reward"]),
        training=TrainingConfig(**values["training"]),
        curriculum=tuple(CurriculumStage(**s) for s in values["curriculum"]),
        uav=UAVConfig(**values["uav"]),
        communication=TransportConfig(**communication),
        evaluation=EvaluationConfig(**evaluation),
    )
    if hasattr(ProjectConfig, "from_dict"):
        config = ProjectConfig.from_dict(values)
    if json.dumps(config.to_dict(), sort_keys=True) != json.dumps(values, sort_keys=True):
        raise ValueError("Worker configuration differs from the clean reference")
    outcome = train_policy(
        config, condition, 42, destination, resume=False,
        device_name="cuda", run_id=root.name,
    )
    actual = read_json(destination / "manifest.json")
    for key in ("initial_weights_sha256", "map_sha256", "code_sha256", "config"):
        if actual[key] != manifest[key]:
            raise ValueError(f"Condition reference mismatch: {key}")
    print(json.dumps({"condition": condition, "status": outcome.status}), flush=True)


def run_queue(root):
    manifest = preflight(root)
    claim = root / ".delay_comparison.claim"
    with claim.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "created_utc": utc_now()}))
    plan = {
        "run_id": root.name,
        "clean_reference": str(root / "clean"),
        "clean_qualified": read_json(root / "clean/status.json")["qualified"],
        "conditions": list(CONDITIONS),
        "seed": 42,
        "fresh_initializations": True,
        "execution": "sequential, one CUDA learner at a time",
        "budget_seconds_per_condition": manifest["config"]["training"]["budget_seconds"],
        "communication_profile": "controlled",
        "delay_steps": manifest["config"]["communication"]["delay_steps"],
        "mission_information": "fresh",
        "frozen_source": str(root / "code"),
        "code_sha256": manifest["code_sha256"],
        "expected_initial_weights_sha256": manifest["initial_weights_sha256"],
        "launcher_sha256": digest(__file__),
        "automatic_mission_inference": False,
    }
    write_json(root / "delay_comparison_plan.json", plan)
    write_json(root / "delay_comparison_launch.json", {
        "pid": os.getpid(), "started_utc": process_creation(os.getpid()),
        "command": subprocess.list2cmdline([sys.executable, "-u", *sys.argv]),
        "command_args": [sys.executable, "-u", *sys.argv],
        "run_id": root.name, "role": "sequential learning-delay controller",
    })
    state = {
        "status": "running", "active_condition": None,
        "queued_conditions": list(CONDITIONS), "completed_conditions": [],
        "started_utc": utc_now(), "mission_comparison_launched": False,
    }
    state_path = root / "delay_comparison_status.json"
    try:
        for condition in CONDITIONS:
            state["active_condition"] = condition
            state["queued_conditions"] = [c for c in CONDITIONS if c not in state["completed_conditions"] and c != condition]
            write_json(state_path, state)
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--run-root", str(root), "--worker-condition", condition]
            with (root / f"{condition}.stdout.log").open("wb") as stdout, \
                 (root / f"{condition}.stderr.log").open("wb") as stderr:
                process = subprocess.Popen(
                    command, cwd=manifest["config"]["project_root"],
                    stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                write_json(root / f"launch-{condition}.json", {
                    "pid": process.pid, "started_utc": process_creation(process.pid),
                    "command": subprocess.list2cmdline(command), "command_args": command,
                    "condition": condition, "run_id": root.name,
                    "budget_seconds": plan["budget_seconds_per_condition"],
                    "fresh_initialization": True,
                })
                state["active_pid"] = process.pid
                write_json(state_path, state)
                print("START " + condition, flush=True)
                return_code = process.wait()
            if return_code:
                raise RuntimeError(f"{condition} exited with code {return_code}")
            result = read_json(root / condition / "status.json")
            if result["status"] not in ("budget_exhausted", "qualified", "phase_episode_budget_exhausted", "generated_step_budget_exhausted"):
                raise RuntimeError(f"Unexpected terminal status: {result['status']}")
            state["completed_conditions"].append(condition)
            state["active_pid"] = None
            print(json.dumps({"condition": condition, **result}), flush=True)
            write_json(state_path, state)
        state.update(status="reporting", active_condition=None, queued_conditions=[])
        write_json(state_path, state)
        environment = dict(os.environ, PYTHONPATH=str(root / "code"))
        subprocess.run(
            [sys.executable, "-u", "-m", "panama_uav", "report", "--path", str(root)],
            cwd=manifest["config"]["project_root"], env=environment, check=True,
            timeout=600, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        state.update(status="complete", finished_utc=utc_now(), reports_refreshed=True)
        write_json(state_path, state)
    except BaseException as error:
        state.update(status="error", error=str(error), failed_utc=utc_now())
        write_json(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--worker-condition", choices=CONDITIONS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.worker_condition:
        run_worker(root, args.worker_condition)
    elif args.check_only:
        manifest = preflight(root)
        print(json.dumps({"preflight": "passed", "conditions": CONDITIONS,
                          "budget_seconds_per_condition": manifest["config"]["training"]["budget_seconds"],
                          "expected_initial_weights_sha256": manifest["initial_weights_sha256"]}))
    else:
        run_queue(root)


if __name__ == "__main__":
    main()
