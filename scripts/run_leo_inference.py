"""Qualified, frozen-baseline mission-information inference; paired held-out cases.

Controlled staleness and orbital transport are separate conditions. Independent
episodes run in Python workers; each robot population uses batched inference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
from datetime import UTC, datetime
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
COUNTS = (2, 4, 6, 8)
FIELDS = (("task",), ("intentions",), ("priorities",), ("task", "intentions", "priorities"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def utc():
    return datetime.now(UTC).isoformat()


def config_from(values):
    from panama_uav.config import ProjectConfig
    if hasattr(ProjectConfig, "from_dict"):
        return ProjectConfig.from_dict(values)
    from panama_uav.config import (ProjectConfig, MapConfig, RewardConfig, TrainingConfig,
        CurriculumStage, UAVConfig, TransportConfig, EvaluationConfig)
    comm = dict(values["communication"])
    comm["stale_information"] = tuple(comm["stale_information"])
    ev = dict(values["evaluation"])
    for key in ("agent_counts", "delay_steps"):
        ev[key] = tuple(ev[key])
    return ProjectConfig(Path(values["project_root"]), MapConfig(**values["map"]),
        RewardConfig(**values["reward"]), TrainingConfig(**values["training"]),
        tuple(CurriculumStage(**s) for s in values["curriculum"]), UAVConfig(**values["uav"]),
        TransportConfig(**comm), EvaluationConfig(**ev))


def conditions(config):
    perfect = replace(config.communication, profile="perfect", stale_information=(),
                      handover_enabled=False, contention_enabled=False)
    result = [("perfect", perfect)]
    # Physical-profile cases do not add artificial contention or delay windows.
    for route in ("cloud", "uav"):
        for fields in FIELDS:
            name = "all" if len(fields) == 3 else fields[0]
            result.append((f"orbital_{route}_{name}", replace(config.communication,
                profile="physical", mission_route=route, stale_information=fields,
                handover_enabled=True, contention_enabled=False)))
    for route in ("cloud", "uav"):
        for delay in (1, 2, 5, 10, 20):
            for fields in FIELDS:
                name = "all" if len(fields) == 3 else fields[0]
                result.append((f"controlled_{route}_{name}_{delay}", replace(config.communication,
                    profile="controlled", mission_route=route, stale_information=fields,
                    delay_steps=delay, handover_enabled=False, contention_enabled=True)))
    return result


def prepare(args):
    training = args.training_root.resolve()
    clean = training/"clean"
    status = json.loads((clean/"status.json").read_text())
    if not status["qualified"] or len(status["qualification"]) != 4:
        raise ValueError("Inference requires the fully qualified Clean baseline")
    root = (PROJECT/"output/inference"/args.run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(PROJECT/"src/panama_uav", root/"code/panama_uav", ignore=shutil.ignore_patterns("__pycache__"))
    for script in ("run_leo_inference.py", "plot_leo_inference.py"):
        shutil.copy2(PROJECT/"scripts"/script, root/"code"/script)
    shutil.copy2(clean/"final_model.pt", root/"baseline.pt")
    manifest = json.loads((clean/"manifest.json").read_text())
    (root/"assets/maps").mkdir(parents=True)
    shutil.copy2(Path(manifest["config"]["project_root"])/"maps/random-32-32-10.map", root/"assets/maps/random-32-32-10.map")
    shutil.copy2(manifest["config"]["communication"]["leo_timeline"], root/"assets/orbit.npz")
    values = manifest["config"]
    values["project_root"] = str(root/"assets")
    values["communication"]["leo_timeline"] = str(root/"assets/orbit.npz")
    plan = {"run_id": root.name, "created_utc": utc(), "training_root": str(training),
        "baseline_model_sha256": sha(root/"baseline.pt"), "baseline_qualified": True,
        "baseline_run_id": json.loads((clean/"manifest.json").read_text())["run_id"],
        "baseline_training_code_sha256": manifest["code_sha256"],
        "code_sha256": {p.name: sha(p) for p in (root/"code/panama_uav").glob("*.py")},
        "configuration": values, "counts": list(COUNTS), "cases_per_stage": 50,
        "episode_cap": 250, "controlled_delays_steps": [1, 2, 5, 10, 20],
        "routes": ["cloud", "uav"], "fields": [list(f) for f in FIELDS],
        "workers": args.workers, "device": args.device,
        "map_sha256": sha(root/"assets/maps/random-32-32-10.map"),
        "orbit_sha256": sha(root/"assets/orbit.npz"),
        "runner_sha256": sha(root/"code/run_leo_inference.py"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "pairing": "Same 200 held-out starts/goals/case seeds; seeded reservations paired across controlled conditions",
        "policy": "Frozen fully qualified Clean final model, greedy; no learning or weight deployment",
        "isolation": "Only selected receiver caches are delayed; local sensing, truth collision resolution and unselected fields stay fresh",
        "profile_separation": "Controlled 1–20-step contention is a stress test, not a measured handover duration; orbital tests use library-backed service without artificial reservations"}
    write(root/"plan.json", plan)
    return root


def init_worker(root, device):
    global MODEL, PAYLOAD, DEVICE, ROOT
    ROOT = Path(root)
    sys.path.insert(0, str(ROOT/"code"))
    import torch
    from panama_uav.models import choose_device
    from panama_uav.training import load_model
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    plan = json.loads((ROOT/"plan.json").read_text())
    assert {p.name: sha(p) for p in (ROOT/"code/panama_uav").glob("*.py")} == plan["code_sha256"]
    DEVICE = choose_device(device)
    MODEL, PAYLOAD = load_model(ROOT/"baseline.pt", config_from(plan["configuration"]), DEVICE, require_qualified=True)
    MODEL.eval()


def episode_job(task):
    import numpy as np
    from panama_uav.evaluation import run_episode
    config, name, case = task
    row, trace = run_episode(config, MODEL, case, DEVICE, PAYLOAD["learner_updates"])
    assert trace["frames"][0]["positions"] == case["starts"] and trace["goals"] == case["goals"]
    comm = config.communication
    row.update(condition=name, stage=COUNTS.index(case["num_agents"])+1,
               mission_route=comm.mission_route, communication_profile=comm.profile,
               stale_information=list(comm.stale_information),
               delay_steps=comm.delay_steps if comm.profile == "controlled" else 0,
               diagnostic=False, policy_model_sha256=sha(ROOT/"baseline.pt"))
    for field in ("task", "intentions", "priorities"):
        ages = [v[field] for frame in trace["frames"] for v in frame["information_age"].values()]
        row[f"maximum_{field}_age_seconds"] = float(max(ages))
        row[f"p95_{field}_age_seconds"] = float(np.quantile(ages, .95))
        if field not in comm.stale_information or comm.profile == "perfect":
            assert max(ages) == 0, f"Unselected {field} was made stale"
    row["actual_stale_agents"] = sum(any(any(frame["information_age"][agent][field] > 0
        for field in comm.stale_information) for frame in trace["frames"])
        for agent in range(case["num_agents"]))
    row["maximum_queue_bytes"] = max(sum(f["queue_bytes"].values()) for f in trace["frames"])
    assert row["steps"] <= 250 and row["par2"] == (row["steps"] if row["success"] else 500)
    target = ROOT/name/f"{case['case_id']}.json.gz"
    temporary = target.with_suffix(target.suffix+".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream:
        json.dump(trace, stream, separators=(",", ":"), allow_nan=False)
    os.replace(temporary, target)
    return row


def run(root):
    import psutil
    from dataclasses import asdict
    root = root.resolve()
    sys.path.insert(0, str(root/"code"))
    from panama_uav.evaluation import case_bank, paired_summary
    from panama_uav.records import append_jsonl, export_csv
    plan = json.loads((root/"plan.json").read_text())
    assert sha(root/"baseline.pt") == plan["baseline_model_sha256"]
    assert sha(root/"assets/orbit.npz") == plan["orbit_sha256"]
    assert sha(root/"assets/maps/random-32-32-10.map") == plan["map_sha256"]
    config = config_from(plan["configuration"])
    choices = conditions(config)
    bank = case_bank(replace(config, communication=choices[0][1]))
    bank_path = root/"held_out_cases.json"
    if bank_path.exists():
        assert json.loads(bank_path.read_text()) == bank
    else:
        write(bank_path, bank)
    write(root/"manifest.json", {**plan, "case_bank_sha256": sha(bank_path),
        "conditions": [{"name": name, "transport": asdict(comm)} for name, comm in choices],
        "unique_conditions": len(choices), "planned_episode_executions": len(bank)*len(choices)})
    write(root/"launch.json", {"pid": os.getpid(), "creation_utc": datetime.fromtimestamp(psutil.Process().create_time(), UTC).isoformat(),
        "command_args": [sys.executable, *sys.argv]})
    state = {"status": "running", "started_utc": utc(), "active_condition": None,
             "completed_conditions": [], "completed_episodes": 0,
             "planned_episodes": len(bank)*len(choices), "workers": plan["workers"]}
    start = time.monotonic()
    reference, all_summaries = [], []
    executor = ProcessPoolExecutor(plan["workers"], mp_context=multiprocessing.get_context("spawn"),
                                   initializer=init_worker, initargs=(str(root), plan["device"]))
    try:
        for name, comm in choices:
            directory = root/name
            directory.mkdir(exist_ok=True)
            path = directory/"episodes.jsonl"
            existing = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
            done = {r["case_id"] for r in existing}
            assert len(done) == len(existing)
            state.update(active_condition=name, condition_episodes=len(existing), elapsed_seconds=time.monotonic()-start)
            state["completed_episodes"] += len(existing)
            write(root/"status.json", state)
            print(f"START {name} ({len(existing)}/{len(bank)} retained)", flush=True)
            condition_config = replace(config, communication=comm)
            pending = {executor.submit(episode_job, (condition_config, name, case))
                       for case in bank if case["case_id"] not in done}
            last_progress = time.monotonic()
            while pending:
                finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                if not finished and time.monotonic()-last_progress > 600:
                    raise RuntimeError("No inference episode completed for ten minutes")
                for future in finished:
                    row = future.result()
                    append_jsonl(path, row)
                    existing.append(row)
                    state["condition_episodes"] += 1
                    state["completed_episodes"] += 1
                    state.update(latest_case=row["case_id"], latest_stage=row["stage"], elapsed_seconds=time.monotonic()-start)
                    last_progress = time.monotonic()
                    write(root/"status.json", state)
                    if state["completed_episodes"] % 50 == 0:
                        print(json.dumps(state), flush=True)
            existing.sort(key=lambda r: (r["num_agents"], r["case"]))
            assert len(existing) == len(bank)
            if name == "perfect": reference = existing
            summary = paired_summary(existing, reference)
            baseline = {r["case_id"]: r for r in reference}
            import numpy as np
            for item in summary:
                item.update(condition=name, stage=COUNTS.index(item["num_agents"])+1,
                    communication_profile=comm.profile, mission_route=comm.mission_route,
                    fields="all" if len(comm.stale_information)==3 else comm.stale_information[0] if comm.stale_information else "perfect",
                    delay_steps=comm.delay_steps if comm.profile == "controlled" else 0)
                group = [r for r in existing if r["num_agents"] == item["num_agents"]]
                both = [r for r in group if r["success"] and baseline[r["case_id"]]["success"]]
                item.update(mean_episode_steps=float(np.mean([r["steps"] for r in group])),
                    common_success_cases=len(both), paired_common_success_makespan_delta=
                        float(np.mean([r["steps"]-baseline[r["case_id"]]["steps"] for r in both])) if both else None,
                    actual_stale_agents=sum(r["actual_stale_agents"] for r in group),
                    exposed_agents=sum(r["exposed_agents"] for r in group))
            write(directory/"summary.json", summary)
            export_csv(path, directory/"episodes.csv")
            all_summaries.extend(summary)
            write(root/"summary.json", all_summaries)
            state["completed_conditions"].append(name)
            write(root/"status.json", state)
        executor.shutdown()
        state.update(status="complete", active_condition=None, finished_utc=utc(), elapsed_seconds=time.monotonic()-start)
        write(root/"status.json", state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status="error", error=repr(error), failed_utc=utc())
        write(root/"status.json", state)
        # This pool owns only its independently frozen evaluation workers.
        for process in (executor._processes or {}).values():
            if process.is_alive(): process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=Path("output/leo-learning/leo-learning-20261009-01-recovery-03"))
    parser.add_argument("--run-id")
    parser.add_argument("--root", type=Path, help="Resume a frozen run by completed case IDs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    if not 1 <= args.workers <= 8:
        parser.error("Use between one and eight inference workers")
    if args.root:
        run(args.root)
    elif args.run_id:
        run(prepare(args))
    else:
        parser.error("--run-id or --root is required")


if __name__ == "__main__":
    main()
