"""Parameter-only pilot: existing allocator and fixed routes at 1/6 GHz UAV."""
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib
import traceback

import numpy as np

from panama_uav.communication_effects import condition_config, measurements, validate_configuration
from panama_uav.communication_report import csv_table, paired_changes
from panama_uav.config import FIXED_MAP, ProjectConfig, load_config
from panama_uav.evaluation import run_episode
from panama_uav.inference_controls import InferenceCondition, inference_class
from panama_uav.records import append_jsonl, sha256, write_json
from panama_uav.resource_study import study_cases
from panama_uav.scenarios import Scenario
from panama_uav.staleness_audit import audit_stress

PROTOCOL = "parameter_only_uav6_v1"
LABEL = "Parameter-only pilot results"
REQUESTED = (2, 4, 8, 16)
CONDITIONS = (
    dict(name="allocated-1ghz", route="allocated", cpu=1e9),
    dict(name="allocated-6ghz", route="allocated", cpu=6e9),
    dict(name="uav-6ghz", route="edge", cpu=6e9),
    dict(name="cloud-6ghz", route="cloud", cpu=6e9),
)
IDENTITY = ("case_id", "num_agents", "seed", "starts", "goals", "orbital_offset_seconds", "map_sha256")


def parameter_change(base, pilot):
    """Reject every parsed configuration change except the declared UAV capacity."""
    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            name = prefix + key
            result.update(flatten(item, name + ".") if isinstance(item, dict) else {name: item})
        return result
    old = flatten(tomllib.loads(Path(base).read_text(encoding="utf-8")))
    new = flatten(tomllib.loads(Path(pilot).read_text(encoding="utf-8")))
    changed = {key: [old.get(key), new.get(key)] for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    if changed != {"resource_allocation.drone_cycles_per_second": [1e9, 6e9]}:
        raise ValueError(f"Pilot must change only UAV capacity 1e9 -> 6e9: {changed}")
    return changed


def population_scope(payload, requested):
    supported = {q["num_agents"] for q in payload.get("qualification", [])
                 if q.get("episodes") == 200 and q.get("success_rate", 0) >= .95}
    actual = [n for n in requested if n in supported]
    excluded = [dict(num_agents=n, reason="No qualification for this population in the frozen checkpoint")
                for n in requested if n not in supported]
    if not actual:
        raise ValueError("No requested population is qualified")
    return actual, excluded


def execution_config(config, condition, root, case):
    configured = replace(config, resource_allocation=replace(config.resource_allocation,
                                                            drone_cycles_per_second=condition["cpu"]))
    return condition_config(configured, InferenceCondition(condition["name"], condition["route"]), root, case)


def pool_profile(events, horizon, pool):
    history = sorted((e for e in events if e["event"] == "compute_allocation"
                      and e["pool"] == pool and 0 <= e["time"] <= horizon), key=lambda e: e["time"])
    used = concurrent = occupied = 0.
    maximum = 0
    for i, event in enumerate(history):
        until = history[i+1]["time"] if i+1 < len(history) else horizon
        duration = max(0., until - event["time"])
        utilization = event["utilization"]
        if not -1e-9 <= utilization <= 1+1e-7:
            raise ValueError("CPU capacity exceeded")
        jobs = len(event["jobs"])
        used += duration * utilization
        concurrent += duration * jobs
        occupied += duration * bool(jobs)
        maximum = max(maximum, jobs)
    return dict(utilization_seconds=used, concurrent_job_seconds=concurrent,
                occupied_seconds=occupied, max_concurrent_jobs=maximum)


def overlaps(left, right, windows):
    return any(left < end and right > start for start, end in windows)


def enrich(row, tasks, rounds, trace):
    """Read-only extraction; production telemetry and decisions are not modified."""
    horizon = trace["frames"][-1]["time"]
    windows = trace["handover_windows"]
    originals = {t["job_id"]: t for t in trace["resource_tasks"]}
    allocations = {e["job_id"]: e for e in trace["remote_events"] if e["event"] == "allocation_event"}
    task_index = {t["job_id"]: t for t in tasks}
    for task in tasks:
        original = originals[task["job_id"]]
        allocation = allocations.get(task["job_id"], {})
        previous = [t for t in allocation.get("snapshot", {}).get("tasks", []) if t["task_id"] != task["job_id"]]
        by_id = {a["task_id"]: a for a in allocation.get("assignments", [])}
        if any(t["fixed_x"] is not None and by_id[t["task_id"]]["x"] != t["fixed_x"] for t in previous):
            raise ValueError("A previous assignment changed")
        source, admitted = original["source_time"], original.get("allocation_at")
        task.update(backhaul_available_at_admission=original.get("backhaul_radio", {}).get("available"),
            fixed_previous_tasks=len(previous),
            fixed_previous_uav=sum(t["fixed_x"] == 1 for t in previous),
            fixed_previous_cloud=sum(t["fixed_x"] == 0 for t in previous),
            handover_at_sensing=any(a <= source < b for a, b in windows),
            handover_at_admission=admitted is not None and any(a <= admitted < b for a, b in windows),
            handover_overlaps_lifetime=overlaps(source, original.get("delivered_at", horizon), windows))
        stages = {k: v for k, v in original.get("service_stages", {}).items()
                  if k.split(":")[0] in ("remote_robot_observation", "remote_cloud_observation", "remote_action_command")}
        for name in ("queue_seconds", "interruption_seconds", "guard_seconds"):
            task["command_path_" + name] = sum(max(0., v[name]) for v in stages.values())
        task["satellite_queue_seconds"] = sum(v["queue_seconds"] for k, v in stages.items() if ":satellite_" in k)
        task["active_peer_aoi_sum"] = 0.
        task["active_peer_aoi_samples"] = 0
    for event in trace["oracle_events"]:
        if event["event"] != "controlled_peer_shadow":
            continue
        task = task_index[event["job_id"]]
        for pair in event["pairs"]:
            if not event["fresh_states"][str(pair["peer"])]["completed"]:
                task["active_peer_aoi_sum"] += pair["peer_age_seconds"]
                task["active_peer_aoi_samples"] += 1
    for pool in ("drone", "cloud"):
        for name, value in pool_profile(trace["transport_events"], horizon, pool).items():
            row[pool + "_" + name] = value
    for cycle in rounds:
        group = [t for t in tasks if t["cycle"] == cycle["cycle"]]
        cycle.update(uav_tasks=sum(t.get("executed_x") == 1 for t in group),
                     cloud_tasks=sum(t.get("executed_x") == 0 for t in group))
    row.update(active_peer_aoi_sum=sum(t["active_peer_aoi_sum"] for t in tasks),
               active_peer_aoi_samples=sum(t["active_peer_aoi_samples"] for t in tasks),
               satellite_attachment_events=sum(e["event"] == "orbit_handover" and
                   e.get("source_window", {}).get("attachment_changed", False) for e in trace["transport_events"]),
               handover_service_windows=len(windows))
    return row, tasks, rounds


def preserved_files(project):
    paths = list((project / "src/panama_uav").glob("*.py"))
    paths.append(project / "configs/paper_remote_dt_250ms.toml")
    for name in ("250ms", "staleness", "deadlines", "communication"):
        paths.extend(p for p in (project / "results" / name).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    return {p.relative_to(project).as_posix(): sha256(p) for p in paths}


def prepare(root, base, pilot, model, reference, *, cases=5, counts=REQUESTED, cap=250, workers=4, budget=3600):
    root, base, pilot, model = [Path(p).resolve() for p in (root, base, pilot, model)]
    change = parameter_change(base, pilot)
    config = load_config(pilot)
    validate_configuration(config)
    if root.exists():
        raise ValueError("Use a new pilot directory; existing results are immutable")
    if not 1 <= cases <= 10 or not 1 <= cap <= 250 or not 1 <= workers <= 8 or budget <= 0:
        raise ValueError("Use bounded diagnostic case, step, worker and runtime limits")
    if len(set(counts)) != len(counts) or not set(counts) <= set(REQUESTED):
        raise ValueError("Pilot populations are 2/4/8/16")
    import torch
    from panama_uav.training import load_model
    torch.set_num_threads(1)
    _, payload = load_model(model, config, torch.device("cpu"), require_qualified=True)
    actual, excluded = population_scope(payload, counts)
    assets = root / "assets"
    (assets / "maps").mkdir(parents=True)
    shutil.copy2(model, assets / "checkpoint.pt")
    shutil.copy2(config.resolve(FIXED_MAP), assets / FIXED_MAP)
    shutil.copy2(config.resolve(config.communication.leo_timeline), assets / "orbit.npz")
    shutil.copy2(base, root / "base.toml")
    shutil.copy2(pilot, root / "pilot.toml")
    frozen = replace(config, project_root=assets,
                     communication=replace(config.communication, leo_timeline=str(assets / "orbit.npz")))
    bank = study_cases(frozen, cases, tuple(actual), cap)
    for case in bank:
        case.update(map_path=str(assets / FIXED_MAP), map_sha256=sha256(assets / FIXED_MAP), scenario="random_mapf")
    reference = Path(reference).resolve()
    previous = {r["case_id"]: r for r in json.loads(reference.read_text(encoding="utf-8"))}
    for case in bank:
        if case["case_id"] not in previous or any(case[k] != previous[case["case_id"]][k] for k in IDENTITY):
            raise ValueError("Historical paired case identity changed")
    shutil.copy2(reference, root / "reference_cases.json")
    write_json(root / "cases.json", bank)
    source = Path(__file__).resolve().parent.parent / "src/panama_uav"
    shutil.copytree(source, root / "code/panama_uav", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, root / "code/run_uav6_pilot.py")
    project = config.project_root
    manifest = dict(protocol=PROTOCOL, label=LABEL, created_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
        parameter_changes=change, configuration=frozen.to_dict(), requested_populations=list(counts),
        populations=actual, excluded_populations=excluded, cases_per_population=cases, episode_cap=cap,
        conditions=list(CONDITIONS), expected_episodes=len(bank)*len(CONDITIONS), workers=workers,
        budget_seconds=budget, model_origin=str(model), model_sha256=sha256(assets / "checkpoint.pt"),
        model_qualification=payload["qualification"], model_map_compatibility=payload["map_compatibility"],
        learner_updates=payload.get("learner_updates", 0), policy_updates=0,
        timeline_sha256=sha256(assets / "orbit.npz"), map_sha256=sha256(assets / FIXED_MAP),
        cases_sha256=sha256(root / "cases.json"), reference_sha256=sha256(root / "reference_cases.json"),
        base_config_sha256=sha256(root / "base.toml"), pilot_config_sha256=sha256(root / "pilot.toml"),
        harness_sha256=sha256(root / "code/run_uav6_pilot.py"),
        code_sha256={p.name: sha256(p) for p in (root / "code/panama_uav").glob("*.py")},
        protected_sha256=preserved_files(project), project_root=str(project),
        inference_device="cpu", cpu_threads_per_worker=1,
        packages={name: importlib.metadata.version(name) for name in
                  ("numpy", "scipy", "torch", "matplotlib", "simpy", "psutil", "cvxpy", "clarabel")})
    write_json(root / "manifest.json", manifest)
    write_json(root / "status.json", dict(status="prepared", completed_episodes=0, expected_episodes=manifest["expected_episodes"]))
    return manifest


def verify_inputs(root):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert m["protocol"] == PROTOCOL
    for name, key in (("assets/checkpoint.pt", "model_sha256"), ("assets/orbit.npz", "timeline_sha256"),
        ("assets/" + FIXED_MAP, "map_sha256"), ("cases.json", "cases_sha256"),
        ("reference_cases.json", "reference_sha256"), ("base.toml", "base_config_sha256"),
        ("pilot.toml", "pilot_config_sha256"), ("code/run_uav6_pilot.py", "harness_sha256")):
        if sha256(root / name) != m[key]:
            raise ValueError(f"Frozen input changed: {name}")
    if any(sha256(root / "code/panama_uav" / name) != value for name, value in m["code_sha256"].items()):
        raise ValueError("Frozen production source changed")
    parameter_change(root / "base.toml", root / "pilot.toml")
    return m


def initialize(root):
    import torch
    from panama_uav.training import load_model
    global ROOT, MANIFEST, CONFIG, MODEL
    ROOT = Path(root)
    MANIFEST = verify_inputs(ROOT)
    CONFIG = ProjectConfig.from_dict(MANIFEST["configuration"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    MODEL, payload = load_model(ROOT / "assets/checkpoint.pt", CONFIG, torch.device("cpu"), require_qualified=True)
    actual, excluded = population_scope(payload, MANIFEST["populations"])
    if excluded or actual != MANIFEST["populations"]:
        raise ValueError("Worker population lacks checkpoint qualification")


def infer(location, robot, observation):
    import torch
    from panama_uav.models import select_action
    return select_action(MODEL, observation, 0., np.random.default_rng(0), torch.device("cpu"))


def episode(argument):
    import torch
    condition, case = argument
    output = ROOT / "episodes" / condition["name"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / f"worker-{os.getpid()}.json", dict(condition=condition["name"], case_id=case["case_id"], status="running"))
    started = time.monotonic()
    try:
        c = InferenceCondition(condition["name"], condition["route"])
        config = execution_config(CONFIG, condition, ROOT, case)
        scenario = Scenario(Path(case["case_id"]), Path(case["map_path"]),
                            tuple(map(tuple, case["starts"])), tuple(map(tuple, case["goals"])))
        row, trace = run_episode(config, MODEL, case, torch.device("cpu"), MANIFEST["learner_updates"],
            max_steps=MANIFEST["episode_cap"], scenario=scenario, remote_communication_class=inference_class(c))
        if trace["frames"][0]["positions"] != case["starts"] or trace["goals"] != case["goals"]:
            raise ValueError("Paired start/goal identity changed")
        row, tasks, rounds = measurements(row, trace, c)
        row, tasks, rounds = enrich(row, tasks, rounds, trace)
        row.update(drone_capacity=condition["cpu"], label=LABEL)
        trace["pilot_condition"] = condition
        trace["episode_metrics"] = row
        trace["provenance"] = dict(protocol=PROTOCOL, manifest_sha256=sha256(ROOT / "manifest.json"),
            model_sha256=MANIFEST["model_sha256"], timeline_sha256=MANIFEST["timeline_sha256"])
        audit = audit_stress(trace, infer)
        archive = output / (case["case_id"] + ".json.gz")
        with gzip.open(archive.with_suffix(".pending"), "wb", compresslevel=3) as stream:
            stream.write(json.dumps(trace, allow_nan=False, separators=(",", ":")).encode())
        os.replace(archive.with_suffix(".pending"), archive)
        audit.update(condition=condition["name"], case_id=case["case_id"], archive=str(archive),
                     archive_sha256=sha256(archive), wall_seconds=time.monotonic()-started)
        packet = dict(row=row, tasks=tasks, rounds=rounds, audit=audit, manifest_sha256=sha256(ROOT / "manifest.json"))
        target = output / (case["case_id"] + ".result.json")
        write_json(target.with_suffix(".pending"), packet)
        os.replace(target.with_suffix(".pending"), target)
        write_json(ROOT / f"worker-{os.getpid()}.json", dict(status="complete", condition=condition["name"], case_id=case["case_id"]))
        return packet
    except Exception as error:
        write_json(output / (case["case_id"] + ".error.json"), dict(error=str(error), traceback=traceback.format_exc(),
            condition=condition, case=case, snapshot=getattr(error, "snapshot", None)))
        raise


def run(root):
    root = Path(root).resolve()
    manifest = verify_inputs(root)
    if (root / "episodes").exists():
        raise ValueError("This bounded pilot cannot overwrite or restart previous episodes")
    bank = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    started, complete = time.monotonic(), 0
    def status(state, **values):
        write_json(root / "status.json", dict(status=state, completed_episodes=complete,
            expected_episodes=manifest["expected_episodes"], wall_seconds=time.monotonic()-started,
            updated_utc=datetime.now(timezone.utc).isoformat(), **values))
    status("running")
    try:
        arguments = iter((condition, case) for case in bank for condition in manifest["conditions"])
        with ProcessPoolExecutor(max_workers=manifest["workers"], initializer=initialize, initargs=(str(root),)) as executor:
            pending = deque()
            for _ in range(manifest["workers"]):
                item = next(arguments, None)
                if item is not None:
                    pending.append(executor.submit(episode, item))
            while pending:
                packet = pending.popleft().result(timeout=max(1., manifest["budget_seconds"]-(time.monotonic()-started)))
                for name, key in (("episodes", "row"), ("tasks", "tasks"), ("rounds", "rounds"), ("audits", "audit")):
                    values = packet[key] if isinstance(packet[key], list) else [packet[key]]
                    for value in values:
                        append_jsonl(root / (name + ".jsonl"), value)
                complete += 1
                status("running", latest_case=packet["row"]["case_id"], latest_condition=packet["row"]["condition"])
                if time.monotonic()-started > manifest["budget_seconds"]:
                    raise TimeoutError("Bounded pilot runtime reached")
                item = next(arguments, None)
                if item is not None:
                    pending.append(executor.submit(episode, item))
        report(root)
        status("complete")
    except Exception as error:
        status("error", error=str(error))
        raise


def describe(tasks):
    delivered = [t["command_delivery_seconds"] for t in tasks if t.get("command_delivery_seconds") is not None]
    predicted = [t["predicted_latency_seconds"] for t in tasks if t.get("predicted_latency_seconds") is not None]
    errors = [t["command_delivery_seconds"]-t["predicted_latency_seconds"] for t in tasks
              if t.get("command_delivery_seconds") is not None and t.get("predicted_latency_seconds") is not None]
    def mean(values):
        return float(np.mean(values)) if values else None
    def quantile(values, q):
        return float(np.quantile(values, q)) if values else None
    uav = sum(t.get("executed_x") == 1 for t in tasks)
    cloud = sum(t.get("executed_x") == 0 for t in tasks)
    active = sum(t["active_peer_aoi_samples"] for t in tasks)
    return dict(tasks=len(tasks), uav_tasks=uav, cloud_tasks=cloud,
        assigned_tasks=uav+cloud, uav_fraction=uav/(uav+cloud) if uav+cloud else None,
        cloud_fraction=cloud/(uav+cloud) if uav+cloud else None,
        on_time=sum(t["deadline_status"] == "on_time" for t in tasks),
        late=sum(t["deadline_status"] == "late" for t in tasks),
        undelivered=sum(t["deadline_status"] == "undelivered" for t in tasks),
        on_time_fraction=sum(t["deadline_status"] == "on_time" for t in tasks)/len(tasks) if tasks else None,
        delivered_tasks=len(delivered), observed_mean_s=mean(delivered), observed_p50_s=quantile(delivered, .5),
        observed_p95_s=quantile(delivered, .95), predicted_snapshot_mean_s=mean(predicted),
        predicted_snapshot_p50_s=quantile(predicted, .5), predicted_snapshot_p95_s=quantile(predicted, .95),
        observed_minus_predicted_mean_s=mean(errors),
        command_queue_mean_s=mean([t["command_path_queue_seconds"] for t in tasks]),
        satellite_queue_mean_s=mean([t["satellite_queue_seconds"] for t in tasks]),
        command_interruption_mean_s=mean([t["command_path_interruption_seconds"] for t in tasks]),
        fixed_previous_mean=mean([t["fixed_previous_tasks"] for t in tasks]),
        fixed_previous_max=max((t["fixed_previous_tasks"] for t in tasks), default=0),
        active_peer_aoi_s=sum(t["active_peer_aoi_sum"] for t in tasks)/active if active else None,
        active_peer_samples=active)


def report(root):
    root = Path(root).resolve()
    manifest = verify_inputs(root)
    bank = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in bank}
    expected = {(c["name"], case) for c in manifest["conditions"] for case in cases}
    packets = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "episodes").glob("*/*.result.json"))]
    keys = [(p["row"]["condition"], p["row"]["case_id"]) for p in packets]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete or duplicated paired case cohort")
    grouped = defaultdict(list)
    for packet in packets:
        row, audit, tasks = packet["row"], packet["audit"], packet["tasks"]
        if packet["manifest_sha256"] != sha256(root / "manifest.json") or audit["status"] != "passed" or sha256(audit["archive"]) != audit["archive_sha256"]:
            raise ValueError("Case provenance or replay failed")
        if any(row[k] != cases[row["case_id"]][k] for k in IDENTITY):
            raise ValueError("Paired case identity changed")
        if len(tasks) != row["tasks_generated"] or len({t["job_id"] for t in tasks}) != len(tasks):
            raise ValueError("Task denominator or unique ID mismatch")
        counts = Counter(t["deadline_status"] for t in tasks)
        if any(counts[state] != row["deadline_commands_" + state] for state in ("on_time", "late", "undelivered")):
            raise ValueError("Exclusive command partition mismatch")
        if row["command_cutoff_misses"] != sum(t["cutoff_induced_wait"] for t in tasks):
            raise ValueError("Cutoff WAIT denominator mismatch")
        grouped[row["condition"], row["num_agents"]].append(packet)
    summary, handover, contrasts = [], [], []
    for (condition, population), group in sorted(grouped.items()):
        rows = [p["row"] for p in group]
        tasks = [t for p in group for t in p["tasks"]]
        cycles = [r for p in group for r in p["rounds"] if r["cycle"] == 0]
        horizon = sum(r["observed_seconds"] for r in rows)
        result = dict(label=LABEL, condition=condition, num_agents=population, cases=len(rows),
            completed_missions=sum(r["success"] for r in rows), **describe(tasks))
        result.update(first_round_uav_mean=float(np.mean([r["uav_tasks"] for r in cycles])),
            first_round_cloud_mean=float(np.mean([r["cloud_tasks"] for r in cycles])),
            makespan_steps=_mean(r["makespan_seconds"] for r in rows),
            waiting_steps=_mean(r["waiting_completed_steps"] for r in rows),
            cutoff_waits=sum(r["command_cutoff_misses"] for r in rows),
            modeled_energy_total_j=sum(r["modeled_energy_lower_j"] for r in rows),
            modeled_energy_mean_episode_j=_mean(r["modeled_energy_lower_j"] for r in rows),
            robot_tx_energy_j=sum(r["robot_tx_energy_j"] for r in rows),
            uav_tx_energy_j=sum(r["drone_tx_energy_j"] for r in rows),
            uav_compute_energy_j=sum(r["drone_compute_energy_j"] for r in rows),
            unavailable_admission_tasks=sum(t["backhaul_available_at_admission"] is False for t in tasks),
            handover_windows=sum(r["handover_service_windows"] for r in rows),
            satellite_attachment_events=sum(r["satellite_attachment_events"] for r in rows))
        for pool in ("drone", "cloud"):
            result[pool+"_utilization"] = sum(r[pool+"_utilization_seconds"] for r in rows)/horizon
            result[pool+"_concurrent_mean"] = sum(r[pool+"_concurrent_job_seconds"] for r in rows)/horizon
            occupied = sum(r[pool+"_occupied_seconds"] for r in rows)
            result[pool+"_concurrent_when_busy"] = sum(r[pool+"_concurrent_job_seconds"] for r in rows)/occupied if occupied else None
            result[pool+"_concurrent_max"] = max(r[pool+"_max_concurrent_jobs"] for r in rows)
        summary.append(result)
        for name, subset in (
            ("handover_at_sensing", [t for t in tasks if t["handover_at_sensing"]]),
            ("outside_handover_at_sensing", [t for t in tasks if not t["handover_at_sensing"]]),
            ("unavailable_at_admission", [t for t in tasks if t["backhaul_available_at_admission"] is False]),
            ("available_at_admission", [t for t in tasks if t["backhaul_available_at_admission"] is True]),
            ("fixed_previous_present", [t for t in tasks if t["fixed_previous_tasks"]]),
            ("no_previous_jobs", [t for t in tasks if not t["fixed_previous_tasks"]])):
            handover.append(dict(label=LABEL, condition=condition, num_agents=population, subset=name, **describe(subset)))
        if condition != "allocated-1ghz":
            baseline = [p["row"] for p in grouped["allocated-1ghz", population]]
            for metric in ("mission_success_rate", "command_on_time_rate", "makespan_seconds", "waiting_steps", "modeled_energy_j"):
                contrasts.append(dict(condition=condition, reference="allocated-1ghz", num_agents=population,
                                      **paired_changes(rows, baseline, metric)))
    for name, values in (("summary", summary), ("handover", handover), ("comparisons", contrasts)):
        csv_table(root / (name + ".csv"), values)
    write_json(root / "summary.json", summary)
    for name, key in (("episodes", "row"), ("tasks", "tasks"), ("rounds", "rounds"), ("audits", "audit")):
        values = [item for p in packets for item in (p[key] if isinstance(p[key], list) else [p[key]])]
        csv_table(root / (name + ".csv"), values)
    project = Path(manifest["project_root"])
    changed = [name for name, digest in manifest["protected_sha256"].items() if sha256(project / name) != digest]
    if changed:
        raise ValueError(f"Protected production or historical files changed: {changed}")
    write_json(root / "validation.json", dict(status="passed", label=LABEL, episodes=len(packets),
        tasks=sum(len(p["tasks"]) for p in packets), replay_audits=len(packets),
        model_qualification_enforced=True, excluded_populations=manifest["excluded_populations"],
        paired_case_identity_verified=True, partition_and_cutoff_denominators_verified=True,
        fixed_previous_assignments_verified=True, protected_files_verified=len(manifest["protected_sha256"]),
        only_parameter_change=manifest["parameter_changes"], policy_updates=0))
    plots(root, summary)
    write_report(root, manifest, summary, contrasts)
    return summary


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def plots(root, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    colors = ("#666666", "#009E73", "#0072B2", "#D55E00")
    labels = ("Allocator 1 GHz", "Allocator 6 GHz", "All UAV 6 GHz", "All cloud")
    for condition, color, label in zip(CONDITIONS, colors, labels):
        group = sorted((r for r in summary if r["condition"] == condition["name"]), key=lambda r: r["num_agents"])
        x = [r["num_agents"] for r in group]
        for ax, field in ((axes[0, 0], "cloud_fraction"), (axes[0, 1], "observed_p95_s"), (axes[1, 1], "on_time_fraction")):
            ax.plot(x, [r[field] for r in group], color=color, marker="o", label=label)
        for pool, style in (("drone", "-"), ("cloud", "--")):
            axes[1, 0].plot(x, [r[pool+"_utilization"] for r in group], color=color, ls=style, marker="o", label=label+" / "+pool)
    axes[0, 0].set(ylabel="Cloud task fraction", ylim=(-.03, 1.03))
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set(ylabel="Observed E2E command p95 (s)")
    axes[0, 1].axhline(.25, color="black", ls=":", lw=1)
    axes[1, 0].set(ylabel="CPU utilization (solid UAV, dashed cloud)", ylim=(-.03, 1.03))
    axes[1, 1].set(ylabel="Commands within 250 ms", ylim=(-.03, 1.03))
    for ax in axes.flat:
        ax.set(xlabel="Robots", xticks=sorted({r["num_agents"] for r in summary}))
        ax.grid(alpha=.2)
    fig.suptitle(LABEL+" — 16 robots outside checkpoint qualification", fontsize=11)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(folder / ("diagnostics."+ext), dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


def write_report(root, manifest, summary, contrasts):
    def fmt(value):
        return "—" if value is None else f"{value:.4g}"
    text = "# Parameter-only pilot results\n\n"
    text += f"{manifest['cases_per_population']} paired cases per supported population; {manifest['expected_episodes']} simulations. "
    text += "The only experimental parameter change is UAV capacity from 1 to 6 GHz. All other configured parameters remain unchanged.\n\n"
    text += "16 robots were excluded: the frozen checkpoint is qualified for 2/4/6/8 robots, with no 16-robot qualification. No unqualified model was substituted.\n\n"
    text += "| Robots | Condition | UAV/cloud tasks | Cloud % | First-round UAV/cloud | E2E mean / p50 / p95 (ms) | On time % | Missions | Makespan | Waiting | Energy/episode J |\n"
    text += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for r in sorted(summary, key=lambda r: (r["num_agents"], [c["name"] for c in CONDITIONS].index(r["condition"]))):
        values = (r["num_agents"], r["condition"], f"{r['uav_tasks']}/{r['cloud_tasks']}", fmt(100*r["cloud_fraction"]),
            f"{fmt(r['first_round_uav_mean'])}/{fmt(r['first_round_cloud_mean'])}",
            "/".join(fmt(1000*r[k]) for k in ("observed_mean_s", "observed_p50_s", "observed_p95_s")),
            fmt(100*r["on_time_fraction"]), f"{r['completed_missions']}/{r['cases']}",
            fmt(r["makespan_steps"]), fmt(r["waiting_steps"]), fmt(r["modeled_energy_mean_episode_j"]))
        text += "| "+" | ".join(map(str, values))+" |\n"
    proposed = sorted((r for r in summary if r["condition"] == "allocated-6ghz"), key=lambda r: r["num_agents"])
    text += "\n## Placement evidence\n\n"
    text += "Measured cloud fractions under the 6 GHz allocator: "+", ".join(f"{r['num_agents']} robots: {100*r['cloud_fraction']:.2f}%" for r in proposed)+". "
    increasing = all(a["cloud_fraction"] < b["cloud_fraction"] for a, b in zip(proposed, proposed[1:]))
    text += ("Cloud share increases across the tested populations. " if increasing else "Cloud share does not increase strictly across all tested populations. ")
    text += "The 16-robot cloud-dominance hypothesis remains untested. First-round counts are reported separately from mission-wide task percentages; admissions are sequential and existing assignments remain fixed.\n\n"
    text += "## Timing, contention and handover diagnostics\n\n"
    text += "`summary.csv` separates predicted solver-snapshot latency from observed sensing-to-robot command latency, including means and p50/p95. The snapshot does not include future arrivals, changing CPU shares, complete wire serialization or all response queuing. `handover.csv` compares tasks sensed within native handover/service windows, outside those windows, admitted with unavailable/available backhaul, and with/without prior fixed jobs. These are exposure associations; no handover model was disabled.\n\n"
    text += "CPU utilization integrates actual allocated rates over simulated time; concurrent-job means and maxima come from existing compute-allocation events. All-UAV/cloud controls reuse the existing forced binary route harness. The cloud route can queue through temporary backhaul unavailability. Active-peer AoI excludes peers marked completed in the fresh physical reference at inference; its sample denominator is saved.\n\n"
    text += "Energy includes robot uplink, UAV backhaul uplink and UAV dynamic compute. Cloud electricity, downlink power, receivers, static power and propulsion remain unmodeled. Increasing the UAV capacity does not force it to run at the limit: the existing energy-aware processor-sharing rule can leave capacity idle.\n\n"
    text += "## Provenance and limitations\n\n"
    text += f"Frozen checkpoint SHA-256: `{manifest['model_sha256']}`. Archived timeline SHA-256: `{manifest['timeline_sha256']}`. "
    text += f"Source base: `{manifest['source_commit']}` plus the hash-recorded isolated pilot harness. `manifest.json` stores complete conditions, case identities, config/source/input hashes and package versions. No policy updates occur.\n\n"
    text += "This small, single-map diagnostic is parameter-only pilot evidence, not validated paper findings. Existing solver, production source, configurations and historical result bundles are hash-checked without edits. All episodes require causal replay and frozen-model action verification. `comparisons.csv` contains case-paired 95% bootstrap intervals against the 1 GHz allocator. Sixteen-robot suitability cannot be established with this checkpoint.\n\n"
    text += "Exact preparation/launch commands and external-input requirements are documented in `UAV6_PILOT.md` in the repository. Original per-case archives and the frozen source/input snapshot remain local.\n"
    (root / "REPORT.md").write_text(text, encoding="utf-8")


def launch(root):
    import psutil
    root = Path(root).resolve()
    verify_inputs(root)
    if (root / "launch.json").exists():
        raise ValueError("A launch is already recorded; inspect it rather than duplicating workers")
    env = dict(os.environ, PYTHONPATH=str(root / "code"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, str(root / "code/run_uav6_pilot.py"), "--worker", str(root)]
    with (root / "stdout.log").open("a") as out, (root / "stderr.log").open("a") as err:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=out, stderr=err,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    record = dict(pid=process.pid, creation_time=psutil.Process(process.pid).create_time(), command=command)
    write_json(root / "launch.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--base", type=Path, default=Path("configs/paper_remote_dt_250ms.toml"))
    parser.add_argument("--config", type=Path, default=Path("configs/paper_remote_dt_250ms_uav6.toml"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/inputs/final_model.pt"))
    parser.add_argument("--reference-cases", type=Path, default=Path("results/communication/cases.json"))
    parser.add_argument("--counts", type=int, nargs="+", default=REQUESTED)
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--cap", type=int, default=250)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-seconds", type=int, default=3600)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        return run(args.worker)
    if args.root is None:
        parser.error("--root is required")
    if args.prepare:
        manifest = prepare(args.root, args.base, args.config, args.model, args.reference_cases,
            cases=args.cases, counts=args.counts, cap=args.cap, workers=args.workers, budget=args.budget_seconds)
        print(json.dumps(dict(expected_episodes=manifest["expected_episodes"], populations=manifest["populations"],
                              excluded=manifest["excluded_populations"])))
    if args.launch:
        print(json.dumps(launch(args.root)))
    if args.report:
        report(args.root)


if __name__ == "__main__":
    main()
