"""Read-only, case-clustered analysis of the accepted 250 ms remote-DT study."""
from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np

from .allocation import priority_weights
from .config import FIXED_MAP, ProjectConfig
from .deadline_study import task_location, task_statistics
from .records import read_jsonl, sha256, write_json
from .remote_audit import replay_trace
from .remote_state import observation_hash

BASELINE = "up-1-cpu-1__equal"
PRIMARY = "up-1-cpu-1__priority"
BASES = ("communicated", "physical")
LEVELS = ("low", "middle", "high", "unknown")
MISSION_METRICS = ("success", "restricted_makespan_seconds", "successful_makespan", "waiting_steps",
    "command_cutoff_misses", "mean_task_latency_seconds", "task_latency_p50_seconds", "task_latency_p95_seconds",
    "task_latency_p99_seconds", "executed_drone_tasks", "executed_cloud_tasks", "mean_drone_cpu_utilization",
    "mean_cloud_cpu_utilization", "modeled_energy_lower_j", "modeled_energy_upper_j", "radio_energy_lower_j",
    "drone_compute_energy_j", "radio_queue_seconds", "mean_cpu_zero_rate_wait_seconds",
    "mean_cpu_shared_elapsed_seconds", "mean_cpu_capacity_floor_excess_seconds", "attempted_vertex_conflicts",
    "attempted_edge_swaps", "physical_conflict_events", "missed_vertex_conflicts", "false_positive_vertex_conflicts",
    "decision_action_shadow_disagreements", "decision_action_shadow_cutoff_miss_disagreements")
RATIOS = {"on_time_rate": ("deadline_commands_on_time", "deadline_tasks_monitored"),
    "late_rate": ("deadline_commands_late", "deadline_tasks_monitored"),
    "undelivered_rate": ("deadline_commands_undelivered", "deadline_tasks_monitored"),
    "deadline_compliance": ("deadline_commands_on_time", "deadline_tasks_assessed"),
    "drone_fraction": ("offload_drone_tasks", "offloading_assigned_tasks"),
    "cloud_fraction": ("offload_cloud_tasks", "offloading_assigned_tasks"),
    "xi_fov_supported": ("fov_errors", "fov_relevant_pairs"),
    "xi_vertex_supported": ("vertex_e2e_errors", "vertex_relevant_pairs"),
    "cutoff_action_disagreement_rate": ("decision_action_shadow_disagreements", "decision_action_shadow_comparisons")}


def level(weight):
    """Fixed reference-weight bins; never stratify equal policy by its equal weights."""
    if weight is None:
        return "unknown"
    if not np.isfinite(weight) or not 1 <= weight <= 4:
        raise ValueError("Reference priority weights must lie in [1,4]")
    return "high" if weight >= 3 else "middle" if weight >= 2 else "low"


def reference_weights(priorities, completed):
    # Match the float-valued simulation configuration. Integer bounds would
    # make np.full in the existing mapper truncate fractional reference weights.
    return priority_weights(priorities,completed,minimum=1.0,maximum=4.0)


def paired_interval(rows, reference, metric, *, ratio=None, samples=2000, seed=42):
    """Resample matched CASES, including zero-exposure cases for pooled rates."""
    left = {r["case_id"]:r for r in rows}
    right = {r["case_id"]:r for r in reference}
    if len(left) != len(rows) or len(right) != len(reference) or set(left) != set(right):
        raise ValueError("Paired inference requires the same unique cases")
    keys = sorted(left)
    if ratio is None:
        keys = [k for k in keys if left[k].get(metric) is not None and right[k].get(metric) is not None]
        if not keys:
            return dict(metric=metric, estimate=None, ci95_low=None, ci95_high=None, pairs=0,
                excluded_pairs=len(left), bootstrap_valid=0, estimator="mean paired case difference")
        case_deltas = np.array([float(left[k][metric])-float(right[k][metric]) for k in keys])
        estimate = float(case_deltas.mean())
        evaluate = lambda indices: case_deltas[indices].mean(axis=1)
    else:
        numerator, denominator = ratio
        ln = np.array([left[k][numerator] or 0 for k in keys],float)
        ld = np.array([left[k][denominator] or 0 for k in keys],float)
        rn = np.array([right[k][numerator] or 0 for k in keys],float)
        rd = np.array([right[k][denominator] or 0 for k in keys],float)
        if any(not np.all(np.isfinite(v)) or np.any(v < 0) for v in (ln,ld,rn,rd)) or np.any(ln > ld) or np.any(rn > rd):
            raise ValueError("Rate counts require finite nonnegative numerator <= denominator")
        estimate = float(ln.sum()/ld.sum()-rn.sum()/rd.sum()) if ld.sum() and rd.sum() else None
        def evaluate(indices):
            a,b = ld[indices].sum(1),rd[indices].sum(1)
            good = (a > 0) & (b > 0)
            return ln[indices].sum(1)[good]/a[good]-rn[indices].sum(1)[good]/b[good]
    values = np.array([])
    if len(keys) > 1 and estimate is not None:
        indices = np.random.default_rng(seed).integers(0,len(keys),(samples,len(keys)))
        values = evaluate(indices)
    bounds = np.quantile(values,[.025,.975]).tolist() if len(values) else [None,None]
    return dict(metric=metric, estimate=estimate, ci95_low=bounds[0], ci95_high=bounds[1], pairs=len(keys),
        excluded_pairs=len(left)-len(keys), bootstrap_valid=len(values),
        estimator="difference of pooled task/pair rates, case-cluster bootstrap" if ratio else "mean paired case difference")


def cpu_intervals(events, horizon):
    """Processor sharing has zero-rate waits, not an invented FIFO queue."""
    per_job = defaultdict(lambda: dict(zero_rate_wait_seconds=0., positive_service_seconds=0.))
    pools = {}
    for name in ("drone", "cloud"):
        allocations = [e for e in events if e["event"] == "compute_allocation" and e["pool"] == name]
        area = jobs_area = 0.
        for index,event in enumerate(allocations):
            if not np.isfinite(event["utilization"]) or not -1e-9 <= event["utilization"] <= 1+1e-8:
                raise ValueError("CPU utilization violates pool capacity")
            end = min(horizon,allocations[index+1]["time"] if index+1 < len(allocations) else horizon)
            duration = max(0.,end-event["time"])
            area += duration*event["utilization"]
            jobs_area += duration*len(event["jobs"])
            for job in event["jobs"]:
                field = "positive_service_seconds" if job["cycles_per_second"] > 0 else "zero_rate_wait_seconds"
                per_job[job["job_id"]][field] += duration
        pools[name] = dict(utilization=area/horizon if horizon else 0.,mean_resident_jobs=jobs_area/horizon if horizon else 0.)
    return dict(per_job),pools


def _initialize(model_path, configuration):
    import torch
    from .training import load_model
    global _MODEL, _DEVICE
    torch.set_num_threads(1)
    _DEVICE = torch.device("cpu")
    cfg = ProjectConfig.from_dict(configuration)
    _MODEL,_ = load_model(model_path,cfg,_DEVICE,require_qualified=False)


def _infer(location, robot, observation):
    from .models import select_action
    return select_action(_MODEL,observation,0,np.random.default_rng(0),_DEVICE)


def inspect_episode(argument):
    path,row,configuration,identity = argument
    cfg = ProjectConfig.from_dict(configuration)
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        trace = json.load(stream)
    if any(trace["case"][k] != row[k] for k in trace["case"]):
        raise ValueError("Episode archive case mismatch")
    if any(trace["provenance"].get(k) != v for k,v in identity.items()):
        raise ValueError("Episode provenance differs from shared study inputs")
    audit = replay_trace(trace,_infer)
    events,tasks = trace["remote_events"],trace["resource_tasks"]
    if trace["frames"][0]["positions"] != row["starts"]:
        raise ValueError("Archived physical starts differ from paired case")
    for motion in (e for e in events if e["event"] == "motion_event"):
        before = {k:tuple(v) for k,v in motion["before"].items()}
        after = {k:tuple(v) for k,v in motion["after"].items()}
        if len(set(after.values())) != len(after) or any(
            before[a] == after[b] and before[b] == after[a] and before[a] != before[b]
            for a in after for b in after if a < b):
            raise ValueError("Post-resolution physical overlap or swap")
    arrivals = defaultdict(list)
    for event in events:
        if event["event"] == "command_arrived":
            arrivals[event["command"]["job_id"]].append(event["arrived_at"])
    for task in tasks:
        first = min(arrivals[task["job_id"]]) if arrivals[task["job_id"]] else None
        if first != task.get("delivered_at") or abs(task["deadline_at"]-task["source_time"]-.25) > 1e-12:
            raise ValueError("First command arrival or task deadline mismatch")
    horizon = trace["frames"][-1]["time"]
    statistics = task_statistics(tasks,horizon)
    for key in ("deadline_tasks_monitored", "deadline_commands_on_time", "deadline_commands_late",
                "deadline_commands_undelivered", "task_deadline_misses", "deadline_cutoff_wait_tasks"):
        if statistics[key] != row[key]:
            raise ValueError(f"Deadline denominator/count mismatch: {key}")
    active = {(e["cycle"],e["robot_id"]):e for e in events if e["event"] == "decision_event" and not e["completed_local_control"]}
    shadows = [e for e in trace["oracle_events"] if e["event"] == "decision_action_shadow"]
    if len(shadows) != len(active) or {(e["cycle"],e["robot_id"]) for e in shadows} != set(active):
        raise ValueError("Missing or duplicate active cutoff oracle")
    for shadow in shadows:
        decision = active[(shadow["cycle"],shadow["robot_id"])]
        if (observation_hash(shadow["observation"]) != shadow["observation_hash"] or
            _infer("cloud",shadow["robot_id"],shadow["observation"]) != shadow["oracle_action"] or
            decision["action"] != shadow["actual_action"] or
            decision["command_cutoff_missed"] != shadow["command_cutoff_missed"] or
            decision["cutoff_time"] != shadow["cutoff_time"]):
            raise ValueError("Cutoff action/oracle audit failed")
    states = {e["event_id"]:e["state"] for e in events if e["event"] == "dt_state_event"}
    allocations = {e["job_id"]:e for e in events if e["event"] == "allocation_event"}
    truth = {e["new_job"]:e for e in trace["oracle_events"] if e["event"] == "allocation_shadow"}
    physical = {}
    for event in events:
        if event["event"] == "robot_state_event":
            physical[str(event["report"]["robot_id"])] = event["physical"]
        elif event["event"] == "motion_event":
            physical = event["physical"]
        elif event["event"] == "allocation_event":
            shadow = truth[event["job_id"]]
            ids = sorted(physical,key=int)
            expected = reference_weights([physical[a]["priority"] for a in ids],
                [physical[a]["completed"] for a in ids])
            if (set(ids) != set(shadow["physical_weights"]) or
                any(physical[a]["priority"] != shadow["physical_priorities"][a] for a in ids) or
                not np.allclose(expected,[shadow["physical_weights"][a] for a in ids],rtol=0,atol=1e-12)):
                raise ValueError("Physical priority reference differs from replayed physical states")
    service,pools = cpu_intervals(trace["transport_events"],horizon)
    enriched = []
    for task in tasks:
        job,robot = task["job_id"],str(task["agent"])
        assignment = allocations.get(job)
        dt_weight = physical_weight = dt_priority = physical_priority = None
        dt_active = physical_active = None
        if assignment:
            ids = sorted(assignment["dt_priorities"],key=int)
            if any(assignment["dt_priorities"][a] != states[assignment["priority_versions"][a]]["priority"] for a in ids):
                raise ValueError("Communicated priority reference differs from delivered DT states")
            weights = reference_weights([assignment["dt_priorities"][a] for a in ids],
                [states[assignment["priority_versions"][a]]["completed"] for a in ids])
            dt_weight = float(weights[ids.index(robot)])
            dt_priority = assignment["dt_priorities"][robot]
            physical_weight = truth[job]["physical_weights"][robot]
            physical_priority = truth[job]["physical_priorities"][robot]
            dt_active = sum(not states[assignment["priority_versions"][a]]["completed"] for a in ids)
            physical_active = sum(v < np.size(trace["grid"]) for v in truth[job]["physical_priorities"].values())
        elapsed = task.get("compute_elapsed_seconds")
        capacity = (cfg.resource_allocation.drone_cycles_per_second if task_location(task) == "drone"
                    else cfg.resource_allocation.cloud_cycles_per_second)
        ideal = row["policy"] == "fresh"
        detail = dict(case_id=row["case_id"],condition=row["condition"],num_agents=row["num_agents"],
            job_id=job,robot_id=task["agent"],source_time=task["source_time"],
            allocation_at=task.get("allocation_at"),location=task_location(task),
            communicated_priority=dt_priority,physical_priority=physical_priority,
            communicated_reference_weight=dt_weight,physical_reference_weight=physical_weight,
            communicated_active_robots=dt_active,physical_active_robots=physical_active,
            communicated_level=level(dt_weight),physical_level=level(physical_weight),
            actual_resource_weight=task.get("weight"),deadline_status=task["deadline_status"],
            deadline_met=task["deadline_met"],deadline_at=task["deadline_at"],
            delivered_at=task.get("delivered_at"),latency_seconds=task.get("observed_latency_seconds"),
            predicted_latency_seconds=task.get("predicted_latency_seconds"),
            cutoff_wait=task["cutoff_induced_wait"],deadline_censored=task["deadline_censored"],
            cpu_zero_rate_wait_seconds=service.get(job,{}).get("zero_rate_wait_seconds",0.) if task.get("compute_arrived_at") is not None else None,
            cpu_elapsed_seconds=elapsed,
            cpu_elapsed_censored=task.get("compute_arrived_at") is not None and elapsed is None,
            cpu_observed_elapsed_seconds=(task.get("compute_completed_at",horizon)-task["compute_arrived_at"])
                if task.get("compute_arrived_at") is not None else None,
            cpu_capacity_floor_excess_seconds=max(0.,elapsed-task["required_cycles"]/capacity) if elapsed is not None and not ideal else None)
        enriched.append(detail)
    extra = {}
    for field,target in (("cpu_zero_rate_wait_seconds","mean_cpu_zero_rate_wait_seconds"),
                         ("cpu_elapsed_seconds","mean_cpu_shared_elapsed_seconds"),
                         ("cpu_capacity_floor_excess_seconds","mean_cpu_capacity_floor_excess_seconds")):
        values = [t[field] for t in enriched if t[field] is not None]
        extra[target] = float(np.mean(values)) if values else None
    conflicts = [c for e in events if e["event"] in ("physical_conflict","physical_collision") for c in e["conflicts"]]
    extra.update(attempted_vertex_conflicts=sum(c["kind"] == "vertex" for c in conflicts),
        attempted_edge_swaps=sum(c["kind"] == "edge_swap" for c in conflicts))
    for pool,metrics in pools.items():
        if abs(metrics["utilization"]-row[f"mean_{pool}_cpu_utilization"]) > 1e-10:
            raise ValueError("CPU utilization integration mismatch")
        extra[f"mean_{pool}_resident_jobs"] = metrics["mean_resident_jobs"]
    audit = {k:v for k,v in audit.items() if k not in ("pair_consistency","assignments")}
    audit.update(case_id=row["case_id"],condition=row["condition"],cutoff_oracles_verified=len(shadows),archive_sha256=sha256(path))
    return dict(row,**statistics,**extra),enriched,audit


def priority_case_rows(row, tasks):
    output = []
    for basis in BASES:
        for name in LEVELS:
            group = [t for t in tasks if t[f"{basis}_level"] == name]
            delivered = [t["latency_seconds"] for t in group if t["delivered_at"] is not None]
            on_time = sum(t["deadline_status"] == "on_time" for t in group)
            assessed = sum(t["deadline_met"] is not None for t in group)
            output.append(dict(case_id=row["case_id"],condition=row["condition"],num_agents=row["num_agents"],
                basis=basis,priority_level=name,tasks=len(group),on_time=on_time,
                late=sum(t["deadline_status"] == "late" for t in group),
                undelivered=sum(t["deadline_status"] == "undelivered" for t in group),assessed=assessed,
                compliance=on_time/assessed if assessed else None,
                mean_latency=float(np.mean(delivered)) if delivered else None,
                latency_p50=float(np.quantile(delivered,.5)) if delivered else None,
                latency_p95=float(np.quantile(delivered,.95)) if delivered else None,
                latency_p99=float(np.quantile(delivered,.99)) if delivered else None,
                cutoff_waits=sum(t["cutoff_wait"] is True for t in group),
                drone_tasks=sum(t["location"] == "drone" for t in group),
                cloud_tasks=sum(t["location"] == "cloud" for t in group)))
    return output


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def _bounded_inspections(executor, arguments, window):
    iterator = iter(arguments)
    pending = deque()
    for _ in range(window):
        item = next(iterator,None)
        if item is None:
            break
        pending.append(executor.submit(inspect_episode,item))
    while pending:
        yield pending.popleft().result()
        item = next(iterator,None)
        if item is not None:
            pending.append(executor.submit(inspect_episode,item))


def comparisons(episodes, strata, samples=2000):
    results = []
    for count in sorted({r["num_agents"] for r in episodes}):
        baseline = [r for r in episodes if r["num_agents"] == count and r["condition"] == BASELINE]
        for name in dict.fromkeys(r["condition"] for r in episodes):
            rows = [r for r in episodes if r["num_agents"] == count and r["condition"] == name]
            for metric in (*MISSION_METRICS,*RATIOS):
                result = paired_interval(rows,baseline,metric,ratio=RATIOS.get(metric),samples=samples)
                results.append(dict(condition=name,reference=BASELINE,num_agents=count,scope="mission",
                    primary_comparison=name == PRIMARY,**result))
            for basis in BASES:
                for group in LEVELS:
                    a = [r for r in strata if r["num_agents"] == count and r["condition"] == name and r["basis"] == basis and r["priority_level"] == group]
                    b = [r for r in strata if r["num_agents"] == count and r["condition"] == BASELINE and r["basis"] == basis and r["priority_level"] == group]
                    for metric,ratio in (("priority_compliance",("on_time","assessed")),("mean_latency",None),
                                         ("latency_p95",None),("cutoff_waits",None)):
                        results.append(dict(condition=name,reference=BASELINE,num_agents=count,scope="priority",
                            basis=basis,priority_level=group,primary_comparison=name == PRIMARY,
                            **paired_interval(a,b,metric,ratio=ratio,samples=samples)))
    return results


def preferential_service(strata, samples=2000):
    rows = []
    for count in sorted({r["num_agents"] for r in strata}):
        for basis in BASES:
            for name in (PRIMARY,BASELINE):
                records = [r for r in strata if r["num_agents"] == count and r["basis"] == basis and r["condition"] == name]
                high = {r["case_id"]:r for r in records if r["priority_level"] == "high"}
                low = {r["case_id"]:r for r in records if r["priority_level"] == "low"}
                for case in high:
                    rows.append(dict(case_id=case,condition=name,num_agents=count,basis=basis,
                        high_minus_low_latency=high[case]["mean_latency"]-low[case]["mean_latency"]
                            if high[case]["mean_latency"] is not None and low[case]["mean_latency"] is not None else None,
                        high_minus_low_compliance=high[case]["compliance"]-low[case]["compliance"]
                            if high[case]["compliance"] is not None and low[case]["compliance"] is not None else None))
    differences = []
    for count in sorted({r["num_agents"] for r in rows}):
        for basis in BASES:
            actual = [r for r in rows if r["num_agents"] == count and r["basis"] == basis and r["condition"] == PRIMARY]
            equal = [r for r in rows if r["num_agents"] == count and r["basis"] == basis and r["condition"] == BASELINE]
            for metric in ("high_minus_low_latency","high_minus_low_compliance"):
                differences.append(dict(num_agents=count,basis=basis,condition=PRIMARY,reference=BASELINE,
                    **paired_interval(actual,equal,metric,samples=samples)))
    return rows,differences


def report_priority_study(root, model_path, *, workers=1, samples=2000, sensitivity_path=None):
    root = Path(root).resolve()
    manifest = json.loads((root/"manifest.json").read_text())
    cfg = ProjectConfig.from_dict(manifest["configuration"])
    p = cfg.resource_allocation
    if (p.execution_mode != "remote_dt" or cfg.communication.seconds_per_step != 1 or p.enforce_task_deadlines
        or (p.decision_budget_seconds,p.command_cutoff_seconds,p.task_deadline_seconds) != (.25,.25,.25)
        or (p.weight_min,p.weight_max) != (1,4) or not p.oracle_shadow):
        raise ValueError("Priority evaluation requires the unchanged 250 ms protocol with reference weights 1–4 and oracle diagnostics")
    if json.loads((root/"status.json").read_text())["status"] != "complete":
        raise ValueError("All study conditions must finish before the research report")
    model_path = Path(model_path).resolve()
    if sha256(model_path) != manifest["model"]["sha256"]:
        raise ValueError("Checkpoint identity differs from evaluation")
    from .training import load_model
    import torch
    load_model(model_path,cfg,torch.device("cpu"),require_qualified=not manifest["diagnostic"])
    if not 1 <= workers <= 8 or not 100 <= samples <= 10000:
        raise ValueError("Use 1–8 audit workers and 100–10000 bootstrap samples")
    if sha256(cfg.resolve(FIXED_MAP)) != manifest["map_sha256"] or sha256(cfg.resolve(cfg.communication.leo_timeline)) != manifest["timeline_sha256"]:
        raise ValueError("Map or orbital asset identity changed")
    expected = ("fresh",BASELINE,"up-1-cpu-1__static","up-1-cpu-1__aoi",PRIMARY,"up-1-cpu-1__equal_legacy")
    if tuple(c["name"] for c in manifest["conditions"]) != expected:
        raise ValueError("All six nominal treatments are required")
    from .resource_study import conditions
    for (name,condition,tags),record in zip(conditions(cfg),manifest["conditions"]):
        condition = replace(condition,resource_allocation=replace(condition.resource_allocation,
            failure_directory=str(root/name/"failures")))
        if json.loads(json.dumps(condition.to_dict())) != record["configuration"] or tags != record["tags"]:
            raise ValueError("Unexpected treatment configuration change")
    if not manifest["diagnostic"] and (not p.remote_acceptance_approved or manifest["cases_per_count"] != 50
        or manifest["counts"] != [2,4,6,8] or manifest["episode_cap"] != 250):
        raise ValueError("Full research report requires accepted 50-case study at all four populations")
    bank = json.loads((root/"held_out_cases.json").read_text())
    if sha256(root/"held_out_cases.json") != manifest["case_bank_sha256"]:
        raise ValueError("Paired case bank hash mismatch")
    case_map = {r["case_id"]:r for r in bank}
    identity = {k:manifest[k] for k in ("git_commit","code_sha256","model","map_sha256","timeline_sha256","case_bank_sha256")}
    arguments = []
    for condition in manifest["conditions"]:
        rows = read_jsonl(root/condition["name"]/"episodes.jsonl")
        if len(rows) != len(case_map) or {r["case_id"] for r in rows} != set(case_map):
            raise ValueError("Missing or duplicate paired treatment cases")
        for row in rows:
            if any(row[k] != v for k,v in case_map[row["case_id"]].items()):
                raise ValueError("Paired seeds, starts/goals or orbital offsets differ")
            arguments.append((root/condition["name"]/(row["case_id"]+".json.gz"),row,cfg.to_dict(),identity))
    output = root/"priority_analysis"
    output.mkdir(exist_ok=True)
    write_json(output/"status.json",dict(status="running",completed=0,planned=len(arguments)))
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit = None
    write_json(output/"methodology.json",dict(source_run_commit=manifest["git_commit"],analysis_commit=commit,
        analysis_sha256=sha256(__file__),bootstrap_samples=samples,bootstrap_seed=42,
        bootstrap_unit="paired case; never independent tasks",primary_comparison=[PRIMARY,BASELINE],
        priority_bins="fixed reference weight [1,2): low; [2,3): middle; [3,4]: high; missing: unknown",
        stratification="communicated pre-policy PANAMA reference and separate physical oracle reference at admission",
        note="Post-treatment strata differ across trajectories; service contrasts are descriptive, not a causal same-task effect"))
    if sensitivity_path:
        sensitivity_path = Path(sensitivity_path).resolve()
        evidence = json.loads((sensitivity_path/"manifest.json").read_text())
        if any(evidence[key] != manifest[key] for key in ("map_sha256","timeline_sha256")) or evidence["model_sha256"] != manifest["model"]["sha256"]:
            raise ValueError("Sensitivity evidence asset mismatch")
        write_json(output/"sensitivity_evidence.json",dict(path=str(sensitivity_path),
            summary_sha256=sha256(sensitivity_path/"summary.csv"),use="separate pilot context; never pooled into full-study inference"))
    episodes,strata,audits = [],[],[]
    executor = None
    try:
        if workers > 1:
            executor = ProcessPoolExecutor(max_workers=workers,initializer=_initialize,initargs=(str(model_path),cfg.to_dict()))
            # Bounded batches keep large decoded archives out of the parent.
            results = _bounded_inspections(executor,arguments,2*workers)
        else:
            _initialize(str(model_path),cfg.to_dict())
            results = map(inspect_episode,arguments)
        with (output/"tasks_by_priority.csv").open("w",newline="",encoding="utf-8") as stream:
            writer = None
            for row,tasks,audit in results:
                if writer is None and tasks:
                    writer = csv.DictWriter(stream,fieldnames=list(tasks[0]))
                    writer.writeheader()
                if tasks:
                    writer.writerows(tasks)
                episodes.append(row)
                strata.extend(priority_case_rows(row,tasks))
                audits.append(audit)
                write_json(output/"status.json",dict(status="running",completed=len(episodes),planned=len(arguments)))
        differences = comparisons(episodes,strata,samples)
        service_rows,service_differences = preferential_service(strata,samples)
        for name,rows in (("episodes",episodes),("priority_case_metrics",strata),("paired_treatment_differences",differences),
                          ("preferential_service_cases",service_rows),("preferential_service_differences",service_differences),
                          ("replay_audits",audits)):
            _write_csv(output/f"{name}.csv",rows)
        write_json(output/"validation.json",dict(status="passed",episodes=len(episodes),
            actual_model_actions_verified=sum(r["model_actions_verified"] for r in audits),
            cutoff_oracle_actions_verified=sum(r["cutoff_oracles_verified"] for r in audits),
            ordering_pairs_verified=sum(r["ordering_pairs_verified"] for r in audits)))
        _plots_and_report(output,manifest,episodes,strata,differences,service_differences)
        latency_distributions(output,episodes)
    except Exception as exc:
        write_json(output/"status.json",dict(status="failed",completed=len(episodes),error=str(exc)))
        raise
    finally:
        if executor:
            executor.shutdown(wait=True,cancel_futures=True)
    write_json(output/"status.json",dict(status="complete",episodes=len(episodes)))
    return differences


def _plots_and_report(output, manifest, episodes, strata, differences, service):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = sorted({r["num_agents"] for r in episodes})
    primary = [r for r in differences if r["primary_comparison"] and r["scope"] == "mission"]
    for metric in ("success","restricted_makespan_seconds","waiting_steps","on_time_rate","command_cutoff_misses",
                   "modeled_energy_lower_j","xi_fov_supported","cutoff_action_disagreement_rate"):
        group = [next(r for r in primary if r["num_agents"] == n and r["metric"] == metric) for n in counts]
        fig,ax = plt.subplots(figsize=(6,4),layout="constrained")
        for n,row in zip(counts,group):
            if row["estimate"] is None:
                continue
            ax.plot(n,row["estimate"],"o",color="tab:blue")
            if row["ci95_low"] is not None:
                ax.vlines(n,row["ci95_low"],row["ci95_high"],color="tab:blue")
        ax.axhline(0,color="black",lw=.7)
        ax.set(xticks=counts,xlabel="Robots",ylabel=f"Current priority − matched equal: {metric}",
            title="250 ms paired case differences (95% CI)")
        ax.grid(alpha=.2)
        fig.savefig(output/f"paired_{metric}.png",dpi=160,bbox_inches="tight")
        fig.savefig(output/f"paired_{metric}.svg",bbox_inches="tight")
        plt.close(fig)
    lines = ["# "+("Diagnostic" if manifest["diagnostic"] else "Full")+" 250 ms priority-aware evaluation", "",
        f"Run commit: `{manifest['git_commit']}`. Cases per population: {manifest['cases_per_count']}. "
        f"Diagnostic: {manifest['diagnostic']}. Episodes: {len(episodes)}.", "",
        "Primary contrasts are current-priority minus matched equal-weight allocation. "
        "95% percentile intervals resample matched cases, preserving all robots/tasks within a case. "
        "Successful makespan is conditional on both missions succeeding; capped duration includes failures. "
        "Pooled rates preserve task/pair numerator and denominator counts; unsupported rates remain missing.", "",
        "## Primary mission comparison", "",
        "| Robots | Metric | Difference | 95% CI | Paired cases |", "| ---: | --- | ---: | --- | ---: |"]
    report_metrics = ("success","restricted_makespan_seconds","waiting_steps","on_time_rate",
        "command_cutoff_misses","task_latency_p95_seconds","modeled_energy_lower_j",
        "xi_fov_supported","cutoff_action_disagreement_rate")
    for row in (r for r in primary if r["metric"] in report_metrics):
        estimate = "unsupported" if row["estimate"] is None else f"{row['estimate']:.6g}"
        interval = "n/a" if row["ci95_low"] is None else f"[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}]"
        lines.append(f"| {row['num_agents']} | {row['metric']} | {estimate} | {interval} | {row['pairs']} |")
    lines += ["", "## Preferential service and limits", "",
        "Priority-level tables use fixed reference PANAMA weights: low [1,2), middle [2,3), high [3,4]. "
        "The communicated reference is reconstructed before the treatment replaces weights; physical oracle "
        "levels are separate. Equal allocations therefore retain meaningful PANAMA strata. Missing/unadmitted "
        "tasks remain unknown. All cases, including zero-exposure cases, enter pooled-rate bootstrap samples.", "",
        "Preferential-service differences compare the high-minus-low service gap under current priority with "
        "the same gap under matched equal allocation. Negative latency differences and positive compliance "
        "differences indicate stronger preference for high-priority tasks. These contrasts are separate from "
        "mission success/makespan effects. Different trajectories produce different post-treatment task "
        "strata, so this is descriptive service evidence rather than a causal same-task comparison. "
        "Reference weights also change with the active population: a lone active robot receives weight 4. "
        "Active-team sizes are retained in task records; high/low contrasts alone do not isolate causality.", "",
        "| Robots | Priority source | Difference in high-minus-low gap | Estimate | 95% CI | Cases |",
        "| ---: | --- | --- | ---: | --- | ---: |"]
    for row in service:
        estimate = "unsupported" if row["estimate"] is None else f"{row['estimate']:.6g}"
        interval = "n/a" if row["ci95_low"] is None else f"[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}]"
        lines.append(f"| {row['num_agents']} | {row['basis']} | {row['metric']} | {estimate} | {interval} | {row['pairs']} |")
    lines += ["",
        "CPU service uses processor sharing. Zero-rate waiting is measured from allocation intervals; it is "
        "not a FIFO queue. Elapsed service beyond C/F includes sharing and DVFS, and is reported separately. "
        "Delivered-task latency excludes right-censored tasks; undelivered counts remain explicit. "
        "FoV/vertex values with no relevant directed pairs do not establish consistency. Vertex-only results "
        "remain exploratory. Energy excludes propulsion, static electronics and cloud electricity.", "",
        "PANAMA priorities include the existing robot-ID tie-break. When path distances are equal, very "
        "small numerical priority differences can span the reference weight range. Service bins describe "
        "that implemented mapping; raw priorities and weights remain available for interpretation. "
        "Intervals are pointwise without multiplicity correction and are conditional on this fixed map, "
        "checkpoint, synthetic workload and orbital archive.", "",
        "All six treatments and every measured mission contrast are retained in CSVs, including null, "
        "negative and mixed findings. Sensitivity pilots remain separate evidence and are not pooled into "
        "full-study inference. Actual-model inference, cutoff oracle coverage/actions, event checksums, "
        "case identity, CPU utilization and deadline denominators are replay-audited. See validation.json.", ""]
    (output/"RESEARCH_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def latency_distributions(output, episodes):
    """Exact pooled quantiles, with censored counts and separate priority/site CDFs."""
    groups = defaultdict(list)
    counts = defaultdict(lambda: defaultdict(int))
    for episode in episodes:
        condition,n = episode["condition"],episode["num_agents"]
        for basis,names in (("all",("all",)),("location",("drone","cloud","unassigned")),
                            ("communicated",LEVELS),("physical",LEVELS)):
            for name in names:
                counts[(condition,n,basis,name)]  # Retain explicit no-exposure groups.
    with (output/"tasks_by_priority.csv").open(newline="",encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            keys = [(row["condition"],int(row["num_agents"]),"all","all"),
                (row["condition"],int(row["num_agents"]),"location",row["location"])]
            keys += [(row["condition"],int(row["num_agents"]),basis,row[f"{basis}_level"]) for basis in BASES]
            for key in keys:
                counts[key]["tasks"] += 1
                counts[key][row["deadline_status"]] += 1
                counts[key]["assessed"] += row["deadline_met"] != ""
                counts[key]["cutoff_waits"] += row["cutoff_wait"] == "True"
                counts[key]["cpu_elapsed_censored_tasks"] += row["cpu_elapsed_censored"] == "True"
                for field in ("cpu_zero_rate_wait_seconds","cpu_elapsed_seconds","cpu_observed_elapsed_seconds",
                              "cpu_capacity_floor_excess_seconds"):
                    if row[field] != "":
                        counts[key][field+"_sum"] += float(row[field])
                        counts[key][field+"_samples"] += 1
                if row["latency_seconds"]:
                    groups[key].append(float(row["latency_seconds"]))
    rows = []
    for (condition,count,basis,group),counter in counts.items():
        values = groups[(condition,count,basis,group)]
        item = dict(condition=condition,num_agents=count,basis=basis,group=group,
            tasks=counter["tasks"],on_time=counter["on_time"],late=counter["late"],undelivered=counter["undelivered"],
            assessed=counter["assessed"],cutoff_waits=counter["cutoff_waits"],
            on_time_rate=counter["on_time"]/counter["tasks"] if counter["tasks"] else None,
            late_rate=counter["late"]/counter["tasks"] if counter["tasks"] else None,
            undelivered_rate=counter["undelivered"]/counter["tasks"] if counter["tasks"] else None,
            deadline_compliance=counter["on_time"]/counter["assessed"] if counter["assessed"] else None,
            delivered_latency_samples=len(values),cpu_elapsed_censored_tasks=counter["cpu_elapsed_censored_tasks"],
            pooled_mean_latency_seconds=float(np.mean(values)) if values else None)
        for field in ("cpu_zero_rate_wait_seconds","cpu_elapsed_seconds","cpu_observed_elapsed_seconds",
                      "cpu_capacity_floor_excess_seconds"):
            denominator = counter[field+"_samples"]
            item["mean_"+field] = counter[field+"_sum"]/denominator if denominator else None
            item[field+"_samples"] = denominator
        for quantile in (50,75,90,95,99):
            item[f"latency_p{quantile}_seconds"] = float(np.quantile(values,quantile/100)) if values else None
        rows.append(item)
    _write_csv(output/"latency_distributions.csv",rows)
    _write_csv(output/"priority_summary.csv",[r for r in rows if r["basis"] in BASES])
    overall,locations = [],[]
    for item in rows:
        case_rows = [r for r in episodes if r["condition"] == item["condition"] and r["num_agents"] == item["num_agents"]]
        if item["basis"] == "all":
            summary = dict(item,mission_success_rate=float(np.mean([r["success"] for r in case_rows])),cases=len(case_rows))
            for metric in MISSION_METRICS:
                values = [r[metric] for r in case_rows if r.get(metric) is not None]
                summary["mean_case_"+metric] = float(np.mean(values)) if values else None
            overall.append(summary)
        elif item["basis"] == "location":
            summary = dict(item,cases=len(case_rows))
            if item["group"] in ("drone","cloud"):
                summary["mean_case_cpu_utilization"] = float(np.mean([r[f"mean_{item['group']}_cpu_utilization"] for r in case_rows]))
                summary["mean_case_resident_jobs"] = float(np.mean([r[f"mean_{item['group']}_resident_jobs"] for r in case_rows]))
            locations.append(summary)
    _write_csv(output/"overall_summary.csv",overall)
    _write_csv(output/"execution_location_summary.csv",locations)
    import matplotlib.pyplot as plt
    populations = sorted({k[1] for k in counts})
    for view in ("all","communicated","physical","location"):
        fig,axes = plt.subplots(1,len(populations),figsize=(4*len(populations),4.2),squeeze=False,layout="constrained")
        for ax,count in zip(axes[0],populations):
            for (condition,n,basis,group),values in groups.items():
                if n != count or basis != view or not values:
                    continue
                if view != "all" and condition not in (PRIMARY,BASELINE):
                    continue
                sorted_values = np.sort(values)
                indices = np.unique(np.linspace(0,len(values)-1,min(1500,len(values))).astype(int))
                caption = condition.split("__")[-1]+(" / "+group if view != "all" else "")
                ax.step(sorted_values[indices]*1000,(indices+1)/len(values),where="post",label=caption,
                    linestyle="--" if condition == BASELINE else "-")
            ax.axvline(250,color="black",lw=.7,ls=":")
            ax.set(xscale="symlog",xlabel="Command latency (ms; symlog)",ylabel="Delivered-command CDF",
                title=f"{count} robots",ylim=(0,1.02))
            ax.grid(alpha=.2)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=3,fontsize=8)
        for suffix in ("png","svg"):
            fig.savefig(output/f"latency_cdf_{view}.{suffix}",dpi=160,bbox_inches="tight")
        plt.close(fig)
