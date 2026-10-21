"""Paired command-cutoff sensitivity; simulation and allocation kernels are reused."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess

import numpy as np

from .config import FIXED_MAP
from .records import export_csv, read_jsonl, sha256, write_json
from .resource_study import conditions, run_study, study_cases
from .task_deadlines import command_deadline_metrics

DEADLINES_MS = (100, 250, 300, 500)
POLICY_LABELS = {"fresh": "Fresh reference", "equal": "Equal (matched)",
    "static": "Static priority", "aoi": "AoI only", "priority": "Current priority",
    "equal_legacy": "Equal (weight 1)"}
LOCATIONS = ("drone", "cloud", "unassigned")
COUNT_FIELDS = ("deadline_tasks_monitored", "deadline_tasks_assessed", "deadline_commands_on_time",
    "deadline_commands_late", "deadline_commands_undelivered", "deadline_undelivered_overdue_tasks",
    "deadline_pending_tasks", "task_deadline_misses", "deadline_censored_tasks", "deadline_cutoff_wait_tasks")


def deadline_config(config, milliseconds):
    if not config.resource_allocation.enabled or config.resource_allocation.execution_mode != "remote_dt":
        raise ValueError("Deadline sensitivity requires enabled remote_dt execution")
    if config.communication.seconds_per_step != 1.0:
        raise ValueError("Deadline sensitivity requires a fixed 1-second control interval")
    if milliseconds not in DEADLINES_MS:
        raise ValueError(f"Use command cutoffs {DEADLINES_MS} ms")
    cutoff = milliseconds/1000
    return replace(config, resource_allocation=replace(config.resource_allocation,
        decision_budget_seconds=cutoff, command_cutoff_seconds=cutoff,
        task_deadline_seconds=cutoff, enforce_task_deadlines=False))


def _ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def _rates(counts):
    total = counts["deadline_tasks_monitored"]
    return dict(command_rate_denominator=total,
        command_on_time_rate=_ratio(counts["deadline_commands_on_time"] or 0, total),
        command_late_rate=_ratio(counts["deadline_commands_late"] or 0, total),
        command_undelivered_rate=_ratio(counts["deadline_commands_undelivered"] or 0, total))


def task_location(task):
    if task.get("executed_x") == 1:
        return "drone"
    if task.get("executed_x") == 0:
        return "cloud"
    return "unassigned"


def task_statistics(tasks, observed_at):
    """One task, one arrival category; censor/WAIT tags remain separate."""
    if any(t.get("deadline_at") is None for t in tasks):
        raise ValueError("Sensitivity records require monitored task deadlines")
    metrics = command_deadline_metrics(tasks, observed_at)
    metrics.update(_rates(metrics))
    metrics.update(_work_statistics(tasks))
    return metrics


def _work_statistics(tasks):
    metrics = {}
    delivered = [t["observed_latency_seconds"] for t in tasks if t.get("delivered_at") is not None]
    metrics["pooled_task_latency_p95_seconds"] = float(np.quantile(delivered, .95)) if delivered else None
    assigned = sum(task_location(t) != "unassigned" for t in tasks)
    metrics["offloading_assigned_tasks"] = assigned
    for location in LOCATIONS:
        count = sum(task_location(t) == location for t in tasks)
        metrics[f"offload_{location}_tasks"] = count
        metrics[f"offload_{location}_fraction"] = _ratio(count, len(tasks) if location == "unassigned" else assigned)
    return metrics


def _pool_statistics(rows, tasks):
    # Sum final classifications from each episode's own observation horizon.
    counts = {k:sum(r[k] or 0 for r in rows) for k in COUNT_FIELDS}
    return dict(counts, **_rates(counts), **_work_statistics(tasks),
        deadline_compliance_rate=_ratio(counts["deadline_commands_on_time"], counts["deadline_tasks_assessed"]))


def _write_table(root, name, rows):
    path = root/f"{name}.jsonl"
    path.write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in rows), encoding="utf-8")
    export_csv(path, root/f"{name}.csv")


def _verify_child(root, milliseconds, manifest):
    directory = root/f"deadline-{milliseconds}ms"
    child = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if json.loads((directory/"status.json").read_text())["status"] != "complete":
        raise ValueError(f"Incomplete deadline condition: {milliseconds} ms")
    if sha256(directory/"held_out_cases.json") != manifest["case_bank_sha256"]:
        raise ValueError("Paired case bank mismatch across deadlines")
    for actual, expected, label in ((child["model"]["sha256"], manifest["model_sha256"], "model"),
        (child["map_sha256"], manifest["map_sha256"], "map"),
        (child["timeline_sha256"], manifest["timeline_sha256"], "timeline"),
        (child["code_sha256"], manifest["code_sha256"], "source")):
        if actual != expected:
            raise ValueError(f"Paired {label} identity mismatch")
    expected = json.loads(json.dumps(manifest["configuration"]))
    expected["resource_allocation"].update(decision_budget_seconds=milliseconds/1000,
        command_cutoff_seconds=milliseconds/1000, task_deadline_seconds=milliseconds/1000,
        enforce_task_deadlines=False)
    if child["configuration"] != expected:
        raise ValueError("Unexpected configuration change in deadline condition")
    if [c["name"] for c in child["conditions"]] != manifest["treatments"]:
        raise ValueError("Deadline study must retain all six treatments")
    from .config import ProjectConfig
    for (name,cfg,tags), actual in zip(conditions(ProjectConfig.from_dict(expected)),child["conditions"]):
        cfg = replace(cfg,resource_allocation=replace(cfg.resource_allocation,
            failure_directory=str(directory/name/"failures")))
        if json.loads(json.dumps(cfg.to_dict())) != actual["configuration"] or tags != actual["tags"]:
            raise ValueError("Unexpected treatment configuration change")
    return directory, child


def report_study(root):
    root = Path(root).resolve()
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if sha256(root/"held_out_cases.json") != manifest["case_bank_sha256"]:
        raise ValueError("Shared paired case bank mismatch")
    bank = {c["case_id"]: c for c in json.loads((root/"held_out_cases.json").read_text())}
    episodes, summaries, locations = [], [], []
    grouped_rows, grouped_tasks = defaultdict(list), defaultdict(list)
    for milliseconds in manifest["deadlines_ms"]:
        directory, child = _verify_child(root, milliseconds, manifest)
        for condition in child["conditions"]:
            folder = directory/condition["name"]
            tasks_by_case = defaultdict(list)
            for task in read_jsonl(folder/"tasks.jsonl"):
                tasks_by_case[task["case_id"]].append(task)
            rows = read_jsonl(folder/"episodes.jsonl")
            if len(rows) != len(bank) or {r["case_id"] for r in rows} != set(bank):
                raise ValueError("Incomplete or duplicate paired episodes")
            if set(tasks_by_case)-set(bank):
                raise ValueError("Unpaired task records")
            for row in rows:
                case = bank[row["case_id"]]
                if any(row.get(k) != v for k, v in case.items()):
                    raise ValueError("Paired episode geometry/seed/orbit mismatch")
                tasks = tasks_by_case[row["case_id"]]
                if len(tasks) != row["tasks_generated"]:
                    raise ValueError("Missing task records in deadline report")
                if any(abs(t["deadline_at"]-t["source_time"]-milliseconds/1000) > 1e-12 for t in tasks):
                    raise ValueError("Task sensing-to-command deadline differs from scenario")
                # Each episode ends at an integer control epoch, even if the
                # last command cutoff occurred earlier in that interval.
                horizon = row["restricted_makespan_seconds"]
                metrics = task_statistics(tasks, horizon)
                for key in ("deadline_tasks_monitored", "deadline_tasks_assessed",
                    "deadline_commands_on_time", "deadline_commands_late", "deadline_commands_undelivered",
                    "task_deadline_misses", "deadline_cutoff_wait_tasks", "deadline_censored_tasks"):
                    if metrics[key] != row[key]:
                        raise ValueError(f"Deadline accounting mismatch: {key}")
                if metrics["deadline_cutoff_wait_tasks"] != row["command_cutoff_misses"]:
                    raise ValueError("Command cutoff and monitored deadline differ")
                item = dict(row, deadline_ms=milliseconds, **metrics)
                episodes.append(item)
                key = (milliseconds, condition["name"], row["num_agents"])
                grouped_rows[key].append(item)
                # Make logical IDs unique only in the derived pooled table;
                # per-episode task and archive identities stay untouched.
                grouped_tasks[key].extend(dict(t, job_id=row["case_id"]+"/"+t["job_id"]) for t in tasks)
    for (milliseconds, name, count), rows in grouped_rows.items():
        tasks = grouped_tasks[(milliseconds, name, count)]
        stats = _pool_statistics(rows,tasks)
        successful = [r["successful_makespan"] for r in rows if r["successful_makespan"] is not None]
        identity = dict(deadline_ms=milliseconds, treatment=name, policy=rows[0]["policy"], num_agents=count)
        summary = dict(identity, cases=len(rows), mission_success_rate=float(np.mean([r["success"] for r in rows])),
            mean_successful_makespan_seconds=float(np.mean(successful)) if successful else None,
            mean_restricted_makespan_seconds=float(np.mean([r["restricted_makespan_seconds"] for r in rows])),
            mean_waiting_steps=float(np.mean([r["waiting_steps"] for r in rows])),
            mean_cutoff_waits=float(np.mean([r["command_cutoff_misses"] for r in rows])),
            mean_episode_task_latency_p95_seconds=float(np.mean([r["task_latency_p95_seconds"] for r in rows
                if r["task_latency_p95_seconds"] is not None])) if any(r["task_latency_p95_seconds"] is not None for r in rows) else None,
            **stats)
        summaries.append(summary)
        for location in LOCATIONS:
            location_rows = []
            for row in rows:
                subset = [t for t in tasks if t["case_id"] == row["case_id"] and task_location(t) == location]
                location_rows.append(task_statistics(subset, row["restricted_makespan_seconds"]))
            counts = {k:sum(r[k] or 0 for r in location_rows) for k in COUNT_FIELDS}
            latencies = [t["observed_latency_seconds"] for t in tasks if task_location(t) == location and t.get("delivered_at") is not None]
            locations.append(dict(identity, execution_location=location, cases=len(rows), **counts, **_rates(counts),
                deadline_compliance_rate=_ratio(counts["deadline_commands_on_time"], counts["deadline_tasks_assessed"]),
                offloading_fraction=stats[f"offload_{location}_fraction"],
                pooled_task_latency_p95_seconds=float(np.quantile(latencies, .95)) if latencies else None))
    pairs = paired_deadline_changes(episodes)
    for name, rows in (("episodes", episodes), ("summary", summaries), ("execution_locations", locations),
                       ("paired_vs_100ms", pairs)):
        _write_table(root, name, rows)
    write_json(root/"summary.json", summaries)
    plot_results(root, summaries, locations)
    return summaries


def paired_deadline_changes(episodes):
    baseline = {(r["condition"], r["case_id"]):r for r in episodes if r["deadline_ms"] == 100}
    grouped = defaultdict(list)
    for row in episodes:
        grouped[(row["deadline_ms"], row["condition"], row["num_agents"])].append(row)
    result = []
    for (deadline, treatment, count), rows in grouped.items():
        for metric in ("deadline_compliance_rate", "command_on_time_rate", "command_late_rate",
            "command_undelivered_rate", "command_cutoff_misses", "success", "restricted_makespan_seconds",
            "successful_makespan", "waiting_steps", "task_latency_p95_seconds", "offload_drone_fraction",
            "offload_cloud_fraction"):
            differences = [float(r[metric])-float(baseline[(treatment,r["case_id"])][metric]) for r in rows
                if r.get(metric) is not None and baseline[(treatment,r["case_id"])].get(metric) is not None]
            if not differences:
                continue
            interval = [None, None]
            if len(differences) > 1:
                values = np.asarray(differences)
                indices = np.random.default_rng(42).integers(0,len(values),(2000,len(values)))
                interval = np.quantile(values[indices].mean(1),[.025,.975]).tolist()
            result.append(dict(deadline_ms=deadline, treatment=treatment, num_agents=count, metric=metric,
                pairs=len(differences), mean_delta=float(np.mean(differences)), ci95_low=interval[0], ci95_high=interval[1]))
    return result


def plot_results(root, summaries, locations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = root/"figures"
    output.mkdir(exist_ok=True)
    counts = sorted({r["num_agents"] for r in summaries})
    title = ("Deadline sensitivity — paired diagnostic" if json.loads((root/"manifest.json").read_text())["diagnostic"]
             else "Deadline sensitivity — paired frozen-policy study")
    metrics = (("command_on_time_rate", "Commands within cutoff (fraction)"),
        ("command_late_rate", "Late commands (fraction)"),
        ("command_undelivered_rate", "Undelivered commands (fraction)"),
        ("mean_cutoff_waits", "Cutoff WAITs / episode"),
        ("mean_restricted_makespan_seconds", "Mission duration, capped (s)"),
        ("mean_waiting_steps", "Active waiting steps / episode"),
        ("mission_success_rate", "Mission success (fraction)"),
        ("pooled_task_latency_p95_seconds", "Delivered task latency p95 (s)"),
        ("offload_drone_fraction", "UAV share of assigned tasks"))
    for metric, label in metrics:
        fig, axes = plt.subplots(1,len(counts),figsize=(4*len(counts),3.8),squeeze=False,layout="constrained")
        for ax,count in zip(axes[0],counts):
            for index,(policy,caption) in enumerate(POLICY_LABELS.items()):
                group = sorted((r for r in summaries if r["num_agents"] == count and r["policy"] == policy
                    and r[metric] is not None),key=lambda r:r["deadline_ms"])
                ax.plot([r["deadline_ms"] for r in group],[r[metric] for r in group],
                    marker=("o","s","^","v","D","x")[index], linestyle="-" if index%2 == 0 else "--",label=caption)
            ax.set(xlabel="Command cutoff (ms)",ylabel=label,title=f"{count} robots",xticks=DEADLINES_MS)
            ax.set_ylim(0,1.05 if "fraction" in label or metric == "offload_drone_fraction" else None)
            ax.grid(alpha=.25)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=3,fontsize=8)
        fig.suptitle(title,fontsize=11)
        for suffix in ("png","svg"):
            fig.savefig(output/f"{metric}.{suffix}",dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(2,len(counts),figsize=(4*len(counts),6),squeeze=False,layout="constrained")
    for line,location in enumerate(("drone","cloud")):
        for ax,count in zip(axes[line],counts):
            for index,(policy,caption) in enumerate(POLICY_LABELS.items()):
                group=sorted((r for r in locations if r["num_agents"] == count and r["policy"] == policy
                    and r["execution_location"] == location and r["deadline_compliance_rate"] is not None),
                    key=lambda r:r["deadline_ms"])
                # Missing locations stay missing; do not interpolate across a
                # cutoff at which no work ran at that processor.
                by_deadline={r["deadline_ms"]:r["deadline_compliance_rate"] for r in group}
                ax.plot(DEADLINES_MS,[by_deadline.get(d,np.nan) for d in DEADLINES_MS],
                    marker=("o","s","^","v","D","x")[index],label=caption)
            ax.set(xlabel="Command cutoff (ms)",ylabel=f"{location}: deadline compliance",title=f"{count} robots",
                xticks=DEADLINES_MS,ylim=(0,1.05))
            ax.grid(alpha=.25)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=3,fontsize=8)
    fig.suptitle(title,fontsize=11)
    for suffix in ("png","svg"):
        fig.savefig(output/f"location_compliance.{suffix}",dpi=160)
    plt.close(fig)


def run_deadline_study(config, output, model_path, *, diagnostic=False, cases=1,
                       counts=(2,4,6,8), max_steps=250, workers=1, device_name="cpu", resume=False):
    """Execute real cutoff changes; never reclassify a fixed 100 ms trajectory."""
    deadline_config(config,100)  # Validate execution mode and fixed control interval before creating output.
    if not 1 <= cases <= 50 or not counts or len(set(counts)) != len(counts) or not set(counts) <= {2,4,6,8}:
        raise ValueError("Use 1–50 paired cases and populations 2/4/6/8")
    if not 1 <= max_steps <= 250 or not 1 <= workers <= 32:
        raise ValueError("Use a 1–250 step cap and 1–32 workers")
    if not diagnostic and (not config.resource_allocation.remote_acceptance_approved or
                          cases != 50 or tuple(counts) != (2,4,6,8) or max_steps != 250):
        raise ValueError("Full deadline studies require acceptance approval and 50 cases; use --diagnostic for pilots")
    model_path = Path(model_path).resolve()
    if not model_path.is_file() or not config.communication.leo_timeline:
        raise ValueError("Deadline sensitivity requires a checkpoint and archived orbital timeline")
    timeline = config.resolve(config.communication.leo_timeline)
    if not timeline.is_file():
        raise FileNotFoundError(timeline)
    config = replace(config, communication=replace(config.communication,leo_timeline=str(timeline)))
    variants = [(ms,deadline_config(config,ms)) for ms in DEADLINES_MS]
    bank = study_cases(config,cases,counts,max_steps)
    if any(study_cases(cfg,cases,counts,max_steps) != bank for _,cfg in variants):
        raise AssertionError("Changing cutoffs changed the paired case bank")
    output = Path(output).resolve()
    identity = dict(schema=1,kind="command_deadline_sensitivity",deadlines_ms=list(DEADLINES_MS),
        diagnostic=diagnostic,configuration=config.to_dict(),cases_per_count=cases,counts=list(counts),episode_cap=max_steps,
        model_path=str(model_path),model_sha256=sha256(model_path),map_sha256=sha256(config.resolve(FIXED_MAP)),
        timeline_sha256=sha256(timeline),code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob("*.py")},
        treatments=[name for name,_,_ in conditions(config)],
        rate_denominators="Arrival rates: all monitored tasks; compliance: assessed tasks; offload shares: assigned tasks; no-work locations: null")
    identity = json.loads(json.dumps(identity,allow_nan=False))
    if resume:
        previous = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        if any(previous.get(k) != v for k,v in identity.items()):
            raise ValueError("Deadline study resume identity mismatch")
        if json.loads((output/"held_out_cases.json").read_text()) != bank:
            raise ValueError("Deadline study resume case bank mismatch")
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/"held_out_cases.json",bank)
        try:
            commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            commit=None
        write_json(output/"manifest.json",dict(identity,created_utc=datetime.now(UTC).isoformat(),
            git_commit=commit,case_bank_sha256=sha256(output/"held_out_cases.json")))
    completed = []
    try:
        for ms,cfg in variants:
            write_json(output/"status.json",dict(status="running",active_deadline_ms=ms,
                completed_deadlines_ms=completed,planned_episodes=4*6*len(bank)))
            directory=output/f"deadline-{ms}ms"
            run_study(cfg,directory,model_path,diagnostic=diagnostic,cases=cases,counts=counts,
                max_steps=max_steps,workers=workers,device_name=device_name,resume=resume and directory.exists())
            manifest=json.loads((output/"manifest.json").read_text())
            _verify_child(output,ms,manifest)
            completed.append(ms)
        summaries=report_study(output)
    except Exception as exc:
        write_json(output/"status.json",dict(status="failed",active_deadline_ms=ms,
            completed_deadlines_ms=completed,error=str(exc)))
        raise
    write_json(output/"status.json",dict(status="complete",completed_deadlines_ms=completed,
        completed_episodes=4*6*len(bank),diagnostic=diagnostic))
    return summaries
