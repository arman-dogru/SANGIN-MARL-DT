"""Reproducible frozen-policy resource sweeps; diagnostics never qualify a model."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
import gzip
import json
from pathlib import Path
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .allocation import AllocationFailure, AllocationSnapshot, compare_allocations
from .config import FIXED_MAP
from .records import append_jsonl, export_csv, sha256, write_json


def _worker_initialize(model_path, configuration, diagnostic, device_name):
    import torch
    from .config import ProjectConfig
    from .models import DQN, choose_device
    from .training import load_model
    global _WORKER_MODEL, _WORKER_DEVICE
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = ProjectConfig.from_dict(configuration)
    _WORKER_DEVICE = choose_device(device_name)
    if model_path is None:
        _WORKER_MODEL = DQN((3, 15, 15), config.vector_length).to(_WORKER_DEVICE).eval()
    else:
        _WORKER_MODEL, _ = load_model(model_path, config, _WORKER_DEVICE, require_qualified=not diagnostic)


def _worker_episode(arguments):
    from .evaluation import run_episode
    config, case, version, max_steps = arguments
    return run_episode(config, _WORKER_MODEL, case, _WORKER_DEVICE, version,
        decision_deadline_seconds=config.resource_allocation.decision_budget_seconds, max_steps=max_steps)


def _case_config(config, case):
    if "orbital_offset_seconds" not in case:
        return config
    return replace(config, communication=replace(config.communication,
        backhaul_time_offset_seconds=config.communication.backhaul_time_offset_seconds+case["orbital_offset_seconds"]))


def optimize_snapshot(input_path, output_path):
    snapshot = AllocationSnapshot.from_dict(json.loads(Path(input_path).read_text(encoding="utf-8")))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = compare_allocations(snapshot)
    except AllocationFailure as exc:
        write_json(target.with_suffix(".failed.json"), dict(error=str(exc), snapshot=exc.snapshot, details=exc.details))
        raise
    write_json(target, result)
    return result


def conditions(config, secondary=True):
    """Each physical variant has identical radio/compute assumptions across policies."""
    comm = replace(config.communication, profile="physical", contention_enabled=False,
        packet_payload_bytes=config.communication.packet_payload_bytes or 65536,
        stale_information=("task", "intentions", "priorities"))
    resource = replace(config.resource_allocation, enabled=True)
    base = replace(config, communication=comm, resource_allocation=resource)
    result = [("fresh", replace(base, communication=replace(comm, profile="perfect")),
               dict(variant="fresh", policy="fresh", kind="reference", uplink_factor=1.0, cpu_factor=1.0))]
    variants = []
    for up in (.25, 1., 4.):
        for cpu in (.25, 1., 4.):
            variants.append((f"up-{up:g}-cpu-{cpu:g}", replace(base,
                communication=replace(comm, satellite_uplink_mbps=comm.satellite_uplink_mbps*up),
                resource_allocation=replace(resource, drone_cycles_per_second=resource.drone_cycles_per_second*cpu,
                    cloud_cycles_per_second=resource.cloud_cycles_per_second*cpu)),
                dict(kind="primary", uplink_factor=up, cpu_factor=cpu)))
    if secondary:
        for variable in ("drone_cpu", "cloud_cpu", "downlink", "task_bits", "cycles_per_bit"):
            for factor in (.25, 4.):
                values, radio = {}, {}
                if variable == "drone_cpu":
                    values["drone_cycles_per_second"] = resource.drone_cycles_per_second*factor
                elif variable == "cloud_cpu":
                    values["cloud_cycles_per_second"] = resource.cloud_cycles_per_second*factor
                elif variable == "downlink":
                    radio = dict(satellite_downlink_mbps=comm.satellite_downlink_mbps*factor,
                                 access_downlink_mbps=comm.access_downlink_mbps*factor)
                elif variable == "task_bits":
                    values["task_input_bits"] = int(resource.task_input_bits*factor)
                else:
                    values["cycles_per_bit"] = resource.cycles_per_bit*factor
                variants.append((f"{variable}-{factor:g}", replace(base,
                    communication=replace(comm, **radio), resource_allocation=replace(resource, **values)),
                    dict(kind="secondary", variable=variable, factor=factor)))
        for ratio in (1., 16.):
            variants.append((f"weight-ratio-{ratio:g}", replace(base,
                resource_allocation=replace(resource, weight_max=resource.weight_min*ratio)),
                dict(kind="secondary", variable="weight_ratio", factor=ratio)))
        for energy in (0., .1, 10.):
            variants.append((f"energy-{energy:g}", replace(base,
                resource_allocation=replace(resource, energy_coefficient=energy)),
                dict(kind="secondary", variable="energy_coefficient", factor=energy)))
    for variant, cfg, tags in variants:
        for policy in ("equal", "static", "aoi", "priority"):
            result.append((f"{variant}__{policy}", replace(cfg,
                resource_allocation=replace(cfg.resource_allocation, policy=policy)),
                dict(tags, variant=variant, policy=policy)))
    return result


def episode_metrics(row, trace):
    frames, tasks = trace["frames"], trace["resource_tasks"]
    delivered = [t for t in tasks if "delivered_at" in t]
    metrics = dict(tasks_generated=len(tasks), tasks_delivered=len(delivered),
        tasks_unfinished=len(tasks)-len(delivered), waiting_steps=0, sum_path_costs=0,
        mean_task_latency_seconds=None, mean_completed_task_energy_j=None,
        mean_predicted_latency_seconds=None, mean_compute_seconds=None,
        mean_uplink_queue_seconds=None, mean_batch_wait_seconds=None,
        completed_task_energy_j=sum(t["robot_energy_j"]+t["backhaul_energy_j"]+t.get("drone_compute_energy_j", 0)
                                    for t in delivered),
        completed_task_robot_energy_j=sum(t["robot_energy_j"] for t in delivered),
        completed_task_backhaul_energy_j=sum(t["backhaul_energy_j"] for t in delivered),
        completed_task_compute_energy_j=sum(t.get("drone_compute_energy_j", 0) for t in delivered),
        voluntary_wait_steps=0, travel_steps=0)
    if delivered:
        for target, source in (("mean_task_latency_seconds", "observed_latency_seconds"),
            ("mean_predicted_latency_seconds", "predicted_latency_seconds"),
            ("mean_compute_seconds", "compute_elapsed_seconds"),
            ("mean_uplink_queue_seconds", "uplink_queue_seconds"),
            ("mean_batch_wait_seconds", "batch_wait_seconds")):
            metrics[target] = float(np.mean([t[source] for t in delivered]))
        metrics["mean_completed_task_energy_j"] = metrics["completed_task_energy_j"]/len(delivered)
    for before, after in zip(frames, frames[1:]):
        for agent, completed in before["completed_flags"].items():
            if not completed:
                metrics["sum_path_costs"] += 1
                actions = after["actions"]
                if actions.get(agent, actions.get(str(agent))) == 4:
                    metrics["voluntary_wait_steps"] += 1
                i = int(agent)
                if before["positions"][i] == after["positions"][i]:
                    metrics["waiting_steps"] += 1
                else:
                    metrics["travel_steps"] += 1
    snapshots = [f["consistency"] for f in frames]
    relevant = sum(s["relevant_pairs"] for s in snapshots)
    errors = sum(s["fov_errors"] for s in snapshots)
    allocations = [e for e in trace["transport_events"] if e["event"] == "paper_allocation"]
    metrics.update(fov_relevant_pairs=relevant, fov_errors=errors,
        xi_fov=errors/max(1, relevant), mean_xi_fov=float(np.mean([s["xi_fov"] for s in snapshots])),
        mean_xi_aoi=float(np.mean([s["xi_aoi"] for s in snapshots])),
        ordering_errors=sum(s["ordering_errors"] for s in snapshots),
        allocation_solves=len(allocations),
        mean_relaxation_gap=float(np.mean([a["absolute_gap"] for a in allocations])) if allocations else 0,
        solver_seconds=sum(a["relaxed"]["solve_seconds"]+a["binary"]["solve_seconds"] for a in allocations))
    metrics.update(fov_uplink_errors=sum(s.get("fov_uplink_errors", 0) for s in snapshots),
        fov_downlink_errors=sum(s.get("fov_downlink_errors", 0) for s in snapshots),
        xi_fov_uplink=sum(s.get("fov_uplink_errors", 0) for s in snapshots)/max(1, relevant),
        xi_fov_downlink=sum(s.get("fov_downlink_errors", 0) for s in snapshots)/max(1, relevant),
        restricted_makespan_seconds=row["steps"]*trace["configuration_seconds_per_step"]
            if "configuration_seconds_per_step" in trace else row["steps"],
        latency_p95_seconds=float(np.quantile([t["observed_latency_seconds"] for t in delivered], .95)) if delivered else None,
        mean_prediction_error_seconds=float(np.mean([t["observed_latency_seconds"]-t["predicted_latency_seconds"] for t in delivered]))
            if delivered else None,
        allocation_loop_steps=len(trace.get("allocation_loop", [])))
    duration = frames[-1]["time"]
    for pool in ("drone", "cloud"):
        events = [e for e in trace["transport_events"] if e["event"] == "compute_allocation" and e["pool"] == pool]
        area = sum((min(duration, events[i+1]["time"] if i+1 < len(events) else duration)-event["time"])
                   *event["utilization"] for i, event in enumerate(events))
        metrics[f"mean_{pool}_cpu_utilization"] = area/duration if duration else 0.0
    return {**row, **metrics}


METRICS = ("par2", "success", "waiting_steps", "sum_path_costs", "xi_fov", "mean_xi_aoi",
           "mean_task_latency_seconds", "mean_completed_task_energy_j", "completed_task_energy_j",
           "radio_energy_lower_j", "radio_energy_upper_j", "mean_task_age_seconds",
           "mean_intentions_age_seconds", "mean_predicted_latency_seconds", "mean_compute_seconds",
           "completion_fraction", "ever_completion_fraction", "successful_makespan", "restricted_makespan_seconds",
           "tasks_generated", "tasks_delivered", "tasks_unfinished", "latency_p95_seconds", "mean_prediction_error_seconds",
           "xi_fov_uplink", "xi_fov_downlink", "fov_relevant_pairs", "fov_errors", "fov_uplink_errors", "fov_downlink_errors",
           "drone_compute_energy_j", "modeled_energy_lower_j", "modeled_energy_upper_j", "collision_attempts",
           "voluntary_wait_steps", "travel_steps", "completed_task_robot_energy_j", "completed_task_backhaul_energy_j",
           "completed_task_compute_energy_j", "mean_drone_cpu_utilization", "mean_cloud_cpu_utilization")


def paired_metrics(rows, reference, samples=2000):
    baseline = {r["case_id"]: r for r in reference}
    result = {}
    for metric in METRICS:
        differences = [float(r[metric])-float(baseline[r["case_id"]][metric]) for r in rows
            if r["case_id"] in baseline and r.get(metric) is not None
            and baseline[r["case_id"]].get(metric) is not None]
        if not differences:
            continue
        values = np.array(differences)
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(values), (samples, len(values)))
        result[metric] = dict(delta=float(values.mean()), pairs=len(values),
            ci95=np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist())
    return result


def summarize(root):
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    groups = {}
    for condition in manifest["conditions"]:
        path = root/condition["name"]/"episodes.jsonl"
        if path.exists():
            groups[condition["name"]] = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]
    summaries = []
    for condition in manifest["conditions"]:
        name = condition["name"]
        for count in manifest["counts"]:
            rows = [r for r in groups.get(name, []) if r["num_agents"] == count]
            if not rows:
                continue
            fresh = [r for r in groups.get("fresh", []) if r["num_agents"] == count]
            equal_name = condition["tags"]["variant"]+"__equal"
            equal = [r for r in groups.get(equal_name, []) if r["num_agents"] == count]
            item = dict(condition=name, num_agents=count, cases=len(rows), **condition["tags"])
            for metric in METRICS:
                values = [float(r[metric]) for r in rows if r.get(metric) is not None]
                item[metric] = float(np.mean(values)) if values else None
                if values:
                    rng = np.random.default_rng(42)
                    indices = rng.integers(0, len(values), (manifest["bootstrap_samples"], len(values)))
                    item[metric+"_ci95"] = np.quantile(np.asarray(values)[indices].mean(axis=1), [.025, .975]).tolist()
            item["pooled_relevant_pairs"] = sum(r["fov_relevant_pairs"] for r in rows)
            item["pooled_fov_errors"] = sum(r["fov_errors"] for r in rows)
            item["pooled_xi_fov"] = item["pooled_fov_errors"]/max(1, item["pooled_relevant_pairs"])
            item["paired_fresh"] = paired_metrics(rows, fresh, manifest["bootstrap_samples"])
            item["paired_equal"] = paired_metrics(rows, equal, manifest["bootstrap_samples"])
            summaries.append(item)
    write_json(root/"summary.json", summaries)
    # Reuse the repository's CSV serializer for nested comparison intervals.
    path = root/"summary.jsonl"
    path.write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in summaries), encoding="utf-8")
    export_csv(path, root/"summary.csv")
    return summaries


def plot_results(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    rows = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((root/"manifest.json").read_text(encoding="utf-8"))["diagnostic"]
    counts = sorted({r["num_agents"] for r in rows})
    if not counts:
        return []
    output = root/"figures"
    output.mkdir(exist_ok=True)
    paths = []
    for metric, label in (("mean_task_latency_seconds", "Observed task latency (s)"),
                          ("xi_fov", "FoV-relevant ordering error fraction")):
        fig, axes = plt.subplots(1, len(counts), figsize=(4*len(counts), 3.5), squeeze=False)
        for ax, count in zip(axes[0], counts):
            for policy in ("equal", "static", "aoi", "priority"):
                group = sorted([r for r in rows if r.get("kind") == "primary" and
                    r.get("uplink_factor") == 1 and r["num_agents"] == count and r["policy"] == policy],
                    key=lambda r: r["cpu_factor"])
                group = [r for r in group if r.get(metric) is not None]
                if group:
                    ax.plot([r["cpu_factor"] for r in group], [r[metric] for r in group], "o-", label=policy)
            ax.set(xscale="log", xlabel="CPU capacity multiplier", ylabel=label, title=f"{count} robots")
            ax.set_xticks([.25, 1, 4], labels=["0.25", "1", "4"])
            ax.set_ylim(bottom=0)
            if metric == "xi_fov":
                ax.set_ylim(0, 1)
            ax.grid(alpha=.25)
            if ax.lines:
                ax.legend()
        if diagnostic:
            fig.suptitle("DIAGNOSTIC - frozen policy; not paper results", fontsize=10)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            target = output/f"{metric}_vs_compute.{suffix}"
            fig.savefig(target, dpi=170)
            paths.append(str(target))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    for policy in ("equal", "static", "aoi", "priority"):
        group = [r for r in rows if r["policy"] == policy and r["num_agents"] == max(counts)
            and r.get("mean_task_latency_seconds") is not None]
        if group:
            ax.scatter([r["mean_task_latency_seconds"] for r in group],
                [r["mean_completed_task_energy_j"] for r in group], label=policy, alpha=.7)
    ax.set(xlabel="Observed task latency (s)", ylabel="Completed-task transmit energy (J)",
           title=f"{max(counts)} robots; explicit simulation assumptions")
    if ax.collections:
        ax.legend()
    if diagnostic:
        fig.suptitle("DIAGNOSTIC - frozen policy; not paper results", fontsize=10)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        target = output/f"energy_latency.{suffix}"
        fig.savefig(target, dpi=170)
        paths.append(str(target))
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)
    factors = [.25, 1., 4.]
    for ax, policy in zip(axes, ("equal", "static", "aoi", "priority")):
        matrix = np.full((3, 3), np.nan)
        for row in rows:
            if row.get("kind") == "primary" and row["policy"] == policy and row["num_agents"] == max(counts):
                matrix[factors.index(row["uplink_factor"]), factors.index(row["cpu_factor"])] = row["xi_fov"]
        im = ax.imshow(matrix, origin="lower", vmin=0, vmax=1, cmap="magma")
        ax.set(xticks=range(3), xticklabels=["0.25", "1", "4"], yticks=range(3),
            yticklabels=["0.25", "1", "4"], xlabel="CPU multiplier", ylabel="Uplink multiplier", title=policy)
    fig.colorbar(im, ax=list(axes), label="FoV ordering error fraction", shrink=.8)
    fig.suptitle(("DIAGNOSTIC - " if diagnostic else "")+f"{max(counts)} robots: rate and compute sweep")
    for suffix in ("png", "svg"):
        target = output/f"rate_compute_xi_fov.{suffix}"
        fig.savefig(target, dpi=170)
        paths.append(str(target))
    plt.close(fig)
    return paths


def run_study(config, output, model_path=None, diagnostic=False, cases=50,
              counts=(2, 4, 6, 8), secondary=True, selected=None, max_steps=250, device_name="cpu", workers=1):
    import torch
    from .evaluation import case_bank, run_episode
    from .models import DQN, choose_device
    from .training import load_model
    if not 1 <= cases <= 50 or not set(counts) <= {2, 4, 6, 8} or not counts or len(counts) != len(set(counts)):
        raise ValueError("Use 1--50 cases and populations 2/4/6/8")
    if not 1 <= max_steps <= 250:
        raise ValueError("Episode cap must be 1--250")
    if not 1 <= workers <= 8:
        raise ValueError("Use 1--8 independent worker processes")
    if not diagnostic and (cases != 50 or tuple(counts) != (2, 4, 6, 8) or max_steps != 250):
        raise ValueError("Paper results require 50 cases, all populations, and 250 steps")
    if not diagnostic and (model_path is None or config.communication.leo_timeline is None):
        raise ValueError("Paper runs require a qualified checkpoint and archived orbital timeline")
    if model_path is not None and not Path(model_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if config.communication.leo_timeline:
        timeline = config.resolve(config.communication.leo_timeline)
        if not timeline.is_file():
            raise FileNotFoundError(f"Archived orbital timeline not found: {timeline}")
        config = replace(config, communication=replace(config.communication, leo_timeline=str(timeline)))
    choices = conditions(config, secondary)
    if selected:
        unknown = set(selected)-{name for name, _, _ in choices}
        if unknown:
            raise ValueError(f"Unknown resource conditions: {sorted(unknown)}")
        choices = [c for c in choices if c[0] == "fresh" or c[0] in selected]
    output = Path(output).resolve()
    choices = [(name, replace(cfg, resource_allocation=replace(cfg.resource_allocation,
        failure_directory=str(output/name/"failures"))), tags) for name, cfg, tags in choices]
    output.mkdir(parents=True, exist_ok=False)
    device = choose_device(device_name)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if model_path is None:
        model = DQN((3, 15, 15), config.vector_length).to(device)
        model.eval()
        torch.save(model.state_dict(), output/"diagnostic_model.pt")
        model_identity = dict(kind="untrained_frozen_diagnostic", sha256=sha256(output/"diagnostic_model.pt"))
        version = 0
    else:
        model, payload = load_model(model_path, config, device, require_qualified=not diagnostic)
        model_identity = dict(kind="checkpoint", path=str(Path(model_path).resolve()), sha256=sha256(model_path))
        model_identity.update(qualification=payload.get("qualification"), map_compatibility=payload.get("map_compatibility"))
        version = payload.get("learner_updates", 0)
    bank = case_bank(config, cases, counts)
    if config.communication.leo_timeline:
        from .leo import OrbitTimeline
        orbital = OrbitTimeline.load(config.communication.leo_timeline)
        available = orbital.period-config.communication.backhaul_time_offset_seconds-2*max_steps-1
        if available <= 0:
            raise ValueError("Orbital timeline is too short for the case bank and episode cap")
        for case in bank:
            case["orbital_offset_seconds"] = float(((case["num_agents"]//2-1)*5000+case["case"]*100)%available)
    write_json(output/"held_out_cases.json", bank)
    code_root = Path(__file__).parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    from importlib.metadata import version as package_version
    manifest = dict(schema=1, created_utc=datetime.now(UTC).isoformat(), diagnostic=diagnostic,
        loop_schema=2, workers=workers,
        configuration=config.to_dict(), model=model_identity, counts=list(counts), cases_per_count=cases,
        episode_cap=max_steps, decision_budget_seconds=config.resource_allocation.decision_budget_seconds,
        bootstrap_samples=config.evaluation.bootstrap_samples, git_commit=commit,
        code_sha256={p.name: sha256(p) for p in code_root.glob("*.py")},
        python=platform.python_version(), packages={n: package_version(n) for n in
            ("numpy", "torch", "simpy", "cvxpy", "clarabel")},
        map_sha256=sha256(config.resolve(FIXED_MAP)), case_bank_sha256=sha256(output/"held_out_cases.json"),
        timeline_sha256=sha256(config.communication.leo_timeline) if config.communication.leo_timeline else None,
        assumptions="Synthetic task/RF assumptions; fractional solutions are bounds; runtime executes binary routes",
        conditions=[dict(name=name, tags=tags, configuration=cfg.to_dict()) for name, cfg, tags in choices])
    write_json(output/"manifest.json", manifest)
    write_json(output/"status.json", dict(status="running", completed_episodes=0))
    completed = 0
    executor = (ProcessPoolExecutor(max_workers=workers, initializer=_worker_initialize,
        initargs=(str(Path(model_path).resolve()) if model_path else None, config.to_dict(), diagnostic, device_name))
        if workers > 1 else None)
    try:
        for name, cfg, tags in choices:
            directory = output/name
            directory.mkdir()
            cfg = replace(cfg, resource_allocation=replace(cfg.resource_allocation,
                failure_directory=str(directory/"failures")))
            if executor:
                results = executor.map(_worker_episode, [(_case_config(cfg, case), case, version, max_steps) for case in bank])
            else:
                results = (run_episode(_case_config(cfg, case), model, case, device, version,
                    decision_deadline_seconds=cfg.resource_allocation.decision_budget_seconds,
                    max_steps=max_steps) for case in bank)
            for case, (row, trace) in zip(bank, results):
                if trace["frames"][0]["positions"] != case["starts"]:
                    raise AssertionError("Paired case starts changed")
                row = episode_metrics(row, trace)
                row.update(condition=name, diagnostic=diagnostic, policy_model_sha256=model_identity["sha256"], **tags)
                with gzip.open(directory/f"{case['case_id']}.json.gz", "wt", encoding="utf-8") as stream:
                    json.dump(trace, stream, allow_nan=False, separators=(",", ":"))
                append_jsonl(directory/"episodes.jsonl", row)
                for job in trace["resource_tasks"]:
                    append_jsonl(directory/"tasks.jsonl", dict(case_id=case["case_id"], **job))
                for event in trace["transport_events"]:
                    if event["event"] == "paper_allocation":
                        append_jsonl(directory/"allocations.jsonl", dict(case_id=case["case_id"], **event))
                completed += 1
                write_json(output/"status.json", dict(status="running", condition=name,
                    completed_episodes=completed, planned_episodes=len(bank)*len(choices)))
                if completed % 50 == 0:
                    print(f"{completed}/{len(bank)*len(choices)} episodes; {name}", flush=True)
            export_csv(directory/"episodes.jsonl", directory/"episodes.csv")
            if (directory/"tasks.jsonl").exists():
                export_csv(directory/"tasks.jsonl", directory/"tasks.csv")
        summary = summarize(output)
        plot_results(output)
    except Exception as exc:
        write_json(output/"status.json", dict(status="failed", error=str(exc), completed_episodes=completed))
        raise
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    write_json(output/"status.json", dict(status="complete", completed_episodes=completed,
        diagnostic=diagnostic))
    return summary
