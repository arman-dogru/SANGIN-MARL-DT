"""Editable scientific figures from recorded learning/inference JSONL only.

Customize COLORS, LABELS, figure sizes and the plotting functions below.
Example:
  python scripts/plot_learning_and_inference.py --rolling 50 --dpi 220
Raw experiment records are never edited or extended by this script.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html as html_lib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ("clean", "uplink_delayed", "downlink_delayed", "combined")
COLORS = {"clean": "#2474a8", "uplink_delayed": "#d68a22",
          "downlink_delayed": "#8b62ae", "combined": "#239777"}
LABELS = {"clean": "Clean", "uplink_delayed": "Uplink delayed",
          "downlink_delayed": "Downlink delayed", "combined": "Combined"}
FIELDS = {"task": "Stale A* guidance", "intentions": "Stale intentions",
          "priorities": "Stale priorities", "all": "All three stale"}
ROUTE_COLORS = {"uav": "#2474a8", "cloud": "#d68a22"}
INFERENCE_NOTE = "Exploratory frozen Clean checkpoint; eight-agent training qualification was not reached."
MINIMUM_STEP_AXIS_SPAN = 5.0


def read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def export(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def rolling(values, window):
    result = []
    for index in range(len(values)):
        segment = np.asarray(values[max(0, index-window+1):index+1], dtype=float)
        finite = segment[np.isfinite(segment)]
        result.append(float(finite.mean()) if len(finite) else np.nan)
    return result


def finish(fig, output, name, dpi, inference=False):
    if inference:
        fig.text(.5, .006, INFERENCE_NOTE, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .025 if inference else 0, 1, .96))
    fig.savefig(output / f"{name}.png", dpi=dpi)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def stage_data(training_root):
    datasets, summaries = {}, []
    for condition in CONDITIONS:
        folder = training_root / condition
        episodes = read(folder / "training.jsonl")
        validations = read(folder / "validation.jsonl")
        cases = read(folder / "validation_cases.jsonl")
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        datasets[condition] = {"episodes": episodes, "validations": validations,
                               "validation_cases": cases, "status": status}
        for stage, count in enumerate((2, 4, 6, 8)):
            group = [r for r in episodes if r["stage_index"] == stage]
            checks = [r for r in validations if r["stage_index"] == stage and r["complete"]]
            last = checks[-1] if checks else None
            qualified = next((r for r in checks if r["success_rate"] >= .95), None)
            val_cases = [r for r in cases if last and r["stage_index"] == stage
                         and r["stage_episode"] == last["stage_episode"]]
            successes = [r["steps"] for r in group if r["success"]]
            val_successes = [r["steps"] for r in val_cases if r["success"]]
            summaries.append({
                "condition": condition, "agents": count,
                "completed_training_episodes": len(group),
                "training_successes": sum(r["success"] for r in group),
                "training_success_rate": sum(r["success"] for r in group)/len(group) if group else None,
                "training_mean_successful_makespan": float(np.mean(successes)) if successes else None,
                "training_mean_par2": float(np.mean([r["par2"] for r in group])) if group else None,
                "completed_training_joint_steps": sum(r["steps"] for r in group),
                "qualified": qualified is not None,
                "episodes_to_qualification": qualified["stage_episode"] if qualified else None,
                "cumulative_episodes_to_qualification": sum(r["stage_index"] <= stage for r in episodes)
                    if qualified else None,
                "global_steps_at_qualification": qualified["generated_joint_steps"] if qualified else None,
                "wall_seconds_at_qualification": qualified["wall_seconds"] if qualified else None,
                "validation_checks": len(checks), "validation_episodes": sum(r["episodes"] for r in checks),
                "best_validation_success_rate": max((r["success_rate"] for r in checks), default=None),
                "last_validation_success_rate": last["success_rate"] if last else None,
                "last_validation_stage_episode": last["stage_episode"] if last else None,
                "last_validation_successful_makespan": float(np.mean(val_successes)) if val_successes else None,
                "last_validation_par2": last["par2"] if last else None,
                "last_validation_completion_fraction": float(np.mean([r["completion_fraction"] for r in val_cases])) if val_cases else None,
                "last_validation_collision_attempts": float(np.mean([r["collision_attempts"] for r in val_cases])) if val_cases else None,
                "last_validation_uav_travel_m": float(np.mean([r["uav_travel_m"] for r in val_cases])) if val_cases else None,
            })
    return datasets, summaries


def training_plots(datasets, summary, output, window, dpi):
    for stage, count in enumerate((2, 4, 6, 8)):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for condition, data in datasets.items():
            group = [r for r in data["episodes"] if r["stage_index"] == stage]
            checks = [r for r in data["validations"] if r["stage_index"] == stage and r["complete"]]
            if not group:
                continue
            start = group[0]["generated_joint_steps"] - group[0]["steps"]
            color, label = COLORS[condition], LABELS[condition]
            episodes = [r["stage_episode"] for r in group]
            local_steps = [r["generated_joint_steps"]-start for r in group]
            success = 100*np.array(rolling([r["success"] for r in group], window))
            for ax, x, xkey in ((axes[0, 0], episodes, "stage_episode"),
                               (axes[0, 1], local_steps, "generated_joint_steps")):
                ax.plot(x, success, color=color, label=label, linewidth=1.4)
                ax.scatter([r[xkey]-(start if xkey == "generated_joint_steps" else 0) for r in checks],
                           [100*r["success_rate"] for r in checks], color=color, s=30, zorder=5,
                           edgecolors="white", linewidths=.5)
                ax.set_ylim(0, 100)
            axes[1, 0].plot(episodes, rolling([r["par2"] for r in group], window), color=color, label=label)
            axes[1, 0].scatter([r["stage_episode"] for r in checks], [r["par2"] for r in checks], color=color, s=28)
            makespan = [r["steps"] if r["success"] else np.nan for r in group]
            axes[1, 1].plot(episodes, rolling(makespan, window), color=color, label=label)
            for check in checks:
                successful = [r["steps"] for r in data["validation_cases"]
                              if r["stage_index"] == stage and r["stage_episode"] == check["stage_episode"]
                              and r["success"]]
                if successful:
                    axes[1, 1].scatter(check["stage_episode"], np.mean(successful),
                                       color=color, s=28, edgecolors="white", linewidths=.5)
        axes[0, 0].set(xlabel="Training episodes in phase", ylabel="Success (%)")
        axes[0, 1].set(xlabel="Generated joint steps in phase", ylabel="Success (%)")
        axes[1, 0].set(xlabel="Training episodes in phase", ylabel="Failure-aware PAR-2 (steps)")
        axes[1, 1].set(xlabel="Training episodes in phase", ylabel="Successful makespan (steps)")
        for ax in axes.flat:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"{count} agents — recorded training means ({window} episodes); greedy validation dots")
        finish(fig, output, f"learning_stage_{count}_agents", dpi)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(4)
    for index, condition in enumerate(CONDITIONS):
        group = [r for r in summary if r["condition"] == condition]
        positions = x+(index-1.5)*.19
        bars = axes[0].bar(positions, [r["completed_training_episodes"] for r in group],
                           width=.18, color=COLORS[condition], label=LABELS[condition])
        for bar, row in zip(bars, group):
            if not row["qualified"]:
                bar.set_hatch("//")
            axes[0].annotate(str(row["completed_training_episodes"]),
                             (bar.get_x()+bar.get_width()/2, bar.get_height()), ha="center", va="bottom", fontsize=7)
        axes[1].bar(positions, [100*r["last_validation_success_rate"] for r in group],
                    width=.18, color=COLORS[condition], label=LABELS[condition])
    for ax in axes:
        ax.set_xticks(x, ["2 agents", "4 agents", "6 agents", "8 agents"])
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Completed training episodes (hatched = unqualified)")
    axes[0].set_ylim(0, max(r["completed_training_episodes"] for r in summary)*1.14)
    axes[1].set(ylabel="Final greedy validation success (%)", ylim=(0, 100))
    axes[1].axhline(95, color="black", linestyle=":", linewidth=1)
    fig.suptitle("Curriculum effort and final validation — 200 cases per check")
    finish(fig, output, "curriculum_comparison", dpi)
    common_steps = min(data["status"]["generated_joint_steps"] for data in datasets.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for condition, data in datasets.items():
        group = [r for r in data["episodes"] if r["generated_joint_steps"] <= common_steps]
        checks = [r for r in data["validations"] if r["complete"] and r["generated_joint_steps"] <= common_steps]
        color = COLORS[condition]
        axes[0].plot([r["generated_joint_steps"] for r in group],
                     100*np.asarray(rolling([r["success"] for r in group], window)),
                     color=color, label=LABELS[condition])
        for stage, marker in enumerate(("o", "s", "^", "D")):
            subset = [r for r in checks if r["stage_index"] == stage]
            axes[0].scatter([r["generated_joint_steps"] for r in subset],
                            [100*r["success_rate"] for r in subset], marker=marker,
                            color=color, edgecolors="white", s=30, linewidths=.5)
        qualified = [r for r in summary if r["condition"] == condition and r["qualified"]]
        axes[1].step([0]+[r["global_steps_at_qualification"] for r in qualified]+[common_steps],
                     [0]+list(range(1,len(qualified)+1))+[len(qualified)], where="post",
                     color=color, label=LABELS[condition])
    axes[0].set(xlabel="Generated joint steps", ylabel="Success (%)", ylim=(0,100))
    axes[1].set(xlabel="Generated joint steps", ylabel="Qualified phases", yticks=range(5), ylim=(-.1,4.1))
    for ax in axes:
        ax.set_xlim(0,common_steps)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Common generated-step range: 0–{common_steps:,}; validation markers: 2 / 4 / 6 / 8 agents")
    finish(fig, output, "learning_common_generated_steps", dpi)
    fig,axes = plt.subplots(2,2,figsize=(12,8))
    for stage,ax in enumerate(axes.flat):
        for condition,data in datasets.items():
            checks = [r for r in data["validations"] if r["stage_index"] == stage and r["complete"]]
            ax.plot([r["stage_episode"] for r in checks],[100*r["success_rate"] for r in checks],
                    "o-",color=COLORS[condition],label=LABELS[condition],markersize=5)
        ax.axhline(95,color="black",linestyle=":",linewidth=1)
        ax.set(title=f"{(2,4,6,8)[stage]} agents",xlabel="Training episodes in phase",
               ylabel="Greedy validation success (%)",ylim=(0,100))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Learning under each transport condition — 200-case greedy validation at recorded checks")
    finish(fig,output,"greedy_validation_by_stage",dpi)
    fig,axes = plt.subplots(2,2,figsize=(13,8))
    common_received = min(d["status"]["delivered_experiences"] for d in datasets.values())
    for name,data in datasets.items():
        color = COLORS[name]
        for column,key,scale in ((0,"wall_seconds",3600),(1,"delivered_experiences",1)):
            group = [r for r in data["episodes"] if key != "delivered_experiences" or r[key] <= common_received]
            axes[0,column].plot([r[key]/scale for r in group],
                                100*np.asarray(rolling([r["success"] for r in group],window)),
                                color=color,label=LABELS[name])
            qualified = [r for r in summary if r["condition"] == name and r["qualified"]]
            points = []
            for r in qualified:
                episode = next(e for e in data["episodes"] if e["stage_index"] == (r["agents"]-2)//2 and e["stage_episode"] == r["episodes_to_qualification"])
                points.append(r["wall_seconds_at_qualification"]/scale if key == "wall_seconds" else episode[key]/scale)
            bound = data["status"]["wall_seconds"]/3600 if key == "wall_seconds" else common_received
            axes[1,column].step([0]+points+[bound],[0]+list(range(1,len(points)+1))+[len(points)],where="post",color=color,label=LABELS[name])
            for stage,marker in enumerate(("o","s","^","D")):
                checks = [r for r in data["validations"] if r["stage_index"] == stage and r["complete"]]
                x,y = [],[]
                for check in checks:
                    ep = next(e for e in data["episodes"] if e["stage_index"] == stage and e["stage_episode"] == check["stage_episode"])
                    value = check["wall_seconds"] if key == "wall_seconds" else ep[key]
                    if key == "delivered_experiences" and value > common_received:continue
                    x.append(value/scale);y.append(100*check["success_rate"])
                axes[0,column].scatter(x,y,color=color,marker=marker,s=27,edgecolors="white",linewidths=.5)
    for column,label in enumerate(("Elapsed wall-clock hours (including validation)","Received agent experiences")):
        axes[0,column].set(ylabel="Success (%)",ylim=(0,100))
        axes[1,column].set(ylabel="Qualified phases",yticks=range(5),ylim=(-.1,4.1))
        for ax in axes[:,column]:
            ax.set_xlabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
            if column == 1:ax.set_xlim(0,common_received)
    fig.suptitle("Learning progress by real time and received samples — training curves and greedy validation dots")
    finish(fig,output,"learning_wallclock_and_received",dpi)


def inference_data(root):
    summary, records = [], {}
    audited = {}
    audit_path = root/"transport_event_audit.csv"
    if audit_path.exists():
        with audit_path.open(encoding="utf-8",newline="") as stream:
            audited = {(r["suite"],r["condition"],r["case_id"]):r for r in csv.DictReader(stream)}
    for suite in ("planned", "orbital", "peer-cutoff", "latency-ablation", "empirical"):
        suite_root = root/suite if suite != "empirical" else root/"data-trace"/"empirical"
        path = suite_root/"summary.json"
        if not path.exists():
            continue
        manifest = json.loads((suite_root/"manifest.json").read_text(encoding="utf-8"))
        protocols = {r["name"]:r["transport"] for r in manifest["conditions"]}
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = (suite, row["condition"])
            if key not in records:
                records[key] = read(suite_root/row["condition"]/"episodes.jsonl")
                for case in records[key]:
                    event_audit = audited.get((*key,case["case_id"]))
                    if event_audit:
                        for metric in ("transmissions_interrupted_by_handover", "handover_wait_seconds", "handover_windows_elapsed"):
                            case[metric] = float(event_audit[metric])
                        case["maximum_delivery_age_seconds"] = float(event_audit["maximum_delivery_age_seconds"])
            cases = records[key]
            group = [r for r in cases if r["num_agents"] == row["num_agents"]]
            row = {"suite": suite, **row}
            protocol = protocols[row["condition"]]
            row.update(configured_additional_delay_steps=protocol["delay_steps"],
                       reservation_window_steps=protocol["reservation_steps"],
                       contention_enabled=protocol["contention_enabled"],
                       handover_enabled=protocol["handover_enabled"],
                       seconds_per_robot_step=protocol["seconds_per_step"])
            for metric in ("mean_task_age_seconds", "mean_intentions_age_seconds", "mean_priorities_age_seconds",
                           "maximum_task_age_seconds", "maximum_intentions_age_seconds", "maximum_priorities_age_seconds",
                           "exposed_agents", "exposed_messages", "maximum_queue_bytes", "handover_wait_seconds",
                           "transmissions_interrupted_by_handover", "handover_windows_elapsed", "peer_messages_dropped",
                           "maximum_delivery_age_seconds"):
                values = [r[metric] for r in group if metric in r]
                if metric.startswith("maximum_"):
                    row[metric] = float(max(values)) if values else None
                elif metric.startswith("mean_"):
                    row[metric] = float(np.mean(values)) if values else None
                else:
                    row["mean_"+metric] = float(np.mean(values)) if values else None
            row["transport_statistics_source"] = "recorded event audit" if row["condition"].startswith("physical_") and audited else "episode records"
            summary.append(row)
    return summary, records


def inference_plots(summary, output, dpi):
    planned = [r for r in summary if r["suite"] == "planned"]
    for count in (2, 4, 6, 8):
        base = next((r for r in planned if r["condition"] == "perfect" and r["num_agents"] == count), None)
        if not base:
            continue
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        for col, field in enumerate(FIELDS):
            for route in ROUTE_COLORS:
                group = [r for r in planned if r["num_agents"] == count
                         and r["condition"].startswith(f"controlled_{route}_{field}_")]
                group.sort(key=lambda r: float(r["condition"].rsplit("_", 1)[1]))
                points = [base]+group
                x = [0]+[float(r["condition"].rsplit("_", 1)[1]) for r in group]
                for ax, key, scale in zip(axes[:, col], ("success_rate", "par2", "successful_makespan"), (100, 1, 1)):
                    ax.plot(x, [r[key]*scale if r[key] is not None else np.nan for r in points],
                            color=ROUTE_COLORS[route], linestyle="-" if route == "uav" else "--",
                            marker="o" if route == "uav" else "s", markersize=4,
                            markerfacecolor=ROUTE_COLORS[route] if route == "uav" else "none",
                            label="UAV edge" if route == "uav" else "Cloud")
                    ax.grid(alpha=.2)
                    ax.set_xlabel("Additional delay (steps)")
                axes[0, col].set(title=FIELDS[field], ylim=(0, 100))
                axes[0, col].legend(fontsize=8)
        for row, label in enumerate(("Success (%)", "Failure-aware PAR-2 (steps)", "Successful makespan (steps)")):
            axes[row, 0].set_ylabel(label)
            if row:
                values = [float(v) for ax in axes[row] for line in ax.lines
                          for v in line.get_ydata() if np.isfinite(v)]
                if values:
                    low,high = min(values),max(values)
                    span = max(high-low,MINIMUM_STEP_AXIS_SPAN)
                    center = (high+low)/2
                    for ax in axes[row]:
                        ax.set_ylim(max(0,center-.6*span),center+.6*span)
        fig.suptitle(f"{count} agents — frozen Clean policy; 50 paired held-out cases per condition")
        finish(fig, output, f"inference_staleness_{count}_agents", dpi, True)
    for suite in ("planned", "orbital", "peer-cutoff", "empirical"):
        rows = [r for r in summary if r["suite"] == suite]
        if not rows:
            continue
        names = ["perfect"]+sorted({r["condition"] for r in rows if
                 (r["condition"].startswith("physical_") if suite != "peer-cutoff" else r["condition"].startswith("peer_cutoff_"))},
                 key=lambda n: int(n.rsplit("_", 1)[1]) if suite == "peer-cutoff" else n)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, count in zip(axes.flat, (2, 4, 6, 8)):
            group = [next((r for r in rows if r["condition"] == name and r["num_agents"] == count),None) for name in names]
            if any(r is None for r in group):
                ax.set_visible(False)
                continue
            ax.bar(np.arange(len(group)), [100*r["success_rate"] for r in group], color="#2474a8")
            ax.set_xticks(np.arange(len(group)), [n.replace("physical_", "").replace("peer_cutoff_", "cutoff ").replace("_", " ") for n in names], rotation=35, ha="right", fontsize=8)
            ax.set(title=f"{count} agents", ylabel="Success (%)", ylim=(0, 100))
            ax.grid(axis="y", alpha=.2)
            for i, r in enumerate(group):
                ax.text(i, r["success_rate"]*100-5, f"{100*r['success_rate']:g}", ha="center", fontsize=8, color="white")
        fig.suptitle({"planned": "Physical latency, handover and resource contention (parameterized model)",
                      "orbital": "Mont-Tremblant archived orbital simulation (assumed handshake and capacities)",
                      "empirical": "Imported Starlink one-way delays (Karlstad measurement; not a Quebec measurement)",
                      "peer-cutoff": "Temporary loss of peer intentions/priorities; local sensing and goals stay fresh"}[suite])
        finish(fig, output, f"inference_{suite}_comparison", dpi, True)
    peer_rows = [r for r in summary if r["suite"] == "peer-cutoff"]
    if peer_rows:
        fig,axes = plt.subplots(3,4,figsize=(15,9),sharex="col")
        for column,count in enumerate((2,4,6,8)):
            base = next(r for r in peer_rows if r["condition"]=="perfect" and r["num_agents"]==count)
            group = sorted([r for r in peer_rows if r["num_agents"]==count and r["condition"].startswith("peer_cutoff_")],
                           key=lambda r:int(r["condition"].rsplit("_",1)[1]))
            points = [base]+group;x = [0]+[int(r["condition"].rsplit("_",1)[1]) for r in group]
            for row,(key,scale,label) in enumerate((("success_rate",100,"Success (%)"),
                                                   ("par2",1,"Failure-aware PAR-2 (steps)"),
                                                   ("successful_makespan",1,"Successful makespan (steps)"))):
                ax=axes[row,column]
                y=[r[key]*scale if r[key] is not None else np.nan for r in points]
                ax.plot(x,y,"o-",color=COLORS["clean"])
                ax.axhline(y[0],color="#555555",linestyle=":",linewidth=1)
                ax.grid(alpha=.2)
                if row==0:ax.set(title=f"{count} agents",ylim=(0,100))
                elif any(np.isfinite(y)):
                    finite=[v for v in y if np.isfinite(v)];center=(min(finite)+max(finite))/2
                    span=max(max(finite)-min(finite),MINIMUM_STEP_AXIS_SPAN)
                    ax.set_ylim(max(0,center-.6*span),center+.6*span)
                if column==0:ax.set_ylabel(label)
                if row==2:ax.set_xlabel("Peer cutoff duration (steps)")
        fig.suptitle("Temporary loss of peer updates — frozen Clean policy; dotted line = perfect communication")
        finish(fig,output,"peer_cutoff_outcomes",dpi,True)
        fig,axes = plt.subplots(1,3,figsize=(14,5))
        for count,color in zip((2,4,6,8),COLORS.values()):
            group = sorted([r for r in peer_rows if r["num_agents"]==count and r["condition"].startswith("peer_cutoff_")],
                           key=lambda r:int(r["condition"].rsplit("_",1)[1]))
            x=[0]+[int(r["condition"].rsplit("_",1)[1]) for r in group]
            metrics=([0]+[100*r["mean_exposed_agents"]/count for r in group],
                     [0]+[r["mean_intentions_age_seconds"] for r in group],
                     [0]+[r["maximum_intentions_age_seconds"] for r in group])
            for ax,y in zip(axes,metrics):
                ax.plot(x,y,"o-",color=color,label=f"{count} agents")
        for ax,label in zip(axes,("Agents with realized cutoff exposure (%)","Mean intentions age per mission (s)",
                                  "Maximum observed intentions age (s)")):
            ax.set(xlabel="Peer cutoff duration (steps)",ylabel=label)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0].set_ylim(0,100)
        fig.suptitle("Recorded peer-update losses and resulting information age — local sensing and task guidance stay fresh")
        finish(fig,output,"peer_cutoff_exposure",dpi,True)
    ablation = [r for r in summary if r["suite"] == "latency-ablation"]
    if ablation:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for name,label,color in (("perfect","Perfect","#555555"),
                                  ("physical_uav_task","A* only",COLORS["clean"]),
                                  ("physical_uav_intentions","Intentions only",COLORS["uplink_delayed"]),
                                  ("physical_uav_priorities","Priorities only",COLORS["downlink_delayed"]),
                                  ("physical_uav_latency","All three",COLORS["combined"])):
            group = [r for r in (planned if name == "physical_uav_latency" else ablation) if r["condition"] == name]
            group.sort(key=lambda r:r["num_agents"])
            if not group:
                continue
            for ax,key,scale in zip(axes.flat,("success_rate","par2","completion_fraction","collision_attempts"),(100,1,100,1)):
                ax.plot([r["num_agents"] for r in group],[scale*r[key] for r in group],
                        "o-",color=color,label=label)
        axes[0,0].set(ylabel="Success (%)",ylim=(0,100))
        axes[0,1].set_ylabel("Failure-aware PAR-2 (steps)")
        axes[1,0].set(ylabel="Agents at goals at episode end (%)",ylim=(0,100))
        axes[1,1].set_ylabel("Blocked collision attempts per episode")
        for ax in axes.flat:
            ax.set(xlabel="Number of robots",xticks=[2,4,6,8])
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle("Continuous physical access latency — selected information fields; UAV-local route")
        finish(fig,output,"physical_latency_field_isolation",dpi,True)
    if planned:
        fig,axes = plt.subplots(1,2,figsize=(13,5))
        for route,color in ROUTE_COLORS.items():
            for suffix,style in (("latency","-"),("handover","--"),("contention",":"),("combined","-.")):
                group = sorted([r for r in planned if r["condition"] == f"physical_{route}_{suffix}"],key=lambda r:r["num_agents"])
                for ax,key,scale in zip(axes,("success_rate","par2"),(100,1)):
                    ax.plot([r["num_agents"] for r in group],[scale*r[key] for r in group],
                            color=color,linestyle=style,marker="o",label=f"{route.title()} {suffix}")
        axes[0].set(ylabel="Success (%)",ylim=(0,100));axes[1].set_ylabel("Failure-aware PAR-2 (steps)")
        for ax in axes:
            ax.set(xlabel="Number of robots",xticks=[2,4,6,8]);ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
        fig.suptitle("Persistent latency versus added handover or contention — physical-profile controls")
        finish(fig,output,"physical_transport_effects",dpi,True)


def paired_contrast(group, reference):
    baseline = {r["case_id"]: r for r in reference}
    assert len(group) == len(reference) and set(baseline) == {r["case_id"] for r in group}
    rng = np.random.default_rng(4200)
    indices = rng.integers(0, len(group), (2000, len(group)))
    result = {}
    for key, label in (("success", "success"), ("par2", "par2"),
                       ("completion_fraction","completion_fraction"),
                       ("collision_attempts","collision_attempts"),("uav_travel_m","uav_travel_m")):
        differences = np.array([r[key]-baseline[r["case_id"]][key] for r in group], dtype=float)
        interval = np.quantile(differences[indices].mean(axis=1), [.025, .975])
        result[label+"_delta"] = float(differences.mean())
        result[label+"_ci_low"] = float(interval[0])
        result[label+"_ci_high"] = float(interval[1])
    common = [r for r in group if r["success"] and baseline[r["case_id"]]["success"]]
    result["common_successful_cases"] = len(common)
    result["makespan_delta_common_successes"] = float(np.mean([r["steps"]-baseline[r["case_id"]]["steps"] for r in common])) if common else None
    result["baseline_makespan_common_successes"] = float(np.mean([baseline[r["case_id"]]["steps"] for r in common])) if common else None
    result["lost_successes"] = sum(not r["success"] and baseline[r["case_id"]]["success"] for r in group)
    result["rescued_failures"] = sum(r["success"] and not baseline[r["case_id"]]["success"] for r in group)
    return result


def inference_effects(summary, records, inference_root, training_root, output, dpi):
    contrasts = []
    for row in summary:
        suite, condition, count = row["suite"], row["condition"], row["num_agents"]
        if condition == "perfect":
            continue
        reference_name = "perfect"
        if condition.startswith("physical_") and condition.endswith(("handover", "contention")):
            reference_name = condition.rsplit("_", 1)[0]+"_latency"
        elif condition.startswith("physical_") and condition.endswith("combined"):
            reference_name = condition.rsplit("_", 1)[0]+"_contention"
        group = [r for r in records[(suite, condition)] if r["num_agents"] == count]
        reference = [r for r in records[(suite, reference_name)] if r["num_agents"] == count]
        contrasts.append({"suite": suite, "condition": condition, "agents": count,
                          "reference_condition": reference_name, "paired_cases": len(group),
                          **paired_contrast(group, reference)})
    export(output/"paired_inference_effects.csv", contrasts)
    if not contrasts or ("planned","perfect") not in records:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for index, field in enumerate(FIELDS):
        group = [r for r in contrasts if r["suite"] == "planned"
                 and r["condition"] == f"controlled_cloud_{field}_20"]
        group.sort(key=lambda r: r["agents"])
        if not group:
            continue
        x = np.arange(len(group))+(index-1.5)*.13
        for ax, key, scale in zip(axes, ("success", "par2"), (100, 1)):
            y = np.array([r[key+"_delta"]*scale for r in group])
            low = np.array([r[key+"_ci_low"]*scale for r in group])
            high = np.array([r[key+"_ci_high"]*scale for r in group])
            ax.errorbar(x, y, yerr=[y-low, high-y], fmt="o", capsize=3,
                        label=FIELDS[field], color=list(COLORS.values())[index])
    for ax, label in zip(axes, ("Success change (percentage points)", "PAR-2 change (steps)")):
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(4), ["2 agents", "4 agents", "6 agents", "8 agents"])
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("20-step cloud-route staleness versus perfect communication — paired bootstrap 95% intervals")
    finish(fig, output, "paired_mission_effects_20_steps", dpi, True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for field in FIELDS:
        group = [r for r in summary if r["suite"] == "planned" and r["num_agents"] == 6
                 and r["condition"].startswith(f"controlled_cloud_{field}_")]
        group.sort(key=lambda r: int(r["condition"].rsplit("_", 1)[1]))
        if not group:
            continue
        x = [int(r["condition"].rsplit("_", 1)[1]) for r in group]
        axes[0].plot(x, [100*r["mean_exposed_agents"]/6 for r in group], "o-", label=FIELDS[field])
        age_key = "maximum_task_age_seconds" if field in ("task", "all") else f"maximum_{field}_age_seconds"
        axes[1].plot(x, [r[age_key] for r in group], "o-", label=FIELDS[field])
    axes[0].set(xlabel="Additional delay (steps)", ylabel="Agents with realized reservation exposure (%)", ylim=(0,100))
    axes[1].set(xlabel="Additional delay (steps)", ylabel="Maximum observed selected-information age (s)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Six agents — scheduled windows are not assumed to be experienced")
    finish(fig, output, "realized_staleness_exposure", dpi, True)
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex="col")
    names = ["perfect", "controlled_cloud_all_1", "controlled_cloud_all_5", "controlled_cloud_all_20"]
    for col, count in enumerate((2, 4, 6, 8)):
        for row, field in enumerate(("task", "intentions", "priorities")):
            samples = [[r[f"mean_{field}_age_seconds"] for r in records[("planned", name)] if r["num_agents"] == count] for name in names]
            axes[row, col].boxplot(samples, tick_labels=["0", "1", "5", "20"], showfliers=True)
            axes[row, col].grid(axis="y", alpha=.2)
            if col == 0:
                axes[row, col].set_ylabel(field+" mean age per mission (s)")
        axes[0, col].set_title(f"{count} agents")
        axes[-1, col].set_xlabel("All-field additional delay (steps)")
    fig.suptitle("Distribution of episode-mean information age — 50 paired missions per agent count")
    finish(fig, output, "information_age_distributions", dpi, True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, count in zip(axes.flat, (2, 4, 6, 8)):
        base = next(r for r in summary if r["suite"] == "planned" and r["condition"] == "perfect" and r["num_agents"] == count)
        for field, color in zip(FIELDS, COLORS.values()):
            group = [r for r in summary if r["suite"] == "planned" and r["num_agents"] == count
                     and r["condition"].startswith(f"controlled_cloud_{field}_")]
            group.sort(key=lambda r: int(r["condition"].rsplit("_", 1)[1]))
            ax.plot([0]+[int(r["condition"].rsplit("_", 1)[1]) for r in group],
                    [base["uav_travel_m"]]+[r["uav_travel_m"] for r in group], "o-", label=FIELDS[field], color=color)
        ax.set(title=f"{count} agents", xlabel="Additional delay (steps)", ylabel="Mean episode UAV travel (m)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Measured UAV travel under frozen-policy information staleness — cloud route")
    finish(fig, output, "inference_uav_travel", dpi, True)
    fig,axes = plt.subplots(2,4,figsize=(16,8))
    for column,count in enumerate((2,4,6,8)):
        base = next(r for r in summary if r["suite"] == "planned" and r["condition"] == "perfect" and r["num_agents"] == count)
        for field,color in zip(FIELDS,COLORS.values()):
            group = sorted([r for r in summary if r["suite"] == "planned" and r["num_agents"] == count
                            and r["condition"].startswith(f"controlled_cloud_{field}_")],
                           key=lambda r:int(r["condition"].rsplit("_",1)[1]))
            points = [base]+group;x = [0]+[int(r["condition"].rsplit("_",1)[1]) for r in group]
            for ax,key,scale in zip(axes[:,column],("completion_fraction","collision_attempts"),(100,1)):
                ax.plot(x,[r[key]*scale for r in points],"o-",color=color,label=FIELDS[field])
                ax.set_xlabel("Additional delay (steps)");ax.grid(alpha=.2);ax.legend(fontsize=7)
        axes[0,column].set(title=f"{count} agents",ylim=(0,100))
        axes[1,column].set_ylim(bottom=0)
    axes[0,0].set_ylabel("Agents at goals at episode end (%)")
    axes[1,0].set_ylabel("Blocked collision attempts per episode")
    fig.suptitle("Mission completion and attempted conflicts — frozen Clean policy, cloud controlled-delay route")
    finish(fig,output,"mission_completion_and_conflicts",dpi,True)
    if (inference_root/"transport_event_audit.csv").exists():
        available = [suite for suite in ("planned","orbital") if any(r["suite"] == suite for r in summary)]
        fig,axes = plt.subplots(len(available),3,figsize=(15,4*len(available)),squeeze=False)
        for row,suite in enumerate(available):
            for suffix,color in zip(("latency","handover","contention","combined"),COLORS.values()):
                group = sorted([r for r in summary if r["suite"] == suite and r["condition"] == f"physical_cloud_{suffix}"],key=lambda r:r["num_agents"])
                for ax,key,scale in zip(axes[row],("mean_handover_wait_seconds","maximum_queue_bytes","maximum_delivery_age_seconds"),(1000,1,1)):
                    ax.plot([r["num_agents"] for r in group],[scale*r[key] for r in group],"o-",color=color,label=suffix.title())
                    ax.set_xticks([2,4,6,8]);ax.set_xlabel("Number of robots");ax.grid(alpha=.2);ax.legend(fontsize=7)
            for ax,label in zip(axes[row],("Mean cumulative satellite hold per episode (ms)","Maximum recorded queued / in-flight bytes","Maximum complete-message delivery age (s)")):
                ax.set_ylabel(label)
            axes[row,0].set_title("Parameterized satellite" if suite == "planned" else "Archived Mont-Tremblant orbit")
        fig.suptitle("Cloud-route transport telemetry from recorded packet events")
        finish(fig,output,"physical_transport_telemetry",dpi,True)
    # Use recorded traces, not rerun actions, for a clearly identified example.
    key = ("planned", "controlled_cloud_all_20")
    if key not in records:
        return
    stale_cases = [r for r in records[key] if r["num_agents"] == 6]
    fresh_cases = {r["case_id"]: r for r in records[("planned", "perfect")] if r["num_agents"] == 6}
    example = max(stale_cases, key=lambda r: (r["par2"]-fresh_cases[r["case_id"]]["par2"], r["steps"]-fresh_cases[r["case_id"]]["steps"]))
    traces = []
    for condition, result in (("perfect", fresh_cases[example["case_id"]]), (key[1], example)):
        path = inference_root/"planned"/condition/(example["case_id"]+".json.gz")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            trace = json.load(stream)
        trace["result"] = result
        trace["contention_enabled"] = condition != "perfect"
        traces.append(trace)
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex="col")
    for column, trace in enumerate(traces):
        for row, field in enumerate(("task", "intentions", "priorities")):
            for agent in trace["frames"][0]["information_age"]:
                axes[row, column].plot([f["time"] for f in trace["frames"]],
                                      [f["information_age"][agent][field] for f in trace["frames"]],
                                      linewidth=1, label=f"Robot {agent}")
            axes[row, column].set_ylabel(field+" age (s)")
            axes[row, column].grid(alpha=.2)
        axes[0, column].set_title("Perfect" if column == 0 else "All fields delayed 20 steps")
        axes[-1, column].set_xlabel("Robot mission time (seconds)")
    axes[0, 1].legend(fontsize=7, ncol=3)
    fig.suptitle(f"Recorded six-agent example: {example['case_id']} — largest paired PAR-2 increase")
    finish(fig, output, "recorded_information_age_timeline", dpi, True)
    # The template itself is read directly; importing would require package state.
    source = (training_root/"code/panama_uav/replay_viewer.py").read_text(encoding="utf-8")
    template = source.split('HTML = r"""',1)[1].rsplit('"""',1)[0]
    template = template.replace("data.label+' · '+data.route+' mission route · additional contention delay '+data.delay+' steps'",
                                "data.label+' · '+data.route+' mission route · '+(data.description||('additional message delay '+data.delay+' steps'))")
    template = template.replace("filter(e=>e.event==='scheduled_reservation'", "filter(e=>t.contention_enabled!==false&&e.event==='scheduled_reservation'")
    template = template.replace("trace.result.par2+'</pre>'", "trace.result.par2+' · blocked collision attempts '+trace.result.collision_attempts+'</pre>'")
    html = template
    data = {"label": "Exploratory frozen Clean checkpoint; largest paired PAR-2 increase among 50 six-agent cases",
            "route": "cloud", "delay": 20, "traces": traces}
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    html = html.replace("Satellite handover + resource contention", "All mission information delayed 20 steps")
    html = html.replace("This replay records neural policy actions; no stopping or recovery controller is added.",
                        "Recorded neural-policy actions and receiver-cache ages are shown.")
    (output/"recorded_inference_replay.html").write_text(html, encoding="utf-8")
    (output/"recorded_example.json").write_text(json.dumps({"case_id":example["case_id"],
        "selection":"largest paired PAR-2 increase among 50 six-agent cases",
        "perfect_result":fresh_cases[example["case_id"]],"stale_result":example},indent=2),encoding="utf-8")
    # A separately identified continuous-latency failure; no case cherry-picked
    # to replace aggregate statistics. Use the first paired two-agent failure.
    latency_rows = records.get(("planned","physical_uav_latency"),[])
    all_fresh = {r["case_id"]:r for r in records[("planned","perfect")]}
    failure = next((r for r in latency_rows if r["num_agents"] == 2 and not r["success"] and all_fresh[r["case_id"]]["success"]),None)
    if failure:
        latency_traces = []
        for condition,result in (("perfect",all_fresh[failure["case_id"]]),("physical_uav_latency",failure)):
            with gzip.open(inference_root/"planned"/condition/(failure["case_id"]+".json.gz"),"rt",encoding="utf-8") as stream:
                trace = json.load(stream)
            trace["result"] = result;trace["contention_enabled"] = False;latency_traces.append(trace)
        payload = {"label":"First paired two-agent continuous-latency failure; exploratory frozen Clean model",
                   "route":"uav","delay":0,"traces":latency_traces,
                   "description":"positive access latency; no reservation or satellite handover"}
        replay = template
        replay = replay.replace("__DATA__",json.dumps(payload,separators=(",",":")).replace("</","<\\/"))
        replay = replay.replace("Satellite handover + resource contention","Continuous access latency; no satellite hop")
        replay = replay.replace("This replay records neural policy actions; no stopping or recovery controller is added.",
                                "Recorded neural-policy actions and receiver-cache ages are shown.")
        (output/"recorded_latency_replay.html").write_text(replay,encoding="utf-8")
    handover_cases = records.get(("planned","physical_cloud_handover"),[])
    controls = {r["case_id"]:r for r in records.get(("planned","physical_cloud_latency"),[])}
    example = next((r for r in handover_cases if r["num_agents"] == 2 and r["steps"]>15
                    and r["success"] and r["case_id"] in controls),None)
    if example:
        handover_traces = []
        for condition,result in (("physical_cloud_latency",controls[example["case_id"]]),("physical_cloud_handover",example)):
            with gzip.open(inference_root/"planned"/condition/(example["case_id"]+".json.gz"),"rt",encoding="utf-8") as stream:
                trace = json.load(stream)
            trace["result"] = result;trace["contention_enabled"] = False;handover_traces.append(trace)
        payload = {"label":"Recorded paired handover control; exploratory frozen Clean model",
                   "route":"cloud","delay":0,"traces":handover_traces,
                   "description":"15-second scheduling interval; 50 ms simulated interruption"}
        replay = template.replace("__DATA__",json.dumps(payload,separators=(",",":")).replace("</","<\\/"))
        replay = replay.replace('<h2>Perfect communication</h2>','<h2>Ordinary physical latency</h2>')
        replay = replay.replace("Satellite handover + resource contention","Physical latency + satellite handover")
        replay = replay.replace("This replay records neural policy actions; no stopping or recovery controller is added.",
                                "Recorded neural-policy actions and receiver-cache ages are shown.")
        (output/"recorded_handover_replay.html").write_text(replay,encoding="utf-8")


def report_page(stages, inference, output):
    def pct(value):
        return f"{100*value:g}%" if value is not None else "—"
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Measured learning and inference</title>',
            '<style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 24px;color:#24313f}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:24px 0}th,td{text-align:left;padding:10px;border-bottom:1px solid #ccd5de}a{color:#176799}small{color:#566675}.scroll{overflow:auto}h1,h2{margin-top:32px}</style>',
            '<h1>Measured learning and inference comparison</h1>',
            '<p>Four matched-initialization learning arms, one training seed (42), three hours per arm. Validation uses 200 greedy cases per check. Qualification requires 95%.</p>',
            '<p><a href="RESULTS.md">Findings and interpretation</a> · <a href="training_stage_summary.csv">Stage CSV</a> · <a href="training_totals.csv">Totals CSV</a> · <a href="inference_summary.csv">Inference CSV</a> · <a href="paired_inference_effects.csv">Paired effects CSV</a></p>',
            '<table><tr><th>Condition</th><th>2 agents</th><th>4 agents</th><th>6 agents</th><th>8 agents</th><th>Total episodes</th></tr>']
    for condition in CONDITIONS:
        group = [r for r in stages if r["condition"] == condition]
        cells = [f'{r["completed_training_episodes"]:,} / {pct(r["last_validation_success_rate"])}'+(' (partial)' if not r['qualified'] else '') for r in group]
        body.append('<tr><th>'+LABELS[condition]+'</th>'+''.join('<td>'+c+'</td>' for c in cells)+f'<td>{sum(r["completed_training_episodes"] for r in group):,}</td></tr>')
    body += ['</table><p>Cells show completed phase training episodes / last greedy validation success. For qualified phases, episodes also equal effort to qualification; eight-agent counts are censored by budget exhaustion.</p>',
             '<img src="curriculum_comparison.png" alt="Curriculum effort and validation">',
             '<img src="learning_common_generated_steps.png" alt="Common generated-step comparison">',
             '<img src="learning_wallclock_and_received.png" alt="Wall-clock and received-sample learning comparison">',
             '<img src="greedy_validation_by_stage.png" alt="Greedy validation by stage">']
    for n in (2,4,6,8):
        body.append(f'<h2>{n}-agent learning phase</h2><img src="learning_stage_{n}_agents.png" alt="{n}-agent learning curves">')
    if inference:
        body += ['<h2>Frozen-policy inference</h2><p>50 paired held-out cases per agent count and treatment; maximum 250 steps. Successful makespan conditions on success. PAR-2 includes failures at 500 steps.</p>',
                 f'<p>{INFERENCE_NOTE} These tests assess the retained partial model; they do not qualify the eight-agent curriculum phase.</p>',
                 '<p>Occasional controlled holds and continuous physical latency are different treatments. Positive physical latency introduces persistent one-step-old inputs under the synchronous action schedule, even when packet delay is below one second. Compare handovers against the latency-only control to isolate their additional effect.</p>']
        replays = [(filename,label) for filename,label in (
            ("recorded_inference_replay.html","Recorded occasional-delay replay"),
            ("recorded_latency_replay.html","Recorded continuous-latency failure"),
            ("recorded_handover_replay.html","Recorded handover control")) if (output/filename).exists()]
        if replays:
            body.append('<p>'+ ' · '.join(f'<a href="{filename}">{label}</a>' for filename,label in replays)+'</p>')
        for filename in ('physical_latency_field_isolation','physical_transport_effects','paired_mission_effects_20_steps',
                         'peer_cutoff_outcomes','peer_cutoff_exposure','inference_orbital_comparison','realized_staleness_exposure','physical_transport_telemetry','mission_completion_and_conflicts'):
            if (output/(filename+'.png')).exists():
                body.append(f'<img src="{filename}.png" alt="{html_lib.escape(filename)}">')
        for n in (2,4,6,8):
            if (output/f"inference_staleness_{n}_agents.png").exists():
                body.append(f'<h2>{n}-agent staleness sweep</h2><img src="inference_staleness_{n}_agents.png" alt="{n}-agent staleness sweep">')
        body += ['<h2>All inference treatments</h2><div class="scroll"><table><tr><th>Suite</th><th>Condition</th><th>Agents</th><th>Success</th><th>Successful makespan</th><th>PAR-2</th><th>Completion</th><th>Blocked attempts</th><th>UAV travel (m)</th></tr>']
        for r in inference:
            makespan = f'{r["successful_makespan"]:.2f}' if r['successful_makespan'] is not None else '—'
            body.append(f'<tr><td>{html_lib.escape(r["suite"])}</td><td>{html_lib.escape(r["condition"])}</td><td>{r["num_agents"]}</td><td>{pct(r["success_rate"])}</td><td>{makespan}</td><td>{r["par2"]:.2f}</td><td>{pct(r["completion_fraction"])}</td><td>{r["collision_attempts"]:.2f}</td><td>{r["uav_travel_m"]:.2f}</td></tr>')
        body += ['</table></div>',f'<p><small>{INFERENCE_NOTE} Archived orbital geometry, handshake assumptions and controlled holds are separate experiments; the short imported packet-delay traces do not cover these missions. All observations are recorded; no synthetic extension.</small></p>']
    body.append('<h2>Figure downloads</h2><ul>')
    for path in sorted(output.glob('*.png')):
        label=html_lib.escape(path.stem.replace('_',' '))
        pdf=path.with_suffix('.pdf')
        body.append(f'<li><a href="{path.name}">{label} — PNG</a>'+
                    (f' · <a href="{pdf.name}">PDF</a>' if pdf.exists() else '')+'</li>')
    body.append('</ul>')
    (output/'results.html').write_text('\n'.join(body),encoding='utf-8')


def write_findings(stages, inference, records, output):
    lines = ['# Measured learning and inference results', '',
             'All four learning conditions qualified 2, 4 and 6 agents; none qualified 8 agents within its three-hour allocation.', '',
             '## Learning effort', '',
             'Cells show completed phase training episodes / last 200-case greedy validation success. For qualified phases this is also the episode count to qualification. Eight-agent counts are partial.', '',
             '| Condition | 2 agents | 4 agents | 6 agents | 8 agents | Total completed |',
             '|---|---:|---:|---:|---:|---:|']
    for name in CONDITIONS:
        group = [r for r in stages if r['condition']==name]
        cells = [f'{r["completed_training_episodes"]:,} / {100*r["last_validation_success_rate"]:g}%'+(' (partial)' if not r['qualified'] else '') for r in group]
        lines.append('| '+LABELS[name]+' | '+' | '.join(cells)+f' | {sum(r["completed_training_episodes"] for r in group):,} |')
    lines += ['', 'At six agents, qualification took 2.6× as many episodes for uplink delays, 2.0× for downlink delays, and 2.2× for combined delays compared with Clean. Earlier phases do not show uniformly slower learning. These are matched runs with one training seed, not multi-seed causal estimates.', '',
              'Training curves include exploration; validation dots use the current greedy learner. Different episode totals follow from phase graduation, episode lengths and validation time within equal wall-clock budgets.', '',
              '## Frozen-policy mission tests', '',
              'The final Clean checkpoint is exploratory because eight-agent qualification was not achieved. It is evaluated greedily on 50 paired held-out cases per count, distinct from the curriculum validation bank. Goals, local sensing and collision truth remain available.', '',
              '| Agents | Perfect SR | A* delay 20 SR / makespan | All fields delay 20 SR / makespan | Peer cutoff 20 SR / makespan |',
              '|---|---:|---:|---:|---:|']
    lookup = {(r['suite'],r['condition'],r['num_agents']):r for r in inference}
    def value(suite,name,n):
        r = lookup.get((suite,name,n))
        if not r:return 'pending'
        ms = f'{r["successful_makespan"]:.2f}' if r['successful_makespan'] is not None else '—'
        return f'{100*r["success_rate"]:g}% / {ms}'
    for n in (2,4,6,8):
        base = lookup.get(('planned','perfect',n))
        if not base:continue
        lines.append(f'| {n} | {100*base["success_rate"]:g}% | '+
            value('planned','controlled_cloud_task_20',n)+' | '+
            value('planned','controlled_cloud_all_20',n)+' | '+
            value('peer-cutoff','peer_cutoff_20',n)+' |')
    lines += ['', 'Makespan above is conditional on success. Use `inference_summary.csv` and `paired_inference_effects.csv` for failure-aware PAR-2 and paired confidence intervals; one rescued or lost failure can dominate PAR-2.', '',
              'For a comparison that keeps the successful case set fixed:', '',
              '| Agents | Intervention | Lost successes | Rescued failures | Makespan increase on common successes |',
              '|---|---|---:|---:|---:|']
    for n in (2,4,6,8):
        for suite,name,label in [('planned','controlled_cloud_all_20','All fields delayed 20 steps'),
                                 ('peer-cutoff','peer_cutoff_20','Peer cutoff 20 steps')]:
            reference = {r['case_id']:r for r in records.get((suite,'perfect'),[]) if r['num_agents']==n}
            treatment = [r for r in records.get((suite,name),[]) if r['num_agents']==n]
            if len(reference)!=50 or len(treatment)!=50:continue
            lost = sum(reference[r['case_id']]['success'] and not r['success'] for r in treatment)
            rescued = sum(not reference[r['case_id']]['success'] and r['success'] for r in treatment)
            common = [r for r in treatment if r['success'] and reference[r['case_id']]['success']]
            delta = float(np.mean([r['steps']-reference[r['case_id']]['steps'] for r in common])) if common else None
            delta_text = f'{delta:+.2f} steps ({len(common)} paired successes)' if delta is not None else 'No common successes'
            lines.append(f'| {n} | {label} | {lost} | {rescued} | {delta_text} |')
    lines += ['',
              '## Continuous latency and handover', '',
              'Positive physical delay means reports generated at an integer robot boundary are unavailable to that boundary’s next action. Thus even subsecond access latency gives one-step-old inputs under this synchronous action model. It is a persistent lag, unlike the occasional controlled reservation windows. Do not interpret the difference as the effect of a 50 ms satellite handshake.', '',
              '| Agents | Perfect SR | Guidance-only physical SR | Intentions-only physical SR | Priorities-only physical SR | All-field physical SR |',
              '|---|---:|---:|---:|---:|---:|']
    for n in (2,4,6,8):
        cells = []
        for suite,name in [('planned','perfect'),('latency-ablation','physical_uav_task'),
                           ('latency-ablation','physical_uav_intentions'),('latency-ablation','physical_uav_priorities'),('planned','physical_uav_latency')]:
            r = lookup.get((suite,name,n));cells.append(f'{100*r["success_rate"]:g}%' if r else 'pending')
        lines.append(f'| {n} | '+' | '.join(cells)+' |')
    lines += ['', 'The recorded latency diagnostic reproduces a cached-guidance oscillation with monotonic timestamps and no satellite traffic. The latency field-isolation plot identifies which input drives aggregate sensitivity. A continuously delayed, fresh-trained controller can therefore be less robust than one exposed to isolated longer delays.', '',
              'Handovers are compared against latency-only controls. Contention is compared against latency-only; combined is compared against contention to isolate the added handover. UAV-local exchange has no satellite hop.', '']
    for suite in ('planned','orbital'):
        for route in ('uav','cloud'):
            reference = {r['case_id']:r for r in records.get((suite,f'physical_{route}_latency'),[])}
            treatment = records.get((suite,f'physical_{route}_handover'),[])
            if len(reference)!=200 or len(treatment)!=200:continue
            changed = sum(any(r[k]!=reference[r['case_id']][k] for k in ('success','steps','collision_attempts')) for r in treatment)
            lines.append(f'- {suite.capitalize()}, {route} route: {changed} of 200 paired cases changed success, episode length or collision attempts when handovers were added to ordinary latency.')
    lines += ['',
              'The inherited physical-contention setting adds a five-step message hold during seeded two-step service windows (one or two windows per agent). This is an assumed resource intervention, not a measured satellite outage. Controlled sweeps replace the additional hold with 1/2/5/10/20 steps. Peer-cutoff sweeps use 1/2/5/10/20-step windows and discard peer messages instead of buffering them.', '',
              'The orbital suite uses archived Mont-Tremblant orbital geometry with simulated attachment and handshake assumptions at the same 1,000-second timeline offset. The short 50-second empirical one-way recordings cannot cover 250-second missions and were not looped or extrapolated.', '',
              '## Outputs and customization', '',
              'Open `results.html` for figures and recorded side-by-side replays. PNG/PDF files, measured summaries and paired-effect CSVs are alongside it.', '',
              'Rebuild with `python scripts/plot_learning_and_inference.py --rolling 50 --dpi 220`. Edit its colors, labels, sizes and plotting functions directly. The script reads recorded JSONL; it never manufactures raw observations.', '',
              'Raw records and full episode traces: `output/inference/baseline-20261006-01`. The audit checks pairing, episode caps, selected input isolation, frozen hashes and identical perfect references. `transport_event_audit.csv` corrects only derived handover statistics for numerical roundoff and orbital window bookkeeping; raw records and outcomes remain intact.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=Path("output/learning-pilots/fast-curriculum-20261005-01"))
    parser.add_argument("--inference-root", type=Path, default=Path("output/inference/baseline-20261006-01"))
    parser.add_argument("--output", type=Path, default=Path("output/results/learning-and-inference-20261006-01"))
    parser.add_argument("--rolling", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.rolling < 1:
        parser.error("--rolling must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    datasets, stages = stage_data(args.training_root)
    export(args.output/"training_stage_summary.csv", stages)
    totals = [{"condition": name, **data["status"]} for name, data in datasets.items()]
    export(args.output/"training_totals.csv", totals)
    training_plots(datasets, stages, args.output, args.rolling, args.dpi)
    inference, records = inference_data(args.inference_root)
    export(args.output/"inference_summary.csv", inference)
    inference_plots(inference, args.output, args.dpi)
    inference_effects(inference, records, args.inference_root, args.training_root, args.output, args.dpi)
    report_page(stages, inference, args.output)
    write_findings(stages, inference, records, args.output)
    (args.output/"plot_manifest.json").write_text(json.dumps({
        "training_root": str(args.training_root.resolve()), "inference_root": str(args.inference_root.resolve()),
        "rolling_recorded_training_episodes": args.rolling, "success_axes_percent": [0, 100],
        "minimum_inference_step_axis_span": MINIMUM_STEP_AXIS_SPAN,
        "common_generated_step_end": min(d["status"]["generated_joint_steps"] for d in datasets.values()),
        "common_received_sample_end": min(d["status"]["delivered_experiences"] for d in datasets.values()),
        "no_extrapolation_or_synthetic_extension": True,
        "inference_checkpoint_status": INFERENCE_NOTE, "stage_rows": len(stages),
        "inference_summary_rows": len(inference), "plot_script": str(Path(__file__).resolve()),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_input_sha256": {name:{filename:hashlib.sha256((args.training_root/name/filename).read_bytes()).hexdigest()
             for filename in ("training.jsonl","validation.jsonl","validation_cases.jsonl","status.json")} for name in CONDITIONS},
        "transport_event_audit_sha256": hashlib.sha256((args.inference_root/"transport_event_audit.csv").read_bytes()).hexdigest()
             if (args.inference_root/"transport_event_audit.csv").exists() else None,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "stage_rows": len(stages),
                      "inference_rows": len(inference), "training_totals": {n: len(d["episodes"]) for n,d in datasets.items()}}))


if __name__ == "__main__":
    main()
