"""Publish only a complete, qualified paired study into tracked paper artifacts."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from panama_uav.records import sha256, write_json
from panama_uav.resource_study import summarize

POLICIES = ("equal", "static", "aoi", "priority")
LABELS = dict(equal="Equal weights", static="Static priority", aoi="AoI only", priority="PANAMA priority", fresh="Fresh reference")
COLORS = dict(equal="#3b6280", static="#a26825", aoi="#6e5891", priority="#187860", fresh="#444444")


def save_figure(fig, directory, name):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(directory/f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def overview(rows, figures):
    metrics = [("success", "Mission success (%)", 100), ("par2", "PAR-2 (steps)", 1),
        ("mean_task_latency_seconds", "Observed task latency (s)", 1),
        ("mean_completed_task_energy_j", "Modeled energy / delivered task (J)", 1),
        ("mean_intentions_age_seconds", "Coordination AoI (s)", 1),
        ("xi_fov", "Mean episode FoV error (%)", 100),
        ("waiting_steps", "Stationary active-agent steps", 1),
        ("sum_path_costs", "Sum of first-arrival costs (steps)", 1)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), constrained_layout=True)
    for ax, (metric, label, scale) in zip(axes.flat, metrics):
        for policy in (*POLICIES, "fresh"):
            group = sorted([r for r in rows if r["policy"] == policy and
                (r["variant"] == "up-1-cpu-1" or r["variant"] == "fresh") and r[metric] is not None],
                key=lambda r:r["num_agents"])
            if not group:
                continue
            y = np.array([r[metric]*scale for r in group])
            intervals = np.array([r[metric+"_ci95"] for r in group]).T*scale
            ax.errorbar([r["num_agents"] for r in group], y,
                yerr=np.maximum(0, np.vstack((y-intervals[0], intervals[1]-y))),
                fmt="o--" if policy == "fresh" else "o-", color=COLORS[policy],
                capsize=2, markersize=3, linewidth=1, label=LABELS[policy])
        ax.set(xlabel="Robots", ylabel=label, xticks=[2,4,6,8]); ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
    save_figure(fig, figures, "mission_comparison")


def grid_effects(rows, figures):
    factors = [.25, 1., 4.]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6), constrained_layout=True)
    for row_index, (metric, label, scale) in enumerate((("xi_fov", "FoV error (%)",100),
                                                       ("mean_task_latency_seconds", "Task latency (s)",1))):
        values = [r[metric]*scale for r in rows if r["num_agents"]==8 and r.get("kind")=="primary" and r[metric] is not None]
        vmax=max(.001,max(values))
        for ax, policy in zip(axes[row_index], POLICIES):
            matrix=np.full((3,3),np.nan)
            for r in rows:
                if r["num_agents"]==8 and r.get("kind")=="primary" and r["policy"]==policy:
                    matrix[factors.index(r["uplink_factor"]),factors.index(r["cpu_factor"])]=r[metric]*scale
            im=ax.imshow(matrix,origin="lower",vmin=0,vmax=vmax,cmap="viridis")
            for y in range(3):
                for x in range(3):
                    ax.text(x,y,f"{matrix[y,x]:.2f}",ha="center",va="center",fontsize=8,
                            color="white" if matrix[y,x]<vmax*.55 else "black")
            ax.set(title=LABELS[policy],xlabel="CPU multiplier",ylabel="Uplink multiplier",
                xticks=range(3),yticks=range(3),xticklabels=["0.25","1","4"],yticklabels=["0.25","1","4"])
        fig.colorbar(im,ax=list(axes[row_index]),label=label,shrink=.85)
    save_figure(fig,figures,"rate_compute_grid")


def paired_effects(rows,figures):
    group=[r for r in rows if r["num_agents"]==8 and r["policy"]=="priority"]
    fig,axes=plt.subplots(1,3,figsize=(13,max(4,len(group)*.19)),constrained_layout=True)
    for ax,(metric,label,scale) in zip(axes,(("success","Success difference (percentage points)",100),
                                           ("mean_task_latency_seconds","Task latency difference (s)",1),
                                           ("xi_fov","FoV error difference (percentage points)",100))):
        for i,r in enumerate(group):
            result=r["paired_equal"].get(metric)
            if result:
                mean=result["delta"]*scale; low,high=np.array(result["ci95"])*scale
                ax.errorbar(mean,i,xerr=np.maximum(0,[[mean-low],[high-mean]]),fmt="o",color=COLORS["priority"],capsize=2,markersize=3)
        ax.axvline(0,color="black",lw=.8,ls="--")
        ax.set(xlabel=label,yticks=range(len(group)),yticklabels=[r["variant"] for r in group] if ax is axes[0] else [])
        ax.invert_yaxis(); ax.grid(axis="x",alpha=.2)
    save_figure(fig,figures,"paired_priority_minus_equal")


def ordering_components(rows,figures):
    group=sorted([r for r in rows if r["variant"]=="up-1-cpu-1" and r["policy"]=="priority"],key=lambda r:r["num_agents"])
    fig,axes=plt.subplots(1,2,figsize=(8,3.3),constrained_layout=True)
    x=np.arange(len(group))
    for shift,(metric,label,color) in enumerate((("xi_fov_uplink","Uplink","#3b6280"),
                                                ("xi_fov_downlink","Delivery","#a26825"),("xi_fov","End to end","#187860"))):
        axes[0].bar(x+(shift-1)*.23,[r[metric]*100 for r in group],width=.23,label=label,color=color)
    axes[0].set(xticks=x,xticklabels=[r["num_agents"] for r in group],xlabel="Robots",ylabel="Mean episode FoV error (%)")
    axes[0].legend(frameon=False,fontsize=8)
    for metric,label in (("mean_drone_cpu_utilization","Drone"),("mean_cloud_cpu_utilization","Cloud")):
        axes[1].plot([r["num_agents"] for r in group],[r[metric] for r in group],"o-",label=label)
    axes[1].set(xlabel="Robots",ylabel="Time-average CPU utilization",ylim=(0,1),xticks=[2,4,6,8]); axes[1].legend(frameon=False)
    for ax in axes: ax.grid(axis="y",alpha=.2)
    save_figure(fig,figures,"ordering_and_utilization")


def latex_results(rows, paper, manifest, totals):
    baseline=[r for r in rows if r["variant"]=="up-1-cpu-1"]
    lines=[r"\subsection{Measured Comparative Results}",
        f"The completed study contains {totals['episodes']:,} qualified-policy episodes and "
        f"{totals['allocation_snapshots']:,} allocation snapshots. All conditions use 50 cases per population. "
        "The primary and secondary results are generated directly from saved records; no missing task latency is replaced by zero.",
        r"\begin{table*}[t]\centering\caption{Nominal uplink/CPU condition: per-episode means over 50 held-out cases. Energy covers delivered tasks only; unfinished counts and full energy bounds accompany the machine-readable tables.}\label{tab:measured_baseline}\scriptsize",
        r"\begin{tabular}{rrlllll}\toprule Robots & Policy & Success (\%) & Task latency (s) & Energy/task (J) & FoV error (\%) & Waiting steps\\\midrule"]
    for count in (2,4,6,8):
        for policy in POLICIES:
            r=next(r for r in baseline if r["num_agents"]==count and r["policy"]==policy)
            lines.append(f"{count} & {LABELS[policy]} & {r['success']*100:.1f} & {r['mean_task_latency_seconds']:.4f} & "
                         rf"{r['mean_completed_task_energy_j']:.5f} & {r['xi_fov']*100:.3f} & {r['waiting_steps']:.1f} \\")
    lines += [r"\bottomrule\end{tabular}\end{table*}",
        r"\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{figures/mission_comparison.pdf}",
        r"\caption{Nominal resources: measured mission, latency, energy, AoI and inconsistency outcomes. Error bars are pointwise 95\% episode-bootstrap intervals. The ideal fresh reference has no support-task latency or energy samples.}\label{fig:mission_comparison}\end{figure*}",
        r"\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{figures/rate_compute_grid.pdf}",
        r"\caption{Eight robots: primary uplink/CPU grid. Both FoV inconsistency and observed task delay are measured outcomes.}\label{fig:grid}\end{figure*}",
        r"\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{figures/paired_priority_minus_equal.pdf}",
        r"\caption{Eight robots: paired PANAMA-priority minus equal-weight differences across the primary and secondary conditions, with pointwise 95\% bootstrap intervals. A lower latency/error difference and a higher success difference are favorable. Intervals crossing zero do not establish a directional advantage.}\label{fig:paired}\end{figure*}",
        r"\begin{figure*}[t]\centering\includegraphics[width=.8\textwidth]{figures/ordering_and_utilization.pdf}",
        r"\caption{Nominal PANAMA-priority condition: UL, delivery and E2E ordering errors are separate quantities; CPU utilization includes idle intervals and unfinished service.}\label{fig:components}\end{figure*}"]
    r=next(r for r in baseline if r["num_agents"]==8 and r["policy"]=="priority")
    parts=[]
    for metric,label,scale in (("success","completion probability",100),("mean_task_latency_seconds","mean task latency",1),("xi_fov","mean episode FoV inconsistency",100)):
        paired=r["paired_equal"][metric]; lo,hi=np.array(paired['ci95'])*scale
        units="percentage points" if scale==100 else "s"
        parts.append(f"The nominal eight-robot priority-minus-equal difference in {label} is {paired['delta']*scale:.4f} {units} "
                     f"(95\\% paired interval [{lo:.4f}, {hi:.4f}]; {paired['pairs']} pairs).")
    lines += parts
    lines += ["These results quantify the complete closed loop rather than only the static optimizer. "
        "Queueing, changing CPU service, delayed self and peer state, discrete route changes and the frozen policy can change mission outcomes independently of snapshot objective gains. "
        "The confidence intervals and sensitivity figures must therefore be considered alongside the mean values; superiority is not assumed."]
    (paper/"generated/results.tex").write_text("\n\n".join(lines)+"\n",encoding="utf-8")


def publish(study,paper,output):
    study,paper,output=Path(study),Path(paper),Path(output)
    status=json.loads((study/"status.json").read_text())
    manifest=json.loads((study/"manifest.json").read_text())
    if status["status"]!="complete" or manifest["diagnostic"]:
        raise ValueError("Only a completed qualified study can populate the paper results")
    bank=json.loads((study/"held_out_cases.json").read_text()); ids={r["case_id"] for r in bank}
    totals=dict(episodes=0,tasks=0,delivered_tasks=0,allocation_snapshots=0)
    for condition in manifest["conditions"]:
        directory=study/condition["name"]
        episodes=[json.loads(s) for s in (directory/"episodes.jsonl").read_text().splitlines()]
        if len(episodes)!=len(ids) or {r["case_id"] for r in episodes}!=ids:
            raise ValueError(f"Incomplete/duplicate case bank: {condition['name']}")
        for r in episodes:
            assert r["policy_model_sha256"]==manifest["model"]["sha256"] and not r["diagnostic"]
            assert r["tasks_generated"]==r["tasks_delivered"]+r["tasks_unfinished"]
            assert 0 <= r["xi_fov"] <= 1
            totals["episodes"]+=1; totals["tasks"]+=r["tasks_generated"]
            totals["delivered_tasks"]+=r["tasks_delivered"]; totals["allocation_snapshots"]+=r["allocation_solves"]
        for filename,expected in (("tasks.jsonl",sum(r["tasks_generated"] for r in episodes)),
                                  ("allocations.jsonl",sum(r["allocation_solves"] for r in episodes))):
            target=directory/filename
            count=sum(1 for _ in target.open(encoding="utf-8")) if target.exists() else 0
            if count!=expected:
                raise ValueError(f"Intermediate record count mismatch: {condition['name']}/{filename}")
    rows=summarize(study)
    plt.rcParams.update({"font.size":8,"axes.titlesize":9,"svg.fonttype":"none","pdf.fonttype":42})
    figures=paper/"figures"; figures.mkdir(exist_ok=True)
    overview(rows,figures); grid_effects(rows,figures); paired_effects(rows,figures); ordering_components(rows,figures)
    latex_results(rows,paper,manifest,totals)
    output.mkdir(parents=True,exist_ok=True)
    for filename in ("summary.json","summary.csv","manifest.json","held_out_cases.json","status.json"):
        shutil.copyfile(study/filename,output/filename)
    write_json(output/"verification.json",dict(status="passed",**totals,conditions=len(manifest["conditions"]),
        checkpoint_sha256=manifest["model"]["sha256"],timeline_sha256=manifest["timeline_sha256"],
        study_path=str(study.resolve()),manifest_sha256=sha256(study/"manifest.json")))
    baseline=[r for r in rows if r["variant"]=="up-1-cpu-1"]
    fields=["num_agents","policy","success","par2","mean_task_latency_seconds","mean_completed_task_energy_j","xi_fov","xi_fov_uplink","xi_fov_downlink","mean_intentions_age_seconds","waiting_steps","sum_path_costs","tasks_unfinished"]
    with (output/"baseline.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(baseline)
    print(json.dumps(totals,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--paper",default=str(ROOT/"paper"))
    parser.add_argument("--output",default=str(ROOT/"research/results"))
    args=parser.parse_args(); publish(args.study,args.paper,args.output)
