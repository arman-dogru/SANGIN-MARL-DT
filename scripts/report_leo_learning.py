"""Editable plotting entry point. Uses recorded episodes and validations only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    if not path.exists():
        return []
    with path.open() as stream:
        result = []
        for line in stream:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                break # the writer can have an incomplete last line
        return result


def rolling(values, window=50):
    a = np.asarray(values,float)
    totals = np.r_[0.,a.cumsum()]
    right = np.arange(1,len(a)+1)
    left = np.maximum(0,right-window)
    return (totals[right]-totals[left])/(right-left)


def report(root):
    root = Path(root)
    output = root/"plots"
    output.mkdir(exist_ok=True)
    arms = [(name,read(root/name/"training.jsonl"),read(root/name/"validation.jsonl"))
            for name in ("clean","combined")]
    colors = {"clean":"#2476b6","combined":"#dc7936"}
    labels = {"clean":"Perfect communication","combined":"Full LEO transport"}
    summary=[]
    for phase in range(4):
        fig, axes = plt.subplots(3,3,figsize=(14,10),layout="constrained")
        for name, rows, vals in arms:
            group=[r for r in rows if r["stage_index"]==phase]
            dots=[r for r in vals if r["stage_index"]==phase and r["complete"]]
            summary.append({"condition":name,"stage":phase+1,"episodes_completed":len(group),
                            "training_success_rate":float(np.mean([r["success"] for r in group])) if group else None,
                            "validations":dots})
            if not group:
                continue
            for col,key in enumerate(("stage_episode","generated_joint_steps","wall_seconds")):
                scale=3600 if key=="wall_seconds" else 1
                x=[r[key]/scale for r in group]
                axes[0,col].plot(x,100*rolling([r["success"] for r in group]),color=colors[name],label=labels[name])
                axes[0,col].scatter([r[key]/scale for r in dots],[100*r["success_rate"] for r in dots],color=colors[name],zorder=4)
                axes[1,col].plot(x,rolling([r["par2"] for r in group]),color=colors[name],label=labels[name])
                successful=[r for r in group if r["success"]]
                if successful:
                    axes[2,col].plot([r[key]/scale for r in successful],rolling([r["steps"] for r in successful]),
                                     color=colors[name],label=labels[name])
        for col,label in enumerate(("Training episodes in stage","Generated environment steps","Wall-clock hours")):
            axes[0,col].set(ylim=(0,100),ylabel="Success (%)")
            axes[1,col].set(xlabel=label,ylabel="Failure-aware PAR-2 (steps)")
            axes[2,col].set(xlabel=label,ylabel="Successful makespan (steps)")
            for row in range(3):
                axes[row,col].grid(alpha=.2)
                if axes[row,col].lines: axes[row,col].legend(fontsize=8)
        fig.suptitle(f"Stage {phase+1} · measured training; dots are 200-case greedy validations")
        fig.savefig(output/f"stage_{phase+1}_learning.png",dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for name,rows,vals in arms:
        for col,key in enumerate(("generated_joint_steps","wall_seconds")):
            dots=[r for r in vals if r["complete"] and r["success_rate"]>=.95]
            scale=3600 if key=="wall_seconds" else 1
            x=[0]+[r[key]/scale for r in dots]
            y=[0]+[r["stage_index"]+1 for r in dots]
            if rows:
                x.append(rows[-1][key]/scale)
                y.append(y[-1])
            axes[col].step(x,y,where="post",label=labels[name],color=colors[name])
            axes[col].set(ylim=(-.05,4.1),yticks=range(5),ylabel="Qualified stages",xlabel="Generated environment steps" if col==0 else "Wall-clock hours")
            axes[col].grid(alpha=.2)
            axes[col].legend()
    fig.savefig(output/"stage_progress.png",dpi=180)
    plt.close(fig)
    (output/"stage_summary.json").write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout="constrained")
    latest={}
    for name,_,_ in arms:
        path=root/name/"learning_steps.jsonl"
        if not path.exists(): continue
        # Two streaming passes bound plot memory independently of run length.
        with path.open() as stream: count=sum(1 for line in stream)
        stride=max(1,int(np.ceil(count/2000)))
        sampled=[]
        with path.open() as stream:
            for index,line in enumerate(stream):
                if index%stride and index!=count-1: continue
                try: sampled.append(json.loads(line))
                except json.JSONDecodeError: break
        if not sampled: continue
        latest[name]=sampled[-1]["generated_joint_steps"]
        x=[r["generated_joint_steps"] for r in sampled]
        ys=[
            [r["delivered_experiences"] for r in sampled],
            [r["learner_updates"] for r in sampled],
            [np.mean(r["experience_delivery_age_seconds"]) if r["experience_delivery_age_seconds"] else np.nan for r in sampled],
            [np.mean(list(r["deployed_model_age_seconds"].values())) for r in sampled],
            [np.mean(list(r["model_version_lag"].values())) for r in sampled],
            [sum(r["queue_bytes"].values())/1e6 for r in sampled],
        ]
        for ax,y in zip(axes.flat,ys): ax.plot(x,y,label=labels[name],color=colors[name])
    for ax,label in zip(axes.flat,("Received experiences","Learner updates","Experience delivery age (s)","Deployed model age (s)","Model version lag","Pending message bytes (MB)")):
        ax.set(xlabel="Generated environment steps",ylabel=label);ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    fig.savefig(output/"learning_transport.png",dpi=180);plt.close(fig)
    (output/"plot_manifest.json").write_text(json.dumps({"common_generated_step_range":min(latest.values()) if len(latest)==2 else None,
        "observed_max_steps":latest,"smoothing":"trailing 50 recorded episodes", "extrapolation":False,
        "recovery":json.loads((root/"recovery.json").read_text()) if (root/"recovery.json").exists() else None},indent=2))
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    print(report(parser.parse_args().root))
