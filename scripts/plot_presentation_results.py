"""Slide-ready figures from measured records, with anonymous stage labels.

Run from the repository root:
    python scripts/plot_presentation_results.py --dpi 220 --rolling 50

Edit COLORS, TRAINING_LABELS, COMPARISONS and styling below to customize.
The source experiment records are read-only. Stage identities and full method
details are kept in README.md and manifest.json, outside the exported figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("clean", "uplink_delayed", "downlink_delayed", "combined")
TRAINING_LABELS = {"clean": "No delay", "uplink_delayed": "Upload delayed",
                   "downlink_delayed": "Deployment delayed", "combined": "Both delayed"}
COLORS = ("#2671A8", "#D98727", "#8255A1", "#26957D", "#BB5567", "#637487")
COUNTS = (2, 4, 6, 8)
STAGES = [f"Stage {i+1}" for i in range(4)]
COMPARISONS = [
    ("inference_mission_delay_5", "Mission information: no delay vs 5-step delay", "planned",
     [("perfect", "No delay"), ("controlled_cloud_all_5", "5-step delay")]),
    ("inference_mission_delay_20", "Mission information: no delay vs 20-step delay", "planned",
     [("perfect", "No delay"), ("controlled_cloud_all_20", "20-step delay")]),
    ("inference_mission_delay_sweep", "Mission information: controlled delay comparison", "planned",
     [("perfect", "No delay")] + [(f"controlled_cloud_all_{d}", f"{d}-step delay") for d in (1, 2, 5, 10, 20)]),
    ("inference_stale_fields_20", "Mission information: which delayed input matters?", "planned",
     [("perfect", "No delay"), ("controlled_cloud_task_20", "A* guidance delayed"),
      ("controlled_cloud_intentions_20", "Intentions delayed"),
      ("controlled_cloud_priorities_20", "Priorities delayed"),
      ("controlled_cloud_all_20", "All three delayed")]),
    ("inference_peer_cutoff_20", "Peer information: connected vs 20-step cutoff", "peer-cutoff",
     [("perfect", "Connected"), ("peer_cutoff_20", "20-step cutoff")]),
    ("inference_orbital_delay", "Cloud mission information: zero delay vs orbital transport", "orbital",
     [("perfect", "No delay"), ("physical_cloud_latency", "Orbital latency"),
      ("physical_cloud_handover", "Orbital latency + handover")]),
    ("inference_handover_increment", "Cloud transport: additional handover effect", "orbital",
     [("physical_cloud_latency", "Latency only"), ("physical_cloud_handover", "Latency + handover")]),
    ("inference_orbital_contention", "Cloud transport: contention and handover", "orbital",
     [("perfect", "No delay"), ("physical_cloud_latency", "Latency only"),
      ("physical_cloud_contention", "Latency + contention"),
      ("physical_cloud_combined", "Latency + contention + handover")]),
]


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smooth(values, window):
    values = np.asarray(values, dtype=float)
    total = np.r_[0., values.cumsum()]
    index = np.arange(len(values))
    start = np.maximum(0, index-window+1)
    return (total[index+1]-total[start])/(index-start+1)


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5EAF0", linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9BA7B3")
    ax.tick_params(length=0, pad=9)


def save(fig, out, name, dpi, files):
    # Inspect visible text before export: no agent counts or map identifiers.
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    forbidden = ("agents", "agent count", "32x32", "32 × 32", ".map", "random-32", "64x64")
    assert not any(term in text.lower() for term in forbidden for text in texts), name
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out/f"{name}.{ext}", dpi=dpi, facecolor="white")
    files.append({"name": name, "visible_text": [t for t in texts if t]})
    plt.close(fig)


def qualification(data):
    return [next((r for r in data["validation"] if r["complete"] and
                  r["stage_index"] == stage and r["success_rate"] >= .95), None)
            for stage in range(4)]


def plot_progress(training, out, dpi, files):
    common = min(d["status"]["generated_joint_steps"] for d in training.values())
    for mode in ("combined", "environment_steps", "wallclock"):
        fig, axes = plt.subplots(1, 2 if mode == "combined" else 1,
                                 figsize=(16, 8.6), squeeze=False)
        keys = ("steps", "wall") if mode == "combined" else (("steps",) if mode == "environment_steps" else ("wall",))
        for ax, key in zip(axes.flat, keys):
            for i, (name, data) in enumerate(training.items()):
                qualified = [r for r in qualification(data) if r is not None]
                coordinates = [r["generated_joint_steps"] if key == "steps" else r["wall_seconds"]/3600 for r in qualified]
                bound = common if key == "steps" else data["status"]["wall_seconds"]/3600
                points = [(x, j+1) for j, x in enumerate(coordinates) if x <= bound]
                xx = [0]+[x for x, _ in points]+[bound]
                yy = [0]+[y for _, y in points]+[points[-1][1] if points else 0]
                ax.step(xx, yy, where="post", color=COLORS[i], label=TRAINING_LABELS[name],
                        linewidth=2.8, linestyle=("-", "--", "-.", ":")[i], alpha=.95)
                ax.scatter([x for x, _ in points], [y for _, y in points],
                           color=COLORS[i], marker=("o", "s", "^", "D")[i], s=70, zorder=4,
                           edgecolor="white", linewidth=.8)
            ax.set_yticks(range(5), ["Start"]+STAGES)
            ax.set_ylim(-.08, 4.3)
            ax.set_ylabel("Highest stage qualified")
            ax.set_xlabel("Generated environment steps" if key == "steps" else "Elapsed wall-clock time (hours)")
            ax.set_xlim(0, common if key == "steps" else 3)
            if key == "steps":
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
            style(ax)
        fig.suptitle("Training: curriculum progress", x=.075, ha="left", y=.965, fontsize=25, weight="bold")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.065, .9), ncol=4, frameon=False, fontsize=13)
        fig.subplots_adjust(left=.1, right=.975, bottom=.12, top=.77, wspace=.32)
        save(fig, out, "training_stage_progress"+("" if mode == "combined" else "_"+mode), dpi, files)


def plot_training_success(training, out, dpi, window, files):
    for mode in ("environment_steps", "wallclock"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        for stage, ax in enumerate(axes.flat):
            for i, (name, data) in enumerate(training.items()):
                episodes = [r for r in data["episodes"] if r["stage_index"] == stage]
                validations = [r for r in data["validation"] if r["stage_index"] == stage and r["complete"]]
                if not episodes:
                    continue
                key, scale = ("generated_joint_steps", 1) if mode == "environment_steps" else ("wall_seconds", 3600)
                offset = (episodes[0][key]-episodes[0]["steps"]) if mode == "environment_steps" else (
                    max([r["wall_seconds"] for r in data["validation"] if r["stage_index"] < stage and r["complete"]], default=0))
                x = [(r[key]-offset)/scale for r in episodes]
                ax.plot(x, 100*smooth([r["success"] for r in episodes], window),
                        color=COLORS[i], linewidth=1.7, alpha=.9, label=TRAINING_LABELS[name])
                ax.scatter([(r[key]-offset)/scale for r in validations],
                           [100*r["success_rate"] for r in validations], s=37,
                           color=COLORS[i], edgecolors="white", linewidth=.6, zorder=5)
            ax.axhline(95, color="#4E5965", linestyle=":", linewidth=1)
            ax.set(title=STAGES[stage], ylabel="Success (%)", ylim=(0, 100),
                   xlabel="Environment steps within stage" if mode == "environment_steps" else "Wall-clock time within stage (hours)")
            ax.set_xlim(left=0)
            if mode == "environment_steps":
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
            style(ax)
        fig.suptitle("Training success and validation", x=.075, ha="left", y=.976, fontsize=25, weight="bold")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.065, .925), ncol=4, frameon=False, fontsize=13)
        fig.text(.075, .024, f"Lines: {window}-episode training mean    •    Dots: greedy validation    •    Dotted line: 95% target",
                 fontsize=11, color="#56616D")
        fig.subplots_adjust(left=.075, right=.97, bottom=.115, top=.815, hspace=.55, wspace=.2)
        save(fig, out, "training_success_"+mode, dpi, files)


def interval(values, rng, repeats):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan, np.nan
    means = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return float(values.mean()), *np.percentile(means, (2.5, 97.5)).tolist()


def wilson(success, total):
    p = success/total
    z = 1.959963984540054
    d = 1+z*z/total
    center = (p+z*z/(2*total))/d
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return p*100, max(0, center-half)*100, min(1, center+half)*100


def aggregate(records, suite, name, rng, repeats):
    rows = []
    for stage, count in enumerate(COUNTS):
        group = sorted([r for r in records if r["num_agents"] == count], key=lambda r: r["case_id"])
        assert len(group) == 50 and len({r["case_id"] for r in group}) == 50
        successful = [r["steps"] for r in group if r["success"]]
        row = {"suite": suite, "condition": name, "stage": STAGES[stage],
               "num_agents": count, "cases": len(group), "successes": len(successful)}
        for metric_index, (key, values) in enumerate((("mean_environment_steps", [r["steps"] for r in group]),
                            ("successful_makespan", successful), ("par2", [r["par2"] for r in group]))):
            row[key], row[key+"_ci_low"], row[key+"_ci_high"] = interval(
                values, np.random.default_rng(42+stage*10+metric_index), repeats)
        row["success_rate"], row["success_rate_ci_low"], row["success_rate_ci_high"] = wilson(len(successful), len(group))
        rows.append(row)
    return rows


def plot_bars(name, title, suite, conditions, summary, out, dpi, files):
    fig, axes = plt.subplots(1, 3, figsize=(16, 8.6))
    keys = ("mean_environment_steps", "success_rate", "successful_makespan")
    labels = ("Average episode length", "Success rate", "Successful makespan")
    x = np.arange(4)
    width = .76/len(conditions)
    for ax, key, label in zip(axes, keys, labels):
        upper = 0.
        for i, (condition, legend) in enumerate(conditions):
            group = [summary[(suite, condition, stage)] for stage in STAGES]
            values = np.array([r[key] for r in group])
            lo = np.array([r[key+"_ci_low"] for r in group])
            hi = np.array([r[key+"_ci_high"] for r in group])
            positions = x+(i-(len(conditions)-1)/2)*width
            ax.bar(positions, values, width=width*.9, color=COLORS[i], label=legend,
                   yerr=np.maximum(0, np.vstack([values-lo, hi-values])),
                   error_kw={"ecolor": "#485360", "elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
                   zorder=3)
            for position, value, ceiling in zip(positions, values, hi):
                text = f"{value:.0f}" if key == "success_rate" else f"{value:.1f}"
                if key == "success_rate":
                    ax.text(position, max(3, value-9), text, ha="center", va="center", fontsize=10.5 if len(conditions)<5 else 8.5,
                            color="white", weight="bold", rotation=90 if len(conditions)>=4 else 0)
                else:
                    ax.annotate(text, (position, ceiling), xytext=(0, 5), textcoords="offset points",
                                ha="center", va="bottom", fontsize=10.5 if len(conditions)<5 else 8.5,
                                rotation=90 if len(conditions)>=3 else 0)
            upper = max(upper, float(hi.max()))
        ax.set_title(label, fontsize=16, pad=18)
        ax.set_xticks(x, STAGES)
        ax.set_ylabel("Success (%)" if key == "success_rate" else "Environment steps")
        ax.set_ylim(0, 100 if key == "success_rate" else max(5, upper*1.22))
        style(ax)
    fig.suptitle(title, x=.06, ha="left", y=.968, fontsize=23, weight="bold")
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper left", bbox_to_anchor=(.052, .902),
               ncol=min(3, len(conditions)) if len(conditions)>4 else len(conditions),
               frameon=False, fontsize=12)
    fig.text(.06, .035, "Episode length: all cases, including failures    •    Makespan: successful cases    •    Whiskers: 95% intervals",
             fontsize=11, color="#56616D")
    fig.subplots_adjust(left=.06, right=.98, bottom=.16, top=.72 if len(conditions)>4 else .77, wspace=.33)
    save(fig, out, name, dpi, files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=ROOT/"output/learning-pilots/fast-curriculum-20261005-01")
    parser.add_argument("--inference-root", type=Path, default=ROOT/"output/inference/baseline-20261006-01")
    parser.add_argument("--output", type=Path, default=ROOT/"output/results/presentation-20261007-01")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--rolling", type=int, default=50)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 13,
                         "axes.titlesize": 15, "svg.fonttype": "none", "pdf.fonttype": 42})
    files, sources, training = [], [], {}
    for name in CONDITIONS:
        folder = args.training_root/name
        training[name] = {"episodes": read(folder/"training.jsonl"),
                          "validation": read(folder/"validation.jsonl"),
                          "status": json.loads((folder/"status.json").read_text(encoding="utf-8"))}
        for filename in ("training.jsonl", "validation.jsonl", "status.json"):
            path = folder/filename
            sources.append({"path": str(path.resolve()), "sha256": sha(path)})
    plot_progress(training, args.output, args.dpi, files)
    plot_training_success(training, args.output, args.dpi, args.rolling, files)
    rng = np.random.default_rng(42)
    summary, rows, pairing = {}, [], []
    for suite in dict.fromkeys(r[2] for r in COMPARISONS):
        conditions = list(dict.fromkeys(condition for comparison in COMPARISONS if comparison[2] == suite for condition, _ in comparison[3]))
        reference = read(args.inference_root/suite/"perfect/episodes.jsonl")
        reference_cases = {r["case_id"]: r for r in reference}
        for condition in conditions:
            path = args.inference_root/suite/condition/"episodes.jsonl"
            records = read(path)
            assert len(records) == 200 and len({r["case_id"] for r in records}) == 200
            assert all(1 <= r["steps"] <= 250 for r in records)
            for r in records:
                base = reference_cases[r["case_id"]]
                assert all(r[k] == base[k] for k in ("seed", "starts", "goals", "num_agents"))
            pairing.append({"suite": suite, "condition": condition, "paired_cases": len(records)})
            sources.append({"path": str(path.resolve()), "sha256": sha(path)})
            for row in aggregate(records, suite, condition, rng, args.bootstrap):
                summary[(suite, condition, row["stage"])] = row
                rows.append(row)
    for comparison in COMPARISONS:
        plot_bars(*comparison, summary, args.output, args.dpi, files)
    export(args.output/"inference_bar_data.csv", rows)
    progress_rows = []
    for name, data in training.items():
        for stage, check in enumerate(qualification(data)):
            progress_rows.append({"condition": name, "stage": STAGES[stage], "qualified": bool(check),
                                  "generated_environment_steps": check["generated_joint_steps"] if check else None,
                                  "wall_seconds": check["wall_seconds"] if check else None,
                                  "training_episodes_in_stage": sum(r["stage_index"] == stage for r in data["episodes"]),
                                  "validation_success_rate": check["success_rate"] if check else None})
    export(args.output/"training_stage_progress.csv", progress_rows)
    manifest = {"source_script": str(Path(__file__).resolve()), "source_script_sha256": sha(Path(__file__)),
                "stage_mapping": dict(zip(STAGES, COUNTS)), "map": "random-32-32-10.map",
                "inference_cases_per_stage": 50, "episode_cap_steps": 250,
                "training_seed": 42, "rolling_episodes": args.rolling, "bootstrap_resamples": args.bootstrap,
                "confidence_intervals": "Wilson for success; percentile bootstrap of cases for lengths. Bar intervals are marginal, not paired difference intervals.",
                "common_generated_steps": min(d["status"]["generated_joint_steps"] for d in training.values()),
                "baseline_scope": "Exploratory final Clean checkpoint; Stage 4 failed curriculum qualification.",
                "sources": sources, "pairing_checks": pairing, "figures": files}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    items = "\n".join(f'<article><h2>{html.escape(r["name"].replace("_", " "))}</h2><img src="{r["name"]}.png"><p><a href="{r["name"]}.png">PNG</a> · <a href="{r["name"]}.pdf">PDF</a> · <a href="{r["name"]}.svg">SVG</a></p></article>' for r in files)
    page = '<!doctype html><html><meta charset="utf-8"><title>Training and inference figures</title><style>body{font:16px system-ui;background:#eef2f6;color:#172334;max-width:1450px;margin:40px auto;padding:0 20px}article{background:white;padding:20px;margin:25px 0;border-radius:12px}img{width:100%;height:auto}a{color:#2671a8}</style><h1>Training and inference figures</h1>'+items+'<p>Measured exploratory results. Training did not qualify the final stage; inference uses its frozen final checkpoint. Protocol, source hashes and stage identities are retained separately in README.md and manifest.json.</p></html>'
    (args.output/"index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "figures": len(files), "summary_rows": len(rows), "paired_conditions": len(pairing)}))


if __name__ == "__main__":
    main()
