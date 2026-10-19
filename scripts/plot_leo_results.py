"""Plot the completed measured Clean/LEO learning comparison.

Edit STYLE, WINDOW, COLORS and plotting functions to customize the figures.
Inputs are training.jsonl, validation.jsonl, validation_cases.jsonl,
learning_steps.jsonl, transport.jsonl and status.json under each arm.
No results are extrapolated. The source experiment is never modified.
"""
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_left
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

ARMS = ("clean", "combined")
COLORS = {"clean": "#2563b5", "combined": "#d96b20"}
LABELS = {"clean": "Perfect communication", "combined": "LEO communication"}
WINDOW = 100
TELEMETRY_WINDOW = 50
STYLE = {"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
         "axes.spines.right": False, "axes.titleweight": "bold", "axes.labelcolor": "#334155",
         "text.color": "#172b45", "axes.edgecolor": "#94a3b8", "legend.frameon": False,
         "savefig.facecolor": "white", "figure.facecolor": "white"}


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def rolling(values, window=WINDOW):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    total = np.r_[0., np.where(valid, values, 0.).cumsum()]
    count = np.r_[0, valid.cumsum()]
    right = np.arange(1, len(values)+1)
    left = np.maximum(0, right-window)
    denominator = count[right]-count[left]
    return np.divide(total[right]-total[left], denominator,
                     out=np.full(len(values), np.nan), where=denominator > 0)


def csv_file(path, data):
    if not data:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def compact(value):
    if value >= 100000:
        return f"{value/1000:.0f}k"
    return f"{value/1000:.1f}k" if value >= 10000 else f"{value:,.0f}"


def save(fig, output, name, figures):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output/f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)


def prepare(root):
    data, summary, audit = {}, [], {}
    for arm in ARMS:
        folder = root/arm
        status = json.loads((folder/"status.json").read_text())
        training = rows(folder/"training.jsonl")
        validation = rows(folder/"validation.jsonl")
        cases = rows(folder/"validation_cases.jsonl")
        assert status["qualified"] and len(status["qualification"]) == 4
        assert len(training) == status["episodes"]
        assert [r["global_episode"] for r in training] == list(range(1, len(training)+1))
        assert sum(r["steps"] for r in training) == status["generated_joint_steps"]
        assert all(0 < r["steps"] <= 250 for r in training+cases)
        groups = {}
        for case in cases:
            key = (case["stage_index"], case["stage_episode"], case["generated_joint_steps"])
            groups.setdefault(key, []).append(case)
        for val in validation:
            group = groups[(val["stage_index"], val["stage_episode"], val["generated_joint_steps"])]
            assert val["complete"] and val["episodes"] == len(group) == 200
            assert sorted(r["case"] for r in group) == list(range(200))
            assert np.isclose(np.mean([r["success"] for r in group]), val["success_rate"])
            assert np.isclose(np.mean([r["par2"] for r in group]), val["par2"])
            assert all(r["par2"] == (r["steps"] if r["success"] else 500) for r in group)
        prior_steps = prior_wall = 0
        for stage, q in enumerate(status["qualification"]):
            assert q["stage_index"] == stage and q["stage_episode"] >= 500 and q["success_rate"] >= .95
            group = [r for r in training if r["stage_index"] == stage]
            assert len(group) == q["stage_episode"]
            group_cases = groups[(stage, q["stage_episode"], q["generated_joint_steps"])]
            successful = [r for r in group_cases if r["success"]]
            summary.append({"condition": arm, "stage": stage+1, "training_episodes": len(group),
                "environment_steps": q["generated_joint_steps"]-prior_steps,
                "recorded_hours": (q["wall_seconds"]-prior_wall)/3600,
                "cumulative_environment_steps": q["generated_joint_steps"],
                "cumulative_recorded_hours": q["wall_seconds"]/3600,
                "training_success_pct": 100*np.mean([r["success"] for r in group]),
                "qualifying_validation_success_pct": 100*q["success_rate"],
                "qualifying_validation_cases": 200,
                "validation_successful_makespan": np.mean([r["steps"] for r in successful]),
                "validation_mean_episode_steps": np.mean([r["steps"] for r in group_cases]),
                "validation_par2": q["par2"],
                "validation_completion_fraction": np.mean([r["completion_fraction"] for r in group_cases]),
                "validation_collision_attempts": np.mean([r["collision_attempts"] for r in group_cases]),
                "validation_uav_travel_m": np.mean([r["uav_travel_m"] for r in group_cases])})
            prior_steps, prior_wall = q["generated_joint_steps"], q["wall_seconds"]
        data[arm] = dict(status=status, training=training, validation=validation, groups=groups)
        audit[arm] = {"training_episodes": len(training), "training_steps": sum(r["steps"] for r in training),
                      "complete_validation_batches": len(validation), "validation_cases": len(cases),
                      "all_four_phases_qualified": True, "max_episode_steps": max(r["steps"] for r in training+cases)}
    for stage in range(4):
        seeds = []
        for arm in ARMS:
            q = data[arm]["status"]["qualification"][stage]
            group = data[arm]["groups"][(stage, q["stage_episode"], q["generated_joint_steps"])]
            seeds.append([r["seed"] for r in sorted(group, key=lambda r: r["case"])])
        assert seeds[0] == seeds[1], "Validation cases must be paired"
    manifests = [json.loads((root/a/"manifest.json").read_text()) for a in ARMS]
    for key in ("initial_weights_sha256", "map_sha256", "config"):
        assert manifests[0][key] == manifests[1][key], key
    audit.update(paired_validation_seeds=True, matching_initial_weights_and_configuration=True)
    return data, summary, audit


def success_curves(data, output, figures):
    for key, scale, axis_label, name in (
            ("stage_episode", 1, "Training episodes in stage", "success_episodes"),
            ("generated_joint_steps", 1, "Environment steps in stage", "success_environment_steps"),
            ("wall_seconds", 3600, "Recorded run hours in stage", "success_training_time")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
        for stage, ax in enumerate(axes.flat):
            for arm in ARMS:
                training = [r for r in data[arm]["training"] if r["stage_index"] == stage]
                validation = [r for r in data[arm]["validation"] if r["stage_index"] == stage]
                previous = data[arm]["status"]["qualification"][stage-1] if stage else {}
                offset = previous.get(key, 0) if key != "stage_episode" else 0
                ax.plot([(r[key]-offset)/scale for r in training],
                        100*rolling([r["success"] for r in training]), color=COLORS[arm], lw=1.7, alpha=.85)
                ax.scatter([(r[key]-offset)/scale for r in validation],
                           [100*r["success_rate"] for r in validation], color=COLORS[arm],
                           edgecolors="white", linewidths=.8, s=42, zorder=5)
            ax.axhline(95, color="#64748b", linestyle="--", lw=1, alpha=.7)
            ax.set(title=f"Stage {stage+1}", ylim=(0, 100), xlabel=axis_label, ylabel="Success (%)")
            ax.grid(axis="y", alpha=.18)
            if key == "generated_joint_steps":
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: compact(v)))
        handles = [Line2D([], [], color=COLORS[a], lw=2, label=LABELS[a]) for a in ARMS]
        handles += [Line2D([], [], color="#64748b", marker="o", linestyle="none", label="200-case validation")]
        fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=10)
        save(fig, output, name, figures)


def grouped_bars(ax, summary, field, ylabel, formatter):
    x = np.arange(4)
    for i, arm in enumerate(ARMS):
        vals = [r[field] for r in summary if r["condition"] == arm]
        bars = ax.bar(x+(i-.5)*.36, vals, .34, color=COLORS[arm], label=LABELS[arm])
        ax.bar_label(bars, labels=[formatter(v) for v in vals], padding=4, fontsize=9)
    ax.set(xticks=x, xticklabels=[f"Stage {i+1}" for i in x], ylabel=ylabel)
    top = 100 if field.endswith("success_pct") else max(r[field] for r in summary)*1.23
    ax.set_ylim(0, top)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)


def task_curves(data, output, figures):
    for field, label, name in (("steps", "Successful makespan (steps)", "makespan_learning"),
                                ("par2", "Failure-aware PAR-2 (steps)", "failure_aware_learning"),
                                ("uav_travel_m", "UAV travel per episode (m)", "uav_travel_learning")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
        for stage, ax in enumerate(axes.flat):
            for arm in ARMS:
                training = [r for r in data[arm]["training"] if r["stage_index"] == stage
                            and (field != "steps" or r["success"])]
                validation = [r for r in data[arm]["validation"] if r["stage_index"] == stage]
                ax.plot([r["stage_episode"] for r in training], rolling([r[field] for r in training]),
                        color=COLORS[arm], lw=1.7, alpha=.85)
                points = []
                for val in validation:
                    group = data[arm]["groups"][(stage, val["stage_episode"], val["generated_joint_steps"])]
                    values = [r[field] for r in group if field != "steps" or r["success"]]
                    if values:
                        points.append((val["stage_episode"], np.mean(values)))
                ax.scatter([p[0] for p in points], [p[1] for p in points], s=42,
                           color=COLORS[arm], edgecolors="white", linewidths=.8, zorder=5)
            ax.set(title=f"Stage {stage+1}", xlabel="Training episodes in stage", ylabel=label)
            ax.grid(axis="y", alpha=.18)
            ax.margins(y=.1)
        handles = [Line2D([], [], color=COLORS[a], lw=2, label=LABELS[a]) for a in ARMS]
        handles += [Line2D([], [], color="#64748b", marker="o", linestyle="none", label="200-case validation")]
        fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=10)
        save(fig, output, name, figures)


def cost_and_validation(data, summary, output, figures):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    for ax, field, label, fmt in zip(axes,
            ("training_episodes", "environment_steps", "recorded_hours"),
            ("Training episodes", "Environment steps", "Recorded run hours"),
            (lambda v: f"{v:,.0f}", compact, lambda v: f"{v:.2f}")):
        grouped_bars(ax, summary, field, label, fmt)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, output, "training_cost_by_stage", figures)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    for ax, field, label, fmt in zip(axes,
            ("qualifying_validation_success_pct", "validation_successful_makespan", "validation_par2"),
            ("Validation success (%)", "Successful makespan (steps)", "Failure-aware PAR-2 (steps)"),
            (lambda v: f"{v:g}", lambda v: f"{v:.1f}", lambda v: f"{v:.1f}")):
        grouped_bars(ax, summary, field, label, fmt)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, output, "qualifying_validation_outcomes", figures)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    for arm in ARMS:
        q = data[arm]["status"]["qualification"]
        for ax, key, scale, label in zip(axes, ("generated_joint_steps", "wall_seconds"), (1, 3600),
                                       ("Cumulative environment steps", "Recorded run hours")):
            ax.step([0]+[r[key]/scale for r in q], [0, 1, 2, 3, 4], where="post", color=COLORS[arm],
                    lw=2.3, label=LABELS[arm])
            ax.scatter([r[key]/scale for r in q], [1, 2, 3, 4], color=COLORS[arm], s=35, zorder=4)
            ax.set(xlabel=label, ylabel="Qualified stages", ylim=(-.05, 4.2), yticks=range(5))
            ax.grid(alpha=.18)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, p: compact(v)))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, output, "curriculum_progress", figures)


def telemetry(root, data, output, figures, common):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
    audit = {}
    for arm in ARMS:
        path = root/arm/"learning_steps.jsonl"
        expected = data[arm]["status"]["generated_joint_steps"]
        stride = max(1, int(np.ceil(expected/3000)))
        # Fixed-stride sampling aliases the 16-step model publication cycle.
        # Seeded stratification selects actual records across all cycle phases.
        rng = np.random.default_rng(42)
        chosen = {0, expected-1, common-1}
        chosen.update(int(rng.integers(left, min(expected, left+stride)))
                      for left in range(0, expected, stride))
        sampled, digest = [], hashlib.sha256()
        with path.open("rb") as stream:
            for index, line in enumerate(stream):
                digest.update(line)
                if index in chosen:
                    r = json.loads(line)
                    assert r["generated_joint_steps"] == index+1
                    sampled.append(r)
        assert index+1 == expected
        audit[arm] = {"sha256": digest.hexdigest(), "rows": expected, "sampled_rows": len(sampled),
                      "sampling": "Seed-42 stratified observations", "stratum_width": stride,
                      "plotted_smoothing_observations": TELEMETRY_WINDOW}
        csv_file(output/f"{arm}_telemetry_sample.csv", [{
            "environment_steps": r["generated_joint_steps"], "stage": r["stage_index"]+1,
            "experience_age_ms": np.mean(r["experience_delivery_age_seconds"])*1000 if r["experience_delivery_age_seconds"] else None,
            "deployed_model_age_s": np.mean(list(r["deployed_model_age_seconds"].values())),
            "model_version_lag": np.mean(list(r["model_version_lag"].values())),
            "pending_bytes": sum(r["queue_bytes"].values())} for r in sampled])
        series = ([np.mean(r["experience_delivery_age_seconds"])*1000 if r["experience_delivery_age_seconds"] else np.nan for r in sampled],
                  [np.mean(list(r["deployed_model_age_seconds"].values())) for r in sampled],
                  [np.mean(list(r["model_version_lag"].values())) for r in sampled],
                  [sum(r["queue_bytes"].values())/1e6 for r in sampled])
        for ax, y in zip(axes.flat, series):
            ax.plot([r["generated_joint_steps"] for r in sampled], rolling(y, TELEMETRY_WINDOW),
                    color=COLORS[arm], lw=1.7, label=LABELS[arm])
    for ax, label in zip(axes.flat, ("Experience delivery age (ms)", "Age of deployed model (s)", "Model version lag", "Reported pending traffic (MB)")):
        ax.set(xlabel="Cumulative environment steps", ylabel=label)
        ax.grid(alpha=.18)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: compact(v)))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, output, "learning_transport_telemetry", figures)
    return audit


def delivery_delays(root, data, output, figures, cached=None):
    summaries, audit = [], {}
    if cached:
        assert cached["source_root"] == str(root)
        # Customizing figures can reuse exact prior aggregates only if the
        # complete source files still have their audited content hashes.
        for arm in ARMS:
            with (root/arm/"transport.jsonl").open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            assert digest == cached["transport_log_audit"][arm]["sha256"]
        summaries, audit = cached["message_delivery_summary"], cached["transport_log_audit"]
    for arm in (() if cached else ARMS):
        samples = {(stage, kind): array("d") for stage in range(4) for kind in ("experience", "model")}
        edges = [q["generated_joint_steps"] for q in data[arm]["status"]["qualification"]]
        digest = hashlib.sha256()
        path = root/arm/"transport.jsonl"
        print(f"Reading complete-message delivery records: {arm}", flush=True)
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if b'"event":"delivered"' not in line:
                    continue
                r = json.loads(line)
                if r["kind"] not in ("experience", "model"):
                    continue
                phase = min(3, bisect_left(edges, r["source_time"]))
                samples[phase, r["kind"]].append(r["age_seconds"])
        count = 0
        for (stage, kind), values in samples.items():
            a = np.frombuffer(values, dtype=np.float64)
            assert len(a) and np.all(a >= 0)
            summaries.append({"condition": arm, "stage": stage+1, "message_kind": kind,
                "delivered_messages": len(a), "mean_seconds": float(a.mean()),
                "p50_seconds": float(np.quantile(a, .5)), "p95_seconds": float(np.quantile(a, .95)),
                "max_seconds": float(a.max())})
            if kind == "experience": count += len(a)
        assert count == data[arm]["status"]["delivered_experiences"]
        audit[arm] = {"sha256": digest.hexdigest(), "delivered_experiences_audited": count}
    csv_file(output/"message_delivery_summary.csv", summaries)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for ax, kind, scale, label in zip(axes, ("experience", "model"), (1000, 1),
                                    ("Experience upload delivery (ms)", "Complete model delivery to robot (s)")):
        x = np.arange(4)
        for i, arm in enumerate(ARMS):
            group = [r for r in summaries if r["condition"] == arm and r["message_kind"] == kind]
            means = np.array([r["mean_seconds"]*scale for r in group])
            p95 = np.array([r["p95_seconds"]*scale for r in group])
            bars = ax.bar(x+(i-.5)*.36, means, .34, color=COLORS[arm], label=LABELS[arm])
            ax.scatter(x+(i-.5)*.36, p95, color=COLORS[arm], marker="_", s=180, linewidths=2, zorder=4)
            ax.bar_label(bars, labels=[f"{v:.1f}" if v else "0" for v in means], padding=4, fontsize=9)
        ax.set(xticks=x, xticklabels=[f"Stage {i+1}" for i in x], ylabel=label, ylim=(0, max(p95.max(), means.max())*1.25))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#64748b", marker="_", linestyle="none", markersize=12))
    fig.legend(handles, labels+["95th percentile"], loc="outside upper center", ncol=3)
    save(fig, output, "complete_message_delays", figures)
    return summaries, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-delivery-summary", action="store_true",
                        help="Reuse prior exact delivery summaries after verifying source hashes")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root/"results"
    output.mkdir(parents=True, exist_ok=True)
    cached = json.loads((output/"results.json").read_text()) if args.reuse_delivery_summary else None
    plt.rcParams.update(STYLE)
    data, summary, audit = prepare(root)
    print("Episode/validation audit passed; rendering learning curves", flush=True)
    csv_file(output/"stage_results.csv", summary)
    totals = [{"condition": a, **{k: data[a]["status"][k] for k in
        ("episodes", "generated_joint_steps", "wall_seconds", "delivered_experiences", "learner_updates", "pending_messages")}}
        for a in ARMS]
    csv_file(output/"run_totals.csv", totals)
    figures = []
    success_curves(data, output, figures)
    cost_and_validation(data, summary, output, figures)
    task_curves(data, output, figures)
    common = min(d["status"]["generated_joint_steps"] for d in data.values())
    step_audit = telemetry(root, data, output, figures, common)
    deliveries, transport_audit = delivery_delays(root, data, output, figures, cached)
    metadata = {"source_root": str(root), "seed": 42, "replicated_seeds": 1,
        "source_files_per_arm": ["training.jsonl", "validation.jsonl", "validation_cases.jsonl",
                                  "learning_steps.jsonl", "transport.jsonl", "status.json"],
        "smoothing": f"Trailing {WINDOW} observed training episodes", "synthetic_results": False,
        "extrapolation": False, "success_axes": [0, 100], "common_generated_step_range": [0, common],
        "stage_labels": "Stage 1–4; no map names or agent counts in figures",
        "audit": audit, "step_log_audit": step_audit, "transport_log_audit": transport_audit,
        "figures": figures, "stage_results": summary, "totals": totals,
        "message_delivery_summary": deliveries,
        "interpretation": ["One matched seed; no across-seed uncertainty is established.",
            "Validation always uses fresh mission information and the current learner greedily.",
            "Qualifying results are selected curriculum checkpoints, not independent mission-inference tests.",
            "Recorded run time includes computation, transport simulation and validation; infrastructure hangs and recovery downtime are excluded.",
            "Deployed-model age includes snapshot publication interval; it is distinct from complete-message delivery latency.",
            "Delivery latency statistics use every completed robot-bound model message and every completed experience message, including old model messages if delivered.",
            "Stage assignment for delivery statistics uses message generation time.",
            "The final LEO run retains pending messages at qualification; they are not counted as delivered."],
        "recovery": json.loads((root/"recovery.json").read_text())}
    (output/"results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    captions = {"success_episodes": "Training success (rolling 100 episodes) with 200-case greedy validation dots.",
        "success_environment_steps": "Success against environment steps within each stage.",
        "success_training_time": "Success against recorded run time within each stage.",
        "training_cost_by_stage": "Measured training required to qualify each stage.",
        "qualifying_validation_outcomes": "Selected qualifying validation batches; failures receive 500 steps in PAR-2.",
        "curriculum_progress": "Qualification progress over generated steps and recorded run hours.",
        "makespan_learning": "Successful-episode makespan; training uses a rolling 100 successful episodes and dots summarize successful validation cases.",
        "failure_aware_learning": "Failure-aware learning curve: unsuccessful episodes receive 500 steps (PAR-2).",
        "uav_travel_learning": "Recorded UAV travel over training and validation episodes.",
        "learning_transport_telemetry": "Seeded stratified recorded telemetry with a trailing 50-observation mean; deployed-model age includes the publication interval. Raw sampled values are in the CSV files.",
        "complete_message_delays": "Mean complete-message delivery latency; horizontal ticks mark the 95th percentile."}
    blocks = ['<!doctype html><meta charset="utf-8"><title>LEO learning results</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;color:#172b45;padding:0 20px}img{width:100%}figure{margin:40px 0}p{line-height:1.6}a{color:#2563b5}table{border-collapse:collapse}td,th{padding:10px 18px;border-bottom:1px solid #ddd}</style><h1>LEO learning comparison</h1><p>Both conditions qualified all four stages. Blue: perfect communication. Orange: LEO learning transport.</p>']
    blocks.append('<table><tr><th>Condition</th><th>Episodes</th><th>Environment steps</th><th>Recorded hours</th></tr>')
    for a, t in zip(ARMS, totals):
        blocks.append(f'<tr><td>{LABELS[a]}</td><td>{t["episodes"]:,}</td><td>{t["generated_joint_steps"]:,}</td><td>{t["wall_seconds"]/3600:.2f}</td></tr>')
    blocks.append('</table>')
    for name in figures:
        blocks.append(f'<figure><img src="{name}.png"><figcaption>{html.escape(captions[name])} <a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></figcaption></figure>')
    blocks.append('<p>Single matched seed. Validation uses fresh communication. Recorded hours exclude infrastructure hangs and recovery downtime and include simulator overhead. See <a href="results.json">results.json</a> for audit/provenance and CSV files for the numerical results.</p>')
    (output/"index.html").write_text('\n'.join(blocks), encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": figures, "totals": totals}, indent=2), flush=True)


if __name__ == "__main__":
    main()
