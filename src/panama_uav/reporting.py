"""Plots consume measured records only; missing phases stay explicitly missing."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .records import export_csv, read_jsonl, write_json

COLORS = ["#2878b5", "#e5952d", "#8a65b0", "#259d80"]


def rolling(values, window=50):
    values = np.asarray(values, float)
    return np.array(
        [values[max(0, i - window + 1) : i + 1].mean() for i in range(len(values))]
    )


def save(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / (name + ".png"), dpi=180)
    fig.savefig(directory / (name + ".pdf"))
    plt.close(fig)


def sampled_records(path, max_rows=4000):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as stream:
        count = sum(1 for _ in stream)
    stride = max(1, int(np.ceil(count / max_rows)))
    with path.open() as stream:
        return [
            json.loads(line)
            for i, line in enumerate(stream)
            if i % stride == 0 or i == count - 1
        ]


def report_training(root):
    root = Path(root)
    output = root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    arms = [
        p
        for p in root.iterdir()
        if p.is_dir()
        and (p / "manifest.json").exists()
        and (p / "status.json").exists()
    ]
    data = {p.name: read_jsonl(p / "training.jsonl") for p in arms}
    validations = {p.name: read_jsonl(p / "validation.jsonl") for p in arms}
    for phase in range(4):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        available = False
        for index, (name, rows) in enumerate(data.items()):
            group = [r for r in rows if r["stage_index"] == phase]
            if not group:
                continue
            available = True
            color = COLORS[index % 4]
            dots = [
                r
                for r in validations[name]
                if r["stage_index"] == phase and r["complete"]
            ]
            for column, xkey in enumerate(("stage_episode", "generated_joint_steps")):
                x = [r[xkey] for r in group]
                axes[0, column].plot(
                    x,
                    100 * rolling([r["success"] for r in group]),
                    label=name,
                    color=color,
                    marker="." if len(group) == 1 else None,
                )
                axes[0, column].scatter(
                    [r[xkey] for r in dots],
                    [100 * r["success_rate"] for r in dots],
                    color=color,
                    s=35,
                    zorder=4,
                )
                axes[1, column].plot(
                    x,
                    rolling([r["par2"] for r in group]),
                    label=name,
                    color=color,
                    marker="." if len(group) == 1 else None,
                )
                successes = [r for r in group if r["success"]]
                axes[1, column].scatter(
                    [r[xkey] for r in successes],
                    [r["steps"] for r in successes],
                    color=color,
                    s=4,
                    alpha=0.3,
                )
        for col, label in enumerate(
            ("Training episode in phase", "Generated joint steps")
        ):
            axes[0, col].set_ylim(0, 100)
            axes[0, col].set_ylabel("Success (%) · rolling 50; validation dots")
            axes[1, col].set_ylabel("PAR-2 (steps); successful makespan dots")
            axes[1, col].set_xlabel(label)
        for ax in axes.flat:
            ax.grid(alpha=0.2)
            if available:
                ax.legend(fontsize=8)
        fig.suptitle(
            f"{(2, 4, 6, 8)[phase]} agents · fixed random 32 × 32 map"
            if available
            else f"{(2, 4, 6, 8)[phase]} agents · no measured completed training episodes"
        )
        save(fig, output, f"training_phase_{phase}")
    # Plot the common generated-step range without extending any run.
    series = {p.name: sampled_records(p / "learning_steps.jsonl") for p in arms}
    limits = [rows[-1]["generated_joint_steps"] for rows in series.values() if rows]
    common = min(limits) if limits else 0
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for index, (name, rows) in enumerate(series.items()):
        group = [r for r in rows if r["generated_joint_steps"] <= common]
        if not group:
            continue
        x = [r["generated_joint_steps"] for r in group]
        color = COLORS[index % 4]
        arrays = [
            [r["delivered_experiences"] for r in group],
            [r["learner_updates"] for r in group],
            [np.mean(list(r["deployed_model_age_seconds"].values())) for r in group],
            [np.mean(list(r["model_version_lag"].values())) for r in group],
            [
                np.mean(r["replay_age_seconds"]) if r["replay_age_seconds"] else np.nan
                for r in group
            ],
            [r["wall_seconds"] for r in group],
        ]
        for ax, y in zip(axes.flat, arrays):
            ax.plot(x, y, label=name, color=color)
    labels = (
        "Delivered experiences",
        "Learner updates",
        "Deployed model age (s)",
        "Model version lag",
        "Sampled replay age (s)",
        "Wall-clock progress (s)",
    )
    for ax, label in zip(axes.flat, labels):
        ax.set_xlabel("Generated joint steps")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        if common:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"Learning transport · common measured range: 0–{common} generated steps"
    )
    save(fig, output, "learning_common_steps")
    for p in arms:
        rows = series[p.name]
        if not rows:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        x = [r["generated_joint_steps"] for r in rows]
        axes[0, 0].plot(
            x,
            [
                np.mean(r["experience_delivery_age_seconds"])
                if r["experience_delivery_age_seconds"]
                else np.nan
                for r in rows
            ],
        )
        axes[0, 0].set_ylabel("Delivered experience age (s)")
        keys = list(dict.fromkeys(k for r in rows for k in r["queue_bytes"]))
        for key in keys:
            axes[0, 1].plot(x, [r["queue_bytes"].get(key, 0) for r in rows], label=key)
        axes[0, 1].set_ylabel("Queued / buffered bytes")
        axes[0, 1].legend(fontsize=7)
        axes[1, 0].plot(x, [r["uav_travel_m"] for r in rows])
        axes[1, 0].set_ylabel("Episode UAV travel (m)")
        episodes = data[p.name]
        axes[1, 1].plot(
            [r["wall_seconds"] / 3600 for r in episodes],
            100 * rolling([r["success"] for r in episodes]),
        )
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].set_ylabel("Training success (%)")
        axes[1, 1].set_xlabel("Wall-clock hours")
        for ax in axes.flat:
            ax.grid(alpha=0.2)
        for ax in axes.flat:
            if ax is not axes[1, 1]:
                ax.set_xlabel("Generated joint steps")
        fig.suptitle(p.name)
        save(fig, output, p.name + "_transport")
        for name in ("training", "validation"):
            export_csv(p / (name + ".jsonl"), p / (name + ".csv"))
    write_json(
        output / "plot_manifest.json",
        {
            "input_directories": [str(p) for p in arms],
            "common_generated_steps": common,
            "smoothing": "trailing 50 recorded episodes; no extrapolation",
            "learning_step_display": "recorded stride sample, at most 4000 rows per run",
            "missing_phases": "not trained or no completed episodes; never filled",
        },
    )
    return output


def report_evaluation(root):
    root = Path(root)
    output = root / "plots"
    output.mkdir(exist_ok=True)
    rows = json.loads((root / "summary.json").read_text())
    exposure_rows = {
        directory.name: read_jsonl(directory / "episodes.jsonl")
        for directory in root.iterdir()
        if directory.is_dir() and (directory / "episodes.jsonl").exists()
    }
    for count in sorted({r["num_agents"] for r in rows}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for route, color in zip(("uav", "cloud"), COLORS):
            for field, style in zip(
                ("task", "intentions", "priorities", "all"), ("-", "--", ":", "-.")
            ):
                group = [
                    r
                    for r in rows
                    if r["num_agents"] == count
                    and r["condition"].startswith(f"controlled_{route}_{field}_")
                ]
                group.sort(key=lambda r: float(r["condition"].rsplit("_", 1)[1]))
                if not group:
                    continue
                base = next(
                    r
                    for r in rows
                    if r["condition"] == "perfect" and r["num_agents"] == count
                )
                points = [base] + group
                x = [0] + [float(r["condition"].rsplit("_", 1)[1]) for r in group]
                for ax, key, multiplier in zip(
                    axes.flat,
                    (
                        "success_rate",
                        "successful_makespan",
                        "par2",
                        "completion_fraction",
                    ),
                    (100, 1, 1, 100),
                ):
                    ax.plot(
                        x,
                        [
                            r[key] * multiplier if r[key] is not None else np.nan
                            for r in points
                        ],
                        style,
                        color=color,
                        label=f"{route} · {field}",
                        marker="o",
                        markersize=3,
                    )
        for ax, label in zip(
            axes.flat,
            (
                "Success (%)",
                "Successful makespan (steps)",
                "Failure-aware PAR-2 (steps)",
                "Completion at termination (%)",
            ),
        ):
            ax.set_xlabel("Additional contention delay (steps)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
        axes[0, 0].set_ylim(0, 100)
        axes[1, 1].set_ylim(0, 100)
        if any(r["condition"].startswith("controlled") for r in rows):
            axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Frozen policy · {count} agents · paired held-out cases")
        save(fig, output, f"inference_agents_{count}")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for route, color in zip(("uav", "cloud"), COLORS):
            for field, style in zip(
                ("task", "intentions", "priorities", "all"), ("-", "--", ":", "-.")
            ):
                group = [
                    r
                    for r in rows
                    if r["num_agents"] == count
                    and r["condition"].startswith(f"controlled_{route}_{field}_")
                ]
                group.sort(key=lambda r: float(r["condition"].rsplit("_", 1)[1]))
                if not group:
                    continue
                base = next(
                    r
                    for r in rows
                    if r["condition"] == "perfect" and r["num_agents"] == count
                )
                points = [base] + group
                x = [0] + [float(r["condition"].rsplit("_", 1)[1]) for r in group]
                metrics = [[], [], [], []]
                for point in points:
                    cases = [
                        r
                        for r in exposure_rows[point["condition"]]
                        if r["num_agents"] == count
                    ]
                    metrics[0].append(point["collision_attempts"])
                    metrics[1].append(point["uav_travel_m"])
                    metrics[2].append(
                        100 * np.mean([r["exposed_agents"] / count for r in cases])
                    )
                    metrics[3].append(
                        np.mean(
                            [
                                sum(r.get("realized_windows_by_agent", {}).values())
                                / count
                                for r in cases
                            ]
                        )
                    )
                for ax, values in zip(axes.flat, metrics):
                    ax.plot(
                        x,
                        values,
                        style,
                        color=color,
                        marker="o",
                        markersize=3,
                        label=f"{route} - {field}",
                    )
        for ax, label in zip(
            axes.flat,
            (
                "Collision attempts",
                "UAV travel (m)",
                "Agents actually exposed (%)",
                "Realized windows per agent",
            ),
        ):
            ax.set_xlabel("Additional contention delay (steps)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
        axes[1, 0].set_ylim(0, 100)
        if any(r["condition"].startswith("controlled") for r in rows):
            axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{count} agents - measured movement and realized exposure")
        save(fig, output, f"inference_agents_{count}_movement_exposure")
    # Timelines and traffic use the first actually recorded case in every treatment.
    age_distributions = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        traces = sorted(directory.glob("*.json.gz"))
        if not traces:
            continue
        hist = {key: [] for key in ("task", "intentions", "priorities")}
        for path in traces:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
            for frame in trace["frames"]:
                for ages in frame["information_age"].values():
                    for field, values in hist.items():
                        values.append(ages[field])
        age_distributions[directory.name] = {
            key: {
                "mean": float(np.mean(v)),
                "p50": float(np.quantile(v, 0.5)),
                "p95": float(np.quantile(v, 0.95)),
                "max": max(v),
                "samples": len(v),
            }
            for key, v in hist.items()
        }
        with gzip.open(traces[0], "rt", encoding="utf-8") as stream:
            trace = json.load(stream)
        frames = trace["frames"]
        x = [f["time"] for f in frames]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for field, values in hist.items():
            axes[0, 0].hist(values, bins=30, histtype="step", density=True, label=field)
        axes[0, 0].set_xlabel("Information age (s)")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].legend()
        for agent in frames[0]["information_age"]:
            axes[0, 1].plot(
                x,
                [f["information_age"][agent]["task"] for f in frames],
                label=f"Agent {agent}",
            )
        axes[0, 1].set_ylabel("Task guidance age (s)")
        axes[0, 1].legend(fontsize=7)
        for key in frames[0]["queue_bytes"]:
            axes[1, 0].plot(
                x, [f["queue_bytes"].get(key, 0) for f in frames], label=key
            )
        axes[1, 0].set_ylabel("Queued / buffered bytes")
        axes[1, 0].legend(fontsize=6)
        events = [e for e in trace["transport_events"] if e["event"] == "delivered"]
        axes[1, 1].step(
            [e["time"] for e in events],
            np.cumsum([e.get("bytes", 0) for e in events]),
            where="post",
        )
        axes[1, 1].set_ylabel("Delivered bytes (all hops' final receipts)")
        for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
            ax.set_xlabel("Simulated seconds")
            for start, end in trace["handover_windows"]:
                ax.axvspan(start, end, color="red", alpha=0.3)
        for ax in axes.flat:
            ax.grid(alpha=0.2)
        fig.suptitle(directory.name + " · measured information / transport")
        save(fig, output, directory.name + "_information")
        fig, timeline_axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for ax, field in zip(timeline_axes, ("task", "intentions", "priorities")):
            for agent in frames[0]["information_age"]:
                ax.plot(
                    x,
                    [f["information_age"][agent][field] for f in frames],
                    label=f"Agent {agent}",
                )
            for begin, finish in trace["handover_windows"]:
                ax.axvspan(begin, finish, color="red", alpha=0.3)
            windows = [
                event
                for event in trace["transport_events"]
                if event["event"] == "scheduled_reservation"
            ]
            for event in windows:
                ax.axvspan(event["start"], event["end"], color="gold", alpha=0.06)
            ax.set_ylabel(field + " age (s)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        timeline_axes[-1].set_xlabel("Simulated seconds")
        fig.suptitle(directory.name + " - all receiver cache timelines")
        save(fig, output, directory.name + "_timelines")
    physical = [r for r in rows if r["condition"].startswith("physical")]
    if physical:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for count in sorted({r["num_agents"] for r in physical}):
            group = [r for r in physical if r["num_agents"] == count]
            x = np.arange(len(group))
            for ax, key in zip(axes, ("success_rate", "par2", "uav_travel_m")):
                ax.plot(
                    x,
                    [r[key] * (100 if key == "success_rate" else 1) for r in group],
                    marker="o",
                    label=f"{count} agents",
                )
                ax.set_xticks(
                    x,
                    [r["condition"].replace("physical_", "") for r in group],
                    rotation=40,
                    ha="right",
                )
        for ax, label in zip(axes, ("Success (%)", "PAR-2 (steps)", "UAV travel (m)")):
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            ax.legend()
        axes[0].set_ylim(0, 100)
        save(fig, output, "physical_treatments")
    write_json(output / "information_age_summary.json", age_distributions)
    return output
