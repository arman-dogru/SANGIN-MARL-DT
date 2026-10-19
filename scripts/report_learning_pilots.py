"""Plot recorded screening checks and episode outcomes, without extension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A running writer may have an incomplete final line.
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output/learning-pilots")
    args = parser.parse_args()
    root = Path(args.root)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    training_fig, training_ax = plt.subplots(figsize=(9, 4), layout="constrained")
    summary = []
    for run in sorted(root.glob("*learning-20261005-*")):
        manifest_path = run / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        gamma = manifest.get("gamma", 0.99)
        for candidate in manifest["candidates"]:
            name, learning_rate, decay = candidate
            checks = read(run / name / "screening_checks.jsonl")
            if not checks:
                continue
            label = f"γ={gamma:g}, lr={learning_rate:g}, decay={decay // 1000}k"
            x = [c["generated_joint_steps"] for c in checks]
            for ax, key, scale in zip(
                axes,
                ("success_rate", "mean_par2", "distance_fraction_remaining"),
                (100, 1, 100),
            ):
                ax.plot(
                    x, [c[key] * scale for c in checks], "o-", label=label, markersize=4
                )
            episodes = read(run / name / "training_episodes.jsonl")
            successes = np.array([e["success"] for e in episodes], dtype=float)
            if len(successes):
                window = min(50, len(successes))
                rolling = (
                    np.convolve(successes, np.ones(window) / window, mode="valid") * 100
                )
                training_ax.plot(
                    [e["global_episode"] for e in episodes][window - 1 :],
                    rolling,
                    label=label,
                )
            summary.append(
                {
                    "run_id": run.name,
                    "candidate": name,
                    "gamma": gamma,
                    "learning_rate": learning_rate,
                    "epsilon_decay_steps": decay,
                    **checks[-1],
                }
            )
    for ax, ylabel in zip(
        axes,
        (
            "Greedy screening success (%)",
            "Failure-aware PAR-2 (steps)",
            "Goal distance remaining (%)",
        ),
    ):
        ax.set(xlabel="Generated joint steps", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 510)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Two-agent hyperparameter screening · 30 separate tuning cases per check",
        fontsize=11,
    )
    fig.savefig(output / "hyperparameter_screening.png", dpi=190)
    fig.savefig(output / "hyperparameter_screening.pdf")
    plt.close(fig)
    training_ax.set(
        xlabel="Training episode", ylabel="Training success (%)", ylim=(0, 100)
    )
    training_ax.grid(alpha=0.2)
    training_ax.legend(fontsize=8)
    training_ax.set_title("Measured training outcomes · trailing 50-episode mean")
    training_fig.savefig(output / "training_screening.png", dpi=190)
    plt.close(training_fig)
    (root / "screening_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if summary:
        keys = list(dict.fromkeys(k for row in summary for k in row))
        with (root / "screening_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
