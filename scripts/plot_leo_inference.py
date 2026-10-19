"""Editable plots and audit of paired frozen-policy mission staleness tests."""
import argparse
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COUNTS = (2, 4, 6, 8)
FIELDS = ("task", "intentions", "priorities", "all")
LABELS = {"task": "Stale A*", "intentions": "Stale intentions", "priorities": "Stale priorities", "all": "All stale"}
COLORS = {"task": "#2563b5", "intentions": "#d96b20", "priorities": "#8b5abf", "all": "#cb3348"}
METRICS = (("success_rate", "Success (%)", 100),
           ("successful_makespan", "Successful makespan (steps)", 1),
           ("par2", "Failure-aware PAR-2 (steps)", 1))


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def audit(root, manifest, partial=False):
    bank = json.loads((root/"held_out_cases.json").read_text())
    reference = {r["case_id"]: r for r in bank}
    assert len(reference) == 200 and manifest["baseline_qualified"]
    assert sha(root/"baseline.pt") == manifest["baseline_model_sha256"]
    assert sha(root/"held_out_cases.json") == manifest["case_bank_sha256"]
    assert {p.name: sha(p) for p in (root/"code/panama_uav").glob("*.py")} == manifest["code_sha256"]
    summaries = {(r["condition"], r["num_agents"]): r
                 for r in json.loads((root/"summary.json").read_text())}
    checks, total = [], 0
    for spec in manifest["conditions"]:
        name = spec["name"]
        path = root/name/"episodes.jsonl"
        if not path.exists():
            if partial: continue
            raise ValueError(f"Missing condition: {name}")
        rows = read(path)
        if len(rows) < 200 and partial: continue
        assert len(rows) == 200 and {r["case_id"] for r in rows} == set(reference)
        for row in rows:
            case = reference[row["case_id"]]
            assert all(row[key] == case[key] for key in ("seed", "starts", "goals", "num_agents"))
            assert 0 < row["steps"] <= 250
            assert row["par2"] == (row["steps"] if row["success"] else 500)
            assert row["policy_model_sha256"] == manifest["baseline_model_sha256"]
            assert row["stale_information"] == spec["transport"]["stale_information"]
            for field in ("task", "intentions", "priorities"):
                if field not in row["stale_information"] or name == "perfect":
                    assert row[f"maximum_{field}_age_seconds"] == 0
        for count in COUNTS:
            group = [r for r in rows if r["num_agents"] == count]
            assert len(group) == 50
            summary = summaries[name, count]
            successful = [r["steps"] for r in group if r["success"]]
            for key, measured in (("success_rate", np.mean([r["success"] for r in group])),
                                  ("par2", np.mean([r["par2"] for r in group]))):
                assert np.isclose(summary[key], measured, rtol=0, atol=1e-12)
            if successful:
                assert np.isclose(summary["successful_makespan"], np.mean(successful), rtol=0, atol=1e-12)
            else:
                assert summary["successful_makespan"] is None
            with gzip.open(root/name/f"{group[0]['case_id']}.json.gz", "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
            assert trace["frames"][0]["positions"] == group[0]["starts"]
            assert len({v for f in trace["frames"] for v in f["model_versions"].values()}) == 1
        total += len(rows)
        checks.append({"condition": name, "episode_records": len(rows), "trace_samples_verified": 4,
                       "sha256": sha(path)})
    if not partial: assert total == manifest["planned_episode_executions"]
    return {"status": "passed", "completed_conditions": len(checks), "episode_records": total,
            "paired_cases": 200, "same_seeds_starts_goals_verified": True,
            "selected_input_isolation_verified": True, "frozen_policy_verified": True,
            "episode_cap_250_verified": True, "summary_values_recomputed": True, "conditions": checks}


def save(fig, output, name, figures):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output/f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures.append(name)


def figures_for(rows, output):
    figures = []
    perfect = {r["stage"]: r for r in rows if r["condition"] == "perfect"}
    for route in ("cloud", "uav"):
        route_label = "Cloud processing" if route == "cloud" else "UAV-edge processing"
        for metric, label, scale in METRICS:
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            fig.subplots_adjust(left=.08, right=.985, bottom=.08, top=.84, wspace=.22, hspace=.36)
            for stage, ax in enumerate(axes.flat, 1):
                for field in FIELDS:
                    group = sorted((r for r in rows if r["communication_profile"] == "controlled"
                                    and r["mission_route"] == route and r["fields"] == field and r["stage"] == stage),
                                   key=lambda r: r["delay_steps"])
                    ax.plot([r["delay_steps"] for r in group],
                            [r[metric]*scale if r[metric] is not None else np.nan for r in group],
                            color=COLORS[field], marker="o", lw=1.7, label=LABELS[field])
                baseline = perfect[stage][metric]
                if baseline is not None:
                    ax.axhline(baseline*scale, color="#334155", ls="--", lw=1.7, label="Perfect communication")
                ax.set(title=f"Stage {stage}", xlabel="Additional controlled delay (steps)", ylabel=label,
                       xticks=[1, 2, 5, 10, 20])
                if metric == "success_rate": ax.set_ylim(0, 100)
                ax.grid(alpha=.18)
            handles = [Line2D([], [], color="#334155", ls="--", label="Perfect communication")]
            handles += [Line2D([], [], color=COLORS[f], marker="o", label=LABELS[f]) for f in FIELDS]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .947), ncol=3, fontsize=10)
            fig.suptitle(f"{route_label} · controlled mission-information staleness", fontsize=13, y=.995)
            save(fig, output, f"controlled_{route}_{metric}", figures)
        for profile_name, profile, severity in (("orbital", "physical", None), ("controlled20", "controlled", 20)):
            for metric, label, scale in METRICS:
                comparison_bars(rows, output, route, route_label, profile_name, profile, severity,
                                metric, label, scale, perfect, figures)
    return figures


def comparison_bars(rows, output, route, route_label, profile_name, profile, severity,
                    metric, label, scale, perfect, figures):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for stage, ax in enumerate(axes.flat, 1):
        values = [perfect[stage][metric]]
        for field in FIELDS:
            group = [r for r in rows if r["communication_profile"] == profile
                     and r["mission_route"] == route and r["fields"] == field and r["stage"] == stage
                     and (severity is None or r["delay_steps"] == severity)]
            values.append(group[0][metric] if group else None)
        colors = ["#64748b"]+[COLORS[f] for f in FIELDS]
        bars = ax.bar(range(5), [v*scale if v is not None else 0 for v in values], color=colors, width=.68)
        for bar, value in zip(bars, values):
            height = 0 if value is None else value*scale
            inside = metric == "success_rate" and height >= 94
            ax.annotate(f"{height:.1f}" if value is not None else "N/A",
                        (bar.get_x()+bar.get_width()/2, height),
                        xytext=(0, -8 if inside else 4), textcoords="offset points",
                        ha="center", va="top" if inside else "bottom", fontsize=10,
                        color="white" if inside else "#172b45")
        ax.set(title=f"Stage {stage}", xticks=range(5), xticklabels=["Perfect", "A*", "Intentions", "Priorities", "All"], ylabel=label)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        if metric == "success_rate": ax.set_ylim(0, 100)
        else: ax.set_ylim(0, max([v*scale for v in values if v is not None] or [1])*1.2)
    title = "orbital LEO transport" if severity is None else f"controlled {severity}-step update holds"
    fig.suptitle(f"{route_label} · {title}", fontsize=14)
    save(fig, output, f"{profile_name}_{route}_{metric}", figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root/"plots"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/"manifest.json").read_text())
    state = json.loads((root/"status.json").read_text())
    if not args.partial: assert state["status"] == "complete"
    verified = audit(root, manifest, args.partial)
    rows = json.loads((root/"summary.json").read_text())
    export(output/"inference_summary.csv", rows)
    episodes = []
    for spec in manifest["conditions"]:
        path = root/spec["name"]/"episodes.jsonl"
        if path.exists(): episodes.extend(read(path))
    export(output/"paired_episode_results.csv", episodes)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "legend.frameon": False})
    figures = figures_for(rows, output)
    shutil.copy2(Path(__file__), output/"plot_leo_inference.py")
    report = {"source_root": str(root), "model_sha256": manifest["baseline_model_sha256"],
        "case_bank_sha256": manifest["case_bank_sha256"], "figures": figures, "audit": verified,
        "cases_per_stage": 50, "episode_cap": 250,
        "makespan": "Mean steps among successful episodes; N/A if none. PAR-2 assigns 500 to failures.",
        "reference": "One shared perfect-communication case bank and frozen model for all routes and severities",
        "independent_variables": "Controlled delay severities are additional seeded contention, not measured handover durations. Orbital plots contain no artificial contention.",
        "local_inputs": "Local obstacles, local robots and self-position stay fresh; only named mission caches are delayed",
        "decision_timing": "Reports and immediate decisions occur at step boundaries. Positive physical delivery times can cause previous-tick guidance even below one second; controlled transport is immediate outside held FIFO updates.",
        "synthetic_extension": False, "summaries": rows}
    (output/"results.json").write_text(json.dumps(report, indent=2))
    (root/"audit.json").write_text(json.dumps(verified, indent=2))
    blocks = ['<!doctype html><meta charset="utf-8"><title>Baseline mission inference</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 20px;color:#172b45}img{width:100%}figure{margin:40px 0}p{line-height:1.6}</style><h1>Frozen-baseline mission inference</h1><p>50 paired held-out cases per stage. The same model and cases are used throughout. Controlled stress tests and orbital LEO transport are separate.</p>']
    for name in figures:
        blocks.append(f'<figure><img src="{name}.png"><figcaption>{html.escape(name.replace("_", " "))} · <a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></figcaption></figure>')
    blocks.append('<p>Makespan averages successful episodes. PAR-2 includes failures as 500 steps. Plots show measured outcomes for one trained seed; paired case confidence intervals are in inference_summary.csv. Unselected mission inputs and all local sensing remain fresh.</p>')
    (output/"index.html").write_text('\n'.join(blocks), encoding="utf-8")
    with zipfile.ZipFile(root/"leo_inference_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file(): archive.write(path, path.name)
    print(json.dumps({"plots": str(output), "figures": figures, "audit": {k: v for k, v in verified.items() if k != "conditions"}}, indent=2))


if __name__ == "__main__":
    main()
