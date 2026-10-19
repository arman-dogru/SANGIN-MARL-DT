"""Paired held-out frozen-policy experiments. Every output is measured."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import COMMUNICATION_SCHEMA, FIXED_MAP, ExperimentMode
from .environment import ExperimentEnvironment
from .models import choose_device, select_actions
from .records import append_jsonl, export_csv, read_jsonl, sha256, write_json
from .training import load_model


def case_bank(config, cases=50, counts=(2, 4, 6, 8)):
    records = []
    for count in counts:
        for case in range(cases):
            seed = 50_000_000 + count * 10_000 + case
            env = ExperimentEnvironment(
                config, ExperimentMode.UAV, FIXED_MAP, count, 250, seed
            )
            env.reset()
            records.append(
                {
                    "case_id": f"agents-{count}-case-{case:03d}",
                    "case": case,
                    "num_agents": count,
                    "seed": seed,
                    "starts": [list(a.position) for a in env.agents],
                    "goals": [list(a.goal) for a in env.agents],
                }
            )
    return records


def trace_frame(env, versions):
    return {
        "step": env.step_count,
        "time": env.step_count * env.config.communication.seconds_per_step,
        "positions": [list(a.position) for a in env.agents],
        "uav_position": env.uav.position.tolist(),
        "uav_travel_m": env.uav.total_distance_m,
        "model_versions": versions,
        "agents_at_goal": sum(a.at_goal for a in env.agents),
        **env.communication.snapshot(),
    }


def run_episode(config, model, case, device, model_version=0, *, decision_deadline_seconds=0.0):
    env = ExperimentEnvironment(
        config, ExperimentMode.UAV, FIXED_MAP, case["num_agents"], 250, case["seed"]
    )
    observations, _ = env.reset()
    deadline = decision_deadline_seconds
    if deadline:
        observations = env.prepare_decision(deadline)
    versions = {a: 0 for a in observations}
    display_versions = {a: model_version for a in observations}
    frames = [trace_frame(env, display_versions)]
    events = list(env.communication.transport.log)
    env.communication.transport.log.clear()
    collisions = 0
    while True:
        actions = select_actions(
            {0: model},
            versions,
            observations,
            0.0,
            np.random.default_rng(case["seed"]),
            device,
        )
        result = env.step(actions)
        observations = result.observations
        if deadline and not (result.terminated or result.truncated):
            observations = env.prepare_decision(deadline)
        collisions += result.info.get("collisions_obstacle_step", 0) + result.info.get(
            "collisions_agent_agent_step", 0
        )
        frame = trace_frame(env, display_versions)
        frame["actions"] = actions
        frames.append(frame)
        events.extend(env.communication.transport.log)
        env.communication.transport.log.clear()
        if result.terminated or result.truncated:
            break
    exposures = [event for event in events if event["event"] == "reservation_exposure"]
    by_agent = {a: sum(e["agent"] == a for e in exposures) for a in versions}
    realized_windows = {
        a: len({(e["start"], e["end"]) for e in exposures if e["agent"] == a})
        for a in versions
    }
    ages = {
        field: [f["information_age"][a][field] for f in frames for a in versions]
        for field in ("task", "intentions", "priorities")
    }
    row = {
        **case,
        "steps": env.step_count,
        "success": result.terminated,
        "successful_makespan": env.step_count if result.terminated else None,
        "par2": env.step_count if result.terminated else 500,
        "completion_fraction": sum(a.at_goal for a in env.agents) / env.num_agents,
        "ever_completion_fraction": sum(a.ever_reached_goal for a in env.agents)
        / env.num_agents,
        "collision_attempts": collisions,
        "uav_travel_m": env.uav.total_distance_m,
        "exposed_messages": len(exposures),
        "exposed_agents": sum(n > 0 for n in by_agent.values()),
        "exposure_by_agent": by_agent,
        "realized_windows_by_agent": realized_windows,
        **{
            f"mean_{key}_age_seconds": float(np.mean(values))
            for key, values in ages.items()
        },
    }
    trace = {
        "case": case,
        "grid": env.grid.tolist(),
        "goals": case["goals"],
        "frames": frames,
        "transport_events": events,
        "handover_windows": env.communication.transport.handover_windows(
            0.0, env.communication.transport.now
        ),
    }
    return row, trace


def treatments(config):
    result = [
        (
            "perfect",
            replace(
                config.communication,
                profile="perfect",
                stale_information=(),
                handover_enabled=False,
                contention_enabled=False,
            ),
        )
    ]
    for route in ("uav", "cloud"):
        for fields in (
            ("task",),
            ("intentions",),
            ("priorities",),
            ("task", "intentions", "priorities"),
        ):
            field_name = "all" if len(fields) == 3 else fields[0]
            for delay in config.evaluation.delay_steps:
                result.append(
                    (
                        f"controlled_{route}_{field_name}_{delay}",
                        replace(
                            config.communication,
                            mission_route=route,
                            profile="controlled",
                            stale_information=fields,
                            delay_steps=delay,
                            handover_enabled=False,
                            contention_enabled=True,
                        ),
                    )
                )
        for treatment, handover, contention in (
            ("handover", True, False),
            ("contention", False, True),
            ("combined", True, True),
        ):
            result.append(
                (
                    f"physical_{route}_{treatment}",
                    replace(
                        config.communication,
                        mission_route=route,
                        profile="physical",
                        stale_information=("task", "intentions", "priorities"),
                        handover_enabled=handover,
                        contention_enabled=contention,
                    ),
                )
            )
    return result


def paired_summary(rows, reference, bootstrap_samples=2000):
    baseline = {r["case_id"]: r for r in reference}
    rng = np.random.default_rng(4200)
    result = []
    for count in sorted({r["num_agents"] for r in rows}):
        group = [r for r in rows if r["num_agents"] == count]
        success = np.array([r["success"] for r in group], float)
        par2 = np.array([r["par2"] for r in group], float)
        matched = [r for r in group if r["case_id"] in baseline]
        deltas = np.array(
            [r["par2"] - baseline[r["case_id"]]["par2"] for r in matched], float
        )
        success_delta = np.array(
            [r["success"] - baseline[r["case_id"]]["success"] for r in matched], float
        )
        if len(matched):
            indices = rng.integers(0, len(matched), (bootstrap_samples, len(matched)))
            interval = np.quantile(deltas[indices].mean(1), [0.025, 0.975]).tolist()
            success_interval = np.quantile(
                success_delta[indices].mean(1), [0.025, 0.975]
            ).tolist()
        else:
            interval = success_interval = None
        successful = [r["steps"] for r in group if r["success"]]
        result.append(
            {
                "num_agents": count,
                "cases": len(group),
                "success_rate": float(success.mean()),
                "successful_makespan": float(np.mean(successful))
                if successful
                else None,
                "par2": float(par2.mean()),
                "completion_fraction": float(
                    np.mean([r["completion_fraction"] for r in group])
                ),
                "collision_attempts": float(
                    np.mean([r["collision_attempts"] for r in group])
                ),
                "uav_travel_m": float(np.mean([r["uav_travel_m"] for r in group])),
                "paired_cases": len(matched),
                "paired_par2_delta": float(deltas.mean()) if len(matched) else None,
                "paired_par2_ci95": interval,
                "paired_success_delta": float(success_delta.mean())
                if len(matched)
                else None,
                "paired_success_ci95": success_interval,
            }
        )
    return result


def evaluate_suite(
    config,
    model_path,
    output,
    device_name="auto",
    cases=50,
    counts=(2, 4, 6, 8),
    diagnostic=False,
    selected=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(device_name)
    import torch

    torch.set_num_threads(1)
    model, payload = load_model(
        model_path, config, device, require_qualified=not diagnostic
    )
    if not diagnostic and (cases != 50 or tuple(counts) != (2, 4, 6, 8)):
        raise ValueError(
            "Full study requires 50 held-out cases at 2/4/6/8 agents; use --diagnostic for smaller checks"
        )
    cases_data = case_bank(config, cases, counts)
    bank = {"map_sha256": sha256(config.resolve(FIXED_MAP)), "cases": cases_data}
    bank_path = output / "held_out_cases.json"
    if bank_path.exists() and json.loads(bank_path.read_text()) != bank:
        raise ValueError("Existing held-out case bank differs; choose new output")
    if not bank_path.exists():
        write_json(bank_path, bank)
    identity = {
        "communication_schema": COMMUNICATION_SCHEMA,
        "model_sha256": sha256(model_path),
        "model_run_id": payload["run_id"],
        "diagnostic": diagnostic,
        "case_bank_sha256": sha256(output / "held_out_cases.json"),
        "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "config": json.loads(json.dumps(config.to_dict())),
        "trace_sha256": {
            name: sha256(path)
            for name, path in (
                ("uplink", config.communication.uplink_trace),
                ("downlink", config.communication.downlink_trace),
                ("orbital_timeline", config.communication.leo_timeline),
            )
            if path
        },
    }
    manifest_path = output / "manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Evaluation output identity mismatch; choose new output")
    choices = treatments(config)
    if selected:
        choices = [item for item in choices if item[0] in set(selected) | {"perfect"}]
    condition_manifest = {item["name"]: item for item in previous.get("conditions", [])}
    for name, comm in choices:
        condition_manifest[name] = {"name": name, "transport": asdict(comm)}
    write_json(
        manifest_path,
        {
            **identity,
            "config": config.to_dict(),
            "conditions": list(condition_manifest.values()),
            "results": "measured; no synthetic extension",
            "pairing": "identical held-out starts, goals and reservation seeds",
            "reference": "one perfect zero-delay reference shared by routes",
        },
    )
    reference = read_jsonl(output / "perfect" / "episodes.jsonl")
    all_summaries = []
    for name, comm in choices:
        directory = output / name
        directory.mkdir(exist_ok=True)
        existing = read_jsonl(directory / "episodes.jsonl")
        done = {r["case_id"] for r in existing}
        condition_config = replace(config, communication=comm)
        for case in cases_data:
            if case["case_id"] in done:
                continue
            row, trace = run_episode(
                condition_config, model, case, device, payload.get("learner_updates", 0)
            )
            row.update(
                {
                    "condition": name,
                    "mission_route": comm.mission_route,
                    "communication_profile": comm.profile,
                    "stale_information": list(comm.stale_information),
                    "delay_steps": comm.delay_steps
                    if comm.profile == "controlled"
                    else 0,
                    "diagnostic": diagnostic,
                }
            )
            # One complete, independently compressed episode file; safe resumability by case ID.
            with gzip.open(
                directory / f"{case['case_id']}.json.gz", "wt", encoding="utf-8"
            ) as stream:
                json.dump(trace, stream, separators=(",", ":"), allow_nan=False)
            append_jsonl(directory / "episodes.jsonl", row)
            existing.append(row)
        if name == "perfect":
            reference = existing
        summary = paired_summary(
            existing, reference, config.evaluation.bootstrap_samples
        )
        for item in summary:
            item["condition"] = name
        all_summaries.extend(summary)
        write_json(directory / "summary.json", summary)
        export_csv(directory / "episodes.jsonl", directory / "episodes.csv")
    all_summaries = [
        item
        for directory in sorted(output.iterdir())
        if directory.is_dir() and (directory / "summary.json").exists()
        for item in json.loads((directory / "summary.json").read_text())
    ]
    write_json(output / "summary.json", all_summaries)
    return all_summaries
