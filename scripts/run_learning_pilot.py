"""Measured hyperparameter screening, followed by a fresh qualified curriculum.

Screening cases are tuning cases, never the final held-out mission cases.
The 200-case / 95% curriculum gate remains in the production trainer.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trial-steps", type=int, default=40_000)
    parser.add_argument("--trial-seconds", type=float, default=1_200)
    parser.add_argument("--followup-seconds", type=float, default=10_800)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--candidate-set", choices=("standard", "bootstrap"), default="standard"
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project / "output" / "learning-pilots" / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    # Freeze the actual source used by every candidate and the follow-up.
    # Independent orbit work cannot change a running learner or its resume hash.
    snapshot = root / "code" / "panama_uav"
    shutil.copytree(
        project / "src" / "panama_uav",
        snapshot,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(__file__, root / "code" / "run_learning_pilot.py")
    sys.path.insert(0, str(snapshot.parent))
    from dataclasses import replace

    import numpy as np
    import torch

    from panama_uav.config import ExperimentMode, load_config
    from panama_uav.environment import ExperimentEnvironment
    from panama_uav.models import choose_device, select_actions
    from panama_uav.records import append_jsonl, export_csv, sha256, write_json
    from panama_uav.training import (
        TrainingSession,
        atomic_save,
        model_payload,
        train_policy,
    )

    torch.set_num_threads(1)
    device = choose_device("auto")
    base = load_config(project / "configs" / "base.toml")
    candidates = [
        ("lr1e4_decay20k", 1e-4, 20_000),
        ("lr3e4_decay20k", 3e-4, 20_000),
        ("lr1e4_decay5k", 1e-4, 5_000),
        ("lr3e4_decay5k", 3e-4, 5_000),
    ]
    if args.candidate_set == "bootstrap":
        candidates = [
            ("lr3e4_decay20k", 3e-4, 20_000),
            ("lr1e3_decay20k", 1e-3, 20_000),
        ]
    write_json(
        root / "manifest.json",
        {
            "run_id": args.run_id,
            "started_utc": datetime.now(UTC).isoformat(),
            "seed": 42,
            "gamma": args.gamma,
            "purpose": "hyperparameter screening followed by fresh training",
            "candidates": candidates,
            "trial_joint_steps": args.trial_steps,
            "trial_wall_budget_seconds": args.trial_seconds,
            "screening_cases": 30,
            "screening_seed_start": 8_000_042,
            "screening_is_qualification": False,
            "production_validation_cases": 200,
            "production_graduation_success": 0.95,
            "source_sha256": {p.name: sha256(p) for p in snapshot.glob("*.py")},
            "reward_and_observations": "unchanged retained 0.1",
            "replay_capacity": 50_000,
            "results": "measured only",
        },
    )

    def evaluate(session, directory):
        outcomes = []
        action_counts = np.zeros(5, dtype=int)
        for case in range(30):
            seed = 8_000_042 + case
            env = ExperimentEnvironment(
                session.config,
                ExperimentMode.UAV,
                base.curriculum[0].map_path,
                2,
                250,
                seed,
            )
            obs, _ = env.reset()
            initial_distance = sum(a.distance_to_goal for a in env.agents)
            rng = np.random.default_rng(seed)
            while True:
                actions = select_actions(
                    {0: session.policy}, {a: 0 for a in obs}, obs, 0, rng, device
                )
                for action in actions.values():
                    action_counts[action] += 1
                result = env.step(actions)
                obs = result.observations
                if result.terminated or result.truncated:
                    break
            record = {
                "generated_joint_steps": session.generated_steps,
                "case": case,
                "seed": seed,
                "success": result.terminated,
                "steps": env.step_count,
                "par2": env.step_count if result.terminated else 500,
                "completion_fraction": sum(a.at_goal for a in env.agents) / 2,
                "distance_fraction_remaining": sum(
                    a.distance_to_goal for a in env.agents
                )
                / max(initial_distance, 1),
            }
            outcomes.append(record)
            append_jsonl(directory / "screening_episodes.jsonl", record)
        return {
            "generated_joint_steps": session.generated_steps,
            "episodes": session.episodes,
            "success_rate": float(np.mean([x["success"] for x in outcomes])),
            "mean_par2": float(np.mean([x["par2"] for x in outcomes])),
            "completion_fraction": float(
                np.mean([x["completion_fraction"] for x in outcomes])
            ),
            "distance_fraction_remaining": float(
                np.mean([x["distance_fraction_remaining"] for x in outcomes])
            ),
            "greedy_action_counts": action_counts.tolist(),
            "cases": len(outcomes),
        }

    summaries = []
    for name, learning_rate, decay in candidates:
        directory = root / name
        directory.mkdir()
        config = replace(
            base,
            training=replace(
                base.training,
                learning_rate=learning_rate,
                epsilon_decay_steps=decay,
                replay_capacity=50_000,
                checkpoint_interval_steps=50_000,
                gamma=args.gamma,
            ),
        )
        start = time.monotonic()
        session = TrainingSession(config, "clean", 42, args.run_id + "-" + name, device)
        write_json(
            directory / "manifest.json",
            {
                "config": config.to_dict(),
                "initial_weights_sha256": session.initial_weights_sha256,
                "diagnostic_only": True,
            },
        )
        next_check = 10_000
        latest = None
        print(f"START {name}", flush=True)
        with (directory / "learning_steps.jsonl").open(
            "w", encoding="utf-8"
        ) as step_log:
            while (
                session.generated_steps < args.trial_steps
                and time.monotonic() - start < args.trial_seconds - 30
            ):
                step, episode = session.step()
                step["wall_seconds"] = time.monotonic() - start
                step_log.write(json.dumps(step) + "\n")
                if episode is not None:
                    append_jsonl(directory / "training_episodes.jsonl", episode)
                    session.start_episode()
                if session.generated_steps >= next_check:
                    step_log.flush()
                    latest = evaluate(session, directory)
                    latest["wall_seconds"] = time.monotonic() - start
                    append_jsonl(directory / "screening_checks.jsonl", latest)
                    atomic_save(model_payload(session), directory / "latest_model.pt")
                    write_json(
                        root / "status.json",
                        {"status": "screening", "candidate": name, **latest},
                    )
                    print(json.dumps({"candidate": name, **latest}), flush=True)
                    next_check += 10_000
        if latest is None or latest["generated_joint_steps"] != session.generated_steps:
            latest = evaluate(session, directory)
            append_jsonl(directory / "screening_checks.jsonl", latest)
        atomic_save(model_payload(session), directory / "latest_model.pt")
        summary = {
            "candidate": name,
            "learning_rate": learning_rate,
            "epsilon_decay_steps": decay,
            "wall_seconds": time.monotonic() - start,
            **latest,
        }
        summaries.append(summary)
        write_json(directory / "summary.json", summary)
        export_csv(
            directory / "training_episodes.jsonl", directory / "training_episodes.csv"
        )
        write_json(root / "screening_summary.json", summaries)
        del session
        gc.collect()
        torch.cuda.empty_cache()
    best = max(
        summaries,
        key=lambda s: (
            s["success_rate"],
            -s["mean_par2"],
            s["completion_fraction"],
            -s["distance_fraction_remaining"],
        ),
    )
    write_json(
        root / "selection.json",
        {
            "selected": best,
            "criterion": "success, PAR-2, completion, remaining distance",
            "tuning_cases_are_not_final_test_cases": True,
        },
    )
    if not args.screen_only:
        selected = replace(
            base,
            training=replace(
                base.training,
                learning_rate=best["learning_rate"],
                epsilon_decay_steps=best["epsilon_decay_steps"],
                replay_capacity=50_000,
                checkpoint_interval_steps=50_000,
                budget_seconds=min(args.followup_seconds, 10_800),
                gamma=args.gamma,
            ),
        )
        write_json(
            root / "status.json",
            {"status": "fresh_followup_training", "selected": best},
        )
        print("START fresh curriculum " + best["candidate"], flush=True)
        outcome = train_policy(
            selected,
            "clean",
            42,
            root / "selected" / "clean",
            False,
            str(device),
            args.run_id + "-selected",
        )
        write_json(
            root / "status.json",
            {
                "status": "complete",
                "followup_status": outcome.status,
                "generated_joint_steps": outcome.environment_steps,
                "episodes": outcome.episodes,
                "final_model": str(outcome.final_model),
                "selected": best,
            },
        )
    else:
        write_json(
            root / "status.json", {"status": "complete_screening", "selected": best}
        )
    print("FINISHED " + str(root), flush=True)


if __name__ == "__main__":
    main()
