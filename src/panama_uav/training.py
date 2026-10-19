"""Single-mission online learning with versioned, virtual-time transport."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import COMMUNICATION_SCHEMA, FIXED_MAP, ExperimentMode, LearningCondition
from .environment import ExperimentEnvironment
from .learning import _learn
from .models import DQN, choose_device, cpu_state_dict, select_actions
from .records import append_jsonl, export_csv, sha256, write_json
from .replay import PrioritizedReplayBuffer
from .transport import Transport, encode


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    # Windows indexers can briefly lock the previous checkpoint. This is only
    # filesystem retry time; communication never advances through wall sleeps.
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.02 * (attempt + 1))


def fingerprint(config):
    return hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True).encode()
    ).hexdigest()


def new_replay(config, seed):
    t = config.training
    return PrioritizedReplayBuffer(
        t.replay_capacity,
        t.per_alpha,
        t.per_beta_start,
        t.per_beta_steps,
        t.per_epsilon,
        seed,
    )


def qualifies(stage, stage_episodes, validation):
    return (
        stage_episodes >= stage.min_episodes
        and validation["episodes"] == 200
        and validation["complete"]
        and validation["success_rate"] >= max(0.95, stage.graduation_success_rate)
    )


def model_payload(session):
    config = session.config
    return {
        "communication_schema": COMMUNICATION_SCHEMA,
        "policy_family": "panama_0.1_task",
        "task_reward": session.config.to_dict()["reward"],
        "model_state_dict": cpu_state_dict(session.policy),
        "seed": session.seed,
        "condition": session.condition.value,
        "run_id": session.run_id,
        "spatial_shape": (
            3,
            2 * config.map.fov_radius + 1,
            2 * config.map.fov_radius + 1,
        ),
        "vector_length": config.vector_length,
        "config_fingerprint": fingerprint(config),
        "map_sha256": sha256(config.resolve(FIXED_MAP)),
        "curriculum_completed": session.stage_index == len(config.curriculum),
        "qualification": session.qualification,
        "generated_joint_steps": session.generated_steps,
        "learner_updates": session.learner_updates,
    }


def load_model(path, config, device, require_qualified=False):
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("communication_schema") != COMMUNICATION_SCHEMA
        or payload.get("policy_family") != "panama_0.1_task"
    ):
        raise ValueError(
            "Legacy checkpoint rejected; a new communication-study run ID is required"
        )
    if payload.get("map_sha256") != sha256(config.resolve(FIXED_MAP)):
        raise ValueError("Model map mismatch")
    if payload.get("task_reward") != config.to_dict()["reward"]:
        raise ValueError("Model task-reward definition mismatch; use a fresh run ID")
    if (
        tuple(payload["spatial_shape"])
        != (3, 2 * config.map.fov_radius + 1, 2 * config.map.fov_radius + 1)
        or payload["vector_length"] != config.vector_length
    ):
        raise ValueError("Model observation mismatch")
    if require_qualified and not (
        payload.get("condition") == "clean"
        and payload.get("curriculum_completed")
        and [q["num_agents"] for q in payload.get("qualification", [])] == [2, 4, 6, 8]
        and all(
            q["episodes"] == 200 and q["success_rate"] >= 0.95
            for q in payload["qualification"]
        )
    ):
        raise ValueError(
            "Mission comparison requires the clean baseline qualified at all four phases"
        )
    model = DQN(tuple(payload["spatial_shape"]), payload["vector_length"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


@dataclass
class TrainingOutcome:
    run_directory: Path
    final_model: Path
    status: str
    environment_steps: int
    episodes: int


class TrainingSession:
    def __init__(self, config, condition, seed, run_id, device):
        self.config = config
        self.condition = LearningCondition(condition)
        self.seed = seed
        self.run_id = run_id
        self.device = device
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.policy = DQN(
            (3, 2 * config.map.fov_radius + 1, 2 * config.map.fov_radius + 1),
            config.vector_length,
        ).to(device)
        self.target = DQN(
            (3, 2 * config.map.fov_radius + 1, 2 * config.map.fov_radius + 1),
            config.vector_length,
        ).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=config.training.learning_rate, amsgrad=True
        )
        self.replay = new_replay(config, seed)
        self.initial_weights_sha256 = hashlib.sha256(
            encode({k: v.numpy() for k, v in cpu_state_dict(self.policy).items()})
        ).hexdigest()
        comm = config.communication
        if self.condition == LearningCondition.CLEAN:
            comm = replace(comm, profile="perfect")
        self.transport = Transport(comm)
        self.stage_index = 0
        self.stage_episodes = 0
        self.episodes = 0
        self.generated_steps = 0
        self.delivered_experiences = 0
        self.learner_updates = 0
        self.model_publications = 0
        self.discarded_previous_phase = 0
        self.learning_opportunities = 0
        self.published_version = 0
        self.latest_published_weights = {
            k: v.numpy() for k, v in cpu_state_dict(self.policy).items()
        }
        self.deployed_versions = {}
        self.deployed_times = {}
        self.actor_models = {}
        self.environment = None
        self.observations = None
        self.action_rng = None
        self.episode_reward = 0.0
        self.episode_collisions = 0
        self.qualification = []
        self.elapsed_seconds = 0.0
        self.log_offsets = {}
        self.last_loss = None
        self.delivery_ages = []
        self.pending_validation = False
        self.start_episode()

    def _actor(self, weights):
        # Constructing receivers must not advance the learner's random stream.
        state = torch.get_rng_state()
        model = DQN(
            (3, 2 * self.config.map.fov_radius + 1, 2 * self.config.map.fov_radius + 1),
            self.config.vector_length,
        ).to(self.device)
        torch.set_rng_state(state)
        model.load_state_dict(
            {k: torch.as_tensor(v, device=self.device) for k, v in weights.items()}
        )
        model.eval()
        return model

    def start_episode(self):
        stage = self.config.curriculum[self.stage_index]
        seed = self.seed + self.stage_index * 1_000_000 + self.stage_episodes + 1
        fresh_config = replace(
            self.config,
            communication=replace(
                self.config.communication, profile="perfect", stale_information=()
            ),
        )
        self.environment = ExperimentEnvironment(
            fresh_config,
            ExperimentMode.UAV,
            stage.map_path,
            stage.num_agents,
            250,
            seed,
        )
        self.observations, _ = self.environment.reset(seed)
        self.action_rng = np.random.default_rng(seed)
        self.transport.schedule_episode(seed, stage.num_agents, 250)
        new_agents = set(range(stage.num_agents)) - set(self.deployed_versions)
        if new_agents:
            # New agents join with the last complete published model; existing agents keep their deployments.
            if self.published_version not in self.actor_models:
                self.actor_models[self.published_version] = self._actor(
                    self.latest_published_weights
                )
            for agent in new_agents:
                self.deployed_versions[agent] = self.published_version
                self.deployed_times[agent] = self.transport.now
        self.episode_reward = 0.0
        self.episode_collisions = 0

    def receive(self, packet):
        body = packet.payload
        if packet.kind == "model_relay":
            if body["stage"] != self.stage_index:
                return
            for agent in body["recipients"]:
                self.transport.send(
                    agent, "model", wire=packet.wire, version=packet.version,
                    generated_at=packet.generated_at, route=("access_down",), delayed=False,
                )
        elif packet.kind == "experience":
            self.delivered_experiences += 1
            self.delivery_ages.append(self.transport.now - packet.generated_at)
            if body["stage"] != self.stage_index:
                self.discarded_previous_phase += 1
                return
            self.replay.add(body["transition"], packet.generated_at)
        elif packet.kind == "model":
            agent = packet.agent
            if agent not in self.deployed_versions or body["stage"] != self.stage_index:
                return
            if packet.version <= self.deployed_versions[agent]:
                return
            if packet.version not in self.actor_models:
                self.actor_models[packet.version] = self._actor(body["weights"])
            self.deployed_versions[agent] = packet.version
            self.deployed_times[agent] = packet.generated_at

    def step(self):
        self.delivery_ages = []
        self.transport.advance(self.transport.now, self.receive)
        t = self.config.training
        epsilon = t.epsilon_end + (t.epsilon_start - t.epsilon_end) * math.exp(
            -self.generated_steps / t.epsilon_decay_steps
        )
        versions_before = dict(self.deployed_versions)
        actions = select_actions(
            self.actor_models,
            self.deployed_versions,
            self.observations,
            epsilon,
            self.action_rng,
            self.device,
        )
        result = self.environment.step(actions)
        self.generated_steps += 1
        now = self.generated_steps * self.config.communication.seconds_per_step
        self.transport.advance(now, self.receive)
        done = result.terminated or result.truncated
        upload_delay = self.condition in (
            LearningCondition.UPLINK,
            LearningCondition.COMBINED,
        )
        for agent in self.observations:
            # Encode the complete tuple immediately, before any later movement or update.
            transition = (
                self.observations[agent],
                actions[agent],
                result.rewards[agent],
                result.observations[agent],
                done,
            )
            self.transport.send(
                agent,
                "experience",
                {
                    "transition": transition,
                    "stage": self.stage_index,
                    "joint_step": self.generated_steps,
                    "episode": self.episodes + 1,
                    "deployed_version": versions_before[agent],
                },
                route=("access_up", "satellite_up") if upload_delay else (),
                delayed=upload_delay,
                version=versions_before[agent],
            )
        self.transport.advance(now, self.receive)
        self.learning_opportunities += 1
        self.last_loss = _learn(
            self.policy,
            self.target,
            self.optimizer,
            self.replay,
            t.batch_size,
            t.gamma,
            t.tau,
            self.device,
        )
        if self.last_loss is not None:
            self.learner_updates += 1
        if self.generated_steps % t.model_broadcast_steps == 0:
            self.published_version += 1
            self.model_publications += 1
            weights = {k: v.numpy() for k, v in cpu_state_dict(self.policy).items()}
            self.latest_published_weights = weights
            wire = encode({"weights": weights, "stage": self.stage_index,
                           "recipients": tuple(self.observations)})
            deployment_delay = self.condition in (
                LearningCondition.DOWNLINK,
                LearningCondition.COMBINED,
            )
            if deployment_delay and self.config.communication.model_distribution == "uav_fanout":
                self.transport.send(-1,"model_relay",wire=wire,version=self.published_version,
                                    route=("satellite_down",),delayed=True)
            else:
                for agent in self.observations:
                    self.transport.send(
                        agent, "model", wire=wire, version=self.published_version,
                        route=("satellite_down", "access_down") if deployment_delay else (),
                        delayed=deployment_delay,
                    )
            self.transport.advance(now, self.receive)
        used = set(self.deployed_versions.values())
        self.actor_models = {v: m for v, m in self.actor_models.items() if v in used}
        self.observations = result.observations
        self.episode_reward += sum(result.rewards.values())
        self.episode_collisions += result.info.get(
            "collisions_agent_agent_step", 0
        ) + result.info.get("collisions_obstacle_step", 0)
        replay_age = (
            now - self.replay.last_sample_generated_at
            if self.last_loss is not None
            else np.empty(0)
        )
        record = {
            "generated_joint_steps": self.generated_steps,
            "simulated_seconds": now,
            "stage_index": self.stage_index,
            "num_agents": self.environment.num_agents,
            "episode_step": self.environment.step_count,
            "global_episode": self.episodes + 1,
            "delivered_experiences": self.delivered_experiences,
            "learner_updates": self.learner_updates,
            "learning_opportunities": self.learning_opportunities,
            "model_publications": self.model_publications,
            "published_version": self.published_version,
            "deployed_versions": dict(self.deployed_versions),
            "deployed_model_age_seconds": {
                a: now - at for a, at in self.deployed_times.items()
            },
            "model_version_lag": {
                a: self.published_version - v for a, v in self.deployed_versions.items()
            },
            "experience_delivery_age_seconds": self.delivery_ages,
            "replay_age_seconds": replay_age.tolist(),
            "queue_bytes": self.transport.occupancy(),
            "loss": self.last_loss,
            "epsilon": epsilon,
            "replay_size": len(self.replay),
            "discarded_previous_phase": self.discarded_previous_phase,
            "pending_messages": self.transport.counters["messages_generated"]
            - self.transport.counters["messages_delivered"],
            "positions": result.info["agent_positions"],
            "actions": actions,
            "uav_position": result.info["uav_position"],
            "uav_travel_m": self.environment.uav.total_distance_m,
            "leo_state": self.transport.orbit.state(
                now+self.config.communication.backhaul_time_offset_seconds,
                self.config.communication.orbital_replay,
            ) if self.transport.orbit else None,
        }
        episode = None
        if done:
            self.episodes += 1
            self.stage_episodes += 1
            episode = {
                "stage_index": self.stage_index,
                "stage_name": self.config.curriculum[self.stage_index].name,
                "num_agents": self.environment.num_agents,
                "stage_episode": self.stage_episodes,
                "global_episode": self.episodes,
                "generated_joint_steps": self.generated_steps,
                "steps": self.environment.step_count,
                "success": result.terminated,
                "successful_makespan": self.environment.step_count
                if result.terminated
                else None,
                "par2": self.environment.step_count if result.terminated else 500,
                "completion_fraction": sum(a.at_goal for a in self.environment.agents)
                / self.environment.num_agents,
                "ever_completion_fraction": sum(
                    a.ever_reached_goal for a in self.environment.agents
                )
                / self.environment.num_agents,
                "collision_attempts": self.episode_collisions,
                "reward": self.episode_reward,
                "uav_travel_m": self.environment.uav.total_distance_m,
                "epsilon": epsilon,
                "delivered_experiences": self.delivered_experiences,
                "learner_updates": self.learner_updates,
            }
        return record, episode

    def checkpoint(self):
        return {
            "communication_schema": COMMUNICATION_SCHEMA,
            "config_fingerprint": fingerprint(self.config),
            "session": self,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        }

    @classmethod
    def restore(cls, checkpoint, config, condition, seed, run_id, device):
        if checkpoint.get("communication_schema") != COMMUNICATION_SCHEMA:
            raise ValueError("Legacy checkpoint rejected; use a new run ID")
        if checkpoint["config_fingerprint"] != fingerprint(config):
            raise ValueError("Resume configuration mismatch")
        self = checkpoint["session"]
        if (self.condition.value, self.seed, self.run_id) != (
            LearningCondition(condition).value,
            seed,
            run_id,
        ):
            raise ValueError("Resume identity mismatch")
        self.device = device
        for model in [self.policy, self.target, *self.actor_models.values()]:
            model.to(device)
        for values in self.optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.cpu() if key == "step" else value.to(device)
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if checkpoint["cuda_rng"] and torch.cuda.is_available():
            # map_location=device also moves saved RNG ByteTensors to CUDA;
            # generator state setters require their original CPU byte storage.
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        return self


def validate(
    config,
    stage,
    model,
    seed,
    device,
    deadline=float("inf"),
    log_path=None,
    context=None,
    stop_requested=lambda: False,
):
    successes = 0
    steps = []
    completed = 0
    clean = replace(
        config,
        communication=replace(
            config.communication, profile="perfect", stale_information=()
        ),
    )
    for case in range(200):
        if time.monotonic() >= deadline or stop_requested():
            break
        case_seed = seed + 9_000_000 + case
        env = ExperimentEnvironment(
            clean, ExperimentMode.UAV, stage.map_path, stage.num_agents, 250, case_seed
        )
        observations, _ = env.reset()
        collisions = 0
        while time.monotonic() < deadline and not stop_requested():
            actions = select_actions(
                {0: model},
                {a: 0 for a in observations},
                observations,
                0.0,
                np.random.default_rng(case_seed),
                device,
            )
            result = env.step(actions)
            observations = result.observations
            collisions += result.info.get(
                "collisions_agent_agent_step", 0
            ) + result.info.get("collisions_obstacle_step", 0)
            if result.terminated or result.truncated:
                successes += int(result.terminated)
                completed += 1
                steps.append(env.step_count if result.terminated else 500)
                if log_path:
                    append_jsonl(
                        log_path,
                        {
                            **(context or {}),
                            "seed": case_seed,
                            "num_agents": stage.num_agents,
                            "case": case,
                            "success": result.terminated,
                            "steps": env.step_count,
                            "par2": steps[-1],
                            "completion_fraction": sum(a.at_goal for a in env.agents)
                            / stage.num_agents,
                            "uav_travel_m": env.uav.total_distance_m,
                            "collision_attempts": collisions,
                        },
                    )
                break
        else:
            break
    return {
        "episodes": completed,
        "complete": completed == 200,
        "success_rate": successes / max(completed, 1),
        "par2": float(np.mean(steps)) if steps else None,
        "num_agents": stage.num_agents,
        "validation_policy": "current_learner_greedy",
    }


def train_policy(
    config, condition, seed, run_directory, resume=True, device_name="auto", run_id=None
):
    start = time.monotonic()
    run_directory = Path(run_directory)
    run_id = run_id or run_directory.parent.name
    if not run_id or run_id in ("legacy", "default"):
        raise ValueError("A fresh study run ID is required")
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "checkpoint.pt"
    device = choose_device(device_name)
    torch.set_num_threads(1)
    code_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    dataset_hashes = {
        name: sha256(config.resolve(path))
        for name, path in (
            ("uplink_trace", config.communication.uplink_trace),
            ("downlink_trace", config.communication.downlink_trace),
            ("orbital_timeline", config.communication.leo_timeline),
        )
        if path
    }
    if (run_directory / "manifest.json").exists():
        previous_manifest = json.loads((run_directory / "manifest.json").read_text())
        if previous_manifest["code_sha256"] != code_hashes:
            raise ValueError(
                "Implementation changed since this run; choose a fresh run ID"
            )
        if previous_manifest.get("backhaul_dataset_sha256", {}) != dataset_hashes:
            raise ValueError(
                "Backhaul dataset changed since this run; choose a fresh run ID"
            )
    if checkpoint_path.exists() and not resume:
        raise ValueError("Run already exists; choose a new run ID")
    if checkpoint_path.exists():
        session = TrainingSession.restore(
            torch.load(checkpoint_path, map_location=device, weights_only=False),
            config,
            condition,
            seed,
            run_id,
            device,
        )
        for path in run_directory.glob("*.jsonl"):
            with path.open("r+b") as stream:
                stream.truncate(session.log_offsets.get(path.name, 0))
        if (run_directory / "status.json").exists():
            previous_status = json.loads((run_directory / "status.json").read_text())
            if previous_status["generated_joint_steps"] == session.generated_steps:
                session.elapsed_seconds = max(
                    session.elapsed_seconds, previous_status["wall_seconds"]
                )
    else:
        if any(run_directory.iterdir()):
            raise ValueError(
                "Run directory contains prior artifacts; choose a new run ID"
            )
        session = TrainingSession(config, condition, seed, run_id, device)
    initial_elapsed = session.elapsed_seconds
    deadline = start + max(
        0.0,
        config.training.budget_seconds
        - initial_elapsed
        - config.training.shutdown_reserve_seconds,
    )
    trace_metadata = {
        name: trace.metadata for name, trace in session.transport.traces.items()
    }
    write_json(
        run_directory / "manifest.json",
        {
            "communication_schema": COMMUNICATION_SCHEMA,
            "run_id": run_id,
            "condition": LearningCondition(condition).value,
            "seed": seed,
            "config": config.to_dict(),
            "map_sha256": sha256(config.resolve(FIXED_MAP)),
            "initial_weights_sha256": session.initial_weights_sha256,
            "code_sha256": code_hashes,
            "imported_traces": trace_metadata,
            "backhaul_dataset_sha256": dataset_hashes,
            "orbital_metadata": session.transport.orbit.metadata
            if session.transport.orbit
            else None,
            "software": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "simpy": __import__("simpy").__version__,
            },
            "learner_location": "cloud",
            "mission_information": "fresh",
            "model_transmission": config.communication.model_distribution,
            "budget_includes": "initialization, training, validation, checkpoints",
            "results": "measured only",
            "scheduler": "SimPy calendar; FIFO per agent, round-robin service chunks; framed satellite airtime",
            "archived_orbit_replay": config.communication.orbital_replay,
        },
    )

    def flush_events():
        for event in session.transport.log:
            append_jsonl(run_directory / "transport.jsonl", event)
        session.transport.log.clear()
        if session.environment:
            session.environment.communication.transport.log.clear()

    def save():
        flush_events()
        session.elapsed_seconds = initial_elapsed + time.monotonic() - start
        session.log_offsets = {
            p.name: p.stat().st_size for p in run_directory.glob("*.jsonl")
        }
        atomic_save(session.checkpoint(), checkpoint_path)

    interruption = {"requested": False}

    def request_stop(_signal, _frame):
        interruption["requested"] = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)

    def perform_validation():
        stage = config.curriculum[session.stage_index]
        context = {
            "stage_index": session.stage_index,
            "stage_episode": session.stage_episodes,
            "generated_joint_steps": session.generated_steps,
        }
        validation = validate(
            config,
            stage,
            session.policy,
            seed + session.stage_index * 100_000,
            device,
            deadline,
            run_directory / "validation_cases.jsonl",
            context,
            lambda: interruption["requested"],
        )
        validation.update(context)
        validation["wall_seconds"] = initial_elapsed + time.monotonic() - start
        append_jsonl(run_directory / "validation.jsonl", validation)
        session.pending_validation = not validation["complete"]
        if qualifies(stage, session.stage_episodes, validation):
            session.qualification.append(dict(validation))
            atomic_save(
                model_payload(session),
                run_directory / f"stage_{session.stage_index}_model.pt",
            )
            session.stage_index += 1
            session.stage_episodes = 0
            session.replay = new_replay(config, seed + session.stage_index)

    reason = (
        "qualified"
        if session.stage_index == len(config.curriculum)
        else "budget_exhausted"
    )
    progress = tqdm(
        total=max(
            0.0,
            config.training.budget_seconds - config.training.shutdown_reserve_seconds,
        ),
        initial=initial_elapsed,
        desc=f"train {session.condition.value} (wall budget)",
        unit="s",
        bar_format="{desc}: {elapsed} elapsed | {remaining} remaining | {postfix}",
        disable=None,
    )
    try:
        while (
            session.stage_index < len(config.curriculum)
            and time.monotonic() < deadline
            and not interruption["requested"]
        ):
            if session.pending_validation:
                perform_validation()
                if session.stage_index == len(config.curriculum):
                    reason = "qualified"
                    break
                if time.monotonic() >= deadline:
                    break
            if (
                session.stage_episodes
                >= config.curriculum[session.stage_index].max_episodes
            ):
                reason = "phase_episode_budget_exhausted"
                break
            if session.generated_steps >= config.training.target_environment_steps:
                reason = "generated_step_budget_exhausted"
                break
            if (
                session.environment.all_agents_at_goals
                or session.environment.step_count >= 250
            ):
                session.start_episode()
            record, episode = session.step()
            progress.update(
                max(0.0, initial_elapsed + time.monotonic() - start - progress.n)
            )
            if session.generated_steps % 64 == 0:
                progress.set_postfix(
                    phase=session.stage_index,
                    episodes=session.stage_episodes,
                    generated=session.generated_steps,
                    received=session.delivered_experiences,
                    remaining_s=max(0, int(deadline - time.monotonic())),
                )
            record["wall_seconds"] = initial_elapsed + time.monotonic() - start
            append_jsonl(run_directory / "learning_steps.jsonl", record)
            flush_events()
            if episode:
                episode["wall_seconds"] = record["wall_seconds"]
                append_jsonl(run_directory / "training.jsonl", episode)
                stage = config.curriculum[session.stage_index]
                if (
                    session.stage_episodes >= stage.min_episodes
                    and session.stage_episodes % 250 == 0
                ):
                    session.pending_validation = True
                if (
                    session.stage_episodes >= stage.max_episodes
                    and not session.pending_validation
                ):
                    reason = "phase_episode_budget_exhausted"
                    break
            if session.generated_steps % config.training.checkpoint_interval_steps == 0:
                save()
        if interruption["requested"]:
            reason = "interrupted_partial"
    except KeyboardInterrupt:
        reason = "interrupted_partial"
    except Exception:
        reason = "error_partial"
        raise
    finally:
        progress.close()
        save()
        final_model = run_directory / "final_model.pt"
        atomic_save(model_payload(session), final_model)
        session.elapsed_seconds = initial_elapsed + time.monotonic() - start
        status = {
            "status": reason,
            "qualified": session.stage_index == len(config.curriculum),
            "phase": session.stage_index,
            "stage_episodes": session.stage_episodes,
            "episodes": session.episodes,
            "generated_joint_steps": session.generated_steps,
            "delivered_experiences": session.delivered_experiences,
            "learner_updates": session.learner_updates,
            "model_publications": session.model_publications,
            "wall_seconds": session.elapsed_seconds,
            "budget_seconds": config.training.budget_seconds,
            "pending_messages": session.transport.counters["messages_generated"]
            - session.transport.counters["messages_delivered"],
            "qualification": session.qualification,
        }
        write_json(run_directory / "status.json", status)
        # Account for final artifacts in the persisted wall budget as well.
        save()
        for name in ("training", "validation"):
            export_csv(run_directory / f"{name}.jsonl", run_directory / f"{name}.csv")
        status["wall_seconds"] = initial_elapsed + time.monotonic() - start
        write_json(run_directory / "status.json", status)
        signal.signal(signal.SIGINT, previous_sigint)
    return TrainingOutcome(
        run_directory, final_model, reason, session.generated_steps, session.episodes
    )


def compare_training(config, seed, root, device_name="auto"):
    outcomes = []
    for condition in LearningCondition:
        outcome = train_policy(
            config,
            condition,
            seed,
            Path(root) / condition.value,
            device_name=device_name,
            run_id=Path(root).name,
        )
        outcomes.append(outcome)
        if condition == LearningCondition.CLEAN and outcome.status != "qualified":
            write_json(
                Path(root) / "comparison_status.json",
                {
                    "status": "mission_comparison_blocked_by_unqualified_clean_baseline",
                    "learning_comparison": "continues with explicitly partial phase outcomes",
                    "mission_comparison_launched": False,
                },
            )
    return outcomes
