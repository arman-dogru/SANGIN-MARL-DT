from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .config import ExperimentMode, ProjectConfig
from .information import MissionCommunication
from .radio import RadioSnapshot
from .scenarios import Scenario, load_map
from .uav import UAVController

Position = tuple[int, int]


def manhattan(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def astar_path(grid: np.ndarray, start: Position, goal: Position) -> list[Position]:
    if start == goal:
        return [start]
    height, width = grid.shape
    frontier: list[tuple[int, int, Position]] = [(manhattan(start, goal), 0, start)]
    came_from: dict[Position, Position] = {}
    costs: dict[Position, int] = {start: 0}
    counter = 0
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            path = [current]
            while current != start:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))
        next_cost = costs[current] + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width):
                continue
            if grid[neighbor] != 0 or next_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = next_cost
            came_from[neighbor] = current
            counter += 1
            heapq.heappush(
                frontier, (next_cost + manhattan(neighbor, goal), counter, neighbor)
            )
    return []


@dataclass
class Agent:
    id: int
    position: Position
    goal: Position
    priority: float = math.inf
    path: list[Position] | None = None
    distance_to_goal: float = math.inf
    previous_distance_to_goal: float = math.inf
    initial_distance_to_goal: float = math.inf
    ever_reached_goal: bool = False
    first_completion_time: float | None = None

    @property
    def at_goal(self) -> bool:
        return self.position == self.goal


@dataclass(frozen=True)
class StepResult:
    observations: dict[int, dict[str, np.ndarray]]
    rewards: dict[int, float]
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class ExperimentEnvironment:
    """Retained 0.1 task dynamics with separate received mission information."""

    ACTION_DELTAS: ClassVar[dict[int, Position]] = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
        4: (0, 0),
    }

    def __init__(
        self,
        config: ProjectConfig,
        mode: ExperimentMode,
        map_path: str | Path,
        num_agents: int,
        max_episode_steps: int,
        seed: int,
        scenario: Scenario | None = None,
    ) -> None:
        self.config = config
        self.mode = mode
        self.map_path = config.resolve(map_path)
        self.grid = load_map(self.map_path)
        self.height, self.width = self.grid.shape
        self.num_agents = num_agents
        self.max_episode_steps = max_episode_steps
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.scenario = scenario.first(num_agents) if scenario is not None else None
        self.fov_radius = config.map.fov_radius
        self.fov_size = 2 * self.fov_radius + 1
        self.communication_range = int(
            self.fov_radius * config.map.communication_range_factor
        )
        self.padded_grid = np.pad(self.grid, self.fov_radius, constant_values=1)
        self.agents: list[Agent] = []
        self.agent_by_id: dict[int, Agent] = {}
        self.communicated_paths: dict[int, list[Position]] = {}
        self.step_count = 0
        self.radio_snapshot: RadioSnapshot | None = None
        self.uav: UAVController | None = None
        if max_episode_steps > 250:
            raise ValueError("Study episodes cannot exceed 250 steps")
        self.uav = UAVController(
            self.grid.shape, config.map.meters_per_cell, config.uav
        )
        self.communication = None
        self.remote_inference = None
        self.remote_shadow_inference = None
        self.model_version = 0
        self.failure_reason = None

    @property
    def observation_shape(self) -> tuple[tuple[int, int, int], int]:
        return (
            self.mode.spatial_channels,
            self.fov_size,
            self.fov_size,
        ), self.config.vector_length

    @property
    def all_agents_at_goals(self) -> bool:
        return bool(self.agents) and all(agent.at_goal for agent in self.agents)

    def _valid(self, position: Position) -> bool:
        row, col = position
        return (
            0 <= row < self.height
            and 0 <= col < self.width
            and self.grid[position] == 0
        )

    def _refresh_path(self, agent: Agent) -> None:
        path = astar_path(self.grid, agent.position, agent.goal)
        agent.path = path
        agent.distance_to_goal = float(len(path) - 1) if path else math.inf
        self.communicated_paths[agent.id] = path[
            : self.config.map.communicated_path_steps + 1
        ]

    def _refresh_priorities(self) -> None:
        settled = float(self.width * self.height)
        for agent in self.agents:
            if agent.ever_reached_goal:
                agent.priority = settled + agent.id * 1e-6
            else:
                distance = agent.distance_to_goal if math.isfinite(agent.distance_to_goal) else settled-1
                agent.priority = distance + agent.id * 1e-6

    def _random_starts_and_goals(self) -> tuple[list[Position], list[Position]]:
        cells = [
            tuple(int(value) for value in cell) for cell in np.argwhere(self.grid == 0)
        ]
        self.rng.shuffle(cells)
        if len(cells) < self.num_agents:
            raise RuntimeError("Not enough traversable cells for the requested agents")
        starts: list[Position] = []
        goals: list[Position] = []
        for _ in range(self.num_agents // 2):
            first, second = cells.pop(), cells.pop()
            starts.extend((first, second))
            goals.extend((second, first))
        if self.num_agents % 2:
            starts.append(cells.pop())
            goals.append(cells.pop())
        return starts, goals

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.failure_reason = None
        if self.scenario is None:
            starts, goals = self._random_starts_and_goals()
        else:
            starts, goals = list(self.scenario.starts), list(self.scenario.goals)
        if len(set(starts)) != len(starts):
            raise ValueError("Scenario contains duplicate start locations")
        self.agents = [
            Agent(index, starts[index], goals[index])
            for index in range(self.num_agents)
        ]
        self.agent_by_id = {agent.id: agent for agent in self.agents}
        self.communicated_paths = {}
        for agent in self.agents:
            self._refresh_path(agent)
            agent.previous_distance_to_goal = agent.distance_to_goal
            agent.initial_distance_to_goal = agent.distance_to_goal
            if agent.at_goal:
                agent.ever_reached_goal = True
                agent.first_completion_time = 0.0
        self._refresh_priorities()
        if self.uav is not None:
            self.uav.reset()
            self.radio_snapshot = self.uav.snapshot()
        if self.config.resource_allocation.enabled:
            if self.config.resource_allocation.execution_mode == "remote_dt":
                from .remote import RemoteDTCommunication
                self.communication = RemoteDTCommunication(self.config, self.grid, self.agents,
                    astar_path, self.base_seed if seed is None else seed, self.max_episode_steps,
                    self.uav, self.remote_inference, self.model_version)
                self.communication.shadow_inference = self.remote_shadow_inference
                self.communication.model_version = self.model_version
                self.communication.update(self.agents, 0.0)
            else:
                from .paper import PaperMissionCommunication
                self.communication = PaperMissionCommunication(self.config, self.grid, self.agents,
                    astar_path, self.base_seed if seed is None else seed, self.max_episode_steps, self.uav)
        else:
            self.communication = MissionCommunication(
                self.config.communication, self.grid, self.agents, astar_path,
                self.base_seed if seed is None else seed, self.max_episode_steps)
        finite = [
            agent.initial_distance_to_goal
            for agent in self.agents
            if math.isfinite(agent.initial_distance_to_goal)
        ]
        return self.observations(), {
            "max_initial_astar_distance": max(finite, default=0.0)
        }

    def observation(self, agent: Agent) -> dict[str, np.ndarray]:
        path, received_distance, received_intentions, received_priorities = (
            self.communication.observation_values(agent)
        )
        row, col = agent.position
        obstacles = self.padded_grid[
            row : row + self.fov_size, col : col + self.fov_size
        ].astype(np.float32)
        others = np.zeros((self.fov_size, self.fov_size), dtype=np.float32)
        intentions = np.zeros_like(others)
        for other in self.agents:
            if other.id == agent.id:
                continue
            delta_row, delta_col = other.position[0] - row, other.position[1] - col
            if abs(delta_row) <= self.fov_radius and abs(delta_col) <= self.fov_radius:
                others[delta_row + self.fov_radius, delta_col + self.fov_radius] = 1.0
            if (
                not received_intentions[other.id]["completed"]
                and received_priorities[other.id] < received_priorities[agent.id]
                and (self.config.resource_allocation.enabled or
                     manhattan(agent.position, other.position) <= self.communication_range)
            ):
                for path_row, path_col in received_intentions[other.id]["path"]:
                    rel_row, rel_col = path_row - row, path_col - col
                    if (
                        abs(rel_row) <= self.fov_radius
                        and abs(rel_col) <= self.fov_radius
                    ):
                        intentions[
                            rel_row + self.fov_radius, rel_col + self.fov_radius
                        ] = 1.0
        channels = [obstacles, others, intentions]
        normalizer = (
            agent.initial_distance_to_goal
            if agent.initial_distance_to_goal > 0
            and math.isfinite(agent.initial_distance_to_goal)
            else max(self.width, self.height)
        )
        normalized_distance = np.clip(
            received_distance / normalizer if math.isfinite(received_distance) else 1.0,
            0.0,
            1.0,
        )
        completed = (self.communication.states[agent.id][agent.id].value["completed"]
                     if self.config.resource_allocation.enabled else agent.ever_reached_goal)
        vector = [float(normalized_distance), float(completed)]
        waypoint_normalizer = float(max(self.fov_radius, 1))
        for offset in range(1, self.config.map.astar_waypoints + 1):
            if offset < len(path):
                waypoint = path[offset]
                vector.extend(
                    (
                        float(
                            np.clip(
                                (waypoint[0] - row) / waypoint_normalizer, -1.0, 1.0
                            )
                        ),
                        float(
                            np.clip(
                                (waypoint[1] - col) / waypoint_normalizer, -1.0, 1.0
                            )
                        ),
                    )
                )
            else:
                vector.extend((0.0, 0.0))
        return {
            "spatial": np.stack(channels, axis=0).astype(np.float32),
            "vector": np.asarray(vector, dtype=np.float32),
        }

    def observations(self) -> dict[int, dict[str, np.ndarray]]:
        return {agent.id: self.observation(agent) for agent in self.agents}

    def prepare_decision(self, deadline_seconds):
        if not 0 <= deadline_seconds < self.config.communication.seconds_per_step:
            raise ValueError("Decision deadline must fit inside one robot control cycle")
        self.communication.prepare_decision(
            self.agents, self.step_count*self.config.communication.seconds_per_step, deadline_seconds
        )
        return self.observations()

    def _resolve_positions(
        self,
        actions: dict[int, int],
        rewards: dict[int, float],
        info: defaultdict[str, float],
    ) -> tuple[dict[int, Position], set[int]]:
        before = {agent.id: agent.position for agent in self.agents}
        intended: dict[int, Position] = {}
        for agent in self.agents:
            delta = self.ACTION_DELTAS.get(actions.get(agent.id, 4), (0, 0))
            target = (agent.position[0] + delta[0], agent.position[1] + delta[1])
            if self._valid(target):
                intended[agent.id] = target
            else:
                intended[agent.id] = agent.position
                rewards[agent.id] += self.config.reward.collision_obstacle
                info["collisions_obstacle_step"] += 1
        final = dict(intended)
        collided: set[int] = set()
        while True:
            conflicts: defaultdict[Position, list[int]] = defaultdict(list)
            for agent_id, target in final.items():
                conflicts[target].append(agent_id)
            vertex = next(
                ((target, ids) for target, ids in conflicts.items() if len(ids) > 1),
                None,
            )
            if vertex is not None:
                _, ids = vertex
                stayers = [
                    agent_id for agent_id in ids if final[agent_id] == before[agent_id]
                ]
                candidates = stayers or ids
                winner = min(
                    candidates, key=lambda agent_id: self.agent_by_id[agent_id].priority
                )
                for agent_id in ids:
                    if agent_id != winner:
                        final[agent_id] = before[agent_id]
                collided.update(ids)
                continue
            swap: tuple[int, int] | None = None
            ids = list(final)
            for index, first in enumerate(ids):
                for second in ids[index + 1 :]:
                    if (
                        final[first] == before[second]
                        and final[second] == before[first]
                    ):
                        swap = (first, second)
                        break
                if swap:
                    break
            if swap:
                final[swap[0]], final[swap[1]] = before[swap[0]], before[swap[1]]
                collided.update(swap)
                continue
            break
        return final, collided

    def step(self, actions: dict[int, int]) -> StepResult:
        if not self.agents:
            raise RuntimeError("reset() must be called before step()")
        if self.step_count >= self.max_episode_steps:
            raise RuntimeError("Episode cap reached; reset before stepping again")
        if self.failure_reason:
            raise RuntimeError("Episode terminated after a physical collision")
        remote = (self.config.resource_allocation.enabled
                  and self.config.resource_allocation.execution_mode == "remote_dt")
        if remote:
            if not self.communication.command_contexts or any(
                c["cycle"] != self.step_count for c in self.communication.command_contexts.values()
            ):
                raise RuntimeError("Prepare the current remote decision before robot motion")
            if actions != self.communication.execution_actions:
                raise ValueError("Remote motion must use the commands delivered by this cycle's cutoff")
        self._refresh_priorities()
        self.step_count += 1
        reward_cfg = self.config.reward
        rewards = {agent.id: reward_cfg.time_step for agent in self.agents}
        info: defaultdict[str, Any] = defaultdict(float)
        before = {agent.id: agent.position for agent in self.agents}
        path_occupancy_before: defaultdict[Position, list[Agent]] = defaultdict(list)
        for other in self.agents:
            if other.ever_reached_goal:
                continue
            for position in self.communicated_paths.get(other.id, [])[1:]:
                path_occupancy_before[position].append(other)
        if remote:
            from .remote_control import conflicts, resolve_delivered
            intended = {}
            for agent in self.agents:
                dr, dc = self.ACTION_DELTAS[actions[agent.id]]
                target = (agent.position[0]+dr, agent.position[1]+dc)
                if not self._valid(target):
                    target = agent.position
                    rewards[agent.id] += reward_cfg.collision_obstacle
                    info["collisions_obstacle_step"] += 1
                intended[agent.id] = target
            physical_conflicts = conflicts(before, intended)
            collided = {r for event in physical_conflicts for r in event["robots"]}
            if physical_conflicts:
                self.communication._record("physical_collision" if self.config.resource_allocation.on_collision == "terminate"
                    else "physical_conflict", cycle=self.step_count-1,
                    conflicts=physical_conflicts, before=before, intended=intended,
                    command_contexts=self.communication.command_contexts,
                    response=self.config.resource_allocation.on_collision)
            if collided and self.config.resource_allocation.on_collision == "terminate":
                self.failure_reason = "robot_collision"
                final = before.copy()
            else:
                priorities = {a.id: self.communication.command_contexts[a.id]["resolution_priority"]
                              for a in self.agents}
                final = resolve_delivered(before, intended, priorities)
            info["actual_agent_overlap_after_resolution"] = len(conflicts(before, final))
            info["physical_conflicts"] = physical_conflicts
            info["failure_reason"] = self.failure_reason
        else:
            final, collided = self._resolve_positions(actions, rewards, info)
        info["collisions_agent_agent_step"] = len(collided)
        for agent_id in collided:
            rewards[agent_id] += reward_cfg.collision_agent
        for agent in self.agents:
            agent.position = final[agent.id]
            agent.previous_distance_to_goal = agent.distance_to_goal
            self._refresh_path(agent)

        path_occupancy_after: defaultdict[Position, list[int]] = defaultdict(list)
        for other in self.agents:
            for position in self.communicated_paths.get(other.id, [])[1:]:
                path_occupancy_after[position].append(other.id)
        arrivals: list[int] = []
        for agent in self.agents:
            if (
                math.isfinite(agent.previous_distance_to_goal)
                and math.isfinite(agent.distance_to_goal)
                and not agent.at_goal
            ):
                rewards[agent.id] += (
                    agent.previous_distance_to_goal - agent.distance_to_goal
                ) * reward_cfg.distance_improvement
            occupied_before = path_occupancy_before.get(before[agent.id], [])
            was_on_priority_path = any(
                other.priority < agent.priority for other in occupied_before
            )
            occupied_after_move = path_occupancy_before.get(agent.position, [])
            if (
                was_on_priority_path
                and before[agent.id] != agent.position
                and not any(
                    other.priority < agent.priority for other in occupied_after_move
                )
            ):
                rewards[agent.id] += reward_cfg.active_yield
                info["active_yield_events_step"] += 1
            blocking = any(
                self.agent_by_id[other_id].priority < agent.priority
                and manhattan(agent.position, self.agent_by_id[other_id].position)
                <= self.communication_range
                for other_id in path_occupancy_after.get(agent.position, [])
                if other_id != agent.id
            )
            if agent.at_goal:
                if not agent.ever_reached_goal:
                    agent.ever_reached_goal = True
                    agent.first_completion_time = (self.communication.transport.now if remote else
                        self.step_count*self.config.communication.seconds_per_step)
                    rewards[agent.id] += reward_cfg.goal
                    arrivals.append(self.step_count)
                    info["targets_reached_step"] += 1
                if blocking:
                    rewards[agent.id] += reward_cfg.path_conflict
                    info["path_conflict_penalty_step"] += 1
                else:
                    rewards[agent.id] += reward_cfg.hold
            elif blocking:
                rewards[agent.id] += reward_cfg.path_conflict
                info["path_conflict_penalty_step"] += 1

        self._refresh_priorities()
        if remote:
            self.communication._record("motion_event", cycle=self.step_count-1, actions=actions,
                before=before, after=final, physical={a.id: self.communication._report(a) for a in self.agents},
                first_completion_times={a.id: a.first_completion_time for a in self.agents},
                failure_reason=self.failure_reason)
            self.communication.physical_time = self.communication.transport.now
        if remote:
            self.communication.update(self.agents, self.step_count*self.config.communication.seconds_per_step,
                generate_tasks=not (self.all_agents_at_goals or self.failure_reason or self.step_count >= self.max_episode_steps))
        else:
            self.communication.update(self.agents, self.step_count*self.config.communication.seconds_per_step)
        # UAV positioning sees only successfully received access reports.
        positions = [
            tuple(self.communication.reports[a.id]["position"]) for a in self.agents
        ]
        uav_diagnostics = self.uav.update(self.uav.positions_to_xy_m(positions))
        self.radio_snapshot = self.uav.snapshot()
        info.update(
            {
                "goal_reached_steps_list": arrivals,
                "agents_at_goal_step": int(sum(agent.at_goal for agent in self.agents)),
                "agent_positions": [list(agent.position) for agent in self.agents],
                "uav_distance_moved_step": uav_diagnostics.distance_moved,
                "uav_position": self.uav.position.astype(float).tolist(),
                **self.communication.snapshot(),
            }
        )
        terminated = self.all_agents_at_goals or self.failure_reason is not None
        truncated = self.step_count >= self.max_episode_steps and not terminated
        return StepResult(
            self.observations(), rewards, terminated, truncated, dict(info)
        )
