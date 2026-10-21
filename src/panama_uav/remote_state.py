"""Causal report, replica and command contracts for remote PANAMA execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import numpy as np

from .information import InformationCache

REMOTE_SCHEMA = "remote_dt_v1"


def source_min(sources):
    if not sources or not all(math.isfinite(float(t)) for t in sources.values()):
        raise ValueError("Dynamic information requires finite per-source timestamps")
    return min(map(float, sources.values()))


@dataclass(frozen=True)
class PhysicalReport:
    robot_id: int
    cycle: int
    sensed_at: float
    position: tuple
    goal: tuple
    completed: bool
    initial_distance: float
    obstacles: list
    occupancy: list
    report_id: str

    def __post_init__(self):
        if self.cycle < 0 or not math.isfinite(self.sensed_at) or self.sensed_at < 0:
            raise ValueError("Invalid report epoch")
        if np.shape(self.obstacles) != np.shape(self.occupancy) or len(np.shape(self.obstacles)) != 2:
            raise ValueError("Sensing channels must have equal two-dimensional shapes")

    def payload(self):
        return asdict(self)


@dataclass(frozen=True)
class DTState:
    robot_id: int
    position: tuple
    goal: tuple
    path: list
    priority: float
    completed: bool
    generated_at: float
    version: int
    sources: dict
    report_id: str

    def __post_init__(self):
        source_min(self.sources)
        if (self.version < 0 or not math.isfinite(self.priority) or not math.isfinite(self.generated_at)
            or self.generated_at < 0 or max(map(float, self.sources.values())) > self.generated_at+1e-12):
            raise ValueError("DT timestamp causality or priority violated")

    @property
    def source_time(self):
        return source_min(self.sources)

    def payload(self):
        return asdict(self)


@dataclass
class DTReplica:
    location: str
    states: dict[int, InformationCache] = field(default_factory=dict)
    state_ids: dict[int, str] = field(default_factory=dict)

    def accept(self, state: DTState, available_at, event_id):
        if self.location not in ("drone", "cloud"):
            raise ValueError("Unknown DT location")
        if state.source_time > state.generated_at+1e-12 or state.generated_at > available_at+1e-12:
            raise ValueError("DT timestamp causality violated")
        cache = self.states.setdefault(state.robot_id, InformationCache())
        if cache.source_timestamps and state.source_time < source_min(cache.source_timestamps):
            return False
        accepted = cache.accept(state.generated_at, state.payload(), state.version, dict(state.sources))
        if accepted:
            self.state_ids[state.robot_id] = event_id
        return accepted


@dataclass(frozen=True)
class ActionCommand:
    robot_id: int
    cycle: int
    action: int
    generated_at: float
    expires_at: float
    version: int
    location: str
    sources: dict
    job_id: str
    observation_hash: str
    dt_versions: dict
    dt_priorities: dict

    def __post_init__(self):
        if self.action not in range(5) or self.location not in ("drone", "cloud"):
            raise ValueError("Invalid remote movement command")
        if self.cycle < 0 or self.version < 0 or not all(math.isfinite(t) and t >= 0
                for t in (self.generated_at, self.expires_at)):
            raise ValueError("Invalid command epoch or timestamp")
        source_min(self.sources)
        if max(map(float, self.sources.values())) > self.generated_at+1e-12:
            raise ValueError("Command generated before its dynamic sources")

    def payload(self):
        return asdict(self)


class CommandInbox:
    """Delivered commands only; same-cycle validity and no movement replay."""
    def __init__(self):
        self.commands = {}
        self.closed_cycle = -1

    def accept(self, command, arrived_at):
        if command.generated_at > arrived_at+1e-12:
            raise ValueError("Command arrived before generation")
        if command.cycle <= self.closed_cycle or arrived_at > command.expires_at+1e-12:
            return False, "rejected_obsolete"
        previous = self.commands.get(command.cycle)
        if previous and (command.generated_at, command.version) <= (
            previous[0].generated_at, previous[0].version
        ):
            return False, "rejected_order"
        self.commands[command.cycle] = command, float(arrived_at)
        return True, "accepted"

    def decision(self, cycle, cutoff):
        if cycle <= self.closed_cycle:
            raise ValueError("Decision epoch already closed")
        candidate = self.commands.get(cycle)
        self.closed_cycle = cycle
        self.commands = {n: c for n, c in self.commands.items() if n > cycle}
        if candidate and candidate[1] <= cutoff+1e-12 and cutoff <= candidate[0].expires_at+1e-12:
            return candidate
        return None


def remote_observation(report, own_state, replica, radius, waypoints, horizon=5, map_extent=32):
    """Same network shape, using uploaded sensing and delivered replica state only."""
    side = 2*radius+1
    obstacles = np.asarray(report.obstacles, dtype=np.float32)
    occupancy = np.asarray(report.occupancy, dtype=np.float32)
    if obstacles.shape != (side, side) or occupancy.shape != (side, side):
        raise ValueError("Report FoV differs from the frozen network input")
    intentions = np.zeros((side, side), dtype=np.float32)
    row, col = report.position
    dependencies = {report.robot_id: report.sensed_at}
    versions, priorities = {}, {}
    for peer, cache in sorted(replica.states.items()):
        state = own_state.payload() if peer == report.robot_id else cache.value
        versions[peer] = dict(version=state["version"], generated_at=state["generated_at"],
            state_event=replica.state_ids.get(peer), source_min=source_min(state["sources"]))
        priorities[peer] = state["priority"]
        # Even a suppressed peer intention depends on the compared priorities.
        dependencies.update({int(k): min(dependencies.get(int(k), float(t)), float(t))
                             for k, t in state["sources"].items()})
        if peer == report.robot_id or state["completed"] or state["priority"] >= own_state.priority:
            continue
        for point in state["path"][:horizon+1]:
            dr, dc = point[0]-row, point[1]-col
            if abs(dr) <= radius and abs(dc) <= radius:
                intentions[dr+radius, dc+radius] = 1
    normalizer = report.initial_distance if report.initial_distance > 0 and math.isfinite(report.initial_distance) else map_extent
    distance = (len(own_state.path)-1 if own_state.path and (
        len(own_state.path) > 1 or tuple(report.position) == tuple(report.goal)) else math.inf)
    vector = [float(np.clip(distance/normalizer, 0, 1)), float(report.completed)]
    path = own_state.path
    for k in range(1, waypoints+1):
        point = path[k] if k < len(path) else (row, col)
        vector.extend((float(np.clip((point[0]-row)/max(1, radius), -1, 1)),
                       float(np.clip((point[1]-col)/max(1, radius), -1, 1))))
    observation = dict(spatial=np.stack((obstacles, occupancy, intentions)), vector=np.asarray(vector, np.float32))
    return observation, dependencies, versions, priorities


def observation_hash(observation):
    # Canonical arrays make replay independent of JSON whitespace and host endianness.
    digest = hashlib.sha256()
    for name in ("spatial", "vector"):
        array = np.asarray(observation[name], dtype="<f4")
        digest.update(name.encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
