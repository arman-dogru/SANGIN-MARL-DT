from __future__ import annotations

import math
from dataclasses import dataclass

from .transport import Transport


@dataclass
class InformationCache:
    timestamp: float = -math.inf
    value: object = None
    version: int = -1
    source_timestamps: dict | None = None

    def accept(self, timestamp, value, version=0, source_timestamps=None):
        if timestamp < self.timestamp or (
            timestamp == self.timestamp and version <= self.version
        ):
            return False
        self.timestamp = timestamp
        self.value = value
        self.version = version
        self.source_timestamps = source_timestamps
        return True

    def age(self, now):
        oldest = (
            min(self.source_timestamps.values())
            if self.source_timestamps
            else self.timestamp
        )
        return max(0.0, now - oldest)


@dataclass
class Guidance:
    route: list
    reported_distance: float

    def consume(self, position):
        position = tuple(position)
        if position in self.route:
            index = self.route.index(position)
            self.route = self.route[index + 1 :]
        # Remaining absolute waypoints are interpreted from fresh self-position.
        return [position] + self.route


class MissionCommunication:
    def __init__(self, config, grid, agents, pathfinder, seed, cap):
        self.config = config
        self.transport = Transport(config)
        self.transport.schedule_episode(seed, len(agents), cap)
        self.grid = grid
        self.pathfinder = pathfinder
        self.goals = {a.id: a.goal for a in agents}
        self.reports = {}
        self.report_timestamps = {}
        self.caches = {
            a.id: {
                key: InformationCache() for key in ("task", "intentions", "priorities")
            }
            for a in agents
        }
        self.sequence = 0
        self.decision = None
        self.position_history = {a.id: [(0.0, tuple(a.position))] for a in agents}
        # Initial mission briefing is fresh and explicit, before the clock starts.
        for a in agents:
            self.reports[a.id] = self._report(a)
            self.report_timestamps[a.id] = 0.0
        for a in agents:
            for key in self.caches[a.id]:
                self.caches[a.id][key].accept(
                    0.0, self._value(key, a.id, self.reports), 0
                )

    def _report(self, a):
        return {
            "position": tuple(a.position),
            "goal": tuple(a.goal),
            "path": list(a.path or []),
            "distance": a.distance_to_goal,
            "priority": a.priority,
            "completed": a.ever_reached_goal,
        }

    def _value(self, key, receiver, reports):
        if key == "task":
            r = reports[receiver]
            return Guidance(list(r["path"]), r["distance"])
        if key == "intentions":
            return {
                peer: {"path": list(r["path"][:6]), "completed": r["completed"]}
                for peer, r in reports.items()
            }
        return {peer: r["priority"] for peer, r in reports.items()}

    def _receive(self, packet):
        payload = packet.payload
        if packet.kind == "report":
            if packet.generated_at < self.report_timestamps[packet.agent]:
                return
            # Remote processing uses the reported state, never the agent's current truth.
            path = self.pathfinder(
                self.grid, tuple(payload["position"]), tuple(payload["goal"])
            )
            payload["path"] = path
            payload["distance"] = float(len(path) - 1) if path else math.inf
            self.reports[packet.agent] = payload
            self.report_timestamps[packet.agent] = packet.generated_at
        elif packet.kind == "mission_request":
            field = payload["field"]
            reports = {peer: dict(report) for peer, report in self.reports.items()}
            reports[packet.agent] = payload["report"]
            sources = dict(self.report_timestamps)
            sources[packet.agent] = packet.generated_at
            route = (f"process:{self.config.edge_processing_seconds}",)
            if self.config.mission_route == "cloud":
                route = (
                    "satellite_up",
                    f"process:{self.config.cloud_processing_seconds}",
                )
            self.transport.send(
                packet.agent,
                "mission_compute",
                {"field": field, "reports": reports, "sources": sources},
                route=route,
                generated_at=packet.generated_at,
                version=packet.version,
                delayed=False,
            )
        elif packet.kind == "mission_compute":
            field = payload["field"]
            reports = payload["reports"]
            report = reports[packet.agent]
            route_path = self.pathfinder(
                self.grid, tuple(report["position"]), tuple(report["goal"])
            )
            report["path"] = route_path
            report["distance"] = float(len(route_path) - 1) if route_path else math.inf
            route = (
                ("access_down",)
                if self.config.mission_route == "uav"
                else ("satellite_down", "access_down")
            )
            sources = (
                {packet.agent: packet.generated_at}
                if field == "task"
                else payload["sources"]
            )
            self.transport.send(
                packet.agent,
                "information",
                {
                    field: {
                        "value": self._value(field, packet.agent, reports),
                        "sources": sources,
                    }
                },
                route=route,
                generated_at=packet.generated_at,
                version=packet.version,
                delayed=True,
            )
        elif packet.kind == "information":
            for key, item in payload.items():
                accepted = self.caches[packet.agent][key].accept(
                    packet.generated_at, item["value"], packet.version, item["sources"]
                )
                if accepted and key == "task":
                    for at, position in self.position_history[packet.agent]:
                        if packet.generated_at <= at <= self.transport.now:
                            item["value"].consume(position)

    def update(self, agents, now):
        # Forward reports to the UAV independently of backhaul availability.
        t = self.transport
        self.decision = None
        for a in agents:
            self.position_history[a.id].append((now, tuple(a.position)))
        t.advance(now, self._receive)
        for a in agents:
            t.send(
                a.id,
                "report",
                self._report(a),
                route=("access_up",),
                delayed=False,
                version=self.sequence,
            )
        t.advance(now, self._receive)
        self.sequence += 1
        for a in agents:
            for key, cache in self.caches[a.id].items():
                if (
                    key not in self.config.stale_information
                    or self.config.profile == "perfect"
                ):
                    cache.accept(
                        now,
                        self._value(
                            key, a.id, {peer.id: self._report(peer) for peer in agents}
                        ),
                        self.sequence,
                        {peer.id: now for peer in agents},
                    )
                    continue
                t.send(
                    a.id,
                    "mission_request",
                    {"field": key, "report": self._report(a)},
                    route=("access_up",),
                    version=self.sequence,
                    delayed=False,
                )
        t.advance(now, self._receive)
        for a in agents:
            self.caches[a.id]["task"].value.consume(a.position)

    def prepare_decision(self, agents, cycle_start, deadline_seconds):
        """Process real arrivals inside a fixed control-cycle budget, without moving robots."""
        at = cycle_start + deadline_seconds
        self.transport.advance(at, self._receive)
        fresh = {a.id: self._report(a) for a in agents}
        self.sequence += 1
        details = {}
        for a in agents:
            details[a.id] = {}
            for field, cache in self.caches[a.id].items():
                if field not in self.config.stale_information or self.config.profile == "perfect":
                    cache.accept(at, self._value(field, a.id, fresh), self.sequence,
                                 {peer.id: at for peer in agents})
                oldest = min(cache.source_timestamps.values()) if cache.source_timestamps else cache.timestamp
                details[a.id][field] = {
                    "packet_source_time": cache.timestamp,
                    "oldest_source_time": oldest,
                    "age_seconds": cache.age(at),
                    "response_missed_deadline": cache.timestamp < cycle_start-1e-9,
                    "source_from_previous_cycle": oldest < cycle_start-1e-9,
                }
            self.caches[a.id]["task"].value.consume(a.position)
        self.decision = {"cycle_start": cycle_start, "at": at, "budget_seconds": deadline_seconds,
                         "initial_briefing": cycle_start == 0, "inputs": details}

    def observation_values(self, agent):
        caches = self.caches[agent.id]
        guidance = caches["task"].value
        return (
            guidance.consume(agent.position),
            guidance.reported_distance,
            caches["intentions"].value,
            caches["priorities"].value,
        )

    def ages(self):
        return {
            agent: {key: cache.age(self.transport.now) for key, cache in caches.items()}
            for agent, caches in self.caches.items()
        }

    def snapshot(self):
        return {
            "time": self.transport.now,
            "decision": getattr(self, "decision", None),
            "information_age": self.ages(),
            "queue_bytes": self.transport.occupancy(),
            "cached_guidance": {
                a: list(c["task"].value.route) for a, c in self.caches.items()
            },
            "peer_information_age": {
                a: {
                    key: {
                        peer: max(0.0, self.transport.now - at)
                        for peer, at in (cache.source_timestamps or {}).items()
                    }
                    for key, cache in caches.items()
                }
                for a, caches in self.caches.items()
            },
            "uav_reports": {
                a: {
                    "position": r["position"],
                    "age_seconds": self.transport.now - self.report_timestamps[a],
                }
                for a, r in self.reports.items()
            },
        }
