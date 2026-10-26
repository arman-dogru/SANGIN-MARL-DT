"""Network-delivered observations, location-specific DT inference and action commands.

Physical sensing is performed only at report generation. Selected DT processors
consume uploaded sensing and their own delivered replica stores at completion.
"""
from dataclasses import asdict, replace
import hashlib
import json
import math
from time import perf_counter

import numpy as np

from .allocation import (AllocationTask, AllocationSnapshot, AllocationFailure, compare_allocations,
                         priority_weights, score_allocation, solve_binary)
from .information import InformationCache
from .consistency import CONSISTENCY_SCHEMA, consistency_snapshot
from .paper import PaperMissionCommunication
from .paper_radio import access_budget, backhaul_budget
from .records import write_json
from .remote_state import (REMOTE_SCHEMA, ActionCommand, CommandInbox, DTReplica, DTState,
                           PhysicalReport, observation_hash, remote_observation, source_min)
from .transport import encode
from .task_deadlines import command_deadline_fields


class RemoteDTCommunication(PaperMissionCommunication):
    def __init__(self, project, grid, agents, pathfinder, seed, cap, uav, inference, model_version=0):
        if inference is None:
            raise ValueError("Remote DT execution requires an explicit frozen inference callback")
        super().__init__(project, grid, agents, pathfinder, seed, cap, uav)
        self.inference = inference
        self.shadow_inference = None
        self.model_version = model_version
        self.remote_events, self.oracle_events = [], []
        self.events_by_id = {}
        self.event_checksum = "0"*64
        self.replicas = {name: DTReplica(name) for name in ("drone", "cloud")}
        self.inboxes = {a.id: CommandInbox() for a in agents}
        self.receiver_refs = {a.id: {} for a in agents}
        self.physical_refs = {}
        self.execution_actions, self.command_contexts = {}, {}
        self.latest_commands = {}
        self.completion_sent = set()
        self.radio_energy_j = 0.0
        self.control_energy_j = 0.0
        self._agents = agents
        self.current_cycle = 0
        for agent in agents:
            report = self._sense(agent, agents, 0, 0)
            state = self._state_for_report(report, 0, 0)
            for location, replica in self.replicas.items():
                event = self._record("dt_state_event", location=location, state=state.payload(),
                    available_at=0.0, bootstrap=True, accepted=True)
                replica.accept(state, 0, event["event_id"])
        for receiver, states in self.states.items():
            for peer, cache in states.items():
                event = self._record("received_state_event", receiver=receiver, peer=peer,
                    state_event=self.replicas["cloud"].state_ids[peer], state=cache.value,
                    arrived_at=0.0, generated_at=0.0, source_min=0.0, sources={peer: 0.0},
                    version=0, accepted=True, rejection_reason=None, bootstrap=True)
                self.receiver_refs[receiver][peer] = event["event_id"]
        self.dt_states = self.replicas["cloud"].states
        self._record("bootstrap", assumption="Fresh initial briefing at both replicas and robot caches; no free commands",
            model_version=self.model_version, synthetic_cpu_cycles=True)

    def _record(self, event_name, **values):
        event = dict(schema=REMOTE_SCHEMA, event=event_name, event_id=f"rdt:{len(self.remote_events)}",
            time=self.transport.now, previous_checksum=self.event_checksum, **values)
        # Freeze nested payloads and normalize JSON key types before hashing.
        # Later cache/controller changes cannot rewrite historical evidence.
        event = json.loads(json.dumps(event, allow_nan=False))
        canonical = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.event_checksum = hashlib.sha256(canonical).hexdigest()
        event["checksum"] = self.event_checksum
        self.remote_events.append(event)
        self.events_by_id[event["event_id"]] = event
        return event

    def _sense(self, agent, agents, cycle, now):
        radius = self.project.map.fov_radius
        padded = np.pad(self.grid, radius, constant_values=1)
        row, col = agent.position
        obstacles = padded[row:row+2*radius+1, col:col+2*radius+1].tolist()
        occupancy = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
        for peer in agents:
            dr, dc = peer.position[0]-row, peer.position[1]-col
            if peer.id != agent.id and abs(dr) <= radius and abs(dc) <= radius:
                occupancy[dr+radius, dc+radius] = 1
        normalizer = agent.initial_distance_to_goal
        if not math.isfinite(normalizer):
            normalizer = float(max(self.grid.shape))
        return PhysicalReport(agent.id, cycle, now, tuple(agent.position), tuple(agent.goal),
            agent.ever_reached_goal, normalizer, obstacles, occupancy.tolist(), f"report:{cycle}:{agent.id}")

    def _state_for_report(self, report, generated, version):
        path = self.pathfinder(self.grid, tuple(report.position), tuple(report.goal))
        distance = len(path)-1 if path else self.grid.size-1
        priority = (self.grid.size if report.completed else min(distance, self.grid.size-1))+report.robot_id*1e-6
        return DTState(report.robot_id, tuple(report.position), tuple(report.goal),
            list(path or [tuple(report.position)]), float(priority), report.completed,
            generated, version, {report.robot_id: report.sensed_at}, report.report_id)

    def _send_remote(self, agent, kind, payload, route, *, version=0, pad=False):
        body = encode(payload)
        if pad:
            body = body.ljust(math.ceil(self.parameters.task_input_bits/8), b"\0")
        packet_id = self.transport.send(agent, kind, route=route, wire=body,
            generated_at=self.transport.now, version=version, delayed=False)
        self._record("message_sent", packet_id=packet_id, kind=kind, route=list(route),
            job_id=payload.get("job_id"), wire_payload_bits=len(body)*8, version=version)
        return packet_id

    def _transmitted(self, packet, link, details):
        if not packet.kind.startswith("remote_"):
            return
        payload = packet.payload
        job = self.jobs.get(payload.get("job_id"))
        power = (self.parameters.robot_power_w if link == "access_up" else
                 self.parameters.drone_power_w if link == "satellite_up" else 0)
        energy = power*details["transmission_seconds"]
        self.radio_energy_j += energy
        if job is None:
            self.control_energy_j += energy
        else:
            job["robot_energy_j" if link == "access_up" else "backhaul_energy_j"] += energy
            if packet.kind in ("remote_robot_observation", "remote_cloud_observation"):
                for quantity in ("transmission_seconds", "queue_seconds", "guard_seconds", "interruption_seconds"):
                    job["uplink_"+quantity] += details[quantity]
                if link == "satellite_up" and details["last_chunk"]:
                    job["backhaul_transmission_finish"] = details["finish"]
            stage = job.setdefault("service_stages", {}).setdefault(f"{packet.kind}:{link}",
                dict(transmission_seconds=0.0, queue_seconds=0.0, guard_seconds=0.0, interruption_seconds=0.0))
            for key in ("transmission_seconds", "queue_seconds", "guard_seconds", "interruption_seconds"):
                stage[key] += details[key]
            stage["last_transmission_finish"] = details["finish"]
            wire_key = "access_wire_bits" if link == "access_up" else "backhaul_wire_bits" if link == "satellite_up" else "response_wire_bits"
            job[wire_key] = job.get(wire_key, 0)+details["wire_bits"]
        category = ("synchronization" if packet.kind == "remote_dt_sync" else
                    "upload" if packet.kind.endswith("observation") else "response")
        self.transport.log.append(dict(event="remote_transmit", job_id=payload.get("job_id"),
            id=packet.id, kind=packet.kind, link=link, energy_j=energy, category=category, **details))

    def _publish(self, state, location, job_id=None):
        short = replace(state, path=state.path[:self.project.map.communicated_path_steps+1])
        event = self._record("dt_state_event", location=location, state=short.payload(),
            available_at=self.transport.now, bootstrap=False)
        accepted = self.replicas[location].accept(short, self.transport.now, event["event_id"])
        self._record("dt_state_acceptance", state_event=event["event_id"], accepted=accepted)
        if not accepted:
            return event["event_id"]
        other = "cloud" if location == "drone" else "drone"
        common = dict(job_id=job_id, state=short.payload(), state_event=event["event_id"])
        self._send_remote(state.robot_id, "remote_dt_sync", dict(common, destination=other),
            ("satellite_up",) if location == "drone" else ("satellite_down",), version=state.version)
        down = ("access_down",) if location == "drone" else ("satellite_down", "access_down")
        for receiver in self.goals:
            self._send_remote(receiver, "remote_state", common, down, version=state.version)
        return event["event_id"]

    def _receive(self, packet):
        payload, t = packet.payload, self.transport
        if packet.kind in ("remote_robot_observation", "remote_completion"):
            report = PhysicalReport(**payload["report"])
            if report.sensed_at > t.now+1e-12:
                raise ValueError("Report arrived before sensing")
            if report.sensed_at >= self.report_timestamps[report.robot_id]:
                self.reports[report.robot_id] = {**self.reports[report.robot_id], "position": report.position}
                self.report_timestamps[report.robot_id] = report.sensed_at
            self._record("report_arrived", report_id=report.report_id, robot_id=report.robot_id,
                location="drone", source_time=report.sensed_at, available_at=t.now, packet_id=packet.id)
            if packet.kind == "remote_completion":
                self._publish(self._state_for_report(report, t.now, packet.version), "drone")
            else:
                job = self.jobs[payload["job_id"]]
                job["access_arrived_at"] = t.now
                self._admit(payload)
        elif packet.kind == "remote_cloud_observation":
            job = self.jobs[payload["job_id"]]
            job.update(stage="computing", compute_arrived_at=t.now,
                uplink_propagation_seconds=t.now-job["backhaul_transmission_finish"])
            self._record("report_arrived", report_id=job["report"]["report_id"], robot_id=job["agent"],
                location="cloud", source_time=job["source_time"], available_at=t.now, packet_id=packet.id)
            t.compute_pools["cloud"].submit(job["job_id"], job["required_cycles"], job["weight"], job["job_id"])
        elif packet.kind == "remote_dt_sync":
            state = DTState(**payload["state"])
            accepted = self.replicas[payload["destination"]].accept(state, t.now, payload["state_event"])
            self._record("replica_sync_arrived", location=payload["destination"], state_event=payload["state_event"],
                state=payload["state"], arrived_at=t.now, accepted=accepted, packet_id=packet.id)
        elif packet.kind == "remote_state":
            state = DTState(**payload["state"])
            if state.generated_at > t.now+1e-12:
                raise ValueError("State arrived before generation")
            cache = self.states[packet.agent][state.robot_id]
            accepted = False
            if state.source_time >= source_min(cache.source_timestamps):
                accepted = cache.accept(state.generated_at, state.payload(), state.version, dict(state.sources))
            event = self._record("received_state_event", receiver=packet.agent, peer=state.robot_id,
                state_event=payload["state_event"], state=state.payload(), sources=state.sources,
                arrived_at=t.now, accepted=accepted,
                generated_at=state.generated_at, source_min=state.source_time, version=state.version,
                rejection_reason=None if accepted else "obsolete_source_or_order", packet_id=packet.id)
            if accepted:
                self.receiver_refs[packet.agent][state.robot_id] = event["event_id"]
                self._sync(packet.agent)
        elif packet.kind == "remote_action_command":
            command = ActionCommand(**payload["command"])
            accepted, reason = self.inboxes[packet.agent].accept(command, t.now)
            if accepted:
                self.latest_commands[packet.agent] = command
            job = self.jobs[command.job_id]
            # A duplicate must not turn an on-time first arrival into a late
            # completion or erase a previously accepted logical command.
            if "delivered_at" not in job:
                job.update(delivered_at=t.now, observed_latency_seconds=t.now-job["source_time"],
                    response_seconds=t.now-job["compute_completed_at"])
            job["command_accepted"] = job.get("command_accepted", False) or accepted
            job["command_arrival_count"] = job.get("command_arrival_count", 0)+1
            self._record("command_arrived", robot_id=packet.agent, command=command.payload(),
                arrived_at=t.now, accepted=accepted, reason=reason, packet_id=packet.id)

    def _available_weights(self):
        cache = self.replicas["drone"].states
        reports = {a: c.value for a, c in cache.items()}
        ages = {a: c.age(self.transport.now) for a, c in cache.items()}
        if self.parameters.policy == "equal" and self.parameters.equal_weight_mode == "matched_active_mean":
            ids = sorted(reports)
            bounded = priority_weights([reports[a]["priority"] for a in ids],
                [reports[a]["completed"] for a in ids], self.parameters.weight_min, self.parameters.weight_max)
            active = [i for i, a in enumerate(ids) if not reports[a]["completed"]]
            mean = float(np.mean(bounded[active])) if active else self.parameters.weight_min
            return {a: self.parameters.weight_min if reports[a]["completed"] else mean for a in ids}
        return self._weights(reports, ages)

    def _admit(self, payload):
        t, p = self.transport, self.parameters
        job = self.jobs[payload["job_id"]]
        weights, radio = self._available_weights(), backhaul_budget(t, p)
        if t.orbit is None:
            t.links["satellite_up"].capacity_bps = radio["applied_bps"] or self.config.satellite_uplink_mbps*1e6
        task = AllocationTask(job["job_id"], p.task_input_bits, p.cycles_per_bit,
            weights[job["agent"]], payload["access_radio"]["applied_bps"], radio["applied_bps"],
            radio["round_trip_seconds"], p.robot_power_w, p.drone_power_w,
            deadline_seconds=(job["deadline_at"]-t.now+p.task_input_bits/payload["access_radio"]["applied_bps"]
                              if p.enforce_task_deadlines else None))
        tasks = [task]
        remaining = {r["job_id"]: r["remaining_cycles"] for pool in t.compute_pools.values() for r in pool.snapshot()}
        for previous in self.jobs.values():
            if previous["job_id"] == job["job_id"] or previous["stage"] not in ("computing", "backhaul"):
                continue
            tasks.append(AllocationTask(previous["job_id"], 0, 0, previous["weight"], 1, 1, 0,
                p.robot_power_w, p.drone_power_w, fixed_x=previous["executed_x"],
                required_cycles=remaining.get(previous["job_id"], previous["required_cycles"]),
                deadline_seconds=previous["deadline_at"]-t.now if p.enforce_task_deadlines else None))
        snapshot = AllocationSnapshot(tuple(tasks), p.drone_cycles_per_second, p.cloud_cycles_per_second,
            p.energy_coefficient, p.latency_coefficient, p.drone_kappa, p.include_robot_upload_in_objective)
        try:
            comparison = compare_allocations(snapshot)
        except AllocationFailure as exc:
            directory = self.project.resolve(p.failure_directory)
            write_json(directory/f"remote-seed-{self.seed}-{self.allocations}.json",
                dict(error=str(exc), snapshot=exc.snapshot, details=exc.details))
            raise
        self.allocations += 1
        event = dict(event="paper_allocation", schema=REMOTE_SCHEMA, time=t.now, radio=radio,
            new_job=job["job_id"], priority_versions=dict(self.replicas["drone"].state_ids),
            received_priorities={a: c.value["priority"] for a, c in self.replicas["drone"].states.items()},
            resource_weights=weights, **comparison)
        self.transport.log.append(event)
        if p.oracle_shadow:
            self._allocation_shadow(snapshot, comparison["binary"], job["job_id"])
        self._record("allocation_event", job_id=job["job_id"], allocation_version=self.allocations,
            new_vs_fixed={x.task_id: x.fixed_x is None for x in tasks},
            priority_versions=event["priority_versions"], transport_allocation_index=self.allocations,
            assignments=comparison["binary"]["tasks"], snapshot=asdict(snapshot),
            dt_priorities=event["received_priorities"], resource_weights=weights)
        binary = next(r for r in comparison["binary"]["tasks"] if r["task_id"] == job["job_id"])
        relaxed = next(r for r in comparison["relaxed"]["tasks"] if r["task_id"] == job["job_id"])
        job.update(executed_x=int(binary["x"]), relaxed_x=relaxed["x"], weight=task.weight,
            predicted_latency_seconds=binary["latency_seconds"], predicted_energy_j=binary["energy_j"],
            predicted_allocation=binary, relaxed_prediction=relaxed, allocation_at=t.now,
            backhaul_radio=radio, relaxation_gap=comparison["absolute_gap"], batch_wait_seconds=0.0)
        if job["executed_x"] or self.config.profile == "perfect":
            location = "drone" if job["executed_x"] else "cloud"
            job.update(stage="computing", compute_arrived_at=t.now, location=location)
            if self.config.profile == "perfect":
                self._finish(job["job_id"], 0, 0)
            else:
                t.compute_pools[location].submit(job["job_id"], job["required_cycles"], task.weight, job["job_id"])
        else:
            job.update(stage="backhaul", location="cloud")
            self._send_remote(job["agent"], "remote_cloud_observation", payload, ("satellite_up",),
                version=job["version"], pad=True)

    def _allocation_shadow(self, actual_snapshot, actual, new_job):
        """Evaluation-only oracle. No transport, cache, KKT pool or real decision is changed."""
        ids = sorted(self.truth)
        weights = priority_weights([self.truth[a]["priority"] for a in ids],
            [self.truth[a]["completed"] for a in ids], self.parameters.weight_min, self.parameters.weight_max)
        physical_weights = dict(zip(ids, map(float, weights)))
        reference = replace(actual_snapshot, tasks=tuple(replace(task,
            weight=physical_weights[self.jobs[task.task_id]["agent"]]) for task in actual_snapshot.tasks))
        # Binary enumeration is pure and does not change the actual conic template's warm start.
        oracle = solve_binary(reference)
        scored = score_allocation(reference, actual["tasks"])
        regret = scored.objective-oracle.objective
        if regret < -1e-6*max(1, abs(oracle.objective)):
            raise AssertionError("Truth-scored oracle regret is negative")
        by_id = {r["task_id"]: r for r in oracle.tasks}
        self.oracle_events.append(dict(event="allocation_shadow", time=self.transport.now,
            allocation_version=self.allocations, new_job=new_job,
            actual_priorities={a: c.value["priority"] for a, c in self.replicas["drone"].states.items()},
            physical_priorities={a: r["priority"] for a, r in self.truth.items()},
            physical_weights=physical_weights, reference_snapshot=asdict(reference),
            native_actual_objective=actual["objective"], native_oracle_objective=oracle.objective,
            truth_actual_objective=scored.objective, truth_oracle_objective=oracle.objective,
            truth_scored_regret=max(0.0, regret), oracle=oracle.to_dict(),
            offload_switches=[r["task_id"] for r in actual["tasks"] if r["x"] != by_id[r["task_id"]]["x"]],
            cpu_differences={r["task_id"]: dict(drone=by_id[r["task_id"]]["drone_share"]-r["drone_share"],
                cloud=by_id[r["task_id"]]["cloud_share"]-r["cloud_share"]) for r in actual["tasks"]}))

    def _computed(self, completed):
        self._finish(completed.job_id, completed.service_seconds, completed.compute_energy_j)

    def _inference_replica(self, location, report):
        """Delivered peer inputs; controlled experiments may buffer this read only."""
        return self.replicas[location]

    def _after_inference(self, report, own, location, observation, action, event):
        """Optional observational diagnostics; never advances simulated service."""

    def _finish(self, job_id, service, energy):
        t, job = self.transport, self.jobs[job_id]
        job.update(stage="computed", compute_completed_at=t.now, compute_service_seconds=service,
            drone_compute_energy_j=energy, compute_elapsed_seconds=t.now-job["compute_arrived_at"])
        report = PhysicalReport(**job["report"])
        own = self._state_for_report(report, t.now, job["version"])
        # Own generated state is causal at completion; peer state remains replica-specific.
        own_event = self._publish(own, job["location"], job_id)
        obs, sources, versions, priorities = remote_observation(report, own, self._inference_replica(job["location"], report),
            self.project.map.fov_radius, self.project.map.astar_waypoints,
            self.project.map.communicated_path_steps, max(self.grid.shape))
        versions[job["agent"]]["state_event"] = own_event
        start = perf_counter()
        action = int(self.inference(job["location"], job["agent"], obs))
        wall = perf_counter()-start
        event = self._record("remote_inference", job_id=job_id, robot_id=job["agent"], location=job["location"],
            observation={k: v.tolist() for k, v in obs.items()}, observation_hash=observation_hash(obs),
            sources=sources, source_min=source_min(sources), dt_versions=versions, dt_priorities=priorities,
            action=action, model_version=self.model_version)
        job["inference_wall_seconds"] = wall  # Measurement only; never advances virtual service.
        job["inference_event"] = event["event_id"]
        self._after_inference(report, own, job["location"], obs, action, event)
        if self.parameters.oracle_shadow and self.shadow_inference is not None:
            # Evaluation-only fresh replica built from physical state at this
            # instant. It is never installed at either actual processor.
            shadow_replica = DTReplica("cloud")
            source_time = t.now  # Ideal evaluation-only sensing at the shadow decision instant.
            for agent in self._agents:
                fresh_report = self._sense(agent, self._agents, report.cycle, source_time)
                fresh = self._state_for_report(fresh_report, t.now, report.cycle)
                shadow_replica.accept(fresh, t.now, f"oracle:{report.cycle}:{agent.id}")
            fresh_report = self._sense(self._agents[job["agent"]], self._agents, report.cycle, source_time)
            fresh_own = self._state_for_report(fresh_report, t.now, report.cycle)
            oracle_obs, oracle_sources, _, _ = remote_observation(fresh_report, fresh_own, shadow_replica,
                self.project.map.fov_radius, self.project.map.astar_waypoints,
                self.project.map.communicated_path_steps, max(self.grid.shape))
            oracle_action = int(self.shadow_inference(job["location"], job["agent"], oracle_obs))
            self.oracle_events.append(dict(event="action_shadow", time=t.now, job_id=job_id,
                actual_action=action, oracle_action=oracle_action, disagrees=action != oracle_action,
                observation={k: v.tolist() for k, v in oracle_obs.items()},
                observation_hash=observation_hash(oracle_obs), sources=oracle_sources))
        command = ActionCommand(job["agent"], report.cycle, action, t.now,
            report.cycle*self.config.seconds_per_step+self.parameters.decision_budget_seconds,
            job["version"], job["location"], sources, job_id, observation_hash(obs), versions, priorities)
        self._send_remote(job["agent"], "remote_action_command", dict(job_id=job_id, command=command.payload()),
            ("access_down",) if job["location"] == "drone" else ("satellite_down", "access_down"),
            version=job["version"])

    def update(self, agents, now, generate_tasks=True):
        self._agents = agents
        self.truth = {a.id: self._report(a) for a in agents}
        self.positions = {a.id: tuple(a.position) for a in agents}
        self.transport.advance(now, self._receive)
        self.current_cycle = int(round(now/self.config.seconds_per_step))
        if self.config.profile == "perfect":
            # Explicit ideal-reference exception; physical treatments never refresh this way.
            for agent in agents:
                fresh = self._state_for_report(self._sense(agent, agents, self.current_cycle, now), now, self.current_cycle)
                for location, replica in self.replicas.items():
                    event = self._record("dt_state_event", location=location, state=fresh.payload(),
                        available_at=now, ideal_reference=True)
                    replica.accept(fresh, now, event["event_id"])
                # Zero-delay inference republishes the same state/version, so
                # _publish correctly rejects it as a duplicate. Refresh robot
                # caches explicitly as part of the ideal-reference exception.
                for receiver, states in self.states.items():
                    accepted = states[agent.id].accept(now, fresh.payload(), self.current_cycle,
                        dict(fresh.sources))
                    received = self._record("received_state_event", receiver=receiver, peer=agent.id,
                        state_event=self.replicas["cloud"].state_ids[agent.id], state=fresh.payload(),
                        arrived_at=now, generated_at=now, source_min=fresh.source_time,
                        sources=fresh.sources, version=self.current_cycle, accepted=accepted,
                        rejection_reason=None if accepted else "obsolete_source_or_order", ideal_reference=True)
                    if accepted:
                        self.receiver_refs[receiver][agent.id] = received["event_id"]
                        self._sync(receiver)
        for agent in agents:
            self.position_history[agent.id].append((now, tuple(agent.position)))
            if agent.ever_reached_goal and agent.id in self.completion_sent:
                continue
            report = self._sense(agent, agents, self.current_cycle, now)
            event = self._record("robot_state_event", report=report.payload(), physical=self.truth[agent.id])
            self.physical_refs[agent.id] = event["event_id"]
            if agent.ever_reached_goal:
                self.completion_sent.add(agent.id)
                self._send_remote(agent.id, "remote_completion", dict(report=report.payload()),
                    ("access_up",), version=self.current_cycle)
                continue
            if not generate_tasks:
                continue
            xy = self.uav.positions_to_xy_m([tuple(agent.position)])[0]
            horizontal = float(np.linalg.norm(xy-self.uav.position[:2]))
            loss = float(self.uav.model.path_loss_db(self.uav.position[2], horizontal))
            radio = access_budget(loss, self.parameters, self.config.access_uplink_mbps*1e6)
            radio.update(height_m=float(self.uav.position[2]), horizontal_distance_m=horizontal,
                los_probability=float(self.uav.model.los_probability(self.uav.position[2]/max(horizontal, 1e-6))))
            self.transport.agent_rate_caps[(agent.id, "access_up")] = radio["applied_bps"]
            job_id = f"{self.current_cycle}:{agent.id}"
            self.jobs[job_id] = dict(job_id=job_id, agent=agent.id, report=report.payload(), version=self.current_cycle,
                source_time=now, stage="access", input_bits=self.parameters.task_input_bits,
                deadline_at=(now+self.parameters.task_deadline_seconds
                             if self.parameters.task_deadline_seconds is not None else None),
                cycles_per_bit=self.parameters.cycles_per_bit,
                required_cycles=self.parameters.task_input_bits*self.parameters.cycles_per_bit,
                access_radio=radio, robot_energy_j=0.0, backhaul_energy_j=0.0, drone_compute_energy_j=0.0,
                **{"uplink_"+k: 0.0 for k in ("transmission_seconds", "queue_seconds", "guard_seconds", "interruption_seconds")})
            self._send_remote(agent.id, "remote_robot_observation", dict(job_id=job_id,
                report=report.payload(), access_radio=radio), ("access_up",), version=self.current_cycle, pad=True)
        self.transport.advance(now, self._receive)

    def prepare_decision(self, agents, cycle_start, deadline_seconds):
        if deadline_seconds != self.parameters.decision_budget_seconds:
            raise ValueError("Remote command cutoff differs from its configured epoch")
        cutoff = cycle_start+deadline_seconds
        self.transport.advance(cutoff, self._receive)
        cycle = int(round(cycle_start/self.config.seconds_per_step))
        self.execution_actions, self.command_contexts = {}, {}
        for agent in agents:
            delivered = self.inboxes[agent.id].decision(cycle, cutoff)
            command, arrival = delivered if delivered else (None, None)
            action = command.action if command is not None and not agent.ever_reached_goal else 4
            local_sources, local_details = None, None
            own_priority = self.states[agent.id][agent.id].value["priority"]
            cached_fallback = (command is None and not agent.ever_reached_goal
                               and self.parameters.missed_command_fallback == "cached_astar")
            if cached_fallback:
                from .remote_control import cached_path_action
                local_report = self._sense(agent, agents, cycle, cutoff)
                action, local_sources, local_details = cached_path_action(
                    local_report, self.states[agent.id], self.project.map.communicated_path_steps)
            if agent.ever_reached_goal:
                from .remote_control import completed_action
                local_report = self._sense(agent, agents, cycle, cutoff)
                # Completion is known locally; never promote it from a stale self cache.
                own_priority = self.grid.size+agent.id*1e-6
                action, local_sources, local_details = completed_action(
                    local_report, self.states[agent.id], own_priority)
                if self.parameters.oracle_shadow:
                    fresh = {}
                    for peer, physical in self.truth.items():
                        cache = InformationCache()
                        cache.accept(cutoff, self._state(physical), cycle, {peer: cutoff})
                        fresh[peer] = cache
                    shadow_action, _, _ = completed_action(local_report, fresh, own_priority)
                    from .remote_control import DELTAS
                    def goal_distance(selected):
                        dr, dc = DELTAS[selected]
                        return abs(agent.position[0]+dr-agent.goal[0])+abs(agent.position[1]+dc-agent.goal[1])
                    distance = goal_distance(4)
                    self.oracle_events.append(dict(event="local_control_shadow", time=cutoff,
                        robot_id=agent.id, cycle=cycle, actual_action=action, oracle_action=shadow_action,
                        wrong_yield=goal_distance(action) > distance and goal_distance(shadow_action) <= distance,
                        unnecessary_wait=action == 4 and shadow_action != 4))
            self.execution_actions[agent.id] = action
            context = dict(robot_id=agent.id, cycle=cycle, cutoff_time=cutoff,
                action=action, command=command.payload() if command else None, arrived_at=arrival,
                command_cutoff_missed=command is None and not agent.ever_reached_goal,
                stale_source=command is not None and source_min(command.sources) < cycle_start-1e-12,
                timeout_reason="no_current_command" if command is None and not agent.ever_reached_goal else None,
                completed_local_control=agent.ever_reached_goal)
            context.update(local_sources=local_sources, local_control=local_details,
                resolution_priority=(command.dt_priorities.get(agent.id, own_priority)
                                     if command is not None and not agent.ever_reached_goal else own_priority))
            if cached_fallback:
                context.update(cached_fallback=True, fallback_source_min=source_min(local_sources),
                    fallback_source_age_seconds=cutoff-source_min(local_sources))
            self.command_contexts[agent.id] = context
            self._record("decision_event", **context)
            if command is not None:
                self.jobs[command.job_id]["used_at"] = cutoff
        self.decision = dict(cycle_start=cycle_start, at=cutoff, budget_seconds=deadline_seconds,
            initial_briefing=cycle == 0, remote=True, inputs=self.command_contexts)
        consistency = self._consistency()
        self._record("consistency_event", cycle=cycle, consistency_schema=CONSISTENCY_SCHEMA,
            summary={k: v for k, v in consistency.items() if k != "pairs"},
            consistency_sha256=hashlib.sha256(json.dumps(consistency, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest())
        if self.parameters.oracle_shadow and self.shadow_inference is not None:
            # Evaluate the current physical state at the decision epoch, including
            # timeout WAITs. This fresh replica is diagnostic only: no actual
            # cache, job, command, or transport event is changed.
            fresh_reports = {a.id: self._sense(a, agents, cycle, cutoff) for a in agents}
            fresh_states = {a: self._state_for_report(r, cutoff, cycle) for a, r in fresh_reports.items()}
            oracle_replica = DTReplica("cloud")
            for robot, state in fresh_states.items():
                oracle_replica.accept(state, cutoff, f"decision-oracle:{cycle}:{robot}")
            for agent in agents:
                if agent.ever_reached_goal:
                    continue  # Completed robots retain the separate local-control shadow.
                observation, sources, versions, priorities = remote_observation(
                    fresh_reports[agent.id], fresh_states[agent.id], oracle_replica,
                    self.project.map.fov_radius, self.project.map.astar_waypoints,
                    self.project.map.communicated_path_steps, max(self.grid.shape))
                oracle_action = int(self.shadow_inference("cloud", agent.id, observation))
                context = self.command_contexts[agent.id]
                actual_action = self.execution_actions[agent.id]
                self.oracle_events.append(dict(event="decision_action_shadow", time=cutoff,
                    cycle=cycle, robot_id=agent.id, cutoff_time=cutoff, evaluation_only=True,
                    actual_action=actual_action, oracle_action=oracle_action,
                    disagrees=actual_action != oracle_action,
                    command_cutoff_missed=context["command_cutoff_missed"],
                    timeout_reason=context["timeout_reason"],
                    actual_command_job_id=context["command"]["job_id"] if context["command"] else None,
                    actual_command_arrived_at=context["arrived_at"], model_version=self.model_version,
                    observation={k: v.tolist() for k, v in observation.items()},
                    observation_hash=observation_hash(observation), sources=sources,
                    source_min=source_min(sources), dt_versions=versions, dt_priorities=priorities))

    def _consistency(self):
        views = {}
        for receiver, context in self.command_contexts.items():
            command = context["command"]
            if context["cycle"] != self.current_cycle:
                continue
            if command is None:
                if context.get("cached_fallback"):
                    caches = self.states[receiver]
                    views[receiver] = dict(priorities={p: c.value["priority"] for p,c in caches.items()},
                        states={p: c.value for p,c in caches.items()})
                continue
            priorities = {int(k): v for k, v in command["dt_priorities"].items()}
            states = {int(peer): self.events_by_id[value["state_event"]]["state"]
                      for peer, value in command["dt_versions"].items()}
            views[receiver] = dict(priorities=priorities, states=states)
        metadata = {receiver: {peer: self.events_by_id[ref] for peer, ref in refs.items()}
                    for receiver, refs in self.receiver_refs.items()}
        return consistency_snapshot(self.positions, self.truth, self.states, self.transport.now,
            self.project.map.fov_radius, self.project.map.communicated_path_steps,
            self.parameters.age_reference_seconds, self.dt_states, views, metadata)

    def snapshot(self):
        result = super().snapshot()
        result["consistency"] = self._consistency()
        result.update(remote_schema=REMOTE_SCHEMA, execution_mode="remote_dt", event_checksum=self.event_checksum,
            remote_refs=dict(physical=dict(self.physical_refs),
                replicas={name: dict(replica.state_ids) for name, replica in self.replicas.items()},
                receivers=self.receiver_refs), command_contexts=self.command_contexts)
        return result

    def task_records(self):
        records = super().task_records()
        decisions = {(e["cycle"], e["robot_id"]): e for e in self.remote_events if e["event"] == "decision_event"}
        for record in records:
            record.update(command_deadline_fields(record, self.transport.now))
            decision = decisions.get((record["version"], record["agent"]))
            record["command_cutoff_at"] = decision["cutoff_time"] if decision else None
            record["cutoff_induced_wait"] = (decision["command_cutoff_missed"] and decision["action"] == 4
                                             if decision else None)
        return records

    def ages(self):
        result = super().ages()
        for receiver, fields in result.items():
            command = self.latest_commands.get(receiver)
            fields["task"] = self.transport.now-source_min(command.sources) if command else self.transport.now
        return result

    def energy_bounds(self):
        result = super().energy_bounds()
        pending = result["unfinished_airtime_energy_bound_j"]
        compute = result["drone_compute_energy_j"]
        result.update(radio_energy_lower_j=self.radio_energy_j, radio_energy_upper_j=self.radio_energy_j+pending,
            modeled_energy_lower_j=self.radio_energy_j+compute,
            modeled_energy_upper_j=self.radio_energy_j+compute+pending, completion_control_energy_j=self.control_energy_j)
        return result
