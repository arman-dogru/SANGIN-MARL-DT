"""Opt-in paper experiment: causal allocation, finite compute, atomic S_j caches."""
from __future__ import annotations

import math

import numpy as np

from .allocation import AllocationFailure, AllocationSnapshot, AllocationTask, compare_allocations, priority_weights
from .compute import ComputePool
from .consistency import consistency_snapshot
from .information import Guidance, InformationCache, MissionCommunication
from .paper_radio import access_budget, backhaul_budget
from .records import write_json
from .transport import encode


class PaperMissionCommunication(MissionCommunication):
    def __init__(self, project, grid, agents, pathfinder, seed, cap, uav):
        super().__init__(project.communication, grid, agents, pathfinder, seed, cap)
        self.project, self.parameters, self.uav = project, project.resource_allocation, uav
        self.seed = seed
        if self.config.profile not in ("physical", "perfect"):
            raise ValueError("Paper mode requires physical radio service or a perfect reference")
        if self.config.uplink_trace or self.config.downlink_trace:
            raise ValueError("Paper mode requires modeled radio service, not delay traces")
        if ("intentions" in self.config.stale_information) != ("priorities" in self.config.stale_information):
            raise ValueError("Paper coordination streams must be delayed atomically")
        if not self.parameters.decision_budget_seconds < self.config.seconds_per_step:
            raise ValueError("Control decision budget must fit within one control cycle")
        if self.transport.orbit is not None and self.config.packet_payload_bytes == 0:
            raise ValueError("Paper orbital service requires packet_payload_bytes > 0")
        self.states = {a.id: {b.id: InformationCache() for b in agents} for a in agents}
        self.truth = {a.id: self._report(a) for a in agents}
        self.positions = {a.id: tuple(a.position) for a in agents}
        for receiver in self.states:
            self.caches[receiver]["task"].source_timestamps = {receiver: 0.0}
            for peer in self.states[receiver]:
                self.states[receiver][peer].accept(0.0, self._state(self.truth[peer]), 0, {peer: 0.0})
            self._sync(receiver)
        self.jobs = {}
        self.pending_batches = {}
        self.allocations = 0
        self.static_weights = self._weights(self.truth, {a.id: 0.0 for a in agents})
        self.transport.transmission_observer = self._transmitted
        p = self.parameters
        self.transport.compute_pools = {
            "drone": ComputePool("drone", p.drone_cycles_per_second, self.transport, self._computed,
                kappa=p.drone_kappa, energy_coefficient=p.energy_coefficient, latency_coefficient=p.latency_coefficient),
            "cloud": ComputePool("cloud", p.cloud_cycles_per_second, self.transport, self._computed)}

    def _report(self, agent):
        report = super()._report(agent)
        if not math.isfinite(report["distance"]):
            report["distance"] = float(self.grid.size)
        return report

    def _state(self, report):
        return {key: (list(report["path"][:self.project.map.communicated_path_steps+1])
                      if key == "path" else report[key])
                for key in ("position", "path", "priority", "completed")}

    def _sync(self, receiver):
        values = self.states[receiver]
        sources = {peer: min(c.source_timestamps.values()) for peer, c in values.items()}
        for field in ("intentions", "priorities"):
            cache = self.caches[receiver][field]
            cache.timestamp = min(c.timestamp for c in values.values())
            cache.version = min(c.version for c in values.values())
            cache.source_timestamps = sources
            cache.value = ({peer: dict(path=c.value["path"], completed=c.value["completed"])
                            for peer, c in values.items()} if field == "intentions"
                           else {peer: c.value["priority"] for peer, c in values.items()})

    def _weights(self, reports, ages):
        ids = sorted(reports)
        p = self.parameters
        values = priority_weights([reports[a]["priority"] for a in ids],
            [reports[a]["completed"] for a in ids], p.weight_min, p.weight_max)
        if p.policy == "equal":
            values[:] = p.weight_min
        elif p.policy == "static" and hasattr(self, "static_weights"):
            values = np.array([self.static_weights[a] for a in ids])
        elif p.policy == "aoi":
            values = np.array([p.weight_min+(p.weight_max-p.weight_min)
                *min(1.0, ages[a]/p.age_reference_seconds) for a in ids])
        for i, agent in enumerate(ids):
            if reports[agent]["completed"]:
                values[i] = p.weight_min
        return dict(zip(ids, map(float, values)))

    def _send(self, agent, kind, payload, route, generated_at=None, version=0, pad=False):
        wire = encode(payload)
        if pad:
            wire = wire.ljust(math.ceil(self.parameters.task_input_bits/8), b"\0")
        return self.transport.send(agent, kind, route=route, wire=wire,
            generated_at=generated_at, version=version, delayed=False)

    def _transmitted(self, packet, link, details):
        if not packet.kind.startswith("paper_"):
            return
        job_id = packet.payload["job_id"]
        job = self.jobs[job_id]
        p = self.parameters
        robot_energy = p.robot_power_w*details["transmission_seconds"] if link == "access_up" else 0.0
        drone_energy = p.drone_power_w*details["transmission_seconds"] if link == "satellite_up" else 0.0
        job["robot_energy_j"] += robot_energy
        job["backhaul_energy_j"] += drone_energy
        wire_key = ("access_wire_bits" if link == "access_up" else
                    "backhaul_wire_bits" if link == "satellite_up" else "response_all_recipients_wire_bits")
        job[wire_key] = job.get(wire_key, 0)+details["wire_bits"]
        if packet.kind in ("paper_input", "paper_cloud_input"):
            for quantity in ("transmission_seconds", "queue_seconds",
                             "interruption_seconds", "guard_seconds"):
                job["uplink_"+quantity] += details[quantity]
            if link == "satellite_up" and details["last_chunk"]:
                job["backhaul_transmission_finish"] = details["finish"]
        self.transport.log.append(dict(event="paper_transmit", job_id=job_id, id=packet.id,
            agent=packet.agent, link=link, kind=packet.kind, robot_energy_j=robot_energy,
            backhaul_energy_j=drone_energy, **details))

    def _receive(self, packet):
        payload = packet.payload
        t = self.transport
        if packet.kind == "paper_input":
            agent, report = packet.agent, payload["report"]
            if packet.generated_at >= self.report_timestamps[agent]:
                self.reports[agent] = dict(report)
                self.report_timestamps[agent] = packet.generated_at
            batch = self.pending_batches.setdefault(packet.version, {})
            batch[agent] = payload
            self.jobs[payload["job_id"]]["access_arrived_at"] = t.now
            if len(batch) == len(self.goals):
                self._allocate(batch)
                del self.pending_batches[packet.version]
        elif packet.kind == "paper_cloud_input":
            job = self.jobs[payload["job_id"]]
            job["uplink_propagation_seconds"] = t.now-job["backhaul_transmission_finish"]
            job["stage"] = "computing"
            job["compute_arrived_at"] = t.now
            t.compute_pools["cloud"].submit(payload["job_id"], job["required_cycles"], job["weight"], payload["job_id"])
        elif packet.kind == "paper_guidance":
            cache = self.caches[packet.agent]["task"]
            if payload["source_time"] >= min(cache.source_timestamps.values()):
                accepted = cache.accept(packet.generated_at,
                    Guidance(payload["path"], payload["distance"]), packet.version,
                    {packet.agent: payload["source_time"]})
                if accepted:
                    for at, position in self.position_history[packet.agent]:
                        if payload["source_time"] <= at <= t.now:
                            cache.value.consume(position)
            job = self.jobs[payload["job_id"]]
            job.update(delivered_at=t.now, observed_latency_seconds=t.now-job["source_time"],
                response_seconds=t.now-job["compute_completed_at"])
            self.transport.log.append(dict(event="paper_task_delivered", time=t.now, **job))
        elif packet.kind == "paper_state":
            peer = payload["peer"]
            cache = self.states[packet.agent][peer]
            source = payload["source_time"]
            accepted = False
            if source >= min(cache.source_timestamps.values()):
                accepted = cache.accept(packet.generated_at, payload["state"], packet.version, {peer: source})
            if accepted:
                self._sync(packet.agent)
            self.transport.log.append(dict(event="paper_state_arrived", time=t.now,
                receiver=packet.agent, peer=peer, source_time=source,
                generation_time=packet.generated_at, accepted=accepted, job_id=payload["job_id"]))

    def _allocate(self, batch):
        t, p = self.transport, self.parameters
        reports = {a: payload["report"] for a, payload in batch.items()}
        ages = {a: max(0.0, t.now-payload["reported_cache_source_time"]) for a, payload in batch.items()}
        weights = self._weights(reports, ages)
        radio = backhaul_budget(t, p)
        # The fixed-geometry bound and configured cap both constrain actual service.
        if t.orbit is None:
            t.links["satellite_up"].capacity_bps = radio["applied_bps"]
        tasks = []
        for agent, payload in sorted(batch.items()):
            tasks.append(AllocationTask(payload["job_id"], p.task_input_bits, p.cycles_per_bit,
                weights[agent], payload["access_radio"]["applied_bps"], radio["applied_bps"],
                radio["round_trip_seconds"], p.robot_power_w, p.drone_power_w))
        remaining = {j["job_id"]: j["remaining_cycles"]
                     for pool in t.compute_pools.values() for j in pool.snapshot()}
        # Earlier offloading decisions remain fixed. In-flight remote jobs retain
        # their full compute demand until they reach the cloud pool.
        for job_id, job in self.jobs.items():
            if job.get("stage") not in ("computing", "backhaul"):
                continue
            tasks.append(AllocationTask(job_id, 0, 0, job["weight"], 1, 1, 0,
                p.robot_power_w, p.drone_power_w, fixed_x=job["executed_x"],
                required_cycles=remaining.get(job_id, job["required_cycles"])))
        snapshot = AllocationSnapshot(tuple(tasks), p.drone_cycles_per_second,
            p.cloud_cycles_per_second, p.energy_coefficient, p.latency_coefficient,
            p.drone_kappa, p.include_robot_upload_in_objective)
        try:
            comparison = compare_allocations(snapshot)
        except AllocationFailure as exc:
            directory = self.project.resolve(p.failure_directory)
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory/f"seed-{self.seed}-allocation-{self.allocations}.json",
                dict(error=str(exc), snapshot=exc.snapshot, details=exc.details,
                     time=t.now, radio=radio))
            raise
        self.allocations += 1
        self.transport.log.append(dict(event="paper_allocation", time=t.now, radio=radio,
            input_source_times={a: v["source_time"] for a, v in batch.items()}, **comparison))
        binary = {r["task_id"]: r for r in comparison["binary"]["tasks"]}
        relaxed = {r["task_id"]: r for r in comparison["relaxed"]["tasks"]}
        for agent, payload in sorted(batch.items()):
            job_id = payload["job_id"]
            job = self.jobs[job_id]
            row = binary[job_id]
            job.update(executed_x=int(row["x"]), relaxed_x=relaxed[job_id]["x"],
                predicted_latency_seconds=row["latency_seconds"],
                predicted_energy_j=row["energy_j"], predicted_allocation=row,
                relaxed_prediction=relaxed[job_id], weight=weights[agent],
                allocation_at=t.now, backhaul_radio=radio, relaxation_gap=comparison["absolute_gap"])
            if job["executed_x"]:
                job.update(stage="computing", compute_arrived_at=t.now)
                t.compute_pools["drone"].submit(job_id, job["required_cycles"], weights[agent], job_id)
            else:
                job["stage"] = "backhaul"
                self._send(agent, "paper_cloud_input", payload, ("satellite_up",),
                    generated_at=job["source_time"], version=job["version"], pad=True)

    def _computed(self, completed):
        t = self.transport
        job_id = completed.job_id
        job = self.jobs[job_id]
        job.update(stage="computed", compute_completed_at=t.now,
            compute_service_seconds=completed.service_seconds,
            drone_compute_energy_j=completed.compute_energy_j,
            compute_elapsed_seconds=t.now-completed.arrived_at,
            batch_wait_seconds=job["allocation_at"]-job["access_arrived_at"])
        report = dict(job["report"])
        route = self.pathfinder(self.grid, tuple(report["position"]), tuple(report["goal"]))
        report["path"] = list(route or [tuple(report["position"])])
        report["distance"] = float(len(route)-1) if route else float(self.grid.size)
        hops = ("access_down",) if job["executed_x"] else ("satellite_down", "access_down")
        common = dict(job_id=job_id, source_time=job["source_time"])
        self._send(job["agent"], "paper_guidance", dict(common, path=report["path"], distance=report["distance"]),
            hops, version=job["version"])
        state = self._state(report)
        for receiver in self.goals:
            self._send(receiver, "paper_state", dict(common, peer=job["agent"], state=state),
                hops, version=job["version"])

    def _refresh_oracles(self, agents, now):
        fresh = {a.id: self._report(a) for a in agents}
        for agent in agents:
            if self.config.profile == "perfect" or "task" not in self.config.stale_information:
                self.caches[agent.id]["task"].accept(now,
                    Guidance(list(fresh[agent.id]["path"]), fresh[agent.id]["distance"]),
                    self.sequence, {agent.id: now})
            if self.config.profile == "perfect" or "priorities" not in self.config.stale_information:
                for peer in self.states[agent.id]:
                    self.states[agent.id][peer].accept(now, self._state(fresh[peer]), self.sequence, {peer: now})
                self._sync(agent.id)

    def update(self, agents, now):
        self.decision = None
        self.truth = {a.id: self._report(a) for a in agents}
        self.positions = {a.id: tuple(a.position) for a in agents}
        for agent in agents:
            self.position_history[agent.id].append((now, tuple(agent.position)))
        self.transport.advance(now, self._receive)
        self.sequence += 1
        self._refresh_oracles(agents, now)
        if self.config.profile != "perfect":
            for agent in agents:
                report = self._report(agent)
                xy = self.uav.positions_to_xy_m([tuple(agent.position)])[0]
                horizontal = float(np.linalg.norm(xy-self.uav.position[:2]))
                loss = float(self.uav.model.path_loss_db(self.uav.position[2], horizontal))
                radio = access_budget(loss, self.parameters, self.config.access_uplink_mbps*1e6)
                self.transport.agent_rate_caps[(agent.id, "access_up")] = radio["applied_bps"]
                job_id = f"{self.sequence}:{agent.id}"
                payload = dict(job_id=job_id, report=report, source_time=now,
                    access_radio=radio, reported_cache_source_time=min(
                        self.states[agent.id][agent.id].source_timestamps.values()))
                self.jobs[job_id] = dict(job_id=job_id, agent=agent.id, report=report,
                    version=self.sequence, source_time=now, stage="access",
                    input_bits=self.parameters.task_input_bits,
                    cycles_per_bit=self.parameters.cycles_per_bit,
                    required_cycles=self.parameters.task_input_bits*self.parameters.cycles_per_bit,
                    access_radio=radio, robot_energy_j=0.0, backhaul_energy_j=0.0, drone_compute_energy_j=0.0,
                    access_wire_bits=0, backhaul_wire_bits=0, response_all_recipients_wire_bits=0,
                    **{"uplink_"+k: 0.0 for k in ("transmission_seconds", "queue_seconds", "propagation_seconds",
                                                    "interruption_seconds", "guard_seconds")})
                self._send(agent.id, "paper_input", payload, ("access_up",),
                    generated_at=now, version=self.sequence, pad=True)
        self.transport.advance(now, self._receive)
        for agent in agents:
            self.caches[agent.id]["task"].value.consume(agent.position)

    def prepare_decision(self, agents, cycle_start, deadline_seconds):
        self.transport.advance(cycle_start+deadline_seconds, self._receive)
        self.sequence += 1
        self._refresh_oracles(agents, self.transport.now)
        details = {}
        for agent in agents:
            details[agent.id] = {}
            for field, cache in self.caches[agent.id].items():
                source = min(cache.source_timestamps.values())
                details[agent.id][field] = dict(packet_source_time=cache.timestamp,
                    oldest_source_time=source, age_seconds=cache.age(self.transport.now),
                    response_missed_deadline=source < cycle_start-1e-9,
                    source_from_previous_cycle=source < cycle_start-1e-9)
            self.caches[agent.id]["task"].value.consume(agent.position)
        self.decision = dict(cycle_start=cycle_start, at=self.transport.now,
            budget_seconds=deadline_seconds, initial_briefing=cycle_start == 0, inputs=details)

    def snapshot(self):
        result = super().snapshot()
        result.update(paper_schema=1,
            consistency=consistency_snapshot(self.positions, self.truth, self.states, self.transport.now,
                self.project.map.fov_radius, self.project.map.communicated_path_steps,
                self.parameters.age_reference_seconds),
            compute_pools={name: pool.snapshot() for name, pool in self.transport.compute_pools.items()},
            paper_jobs=[{key: job[key] for key in ("job_id", "agent", "stage", "source_time",
                "required_cycles", "robot_energy_j", "backhaul_energy_j")}
                for job in self.jobs.values() if "delivered_at" not in job],
            completed_flags={a: r["completed"] for a, r in self.truth.items()},
            coordination_sources={receiver: {peer: dict(generation_time=c.timestamp,
                source_time=min(c.source_timestamps.values()), version=c.version)
                for peer, c in states.items()} for receiver, states in self.states.items()})
        return result

    def task_records(self):
        return [dict(job) for job in self.jobs.values()]

    def energy_bounds(self):
        """Completed airtime is exact; unfinished chunks expose an honest bound."""
        lower = sum(j["robot_energy_j"]+j["backhaul_energy_j"] for j in self.jobs.values())
        pending = 0.0
        for _, _, kind, value in self.transport.events:
            if kind != "transmission_observed":
                continue
            _, link, details = value
            power = (self.parameters.robot_power_w if link == "access_up" else
                     self.parameters.drone_power_w if link == "satellite_up" else 0.0)
            pending += power*details["transmission_seconds"]
        compute = self.transport.compute_pools["drone"].energy()
        return dict(radio_energy_lower_j=lower, radio_energy_upper_j=lower+pending,
                    unfinished_airtime_energy_bound_j=pending, drone_compute_energy_j=compute,
                    modeled_energy_lower_j=lower+compute, modeled_energy_upper_j=lower+pending+compute)
