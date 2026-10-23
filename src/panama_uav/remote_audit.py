"""Deterministic event replay for the versioned remote-DT trace schema.

Rebuilds delivered stores, source AoI, directed conflicts, commands, routes and
physical motion. Optional frozen inference verifies actions from reconstructed
observations. This is an evidence audit, not a second communication simulator.
"""
from dataclasses import replace
import gzip
import hashlib
import json

import numpy as np

from .allocation import AllocationSnapshot, score_allocation
from .consistency import CONSISTENCY_SCHEMA, consistency_snapshot
from .environment import astar_path
from .information import InformationCache
from .remote_state import (REMOTE_SCHEMA, ActionCommand, CommandInbox, DTReplica, DTState,
                           PhysicalReport, observation_hash, remote_observation, source_min)
from .remote_control import DELTAS, completed_action, conflicts, resolve_delivered


def digest(value):
    # Recorded events already underwent JSON round-trip normalization in
    # RemoteDTCommunication._record. Rebuilt dictionaries may still have integer
    # robot IDs; normalize them before sorting (10 sorts differently as a key
    # of type int versus str). Existing string-key archive checksums are stable.
    normalized=json.loads(json.dumps(value,allow_nan=False))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def replay_trace(trace, inference=None):
    if trace.get("remote_schema") != REMOTE_SCHEMA:
        raise ValueError("A versioned remote-DT episode archive is required")
    settings = trace["remote_settings"]
    replicas = {name: DTReplica(name) for name in ("drone", "cloud")}
    caches, metadata, reports, states, inboxes = {}, {}, {}, {}, {}
    physical, positions, decisions, assignments, consistencies = {}, {}, {}, {}, []
    previous, now, failure, last_cycle = "0"*64, 0.0, None, -1
    by_id, inference_count, verified_actions = {}, 0, 0
    controlled = settings.get("controlled_staleness")
    if controlled:
        from .staleness_state import ReplicaHistory
        history = ReplicaHistory()

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    for index, event in enumerate(trace["remote_events"]):
        require(event["schema"] == REMOTE_SCHEMA and event["event_id"] == f"rdt:{index}", "Invalid event identity")
        require(event["previous_checksum"] == previous, "Event chain predecessor mismatch")
        require(digest({k: v for k, v in event.items() if k != "checksum"}) == event["checksum"], "Event checksum mismatch")
        require(event["time"] >= now-1e-12, "Event time went backwards")
        now, previous = event["time"], event["checksum"]
        by_id[event["event_id"]] = event
        kind = event["event"]
        if kind == "dt_state_event":
            state = DTState(**event["state"])
            states[event["event_id"]] = state
            if event.get("bootstrap") or event.get("ideal_reference"):
                replicas[event["location"]].accept(state, event["available_at"], event["event_id"])
        elif kind == "dt_state_acceptance":
            source = by_id[event["state_event"]]
            accepted = replicas[source["location"]].accept(states[event["state_event"]], now, event["state_event"])
            require(accepted == event["accepted"], "DT acceptance replay mismatch")
        elif kind == "replica_sync_arrived":
            accepted = replicas[event["location"]].accept(DTState(**event["state"]), now, event["state_event"])
            require(accepted == event["accepted"], "Replica synchronization replay mismatch")
        elif kind == "received_state_event":
            receiver, peer = event["receiver"], event["peer"]
            cache = caches.setdefault(receiver, {}).setdefault(peer, InformationCache())
            sources = event["sources"]
            require(source_min(sources) <= event["generated_at"]+1e-12 <= event["arrived_at"]+2e-12,
                    "Received-state causality violated")
            accepted = False
            if not cache.source_timestamps or source_min(sources) >= source_min(cache.source_timestamps):
                accepted = cache.accept(event["generated_at"], event["state"], event["version"], sources)
            require(accepted == event["accepted"], "Robot cache acceptance replay mismatch")
            if accepted:
                metadata.setdefault(receiver, {})[peer] = event
        elif kind == "robot_state_event":
            report = PhysicalReport(**event["report"])
            reports[report.report_id] = report
            physical[report.robot_id] = event["physical"]
            positions[report.robot_id] = tuple(report.position)
        elif kind == "remote_inference":
            require(event["source_min"] <= now+1e-12, "Inference depends on future sensing")
            robot = event["robot_id"]
            own_ref = event["dt_versions"][str(robot)]["state_event"]
            state = states[own_ref]
            report = reports[state.report_id]
            full_path = astar_path(np.asarray(trace["grid"]), tuple(report.position), tuple(report.goal))
            own = replace(state, path=full_path or [tuple(report.position)])
            replica = (history.view(event["location"], robot, now, controlled["peer_delay_steps"])
                       if controlled else replicas[event["location"]])
            observation, sources, versions, priorities = remote_observation(report, own, replica,
                settings["radius"], settings["waypoints"], settings["horizon"], settings["map_extent"])
            versions[robot]["state_event"] = own_ref
            require(observation_hash(observation) == event["observation_hash"], "Rebuilt observation differs from inference input")
            require(observation_hash(event["observation"]) == event["observation_hash"], "Recorded observation hash mismatch")
            require(digest(sources) == digest(event["sources"]), "Inference source provenance mismatch")
            require(digest(versions) == digest(event["dt_versions"]), "Inference used unavailable DT state")
            require(digest(priorities) == digest(event["dt_priorities"]), "Inference priority replay mismatch")
            if inference is not None:
                require(int(inference(event["location"], robot, observation)) == event["action"], "Frozen DQN action mismatch")
                verified_actions += 1
            inference_count += 1
        elif kind == "command_arrived":
            command = ActionCommand(**event["command"])
            accepted, reason = inboxes.setdefault(event["robot_id"], CommandInbox()).accept(command, now)
            require((accepted, reason) == (event["accepted"], event["reason"]), "Command acceptance replay mismatch")
        elif kind == "decision_event":
            robot, cycle = event["robot_id"], event["cycle"]
            if cycle != last_cycle:
                decisions, last_cycle = {}, cycle
            delivery = inboxes.setdefault(robot, CommandInbox()).decision(cycle, event["cutoff_time"])
            require((delivery is not None) == (event["command"] is not None), "Decision used a command absent from inbox")
            if delivery:
                command, arrival = delivery
                require(digest(command.payload()) == digest(event["command"]), "Executed command mismatch")
                require(source_min(command.sources) <= command.generated_at <= arrival+1e-12 <= now+2e-12,
                        "Executed-command causality violated")
                if not event["completed_local_control"]:
                    require(command.action == event["action"], "Robot bypassed delivered action")
            elif not event["completed_local_control"]:
                require(event["action"] == 4 and event["command_cutoff_missed"], "Timeout did not execute WAIT")
            if event["completed_local_control"]:
                grid, radius = np.asarray(trace["grid"]), settings["radius"]
                row, col = positions[robot]
                obstacles = np.pad(grid, radius, constant_values=1)[row:row+2*radius+1, col:col+2*radius+1]
                occupancy = np.zeros_like(obstacles)
                for peer, point in positions.items():
                    dr, dc = point[0]-row, point[1]-col
                    if peer != robot and max(abs(dr), abs(dc)) <= radius:
                        occupancy[dr+radius, dc+radius] = 1
                report = PhysicalReport(robot, cycle, now, (row, col), tuple(physical[robot]["goal"]), True,
                    0., obstacles.tolist(), occupancy.tolist(), f"report:{cycle}:{robot}")
                action, sources, _ = completed_action(report, caches[robot], grid.size+robot*1e-6)
                require(action == event["action"] and digest(sources) == digest(event["local_sources"]),
                        "Local completion controller replay mismatch")
            decisions[robot] = event
        elif kind == "allocation_event":
            snapshot = AllocationSnapshot.from_dict(event["snapshot"])
            score_allocation(snapshot, event["assignments"])
            assignments[event["job_id"]] = next(r for r in event["assignments"] if r["task_id"] == event["job_id"])
        elif kind == "consistency_event":
            views = {}
            for robot, decision in decisions.items():
                command = decision["command"]
                if command:
                    views[robot] = dict(priorities={int(k): v for k, v in command["dt_priorities"].items()},
                        states={int(k): by_id[v["state_event"]]["state"] for k, v in command["dt_versions"].items()})
            snapshot = consistency_snapshot(positions, physical, caches, now,
                settings["radius"], settings["horizon"], settings["age_reference"], replicas["cloud"].states, views, metadata,
                schema_version=event.get("consistency_schema", 1))
            require(digest(snapshot) == event["consistency_sha256"], "Consistency/AoI replay mismatch")
            require(digest({k: v for k, v in snapshot.items() if k != "pairs"}) == digest(event["summary"]), "Consistency summary mismatch")
            if snapshot.get("consistency_schema", 1) < CONSISTENCY_SCHEMA:
                # Old evidence keeps its original checksums. New observational
                # metrics are derived from the same reconstructed epoch only.
                snapshot = consistency_snapshot(positions, physical, caches, now,
                    settings["radius"], settings["horizon"], settings["age_reference"], replicas["cloud"].states, views, metadata)
            consistencies.append(snapshot)
        elif kind == "motion_event":
            require({int(k): tuple(v) for k, v in event["before"].items()} == positions, "Motion starts from unknown physical state")
            require(all(event["actions"][str(robot)] == decision["action"] for robot, decision in decisions.items()), "Motion differs from cutoff decision")
            grid = np.asarray(trace["grid"])
            targets = {}
            for robot, point in positions.items():
                dr, dc = DELTAS[event["actions"][str(robot)]]
                target = point[0]+dr, point[1]+dc
                targets[robot] = target if (0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1]
                    and not grid[target]) else point
            collision = conflicts(positions, targets)
            if collision and settings["on_collision"] == "terminate":
                expected_positions = positions
            else:
                expected_positions = resolve_delivered(positions, targets,
                    {r: d["resolution_priority"] for r, d in decisions.items()})
            require(expected_positions == {int(k): tuple(v) for k, v in event["after"].items()},
                    "Physical motion/resolution replay mismatch")
            positions = {int(k): tuple(v) for k, v in event["after"].items()}
            physical = {int(k): v for k, v in event["physical"].items()}
            failure = event["failure_reason"]
        if controlled:
            history.observe(event, by_id)

    outcome = trace.get("outcome")
    if outcome:
        require(positions == {int(k): tuple(v) for k, v in outcome["final_positions"].items()}, "Outcome position mismatch")
        require(failure == outcome["failure_reason"], "Outcome failure mismatch")
    vertex_pairs = sum(s["vertex_relevant_pairs"] for s in consistencies)
    vertex_errors = sum(s["vertex_e2e_errors"] for s in consistencies)
    vertex_aoi = sum(s["vertex_aoi_error_sum"] for s in consistencies)
    return dict(schema=REMOTE_SCHEMA, derived_consistency_schema=CONSISTENCY_SCHEMA,
        status="passed", events=len(by_id), final_checksum=previous,
        rebuilt_inferences=inference_count, model_actions_verified=verified_actions,
        decisions=sum(e["event"] == "decision_event" for e in trace["remote_events"]),
        allocations=len(assignments), consistency_decisions=len(consistencies),
        relevant_pairs=sum(s["relevant_pairs"] for s in consistencies),
        fov_errors=sum(s["fov_errors"] for s in consistencies),
        missed_physical_conflicts=sum(s["missed_physical_conflicts"] for s in consistencies),
        ordering_pairs_verified=sum(s["ordering_pairs_verified"] for s in consistencies),
        vertex_relevant_pairs=vertex_pairs, vertex_e2e_errors=vertex_errors, vertex_aoi_error_sum=vertex_aoi,
        xi_vertex_e2e=vertex_errors/max(1, vertex_pairs), xi_vertex_aoi=vertex_aoi/max(1, vertex_pairs),
        missed_vertex_conflicts=sum(s["missed_vertex_conflicts"] for s in consistencies),
        false_positive_vertex_conflicts=sum(s["false_positive_vertex_conflicts"] for s in consistencies),
        assignments=assignments, final_positions=positions, failure_reason=failure,
        pair_consistency=consistencies)


def audit_archive(path, inference=None, expected_model_sha256=None):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    if expected_model_sha256 is not None:
        if trace.get("provenance", {}).get("model", {}).get("sha256") != expected_model_sha256:
            raise ValueError("Audit checkpoint does not match the archived model identity")
    result = replay_trace(trace, inference)
    result["model_identity_verified"] = expected_model_sha256 is not None
    return result
