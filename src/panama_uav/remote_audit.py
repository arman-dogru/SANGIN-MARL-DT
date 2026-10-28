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
from .remote_control import DELTAS, cached_path_action, completed_action, conflicts, resolve_delivered


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
    fallback_diagnostics = {}
    for diagnostic in trace.get("oracle_events",[]):
        if diagnostic["event"] == "fallback_shadow":
            fallback_diagnostics.setdefault(diagnostic["causal_anchor_event_id"],[]).append(diagnostic)
    controlled = settings.get("controlled_staleness")
    extra_decisions,last_round = {},(-1,-1)
    executions,bundles,bundle_inferences,delayed={},{},{},{}
    if settings.get('discrete_command_protocol'):
        from .bundled_actions import DelayedCommandInbox
        delayed={int(r):DelayedCommandInbox(settings['seconds_per_step'],settings['cutoff_seconds'])
            for r,v in settings['speed_assignments'].items() if v['speed_class']=='slow'}
    reversal_records={(e['cycle'],e['substep'],e['first'],e['second']):e
        for e in trace.get('oracle_events',[]) if e['event']=='priority_reversal_event'}
    pending_reversals={};expected_durations=[]
    def close_reversal(key,at,reason,censored):
        record=pending_reversals.pop(key)
        expected_durations.append(dict(event='priority_reversal_cache_duration',time=at,
            reversal_time=record['time'],receiver=key[0],peer=key[1],duration_seconds=at-record['time'],
            censored=censored,reason=reason,physically_relevant=record['physically_relevant']))

    def check_reversal_caches(at):
        for (receiver,peer),record in list(pending_reversals.items()):
            if (caches[receiver][peer].value['priority']<caches[receiver][receiver].value['priority'])==record['physical_peer_higher']:
                close_reversal((receiver,peer),at,'cached_order_matches',False)
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
                check_reversal_caches(now)
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
        elif kind == 'bundle_inference':
            from types import SimpleNamespace
            from .bundled_actions import second_input
            from .remote import RemoteDTCommunication
            parent=by_id[event['parent_inference_event']];robot=event['robot_id']
            require(parent['event']=='remote_inference' and parent['job_id']==event['job_id'] and parent['time']==now,
                'Second inference does not belong to its single remote task')
            speed=settings['speed_assignments'].get(str(robot),settings['speed_assignments'].get(robot))
            require(speed['speed_class']=='fast','Non-fast task received two remote actions')
            own_ref=parent['dt_versions'][str(robot)]['state_event'];report=reports[states[own_ref].report_id]
            grid=np.asarray(trace['grid']);builder=SimpleNamespace(grid=grid,pathfinder=astar_path)
            obs,sources,versions,priorities,prediction=second_input(report,replicas[event['location']],grid,parent['action'],
                lambda r,t,v:RemoteDTCommunication._state_for_report(builder,r,t,v),now,settings)
            versions[robot]['state_event']=own_ref
            for actual,expected,label in ((event['sources'],sources,'sources'),(event['dt_versions'],versions,'versions'),
                (event['dt_priorities'],priorities,'priorities'),(event['prediction'],prediction,'predicted DT state')):
                require(digest(actual)==digest(expected),'Second inference '+label+' mismatch')
            require(observation_hash(obs)==event['observation_hash']==observation_hash(event['observation']),
                'Second remote observation mismatch')
            require(event['first_action']==parent['action'] and event['required_cycles_unchanged'], 'Remote bundle workload mismatch')
            if inference is not None:
                require(int(inference(event['location'],robot,obs))==event['action'],'Second frozen DQN action mismatch')
            bundle_inferences[event['job_id']]=event
        elif kind == 'action_bundle_arrived':
            second=bundle_inferences[event['job_id']];bundle=event['bundle']
            expected=dict(actions=[second['first_action'],second['action']],second_inference_event=second['event_id'],
                predicted_position=second['prediction']['predicted_position'],sources=second['sources'],
                dt_versions=second['dt_versions'],dt_priorities=second['dt_priorities'])
            require(digest(bundle)==digest(expected),'Delivered action bundle differs from remote inference')
            previous_event=by_id[f'rdt:{index-1}']
            require(previous_event['event']=='command_arrived' and previous_event['packet_id']==event['packet_id']
                and previous_event['command']['job_id']==event['job_id'] and event['arrived_at']==now,
                'Action bundle did not arrive in the original command packet')
            bundles[event['job_id']]=bundle
        elif kind == 'delayed_command_arrived':
            command=ActionCommand(**event['command'])
            accepted,reason=delayed[event['robot_id']].accept(command,now,bundles.get(command.job_id))
            require((accepted,reason)==(event['accepted'],event['reason']) and event['execution_cycle']==command.cycle+1,
                'Delayed command acceptance mismatch')
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
            if delivery is None and not event["completed_local_control"] and settings.get("missed_command_fallback") == "cached_astar":
                require(event.get("cached_fallback", False), "Cached fallback activation missing")
            if delivery:
                command, arrival = delivery
                require(digest(command.payload()) == digest(event["command"]), "Executed command mismatch")
                require(source_min(command.sources) <= command.generated_at <= arrival+1e-12 <= now+2e-12,
                        "Executed-command causality violated")
                if not event["completed_local_control"]:
                    require(command.action == event["action"], "Robot bypassed delivered action")
            elif not event["completed_local_control"] and not event.get("cached_fallback"):
                require(event["action"] == 4 and event["command_cutoff_missed"], "Timeout did not execute WAIT")
            cached_fallback = event.get("cached_fallback", False)
            if cached_fallback:
                require(settings.get("missed_command_fallback") == "cached_astar"
                        and delivery is None and not event["completed_local_control"]
                        and event["command_cutoff_missed"], "Invalid cached fallback activation")
            if event["completed_local_control"] or cached_fallback:
                grid, radius = np.asarray(trace["grid"]), settings["radius"]
                row, col = positions[robot]
                obstacles = np.pad(grid, radius, constant_values=1)[row:row+2*radius+1, col:col+2*radius+1]
                occupancy = np.zeros_like(obstacles)
                for peer, point in positions.items():
                    dr, dc = point[0]-row, point[1]-col
                    if peer != robot and max(abs(dr), abs(dc)) <= radius:
                        occupancy[dr+radius, dc+radius] = 1
                report = PhysicalReport(robot, cycle, now, (row, col), tuple(physical[robot]["goal"]), not cached_fallback,
                    0., obstacles.tolist(), occupancy.tolist(), f"report:{cycle}:{robot}")
                if cached_fallback:
                    action, sources, details = cached_path_action(report, caches[robot], settings["horizon"])
                    require(digest(details) == digest(event["local_control"]), "Cached fallback input replay mismatch")
                    require(event["resolution_priority"] == caches[robot][robot].value["priority"],
                            "Cached fallback used an unavailable priority")
                    require(event["fallback_source_min"] == source_min(sources)
                            and abs(event["fallback_source_age_seconds"]-(now-source_min(sources))) < 1e-9,
                            "Cached fallback source age mismatch")
                else:
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
            actual_decisions={r:dict(d) for r,d in decisions.items()}
            halfrate=settings.get('discrete_command_protocol')=='dt_bundle_halfrate_slow_v1'
            if halfrate:
                # leo_study: off-step slow robots have no decision at this cycle.
                from .frequency_motion import active_step
                speeds={int(r):v for r,v in settings['speed_assignments'].items()}
                actual_decisions={r:d for r,d in actual_decisions.items() if d.get('cycle')==event['cycle']}
            if settings.get('discrete_command_protocol') and not halfrate:
                for robot,inbox in delayed.items():
                    candidate=inbox.peek(event['cycle'],now)
                    actual_decisions[robot]=dict(command=candidate[0].payload() if candidate else None,
                        cached_fallback=candidate is None and event['cycle']>0 and settings['missed_command_fallback']=='cached_astar')
            for robot, decision in actual_decisions.items():
                command = decision["command"]
                if command:
                    views[robot] = dict(priorities={int(k): v for k, v in command["dt_priorities"].items()},
                        states={int(k): by_id[v["state_event"]]["state"] for k, v in command["dt_versions"].items()})
                elif decision.get("cached_fallback"):
                    views[robot] = dict(priorities={p:c.value["priority"] for p,c in caches[robot].items()},
                        states={p:c.value for p,c in caches[robot].items()})
            snapshot = consistency_snapshot(positions, physical, caches, now,
                settings["radius"], settings["horizon"], settings["age_reference"], replicas["cloud"].states, views, metadata,
                schema_version=event.get("consistency_schema", 1),
                dt_references={r: (d["command"]["location"] if d.get("command") else "drone",
                    replicas[d["command"]["location"] if d.get("command") else "drone"].states)
                    for r,d in ({p: actual_decisions.get(p, {}) for p in positions} if halfrate else actual_decisions).items()}
                    if settings.get("dt_reference") else None)
            if halfrate:
                from .consistency import summarize_pairs
                kept=[q for q in snapshot["pairs"] if active_step(speeds[q["receiver"]],event["cycle"])]
                if len(kept)!=len(snapshot["pairs"]):
                    snapshot=summarize_pairs(kept,physical,dt_references={p:None for p in positions})
            require(digest(snapshot) == event["consistency_sha256"], "Consistency/AoI replay mismatch")
            require(digest({k: v for k, v in snapshot.items() if k != "pairs"}) == digest(event["summary"]), "Consistency summary mismatch")
            if snapshot.get("consistency_schema", 1) < CONSISTENCY_SCHEMA:
                # Old evidence keeps its original checksums. New observational
                # metrics are derived from the same reconstructed epoch only.
                snapshot = consistency_snapshot(positions, physical, caches, now,
                    settings["radius"], settings["horizon"], settings["age_reference"], replicas["cloud"].states, views, metadata)
            consistencies.append(snapshot)
        elif kind == 'extra_local_decision_event':
            robot,cycle=event['robot_id'],event['cycle']
            speed=settings['speed_assignments'].get(str(robot),settings['speed_assignments'].get(robot))
            require(speed['decisions_per_step']==2 and last_round==(cycle,0),'Extra decision is not a fast second round')
            grid,radius=np.asarray(trace['grid']),settings['radius'];row,col=positions[robot]
            obstacles=np.pad(grid,radius,constant_values=1)[row:row+2*radius+1,col:col+2*radius+1]
            occupancy=np.zeros_like(obstacles)
            for peer,point in positions.items():
                dr,dc=point[0]-row,point[1]-col
                if peer!=robot and max(abs(dr),abs(dc))<=radius:occupancy[dr+radius,dc+radius]=1
            report=PhysicalReport(robot,cycle,now,(row,col),tuple(physical[robot]['goal']),physical[robot]['completed'],
                0.,obstacles.tolist(),occupancy.tolist(),f'report:{cycle}:{robot}')
            completed=physical[robot]['completed']
            priority=grid.size+robot*1e-6 if completed else caches[robot][robot].value['priority']
            action,sources,details=(completed_action(report,caches[robot],priority) if completed
                else cached_path_action(report,caches[robot],settings['horizon']))
            require(action==event['action'] and digest(sources)==digest(event['local_sources'])
                and digest(details)==digest(event['local_control']) and priority==event['resolution_priority'],
                'Extra local action did not use available sensing and caches')
            require(event['command'] is None and not event['command_cutoff_missed'],'Extra local action created a command deadline')
            extra_decisions[robot]=event
        elif kind == 'execution_decision_event':
            from .redesign_audit import replay_execution
            executions[event['robot_id']]=replay_execution(event,positions,physical,caches,decisions,executions,delayed,bundles,settings,np.asarray(trace['grid']))
        elif kind == 'frequency_motion_event':
            from .redesign_audit import replay_frequency
            phase=event['cycle'],event['substep']
            require(phase>(last_round) and (phase[1]==0 or last_round==(phase[0],0)), 'Invalid movement-round ordering')
            prior_physical=physical
            positions,physical=replay_frequency(event,positions,decisions,executions,settings,np.asarray(trace['grid']),physical)
            for i,first in enumerate(sorted(physical)):
                for second in sorted(physical)[i+1:]:
                    a,b=first,second
                    old_order=prior_physical[a]['priority']<prior_physical[b]['priority']
                    new_order=physical[a]['priority']<physical[b]['priority']
                    if old_order==new_order or prior_physical[a]['completed'] or prior_physical[b]['completed']:continue
                    record=reversal_records.pop((phase[0],phase[1],a,b),None)
                    require(record is not None,'Missing physical priority reversal')
                    require(record['time']==now and record['old_order']==int(old_order) and record['new_order']==int(new_order),
                        'Priority reversal timing or ordering mismatch')
                    completion=physical[a]['completed'] or physical[b]['completed']
                    require(record['cause']==('goal_completion' if completion else 'movement'),'Priority reversal cause mismatch')
                    cached_orders=[dict(receiver=r,peer=p,
                        cached_peer_higher=caches[r][p].value['priority']<caches[r][r].value['priority'],
                        stale=(caches[r][p].value['priority']<caches[r][r].value['priority'])!=(physical[p]['priority']<physical[r]['priority']),
                        self_source_time=source_min(caches[r][r].source_timestamps),peer_source_time=source_min(caches[r][p].source_timestamps))
                        for r,p in ((a,b),(b,a))]
                    require(digest(cached_orders)==digest(record['cached_orders']),'Reversal cached ordering provenance mismatch')
                    if completion:
                        for key in ((a,b),(b,a)):
                            if key in pending_reversals:close_reversal(key,now,'goal_completion',True)
                        continue
                    for receiver,peer in ((a,b),(b,a)):
                        key=receiver,peer
                        if key in pending_reversals:close_reversal(key,now,'superseded',True)
                        pending_reversals[key]=dict(record,physical_peer_higher=physical[peer]['priority']<physical[receiver]['priority'])
            check_reversal_caches(now)
            last_round=phase
            failure = event["failure_reason"]
        for diagnostic in fallback_diagnostics.pop(event["event_id"],[]):
            require(diagnostic["time"] == now,"Fallback diagnostic is not anchored at its cutoff")
            report = PhysicalReport(**diagnostic["local_report"])
            robot = diagnostic["robot_id"]
            require(tuple(report.position) == positions[robot] and report.cycle == diagnostic["cycle"]
                    and report.sensed_at == now,"Fallback diagnostic local pose or timestamp mismatch")
            grid,radius = np.asarray(trace["grid"]),settings["radius"]
            row,col = positions[robot]
            obstacles = np.pad(grid,radius,constant_values=1)[row:row+2*radius+1,col:col+2*radius+1]
            occupancy = np.zeros_like(obstacles)
            for peer,point in positions.items():
                dr,dc = point[0]-row,point[1]-col
                if peer != robot and max(abs(dr),abs(dc)) <= radius:occupancy[dr+radius,dc+radius] = 1
            require(np.array_equal(report.obstacles,obstacles) and np.array_equal(report.occupancy,occupancy),
                    "Fallback diagnostic accessed incorrect local sensing")
            action,sources,details = cached_path_action(report,caches[diagnostic["robot_id"]],settings["horizon"])
            require(action == diagnostic["stale_action"] and digest(sources) == digest(diagnostic["stale_sources"])
                    and digest(details) == digest(diagnostic["stale_details"]), "Stale fallback diagnostic replay mismatch")
            fresh = {}
            for peer,values in physical.items():
                cache = InformationCache()
                value = {k:values[k] for k in ("position","path","priority","completed")}
                value["path"] = value["path"][:settings["horizon"]+1]
                cache.accept(now,value,diagnostic["cycle"],{peer:now});fresh[peer] = cache
            action,sources,details = cached_path_action(report,fresh,settings["horizon"])
            require(digest({p:c.value for p,c in fresh.items()}) == digest(diagnostic["fresh_states"])
                and action == diagnostic["fresh_action"] and digest(details) == digest(diagnostic["fresh_details"]),
                "Fresh fallback diagnostic truth or action mismatch")
            require(diagnostic["stale_vs_fresh_disagreement"] == (diagnostic["stale_action"] != action),
                "Fallback disagreement mismatch")
            if settings.get('discrete_command_protocol'):
                require(tuple(report.goal)==tuple(physical[robot]['goal']) and report.completed==physical[robot]['completed']
                    and report.initial_distance==reports[f'report:0:{robot}'].initial_distance,
                    'Diagnostic changed own goal, completion or observation normalizer')
                fresh_replica=DTReplica('cloud');fresh_states={}
                for peer,values in physical.items():
                    path=astar_path(grid,positions[peer],tuple(values['goal']))
                    state=DTState(peer,positions[peer],tuple(values['goal']),path or [positions[peer]],values['priority'],
                        values['completed'],now,diagnostic['cycle'],{peer:now},f"report:{diagnostic['cycle']}:{peer}")
                    fresh_states[peer]=state;fresh_replica.accept(state,now,f"extra-oracle:{diagnostic['cycle']}:{peer}")
                expected_obs=remote_observation(report,fresh_states[robot],fresh_replica,settings['radius'],settings['waypoints'],
                    settings['horizon'],settings['map_extent'])[0]
                if diagnostic['substep']==0:
                    oracle=next(e for e in trace['oracle_events'] if e['event']=='decision_action_shadow'
                        and e['robot_id']==robot and e['cycle']==diagnostic['cycle'])
                    require(observation_hash(expected_obs)==oracle['observation_hash'],'First-round fresh oracle is not current physical state')
                else:
                    require(observation_hash(expected_obs)==observation_hash(diagnostic['fresh_dqn_observation']),
                        'Second-round fresh oracle is not current physical state')
                if inference is not None:
                    require(int(inference('cloud',robot,expected_obs))==diagnostic['fresh_dqn_action'],'Execution oracle action mismatch')
                require(diagnostic['actual_action']==event['action'] and diagnostic['action_source']==event['action_source']
                    and diagnostic['invoked']==bool(event.get('cached_fallback')),
                    'Diagnostic does not describe its actual executed selection')
        if kind == "motion_event":
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

    require(not fallback_diagnostics,"Fallback diagnostic has an unknown causal anchor")
    require(not reversal_records,'Unsubstantiated priority reversal diagnostic')
    for key in list(pending_reversals):close_reversal(key,trace['frames'][-1]['time'],'mission_observation_end',True)
    recorded_durations=[e for e in trace.get('oracle_events',[]) if e['event']=='priority_reversal_cache_duration']
    require(digest(recorded_durations)==digest(expected_durations),'Priority reversal duration/censoring replay mismatch')

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
