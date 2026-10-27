"""Discrete-frequency robot inference experiments; no learning or motion clock."""
from dataclasses import asdict, replace
import hashlib
import json
import math
from time import perf_counter

import numpy as np

from .consistency import consistency_snapshot
from .environment import ExperimentEnvironment
from .config import ExperimentMode
from .information import InformationCache
from .inference_controls import inference_class
from .frequency_motion import FrequencyMotion, MotionSettings
from .remote_control import cached_path_action
from .remote_state import PhysicalReport


TREATMENTS = {
    "A0": dict(heterogeneous=False, control_interval_seconds=1., command_deadline_seconds=.25, fallback="wait"),
    "A1": dict(heterogeneous=False, control_interval_seconds=.35, command_deadline_seconds=.25, fallback="wait"),
    "A2": dict(heterogeneous=True, control_interval_seconds=.35, command_deadline_seconds=.25, fallback="wait"),
    "A3": dict(heterogeneous=False, control_interval_seconds=.35, command_deadline_seconds=.25, fallback="cached_astar"),
    "A4": dict(heterogeneous=True, control_interval_seconds=.35, command_deadline_seconds=.25, fallback="cached_astar"),
    "A5": dict(heterogeneous=True, control_interval_seconds=1., command_deadline_seconds=.25, fallback="cached_astar"),
    "A6": dict(heterogeneous=True, control_interval_seconds=.35, command_deadline_seconds=.15, fallback="cached_astar"),
    "A7": dict(heterogeneous=True, control_interval_seconds=.35, command_deadline_seconds=.30, fallback="cached_astar"),
}


def settings_for(treatment, horizon=250.):
    return MotionSettings(treatment=treatment,horizon_seconds=horizon,**TREATMENTS[treatment])


def configured(base, settings, latency=1., energy=1.):
    return replace(base, communication=replace(base.communication,seconds_per_step=settings.control_interval_seconds),
        resource_allocation=replace(base.resource_allocation,
            decision_budget_seconds=settings.command_deadline_seconds,
            command_cutoff_seconds=settings.command_deadline_seconds,
            task_deadline_seconds=settings.command_deadline_seconds,
            enforce_task_deadlines=False,missed_command_fallback=settings.fallback,
            latency_coefficient=latency,energy_coefficient=energy))


def action_source(context):
    if context["completed_local_control"]:
        return "completed_local_control"
    if context["command"] is not None:
        return "remote_command"
    if context.get("cached_fallback") and context["action"] != 4:
        return "cached_fallback"
    return "safety_wait"


def experimental_class(condition):
    base = inference_class(condition)

    class TimedRemote(base):
        motion = None

        def __init__(self,*args,**kwargs):
            from .bundled_actions import DelayedCommandInbox
            self.action_bundles={};self.arrived_bundles={}
            super().__init__(*args,**kwargs)
            self.delayed_inboxes={r:DelayedCommandInbox(self.config.seconds_per_step,self.parameters.decision_budget_seconds)
                for r in self.states}

        def _after_inference(self,report,own,location,observation,action,event):
            super()._after_inference(report,own,location,observation,action,event)
            if self.motion is None or self.motion.speeds[report.robot_id]['speed_class']!='fast':return
            from .bundled_actions import second_input
            settings=dict(radius=self.project.map.fov_radius,waypoints=self.project.map.astar_waypoints,
                horizon=self.project.map.communicated_path_steps,map_extent=max(self.grid.shape))
            obs,sources,versions,priorities,prediction=second_input(report,self.replicas[location],self.grid,action,
                self._state_for_report,self.transport.now,settings)
            versions[report.robot_id]['state_event']=event['dt_versions'][str(report.robot_id)]['state_event']
            started=perf_counter();second=int(self.inference(location,report.robot_id,obs));wall=perf_counter()-started
            from .remote_state import observation_hash
            inferred=self._record('bundle_inference',job_id=event['job_id'],robot_id=report.robot_id,location=location,
                parent_inference_event=event['event_id'],first_action=action,action=second,
                observation={k:v.tolist() for k,v in obs.items()},observation_hash=observation_hash(obs),
                sources=sources,dt_versions=versions,dt_priorities=priorities,prediction=prediction,
                required_cycles_unchanged=True,model_version=self.model_version)
            self.action_bundles[event['job_id']]=dict(actions=[action,second],second_inference_event=inferred['event_id'],
                predicted_position=prediction['predicted_position'],sources=sources,dt_versions=versions,dt_priorities=priorities)
            self.jobs[event['job_id']]['bundle_inference_wall_seconds']=wall

        def _send_remote(self,agent,kind,payload,route,**kwargs):
            if kind=='remote_action_command' and payload['job_id'] in self.action_bundles:
                payload=dict(payload,action_bundle=self.action_bundles[payload['job_id']])
            return super()._send_remote(agent,kind,payload,route,**kwargs)

        def _record(self, event_name, **values):
            if event_name == "decision_event":
                robot = values["robot_id"]
                values["action_source"] = action_source(values)
                values["fallback_outcome"] = ("movement_intent" if values["action"] != 4 else
                    "yield" if (values.get("local_control") or {}).get("blocking_peers") else "wait") if values.get("cached_fallback") else None
                values["control_interval_seconds"] = self.config.seconds_per_step
                values["command_deadline_seconds"] = self.parameters.decision_budget_seconds
                if values.get("cached_fallback") and values["action"] != 4:
                    own = self.states[robot][robot].value
                    row,col = self.positions[robot]
                    values["threatened_successors"] = [list(point)
                        for peer,cache in sorted(self.states[robot].items())
                        if peer != robot and not cache.value["completed"] and cache.value["priority"] < own["priority"]
                        for point in cache.value["path"][1:self.project.map.communicated_path_steps+1]
                        if max(abs(point[0]-row),abs(point[1]-col)) <= self.project.map.fov_radius]
                self.command_contexts[robot].update(values)
            return super()._record(event_name,**values)

        def _receive(self, packet):
            result = super()._receive(packet)
            if packet.kind=='remote_action_command' and self.motion is not None:
                from .remote_state import ActionCommand
                command=ActionCommand(**packet.payload['command'])
                bundle=packet.payload.get('action_bundle')
                if bundle is not None:
                    self.arrived_bundles[command.job_id]=bundle
                    self._record('action_bundle_arrived',robot_id=packet.agent,job_id=command.job_id,
                        bundle=bundle,arrived_at=self.transport.now,packet_id=packet.id)
                if self.motion.speeds[packet.agent]['speed_class']=='slow':
                    accepted,reason=self.delayed_inboxes[packet.agent].accept(command,self.transport.now,bundle)
                    self._record('delayed_command_arrived',robot_id=packet.agent,command=command.payload(),
                        execution_cycle=command.cycle+1,arrived_at=self.transport.now,accepted=accepted,reason=reason,
                        packet_id=packet.id)
            if self.motion is not None:
                self.motion.check_cache_updates(self.transport.now)
            return result

        def _consistency(self):
            views, references = {}, {}
            for receiver in self.states:
                context = self.command_contexts.get(receiver,{})
                command = context.get("command")
                if self.motion is not None and self.motion.speeds[receiver]['speed_class']=='slow':
                    candidate=self.delayed_inboxes[receiver].peek(self.current_cycle,self.transport.now)
                    command=candidate[0].payload() if candidate else None
                    context=dict(command=command,cached_fallback=command is None and self.current_cycle>0
                        and self.parameters.missed_command_fallback=='cached_astar')
                location = command["location"] if command else "drone"
                references[receiver] = location,self.replicas[location].states
                if command:
                    views[receiver] = dict(priorities={int(k):v for k,v in command["dt_priorities"].items()},
                        states={int(p):self.events_by_id[v["state_event"]]["state"] for p,v in command["dt_versions"].items()})
                elif context.get("cached_fallback"):
                    views[receiver] = dict(priorities={p:c.value["priority"] for p,c in self.states[receiver].items()},
                        states={p:c.value for p,c in self.states[receiver].items()})
            metadata = {r:{p:self.events_by_id[ref] for p,ref in refs.items()} for r,refs in self.receiver_refs.items()}
            return consistency_snapshot(self.positions,self.truth,self.states,self.transport.now,
                self.project.map.fov_radius,self.project.map.communicated_path_steps,
                self.parameters.age_reference_seconds,None,views,metadata,dt_references=references)

        def prepare_decision(self, agents, cycle_start, deadline_seconds):
            super().prepare_decision(agents,cycle_start,deadline_seconds)
            if self.motion is not None:
                for robot,context in self.command_contexts.items():
                    if self.motion.speeds[robot]['speed_class']=='slow' and context['command']:
                        job=self.jobs[context['command']['job_id']]
                        job['nominal_selected_at']=job.pop('used_at')

        def fallback_diagnostic(self,agent,local,cycle,now,context,substep=0):
            fresh = {}
            for peer in self._agents:
                cache = InformationCache()
                cache.accept(now,self._state(self.truth[peer.id]),cycle,{peer.id:now})
                fresh[peer.id] = cache
            stale_action,stale_sources,details = cached_path_action(local,self.states[agent.id],self.project.map.communicated_path_steps)
            fresh_action,fresh_sources,fresh_details = cached_path_action(local,fresh,self.project.map.communicated_path_steps)
            if substep==0:
                dqn = next(e for e in reversed(self.oracle_events) if e["event"] == "decision_action_shadow"
                           and e["cycle"] == cycle and e["robot_id"] == agent.id)
                dqn_action=dqn['oracle_action']
                extra_observation=None
            else:
                # A fresh second-round diagnostic never changes the executed
                # remote bundle, task, request or deadline.
                from .remote_state import DTReplica,remote_observation
                replica=DTReplica('cloud')
                reports={a.id:self._sense(a,self._agents,cycle,now) for a in self._agents}
                states={r:self._state_for_report(report,now,cycle) for r,report in reports.items()}
                for r,state in states.items():replica.accept(state,now,f'extra-oracle:{cycle}:{r}')
                observation=remote_observation(reports[agent.id],states[agent.id],replica,
                    self.project.map.fov_radius,self.project.map.astar_waypoints,
                    self.project.map.communicated_path_steps,max(self.grid.shape))[0]
                dqn_action=int(self.shadow_inference('cloud',agent.id,observation))
                extra_observation={k:v.tolist() for k,v in observation.items()}
            diagnostic = dict(event="fallback_shadow",time=now,cycle=cycle,substep=substep,robot_id=agent.id,evaluation_only=True,
                    causal_anchor_event_id=self.remote_events[-1]["event_id"],
                    local_report=local.payload(),stale_action=stale_action,fresh_action=fresh_action,
                    stale_sources=stale_sources,fresh_sources=fresh_sources,
                    stale_details=details,fresh_details=fresh_details,
                    fresh_states={p:c.value for p,c in fresh.items()},
                    stale_vs_fresh_disagreement=stale_action != fresh_action,
                    fallback_vs_fresh_dqn_disagreement=stale_action != dqn_action,
                    fresh_fallback_vs_fresh_dqn_disagreement=fresh_action != dqn_action,
                    invoked=bool(context.get("cached_fallback")),actual_action=context['action'],
                    action_source=context['action_source'],actual_vs_fresh_dqn_disagreement=context['action']!=dqn_action,
                    decision_reason=context['decision_reason'],fresh_dqn_action=dqn_action)
            if extra_observation is not None:diagnostic['fresh_dqn_observation']=extra_observation
            self.oracle_events.append(json.loads(json.dumps(diagnostic,allow_nan=False)))

    return TimedRemote


def run_frequency_episode(config, model, case, device, version, scenario, condition, settings, progress=None):
    """Source at nΔ; select and execute up to two discrete rounds at nΔ+D.

    Only epochs whose cutoff is <= horizon generate work. At 350 ms and a
    250 ms cutoff this gives 714 opportunities (n=0..713). Motion/transport stop
    at the exact completion event or 250 s, whichever occurs first.
    """
    from .models import select_action
    env = ExperimentEnvironment(config,ExperimentMode.UAV,scenario.map_path,case["num_agents"],250,case["seed"],scenario)
    env.remote_communication_class = experimental_class(condition)
    def infer(location,robot,observation):
        return select_action(model,observation,0.,np.random.default_rng(0),device)
    env.remote_inference = env.remote_shadow_inference = infer
    env.model_version = version
    env.reset()
    c = env.communication
    motion = FrequencyMotion(env,settings)
    c.motion = motion
    frames,loop = [],[]
    def frame():
        frames.append(dict(step=env.step_count,time=c.transport.now,positions=[list(a.position) for a in env.agents],
            uav_position=env.uav.position.tolist(),uav_travel_m=env.uav.total_distance_m,
            agents_at_goal=sum(a.at_goal for a in env.agents),
            completed_flags={a.id:a.ever_reached_goal for a in env.agents}))
    frame()
    cycle = 0
    next_uav_update = 1.
    while cycle*settings.control_interval_seconds+settings.command_deadline_seconds <= settings.horizon_seconds+1e-12:
        env.step_count = cycle
        if cycle:
            now = cycle*settings.control_interval_seconds
            c.transport.advance(now,c._receive)
            c.update(env.agents,now)
            while now+1e-12 >= next_uav_update:
                positions=[tuple(c.reports[a.id]['position']) for a in env.agents]
                env.uav.update(env.uav.positions_to_xy_m(positions))
                next_uav_update+=1.
        c.prepare_decision(env.agents,cycle*settings.control_interval_seconds,settings.command_deadline_seconds)
        motion.step(cycle)
        loop.append(dict(step=cycle,decision_at=c.transport.now,actions=dict(c.execution_actions)))
        frame()
        cycle += 1
        if env.all_agents_at_goals or env.failure_reason:break
        if progress and cycle % 25 == 0:
            progress(cycle,c.transport.now,len(c.jobs))
    if not (env.all_agents_at_goals or env.failure_reason):
        c.transport.advance(settings.horizon_seconds,c._receive)
    motion.finish(c.transport.now)
    env.step_count = cycle
    if not frames or frames[-1]["time"] != c.transport.now:
        frame()
    success = env.all_agents_at_goals and env.failure_reason is None
    row = dict(case,steps=cycle,success=success,failure_reason=env.failure_reason or (None if success else "physical_time_horizon"),
        successful_makespan=c.transport.now if success else None,makespan_seconds=c.transport.now if success else None,
        observed_seconds=c.transport.now,completion_fraction=sum(a.at_goal for a in env.agents)/env.num_agents,
        ever_completion_fraction=sum(a.ever_reached_goal for a in env.agents)/env.num_agents,
        uav_travel_m=env.uav.total_distance_m,**c.energy_bounds())
    trace = dict(configuration_seconds_per_step=settings.control_interval_seconds,case=case,grid=env.grid.tolist(),
        goals=case["goals"],frames=frames,transport_events=c.transport.log,resource_tasks=c.task_records(),allocation_loop=loop,
        remote_schema="remote_dt_v1",remote_events=c.remote_events,oracle_events=c.oracle_events,
        handover_windows=c.transport.handover_windows(0,c.transport.now),
        remote_settings=dict(radius=config.map.fov_radius,horizon=config.map.communicated_path_steps,
            waypoints=config.map.astar_waypoints,map_extent=max(env.grid.shape),age_reference=config.resource_allocation.age_reference_seconds,
            meters_per_cell=config.map.meters_per_cell,seconds_per_step=settings.control_interval_seconds,
            cutoff_seconds=settings.command_deadline_seconds,on_collision=config.resource_allocation.on_collision,
            missed_command_fallback=settings.fallback,controlled_staleness=c.stress_settings,
            frequency_motion=asdict(settings),speed_assignments=motion.speeds,
            dt_reference="executed_command_location_else_drone",
            discrete_command_protocol="dt_bundle_slow_pipeline_v2"),
        outcome=dict(success=success,failure_reason=env.failure_reason,steps=cycle,
            final_positions={a.id:list(a.position) for a in env.agents}),
        physical_summary=dict(distance_cells=motion.distance,active_pair_seconds=motion.active_pair_seconds,
            movement_rounds=motion.rounds))
    return row,trace
