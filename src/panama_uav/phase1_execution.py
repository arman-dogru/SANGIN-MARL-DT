"""Opt-in clean execution matching the remote-DT physical controllers.

This module does not run the SAGIN scheduler.  Every decision consumes fresh,
instantaneous state; the shared remote observation and local completed-robot
controller retain their existing interfaces.  The original Phase 1 environment
remains available for replaying the failed V1 experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np

from .information import InformationCache
from .phase1_environment import CleanMAPFEnvironment, epoch_metrics
from .remote_control import completed_action, conflicts, resolve_delivered
from .remote_state import PhysicalReport, REMOTE_SCHEMA


EXECUTION_PROTOCOL = "clean_remote_execution_v2"


@dataclass(frozen=True)
class ActionSelection:
    """Separate policy proposals from the actions that physically execute."""

    proposed: dict[int, int]
    executed: dict[int, int]
    active: tuple[int, ...]
    completed: tuple[int, ...]
    priorities: dict[int, float]
    local_details: dict[int, dict]
    source_times: dict[int, dict[int, float]]


class RemoteAlignedCleanMAPFEnvironment(CleanMAPFEnvironment):
    """Fresh clean MAPF with remote-DT movement and post-completion semantics.

    ``step`` accepts PANAMA's unchanged five-action dictionary. Completed
    robots' proposals are ignored and replaced by ``completed_action``.  Train
    only ``selection.active`` transitions, ending each robot's learning return
    on its first completion.  Episode success still requires all robots to be
    physically at their goals together, including completed robots that yielded.

    Clean physical time remains the inherited integer control tick.  It models
    no sensing-to-command elapsed time and makes no network deadline claim.
    """

    execution_protocol = EXECUTION_PROTOCOL

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.resource_allocation.enabled or self.config.communication.profile != "perfect":
            raise ValueError("Aligned clean execution requires allocation disabled and perfect communication")
        self.last_selection: ActionSelection | None = None

    @property
    def active_robot_ids(self) -> tuple[int, ...]:
        return tuple(a.id for a in self.agents if not a.ever_reached_goal)

    def reset(self, seed=None):
        self.last_selection = None
        return super().reset(seed)

    def prepare_actions(self, proposed: dict[int, int]) -> ActionSelection:
        """Pure, repeatable selection from the current fresh decision epoch.

        Unspecified actions default to WAIT, as in the legacy environment.
        Peer paths match remote-DT's perfect-reference cache: its bootstrap
        stores the published horizon, then subsequent epochs store full fresh
        paths. ``remote_observation`` independently applies its own intention
        horizon. Own completion overrides proposed actions and priority.
        """
        if not self.agents:
            raise RuntimeError("reset() must be called before selecting actions")
        if set(proposed) - set(self.agent_by_id):
            raise ValueError("Action dictionary includes an unknown robot")
        if any(action not in self.ACTION_DELTAS for action in proposed.values()):
            raise ValueError("PANAMA actions must be integers from zero through four")
        now = self.step_count*self.config.communication.seconds_per_step
        priorities = {}
        cached = {}
        for agent in self.agents:
            distance = len(agent.path)-1 if agent.path else self.grid.size-1
            priority = (self.grid.size if agent.ever_reached_goal else min(distance, self.grid.size-1))
            priorities[agent.id] = float(priority+agent.id*1e-6)
            path = list(agent.path or [agent.position])
            if self.step_count == 0:
                path = path[:self.config.map.communicated_path_steps+1]
            state = dict(position=agent.position, goal=agent.goal, path=path,
                priority=priorities[agent.id], completed=agent.ever_reached_goal)
            cache = InformationCache()
            cache.accept(now, state, self.step_count, {agent.id: now})
            cached[agent.id] = cache
        executed = {a.id: proposed.get(a.id, 4) for a in self.agents}
        details, sources = {}, {}
        for agent in self.agents:
            if not agent.ever_reached_goal:
                continue
            row, col = agent.position
            side = self.fov_size
            obstacles = self.padded_grid[row:row+side, col:col+side].tolist()
            occupancy = [[0.0]*side for _ in range(side)]
            for peer in self.agents:
                dr, dc = peer.position[0]-row, peer.position[1]-col
                if peer.id != agent.id and max(abs(dr), abs(dc)) <= self.fov_radius:
                    occupancy[dr+self.fov_radius][dc+self.fov_radius] = 1.0
            report = PhysicalReport(agent.id, self.step_count, now, agent.position, agent.goal,
                True, agent.initial_distance_to_goal, obstacles, occupancy,
                f"clean-v2:{self.step_count}:{agent.id}")
            executed[agent.id], sources[agent.id], details[agent.id] = completed_action(
                report, cached, priorities[agent.id])
        return ActionSelection(dict(proposed), executed, self.active_robot_ids,
            tuple(a.id for a in self.agents if a.ever_reached_goal), priorities, details, sources)

    def epoch_metrics(self, proposed):
        """Measure executed actions, excluding ignored completed-DQN proposals."""
        return epoch_metrics(self, self.prepare_actions(proposed).executed)

    def _resolve_positions(self, actions, rewards, info):
        before = {a.id: a.position for a in self.agents}
        intended = {}
        for agent in self.agents:
            dr, dc = self.ACTION_DELTAS[actions[agent.id]]
            target = (agent.position[0]+dr, agent.position[1]+dc)
            if not self._valid(target):
                target = agent.position
                rewards[agent.id] += self.config.reward.collision_obstacle
                info["collisions_obstacle_step"] += 1
            intended[agent.id] = target
        events = conflicts(before, intended)
        collided = {robot for event in events for robot in event["robots"]}
        final = resolve_delivered(before, intended, self.last_selection.priorities)
        info["physical_conflicts"] = events
        info["actual_agent_overlap_after_resolution"] = len(conflicts(before, final))
        info["failure_reason"] = None
        return final, collided

    def step(self, actions):
        selection = self.prepare_actions(actions)
        self.last_selection = selection
        result = super().step(selection.executed)
        result.info.update(execution_protocol=EXECUTION_PROTOCOL,
            policy_actions=selection.proposed, execution_actions=selection.executed,
            policy_controlled_robots=list(selection.active),
            completed_local_actions={r: selection.executed[r] for r in selection.completed},
            resolution_priorities=selection.priorities, local_control=selection.local_details,
            local_source_timestamps=selection.source_times)
        return result


def _paired_trajectory(config, case, inference, root, cap):
    """Exercise the actual remote implementation and audit its saved event chain."""
    from .config import ExperimentMode
    from .environment import ExperimentEnvironment
    from .phase1_data import scenario_for
    from .remote_audit import replay_trace
    from .records import write_json

    remote_config = replace(config, resource_allocation=replace(config.resource_allocation,
        enabled=True, execution_mode='remote_dt', on_collision='dt_resolve'))
    arguments = (ExperimentMode.UAV, case['map_path'], case['num_agents'], 250,
                 case['seed'], scenario_for(case))
    clean = RemoteAlignedCleanMAPFEnvironment(config, *arguments)
    remote = ExperimentEnvironment(remote_config, *arguments)
    remote.remote_inference = inference
    clean.reset(); remote.reset()
    checked, local, arrivals, departures = 0, 0, 0, 0
    for _ in range(cap):
        observations = clean.observations()
        proposed = {r: int(inference('cloud', r, observations[r])) for r in clean.active_robot_ids}
        cutoff = remote_config.resource_allocation.decision_budget_seconds
        remote.prepare_decision(cutoff)
        epoch = clean.step_count
        events = [e for e in remote.communication.remote_events
                  if e['event']=='remote_inference' and e['job_id'].startswith(f'{epoch}:')]
        if {e['robot_id'] for e in events} != set(clean.active_robot_ids):
            raise ValueError('Aligned compatibility omitted active inference')
        for event in events:
            if any(not np.array_equal(observations[event['robot_id']][k], event['observation'][k])
                   for k in ('spatial','vector')):
                raise ValueError('Aligned multi-step observation differs from remote')
            checked += 1
        selection = clean.prepare_actions(proposed)
        priorities = {r: c['resolution_priority'] for r,c in remote.communication.command_contexts.items()}
        if selection.executed != remote.communication.execution_actions or selection.priorities != priorities:
            raise ValueError('Aligned action/controller priority differs from remote')
        before = {a.id: (a.position, a.ever_reached_goal) for a in clean.agents}
        first = clean.step(proposed)
        second = remote.step(remote.communication.execution_actions.copy())
        if ([(a.position,a.ever_reached_goal) for a in clean.agents] !=
            [(a.position,a.ever_reached_goal) for a in remote.agents] or
            first.rewards != second.rewards or
            (first.terminated,first.truncated) != (second.terminated,second.truncated) or
            first.info['physical_conflicts'] != second.info['physical_conflicts']):
            raise ValueError('Aligned multi-step physical execution/reward differs from remote')
        local += len(selection.completed)
        arrivals += sum(a.ever_reached_goal and not before[a.id][1] for a in clean.agents)
        departures += sum(before[a.id][1] and before[a.id][0]==a.goal and a.position!=a.goal
                          for a in clean.agents)
        if first.terminated or first.truncated:
            break
    trace = dict(remote_schema=REMOTE_SCHEMA, grid=remote.grid.tolist(),
        remote_events=remote.communication.remote_events,
        remote_settings=dict(radius=config.map.fov_radius, horizon=config.map.communicated_path_steps,
            waypoints=config.map.astar_waypoints, map_extent=max(remote.grid.shape),
            age_reference=config.resource_allocation.age_reference_seconds,
            meters_per_cell=config.map.meters_per_cell,
            seconds_per_step=config.communication.seconds_per_step,
            cutoff_seconds=remote_config.resource_allocation.decision_budget_seconds,
            on_collision='dt_resolve'),
        outcome=dict(final_positions={a.id:list(a.position) for a in remote.agents}, failure_reason=remote.failure_reason))
    audit = replay_trace(trace,inference)
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with gzip.open(root/(case['case_id']+'.json.gz'),'wt',encoding='utf-8') as stream:
        json.dump(trace,stream,allow_nan=False)
    row=dict(num_agents=case['num_agents'], case_id=case['case_id'],
        epochs=clean.step_count, observations_equal=checked, active_actions_equal=checked,
        local_completed_actions_equal=local, first_completions=arrivals,
        completed_goal_departures=departures, positions_rewards_priorities_equal=True,
        replay_status=audit['status'], model_actions_verified=audit['model_actions_verified'],
        event_checksum=audit['final_checksum'])
    write_json(root/(case['case_id']+'_audit.json'),row)
    return row


def verify_execution_compatibility(config, model, bank, device, root):
    """Bounded fresh multi-step comparison, plus deterministic completion fixture.

    This certifies execution compatibility, never policy qualification or a
    communication study. Four validation cases run for at most 16 epochs each;
    a scripted completed robot must yield and return in four more epochs.
    """
    from .models import select_action
    from .records import write_json

    root = Path(root)
    audit_root = root/'execution_compatibility'
    inference = lambda location,robot,obs: select_action(model,obs,0.,np.random.default_rng(0),device)
    checks = []
    for count in (2,4,6,8):
        case = next(c for c in bank if c['num_agents']==count)
        checks.append(_paired_trajectory(config,case,inference,audit_root,16))
    fixture_map = audit_root/'completed_control.map'
    fixture_map.write_text('type octile\nheight 7\nwidth 7\nmap\n'+
        '\n'.join(['@@@@@@@']+['@.....@']*5+['@@@@@@@'])+'\n')
    fixture = dict(map_path=str(fixture_map),num_agents=2,seed=773,
        case_id='completed_control',starts=[[3,3],[3,1]],goals=[[3,3],[3,5]])
    completed = _paired_trajectory(config,fixture,lambda location,robot,obs: {0:1,1:3}[robot],audit_root,4)
    if completed['local_completed_actions_equal']!=4 or completed['completed_goal_departures']!=1:
        raise ValueError('Compatibility fixture failed to exercise completed yield/return')
    result = dict(status='passed', execution_contract='remote-aligned-v2',
        execution_protocol=EXECUTION_PROTOCOL, checks=checks, completed_control_fixture=completed,
        scope='bounded multi-step fresh remote observations/actions/priority/physical movement/rewards; '
              'scripted completed-goal yield and return; event replay; no communication impairment')
    write_json(root/'compatibility.json',result)
    return result
