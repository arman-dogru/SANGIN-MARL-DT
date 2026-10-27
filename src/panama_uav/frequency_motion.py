"""Discrete movements: delayed slow commands and bundled fast DQN actions.

Each robot uploads once per global step. Slow commands execute one epoch later;
fast commands contain two DT-based actions. There is no separate motion clock.
"""
from dataclasses import dataclass
import math
import numpy as np

from .remote_control import DELTAS,cached_path_action,completed_action,conflicts,resolve_delivered
from .information import InformationCache


@dataclass(frozen=True)
class MotionSettings:
    control_interval_seconds: float = .35
    command_deadline_seconds: float = .25
    horizon_seconds: float = 250.
    heterogeneous: bool = True
    fallback: str = 'cached_astar'
    treatment: str = 'A4'

    def __post_init__(self):
        if not (0 < self.command_deadline_seconds < self.control_interval_seconds
                and math.isfinite(self.horizon_seconds) and self.horizon_seconds>0):
            raise ValueError('Require a finite horizon and 0 < deadline < interval')
        if self.fallback not in ('wait','cached_astar'):raise ValueError('Unknown fallback')


def speed_assignment(count,seed,heterogeneous=True):
    if not heterogeneous:
        return {i:dict(speed_class='normal',decisions_per_step=1.,command_lag_steps=0) for i in range(count)}
    classes=(('slow',1.,1),('normal',1.,0),('fast',2.,0))
    order=np.random.default_rng(int(seed)+918273).permutation(count)
    return {int(robot):dict(speed_class=classes[k%3][0],decisions_per_step=classes[k%3][1],command_lag_steps=classes[k%3][2])
            for k,robot in enumerate(order)}


def eligible(speed_class,cycle,substep):
    return substep==0 or speed_class=='fast'


class FrequencyMotion:
    def __init__(self,environment,settings):
        self.env,self.settings=environment,settings
        self.speeds=speed_assignment(environment.num_agents,environment.base_seed,settings.heterogeneous)
        self.distance={a.id:0 for a in environment.agents}
        self.reversal_events=[];self.pending_reversals={}
        self.active_pair_seconds=0.;self.last_time=0.;self.rounds=0

    def account_time(self,now):
        active=sum(not a.ever_reached_goal for a in self.env.agents)
        self.active_pair_seconds+=active*(active-1)/2*(now-self.last_time)
        self.last_time=now

    def select(self,agent,cycle,now,substep):
        c=self.env.communication
        report=c._sense(agent,self.env.agents,cycle,now)
        speed_class=self.speeds[agent.id]['speed_class']
        candidate=None;bundle=None
        if substep==0 and speed_class=='slow':
            candidate=c.delayed_inboxes[agent.id].decision(cycle,now)
        elif substep==0:
            nominal=c.command_contexts[agent.id]
            if nominal['command'] is not None:
                from .remote_state import ActionCommand
                candidate=ActionCommand(**nominal['command']),nominal['arrived_at'],c.arrived_bundles.get(nominal['command']['job_id'])
        else:
            first=self.first_contexts[agent.id]
            if first['command'] is not None:
                from .remote_state import ActionCommand
                candidate=ActionCommand(**first['command']),first['arrived_at'],c.arrived_bundles.get(first['command']['job_id'])
        command,arrival,bundle=candidate if candidate else (None,None,None)
        cached=False;details=None;sources={};prediction_mismatch=None
        warmup=speed_class=='slow' and cycle==0
        if agent.ever_reached_goal:
            priority=self.env.grid.size+agent.id*1e-6
            action,sources,details=completed_action(report,c.states[agent.id],priority)
            source='completed_local_control';command=None;arrival=None
        elif command is not None:
            if substep:
                if bundle is None or bundle['actions'][0]!=command.action:
                    raise ValueError('Fast remote command lacks its matching two-action bundle')
                action=bundle['actions'][1]
                priorities=bundle['dt_priorities']
                prediction_mismatch=list(agent.position)!=bundle['predicted_position']
            else:action=command.action;priorities=command.dt_priorities
            priority=priorities.get(agent.id,priorities.get(str(agent.id)))
            if priority is None:raise ValueError('Remote action lacks a DT self priority')
            source='remote_command';sources=command.sources if not substep else bundle['sources']
        else:
            priority=c.states[agent.id][agent.id].value['priority']
            if self.settings.fallback=='cached_astar' and not warmup:
                action,sources,details=cached_path_action(report,c.states[agent.id],self.env.config.map.communicated_path_steps)
                cached=True
            else:action=4
            source='cached_fallback' if cached and action!=4 else 'safety_wait'
        context=dict(robot_id=agent.id,cycle=cycle,substep=substep,cutoff_time=now,action=action,
            command=command.payload() if command else None,arrived_at=arrival,
            source_cycle=command.cycle if command else None,
            scheduled_source_cycle=cycle-1 if speed_class=='slow' else cycle,
            command_cutoff_missed=command is None and not agent.ever_reached_goal and not warmup,
            slow_pipeline_warmup=warmup,completed_local_control=agent.ever_reached_goal,cached_fallback=cached,
            action_source=source,local_sources=sources,local_control=details,resolution_priority=priority,
            decision_reason='slow_previous_command' if speed_class=='slow' else 'fast_bundle_second' if substep else 'current_command',
            speed_class=speed_class,bundle_prediction_mismatch=prediction_mismatch,
            bundle_inference_event=bundle['second_inference_event'] if bundle and substep else None)
        c._record('execution_decision_event',**context)
        if command is not None:
            c.jobs[command.job_id].setdefault('executed_at',now)
            c.jobs[command.job_id]['used_at']=now
            c.jobs[command.job_id]['execution_cycle']=cycle
        if not agent.ever_reached_goal:c.fallback_diagnostic(agent,report,cycle,now,context,substep=substep)
        return context

    def step(self,cycle):
        env,c=self.env,self.env.communication;now=c.transport.now
        self.account_time(now)
        self.first_contexts={}
        for substep in (0,1):
            due=[a.id for a in env.agents if eligible(self.speeds[a.id]['speed_class'],cycle,substep)]
            if not due and substep==1:continue
            before={a.id:a.position for a in env.agents}
            prior={a.id:dict(priority=a.priority,completed=a.ever_reached_goal,path=list(a.path or []),position=a.position) for a in env.agents}
            contexts={a.id:(self.select(a,cycle,now,substep) if a.id in due else
                dict(self.first_contexts[a.id],action=4,substep=substep,command=None,action_source='not_scheduled')) for a in env.agents}
            if substep==0:self.first_contexts=contexts
            intended,actions,reasons={},{},{}
            for agent in env.agents:
                robot=agent.id;action=contexts[robot]['action'] if robot in due else 4
                actions[robot]=action
                dr,dc=DELTAS[action];target=agent.position[0]+dr,agent.position[1]+dc
                reason='not_scheduled' if robot not in due else 'slow_pipeline_warmup' if contexts[robot]['slow_pipeline_warmup'] else 'controller_wait' if action==4 else 'move_attempt'
                if not env._valid(target):target=agent.position;reason='local_obstacle'
                intended[robot]=target;reasons[robot]=reason
            attempted=conflicts(before,intended)
            priorities={r:context['resolution_priority'] for r,context in contexts.items()}
            if attempted and env.config.resource_allocation.on_collision=='terminate':
                after=before.copy();env.failure_reason='robot_collision'
            else:after=resolve_delivered(before,intended,priorities)
            for agent in env.agents:
                robot=agent.id;agent.position=after[robot]
                if before[robot]!=after[robot]:self.distance[robot]+=1
                elif reasons[robot]=='move_attempt':reasons[robot]='physical_resolver_block'
                env._refresh_path(agent)
                if agent.at_goal and not agent.ever_reached_goal:
                    agent.ever_reached_goal=True;agent.first_completion_time=now
            env._refresh_priorities()
            c.truth={a.id:c._report(a) for a in env.agents};c.positions={a.id:a.position for a in env.agents};c.physical_time=now
            for a in env.agents:c.position_history[a.id].append((now,a.position))
            c._record('frequency_motion_event',cycle=cycle,substep=substep,due=due,actions=actions,
                before=before,intended=intended,after=after,reasons=reasons,contexts=contexts,priorities=priorities,
                completed_before={r:v['completed'] for r,v in prior.items()},
                conflicts_before_resolution=attempted,conflicts_after_resolution=conflicts(before,after),
                physical=c.truth,first_completion_times={a.id:a.first_completion_time for a in env.agents},failure_reason=env.failure_reason)
            if substep==0:c.execution_actions={r:contexts[r]['action'] for r in due}
            self._reversals(prior,now,cycle,substep);self.check_cache_updates(now);self.rounds+=1
            if env.all_agents_at_goals or env.failure_reason:return

    def _reversals(self,prior,now,cycle,substep):
        from .consistency import consistency_snapshot
        agents=self.env.agents;c=self.env.communication
        for i,first in enumerate(agents):
            for second in agents[i+1:]:
                a,b=first.id,second.id
                old_order=prior[a]['priority']<prior[b]['priority'];new_order=first.priority<second.priority
                if old_order==new_order or prior[a]['completed'] or prior[b]['completed']:continue
                completion=first.ever_reached_goal or second.ever_reached_goal
                truth={r:c.truth[r] for r in (a,b)};stores={}
                for r in (a,b):
                    stores[r]={}
                    for p in (a,b):
                        cache=InformationCache();cache.accept(now,truth[p],0,{p:now});stores[r][p]=cache
                s=consistency_snapshot({r:tuple(truth[r]['position']) for r in (a,b)},truth,stores,now,
                    self.env.fov_radius,self.env.config.map.communicated_path_steps)
                record=dict(time=now,cycle=cycle,substep=substep,first=a,second=b,old_order=int(old_order),new_order=int(new_order),
                    cause='goal_completion' if completion else 'movement',physically_relevant=bool(s['relevant_pairs']),
                    before_priorities=[prior[a]['priority'],prior[b]['priority']],after_priorities=[first.priority,second.priority],
                    cached_orders=[dict(receiver=r,peer=p,
                        cached_peer_higher=c.states[r][p].value['priority']<c.states[r][r].value['priority'],
                        stale=(c.states[r][p].value['priority']<c.states[r][r].value['priority'])!=(truth[p]['priority']<truth[r]['priority']),
                        self_source_time=min(c.states[r][r].source_timestamps.values()),
                        peer_source_time=min(c.states[r][p].source_timestamps.values())) for r,p in ((a,b),(b,a))])
                self.reversal_events.append(record);c.oracle_events.append(dict(event='priority_reversal_event',**record))
                if completion:
                    for key in ((a,b),(b,a)):
                        if key in self.pending_reversals:self._close_reversal(key,now,'goal_completion',True)
                    continue
                for receiver,peer in ((a,b),(b,a)):
                    key=receiver,peer
                    if key in self.pending_reversals:self._close_reversal(key,now,'superseded',True)
                    self.pending_reversals[key]=dict(record,receiver=receiver,peer=peer,
                        physical_peer_higher=truth[peer]['priority']<truth[receiver]['priority'])

    def _close_reversal(self,key,now,reason,censored):
        record=self.pending_reversals.pop(key)
        self.env.communication.oracle_events.append(dict(event='priority_reversal_cache_duration',time=now,
            reversal_time=record['time'],receiver=key[0],peer=key[1],duration_seconds=now-record['time'],
            censored=censored,reason=reason,physically_relevant=record['physically_relevant']))

    def check_cache_updates(self,now):
        for (receiver,peer),record in list(self.pending_reversals.items()):
            caches=self.env.communication.states[receiver]
            if (caches[peer].value['priority']<caches[receiver].value['priority'])==record['physical_peer_higher']:
                self._close_reversal((receiver,peer),now,'cached_order_matches',False)

    def finish(self,now):
        self.account_time(now)
        for key in list(self.pending_reversals):self._close_reversal(key,now,'mission_observation_end',True)
