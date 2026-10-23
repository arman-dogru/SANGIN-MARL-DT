"""Opt-in allocated and fixed-placement inference experiments."""
from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from .allocation import AllocationSnapshot, AllocationTask, solve_binary
from .paper_radio import backhaul_budget
from .staleness_remote import ControlledRemoteDT
from .transport import encode


@dataclass(frozen=True)
class InferenceCondition:
    name: str
    route: str
    axis: str = 'nominal'
    level: float = 0.
    uplink_seconds: float = 0.
    downlink_seconds: float = 0.
    backhaul_seconds: float = 0.
    peer_delay_steps: int = 0
    compute_seconds: float = 0.
    interruption_seconds: float = 0.
    interruption_period_seconds: float = 10.
    interruption_start_seconds: float = 5.
    ideal_communication: bool = False

    def __post_init__(self):
        if self.route not in ('ideal', 'allocated', 'edge', 'cloud'):
            raise ValueError('Execution route must be ideal, allocated, edge or cloud')
        values = (self.level, self.uplink_seconds, self.downlink_seconds, self.backhaul_seconds,
                  self.compute_seconds, self.interruption_seconds, self.interruption_start_seconds)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Perturbations require finite nonnegative values')
        if type(self.peer_delay_steps) is not int or self.peer_delay_steps not in range(6):
            raise ValueError('Peer age uses 0–5 complete control steps')
        if not math.isfinite(self.interruption_period_seconds) or not 0 <= self.interruption_seconds < self.interruption_period_seconds:
            raise ValueError('Outages must leave available service in every period')
        if self.route == 'ideal' and any((self.uplink_seconds,self.downlink_seconds,self.backhaul_seconds,
                self.compute_seconds,self.interruption_seconds,self.peer_delay_steps)):
            raise ValueError('The ideal reference has no injected impairment')
        if self.backhaul_seconds and self.route not in ('cloud','allocated'):
            raise ValueError('A cloud-capable route is required for LEO transit')
        if self.ideal_communication and (self.route=='ideal' or any((self.uplink_seconds,
                self.downlink_seconds,self.backhaul_seconds,self.peer_delay_steps,
                self.compute_seconds,self.interruption_seconds))):
            raise ValueError('Matched ideal-network references retain only the finite CPU stage')


def condition_grid():
    conditions = [InferenceCondition('ideal', 'ideal')]
    for route in ('allocated', 'edge', 'cloud'):
        conditions.append(InferenceCondition(route+'__nominal', route))
        for axis, field, levels in (
            ('uplink', 'uplink_seconds', (.05,.15)),
            ('downlink', 'downlink_seconds', (.05,.15)),
            ('aoi', 'peer_delay_steps', (1,3)),
            ('compute', 'compute_seconds', (.05,.15)),
            ('interruption', 'interruption_seconds', (.25,1.))):
            for level in levels:
                conditions.append(InferenceCondition(f'{route}__{axis}-{level:g}', route,
                    axis, level, **{field:level}))
    for route in ('allocated','cloud'):
        for delay in (.05,.15):
            conditions.append(InferenceCondition(f'{route}__backhaul-{delay:g}',route,
                'backhaul',delay,backhaul_seconds=delay))
    for route in ('allocated','edge','cloud'):
        conditions.append(InferenceCondition(route+'__ideal_network',route,
            axis='network_reference',ideal_communication=True))
    return conditions


def outage_windows(condition, horizon):
    if not condition.interruption_seconds:
        return []
    return [(float(start),min(float(start)+condition.interruption_seconds,horizon))
            for start in np.arange(condition.interruption_start_seconds,horizon,
                                   condition.interruption_period_seconds)]


def interrupted_finish(start, duration, windows):
    """Preempt/resume service; only active transmission time consumes RF energy."""
    if duration < 0 or start < 0 or not math.isfinite(start+duration):
        raise ValueError('Invalid service interval')
    if duration==0:
        return start
    at, remaining = start, duration
    for left,right in windows:
        if right <= at or left == right:
            continue
        if at+remaining <= left:
            break
        remaining -= max(0.,left-at)
        at = right
    return at+remaining


def union_windows(windows):
    merged=[]
    for left,right in sorted(windows):
        if right <= left:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1][1]=max(merged[-1][1],right)
        else:
            merged.append([float(left),float(right)])
    return merged


class FixedRouteInference(ControlledRemoteDT):
    """Route is experimental placement; finite priority sharing stays unchanged."""
    condition = InferenceCondition('ideal','ideal')

    def __init__(self,project,grid,agents,pathfinder,seed,cap,uav,inference,model_version=0):
        super().__init__(project,grid,agents,pathfinder,seed,cap,uav,inference,model_version)
        self.effects = self.condition
        t=self.transport
        if self.effects.ideal_communication:
            # Only transport becomes instantaneous. Controller/pool semantics
            # stay physical, so finite CPU service and causal DT updates remain.
            t.config=replace(t.config,profile='perfect',handover_enabled=False,contention_enabled=False)
        self.controlled_outages=outage_windows(self.effects,cap*self.config.seconds_per_step)
        if self.controlled_outages:
            original=t._finish_transmission
            windows=self.controlled_outages
            # Instance-local adapter, installed before the first physical task.
            # Orbital links use OrbitTimeline.service and the combined windows below.
            def finish(start,duration,satellite):
                if not satellite:
                    return interrupted_finish(start,duration,windows)
                return original(start,duration,satellite)
            t._finish_transmission=finish
            if t.orbit is not None:
                offset=self.config.backhaul_time_offset_seconds
                all_windows=[(a,b) for a,b,_ in t.orbit.blocked]
                all_windows.extend((a+offset,b+offset) for a,b in windows)
                # Both native and injected windows are active in these treatments.
                t.orbit.blocked=np.asarray([(a,b,0) for a,b in union_windows(all_windows)],dtype=float).reshape(-1,3)
        # Metadata stays outside the causal event chain: inserting an event
        # would change state-reference IDs and serialized network payload sizes.
        self.inference_experiment=dict(condition=asdict(self.effects),
            outage_windows=self.controlled_outages,
            placement='production binary allocator' if self.condition.route=='allocated' else 'fixed whole-task execution')

    def _send_remote(self,agent,kind,payload,route,**kwargs):
        expanded=[]
        c=self.condition
        extra=dict(access_up=c.uplink_seconds,access_down=c.downlink_seconds,
                   satellite_up=c.backhaul_seconds,satellite_down=c.backhaul_seconds)
        for link in route:
            expanded.append(link)
            if extra.get(link,0):
                expanded.append(f'process:{extra[link]:.17g}')
        return super()._send_remote(agent,kind,payload,tuple(expanded),**kwargs)

    def _admit(self,payload):
        if self.condition.route=='allocated':
            return super()._admit(payload)
        t,p=self.transport,self.parameters
        job=self.jobs[payload['job_id']]
        x=int(self.condition.route!='cloud')
        weights=self._available_weights()
        radio=backhaul_budget(t,p)
        if t.orbit is None:
            t.links['satellite_up'].capacity_bps=radio['applied_bps'] or self.config.satellite_uplink_mbps*1e6
        # Predicted fixed-placement snapshots condition on available service.
        # Actual unavailable routes remain queued in the unchanged transport.
        rate=radio['applied_bps']
        if rate <= 0:
            rates=t.orbit.series.get('satellite_up_bps',[]) if t.orbit else []
            rate=min(self.config.satellite_uplink_mbps*1e6,
                max((float(v) for v in rates if math.isfinite(v) and v>0),default=self.config.satellite_uplink_mbps*1e6))
        tasks=[AllocationTask(job['job_id'],p.task_input_bits,p.cycles_per_bit,
            weights[job['agent']],payload['access_radio']['applied_bps'],rate,
            radio['round_trip_seconds'],p.robot_power_w,p.drone_power_w,fixed_x=x)]
        remaining={r['job_id']:r['remaining_cycles'] for pool in t.compute_pools.values() for r in pool.snapshot()}
        for old in self.jobs.values():
            if old['job_id']==job['job_id'] or old['stage'] not in ('computing','backhaul'):
                continue
            tasks.append(AllocationTask(old['job_id'],0,0,old['weight'],1,1,0,
                p.robot_power_w,p.drone_power_w,fixed_x=old['executed_x'],
                required_cycles=remaining.get(old['job_id'],old['required_cycles'])))
        snapshot=AllocationSnapshot(tuple(tasks),p.drone_cycles_per_second,p.cloud_cycles_per_second,
            p.energy_coefficient,p.latency_coefficient,p.drone_kappa,p.include_robot_upload_in_objective)
        solved=solve_binary(snapshot).to_dict()
        equivalent=dict(solved,mode='fixed_assignment_equivalent',solve_seconds=0.)
        self.allocations+=1
        priorities={a:c.value['priority'] for a,c in self.replicas['drone'].states.items()}
        comparison=dict(binary=solved,relaxed=equivalent,absolute_gap=0.,relative_gap=0.)
        t.log.append(dict(event='paper_allocation',time=t.now,new_job=job['job_id'],radio=radio,
            fixed_placement=True,resource_weights=weights,**comparison))
        if p.oracle_shadow:
            self._allocation_shadow(snapshot,solved,job['job_id'])
        self._record('allocation_event',job_id=job['job_id'],allocation_version=self.allocations,
            new_vs_fixed={task.task_id:False for task in tasks},
            priority_versions=dict(self.replicas['drone'].state_ids),
            transport_allocation_index=self.allocations,assignments=solved['tasks'],snapshot=asdict(snapshot),
            dt_priorities=priorities,resource_weights=weights)
        selected=next(r for r in solved['tasks'] if r['task_id']==job['job_id'])
        job.update(executed_x=x,relaxed_x=float(x),weight=tasks[0].weight,
            predicted_latency_seconds=selected['latency_seconds'],predicted_energy_j=selected['energy_j'],
            predicted_allocation=selected,relaxed_prediction=selected,allocation_at=t.now,
            backhaul_radio=radio,relaxation_gap=0.,batch_wait_seconds=0.,
            prediction_scope='fixed-placement snapshot conditional on available link service')
        if x or self.config.profile=='perfect':
            location='drone' if x else 'cloud'
            job.update(stage='computing',compute_arrived_at=t.now,location=location)
            if self.config.profile=='perfect' and self.condition.route=='ideal':
                self._finish(job['job_id'],0.,0.)
            else:
                t.compute_pools[location].submit(job['job_id'],job['required_cycles'],job['weight'],job['job_id'])
        else:
            job.update(stage='backhaul',location='cloud')
            self._send_remote(job['agent'],'remote_cloud_observation',payload,('satellite_up',),
                              version=job['version'],pad=True)

    def _computed(self,completed):
        if not self.condition.compute_seconds:
            return super()._computed(completed)
        job=self.jobs[completed.job_id]
        job.update(stage='inference_overhead',processor_completed_at=self.transport.now,
                   controlled_compute_overhead_seconds=self.condition.compute_seconds,
                   drone_compute_energy_j=completed.compute_energy_j)
        body=dict(job_id=completed.job_id,service=completed.service_seconds,energy=completed.compute_energy_j)
        self.transport.send(job['agent'],'remote_compute_overhead_done',
            route=(f'process:{self.condition.compute_seconds:.17g}',),wire=encode(body),
            generated_at=self.transport.now,version=job['version'],delayed=False)

    def _receive(self,packet):
        if packet.kind=='remote_compute_overhead_done':
            value=packet.payload
            return self._finish(value['job_id'],value['service'],value['energy'])
        if packet.kind=='remote_cloud_observation' and self.condition.ideal_communication:
            # Perfect transport has no transmission callback. The matched
            # reference still enters the real cloud pool, with zero transit.
            self.jobs[packet.payload['job_id']]['backhaul_transmission_finish']=self.transport.now
        return super()._receive(packet)

def inference_class(condition):
    if not isinstance(condition,InferenceCondition):
        raise ValueError('An immutable inference condition is required')
    return type('FixedRoute_'+condition.name.replace('-','_'),(FixedRouteInference,),
                dict(condition=condition,delay_steps=condition.peer_delay_steps))
