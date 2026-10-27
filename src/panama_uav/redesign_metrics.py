"""Existing task clocks plus physical-motion and fallback exposure accounting."""
from collections import Counter,defaultdict
import math
import numpy as np

from .remote_metrics import remote_metrics


def ratio(a,b):
    return a/b if b else None


def measure(row,trace,treatment,weight,condition):
    row=dict(row,**remote_metrics(trace))
    settings=trace['remote_settings'];duration=trace['frames'][-1]['time']
    decisions=[e for e in trace['remote_events'] if e['event']=='decision_event']
    active=[e for e in decisions if not e['completed_local_control']]
    executions=[e for e in trace['remote_events'] if e['event']=='execution_decision_event']
    executable=[e for e in executions if not e['completed_local_control'] and not e['slow_pipeline_warmup']]
    first={(e['cycle'],e['robot_id']):e for e in executions if e['substep']==0}
    used=defaultdict(list)
    for e in executions:
        if e['command'] and e['action_source']=='remote_command':used[e['command']['job_id']].append(e)
    row['nominal_commands_selected']=row['commands_used']
    row['commands_used']=len(used)
    row['tasks_used']=len(used)
    motion=[e for e in trace['remote_events'] if e['event']=='frequency_motion_event']
    by_cycle={(e['cycle'],e['robot_id']):e for e in decisions}
    command_motion=Counter();fallback_motion=Counter();blocked=Counter();reasons=Counter()
    for event in motion:
        for r,reason in event['reasons'].items():
            reasons[reason]+=1
            instruction=event['contexts'].get(r)
            if event['before'][r]!=event['after'][r] and instruction:
                if instruction['command'] and instruction['action_source']=='remote_command':
                    command_motion[instruction['command']['job_id']]+=1
                if instruction['action_source']=='cached_fallback':
                    fallback_motion[instruction['cycle'],event['substep'],int(r)]+=1
            elif instruction and reason=='physical_resolver_block':blocked[instruction['cycle'],int(r)]+=1
    identity=dict(case_id=row['case_id'],num_agents=row['num_agents'],treatment=treatment,
        weight_id=weight['id'],condition=condition.name,route=condition.route,
        control_interval_seconds=settings['seconds_per_step'],command_deadline_seconds=settings['cutoff_seconds'])
    tasks=[]
    delayed_accepted={e['command']['job_id'] for e in trace['remote_events'] if e['event']=='delayed_command_arrived' and e['accepted']}
    inference_by_job={e['job_id']:e for e in trace['remote_events'] if e['event']=='remote_inference'}
    for raw in trace['resource_tasks']:
        cycle=raw['report']['cycle'];decision=by_cycle.get((cycle,raw['agent']))
        delivered=raw.get('delivered_at');source=raw['source_time']
        generated=inference_by_job.get(raw['job_id'],{}).get('time')
        speed=settings['speed_assignments'][raw['agent']]
        execution_cycle=cycle+(1 if speed['speed_class']=='slow' else 0)
        execution=first.get((execution_cycle,raw['agent']))
        uses=used[raw['job_id']]
        task=dict({k:v for k,v in raw.items() if not isinstance(v,(dict,list))},**identity,cycle=cycle,
            speed_class=speed['speed_class'],movement_decisions_per_step=speed['decisions_per_step'],
            command_lag_steps=execution_cycle-cycle,scheduled_execution_cycle=execution_cycle,
            scheduled_execution_cutoff=execution_cycle*settings['seconds_per_step']+settings['cutoff_seconds'],
            nominal_max_cells_per_second=speed['decisions_per_step']/settings['seconds_per_step'],
            cycle_start=cycle*settings['seconds_per_step'],command_generated_at=generated,
            e2e_latency_seconds=delivered-source if delivered is not None else None,
            cycle_latency_seconds=delivered-cycle*settings['seconds_per_step'] if delivered is not None else None,
            transport_latency_seconds=delivered-generated if delivered is not None else None,
            action_source=execution['action_source'] if execution else None,
            nominal_cutoff_action_source=decision['action_source'] if decision else None,
            cached_fallback_invoked=bool(execution and execution.get('cached_fallback')),
            command_considered=bool(decision and decision['action_source']=='remote_command'),
            command_used=bool(uses),execution_at=uses[0]['time'] if uses else None,
            accepted_for_delayed_execution=raw['job_id'] in delayed_accepted,
            command_actions_executed=len(uses),execution_censored=execution is None,
            command_influenced_movement=bool(command_motion[raw['job_id']]),
            command_movement_cells=command_motion[raw['job_id']],mission_success=row['success'])
        if delivered is not None and not math.isclose(task['e2e_latency_seconds'],raw['observed_latency_seconds'],abs_tol=1e-10):
            raise ValueError('Existing E2E latency disagrees with task timestamps')
        tasks.append(task)
    snapshots=[e['summary'] for e in trace['remote_events'] if e['event']=='consistency_event']
    counters=('directed_pairs','ordering_errors','uplink_errors','downlink_errors','relevant_pairs','fov_errors',
        'fov_uplink_errors','fov_downlink_errors','physical_relevant_pairs','missed_physical_conflicts',
        'false_positive_conflicts','vertex_relevant_pairs','edge_relevant_pairs','action_order_known_pairs',
        'action_dt_ordering_errors','vertex_e2e_errors','vertex_aoi_error_sum','ordering_pairs_verified',
        'physical_relevant_ordering_errors','missed_vertex_conflicts','false_positive_vertex_conflicts',
        'active_peer_aoi_sum','active_peer_aoi_samples')
    for name in counters:row[name]=sum(s[name] for s in snapshots)
    # The existing summary defines xi_aoi as sum(age-weighted errors)/max(1,relevant).
    row['aoi_weighted_error_sum']=sum(s['xi_aoi']*max(1,s['relevant_pairs']) for s in snapshots)
    row.update(xi_fov=ratio(row['fov_errors'],row['relevant_pairs']),
        xi_aoi=ratio(row['aoi_weighted_error_sum'],row['relevant_pairs']),
        xi_fov_ul=ratio(row['fov_uplink_errors'],row['relevant_pairs']),
        xi_fov_dl=ratio(row['fov_downlink_errors'],row['relevant_pairs']),
        unrestricted_ordering_error_rate=ratio(row['ordering_errors'],row['directed_pairs']))
    shadows=[e for e in trace['oracle_events'] if e['event']=='controlled_peer_shadow']
    row.update(inferences=len(shadows),inference_pairs=sum(e['summary']['directed_pairs'] for e in shadows),
        inference_relevant_pairs=sum(e['summary']['relevant_pairs'] for e in shadows),
        inference_fov_errors=sum(e['summary']['fov_errors'] for e in shadows),
        stale_fresh_dqn_disagreements=sum(e['stale_vs_fresh_disagreement'] for e in shadows),
        peer_aoi_sum=sum(p['peer_age_seconds'] for e in shadows for p in e['pairs']),
        peer_aoi_samples=sum(len(e['pairs']) for e in shadows),
        physical_fov_errors=sum(p['physical_relevant']*p['ordering_error'] for e in shadows for p in e['pairs']))
    fallback=[e for e in trace['oracle_events'] if e['event']=='fallback_shadow']
    fallback_reasons=Counter((e.get('local_control') or {}).get('reason') for e in executable if e['cached_fallback'])
    invoked=[e for e in fallback if e['invoked']]
    extra=[e for e in executable if e['substep']==1]
    reversals=[e for e in trace['oracle_events'] if e['event']=='priority_reversal_event']
    row.update(identity,observed_seconds=duration,
        makespan_seconds=duration if row['success'] else None,eligible_decisions=len(executable),
        nominal_remote_opportunities=len(active),eligible_first_movements=sum(e['substep']==0 for e in executable),
        command_decisions=sum(e['action_source']=='remote_command' for e in executable),
        cached_fallback_decisions=sum(e['cached_fallback'] for e in executable),
        safety_wait_decisions=sum(e['action_source']=='safety_wait' for e in executable),
        controller_wait_decisions=sum(e['action']==4 for e in executable),
        fallback_movements=sum(fallback_motion.values()),fallback_decisions_with_movement=len(fallback_motion),
        speed_limited_nonmovement=0,slow_pipeline_warmup_waits=sum(e['slow_pipeline_warmup'] for e in executions),
        slow_delayed_command_uses=sum(e['speed_class']=='slow' and e['command'] is not None for e in executable),
        slow_late_command_uses=sum(e['speed_class']=='slow' and e['command'] is not None
            and e['arrived_at']>e['command']['expires_at']+1e-12 for e in executable),
        fast_bundled_second_actions=sum(e['action_source']=='remote_command' for e in extra),
        fast_bundle_prediction_mismatches=sum(bool(e['bundle_prediction_mismatch']) for e in extra),
        extra_local_decisions=sum(e['cached_fallback'] for e in extra),extra_remote_requests=0,
        extra_cached_movements=sum(v for (cycle,substep,robot),v in fallback_motion.items() if substep==1),
        deadline_fallback_movements=sum(v for (cycle,substep,robot),v in fallback_motion.items() if substep==0),
        extra_stale_fresh_disagreements=sum(e['stale_vs_fresh_disagreement'] for e in fallback if e['substep']==1),
        extra_fallback_diagnostic_pairs=sum(e['substep']==1 for e in fallback),
        physical_resolver_blocks=reasons['physical_resolver_block'],
        local_obstacle_blocks=reasons['local_obstacle']+fallback_reasons['local_obstacle'],
        local_occupancy_waits=fallback_reasons['local_occupancy'],
        invalid_cached_route_stops=sum(fallback_reasons[k] for k in ('cached_path_unavailable','cached_path_exhausted','cached_step_not_adjacent')),
        fallback_shadow_opportunities=len(fallback),fallback_stale_fresh_disagreements=sum(e['stale_vs_fresh_disagreement'] for e in fallback),
        invoked_stale_fresh_disagreements=sum(e['stale_vs_fresh_disagreement'] for e in invoked),
        invoked_fallback_diagnostic_pairs=len(invoked),
        fallback_fresh_dqn_disagreements=sum(e['fallback_vs_fresh_dqn_disagreement'] for e in fallback),
        fresh_fallback_fresh_dqn_disagreements=sum(e['fresh_fallback_vs_fresh_dqn_disagreement'] for e in fallback),
        movement_priority_reversals=sum(e['cause']=='movement' for e in reversals),
        relevant_movement_priority_reversals=sum(e['cause']=='movement' and e['physically_relevant'] for e in reversals),
        completion_priority_reversals=sum(e['cause']=='goal_completion' for e in reversals),
        active_pair_seconds=trace['physical_summary']['active_pair_seconds'],
        physical_distance_cells=sum(trace['physical_summary']['distance_cells'].values()),
        attempted_vertex_conflicts=sum(c['kind']=='vertex' for e in motion for c in e['conflicts_before_resolution']),
        attempted_edge_swaps=sum(c['kind']=='edge_swap' for e in motion for c in e['conflicts_before_resolution']),
        post_resolution_conflicts=sum(len(e['conflicts_after_resolution']) for e in motion),
        waiting_cell_opportunities=sum(reason in ('controller_wait','physical_resolver_block','local_obstacle',
            'cached_route_unalignable_or_exhausted','cached_route_nonadjacent','cached_priority_yield')
            and not e['completed_before'][r] for e in motion for r,reason in e['reasons'].items()),
        controller_wait_seconds=sum(e['action']==4 and e['substep']==0 for e in executable)*settings['seconds_per_step'],
        executed_fresh_dqn_disagreements=sum(e['actual_vs_fresh_dqn_disagreement'] for e in fallback))
    row['waiting_steps']=sum(e['substep']==0 and not e['completed_before'][r] and e['before'][r]==e['after'][r]
        for e in motion for r in e['before'])
    row['scheduled_command_misses']=sum(e['command_cutoff_missed'] for e in executable)
    row['scheduled_cutoff_waits']=sum(e['command_cutoff_missed'] and e['action']==4 for e in executable)
    row['physical_distance_m']=row['physical_distance_cells']*settings['meters_per_cell']
    row['delayed_commands_accepted']=len(delayed_accepted)
    row['mean_first_completion_time_seconds']=float(np.mean([v for v in motion[-1]['first_completion_times'].values()
        if v is not None])) if motion and any(v is not None for v in motion[-1]['first_completion_times'].values()) else None
    row['extra_wait_decisions']=sum(e['action']==4 and not e['completed_local_control'] for e in extra)
    row['deadline_fallback_yields']=sum(e['cached_fallback'] and bool((e.get('local_control') or {}).get('blocking_peers')) for e in executable)
    for pool in ('drone','cloud'):
        events=[e for e in trace['transport_events'] if e['event']=='compute_allocation' and e['pool']==pool]
        area=jobs=busy=0.;peak=0
        for i,e in enumerate(events):
            elapsed=max(0.,min(duration,events[i+1]['time'] if i+1<len(events) else duration)-e['time'])
            area+=elapsed*e['utilization'];jobs+=elapsed*len(e['jobs']);busy+=elapsed*bool(e['jobs'])
            peak=max(peak,len(e['jobs']))
        row.update({pool+'_utilization_seconds':area,pool+'_cpu_utilization':ratio(area,duration),
            pool+'_concurrent_job_seconds':jobs,pool+'_peak_jobs':peak,pool+'_mean_active_jobs':ratio(jobs,busy)})
    row['inference_wall_seconds']=sum(t.get('inference_wall_seconds',0.)+t.get('bundle_inference_wall_seconds',0.) for t in tasks)
    row['allocation_solver_wall_seconds']=sum(e['binary']['solve_seconds']+e['relaxed']['solve_seconds']
        for e in trace['transport_events'] if e['event']=='paper_allocation')
    if row['deadline_commands_on_time']+row['deadline_commands_late']+row['deadline_commands_undelivered']!=len(tasks):
        raise ValueError('Command partition does not reconcile')
    if abs(row['modeled_energy_lower_j']-row['radio_energy_lower_j']-row['drone_compute_energy_j'])>1e-7:
        raise ValueError('Energy scope mismatch')
    return row,tasks


def extend_legacy(row,trace,observations):
    """Add comparable read-only metrics after the exact A0 signature is saved."""
    from .remote_control import DELTAS,conflicts
    row=dict(row,**observations);prior={};reversals=completion_reversals=attempted_vertex=attempted_edges=post=0
    active_pairs=0.;last=0.
    for event in trace['remote_events']:
        if event['event']=='robot_state_event':prior[event['report']['robot_id']]=event['physical']
        elif event['event']=='motion_event':
            after={int(r):v for r,v in event['physical'].items()}
            active=sum(not p['completed'] for p in prior.values())
            active_pairs+=active*(active-1)/2*(event['time']-last);last=event['time']
            for a in prior:
                for b in prior:
                    if a>=b or prior[a]['completed'] or prior[b]['completed']:continue
                    if (prior[a]['priority']<prior[b]['priority'])!=(after[a]['priority']<after[b]['priority']):
                        if after[a]['completed'] or after[b]['completed']:completion_reversals+=1
                        else:reversals+=1
            before={int(r):tuple(p) for r,p in event['before'].items()};targets={}
            grid=np.asarray(trace['grid'])
            for r,point in before.items():
                dr,dc=DELTAS[event['actions'][str(r)]];target=point[0]+dr,point[1]+dc
                targets[r]=target if 0<=target[0]<grid.shape[0] and 0<=target[1]<grid.shape[1] and not grid[target] else point
            attempted=conflicts(before,targets)
            attempted_vertex+=sum(e['kind']=='vertex' for e in attempted);attempted_edges+=sum(e['kind']=='edge_swap' for e in attempted)
            post+=len(conflicts(before,{int(r):tuple(p) for r,p in event['after'].items()}));prior=after
    decisions=[e for e in trace['remote_events'] if e['event']=='decision_event' and not e['completed_local_control']]
    row.update(eligible_decisions=len(decisions),command_decisions=sum(e['command'] is not None for e in decisions),
        cached_fallback_decisions=0,fallback_movements=0,safety_wait_decisions=sum(e['action']==4 and e['command'] is None for e in decisions),
        controller_wait_seconds=sum(e['action']==4 for e in decisions)*trace['remote_settings']['seconds_per_step'],
        movement_priority_reversals=reversals,completion_priority_reversals=completion_reversals,active_pair_seconds=active_pairs,
        attempted_vertex_conflicts=attempted_vertex,attempted_edge_swaps=attempted_edges,post_resolution_conflicts=post,
        xi_fov=ratio(row['fov_errors'],row['relevant_pairs']))
    for pool in ('drone','cloud'):
        row[pool+'_cpu_utilization']=ratio(row[pool+'_utilization_seconds'],row['observed_seconds'])
    return row
