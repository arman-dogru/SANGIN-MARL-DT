"""Independent discrete-frequency and diagnostic replay for redesigned episodes."""
import math
import numpy as np

from .environment import astar_path
from .information import InformationCache
from .remote_control import DELTAS, cached_path_action, completed_action, conflicts, resolve_delivered
from .remote_state import PhysicalReport, observation_hash


def replay_execution(event,positions,physical,caches,decisions,executions,delayed,bundles,settings,grid):
    """Rebuild action selection from delivered commands and locally sensed cells."""
    from .remote_audit import digest
    robot,cycle,substep=event['robot_id'],event['cycle'],event['substep']
    speed=settings['speed_assignments'].get(str(robot),settings['speed_assignments'].get(robot))['speed_class']
    if substep and speed!='fast':raise ValueError('A non-fast robot received a second action')
    row,col=positions[robot];radius=settings['radius'];now=event['time']
    occupancy=np.zeros((2*radius+1,2*radius+1),dtype=int)
    for peer,point in positions.items():
        dr,dc=point[0]-row,point[1]-col
        if peer!=robot and max(abs(dr),abs(dc))<=radius:occupancy[dr+radius,dc+radius]=1
    obstacles=np.pad(grid,radius,constant_values=1)[row:row+2*radius+1,col:col+2*radius+1]
    report=PhysicalReport(robot,cycle,now,(row,col),tuple(physical[robot]['goal']),physical[robot]['completed'],
        0.,obstacles.tolist(),occupancy.tolist(),f'report:{cycle}:{robot}')
    if True:  # leo_study: half-rate slow robots use the current-cycle command
        parent=executions[robot] if substep else decisions[robot]
        command=parent['command'];arrival=parent['arrived_at']
    bundle=bundles.get(command['job_id']) if command else None
    completed=physical[robot]['completed'];warmup=False
    cached=False;sources={};details=None;mismatch=None
    if completed:
        command=None;arrival=None;priority=grid.size+robot*1e-6
        action,sources,details=completed_action(report,caches[robot],priority)
        source='completed_local_control'
    elif command:
        if substep:
            if not bundle or bundle['actions'][0]!=command['action']:raise ValueError('Missing fast command bundle')
            action=bundle['actions'][1];priority=bundle['dt_priorities'][str(robot)]
            sources=bundle['sources'];mismatch=list(positions[robot])!=bundle['predicted_position']
        else:action=command['action'];priority=command['dt_priorities'][str(robot)];sources=command['sources']
        source='remote_command'
    else:
        priority=caches[robot][robot].value['priority']
        if settings['missed_command_fallback']=='cached_astar' and not warmup:
            action,sources,details=cached_path_action(report,caches[robot],settings['horizon']);cached=True
        else:action=4
        source='cached_fallback' if cached and action!=4 else 'safety_wait'
    expected=dict(robot_id=robot,cycle=cycle,substep=substep,cutoff_time=now,action=action,command=command,
        arrived_at=arrival,source_cycle=command['cycle'] if command else None,
        scheduled_source_cycle=cycle,
        command_cutoff_missed=command is None and not completed and not warmup,slow_pipeline_warmup=warmup,
        completed_local_control=completed,cached_fallback=cached,action_source=source,local_sources=sources,
        local_control=details,resolution_priority=priority,
        decision_reason='slow_active_step' if speed=='slow' else 'fast_bundle_second' if substep else 'current_command',
        speed_class=speed,bundle_prediction_mismatch=mismatch,
        bundle_inference_event=bundle['second_inference_event'] if bundle and substep else None)
    actual={k:v for k,v in event.items() if k not in ('event','schema','event_id','time','checksum','previous_checksum')}
    if digest(actual)!=digest(expected):raise ValueError('Execution used an unavailable command, priority or local input')
    return expected


def replay_frequency(event,positions,decisions,executions,settings,grid,physical):
    from .remote_audit import digest
    from .frequency_motion import eligible
    now = event["time"]
    if digest(event["before"]) != digest(positions):
        raise ValueError("Physical motion starts from an unknown state")
    if abs(now-(event['cycle']*settings['seconds_per_step']+settings['cutoff_seconds']))>1e-10:
        raise ValueError('Movement does not occur at the existing cycle cutoff')
    targets,reasons,due,expected_contexts=dict(positions),{},[],{}
    for robot,point in positions.items():
        assignment=settings['speed_assignments'].get(str(robot),settings['speed_assignments'].get(robot))
        speed=assignment['speed_class']
        participates=eligible(speed,event['cycle'],event['substep'],assignment.get('phase',0))
        off_step=speed=='slow' and not eligible(speed,event['cycle'],0,assignment.get('phase',0))
        decision=executions.get(robot)
        if off_step:
            # Half-rate off step: verify the invariants of the recorded speed-limited context.
            recorded=event['contexts'][str(robot)]
            if (recorded['action']!=4 or recorded['command'] is not None or recorded['command_cutoff_missed']
                    or recorded['cached_fallback'] or recorded['cycle']!=event['cycle']
                    or recorded['action_source']!=('speed_limited' if event['substep']==0 else 'not_scheduled')):
                raise ValueError('Off-step slow robot context violates the half-rate protocol')
            expected_contexts[robot]=dict(recorded)
        else:
            expected_contexts[robot]=dict(decision) if participates else dict(decision,action=4,substep=event['substep'],command=None,action_source='not_scheduled')
        action=decision['action'] if participates else 4
        if event['actions'][str(robot)]!=action:raise ValueError('Movement violates decision frequency or selected action')
        if participates:due.append(robot)
        reasons[robot]=('speed_limited' if event['substep']==0 else 'not_scheduled') if not participates else 'slow_pipeline_warmup' if decision['slow_pipeline_warmup'] else 'controller_wait' if action==4 else 'move_attempt'
        dr,dc=DELTAS[action];target=point[0]+dr,point[1]+dc
        if not (0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1] and not grid[target]):
            target=point;reasons[robot]='local_obstacle'
        targets[robot]=target
    collisions = conflicts(positions,targets)
    priorities = {r:context['resolution_priority'] for r,context in expected_contexts.items()}
    final = positions if collisions and settings["on_collision"] == "terminate" else resolve_delivered(positions,targets,priorities)
    for r in positions:
        if final[r] == positions[r] and reasons[r] == "move_attempt":
            reasons[r] = "physical_resolver_block"
    for actual,expected,name in ((event["intended"],targets,"targets"),(event["after"],final,"positions"),
        (event["due"],due,"opportunities"),(event["reasons"],reasons,"motion reasons"),
        (event["conflicts_before_resolution"],collisions,"attempted conflicts"),
        (event["conflicts_after_resolution"],conflicts(positions,final),"resolved conflicts"),
        (event['contexts'],expected_contexts,'action contexts')):
        if digest(actual) != digest(expected):raise ValueError("Physical replay mismatch: "+name)
    new_physical = {int(r):v for r,v in event["physical"].items()}
    for r,values in new_physical.items():
        goal = tuple(physical[r]["goal"])
        path = astar_path(grid,final[r],goal)
        completed = physical[r]["completed"] or final[r] == goal
        priority = (grid.size if completed else len(path)-1 if path else grid.size-1)+r*1e-6
        if tuple(values["position"]) != final[r] or values["priority"] != priority or values["completed"] != completed:
            raise ValueError("Physical path-distance priority or completion mismatch")
        if [tuple(p) for p in values["path"]] != path:
            raise ValueError("Physical A* path mismatch")
    return dict(final),new_physical


def audit_redesign(trace,inference=None):
    from .remote_audit import digest
    from .staleness_audit import audit_stress
    result = audit_stress(trace,inference)
    active = {(e["cycle"],e["robot_id"]) for e in trace["remote_events"]
              if e["event"] == "decision_event" and not e["completed_local_control"]}
    executions={(e['cycle'],e['substep'],e['robot_id']) for e in trace['remote_events']
        if e['event']=='execution_decision_event' and not e['completed_local_control']}
    diagnostics = [e for e in trace["oracle_events"] if e["event"] == "fallback_shadow"]
    if {(e["cycle"],e['substep'],e["robot_id"]) for e in diagnostics} != executions or len(diagnostics) != len(executions):
        raise ValueError("Fallback diagnostic coverage mismatch")
    dqn = {(e["cycle"],e["robot_id"]):e for e in trace["oracle_events"] if e["event"] == "decision_action_shadow"}
    if set(dqn) != active:raise ValueError("DQN oracle coverage mismatch")
    for shadow in dqn.values():
        if observation_hash(shadow["observation"]) != shadow["observation_hash"]:
            raise ValueError("DQN diagnostic observation hash mismatch")
        if inference and int(inference("cloud",shadow["robot_id"],shadow["observation"])) != shadow["oracle_action"]:
            raise ValueError("DQN diagnostic action mismatch")
    for shadow in diagnostics:
        if 'fresh_dqn_observation' in shadow and inference:
            if int(inference('cloud',shadow['robot_id'],shadow['fresh_dqn_observation']))!=shadow['fresh_dqn_action']:
                raise ValueError('Second-round fresh DQN diagnostic mismatch')
        if shadow['actual_vs_fresh_dqn_disagreement']!=(shadow['actual_action']!=shadow['fresh_dqn_action']):
            raise ValueError('Executed action disagreement mismatch')
    bundles=[e for e in trace['remote_events'] if e['event']=='bundle_inference']
    task_ids={t['job_id'] for t in trace['resource_tasks']}
    if len(task_ids)!=len(trace['resource_tasks']):raise ValueError('Duplicate inference task')
    expected_tasks={f"{e['report']['cycle']}:{e['report']['robot_id']}" for e in trace['remote_events']
        if e['event']=='robot_state_event' and not e['report']['completed']}
    if task_ids!=expected_tasks:raise ValueError('Task workload differs from one request per active robot per step')
    for task in trace['resource_tasks']:
        if task['required_cycles']!=task['input_bits']*task['cycles_per_bit']:
            raise ValueError('Bundled execution changed logical compute work')
    fast={int(r) for r,speed in trace['remote_settings']['speed_assignments'].items() if speed['speed_class']=='fast'}
    expected_bundles={e['job_id'] for e in trace['remote_events'] if e['event']=='remote_inference' and e['robot_id'] in fast}
    if {e['job_id'] for e in bundles}!=expected_bundles or len(bundles)!=len(expected_bundles):
        raise ValueError('Fast remote inference bundle coverage mismatch')
    result.update(frequency_motion_events=sum(e["event"] == "frequency_motion_event" for e in trace["remote_events"]),
        bundled_second_inferences_verified=len(bundles),
        fallback_shadows_verified=len(diagnostics),cutoff_dqn_oracles_verified=len(dqn),policy_updates=0)
    return result
