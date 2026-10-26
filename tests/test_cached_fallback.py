"""Cached-path fallback changes only active missed-command execution."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from panama_uav.communication_effects import measurements
from panama_uav.evaluation import case_bank, run_episode
from panama_uav.inference_controls import InferenceCondition, inference_class
from panama_uav.information import InformationCache
from panama_uav.models import DQN
from panama_uav.remote_audit import digest, replay_trace
from panama_uav.remote_control import cached_path_action
from panama_uav.remote_state import ActionCommand, PhysicalReport
from test_remote_execution import environment, remote_config


def cache(robot, path, priority=1., completed=False, source=.02):
    result=InformationCache()
    result.accept(.05,dict(position=path[0] if path else (0,0),path=path,
        priority=priority,completed=completed),1,{robot:source})
    return result


def sensing(position=(0,0),radius=2):
    blank=np.zeros((radius*2+1,radius*2+1)).tolist()
    return PhysicalReport(0,0,.25,position,(0,3),False,3.,blank,deepcopy(blank),'fixture')


def test_cached_path_follows_successor_without_refreshing_state():
    states={0:cache(0,[(0,0),(0,1),(0,2)])}
    before=deepcopy(vars(states[0]))
    action,sources,details=cached_path_action(sensing(),states)
    assert action==3 and details['reason']=='cached_successor' and sources=={0:.02}
    assert vars(states[0])==before
    action,_,details=cached_path_action(sensing((0,1)),states)
    assert action==3 and details['path_index']==1


@pytest.mark.parametrize('path,position,reason',[
    ([],(0,0),'cached_path_unavailable'),
    ([(0,1),(0,2)],(0,0),'cached_path_unavailable'),
    ([(0,0)],(0,0),'cached_path_exhausted'),
    ([(0,0),(0,2)],(0,0),'cached_step_not_adjacent')])
def test_missing_stale_or_exhausted_paths_wait(path,position,reason):
    action,_,details=cached_path_action(sensing(position),{0:cache(0,path)})
    assert action==4 and details['reason']==reason


@pytest.mark.parametrize('priority,completed,expected',[(.5,False,4),(2.,False,3),(.5,True,3)])
def test_yielding_uses_cached_active_peer_priority(priority,completed,expected):
    states={0:cache(0,[(0,0),(0,1)],1.),1:cache(1,[(1,1),(0,1)],priority,completed)}
    assert cached_path_action(sensing(),states)[0]==expected


@pytest.mark.parametrize('field,reason',[('obstacles','local_obstacle'),('occupancy','local_occupancy')])
def test_local_obstacles_and_occupancy_prevent_invalid_fallback(field,reason):
    report=sensing()
    getattr(report,field)[2][3]=1
    action,_,details=cached_path_action(report,{0:cache(0,[(0,0),(0,1)])})
    assert action==4 and details['reason']==reason


def test_peer_intention_horizon_is_not_extended_by_the_controller():
    states={0:cache(0,[(0,0),(0,1)]),1:cache(1,[(1,2),(0,2),(0,1)],.5)}
    assert cached_path_action(sensing(),states,horizon=1)[0]==3
    assert cached_path_action(sensing(),states,horizon=2)[0]==4


@pytest.mark.parametrize('value',['unknown',None])
def test_unknown_fallback_configuration_rejected(value):
    with pytest.raises(ValueError,match='fallback'):
        remote_config(missed_command_fallback=value)


def test_on_time_remote_command_has_identical_execution_in_both_modes():
    traces=[]
    for mode in ('wait','cached_astar'):
        env=environment(remote_config(drone_cycles_per_second=1e11,missed_command_fallback=mode))
        env.prepare_decision(.1)
        assert all(c['command'] is not None for c in env.communication.command_contexts.values())
        traces.append((env.communication.remote_events,env.communication.execution_actions))
    assert traces[0]==traces[1]


def test_fallback_mode_cannot_change_paired_case_generation():
    wait=remote_config()
    cached=remote_config(missed_command_fallback='cached_astar')
    assert case_bank(wait,1,(2,4,8))==case_bank(cached,1,(2,4,8))


@pytest.mark.parametrize('arrival,on_time',[(.249,True),(.250,True),(.251,False)])
def test_fallback_preserves_249_250_251_ms_delivery_boundary(arrival,on_time):
    env=environment(remote_config(drone_cycles_per_second=1e5,cloud_cycles_per_second=1e5,
        missed_command_fallback='cached_astar',task_deadline_seconds=.25,
        command_cutoff_seconds=.25,decision_budget_seconds=.25))
    comm=env.communication
    comm.transport.advance(.05,comm._receive)
    comm.jobs['0:0']['compute_completed_at']=.05
    command=ActionCommand(0,0,4,.05,.25,0,'drone',{0:0.},'0:0','fixture',{}, {})
    comm._send_remote(0,'remote_action_command',dict(job_id='0:0',command=command.payload()),
        (f'process:{arrival-.05}',),version=0)
    env.prepare_decision(.25)
    context=comm.command_contexts[0]
    assert context.get('cached_fallback',False)==(not on_time)
    assert context['command_cutoff_missed']==(not on_time)
    if not on_time:comm.transport.advance(arrival,comm._receive)
    task=next(t for t in comm.task_records() if t['job_id']=='0:0')
    assert task['deadline_status']==('on_time' if on_time else 'late')
    assert task['command_accepted']==on_time
    assert ('used_at' in task)==on_time
    assert task['cutoff_induced_wait']==(not on_time and context['action']==4)


def fallback_trace():
    config=remote_config(drone_cycles_per_second=1e5,cloud_cycles_per_second=1e5,
        missed_command_fallback='cached_astar',task_deadline_seconds=.1)
    torch.manual_seed(42)
    model=DQN((3,15,15),config.vector_length).eval()
    condition=InferenceCondition('fallback-fixture','allocated')
    row,trace=run_episode(config,model,case_bank(config,1,(2,))[0],torch.device('cpu'),max_steps=3,
        remote_communication_class=inference_class(condition))
    return row,trace,condition


def test_multistep_fallback_replay_and_deadline_energy_accounting():
    row,trace,condition=fallback_trace()
    assert replay_trace(trace)['status']=='passed'
    row,tasks,_=measurements(row,trace,condition)
    decisions=[e for e in trace['remote_events'] if e['event']=='decision_event']
    assert row['cached_fallback_decisions']==row['command_cutoff_misses']==6
    assert row['cached_fallback_move_intents']>0
    assert row['cached_fallback_wait_intents']==row['command_cutoff_waits']==row['deadline_cutoff_wait_tasks']
    assert row['commands_used']==row['commands_delivered']==0
    assert row['deadline_commands_undelivered']==len(tasks)==6
    assert row['modeled_energy_lower_j']==pytest.approx(row['radio_energy_lower_j']+row['drone_compute_energy_j'])
    assert all(d['local_sources']=={'0':0.,'1':0.} for d in decisions)
    assert all(d['resolution_priority']==d['local_control']['cached_priority'] for d in decisions)


def test_replay_rejects_fallback_action_tampering_even_with_rehashed_events():
    _,trace,_=fallback_trace()
    event=next(e for e in trace['remote_events'] if e['event']=='decision_event' and e['action']!=4)
    event['action']=4
    previous='0'*64
    for event in trace['remote_events']:
        event['previous_checksum']=previous
        event['checksum']=digest({k:v for k,v in event.items() if k!='checksum'})
        previous=event['checksum']
    with pytest.raises(ValueError,match='controller replay'):
        replay_trace(trace)
