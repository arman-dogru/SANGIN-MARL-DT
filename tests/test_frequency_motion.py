"""Single-request DT bundles, delayed slow commands and causal replay."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch

from panama_uav.config import ExperimentMode
from panama_uav.environment import ExperimentEnvironment
from panama_uav.frequency_motion import FrequencyMotion,MotionSettings,speed_assignment
from panama_uav.inference_controls import InferenceCondition
from panama_uav.models import DQN,select_action
from panama_uav.redesign import configured,settings_for,experimental_class,run_frequency_episode
from panama_uav.redesign_audit import audit_redesign
from panama_uav.scenarios import Scenario
from test_remote_execution import remote_config


def fixture(tmp_path,starts=((1,1),(3,1),(5,1)),goals=((1,15),(3,15),(5,15)),settings=None,parameters=None):
    settings=settings or settings_for('A4',3.)
    path=tmp_path/'frequency.map'
    path.write_text('type octile\nheight 7\nwidth 20\nmap\n'+'\n'.join(['.'*20]*7)+'\n')
    scenario=Scenario(path,path,tuple(starts),tuple(goals));config=configured(remote_config(**(parameters or {})),settings)
    env=ExperimentEnvironment(config,ExperimentMode.UAV,path,len(starts),250,17,scenario)
    env.remote_communication_class=experimental_class(InferenceCondition('edge','edge'))
    env.remote_inference=env.remote_shadow_inference=lambda *args:3
    env.reset();motion=FrequencyMotion(env,settings);env.communication.motion=motion
    return env,motion,scenario,config


def cycle(env,motion,n):
    c=env.communication;start=n*env.config.communication.seconds_per_step
    if n:c.update(env.agents,start)
    c.prepare_decision(env.agents,start,env.config.resource_allocation.decision_budget_seconds)
    motion.step(n)


@pytest.mark.parametrize('interval',[.35,1.])
def test_frequencies_are_per_step_not_a_continuous_motion_clock(tmp_path,interval):
    env,motion,_,_=fixture(tmp_path,settings=MotionSettings(control_interval_seconds=interval,horizon_seconds=5.))
    motion.speeds={0:dict(speed_class='slow',decisions_per_step=.5),1:dict(speed_class='normal',decisions_per_step=1.),
                   2:dict(speed_class='fast',decisions_per_step=2.)}
    for n in range(4):cycle(env,motion,n)
    assert motion.distance=={0:3,1:4,2:8}
    assert len(env.communication.jobs)==12
    assert sum(e['event']=='remote_inference' for e in env.communication.remote_events)==12
    extra=[e for e in env.communication.remote_events if e['event']=='bundle_inference']
    assert len(extra)==4
    slow=[e for e in env.communication.remote_events if e['event']=='execution_decision_event' and e['robot_id']==0]
    assert slow[0]['slow_pipeline_warmup'] and slow[0]['action']==4
    assert [e['source_cycle'] for e in slow[1:]]==[0,1,2]


def test_fast_second_move_checks_intermediate_cell_and_local_obstacle(tmp_path):
    env,motion,_,_=fixture(tmp_path)
    motion.speeds={r:dict(speed_class='fast',decisions_per_step=2.) for r in range(3)}
    env.grid[1,3]=1
    cycle(env,motion,0)
    events=[e for e in env.communication.remote_events if e['event']=='frequency_motion_event']
    assert events[0]['after']['0']==[1,2]
    assert all(e['after']['0']!=[1,3] for e in events)
    assert all(sum(abs(a-b) for a,b in zip(e['before']['0'],e['after']['0']))<=1 for e in events)
    assert not any(e['conflicts_after_resolution'] for e in env.communication.remote_events if e['event']=='frequency_motion_event')


def test_slow_previous_command_is_used_without_a_replacement_request(tmp_path):
    env,motion,_,_=fixture(tmp_path)
    motion.speeds={r:dict(speed_class='slow',decisions_per_step=.5) for r in range(3)}
    cycle(env,motion,0);cycle(env,motion,1)
    event=[e for e in env.communication.remote_events if e['event']=='frequency_motion_event'][-1]
    assert event['due']==[0,1,2] and set(event['reasons'].values())=={'move_attempt'}
    assert all(v['source_cycle']==0 and v['cycle']==1 for v in event['contexts'].values())
    assert len(env.communication.jobs)==6


def test_movement_frequency_naturally_reverses_astar_distance_priorities(tmp_path):
    env,motion,_,_=fixture(tmp_path,goals=((1,8),(3,10),(5,15)))
    motion.speeds={0:dict(speed_class='slow',decisions_per_step=.5),1:dict(speed_class='fast',decisions_per_step=2.),
                   2:dict(speed_class='normal',decisions_per_step=1.)}
    for n in range(4):cycle(env,motion,n)
    assert any(e['cause']=='movement' and {e['first'],e['second']}=={0,1} for e in motion.reversal_events)
    assert all(a.priority==a.distance_to_goal+a.id*1e-6 for a in env.agents)


def test_speed_assignments_are_balanced_and_paired():
    values=speed_assignment(32,123)
    assert values==speed_assignment(32,123)
    counts=[sum(v['speed_class']==c for v in values.values()) for c in ('slow','normal','fast')]
    assert max(counts)-min(counts)<=1


def test_no_new_deadline_or_requests_for_second_bundled_action(tmp_path):
    env,motion,_,_=fixture(tmp_path)
    motion.speeds={r:dict(speed_class='fast',decisions_per_step=2.) for r in range(3)}
    c=env.communication;c.prepare_decision(env.agents,0.,.25)
    jobs=deepcopy(c.jobs);network_generated=c.transport.counters['messages_generated']
    motion.step(0)
    for task in c.jobs:
        assert c.jobs[task]['required_cycles']==jobs[task]['required_cycles']
    assert len(c.jobs)==len(jobs)
    assert c.transport.counters['messages_generated']==network_generated
    assert c.transport.now==.25


def test_episode_replay_telemetry_and_energy(tmp_path):
    settings=settings_for('A4',2.)
    _,_,scenario,config=fixture(tmp_path,settings=settings)
    torch.set_num_threads(1);torch.manual_seed(5);model=DQN((3,15,15),config.vector_length).eval()
    case=dict(case_id='fixture',num_agents=3,seed=17,starts=list(map(list,scenario.starts)),goals=list(map(list,scenario.goals)))
    row,trace=run_frequency_episode(config,model,case,torch.device('cpu'),0,scenario,InferenceCondition('edge','edge'),settings)
    infer=lambda location,robot,obs:select_action(model,obs,0.,np.random.default_rng(0),torch.device('cpu'))
    audit=audit_redesign(trace,infer)
    assert audit['status']=='passed'
    from panama_uav.redesign_metrics import measure
    row,tasks=measure(row,trace,'A4',dict(id='W1'),InferenceCondition('edge','edge'))
    assert row['post_resolution_conflicts']==0 and row['extra_remote_requests']==0
    assert len(tasks)==18
    assert row['modeled_energy_lower_j']==pytest.approx(row['radio_energy_lower_j']+row['drone_compute_energy_j'])
    assert all(e['time']==pytest.approx(e['cycle']*.35+.25) for e in trace['remote_events'] if e['event']=='frequency_motion_event')


def test_final_epoch_convention():
    s=settings_for('A4')
    assert len([n for n in range(1000) if n*s.control_interval_seconds+s.command_deadline_seconds<=250+1e-12])==714


@pytest.mark.parametrize('starts,goals,kind',[
    (((1,1),(1,3)),((1,15),(1,0)),'vertex'),
    (((1,1),(1,2)),((1,15),(1,0)),'edge_swap')])
def test_each_discrete_round_checks_vertex_and_edge_conflicts(tmp_path,starts,goals,kind):
    env,motion,_,_=fixture(tmp_path,starts,goals)
    env.communication.inference=env.communication.shadow_inference=lambda location,robot,obs:3 if robot==0 else 2
    motion.speeds={r:dict(speed_class='fast',decisions_per_step=2.) for r in range(2)}
    cycle(env,motion,0)
    events=[e for e in env.communication.remote_events if e['event']=='frequency_motion_event']
    assert any(c['kind']==kind for c in events[0]['conflicts_before_resolution'])
    assert all(not e['conflicts_after_resolution'] for e in events)


@pytest.mark.parametrize('deadline',[.15,.25,.30])
def test_configured_cutoff_is_single_shared_timestamp(tmp_path,deadline):
    settings=MotionSettings(command_deadline_seconds=deadline)
    env,motion,_,_=fixture(tmp_path,settings=settings)
    motion.speeds={r:dict(speed_class='fast',decisions_per_step=2.) for r in range(3)}
    cycle(env,motion,0)
    events=[e for e in env.communication.remote_events if e['event']=='frequency_motion_event']
    assert {e['time'] for e in events}=={deadline}
    assert len(env.communication.jobs)==3
