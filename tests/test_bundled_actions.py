"""Command applicability is separate from nominal delivery compliance."""
from dataclasses import replace
import numpy as np
import pytest

from panama_uav.bundled_actions import DelayedCommandInbox,second_input
from panama_uav.remote_state import ActionCommand,CommandInbox,PhysicalReport,observation_hash
from test_frequency_motion import fixture,cycle


def command(cycle=0):
    return ActionCommand(0,cycle,3,.1+cycle*.35,.25+cycle*.35,cycle,'drone',
        {0:cycle*.35},f'{cycle}:0','hash',{}, {0:5.})


@pytest.mark.parametrize('arrival,accepted',[(.599,True),(.600,True),(.601,False)])
def test_delayed_cutoff_boundary_does_not_extend_nominal_deadline(arrival,accepted):
    original=CommandInbox();delayed=DelayedCommandInbox(.35,.25);cmd=command()
    assert original.accept(cmd,arrival)[0] is False
    assert delayed.accept(cmd,arrival)[0] is accepted
    candidate=delayed.decision(1,.6)
    assert bool(candidate) is accepted
    assert cmd.expires_at==.25
    assert not delayed.accept(cmd,.6)[0]


def test_delayed_buffer_never_substitutes_current_or_older_source_cycles():
    inbox=DelayedCommandInbox(.35,.25)
    assert inbox.accept(command(1),.5)[0]
    assert inbox.decision(1,.6) is None
    assert inbox.decision(2,.95)[0].cycle==1
    assert inbox.decision(3,1.3) is None
    with pytest.raises(ValueError,match='already closed'):inbox.decision(3,1.3)


@pytest.mark.parametrize('fallback,expected_move',[('wait',False),('cached_astar',True)])
def test_slow_missing_current_command_preserves_treatment_rule(tmp_path,fallback,expected_move):
    from panama_uav.frequency_motion import MotionSettings
    env,motion,_,_=fixture(tmp_path,settings=MotionSettings(fallback=fallback),parameters=dict(drone_cycles_per_second=1e5,cloud_cycles_per_second=1e5))
    motion.speeds={r:dict(speed_class='slow',decisions_per_step=.5,command_lag_steps=0,phase=0) for r in range(3)}
    c=env.communication
    # Keep all uploads in flight; the initial briefing remains the only cache.
    for r in range(3):c.transport.agent_rate_caps[(r,'access_up')]=100.
    cycle(env,motion,0)
    actual=[e for e in c.remote_events if e['event']=='execution_decision_event' and e['cycle']==0]
    assert len(actual)==3 and all(e['command'] is None and e['cached_fallback']==expected_move for e in actual)
    assert all((e['action']!=4)==expected_move for e in actual)
    assert all((a.position[1]!=1)==expected_move for a in env.agents)
    positions=[a.position for a in env.agents]
    cycle(env,motion,1)  # off step: no movement, no uploads, no decisions
    assert [a.position for a in env.agents]==positions
    assert not [e for e in c.remote_events if e['event'] in ('execution_decision_event','decision_event') and e['cycle']==1]
    assert len(c.jobs)==3


def test_fast_actions_are_both_remote_dt_inferences_in_one_response(tmp_path):
    env,motion,_,_=fixture(tmp_path)
    motion.speeds={r:dict(speed_class='fast',decisions_per_step=2.,command_lag_steps=0) for r in range(3)}
    calls=[]
    def infer(location,robot,obs):
        calls.append((location,robot,obs['vector'].copy()))
        return 3
    env.communication.inference=infer
    cycle(env,motion,0)
    events=env.communication.remote_events
    first=[e for e in events if e['event']=='remote_inference']
    second=[e for e in events if e['event']=='bundle_inference']
    sent=[e for e in events if e['event']=='message_sent' and e['kind']=='remote_action_command']
    assert len(first)==len(second)==len(sent)==3 and len(calls)==6
    assert all(e['prediction']['predicted_position']==[env.agents[e['robot_id']].position[0],2] for e in second)
    for r in range(3):
        actions=[e for e in events if e['event']=='execution_decision_event' and e['robot_id']==r]
        assert len(actions)==2 and all(e['action_source']=='remote_command' for e in actions)
        assert actions[1]['bundle_inference_event'] is not None
        assert actions[0]['resolution_priority']-actions[1]['resolution_priority']==1.
    assert all(j['required_cycles']==1e6 for j in env.communication.jobs.values())


def test_second_input_uses_dt_peers_without_changing_replica(tmp_path):
    env,motion,_,_=fixture(tmp_path)
    c=env.communication;report=c._sense(env.agents[0],env.agents,0,0.)
    replica=c.replicas['drone'];before={r:cache.value.copy() for r,cache in replica.states.items()}
    params=dict(radius=env.fov_radius,waypoints=env.config.map.astar_waypoints,
        horizon=env.config.map.communicated_path_steps,map_extent=max(env.grid.shape))
    first=second_input(report,replica,env.grid,3,c._state_for_report,.1,params)
    env.agents[1].position=(1,2)  # unavailable physical truth must not enter the rollout
    again=second_input(report,replica,env.grid,3,c._state_for_report,.1,params)
    assert observation_hash(first[0])==observation_hash(again[0])
    assert first[4]['predicted_position']==[1,2]
    assert {r:cache.value for r,cache in replica.states.items()}==before


@pytest.mark.parametrize('fallback',['wait','cached_astar'])
def test_fast_missing_bundle_applies_same_fallback_to_both_rounds(tmp_path,fallback):
    from panama_uav.frequency_motion import MotionSettings
    env,motion,_,_=fixture(tmp_path,settings=MotionSettings(fallback=fallback),parameters=dict(drone_cycles_per_second=1e5,cloud_cycles_per_second=1e5))
    motion.speeds={r:dict(speed_class='fast',decisions_per_step=2.,command_lag_steps=0) for r in range(3)}
    for r in range(3):env.communication.transport.agent_rate_caps[(r,'access_up')]=100.
    cycle(env,motion,0)
    decisions=[e for e in env.communication.remote_events if e['event']=='execution_decision_event']
    assert len(decisions)==6 and len(env.communication.jobs)==3
    assert all(e['command'] is None and e['cached_fallback']==(fallback=='cached_astar') for e in decisions)
