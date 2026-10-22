"""Execution equivalence uses the actual remote-DT environment as reference."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from panama_uav.config import ExperimentMode, load_config
from panama_uav.environment import ExperimentEnvironment
from panama_uav.phase1 import clean_config
from panama_uav.phase1_environment import CleanMAPFEnvironment
from panama_uav.phase1_execution import RemoteAlignedCleanMAPFEnvironment
from panama_uav.scenarios import Scenario


ROOT = Path(__file__).resolve().parents[1]


def configurations():
    remote = load_config(ROOT/'configs/paper_remote_dt_250ms.toml')
    remote = replace(remote, communication=replace(remote.communication,
        profile='perfect', leo_timeline=None), resource_allocation=replace(
        remote.resource_allocation, on_collision='dt_resolve'))
    return clean_config(remote), remote


def environments(tmp_path, starts, goals, script, rows=None, cap=20):
    path = tmp_path/'execution.map'
    rows = rows or ['@@@@@@@']+['@.....@']*5+['@@@@@@@']
    path.write_text(f'type octile\nheight {len(rows)}\nwidth {len(rows[0])}\nmap\n'+'\n'.join(rows)+'\n')
    clean_cfg, remote_cfg = configurations()
    scenario = Scenario(tmp_path/'fixture.scen', path, tuple(starts), tuple(goals))
    args = (ExperimentMode.UAV, path, len(starts), cap, 91, scenario)
    clean = RemoteAlignedCleanMAPFEnvironment(clean_cfg, *args)
    remote = ExperimentEnvironment(remote_cfg, *args)
    remote.remote_inference = lambda location, robot, obs: script[min(remote.step_count,len(script)-1)][robot]
    clean.reset(); remote.reset()
    return clean, remote


def replay_equivalent(clean, remote, script):
    traces = []
    for proposed in script:
        observations = clean.observations()
        remote.prepare_decision(.25)
        epoch = clean.step_count
        events = [e for e in remote.communication.remote_events
            if e['event']=='remote_inference' and e['job_id'].startswith(f'{epoch}:')]
        assert {e['robot_id'] for e in events} == set(clean.active_robot_ids)
        for event in events:
            for name in ('spatial', 'vector'):
                np.testing.assert_array_equal(observations[event['robot_id']][name], event['observation'][name])
        selection = clean.prepare_actions(proposed)
        assert selection.executed == remote.communication.execution_actions
        assert selection.priorities == {r: c['resolution_priority']
            for r,c in remote.communication.command_contexts.items()}
        first = clean.step(proposed)
        second = remote.step(remote.communication.execution_actions.copy())
        assert [a.position for a in clean.agents] == [a.position for a in remote.agents]
        assert [a.ever_reached_goal for a in clean.agents] == [a.ever_reached_goal for a in remote.agents]
        assert first.rewards == second.rewards
        assert (first.terminated, first.truncated) == (second.terminated, second.truncated)
        for name in ('physical_conflicts', 'collisions_agent_agent_step', 'collisions_obstacle_step',
                     'targets_reached_step', 'active_yield_events_step', 'path_conflict_penalty_step'):
            assert first.info.get(name,0) == second.info.get(name,0)
        traces.append(first.info)
        if first.terminated or first.truncated:
            break
    return traces


def test_completed_robot_yields_and_returns_like_actual_remote_execution(tmp_path):
    script = [{0:1, 1:3}]*4
    clean, remote = environments(tmp_path, [(3,3),(3,1)], [(3,3),(3,5)], script)
    traces = replay_equivalent(clean, remote, script)
    assert [t['completed_local_actions'][0] for t in traces] == [0,4,4,1]
    assert all(t['policy_controlled_robots']==[1] for t in traces)
    assert clean.all_agents_at_goals
    assert clean.agents[0].first_completion_time == 0
    assert not any(j['agent']==0 for j in remote.communication.jobs.values())


def test_new_completion_switches_control_on_the_following_epoch(tmp_path):
    script = [{0:3,1:4}, {0:1,1:4}, {0:1,1:3}]
    clean, remote = environments(tmp_path, [(2,1),(4,1)], [(2,2),(4,5)], script)
    traces = replay_equivalent(clean, remote, script)
    assert traces[0]['policy_controlled_robots']==[0,1]
    assert traces[1]['policy_controlled_robots']==[1]
    assert traces[1]['completed_local_actions'][0]==4
    assert clean.agents[0].position==(2,2)


def test_vertex_swaps_and_stationary_cascades_match_remote_rewards(tmp_path):
    script = [{0:3,1:3,2:4}, {0:3,1:2,2:4}, {0:0,1:1,2:0},
              {0:3,1:3,2:3}, {0:1,1:2,2:4}]
    clean, remote = environments(tmp_path, [(3,1),(3,2),(3,3)],
        [(3,5),(3,4),(4,3)], script)
    traces = replay_equivalent(clean, remote, script)
    assert traces[0]['collisions_agent_agent_step']==2
    assert traces[0]['physical_conflicts'][0]['robots']==[1,2]
    assert traces[0]['actual_agent_overlap_after_resolution']==0


def test_v1_remains_replayable_and_its_completed_dqn_departure_is_distinct(tmp_path):
    script=[{0:1,1:4}]
    aligned,_=environments(tmp_path, [(2,2),(4,1)], [(2,2),(4,5)], script)
    legacy=CleanMAPFEnvironment(aligned.config, ExperimentMode.UAV, aligned.map_path,2,20,91,aligned.scenario)
    legacy.reset()
    legacy.step(script[0]); aligned.step(script[0])
    assert legacy.agents[0].position==(3,2)
    assert aligned.agents[0].position==(2,2)
    assert legacy.agents[0].ever_reached_goal and aligned.agents[0].ever_reached_goal


def test_completed_priority_remains_below_unreachable_active_robot(tmp_path):
    rows=['@@@@@@@','@..@..@','@..@..@','@@@@@@@','@.....@','@.....@','@@@@@@@']
    clean,remote=environments(tmp_path,[(1,1),(1,4)],[(1,1),(4,4)],[{0:0,1:4}],rows)
    selection=clean.prepare_actions({0:0,1:4})
    assert np.isfinite(selection.priorities[1])
    assert selection.priorities[0]>selection.priorities[1]
    replay_equivalent(clean,remote,[{0:0,1:4}])


def test_selection_is_pure_metrics_use_execution_and_sources_are_current(tmp_path):
    clean,_=environments(tmp_path,[(3,3),(3,1)],[(3,3),(3,5)],[{0:4,1:4}])
    before=[a.position for a in clean.agents]
    first=clean.prepare_actions({0:4,1:4}); second=clean.prepare_actions({0:1,1:4})
    assert first.executed==second.executed=={0:0,1:4}
    assert [a.position for a in clean.agents]==before
    assert first.source_times[0]=={0:0.,1:0.}
    assert clean.epoch_metrics({0:4,1:4})['waiting_steps']==1
    assert clean.last_selection is None
    with pytest.raises(ValueError,match='unknown'):clean.prepare_actions({9:0})
    with pytest.raises(ValueError,match='zero through four'):clean.prepare_actions({1:5})


def test_perfect_reference_matches_bootstrap_then_full_path_local_control(tmp_path):
    # A goal can lie beyond the active peer's published horizon but inside FoV.
    rows=['@@@@@@@@@@@@']+['@..........@']*3+['@@@@@@@@@@@@']
    script=[{0:4,1:4}]*2
    clean,remote=environments(tmp_path,[(2,8),(2,1)],[(2,8),(2,10)],script,rows)
    assert clean.config.map.communicated_path_steps==5
    assert clean.prepare_actions(script[0]).executed[0]==4
    traces=replay_equivalent(clean,remote,script)
    assert traces[1]['completed_local_actions'][0]!=4


def test_aligned_environment_rejects_impairments(tmp_path):
    clean,_=environments(tmp_path,[(2,2),(4,1)],[(2,2),(4,5)],[{0:4,1:4}])
    impaired=replace(clean.config,communication=replace(clean.config.communication,profile='physical'))
    with pytest.raises(ValueError,match='perfect communication'):
        RemoteAlignedCleanMAPFEnvironment(impaired,ExperimentMode.UAV,clean.map_path,2,20,91,clean.scenario)


def test_extended_compatibility_has_all_populations_and_completed_control_replay(tmp_path):
    import torch
    from panama_uav.models import DQN
    from panama_uav.phase1_execution import verify_execution_compatibility
    torch.set_num_threads(1)
    clean,_=environments(tmp_path,[(2,2),(4,1)],[(2,2),(4,5)],[{0:4,1:4}])
    model=DQN((3,15,15),clean.config.vector_length)
    for parameter in model.parameters():parameter.data.zero_()
    model.head[-1].bias.data[4]=1
    free=[(r,c) for r in range(1,6) for c in range(1,6)]
    bank=[dict(case_id=f'compatibility-{count}',num_agents=count,map_path=str(clean.map_path),
        starts=free[:count],goals=free[-count:],seed=91) for count in (2,4,6,8)]
    result=verify_execution_compatibility(clean.config,model,bank,torch.device('cpu'),tmp_path/'checks')
    assert result['status']=='passed' and result['execution_contract']=='remote-aligned-v2'
    assert [c['num_agents'] for c in result['checks']]==[2,4,6,8]
    assert all(c['epochs']==16 and c['replay_status']=='passed' for c in result['checks'])
    assert result['completed_control_fixture']['completed_goal_departures']==1
    assert result['completed_control_fixture']['local_completed_actions_equal']==4
