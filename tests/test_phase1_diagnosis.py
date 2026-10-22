import numpy as np
from itertools import product

from panama_uav.mapf_reference import solve_mapf, validate_paths, first_conflict
from panama_uav.phase1_diagnosis import planner_sample, stratified_sample, summarize_trace


def test_cbs_constructs_verified_swap_detour_without_panama_policy():
    grid = np.zeros((2,3),dtype=int)
    starts = [(0,0),(0,2)]; goals = starts[::-1]
    result = solve_mapf(grid,starts,goals,timeout_seconds=2.)
    assert result['status']=='feasible'
    assert validate_paths(grid,starts,goals,result['paths'])
    assert result['makespan']>=4


def test_reference_rejects_vertex_swaps_and_goal_occupancy():
    assert first_conflict([[(0,0),(0,1)],[(0,1),(0,0)]])[0]=='edge'
    assert first_conflict([[(0,0)],[(0,2),(0,1),(0,0)]])[0]=='vertex'
    assert not validate_paths(np.zeros((1,3)),[(0,0)],[(0,2)],[[(0,0),(0,2)]])


def test_reference_distinguishes_limits_and_static_from_horizon_infeasibility():
    grid = np.zeros((1,3),dtype=int)
    assert solve_mapf(grid,[(0,0)],[(0,2)],timeout_seconds=0)['status']=='timeout'
    assert solve_mapf(grid,[(0,0)],[(0,2)],node_limit=0)['status']=='node_limit'
    assert solve_mapf(grid,[(0,0)],[(0,2)],horizon=1)['status']=='infeasible_within_horizon'
    grid[0,1]=1
    assert solve_mapf(grid,[(0,0)],[(0,2)])['status']=='infeasible_static'


def test_reference_proves_finite_horizon_impossibility_for_single_lane_swap():
    result = solve_mapf(np.zeros((1,2),dtype=int),[(0,0),(0,1)],[(0,1),(0,0)],horizon=3)
    assert result['status']=='infeasible_within_horizon'


def test_cbs_agrees_with_independent_joint_breadth_first_feasibility():
    def exhaustive(grid, starts, goals, horizon):
        states = {tuple(starts)}
        for _ in range(horizon+1):
            if tuple(goals) in states:
                return True
            upcoming = set()
            for a,b in states:
                choices = []
                for point in (a,b):
                    choices.append([(point[0]+dr,point[1]+dc) for dr,dc in
                        ((0,0),(1,0),(-1,0),(0,1),(0,-1))
                        if 0 <= point[0]+dr < 3 and 0 <= point[1]+dc < 3
                        and not grid[point[0]+dr,point[1]+dc]])
                for c,d in product(*choices):
                    if c != d and not (c == b and d == a):
                        upcoming.add((c,d))
            states = upcoming
        return False
    rng = np.random.default_rng(14)
    for _ in range(15):
        grid = np.zeros((3,3),dtype=int); grid[1,1] = int(rng.integers(2))
        cells = list(map(tuple,np.argwhere(grid==0)))
        selected = rng.choice(len(cells),4,replace=False)
        starts = [cells[i] for i in selected[:2]]; goals = [cells[i] for i in selected[2:]]
        feasible = exhaustive(grid,starts,goals,4)
        result = solve_mapf(grid,starts,goals,horizon=4)
        assert result['status'] in ('feasible','infeasible_within_horizon')
        assert (result['status']=='feasible') == feasible


def test_stratified_replay_sample_uses_deterministic_case_order_and_caps():
    rows = [dict(num_agents=2,family='corridor',success=success,case_id=f'{success}-{i}')
        for success in (False,True) for i in (2,0,1)]
    sample = stratified_sample(rows,2,1)
    assert [r['case_id'] for r in sample]==['False-0','False-1','True-0']


def test_reference_sample_balances_populations_and_then_families():
    rows = [dict(num_agents=n,family=f,success=False,case_id=f'{n}-{f}-{i}')
        for n in (2,4,6,8) for f in ('a','b','c','d','e') for i in range(3)]
    sample = planner_sample(rows,24)
    assert all(sum(r['num_agents']==n for r in sample)==6 for n in (2,4,6,8))
    assert all(len({r['family'] for r in sample if r['num_agents']==n})==5 for n in (2,4,6,8))


def test_trace_separates_explicit_wait_collision_stops_and_goal_departures():
    trace = [dict(before=[[0,0],[0,1]],after=[[0,1],[0,2]],intended=[[0,1],[0,2]],
        actions=[3,3],completed_before=[True,False],completed_after=[True,False],
        vertex_conflicts=0,edge_conflicts=0),
        dict(before=[[0,1],[0,2]],after=[[0,1],[0,2]],intended=[[0,1],[0,1]],
        actions=[4,2],completed_before=[True,False],completed_after=[True,False],
        vertex_conflicts=1,edge_conflicts=0)]
    result = summarize_trace(trace,[(0,0),(0,0)])
    assert result['goal_departures']==1 and result['completed_off_goal_robot_epochs']==1
    assert result['completed_explicit_waits']==1 and result['active_explicit_waits']==0
    assert result['collision_blocked_moves']==1 and result['obstacle_blocked_moves']==0
    assert result['active_robot_epochs']==2 and result['terminal_stationary_suffix']==1


def test_trace_detects_two_step_cycle_and_terminal_deadlock_separately():
    trace = []
    for before,after,action in [([0,0],[0,1],3),([0,1],[0,0],2),([0,0],[0,1],3)]:
        trace.append(dict(before=[before],after=[after],intended=[after],actions=[action],
            completed_before=[False],completed_after=[False],vertex_conflicts=0,edge_conflicts=0))
    result = summarize_trace(trace,[(0,2)])
    assert result['deterministic_cycle']==dict(start_epoch=0,repeat_epoch=2,period=2)
    assert result['joint_two_step_returns']==2 and result['terminal_stationary_suffix']==0
