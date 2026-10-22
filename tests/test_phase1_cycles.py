from types import SimpleNamespace
import numpy as np
from panama_uav.phase1_cycles import CycleTracker


def env(step,point=(0,0),completed=False):
    robot=SimpleNamespace(id=0,position=point,goal=(0,0),ever_reached_goal=completed,priority=1.)
    return SimpleNamespace(step_count=step,agents=[robot])


def obs():
    return {0:dict(spatial=np.zeros((3,15,15),np.float32),vector=np.zeros(12,np.float32))}


def test_bootstrap_is_not_misclassified_as_a_fresh_state_cycle():
    tracker=CycleTracker()
    tracker.observe(env(0),obs());tracker.observe(env(1),obs())
    assert not tracker.record()['repeated_state']
    tracker.observe(env(2),obs())
    assert tracker.record()['cycle_period']==1


def test_two_step_cycle_and_observation_changes_are_distinct():
    tracker=CycleTracker()
    tracker.observe(env(1,(0,1)),obs());tracker.observe(env(2,(0,2)),obs())
    different=obs();different[0]['vector'][0]=1
    tracker.observe(env(3,(0,1)),different)
    assert not tracker.record()['repeated_state']
    tracker.observe(env(4,(0,2)),obs())
    assert tracker.record()['cycle_period']==2


def test_local_goal_yield_and_active_no_motion_have_separate_denominators():
    tracker=CycleTracker()
    robot=env(1,(0,1),True)
    tracker.moved({0:((0,0),True)},robot,set())
    assert tracker.record()['local_goal_departures']==1
    assert tracker.record()['active_robot_epochs']==0
    tracker.moved({0:((0,1),True)},robot,{0})
    assert tracker.record()['active_no_motion_steps']==1
    assert tracker.record()['terminal_stationary_steps']==1
