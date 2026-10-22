from copy import deepcopy
import pytest
from panama_uav.phase1_stopping import covered,may_open_heldout,training_stop
from panama_uav.phase1_data import FAMILIES


def history(scores):
    return [dict(generated_joint_steps=(i+1)*30000,populations=[dict(num_agents=n,
        complete=True,success_rate=s,per_family={f:dict(success_rate=s) for f in FAMILIES})
        for n in (2,4,6,8)]) for i,s in enumerate(scores)]


def test_plateau_cannot_bypass_population_coverage_floor():
    values=history([.6]*8)
    assert training_stop(values,{2:500,4:500,6:500,8:499}) is None
    assert training_stop(values,dict.fromkeys((2,4,6,8),500))=='validation_plateau'


def test_improving_validation_is_not_stagnation():
    assert training_stop(history([.4,.45,.5,.55,.6,.65,.7,.75]),dict.fromkeys((2,4,6,8),500)) is None
    assert training_stop(history([.9,.4,.45,.5,.6,.7,.8]),dict.fromkeys((2,4,6,8),500)) is None


def test_ready_small_validation_freezes_but_does_not_qualify():
    assert training_stop(history([1.]),dict.fromkeys((2,4,6,8),500))=='validation_ready'
    assert not may_open_heldout([],dict.fromkeys((2,4,6,8),500),dict(status='passed',execution_contract='remote-aligned-v2'))


def test_partial_validation_does_not_trigger_early_stop():
    values=history([.6]*8)
    for item in values:item['populations'][3]['complete']=False
    assert training_stop(values,dict.fromkeys((2,4,6,8),500)) is None


def test_test_bank_opens_only_after_full_strict_validation_and_compatibility():
    val=[dict(num_agents=n,episodes=200,complete=True,passed=True,success_rate=.95,
        exposure_passed=True,relevant_pairs=200,per_family={f:dict(episodes=40,success_rate=.95,exposed_cases=20) for f in FAMILIES})
        for n in (2,4,6,8)]
    counts=dict.fromkeys((2,4,6,8),500)
    compat=dict(status='passed',execution_contract='remote-aligned-v2')
    assert may_open_heldout(val,counts,compat)
    bad=deepcopy(val);bad[-1]['passed']=False
    assert not may_open_heldout(bad,counts,compat)
    assert not may_open_heldout(val,counts,None)
    assert not covered(counts,499)
