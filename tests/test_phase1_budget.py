from copy import deepcopy
import math
import pytest

from panama_uav.phase1_budget import measure_training,project_budget
from panama_uav.phase1_revised import COUNTS


def evidence(seconds_per_step=.02,censored=0):
    learning=[];episodes=[];wall=0.;joint=0
    for n in COUNTS:
        for _ in range(250):
            joint+=1;wall+=seconds_per_step
            learning.append(dict(generated_joint_steps=joint,wall_seconds=wall,num_agents=n,
                case_id=f'train-{n}',communication_delay_seconds=0.))
        episodes.append(dict(num_agents=n,case_id=f'train-{n}',split='train',steps=250,
            success=False,family='intersection'))
    for _ in range(censored):
        joint+=1;wall+=seconds_per_step
        learning.append(dict(generated_joint_steps=joint,wall_seconds=wall,num_agents=2,
            case_id='unfinished',communication_delay_seconds=0.))
    return learning,episodes


def test_throughput_is_conditional_not_four_simultaneous_rates():
    rows,episodes=evidence()
    measured=measure_training(rows,episodes)
    assert measured['total_joint_steps']==1000 and measured['completed_episodes']==4
    assert sum(r['seconds'] for r in measured['populations'].values())==pytest.approx(20)
    for n in COUNTS:
        row=measured['populations'][n]
        assert row['joint_steps_per_hour']==pytest.approx(180000)
        assert row['completed_episodes_per_hour']==pytest.approx(720)
        assert row['failed_250_step_episodes']==1


def test_censored_task_is_charged_without_inventing_completed_episode():
    rows,episodes=evidence(censored=20)
    result=measure_training(rows,episodes)
    assert result['total_joint_steps']==1020 and result['completed_episodes']==4
    assert result['populations'][2]['steps']==270
    assert result['populations'][2]['completed_episodes']==1


def test_checkpoint_runtime_gaps_are_included_in_measured_rate():
    rows,episodes=evidence()
    for row in rows[500:]:row['wall_seconds']+=10
    result=measure_training(rows,episodes)
    assert result['training_window_seconds']==pytest.approx(30)
    assert result['populations'][6]['seconds']==pytest.approx(15)


def test_budget_includes_250_failures_validation_final_evaluation_and_margin():
    measured=measure_training(*evidence())
    plan=project_budget([measured],dict.fromkeys(COUNTS,.002))
    assert plan['target_completed_episodes']==2000
    assert plan['all_250_step_target_joint_steps']==500000
    assert plan['periodic_validation_checks']==17
    assert plan['periodic_validation_seconds_each']==pytest.approx(80)
    assert plan['all_250_step_training_seconds']==pytest.approx(10000)
    assert plan['final_evaluation_reserve_seconds']>=800
    assert plan['proposed_budget_seconds']%1800==0
    assert plan['proposed_budget_seconds']>10800
    assert not plan['three_hours_sufficient_for_conservative_coverage']


def test_fast_evidence_does_not_extend_budget_without_need():
    measured=measure_training(*evidence(.001))
    plan=project_budget([measured],dict.fromkeys(COUNTS,.0001))
    assert plan['three_hours_sufficient_for_conservative_coverage']
    assert plan['proposed_budget_seconds']==10800


def test_slower_eight_robot_measurements_increase_planned_runtime():
    measured=measure_training(*evidence())
    slower=deepcopy(measured)
    slower['populations'][8]['conservative_seconds_per_joint_step']*=2
    base=project_budget([measured],dict.fromkeys(COUNTS,.002))
    new=project_budget([measured,slower],dict.fromkeys(COUNTS,.002))
    assert new['all_250_step_training_seconds']>base['all_250_step_training_seconds']
    assert new['proposed_budget_seconds']>=base['proposed_budget_seconds']


@pytest.mark.parametrize('kind',['steps','time','delay','split'])
def test_invalid_training_evidence_is_rejected(kind):
    rows,episodes=evidence()
    if kind=='steps':rows[10]['generated_joint_steps']=1
    if kind=='time':rows[10]['wall_seconds']=-1
    if kind=='delay':rows[10]['communication_delay_seconds']=1.
    if kind=='split':episodes[0]['split']='test'
    with pytest.raises(ValueError):measure_training(rows,episodes)


def test_budget_cannot_lower_population_coverage_or_hide_unmeasured_costs():
    measured=measure_training(*evidence())
    with pytest.raises(ValueError):project_budget([measured],dict.fromkeys(COUNTS,.002),minimum_episodes=499)
    with pytest.raises(ValueError):project_budget([measured],{2:.002})
