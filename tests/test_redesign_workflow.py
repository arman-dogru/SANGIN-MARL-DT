"""Registered cohort dimensions, timing isolation and historical case pairing."""
import importlib.util
from pathlib import Path
import sys
import json

from panama_uav.config import ProjectConfig
from panama_uav.redesign import TREATMENTS,configured,settings_for
from panama_uav.resource_study import study_cases

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_redesign as study


def test_primary_matrix_reuses_identical_A4_W1_jobs():
    cases=[dict(case_id=f'agents-{n}-case-{i:03d}',case=i,num_agents=n) for n in (2,4,8,16,32) for i in range(10)]
    a=study.jobs_for(cases,'A');b=study.jobs_for(cases,'B')
    assert len(a)==720 and len(b)==750
    assert len(study.jobs_for(cases,'A',True))==360
    assert len(set(map(study.job_id,a))&set(map(study.job_id,b)))==90
    assert len(set(map(study.job_id,a+b)))==1380


def test_registered_timing_and_only_allowed_parameter_changes():
    from panama_uav.config import load_config
    from run_omega_sensitivity import config_changes
    base=load_config(ROOT/'configs/paper_remote_dt_250ms_uav6.toml')
    for treatment in TREATMENTS:
        c=configured(base,settings_for(treatment))
        changed=config_changes(base.to_dict(),c.to_dict())
        assert set(changed)<={'communication.seconds_per_step','resource_allocation.decision_budget_seconds',
            'resource_allocation.command_cutoff_seconds','resource_allocation.task_deadline_seconds',
            'resource_allocation.missed_command_fallback'}
        assert not c.resource_allocation.enforce_task_deadlines
        assert c.resource_allocation.drone_cycles_per_second==6e9


def test_historical_cases_are_not_reselected_by_outcomes():
    source=ROOT/'output/omega_sensitivity/omega-20261025-01'
    if not source.exists():
        import pytest
        pytest.skip('Local historical assets unavailable')
    original=json.loads((source/'manifest.json').read_text())
    old=json.loads((source/'cases.json').read_text())
    config=ProjectConfig.from_dict(original['configuration'])
    generated={c['case_id']:c for c in study_cases(config,10,(2,4,8,16,32),250)}
    for case in old:
        for field in ('seed','starts','goals','orbital_offset_seconds'):
            assert case[field]==generated[case['case_id']][field]


def test_partial_episode_is_not_silently_counted_as_committed(tmp_path):
    from panama_uav.records import sha256
    import pytest
    path=tmp_path/'result.json';path.write_text('{"audit":{"status":"passed"}}')
    assert not study.committed_output(path)
    marker=path.with_name('complete.json');marker.write_text(json.dumps(dict(sha256=sha256(path))))
    assert study.committed_output(path)
    path.write_text('{}')
    with pytest.raises(ValueError,match='transaction'):study.committed_output(path)


def test_worker_identity_ignores_both_inference_host_timers_only():
    a=dict(row=dict(success=True,inference_wall_seconds=1.,allocation_solver_wall_seconds=2.),
        tasks=[dict(job_id='0:0',executed_x=1,bundle_inference_wall_seconds=3.)])
    b=dict(row=dict(a['row'],inference_wall_seconds=8.),tasks=[dict(a['tasks'][0],bundle_inference_wall_seconds=99.)])
    assert study.stable_signature(a)==study.stable_signature(b)
    b['tasks'][0]['executed_x']=0
    assert study.stable_signature(a)!=study.stable_signature(b)


def test_legacy_observation_aggregates_and_resource_denominators(tmp_path):
    import torch
    from panama_uav.evaluation import run_episode
    from panama_uav.models import DQN
    from panama_uav.staleness_remote import controlled_class
    from panama_uav.staleness_audit import audit_stress
    from panama_uav.redesign_metrics import extend_legacy
    from panama_uav.communication_effects import measurements
    from panama_uav.inference_controls import InferenceCondition
    from test_frequency_motion import fixture
    _,_,scenario,config=fixture(tmp_path,settings=settings_for('A0',3.))
    torch.set_num_threads(1);torch.manual_seed(6);model=DQN((3,15,15),config.vector_length).eval()
    case=dict(case_id='legacy',num_agents=3,seed=17,starts=list(map(list,scenario.starts)),goals=list(map(list,scenario.goals)))
    row,trace=run_episode(config,model,case,torch.device('cpu'),scenario=scenario,max_steps=2,remote_communication_class=controlled_class(0))
    metrics=audit_stress(trace,include_observations=True)['observation_aggregates']
    assert metrics['active_peer_aoi_samples']==12
    row,_,_=measurements(row,trace,InferenceCondition('current','allocated'))
    row.update(drone_utilization_seconds=.5,cloud_utilization_seconds=.2)
    extended=extend_legacy(row,trace,metrics)
    assert extended['drone_cpu_utilization']==.5/row['observed_seconds']
    assert extended['relevant_pairs']==metrics['relevant_pairs']
