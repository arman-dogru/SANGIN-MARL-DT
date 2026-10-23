from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from panama_uav.communication_effects import (condition_config, measurements, prepare,
    run, validate_configuration, verify_inputs)
from panama_uav.communication_report import paired_changes, summary_row
from panama_uav.deadline_study import deadline_config
from panama_uav.evaluation import case_bank, run_episode
from panama_uav.inference_controls import (InferenceCondition, condition_grid, inference_class,
    interrupted_finish, outage_windows, union_windows)
from panama_uav.models import DQN, select_action
from panama_uav.staleness_audit import audit_stress
from panama_uav.staleness_remote import controlled_class
from test_deadline_study import inputs


@pytest.fixture
def config(inputs):
    original,_=inputs
    config=deadline_config(original,250)
    return replace(config,resource_allocation=replace(config.resource_allocation,remote_acceptance_approved=True))


def test_grid_is_unique_one_factor_at_a_time_and_keeps_nominal_references():
    conditions=condition_grid()
    assert len(conditions)==41 and len({c.name for c in conditions})==41
    assert {c.route for c in conditions}=={'ideal','allocated','edge','cloud'}
    fields=('uplink_seconds','downlink_seconds','backhaul_seconds','peer_delay_steps','compute_seconds','interruption_seconds')
    for c in conditions:
        assert sum(bool(getattr(c,k)) for k in fields)==(c.axis not in ('nominal','network_reference'))
    with pytest.raises(ValueError):InferenceCondition('bad','cloud',peer_delay_steps=True)
    with pytest.raises(ValueError):InferenceCondition('bad','edge',backhaul_seconds=.1)
    with pytest.raises(ValueError):InferenceCondition('bad','ideal',compute_seconds=.1)
    with pytest.raises(ValueError):InferenceCondition('bad','edge',interruption_seconds=10)


def test_interventions_preserve_policy_physics_radio_workload_and_cpu(config,tmp_path):
    p=config.resource_allocation
    for condition in condition_grid():
        cfg=condition_config(config,condition,tmp_path)
        assert cfg.map==config.map and cfg.reward==config.reward and cfg.uav==config.uav
        for field in ('task_input_bits','cycles_per_bit','drone_cycles_per_second','cloud_cycles_per_second',
                      'robot_power_w','drone_power_w','drone_kappa','oracle_shadow','on_collision'):
            assert getattr(cfg.resource_allocation,field)==getattr(p,field)
        assert cfg.communication.seconds_per_step==1
        assert cfg.resource_allocation.command_cutoff_seconds==.25
    with pytest.raises(ValueError,match='gate'):
        validate_configuration(replace(config,resource_allocation=replace(p,remote_acceptance_approved=False)))


def test_service_outages_preempt_resume_and_respect_boundaries():
    assert interrupted_finish(0,.1,[(.1,.3)])==pytest.approx(.1)
    assert interrupted_finish(.05,.1,[(.1,.3)])==pytest.approx(.35)
    assert interrupted_finish(.1,.1,[(.1,.3)])==pytest.approx(.4)
    assert interrupted_finish(.3,.1,[(.1,.3)])==pytest.approx(.4)
    assert union_windows([(0,.2),(.1,.3),(.3,.4),(.8,.9)])==[[0.,.4],[.8,.9]]
    condition=InferenceCondition('outage','edge',interruption_seconds=1.)
    assert outage_windows(condition,26)==[(5.,6.),(15.,16.),(25.,26.)]


@pytest.mark.parametrize('condition',[
    InferenceCondition('ideal','ideal'),InferenceCondition('edge','edge'),InferenceCondition('cloud','cloud'),
    InferenceCondition('allocated','allocated'),
    InferenceCondition('allocated_ideal','allocated',ideal_communication=True),
    InferenceCondition('edge_ideal','edge',ideal_communication=True),
    InferenceCondition('cloud_ideal','cloud',ideal_communication=True),
    InferenceCondition('up','cloud',uplink_seconds=.05),InferenceCondition('down','edge',downlink_seconds=.05),
    InferenceCondition('age','cloud',peer_delay_steps=1),InferenceCondition('compute','edge',compute_seconds=.05),
    InferenceCondition('outage','cloud',interruption_seconds=.1,interruption_start_seconds=0.)])
def test_fixed_location_causality_deadline_partition_and_replay(config,tmp_path,condition):
    torch.set_num_threads(1);torch.manual_seed(42)
    model=DQN((3,15,15),config.vector_length).eval()
    case=case_bank(config,1,(2,))[0]
    cfg=condition_config(config,condition,tmp_path)
    row,trace=run_episode(cfg,model,case,torch.device('cpu'),max_steps=2,remote_communication_class=inference_class(condition))
    row,tasks,rounds=measurements(row,trace,condition)
    if condition.route!='allocated':
        assert all(t['executed_x']==(0 if condition.route=='cloud' else 1) for t in tasks if 'executed_x' in t)
    assert row['deadline_commands_on_time']+row['deadline_commands_late']+row['deadline_commands_undelivered']==len(tasks)
    assert len({(t['cycle'],t['agent']) for t in tasks})==len(tasks)
    assert all(r['on_time_commands']<=r['tasks'] for r in rounds)
    inference=lambda location,robot,obs:select_action(model,obs,0,np.random.default_rng(0),torch.device('cpu'))
    assert audit_stress(trace,inference)['status']=='passed'
    if condition.compute_seconds:
        completed={e['job_id']:e['time'] for e in trace['transport_events'] if e['event']=='compute_complete'}
        for task in tasks:
            if 'compute_completed_at' in task:
                assert task['compute_completed_at']-completed[task['job_id']]==pytest.approx(condition.compute_seconds)
    if condition.route=='ideal':
        assert all(t['command_delivery_seconds']==0 for t in tasks)
    if condition.ideal_communication:
        assert all(t['ingress_seconds']==0 for t in tasks)
        assert all(t['processing_seconds']>0 for t in tasks if t['processing_seconds'] is not None)


def test_paired_changes_use_case_identity_and_preserve_missing_values():
    actual=[dict(case_id='a',makespan_seconds=10.,success=True),dict(case_id='b',makespan_seconds=None,success=False)]
    base=[dict(case_id='b',makespan_seconds=9.,success=True),dict(case_id='a',makespan_seconds=8.,success=True)]
    result=paired_changes(actual,base,'makespan_seconds',100)
    assert result['paired_cases']==1 and result['difference']==2
    assert paired_changes(actual,base,'mission_success_rate',100)['difference']==-.5
    assert paired_changes([actual[1]],[base[0]],'makespan_seconds',100)['difference'] is None
    with pytest.raises(ValueError):paired_changes(actual,base[:1],'mission_success_rate')


def test_nominal_allocated_execution_matches_production_without_interventions(config,tmp_path):
    torch.set_num_threads(1);torch.manual_seed(42)
    model=DQN((3,15,15),config.vector_length).eval()
    case=case_bank(config,1,(2,))[0]
    condition=InferenceCondition('allocated__nominal','allocated')
    cfg=condition_config(config,condition,tmp_path)
    _,original=run_episode(cfg,model,case,torch.device('cpu'),max_steps=3,
        remote_communication_class=controlled_class(0))
    _,actual=run_episode(cfg,model,case,torch.device('cpu'),max_steps=3,
        remote_communication_class=inference_class(condition))
    assert actual['frames']==original['frames']
    assert actual['remote_events']==original['remote_events']
    def without_wall_timing(value):
        if isinstance(value,dict):
            return {k:without_wall_timing(v) for k,v in value.items() if k!='solve_seconds'}
        if isinstance(value,(list,tuple)):return [without_wall_timing(v) for v in value]
        return value
    assert without_wall_timing(actual['oracle_events'])==without_wall_timing(original['oracle_events'])


def test_workflow_freezes_inputs_exports_tables_and_rejects_tampering(config,inputs,tmp_path,monkeypatch):
    import panama_uav.communication_effects as workflow
    import panama_uav.communication_report as reporting
    _,checkpoint=inputs
    monkeypatch.setattr(workflow,'condition_grid',lambda:[InferenceCondition('ideal','ideal'),
        InferenceCondition('edge__nominal','edge'),InferenceCondition('cloud__nominal','cloud')])
    monkeypatch.setattr(reporting,'_plots',lambda *args:None)
    root=tmp_path/'study'
    manifest=prepare(root,config,checkpoint,cases=1,counts=(2,),cap=2,workers=1,diagnostic=True)
    assert manifest['expected_episodes']==3
    frozen=verify_inputs(root)
    assert frozen['model_sha256']==manifest['model_sha256']
    run(root)
    assert json.loads((root/'status.json').read_text())['status']=='complete'
    assert len(json.loads((root/'summary.json').read_text()))==3
    assert (root/'tasks.csv').exists() and (root/'rounds.csv').exists()
    assert json.loads((root/'validation.json').read_text())['model_updates']==0
    # Aggregate-write recovery uses immutable per-case transactions.
    with (root/'episodes.jsonl').open('a') as stream:stream.write('{partial')
    run(root)
    assert len((root/'episodes.jsonl').read_text().splitlines())==3
    with pytest.raises(ValueError,match='immutable'):
        prepare(root,config,checkpoint,diagnostic=True)
    (root/'assets/orbit.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='input changed'):verify_inputs(root)
