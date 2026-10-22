"""V2 regression: leakage, real exposure, correct roles, resume and selection."""
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from panama_uav.config import load_config
from panama_uav.phase1_data import prepare_dataset
from panama_uav.phase1_revised import (COUNTS, METRICS, PROTOCOL, RevisedSession,
    balanced_population, evaluate, load_protocol, paired_differences, prepare_data, subset,
    validate_data, variant_config, verify_payload, summarize, reconcile_logs, resume_elapsed)
from panama_uav.training import TrainingSession, load_model
from panama_uav.records import sha256

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def spec():
    return load_protocol(ROOT/'configs/phase1_revised.toml')


@pytest.fixture
def dataset(tmp_path,spec):
    historical=prepare_dataset(tmp_path/'historical',seed=19,train_maps=1,eval_maps=1,eval_cases=2)
    config=dict(spec,train_maps_per_family=1,evaluation_maps_per_family=1,cases_per_family=2)
    data=prepare_data(tmp_path/'new',config,tmp_path/'historical/dataset.json')
    return data


def cfg(spec):
    base=load_config(ROOT/spec['base_config'])
    base=replace(base,training=replace(base.training,replay_capacity=128,batch_size=4))
    return variant_config(base,spec['variants'][0],360)


def test_new_split_excludes_all_old_and_stress_maps(dataset):
    validate_data(dataset)
    hashes={m['canonical'] for m in dataset['maps']}
    assert not hashes & set(dataset['excluded_historical_maps'])
    assert len(hashes)==15
    assert all(c['split']=='test' for c in dataset['banks']['test'])
    broken=deepcopy(dataset)
    broken['excluded_historical_maps'].append(broken['maps'][0]['canonical'])
    with pytest.raises(ValueError,match='Historical'):validate_data(broken)


def test_population_schedule_cannot_be_blocked_by_failed_qualification():
    assert [balanced_population(i) for i in range(12)]==list(COUNTS)*3


def test_resume_removes_log_rows_created_after_initial_checkpoint(tmp_path):
    (tmp_path/'training.jsonl').write_text('uncommitted episode\n')
    (tmp_path/'learning_steps.jsonl').write_text('uncommitted steps\n')
    (tmp_path/'monitor_history.jsonl').write_text('external monitoring\n')
    reconcile_logs(tmp_path,{})
    assert (tmp_path/'training.jsonl').read_bytes()==b''
    assert (tmp_path/'learning_steps.jsonl').read_bytes()==b''
    assert (tmp_path/'monitor_history.jsonl').read_text()=='external monitoring\n'
    (tmp_path/'training.jsonl').write_bytes(b'committed\nuncommitted\n')
    reconcile_logs(tmp_path,{'training.jsonl':10})
    assert (tmp_path/'training.jsonl').read_bytes()==b'committed\n'
    with pytest.raises(ValueError,match='shorter'):
        reconcile_logs(tmp_path,{'training.jsonl':11})
    with pytest.raises(ValueError,match='path'):
        reconcile_logs(tmp_path,{'../outside.jsonl':1})


def test_crash_resume_never_refunds_work_or_grants_extra_budget(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'wall_seconds':40}))
    assert resume_elapsed(tmp_path,10,200)==40
    lease=dict(active=True,previous_elapsed=10,started_unix=100,charged_seconds=10)
    (tmp_path/'budget_lease.json').write_text(json.dumps(lease))
    assert resume_elapsed(tmp_path,10,200)==110
    lease.update(active=False,charged_seconds=55)
    (tmp_path/'budget_lease.json').write_text(json.dumps(lease))
    assert resume_elapsed(tmp_path,10,200)==55


def test_validation_subsets_include_designed_and_random_endpoints(dataset):
    bank=subset(dataset['banks']['validation'],2)
    assert {r['designed_coordination'] for r in bank}=={True,False}
    assert len(bank)==40


def test_real_training_exposes_all_counts_without_any_gate(dataset,spec):
    torch.set_num_threads(1)
    session=RevisedSession(cfg(spec),dataset,42,'v2-test',torch.device('cpu'))
    seen=[]
    for n in COUNTS:
        assert session.environment.num_agents==n
        seen.append(session.current_case)
        # Complete a bounded episode; no training performance gate is consulted.
        session.environment.max_episode_steps=1
        _,episode=session.step()
        assert episode is not None
        session.start_episode()
    assert session.population_episodes==dict.fromkeys(COUNTS,1)
    assert all(c['split']=='train' for c in seen)
    assert session.qualification==[]


def test_completed_local_controller_actions_do_not_enter_replay(dataset,spec):
    torch.set_num_threads(1)
    session=RevisedSession(cfg(spec),dataset,42,'v2-role',torch.device('cpu'))
    robot=session.environment.agents[0]
    robot.ever_reached_goal=True
    robot.position=robot.goal
    before=session.delivered_experiences
    row,_=session.step()
    assert row['active_replay_transitions']==1
    assert session.delivered_experiences==before+1


def test_revised_resume_preserves_policy_optimizer_replay_rng_and_roles(dataset,spec):
    torch.set_num_threads(1)
    config=cfg(spec)
    original=RevisedSession(config,dataset,73,'v2-resume',torch.device('cpu'))
    for _ in range(8):original.step()
    buffer=io.BytesIO();torch.save(original.checkpoint(),buffer)
    expected=[original.step()[0] for _ in range(5)]
    weights={k:v.clone() for k,v in original.policy.state_dict().items()}
    buffer.seek(0)
    restored=TrainingSession.restore(torch.load(buffer,weights_only=False),config,'clean',73,'v2-resume',torch.device('cpu'))
    assert [restored.step()[0] for _ in range(5)]==expected
    assert all(torch.equal(v,restored.policy.state_dict()[k]) for k,v in weights.items())
    assert restored.population_steps==original.population_steps


def test_ablation_alternatives_change_learning_only(spec):
    base=load_config(ROOT/spec['base_config'])
    alternatives=[variant_config(base,v,360) for v in spec['variants']]
    assert [c.training.gamma for c in alternatives]==[.9,.9,.99,.99]
    assert [c.training.learning_rate for c in alternatives]==[.001,.0003,.0003,.0003]
    for candidate in alternatives:
        assert candidate.reward==base.reward and candidate.map==base.map
        assert candidate.curriculum==base.curriculum
        assert not candidate.resource_allocation.enabled
        assert candidate.resource_allocation.drone_cycles_per_second==base.resource_allocation.drone_cycles_per_second
        assert candidate.communication.profile=='perfect' and candidate.communication.stale_information==()


def pair_rows():
    return [dict(training_seed=s,case_id=f'c{i}',map_id=f'm{i%2}',map_canonical=f'h{i%2}',
        starts=[[0,0]],goals=[[0,1]],seed=i,num_agents=2,success=False)
        for s in (42,73) for i in range(5)]


def test_paired_comparison_preserves_case_identity_and_map_denominators():
    old=pair_rows();new=deepcopy(old)
    # Unequal map sizes: first map has six of ten cases, not five.
    for row in new:row['success']=row['map_id']=='m0'
    result=paired_differences(new,old,samples=100)
    assert result['mean']==.6 and result['seed_blocks']==2 and result['map_blocks']==2
    changed=deepcopy(new);changed[0]['goals']=[[1,0]]
    with pytest.raises(ValueError,match='identity'):paired_differences(changed,old)
    with pytest.raises(ValueError,match='unique'):paired_differences(new[:-1],old)


def test_deadline_expired_evaluation_never_counts_unfinished_case(dataset,spec,tmp_path):
    session=RevisedSession(cfg(spec),dataset,42,'v2-eval',torch.device('cpu'))
    result=evaluate(session.config,session.policy,dataset['banks']['validation'],session.device,
        tmp_path/'empty.jsonl','expired',time.monotonic()-1)
    assert not result and session.generated_steps==0
    assert not (tmp_path/'empty.jsonl').exists()


def test_old_or_ablation_payloads_cannot_qualify_v2():
    assert not verify_payload(dict(training_protocol='clean_mapf_phase1_v1'))
    assert not verify_payload(dict(training_protocol=PROTOCOL,metric_version=METRICS,
        minimum_episodes_per_population=500,ablation=True))


def test_interrupted_ablation_stops_batch_before_starting_next_variant(dataset,spec,tmp_path,monkeypatch):
    import panama_uav.phase1_revised as module
    path=Path(dataset['maps'][0]['path']).parent/'dataset.json'
    setup=dict(spec=spec,base_configuration=cfg(spec).to_dict(),
        code_sha256={p.name:sha256(p) for p in Path(module.__file__).parent.glob('*.py')},
        dataset_path=str(path),dataset_sha256=sha256(path),source_commit='test-frozen-source')
    (tmp_path/'study.json').write_text(json.dumps(setup))
    called=[]
    def interrupted(folder,manifest,device,resume):
        called.append(manifest['variant'])
        return dict(status='interrupted_partial')
    monkeypatch.setattr(module,'run_one',interrupted)
    result=module.worker(tmp_path,'ablations','cpu')
    assert called==['original_hp'] and result['status']=='interrupted_partial'
    state=json.loads((tmp_path/'status.json').read_text())
    assert state['status']=='interrupted_partial'
    assert len(list((tmp_path/'ablations').iterdir()))==1


def test_full_v2_gate_requires_current_both_splits_every_family_and_coverage():
    families=('random_obstacles','intersection','corridor','bottleneck','warehouse')
    cases=[dict(family=f,map_id=f,success=i<38,steps=10,waiting_steps=0,
        vertex_conflicts=0,edge_conflicts=0,relevant_pairs=2,directed_pairs=4)
        for f in families for i in range(40)]
    summaries=[dict(summarize(cases),num_agents=n) for n in COUNTS]
    valid=dict(training_protocol=PROTOCOL,metric_version=METRICS,
        phase1_qualified=True,curriculum_completed=True,smoke=False,ablation=False,
        minimum_episodes_per_population=500,population_episodes=dict.fromkeys(COUNTS,500),
        qualification=deepcopy(summaries),held_out=deepcopy(summaries),
        compatibility=dict(status='passed',execution_contract='remote-aligned-v2',
            checks=[dict(num_agents=n) for n in COUNTS]))
    assert verify_payload(valid)
    altered=deepcopy(valid);altered['population_episodes'][8]=499
    assert not verify_payload(altered)
    altered=deepcopy(valid);altered['held_out'][3]['per_family']['corridor']['success_rate']=.925
    assert not verify_payload(altered)
    altered=deepcopy(valid);altered['held_out'][0]['complete']=False
    assert not verify_payload(altered)
    altered=deepcopy(valid);altered['compatibility']['execution_contract']='v1'
    assert not verify_payload(altered)


def test_untouched_final_test_is_not_available_to_ablation_evaluate(dataset,spec,tmp_path,monkeypatch):
    import panama_uav.phase1_revised as module
    config=cfg(spec)
    config=replace(config,training=replace(config.training,budget_seconds=30.,target_environment_steps=2))
    reduced=deepcopy(spec)
    reduced['ablation'].update(joint_steps=2,evaluation_reserve_seconds=1,validation_cases_per_family=1)
    # Each fake cohort row carries valid metrics; intercepting here audits which
    # split reaches evaluation, while the two actual learning steps still run.
    requested=[]
    def inspect(config,model,bank,device,path,tag,deadline,stop):
        assert all(c['split']=='validation' for c in bank)
        requested.extend(c['case_id'] for c in bank)
        return [dict(c,success=False,steps=250,waiting_steps=0,vertex_conflicts=0,edge_conflicts=0,
            relevant_pairs=0,directed_pairs=c['num_agents']*(c['num_agents']-1)*250) for c in bank]
    monkeypatch.setattr(module,'evaluate',inspect)
    path=Path(dataset['maps'][0]['path']).parent/'dataset.json'
    manifest=dict(spec=reduced,mode='ablation',seed=42,variant='original_hp',
        configuration=config.to_dict(),dataset_path=str(path),dataset_sha256=sha256(path))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    result=module.run_one(tmp_path,manifest,'cpu')
    assert result['status']=='ablation_complete' and result['generated_joint_steps']==2
    assert not result['held_out'] and not result['qualified'] and requested
    model,payload=load_model(tmp_path/'final_model.pt',config,torch.device('cpu'))
    assert payload['training_protocol']==PROTOCOL and payload['ablation']
    with pytest.raises(ValueError,match='qualified'):load_model(tmp_path/'final_model.pt',config,torch.device('cpu'),require_qualified=True)
