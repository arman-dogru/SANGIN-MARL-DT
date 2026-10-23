"""Scaling preserves execution and strengthens the six-population gates."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import io
import time

import pytest
import torch

from panama_uav.config import CurriculumStage,load_config
from panama_uav.phase1_data import FAMILIES,prepare_dataset
from panama_uav.phase1_execution import verify_execution_compatibility
from panama_uav.phase1_revised import evaluate,subset
from panama_uav.phase1_scale import (METRICS,POPULATIONS,PROTOCOL,ScaleSession,
    load_protocol,validate_dataset,verify_payload,measure_training_scale,project_runtime,launch)
from panama_uav.phase1_stopping import may_open_heldout,training_stop
from panama_uav.training import TrainingSession
from panama_uav.remote_audit import digest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    data=prepare_dataset(tmp_path/'data',seed=303,train_maps=1,eval_maps=1,eval_cases=2,populations=POPULATIONS)
    data.update(schema='clean_mapf_split_v3_scale16',protocol=PROTOCOL,excluded_historical_maps=[])
    return data


def config():
    base=load_config(ROOT/'configs/phase1_clean.toml')
    return replace(base,curriculum=tuple(CurriculumStage(name=f'n{n}',num_agents=n) for n in POPULATIONS),
        training=replace(base.training,replay_capacity=128,batch_size=4))


def session(data,cap=5):
    torch.set_num_threads(1)
    return ScaleSession(config(),data,42,'scale-test',torch.device('cpu'),
        dict(episode_cap_per_population=cap,introduction_every_episodes=1))


def tick(s):
    if s.environment.step_count>=s.environment.max_episode_steps or s.environment.all_agents_at_goals:
        s.start_episode()
    s.environment.max_episode_steps=1
    return s.step()


def test_protocol_keeps_episode_budget_and_original_learner():
    spec=load_protocol(ROOT/'configs/phase1_scale.toml')
    assert spec['curriculum']['episode_cap_per_population']==2500
    assert spec['learner']==dict(learning_rate=.001,gamma=.9,epsilon_decay_steps=20000)
    assert spec['curriculum']['minimum_episodes_per_population']==500
    assert spec['evaluation']['validation_cases_per_family']==40


def test_json_provenance_hash_handles_two_digit_robot_ids():
    numeric={n:{'sources':{n:1.0}} for n in POPULATIONS}
    textual={str(n):{'sources':{str(n):1.0}} for n in POPULATIONS}
    assert digest(numeric)==digest(textual)
    textual['16']['sources']['16']=2.0
    assert digest(numeric)!=digest(textual)


def test_dataset_contains_disjoint_six_population_cases(dataset):
    assert validate_dataset(dataset)
    for name in ('validation','test'):
        rows=dataset['banks'][name]
        assert len(rows)==60
        assert Counter(r['num_agents'] for r in rows)==dict.fromkeys(POPULATIONS,10)
        assert len(subset(rows,2))==60
        assert all(len(set(map(tuple,r['starts'])))==r['num_agents'] for r in rows)
    bad=deepcopy(dataset);bad['excluded_historical_maps']=[dataset['maps'][0]['canonical']]
    with pytest.raises(ValueError,match='Historical'):validate_dataset(bad)


def test_population_and_family_exposure_cannot_be_gated_by_failure(dataset):
    s=session(dataset);completed=[]
    for _ in range(30):
        row,episode=tick(s);completed.append(episode)
        assert row['communication_delay_seconds']==0
        assert row['actual_collision_errors']==0
    assert s.population_episodes==dict.fromkeys(POPULATIONS,5)
    assert s.episode_budget_complete()
    assert all({e['family'] for e in completed if e['num_agents']==n}==set(FAMILIES) for n in POPULATIONS)
    with pytest.raises(ValueError,match='cap'):s.start_episode()


def test_resume_preserves_scaling_schedule_cycles_and_learning(dataset):
    s=session(dataset)
    for _ in range(7):tick(s)
    stream=io.BytesIO();torch.save(s.checkpoint(),stream)
    expected=[tick(s)[0] for _ in range(5)]
    weights={k:v.clone() for k,v in s.policy.state_dict().items()}
    stream.seek(0)
    restored=TrainingSession.restore(torch.load(stream,weights_only=False),s.config,'clean',42,'scale-test',s.device)
    assert [tick(restored)[0] for _ in range(5)]==expected
    assert all(torch.equal(v,restored.policy.state_dict()[k]) for k,v in weights.items())
    assert restored.population_episodes==s.population_episodes


def gates():
    groups=[dict(num_agents=n,episodes=200,complete=True,success_rate=.95,passed=True,
        exposure_passed=True,relevant_pairs=200,
        per_family={f:dict(episodes=40,success_rate=.95,exposed_cases=20) for f in FAMILIES}) for n in POPULATIONS]
    comp=dict(status='passed',execution_contract='remote-aligned-v2',model_weights_sha256='same',
        checks=[dict(num_agents=n,replay_status='passed',positions_rewards_priorities_equal=True) for n in POPULATIONS])
    return dict(training_protocol=PROTOCOL,metric_version=METRICS,condition='clean',
        curriculum_completed=True,phase1_qualified=True,minimum_episodes_per_population=500,
        population_episodes=dict.fromkeys(POPULATIONS,500),qualification=groups,held_out=deepcopy(groups),
        compatibility=comp,model_weights_sha256='same')


def test_same_model_must_pass_every_population_and_family():
    data=gates();assert verify_payload(data)
    for change in ('missing_population','missing_family','coverage','weights','replay','smoke'):
        bad=deepcopy(data)
        if change=='missing_population':bad['qualification'].pop()
        if change=='missing_family':bad['held_out'][-1]['per_family']['corridor']['success_rate']=.925
        if change=='coverage':bad['population_episodes'][16]=499
        if change=='weights':bad['compatibility']['model_weights_sha256']='different'
        if change=='replay':bad['compatibility']['checks'][-1]['replay_status']='failed'
        if change=='smoke':bad['smoke']=True
        assert not verify_payload(bad),change


def test_old_four_population_compatibility_cannot_open_scale_test():
    data=gates();comp=deepcopy(data['compatibility']);comp['checks']=comp['checks'][:4]
    assert not may_open_heldout(data['qualification'],data['population_episodes'],comp,populations=POPULATIONS)
    history=[dict(generated_joint_steps=100000,populations=[dict(g,complete=False) for g in data['qualification']])]*8
    assert training_stop(history,data['population_episodes'],populations=POPULATIONS) is None


def test_unchanged_observation_and_actual_remote_execution_at_12_and_16(dataset,tmp_path):
    s=session(dataset)
    result=verify_execution_compatibility(s.config,s.policy,dataset['banks']['validation'],s.device,
        tmp_path/'audit',populations=(12,16))
    assert result['status']=='passed'
    assert [r['num_agents'] for r in result['checks']]==[12,16]
    assert all(r['positions_rewards_priorities_equal'] and r['replay_status']=='passed' for r in result['checks'])


def test_evaluation_records_scale_version_and_never_counts_unfinished_cases(dataset,tmp_path):
    s=session(dataset)
    rows=evaluate(s.config,s.policy,dataset['banks']['validation'],s.device,tmp_path/'empty.jsonl',
        'expired',time.monotonic()-1,protocol=PROTOCOL,metrics=METRICS,diagnostics=True)
    assert rows==[] and s.generated_steps==0


def test_measured_budget_charges_all_250_step_failures_and_full_evaluation():
    spec=load_protocol(ROOT/'configs/phase1_scale.toml')
    measured={n:dict(steps=500,seconds=10.,completed_episodes=2,completed_episode_steps=500) for n in POPULATIONS}
    result=project_runtime(measured,dict.fromkeys(POPULATIONS,.003),spec,.5)
    assert result['total_episode_cap']==15000
    assert result['all_250_step_training_seconds']==75000
    assert result['periodic_validation_checks']==15
    assert result['final_evaluation_reserve_seconds']>0
    assert result['proposed_budget_seconds']>=result['conservative_total_seconds']*1.2
    assert result['proposed_budget_seconds']%1800==0
    broken=deepcopy(spec);broken['evaluation']['maximum_wall_seconds']=10
    with pytest.raises(ValueError,match='ceiling'):project_runtime(measured,dict.fromkeys(POPULATIONS,.003),broken,.5)


def test_monitoring_registration_and_resume_are_launch_prerequisites(tmp_path):
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='automation'):launch(tmp_path)


def test_another_full_run_cannot_reuse_the_same_heldout_bank(tmp_path):
    import json
    from panama_uav.records import sha256
    (tmp_path/'budget_plan.json').write_text('{}')
    manifest=dict(protocol=PROTOCOL,budget_plan_sha256=sha256(tmp_path/'budget_plan.json'),
        dataset_origin=str(tmp_path/'dataset.json'))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'qualification_owner.json').write_text(json.dumps(dict(root='different-full-run')))
    with pytest.raises(ValueError,match='test bank belongs'):launch(tmp_path,monitor_registered=True)


def test_calibration_throughput_counts_completed_episodes_only():
    learning=[];episodes=[]
    for n in POPULATIONS:
        for k in range(2):
            name=f'n{n}-{k}';index=len(learning)+1
            learning.append(dict(case_id=name,num_agents=n,generated_joint_steps=index,wall_seconds=float(index),communication_delay_seconds=0))
            episodes.append(dict(case_id=name,num_agents=n,split='train',steps=1))
    learning.append(dict(case_id='censored',num_agents=16,generated_joint_steps=13,wall_seconds=14.,communication_delay_seconds=0))
    result=measure_training_scale(learning,episodes)
    assert result[16]['steps']==3 and result[16]['seconds']==4 and result[16]['completed_episodes']==2
    with pytest.raises(ValueError,match='Duplicate'):measure_training_scale(learning,episodes+[episodes[-1]])


def test_actual_scaled_worker_keeps_test_closed_without_training_coverage(dataset,tmp_path,monkeypatch):
    import json
    import panama_uav.phase1_revised as revised
    import panama_uav.phase1_execution as execution
    from panama_uav.records import sha256
    dataset_path=tmp_path/'scaled.json';dataset_path.write_text(json.dumps(dataset))
    base=config();cfg=replace(base,training=replace(base.training,target_environment_steps=0,budget_seconds=30))
    full=dict(evaluation_reserve_seconds=2,shutdown_reserve_seconds=1,minimum_episodes_per_population=500,
        validation_interval_steps=50000,validation_interval_episodes=1000,validation_cases_per_family=40,
        extended_protocol=PROTOCOL,gate_test_on_validation=True)
    manifest=dict(protocol=PROTOCOL,populations=list(POPULATIONS),mode='full',seed=42,variant='original_hp',
        spec=dict(full=full),configuration=cfg.to_dict(),dataset_path=str(dataset_path),dataset_sha256=sha256(dataset_path),
        curriculum=dict(episode_cap_per_population=2500,introduction_every_episodes=40),exploration_clock='global',
        source_commit='test-source',budget_plan_sha256='test-budget')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    requested=[]
    def fake_evaluation(config,model,bank,device,path,tag,deadline,stop,**kwargs):
        assert kwargs['protocol']==PROTOCOL and all(c['split']=='validation' for c in bank)
        requested.extend(bank)
        return []
    monkeypatch.setattr(revised,'evaluate',fake_evaluation)
    monkeypatch.setattr(execution,'verify_execution_compatibility',lambda *a,**k:dict(status='passed',
        execution_contract='remote-aligned-v2',checks=[]))
    outcome=revised.run_one(tmp_path,manifest,'cpu')
    assert requested and not outcome['qualified'] and outcome['held_out']==[]
    assert 'test remains untouched' in outcome['held_out_skipped_reason']
    assert len(outcome['qualification'])==6
    assert (tmp_path/'checkpoint.pt').exists()
