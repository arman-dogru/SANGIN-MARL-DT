from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from panama_uav.config import load_config, ExperimentMode
from panama_uav.phase1_data import canonical_map, prepare_dataset, verify_dataset, FAMILIES, scenario_for
from panama_uav.phase1 import CleanMAPFSession, clean_config, checkpoint_payload, verify_payload, summarize_cases, evaluate
from panama_uav.phase1_environment import CleanMAPFEnvironment
from panama_uav.phase1_compatibility import verify_compatibility
from panama_uav.training import TrainingSession, load_model
from panama_uav.controlled_scenarios import geometry

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    return prepare_dataset(tmp_path/'data',seed=72,train_maps=1,eval_maps=1,eval_cases=2)


def config():
    base=load_config(ROOT/'configs/phase1_clean.toml')
    return replace(base,training=replace(base.training,replay_capacity=128,batch_size=4))


def test_clean_config_disables_all_delay_and_compute_without_changing_panama():
    original=load_config(ROOT/'configs/paper_remote_dt_250ms.toml');clean=clean_config(original)
    assert clean.communication.profile=='perfect' and clean.communication.stale_information==()
    assert clean.communication.leo_timeline is None and clean.communication.uplink_trace is None
    assert not clean.resource_allocation.enabled
    assert clean.reward==original.reward and clean.map==original.map
    assert clean.resource_allocation.drone_cycles_per_second==original.resource_allocation.drone_cycles_per_second


def test_maps_and_scenarios_are_disjoint_and_stress_pilot_reserved(dataset):
    assert verify_dataset(dataset)
    assert len({m['canonical'] for m in dataset['maps']})==15
    for family in ('intersection','narrow_corridor','shared_bottleneck'):
        assert canonical_map(geometry(family)[0]) not in {m['canonical'] for m in dataset['maps']}
    for split in ('validation','test'):
        assert len(dataset['banks'][split])==40
        assert all(c['split']==split for c in dataset['banks'][split])
        assert all(c['initial_relevant_pairs']>0 for c in dataset['banks'][split] if c['designed_coordination'])


def test_canonical_leakage_guard_catches_rotations_reflections_and_translation():
    grid=geometry('intersection')[0];expected=canonical_map(grid)
    assert canonical_map(np.rot90(grid))==canonical_map(np.fliplr(grid))==expected
    crop=grid[3:29,3:29];shifted=np.ones((40,40),np.uint8);shifted[8:34,7:33]=crop
    assert canonical_map(shifted)==expected


def test_leakage_and_corruption_rejected(dataset):
    broken=deepcopy(dataset);broken['maps'][1]['canonical']=broken['maps'][0]['canonical']
    with pytest.raises(ValueError,match='leakage|checksum'):verify_dataset(broken)
    broken=deepcopy(dataset);broken['banks']['test'][0]['map_id']=broken['maps'][0]['map_id']
    with pytest.raises(ValueError,match='leakage'):verify_dataset(broken)


def test_training_samples_only_training_maps_and_is_reproducible(dataset):
    torch.set_num_threads(1)
    a=CleanMAPFSession(config(),dataset,42,'phase1-test',torch.device('cpu'))
    first=a.current_case
    b=CleanMAPFSession(config(),dataset,42,'phase1-test',torch.device('cpu'))
    assert first==b.current_case and first['split']=='train'
    for k,v in a.policy.state_dict().items():assert torch.equal(v,b.policy.state_dict()[k])
    assert all(age==0 for fields in a.environment.communication.ages().values() for age in fields.values())


def test_full_replay_optimizer_rng_resume_is_exact(dataset):
    torch.set_num_threads(1)
    original=CleanMAPFSession(config(),dataset,42,'phase1-resume',torch.device('cpu'))
    for _ in range(8):original.step()
    stream=io.BytesIO();torch.save(original.checkpoint(),stream)
    expected=[original.step()[0] for _ in range(5)]
    weights={k:v.detach().clone() for k,v in original.policy.state_dict().items()}
    stream.seek(0)
    restored=TrainingSession.restore(torch.load(stream,weights_only=False),clean_config(config()),'clean',42,'phase1-resume',torch.device('cpu'))
    assert [restored.step()[0] for _ in range(5)]==expected
    for k,v in restored.policy.state_dict().items():assert torch.equal(v,weights[k])


def test_remote_observation_and_frozen_model_replay_equivalence(dataset,tmp_path):
    session=CleanMAPFSession(config(),dataset,42,'compatibility',torch.device('cpu'))
    result=verify_compatibility(session.config,session.policy,dataset['banks']['validation'],torch.device('cpu'),tmp_path)
    assert result['status']=='passed'
    assert sum(c['initial_observations_equal'] for c in result['checks'])==20
    assert sum(c['remote_actions_verified'] for c in result['checks'])==60


def test_qualification_does_not_lower_success_or_exposure_thresholds():
    rows=[]
    for f in FAMILIES:
        rows.extend(dict(family=f,map_id=f+'-'+str(i%5),success=i<38,steps=20,waiting_steps=1,
            relevant_pairs=2,directed_pairs=4,vertex_conflicts=0,edge_conflicts=0) for i in range(40))
    result=summarize_cases(rows)
    assert result['passed'] and result['episodes']==200 and result['success_rate']==.95
    assert result['success_ci95'][0]<.95<result['success_ci95'][1]
    rows[0]['success']=False
    assert not summarize_cases(rows)['passed']
    rows[0]['success']=True
    for r in rows[:40]:r['relevant_pairs']=0
    assert not summarize_cases(rows)['exposure_passed']
    assert not summarize_cases(rows[:199])['complete']


def test_unqualified_new_checkpoint_never_passes_legacy_or_new_gates(dataset,tmp_path):
    session=CleanMAPFSession(config(),dataset,42,'payload',torch.device('cpu'))
    payload=checkpoint_payload(session,'dataset-hash',smoke=True)
    assert not verify_payload(payload)
    path=tmp_path/'smoke_model.pt';torch.save(payload,path)
    loaded,_=load_model(path,config(),torch.device('cpu'),require_qualified=False)
    assert loaded is not None
    with pytest.raises(ValueError,match='qualified'):
        load_model(path,config(),torch.device('cpu'),require_qualified=True)


def test_evaluation_partial_and_reused_rows_do_not_leak_into_training(dataset,tmp_path):
    session=CleanMAPFSession(config(),dataset,42,'evaluation',torch.device('cpu'))
    bank=dataset['banks']['validation'][:2];path=tmp_path/'evaluation.jsonl'
    assert not evaluate(session.config,session.policy,bank,session.device,path,'partial',time.monotonic()-1)
    rows=evaluate(session.config,session.policy,bank,session.device,path,'done',time.monotonic()+30,cap=2)
    repeated=evaluate(session.config,session.policy,bank,session.device,path,'done',time.monotonic()+30,cap=2)
    assert rows==repeated and len(rows)==2
    assert session.generated_steps==session.learner_updates==0


def test_report_preserves_failure_split_and_never_claims_smoke_qualification(tmp_path):
    from panama_uav.phase1_report import report
    status=dict(status='smoke_complete',qualified=False,wall_seconds=12.,budget_seconds=120.,
        episodes=0,generated_joint_steps=128,learner_updates=121,qualification=[],held_out=[])
    (tmp_path/'status.json').write_text(json.dumps(status))
    (tmp_path/'final_test_cases.jsonl').write_text(json.dumps(dict(split='test',success=False,
        case_id='test-case',num_agents=2,steps=8))+'\n')
    report(tmp_path)
    assert 'test-case' in (tmp_path/'failures.csv').read_text()
    assert 'Qualified: **False**' in (tmp_path/'PHASE1_REPORT.md').read_text()


def test_driver_smoke_completes_all_artifacts_and_resumes_without_training(dataset,tmp_path):
    from panama_uav.phase1 import run
    from panama_uav.records import sha256
    cfg=clean_config(config())
    cfg=replace(cfg,training=replace(cfg.training,budget_seconds=60.,shutdown_reserve_seconds=1.,
        target_environment_steps=4,checkpoint_interval_steps=2))
    manifest=dict(configuration=cfg.to_dict(),smoke=True,seed=42,setup_seconds=0.,
        dataset_sha256=sha256(tmp_path/'data/dataset.json'),
        code_sha256={p.name:sha256(p) for p in (ROOT/'src/panama_uav').glob('*.py')})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    result=run(tmp_path,'cpu',smoke=True)
    assert result['status']=='smoke_complete' and not result['qualified']
    for name in ('checkpoint.pt','final_model.pt','compatibility.json','PHASE1_REPORT.md','failures.csv'):
        assert (tmp_path/name).exists()
    repeated=run(tmp_path,'cpu',smoke=True,resume=True)
    assert repeated['generated_joint_steps']==4
    with pytest.raises(ValueError,match='resume identity'):
        run(tmp_path,'cpu',smoke=False,resume=True)
