from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from panama_uav.phase1_budget import verify_selection
from panama_uav.phase1_extended import bootstrap_success
from panama_uav.phase1_revised import subset


@pytest.fixture
def comparison(tmp_path):
    bank=[]
    families=('random_obstacles','intersection','corridor','bottleneck','warehouse')
    for n in (2,4,6,8):
        for f in families:
            for i in range(40):
                bank.append(dict(case_id=f'{f}-{n}-{i}',map_id=f'{f}-{i%5}',map_canonical=f'{f}-h{i%5}',
                    family=f,num_agents=n,seed=i,starts=[[0,0]],goals=[[0,1]],split='validation',
                    evaluation_index=i,designed_coordination=i<20))
    (tmp_path/'data.json').write_text(json.dumps(dict(banks={'validation':bank})))
    spec=dict(ablation=dict(joint_steps=30000,seeds=[42,73],minimum_success_gain=.05),
        variants=[dict(name='original_hp'),dict(name='lower_lr')])
    (tmp_path/'study.json').write_text(json.dumps(dict(spec=spec,dataset_path=str(tmp_path/'data.json'))))
    cohort=subset(bank,8)
    for name in ('original_hp','lower_lr'):
        for seed in (42,73):
            folder=tmp_path/'ablations'/f'{name}-seed{seed}';folder.mkdir(parents=True)
            (folder/'status.json').write_text(json.dumps(dict(status='ablation_complete',generated_joint_steps=30000,held_out=[])))
            with (folder/'final_validation_cases.jsonl').open('w') as stream:
                for row in cohort:stream.write(json.dumps(dict(row,success=name=='lower_lr'))+'\n')
    (tmp_path/'ablation_comparison.json').write_text(json.dumps(dict(selected='lower_lr',full_run_justified=True)))
    return tmp_path


def test_primary_selection_is_recomputed_without_changing_its_results(comparison):
    before={str(p):p.read_bytes() for p in comparison.rglob('*') if p.is_file()}
    setup,decision,rows=verify_selection(comparison)
    assert decision['selected']=='lower_lr' and len(rows['lower_lr'])==320
    assert all(Path(p).read_bytes()==content for p,content in before.items())


def test_tampered_selection_cannot_authorize_extended_training(comparison):
    path=comparison/'ablation_comparison.json'
    path.write_text(json.dumps(dict(selected='original_hp',full_run_justified=True)))
    with pytest.raises(ValueError,match='disagrees'):verify_selection(comparison)


def test_incomplete_or_test_contaminated_primary_cannot_authorize_extension(comparison):
    path=comparison/'ablations/lower_lr-seed73/status.json'
    status=json.loads(path.read_text());status['held_out']=[dict(success_rate=1.)]
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError,match='test-contaminated'):verify_selection(comparison)


def test_confidence_interval_resamples_shared_case_blocks():
    rows=[dict(training_seed=s,map_id='a',success=True) for s in (42,73) for _ in range(3)]
    rows += [dict(training_seed=s,map_id='b',success=False) for s in (42,73)]
    lo,hi=bootstrap_success(rows,200)
    assert 0<=lo<=.75<=hi<=1
