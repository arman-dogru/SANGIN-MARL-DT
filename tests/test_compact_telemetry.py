"""Lossless telemetry storage must reproduce actions, clocks and event hashes."""
from dataclasses import replace
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compact_telemetry import DiskEvents,MemoryEvents,EventIndex,install


def test_lazy_records_preserve_json_dict_and_archive_semantics(tmp_path):
    values=[dict(event='state',event_id=f'rdt:{i}',time=i*.1,data=[{'x':i,'payload':'repeat'*100}]) for i in range(5)]
    stored=DiskEvents(tmp_path/'events.sqlite',values)
    assert json.loads(json.dumps(list(stored),sort_keys=True))==values
    assert dict(stored[1])==values[1] and stored[1].copy()==values[1]
    assert EventIndex(stored)['rdt:2']['data']==values[2]['data']
    restored=pickle.loads(pickle.dumps(stored,protocol=5))
    assert isinstance(restored,MemoryEvents)
    assert json.loads(json.dumps(list(restored)))==values


def test_disk_telemetry_matches_original_execution_and_full_replay(tmp_path,monkeypatch):
    from test_frequency_motion import fixture
    import panama_uav.redesign as redesign
    from panama_uav.models import DQN,select_action
    from panama_uav.inference_controls import InferenceCondition
    from panama_uav.redesign_audit import audit_redesign
    from panama_uav.remote_audit import digest
    from run_omega_sensitivity import normalized
    settings=redesign.settings_for('A4',2.)
    _,_,scenario,config=fixture(tmp_path,settings=settings)
    config=replace(config,resource_allocation=replace(config.resource_allocation,failure_directory=str(tmp_path/'compact/solver_failures')))
    torch.set_num_threads(1);torch.manual_seed(5);model=DQN((3,15,15),config.vector_length).eval()
    case=dict(case_id='compact',num_agents=3,seed=17,starts=list(map(list,scenario.starts)),goals=list(map(list,scenario.goals)))
    args=(config,model,case,torch.device('cpu'),0,scenario,InferenceCondition('edge','edge'),settings)
    row,before=redesign.run_frequency_episode(*args)
    original=redesign.experimental_class
    monkeypatch.setattr(redesign,'experimental_class',original)
    monkeypatch.setattr(redesign,'_compact_telemetry_installed',False,raising=False)
    install(None)
    other,after=redesign.run_frequency_episode(*args)
    assert digest(before['remote_events'])==digest(list(after['remote_events']))
    # Solver wall-clock timings are host measurements, not simulated events.
    assert digest(normalized(before['oracle_events']))==digest(normalized(list(after['oracle_events'])))
    assert digest(before['frames'])==digest(after['frames'])
    assert row==other
    infer=lambda loc,robot,obs:select_action(model,obs,0.,np.random.default_rng(0),torch.device('cpu'))
    assert audit_redesign(after,infer)['status']=='passed'
    restored=pickle.loads(pickle.dumps(after,protocol=5))
    assert digest(list(restored['remote_events']))==digest(before['remote_events'])
    assert audit_redesign(restored,infer)['status']=='passed'
