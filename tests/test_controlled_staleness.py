from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from panama_uav.config import load_config
from panama_uav.controlled_scenarios import FAMILIES, make_cases, scenario_for
from panama_uav.evaluation import run_episode
from panama_uav.models import DQN
from panama_uav.remote_audit import digest, replay_trace
from panama_uav.remote_state import DTState, PhysicalReport, remote_observation
from panama_uav.staleness_audit import audit_stress
from panama_uav.staleness_remote import controlled_class
from panama_uav.staleness_state import ReplicaHistory
from panama_uav.staleness_study import run_stress_pilot, stress_metrics, stress_treatments, report_stress, _contrasts
from test_deadline_study import inputs

ROOT=Path(__file__).resolve().parents[1]


def publish(history, by_id, robot, cycle, arrival, path=None):
    source=float(cycle)
    state=DTState(robot,(7,7+robot),(7,10),path or [(7,7+robot),(7,9)],
        float(robot),False,source+.1,cycle,{robot:source},f'report:{cycle}:{robot}')
    ref=f'state:{cycle}:{robot}'
    original=dict(event='dt_state_event',event_id=ref,time=arrival,location='cloud',
        state=state.payload(),available_at=arrival)
    by_id[ref]=original
    accepted=history.replicas['cloud'].states.get(robot)
    accepted=accepted is None or source >= min(accepted.source_timestamps.values()) and cycle>accepted.version
    event=dict(event='dt_state_acceptance',state_event=ref,time=arrival,accepted=accepted)
    history.observe(event,by_id)


@pytest.mark.parametrize('delay',range(6))
def test_delay_selects_only_matured_accepted_peer_arrivals_and_preserves_self(delay):
    history=ReplicaHistory(); refs={}
    for cycle in range(7):
        for robot in range(2):
            publish(history,refs,robot,cycle,cycle+.2)
    view=history.view('cloud',0,6.25,delay)
    assert view.states[0].source_timestamps=={0:6.0}
    assert view.states[1].source_timestamps=={1:float(6-delay)}
    assert view.states[1].timestamp==pytest.approx(6-delay+.1)
    assert view.state_ids[1]==f'state:{6-delay}:1'
    assert history.replicas['cloud'].states[1].version==6


def test_arrival_boundary_inclusive_and_warmup_retains_original_briefing():
    history=ReplicaHistory(); refs={}
    for robot in range(2):
        state=DTState(robot,(7,7+robot),(7,10),[(7,7+robot)],float(robot),False,0.,0,{robot:0.},'boot')
        event=dict(event='dt_state_event',event_id=f'boot:{robot}',state=state.payload(),
            location='cloud',available_at=0.,time=0.,bootstrap=True,accepted=True)
        history.observe(event,refs)
    publish(history,refs,1,1,1.2)
    assert history.view('cloud',0,2.2,1).states[1].version==1
    assert history.view('cloud',0,2.199,1).states[1].version==0
    assert history.view('cloud',0,.25,5).state_ids[1]=='boot:1'
    publish(history,refs,1,0,2.3)
    assert history.replicas['cloud'].states[1].version==1


@pytest.mark.parametrize('bad',[-1,6,1.5,True])
def test_invalid_delay_rejected(bad):
    with pytest.raises(ValueError):
        controlled_class(bad)


def test_delay_changes_actual_intention_channel_without_self_or_radio_changes():
    history=ReplicaHistory(); refs={}
    for cycle in range(3):
        publish(history,refs,0,cycle,cycle+.2)
        publish(history,refs,1,cycle,cycle+.2,path=[(7,8),(7,9+cycle)])
    report=PhysicalReport(0,2,2.,(7,7),(7,10),False,3.,np.zeros((15,15)).tolist(),np.zeros((15,15)).tolist(),'report:2:0')
    own=DTState(0,(7,7),(7,10),[(7,7),(7,8),(7,9),(7,10)],4.,False,2.1,2,{0:2.},report.report_id)
    fresh=remote_observation(report,own,history.view('cloud',0,2.25,0),7,5)[0]
    delayed=remote_observation(report,own,history.view('cloud',0,2.25,1),7,5)[0]
    np.testing.assert_array_equal(fresh['spatial'][:2],delayed['spatial'][:2])
    np.testing.assert_array_equal(fresh['vector'],delayed['vector'])
    assert not np.array_equal(fresh['spatial'][2],delayed['spatial'][2])
    assert history.replicas['cloud'].states[1].version==2


def test_scenarios_are_reproducible_with_reachable_unique_paired_geometry(tmp_path):
    first=make_cases(tmp_path/'one',2,(2,4,6,8))
    second=make_cases(tmp_path/'two',2,(2,4,6,8))
    assert len(first)==24
    for a,b in zip(first,second):
        assert {k:v for k,v in a.items() if k!='map_path'}=={k:v for k,v in b.items() if k!='map_path'}
        assert len(scenario_for(a).starts)==a['num_agents']


@pytest.fixture
def small_trace(tmp_path):
    config=load_config(ROOT/'configs/paper_remote_dt_250ms.toml')
    # Native radio model with a deterministic fixed geometry fixture.
    config=replace(config,communication=replace(config.communication,leo_timeline=None,handover_enabled=False))
    torch.set_num_threads(1); torch.manual_seed(42)
    model=DQN((3,15,15),config.vector_length).eval()
    case=make_cases(tmp_path/'maps',1,(2,))[0]
    return config,model,case


def test_zero_delay_matches_original_causal_event_chain_actions_and_arrivals(small_trace):
    cfg,model,case=small_trace
    _,original=run_episode(cfg,model,case,torch.device('cpu'),scenario=scenario_for(case),max_steps=4)
    _,controlled=run_episode(cfg,model,case,torch.device('cpu'),scenario=scenario_for(case),max_steps=4,
        remote_communication_class=controlled_class(0))
    assert original['remote_events']==controlled['remote_events']
    assert [f['positions'] for f in original['frames']]==[f['positions'] for f in controlled['frames']]
    assert [t.get('delivered_at') for t in original['resource_tasks']]==[t.get('delivered_at') for t in controlled['resource_tasks']]
    assert audit_stress(controlled)['status']=='passed'
    assert all(not e['injected_delay_disagreement'] for e in controlled['oracle_events'] if e['event']=='controlled_peer_shadow')


def test_delay_replay_causal_isolation_and_checksum_tampering(small_trace):
    cfg,model,case=small_trace
    _,trace=run_episode(cfg,model,case,torch.device('cpu'),scenario=scenario_for(case),max_steps=4,
        remote_communication_class=controlled_class(2))
    assert audit_stress(trace)['status']=='passed'
    metrics=stress_metrics(dict(case,steps=4),trace)
    assert metrics['timely_comparisons']+metrics['command_cutoff_misses']==metrics['active_decisions']
    assert metrics['inference_pairs']>0
    broken=deepcopy(trace)
    shadow=next(e for e in broken['oracle_events'] if e['event']=='controlled_peer_shadow')
    shadow['network_hash']='tampered'
    with pytest.raises(ValueError,match='checksum'):
        audit_stress(broken)
    broken=deepcopy(trace)
    # Re-sign malicious causal evidence: the replay still rejects future peers.
    e=next(e for e in broken['remote_events'] if e['event']=='remote_inference' and e['time']>2)
    e['sources']['1']=999.
    previous='0'*64
    for event in broken['remote_events']:
        event['previous_checksum']=previous
        event['checksum']=digest({k:v for k,v in event.items() if k!='checksum'})
        previous=event['checksum']
    with pytest.raises(ValueError,match='provenance|future|source'):
        replay_trace(broken)


def test_treatments_preserve_workload_solver_radio_and_movement_configuration(small_trace):
    config,_,_=small_trace
    for name,cfg in stress_treatments(config):
        assert cfg.resource_allocation.drone_cycles_per_second==config.resource_allocation.drone_cycles_per_second
        assert cfg.resource_allocation.cloud_cycles_per_second==config.resource_allocation.cloud_cycles_per_second
        assert cfg.resource_allocation.task_input_bits==config.resource_allocation.task_input_bits
        assert cfg.resource_allocation.cycles_per_bit==config.resource_allocation.cycles_per_bit
        assert cfg.resource_allocation.on_collision==config.resource_allocation.on_collision
        assert cfg.reward==config.reward and cfg.map==config.map
        if name!='fresh_transport':
            assert cfg.communication==config.communication
    with pytest.raises(ValueError,match='250 ms'):
        stress_treatments(load_config(ROOT/'configs/paper_remote_dt.toml'))


@pytest.mark.parametrize('kwargs',[dict(cases=50),dict(max_steps=250),dict(delays=(1,2))])
def test_full_stress_sweep_is_gated_before_writes(tmp_path,kwargs):
    config=load_config(ROOT/'configs/paper_remote_dt_250ms.toml')
    with pytest.raises(ValueError):
        run_stress_pilot(config,tmp_path/'blocked','missing.pt',**kwargs)
    assert not (tmp_path/'blocked').exists()


def test_bounded_workflow_archives_audits_and_pairing(inputs,tmp_path):
    config,checkpoint=inputs
    config=replace(config,resource_allocation=replace(config.resource_allocation,
        decision_budget_seconds=.25,command_cutoff_seconds=.25,task_deadline_seconds=.25))
    output=tmp_path/'stress'
    rows=run_stress_pilot(config,output,checkpoint,cases=1,counts=(2,),delays=(0,1),max_steps=2,workers=1,diagnostic=True)
    assert len(rows)==18
    validation=json.loads((output/'validation.json').read_text())
    assert validation['episodes']==18 and validation['actual_actions_verified']>0
    assert validation['peer_shadow_actions_verified']==2*validation['actual_actions_verified']
    bank=json.loads((output/'paired_cases.json').read_text())
    bank[0]['seed']+=1
    (output/'paired_cases.json').write_text(json.dumps(bank))
    with pytest.raises(ValueError,match='checksum'):
        report_stress(output)


def test_age_contrasts_allow_more_than_one_second_and_interaction_is_paired():
    from panama_uav.staleness_study import RATIOS, SCALARS, TREATMENTS
    rows=[]
    for delay in (0,5):
        for treatment in TREATMENTS:
            for case in (0,1):
                r=dict(scenario='intersection',num_agents=2,delay_steps=delay,treatment=treatment,case_id=str(case))
                r.update({k:0 for k in (*SCALARS,*RATIOS)})
                for a,b in RATIOS.values():
                    r[a]=0; r[b]=1
                r['mean_peer_aoi']=5.+delay
                r['mean_relevant_peer_aoi']=6.+delay
                r['restricted_makespan_seconds']=20+delay*(1 if treatment=='priority' else 2)
                rows.append(r)
    results=_contrasts(rows,100)
    age=next(r for r in results if r['comparison']=='delay_vs_zero' and r['delay_steps']==5 and r['metric']=='mean_peer_aoi')
    assert age['estimate']==age['ci95_low']==age['ci95_high']==5
    effect=next(r for r in results if r['comparison']=='priority_mitigation_interaction' and r['delay_steps']==5 and r['metric']=='restricted_makespan_seconds')
    assert effect['estimate']==effect['ci95_low']==effect['ci95_high']==-5
