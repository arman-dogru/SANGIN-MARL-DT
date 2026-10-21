from dataclasses import replace
from functools import lru_cache
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from panama_uav.config import COMMUNICATION_SCHEMA, FIXED_MAP, load_config
from panama_uav.deadline_study import (DEADLINES_MS, _pool_statistics, deadline_config,
    paired_deadline_changes, report_study, run_deadline_study, task_statistics)
from panama_uav.evaluation import case_bank, run_episode
from panama_uav.models import DQN
from panama_uav.records import read_jsonl, sha256
from panama_uav.remote_state import ActionCommand
from panama_uav.resource_study import study_cases
from test_remote_execution import environment, remote_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_templates(monkeypatch):
    import panama_uav.allocation as allocation
    monkeypatch.setattr(allocation,"_conic_template",lru_cache(maxsize=24)(allocation._conic_template.__wrapped__))


@pytest.fixture
def inputs(tmp_path):
    config = load_config(ROOT/'configs/paper_remote_dt.toml')
    timeline = tmp_path/'diagnostic-timeline.npz'
    np.savez(timeline,times=np.array([0.,21600.]),propagation_seconds=np.full(2,.002),
        satellite_up_bps=np.full(2,1e7),satellite_down_bps=np.full(2,5e7),
        blocked_windows=np.empty((0,3)),handover_events=json.dumps([]),metadata=json.dumps({}))
    config=replace(config,communication=replace(config.communication,leo_timeline=str(timeline)))
    torch.manual_seed(42)
    model=DQN((3,15,15),config.vector_length).eval()
    checkpoint=tmp_path/'diagnostic-model.pt'
    torch.save(dict(communication_schema=COMMUNICATION_SCHEMA,policy_family='panama_0.1_task',
        map_sha256=sha256(config.resolve(FIXED_MAP)),task_reward=config.to_dict()['reward'],
        spatial_shape=(3,15,15),vector_length=config.vector_length,model_state_dict=model.state_dict()),checkpoint)
    return config,checkpoint


def test_scenarios_change_only_the_three_cutoffs_and_keep_one_second_control():
    base=load_config(ROOT/'configs/paper_remote_dt.toml')
    assert deadline_config(base,100) == base
    for ms in DEADLINES_MS:
        cfg=deadline_config(base,ms)
        p=cfg.resource_allocation
        assert p.decision_budget_seconds == p.command_cutoff_seconds == p.task_deadline_seconds == ms/1000
        assert not p.enforce_task_deadlines and cfg.communication.seconds_per_step == 1
        restored=replace(cfg,resource_allocation=replace(p,decision_budget_seconds=.1,
            command_cutoff_seconds=.1,task_deadline_seconds=.1))
        assert restored == base  # Includes CPU/task/RF/PANAMA/oracle assumptions.
    with pytest.raises(ValueError,match='1-second'):
        deadline_config(replace(base,communication=replace(base.communication,seconds_per_step=.5)),250)
    with pytest.raises(ValueError,match='cutoffs'):
        deadline_config(base,200)
    with pytest.raises(ValueError,match='remote_dt'):
        deadline_config(load_config(ROOT/'configs/base.toml'),100)


@pytest.mark.parametrize('ms',DEADLINES_MS)
@pytest.mark.parametrize('offset',[-.001,0,.001])
def test_all_cutoff_boundaries_use_actual_arrival_and_inclusive_deadline(ms,offset):
    cfg=deadline_config(remote_config(drone_cycles_per_second=1e5,cloud_cycles_per_second=1e5),ms)
    env=environment(cfg)
    comm=env.communication
    cutoff=ms/1000
    arrival=cutoff+offset
    comm.transport.advance(.05,comm._receive)
    comm.jobs['0:0']['compute_completed_at']=.05
    command=ActionCommand(0,0,3,.05,cutoff,0,'drone',{0:0.},'0:0','fixture',{}, {})
    comm._send_remote(0,'remote_action_command',dict(job_id='0:0',command=command.payload()),
        (f'process:{arrival-.05}',),version=0)
    env.prepare_decision(cutoff)
    on_time=offset <= 0
    assert comm.execution_actions[0] == (3 if on_time else 4)
    if not on_time:
        assert comm.task_records()[0]['deadline_status'] == 'undelivered'
        comm.transport.advance(arrival,comm._receive)
    task=next(t for t in comm.task_records() if t['job_id']=='0:0')
    assert task['deadline_status'] == ('on_time' if on_time else 'late')
    assert task['cutoff_induced_wait'] == (not on_time)
    assert task['command_accepted'] == on_time
    assert task['command_cutoff_at'] == cutoff
    stats=task_statistics(comm.task_records(),comm.transport.now)
    assert stats['deadline_tasks_monitored'] == 2
    assert stats['deadline_commands_on_time'] == int(on_time)
    assert stats['deadline_commands_late'] == int(not on_time)
    assert stats['deadline_commands_undelivered'] == 1
    env.step(comm.execution_actions.copy())
    assert env.step_count == 1 and comm.transport.now == 1.0


def test_rates_keep_censoring_unassigned_and_no_work_locations_explicit():
    tasks=[dict(job_id='on',source_time=0.,deadline_at=.25,delivered_at=.25,
                observed_latency_seconds=.25,executed_x=1),
        dict(job_id='late',source_time=0.,deadline_at=.25,delivered_at=.3,
             observed_latency_seconds=.3,executed_x=0,cutoff_induced_wait=True),
        dict(job_id='overdue',source_time=0.,deadline_at=.25,executed_x=1,cutoff_induced_wait=True),
        dict(job_id='pending',source_time=1.,deadline_at=1.25)]
    stats=task_statistics(tasks,1.1)
    assert stats['command_on_time_rate'] == stats['command_late_rate'] == .25
    assert stats['command_undelivered_rate'] == .5
    assert stats['deadline_compliance_rate'] == pytest.approx(1/3)
    assert stats['deadline_censored_tasks'] == stats['task_deadline_misses'] == stats['deadline_cutoff_wait_tasks'] == 2
    assert stats['offload_drone_fraction'] == pytest.approx(2/3)
    assert stats['offload_cloud_fraction'] == pytest.approx(1/3)
    assert stats['offload_unassigned_fraction'] == .25
    assert task_statistics([],1)['deadline_compliance_rate'] is None
    assert task_statistics([],1)['command_on_time_rate'] is None
    with pytest.raises(ValueError,match='one record'):
        task_statistics([tasks[0],tasks[0]],1.1)
    with pytest.raises(ValueError,match='monitored'):
        task_statistics([dict(job_id='unmonitored',source_time=0.)],1.1)
    pending_stats=task_statistics([tasks[-1]],1.1)
    later_stats=task_statistics([tasks[0]],2.)
    pooled=_pool_statistics([pending_stats,later_stats],[tasks[-1],tasks[0]])
    assert pooled['deadline_pending_tasks'] == 1 and pooled['task_deadline_misses'] == 0
    assert pooled['deadline_compliance_rate'] == 1


def test_paired_cases_and_orbital_offsets_do_not_depend_on_deadline(inputs):
    cfg,_=inputs
    bank=study_cases(cfg,2,(2,4,6,8),250)
    assert len(bank) == 8
    assert [r['seed'] for r in bank] == [50_000_000+n*10_000+c for n in (2,4,6,8) for c in (0,1)]
    assert [r['orbital_offset_seconds'] for r in bank] == [0,100,5000,5100,10000,10100,15000,15100]
    for ms in DEADLINES_MS:
        assert study_cases(deadline_config(cfg,ms),2,(2,4,6,8),250) == bank


def test_100_ms_scenario_preserves_events_oracles_and_binary_allocations():
    base=remote_config(drone_cycles_per_second=1e9,cycles_per_bit=1000,
        task_deadline_seconds=.1,command_cutoff_seconds=.1,enforce_task_deadlines=False)
    torch.manual_seed(42)
    model=DQN((3,15,15),base.vector_length).eval()
    case=case_bank(base,1,(2,))[0]
    outputs=[]
    for cfg in (base,deadline_config(base,100)):
        row,trace=run_episode(cfg,model,case,torch.device('cpu'),max_steps=3)
        outputs.append((trace['remote_events'],trace['outcome'],[e for e in trace['oracle_events']
            if e['event'] in ('action_shadow','decision_action_shadow')]))
        assert all(t['executed_x'] in (0,1) for t in trace['resource_tasks'])
        assert all(t['deadline_seconds'] is None for e in trace['transport_events']
            if e['event']=='paper_allocation' for t in e['snapshot']['tasks'])
    assert outputs[0] == outputs[1]


def test_paired_changes_match_cases_and_omit_one_case_confidence_intervals():
    rows=[dict(deadline_ms=100,condition='priority',case_id='b',num_agents=2,waiting_steps=8),
        dict(deadline_ms=250,condition='priority',case_id='a',num_agents=2,waiting_steps=3),
        dict(deadline_ms=100,condition='priority',case_id='a',num_agents=2,waiting_steps=5),
        dict(deadline_ms=250,condition='priority',case_id='b',num_agents=2,waiting_steps=4)]
    result=next(r for r in paired_deadline_changes(rows) if r['deadline_ms']==250)
    assert result['mean_delta'] == -3 and result['pairs']==2
    single=paired_deadline_changes([rows[1],rows[2]])
    assert all(r['ci95_low'] is r['ci95_high'] is None for r in single)


def test_full_study_remains_gated_before_any_output_is_created(tmp_path):
    cfg=load_config(ROOT/'configs/paper_remote_dt.toml')
    with pytest.raises(ValueError,match='acceptance approval'):
        run_deadline_study(cfg,tmp_path/'full','missing.pt',cases=50)
    assert not (tmp_path/'full').exists()


def test_failed_condition_stops_the_workflow_without_fallback(inputs,tmp_path,monkeypatch):
    import panama_uav.deadline_study as study
    cfg,model=inputs
    calls=[]
    def fail(config,directory,*args,**kwargs):
        calls.append(config.resource_allocation.command_cutoff_seconds)
        raise RuntimeError('preserved solver failure')
    monkeypatch.setattr(study,'run_study',fail)
    root=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='preserved solver failure'):
        run_deadline_study(cfg,root,model,diagnostic=True,cases=1,counts=(2,),max_steps=2)
    status=json.loads((root/'status.json').read_text())
    assert calls == [.1] and status['status']=='failed' and status['active_deadline_ms']==100
    assert not (root/'deadline-250ms').exists()


def test_real_paired_workflow_exports_all_six_treatments_and_location_metrics(inputs,tmp_path,monkeypatch):
    cfg,model=inputs
    root=tmp_path/'study'
    summaries=run_deadline_study(cfg,root,model,diagnostic=True,cases=1,counts=(2,),max_steps=2)
    assert len(summaries)==24
    assert json.loads((root/'status.json').read_text())['completed_episodes']==24
    assert len(read_jsonl(root/'execution_locations.jsonl'))==72
    assert (root/'summary.csv').exists() and (root/'paired_vs_100ms.csv').exists()
    assert (root/'figures/location_compliance.png').exists()
    assert (root/'figures/mean_restricted_makespan_seconds.svg').exists()
    bank_hash=sha256(root/'held_out_cases.json')
    for ms in DEADLINES_MS:
        child=root/f'deadline-{ms}ms'
        assert sha256(child/'held_out_cases.json')==bank_hash
        with gzip.open(child/'up-1-cpu-1__priority/agents-2-case-000.json.gz','rt') as stream:
            trace=json.load(stream)
        assert trace['audit']['status']=='passed'
        assert all(e['time']==e['cycle']+ms/1000 for e in trace['remote_events'] if e['event']=='decision_event')
        assert all(t['deadline_at']==t['source_time']+ms/1000 for t in trace['resource_tasks'])
    episodes=(root/'episodes.csv').read_bytes()
    run_deadline_study(cfg,root,model,diagnostic=True,cases=1,counts=(2,),max_steps=2,resume=True)
    assert (root/'episodes.csv').read_bytes()==episodes
    with monkeypatch.context() as patch:
        patch.chdir(tmp_path)
        assert len(report_study('study')) == 24  # The documented CLI accepts relative paths.
    changed=replace(cfg,resource_allocation=replace(cfg.resource_allocation,task_input_bits=200000))
    with pytest.raises(ValueError,match='resume identity'):
        run_deadline_study(changed,root,model,diagnostic=True,cases=1,counts=(2,),max_steps=2,resume=True)
    # Rebuilding comparisons rejects mismatched assets, timing and treatment physics.
    path=root/'deadline-250ms/manifest.json'
    manifest=json.loads(path.read_text())
    import copy
    for field in ('map_sha256','timeline_sha256','model','configuration','conditions'):
        changed=copy.deepcopy(manifest)
        if field == 'model':
            changed[field]['sha256']='wrong'
        elif field == 'configuration':
            changed[field]['resource_allocation']['drone_cycles_per_second'] *= 2
        elif field == 'conditions':
            changed[field][1]['configuration']['resource_allocation']['task_input_bits'] *= 2
        else:
            changed[field]='wrong'
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='identity|configuration'):
            report_study(root)
    path.write_text(json.dumps(manifest))
