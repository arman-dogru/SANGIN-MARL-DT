"""Isolation, accounting and paired analysis for the coefficient-only study."""
from dataclasses import replace
import gzip
import importlib.util
import json
from pathlib import Path
import pickle
import tomllib

import pytest

from panama_uav.config import load_config

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


driver=load('run_omega_sensitivity')
report=load('report_omega_sensitivity')
pilot=load('run_uav6_pilot')
reproduction=load('audit_omega_reproduction')
analysis=load('analyse_omega_sensitivity')
independent=load('audit_omega_sensitivity')


def test_five_configurations_change_only_two_coefficients(tmp_path):
    paths=driver.generate_configs(tmp_path,ROOT/'configs/paper_remote_dt_250ms_uav6.toml')
    original=tomllib.loads((ROOT/'configs/paper_remote_dt_250ms_uav6.toml').read_text())
    for name,latency,energy in driver.WEIGHTS:
        actual=tomllib.loads(Path(paths[name]['path']).read_text())
        assert set(driver.config_changes(original,actual)) <= {'resource_allocation.latency_coefficient','resource_allocation.energy_coefficient'}
        assert actual['resource_allocation']['latency_coefficient']==latency
        assert actual['resource_allocation']['energy_coefficient']==energy
    assert len(paths)==5


@pytest.mark.parametrize('field,value', [('task_input_bits',200000),('cloud_cycles_per_second',2e10),('command_cutoff_seconds',.5)])
def test_nonweight_changes_are_rejected(tmp_path,field,value):
    text=(ROOT/'configs/paper_remote_dt_250ms_uav6.toml').read_text()
    line=next(line for line in text.splitlines() if line.startswith(field+' ='))
    path=tmp_path/'invalid.toml';path.write_text(text.replace(line,field+' = '+str(value)))
    with pytest.raises(ValueError,match='unchanged'):driver.generate_configs(tmp_path/'configs',path)


@pytest.mark.parametrize('condition',driver.CONDITIONS)
def test_existing_reference_routes_keep_the_objective_weights(tmp_path,condition):
    base=load_config(ROOT/'configs/paper_remote_dt_250ms_uav6.toml')
    for _,latency,energy in driver.WEIGHTS:
        actual=replace(base,resource_allocation=replace(base.resource_allocation,latency_coefficient=latency,energy_coefficient=energy))
        configured=pilot.execution_config(actual,condition,tmp_path,dict(orbital_offset_seconds=0.))
        assert configured.resource_allocation.latency_coefficient==latency
        assert configured.resource_allocation.energy_coefficient==energy
        assert configured.resource_allocation.cloud_cycles_per_second==1e10
        assert configured.resource_allocation.drone_cycles_per_second==6e9
        assert not configured.resource_allocation.enforce_task_deadlines


def test_trace_archive_preserves_the_complete_dictionary(tmp_path):
    trace=dict(frames=[dict(positions=[[1,2]],actions={0:4})],remote_events=[dict(time=.1)],
               transport_events=[],extra=dict(timestamp=0.,values=(1,2)))
    path=tmp_path/'trace.pkl.gz';driver.write_trace(path,trace)
    with gzip.open(path,'rb') as stream:assert pickle.load(stream)==trace


def test_fingerprints_ignore_host_timings_but_detect_all_executed_changes():
    before=[dict(job_id='x',executed_x=1,observed_latency_seconds=.1,inference_wall_seconds=1.,relaxed_x=.2)]
    host=[dict(before[0],inference_wall_seconds=2.,relaxed_x=.20000001)]
    assert driver.fingerprint(before)==driver.fingerprint(host)
    for field,value in [('executed_x',0),('observed_latency_seconds',.100001)]:
        assert driver.fingerprint(before)!=driver.fingerprint([dict(host[0],**{field:value})])


def test_stable_dictionary_key_order_does_not_hide_event_order():
    assert driver.fingerprint([{'a':1,'b':2}])==driver.fingerprint([{'b':2,'a':1}])
    assert driver.fingerprint([{'a':1},{'a':2}])!=driver.fingerprint([{'a':2},{'a':1}])


@pytest.fixture
def packet():
    case=dict(case_id='a',num_agents=2,seed=1,starts=[[0,0],[0,1]],goals=[[1,0],[1,1]],orbital_offset_seconds=0.,map_sha256='map')
    tasks=[dict(job_id='0',deadline_status='on_time',executed_x=1),dict(job_id='1',deadline_status='late',executed_x=0)]
    row=dict(case,tasks_generated=2,deadline_commands_on_time=1,deadline_commands_late=1,deadline_commands_undelivered=0,
        executed_drone_tasks=1,executed_cloud_tasks=1,deadline_cutoff_wait_tasks=1,command_cutoff_misses=1,
        drone_utilization_seconds=.5,cloud_utilization_seconds=.4,observed_seconds=1.)
    return dict(row=row,tasks=tasks,rounds=[dict(tasks=2)],audit=dict(status='passed')),case


def test_deadline_and_binary_denominators_reconcile(packet):
    value,case=packet;assert report.validate_packet(value,case)


@pytest.mark.parametrize('change', ['identity','partition','binary','duplicate','cpu','replay'])
def test_invalid_episode_is_never_silently_included(packet,change):
    value,case=packet
    if change=='identity':value['row']['seed']=2
    if change=='partition':value['row']['deadline_commands_on_time']=2
    if change=='binary':value['tasks'][0]['executed_x']=.5
    if change=='duplicate':value['tasks'][1]['job_id']='0'
    if change=='cpu':value['row']['drone_utilization_seconds']=1.01
    if change=='replay':value['audit']['status']='failed'
    with pytest.raises(ValueError):report.validate_packet(value,case)


def test_paired_analysis_preserves_cases_and_reports_differences():
    rows=[dict(case_id=str(i),success=bool(i),makespan_seconds=10.,waiting_completed_steps=1,
        task_latency_p95_seconds=.1,deadline_commands_on_time=2,tasks_generated=2,
        modeled_energy_lower_j=2.,drone_busy_seconds=.1,cloud_busy_seconds=.1,observed_seconds=1.,command_cutoff_misses=0) for i in range(2)]
    groups={('W1',r,2):rows for r in report.ROUTES}
    result=report.comparisons(groups)
    assert len(result)==18 and all(r['difference']==0 for r in result)
    groups[('W1','edge',2)]=rows[:1]
    with pytest.raises(ValueError,match='identical'):report.comparisons(groups)


def test_all_375_cases_are_actually_scheduled():
    assert len(driver.WEIGHTS)*len(driver.CONDITIONS)*len(driver.COUNTS)*5==375
    assert driver.COUNTS==(2,4,8,16,32)


def test_fractional_certificate_uses_solver_accuracy_instead_of_minimizer_distance():
    before=dict(status='optimal',objective=.0969927478800188,residual=7.24e-10,is_dcp=True,
                solver_attempts=[dict(tolerance=1e-8)],tasks=[dict(x=.477856)])
    after=dict(before,objective=.09699274768630715,residual=1.38e-11,tasks=[dict(x=.477719)])
    assert reproduction.certificate(before,after)['passed']
    assert not reproduction.certificate(before,dict(after,objective=.097))['passed']
    assert not reproduction.certificate(before,dict(after,residual=1.01e-6))['passed']
    assert not reproduction.certificate(before,dict(after,status='failed'))['passed']
    assert not reproduction.certificate(before,dict(after,is_dcp=False))['passed']


def test_priority_location_statistics_keep_assignment_denominators(tmp_path):
    import csv
    path=tmp_path/'tasks.csv.gz'
    with gzip.open(path,'wt',newline='') as stream:
        fields=['weight_id','route','num_agents','executed_x','weight','cycle']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for x,weight,cycle in [('1',1.,0),('1',3.,1),('0',4.,0),('',2.,1)]:
            writer.writerow(dict(weight_id='W1',route='allocated',num_agents=2,executed_x=x,weight=weight,cycle=cycle))
    rows=analysis.priority_by_location(path)
    uav=next(r for r in rows if r['location']=='uav');cloud=next(r for r in rows if r['location']=='cloud')
    assert uav['tasks']==2 and uav['priority_resource_weight_mean']==2.
    assert uav['first_round_tasks']==1 and uav['first_round_priority_weight_mean']==1.
    assert cloud['tasks']==1 and cloud['priority_resource_weight_mean']==4.


def test_accounting_refresh_retains_only_matching_readback_evidence():
    current=dict(manifest_sha256='manifest',archive_inventory_sha256='inventory',complete_trace_readbacks=[])
    evidence=dict(current,complete_trace_readbacks=['W1/edge/case'])
    assert independent.retain_readbacks(dict(current),evidence)['complete_trace_readbacks']==['W1/edge/case']
    with pytest.raises(ValueError,match='different inventory'):
        independent.retain_readbacks(dict(current),dict(evidence,archive_inventory_sha256='changed'))
