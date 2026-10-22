from dataclasses import replace
import gzip
import json

import pytest

from panama_uav.config import load_config
from panama_uav.deadline_study import deadline_config
from panama_uav.priority_evaluation import (BASELINE,PRIMARY,cpu_intervals,level,paired_interval,
    preferential_service,priority_case_rows,reference_weights,report_priority_study)
from panama_uav.records import read_jsonl
from panama_uav.resource_study import run_study
from test_deadline_study import inputs


def test_reference_priority_bins_include_boundaries_and_unknown():
    assert [level(v) for v in (1,1.999,2,2.999,3,4,None)] == ['low','low','middle','middle','high','high','unknown']
    with pytest.raises(ValueError):
        level(4.1)


def test_reference_weight_precision_matches_float_configuration_for_four_robots():
    weights=reference_weights([39.,39.000001,17.000002,17.000003],[False]*4)
    assert weights.dtype.kind=='f'
    assert weights[0]==pytest.approx(1.0000001363636422,abs=1e-12)
    assert weights[1]==1 and weights[2]==4 and 3 < weights[3] < 4


def test_paired_bootstrap_resamples_cases_and_matches_identifiers():
    actual=[dict(case_id=str(i),latency=4+i) for i in range(50)]
    equal=[dict(case_id=str(i),latency=2+i) for i in reversed(range(50))]
    result=paired_interval(actual,equal,'latency',samples=100)
    assert result['estimate']==result['ci95_low']==result['ci95_high']==2
    assert result['pairs']==50 and result['bootstrap_valid']==100
    with pytest.raises(ValueError,match='same unique cases'):
        paired_interval(actual,equal[:-1],'latency')


def test_pooled_rate_contrast_preserves_case_exposure_and_empty_denominators():
    a=[dict(case_id='a',on=9,n=10),dict(case_id='b',on=0,n=100)]
    b=[dict(case_id='a',on=0,n=10),dict(case_id='b',on=0,n=100)]
    result=paired_interval(a,b,'rate',ratio=('on','n'),samples=1000)
    assert result['estimate']==pytest.approx(9/110)
    assert result['ci95_low']==0 and result['ci95_high']==.9
    a[0]['n']=a[1]['n']=0
    with pytest.raises(ValueError,match='numerator <= denominator'):
        paired_interval(a,b,'rate',ratio=('on','n'))
    a[0]['on']=0
    result=paired_interval(a,b,'rate',ratio=('on','n'))
    assert result['estimate'] is None and result['ci95_low'] is None


def test_conditional_metrics_expose_excluded_pairs_and_one_case_has_no_ci():
    result=paired_interval([dict(case_id='a',success_time=None),dict(case_id='b',success_time=10)],
        [dict(case_id='a',success_time=4),dict(case_id='b',success_time=12)],'success_time')
    assert result['estimate']==-2 and result['pairs']==result['excluded_pairs']==1
    assert result['ci95_low'] is None


def test_cpu_wait_is_zero_rate_service_and_utilization_is_time_integrated():
    events=[dict(event='compute_allocation',pool='drone',time=0.,utilization=.5,
        jobs=[dict(job_id='a',cycles_per_second=1),dict(job_id='b',cycles_per_second=0)]),
        dict(event='compute_allocation',pool='drone',time=.5,utilization=1.,
        jobs=[dict(job_id='b',cycles_per_second=2)])]
    jobs,pools=cpu_intervals(events,1.)
    assert jobs['a']['positive_service_seconds']==.5
    assert jobs['b']['zero_rate_wait_seconds']==jobs['b']['positive_service_seconds']==.5
    assert pools['drone']['utilization']==.75 and pools['drone']['mean_resident_jobs']==1.5
    assert pools['cloud']['utilization']==0
    events[0]['utilization']=1.01
    with pytest.raises(ValueError,match='pool capacity'):
        cpu_intervals(events,1.)


def test_priority_stratification_preserves_all_tasks_and_separates_reference_bases():
    row=dict(case_id='a',condition=BASELINE,num_agents=2)
    tasks=[dict(communicated_level='high',physical_level='low',latency_seconds=.1,delivered_at=.1,
        deadline_status='on_time',deadline_met=True,cutoff_wait=False,location='drone'),
        dict(communicated_level='unknown',physical_level='unknown',latency_seconds=None,delivered_at=None,
        deadline_status='undelivered',deadline_met=False,cutoff_wait=True,location='unassigned')]
    groups=priority_case_rows(row,tasks)
    for basis in ('communicated','physical'):
        assert sum(r['tasks'] for r in groups if r['basis']==basis)==2
    assert next(r for r in groups if r['basis']=='communicated' and r['priority_level']=='low')['compliance'] is None
    assert next(r for r in groups if r['basis']=='physical' and r['priority_level']=='low')['compliance']==1


def test_preferential_service_is_a_separate_difference_of_high_low_gaps():
    rows=[]
    for case in ('a','b'):
        for basis in ('communicated','physical'):
            for condition in (PRIMARY,BASELINE):
                for group in ('high','low'):
                    latency=(1 if condition==PRIMARY else 3) if group=='high' else 4
                    compliance=(.9 if condition==PRIMARY else .6) if group=='high' else .5
                    rows.append(dict(case_id=case,basis=basis,condition=condition,num_agents=2,
                        priority_level=group,mean_latency=latency,compliance=compliance))
    _,differences=preferential_service(rows,samples=100)
    latency=next(r for r in differences if r['metric']=='high_minus_low_latency')
    compliance=next(r for r in differences if r['metric']=='high_minus_low_compliance')
    assert latency['estimate']==latency['ci95_low']==latency['ci95_high']==-2
    assert compliance['estimate']==pytest.approx(.3)


def test_acceptance_is_scoped_to_the_explicit_250_ms_configuration():
    default=load_config('configs/paper_remote_dt.toml')
    approved=load_config('configs/paper_remote_dt_250ms.toml')
    assert not default.resource_allocation.remote_acceptance_approved
    assert approved.resource_allocation.remote_acceptance_approved
    assert replace(approved,resource_allocation=replace(approved.resource_allocation,
        remote_acceptance_approved=False))==deadline_config(default,250)


def test_report_audits_real_archives_and_rejects_case_tampering(inputs,tmp_path):
    cfg,checkpoint=inputs
    cfg=deadline_config(cfg,250)
    root=tmp_path/'study'
    run_study(cfg,root,checkpoint,diagnostic=True,cases=1,counts=(2,),max_steps=2)
    differences=report_priority_study(root,checkpoint,workers=2,samples=100)
    output=root/'priority_analysis'
    assert any(r['primary_comparison'] for r in differences)
    assert (output/'latency_distributions.csv').exists()
    assert (output/'priority_summary.csv').exists()
    assert (output/'execution_location_summary.csv').exists()
    assert (output/'paired_execution_location_differences.csv').exists()
    assert (output/'overall_summary.csv').exists()
    with (output/'priority_summary.csv').open() as stream:
        levels=list(__import__('csv').DictReader(stream))
    assert len(levels)==6*2*4
    assert all(r['deadline_compliance']=='' for r in levels if r['tasks']=='0')
    with (output/'execution_location_summary.csv').open() as stream:
        assert len(list(__import__('csv').DictReader(stream)))==6*3
    from panama_uav.priority_locations import report_locations
    location_differences=report_locations(output,samples=100)
    assert any(r['location']=='cloud' and r['metric']=='deadline_compliance' for r in location_differences)
    assert all(r['estimate'] is None for r in location_differences if r['location']=='unassigned')
    assert (output/'latency_cdf_communicated.png').exists()
    validation=json.loads((output/'validation.json').read_text())
    assert validation['episodes']==6 and validation['actual_model_actions_verified']>0
    assert validation['cutoff_oracle_actions_verified']>0
    tasks=list(__import__('csv').DictReader((output/'tasks_by_priority.csv').open()))
    assert len(tasks)>0 and all(t['communicated_level'] in ('low','middle','high','unknown') for t in tasks)
    path=root/PRIMARY/'agents-2-case-000.json.gz'
    with gzip.open(path,'rt') as stream:
        trace=json.load(stream)
    trace['case']['seed']+=1
    with gzip.open(path,'wt') as stream:
        json.dump(trace,stream)
    with pytest.raises(ValueError,match='case mismatch'):
        report_priority_study(root,checkpoint,workers=1,samples=100)
    trace['case']['seed']-=1
    shadow=next(e for e in trace['oracle_events'] if e['event']=='allocation_shadow')
    shadow['physical_weights']['0']=1 if shadow['physical_weights']['0']!=1 else 4
    with gzip.open(path,'wt') as stream:
        json.dump(trace,stream)
    with pytest.raises(ValueError,match='Physical priority reference'):
        report_priority_study(root,checkpoint,workers=1,samples=100)
