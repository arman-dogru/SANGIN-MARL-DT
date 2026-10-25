"""Completed-cohort filtering, pairing and local-only success figure exports."""
import importlib.util
from pathlib import Path
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('completed_report',ROOT/'scripts/publish_omega_completed.py')
completed=importlib.util.module_from_spec(spec)
spec.loader.exec_module(completed)


def row(case, success, energy):
    return dict(case_id=case,success=success,makespan_seconds=10.,waiting_completed_steps=2,
        task_latency_p95_seconds=.1,deadline_commands_on_time=2,tasks_generated=2,
        modeled_energy_lower_j=energy,drone_busy_seconds=.1,cloud_busy_seconds=.2,
        observed_seconds=10.,command_cutoff_misses=0)


def test_paired_completed_comparisons_exclude_failed_cases_on_either_side():
    groups={('W1','allocated',2):[row('a',True,3),row('b',True,999),row('c',False,999)],
            ('W1','edge',2):[row('a',True,2),row('b',False,999),row('c',True,999)],
            ('W1','cloud',2):[]}
    result=completed.completed_comparisons(groups)
    edge=next(r for r in result if r['reference_route']=='edge' and r['metric']=='modeled_energy_j')
    assert edge['paired_cases']==1 and edge['paired_case_ids']==['a']
    assert edge['difference']==edge['ci95_low']==edge['ci95_high']==1
    cloud=next(r for r in result if r['reference_route']=='cloud' and r['metric']=='modeled_energy_j')
    assert cloud['paired_cases']==0 and cloud['difference'] is None
    assert all(r['metric']!='mission_success_rate' for r in result)


def test_empty_completed_cell_has_missing_outcomes_instead_of_zero():
    identity=dict(weight_id='W1',latency_coefficient=1.,energy_coefficient=1.,
                  condition='all-uav-6ghz',route='edge',num_agents=32)
    template=dict(cases=5,successful_missions=0,success_rate=0.,tasks=40000,
        mean_episode_energy_j=50.,observed_latency_p95_seconds=.6,deadline_miss_percent=100.,
        drone_cpu_utilization_percent=53.,cloud_cpu_utilization_percent=0.,uav_percent=100.,cloud_percent=0.)
    result=completed.completed_summary([],[],[],identity,template,5)
    assert result['included_cases']==0 and result['excluded_cases']==5 and result['tasks']==0
    assert result['analysis_status']=='no_completed_missions'
    for key in ('mean_episode_energy_j','observed_latency_p95_seconds','deadline_miss_percent',
                'drone_cpu_utilization_percent','cloud_cpu_utilization_percent','uav_percent','cloud_percent'):
        assert result[key] is None
    assert 'success_rate' not in result


def test_unsuccessful_mission_cannot_enter_completed_summary():
    with pytest.raises(ValueError,match='Unsuccessful'):
        completed.completed_summary([row('a',False,999)],[],[],{}, {},1)


def test_duplicate_completed_pair_ids_are_rejected():
    with pytest.raises(ValueError,match='Duplicate'):
        completed.completed_comparisons({('W1','allocated',2):[row('a',True,1)]*2,
                                         ('W1','edge',2):[row('a',True,1)]})


def test_local_success_plots_are_never_packaged(tmp_path):
    (tmp_path/'figures/06_mission_success').mkdir(parents=True)
    (tmp_path/'figures/02_latency').mkdir()
    (tmp_path/'figures/06_mission_success/mission_success.png').write_bytes(b'private')
    (tmp_path/'figures/02_latency/e2e_p95.png').write_bytes(b'public')
    (tmp_path/'summary.csv').write_text('value\n1\n')
    completed.seal_bundle(tmp_path)
    for name in ('charts.zip','results.zip'):
        with zipfile.ZipFile(tmp_path/name) as archive:
            assert not any('mission_success' in item for item in archive.namelist())
    assert 'mission_success' not in (tmp_path/'files.csv').read_text()
    assert (tmp_path/'figures/06_mission_success/mission_success.png').read_bytes()==b'private'


def test_legacy_all_cohort_export_cannot_replace_repository_results(tmp_path):
    import audit_omega_sensitivity
    with pytest.raises(ValueError,match='Full-cohort exports stay local'):
        audit_omega_sensitivity.export(tmp_path,ROOT/'results/omega_sensitivity')


def test_source_audits_and_frozen_code_are_retained_without_changes(tmp_path):
    source,destination=tmp_path/'source',tmp_path/'publication'
    paths=['independent_audit.json','readback_audit.json','status.json','validation_revision.json',
           'analysis/frozen_harness/code.py','analysis/postprocessing/code.py',
           'analysis/parameter_validation.json','analysis/postprocessing_provenance.json']
    for name in paths:
        path=source/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(('original evidence: '+name).encode())
    completed.retain_source_evidence(source,destination)
    for name in paths:
        assert (source/name).read_bytes()==(destination/name).read_bytes()
