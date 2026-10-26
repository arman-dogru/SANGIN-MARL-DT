"""Paired study dimensions, parameter isolation and outcome denominators."""
import importlib.util
from pathlib import Path
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_cached_fallback as study
import report_cached_fallback as reporting


def test_primary_and_sensitivity_workloads_keep_original_dimensions():
    assert study.core.COUNTS==(2,4,8,16,32)
    assert len(study.core.WEIGHTS)*len(study.core.CONDITIONS)*len(study.core.COUNTS)*5==375
    assert study.VARIANTS=={'primary':('cached_astar',.25),'wait250':('wait',.25),
                           'wait100':('wait',.1),'cached100':('cached_astar',.1)}
    assert len(study.VARIANTS)*3*3==36


def test_separate_experiment_config_changes_only_fallback():
    original=tomllib.loads((ROOT/'configs/paper_remote_dt_250ms_uav6.toml').read_text())
    updated=tomllib.loads((ROOT/'configs/paper_remote_dt_250ms_uav6_cached.toml').read_text())
    assert study.core.config_changes(original,updated)=={'resource_allocation.missed_command_fallback':(None,'cached_astar')}


def row(case,success,energy,misses):
    return dict(case_id=case,success=success,makespan_seconds=10. if success else None,
        waiting_completed_steps=2 if success else None,task_latency_p95_seconds=.2,
        deadline_commands_on_time=2,tasks_generated=3,modeled_energy_lower_j=energy,
        drone_busy_seconds=.1,cloud_busy_seconds=.2,observed_seconds=10.,
        command_cutoff_misses=misses,deadline_cutoff_wait_tasks=misses)


def test_success_uses_all_attempts_but_outcomes_require_common_completions():
    actual=[row('a',True,4,0),row('b',True,6,1),row('c',False,999,3)]
    old=[row('a',True,2,0),row('b',False,999,3),row('c',False,999,3)]
    key=('W1','allocated',2)
    results=reporting.comparisons({key:actual},{key:old})
    success=next(r for r in results if r['metric']=='mission_success_rate')
    assert success['paired_cases']==3 and success['cohort']=='all_attempted_missions'
    assert abs(success['difference']-1/3)<1e-12
    energy=next(r for r in results if r['metric']=='modeled_energy_j')
    assert energy['paired_cases']==1 and energy['paired_case_ids']==['a'] and energy['difference']==2


def test_no_common_completed_case_does_not_create_zero_outcomes():
    key=('W1','edge',32)
    values=reporting.comparisons({key:[row('a',True,1,1)]},{key:[row('a',False,999,3)]})
    energy=next(r for r in values if r['metric']=='modeled_energy_j')
    assert energy['paired_cases']==0 and energy['difference'] is None
    success=next(r for r in values if r['metric']=='mission_success_rate')
    assert success['difference']==1 and success['paired_cases']==1
