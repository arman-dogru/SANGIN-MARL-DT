"""Measured, pre-execution budgets for clean Phase 1 V2 training.

Reads training/validation evidence only. A budget estimate is not a promise of
qualification and cannot override the primary comparison's selection gate.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np

from .phase1_revised import COUNTS, read_json, read_rows, paired_differences, subset
from .records import sha256, write_json

BUDGET_PROTOCOL = 'phase1_v2_measured_budget_v1'


def measure_training(learning, episodes):
    if not learning:
        raise ValueError('Completed learning logs required')
    by_count = defaultdict(lambda: dict(steps=0, seconds=0., tail_steps=0, tail_seconds=0.))
    tail_start = max(1, len(learning)-max(1000, len(learning)//3)+1)
    previous = 0.
    for index, row in enumerate(learning, 1):
        now = row['wall_seconds']
        if row['generated_joint_steps'] != index or now < previous or not math.isfinite(now):
            raise ValueError('Noncontiguous learning steps or invalid runtime accounting')
        count = row['num_agents']
        if count not in COUNTS or row.get('communication_delay_seconds') != 0:
            raise ValueError('Budget evidence must be clean MAPF')
        values = by_count[count]
        delta = now-previous
        values['steps'] += 1
        values['seconds'] += delta
        if index >= tail_start:
            values['tail_steps'] += 1
            values['tail_seconds'] += delta
        previous = now
    ids = {r['case_id'] for r in learning}
    if len({e['case_id'] for e in episodes}) != len(episodes):
        raise ValueError('Duplicate completed episodes')
    out = {}
    for count in COUNTS:
        rows = [r for r in episodes if r['num_agents'] == count]
        values = by_count[count]
        if not rows or not values['seconds'] or not values['steps']:
            raise ValueError('Need completed training episodes in every population')
        if any(r['split'] != 'train' or r['case_id'] not in ids or not 0 < r['steps'] <= 250 for r in rows):
            raise ValueError('Invalid training split/episode evidence')
        rate = values['steps']/values['seconds']
        tail_rate = values['tail_steps']/values['tail_seconds'] if values['tail_seconds'] else rate
        out[count] = dict(**values, completed_episodes=len(rows),
            joint_steps_per_hour=rate*3600, completed_episodes_per_hour=len(rows)/values['seconds']*3600,
            tail_joint_steps_per_hour=tail_rate*3600,
            conservative_seconds_per_joint_step=1/min(rate, tail_rate),
            mean_completed_episode_steps=float(np.mean([r['steps'] for r in rows])),
            failed_250_step_episodes=sum(not r['success'] and r['steps']==250 for r in rows),
            success_rate=sum(r['success'] for r in rows)/len(rows),
            families=dict(Counter(r['family'] for r in rows)))
    return dict(populations=out, training_window_seconds=previous,
        total_joint_steps=len(learning), completed_episodes=len(episodes),
        rate_scope='conditional on that population being active; rates include initialization/checkpoint gaps',
        censoring='unfinished final episode contributes measured steps/time but is not a completed episode')


def project_budget(measurements, evaluation_seconds_per_step, *, minimum_episodes=500,
                   interval_steps=30000, validation_cases_per_population=40,
                   final_cases_per_population=400, compatibility_seconds=60.,
                   startup_seconds=30., shutdown_seconds=60., safety_factor=1.2):
    if not measurements or minimum_episodes < 500 or interval_steps <= 0 or safety_factor < 1:
        raise ValueError('Measured evidence and unchanged coverage floor required')
    populations = {}
    for n in COUNTS:
        rows = [m['populations'][n] for m in measurements]
        if n not in evaluation_seconds_per_step or evaluation_seconds_per_step[n] <= 0:
            raise ValueError('Measured evaluation cost required for every population')
        # Use the slower seed and slower complete/tail step rate. This includes
        # checkpoint gaps rather than pretending they are free.
        cost = max(r['conservative_seconds_per_joint_step'] for r in rows)
        mean_steps = max(r['mean_completed_episode_steps'] for r in rows)
        populations[n] = dict(
            joint_steps_per_hour_range=[min(r['joint_steps_per_hour'] for r in rows),max(r['joint_steps_per_hour'] for r in rows)],
            completed_episodes_per_hour_range=[min(r['completed_episodes_per_hour'] for r in rows),max(r['completed_episodes_per_hour'] for r in rows)],
            conditional_seconds_per_joint_step=cost,
            empirical_target_steps=minimum_episodes*mean_steps,
            all_250_step_target_steps=minimum_episodes*250,
            empirical_training_seconds=minimum_episodes*mean_steps*cost,
            all_250_step_training_seconds=minimum_episodes*250*cost,
            evaluation_seconds_per_joint_step=evaluation_seconds_per_step[n])
    empirical_steps=sum(r['empirical_target_steps'] for r in populations.values())
    worst_steps=sum(r['all_250_step_target_steps'] for r in populations.values())
    empirical_training=sum(r['empirical_training_seconds'] for r in populations.values())
    worst_training=sum(r['all_250_step_training_seconds'] for r in populations.values())
    # Every greedy evaluation is conservatively charged as 250 steps/case,
    # including cases that terminate much earlier in the measured benchmark.
    validation_cost=validation_cases_per_population*250*sum(evaluation_seconds_per_step.values())
    final_cost=final_cases_per_population*250*sum(evaluation_seconds_per_step.values())
    validation_checks=math.ceil(worst_steps/interval_steps)
    empirical_checks=math.ceil(empirical_steps/interval_steps)
    final_reserve=math.ceil((safety_factor*final_cost+compatibility_seconds+30)/300)*300
    empirical_total=empirical_training+empirical_checks*validation_cost+final_reserve+startup_seconds+shutdown_seconds
    worst_total=worst_training+validation_checks*validation_cost+final_reserve+startup_seconds+shutdown_seconds
    requested=max(10800,math.ceil(safety_factor*worst_total/1800)*1800)
    if requested > 86400:
        raise ValueError('Measured estimate exceeds the supported one-day bound; report rather than cap silently')
    return dict(populations=populations,minimum_completed_episodes_per_population=minimum_episodes,
        target_completed_episodes=minimum_episodes*len(COUNTS),
        empirical_target_joint_steps=empirical_steps,all_250_step_target_joint_steps=worst_steps,
        empirical_training_seconds=empirical_training,all_250_step_training_seconds=worst_training,
        periodic_validation_checks=validation_checks,periodic_validation_seconds_each=validation_cost,
        final_evaluation_reserve_seconds=final_reserve,compatibility_seconds=compatibility_seconds,
        startup_seconds=startup_seconds,shutdown_reserve_seconds=shutdown_seconds,
        empirical_total_seconds=empirical_total,all_250_step_total_seconds=worst_total,
        safety_factor=safety_factor,three_hours_sufficient_for_empirical_coverage=safety_factor*empirical_total<=10800,
        three_hours_sufficient_for_conservative_coverage=safety_factor*worst_total<=10800,
        proposed_budget_seconds=requested,
        caveat='Coverage/runtime projection only; success cannot be guaranteed. Per-population rates are conditional, not four simultaneous throughputs. Checkpoint overhead is already in training costs. Evaluation reserve assumes 250 steps for every validation/test case.')


def verify_selection(study):
    """Recompute the declared criterion without changing any comparison files."""
    study=Path(study).resolve()
    setup=read_json(study/'study.json')
    decision=read_json(study/'ablation_comparison.json')
    if setup['spec']['ablation']['joint_steps']!=30000:
        raise ValueError('Full training requires the primary 30000-step comparison')
    dataset=read_json(setup['dataset_path'])
    expected={r['case_id'] for r in subset(dataset['banks']['validation'],8)}
    cohorts={}
    for variant in setup['spec']['variants']:
        rows=[]
        for seed in setup['spec']['ablation']['seeds']:
            folder=study/'ablations'/f"{variant['name']}-seed{seed}"
            status=read_json(folder/'status.json')
            values=read_rows(folder/'final_validation_cases.jsonl')
            if status['status']!='ablation_complete' or status['generated_joint_steps']!=30000 or status['held_out']:
                raise ValueError('Incomplete or test-contaminated comparison')
            if len(values)!=160 or {r['case_id'] for r in values}!=expected or any(r['split']!='validation' for r in values):
                raise ValueError('Incomplete or changed primary paired cases')
            rows.extend(dict(r,training_seed=seed) for r in values)
        cohorts[variant['name']]=rows
    selected=max(cohorts,key=lambda name:np.mean([r['success'] for r in cohorts[name]]))
    baseline=cohorts['original_hp']
    effect=paired_differences(cohorts[selected],baseline)
    gains=[np.mean([r['success'] for r in cohorts[selected] if r['num_agents']==n])-
        np.mean([r['success'] for r in baseline if r['num_agents']==n]) for n in COUNTS]
    comparative=selected!='original_hp' and effect['mean']>=setup['spec']['ablation']['minimum_success_gain'] and effect['ci95'][0]>0 and min(gains)>=-.05
    absolute=min(np.mean([r['success'] for r in cohorts[selected] if r['num_agents']==n]) for n in COUNTS)>=.90
    justified=bool(comparative or absolute)
    if decision['selected']!=selected or bool(decision['full_run_justified'])!=justified:
        raise ValueError('Saved primary selection disagrees with recomputed predeclared criterion')
    return setup,decision,cohorts


def comparison_evidence(study):
    study=Path(study).resolve()
    setup,decision,_=verify_selection(study)
    if not decision['full_run_justified']:
        raise ValueError('Predeclared primary selection criteria were not met; stop')
    selected=decision['selected']
    measurements=[]
    hashes={str(study/'study.json'):sha256(study/'study.json'),
        str(study/'ablation_comparison.json'):sha256(study/'ablation_comparison.json')}
    for seed in setup['spec']['ablation']['seeds']:
        folder=study/'ablations'/f'{selected}-seed{seed}'
        status=read_json(folder/'status.json')
        if status['status']!='ablation_complete' or status['generated_joint_steps']!=30000 or status['held_out']:
            raise ValueError('Incomplete or test-contaminated comparison')
        if [r['num_agents'] for r in status['qualification']]!=list(COUNTS) or any(r['episodes']!=40 for r in status['qualification']):
            raise ValueError('Incomplete primary validation cohorts')
        measurements.append(dict(seed=seed,**measure_training(read_rows(folder/'learning_steps.jsonl'),read_rows(folder/'training.jsonl'))))
        for name in ('manifest.json','status.json','learning_steps.jsonl','training.jsonl','final_validation_cases.jsonl'):
            hashes[str(folder/name)]=sha256(folder/name)
    return setup,decision,measurements,hashes


def save_plan(study, benchmark, *, output=None):
    """Benchmark must be recorded from validation-only frozen-model replays."""
    study=Path(study).resolve()
    setup,decision,measurements,hashes=comparison_evidence(study)
    if benchmark.get('split')!='validation' or benchmark.get('learning_updates')!=0:
        raise ValueError('Only read-only validation benchmarks may inform budget')
    costs={n:float(benchmark['seconds_per_step_p90'][str(n)]) for n in COUNTS}
    projection=project_budget(measurements,costs)
    plan=dict(protocol=BUDGET_PROTOCOL,created_utc=datetime.now(timezone.utc).isoformat(),
        comparison=str(study),source_commit=setup['source_commit'],selected=decision['selected'],
        selected_configuration=next(v for v in setup['spec']['variants'] if v['name']==decision['selected']),
        selection=decision,dataset_path=setup['dataset_path'],dataset_sha256=setup['dataset_sha256'],
        input_sha256=hashes,measurements=measurements,evaluation_benchmark=benchmark,
        projection=projection,phase='recorded_before_execution',
        qualification='Unchanged 95% per population/family, coverage, complete independent final cohorts, compatibility/replay')
    path=Path(output) if output else study/'budget_plan.json'
    if path.exists():
        raise ValueError('Budget plan already recorded; use a new versioned plan instead of overwriting')
    write_json(path,plan)
    return plan
