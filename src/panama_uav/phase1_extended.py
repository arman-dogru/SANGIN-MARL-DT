"""Validation-selected, measured-budget extension of the unchanged V2 learner."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .config import ExperimentMode,ProjectConfig
from .models import choose_device,select_actions
from .phase1 import weights_hash
from .phase1_budget import comparison_evidence,save_plan,verify_selection
from .phase1_cycles import CycleTracker
from .phase1_data import FAMILIES,scenario_for
from .phase1_environment import epoch_metrics
from .phase1_execution import RemoteAlignedCleanMAPFEnvironment
from .phase1_revised import COUNTS,read_json,read_rows,run_one,variant_config
from .records import sha256,write_json
from .training import load_model

RUN_PROTOCOL='clean_mapf_phase1_v2_extended_v1'


def table(path,records):
    if not records:
        return
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in records for k in r)))
        writer.writeheader();writer.writerows(records)


def audit_case(config,model,case,device):
    """Read-only greedy validation replay with timings and exact cycle checks."""
    if case['split']!='validation':
        raise ValueError('Budget/model diagnosis must not consume test cases')
    start=time.monotonic()
    env=RemoteAlignedCleanMAPFEnvironment(config,ExperimentMode.UAV,case['map_path'],
        case['num_agents'],250,case['seed'],scenario_for(case))
    observations,_=env.reset()
    tracker=CycleTracker();totals=Counter()
    while True:
        active=set(env.active_robot_ids)
        tracker.observe(env,observations)
        proposed=select_actions({0:model},{r:0 for r in sorted(active)},
            {r:observations[r] for r in sorted(active)},0.,np.random.default_rng(0),device)
        actions=env.prepare_actions(proposed).executed
        totals.update(epoch_metrics(env,actions))
        before={a.id:(a.position,a.ever_reached_goal) for a in env.agents}
        result=env.step(proposed)
        tracker.moved(before,env,active)
        totals['blocked_move_steps']+=sum(actions[a.id]!=4 and before[a.id][0]==a.position for a in env.agents if a.id in active)
        observations=result.observations
        env.communication.transport.log.clear()
        if result.terminated or result.truncated:
            seconds=time.monotonic()-start
            return dict(case,steps=env.step_count,success=result.terminated,
                final_positions=[list(a.position) for a in env.agents],
                ever_completed_robots=sum(a.ever_reached_goal for a in env.agents),
                all_visited_without_completion=all(a.ever_reached_goal for a in env.agents) and not result.terminated,
                wall_seconds=seconds,seconds_per_step=seconds/env.step_count,
                model_weights_sha256=weights_hash(model),**totals,**tracker.record())


def bootstrap_success(records,samples=2000):
    seeds=sorted({r['training_seed'] for r in records})
    maps=sorted({r['map_id'] for r in records})
    sums=np.array([[sum(float(r['success']) for r in records if r['training_seed']==s and r['map_id']==m) for m in maps] for s in seeds])
    sizes=np.array([[sum(r['training_seed']==s and r['map_id']==m for r in records) for m in maps] for s in seeds])
    rng=np.random.default_rng(42)
    values=[]
    for _ in range(samples):
        s=rng.integers(len(seeds),size=len(seeds));m=rng.integers(len(maps),size=len(maps))
        values.append(sums[s[:,None],m[None,:]].sum()/sizes[s[:,None],m[None,:]].sum())
    return np.quantile(values,[.025,.975]).tolist()


def analyse_comparison(study):
    from .phase1_budget import measure_training
    study=Path(study).resolve()
    setup,decision,cohorts=verify_selection(study)
    rows=[];stability=[];throughput=[]
    for name,values in cohorts.items():
        for n in COUNTS:
            for family in ('all',*FAMILIES):
                group=[r for r in values if r['num_agents']==n and (family=='all' or r['family']==family)]
                low,high=bootstrap_success(group)
                rows.append(dict(variant=name,num_agents=n,family=family,episodes=len(group),
                    success_rate=float(np.mean([r['success'] for r in group])),success_ci95_low=low,success_ci95_high=high,
                    failures=sum(not r['success'] for r in group),mean_capped_makespan=float(np.mean([r['steps'] for r in group])),
                    mean_active_waits=float(np.mean([r['waiting_steps'] for r in group])),
                    mean_blocked_moves=float(np.mean([r.get('blocked_move_steps',0) for r in group])),
                    relevant_pairs=sum(r['relevant_pairs'] for r in group),directed_pairs=sum(r['directed_pairs'] for r in group)))
        for seed in setup['spec']['ablation']['seeds']:
            folder=study/'ablations'/f'{name}-seed{seed}'
            learning=read_rows(folder/'learning_steps.jsonl');episodes=read_rows(folder/'training.jsonl')
            measured=measure_training(learning,episodes)
            throughput.extend(dict(variant=name,seed=seed,num_agents=n,**data) for n,data in measured['populations'].items())
            losses=[r['loss'] for r in learning if r['loss'] is not None]
            probes=[r for r in learning if 'q_finite' in r]
            active=sum(r['active_replay_transitions'] for r in learning)
            stability.append(dict(variant=name,seed=seed,learning_steps=len(learning),episodes=len(episodes),
                nonfinite_losses=sum(not np.isfinite(x) for x in losses),
                mean_loss=float(np.mean(losses)),loss_p95=float(np.quantile(losses,.95)),
                nonfinite_q_probes=sum(not r['q_finite'] for r in probes),
                max_abs_q=max((r['q_abs_max'] for r in probes),default=None),
                max_gradient_after_clip=max((r.get('gradient_l2_after_clip',0) for r in probes),default=0),
                epsilon_start=learning[0]['epsilon'],epsilon_end=learning[-1]['epsilon'],
                active_robot_epochs=active,active_waits=sum(r['waiting_steps'] for r in learning),
                active_no_motion=sum(r['waiting_steps']+r.get('blocked_move_steps',0) for r in learning)))
    root=study/'extended_review';root.mkdir(exist_ok=True)
    table(root/'population_family.csv',rows);table(root/'training_throughput.csv',throughput)
    table(root/'stability.csv',stability)
    result=dict(protocol=RUN_PROTOCOL,selection=decision,held_out_cases_evaluated=0,
        stability=stability,scope='All completed primary validation cases; two learning seeds; paired seed/map intervals; no multiplicity correction')
    write_json(root/'comparison_review.json',result)
    return result


def diagnose_models(study,device_name='cuda'):
    """At most one failure and one control per population/model/seed."""
    study=Path(study).resolve();setup,_,_=verify_selection(study)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=choose_device(device_name)
    bank={c['case_id']:c for c in read_json(setup['dataset_path'])['banks']['validation']}
    results=[]
    for variant in setup['spec']['variants']:
        for seed in setup['spec']['ablation']['seeds']:
            source=study/'ablations'/f"{variant['name']}-seed{seed}"
            config=ProjectConfig.from_dict(read_json(source/'manifest.json')['configuration'])
            model,_=load_model(source/'final_model.pt',config,device)
            rows=read_rows(source/'final_validation_cases.jsonl')
            for n in COUNTS:
                for success in (False,True):
                    candidates=sorted((r for r in rows if r['num_agents']==n and r['success']==success),key=lambda r:r['case_id'])
                    if not candidates:continue
                    prior=candidates[0]
                    audited=audit_case(config,model,bank[prior['case_id']],device)
                    if any(audited[k]!=prior[k] for k in ('success','steps','final_positions','waiting_steps','vertex_conflicts','edge_conflicts','relevant_pairs','directed_pairs')):
                        raise ValueError('Comparison diagnostic replay failed integrity check')
                    results.append(dict(variant=variant['name'],training_seed=seed,**audited))
    target=study/'extended_review';target.mkdir(exist_ok=True)
    table(target/'failure_replays.csv',results)
    write_json(target/'failure_replays.json',dict(protocol=RUN_PROTOCOL,cases=results,
        learning_updates=0,split='validation',scope='Outcome-stratified bounded diagnostic; at most64 replays; not a new performance estimate'))
    return results


def benchmark_and_plan(study,device_name='cuda'):
    study=Path(study).resolve()
    setup,decision,_,_=comparison_evidence(study)
    if (study/'budget_plan.json').exists():
        plan=read_json(study/'budget_plan.json')
        write_budget_report(study,plan)
        return plan
    import psutil
    for proc in psutil.process_iter(['cmdline']):
        try:
            args=proc.info['cmdline'] or []
            if 'panama_uav.phase1_revised' in args and str(study) in args:
                raise ValueError('Comparison worker must exit before throughput benchmark')
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    seed=setup['spec']['ablation']['seeds'][0]
    source=study/'ablations'/f"{decision['selected']}-seed{seed}"
    config=ProjectConfig.from_dict(read_json(source/'manifest.json')['configuration'])
    device=choose_device(device_name)
    model,_=load_model(source/'final_model.pt',config,device)
    cohort=read_rows(source/'final_validation_cases.jsonl')
    cases=[]
    for n in COUNTS:
        group=[r for r in cohort if r['num_agents']==n]
        indices=np.linspace(0,len(group)-1,8,dtype=int)
        cases.extend(group[i] for i in indices)
    # Strip prior outcome fields; replay uses only immutable case inputs.
    bank={r['case_id']:r for r in read_json(setup['dataset_path'])['banks']['validation']}
    results=[audit_case(config,model,bank[c['case_id']],device) for c in cases]
    prior={r['case_id']:r for r in cohort}
    for r in results:
        if any(r[k]!=prior[r['case_id']][k] for k in ('success','steps','final_positions','waiting_steps','vertex_conflicts','edge_conflicts','relevant_pairs','directed_pairs')):
            raise ValueError('Selected model validation replay failed integrity check')
    benchmark=dict(protocol='phase1_validation_timing_v1',split='validation',learning_updates=0,
        source_model=str(source/'final_model.pt'),model_weights_sha256=weights_hash(model),
        device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        seconds_per_step_p90={str(n):float(np.quantile([r['seconds_per_step'] for r in results if r['num_agents']==n],.9)) for n in COUNTS},
        cases=results,scope='32 deterministic validation replays; outcome-independent spaced sampling; conditional timing/cycle diagnostic, not a new success estimate')
    review=study/'extended_review';review.mkdir(exist_ok=True)
    write_json(review/'validation_benchmark.json',benchmark)
    table(review/'validation_cycles.csv',results)
    plan=save_plan(study,benchmark)
    write_budget_report(study,plan)
    return plan


def write_budget_report(study,plan):
    p=plan['projection']
    lines=['# Measured full-training budget','',
        f"Recorded before execution: {plan['created_utc']}. Selected: **{plan['selected']}**. "
        'The primary comparison criterion was independently recomputed from validation only.', '',
        '| Robots | Joint steps/hour (conditional range) | Completed episodes/hour (conditional range) | 500-episode all-250-step training hours |',
        '| --- | --- | --- | ---: |']
    for n in COUNTS:
        row=p['populations'].get(n,p['populations'].get(str(n)))
        step='–'.join(f'{v:,.0f}' for v in row['joint_steps_per_hour_range'])
        eps='–'.join(f'{v:,.1f}' for v in row['completed_episodes_per_hour_range'])
        lines.append(f"| {n} | {step} | {eps} | {row['all_250_step_training_seconds']/3600:.2f} |")
    lines+=['',f"Empirical coverage projection including checks/final evaluation: {p['empirical_total_seconds']/3600:.2f} h. "
        f"Conservative all-250-step projection: {p['all_250_step_total_seconds']/3600:.2f} h before the {p['safety_factor']:.0%} multiplier. "
        f"Proposed explicit full-run budget: **{p['proposed_budget_seconds']/3600:.1f} h**. "
        f"Final evaluation reserve: {p['final_evaluation_reserve_seconds']/60:.0f} min. "
        f"Three hours sufficient under the conservative projection: **{p['three_hours_sufficient_for_conservative_coverage']}**.",'',
        p['caveat'],'','No active budget is extended. A fresh versioned run uses this recorded plan; '
        'held-out testing remains closed until strict final validation, coverage and compatibility pass. '
        'Validation readiness or sustained stagnation may end learning before the cap.','']
    (Path(study)/'BUDGET.md').write_text('\n'.join(lines),encoding='utf-8')


def prepare_extended(root,study):
    preparation_start=time.monotonic()
    root=Path(root).resolve();study=Path(study).resolve()
    setup,decision,_,_=comparison_evidence(study)
    plan=read_json(study/'budget_plan.json')
    if plan['phase']!='recorded_before_execution' or plan['selected']!=decision['selected']:
        raise ValueError('A recorded matching budget plan is required before a fresh full run')
    if any(sha256(path)!=value for path,value in plan['input_sha256'].items()):
        raise ValueError('Budget evidence changed since recording')
    root.mkdir(parents=True,exist_ok=False)
    project=Path(setup['base_configuration']['project_root'])
    spec=deepcopy(setup['spec'])
    p=plan['projection']
    spec['full'].update(budget_seconds=p['proposed_budget_seconds'],evaluation_reserve_seconds=p['final_evaluation_reserve_seconds'],
        extended_protocol=RUN_PROTOCOL,gate_test_on_validation=True,
        plateau_patience=6,plateau_min_delta=.01,plateau_min_steps=90000)
    variant=next(v for v in spec['variants'] if v['name']==decision['selected'])
    config=variant_config(ProjectConfig.from_dict(setup['base_configuration']),variant,p['proposed_budget_seconds'])
    shutil.copytree(project/'src/panama_uav',root/'code/panama_uav',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(study/'budget_plan.json',root/'budget_plan.json')
    manifest=dict(protocol=RUN_PROTOCOL,mode='full',variant=decision['selected'],seed=spec['ablation']['seeds'][0],
        spec=spec,configuration=config.to_dict(),dataset_path=setup['dataset_path'],dataset_sha256=setup['dataset_sha256'],
        comparison=str(study),comparison_sha256=sha256(study/'ablation_comparison.json'),
        budget_plan_sha256=sha256(root/'budget_plan.json'),setup_seconds=time.monotonic()-preparation_start,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
        code_sha256={p.name:sha256(p) for p in (root/'code/panama_uav').glob('*.py')},
        created_utc=datetime.now(timezone.utc).isoformat(),initialization='fresh deterministic weights; no historical or ablation checkpoint loaded',
        stopping='current weights; coverage then validation readiness or 6-check stagnation; fixed measured budget cap; strict final validation before test')
    write_json(root/'manifest.json',manifest)
    return manifest


def launch(root,device='cuda',resume=False):
    import psutil
    root=Path(root).resolve()
    manifest=read_json(root/'manifest.json')
    if sha256(root/'budget_plan.json')!=manifest['budget_plan_sha256']:
        raise ValueError('Recorded full budget changed')
    if (root/'checkpoint.pt').exists() and not resume:
        raise ValueError('Existing checkpoints require explicit resume; never restart weights')
    if resume and not (root/'checkpoint.pt').exists():
        raise ValueError('No saved checkpoint to resume; refuse a fresh restart')
    comparison=Path(manifest['comparison'])
    registration=comparison/'extended_launch.json'
    if registration.exists() and read_json(registration)['root']!=str(root):
        raise ValueError('An extended run already owns this comparison/test bank; no second fresh launch')
    for proc in psutil.process_iter(['cmdline']):
        try:
            cmd=proc.info['cmdline'] or []
            if any(module in cmd for module in ('panama_uav.phase1','panama_uav.phase1_revised','panama_uav.phase1_extended')):
                raise ValueError('Existing PANAMA training worker; refusing overlapping launch')
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    project=Path(manifest['configuration']['project_root'])
    env=os.environ.copy();env['PYTHONPATH']=str(root/'code');env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    command=[sys.executable,'-m','panama_uav.phase1_extended','--root',str(root),'--device',device]
    if resume:command.append('--resume')
    with (root/'worker.stdout.log').open('a') as out,(root/'worker.stderr.log').open('a') as err:
        process=subprocess.Popen(command,cwd=project,env=env,stdout=out,stderr=err,
            creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS) if os.name=='nt' else 0,
            start_new_session=os.name!='nt')
    record=dict(pid=process.pid,creation_time=psutil.Process(process.pid).create_time(),command=command,resume=resume)
    write_json(root/'launch.json',record)
    write_json(registration,dict(root=str(root),launch=record,
        budget_plan_sha256=manifest['budget_plan_sha256'],created_utc=datetime.now(timezone.utc).isoformat()))
    return record


def run(root,device='cuda',resume=False):
    root=Path(root).resolve();manifest=read_json(root/'manifest.json')
    if manifest['protocol']!=RUN_PROTOCOL or sha256(root/'budget_plan.json')!=manifest['budget_plan_sha256']:
        raise ValueError('Extended run/budget identity changed')
    if {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}!=manifest['code_sha256']:
        raise ValueError('Frozen extended source changed')
    result=run_one(root,manifest,device,resume)
    if result['mode']=='complete':
        extended_report(root)
    return result


def extended_report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    state=read_json(root/'status.json');manifest=read_json(root/'manifest.json')
    plan=read_json(root/'budget_plan.json')
    actual=read_json(root/'budget_lease.json')['charged_seconds']
    training=read_rows(root/'training.jsonl');validation=read_rows(root/'validation.jsonl')
    cases=[]
    for split in ('validation','test'):
        records=read_rows(root/f'final_{split}_cases.jsonl')
        table(root/f'final_{split}_cases.csv',records)
        cases.extend(records)
    table(root/'failures.csv',[r for r in cases if not r['success']])
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for n in COUNTS:
        cohort=[r for r in training if r['num_agents']==n]
        axes[0].plot([r['generated_joint_steps'] for r in cohort],np.arange(1,len(cohort)+1),label=f'{n} robots')
        points=[(r['generated_joint_steps'],next((g['success_rate'] for g in r['populations'] if g['num_agents']==n),None)) for r in validation]
        if points:axes[1].plot([x for x,y in points],[y for x,y in points],marker='o',label=f'{n} robots')
    axes[0].axhline(500,color='black',ls='--',label='Coverage floor')
    axes[1].axhline(.95,color='black',ls='--',label='Success target')
    axes[0].set(xlabel='Joint learning steps',ylabel='Completed training episodes')
    axes[1].set(xlabel='Joint learning steps',ylabel='Greedy progress-validation success',ylim=(0,1.02))
    for axis in axes:axis.grid(alpha=.2);axis.legend(fontsize=7)
    fig.savefig(root/'validation_curves.png',dpi=160);plt.close(fig)
    lines=['# Extended clean PANAMA results','',
        f"Status: **{state['status']}**. Qualified: **{state['qualified']}**. Termination: {state.get('termination_reason')}.", '',
        f"Selected configuration: `{manifest['variant']}`. Source commit: `{manifest['source_commit']}`. "
        f"Run protocol: `{RUN_PROTOCOL}`. Budget plan SHA256: `{manifest['budget_plan_sha256']}`.", '',
        f"Predeclared budget: {manifest['configuration']['training']['budget_seconds']/3600:.2f} h; "
        f"actual initialization/learning/validation/checkpoint/final-evaluation time: {actual/3600:.2f} h. "
        f"Joint steps: {state['generated_joint_steps']:,}; completed episodes: {state['population_episodes']}. "
        f"Empirical coverage projection: {plan['projection']['empirical_total_seconds']/3600:.2f} h; "
        f"all-250-step projection: {plan['projection']['all_250_step_total_seconds']/3600:.2f} h before safety margin.", '',
        '| Split | Robots | Cases | Success | 95% map-cluster CI | Mean capped steps | Mean active WAITs | Relevant/direct pair epochs |',
        '| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |']
    for split,key in (('validation','qualification'),('test','held_out')):
        for r in state[key]:
            ci=r.get('map_cluster_ci95',{}).get('success')
            label=f'{ci[0]:.1%}–{ci[1]:.1%}' if ci else 'unavailable'
            rate=f"{r['success_rate']:.1%}" if r['success_rate'] is not None else 'unavailable'
            lines.append(f"| {split} | {r['num_agents']} | {r['episodes']} | {rate} | {label} | "
                f"{r['mean_capped_makespan']} | {r['mean_waiting_steps']} | {r['relevant_pairs']}/{r['directed_pairs']} |")
    lines+=['','| Split | Robots | Random | Intersection | Corridor | Bottleneck | Warehouse |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split,key in (('validation','qualification'),('test','held_out')):
        for r in state[key]:
            values=[r['per_family'][f]['success_rate'] for f in FAMILIES]
            labels=' | '.join(f'{v:.1%}' if v is not None else 'unavailable' for v in values)
            lines.append(f"| {split} | {r['num_agents']} | {labels} |")
    failed=[r for r in cases if not r['success']]
    lines+=['',f"Final failures: {len(failed)}; repeated-state failures: {sum(r.get('repeated_state',False) for r in failed)}; "
        f"all-visited-but-unfinished failures: {sum(r.get('all_visited_without_completion',False) for r in failed)}. "
        'These counts cover completed final evaluations only. Goal departures in this protocol are local-controller yielding, '
        'not ignored completed-robot DQN proposals. Conflicts count pre-resolution attempts; WAITs exclude blocked non-WAIT actions.', '',
        f"Held-out status: {state.get('held_out_skipped_reason') or ('evaluated' if state['held_out'] else 'not evaluated')}.", '',
        'Qualification requires all unchanged coverage, per-population/per-family 95%, exposure, complete final cohorts, '
        'and remote execution/replay gates. Validation progress or a historical snapshot cannot confer qualification. '
        'Compatibility is bounded multi-step equivalence and event replay, not a replay of every mission. '
        'No communication impairments or communication-aware learning were used. Checkpoints and results remain local.', '']
    (root/'EXTENDED_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args.root,args.device,args.resume)
