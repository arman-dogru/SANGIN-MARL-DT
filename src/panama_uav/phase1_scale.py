"""Episode-budgeted, fresh PANAMA through 16 robots; historical protocols frozen.

This coverage experiment retains the original learner. The historical V2
comparison and its failed launch-selection criterion remain unchanged.
"""
from collections import Counter
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import json
import math
import os
import shutil
import subprocess
import sys
import time
import tomllib

import numpy as np
import torch

from .config import COMMUNICATION_SCHEMA,CurriculumStage,ExperimentMode,ProjectConfig,load_config
from .models import choose_device
from .phase1 import clean_config,weights_hash
from .phase1_cycles import CycleTracker
from .phase1_data import FAMILIES,case_for,prepare_dataset,scenario_for,verify_dataset
from .phase1_execution import RemoteAlignedCleanMAPFEnvironment
from .phase1_revised import RevisedSession,read_json,read_rows,run_one,summarize,variant_config
from .phase1_stopping import covered,may_open_heldout
from .records import append_jsonl,sha256,write_json
from .scenarios import load_map

PROTOCOL='clean_mapf_phase1_v3_scale16'
METRICS='clean_mapf_metrics_v3_scale16'
POPULATIONS=(2,4,6,8,12,16)


def load_protocol(path):
    with Path(path).open('rb') as stream: spec=tomllib.load(stream)
    if spec.get('protocol')!=PROTOCOL or tuple(spec.get('populations',[]))!=POPULATIONS:
        raise ValueError('New scale protocol requires all six populations')
    if spec['cases_per_family']!=40 or spec['evaluation_maps_per_family']<5:
        raise ValueError('Qualification retains complete independent 40-case families')
    c=spec['curriculum'];e=spec['evaluation']
    if c['minimum_episodes_per_population']!=500 or c['episode_cap_per_population']<500:
        raise ValueError('Do not lower the qualification exposure floor')
    if not 0<c['introduction_every_episodes']<=c['episode_cap_per_population']:
        raise ValueError('Finite episode introduction interval required')
    if c['exploration_clock']!='global':
        raise ValueError('Exploration changes require a separate controlled comparison')
    if e['validation_cases_per_family']!=40 or e['validation_interval_episodes']<=0:
        raise ValueError('Graduation requires full validation cohorts')
    if not 0<e['maximum_wall_seconds']<=259200 or e['safety_factor']<1:
        raise ValueError('Positive explicit bounded runtime required')
    return spec


def validate_dataset(data):
    verify_dataset(data)
    if (data.get('schema')!='clean_mapf_split_v3_scale16' or
        tuple(data.get('populations',[]))!=POPULATIONS or data.get('protocol')!=PROTOCOL):
        raise ValueError('Six-population independent dataset required')
    if {m['canonical'] for m in data['maps']} & set(data['excluded_historical_maps']):
        raise ValueError('Historical map leakage')
    for bank in data['banks'].values():
        if bank and {c['num_agents'] for c in bank}!=set(POPULATIONS):
            raise ValueError('Missing population in evaluation bank')
    return True


def prepare_data(path,spec,project):
    excluded=set();sources={}
    for rel in spec['historical_datasets']:
        source=Path(project)/rel
        previous=read_json(source)
        excluded.update(m['canonical'] for m in previous['maps'])
        excluded.update(previous['reserved_stress_maps'])
        sources[str(source.resolve())]=sha256(source)
    data=prepare_dataset(path,seed=spec['dataset_seed'],train_maps=spec['train_maps_per_family'],
        eval_maps=spec['evaluation_maps_per_family'],eval_cases=spec['cases_per_family'],
        populations=POPULATIONS,excluded_canonicals=excluded)
    data.update(schema='clean_mapf_split_v3_scale16',protocol=PROTOCOL,
        excluded_historical_maps=sorted(excluded),historical_dataset_sha256=sources,
        final_test_policy='closed until current-model complete validation, coverage and compatibility pass')
    validate_dataset(data);write_json(Path(path)/'dataset.json',data)
    return data


class ScaleSession(RevisedSession):
    def __init__(self,config,dataset,seed,run_id,device,curriculum,exploration_clock='global'):
        self.scale_curriculum=dict(curriculum)
        if exploration_clock!='global':raise ValueError('Only unchanged global-clock learner authorized')
        super().__init__(config,dataset,seed,run_id,device)

    def introduced_count(self):
        return min(len(POPULATIONS),1+self.episodes//self.scale_curriculum['introduction_every_episodes'])

    def episode_budget_complete(self):
        cap=self.scale_curriculum['episode_cap_per_population']
        return all(self.population_episodes[n]>=cap for n in POPULATIONS)

    def start_episode(self):
        cap=self.scale_curriculum['episode_cap_per_population']
        available=[n for n in POPULATIONS[:self.introduced_count()] if self.population_episodes[n]<cap]
        if not available:raise ValueError('Episode cap reached; do not start additional training')
        count=min(available,key=lambda n:(self.population_episodes[n],POPULATIONS.index(n)))
        local=self.population_episodes[count]
        self.stage_index=POPULATIONS.index(count);self.stage_episodes=local
        family=FAMILIES[local%len(FAMILIES)]
        options=[m for m in self.dataset['maps'] if m['split']=='train' and m['family']==family]
        episode_seed=self.seed+60_000_000+count*1_000_000+local
        record=options[int(np.random.default_rng(episode_seed).integers(len(options)))]
        self.current_case=case_for(load_map(record['path']),record,count,episode_seed,coordinated=local%2==0)
        self.environment=RemoteAlignedCleanMAPFEnvironment(self.config,ExperimentMode.UAV,
            record['path'],count,250,episode_seed,scenario_for(self.current_case))
        self.observations,_=self.environment.reset();self.action_rng=np.random.default_rng(episode_seed)
        if not self.actor_models:self.actor_models[0]=self._actor(self.latest_published_weights)
        self.deployed_versions={r:max(self.actor_models) for r in self.observations}
        self.episode_reward=0.;self.epoch_totals=Counter();self.cycles=CycleTracker()

    def step(self):
        self.cycles.observe(self.environment,self.observations)
        before={a.id:(a.position,a.ever_reached_goal) for a in self.environment.agents}
        active=self.environment.active_robot_ids
        row,episode=super().step()
        self.cycles.moved(before,self.environment,active)
        row.update(protocol=PROTOCOL,metrics=METRICS,curriculum_stage=self.introduced_count(),
            introduced_populations=list(POPULATIONS[:self.introduced_count()]))
        if episode:episode.update(protocol=PROTOCOL,metrics=METRICS,**self.cycles.record())
        # The inherited resolver records conflicts before resolution; check the
        # physical invariant separately instead of calling them collisions.
        from .remote_control import conflicts
        positions={a.id:a.position for a in self.environment.agents}
        errors=conflicts({r:p for r,(p,_) in before.items()},positions)
        row['actual_collision_errors']=len(errors)
        row.update(repeated_state=self.cycles.period is not None,cycle_period=self.cycles.period)
        if errors:raise ValueError('Physical collision correctness failure')
        return row,episode


def verify_payload(data):
    if (data.get('training_protocol')!=PROTOCOL or data.get('metric_version')!=METRICS or
        data.get('condition')!='clean' or data.get('smoke') or data.get('ablation') or
        not data.get('phase1_qualified') or not data.get('curriculum_completed') or
        data.get('minimum_episodes_per_population')!=500 or
        not covered(data.get('population_episodes',{}),populations=POPULATIONS)):
        return False
    identity=data.get('model_weights_sha256')
    compatibility=data.get('compatibility') or {}
    if (not identity or compatibility.get('model_weights_sha256')!=identity or
        [r.get('num_agents') for r in compatibility.get('checks',[])]!=list(POPULATIONS) or
        not all(r.get('replay_status')=='passed' and r.get('positions_rewards_priorities_equal')
                for r in compatibility.get('checks',[]))):return False
    for key in ('qualification','held_out'):
        groups=data.get(key,[])
        if not may_open_heldout(groups,data['population_episodes'],compatibility,populations=POPULATIONS):return False
    return True


def make_payload(session,manifest,validation=(),test=(),compatibility=None,ablation=False):
    result=dict(communication_schema=COMMUNICATION_SCHEMA,policy_family='panama_0.1_task',
        training_protocol=PROTOCOL,training_run_protocol=PROTOCOL,metric_version=METRICS,condition='clean',
        task_reward=session.config.to_dict()['reward'],model_state_dict={k:v.detach().cpu().clone() for k,v in session.policy.state_dict().items()},
        spatial_shape=(3,15,15),vector_length=session.config.vector_length,dataset_sha256=manifest['dataset_sha256'],
        run_id=session.run_id,seed=session.seed,learner_updates=session.learner_updates,
        generated_joint_steps=session.generated_steps,populations=list(POPULATIONS),
        population_episodes=dict(session.population_episodes),population_steps=dict(session.population_steps),
        minimum_episodes_per_population=500,episode_cap_per_population=manifest['curriculum']['episode_cap_per_population'],
        curriculum_completed=covered(session.population_episodes,populations=POPULATIONS),
        qualification=list(validation),held_out=list(test),compatibility=compatibility,
        model_weights_sha256=weights_hash(session.policy),phase1_qualified=True,smoke=ablation,ablation=ablation,
        source_commit=manifest['source_commit'],budget_plan_sha256=manifest['budget_plan_sha256'],
        configured_budget_seconds=session.config.training.budget_seconds,
        initialization='fresh deterministic weights',execution_contract='remote-aligned-v2',
        observation_contract='unchanged remote_observation with instantaneous fresh state')
    result['phase1_qualified']=verify_payload(result)
    if result['phase1_qualified']:result['qualified_name']='PANAMA-CLEAN'
    return result


def prepare(root,protocol_path,*,episode_authorization,wall_seconds,evaluation_reserve_seconds,
            dataset_path=None,benchmark_path=None):
    begin=time.monotonic();root=Path(root).resolve();project=Path(protocol_path).resolve().parent.parent
    spec=load_protocol(protocol_path)
    cap=spec['curriculum']['episode_cap_per_population']
    if episode_authorization!=cap or episode_authorization!=2500:
        raise ValueError('Episode budget must match the fixed 2500-per-population protocol')
    if not 0<evaluation_reserve_seconds<wall_seconds-60 or wall_seconds>spec['evaluation']['maximum_wall_seconds']:
        raise ValueError('Recorded finite wall cap and final evaluation reserve required')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=project,text=True).strip():
        raise ValueError('Commit source before freezing an experimental run')
    root.mkdir(parents=True,exist_ok=False)
    if dataset_path is None:
        dataset=prepare_data(root/'data',spec,project)
    else:
        dataset=read_json(dataset_path);validate_dataset(dataset)
        write_json(root/'data/dataset.json',dataset)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    shutil.copytree(project/'src/panama_uav',root/'code/panama_uav',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(protocol_path,root/'protocol.toml')
    budget=dict(protocol='phase1_scale_episode_budget_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        budget_basis='2500 completed episodes/population; 15000 total; finite wall cap includes evaluation and checkpoints',
        authorized_episodes_per_population=cap,populations=list(POPULATIONS),
        maximum_completed_episodes=cap*len(POPULATIONS),maximum_joint_training_steps=cap*len(POPULATIONS)*250,
        maximum_worker_wall_seconds=wall_seconds,final_evaluation_reserve_seconds=evaluation_reserve_seconds,
        shutdown_reserve_seconds=60,resume='conservative lease; no budget refund or fresh restart',
        later_communication_budget='same episode cap; blocked until clean qualification and a frozen impairment/control protocol')
    if benchmark_path is not None:
        measured=read_json(benchmark_path)
        if (measured['dataset_sha256']!=sha256(root/'data/dataset.json') or
            measured['code_sha256']!={p.name:sha256(p) for p in (root/'code/panama_uav').glob('*.py')} or
            measured['protocol_sha256']!=sha256(protocol_path) or
            measured['projection']['proposed_budget_seconds']!=wall_seconds or
            measured['projection']['final_evaluation_reserve_seconds']!=evaluation_reserve_seconds):
            raise ValueError('Runtime plan differs from measured dataset/cap')
        budget.update(benchmark_sha256=sha256(benchmark_path),projection=measured['projection'])
        shutil.copy2(benchmark_path,root/'benchmark.json')
    write_json(root/'budget_plan.json',budget)
    config=variant_config(load_config(project/spec['base_config']),dict(name='original_hp',**spec['learner']),wall_seconds)
    config=replace(config,curriculum=tuple(CurriculumStage(name=f'agents-{n}',num_agents=n) for n in POPULATIONS),
        evaluation=replace(config.evaluation,agent_counts=POPULATIONS),
        training=replace(config.training,target_environment_steps=budget['maximum_joint_training_steps']))
    settings=dict(spec['evaluation'],budget_seconds=wall_seconds,evaluation_reserve_seconds=evaluation_reserve_seconds,
        extended_protocol=PROTOCOL,gate_test_on_validation=True,minimum_episodes_per_population=500,
        validation_interval_steps=50000)
    manifest=dict(protocol=PROTOCOL,mode='full',variant='original_hp_episode_coverage',seed=spec['learning_seed'],
        populations=list(POPULATIONS),curriculum=spec['curriculum'],exploration_clock='global',
        configuration=config.to_dict(),spec=dict(full=settings),dataset_path=str(root/'data/dataset.json'),
        dataset_origin=str(Path(dataset_path).resolve()) if dataset_path else str(root/'data/dataset.json'),
        dataset_sha256=sha256(root/'data/dataset.json'),budget_plan_sha256=sha256(root/'budget_plan.json'),
        source_commit=source,code_sha256={p.name:sha256(p) for p in (root/'code/panama_uav').glob('*.py')},
        setup_seconds=time.monotonic()-begin,created_utc=datetime.now(timezone.utc).isoformat(),
        selection='episode-coverage experiment using strongest prior validation settings; historical V2 launch gate remains failed',
        stopping='episode/wall caps; complete-cohort readiness or declared six-check plateau; strict final validation before untouched test')
    write_json(root/'manifest.json',manifest)
    return manifest


def benchmark(root,dataset_path,protocol_path,device_name='cuda',maximum_seconds=180):
    """Fresh disposable 12-episode calibration; validation timing, no test access."""
    from .phase1_extended import audit_case
    from .training import atomic_save
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    spec=load_protocol(protocol_path);project=Path(protocol_path).resolve().parent.parent
    data=read_json(dataset_path);validate_dataset(data)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    code={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=choose_device(device_name);begin=time.monotonic()
    config=variant_config(load_config(project/spec['base_config']),dict(name='original_hp',**spec['learner']),maximum_seconds)
    session=ScaleSession(config,data,spec['learning_seed'],'scale16-calibration',device,
        dict(episode_cap_per_population=2,introduction_every_episodes=1))
    learning=[];episodes=[]
    write_json(root/'manifest.json',dict(protocol='scale16_cost_calibration_v1',source_commit=source,
        code_sha256=code,dataset_sha256=sha256(dataset_path),maximum_seconds=maximum_seconds,
        device=str(device),qualified=False,training_episodes_per_population=2,
        scope='disposable learning/cost smoke; weights are not used to initialize the full run'))
    while not session.episode_budget_complete():
        if time.monotonic()-begin>=maximum_seconds:
            raise ValueError('Calibration budget exhausted; incomplete population timing cannot support launch')
        if session.environment.all_agents_at_goals or session.environment.step_count>=250:session.start_episode()
        row,episode=session.step();row['wall_seconds']=time.monotonic()-begin
        if row['loss'] is not None and not math.isfinite(row['loss']):raise ValueError('Nonfinite calibration loss')
        learning.append(row);append_jsonl(root/'learning_steps.jsonl',row)
        if episode:episodes.append(episode);append_jsonl(root/'training.jsonl',episode)
    measured=measure_training_scale(learning,episodes)
    eval_cost={};timings=[]
    for n in POPULATIONS:
        cases=[c for c in data['banks']['validation'] if c['num_agents']==n]
        selected=[next(c for c in cases if c['family']=='random_obstacles'),
                  next(c for c in cases if c['family']=='intersection')]
        rows=[]
        for case in selected:
            if time.monotonic()-begin>=maximum_seconds:raise ValueError('Calibration evaluation timed out')
            rows.append(audit_case(config,session.policy,case,device))
        eval_cost[n]=max(r['seconds_per_step'] for r in rows);timings+=rows
    checkpoint_begin=time.monotonic();atomic_save(session.checkpoint(),root/'checkpoint.pt')
    checkpoint_seconds=time.monotonic()-checkpoint_begin
    projection=project_runtime(measured,eval_cost,spec,checkpoint_seconds)
    if code!={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}:
        raise ValueError('Calibration source changed while executing')
    result=dict(protocol='scale16_cost_calibration_v1',source_commit=source,code_sha256=code,
        protocol_sha256=sha256(protocol_path),
        dataset_sha256=sha256(dataset_path),training=measured,evaluation_cases=timings,
        checkpoint_seconds=checkpoint_seconds,wall_seconds=time.monotonic()-begin,
        learning_steps=session.generated_steps,completed_training_episodes=len(episodes),
        projection=projection,qualified=False,held_out_cases_evaluated=0)
    write_json(root/'benchmark.json',result);return result


def measure_training_scale(learning,episodes):
    """Conditional cost from all six populations, including startup gaps."""
    by_count={n:dict(steps=0,seconds=0.,completed_episodes=0,completed_episode_steps=0) for n in POPULATIONS}
    previous=0.;ids={r['case_id'] for r in learning}
    if len({r['case_id'] for r in episodes})!=len(episodes):
        raise ValueError('Duplicate completed calibration episodes')
    for index,row in enumerate(learning,1):
        if (row['generated_joint_steps']!=index or not math.isfinite(row['wall_seconds']) or
            row['wall_seconds']<previous or row['communication_delay_seconds']!=0):
            raise ValueError('Noncontiguous or non-clean calibration')
        target=by_count[row['num_agents']];target['steps']+=1
        target['seconds']+=row['wall_seconds']-previous;previous=row['wall_seconds']
    for row in episodes:
        if row['split']!='train' or row['case_id'] not in ids or not 0<row['steps']<=250:
            raise ValueError('Timing requires completed training episodes with matching learning records')
        target=by_count[row['num_agents']];target['completed_episodes']+=1;target['completed_episode_steps']+=row['steps']
    for target in by_count.values():
        if not target['steps'] or not target['seconds'] or target['completed_episodes']<2:
            raise ValueError('Two completed calibration episodes/population required')
        target['joint_steps_per_hour']=3600*target['steps']/target['seconds']
        target['completed_episodes_per_hour']=3600*target['completed_episodes']/target['seconds']
    return by_count


def project_runtime(measured,evaluation_cost,spec,checkpoint_seconds):
    cap=spec['curriculum']['episode_cap_per_population'];evaluation=spec['evaluation']
    training=0.;empirical=0.;populations={}
    for n in POPULATIONS:
        row=measured.get(n,measured.get(str(n)))
        cost=row['seconds']/row['steps'];mean_steps=row['completed_episode_steps']/row['completed_episodes']
        training+=cap*250*cost;empirical+=cap*mean_steps*cost
        populations[n]=dict(**row,all_250_step_training_seconds=cap*250*cost,
            empirical_training_seconds=cap*mean_steps*cost,evaluation_seconds_per_joint_step=evaluation_cost[n])
    check_cost=200*250*sum(evaluation_cost.values())
    checks=math.ceil(cap*len(POPULATIONS)/evaluation['validation_interval_episodes'])
    checkpoint_cost=math.ceil(cap*len(POPULATIONS)*250/5000)*checkpoint_seconds
    reserve=math.ceil((2*check_cost*evaluation['safety_factor']+120)/300)*300
    total=training+checks*check_cost+checkpoint_cost+reserve+120
    proposed=max(10800,math.ceil(total*evaluation['safety_factor']/1800)*1800)
    if proposed>evaluation['maximum_wall_seconds']:
        raise ValueError('Measured coverage exceeds the protocol runtime ceiling; no silent truncation or extension')
    return dict(populations=populations,episode_cap_per_population=cap,total_episode_cap=cap*len(POPULATIONS),
        all_250_step_training_seconds=training,empirical_training_seconds=empirical,
        periodic_validation_checks=checks,periodic_validation_seconds_each=check_cost,
        checkpoint_reserve_seconds=checkpoint_cost,final_evaluation_reserve_seconds=reserve,
        conservative_total_seconds=total,safety_factor=evaluation['safety_factor'],
        proposed_budget_seconds=proposed,
        caveat='Cost forecast from two disposable calibration episodes/population; not a performance guarantee. All failures charged at 250 steps; all complete validation/test cases budgeted.')


def reference_audit(dataset_path,output,timeout_seconds=2.):
    from .mapf_reference import solve_mapf,validate_paths
    data=read_json(dataset_path);validate_dataset(data);rows=[]
    for n in (12,16):
        for family in FAMILIES:
            candidates=[c for c in data['banks']['validation'] if c['num_agents']==n and c['family']==family]
            for designed in (True,False):
                case=next(c for c in candidates if c['designed_coordination']==designed)
                grid=load_map(case['map_path'])
                result=solve_mapf(grid,case['starts'],case['goals'],timeout_seconds=timeout_seconds)
                if result['status']=='feasible' and not validate_paths(grid,case['starts'],case['goals'],result['paths']):
                    raise ValueError('Independent planner returned an invalid witness')
                rows.append(dict(case_id=case['case_id'],num_agents=n,family=family,designed=designed,**result))
    result=dict(protocol='scale16_bounded_reference_v1',dataset_sha256=sha256(dataset_path),
        cases=rows,counts=dict(Counter(r['status'] for r in rows)),test_cases_evaluated=0,
        scope='20 stratified validation references; timeouts/node limits are unresolved, not infeasible; no cases removed')
    write_json(output,result);return result


def launch(root,device='cuda',resume=False,*,monitor_registered=False):
    import psutil
    root=Path(root).resolve();manifest=read_json(root/'manifest.json')
    if not monitor_registered:raise ValueError('Successful hourly automation registration required before launch')
    if manifest['protocol']!=PROTOCOL or sha256(root/'budget_plan.json')!=manifest['budget_plan_sha256']:
        raise ValueError('Frozen budget identity changed')
    if resume and not (root/'checkpoint.pt').exists():raise ValueError('Resume needs a saved checkpoint, never fresh weights')
    if not resume and (root/'checkpoint.pt').exists():raise ValueError('Existing weights require explicit resume')
    owner_path=Path(manifest['dataset_origin']).parent/'qualification_owner.json'
    if owner_path.exists() and read_json(owner_path)['root']!=str(root):
        raise ValueError('The test bank belongs to another full run; use a new independent dataset')
    for proc in psutil.process_iter(['pid','cmdline']):
        try:
            if any(m in (proc.info['cmdline'] or []) for m in
                ('panama_uav.phase1','panama_uav.phase1_revised','panama_uav.phase1_extended','panama_uav.phase1_scale')):
                raise ValueError('Existing PANAMA worker; no overlapping launch')
        except (psutil.AccessDenied,psutil.NoSuchProcess):continue
    command=[sys.executable,'-m','panama_uav.phase1_scale','--worker',str(root),'--device',device]
    if resume:command.append('--resume')
    env=os.environ.copy();env['PYTHONPATH']=str(root/'code');env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    write_json(owner_path,dict(root=str(root),dataset_sha256=manifest['dataset_sha256'],
        source_commit=manifest['source_commit']))
    with (root/'worker.stdout.log').open('a') as out,(root/'worker.stderr.log').open('a') as err:
        p=subprocess.Popen(command,cwd=manifest['configuration']['project_root'],env=env,stdout=out,stderr=err,
            creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS) if os.name=='nt' else 0,
            start_new_session=os.name!='nt')
    record=dict(pid=p.pid,creation_time=psutil.Process(p.pid).create_time(),command=command,
        resume=resume,launched_utc=datetime.now(timezone.utc).isoformat(),hourly_monitor_registered=True)
    write_json(root/'launch.json',record);return record


def worker(root,device='cuda',resume=False):
    root=Path(root);manifest=read_json(root/'manifest.json')
    if (manifest['protocol']!=PROTOCOL or tuple(manifest['populations'])!=POPULATIONS or
        sha256(root/'budget_plan.json')!=manifest['budget_plan_sha256'] or
        {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}!=manifest['code_sha256']):
        raise ValueError('Frozen source or budget changed')
    result=run_one(root,manifest,device,resume)
    postprocessing_start=time.monotonic()
    report(root)
    seconds=time.monotonic()-postprocessing_start
    result.update(report_generation_seconds=seconds,wall_seconds=result['wall_seconds']+seconds)
    write_json(root/'status.json',result)
    lease=read_json(root/'budget_lease.json');lease['charged_seconds']+=seconds
    write_json(root/'budget_lease.json',lease)
    return result


def report(root):
    from .phase1_extended import table
    root=Path(root);manifest=read_json(root/'manifest.json');state=read_json(root/'status.json')
    fields=[]
    for split,key in (('validation','qualification'),('test','held_out')):
        for population in state.get(key,[]):
            for family,group in [('all',population),*population.get('per_family',{}).items()]:
                fields.append(dict(split=split,num_agents=population['num_agents'],family=family,
                    **{k:v for k,v in group.items() if k not in ('per_family','num_agents')}))
    table(root/'per_population_and_scenario.csv',fields)
    cases=[]
    for split in ('validation','test'):
        rows=read_rows(root/f'final_{split}_cases.jsonl');table(root/f'final_{split}_cases.csv',rows);cases+=rows
    table(root/'failures.csv',[r for r in cases if not r['success']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    episodes=read_rows(root/'training.jsonl');history=read_rows(root/'validation.jsonl')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for n in POPULATIONS:
        training=[r for r in episodes if r['num_agents']==n]
        axes[0].plot([r['generated_joint_steps'] for r in training],range(1,len(training)+1),label=str(n))
        points=[(r['generated_joint_steps'],g['success_rate']) for r in history
                for g in r['populations'] if g['num_agents']==n and g['complete']]
        axes[1].plot([p[0] for p in points],[p[1] for p in points],marker='o',label=str(n))
    axes[0].axhline(500,color='black',ls='--');axes[1].axhline(.95,color='black',ls='--')
    axes[0].set(xlabel='Joint training steps',ylabel='Completed episodes per population')
    axes[1].set(xlabel='Joint training steps',ylabel='Complete greedy validation success',ylim=(0,1.02))
    for ax in axes:ax.grid(alpha=.2);ax.legend(title='Robots',fontsize=7)
    fig.savefig(root/'validation_curves.png',dpi=160);plt.close(fig)
    lines=['# Clean MAPF scale-16 results','',f"Status: {state['status']}. Qualified: {state['qualified']}.",'',
        f"Source: {manifest['source_commit']}. Joint steps: {state['generated_joint_steps']}. Runtime: {state['wall_seconds']:.1f} seconds.",
        f"Completed episodes: {state['population_episodes']}. Budget: 2500/population, 15000 total.",'',
        f"Held-out status: {state.get('held_out_skipped_reason','see complete cohort records')}",'',
        'The same current frozen weights must pass all six populations and every family; partial cohorts cannot qualify.','']
    (root/'SCALE_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();worker(args.worker,args.device,args.resume)
