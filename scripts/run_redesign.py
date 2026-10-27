"""Versioned movement-frequency study with audited CPU scheduling."""
import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict,replace
from datetime import datetime,timezone
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from panama_uav.config import FIXED_MAP,ProjectConfig
from panama_uav.records import sha256
from panama_uav.redesign import TREATMENTS,settings_for,configured
import run_omega_sensitivity as core

PROTOCOL='dt_bundle_slow_pipeline_v2'
DEFAULT_SOURCE=Path('output/omega_sensitivity/omega-20261025-01')
DEFAULT_ROOT=Path('output/redesign/redesign-20261026-01')
FORBIDDEN=('allocation.py','compute.py','paper_radio.py','radio.py','transport.py','environment.py',
           'models.py','remote_state.py','inference_controls.py')
THREADS=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')


def job_id(job):
    return '/'.join(str(job[k]) for k in ('treatment','weight_id','route','num_agents','case_id'))


def output_path(root,job,namespace='primary'):
    return Path(root)/namespace/job_id(job)/'result.json'


def committed_output(path):
    path=Path(path);marker=path.with_name('complete.json')
    if not marker.exists():return False
    if not path.exists() or sha256(path)!=core.read(marker)['sha256']:
        raise ValueError('Committed episode transaction is missing or changed: '+str(path))
    packet=core.read(path)
    if packet['audit']['status']!='passed':raise ValueError('Committed episode is not audited')
    return True


def stable_signature(packet):
    excluded=core.HOST_FIELDS|core.RELAXED_FIELDS|{'inference_wall_seconds','bundle_inference_wall_seconds','allocation_solver_wall_seconds'}
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in excluded}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    return core.fingerprint([clean(packet['row']),clean(packet['tasks'])])


def event_export(path,records):
    records=list(records);fields=sorted({k for r in records for k in r}) or ['event']
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    with gzip.open(temporary,'wt',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for record in records:
            writer.writerow({k:json.dumps(v,separators=(',',':'),allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in record.items()})
    os.replace(temporary,path)
    return dict(path=str(path),sha256=sha256(path),records=len(records),fields=fields)


def jobs_for(cases,phase,initial=False):
    result=[]
    for case in cases:
        if initial and case['case']>=5:continue
        if phase=='A' and case['num_agents'] not in (8,16,32):continue
        for treatment in TREATMENTS if phase=='A' else ('A4',):
            for weight in core.WEIGHTS[:1] if phase=='A' else core.WEIGHTS:
                for condition in core.CONDITIONS:
                    result.append(dict(phase=phase,treatment=treatment,weight_id=weight[0],latency=weight[1],energy=weight[2],
                        route=condition['route'],num_agents=case['num_agents'],case_id=case['case_id']))
    return sorted(result,key=lambda j:(-j['num_agents'],j['treatment'],j['route'],j['case_id'],j['weight_id']))


def verify(root,historical=False):
    root=Path(root);m=core.read(root/'manifest.json')
    for name,digest in m['frozen_sha256'].items():
        if sha256(root/name)!=digest:raise ValueError('Frozen input changed: '+name)
    for name,digest in m['forbidden_sha256'].items():
        if sha256(root/'code/panama_uav'/name)!=digest:raise ValueError('Protected implementation changed: '+name)
    if historical:
        for name,digest in m['protected_results_sha256'].items():
            if sha256(Path(m['project_root'])/name)!=digest:raise ValueError('Historical result changed: '+name)
    return m


def prepare(root,source=DEFAULT_SOURCE,budget_seconds=259200.,cases_per_population=10):
    import torch,psutil
    from panama_uav.training import load_model
    from panama_uav.resource_study import study_cases
    root,source=Path(root).resolve(),Path(source).resolve();project=Path.cwd()
    if root.exists():raise ValueError('Use a new versioned directory')
    if cases_per_population not in (10,20) or budget_seconds<=0:raise ValueError('Use a registered cohort and positive wall cap')
    validation_path=project/'tmp/redesign-validation.json'
    validation=core.read(validation_path)
    if validation['status']!='passed' or validation['full_suite_failures'] or validation['full_suite_errors']:
        raise ValueError('Passing full regression evidence is required before preparing source')
    for name,digest in validation['source_sha256'].items():
        if sha256(project/name)!=digest:raise ValueError('Regression evidence does not match current source: '+name)
    original=core.read(source/'manifest.json')
    if core.read(source/'status.json')['status']!='complete':raise ValueError('Original comparison incomplete')
    audit_source=project/'output/redesign/audit-20261026-01/src/panama_uav'
    for name in FORBIDDEN:
        if sha256(project/'src/panama_uav'/name)!=sha256(audit_source/name):
            raise ValueError('Forbidden local production change: '+name)
    old=core.read(source/'cases.json')
    config=ProjectConfig.from_dict(original['configuration'])
    torch.set_num_threads(1)
    _,payload=load_model(source/'assets/checkpoint.pt',config,torch.device('cpu'),require_qualified=True)
    (root/'assets').mkdir(parents=True)
    shutil.copy2(validation_path,root/'regression_validation.json')
    for name in ('checkpoint.pt','orbit.npz',FIXED_MAP):
        target=root/'assets'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/'assets'/name,target)
    generated=study_cases(config,cases_per_population,core.COUNTS,250)
    historical={c['case_id']:c for c in old}
    for case in generated:
        if case['case_id'] in historical:
            for field in ('seed','starts','goals','orbital_offset_seconds'):
                if case[field]!=historical[case['case_id']][field]:raise ValueError('Historical case identity changed: '+field)
            case.update(historical[case['case_id']])
        else:
            case.update(map_path=str(root/'assets'/FIXED_MAP),map_sha256=original['map_sha256'],scenario='random_mapf')
    core.write(root/'cases.json',generated)
    shutil.copytree(project/'src/panama_uav',root/'code/panama_uav',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('run_redesign.py','run_omega_sensitivity.py','report_redesign.py','report_omega_sensitivity.py','audit_omega_reproduction.py'):
        shutil.copy2(project/'scripts'/name,root/'code'/name)
    shutil.copy2(source/'code/pilot_helper.py',root/'code/pilot_helper.py')
    config=replace(config,project_root=root/'assets',communication=replace(config.communication,leo_timeline=str(root/'assets/orbit.npz')))
    for name in TREATMENTS:
        for w,l,e in core.WEIGHTS:
            path=root/'configs'/name/(w.lower()+'.json')
            core.write(path,dict(configuration=configured(config,settings_for(name),l,e).to_dict(),motion=asdict(settings_for(name))))
    jobs_a=jobs_for(generated,'A');jobs_b=jobs_for(generated,'B')
    unique={job_id(j):j for j in jobs_a+jobs_b}
    references={}
    for path in (source/'episodes/W1').glob('*/*.result.json'):
        packet=core.read(path);r=packet['row']
        references[r['route']+'/'+r['case_id']]=dict(packet=str(path),sha256=sha256(path),signatures=packet['signatures'])
    manifest=dict(protocol=PROTOCOL,created_utc=datetime.now(timezone.utc).isoformat(),project_root=str(project),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_dirty=True,
        source_run=str(source),configuration=config.to_dict(),populations=list(core.COUNTS),
        cases_per_population=cases_per_population,horizon_seconds=250.,treatments=TREATMENTS,
        weights=[dict(id=w,latency=l,energy=e) for w,l,e in core.WEIGHTS],conditions=list(core.CONDITIONS),
        jobs_A=jobs_a,jobs_B=jobs_b,unique_primary_jobs=len(unique),phase_A_episodes=len(jobs_a),phase_B_episodes=len(jobs_b),
        phase_B_reused_A4_W1_jobs=len(set(map(job_id,jobs_a))&set(map(job_id,jobs_b))),
        budget_seconds=budget_seconds,workers=16,max_workers=18,heavy_workers=8,large_workers=4,
        ram_reserve_gib=8.,initial_peak_rss_gib={'8':1.5,'16':5.,'32':20.},
        model_sha256=original['model_sha256'],timeline_sha256=original['timeline_sha256'],map_sha256=original['map_sha256'],
        learner_updates=payload.get('learner_updates',0),policy_updates=0,exploratory_populations=[16,32],
        inference_threads=1,thread_environment=THREADS,baseline_references=references,
        forbidden_sha256={name:sha256(project/'src/panama_uav'/name) for name in FORBIDDEN},
        protected_results_sha256={p.relative_to(project).as_posix():sha256(p) for p in (project/'results').rglob('*') if p.is_file()},
        hardware=dict(physical_cores=psutil.cpu_count(False),logical_cores=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total),
        energy_scope='Existing communication and UAV computation only; no locomotion, propulsion or new fallback CPU energy')
    core.write(root/'source_changes.json',dict(changed=[p.name for p in (project/'src/panama_uav').glob('*.py')
        if not (audit_source/p.name).exists() or sha256(p)!=sha256(audit_source/p.name)],forbidden_unchanged=True))
    subprocess.run(['git','diff','--binary','--output='+str(root/'source.patch')],check=True)
    manifest['frozen_sha256']={p.relative_to(root).as_posix():sha256(p) for folder in ('code','assets','configs')
        for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    for name in ('cases.json','source_changes.json','source.patch','regression_validation.json'):manifest['frozen_sha256'][name]=sha256(root/name)
    core.write(root/'manifest.json',manifest);core.write(root/'status.json',dict(status='prepared',primary_completed=0,expected=len(unique)))
    return manifest


def initialize(root):
    import torch
    from panama_uav.training import load_model
    global ROOT,M,CONFIG,MODEL,CASES,HELPER
    ROOT=Path(root);M=verify(ROOT);CONFIG=ProjectConfig.from_dict(M['configuration'])
    CASES={c['case_id']:c for c in core.read(ROOT/'cases.json')}
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(42)
    MODEL,_=load_model(ROOT/'assets/checkpoint.pt',CONFIG,torch.device('cpu'),require_qualified=True)
    HELPER=core.module(ROOT/'code/pilot_helper.py','redesign_pilot_helper');HELPER.MODEL=MODEL


def infer(location,robot,observation):
    import torch
    from panama_uav.models import select_action
    return select_action(MODEL,observation,0.,np.random.default_rng(0),torch.device('cpu'))


def progress(phase,job,**fields):
    import psutil
    p=psutil.Process()
    core.write(ROOT/f'worker-{os.getpid()}.json',dict(phase=phase,job=job_id(job),pid=p.pid,
        creation_time=p.create_time(),time=time.time(),rss_bytes=p.memory_info().rss,**fields))


def episode(argument):
    import torch,psutil
    from panama_uav.inference_controls import InferenceCondition,inference_class
    from panama_uav.evaluation import run_episode
    from panama_uav.communication_effects import measurements
    from panama_uav.scenarios import Scenario
    from panama_uav.redesign import run_frequency_episode
    from panama_uav.redesign_metrics import measure
    from panama_uav.redesign_audit import audit_redesign
    from panama_uav.staleness_audit import audit_stress
    namespace,job,horizon=argument
    output=output_path(ROOT,job,namespace);output.parent.mkdir(parents=True,exist_ok=True)
    if committed_output(output):raise ValueError('Completed output cannot be overwritten')
    if output.with_name('error.json').exists():raise ValueError('A recorded failed attempt requires investigation before retry')
    interrupted=[p for p in output.parent.iterdir() if p.is_file()]
    if interrupted:
        folder=output.parent/'interrupted_attempts'/str(time.time_ns());folder.mkdir(parents=True)
        for path in interrupted:shutil.move(str(path),str(folder/path.name))
    started=time.monotonic();timing={};trace=None
    try:
        progress('simulation',job)
        case=CASES[job['case_id']]
        condition=InferenceCondition(next(c['name'] for c in core.CONDITIONS if c['route']==job['route']),job['route'])
        base=replace(CONFIG,communication=replace(CONFIG.communication,
            mission_route='cloud' if job['route'] in ('allocated','cloud') else 'uav',
            backhaul_time_offset_seconds=CONFIG.communication.backhaul_time_offset_seconds+case['orbital_offset_seconds']),
            resource_allocation=replace(CONFIG.resource_allocation,failure_directory=str(output.parent/'solver_failures')))
        settings=settings_for(job['treatment'],horizon)
        config=configured(base,settings,job['latency'],job['energy'])
        scenario=Scenario(Path(case['case_id']),ROOT/'assets'/FIXED_MAP,tuple(map(tuple,case['starts'])),tuple(map(tuple,case['goals'])))
        weight=dict(id=job['weight_id'],latency=job['latency'],energy=job['energy'])
        if job['treatment']=='A0':
            row,trace=run_episode(config,MODEL,case,torch.device('cpu'),M['learner_updates'],
                max_steps=int(horizon),scenario=scenario,remote_communication_class=inference_class(condition))
            timing['simulation_seconds']=time.monotonic()-started;at=time.monotonic()
            row,tasks,rounds=measurements(row,trace,condition);row,tasks,rounds=HELPER.enrich(row,tasks,rounds,trace)
            row.update(drone_capacity=6e9,label=HELPER.LABEL)
            signatures=core.signatures(row,tasks,rounds,trace)
        else:
            row,trace=run_frequency_episode(config,MODEL,case,torch.device('cpu'),M['learner_updates'],scenario,condition,settings,
                progress=lambda cycle,now,tasks:progress('simulation',job,cycle=cycle,physical_seconds=now,tasks=tasks))
            timing['simulation_seconds']=time.monotonic()-started;at=time.monotonic()
            row,tasks=measure(row,trace,job['treatment'],weight,condition);rounds=[];signatures={}
        row.update(treatment=job['treatment'],weight_id=job['weight_id'],route=job['route'],condition=condition.name)
        timing['measurement_seconds']=time.monotonic()-at;at=time.monotonic();progress('causal_replay',job)
        audit=audit_stress(trace,infer,include_observations=True) if job['treatment']=='A0' else audit_redesign(trace,infer)
        from audit_omega_reproduction import certificate
        allocations=[e for e in trace['transport_events'] if e['event']=='paper_allocation']
        for event in allocations:
            certified=certificate(event['relaxed'],event['relaxed'])
            if not certified['passed'] or event['relaxed']['objective']>event['binary']['objective']+1e-6*max(1.,abs(event['binary']['objective'])):
                raise ValueError('Allocation certificate or relaxation bound failed: '+event['new_job'])
        audit['allocation_certificates_verified']=len(allocations)
        if job['treatment']=='A0':
            from panama_uav.redesign_metrics import extend_legacy
            row=extend_legacy(row,trace,audit.pop('observation_aggregates'))
        timing['audit_seconds']=time.monotonic()-at;at=time.monotonic();progress('archive',job)
        exports={}
        for name,kinds,events in (
            ('decisions',('decision_event','execution_decision_event'),trace['remote_events']),
            ('command_bundles',('bundle_inference','action_bundle_arrived','delayed_command_arrived'),trace['remote_events']),
            ('movement',('motion_event','frequency_motion_event'),trace['remote_events']),
            ('priority_reversals',('priority_reversal_event','priority_reversal_cache_duration'),trace['oracle_events']),
            ('fallback_diagnostics',('fallback_shadow',),trace['oracle_events']),
            ('allocations',('allocation_event',),trace['remote_events'])):
            exports[name]=event_export(output.with_name(name+'.csv.gz'),(e for e in events if e['event'] in kinds))
        timing['telemetry_export_seconds']=time.monotonic()-at;at=time.monotonic()
        archive=output.with_name('trace.pkl.gz');core.write_trace(archive,trace)
        timing['archive_seconds']=time.monotonic()-at
        audit.update(archive=str(archive),archive_sha256=sha256(archive))
        packet=dict(job=job,row=row,tasks=tasks,rounds=rounds,audit=audit,event_exports=exports,signatures=signatures,timing=timing,
            manifest_sha256=sha256(ROOT/'manifest.json'),trace_bytes=archive.stat().st_size)
        core.write(output,packet)
        metadata=dict(job=job_id(job),path=str(output),sha256=sha256(output),success=row['success'],tasks=len(tasks),
            wall_seconds=time.monotonic()-started,peak_rss_bytes=psutil.Process().memory_info().peak_wset,**timing)
        core.write(output.with_name('complete.json'),metadata);progress('complete',job,**{k:v for k,v in metadata.items() if k!='job'})
        del trace,packet,tasks;gc.collect()
        return metadata
    except Exception as exc:
        failure=dict(job=job,error=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc(),
            snapshot=getattr(exc,'snapshot',None),solver_details=getattr(exc,'details',None))
        if trace is not None:
            core.write_trace(output.with_name('failed_trace.pkl.gz'),trace)
        core.write(output.with_name('error.json'),failure);progress('error',job,error=str(exc));raise


def dispatch(root,jobs,workers,namespace='primary',horizon=250.):
    """Dynamic, deterministic jobs with live RAM reserve and size-class limits."""
    import psutil
    root=Path(root);m=verify(root);started=time.monotonic();pending={};done=[]
    initial_io=psutil.disk_io_counters()
    queue=[j for j in jobs if not committed_output(output_path(root,j,namespace))]
    telemetry=[];last_sample=0.;launch=core.read(root/'launch.json')
    # Admission reserves worst-case remaining RSS, including idle model workers.
    estimates={2:1.,4:1.,8:1.5,16:5.,32:20.}
    if (root/'memory_estimates.json').exists():estimates.update({int(k):v for k,v in core.read(root/'memory_estimates.json').items()})
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(root),)) as executor:
        while queue or pending:
            if time.time()-launch['first_started_unix']>=m['budget_seconds']:
                core.write(root/'budget_stop.json',dict(status='wall_cap',time=time.time(),
                    interrupted_jobs=[job_id(j) for j in pending.values()],remaining_jobs=[job_id(j) for j in queue]))
                processes=list(executor._processes.values())
                for process in processes:
                    if process.is_alive():process.terminate()
                raise TimeoutError('Fixed experiment wall budget reached; completed transactions preserved')
            for job in list(queue):
                if len(pending)>=workers:break
                active=list(pending.values())
                if job['num_agents']>=16 and sum(j['num_agents']>=16 for j in active)>=m['heavy_workers']:continue
                if job['num_agents']==32 and sum(j['num_agents']==32 for j in active)>=m['large_workers']:continue
                # A conservative reservation protects against peak replay memory.
                planned=sum(estimates[j['num_agents']] for j in active)+estimates[job['num_agents']]
                children=psutil.Process().children(recursive=True)
                actual=sum(p.memory_info().rss for p in children if p.is_running())/2**30
                available=psutil.virtual_memory().available/2**30
                if max(0.,planned-actual)>available-m['ram_reserve_gib']:continue
                future=executor.submit(episode,(namespace,job,horizon));pending[future]=job;queue.remove(job)
            if not pending and queue:raise MemoryError('RAM reserve prevents dispatch; no jobs started or omitted')
            completed,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for future in completed:
                job=pending.pop(future);result=future.result();done.append(result)
            if time.monotonic()-last_sample>=5:
                vm=psutil.virtual_memory();children=psutil.Process().children(recursive=True)
                rss=sum(p.memory_info().rss for p in children if p.is_running())
                sample=dict(time=time.time(),rss_bytes=rss,available_bytes=vm.available,cpu_percent=psutil.cpu_percent(),
                    io=dict(psutil.disk_io_counters()._asdict()),pending=len(pending),remaining=len(queue),completed=len(done))
                telemetry.append(sample);last_sample=time.monotonic()
                core.write(root/'status.json',dict(status='running',namespace=namespace,completed=len(done),remaining=len(queue),
                    active=[job_id(j) for j in pending.values()],last_sample=sample,workers=workers))
                with (root/'scheduler.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(sample)+'\n')
    summary=dict(namespace=namespace,workers=workers,episodes=len(jobs),wall_seconds=time.monotonic()-started,
        completed_audited_episodes_per_hour=len(done)/(time.monotonic()-started)*3600,
        peak_rss_bytes=max((s['rss_bytes'] for s in telemetry),default=0),
        mean_rss_bytes=float(np.mean([s['rss_bytes'] for s in telemetry])) if telemetry else 0,
        min_available_bytes=min((s['available_bytes'] for s in telemetry),default=0),
        mean_cpu_percent=float(np.mean([s['cpu_percent'] for s in telemetry])) if telemetry else 0,
        disk_read_bytes=psutil.disk_io_counters().read_bytes-initial_io.read_bytes,
        disk_write_bytes=psutil.disk_io_counters().write_bytes-initial_io.write_bytes,results=done)
    core.write(root/(namespace.replace('/','_')+'_scheduler.json'),summary)
    return summary


def run(root,stage):
    root=Path(root);m=verify(root,True);cases=core.read(root/'cases.json')
    if stage=='workflow':
        for part in ('preflight','reference','benchmark','full'):
            gate={'preflight':'preflight.json','reference':'reference_validation.json','benchmark':'benchmark.json'}.get(part)
            if gate and (root/gate).exists() and core.read(root/gate)['status']=='passed':continue
            core.write(root/'workflow.json',dict(status='running',stage=part,started_at=time.time()))
            run(root,part)
        core.write(root/'workflow.json',dict(status='complete',finished_at=time.time()))
        return
    if stage=='preflight':
        jobs=[j for j in jobs_for(cases,'B') if j['weight_id']=='W1' and j['route']=='allocated' and j['case_id'].endswith('-000')]
        dispatch(root,jobs,2,'preflight',horizon=10.)
        core.write(root/'preflight.json',dict(status='passed',episodes=len(jobs),horizon=10.))
    elif stage=='reference':
        jobs=[j for j in m['jobs_A'] if j['treatment']=='A0' and j['case_id'].endswith('-000')]
        dispatch(root,jobs,4)
        checks=[]
        for job in jobs:
            packet=core.read(output_path(root,job));reference=m['baseline_references'][job['route']+'/'+job['case_id']]
            if sha256(reference['packet'])!=reference['sha256']:raise ValueError('Preserved original result changed')
            # Host clocks/fractional minimizers are already excluded by signatures.
            matches={k:packet['signatures'][k]==v for k,v in reference['signatures'].items()}
            original=core.read(reference['packet'])
            old_archive=Path(original['audit']['archive'])
            if sha256(old_archive)!=original['audit']['archive_sha256']:raise ValueError('Historical trace changed')
            def relaxed_records(archive):
                with gzip.open(archive,'rb') as stream:trace=pickle.load(stream)
                return {e['new_job']:e['relaxed'] for e in trace['transport_events'] if e['event']=='paper_allocation'}
            old=relaxed_records(old_archive);new=relaxed_records(packet['audit']['archive'])
            from audit_omega_reproduction import certificate
            if old.keys()!=new.keys():raise ValueError('A0 fractional allocation snapshots changed')
            certificates=[certificate(old[k],new[k]) for k in old]
            checks.append(dict(job=job_id(job),matches=matches,fractional_certificates=len(certificates),
                fractional_failed=sum(not c['passed'] for c in certificates),
                passed=all(matches.values()) and all(c['passed'] for c in certificates)))
        core.write(root/'reference_validation.json',dict(status='passed' if all(c['passed'] for c in checks) else 'failed',checks=checks))
        if not all(c['passed'] for c in checks):raise ValueError('A0 reproduction gate failed')
    elif stage=='benchmark':
        if core.read(root/'reference_validation.json')['status']!='passed':raise ValueError('A0 must reproduce before benchmark')
        jobs=[j for j in m['jobs_A'] if j['treatment']=='A4' and int(j['case_id'].split('-')[-1])<2]
        results=[]
        for workers in (12,16,18):
            namespace='benchmark/'+str(workers)
            results.append(dispatch(root,jobs,workers,namespace))
        stable=[r for r in results if r['min_available_bytes']>=m['ram_reserve_gib']*2**30]
        if not stable:raise MemoryError('No stable audited benchmark met the RAM reserve')
        chosen=max(stable,key=lambda r:r['completed_audited_episodes_per_hour'])
        checks=[]
        for job in jobs:
            copies=[core.read(output_path(root,job,'benchmark/'+str(w))) for w in (12,16,18)]
            signatures=[stable_signature(p) for p in copies]
            if len(set(signatures))!=1:raise ValueError('Worker scheduling changed a fixed-seed simulation: '+job_id(job))
            checks.append(dict(job=job_id(job),signature=signatures[0],worker_counts=[12,16,18],passed=True))
            # Promote a verified transaction rather than rerunning identical A4/W1.
            source=output_path(root,job,'benchmark/12');target=output_path(root,job)
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                # Packets retain the original archive path and manifest identity.
                shutil.copy2(source,target);shutil.copy2(source.with_name('complete.json'),target.with_name('complete.json'))
        core.write(root/'benchmark.json',dict(status='passed',selected_workers=chosen['workers'],results=results,
            scheduling_identity_checks=checks,reused_primary_episodes=len(checks)))
        memory={}
        for n in core.COUNTS:
            peaks=[r['peak_rss_bytes']/2**30 for run_result in results for r in run_result['results']
                if int(r['job'].split('/')[3])==n]
            if peaks:memory[str(n)]=max(peaks)*1.3
        core.write(root/'memory_estimates.json',memory)
        # Estimate every registered job from audited population/route timings,
        # respecting the actual measured concurrency rather than nominal cores.
        elapsed=sum(r['wall_seconds'] for r in chosen['results'])
        throughput=chosen['completed_audited_episodes_per_hour']
        remaining=m['unique_primary_jobs']-len(list((root/'primary').rglob('result.json')))
        estimate=dict(measured_audited_episodes_per_hour=throughput,selected_workers=chosen['workers'],
            remaining_primary_jobs=remaining,estimated_primary_seconds=remaining/throughput*3600,
            reporting_and_audit_reserve_seconds=3600.,safety_factor=1.25,
            estimated_remaining_seconds=remaining/throughput*3600*1.25+3600.,
            fixed_wall_cap_seconds=m['budget_seconds'],recorded_before_primary_execution=True)
        core.write(root/'budget_estimate.json',estimate)
    elif stage=='full':
        if core.read(root/'preflight.json')['status']!='passed' or core.read(root/'reference_validation.json')['status']!='passed' or core.read(root/'benchmark.json')['status']!='passed':
            raise ValueError('Validation and benchmark gates required')
        workers=core.read(root/'benchmark.json')['selected_workers']
        estimate=core.read(root/'budget_estimate.json')
        elapsed=time.time()-core.read(root/'launch.json')['first_started_unix']
        if estimate['estimated_remaining_seconds']>m['budget_seconds']-elapsed:
            raise TimeoutError('Measured full-run estimate exceeds the remaining fixed wall cap; estimate retained')
        dispatch(root,jobs_for(cases,'A',True),workers)
        from report_redesign import report
        if not (root/'reports/phase_A_initial').exists():report(root,'phase_A_initial')
        dispatch(root,m['jobs_A'],workers)
        if not (root/'reports/phase_A').exists():report(root,'phase_A')
        dispatch(root,m['jobs_B'],workers)
        if not (root/'reports/complete').exists():report(root,'complete')
        core.write(root/'status.json',dict(status='complete',primary_episodes=m['unique_primary_jobs'],policy_updates=0))


def launch(root,stage,resume=False):
    import psutil
    root=Path(root).resolve();m=verify(root)
    for process in psutil.process_iter(['cmdline']):
        argv=process.info['cmdline'] or []
        if '--worker' in argv and str(root) in argv:raise ValueError('Matching worker already active')
    old=core.read(root/'launch.json') if (root/'launch.json').exists() else None
    if old and not resume:raise ValueError('Existing run requires explicit resume')
    if old and time.time()-old['first_started_unix']>=m['budget_seconds']:raise TimeoutError('Wall budget exhausted')
    command=[sys.executable,str(root/'code/run_redesign.py'),'--worker',str(root),'--stage',stage]
    env=dict(os.environ,PYTHONPATH=str(root/'code'),**THREADS)
    with (root/'stdout.log').open('a') as out,(root/'stderr.log').open('a') as err:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    info=dict(pid=process.pid,creation_time=psutil.Process(process.pid).create_time(),command=command,stage=stage,
        first_started_unix=old['first_started_unix'] if old else time.time())
    core.write(root/'launch.json',info);return info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--launch',action='store_true')
    parser.add_argument('--worker',type=Path);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stage',choices=('preflight','reference','benchmark','full','workflow'),default='preflight')
    parser.add_argument('--budget-seconds',type=float,default=259200.);parser.add_argument('--cases',type=int,default=10)
    args=parser.parse_args()
    if args.worker:
        try:
            run(args.worker,args.stage)
            if args.stage not in ('full','workflow'):
                core.write(args.worker/'status.json',dict(status='stage_complete',stage=args.stage,policy_updates=0))
        except Exception as exc:
            core.write(args.worker/'status.json',dict(status='error',stage=args.stage,error=str(exc),traceback=traceback.format_exc()));raise
    else:
        if args.prepare:prepare(args.root,args.source,args.budget_seconds,args.cases)
        if args.launch:print(json.dumps(launch(args.root,args.stage,args.resume),indent=2))
