"""Parameter-only latency/energy study with a reproduced-baseline launch gate."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import gc
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import tomllib
import traceback

import numpy as np

from panama_uav.config import FIXED_MAP, ProjectConfig
from panama_uav.records import sha256

PROTOCOL = 'omega_sensitivity_v1'
WEIGHTS = (('W1',1.,1.),('W2',1.,2.),('W3',1.,4.),('W4',2.,1.),('W5',4.,1.))
COUNTS = (2,4,8,16,32)
CONDITIONS = (dict(name='allocated-6ghz',route='allocated',cpu=6e9),
              dict(name='uav-6ghz',route='edge',cpu=6e9),
              dict(name='cloud-6ghz',route='cloud',cpu=6e9))
HOST_FIELDS = {'inference_wall_seconds','solver_seconds','solve_seconds'}
RELAXED_FIELDS = {'relaxed_x','relaxation_gap','mean_relaxation_gap'}
DEFAULT_BASE = Path('output/uav6_pilot/uav6-20261024-01')
DEFAULT_ROOT = Path('output/omega_sensitivity/omega-20261025-01')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def normalized(value):
    if isinstance(value,dict):
        return {str(k):normalized(v) for k,v in value.items() if k not in HOST_FIELDS|RELAXED_FIELDS}
    if isinstance(value,(list,tuple)):
        return [normalized(v) for v in value]
    return value


def fingerprint(records):
    digest=hashlib.sha256()
    for record in records:
        digest.update(json.dumps(normalized(record),sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def signatures(row,tasks,rounds,trace):
    return dict(mission=fingerprint([row]),tasks=fingerprint(tasks),rounds=fingerprint(rounds),
        frames=fingerprint(trace['frames']),
        allocations=fingerprint(e for e in trace['remote_events'] if e['event']=='allocation_event'),
        cpu=fingerprint(e for e in trace['transport_events'] if e['event'] in ('compute_allocation','compute_complete')))


def write_trace(path,trace):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=3,mtime=0) as stream:
            pickle.dump(trace,stream,protocol=5)
    os.replace(temporary,path)


def config_changes(original,actual):
    def flatten(value,prefix=''):
        result={}
        for key,item in value.items():
            if isinstance(item,dict):result.update(flatten(item,prefix+key+'.'))
            else:result[prefix+key]=item
        return result
    a,b=flatten(original),flatten(actual)
    return {k:(a.get(k),b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}


def generate_configs(destination,baseline):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    text=Path(baseline).read_text(encoding='utf-8');original=tomllib.loads(text)
    required=dict(drone_cycles_per_second=6e9,cloud_cycles_per_second=1e10,task_input_bits=100000,
        cycles_per_bit=1000.,drone_kappa=1e-28,weight_min=1.,weight_max=4.,command_cutoff_seconds=.25,
        decision_budget_seconds=.25,task_deadline_seconds=.25,enforce_task_deadlines=False)
    if any(original['resource_allocation'].get(k)!=v for k,v in required.items()):
        raise ValueError('The established 6 GHz settings must remain unchanged')
    if original['communication']['seconds_per_step']!=1.:raise ValueError('Control interval changed')
    result={}
    for name,latency,energy in WEIGHTS:
        updated=text.replace('latency_coefficient = 1.0',f'latency_coefficient = {latency:.1f}')
        updated=updated.replace('energy_coefficient = 1.0',f'energy_coefficient = {energy:.1f}')
        data=tomllib.loads(updated)
        changes=config_changes(original,data)
        allowed={'resource_allocation.latency_coefficient','resource_allocation.energy_coefficient'}
        if not set(changes)<=allowed:raise ValueError('Only objective coefficients may change')
        p=data['resource_allocation']
        if (p['latency_coefficient'],p['energy_coefficient'])!=(latency,energy):raise ValueError('Weight mapping failed')
        path=destination/(name.lower()+'.toml')
        if path.exists() and path.read_text(encoding='utf-8')!=updated:raise ValueError('Existing configuration differs')
        path.write_text(updated,encoding='utf-8');result[name]=dict(path=str(path),sha256=sha256(path),latency=latency,energy=energy)
    return result


def protected(project):
    paths=subprocess.check_output(['git','ls-files'],cwd=project,text=True).splitlines()
    return {name:sha256(project/name) for name in paths
        if name.startswith(('src/','configs/','maps/','results/','paper/','research/'))
        and not name.startswith('results/omega_sensitivity/') and (project/name).is_file()}


def verify(root,check_original=False):
    root=Path(root);m=read(root/'manifest.json')
    for name,digest in m['frozen_sha256'].items():
        if sha256(root/name)!=digest:raise ValueError('Frozen input changed: '+name)
    for name,item in m['weight_configs'].items():
        data=tomllib.loads((root/'configs'/(name.lower()+'.toml')).read_text(encoding='utf-8'))
        base=tomllib.loads((root/'configs/w1.toml').read_text(encoding='utf-8'))
        if not set(config_changes(base,data)) <= {'resource_allocation.latency_coefficient','resource_allocation.energy_coefficient'}:
            raise ValueError('A non-weight setting changed')
    if check_original:
        for name,digest in m['protected_sha256'].items():
            if sha256(Path(m['project_root'])/name)!=digest:raise ValueError('Original file changed: '+name)
    return m


def prepare(root,base=DEFAULT_BASE,*,smoke=False,workers=4,heavy_workers=2,budget_seconds=86400):
    import torch,psutil
    from panama_uav.training import load_model
    root,base=Path(root).resolve(),Path(base).resolve();project=Path.cwd().resolve()
    if root.exists():raise ValueError('Use a new versioned run directory')
    if not 1<=heavy_workers<=workers<=6:raise ValueError('Invalid worker limits')
    original=read(base/'manifest.json');extension=base/'extensions/agents16-32-01'
    for name,digest in original['code_sha256'].items():
        if sha256(project/'src/panama_uav'/name)!=digest:raise ValueError('Production source differs from reference: '+name)
    assert read(base/'status.json')['status']=='complete' and read(extension/'status.json')['status']=='complete'
    config=ProjectConfig.from_dict(original['configuration'])
    torch.set_num_threads(1)
    _,payload=load_model(base/'assets/checkpoint.pt',config,torch.device('cpu'),require_qualified=True)
    (root/'assets').mkdir(parents=True)
    for name in ('checkpoint.pt','orbit.npz',FIXED_MAP):
        target=root/'assets'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/'assets'/name,target)
    configurations=generate_configs(root/'configs',project/'configs/paper_remote_dt_250ms_uav6.toml')
    cases=read(base/'cases.json')+read(extension/'cases.json')
    cases=sorted([c for c in cases if c['num_agents'] in COUNTS and (not smoke or c['case_id'].endswith('-000'))],key=lambda c:(c['num_agents'],c['case_id']))
    references={}
    for case in cases:
        if sha256(case['map_path'])!=case['map_sha256']:raise ValueError('Historical map changed')
        source=base if case['num_agents']<=8 else extension
        for condition in CONDITIONS:
            path=source/'episodes'/condition['name']/(case['case_id']+'.result.json')
            packet=read(path)
            references[condition['name']+'/'+case['case_id']]=dict(packet=str(path),packet_sha256=sha256(path),
                archive=packet['audit']['archive'],archive_sha256=packet['audit']['archive_sha256'])
    write(root/'cases.json',cases)
    shutil.copytree(base/'code/panama_uav',root/'code/panama_uav',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(base/'code/run_uav6_pilot.py',root/'code/pilot_helper.py')
    for name in ('run_omega_sensitivity.py','report_omega_sensitivity.py','audit_omega_reproduction.py'):
        shutil.copy2(project/'scripts'/name,root/'code'/name)
    config=replace(config,project_root=root/'assets',communication=replace(config.communication,leo_timeline=str(root/'assets/orbit.npz')))
    selected=list(WEIGHTS[:1] if smoke else WEIGHTS)
    manifest=dict(protocol=PROTOCOL,created_utc=datetime.now(timezone.utc).isoformat(),project_root=str(project),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),configuration=config.to_dict(),
        weights=[dict(id=n,latency=l,energy=e) for n,l,e in selected],weight_configs=configurations,
        conditions=list(CONDITIONS),populations=list(COUNTS),cases_per_population=1 if smoke else 5,
        episode_cap=3 if smoke else 250,expected_episodes=len(cases)*len(CONDITIONS)*len(selected),smoke=smoke,
        workers=workers,heavy_workers=heavy_workers,heavy_min_agents=16,budget_seconds=budget_seconds,
        baseline_references=references,baseline_root=str(base),baseline_extension=str(extension),
        learner_updates=payload.get('learner_updates',0),policy_updates=0,
        model_qualification=payload.get('qualification',[]),exploratory_populations=[16,32],
        model_sha256=original['model_sha256'],timeline_sha256=original['timeline_sha256'],map_sha256=original['map_sha256'],
        protected_sha256=protected(project),inference_device='cpu',threads_per_worker=1,
        archive_format='gzip pickle protocol 5; full unchanged trace dictionary; causal audit before archival',
        baseline_exact_checks=['physical frames and event checksum','executed task records','binary allocations','CPU service','command timing','mission metrics'],
        baseline_host_exclusions=sorted(HOST_FIELDS),fractional_audit='execution_exact_fractional_certificate_v2',
        hardware=dict(physical_cores=psutil.cpu_count(False),logical_cores=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total),
        packages={name:importlib.metadata.version(name) for name in original['packages']})
    manifest['frozen_sha256']={p.relative_to(root).as_posix():sha256(p) for folder in ('code','assets','configs')
        for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    manifest['frozen_sha256']['cases.json']=sha256(root/'cases.json')
    write(root/'manifest.json',manifest);write(root/'status.json',dict(status='prepared',completed_episodes=0,expected_episodes=manifest['expected_episodes']))
    return manifest


def initialize(root):
    import torch
    from panama_uav.training import load_model
    global ROOT,M,H,MODEL
    ROOT=Path(root);M=verify(ROOT);H=module(ROOT/'code/pilot_helper.py','pilot_helper')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
    MODEL,_=load_model(ROOT/'assets/checkpoint.pt',ProjectConfig.from_dict(M['configuration']),torch.device('cpu'),require_qualified=True)
    H.MODEL=MODEL


def progress(phase,**fields):
    import psutil
    process=psutil.Process()
    write(ROOT/f'worker-{os.getpid()}.json',dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),creation_time=process.create_time(),rss_bytes=process.memory_info().rss,**fields))


def inference(location,robot,observation):
    return H.infer(location,robot,observation)


def episode(job):
    import torch
    from panama_uav.communication_effects import measurements
    from panama_uav.evaluation import run_episode
    from panama_uav.inference_controls import InferenceCondition,inference_class
    from panama_uav.scenarios import Scenario
    from panama_uav.staleness_audit import audit_stress
    weight,condition,case=job;identifier=weight['id']+'/'+condition['name']+'/'+case['case_id']
    folder=ROOT/'episodes'/weight['id']/condition['name'];folder.mkdir(parents=True,exist_ok=True)
    output=folder/(case['case_id']+'.result.json')
    if output.exists():raise ValueError('An existing case cannot be overwritten')
    started=time.monotonic();timing={}
    try:
        progress('simulation',job=identifier)
        original=ProjectConfig.from_dict(M['configuration'])
        configured=replace(original,resource_allocation=replace(original.resource_allocation,
            latency_coefficient=weight['latency'],energy_coefficient=weight['energy']))
        config=H.execution_config(configured,condition,ROOT/weight['id'],case)
        c=InferenceCondition(condition['name'],condition['route'])
        scenario=Scenario(Path(case['case_id']),Path(case['map_path']),tuple(map(tuple,case['starts'])),tuple(map(tuple,case['goals'])))
        row,trace=run_episode(config,MODEL,case,torch.device('cpu'),M['learner_updates'],max_steps=M['episode_cap'],scenario=scenario,remote_communication_class=inference_class(c))
        timing['simulation_seconds']=time.monotonic()-started;at=time.monotonic();progress('measurement',job=identifier)
        row,tasks,rounds=measurements(row,trace,c);row,tasks,rounds=H.enrich(row,tasks,rounds,trace)
        row.update(drone_capacity=6e9,label=H.LABEL)
        semantic=signatures(row,tasks,rounds,trace)
        allocations=[];first_snapshots=[];task_by_id={t['job_id']:t for t in tasks}
        for event in trace['remote_events']:
            if event['event']!='allocation_event':continue
            task=next(t for t in event['assignments'] if t['task_id']==event['job_id'])
            previous=[t for t in event['snapshot']['tasks'] if t['task_id']!=event['job_id']]
            allocations.append(dict(job_id=event['job_id'],time=event['time'],**task,
                fixed_previous_tasks=len(previous),fixed_previous_uav=sum(t['fixed_x']==1 for t in previous),
                fixed_previous_cloud=sum(t['fixed_x']==0 for t in previous)))
            if task_by_id[event['job_id']]['cycle']==0:
                first_snapshots.append(dict(job_id=event['job_id'],time=event['time'],snapshot=event['snapshot'],assignments=event['assignments']))
        cpu=[]
        for pool in ('drone','cloud'):
            events=[e for e in trace['transport_events'] if e['event']=='compute_allocation' and e['pool']==pool]
            frequency_time=occupied_time=0.;peak=0.;samples=0
            for index,event in enumerate(events):
                stop=events[index+1]['time'] if index+1<len(events) else row['observed_seconds']
                duration=max(0.,min(stop,row['observed_seconds'])-event['time'])
                for item in event['jobs']:
                    frequency_time+=item['cycles_per_second']*duration;occupied_time+=duration
                    peak=max(peak,item['cycles_per_second']);samples+=1
            cpu.append(dict(pool=pool,frequency_job_seconds=frequency_time,allocated_job_seconds=occupied_time,
                frequency_when_active_cps=frequency_time/occupied_time if occupied_time else None,peak_job_frequency_cps=peak,job_allocation_samples=samples))
        timing['measurement_seconds']=time.monotonic()-at;at=time.monotonic();progress('causal_replay',job=identifier)
        audit=audit_stress(trace,inference)
        timing['audit_seconds']=time.monotonic()-at;at=time.monotonic();progress('archive',job=identifier)
        trace['omega_experiment']=dict(protocol=PROTOCOL,weight=weight,condition=condition,manifest_sha256=sha256(ROOT/'manifest.json'))
        archive=folder/(case['case_id']+'.pkl.gz');write_trace(archive,trace)
        timing['archive_seconds']=time.monotonic()-at
        del trace,task_by_id;gc.collect()
        audit.update(archive=str(archive),archive_sha256=sha256(archive),weight_id=weight['id'],condition=condition['name'],case_id=case['case_id'])
        packet=dict(weight=weight,row=row,tasks=tasks,rounds=rounds,allocations=allocations,first_snapshots=first_snapshots,
            cpu=cpu,audit=audit,signatures=semantic,timing=timing,manifest_sha256=sha256(ROOT/'manifest.json'))
        write(output,packet)
        metadata=dict(job=identifier,path=str(output),sha256=sha256(output),success=row['success'],steps=row['steps'],tasks=len(tasks),wall_seconds=time.monotonic()-started,**timing)
        write(output.with_suffix('.meta.json'),metadata);progress('complete',**metadata)
        del packet,tasks,rounds,allocations;gc.collect()
        return metadata
    except Exception as error:
        details=dict(job=identifier,error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc(),
            snapshot=getattr(error,'snapshot',None),solver_details=getattr(error,'details',None),timing=timing)
        write(folder/(case['case_id']+'.error.json'),details);progress('error',**details);raise


def baseline_gate(root,jobs):
    from audit_omega_reproduction import baseline_gate as certified_gate
    return certified_gate(root,jobs)


def result_path(root,job):
    weight,condition,case=job
    return Path(root)/'episodes'/weight['id']/condition['name']/(case['case_id']+'.result.json')


def run(root):
    import psutil
    root=Path(root).resolve();m=verify(root,True)
    lease=read(root/'launch.json');deadline=lease['first_started_unix']+m['budget_seconds']
    cases=read(root/'cases.json');all_jobs=[(w,c,case) for w in m['weights'] for c in m['conditions'] for case in cases]
    errors=list((root/'episodes').glob('*/*/*.error.json'))
    if errors:raise ValueError('Recorded errors require diagnosis before resuming: '+str(errors[0]))
    completed={}
    for job in all_jobs:
        path=result_path(root,job)
        if path.exists():
            meta=read(path.with_suffix('.meta.json'))
            if sha256(path)!=meta['sha256']:raise ValueError('A committed transaction changed')
            completed[meta['job']]=meta
    def status(phase,**extra):
        write(root/'status.json',dict(status='running',stage=phase,completed_episodes=len(completed),expected_episodes=m['expected_episodes'],
            updated_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-lease['first_started_unix'],
            remaining_seconds=max(0.,deadline-time.time()),**extra))
    try:
        phases=[('baseline_simulation',[j for j in all_jobs if j[0]['id']=='W1']),
                ('weight_sensitivity',[j for j in all_jobs if j[0]['id']!='W1'])]
        if not m['smoke']:
            phases.insert(0,('baseline_preflight',phases[0][1][:1]))
        for phase,jobs in phases:
            queue=[j for j in jobs if not result_path(root,j).exists()];status(phase)
            if queue:
                with ProcessPoolExecutor(max_workers=m['workers'],initializer=initialize,initargs=(str(root),)) as executor:
                    pending={}
                    while queue or pending:
                        if time.time()>deadline:raise TimeoutError('Recorded study wall cap reached')
                        heavy=sum(j[2]['num_agents']>=m['heavy_min_agents'] for j in pending.values())
                        while len(pending)<m['workers'] and psutil.virtual_memory().available>5e9:
                            index=next((i for i,j in enumerate(queue) if j[2]['num_agents']<m['heavy_min_agents'] or heavy<m['heavy_workers']),None)
                            if index is None:break
                            job=queue.pop(index);pending[executor.submit(episode,job)]=job
                            heavy+=job[2]['num_agents']>=m['heavy_min_agents']
                        if not pending:raise MemoryError('Insufficient available RAM to launch a case')
                        done,_=wait(pending,timeout=45,return_when=FIRST_COMPLETED)
                        for future in done:
                            metadata=future.result();completed[metadata['job']]=metadata;del pending[future]
                            status(phase,latest=metadata)
            if phase in ('baseline_preflight','baseline_simulation') and not m['smoke']:
                status('baseline_reproduction_audit');baseline_gate(root,jobs)
        verify(root,True);status('reporting')
        report_module=module(root/'code/report_omega_sensitivity.py','omega_report')
        report_module.report(root)
        write(root/'status.json',dict(status='complete',completed_episodes=len(completed),expected_episodes=m['expected_episodes'],
            elapsed_seconds=time.time()-lease['first_started_unix'],policy_updates=0))
    except Exception as error:
        write(root/'status.json',dict(status='error',completed_episodes=len(completed),expected_episodes=m['expected_episodes'],
            elapsed_seconds=time.time()-lease['first_started_unix'],error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc()))
        raise


def launch(root,resume=False):
    import psutil
    root=Path(root).resolve();m=verify(root,True);path=root/'launch.json';previous=read(path) if path.exists() else None
    if previous:
        if not resume:raise ValueError('Launch exists; use a confirmed stopped run and --resume')
        if psutil.pid_exists(previous['pid']):
            process=psutil.Process(previous['pid'])
            if abs(process.create_time()-previous['creation_time'])<.01:raise ValueError('Recorded launcher is still running')
        for process in psutil.process_iter(['cmdline']):
            command=process.info['cmdline'] or []
            if '--worker' in command and str(root) in command:raise ValueError('A matching worker is still running')
        if time.time()-previous['first_started_unix']>=m['budget_seconds']:raise ValueError('No recorded wall budget remains')
        if read(root/'status.json')['status']=='complete':raise ValueError('The study is already complete')
    env=dict(os.environ,PYTHONPATH=str(root/'code'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,str(root/'code/run_omega_sensitivity.py'),'--worker',str(root)]
    with (root/'stdout.log').open('a') as out,(root/'stderr.log').open('a') as err:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    info=dict(pid=process.pid,creation_time=psutil.Process(process.pid).create_time(),command=command,
        first_started_unix=previous['first_started_unix'] if previous else time.time())
    write(path,info);print(json.dumps(info),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT);parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--launch',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--heavy-workers',type=int,default=2)
    parser.add_argument('--budget-seconds',type=float,default=86400.);parser.add_argument('--worker',type=Path)
    parser.add_argument('--report',action='store_true');args=parser.parse_args()
    if args.worker:run(args.worker)
    else:
        if args.prepare:prepare(args.root,args.base,smoke=args.smoke,workers=args.workers,heavy_workers=args.heavy_workers,budget_seconds=args.budget_seconds)
        if args.launch:launch(args.root,args.resume)
        if args.report:module(Path(__file__).with_name('report_omega_sensitivity.py'),'omega_report').report(args.root)
