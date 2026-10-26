"""Paired cached-A* fallback study with a preserved WAIT baseline and pilot gate."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import difflib
import gzip
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

import run_omega_sensitivity as core
from audit_omega_reproduction import certificate
from panama_uav.config import FIXED_MAP, ProjectConfig
from panama_uav.records import sha256

PROTOCOL='cached_astar_fallback_v1'
DEFAULT_SOURCE=Path('output/omega_sensitivity/omega-20261025-01')
DEFAULT_ROOT=Path('output/cached_fallback/cached-20261026-01')
VARIANTS={'primary':('cached_astar',.25),'wait250':('wait',.25),
          'wait100':('wait',.1),'cached100':('cached_astar',.1)}
ALLOWED_SOURCE_CHANGES={'config.py','remote.py','remote_control.py','remote_audit.py',
                        'remote_metrics.py','evaluation.py','communication_effects.py'}


def variant_root(root,variant):
    return Path(root) if variant=='primary' else Path(root)/'validation'/variant


def verify(root,originals=False):
    root=Path(root)
    manifest=core.verify(root,originals)
    if manifest['protocol']!=PROTOCOL:raise ValueError('Unknown fallback study protocol')
    if sha256(Path(manifest['source_root'])/'manifest.json')!=manifest['source_manifest_sha256']:
        raise ValueError('Historical experiment manifest changed')
    for case in core.read(root/'cases.json'):
        if sha256(case['map_path'])!=case['map_sha256']:raise ValueError('Paired historical map changed')
    return manifest


def prepare(root,source=DEFAULT_SOURCE,*,workers=4,heavy_workers=2,budget_seconds=86400):
    import psutil,torch
    from panama_uav.training import load_model
    root,source=Path(root).resolve(),Path(source).resolve()
    project=Path.cwd().resolve()
    if root.exists():raise ValueError('Use a new run directory; historical results cannot be overwritten')
    if not 1<=heavy_workers<=workers<=6 or budget_seconds<=0:raise ValueError('Invalid bounded worker/budget settings')
    original=core.read(source/'manifest.json')
    if core.read(source/'status.json')['status']!='complete':raise ValueError('Complete original experiment required')
    cases=core.read(source/'cases.json')
    if sorted(original['populations'])!=list(core.COUNTS) or len(cases)!=25:
        raise ValueError('Use the existing five paired cases at every requested population')
    expected_weights=[dict(id=n,latency=l,energy=e) for n,l,e in core.WEIGHTS]
    if original['weights']!=expected_weights:raise ValueError('Historical weights changed')
    # Observation/action, model, solver, transport and movement sources are frozen.
    # Only missed-command execution and its configuration, replay and accounting differ.
    differences=[];changed=[]
    for old in sorted((source/'code/panama_uav').glob('*.py')):
        current=project/'src/panama_uav'/old.name
        if sha256(old)!=sha256(current):
            changed.append(old.name)
            if old.name not in ALLOWED_SOURCE_CHANGES:raise ValueError('Unrelated production change: '+old.name)
            differences.extend(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),
                current.read_text(encoding='utf-8').splitlines(True),fromfile='baseline/'+old.name,tofile='cached/'+old.name))
    config=ProjectConfig.from_dict(original['configuration'])
    torch.set_num_threads(1)
    _,payload=load_model(source/'assets/checkpoint.pt',config,torch.device('cpu'),require_qualified=True)
    (root/'assets').mkdir(parents=True)
    for name in ('checkpoint.pt','orbit.npz',FIXED_MAP):
        target=root/'assets'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'assets'/name,target)
    shutil.copytree(project/'src/panama_uav',root/'code/panama_uav',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(source/'code/pilot_helper.py',root/'code/pilot_helper.py')
    for name in ('run_cached_fallback.py','report_cached_fallback.py','run_omega_sensitivity.py',
                 'report_omega_sensitivity.py','audit_omega_reproduction.py'):
        shutil.copy2(project/'scripts'/name,root/'code'/name)
    config=replace(config,project_root=root/'assets',
        communication=replace(config.communication,leo_timeline=str(root/'assets/orbit.npz')),
        resource_allocation=replace(config.resource_allocation,missed_command_fallback='cached_astar'))
    configs=root/'configs';configs.mkdir()
    weight_configs={}
    for w in expected_weights:
        before=(source/'configs'/(w['id'].lower()+'.toml')).read_text(encoding='utf-8')
        text=before.replace('timeout_action = 4','timeout_action = 4\nmissed_command_fallback = "cached_astar"')
        if text==before:raise ValueError('Cannot locate fallback insertion point')
        path=configs/(w['id'].lower()+'.toml');path.write_text(text,encoding='utf-8')
        weight_configs[w['id']]=dict(path=str(path),sha256=sha256(path),latency=w['latency'],energy=w['energy'])
    baseline=[]
    for path in sorted((source/'episodes').glob('*/*/*.result.json')):
        packet=core.read(path)
        if sha256(path)!=core.read(path.with_suffix('.meta.json'))['sha256']:
            raise ValueError('Historical episode transaction changed')
        baseline.append(dict(weight_id=packet['weight']['id'],condition=packet['row']['condition'],
            case_id=packet['row']['case_id'],packet=str(path),packet_sha256=sha256(path),
            archive=packet['audit']['archive'],archive_sha256=packet['audit']['archive_sha256']))
    if len(baseline)!=375:raise ValueError('The preserved baseline must include all 375 attempted missions')
    core.write(root/'baseline_index.json',baseline)
    core.write(root/'cases.json',cases)
    (root/'production_delta.patch').write_text(''.join(differences),encoding='utf-8')
    manifest=dict(original,protocol=PROTOCOL,created_utc=datetime.now(timezone.utc).isoformat(),
        project_root=str(project),source_root=str(source),source_manifest_sha256=sha256(source/'manifest.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=project,text=True).strip()),
        configuration=config.to_dict(),weight_configs=weight_configs,workers=workers,heavy_workers=heavy_workers,
        budget_seconds=budget_seconds,weights=expected_weights,conditions=list(core.CONDITIONS),
        expected_episodes=375,episode_cap=250,cases_per_population=5,populations=list(core.COUNTS),smoke=False,
        policy_updates=0,learner_updates=payload.get('learner_updates',0),
        protected_sha256={name:sha256(project/name) for name in subprocess.check_output(['git','ls-files'],cwd=project,text=True).splitlines()
            if name.startswith(('src/','configs/','maps/','results/','paper/','research/'))
            and not name.startswith('results/cached_fallback/') and (project/name).is_file()},
        variants={name:dict(fallback=mode,deadline_seconds=deadline) for name,(mode,deadline) in VARIANTS.items()},
        source_changes=changed,pilot_populations=[2,4,8],pilot_cases_per_population=1,pilot_expected_episodes=36,
        baseline_reuse='All 375 original 250 ms WAIT runs retained; nine rerun for exact default-behavior validation',
        pilot_reuse='Nine 250 ms cached-fallback validation cases form part of the full primary matrix',
        primary_cutoff_seconds=.25,sensitivity_cutoff_seconds=.1,control_interval_seconds=1.,
        energy_scope='Existing communication plus UAV computation; no robot movement energy or added local-controller energy',
        timing_reference=dict(original_primary_workflow_seconds=core.read(source/'status.json').get('elapsed_seconds'),
                              hard_wall_budget_seconds=budget_seconds))
    for variant,(mode,deadline) in VARIANTS.items():
        if variant=='primary':continue
        values=deepcopy(manifest)
        parameters=values['configuration']['resource_allocation']
        parameters.update(missed_command_fallback=mode,decision_budget_seconds=deadline,
            command_cutoff_seconds=deadline,task_deadline_seconds=deadline)
        values.update(variant=variant,parent_root=str(root),expected_episodes=9,pilot_only=True)
        core.write(variant_root(root,variant)/'manifest.json',values)
    manifest['frozen_sha256']={p.relative_to(root).as_posix():sha256(p)
        for folder in ('code','assets','configs') for p in (root/folder).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts}
    for name in ('cases.json','baseline_index.json','production_delta.patch'):
        manifest['frozen_sha256'][name]=sha256(root/name)
    for variant in ('wait250','wait100','cached100'):
        path=variant_root(root,variant)/'manifest.json'
        manifest['frozen_sha256'][path.relative_to(root).as_posix()]=sha256(path)
    core.write(root/'manifest.json',manifest)
    core.write(root/'status.json',dict(status='prepared',completed_episodes=0,expected_episodes=375))
    return manifest


def initialize(root):
    global STUDY_ROOT,MANIFEST
    STUDY_ROOT=Path(root);MANIFEST=verify(STUDY_ROOT)
    core.initialize(STUDY_ROOT)
    core.PROTOCOL=PROTOCOL


def episode(job):
    variant,weight,condition,case=job
    directory=variant_root(STUDY_ROOT,variant)
    core.ROOT=directory
    core.M=MANIFEST if variant=='primary' else core.read(directory/'manifest.json')
    core.H.LABEL='Parameter-only pilot results' if variant.startswith('wait') else 'Cached A* fallback results'
    try:
        value=core.episode((weight,condition,case))
        return dict(value,variant=variant)
    finally:
        core.ROOT,core.M=STUDY_ROOT,MANIFEST


def result_path(root,job):
    variant,weight,condition,case=job
    return core.result_path(variant_root(root,variant),(weight,condition,case))


def reproduce_wait(root,jobs):
    """Validate legacy execution fingerprints and conic diagnostic certificates."""
    index={(r['condition'],r['case_id']):r for r in core.read(Path(root)/'baseline_index.json') if r['weight_id']=='W1'}
    results=[]
    for job in jobs:
        _,_,condition,case=job
        reference=index[condition['name'],case['case_id']]
        if sha256(reference['packet'])!=reference['packet_sha256'] or sha256(reference['archive'])!=reference['archive_sha256']:
            raise ValueError('Preserved WAIT baseline changed')
        before,after=core.read(reference['packet']),core.read(result_path(root,job))
        matches={k:v==after['signatures'][k] for k,v in before['signatures'].items()}
        with gzip.open(reference['archive'],'rb') as stream:trace=pickle.load(stream)
        previous={e['new_job']:e['relaxed'] for e in trace['transport_events'] if e['event']=='paper_allocation'}
        del trace
        with gzip.open(after['audit']['archive'],'rb') as stream:trace=pickle.load(stream)
        current={e['new_job']:e['relaxed'] for e in trace['transport_events'] if e['event']=='paper_allocation'}
        del trace
        if previous.keys()!=current.keys():raise ValueError('Legacy snapshot IDs changed')
        certificates=[certificate(previous[k],current[k]) for k in previous]
        result=dict(case_id=case['case_id'],condition=condition['name'],matches=matches,
            fractional_certificates=len(certificates),failed_certificates=sum(not c['passed'] for c in certificates),
            passed=all(matches.values()) and all(c['passed'] for c in certificates))
        results.append(result)
        if not result['passed']:
            core.write(Path(root)/'baseline_validation.json',dict(status='failed',details=results))
            raise ValueError('Default WAIT execution changed: '+str(result))
    core.write(Path(root)/'baseline_validation.json',dict(status='passed',cases=len(results),
        exact_executed_records=True,details=results))


def run(root,stage):
    import psutil
    root=Path(root).resolve();manifest=verify(root,True)
    lease=core.read(root/'launch.json');deadline=lease['first_started_unix']+manifest['budget_seconds']
    cases=core.read(root/'cases.json')
    pilot_cases=[c for c in cases if c['num_agents'] in (2,4,8) and c['case_id'].endswith('-000')]
    def jobs(variant,weights,cases):
        return [(variant,w,c,case) for w in weights for c in manifest['conditions'] for case in cases]
    preflight=jobs('wait250',manifest['weights'][:1],pilot_cases)
    pilots=[('legacy_wait_reproduction',preflight),
            ('cached_250ms_pilot',jobs('primary',manifest['weights'][:1],pilot_cases)),
            ('paired_100ms_pilot',jobs('wait100',manifest['weights'][:1],pilot_cases)+jobs('cached100',manifest['weights'][:1],pilot_cases))]
    if stage=='full':
        if not (root/'pilot_validation.json').exists() or core.read(root/'pilot_validation.json')['status']!='passed':
            raise ValueError('Complete paired pilot validation is required before the primary matrix')
        phases=[('primary_250ms_matrix',jobs('primary',manifest['weights'],cases))]
    else:phases=pilots
    errors=list((root/'episodes').glob('*/*/*.error.json'))+list((root/'validation').glob('*/episodes/*/*/*.error.json'))
    if errors:raise ValueError('Diagnose recorded errors before continuation: '+str(errors[0]))
    def status(phase,**values):
        primary=len(list((root/'episodes').glob('*/*/*.result.json')))
        core.write(root/'status.json',dict(status='running',stage=phase,completed_episodes=primary,
            expected_episodes=375,updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.time()-lease['first_started_unix'],remaining_seconds=max(0.,deadline-time.time()),**values))
    try:
        for phase,work in phases:
            queue=[]
            for job in work:
                path=result_path(root,job)
                if path.exists():
                    if sha256(path)!=core.read(path.with_suffix('.meta.json'))['sha256']:raise ValueError('A committed result changed')
                else:queue.append(job)
            status(phase)
            with ProcessPoolExecutor(max_workers=manifest['workers'],initializer=initialize,initargs=(str(root),)) as executor:
                pending={}
                while queue or pending:
                    if time.time()>=deadline:raise TimeoutError('Fixed experiment wall budget exhausted')
                    heavy=sum(j[3]['num_agents']>=16 for j in pending.values())
                    while len(pending)<manifest['workers'] and psutil.virtual_memory().available>5e9:
                        index=next((i for i,j in enumerate(queue) if j[3]['num_agents']<16 or heavy<manifest['heavy_workers']),None)
                        if index is None:break
                        job=queue.pop(index);pending[executor.submit(episode,job)]=job
                        heavy+=job[3]['num_agents']>=16
                    if not pending:raise MemoryError('Insufficient memory for a diagnostic worker')
                    done,_=wait(pending,timeout=45,return_when=FIRST_COMPLETED)
                    for future in done:
                        value=future.result();del pending[future]
                        status(phase,latest=value)
            if phase=='legacy_wait_reproduction':
                status('legacy_wait_audit');reproduce_wait(root,preflight)
        verify(root,True)
        from report_cached_fallback import pilot_report,report_study
        if stage=='pilot':
            result=pilot_report(root)
            if result['status']!='passed':raise ValueError('Pilot validation failed')
            core.write(root/'status.json',dict(status='pilot_complete',completed_episodes=9,expected_episodes=375,
                elapsed_seconds=time.time()-lease['first_started_unix'],next_stage='primary_250ms_matrix'))
        else:
            status('reporting');report_study(root)
            core.write(root/'status.json',dict(status='complete',completed_episodes=375,expected_episodes=375,
                elapsed_seconds=time.time()-lease['first_started_unix'],policy_updates=0))
    except Exception as error:
        core.write(root/'status.json',dict(status='error',stage=stage,
            completed_episodes=len(list((root/'episodes').glob('*/*/*.result.json'))),expected_episodes=375,
            elapsed_seconds=time.time()-lease['first_started_unix'],error=str(error),traceback=traceback.format_exc()))
        raise


def launch(root,stage='pilot',resume=False):
    import psutil
    root=Path(root).resolve();manifest=verify(root,True)
    previous=core.read(root/'launch.json') if (root/'launch.json').exists() else None
    if previous:
        if not resume:raise ValueError('Use --resume only after confirming the previous worker stopped')
        for process in psutil.process_iter(['pid','create_time','cmdline']):
            argv=process.info['cmdline'] or []
            if '--worker' in argv and str(root) in argv and any('run_cached_fallback.py' in a for a in argv):
                raise ValueError('A matching fallback-study worker is still running')
        if time.time()-previous['first_started_unix']>=manifest['budget_seconds']:raise ValueError('No wall budget remains')
    elif stage!='pilot':raise ValueError('Run the validation pilot first')
    if core.read(root/'status.json')['status']=='complete':raise ValueError('Experiment already complete')
    command=[sys.executable,str(root/'code/run_cached_fallback.py'),'--worker',str(root),'--stage',stage]
    env=dict(os.environ,PYTHONPATH=str(root/'code'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (root/'stdout.log').open('a') as out,(root/'stderr.log').open('a') as err:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    info=dict(pid=process.pid,creation_time=psutil.Process(process.pid).create_time(),command=command,stage=stage,
        first_started_unix=previous['first_started_unix'] if previous else time.time())
    core.write(root/'launch.json',info)
    return info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--launch',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--worker',type=Path)
    parser.add_argument('--stage',choices=('pilot','full'),default='pilot')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--heavy-workers',type=int,default=2)
    parser.add_argument('--budget-seconds',type=float,default=86400.)
    parser.add_argument('--report',action='store_true');args=parser.parse_args()
    if args.worker:run(args.worker,args.stage)
    else:
        if args.prepare:prepare(args.root,args.source,workers=args.workers,heavy_workers=args.heavy_workers,budget_seconds=args.budget_seconds)
        if args.launch:print(json.dumps(launch(args.root,args.stage,args.resume),indent=2))
        if args.report:
            from report_cached_fallback import report_study
            report_study(args.root)
