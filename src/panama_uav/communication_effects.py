"""Paired frozen-policy SAGIN inference simulations and reproducible telemetry."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from .config import FIXED_MAP, ProjectConfig, load_config
from .evaluation import run_episode
from .inference_controls import InferenceCondition, condition_grid, inference_class
from .records import append_jsonl, sha256, write_json
from .resource_study import study_cases
from .scenarios import Scenario
from .staleness_audit import audit_stress
from .staleness_study import stress_metrics

PROTOCOL='frozen_panama_communication_effects_v1'
COUNTS=(2,4,6,8)


def condition_config(base,condition,root,case=None):
    c=base.communication
    if c.seconds_per_step != 1 or base.resource_allocation.enforce_task_deadlines:
        raise ValueError('Use one-second control and monitored deadlines without solver constraints')
    offset=c.backhaul_time_offset_seconds+(case['orbital_offset_seconds'] if case else 0.)
    return replace(base,communication=replace(c,
        profile='perfect' if condition.route=='ideal' else 'physical',
        mission_route='cloud' if condition.route in ('cloud','allocated') else 'uav',
        backhaul_time_offset_seconds=offset),resource_allocation=replace(base.resource_allocation,
        failure_directory=str(Path(root)/condition.name/'failures')))


def validate_configuration(config):
    p=config.resource_allocation
    if not p.enabled or p.execution_mode!='remote_dt' or not p.oracle_shadow:
        raise ValueError('Frozen remote-DT execution and oracle diagnostics are required')
    if not p.remote_acceptance_approved:
        raise ValueError('Existing remote execution acceptance gate is closed')
    if (p.decision_budget_seconds,p.command_cutoff_seconds,p.task_deadline_seconds)!=(.25,.25,.25):
        raise ValueError('Use consistent 250 ms cutoffs')
    if config.communication.seconds_per_step!=1 or p.enforce_task_deadlines:
        raise ValueError('Keep one-second control and hard solver deadlines disabled')
    if not config.communication.leo_timeline:
        raise ValueError('An archived orbital timeline is required')


def measurements(row,trace,condition):
    row=stress_metrics(row,trace)
    tasks=[];by_cycle=defaultdict(list)
    identity=dict(case_id=row['case_id'],condition=condition.name,route=condition.route,
                  num_agents=row['num_agents'])
    for original in trace['resource_tasks']:
        task={k:v for k,v in original.items() if not isinstance(v,(dict,list))}
        computed=original.get('compute_completed_at')
        arrived=original.get('compute_arrived_at')
        source=original['source_time']
        task.update(identity,inference_ready_seconds=computed-source if computed is not None else None,
            ingress_seconds=arrived-source if arrived is not None else None,
            processing_seconds=computed-arrived if computed is not None and arrived is not None else None,
            command_delivery_seconds=original.get('observed_latency_seconds'),
            egress_seconds=original['delivered_at']-computed if 'delivered_at' in original else None,
            cycle=original['report']['cycle'])
        tasks.append(task);by_cycle[task['cycle']].append(task)
    rounds=[]
    for cycle,group in sorted(by_cycle.items()):
        ready=[t['inference_ready_seconds'] for t in group if t['inference_ready_seconds'] is not None]
        delivered=[t['command_delivery_seconds'] for t in group if t['command_delivery_seconds'] is not None]
        rounds.append(dict(identity,cycle=cycle,tasks=len(group),inferences_ready=len(ready),
            delivered_commands=len(delivered),
            inference_round_seconds=max(ready) if len(ready)==len(group) else None,
            command_round_seconds=max(delivered) if len(delivered)==len(group) else None,
            on_time_commands=sum(t['deadline_status']=='on_time' for t in group),
            round_deadline_met=all(t['deadline_status']=='on_time' for t in group)))
    def percentile(values):
        return float(np.quantile(values,.95)) if values else None
    horizon=trace['frames'][-1]['time']
    transmit=[e for e in trace['transport_events'] if e['event']=='remote_transmit']
    row.update(identity,axis=condition.axis,level=condition.level,
        makespan_seconds=float(row['steps']) if row['success'] else None,
        waiting_completed_steps=row['waiting_steps'] if row['success'] else None,
        inference_ready_p95_seconds=percentile([t['inference_ready_seconds'] for t in tasks if t['inference_ready_seconds'] is not None]),
        inference_round_p95_seconds=percentile([r['inference_round_seconds'] for r in rounds if r['inference_round_seconds'] is not None]),
        command_round_p95_seconds=percentile([r['command_round_seconds'] for r in rounds if r['command_round_seconds'] is not None]),
        rounds=len(rounds),rounds_on_time=sum(r['round_deadline_met'] for r in rounds),
        observed_seconds=horizon,
        drone_busy_seconds=row['mean_drone_cpu_utilization']*horizon,
        cloud_busy_seconds=row['mean_cloud_cpu_utilization']*horizon,
        robot_tx_energy_j=sum(e['energy_j'] for e in transmit if e['link']=='access_up'),
        drone_tx_energy_j=sum(e['energy_j'] for e in transmit if e['link']=='satellite_up'))
    if sum(row[k] for k in ('deadline_commands_on_time','deadline_commands_late','deadline_commands_undelivered'))!=len(tasks):
        raise ValueError('Command arrival partition mismatch')
    if row['deadline_cutoff_wait_tasks']!=row['command_cutoff_misses']:
        raise ValueError('Cutoff WAIT denominator mismatch')
    expected_x=0 if condition.route=='cloud' else 1
    if condition.route!='allocated' and any(t.get('executed_x') not in (None,expected_x) for t in tasks):
        raise ValueError('A task changed its fixed execution location')
    return row,tasks,rounds


def _initialize(root):
    import torch
    from .training import load_model
    global _ROOT,_CONFIG,_MODEL,_VERSION,_MANIFEST
    _ROOT=Path(root);_MANIFEST=json.loads((_ROOT/'manifest.json').read_text())
    _CONFIG=ProjectConfig.from_dict(_MANIFEST['configuration'])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
    _MODEL,payload=load_model(_ROOT/'assets/checkpoint.pt',_CONFIG,torch.device('cpu'),
                            require_qualified=not _MANIFEST['diagnostic'])
    _VERSION=payload.get('learner_updates',0)


def _infer(location,robot,observation):
    import torch
    from .models import select_action
    return select_action(_MODEL,observation,0.,np.random.default_rng(0),torch.device('cpu'))


def _episode(argument):
    import torch
    condition_data,case,cap=argument
    condition=InferenceCondition(**condition_data)
    folder=_ROOT/'episodes'/condition.name;folder.mkdir(parents=True,exist_ok=True)
    progress=_ROOT/f'worker-{os.getpid()}.json'
    write_json(progress,dict(status='running',condition=condition.name,case_id=case['case_id']))
    started=time.monotonic()
    try:
        config=condition_config(_CONFIG,condition,_ROOT,case)
        scenario=Scenario(Path(case['case_id']),Path(case['map_path']),
            tuple(map(tuple,case['starts'])),tuple(map(tuple,case['goals'])))
        row,trace=run_episode(config,_MODEL,case,torch.device('cpu'),_VERSION,max_steps=cap,
            scenario=scenario,remote_communication_class=inference_class(condition))
        if trace['frames'][0]['positions']!=case['starts'] or trace['goals']!=case['goals']:
            raise ValueError('Paired start/goal identity changed')
        row,tasks,rounds=measurements(row,trace,condition)
        trace['study_condition']=condition_data
        trace['provenance']=dict(protocol=PROTOCOL,case_id=case['case_id'],
            model_sha256=_MANIFEST['model_sha256'],timeline_sha256=_MANIFEST['timeline_sha256'],
            map_sha256=case['map_sha256'],case_bank_sha256=_MANIFEST['case_bank_sha256'],
            manifest_sha256=sha256(_ROOT/'manifest.json'))
        trace['episode_metrics']=row
        archive=folder/f"{case['case_id']}.json.gz"
        encoded=json.dumps(trace,allow_nan=False,separators=(',',':')).encode()
        pending=archive.with_suffix('.gz.pending')
        with gzip.open(pending,'wb',compresslevel=3) as stream:stream.write(encoded)
        os.replace(pending,archive)
        del encoded
        audit=audit_stress(trace,_infer)
        audit.update(condition=condition.name,case_id=case['case_id'],archive=str(archive),
            archive_sha256=sha256(archive),wall_seconds=time.monotonic()-started)
        result_path=folder/f"{case['case_id']}.result.json"
        pending_result=result_path.with_suffix('.json.pending')
        write_json(pending_result,dict(row=row,tasks=tasks,rounds=rounds,audit=audit,
                                      manifest_sha256=sha256(_ROOT/'manifest.json')))
        os.replace(pending_result,result_path)
        write_json(progress,dict(status='complete',condition=condition.name,case_id=case['case_id']))
        return row,tasks,rounds,audit
    except Exception as exc:
        write_json(folder/f"{case['case_id']}.error.json",dict(error=str(exc),traceback=traceback.format_exc(),
            condition=condition_data,case=case,snapshot=getattr(exc,'snapshot',None)))
        raise


def prepare(root,config,model,*,cases=10,counts=COUNTS,cap=250,workers=4,diagnostic=False,budget_seconds=14400):
    root=Path(root).resolve();model=Path(model).resolve()
    validate_configuration(config)
    if root.exists():raise ValueError('Use a new output directory; existing experiments are immutable')
    if cases<1 or workers<1 or not 1<=cap<=250 or not set(counts)<=set(COUNTS):
        raise ValueError('Invalid paired-case, population, worker or episode limits')
    if budget_seconds<=0:raise ValueError('Use a positive bounded runtime')
    import torch
    from .training import load_model
    torch.set_num_threads(1)
    _,payload=load_model(model,config,torch.device('cpu'),require_qualified=not diagnostic)
    assets=root/'assets';(assets/'maps').mkdir(parents=True)
    shutil.copy2(model,assets/'checkpoint.pt')
    shutil.copy2(config.resolve(FIXED_MAP),assets/FIXED_MAP)
    shutil.copy2(config.resolve(config.communication.leo_timeline),assets/'orbit.npz')
    frozen=replace(config,project_root=assets,
        communication=replace(config.communication,leo_timeline=str(assets/'orbit.npz')))
    bank=study_cases(frozen,cases,tuple(counts),cap)
    for case in bank:
        case.update(map_path=str(assets/FIXED_MAP),map_sha256=sha256(assets/FIXED_MAP),scenario='random_mapf')
    write_json(root/'paired_cases.json',bank)
    source=Path(__file__).resolve().parent
    shutil.copytree(source,root/'code/panama_uav',ignore=shutil.ignore_patterns('__pycache__'))
    conditions=condition_grid()
    project=config.project_root
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=project,text=True).strip()
    manifest=dict(protocol=PROTOCOL,created_utc=datetime.now(timezone.utc).isoformat(),
        configuration=frozen.to_dict(),source_commit=commit,source_dirty=bool(dirty),
        code_sha256={p.name:sha256(p) for p in (root/'code/panama_uav').glob('*.py')},
        model_origin=str(model),model_sha256=sha256(assets/'checkpoint.pt'),
        model_qualification=payload.get('qualification',[]),diagnostic=diagnostic,
        timeline_sha256=sha256(assets/'orbit.npz'),map_sha256=sha256(assets/FIXED_MAP),
        case_bank_sha256=sha256(root/'paired_cases.json'),populations=list(counts),cases_per_population=cases,
        episode_cap=cap,workers=workers,budget_seconds=budget_seconds,conditions=[asdict(c) for c in conditions],
        expected_episodes=len(bank)*len(conditions),python=sys.version,platform=platform.platform(),
        inference_device='cpu',policy_updates=0,bootstrap='Fresh initial state briefing; no free movement commands',
        packages={name:importlib.metadata.version(name) for name in
                  ('numpy','scipy','torch','matplotlib','simpy','psutil','cvxpy','clarabel')},
        cpu_threads_per_worker=1,
        intervention_label='Controlled inference sensitivity; injected delays/outages are experimental assumptions',
        compute_overhead='Additional post-CPU inference service latency; original CPU cycles/capacity and sharing remain fixed',
        energy_scope='Robot access uplink, UAV backhaul uplink, and configured UAV CPU model; other platform power unmodeled')
    write_json(root/'manifest.json',manifest)
    write_json(root/'status.json',dict(status='prepared',completed_episodes=0,expected_episodes=manifest['expected_episodes']))
    return manifest


def verify_inputs(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if m['protocol']!=PROTOCOL:raise ValueError('Unknown study protocol')
    for path,key in ((root/'assets/checkpoint.pt','model_sha256'),(root/'assets/orbit.npz','timeline_sha256'),
                     (root/'assets'/FIXED_MAP,'map_sha256'),(root/'paired_cases.json','case_bank_sha256')):
        if sha256(path)!=m[key]:raise ValueError(f'Frozen input changed: {path}')
    if any(sha256(root/'code/panama_uav'/name)!=value for name,value in m['code_sha256'].items()):
        raise ValueError('Frozen worker source changed')
    return m


def run(root):
    root=Path(root).resolve();manifest=verify_inputs(root)
    bank=json.loads((root/'paired_cases.json').read_text())
    committed=[];manifest_hash=sha256(root/'manifest.json')
    for path in sorted((root/'episodes').glob('*/*.result.json')):
        packet=json.loads(path.read_text())
        if packet['manifest_sha256']!=manifest_hash:raise ValueError('Committed result uses another manifest')
        committed.append(packet)
    done=[packet['audit'] for packet in committed]
    keys={(r['condition'],r['case_id']) for r in done}
    if len(keys)!=len(done) or any(a['status']!='passed' or sha256(a['archive'])!=a['archive_sha256'] for a in done):
        raise ValueError('Invalid completed-episode ledger')
    if committed:
        # Raw per-case transactions are authoritative; interrupted aggregate writes
        # are reconstructed without rerunning or duplicating completed episodes.
        for name,field in (('episodes','row'),('tasks','tasks'),('rounds','rounds'),('audits','audit')):
            temporary=root/(name+'.jsonl.pending')
            with temporary.open('w',encoding='utf-8') as stream:
                for packet in committed:
                    records=packet[field] if isinstance(packet[field],list) else [packet[field]]
                    for record in records:stream.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
            os.replace(temporary,root/(name+'.jsonl'))
    prior=json.loads((root/'status.json').read_text())
    charged=prior.get('wall_seconds',0.)
    started=time.monotonic()
    arguments=[(c,case,manifest['episode_cap']) for case in bank for c in manifest['conditions']
               if (c['name'],case['case_id']) not in keys]
    def status(state,**values):
        write_json(root/'status.json',dict(status=state,completed_episodes=len(done),
            expected_episodes=manifest['expected_episodes'],wall_seconds=charged+time.monotonic()-started,
            updated_utc=datetime.now(timezone.utc).isoformat(),**values))
    status('running')
    try:
        # Bounded submission prevents a slow consumer from retaining large traces.
        from collections import deque
        with ProcessPoolExecutor(max_workers=manifest['workers'],initializer=_initialize,initargs=(str(root),)) as executor:
            pending=deque();iterator=iter(arguments)
            for _ in range(2*manifest['workers']):
                arg=next(iterator,None)
                if arg is not None:pending.append(executor.submit(_episode,arg))
            while pending:
                row,tasks,rounds,audit=pending.popleft().result()
                append_jsonl(root/'episodes.jsonl',row)
                for name,records in (('tasks',tasks),('rounds',rounds)):
                    for record in records:append_jsonl(root/(name+'.jsonl'),record)
                append_jsonl(root/'audits.jsonl',audit);done.append(audit)
                status('running',latest_case=audit['case_id'],latest_condition=audit['condition'])
                if charged+time.monotonic()-started>manifest['budget_seconds']:
                    raise TimeoutError('Recorded inference-study runtime limit reached')
                arg=next(iterator,None)
                if arg is not None:pending.append(executor.submit(_episode,arg))
        from .communication_report import report
        report(root)
        status('complete')
    except Exception as exc:
        status('error',error=str(exc));raise


def launch(root):
    import psutil
    root=Path(root).resolve();verify_inputs(root)
    previous=root/'launch.json'
    if previous.exists():
        p=json.loads(previous.read_text())
        if psutil.pid_exists(p['pid']) and abs(psutil.Process(p['pid']).create_time()-p['creation_time'])<1:
            raise ValueError('This study already has an active worker')
    env=dict(os.environ,PYTHONPATH=str(root/'code'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,'-m','panama_uav.communication_effects','--worker',str(root)]
    with (root/'worker.stdout.log').open('a') as out,(root/'worker.stderr.log').open('a') as err:
        proc=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    record=dict(pid=proc.pid,creation_time=psutil.Process(proc.pid).create_time(),command=command,
                launched_utc=datetime.now(timezone.utc).isoformat())
    write_json(previous,record);return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--config',type=Path,default=Path('configs/paper_remote_dt_250ms.toml'))
    parser.add_argument('--model',type=Path)
    parser.add_argument('--cases',type=int,default=10)
    parser.add_argument('--counts',type=int,nargs='+',default=COUNTS)
    parser.add_argument('--cap',type=int,default=250)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--budget-seconds',type=int,default=14400)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--diagnostic',action='store_true')
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:return run(args.worker)
    if args.root is None:parser.error('--root is required')
    if args.prepare:
        if args.model is None:parser.error('--model is required for preparation')
        prepare(args.root,load_config(args.config),args.model,cases=args.cases,counts=args.counts,
            cap=args.cap,workers=args.workers,diagnostic=args.diagnostic,budget_seconds=args.budget_seconds)
    if args.launch:print(json.dumps(launch(args.root)))
    if args.report:
        from .communication_report import report
        report(args.root)


if __name__=='__main__':main()
