"""Independent episode, exposure, publication and replay audits for cached fallback."""
from collections import Counter,defaultdict
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import zipfile

import numpy as np
import run_cached_fallback as study
import report_cached_fallback as reporting
from panama_uav.records import sha256


def verify_publication(root):
    root=Path(root)
    for row in csv.DictReader((root/'files.csv').open(encoding='utf-8')):
        path=root/row['path']
        if path.stat().st_size!=int(row['bytes']) or sha256(path)!=row['sha256']:
            raise ValueError('Publication file changed: '+row['path'])
    manifest=study.core.read(root/'manifest.json');publication=study.core.read(root/'publication.json')
    episodes=list(csv.DictReader((root/'episodes.csv').open(encoding='utf-8')))
    if any(r['success']!='True' for r in episodes):raise ValueError('Unsuccessful episode in conditional publication')
    ids={(r['weight_id'],r['route'],r['case_id']) for r in episodes}
    if len(ids)!=len(episodes) or ids!={tuple(i) for i in publication['included_identities']}:
        raise ValueError('Conditional episode identities mismatch')
    totals=Counter();counts=defaultdict(Counter)
    with gzip.open(root/'tasks.csv.gz','rt',encoding='utf-8',newline='') as stream:
        for task in csv.DictReader(stream):
            identity=task['weight_id'],task['route'],task['case_id']
            if identity not in ids:raise ValueError('Task outside completed cohort')
            if task['deadline_status'] not in ('on_time','late','undelivered'):raise ValueError('Invalid command partition')
            key=task['weight_id'],task['route'],int(task['num_agents'])
            counts[key].update(tasks=1,uav=int(task['executed_x']=='1'),cloud=int(task['executed_x']=='0'))
            counts[key][task['deadline_status']]+=1
            totals.update(tasks=1)
    summary=list(csv.DictReader((root/'summary.csv').open(encoding='utf-8')))
    if len(summary)!=75:raise ValueError('Missing primary summary cells')
    for row in summary:
        key=row['weight_id'],row['route'],int(row['num_agents'])
        actual=[r for r in episodes if (r['weight_id'],r['route'],int(r['num_agents']))==key]
        if int(row['included_cases'])!=len(actual):raise ValueError('Incorrect conditional denominator')
        for field,counter in (('tasks','tasks'),('uav_tasks','uav'),('cloud_tasks','cloud'),
                              ('on_time_commands','on_time'),('late_commands','late'),('undelivered_commands','undelivered')):
            if int(row[field])!=counts[key][counter]:raise ValueError('Conditional task total mismatch')
        if not actual:
            for field in ('mean_episode_energy_j','observed_latency_p95_seconds','xi_fov','attempted_vertex_conflicts_mean'):
                if row[field]!='':raise ValueError('Empty cohort acquired a synthetic outcome')
        else:
            expected=sum(float(r['modeled_energy_lower_j']) for r in actual)/len(actual)
            if abs(expected-float(row['mean_episode_energy_j']))>1e-8:raise ValueError('Conditional energy denominator mismatch')
    rates=list(csv.DictReader((root/'mission_success_rates.csv').open(encoding='utf-8')))
    if len(rates)!=150 or any(int(r['attempted_missions'])!=5 for r in rates):
        raise ValueError('Mission success must include all five attempted cases')
    for row in rates:
        if row['fallback']=='cached_astar':
            key=row['weight_id'],row['route'],int(row['num_agents'])
            n=sum((r['weight_id'],r['route'],int(r['num_agents']))==key for r in episodes)
            if int(row['completed_missions'])!=n:raise ValueError('Mission-success numerator mismatch')
    if totals['tasks']!=publication['included_tasks']:raise ValueError('Published task count mismatch')
    for name in ('charts.zip','results.zip'):
        with zipfile.ZipFile(root/name) as archive:
            if archive.testzip() is not None:raise ValueError('ZIP integrity failure')
            for name in archive.namelist():
                if 'mission_success.' in name or 'private_figures' in name:raise ValueError('Local success plot leaked into publication')
                if archive.read(name)!=(root/name).read_bytes():raise ValueError('ZIP contains an outdated artifact')
    return dict(status='passed',included_episodes=len(episodes),included_tasks=totals['tasks'],
        summary_cells=len(summary),mission_success_cells=len(rates),mission_success_figures_exported=0)


def audit(root,read_back=False):
    root=Path(root).resolve();manifest=study.verify(root,True)
    if study.core.read(root/'status.json')['status']!='complete':raise ValueError('The primary study is not complete')
    cases={c['case_id']:c for c in study.core.read(root/'cases.json')}
    paths=sorted((root/'episodes').glob('*/*/*.result.json'))
    if len(paths)!=375:raise ValueError('Incomplete primary matrix')
    total=Counter();inventory=[];identities=set()
    selected=[]
    for path in paths:
        data=reporting.packet(path,cases[path.name.removesuffix('.result.json')]);row=data['row'];w=data['weight']['id']
        unique=w,row['route'],row['case_id']
        if unique in identities:raise ValueError('Duplicate primary identity')
        identities.add(unique)
        if data['manifest_sha256']!=sha256(root/'manifest.json'):raise ValueError('Wrong primary manifest')
        if sha256(data['audit']['archive'])!=data['audit']['archive_sha256']:raise ValueError('Trace archive changed')
        inventory.append((unique,sha256(path),data['audit']['archive_sha256']))
        total.update(episodes=1,tasks=row['tasks_generated'],successes=int(row['success']),
            on_time=row['deadline_commands_on_time'],late=row['deadline_commands_late'],undelivered=row['deadline_commands_undelivered'],
            fallback_decisions=row['cached_fallback_decisions'],fallback_moves=row['cached_fallback_motion_steps'])
        if w in ('W1','W5') and row['case_id'].endswith('-000'):selected.append(path)
    if read_back:
        for variant in ('wait100','cached100'):
            selected.extend(sorted((root/'validation'/variant/'episodes/W1').glob('*/*.result.json')))
    records=[]
    evidence=root/'independent_readbacks.json'
    fingerprint=hashlib.sha256(json.dumps(sorted(inventory),separators=(',',':')).encode()).hexdigest()
    if read_back:
        import torch
        from panama_uav.training import load_model
        from panama_uav.models import select_action
        from panama_uav.staleness_audit import audit_stress
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        model,_=load_model(root/'assets/checkpoint.pt',study.ProjectConfig.from_dict(manifest['configuration']),torch.device('cpu'),require_qualified=True)
        def infer(location,robot,observation):return select_action(model,observation,0.,np.random.default_rng(0),torch.device('cpu'))
        for path in selected:
            data=study.core.read(path)
            if sha256(data['audit']['archive'])!=data['audit']['archive_sha256']:raise ValueError('Pilot/primary archive changed')
            with gzip.open(data['audit']['archive'],'rb') as stream:trace=pickle.load(stream)
            result=audit_stress(trace,infer)
            fallback=sum(e.get('cached_fallback',False) for e in trace['remote_events'] if e['event']=='decision_event')
            record=dict(packet=str(path.relative_to(root)),archive_sha256=data['audit']['archive_sha256'],
                cached_decisions_verified=fallback,**result)
            records.append(record)
            study.core.write(root/'readback_progress.json',dict(status='running',completed=len(records),expected=len(selected),latest=record))
            del trace
        study.core.write(evidence,dict(status='passed',primary_inventory_sha256=fingerprint,records=records))
    elif evidence.exists():
        saved=study.core.read(evidence)
        if saved['primary_inventory_sha256']!=fingerprint:raise ValueError('Readback evidence belongs to changed primary results')
        records=saved['records']
    result=dict(status='passed',totals=dict(total),archive_hashes_checked=375,primary_inventory_sha256=fingerprint,
        full_trace_readbacks=len(records),cached_decisions_verified=sum(r['cached_decisions_verified'] for r in records),
        publication=verify_publication(root/'publication'),protected_source_and_history_unchanged=True)
    study.core.write(root/'independent_audit.json',result)
    return result


def export(root,destination):
    root,destination=Path(root).resolve(),Path(destination).resolve()
    evidence=study.core.read(root/'independent_audit.json')
    if evidence['status']!='passed' or evidence['full_trace_readbacks']!=48:
        raise ValueError('All 30 primary and 18 sensitivity trace readbacks are required')
    if destination.exists():raise ValueError('Preserve existing publications; use a new destination')
    shutil.copytree(root/'publication',destination)
    for name in ('independent_audit.json','independent_readbacks.json','regression.json','parameter_isolation.json'):
        shutil.copy2(root/name,destination/name)
    study.core.write(destination/'pilot_configurations.json',{variant:study.core.read(root/'validation'/variant/'manifest.json')['configuration']
        for variant in ('wait250','wait100','cached100')})
    from publish_omega_completed import seal_bundle
    seal_bundle(destination)
    return verify_publication(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--read-back',action='store_true');parser.add_argument('--export',type=Path);args=parser.parse_args()
    print(json.dumps(audit(args.root,args.read_back),indent=2))
    if args.export:print(json.dumps(export(args.root,args.export),indent=2))
