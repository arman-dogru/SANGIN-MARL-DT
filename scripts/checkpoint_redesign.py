"""Incremental, compact publication of committed and audited experiment results.

Raw traces remain local. Failed missions remain in the episode inventory;
conditional performance plots use successful missions and leave empty cells NA.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np

from run_redesign_accelerated import read,write,digest,key,phases

PROJECT=Path(__file__).resolve().parents[1]
CODE_FILES=('EXPERIMENT_REDESIGN.md','.gitattributes',
    'src/panama_uav/consistency.py','src/panama_uav/remote_audit.py','src/panama_uav/staleness_audit.py',
    'src/panama_uav/bundled_actions.py','src/panama_uav/frequency_motion.py','src/panama_uav/redesign.py',
    'src/panama_uav/redesign_audit.py','src/panama_uav/redesign_metrics.py',
    'scripts/run_redesign.py','scripts/report_redesign.py','scripts/record_redesign_validation.py',
    'scripts/run_redesign_accelerated.py','scripts/checkpoint_redesign.py',
    'tests/test_bundled_actions.py','tests/test_frequency_motion.py','tests/test_redesign_workflow.py',
    'tests/test_redesign_acceleration.py')


def ratio(a,b):return a/b if b else None
def csv_file(path,rows):
    import csv
    fields=sorted({k for r in rows for k in r}) or ['case_id']
    with Path(path).open('w',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def cached_episode(path,cache,parent_hash):
    marker=read(path.with_name('complete.json'));name=marker['sha256'];saved=cache/(name+'.json')
    if saved.exists():return read(saved)
    if digest(path)!=name:raise ValueError('Changed committed episode: '+str(path))
    packet=read(path)
    if packet['manifest_sha256']!=parent_hash or packet['audit']['status']!='passed':raise ValueError('Unverified episode provenance')
    job=packet['job'];row=packet['row'];tasks=packet['tasks'];statuses=Counter(t['deadline_status'] for t in tasks)
    if sum(statuses[s] for s in ('on_time','late','undelivered'))!=len(tasks):raise ValueError('Deadline partition does not reconcile')
    assigned=[t['executed_x'] for t in tasks if t.get('executed_x') is not None]
    if any(x not in (0,1) for x in assigned):raise ValueError('Nonbinary executed placement')
    latencies=np.array([t['observed_latency_seconds'] for t in tasks if t.get('delivered_at') is not None],dtype=np.float64)
    np.save(cache/(name+'.npy'),latencies,allow_pickle=False)
    result={k:v for k,v in row.items() if not isinstance(v,(dict,list))}
    result.update({k:job[k] for k in ('treatment','weight_id','route','num_agents','case_id')})
    result.update(result_sha256=name,audit_status='passed',tasks=len(tasks),on_time=statuses['on_time'],late=statuses['late'],
        undelivered=statuses['undelivered'],undelivered_overdue=sum(bool(t.get('deadline_undelivered_overdue')) for t in tasks),
        pending_censored=sum(bool(t.get('deadline_pending')) for t in tasks),delivered_samples=len(latencies),
        uav_tasks=assigned.count(1),cloud_tasks=assigned.count(0),unassigned_tasks=len(tasks)-len(assigned),
        completed_wall_seconds=marker['wall_seconds'],peak_rss_bytes=marker['peak_rss_bytes'])
    write(saved,result);return result


def summaries(rows,cache):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in ('treatment','weight_id','route','num_agents'))].append(row)
    output=[]
    for identity,allrows in sorted(groups.items()):
        selected=[r for r in allrows if r['success']]
        item=dict(zip(('treatment','weight_id','route','num_agents'),identity))
        item.update(attempted=len(allrows),successful=len(selected),failed=len(allrows)-len(selected),conditional_cohort='successful_missions')
        additive=('tasks','on_time','late','undelivered','undelivered_overdue','pending_censored','uav_tasks','cloud_tasks','unassigned_tasks',
            'eligible_decisions','cached_fallback_decisions','fallback_movements','movement_priority_reversals',
            'relevant_movement_priority_reversals','relevant_pairs','fov_errors','fov_uplink_errors','fov_downlink_errors',
            'active_peer_aoi_sum','active_peer_aoi_samples','attempted_vertex_conflicts','attempted_edge_swaps','post_resolution_conflicts')
        for field in additive:item[field]=sum(r.get(field,0) or 0 for r in selected)
        values=np.concatenate([np.load(cache/(r['result_sha256']+'.npy'),allow_pickle=False) for r in selected]) if selected else np.array([])
        item.update(delivered_samples=len(values),e2e_mean_seconds=float(values.mean()) if len(values) else None,
            e2e_p95_seconds=float(np.quantile(values,.95)) if len(values) else None,
            deadline_miss_rate=ratio(item['late']+item['undelivered_overdue'],item['on_time']+item['late']+item['undelivered_overdue']),
            uav_fraction=ratio(item['uav_tasks'],item['uav_tasks']+item['cloud_tasks']),
            fallback_rate=ratio(item['cached_fallback_decisions'],item['eligible_decisions']),
            xi_fov=ratio(item['fov_errors'],item['relevant_pairs']),
            xi_fov_ul=ratio(item['fov_uplink_errors'],item['relevant_pairs']),xi_fov_dl=ratio(item['fov_downlink_errors'],item['relevant_pairs']),
            active_peer_aoi_seconds=ratio(item['active_peer_aoi_sum'],item['active_peer_aoi_samples']))
        for name,field in (('energy_j','modeled_energy_lower_j'),('makespan_seconds','makespan_seconds'),('waiting_steps','waiting_steps')):
            available=[r[field] for r in selected if r.get(field) is not None]
            item[name]=float(np.mean(available)) if available else None
        duration=sum(r['observed_seconds'] for r in selected)
        for pool in ('drone','cloud'):item[pool+'_cpu_utilization']=ratio(sum(r.get(pool+'_utilization_seconds',0) for r in selected),duration)
        output.append(item)
    return output


def figures(folder,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'allocated':'#2878B5','edge':'#E59637','cloud':'#49A078'}
    labels={'allocated':'Allocator','edge':'All UAV','cloud':'All cloud'}
    table={(r['treatment'],r['route'],r['num_agents']):r for r in summary if r['weight_id']=='W1'}
    for name,metrics in (
        ('performance',(('e2e_p95_seconds','Delivered E2E p95 (s)'),('deadline_miss_rate','Assessed command miss fraction'),
            ('energy_j','Modeled energy / episode (J)'),('drone_cpu_utilization','UAV CPU utilization'),
            ('cloud_cpu_utilization','Cloud CPU utilization'),('uav_fraction','UAV task fraction'))),
        ('coordination',(('active_peer_aoi_seconds','Active-peer AoI (s)'),('xi_fov','FoV ordering error fraction'),
            ('xi_fov_ul','FoV uplink ordering errors'),('xi_fov_dl','FoV downlink ordering errors'),
            ('fallback_rate','Cached-fallback fraction'),('movement_priority_reversals','Movement priority reversals (count)')))):
        fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
        for ax,(metric,title) in zip(axes.flat,metrics):
            for index,route in enumerate(colors):
                yy=[]
                for n in (8,16,32):
                    row=table.get(('A4',route,n),{});value=row.get(metric) if row.get('successful') else None
                    yy.append(np.nan if value is None else value)
                ax.bar(np.arange(3)+(index-1)*.24,yy,.24,color=colors[route],label=labels[route])
            if metric=='e2e_p95_seconds':ax.axhline(.25,color='black',lw=.8,ls='--',label='250 ms')
            ax.set(xticks=range(3),xticklabels=['8','16','32'],xlabel='Robots',ylabel=title);ax.grid(axis='y',alpha=.15)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle('A4 / W1 · preliminary · successful missions · missing bars = undefined',fontsize=11)
        fig.savefig(folder/(name+'.png'),dpi=180);plt.close(fig)


def git_publish(project,destination):
    def git(*args):return subprocess.check_output(['git',*args],cwd=project,text=True,stderr=subprocess.STDOUT).strip()
    if git('branch','--show-current')!='main':raise ValueError('Publication requires main')
    git('fetch','origin','main')
    ahead,behind=map(int,git('rev-list','--left-right','--count','HEAD...origin/main').split())
    if behind:
        if ahead:raise ValueError('Remote and local main diverged; saved outputs remain local')
        git('merge','--ff-only','origin/main')
    paths=[name for name in CODE_FILES if (project/name).exists()]
    paths+=['results/redesign']
    for path in (project/'results/redesign').rglob('*'):
        if path.is_file() and path.stat().st_size>5*2**20:raise ValueError('Unexpected large publication artifact: '+str(path))
    git('add','--',*paths)
    staged=git('diff','--cached','--name-only').splitlines()
    allowed=set(CODE_FILES)
    if any(p not in allowed and not p.startswith('results/redesign/') for p in staged):
        raise ValueError('Unrelated staged files are present; publication refused')
    if staged:git('commit','-m','results: publish A4-first experiment checkpoint '+destination.name)
    git('push','origin','main')
    return git('rev-parse','HEAD')


def checkpoint(root,push=False):
    root=Path(root).resolve();cache=root/'accelerated/checkpoint_cache';cache.mkdir(parents=True,exist_ok=True)
    m=read(root/'manifest.json');parent_hash=digest(root/'manifest.json');rows=[]
    for marker in sorted((root/'primary').rglob('complete.json')):
        rows.append(cached_episode(marker.with_name('result.json'),cache,parent_hash))
    primary=[j for n,jobs in phases(m) for j in jobs]
    a4keys={key(j) for j in phases(m)[0][1]};bykey={key(r):r for r in rows};a4=[r for k,r in bykey.items() if k in a4keys]
    status=read(root/'accelerated/status.json') if (root/'accelerated/status.json').exists() else read(root/'status.json')
    completed_new=status.get('completed_new',0);started=status.get('started_at',time.time());elapsed=max(1.,time.time()-started)
    throughput=completed_new/elapsed*3600
    recent_path=root/'accelerated/last_checkpoint.json';previous=read(recent_path) if recent_path.exists() else None
    if previous and completed_new>=previous['completed_new']:
        throughput=(completed_new-previous['completed_new'])/max(1.,time.time()-previous['observed_at'])*3600
    remaining=(45-len(a4)) if len(a4)<45 else (len(primary)-len(rows))
    eta_seconds=remaining/throughput*3600 if throughput else None
    errors=list(root.rglob('error.json'))
    summary=summaries(rows,cache)
    progress=dict(phase=status.get('phase',status.get('stage')),a4_completed=len(a4),a4_total=45,
        completed=len(rows),total=len(primary),successful=sum(r['success'] for r in rows),
        failed_missions=sum(not r['success'] for r in rows),execution_errors=len(errors)+(status.get('status')=='error'),
        measured_episodes_per_hour=throughput,eta_seconds=eta_seconds,eta_scope='A4 first cohort' if len(a4)<45 else 'remaining sweep',
        observed_at=time.time(),completed_new=completed_new,worker_state=status.get('status'),preliminary=True,
        findings=f"A4 successes {sum(r['success'] for r in a4)}/{len(a4)}; failed missions retained; conditional metrics exclude failures")
    fingerprint=hashlib.sha256(json.dumps(sorted(r['result_sha256'] for r in rows)).encode()).hexdigest()
    if previous and previous.get('result_fingerprint')==fingerprint and (PROJECT/'results/redesign/latest.json').exists():
        if push:
            pointer=read(PROJECT/'results/redesign/latest.json');destination=PROJECT/pointer['snapshot']
            progress['commit']=git_publish(PROJECT,destination)
        write(root/'accelerated/last_checkpoint.json',dict(progress,result_fingerprint=fingerprint));return progress
    stamp=datetime.now(ZoneInfo('America/Toronto')).strftime('%Y%m%d-%H%M%S')
    destination=PROJECT/'results/redesign'/stamp;destination.mkdir(parents=True)
    csv_file(destination/'episodes.csv',rows);csv_file(destination/'summary.csv',summary);figures(destination,summary)
    write(destination/'progress.json',progress)
    write(destination/'provenance.json',dict(parent_manifest_sha256=parent_hash,model_sha256=m['model_sha256'],
        timeline_sha256=m['timeline_sha256'],map_sha256=m['map_sha256'],protocol=m['protocol'],
        raw_results='Local committed episode packets and event archives; preserved unchanged',
        validation=dict(full_suite_tests=466,focused_tests=76,preflight=read(root/'preflight.json')['status'],
            reproduction=read(root/'reference_validation.json')['status']),worker_count_comparison='deferred scheduling diagnostic'))
    (destination/'README.md').write_text('# Preliminary communication experiment\n\n'
        'A4/W1 uses 8, 16 and 32 robots, three execution policies, and five paired cases. '
        'The scheduler prioritizes this cohort before registered ablations. Simulation code and inputs are frozen.\n\n'
        '`episodes.csv` retains every completed mission, including failures. `summary.csv` and plots condition on successful missions; '
        'empty cohorts are undefined. Deadline counts retain late, undelivered and pending/censored distinctions. '
        'Energy includes the existing communication and UAV computation components. Raw event traces remain local. '
        'No mission-success plot is included. The incomplete worker-count comparison remains a scheduling diagnostic.\n',encoding='utf-8')
    write(PROJECT/'results/redesign/latest.json',dict(snapshot=destination.relative_to(PROJECT).as_posix(),result_fingerprint=fingerprint))
    if push:
        try:progress['commit']=git_publish(PROJECT,destination)
        except Exception as exc:
            progress['publication_error']=str(exc);write(root/'accelerated/publication_error.json',progress);raise
    write(root/'accelerated/last_checkpoint.json',dict(progress,result_fingerprint=fingerprint));return progress


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--push',action='store_true');args=parser.parse_args()
    print(json.dumps(checkpoint(args.root,args.push),indent=2))
