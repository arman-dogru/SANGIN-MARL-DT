"""Read-only successful-mission reports and whole-cohort completion accounting."""
from collections import defaultdict,Counter
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
from panama_uav.latency_plotting import plot_empirical_latency
from panama_uav.records import sha256
from panama_uav.communication_report import csv_table
import run_omega_sensitivity as core

ROUTES=('allocated','edge','cloud')
COLORS={'allocated':'#2878B5','edge':'#E59637','cloud':'#49A078'}
LABELS={'allocated':'Allocator','edge':'All UAV','cloud':'All cloud'}


def ratio(a,b):return a/b if b else None
def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def report(root,stage):
    root=Path(root);m=core.read(root/'manifest.json');destination=root/'reports'/stage
    if destination.exists():raise ValueError('Existing reports are immutable; choose a new report revision')
    destination.mkdir(parents=True)
    packets=[];allrows=[];groups=defaultdict(list);taskgroups=defaultdict(list);latencies=defaultdict(list)
    selected_exports=defaultdict(list)
    tasks=[];inventory=[]
    for path in sorted((root/'primary').rglob('result.json')):
        packet=core.read(path);meta=core.read(path.with_name('complete.json'))
        if sha256(path)!=meta['sha256']:raise ValueError('Episode transaction hash changed')
        if packet['audit']['status']!='passed':raise ValueError('Unaudited episode')
        row=packet['row'];job=packet['job'];key=job['treatment'],job['weight_id'],job['route'],job['num_agents']
        allrows.append(row);groups[key].append(row)
        if row['success']:
            for name,item in packet.get('event_exports',{}).items():
                if sha256(item['path'])!=item['sha256']:raise ValueError('Derived event export hash changed')
                selected_exports[name].append((job,item))
            for task in packet['tasks']:
                task=dict(task,treatment=job['treatment'],weight_id=job['weight_id'],route=job['route'],
                    num_agents=job['num_agents'],case_id=job['case_id'])
                tasks.append(task);taskgroups[key].append(task)
                latency=task.get('observed_latency_seconds')
                if latency is not None:latencies[key].append(latency)
    if not allrows:raise ValueError('No completed episode records')
    unique={('/'.join(str(j[k]) for k in ('treatment','weight_id','route','num_agents','case_id'))):j for j in m['jobs_A']+m['jobs_B']}
    present={(r['treatment'],r['weight_id'],r['route'],r['case_id']) for r in allrows}
    for identifier,job in unique.items():
        inventory.append(dict(job,status='completed' if (job['treatment'],job['weight_id'],job['route'],job['case_id']) in present else 'not_completed'))
    summary=[];success=[]
    for key,rows in sorted(groups.items()):
        treatment,weight,route,n=key;selected=[r for r in rows if r['success']];tt=taskgroups[key];values=latencies[key]
        statuses=Counter(t['deadline_status'] for t in tt)
        item=dict(treatment=treatment,weight_id=weight,route=route,num_agents=n,attempted_missions=len(rows),
            successful_missions=len(selected),failed_missions=len(rows)-len(selected),conditional_cohort='successful_missions',
            tasks=len(tt),delivered_samples=len(values),on_time=statuses['on_time'],late=statuses['late'],undelivered=statuses['undelivered'],
            on_time_rate=ratio(statuses['on_time'],len(tt)),late_rate=ratio(statuses['late'],len(tt)),undelivered_rate=ratio(statuses['undelivered'],len(tt)),
            p95_latency_seconds=float(np.quantile(values,.95)) if values else None,
            mean_latency_seconds=mean(values),mean_predicted_latency_seconds=mean(t.get('predicted_latency_seconds') for t in tt),
            mean_makespan_seconds=mean(r.get('makespan_seconds') for r in selected),
            mean_modeled_energy_j=mean(r.get('modeled_energy_lower_j') for r in selected),
            mean_waiting_opportunities=mean(r.get('waiting_cell_opportunities',r.get('waiting_steps')) for r in selected),
            mean_controller_wait_seconds=mean(r.get('controller_wait_seconds') for r in selected),
            uav_tasks=sum(t.get('executed_x')==1 for t in tt),cloud_tasks=sum(t.get('executed_x')==0 for t in tt))
        item['undelivered_overdue']=sum(t.get('deadline_undelivered_overdue',False) for t in tt)
        item['pending_censored']=sum(t.get('deadline_pending',False) for t in tt)
        item['assessed_tasks']=item['on_time']+item['late']+item['undelivered_overdue']
        item['deadline_miss_rate']=ratio(item['late']+item['undelivered_overdue'],item['assessed_tasks'])
        additive=('eligible_decisions','command_decisions','cached_fallback_decisions','fallback_movements','safety_wait_decisions',
            'speed_limited_nonmovement','fallback_shadow_opportunities','fallback_stale_fresh_disagreements',
            'invoked_fallback_diagnostic_pairs','invoked_stale_fresh_disagreements','fallback_fresh_dqn_disagreements',
            'fresh_fallback_fresh_dqn_disagreements','movement_priority_reversals','relevant_movement_priority_reversals',
            'completion_priority_reversals','active_pair_seconds','attempted_vertex_conflicts','attempted_edge_swaps',
            'post_resolution_conflicts','directed_pairs','ordering_errors','relevant_pairs','fov_errors','fov_uplink_errors',
            'fov_downlink_errors','aoi_weighted_error_sum','peer_aoi_sum','peer_aoi_samples','inference_pairs','inference_relevant_pairs',
            'inference_fov_errors','physical_fov_errors','vertex_relevant_pairs','vertex_e2e_errors','vertex_aoi_error_sum')
        for field in additive:item[field]=sum(r.get(field,0) or 0 for r in selected) if selected else 0
        for field in ('active_peer_aoi_sum','active_peer_aoi_samples','physical_relevant_pairs','physical_relevant_ordering_errors',
            'slow_delayed_command_uses','slow_late_command_uses','fast_bundled_second_actions','fast_bundle_prediction_mismatches',
            'scheduled_command_misses','scheduled_cutoff_waits'):
            observed=[r.get(field) for r in selected]
            item[field]=sum(v for v in observed if v is not None) if observed and all(v is not None for v in observed) else None
        for target,numerator,denominator in (
            ('cached_fallback_rate','cached_fallback_decisions','eligible_decisions'),
            ('mean_peer_aoi_seconds','peer_aoi_sum','peer_aoi_samples'),
            ('ordering_error_rate','ordering_errors','directed_pairs'),('xi_fov','fov_errors','relevant_pairs'),
            ('xi_fov_ul','fov_uplink_errors','relevant_pairs'),('xi_fov_dl','fov_downlink_errors','relevant_pairs'),
            ('xi_aoi','aoi_weighted_error_sum','relevant_pairs'),
            ('stale_fallback_disagreement_rate','invoked_stale_fresh_disagreements','invoked_fallback_diagnostic_pairs')):
            item[target]=ratio(item[numerator],item[denominator])
        item['mean_active_peer_aoi_seconds']=ratio(item['active_peer_aoi_sum'],item['active_peer_aoi_samples'])
        item['physical_ordering_error_rate']=ratio(item['physical_relevant_ordering_errors'],item['physical_relevant_pairs'])
        for pool in ('drone','cloud'):
            item[pool+'_cpu_utilization']=ratio(sum(r.get(pool+'_utilization_seconds',0) for r in selected),sum(r['observed_seconds'] for r in selected))
        summary.append(item)
        count=len(rows);p=len(selected)/count;z=1.959963984540054;factor=1+z*z/count
        center=(p+z*z/(2*count))/factor;half=z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/factor
        success.append(dict(treatment=treatment,weight_id=weight,route=route,num_agents=n,attempted=count,successful=len(selected),
            success_rate=p,ci95_low=max(0.,center-half),ci95_high=min(1.,center+half),
            failure_reasons=dict(Counter(r.get('failure_reason') or 'unspecified' for r in rows if not r['success']))))
    for name,rows in (('summary',summary),('mission_success',success),('all_completed_episodes',allrows),('inventory',inventory),
        ('successful_episodes',[r for r in allrows if r['success']])):csv_table(destination/(name+'.csv'),rows)
    fields=sorted({k for t in tasks for k in t})
    with gzip.open(destination/'tasks.csv.gz','wt',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(tasks)
    for name,items in selected_exports.items():
        identities=('treatment','weight_id','route','num_agents','case_id')
        fields=sorted(set(identities)|{k for job,item in items for k in item['fields']})
        with gzip.open(destination/(name+'.csv.gz'),'wt',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for job,item in items:
                with gzip.open(item['path'],'rt',encoding='utf-8',newline='') as source:
                    for row in csv.DictReader(source):writer.writerow(dict(row,**{k:job[k] for k in identities}))
    paired=[];rng=np.random.default_rng(7821)
    for key,rows in groups.items():
        t,w,route,n=key
        references=[('A0','W1')] if t!='A0' and w=='W1' else []
        if t=='A4' and w!='W1':references.append(('A4','W1'))
        for ref_t,ref_w in references:
            old={r['case_id']:r for r in groups.get((ref_t,ref_w,route,n),[])}
            for metric in ('success','makespan_seconds','modeled_energy_lower_j','controller_wait_seconds','xi_fov','movement_priority_reversals'):
                matched=[(r,old[r['case_id']]) for r in rows if r['case_id'] in old
                    and (metric=='success' or r['success'] and old[r['case_id']]['success'])
                    and r.get(metric) is not None and old[r['case_id']].get(metric) is not None]
                differences=np.array([float(a[metric])-float(b[metric]) for a,b in matched])
                ci=np.quantile(differences[rng.integers(0,len(matched),(2000,len(matched)))].mean(1),[.025,.975]) if matched else [None,None]
                paired.append(dict(treatment=t,weight_id=w,route=route,num_agents=n,reference_treatment=ref_t,reference_weight=ref_w,
                    metric=metric,paired_cases=len(matched),paired_case_ids=[a['case_id'] for a,b in matched],difference=mean(differences),
                    ci95_low=ci[0],ci95_high=ci[1],cohort='all_attempts' if metric=='success' else 'successful_in_both'))
    csv_table(destination/'paired_comparisons.csv',paired)
    plot(destination,summary,success,latencies,m)
    shutil.copy2(root/'manifest.json',destination/'manifest.json');shutil.copy2(root/'cases.json',destination/'cases.json')
    core.write(destination/'validation.json',dict(status='passed',completed_episodes=len(allrows),successful=sum(r['success'] for r in allrows),
        phase=stage,conditional_tasks=len(tasks),all_saved_episode_audits_passed=True,policy_updates=0))
    text=['# Movement frequency and communication coordination','',f'Report stage: {stage}. Completed episode records: {len(allrows)}. '
        f'Successful missions: {sum(r["success"] for r in allrows)}. Full planned inventory: {m["unique_primary_jobs"]}.','',
        'The frozen policy, original path-distance priorities, priority-to-weight map, optimizer, radio and CPU/energy models are unchanged. '
        'A0 retains original discrete movement for exact reproduction. A1–A7 use discrete execution schedules: '
        'slow robots execute the previous step command, normal robots the current command, and fast robots two bundled actions. '
        'Each action permits one adjacent cell with collision checks. Fast tasks run the frozen DQN twice remotely, '
        'with a DT-only predicted state for call two, one response and no extra task or deadline. '
        'Slow late deliveries remain late even when applicable at their next-step execution cutoff. '
        'Missing responses preserve each treatment fallback in both rounds. There is no separate movement clock. '
        'Both movement rounds occur at nΔ+D with explicit substep ordering. Only sensing epochs with cutoff <=250 s generate tasks. '
        'Termination occurs at the exact physical completion event, with unfinished tasks right-censored.','',
        'Diagnostic fresh fallback and DQN actions never execute. The DT ordering decomposition uses the command processing location; '
        'missing-command epochs explicitly use the UAV ingress replica. The original formal relevance and error definitions remain. '
        'Vertex-only consistency is exploratory and separate.','',
        'Numerical conditional metrics use successful missions. Whole-cohort success and failures are in mission_success.csv. '
        'Paired conditional comparisons use successful cases in both conditions. Empty cohorts remain undefined. '
        'CDFs condition on delivered commands and do not assign latency to undelivered commands. '
        'CPU utilization includes mission idle periods. Energy includes existing communication and UAV computation only.','',
        'Inspect summary.csv for exact reversal, fallback, ordering-pair and task denominators. '
        'All attempted completed outcomes remain in all_completed_episodes.csv; inventory.csv includes unfinished jobs.','']
    (destination/'report.md').write_text('\n'.join(text),encoding='utf-8')


def plot(destination,summary,success,latencies,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=destination/'figures';folder.mkdir()
    def save(fig,name):
        for suffix in ('png','pdf','svg'):fig.savefig(folder/(name+'.'+suffix),dpi=250,bbox_inches='tight')
        plt.close(fig)
    table={(r['treatment'],r['weight_id'],r['route'],r['num_agents']):r for r in summary}
    def shown(key,metric):
        row=table.get(key,{})
        return row.get(metric) if row.get('successful_missions',0) else None
    for t in sorted({k[0] for k in latencies}):
        weights=sorted({k[1] for k in latencies if k[0]==t})
        for w in weights:
            populations=sorted({k[3] for k in latencies if k[:2]==(t,w)})
            deadline=manifest['treatments'][t]['command_deadline_seconds'];interval=manifest['treatments'][t]['control_interval_seconds']
            for ccdf in (False,True):
                fig,axes=plt.subplots(1,len(populations),figsize=(4.5*len(populations),4),squeeze=False,layout='constrained')
                for ax,n in zip(axes[0],populations):
                    for route in ROUTES:
                        key=t,w,route,n;values=latencies[key];r=table.get(key)
                        if not r:continue
                        caption=f'{LABELS[route]} n={len(values)}; L={r["late"]}/{r["tasks"]}; U={r["undelivered"]}/{r["tasks"]}'
                        plot_empirical_latency(ax,values,complementary=ccdf,log_tail=ccdf,unit_scale=1000,label=caption,color=COLORS[route])
                    ax.axvline(deadline*1000,color='black',ls='--',lw=.8,label=f'Cutoff {deadline*1000:g} ms')
                    ax.axvline(interval*1000,color='gray',ls=':',lw=.8,label=f'Interval {interval*1000:g} ms')
                    ax.set(title=f'{n} robots',xlabel='Sensing-to-arrival latency (ms)',ylabel='Delivered-command '+('CCDF' if ccdf else 'CDF'))
                    ax.grid(alpha=.2);ax.legend(fontsize=6)
                fig.suptitle(f'{t} / {w} · successful missions · delivered commands')
                save(fig,f'{t}_{w}_latency_'+('ccdf' if ccdf else 'cdf'))
    metrics=('p95_latency_seconds','mean_modeled_energy_j','mean_makespan_seconds','mean_controller_wait_seconds',
        'cached_fallback_rate','mean_active_peer_aoi_seconds','xi_fov','xi_fov_ul','xi_fov_dl','xi_aoi',
        'movement_priority_reversals','relevant_movement_priority_reversals','attempted_vertex_conflicts','attempted_edge_swaps',
        'drone_cpu_utilization','cloud_cpu_utilization','stale_fallback_disagreement_rate')
    treatments=sorted({r['treatment'] for r in summary if r['weight_id']=='W1'})
    for metric in metrics:
        fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
        for ax,n in zip(axes,(8,16,32)):
            for i,route in enumerate(ROUTES):
                values=[shown((t,'W1',route,n),metric) for t in treatments]
                ax.bar(np.arange(len(treatments))+(i-1)*.25,[np.nan if v is None else v for v in values],.25,color=COLORS[route],label=LABELS[route])
            ax.set(xticks=np.arange(len(treatments)),xticklabels=treatments,title=f'{n} robots',ylabel=metric)
        axes[0].legend(fontsize=8);save(fig,'phase_A_'+metric)
    weights=sorted({r['weight_id'] for r in summary if r['treatment']=='A4'})
    if len(weights)>1:
        for metric in metrics:
            fig,axes=plt.subplots(1,5,figsize=(19,3.8),layout='constrained')
            for ax,n in zip(axes,(2,4,8,16,32)):
                for i,route in enumerate(ROUTES):
                    values=[shown(('A4',w,route,n),metric) for w in weights]
                    ax.bar(np.arange(len(weights))+(i-1)*.25,[np.nan if v is None else v for v in values],.25,color=COLORS[route],label=LABELS[route])
                ax.set(xticks=np.arange(len(weights)),xticklabels=weights,title=f'{n} robots',ylabel=metric)
            axes[0].legend(fontsize=8);save(fig,'phase_B_'+metric)
        fig,axes=plt.subplots(1,5,figsize=(19,3.8),layout='constrained')
        for ax,n in zip(axes,(2,4,8,16,32)):
            for w in weights:
                row=table.get(('A4',w,'allocated',n),{})
                if not row.get('successful_missions'):continue
                x,y=row['mean_modeled_energy_j'],row['p95_latency_seconds']*1000
                ax.scatter(x,y,color=COLORS['allocated']);ax.annotate(w,(x,y),xytext=(3,3),textcoords='offset points',fontsize=8)
            ax.axhline(250,color='black',ls=':',lw=.7)
            ax.set(title=f'{n} robots',xlabel='Mean modeled energy / episode (J)',ylabel='Delivered-command p95 (ms)')
        save(fig,'phase_B_latency_energy_tradeoff')
    # Success plots remain local, outside any eventual publication package.
    fig,ax=plt.subplots(figsize=(9,4))
    for route in ROUTES:
        rows=[r for r in success if r['weight_id']=='W1' and r['route']==route]
        ax.scatter([f'{r["treatment"]}/{r["num_agents"]}' for r in rows],[r['success_rate'] for r in rows],label=LABELS[route],color=COLORS[route])
    ax.tick_params(axis='x',rotation=90);ax.set(ylabel='Success / all completed attempts',ylim=(0,1.05));ax.legend();save(fig,'mission_success_local')
