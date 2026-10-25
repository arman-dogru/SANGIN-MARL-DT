"""Read-only summaries, matched-snapshot controls and figures for omega sensitivity."""
from collections import Counter, defaultdict
from dataclasses import replace
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

from panama_uav.allocation import AllocationSnapshot, solve_binary
from panama_uav.communication_report import csv_table, paired_changes
from panama_uav.records import sha256

COUNTS=(2,4,8,16,32)
ROUTES=('allocated','edge','cloud')
LABELS={'allocated':'Allocated','edge':'All UAV','cloud':'All Cloud'}
COLORS={'allocated':'#2878B5','edge':'#E59637','cloud':'#49A078'}
WEIGHTS=(('W1',1.,1.),('W2',1.,2.),('W3',1.,4.),('W4',2.,1.),('W5',4.,1.))
IDENTITY=('case_id','num_agents','seed','starts','goals','orbital_offset_seconds','map_sha256')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def quantile(values,q):
    values=[v for v in values if v is not None]
    return float(np.quantile(values,q)) if values else None


def ratio(a,b):
    return a/b if b else None


def validate_packet(packet,case):
    row,tasks,rounds=packet['row'],packet['tasks'],packet['rounds']
    if any(row[k]!=case[k] for k in IDENTITY):raise ValueError('Paired case identity changed')
    if packet['audit']['status']!='passed':raise ValueError('Causal replay failed')
    if len({t['job_id'] for t in tasks})!=len(tasks):raise ValueError('Duplicate task IDs')
    statuses=Counter(t['deadline_status'] for t in tasks)
    if set(statuses)-{'on_time','late','undelivered'}:raise ValueError('Unexpected deadline state')
    for status,key in (('on_time','deadline_commands_on_time'),('late','deadline_commands_late'),('undelivered','deadline_commands_undelivered')):
        if statuses[status]!=row[key]:raise ValueError('Command partition mismatch')
    if sum(statuses.values())!=row['tasks_generated']:raise ValueError('Task denominator mismatch')
    if any(t.get('executed_x') not in (None,0,1) for t in tasks):raise ValueError('Nonbinary placement')
    if sum(t.get('executed_x')==1 for t in tasks)!=row['executed_drone_tasks']:raise ValueError('UAV task count mismatch')
    if sum(t.get('executed_x')==0 for t in tasks)!=row['executed_cloud_tasks']:raise ValueError('Cloud task count mismatch')
    if sum(r['tasks'] for r in rounds)!=len(tasks):raise ValueError('Round task counts mismatch')
    if row['deadline_cutoff_wait_tasks']!=row['command_cutoff_misses']:raise ValueError('Cutoff WAIT mismatch')
    for pool in ('drone','cloud'):
        if not 0<=row[pool+'_utilization_seconds']<=row['observed_seconds']+1e-6:raise ValueError('Invalid CPU utilization')
    return True


class StreamCSV:
    """Union schema is fixed from declared telemetry, not inferred from the first job."""
    def __init__(self,path,fields):
        self.stream=gzip.open(path,'wt',encoding='utf-8',newline='')
        self.writer=csv.DictWriter(self.stream,fieldnames=fields,extrasaction='raise')
        self.writer.writeheader()
    def row(self,row):
        self.writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    def close(self):self.stream.close()


def summarize(rows,tasks,cpu):
    first=rows[0]
    result={k:first[k] for k in ('weight_id','latency_coefficient','energy_coefficient','condition','route','num_agents')}
    total=len(tasks);uav=sum(t.get('executed_x')==1 for t in tasks);cloud=sum(t.get('executed_x')==0 for t in tasks)
    first_round=[t for t in tasks if t['cycle']==0]
    horizon=sum(r['observed_seconds'] for r in rows)
    energy=sum(r['modeled_energy_lower_j'] for r in rows)
    completed=[r for r in rows if r['success']]
    on=sum(r['deadline_commands_on_time'] for r in rows);late=sum(r['deadline_commands_late'] for r in rows)
    missing=sum(r['deadline_commands_undelivered'] for r in rows)
    result.update(cases=len(rows),successful_missions=len(completed),success_rate=len(completed)/len(rows),
        exploratory_scale_test=first['num_agents']>=16,tasks=total,uav_tasks=uav,cloud_tasks=cloud,
        unassigned_tasks=total-uav-cloud,uav_percent=100*uav/total if total else None,cloud_percent=100*cloud/total if total else None,
        first_round_tasks=len(first_round),first_round_uav=sum(t.get('executed_x')==1 for t in first_round),
        first_round_cloud=sum(t.get('executed_x')==0 for t in first_round),
        first_round_uav_mean=sum(t.get('executed_x')==1 for t in first_round)/len(rows),
        first_round_cloud_mean=sum(t.get('executed_x')==0 for t in first_round)/len(rows),
        commands=total,on_time_commands=on,late_commands=late,undelivered_commands=missing,
        on_time_percent=100*on/total,deadline_miss_percent=100*(late+missing)/total,
        censored_tasks=sum(r['deadline_censored_tasks'] for r in rows),
        pending_tasks=sum(r['deadline_pending_tasks'] for r in rows),
        cutoff_waits=sum(r['command_cutoff_misses'] for r in rows),
        observed_latency_mean_seconds=mean(t.get('observed_latency_seconds') for t in tasks),
        observed_latency_p50_seconds=quantile((t.get('observed_latency_seconds') for t in tasks),.5),
        observed_latency_p95_seconds=quantile((t.get('observed_latency_seconds') for t in tasks),.95),
        observed_latency_samples=sum(t.get('observed_latency_seconds') is not None for t in tasks),
        predicted_latency_mean_seconds=mean(t.get('predicted_latency_seconds') for t in tasks),
        predicted_latency_p95_seconds=quantile((t.get('predicted_latency_seconds') for t in tasks),.95),
        radio_queue_mean_seconds=mean(t['command_path_queue_seconds'] for t in tasks),
        interruption_mean_seconds=mean(t['command_path_interruption_seconds'] for t in tasks),
        guard_mean_seconds=mean(t['command_path_guard_seconds'] for t in tasks),
        compute_queue_mean_seconds=mean(t.get('compute_queue_seconds',0.) for t in tasks if t.get('compute_completed_at') is not None),
        compute_elapsed_mean_seconds=mean(t.get('compute_elapsed_seconds') for t in tasks),
        total_modeled_energy_j=energy,mean_episode_energy_j=energy/len(rows),
        mean_episode_energy_upper_j=mean(r['modeled_energy_upper_j'] for r in rows),
        energy_per_generated_task_j=energy/total,
        successful_episode_energy_mean_j=mean(r['modeled_energy_lower_j'] for r in completed),
        unsuccessful_episode_energy_mean_j=mean(r['modeled_energy_lower_j'] for r in rows if not r['success']),
        mean_episode_robot_tx_j=mean(r['robot_tx_energy_j'] for r in rows),
        mean_episode_uav_satellite_tx_j=mean(r['drone_tx_energy_j'] for r in rows),
        mean_episode_uav_compute_j=mean(r['drone_compute_energy_j'] for r in rows),
        mean_episode_other_radio_j=mean(r['radio_energy_lower_j']-r['robot_tx_energy_j']-r['drone_tx_energy_j'] for r in rows),
        successful_makespan_mean_seconds=mean(r['makespan_seconds'] for r in completed),
        waiting_steps_mean=mean(r['waiting_steps'] for r in rows),
        successful_waiting_steps_mean=mean(r['waiting_steps'] for r in completed),
        active_peer_aoi_seconds=ratio(sum(r['active_peer_aoi_sum'] for r in rows),sum(r['active_peer_aoi_samples'] for r in rows)),
        active_peer_aoi_samples=sum(r['active_peer_aoi_samples'] for r in rows),
        handover_tasks=sum(t['handover_overlaps_lifetime'] for t in tasks),
        unavailable_backhaul_admissions=sum(t['backhaul_available_at_admission'] is False for t in tasks),
        fixed_previous_tasks_mean=mean(t['fixed_previous_tasks'] for t in tasks),
        observed_mission_seconds=horizon)
    for pool in ('drone','cloud'):
        selected=[c for c in cpu if c['pool']==pool]
        occupied=sum(r[pool+'_occupied_seconds'] for r in rows)
        result.update({pool+'_cpu_utilization_percent':100*sum(r[pool+'_utilization_seconds'] for r in rows)/horizon,
            pool+'_peak_simultaneous_jobs':max(r[pool+'_max_concurrent_jobs'] for r in rows),
            pool+'_mean_concurrent_jobs_while_active':ratio(sum(r[pool+'_concurrent_job_seconds'] for r in rows),occupied),
            pool+'_active_job_frequency_cps':ratio(sum(c['frequency_job_seconds'] for c in selected),sum(c['allocated_job_seconds'] for c in selected)),
            pool+'_peak_job_frequency_cps':max(c['peak_job_frequency_cps'] for c in selected)})
    for place,x in (('uav',1),('cloud',0)):
        group=[t for t in tasks if t.get('executed_x')==x]
        result[place+'_on_time_percent']=100*sum(t['deadline_status']=='on_time' for t in group)/len(group) if group else None
    return result


def snapshots(root,packets):
    """Counterfactual solves change weights while holding the exact admission snapshot fixed."""
    output=root/'analysis/matched_snapshots.csv'
    if output.exists():return list(csv.DictReader(output.open(encoding='utf-8')))
    rows=[]
    for path in packets:
        if path.parts[-3:] and path.parent.name!='allocated-6ghz':continue
        if path.parent.parent.name!='W1':continue
        packet=read(path)
        for saved in packet['first_snapshots']:
            base=AllocationSnapshot.from_dict(saved['snapshot'])
            for name,latency,energy in WEIGHTS:
                model=replace(base,latency_coefficient=latency,energy_coefficient=energy)
                try:
                    result=solve_binary(model)
                    task=next(t for t in result.tasks if t['task_id']==saved['job_id'])
                    actual=next(t for t in saved['assignments'] if t['task_id']==saved['job_id'])
                    if name=='W1' and (task['x']!=actual['x'] or abs(task['latency_seconds']-actual['latency_seconds'])>1e-9):
                        raise ValueError('Matched-snapshot baseline changed')
                    rows.append(dict(case_id=packet['row']['case_id'],num_agents=packet['row']['num_agents'],
                        job_id=saved['job_id'],weight_id=name,latency_coefficient=latency,energy_coefficient=energy,
                        x=task['x'],w1_x=actual['x'],placement_changed=task['x']!=actual['x'],
                        fixed_previous_tasks=len(base.tasks)-1,cloud_route_available=base.tasks[0].backhaul_bps>0,
                        latency_seconds=task['latency_seconds'],energy_j=task['energy_j'],objective=result.objective,status=result.status))
                except Exception as error:
                    write(root/'analysis/matched_snapshot_failure.json',dict(case_id=packet['row']['case_id'],job_id=saved['job_id'],weight_id=name,
                        snapshot=saved['snapshot'],error=str(error),details=getattr(error,'details',None)))
                    raise
    csv_table(output,rows);return rows


def comparisons(groups):
    output=[]
    metrics=('mission_success_rate','makespan_seconds','waiting_steps','case_command_p95_seconds','command_on_time_rate',
             'modeled_energy_j','mean_drone_cpu_utilization','mean_cloud_cpu_utilization','cutoff_waits')
    for (weight,route,n),data in sorted(groups.items()):
        references=[]
        if weight!='W1' and ('W1',route,n) in groups:references.append(('W1',route))
        if route=='allocated':references.extend((weight,r) for r in ('edge','cloud'))
        for bw,br in references:
            base=groups.get((bw,br,n))
            if base is None:continue
            for metric in metrics:
                output.append(dict(weight_id=weight,route=route,num_agents=n,reference_weight=bw,reference_route=br,
                    **paired_changes(data,base,metric)))
    return output


def save_figure(fig,folder,name):
    folder.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf','svg'):fig.savefig(folder/(name+'.'+suffix),dpi=300,bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def figures(root,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'axes.axisbelow':True,'grid.alpha':.2,'svg.fonttype':'none','pdf.fonttype':42})
    table={(r['weight_id'],r['route'],r['num_agents']):r for r in summary}
    weights=[w for w,_,_ in WEIGHTS if any(r['weight_id']==w for r in summary)]
    x=np.arange(len(COUNTS));width=.23
    def panels(folder,name,field,ylabel,deadline=None,ylim=None):
        fig,axes=plt.subplots(1,len(weights),figsize=(3.4*len(weights),3.3),squeeze=False)
        limits=ylim or (0,max([r[field] for r in summary if r.get(field) is not None]+[deadline or 0,1e-9])*1.12)
        for i,w in enumerate(weights):
            ax=axes[0,i]
            for j,route in enumerate(ROUTES):
                values=[table.get((w,route,n),{}).get(field) for n in COUNTS]
                values=[np.nan if v is None else v for v in values]
                ax.bar(x+(j-1)*width,values,width,label=LABELS[route],color=COLORS[route],edgecolor='white',linewidth=.4)
            setting=next(v for v in WEIGHTS if v[0]==w)
            ax.set_title(f'{w}: latency {setting[1]:g}, energy {setting[2]:g}')
            ax.set_xticks(x,[str(n)+('*' if n>=16 else '') for n in COUNTS]);ax.set_xlabel('Robots')
            if i==0:ax.set_ylabel(ylabel)
            if deadline is not None:ax.axhline(deadline,color='#922B21',ls='--',lw=1,label='250 ms deadline')
            ax.set_ylim(*limits)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.04),frameon=False)
        fig.text(.5,-.01,'Parameter-only sensitivity results | * Exploratory scale tests | Five paired cases per bar',ha='center',fontsize=8)
        fig.tight_layout();save_figure(fig,root/'figures'/folder,name)
    # Stacked panels distinguish admission-round counts from full-mission fractions.
    fig,axes=plt.subplots(2,len(weights),figsize=(3.4*len(weights),6.3),squeeze=False)
    for i,w in enumerate(weights):
        for row,(uf,cf,label) in enumerate((('first_round_uav_mean','first_round_cloud_mean','First-round tasks per case (mean)'),
                ('uav_percent','cloud_percent','Mission-wide tasks (%)'))):
            ax=axes[row,i];u=[table[(w,'allocated',n)][uf] for n in COUNTS];c=[table[(w,'allocated',n)][cf] for n in COUNTS]
            ax.bar(x,u,color=COLORS['edge'],label='UAV');ax.bar(x,c,bottom=u,color=COLORS['cloud'],label='Cloud')
            ax.set_xticks(x,[str(n)+('*' if n>=16 else '') for n in COUNTS]);ax.set_xlabel('Robots')
            if row==0:ax.set_title(w)
            if i==0:ax.set_ylabel(label)
            if row==1:ax.set_ylim(0,100)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
    fig.text(.5,-.01,'Allocated policy | First-round counts and mission percentages use different denominators | * Exploratory scale tests',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,0,1,.96));save_figure(fig,root/'figures/01_allocation','allocation_sensitivity')
    panels('02_latency','e2e_p95','observed_latency_p95_seconds','Observed E2E p95 latency (s)',deadline=.25)
    panels('03_energy','episode_energy','mean_episode_energy_j','Modeled energy per episode (J)')
    # Component stacks for every weight/route, one panel per population.
    fig,axes=plt.subplots(1,len(COUNTS),figsize=(18,4),squeeze=False)
    components=(('mean_episode_uav_compute_j','UAV computation','#8367AC'),('mean_episode_robot_tx_j','Robot TX','#75AADB'),
                ('mean_episode_uav_satellite_tx_j','UAV satellite TX','#E59637'),('mean_episode_other_radio_j','Other modeled radio','#A3ADB7'))
    for i,n in enumerate(COUNTS):
        ax=axes[0,i];labels=[w+'\n'+{'allocated':'A','edge':'U','cloud':'C'}[r] for w in weights for r in ROUTES]
        bottom=np.zeros(len(labels))
        for field,label,color in components:
            values=np.array([table[(w,r,n)][field] for w in weights for r in ROUTES])
            ax.bar(np.arange(len(labels)),values,bottom=bottom,label=label,color=color);bottom+=values
        ax.set_xticks(range(len(labels)),labels,fontsize=6);ax.set_title(f'{n} robots'+(' (exploratory)' if n>=16 else ''))
        if i==0:ax.set_ylabel('Mean energy per episode (J)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False)
    fig.text(.5,-.01,'A: Allocated, U: All UAV, C: All Cloud | Cloud computation energy is absent from the preserved model',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,0,1,.9));save_figure(fig,root/'figures/03_energy','energy_components')
    panels('04_deadline_misses','deadline_misses','deadline_miss_percent','Commands missing 250 ms (%)',ylim=(0,105))
    for pool,label in (('drone','UAV'),('cloud','Cloud')):
        upper=max(5.,5.*np.ceil(max(r[pool+'_cpu_utilization_percent'] for r in summary)*1.1/5.))
        panels('05_cpu',('uav' if pool=='drone' else pool)+'_utilization',pool+'_cpu_utilization_percent',
            label+' CPU utilization (%)\nMission-time mean, including idle',ylim=(0,upper))
    panels('06_mission_success','mission_success','successful_missions','Successful missions out of five',ylim=(0,5.4))
    fig,axes=plt.subplots(1,len(COUNTS),figsize=(17,3.6),squeeze=False)
    palette=plt.get_cmap('tab10')
    for i,n in enumerate(COUNTS):
        ax=axes[0,i]
        for j,w in enumerate(weights):
            row=table[(w,'allocated',n)];a,b=row['mean_episode_energy_j'],row['observed_latency_p95_seconds']
            offsets=((5,6),(5,-13),(-23,9),(-23,-15),(8,20))
            ax.scatter(a,b,s=40,color=palette(j),label=w)
            ax.annotate(w,(a,b),xytext=offsets[j],textcoords='offset points',fontsize=8,
                arrowprops=dict(arrowstyle='-',color=palette(j),lw=.6),bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.5))
        ax.axhline(.25,color='#922B21',ls='--',lw=.8);ax.set_title(f'{n} robots'+(' (exploratory)' if n>=16 else ''));ax.set_xlabel('Mean modeled energy / episode (J)')
        if i==0:ax.set_ylabel('Observed E2E p95 latency (s)')
        ax.margins(.22)
    fig.text(.5,-.01,'Allocated policy | Discrete tested weights; no interpolation | 250 ms dashed reference',ha='center',fontsize=8)
    fig.tight_layout();save_figure(fig,root/'figures/07_tradeoff','latency_energy_tradeoff')


def numerical_report(root,m,summary,contrasts,matched):
    table={(r['weight_id'],r['route'],r['num_agents']):r for r in summary}
    weights=[w['id'] for w in m['weights']]
    lines=['# Latency–energy weight sensitivity','',
        'Parameter-only sensitivity results. The policy, workload, radio model, admission, binary allocator and compute scheduler are unchanged. '
        'The 16- and 32-robot cohorts are exploratory scale tests; checkpoint qualification metadata is unchanged.','',
        f"The matrix contains {len(summary)*m['cases_per_population']} simulations: {len(weights)} weights × three routes × five populations × {m['cases_per_population']} paired cases, capped at {m['episode_cap']} steps. No control results were reused.", '',
        '## Comparative results','',
        '| Robots | Weight | Route | UAV / cloud tasks | E2E p95 (ms) | Predicted mean (ms) | Energy / episode (J) | Misses (%) | Success |',
        '|---:|:---:|:---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['num_agents']} | {r['weight_id']} | {LABELS[r['route']]} | {r['uav_tasks']} / {r['cloud_tasks']} | {r['observed_latency_p95_seconds']*1000:.2f} | {(r['predicted_latency_mean_seconds'] or 0)*1000:.2f} | {r['mean_episode_energy_j']:.3f} | {r['deadline_miss_percent']:.2f} | {r['successful_missions']}/{r['cases']} |")
    lines+=['','## Research questions','']
    for n in COUNTS:
        base=table[('W1','allocated',n)]
        text=[]
        for w in weights:
            r=table[(w,'allocated',n)]
            text.append(f"{w}: {r['uav_percent']:.1f}% UAV, p95 {r['observed_latency_p95_seconds']*1000:.1f} ms, {r['mean_episode_energy_j']:.2f} J/episode, misses {r['deadline_miss_percent']:.2f}%, success {r['successful_missions']}/5")
        lines.append(f"- **{n} robots:** "+'; '.join(text)+'.')
    if len(weights)==5:
        lines+=['','1. **Higher latency/energy ratio:** '+ '; '.join(f"{n} robots: W4/W5 p95 changes {(table[('W4','allocated',n)]['observed_latency_p95_seconds']/table[('W1','allocated',n)]['observed_latency_p95_seconds']-1)*100:+.1f}% / {(table[('W5','allocated',n)]['observed_latency_p95_seconds']/table[('W1','allocated',n)]['observed_latency_p95_seconds']-1)*100:+.1f}%" for n in COUNTS)+'.',
            '2. **Higher energy/latency ratio:** '+ '; '.join(f"{n} robots: W2/W3 energy changes {(table[('W2','allocated',n)]['mean_episode_energy_j']/table[('W1','allocated',n)]['mean_episode_energy_j']-1)*100:+.1f}% / {(table[('W3','allocated',n)]['mean_episode_energy_j']/table[('W1','allocated',n)]['mean_episode_energy_j']-1)*100:+.1f}%" for n in COUNTS)+'. Observed latency costs are given above.',
            '3. **UAV preference at 2–4 robots:** '+ '; '.join(f"{n} robots: "+', '.join(f"{w} {table[(w,'allocated',n)]['uav_percent']:.1f}%" for w in weights) for n in (2,4))+'.',
            '4. **Mixed allocation at eight robots:** '+ ', '.join(f"{w}: {table[(w,'allocated',8)]['uav_percent']:.1f}% UAV" for w in weights)+'.',
            '5. **Cloud dominance at scale:** '+ '; '.join(f"{n} robots: "+(', '.join(w for w in weights if table[(w,'allocated',n)]['cloud_percent']>50) or 'none above 50%') for n in (16,32))+'.',
            '6. **Lowest miss rate:** '+ '; '.join(f"{n} robots: "+', '.join(w for w in weights if abs(table[(w,'allocated',n)]['deadline_miss_percent']-min(table[(v,'allocated',n)]['deadline_miss_percent'] for v in weights))<1e-10) for n in COUNTS)+'.',
            '7. **Single balance across populations:** '+', '.join(f"{n} robots minimum energy {min(weights,key=lambda w:table[(w,'allocated',n)]['mean_episode_energy_j'])}, minimum p95 {min(weights,key=lambda w:table[(w,'allocated',n)]['observed_latency_p95_seconds'])}" for n in COUNTS)+'. These empirical minima do not establish a universal setting from five cases.',
            '8. **Objective versus route/assignment constraints:** matched_snapshots.csv re-solves identical first-round admissions using each weight. '+
                '; '.join(f"{w}: {sum(str(r['weight_id'])==w and str(r['placement_changed']) in ('True','true') for r in matched)} placement changes in {sum(str(r['weight_id'])==w for r in matched)} matched admissions" for w in weights)+
                '. These comparisons hold radio inputs, priorities and previous assignments fixed; mission-wide placement also reflects admission availability and changed trajectories. active_count.csv and handover.csv retain those exposure counts.',
            '9. **Strict observed latency-and-energy dominance over both fixed routes:** '+ '; '.join(f"{n} robots: "+(', '.join(w for w in weights if all(table[(w,'allocated',n)]['observed_latency_p95_seconds']<table[(w,r,n)]['observed_latency_p95_seconds'] and table[(w,'allocated',n)]['mean_episode_energy_j']<table[(w,r,n)]['mean_episode_energy_j'] for r in ('edge','cloud'))) or 'none') for n in COUNTS)+'. Paired uncertainty is in comparisons.csv.',
            '10. **Deadline compliance and mission completion:** '+ '; '.join(f"{n} robots: success spans {min(table[(w,'allocated',n)]['successful_missions'] for w in weights)}–{max(table[(w,'allocated',n)]['successful_missions'] for w in weights)}/5 and miss rates {min(table[(w,'allocated',n)]['deadline_miss_percent'] for w in weights):.2f}–{max(table[(w,'allocated',n)]['deadline_miss_percent'] for w in weights):.2f}%" for n in COUNTS)+'. The paired comparisons describe association; finite cases and changing trajectories limit attribution.']
    lines+=['','## Definitions and validation','',
        'Observed E2E latency is robot command arrival minus sensing time. Predicted latency is the solver admission snapshot estimate and excludes future scheduling changes. Quantiles pool delivered tasks within each cell; undelivered tasks remain in deadline and task denominators. A miss is late plus undelivered; pending/censored counts are reported separately. The runtime cutoff is 250 ms and hard solver deadlines remain disabled.', '',
        'Energy is the existing modeled lower/upper mission tally, with transmitter and UAV-computation components. Cloud-computation and platform propulsion energy are not added. Per-task energy divides total mission energy by generated tasks, including unsuccessful missions; successful-only energy and failed-episode energy are separate fields. Longer failed episodes can increase total energy without a change in per-task efficiency.', '',
        'CPU utilization divides allocated-capacity time by the complete observed mission duration, including idle periods. Mean concurrent jobs while active divides job-seconds by occupied pool-seconds. Existing processor sharing has no separate waiting queue: service contention appears in compute elapsed time and concurrent-job telemetry.', '',
        'First-round counts average the five cycle-zero admissions per case; mission-wide percentages use all generated tasks. Active-count and handover tables provide subgroup denominators. Completed-mission makespan and all-episode waiting counts are retained. Paired percentile-bootstrap 95% intervals use 2,000 case resamples; five-case intervals are descriptive and small-sample uncertainty remains substantial.', '',
        'Every episode passes the existing causal/frozen-action replay before it is committed. Baseline reproduction compares physical frames/event chains, binary allocations, CPU service, task timing and mission metrics. Only real host timings are excluded from exact comparison; fractional solver diagnostics have a recorded numerical tolerance. Protected production and historical result hashes are checked before and after execution.', '',
        '## Provenance and reproduction','',f"- Model SHA256: `{m['model_sha256']}`; learner updates: {m['learner_updates']}; policy updates during study: 0.",
        f"- Timeline SHA256: `{m['timeline_sha256']}`; map SHA256: `{m['map_sha256']}`.",
        '- Cases, seeds, starts/goals, orbital offsets and configuration/source hashes are in cases.json and manifest.json.',
        '- Run: `.\\.venv\\Scripts\\python.exe scripts/run_omega_sensitivity.py --root output/omega_sensitivity/<new-id> --prepare --launch`.',
        '- Resume only a stopped incomplete run: append `--resume` to `--launch` with the same root. Existing case transactions cannot be overwritten.',
        '- Rebuild reports: `.\\.venv\\Scripts\\python.exe scripts/run_omega_sensitivity.py --root <run-root> --report`.',
        '- Full causal trace archives and checkpoints remain local under the run directory. Exported CSVs, configurations, report and figures are self-contained summaries.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def report(root):
    root=Path(root).resolve();m=read(root/'manifest.json');cases={c['case_id']:c for c in read(root/'cases.json')}
    paths=sorted((root/'episodes').glob('*/*/*.result.json'))
    if len(paths)!=m['expected_episodes']:raise ValueError('Incomplete experiment matrix')
    analysis=root/'analysis';analysis.mkdir(exist_ok=True)
    groups=defaultdict(list);taskgroups=defaultdict(list);cpugroups=defaultdict(list)
    episodes=[];rounds=[];active=defaultdict(list);handover=defaultdict(list);seen=set();first=[]
    # Discover union schemas before streaming, including unfinished task fields.
    task_fields=set();allocation_fields=set()
    for path in paths:
        p=read(path)
        for t in p['tasks']:task_fields.update(t)
        for a in p['allocations']:allocation_fields.update(a)
    identity_fields=['weight_id','latency_coefficient','energy_coefficient','case_id','num_agents','route','condition']
    taskcsv=StreamCSV(root/'tasks.csv.gz',list(dict.fromkeys(identity_fields+sorted(task_fields))))
    allocationcsv=StreamCSV(root/'allocations.csv.gz',list(dict.fromkeys(identity_fields+sorted(allocation_fields))))
    audit=[]
    try:
        for path in paths:
            p=read(path);row=dict(p['row']);w=p['weight'];case=cases[row['case_id']]
            validate_packet(p,case)
            identity=dict(weight_id=w['id'],latency_coefficient=w['latency'],energy_coefficient=w['energy'],
                case_id=row['case_id'],num_agents=row['num_agents'],route=row['route'],condition=row['condition'])
            key=(w['id'],row['route'],row['num_agents']);unique=(w['id'],row['route'],row['case_id'])
            if unique in seen:raise ValueError('Duplicate episode')
            seen.add(unique);row.update(identity);episodes.append(row);groups[key].append(row)
            taskgroups[key].extend(p['tasks']);cpugroups[key].extend(p['cpu'])
            cyclecounts={r['cycle']:r['tasks'] for r in p['rounds']}
            for t in p['tasks']:
                taskcsv.row(dict(t,**{k:v for k,v in identity.items() if k not in t}))
                active[key+(cyclecounts[t['cycle']],)].append(t)
                subgroup='unavailable_backhaul' if t['backhaul_available_at_admission'] is False else 'available_backhaul'
                handover[key+(subgroup,)].append(t)
                handover[key+('handover_lifetime' if t['handover_overlaps_lifetime'] else 'outside_handover',)].append(t)
            for a in p['allocations']:allocationcsv.row(dict(a,**identity))
            rounds.extend(dict(r,weight_id=w['id']) for r in p['rounds'])
            first.append(dict(identity,tasks=sum(t['cycle']==0 for t in p['tasks']),uav=sum(t['cycle']==0 and t.get('executed_x')==1 for t in p['tasks']),
                cloud=sum(t['cycle']==0 and t.get('executed_x')==0 for t in p['tasks'])))
            audit.append(dict(identity,status=p['audit']['status'],archive=p['audit']['archive'],archive_sha256=p['audit']['archive_sha256'],
                packet_sha256=sha256(path),**p['timing']))
    finally:taskcsv.close();allocationcsv.close()
    summary=[summarize(rows,taskgroups[key],cpugroups[key]) for key,rows in sorted(groups.items())]
    contrasts=comparisons(groups)
    for name,rows in (('summary',summary),('episodes',episodes),('rounds',rounds),('first_round',first),('comparisons',contrasts),('audit',audit)):
        csv_table(root/(name+'.csv'),rows)
    def exposure_rows(source,field):
        result=[]
        for key,tasks in sorted(source.items()):
            weight,route,n,value=key
            result.append(dict(weight_id=weight,route=route,num_agents=n,**{field:value},tasks=len(tasks),
                uav_tasks=sum(t.get('executed_x')==1 for t in tasks),cloud_tasks=sum(t.get('executed_x')==0 for t in tasks),
                on_time=sum(t['deadline_status']=='on_time' for t in tasks),late=sum(t['deadline_status']=='late' for t in tasks),
                undelivered=sum(t['deadline_status']=='undelivered' for t in tasks),
                observed_latency_p95_seconds=quantile((t.get('observed_latency_seconds') for t in tasks),.95),
                fixed_previous_tasks_mean=mean(t['fixed_previous_tasks'] for t in tasks),
                queue_mean_seconds=mean(t['command_path_queue_seconds'] for t in tasks),
                interruption_mean_seconds=mean(t['command_path_interruption_seconds'] for t in tasks),
                active_peer_aoi_seconds=ratio(sum(t['active_peer_aoi_sum'] for t in tasks),sum(t['active_peer_aoi_samples'] for t in tasks)),
                active_peer_aoi_samples=sum(t['active_peer_aoi_samples'] for t in tasks)))
        return result
    csv_table(root/'active_count.csv',exposure_rows(active,'actual_active_robots'))
    csv_table(root/'handover.csv',exposure_rows(handover,'exposure'))
    matched=snapshots(root,paths)
    figures(root,summary);numerical_report(root,m,summary,contrasts,matched)
    write(root/'validation.json',dict(status='passed',episodes=len(paths),expected_episodes=m['expected_episodes'],
        causal_replays=len(audit),paired_case_identities=True,exclusive_deadline_partition=True,binary_placements=True,
        unsuccessful_missions=sum(not e['success'] for e in episodes),solver_failures=len(list((root/'episodes').glob('*/*/*.error.json')))))
    # Small distributable bundles exclude full traces and model assets.
    for name,files in (('charts.zip',list((root/'figures').rglob('*'))),
            ('results.zip',list(root.glob('*.csv'))+list(root.glob('*.csv.gz'))+list((root/'configs').glob('*.toml'))+
             list((root/'analysis').glob('*.csv'))+[root/n for n in ('report.md','manifest.json','cases.json','validation.json')]+
             ([root/'baseline_validation.json'] if (root/'baseline_validation.json').exists() else []))):
        with zipfile.ZipFile(root/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path in files:
                if path.is_file():archive.write(path,path.relative_to(root))
    return summary


if __name__=='__main__':report(Path(sys.argv[1]))
