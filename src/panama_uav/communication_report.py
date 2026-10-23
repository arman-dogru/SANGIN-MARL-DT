"""Audited comparisons and vector figures for frozen-policy communication runs."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .records import sha256, write_json

ROUTES=('ideal','allocated','edge','cloud')
LABELS=dict(ideal='Instant baseline',allocated='Allocated',edge='All UAV',cloud='All cloud')
RATIOS=dict(command_on_time_rate=('deadline_commands_on_time','tasks_generated'),
    command_late_rate=('deadline_commands_late','tasks_generated'),
    command_undelivered_rate=('deadline_commands_undelivered','tasks_generated'),
    priority_fov_error_rate=('physical_fov_order_errors','physical_relevant_pairs'),
    cached_fov_error_rate=('fov_errors','fov_relevant_pairs'),
    peer_aoi_seconds=('peer_aoi_sum','inference_pairs'),
    stale_action_disagreement_rate=('stale_fresh_disagreements','inferences'),
    injected_peer_action_disagreement_rate=('injected_delay_disagreements','inferences'),
    decision_action_disagreement_rate=('decision_action_shadow_disagreements','decision_action_shadow_comparisons'),
    cutoff_wait_rate=('command_cutoff_misses','active_decisions'),
    mean_drone_cpu_utilization=('drone_busy_seconds','observed_seconds'),
    mean_cloud_cpu_utilization=('cloud_busy_seconds','observed_seconds'))
SCALARS=dict(mission_success_rate='success',makespan_seconds='makespan_seconds',
    waiting_steps='waiting_completed_steps',case_command_p95_seconds='task_latency_p95_seconds',
    case_inference_round_p95_seconds='inference_round_p95_seconds',
    cutoff_waits='command_cutoff_misses',modeled_energy_j='modeled_energy_lower_j')


def csv_table(path,rows):
    rows=list(rows)
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v
                         for k,v in row.items()} for row in rows)


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _mean(values):
    values=[float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _quantile(values,q=.95):
    values=[float(v) for v in values if v is not None]
    return float(np.quantile(values,q)) if values else None


def estimate(rows,metric):
    if metric in RATIOS:
        a,b=RATIOS[metric];den=sum(r[b] for r in rows)
        return sum(r[a] for r in rows)/den if den else None
    return _mean(r[SCALARS[metric]] for r in rows)


def paired_changes(rows,reference,metric,samples=2000):
    actual={r['case_id']:r for r in rows};base={r['case_id']:r for r in reference}
    if len(actual)!=len(rows) or len(base)!=len(reference) or set(actual)!=set(base):
        raise ValueError('Contrasts require identical unique paired cases')
    ids=sorted(actual)
    if metric not in RATIOS:
        field=SCALARS[metric]
        ids=[i for i in ids if actual[i].get(field) is not None and base[i].get(field) is not None]
    if not ids:return dict(metric=metric,paired_cases=0,difference=None,ci95_low=None,ci95_high=None)
    rng=np.random.default_rng(20261023)
    indices=rng.integers(0,len(ids),size=(samples,len(ids)))
    if metric in RATIOS:
        a,b=RATIOS[metric]
        values=[]
        for group in (actual,base):
            numerator=np.array([group[i][a] for i in ids],dtype=float)
            denominator=np.array([group[i][b] for i in ids],dtype=float)
            totals=denominator[indices].sum(axis=1)
            values.append(np.divide(numerator[indices].sum(axis=1),totals,
                out=np.full(samples,np.nan),where=totals>0))
        changes=values[0]-values[1]
        first=estimate([actual[i] for i in ids],metric);second=estimate([base[i] for i in ids],metric)
        point=first-second if first is not None and second is not None else None
    else:
        values=np.array([float(actual[i][field])-float(base[i][field]) for i in ids])
        point=float(np.mean(values));changes=values[indices].mean(axis=1)
    changes=changes[np.isfinite(changes)]
    lo,hi=np.quantile(changes,[.025,.975]) if len(changes) else (None,None)
    return dict(metric=metric,paired_cases=len(ids),difference=point,
        ci95_low=float(lo) if lo is not None else None,ci95_high=float(hi) if hi is not None else None)


def summary_row(rows,tasks,rounds):
    first=rows[0]
    result={k:first[k] for k in ('condition','route','axis','level','num_agents')}
    result.update(cases=len(rows),completed_missions=sum(r['success'] for r in rows),
        tasks=len(tasks),rounds=len(rounds),on_time_commands=sum(r['deadline_commands_on_time'] for r in rows),
        late_commands=sum(r['deadline_commands_late'] for r in rows),
        undelivered_commands=sum(r['deadline_commands_undelivered'] for r in rows),
        censored_tasks=sum(r['deadline_censored_tasks'] for r in rows),
        inference_relevant_pairs=sum(r['physical_relevant_pairs'] for r in rows),
        inference_ordering_errors=sum(r['physical_fov_order_errors'] for r in rows),
        cached_fov_relevant_pairs=sum(r['fov_relevant_pairs'] for r in rows),
        cached_fov_errors=sum(r['fov_errors'] for r in rows),
        oracle_decision_comparisons=sum(r['decision_action_shadow_comparisons'] for r in rows),
        oracle_decision_disagreements=sum(r['decision_action_shadow_disagreements'] for r in rows),
        active_decisions=sum(r['active_decisions'] for r in rows),
        inference_peer_pairs=sum(r['inference_pairs'] for r in rows),
        inferences=sum(r['inferences'] for r in rows),
        attempted_vertex_conflicts=sum(r['attempted_vertex_conflicts'] for r in rows),
        attempted_edge_swaps=sum(r['attempted_edge_swaps'] for r in rows),
        robot_tx_energy_j=sum(r['robot_tx_energy_j'] for r in rows),
        drone_tx_energy_j=sum(r['drone_tx_energy_j'] for r in rows),
        drone_compute_energy_j=sum(r['drone_compute_energy_j'] for r in rows),
        assigned_uav_tasks=sum(r['executed_drone_tasks'] for r in rows),
        assigned_cloud_tasks=sum(r['executed_cloud_tasks'] for r in rows),
        modeled_energy_total_j=sum(r['modeled_energy_lower_j'] for r in rows),
        task_latency_p95_seconds=_quantile(t.get('command_delivery_seconds') for t in tasks),
        inference_ready_p95_seconds=_quantile(t.get('inference_ready_seconds') for t in tasks),
        inference_round_p95_seconds=_quantile(r.get('inference_round_seconds') for r in rounds),
        command_round_p95_seconds=_quantile(r.get('command_round_seconds') for r in rounds),
        ingress_mean_seconds=_mean(t.get('ingress_seconds') for t in tasks),
        processing_mean_seconds=_mean(t.get('processing_seconds') for t in tasks),
        egress_mean_seconds=_mean(t.get('egress_seconds') for t in tasks))
    for metric in (*SCALARS,*RATIOS):result[metric]=estimate(rows,metric)
    for metric in ('case_command_p95_seconds','command_on_time_rate','mission_success_rate','makespan_seconds'):
        # A zero baseline provides a reproducible case-clustered interval for the estimate.
        baseline=[dict(r,**{field:0 for field in ('success','makespan_seconds','task_latency_p95_seconds',
                                                   'deadline_commands_on_time') if r.get(field) is not None}) for r in rows]
        interval=paired_changes(rows,baseline,metric)
        result[metric+'_ci95_low']=interval['ci95_low'];result[metric+'_ci95_high']=interval['ci95_high']
    n=len(rows);proportion=result['mission_success_rate'];z=1.959963984540054
    center=(proportion+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(proportion*(1-proportion)/n+z*z/(4*n*n))/(1+z*z/n)
    result['mission_success_rate_ci95_low']=max(0.,float(center-half))
    result['mission_success_rate_ci95_high']=min(1.,float(center+half))
    return result


def _plots(root,summary,tasks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,'legend.fontsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':600})
    folder=root/'figures';folder.mkdir(exist_ok=True)
    colors={'ideal':'#666666','allocated':'#009E73','edge':'#0072B2','cloud':'#D55E00'}
    styles={'allocated':(':','^'),'edge':('-','o'),'cloud':('--','s')}
    counts=sorted({r['num_agents'] for r in summary})
    def save(fig,name):
        for extension in ('pdf','svg','png'):
            metadata={'CreationDate':None} if extension=='pdf' else None
            fig.savefig(folder/f'{name}.{extension}',bbox_inches='tight',metadata=metadata)
        plt.close(fig)
    nominal=[r for r in summary if r['axis']=='nominal']
    fig,axes=plt.subplots(2,3,figsize=(10.5,6.1),layout='constrained')
    fields=[('case_command_p95_seconds','Mean case command p95 (s)'),
        ('command_on_time_rate','On-time command fraction'),('peer_aoi_seconds','Mean peer AoI (s)'),
        ('modeled_energy_j','Mean observed energy (J)'),
        ('makespan_seconds','Makespan (steps)'),('waiting_steps','Waiting (robot steps)')]
    for ax,(field,label) in zip(axes.flat,fields):
        for j,route in enumerate(ROUTES):
            group=sorted([r for r in nominal if r['route']==route],key=lambda r:r['num_agents'])
            valid=[(counts.index(r['num_agents'])+(j-(len(ROUTES)-1)/2)*.2,r) for r in group if r[field] is not None]
            if not valid:continue
            xs=[x for x,r in valid];ys=[r[field] for x,r in valid]
            ax.bar(xs,ys,width=.19,color=colors[route],label=LABELS[route],alpha=.9)
            if field+'_ci95_low' in group[0]:
                lo=[max(0,r[field]-r[field+'_ci95_low']) for x,r in valid]
                hi=[max(0,r[field+'_ci95_high']-r[field]) for x,r in valid]
                ax.errorbar(xs,ys,yerr=[lo,hi],fmt='none',ecolor='black',capsize=2,lw=.6)
        ax.set(xticks=range(len(counts)),xticklabels=counts,xlabel='Robots',ylabel=label)
        if field=='command_on_time_rate':ax.set_ylim(0,1.05)
        if field=='case_command_p95_seconds':ax.axhline(.25,color='black',ls=':',lw=1)
    axes[0,0].legend(frameon=False)
    save(fig,'nominal_routes')
    fig,axes=plt.subplots(2,3,figsize=(10.5,6.2),layout='constrained')
    population_colors={n:c for n,c in zip(counts,('#009E73','#0072B2','#CC79A7','#E69F00'))}
    axes_spec=[('uplink','Additional access uplink (ms)'),('downlink','Additional access downlink (ms)'),
        ('compute','Additional inference service (ms)'),('aoi','Peer-state hold (s)'),
        ('interruption','Interruption / 10 s (ms)'),('backhaul','Additional LEO leg delay (ms)')]
    for ax,(axis,label) in zip(axes.flat,axes_spec):
        for n in counts:
            for route in ('allocated','edge','cloud'):
                if axis=='backhaul' and route=='edge':continue
                rows=[r for r in summary if r['num_agents']==n and r['route']==route and r['axis'] in (axis,'nominal')]
                rows=sorted(rows,key=lambda r:r['level'])
                ax.plot([r['level']*(1 if axis=='aoi' else 1000) for r in rows],
                        [r['command_on_time_rate'] for r in rows],color=population_colors[n],
                        ls=styles[route][0],marker=styles[route][1],ms=3)
        ax.set(xlabel=label,ylabel='On-time command fraction',ylim=(-.03,1.03))
    handles=[Line2D([0],[0],color=c,label=f'{n} robots') for n,c in population_colors.items()]
    handles.extend([Line2D([0],[0],color='black',ls=style[0],marker=style[1],label=LABELS[route]) for route,style in styles.items()])
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False)
    save(fig,'deadline_sensitivity')
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0),layout='constrained')
    for route in ROUTES:
        condition='ideal' if route=='ideal' else route+'__nominal'
        for ax,field in zip(axes,('inference_ready_seconds','command_delivery_seconds')):
            values=sorted(t[field] for t in tasks if t['condition']==condition and t['num_agents']==max(counts) and t.get(field) is not None)
            if values:ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=LABELS[route],color=colors[route])
    for ax,label in zip(axes,('Inference ready (s)','Command arrival (s)')):
        ax.set(xlabel=label,ylabel='Empirical CDF',ylim=(0,1.02));ax.axvline(.25,color='black',ls=':',lw=1)
    axes[0].legend(frameon=False);save(fig,'latency_distributions')
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0),layout='constrained')
    for route in ('allocated','edge','cloud'):
        for n in counts:
            rows=sorted([r for r in summary if r['route']==route and r['num_agents']==n and r['axis'] in ('nominal','aoi')],key=lambda r:r['level'])
            for ax,field in zip(axes,('priority_fov_error_rate','stale_action_disagreement_rate')):
                valid=[r for r in rows if r[field] is not None]
                ax.plot([r['peer_aoi_seconds'] for r in valid],[r[field] for r in valid],
                    color=population_colors[n],ls=styles[route][0],marker=styles[route][1],ms=3)
    for ax,label in zip(axes,('FoV priority-order error fraction','Stale / fresh action disagreement')):
        ax.set(xlabel='Measured mean peer AoI (s)',ylabel=label,ylim=(-.02,1.02))
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False);save(fig,'aoi_coordination')
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0),layout='constrained')
    for route in ROUTES:
        rows=sorted([r for r in nominal if r['route']==route],key=lambda r:r['num_agents'])
        axes[0].errorbar([r['num_agents'] for r in rows],[r['mission_success_rate'] for r in rows],
            yerr=[[max(0,r['mission_success_rate']-r['mission_success_rate_ci95_low']) for r in rows],
                  [max(0,r['mission_success_rate_ci95_high']-r['mission_success_rate']) for r in rows]],
            color=colors[route],label=LABELS[route],marker='o',capsize=2)
        for pool,ls in (('drone','-'),('cloud','--')):
            axes[1].plot([r['num_agents'] for r in rows],[r['mean_'+pool+'_cpu_utilization'] for r in rows],
                color=colors[route],ls=ls,marker='o',ms=3)
    axes[0].set(xlabel='Robots',ylabel='Mission success fraction',ylim=(-.03,1.03),xticks=counts)
    axes[0].legend(frameon=False)
    axes[1].set(xlabel='Robots',ylabel='CPU utilization',ylim=(-.03,1.03),xticks=counts)
    axes[1].legend(handles=[Line2D([0],[0],color='black',ls='-',label='UAV CPU'),Line2D([0],[0],color='black',ls='--',label='Cloud CPU')],frameon=False)
    save(fig,'success_and_utilization')


def _network_plot(root,contrasts):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(10.5,6.1),layout='constrained')
    fields=(('case_command_p95_seconds','Command p95 change (s)'),
        ('command_on_time_rate','On-time command change'),
        ('priority_fov_error_rate','FoV ordering error change'),
        ('mission_success_rate','Mission success change'),
        ('makespan_seconds','Makespan change (steps)'),('waiting_steps','Waiting change (robot steps)'))
    colors={'allocated':'#009E73','edge':'#0072B2','cloud':'#D55E00'}
    for ax,(metric,label) in zip(axes.flat,fields):
        counts=sorted({r['num_agents'] for r in contrasts})
        for j,route in enumerate(('allocated','edge','cloud')):
            rows=sorted([r for r in contrasts if r['condition']==route+'__nominal'
                and r['comparison']=='vs_ideal_network' and r['metric']==metric
                and r['difference'] is not None],key=lambda r:r['num_agents'])
            if not rows:continue
            xs=[counts.index(r['num_agents'])+(j-1)*.16 for r in rows]
            ys=[r['difference'] for r in rows]
            ax.errorbar(xs,ys,yerr=[[max(0,r['difference']-r['ci95_low']) for r in rows],
                [max(0,r['ci95_high']-r['difference']) for r in rows]],color=colors[route],
                label=LABELS[route],marker='o',ls='none',capsize=3)
        ax.axhline(0,color='black',lw=.7)
        ax.set(xticks=range(len(counts)),xticklabels=counts,xlabel='Robots',ylabel=label)
    axes[0,0].legend(frameon=False)
    for ext in ('pdf','svg','png'):
        fig.savefig(root/'figures'/f'network_penalties.{ext}',bbox_inches='tight',
            metadata={'CreationDate':None} if ext=='pdf' else None)
    plt.close(fig)


def report(root):
    from .communication_effects import verify_inputs
    root=Path(root).resolve();m=verify_inputs(root)
    bank=read_rows(root/'episodes.jsonl');tasks=read_rows(root/'tasks.jsonl');rounds=read_rows(root/'rounds.jsonl')
    audits=read_rows(root/'audits.jsonl')
    cases={c['case_id']:c for c in json.loads((root/'paired_cases.json').read_text())}
    expected={(c['name'],case) for c in m['conditions'] for case in cases}
    if len(bank)!=len(expected) or {(r['condition'],r['case_id']) for r in bank}!=expected:
        raise ValueError('Missing or duplicate paired simulations')
    if len(audits)!=len(expected) or {(a['condition'],a['case_id']) for a in audits}!=expected:
        raise ValueError('Replay coverage mismatch')
    if any(a['status']!='passed' or sha256(a['archive'])!=a['archive_sha256'] for a in audits):
        raise ValueError('Replay archive checksum mismatch')
    grouped=defaultdict(list);task_groups=defaultdict(list);round_groups=defaultdict(list)
    for row in bank:
        if any(row[k]!=v for k,v in cases[row['case_id']].items()):raise ValueError('Paired case identity mismatch')
        grouped[row['condition'],row['num_agents']].append(row)
    for task in tasks:task_groups[task['condition'],task['num_agents']].append(task)
    for cycle in rounds:round_groups[cycle['condition'],cycle['num_agents']].append(cycle)
    summary=[];contrasts=[]
    for key,rows in grouped.items():
        group_tasks=task_groups[key]
        if len(group_tasks)!=sum(r['tasks_generated'] for r in rows):raise ValueError('Task telemetry coverage mismatch')
        summary.append(summary_row(rows,group_tasks,round_groups[key]))
        route=rows[0]['route'];n=key[1]
        references=[('vs_ideal',grouped['ideal',n])]
        if route!='ideal':
            references.append(('vs_route_nominal',grouped[route+'__nominal',n]))
            if (route+'__ideal_network',n) in grouped:
                references.append(('vs_ideal_network',grouped[route+'__ideal_network',n]))
        for label,reference in references:
            for metric in (*SCALARS,*RATIOS):
                contrasts.append(dict(condition=key[0],route=route,num_agents=n,comparison=label,
                                      **paired_changes(rows,reference,metric)))
    primary_summary=[r for r in summary if r['axis']=='nominal']
    primary_contrasts=[]
    for route in ('allocated','edge','cloud'):
        for reference in ROUTES:
            if ROUTES.index(reference)>=ROUTES.index(route):continue
            for n in m['populations']:
                name=route+'__nominal';base='ideal' if reference=='ideal' else reference+'__nominal'
                if (name,n) not in grouped or (base,n) not in grouped:continue
                for metric in (*SCALARS,*RATIOS):
                    primary_contrasts.append(dict(condition=name,reference=base,route=route,num_agents=n,
                        **paired_changes(grouped[name,n],grouped[base,n],metric)))
    for name,records in (('episodes',bank),('tasks',tasks),('rounds',rounds),('summary',summary),
            ('paired_comparisons',contrasts),('primary_summary',primary_summary),
            ('primary_paired_comparisons',primary_contrasts)):
        csv_table(root/(name+'.csv'),records)
    write_json(root/'summary.json',summary)
    write_json(root/'validation.json',dict(status='passed',paired_episodes=len(bank),
        task_records=len(tasks),round_records=len(rounds),audited_archives=len(audits),
        frozen_inputs_verified=True,model_updates=0,all_pair_identities_verified=True))
    _plots(root,summary,tasks)
    if all((root/'figures'/('nominal_routes.'+ext)).exists() for ext in ('pdf','svg','png')):
        _network_plot(root,contrasts)
    text='# Frozen PANAMA communication effects\n\n'
    text+=f"Protocol: `{m['protocol']}`. Frozen policy SHA256: `{m['model_sha256']}`.\n\n"
    text+=f"{len(cases)} paired cases; {len(m['conditions'])} conditions; {len(bank)} simulations; 250 ms command cutoff and one-second control.\n\n"
    text+='| Route | Robots | Missions | Completions | Command p95 (s) | On-time | Mean peer AoI (s) | Makespan | Waiting |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'
    def fmt(x):return f'{x:.4g}' if x is not None else '—'
    for row in sorted([s for s in summary if s['axis']=='nominal'],key=lambda r:(ROUTES.index(r['route']),r['num_agents'])):
        text+='| '+' | '.join([LABELS[row['route']],str(row['num_agents']),str(row['cases']),str(row['completed_missions']),
            fmt(row['task_latency_p95_seconds']),fmt(row['command_on_time_rate']),fmt(row['peer_aoi_seconds']),fmt(row['makespan_seconds']),fmt(row['waiting_steps'])])+' |\n'
    text+='\nThe sensitivity matrix adds 50/150 ms access uplink or downlink delay, 1/3-step peer-state holds, 50/150 ms inference service overhead, and 250/1000 ms interruptions every ten seconds. Cloud-capable arms also add 50/150 ms per LEO leg. Each sensitivity changes one factor.\n\n'
    text+='Matched ideal-network controls retain each location\'s finite CPU stage. Comparisons against these controls separate communication costs from computation; the instant ideal reference also removes modeled computation.\n\n'
    text+='`summary.csv`, `paired_comparisons.csv`, `tasks.csv`, and `rounds.csv` preserve counts and paired intervals. `validation.json` records input/replay coverage. Vector PDF/SVG and 600 dpi PNG figures are under `figures/`.\n\n'
    text+='Timing separates sensing-to-inference-ready, inference rounds, and command arrival at the robot. Additional compute service is modeled after the unchanged finite CPU stage. Energy covers robot uplink, UAV backhaul uplink, and configured UAV CPU energy; cloud/server/receive/propulsion power is not parameterized. Injected delays and interruptions are controlled assumptions; orbital geometry and link budgets come from the archived timeline.\n'
    (root/'REPORT.md').write_text(text,encoding='utf-8')
    return summary
