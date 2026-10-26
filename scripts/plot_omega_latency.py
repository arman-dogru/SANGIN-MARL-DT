"""Read-only W1 latency CDF/CCDF from the published omega task table."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np

from panama_uav.latency_plotting import empirical_steps,plot_empirical_latency
from panama_uav.records import sha256
from panama_uav.task_deadlines import DEADLINE_TOLERANCE_SECONDS

COUNTS=(2,4,8,16,32)
ROUTES=('allocated','edge','cloud')
LABELS={'allocated':'Allocator','edge':'All UAV','cloud':'All cloud'}
COLORS={'allocated':'#2878B5','edge':'#E59637','cloud':'#49A078'}


def display(value,scale=1.):
    return '—' if value is None else f'{value*scale:.2f}'


def close(actual,expected,name):
    if not math.isclose(actual,float(expected),rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError(f'Published statistic mismatch: {name}: {actual} != {expected}')


def collect(source,deadline=.25):
    source=Path(source)
    published={(int(r['num_agents']),r['route']):r for r in csv.DictReader((source/'summary.csv').open(encoding='utf-8',newline='')) if r['weight_id']=='W1'}
    episodes=defaultdict(set)
    for row in csv.DictReader((source/'episodes.csv').open(encoding='utf-8',newline='')):
        if row['weight_id']!='W1':continue
        if row['success']!='True':raise ValueError('Published latency cohort includes unsuccessful missions')
        episodes[int(row['num_agents']),row['route']].add(row['case_id'])
    groups=defaultdict(list);counts=defaultdict(Counter);seen=set();task_cases=defaultdict(set)
    with gzip.open(source/'tasks.csv.gz','rt',encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['weight_id']!='W1':continue
            key=int(row['num_agents']),row['route']
            if key[0] not in COUNTS or key[1] not in ROUTES:raise ValueError('Unexpected W1 population or route')
            identity=key+(row['case_id'],row['job_id'])
            if identity in seen:raise ValueError('Duplicate task identity')
            seen.add(identity)
            if row['case_id'] not in episodes[key]:raise ValueError('Task outside published successful cohort')
            task_cases[key].add(row['case_id'])
            source_time=float(row['source_time']);bound=float(row['deadline_at'])
            close(bound-source_time,deadline,'task deadline')
            status=row['deadline_status']
            if status not in ('on_time','late','undelivered'):raise ValueError('Invalid task outcome')
            counts[key].update(tasks=1);counts[key][status]+=1
            if row['delivered_at']:
                latency=float(row['observed_latency_seconds'])
                close(float(row['delivered_at'])-source_time,latency,'sensing-to-arrival latency')
                close(float(row['command_delivery_seconds']),latency,'existing latency alias')
                expected='on_time' if float(row['delivered_at'])<=bound+DEADLINE_TOLERANCE_SECONDS else 'late'
                if status!=expected:raise ValueError('Task deadline classification differs from arrival')
                groups[key].append(latency)
            elif status!='undelivered' or row['observed_latency_seconds'] or row['command_delivery_seconds']:
                raise ValueError('Undelivered task acquired a fabricated latency')
    summary=[]
    for n in COUNTS:
        for route in ROUTES:
            key=n,route;values=groups[key];counter=counts[key];old=published[key]
            for field,c in (('tasks','tasks'),('commands','tasks'),('on_time_commands','on_time'),
                ('late_commands','late'),('undelivered_commands','undelivered'),('observed_latency_samples','delivered')):
                actual=len(values) if c=='delivered' else counter[c]
                if actual!=int(old[field]):raise ValueError('Published count mismatch: '+field)
            if len(episodes[key])!=int(old['included_cases']) or task_cases[key]!=episodes[key]:
                raise ValueError('Successful episode count or task cohort mismatch')
            item=dict(weight_id='W1',num_agents=n,route=route,successful_episodes=len(episodes[key]),
                task_observations=counter['tasks'],delivered_samples=len(values),on_time=counter['on_time'],
                late=counter['late'],undelivered=counter['undelivered'],deadline_seconds=deadline,
                late_rate=counter['late']/counter['tasks'] if counter['tasks'] else None,
                undelivered_rate=counter['undelivered']/counter['tasks'] if counter['tasks'] else None)
            for field,stat in (('mean',np.mean),('p50',lambda x:np.quantile(x,.5)),('p95',lambda x:np.quantile(x,.95))):
                value=float(stat(values)) if values else None
                if value is not None:close(value,old[f'observed_latency_{field}_seconds'],field)
                elif old[f'observed_latency_{field}_seconds']:raise ValueError('Nonempty published latency in empty cohort')
                item[f'latency_{field}_seconds']=value
            if counter['tasks']:
                close(100*counter['on_time']/counter['tasks'],old['on_time_percent'],'on-time percentage')
                close(100*(counter['late']+counter['undelivered'])/counter['tasks'],old['deadline_miss_percent'],'miss percentage')
            summary.append(item)
    return groups,summary


def table(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(source,destination,deadline=.25):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists():raise ValueError('Use a new output directory; existing results are read-only')
    protected={p.relative_to(source).as_posix():sha256(p) for p in source.rglob('*') if p.is_file()}
    groups,summary=collect(source,deadline)
    destination.mkdir(parents=True)
    table(destination/'latency_statistics.csv',summary)
    points=[]
    for (n,route),values in sorted(groups.items()):
        x,y=empirical_steps(values)
        points.extend(dict(num_agents=n,route=route,latency_seconds=float(t),cdf=float(f),ccdf=float(1-f),
            delivered_samples=len(values)) for t,f in zip(x,y))
    table(destination/'empirical_distributions.csv',points)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    indexed={(r['num_agents'],r['route']):r for r in summary}
    for complementary,name in ((False,'cdf'),(True,'ccdf')):
        fig,axes=plt.subplots(1,5,figsize=(17,4.6),layout='constrained')
        for ax,n in zip(axes,COUNTS):
            for route in ROUTES:
                row=indexed[n,route]
                plot_empirical_latency(ax,groups[n,route],complementary=complementary,log_tail=complementary,
                    unit_scale=1000,color=COLORS[route],label=LABELS[route],linewidth=1.5)
            ax.axvline(deadline*1000,color='#333333',linestyle='--',linewidth=1,label='250 ms deadline')
            ax.set(xlabel='Sensing-to-arrival latency (ms)',title=f'{n} robots'+(' (exploratory)' if n>=16 else ''))
            ax.set_xlim(left=0);ax.grid(alpha=.2)
            if complementary:
                maximum=max(1,max(len(groups[n,r]) for r in ROUTES))
                ax.set_ylim(.75/maximum,1.05)
            else:ax.set_ylim(0,1.02)
            captions=[]
            for route in ROUTES:
                r=indexed[n,route]
                captions.append(f'{LABELS[route]}: n={r["delivered_samples"]:,}; missions={r["successful_episodes"]}; '
                    f'late={display(r["late_rate"],100)}%; U={display(r["undelivered_rate"],100)}%')
            ax.text(.0,-.27,'\n'.join(captions),transform=ax.transAxes,fontsize=7.3,va='top',linespacing=1.6)
        axes[0].set_ylabel('P(T > t | delivered)' if complementary else 'P(T ≤ t | delivered)')
        fig.suptitle('W1 · Successful missions · Delivered-command '+('CCDF (log tail)' if complementary else 'CDF'),fontsize=12)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
        for suffix in ('png','pdf','svg'):fig.savefig(destination/f'latency_{name}.{suffix}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    after={p.relative_to(source).as_posix():sha256(p) for p in source.rglob('*') if p.is_file()}
    if protected!=after:raise ValueError('Existing results changed')
    manifest=dict(protocol='omega_w1_latency_distributions_v1',source=str(source),source_sha256=protected,
        weight='W1',populations=list(COUNTS),routes=list(ROUTES),deadline_seconds=deadline,
        cohort='successful_missions',cdf_conditioning='delivered_commands',
        late_and_undelivered_denominator='all_tasks_in_included_successful_missions',
        undelivered_latency='right_censored_not_imputed',ccdf='P(T > t | delivered); terminal zero retained in CSV, omitted from log plot',
        published_statistics_verified=True,source_unchanged=True,simulation_runs=0,
        python_script_sha256=sha256(Path(__file__)),helper_sha256=sha256(Path(sys.modules['panama_uav.latency_plotting'].__file__)))
    (destination/'validation.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    lines=['# W1 command-latency distributions','',
        'Read-only CDF and log-tail CCDF derived from `results/omega_sensitivity/tasks.csv.gz`. '
        'W1 uses latency/energy coefficients 1/1. All curves pool delivered commands from published successful missions. '
        'The late/undelivered fractions divide by all task observations in the same successful-mission cohort. '
        'Undelivered commands remain right-censored and receive no fabricated latency.','',
        'The step-CDF implementation is shared with the existing priority-evaluation plotting code. '
        'Every task identity, successful-episode count, command partition, mean, p50, pooled p95, and deadline percentage '
        'is checked against the existing publication. No simulations are executed; source result hashes are unchanged.','',
        'The CCDF is P(T > t | delivered). Its terminal zero is stored in the numerical distribution table; '
        'the logarithmic plot ends at the positive left limit at the maximum observation. '
        'The 16/32-robot cohorts remain exploratory scale tests.','',
        '| Robots | Policy | Successful missions | Delivered n | p95 (ms) | Late (%) | Undelivered (%) |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["num_agents"]} | {LABELS[r["route"]]} | {r["successful_episodes"]} | {r["delivered_samples"]} | '
            f'{display(r["latency_p95_seconds"],1000)} | {display(r["late_rate"],100)} | {display(r["undelivered_rate"],100)} |')
    lines+=['','Reproduce in a new directory:','',
        '```powershell',r'.\.venv\Scripts\python.exe scripts/plot_omega_latency.py --output results/omega_latency_w1_v2','```','']
    (destination/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status='passed',groups=len(summary),tasks=sum(r['task_observations'] for r in summary),
        delivered=sum(r['delivered_samples'] for r in summary),plots=6,existing_results_unchanged=True),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/omega_sensitivity'))
    parser.add_argument('--output',type=Path,default=Path('results/omega_latency_w1'))
    args=parser.parse_args();run(args.source,args.output)
