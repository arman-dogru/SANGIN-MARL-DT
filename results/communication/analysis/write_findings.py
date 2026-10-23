"""Render the measured findings and an imposed-hold supplement from saved data."""
import csv
import json
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve()
summary=json.loads((root/'summary.json').read_text())
groups={(r['condition'],r['num_agents']):r for r in summary}
primary=list(csv.DictReader((root/'primary_paired_comparisons.csv').open()))
contrasts=list(csv.DictReader((root/'paired_comparisons.csv').open()))
manifest=json.loads((root/'manifest.json').read_text())
status=json.loads((root/'status.json').read_text())
audit=json.loads((root/'denominator_audit.json').read_text())
mechanisms=json.loads((root/'mechanism_audit.json').read_text())['conditions']
labels={'ideal':'Instant baseline','allocated':'Allocated','edge':'All UAV','cloud':'All cloud'}
routes=tuple(labels)
def name(route):return 'ideal' if route=='ideal' else route+'__nominal'
def fmt(value,digits=3):return '—' if value is None else f'{value:.{digits}f}'
def percent(value):return '—' if value is None else f'{100*value:.2f}%'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join(
        '| '+' | '.join(map(str,row))+' |\n' for row in rows)+'\n'

text='# Frozen PANAMA communication inference: findings\n\n'
text+='The nominal cloud, allocated and instant-reference arms each completed 39/40 missions. All-UAV completed 10/40. Command timing and waiting differ even where completion counts match. Deliberate peer staleness changed DQN decisions even with every cloud command on time. The strongest injected LEO delay prevented completion in both cloud-capable arms.\n\n'
text+='## Configuration and evidence\n\n'
text+='Forty paired cases cover 2/4/6/8 robots, ten per population, on the archived 32×32 nominal map. Each case retains identical starts/goals, seed, policy and orbital offset across 41 conditions. There are 160 primary simulations, 120 matched ideal-network simulations and 1,360 single-factor sensitivity simulations. All episodes retain the 250-step cap, one-second control and 250 ms command/task cutoffs; hard optimizer deadlines remain disabled. The instant reference uses the existing fresh-state exception and removes modeled network/CPU service. Matched ideal-network controls retain finite CPU service and causal DT updates.\n\n'
text+=f"Frozen policy: `{manifest['model_sha256']}`. Exact source and package versions are in `manifest.json` and `code/`; source base is `{manifest['source_commit']}` with the recorded working-tree snapshot. Study wall time was {status['wall_seconds']:.1f} seconds ({status['wall_seconds']/3600:.3f} hours), with eight single-thread CPU inference workers. No learning updates occurred.\n\n"
text+='The workload remains 100,000 logical bits and 1,000 cycles/bit; UAV/cloud capacities remain 1e9/1e10 cycles/s. Robot/UAV conducted uplink power remains 0.2/2 W. Native access and LEO capacity caps, link budgets, queues and archived propagation are retained. Additional delays/outages are controlled assumptions, not measured network conditions.\n\n'
text+='## Four primary arms\n\n'
rows=[]
for n in (2,4,6,8):
    for route in routes:
        r=groups[name(route),n]
        rows.append((n,labels[route],f"{r['completed_missions']}/{r['cases']}",percent(r['command_on_time_rate']),
            fmt(1000*r['task_latency_p95_seconds'],1),fmt(r['makespan_seconds'],2),fmt(r['waiting_steps'],2),fmt(r['modeled_energy_j'])))
text+=table(('Robots','Arm','Missions','On-time commands','Command p95 (ms)','Makespan (steps)','Waiting (robot steps)','Modeled energy (J)'),rows)
text+='Command p95 in this table pools observed commands. The figures also report the separately named mean case p95 and its case-bootstrap interval. Full per-population CPU utilization, offloading, source-age, energy and outcome counts are in `primary_summary.csv`.\n\n'
text+='The same case, `agents-8-case-002`, reached the 250-step cap in the instant, allocated and cloud arms. The nominal experiment therefore provides no observed additional success loss for either cloud-capable arm. This small cohort does not establish population-level equivalence.\n\n'
text+='All-UAV command p95 is approximately 402/603/803 ms at 4/6/8 robots. The corresponding finite-CPU ideal-network controls also complete no missions and meet no command deadlines. Under the retained processor sharing and workload, this is a compute-service limit at the selected cutoff. Its flat additional-delay response has a zero-compliance floor.\n\n'
text+='## Paired primary differences\n\n'
text+='Cloud minus allocated, eight robots; intervals use 2,000 paired case-cluster bootstrap resamples, seed 20261023. Absolute mission-success intervals use Wilson bounds.\n\n'
rows=[]
for metric,label,scale in (('mission_success_rate','Mission success (pp)',100),
        ('command_on_time_rate','On-time commands (pp)',100),('makespan_seconds','Makespan (steps)',1),
        ('waiting_steps','Waiting (robot steps)',1),('modeled_energy_j','Modeled energy (J)',1)):
    r=next(x for x in primary if x['condition']=='cloud__nominal' and x['reference']=='allocated__nominal'
           and x['num_agents']=='8' and x['metric']==metric)
    rows.append((label,r['paired_cases'],fmt(scale*float(r['difference'])),
                 f"[{scale*float(r['ci95_low']):.3f}, {scale*float(r['ci95_high']):.3f}]"))
text+=table(('Quantity','Paired cases','Difference','95% CI'),rows)
text+='Cloud serves commands more reliably and reduces waiting/makespan in this cohort, at higher modeled energy. Allocation optimizes its fixed-input weighted-latency/energy snapshot, including transmission and UAV dynamic-compute energy, with no hard command-deadline constraint. It does not establish an end-to-end or mission optimum.\n\n'
text+='## Controlled communication effects\n\n'
rows=[]
for condition in ('cloud__nominal','cloud__uplink-0.15','cloud__downlink-0.15','cloud__compute-0.15',
        'cloud__interruption-0.25','cloud__interruption-1','cloud__backhaul-0.15',
        'allocated__nominal','allocated__uplink-0.05','allocated__backhaul-0.15'):
    r=groups[condition,8]
    rows.append((condition,f"{r['completed_missions']}/10",percent(r['command_on_time_rate']),
                 fmt(1000*r['task_latency_p95_seconds'],1),fmt(r['makespan_seconds'],2),fmt(r['waiting_steps'],2)))
text+=table(('Eight-robot condition','Missions','On-time','Command p95 (ms)','Makespan','Waiting'),rows)
text+='At eight robots, adding 150 ms downlink delay reduces cloud command timeliness by 25.16 pp (95% CI −30.42 to −18.53) and adds 15.78 makespan steps (9.22 to 23.11). One-second interruptions every ten seconds reduce timeliness by 9.68 pp (−9.93 to −9.34) and add 3.44 makespan steps (2.33 to 4.33). Adding 150 ms to each LEO direction produces zero cloud completions and zero on-time cloud commands.\n\n'
text+='The 250 ms and one-second interruptions have the same observed cloud mission outcomes here, despite different command-tail latency. Both cross the cutoff at the affected ticks. Commands received within the cutoff share the same one-second control schedule. Mission effects therefore need not increase smoothly with sub-cutoff delay. All 41 conditions and all populations remain in `summary.csv` and `paired_comparisons.csv`.\n\n'
text+='### Admission timing explains the non-monotone allocated response\n\n'
text+='The nominal allocated arm has 163 eight-robot jobs admitted while the native backhaul budget reports unavailable service. All 163 use the UAV; 125 are late. All 2,173 jobs admitted with available backhaul are on time. In the cloud-only arm, 161 jobs see unavailable backhaul at admission and all arrive on time after queuing. The additional 50 ms access-uplink hold shifts admission past those unavailable instants: no jobs see unavailable backhaul, all 2,215 commands arrive on time, and mean makespan falls by 2.00 steps (95% CI −2.56 to −1.44).\n\n'
text+='These results expose sensitivity to admission phase and instantaneous availability. The retained allocator does not optimize waiting for imminent route recovery; the fixed-cloud route continues through the transport queue. `mechanism_audit.json` contains the admission counts from every eight-robot case in the listed conditions. No allocation algorithm was changed.\n\n'
text+='## Peer staleness and coordination\n\n'
rows=[]
for n in (2,4,6,8):
    for condition in ('cloud__nominal','cloud__aoi-3'):
        r=groups[condition,n]
        rows.append((n,condition,f"{r['completed_missions']}/10",percent(r['command_on_time_rate']),
            f"{r['inference_ordering_errors']}/{r['inference_relevant_pairs']}",
            percent(r['stale_action_disagreement_rate'])))
text+=table(('Robots','Condition','Missions','On-time','Inference FoV errors/exposures','Stale/fresh action disagreement'),rows)
text+='For eight cloud robots, the three-step peer hold increases stale/fresh DQN disagreement from 2.30% to 6.88%, a paired increase of 4.58 pp (95% CI 2.89 to 7.09), while command timeliness remains 100%. The inference FoV error count rises from 1/522 to 11/515. Mean makespan increases by 1.67 steps, with a 95% interval of 0.00 to 3.56; mission completion remains 9/10. The experiment shows changed decisions and ordering, with a modest and uncertain mission-timing effect in this small bank.\n\n'
text+='Inference physical-reference FoV errors and the original robot-cached FoV metric remain separate fields. Original cached FoV error is zero in these nominal runs. Empty relevant-pair denominators remain unavailable, including eight-robot UAV-only runs. Exploratory vertex-only records remain separate in the raw telemetry. Sparse and repeated exposures produce wide ordering-error intervals.\n\n'
text+='### Absolute source age includes completed robots\n\n'
rows=[]
for condition in ('cloud__nominal','cloud__ideal_network','cloud__aoi-3'):
    m=mechanisms[condition]['peer_age']
    rows.append((condition,fmt(m['active_peers']['mean_age_seconds']),m['active_peers']['pairs'],
        fmt(m['completed_peers']['mean_age_seconds']),m['completed_peers']['pairs']))
text+=table(('Eight-robot condition','Active-peer age (s)','Active peer samples','Completed-peer age (s)','Completed peer samples'),rows)
text+='Completed robots retain their last delivered state after reporting stops. The resulting overall nominal cloud peer age of 22.51 seconds is dominated by these retained records; the finite-CPU ideal-network control has a similar overall age. Active-peer age is about 0.55 seconds nominally and 3.14 seconds with the three-step hold. The supplementary `aoi_hold_sensitivity` figure uses imposed hold on the horizontal axis; the original all-peer-age figure and metrics are retained.\n\n'
text+='## Validation, scope and reproducibility\n\n'
text+=f"All 360 regression tests passed, with one existing ITU-R low-elevation warning. All 1,640 archives passed causal replay and frozen-policy action checks. Independent transaction checks cover {audit['totals']['tasks']:,} task records and {audit['totals']['rounds']:,} inference rounds: {audit['totals']['on_time']:,} on-time, {audit['totals']['late']:,} late, and {audit['totals']['undelivered']:,} undelivered commands, plus {audit['totals']['cutoff_waits']:,} cutoff-WAIT tags. The tags are separate from the exclusive arrival partition. CPU bounds, forced/binary assignments, per-round denominators and aggregate/per-case record identities pass.\n\n"
text+='The study uses one nominal map and ten paired cases per population. Delays/outages are controlled assumptions. CPU times derive from logical cycles and finite pools; added compute sensitivity is post-CPU inference overhead. The allocator retains native rate/propagation inputs and does not receive injected transit/overhead as an extra optimization input. Energy covers robot access uplink, UAV backhaul uplink and UAV dynamic compute; cloud compute, downlink transmitters, receivers, static platform and propulsion power are unparameterized. Reported energy is the observed modeled lower-bound component.\n\n'
text+='Reproduction inputs are `manifest.json`, `paired_cases.json`, `assets/` and `code/`. Raw events/observations, source timestamps, actual/oracle actions and task records remain in `episodes/`. `validation.json`, `denominator_audit.json`, `mechanism_audit.json` and `preservation.json` hold the checks. The earlier 1,200-episode study, 216-episode staleness pilot and original checkpoint remain unchanged. Tables and vector/600 dpi figures are local artifacts. `analysis/` contains the supplemental read-only calculations and plotting code.\n'
(root/'FINDINGS.md').write_text(text,encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'serif','font.size':9,'legend.fontsize':8,'pdf.fonttype':42,
    'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
colors={2:'#009E73',4:'#0072B2',6:'#CC79A7',8:'#E69F00'}
styles={'allocated':(':','^'),'edge':('-','o'),'cloud':('--','s')}
fig,axes=plt.subplots(1,2,figsize=(7.1,3.1),layout='constrained')
for route in styles:
    for n,color in colors.items():
        rows=[groups[name(route),n],groups[route+'__aoi-1',n],groups[route+'__aoi-3',n]]
        for ax,field in zip(axes,('priority_fov_error_rate','stale_action_disagreement_rate')):
            valid=[(hold,r[field]) for hold,r in zip((0,1,3),rows) if r[field] is not None]
            ax.plot([p[0] for p in valid],[p[1] for p in valid],color=color,
                ls=styles[route][0],marker=styles[route][1],ms=3)
for ax,label in zip(axes,('Inference FoV\nordering-error fraction','Stale / fresh action\ndisagreement fraction')):
    ax.set(xlabel='Imposed peer-state hold (s)',ylabel=label,xticks=(0,1,3),ylim=(-.02,.5 if ax is axes[0] else .12))
handles=[Line2D([0],[0],color=color,label=f'{n} robots') for n,color in colors.items()]
handles.extend(Line2D([0],[0],color='black',ls=style[0],marker=style[1],label=labels[route]) for route,style in styles.items())
fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
for extension in ('pdf','svg','png'):
    fig.savefig(root/'figures'/('aoi_hold_sensitivity.'+extension),bbox_inches='tight',metadata={'CreationDate':None} if extension=='pdf' else None)
plt.close(fig)
print('Wrote FINDINGS.md and aoi_hold_sensitivity PDF/SVG/600 dpi PNG.')
