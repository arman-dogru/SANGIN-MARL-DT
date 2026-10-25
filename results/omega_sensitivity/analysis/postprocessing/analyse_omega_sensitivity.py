"""Supplemental objective-isolation, invariance and uncertainty checks; no missions run."""
from collections import defaultdict
from dataclasses import replace
import argparse
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sys

if __name__=='__main__' and len(sys.argv)>1:
    frozen=Path(sys.argv[1]).resolve()/'code'
    if frozen.is_dir():sys.path.insert(0,str(frozen))

from panama_uav.allocation import AllocationSnapshot, solve_binary
from panama_uav.communication_report import csv_table
from panama_uav.records import sha256

WEIGHTS=(('W1',1.,1.),('W2',1.,2.),('W3',1.,4.),('W4',2.,1.),('W5',4.,1.))
COUNTS=(2,4,8,16,32)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def same(a,b,tol=1e-9):
    if isinstance(a,(float,int)) and isinstance(b,(float,int)):return abs(a-b)<=tol*max(1,abs(a),abs(b))
    return a==b


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*((p*(1-p)+z*z/(4*n))/n)**.5/den
    return max(0.,center-radius),min(1.,center+radius)


def priority_by_location(path):
    totals=defaultdict(lambda:dict(tasks=0,weight_sum=0.,first_round_tasks=0,first_round_weight_sum=0.))
    with gzip.open(path,'rt',encoding='utf-8',newline='') as stream:
        for task in csv.DictReader(stream):
            if task['executed_x'] not in ('0','1'):continue
            key=(task['weight_id'],task['route'],int(task['num_agents']),'uav' if task['executed_x']=='1' else 'cloud')
            group=totals[key];weight=float(task['weight']);group['tasks']+=1;group['weight_sum']+=weight
            if int(task['cycle'])==0:group['first_round_tasks']+=1;group['first_round_weight_sum']+=weight
    return [dict(weight_id=w,route=r,num_agents=n,location=loc,tasks=d['tasks'],priority_resource_weight_mean=d['weight_sum']/d['tasks'],
        first_round_tasks=d['first_round_tasks'],first_round_priority_weight_mean=d['first_round_weight_sum']/d['first_round_tasks'] if d['first_round_tasks'] else None)
        for (w,r,n,loc),d in sorted(totals.items())]


def analyse(root):
    root=Path(root).resolve();m=read(root/'manifest.json')
    if read(root/'status.json')['status']!='complete' or m['expected_episodes']!=375:raise ValueError('A complete primary matrix is required')
    folder=root/'analysis';folder.mkdir(exist_ok=True);objective_rows=[];monotonic=[];invariant=[]
    cases=read(root/'cases.json')
    for case in cases:
        p=read(root/'episodes/W1/allocated-6ghz'/(case['case_id']+'.result.json'))
        for saved in p['first_snapshots']:
            snapshot=AllocationSnapshot.from_dict(saved['snapshot']);job=next(t for t in snapshot.tasks if t.task_id==saved['job_id'])
            rows=[]
            for name,latency,energy in WEIGHTS:
                result=solve_binary(replace(snapshot,latency_coefficient=latency,energy_coefficient=energy))
                task=next(t for t in result.tasks if t['task_id']==saved['job_id'])
                rows.append(dict(case_id=case['case_id'],num_agents=case['num_agents'],job_id=saved['job_id'],weight_id=name,
                    ratio=latency/energy,latency_coefficient=latency,energy_coefficient=energy,
                    weighted_latency_seconds=result.latency_objective/latency,modeled_objective_energy_j=result.energy_objective/energy,
                    objective=result.objective,x=task['x'],new_assignment_free=job.fixed_x is None,
                    backhaul_available=job.backhaul_bps>0,fixed_previous_tasks=len(snapshot.tasks)-1))
            rows.sort(key=lambda r:r['ratio']);objective_rows.extend(rows)
            for low,high in zip(rows,rows[1:]):
                lat_ok=high['weighted_latency_seconds']<=low['weighted_latency_seconds']+1e-7*max(1,abs(low['weighted_latency_seconds']))
                en_ok=high['modeled_objective_energy_j']>=low['modeled_objective_energy_j']-1e-7*max(1,abs(low['modeled_objective_energy_j']))
                monotonic.append(dict(case_id=case['case_id'],job_id=saved['job_id'],lower_ratio=low['ratio'],higher_ratio=high['ratio'],
                    latency_nonincreasing=lat_ok,energy_nondecreasing=en_ok))
        base=read(root/'episodes/W1/cloud-6ghz'/(case['case_id']+'.result.json'))
        for name,_,_ in WEIGHTS[1:]:
            other=read(root/'episodes'/name/'cloud-6ghz'/(case['case_id']+'.result.json'))
            fields=('success','steps','waiting_steps','executed_drone_tasks','executed_cloud_tasks',
                'deadline_commands_on_time','deadline_commands_late','deadline_commands_undelivered',
                'modeled_energy_lower_j','task_latency_p95_seconds','active_peer_aoi_sum','active_peer_aoi_samples')
            differences=[k for k in fields if not same(base['row'][k],other['row'][k])]
            a={t['job_id']:t for t in base['tasks']};b={t['job_id']:t for t in other['tasks']}
            shared=a.keys()&b.keys();physical_fields=('executed_x','observed_latency_seconds','predicted_latency_seconds','compute_service_seconds',
                                                   'robot_energy_j','backhaul_energy_j','drone_compute_energy_j','deadline_status')
            task_differences=sum(any(not same(a[j].get(k),b[j].get(k)) for k in physical_fields) for j in shared)
            max_latency=max((abs(a[j]['observed_latency_seconds']-b[j]['observed_latency_seconds']) for j in shared
                if a[j].get('observed_latency_seconds') is not None and b[j].get('observed_latency_seconds') is not None),default=0.)
            invariant.append(dict(case_id=case['case_id'],num_agents=case['num_agents'],weight_id=name,matching_tasks=a.keys()==b.keys(),
                row_metric_differences=differences,task_execution_differences=task_differences,max_latency_difference_seconds=max_latency,
                invariant=not differences and a.keys()==b.keys() and task_differences==0,tolerance=1e-9))
    csv_table(folder/'matched_snapshot_objectives.csv',objective_rows);csv_table(folder/'objective_monotonicity.csv',monotonic)
    csv_table(folder/'all_cloud_invariance.csv',invariant)
    summary=list(csv.DictReader((root/'summary.csv').open(encoding='utf-8')))
    table={(r['weight_id'],r['route'],int(r['num_agents'])):r for r in summary}
    pareto=[]
    for n in COUNTS:
        for w,_,_ in WEIGHTS:
            row=table[(w,'allocated',n)];lat=float(row['observed_latency_p95_seconds']);en=float(row['mean_episode_energy_j'])
            dominated=[]
            for v,_,_ in WEIGHTS:
                other=table[(v,'allocated',n)];ol=float(other['observed_latency_p95_seconds']);oe=float(other['mean_episode_energy_j'])
                if ol<=lat and oe<=en and (ol<lat or oe<en):dominated.append(v)
            lo,hi=wilson(int(row['successful_missions']),int(row['cases']))
            pareto.append(dict(weight_id=w,num_agents=n,p95_ms=lat*1000,energy_episode_j=en,energy_task_j=float(row['energy_per_generated_task_j']),
                cloud_percent=float(row['cloud_percent']),miss_percent=float(row['deadline_miss_percent']),successful_missions=int(row['successful_missions']),
                mission_success_wilson95_low=lo,mission_success_wilson95_high=hi,dominated_by=dominated,empirical_pareto=not dominated))
    csv_table(folder/'allocated_tradeoffs.csv',pareto)
    exposures=list(csv.DictReader((root/'handover.csv').open(encoding='utf-8')))
    exposure_table={(r['weight_id'],int(r['num_agents']),r['exposure']):r for r in exposures if r['route']=='allocated'}
    availability=[]
    for n in COUNTS:
        for w,_,_ in WEIGHTS:
            row=table[(w,'allocated',n)];available=exposure_table.get((w,n,'available_backhaul'),{})
            unavailable=exposure_table.get((w,n,'unavailable_backhaul'),{})
            count=int(unavailable.get('tasks',0));forced=int(unavailable.get('uav_tasks',0));free=int(available.get('tasks',0))
            availability.append(dict(weight_id=w,num_agents=n,all_tasks=int(row['tasks']),unavailable_admissions=count,
                unavailable_percent=100*count/int(row['tasks']),unavailable_uav_tasks=forced,
                uav_tasks_with_unavailable_backhaul_percent=100*forced/int(row['uav_tasks']) if int(row['uav_tasks']) else None,
                available_admissions=free,cloud_percent_when_available=100*int(available.get('cloud_tasks',0))/free if free else None,
                fixed_previous_tasks_mean=float(row['fixed_previous_tasks_mean'])))
    csv_table(folder/'availability_attribution.csv',availability)
    priority=priority_by_location(root/'tasks.csv.gz');csv_table(folder/'priority_by_location.csv',priority)
    # Regenerate presentation-only figures without changing a simulation record.
    plot_source=Path(__file__).with_name('report_omega_sensitivity.py')
    spec=importlib.util.spec_from_file_location('omega_publication_figures',plot_source)
    plotting=importlib.util.module_from_spec(spec);spec.loader.exec_module(plotting)
    def numeric(row):
        result={}
        for key,value in row.items():
            if value=='':result[key]=None
            else:
                try:result[key]=float(value)
                except ValueError:result[key]=value
        return result
    plotting.figures(root,[numeric(row) for row in summary])
    (folder/'postprocessing_provenance.json').write_text(json.dumps(dict(analysis_sha256=sha256(__file__),
        figure_source_sha256=sha256(plot_source),simulation_changes=0,solver_changes=0),indent=2)+'\n',encoding='utf-8')
    postprocessing=folder/'postprocessing';postprocessing.mkdir(exist_ok=True)
    for source in (Path(__file__),plot_source,Path(__file__).with_name('audit_omega_sensitivity.py')):
        shutil.copy2(source,postprocessing/source.name)
    lines=['## Interpretation and robustness','',
        'All latency values use the simulator clock. Host inference/solver wall times are diagnostic and do not drive modeled CPU service. '
        'Cloud-computation energy is absent from the preserved objective and mission energy model. Weight effects on modeled energy therefore do not establish whole-system hardware energy savings.','',
        'The paired cases and orbital offsets match within each population. Maps/initial conditions and orbital offsets differ between populations, so cross-population trends cannot be attributed to robot count alone. '
        'First-round admissions, active-count strata and route-availability strata help interpret that limitation.','',
        'Stable placement percentages can hide changes in which priority levels run at each location. '
        'analysis/priority_by_location.csv reports admission resource weights by execution location; the priority policy and range remain unchanged. '
        + '; '.join(f"two robots, {w}: mean UAV resource weight "+f"{next(p['priority_resource_weight_mean'] for p in priority if p['weight_id']==w and p['route']=='allocated' and p['num_agents']==2 and p['location']=='uav'):.2f}, "
            +f"active-job UAV frequency {float(table[(w,'allocated',2)]['drone_active_job_frequency_cps'])/1e9:.2f} GHz" for w in ('W1','W3','W4','W5'))+'.','',
        'The confidence intervals in comparisons.csv resample paired cases. The latency contrast is the mean difference in per-case p95, whereas summary.csv pools delivered tasks to compute each cell p95. '
        'These are distinct estimands. Task counts vary with mission duration and completion, and generated-task denominators remain explicit.','',
        'analysis/availability_attribution.csv separates unavailable-backhaul admissions from available-route choices, including the fraction of UAV assignments exposed to unavailability. '
        'Fixed previous jobs remain in each snapshot. The matched-snapshot solves identify direct objective effects under those fixed inputs; a unique mission-wide decomposition of objective, availability and trajectory feedback is not identified by this parameter-only design.','',
        f"Matched admissions: {len(objective_rows)//5} identical W1 snapshots, each solved at all five weight ratios. "
        f"Of {len(monotonic)} adjacent-ratio checks, {sum(not(r['latency_nonincreasing'] and r['energy_nondecreasing']) for r in monotonic)} violate the expected weighted-latency/energy monotonicity beyond tolerance. "
        'The objective uses a priority-weighted sum of snapshot latencies. The reported tail percentile and command-deadline rate describe other aspects of the realized schedule, including future radio service and admission feedback.','',
        f"All-cloud invariance: {sum(r['invariant'] for r in invariant)}/{len(invariant)} paired weight-versus-W1 cases match the reported execution metrics within 1e-9. "
        'All simulations were executed; no controls were reused. Detailed differences, if present, are retained in analysis/all_cloud_invariance.csv.','',
        '| Robots | Weight | Cloud (%) | E2E p95 (ms) | Energy/episode (J) | Energy/generated task (J) | Misses (%) | Success | Empirical Pareto |',
        '|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|']
    for row in pareto:
        lines.append(f"| {row['num_agents']} | {row['weight_id']} | {row['cloud_percent']:.2f} | {row['p95_ms']:.2f} | {row['energy_episode_j']:.3f} | {row['energy_task_j']:.5f} | {row['miss_percent']:.2f} | {row['successful_missions']}/5 | {'yes' if row['empirical_pareto'] else 'no'} |")
    lines+=['','Pareto membership compares only the tested allocated settings using pooled p95 latency and mean episode energy. It is a descriptive pilot result and does not account for statistical uncertainty or impose a preferred allocation split. '
        'Success Wilson intervals and the full paired contrasts are retained in the CSVs.','']
    if (root/'validation_revision.json').exists():
        audit=read(root/'baseline_validation.json')
        maximum=max(r['relaxed_x_max_absolute_difference'] for r in audit['details'])
        lines+=['### Reproduction audit correction','',
            f"The initial harness rejected an unused fractional minimizer distance; the maximum recorded fractional-x difference is {maximum:.9g}. "
            'The original rejection and status are preserved in baseline_validation_v1.json and status_v1_gate_failure.json. '
            'The revised audit retains exact executed-output matching and checks fractional objectives and residuals against the unchanged solver tolerances. '
            'Production, simulator, policy and solver changes: zero. Completed episodes were not rerun. validation_revision.json records the isolated audit/controller hashes.','']
    status=read(root/'status.json')
    lines += [f"Total study wall time: {status['elapsed_seconds']/3600:.3f} hours, including baseline reproduction, audit recovery and report generation. "
              'The fixed 24-hour lease was preserved during recovery. No learning updates occurred.','']
    marker='\n## Interpretation and robustness\n';text=(root/'report.md').read_text(encoding='utf-8').split(marker)[0]
    (root/'report.md').write_text(text.rstrip()+'\n\n'+'\n'.join(lines),encoding='utf-8')
    # Preserve the executed harness bytes alongside the source/configuration hashes.
    for source in list((root/'code').glob('*.py'))+list((root/'validation_v2').glob('*.py')):
        destination=folder/'frozen_harness'/source.parent.name/source.name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    result=dict(matrix_simulations=375,matched_admissions=len(objective_rows)//5,monotonicity_violations=sum(not(r['latency_nonincreasing'] and r['energy_nondecreasing']) for r in monotonic),
        all_cloud_invariant_pairs=sum(r['invariant'] for r in invariant),all_cloud_comparison_pairs=len(invariant),wall_hours=status['elapsed_seconds']/3600,
        allocated_tradeoffs=pareto,availability_attribution=availability,priority_by_location=priority,policy_updates=0)
    (folder/'findings.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    print(json.dumps(analyse(args.root),indent=2))
