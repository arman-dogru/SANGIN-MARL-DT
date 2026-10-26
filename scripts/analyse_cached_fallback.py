"""Paired conditional effects and first-round causal-isolation checks."""
from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import run_cached_fallback as study
from panama_uav.communication_report import csv_table


METRICS={
    'mission_success_rate':('success',None),
    'makespan_seconds':('makespan_seconds',None),
    'waiting_steps':('waiting_completed_steps',None),
    'mean_case_p95_seconds':('task_latency_p95_seconds',None),
    'modeled_energy_j':('modeled_energy_lower_j',None),
    'energy_per_task_j':('modeled_energy_lower_j','tasks_generated'),
    'active_inference_peer_aoi_seconds':('active_peer_aoi_sum','active_peer_aoi_samples'),
    'xi_fov':('fov_errors','fov_relevant_pairs'),
    'xi_vertex_e2e_exploratory':('vertex_e2e_errors','vertex_relevant_pairs'),
    'physical_reference_order_error_rate':('physical_fov_order_errors','physical_relevant_pairs'),
    'action_order_error_rate':('action_dt_ordering_errors','action_order_known_pairs'),
    'stale_fresh_DQN_disagreement_rate':('stale_fresh_disagreements','inferences'),
    'actual_fresh_DQN_disagreement_rate':('decision_action_shadow_disagreements','decision_action_shadow_comparisons'),
    'command_on_time_rate':('deadline_commands_on_time','tasks_generated'),
    'cutoff_wait_rate':('deadline_cutoff_wait_tasks','active_decisions'),
    'attempted_vertex_conflicts_per_decision':('attempted_vertex_conflicts','active_decisions'),
    'attempted_edge_swaps_per_decision':('attempted_edge_swaps','active_decisions'),
}


def interval(actual,reference,metric,samples=2000):
    """Same case resample in both arms; undefined denominators remain undefined."""
    if metric not in METRICS:raise ValueError('Unknown paired metric')
    a={r['case_id']:r for r in actual};b={r['case_id']:r for r in reference}
    if len(a)!=len(actual) or len(b)!=len(reference) or a.keys()!=b.keys():
        raise ValueError('Identical unique attempted case identities are required')
    conditional=metric!='mission_success_rate'
    field,denominator=METRICS[metric]
    ids=sorted(i for i in a if (not conditional or (a[i]['success'] and b[i]['success']))
               and a[i].get(field) is not None and b[i].get(field) is not None)
    result=dict(metric=metric,cohort='completed_in_both_conditions' if conditional else 'all_attempted_missions',
        attempted_pairs=len(a),paired_cases=len(ids),paired_case_ids=ids,difference=None,
        ci95_low=None,ci95_high=None,valid_bootstrap_samples=0,single_pair_interval=len(ids)==1)
    if not ids:return result
    rng=np.random.default_rng(20261026);indices=rng.integers(0,len(ids),size=(samples,len(ids)))
    estimates=[];bootstrap=[];denominators=[]
    for group in (a,b):
        values=np.array([group[i][field] for i in ids],dtype=float)
        if denominator is None:
            estimates.append(float(values.mean()));bootstrap.append(values[indices].mean(axis=1));denominators.append(len(ids))
        else:
            weights=np.array([group[i].get(denominator,0) or 0 for i in ids],dtype=float)
            totals=weights[indices].sum(axis=1)
            denominators.append(float(weights.sum()))
            estimates.append(float(values.sum()/weights.sum()) if weights.sum()>0 else None)
            bootstrap.append(np.divide(values[indices].sum(axis=1),totals,
                out=np.full(samples,np.nan),where=totals>0))
    result.update(actual_denominator=denominators[0],reference_denominator=denominators[1])
    if any(value is None for value in estimates):return result
    changes=bootstrap[0]-bootstrap[1];changes=changes[np.isfinite(changes)]
    result.update(difference=estimates[0]-estimates[1],valid_bootstrap_samples=len(changes))
    if len(changes):
        lo,hi=np.quantile(changes,[.025,.975]);result.update(ci95_low=float(lo),ci95_high=float(hi))
    return result


def analyse(root):
    root=Path(root).resolve();manifest=study.verify(root,True)
    if study.core.read(root/'status.json')['status']!='complete':raise ValueError('Complete primary results required')
    primary=defaultdict(list);baseline=defaultdict(list);snapshot_checks=[]
    index={(i['weight_id'],i['condition'],i['case_id']):i for i in study.core.read(root/'baseline_index.json')}
    for path in sorted((root/'episodes').glob('*/*/*.result.json')):
        data=study.core.read(path);row=data['row'];weight=data['weight']['id'];key=weight,row['route'],row['num_agents']
        previous=index[weight,row['condition'],row['case_id']]
        if study.sha256(previous['packet'])!=previous['packet_sha256']:raise ValueError('Baseline changed')
        old=study.core.read(previous['packet'])
        for field in ('case_id','num_agents','seed','starts','goals','orbital_offset_seconds','map_sha256'):
            if row[field]!=old['row'][field]:raise ValueError('Paired identity changed: '+field)
        primary[key].append(row);baseline[key].append(old['row'])
        snapshots=data['first_snapshots'];references=old['first_snapshots']
        inputs_equal=study.core.fingerprint([s['snapshot'] for s in snapshots])==study.core.fingerprint([s['snapshot'] for s in references])
        placements_equal=[[s['job_id'],[(r['task_id'],r['x']) for r in s['assignments']]] for s in snapshots]==[
            [s['job_id'],[(r['task_id'],r['x']) for r in s['assignments']]] for s in references]
        snapshot_checks.append(dict(weight_id=weight,route=row['route'],num_agents=row['num_agents'],case_id=row['case_id'],
            initial_snapshot_inputs_identical=inputs_equal,initial_binary_assignments_identical=placements_equal,
            initial_snapshots=len(snapshots)))
        if not inputs_equal or not placements_equal:raise ValueError('Initial allocator/radio behavior changed before fallback motion')
    if sum(map(len,primary.values()))!=375:raise ValueError('Incomplete primary cohort')
    results=[]
    for (weight,route,n),rows in sorted(primary.items()):
        for metric in METRICS:
            results.append(dict(weight_id=weight,route=route,num_agents=n,reference='original_WAIT_same_weight',
                reference_weight=weight,**interval(rows,baseline[weight,route,n],metric)))
            if weight!='W1':
                results.append(dict(weight_id=weight,route=route,num_agents=n,reference='cached_fallback_W1',
                    reference_weight='W1',**interval(rows,primary['W1',route,n],metric)))
    folder=root/'supplementary_analysis';folder.mkdir(exist_ok=True)
    csv_table(folder/'paired_effects.csv',results);csv_table(folder/'first_round_isolation.csv',snapshot_checks)
    weights=['W1','W2','W3','W4','W5']
    lines=['# Cached fallback paired effects','',
        'Success differences use all five attempted paired missions. Other metrics use cases completed in both '
        'conditions. Ratios retain their event/task exposure denominators. Intervals use 2,000 paired case resamples. '
        'One-pair intervals are degenerate, and zero-exposure effects remain undefined.','',
        'All 375 initial-round snapshot input sets and binary assignment sets match their original WAIT references. '
        'Placement and timing changes later in a mission can therefore reflect changed trajectories, task counts '
        'and delivered state, rather than a changed allocator or radio kernel.','',
        'Actual-versus-fresh-DQN disagreements include the change from a remote neural action to a local cached '
        'heuristic when the command is missing. They do not isolate stale-DQN-input effects. The preserved '
        'stale-versus-fresh DQN shadow at computation completion is reported separately.','',
        '## Mission-success differences: cached minus WAIT','',
        '| Robots | Weight | Policy | Difference (percentage points), 95% CI |',
        '|---:|:---:|---|---:|']
    for n in study.core.COUNTS:
        for w in weights:
            for route in study.core.CONDITIONS:
                effect=next(r for r in results if r['weight_id']==w and r['route']==route['route'] and r['num_agents']==n
                    and r['reference']=='original_WAIT_same_weight' and r['metric']=='mission_success_rate')
                lines.append(f"| {n} | {w} | {route['name']} | {effect['difference']*100:+.1f} [{effect['ci95_low']*100:+.1f}, {effect['ci95_high']*100:+.1f}] |")
    (folder/'paired_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result=dict(status='passed',initial_snapshot_case_checks=len(snapshot_checks),all_initial_inputs_and_placements_identical=True,
        paired_effects=len(results),case_bootstrap_samples=2000,source_sha256=study.sha256(__file__))
    study.core.write(folder/'validation.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    print(json.dumps(analyse(args.root),indent=2))
