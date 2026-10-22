"""Bounded, paired controlled stress pilot; never a nominal network study."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import csv
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np

from .config import ProjectConfig
from .controlled_scenarios import FAMILIES, make_cases, scenario_for
from .evaluation import run_episode
from .models import choose_device, select_action
from .priority_evaluation import paired_interval
from .records import sha256, write_json
from .resource_study import conditions, episode_metrics
from .staleness_audit import audit_stress
from .staleness_remote import controlled_class
from .training import load_model

DELAYS = (0, 1, 2, 3, 4, 5)
TREATMENTS = ('fresh_transport', 'equal', 'priority')
RATIOS = dict(inference_fov_rate=('inference_fov_errors','inference_relevant_pairs'),
    physical_fov_order_error_rate=('physical_fov_order_errors','physical_relevant_pairs'),
    stale_fresh_rate=('stale_fresh_disagreements','inferences'),
    injected_delay_rate=('injected_delay_disagreements','inferences'),
    timely_stale_rate=('timely_stale_disagreements','timely_comparisons'),
    timely_injected_rate=('timely_injected_disagreements','timely_comparisons'),
    command_on_time_rate=('deadline_commands_on_time','tasks_generated'),
    command_late_rate=('deadline_commands_late','tasks_generated'),
    command_undelivered_rate=('deadline_commands_undelivered','tasks_generated'),
    relevant_pair_fraction=('inference_relevant_pairs','inference_pairs'),
    physical_pair_fraction=('physical_relevant_pairs','inference_pairs'),
    mean_peer_aoi=('peer_aoi_sum','inference_pairs'),
    mean_relevant_peer_aoi=('relevant_peer_aoi_sum','physical_relevant_pairs'))
SCALARS = ('success', 'restricted_makespan_seconds', 'waiting_steps', 'command_cutoff_misses',
    'physical_conflict_events', 'attempted_vertex_conflicts', 'attempted_edge_swaps')


def stress_treatments(config):
    p = config.resource_allocation
    if (not p.enabled or p.execution_mode != 'remote_dt' or not p.oracle_shadow or
        config.communication.seconds_per_step != 1 or p.enforce_task_deadlines or
        (p.decision_budget_seconds,p.command_cutoff_seconds,p.task_deadline_seconds) != (.25,.25,.25) or
        p.equal_weight_mode != 'matched_active_mean'):
        raise ValueError('Controlled stress requires remote DT, shadow diagnostics, matched equal and 250 ms / 1 s settings')
    selected = {'fresh':'fresh_transport', 'up-1-cpu-1__equal':'equal', 'up-1-cpu-1__priority':'priority'}
    return [(selected[name],cfg) for name,cfg,_ in conditions(config) if name in selected]


def stress_metrics(row, trace):
    row = episode_metrics(row, trace)
    shadows = [e for e in trace['oracle_events'] if e['event']=='controlled_peer_shadow']
    by_job = {e['job_id']:e for e in shadows}
    pairs = [p for e in shadows for p in e['pairs']]
    active = [e for e in trace['remote_events'] if e['event']=='decision_event' and not e['completed_local_control']]
    timely = [by_job[e['command']['job_id']] for e in active if e['command'] is not None]
    motion = [e for e in trace['remote_events'] if e['event']=='motion_event']
    exposed_cycles = {int(e['job_id'].split(':')[0]) for e in shadows
                      if any(p['physical_relevant'] for p in e['pairs'])}
    moving_cycles = {e['cycle'] for e in motion if e['before'] != e['after']}
    from .remote_control import conflicts, DELTAS
    attempted_vertex = attempted_edge = 0
    grid = np.asarray(trace['grid'])
    for event in motion:
        positions = {int(k):tuple(v) for k,v in event['before'].items()}
        targets = {}
        for robot,point in positions.items():
            dr,dc = DELTAS[event['actions'][str(robot)]]
            cell = point[0]+dr,point[1]+dc
            targets[robot] = cell if 0<=cell[0]<grid.shape[0] and 0<=cell[1]<grid.shape[1] and not grid[cell] else point
        for conflict in conflicts(positions,targets):
            attempted_vertex += conflict['kind']=='vertex'
            attempted_edge += conflict['kind']=='edge_swap'
    row.update(inferences=len(shadows), inference_pairs=len(pairs),
        inference_relevant_pairs=sum(p['relevant'] for p in pairs),
        inference_fov_errors=sum(p['fov_error'] for p in pairs),
        physical_relevant_pairs=sum(p['physical_relevant'] for p in pairs),
        physical_fov_order_errors=sum(p['physical_relevant']*p['ordering_error'] for p in pairs),
        inference_vertex_pairs=sum(bool(p['vertex_conflict_steps']) for p in pairs),
        inference_edge_pairs=sum(bool(p['edge_conflict_steps']) for p in pairs),
        physical_vertex_pairs=sum(bool(p['physical_vertex_conflict_steps']) for p in pairs),
        physical_edge_pairs=sum(bool(p['physical_edge_conflict_steps']) for p in pairs),
        peer_aoi_sum=sum(p['peer_age_seconds'] for p in pairs),
        relevant_peer_aoi_sum=sum(p['peer_age_seconds']*p['physical_relevant'] for p in pairs),
        physical_relevant_inferences=sum(any(p['physical_relevant'] for p in e['pairs']) for e in shadows),
        unique_physical_directed_pairs=len({(p['receiver'],p['peer']) for p in pairs if p['physical_relevant']}),
        coordination_exposed_cycles=len(exposed_cycles), progressing_exposure_cycles=len(exposed_cycles & moving_cycles),
        stale_fresh_disagreements=sum(e['stale_vs_fresh_disagreement'] for e in shadows),
        injected_delay_disagreements=sum(e['injected_delay_disagreement'] for e in shadows),
        timely_comparisons=len(timely),
        timely_stale_disagreements=sum(e['stale_vs_fresh_disagreement'] for e in timely),
        timely_injected_disagreements=sum(e['injected_delay_disagreement'] for e in timely),
        active_decisions=len(active), attempted_vertex_conflicts=attempted_vertex,
        attempted_edge_swaps=attempted_edge,
        inference_vertex_only_errors_exploratory=sum(p['vertex_e2e_error'] for p in pairs))
    for metric,(numerator,denominator) in RATIOS.items():
        row[metric] = row[numerator]/row[denominator] if row[denominator] else None
    if row['timely_comparisons'] + row['command_cutoff_misses'] != len(active):
        raise ValueError('Timely / cutoff decision partition mismatch')
    return row


def _initialize(configuration, model_path, diagnostic):
    import torch
    global _MODEL, _VERSION, _MODEL_SHA
    torch.set_num_threads(1)
    torch.manual_seed(42)
    _MODEL,payload = load_model(model_path, ProjectConfig.from_dict(configuration), choose_device('cpu'),
        require_qualified=not diagnostic)
    _VERSION = payload.get('learner_updates',0)
    _MODEL_SHA = sha256(model_path)


def _inference(location, robot, observation):
    return select_action(_MODEL, observation, 0, np.random.default_rng(0), choose_device('cpu'))


def _episode(argument):
    configuration, case, delay, treatment, directory, cap = argument
    config = ProjectConfig.from_dict(configuration)
    config = replace(config, communication=replace(config.communication,
        backhaul_time_offset_seconds=config.communication.backhaul_time_offset_seconds + case['orbital_offset_seconds']))
    folder = Path(directory)
    manifest_path = folder.parent.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    row,trace = run_episode(config, _MODEL, case, choose_device('cpu'), _VERSION, max_steps=cap,
        scenario=scenario_for(case), remote_communication_class=controlled_class(delay))
    trace['provenance'] = dict(model_sha256=_MODEL_SHA,
        case_id=case['case_id'], map_sha256=case['map_sha256'], delay_steps=delay, treatment=treatment,
        manifest_sha256=sha256(manifest_path), timeline_sha256=manifest['timeline_sha256'],
        case_bank_sha256=manifest['case_bank_sha256'])
    trace['stress_configuration'] = config.to_dict()
    if trace['frames'][0]['positions'] != case['starts']:
        raise ValueError('Controlled paired starts changed')
    row = stress_metrics(row,trace)
    row.update(delay_steps=delay, treatment=treatment)
    trace['stress_episode'] = row
    # Preserve a failed audit snapshot rather than substituting a policy.
    path = folder / f"{case['case_id']}.json.gz"
    with gzip.open(path,'wt',encoding='utf-8',compresslevel=4) as stream:
        stream.write(json.dumps(trace,allow_nan=False,separators=(',',':')))
    audit = audit_stress(trace, _inference)
    return row, dict(case_id=case['case_id'],delay_steps=delay,treatment=treatment,
        archive=str(path),archive_sha256=sha256(path),**audit)


def _table(path, rows):
    rows = list(rows)
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def _contrasts(episodes, samples):
    groups = defaultdict(list)
    for r in episodes:
        groups[r['scenario'],r['num_agents'],r['delay_steps'],r['treatment']].append(r)
    result = []
    for (family,n,delay,treatment), rows in groups.items():
        comparisons = [('delay_vs_zero',groups[family,n,0,treatment])]
        if treatment == 'priority':
            comparisons.append(('priority_vs_equal',groups[family,n,delay,'equal']))
        for comparison, reference in comparisons:
            for metric in (*SCALARS,*RATIOS):
                result.append(dict(scenario=family,num_agents=n,delay_steps=delay,treatment=treatment,
                    comparison=comparison,**paired_interval(rows,reference,metric,
                        ratio=None if metric.startswith('mean_') else RATIOS.get(metric),samples=samples)))
        if treatment == 'priority':
            # Case-level difference-in-differences; preserve all four paired arms.
            keys = {r['case_id'] for r in rows}
            zero_p = {r['case_id']:r for r in groups[family,n,0,'priority']}
            equal = {r['case_id']:r for r in groups[family,n,delay,'equal']}
            zero_e = {r['case_id']:r for r in groups[family,n,0,'equal']}
            assert keys == set(zero_p) == set(equal) == set(zero_e)
            for metric in (*SCALARS,*RATIOS):
                first,second = [],[]
                for r in rows:
                    key = r['case_id']
                    values = r[metric],zero_p[key][metric],equal[key][metric],zero_e[key][metric]
                    p,e = (float(values[0])-float(values[1]),float(values[2])-float(values[3])) if all(v is not None for v in values) else (None,None)
                    first.append(dict(case_id=key,value=p))
                    second.append(dict(case_id=key,value=e))
                contrast = paired_interval(first,second,'value',samples=samples)
                contrast['metric'] = metric
                contrast['estimator'] = 'difference of mean paired case changes (priority minus equal); conditional rates are case means'
                result.append(dict(scenario=family,num_agents=n,delay_steps=delay,treatment=treatment,
                    comparison='priority_mitigation_interaction',**contrast))
    return result


def report_stress(root, samples=2000):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    bank = json.loads((root/'paired_cases.json').read_text())
    if sha256(root/'paired_cases.json') != manifest['case_bank_sha256']:
        raise ValueError('Controlled case bank checksum mismatch')
    expected = {(c['case_id'],d,t) for c in bank for d in manifest['delays'] for t in TREATMENTS}
    episodes = [json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines()]
    if len(episodes) != len(expected) or {(r['case_id'],r['delay_steps'],r['treatment']) for r in episodes} != expected:
        raise ValueError('Missing or duplicate controlled paired episodes')
    by_case = {c['case_id']:c for c in bank}
    for r in episodes:
        if any(r[k] != v for k,v in by_case[r['case_id']].items()):
            raise ValueError('Controlled paired case identity mismatch')
    audits = json.loads((root/'audits.json').read_text())
    if len(audits) != len(episodes) or any(a['status'] != 'passed' or sha256(a['archive']) != a['archive_sha256'] for a in audits):
        raise ValueError('Controlled replay archive validation mismatch')
    rows_by_key = {(r['case_id'],r['delay_steps'],r['treatment']):r for r in episodes}
    if {(a['case_id'],a['delay_steps'],a['treatment']) for a in audits} != expected:
        raise ValueError('Controlled audit pairing mismatch')
    treatment_path = root/'treatments.json'
    if sha256(treatment_path) != manifest['treatments_sha256']:
        raise ValueError('Controlled treatment identity mismatch')
    treatments = json.loads(treatment_path.read_text())
    from .scenarios import load_map
    for case in bank:
        if sha256(case['map_path']) != case['map_sha256']:
            raise ValueError('Controlled map identity mismatch')
    for audit in audits:
        with gzip.open(audit['archive'],'rt',encoding='utf-8') as stream:
            trace=json.load(stream)
        key=audit['case_id'],audit['delay_steps'],audit['treatment']
        row=rows_by_key[key]
        if trace['stress_episode'] != row or not np.array_equal(trace['grid'],load_map(row['map_path'])):
            raise ValueError('Controlled episode CSV differs from replay archive')
        provenance=trace['provenance']
        if any(provenance[k] != v for k,v in dict(model_sha256=manifest['model_sha256'],
            timeline_sha256=manifest['timeline_sha256'],case_bank_sha256=manifest['case_bank_sha256'],
            manifest_sha256=sha256(root/'manifest.json'),map_sha256=row['map_sha256']).items()):
            raise ValueError('Controlled archive provenance mismatch')
        expected_config=treatments[key[2]]
        expected_config=json.loads(json.dumps(expected_config))
        expected_config['communication']['backhaul_time_offset_seconds']+=row['orbital_offset_seconds']
        expected_config['resource_allocation']['failure_directory']=str(Path(audit['archive']).parent/'failures')
        if trace['stress_configuration'] != expected_config:
            raise ValueError('Unexpected controlled execution configuration change')
    grouped = defaultdict(list)
    for row in episodes:
        grouped[row['scenario'],row['num_agents'],row['delay_steps'],row['treatment']].append(row)
    summary = []
    count_fields = set(v for pair in RATIOS.values() for v in pair) | {
        'inference_vertex_pairs','inference_edge_pairs','physical_vertex_pairs','physical_edge_pairs',
        'command_cutoff_misses','attempted_vertex_conflicts','attempted_edge_swaps','physical_conflict_events',
        'inference_vertex_only_errors_exploratory','active_decisions'}
    count_fields.update(('physical_relevant_inferences','unique_physical_directed_pairs',
                         'coordination_exposed_cycles','progressing_exposure_cycles'))
    for (family,n,delay,treatment),rows in grouped.items():
        item = dict(scenario=family,num_agents=n,delay_steps=delay,treatment=treatment,cases=len(rows),
            mission_success_rate=float(np.mean([r['success'] for r in rows])),
            mean_capped_makespan=float(np.mean([r['restricted_makespan_seconds'] for r in rows])),
            mean_waiting_steps=float(np.mean([r['waiting_steps'] for r in rows])))
        item.update({k:sum(r[k] for r in rows) for k in count_fields})
        for metric,(a,b) in RATIOS.items():
            item[metric] = item[a]/item[b] if item[b] else None
        summary.append(item)
    contrasts = _contrasts(episodes,samples)
    _table(root/'episodes.csv',episodes)
    _table(root/'summary.csv',summary)
    _table(root/'paired_differences.csv',contrasts)
    write_json(root/'summary.json',summary)
    _plots(root,summary)
    write_json(root/'validation.json',dict(status='passed',episodes=len(episodes),paired_cases=len(bank),
        archive_checksums_verified=True,paired_identity=True,
        actual_actions_verified=sum(r['actual_actions_verified'] for r in audits),
        peer_shadow_actions_verified=sum(r['peer_shadow_actions_verified'] for r in audits),
        cutoff_oracles_verified=sum(r['cutoff_oracles_verified'] for r in audits)))
    write_json(root/'analysis_provenance.json',dict(run_commit=manifest['git_commit'],
        analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
        analysis_source_sha256=sha256(__file__),bootstrap_samples=samples,bootstrap_seed=42,
        episodes_sha256=sha256(root/'episodes.jsonl'),case_bank_sha256=manifest['case_bank_sha256']))
    return summary


def _plots(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(fresh_transport='tab:gray',equal='tab:orange',priority='tab:blue')
    for metric,label in (('timely_stale_rate','Used-command disagreement\n(stale versus fresh peers)'),
        ('physical_pair_fraction','Physically relevant pairs /\nall directed pairs'),
        ('mission_success_rate','Mission success rate'),('mean_capped_makespan','Mean capped makespan (s)'),
        ('command_on_time_rate','On-time command rate'),('mean_peer_aoi','Mean peer source AoI (s)')):
        counts = sorted({r['num_agents'] for r in summary})
        fig,axes = plt.subplots(len(counts),3,figsize=(12,3*len(counts)),squeeze=False,layout='constrained')
        for i,n in enumerate(counts):
            for j,family in enumerate(FAMILIES):
                ax=axes[i,j]
                for treatment in TREATMENTS:
                    rows=sorted((r for r in summary if r['scenario']==family and r['num_agents']==n and r['treatment']==treatment),key=lambda r:r['delay_steps'])
                    style=dict(fresh_transport='s:',equal='^--',priority='o-')[treatment]
                    ax.plot([r['delay_steps'] for r in rows],[np.nan if r[metric] is None else r[metric] for r in rows],
                        style,label=treatment,color=colors[treatment])
                ax.set(title=f'{family.replace("_"," ")}, {n} robots',xlabel='Injected peer delay (1 s steps)',ylabel=label)
                if metric not in ('mean_capped_makespan','mean_peer_aoi'):
                    ax.set_ylim(-.03,1.03)
                ax.grid(alpha=.2)
        fig.suptitle('Controlled stress pilot; not nominal network measurements\nDescriptive curves; paired confidence intervals in CSV')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
        fig.savefig(root/f'{metric}.png',dpi=150,bbox_inches='tight')
        plt.close(fig)


def run_stress_pilot(config, output, model_path, *, cases=2, counts=(2,4), delays=DELAYS,
                     max_steps=60, workers=2, diagnostic=False):
    # Deliberately no full-study override: a new reviewed protocol is required.
    if cases not in (1,2) or not 1<=max_steps<=80 or not 1<=workers<=4:
        raise ValueError('Pilot only: 1--2 cases, at most 80 steps and 1--4 workers; full sweep is gated')
    if not counts or not set(counts)<={2,4,6,8} or len(set(counts))!=len(counts):
        raise ValueError('Use distinct populations 2/4/6/8')
    if not delays or 0 not in delays or not set(delays)<=set(DELAYS) or len(set(delays))!=len(delays):
        raise ValueError('Use distinct delays 0--5, including zero')
    treatments = stress_treatments(config)
    model_path = str(Path(model_path).resolve())
    device = choose_device('cpu')
    _,payload = load_model(model_path,config,device,require_qualified=not diagnostic)
    timeline = config.resolve(config.communication.leo_timeline) if config.communication.leo_timeline else None
    if not timeline or not timeline.is_file():
        raise ValueError('Controlled pilot requires the archived orbital timeline')
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    bank = make_cases(output/'maps',cases,counts)
    write_json(output/'paired_cases.json',bank)
    code_root = Path(__file__).parent
    write_json(output/'treatments.json',{name:cfg.to_dict() for name,cfg in treatments})
    manifest = dict(schema=1,label='controlled stress pilot; not nominal network measurements',
        created_utc=datetime.now(timezone.utc).isoformat(),diagnostic_model=diagnostic,
        qualification_scope='original map only; controlled maps are out-of-distribution stress tests',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code_root,text=True).strip(),
        code_sha256={p.name:sha256(p) for p in code_root.glob('*.py')},
        configuration=config.to_dict(),model_path=model_path,model_sha256=sha256(model_path),
        qualification=payload.get('qualification'),timeline_sha256=sha256(timeline),
        case_bank_sha256=sha256(output/'paired_cases.json'),cases_per_density=cases,
        treatments_sha256=sha256(output/'treatments.json'),
        counts=list(counts),delays=list(delays),max_steps=max_steps,treatments=list(TREATMENTS),
        episodes=len(bank)*len(delays)*len(TREATMENTS),bootstrap_unit='paired case',
        bootstrap_samples=2000,bootstrap_seed=42,confidence='pointwise 95%; no multiplicity correction',
        vertex_only='exploratory, separate from established vertex-plus-edge measures')
    write_json(output/'manifest.json',manifest)
    # Archive exactly the treatment configurations, with unchanged physical kernels.
    arguments=[]
    for delay in delays:
        for treatment,cfg in treatments:
            folder=output/f'delay-{delay}'/treatment
            folder.mkdir(parents=True)
            cfg=replace(cfg,resource_allocation=replace(cfg.resource_allocation,failure_directory=str(folder/'failures')))
            arguments.extend((cfg.to_dict(),case,delay,treatment,str(folder),max_steps) for case in bank)
    audits=[]
    write_json(output/'status.json',dict(status='running',completed=0,episodes=manifest['episodes']))
    try:
        with ProcessPoolExecutor(max_workers=workers,initializer=_initialize,
            initargs=(config.to_dict(),model_path,diagnostic)) as executor:
            # Only compact episode rows return to the parent; large traces stay on disk.
            for row,audit in executor.map(_episode,arguments):
                with (output/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                audits.append(audit)
                write_json(output/'status.json',dict(status='running',completed=len(audits),episodes=manifest['episodes']))
        write_json(output/'audits.json',audits)
        summary=report_stress(output)
    except BaseException as error:
        write_json(output/'status.json',dict(status='failed',completed=len(audits),error=str(error)))
        raise
    write_json(output/'status.json',dict(status='complete',completed=len(audits),episodes=manifest['episodes']))
    return summary
