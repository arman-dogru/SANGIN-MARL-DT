"""Publish completed-mission summaries from immutable, already audited omega runs."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_omega_sensitivity as report

from panama_uav.communication_report import csv_table, paired_changes
from panama_uav.records import sha256

IDENTITY_FIELDS = ('weight_id', 'latency_coefficient', 'energy_coefficient',
                   'case_id', 'num_agents', 'route', 'condition')
GROUP_FIELDS = ('weight_id', 'latency_coefficient', 'energy_coefficient',
                'condition', 'route', 'num_agents')
METRICS = ('makespan_seconds', 'waiting_steps', 'case_command_p95_seconds',
           'command_on_time_rate', 'modeled_energy_j',
           'mean_drone_cpu_utilization', 'mean_cloud_cpu_utilization', 'cutoff_waits')
COUNT_FIELDS = ('tasks', 'uav_tasks', 'cloud_tasks', 'unassigned_tasks',
                'first_round_tasks', 'first_round_uav', 'first_round_cloud',
                'commands', 'on_time_commands', 'late_commands', 'undelivered_commands',
                'censored_tasks', 'pending_tasks', 'cutoff_waits',
                'observed_latency_samples', 'active_peer_aoi_samples',
                'handover_tasks', 'unavailable_backhaul_admissions', 'observed_mission_seconds')


def completed_summary(rows, tasks, cpu, identity, template, attempted):
    """An empty completed cohort has counts of zero and undefined outcome metrics."""
    if any(not row['success'] for row in rows):
        raise ValueError('Unsuccessful mission in completed cohort')
    if rows:
        value = report.summarize(rows, tasks, cpu)
    else:
        if tasks or cpu:
            raise ValueError('Telemetry cannot exist without an included mission')
        value = {key: None for key in template}
        value.update({key: identity[key] for key in GROUP_FIELDS})
        value.update({key: 0 for key in COUNT_FIELDS})
        value['cases'] = 0
        value['exploratory_scale_test'] = identity['num_agents'] >= 16
    for field in ('success_rate', 'successful_missions', 'unsuccessful_episode_energy_mean_j'):
        value.pop(field, None)
    value.update(cohort='completed_missions', attempted_cases=attempted,
                 included_cases=len(rows), excluded_cases=attempted-len(rows),
                 analysis_status='available' if rows else 'no_completed_missions')
    return value


def completed_comparisons(groups):
    """Pair the intersection of completed case identities, separately per contrast."""
    result = []
    for (weight, route, n), rows in sorted(groups.items()):
        references = ([('W1', route)] if weight != 'W1' else [])
        if route == 'allocated':
            references.extend((weight, other) for other in ('edge', 'cloud'))
        for reference_weight, reference_route in references:
            reference = groups.get((reference_weight, reference_route, n), [])
            actual = {row['case_id']: row for row in rows if row['success']}
            baseline = {row['case_id']: row for row in reference if row['success']}
            if len(actual) != sum(bool(r['success']) for r in rows) or len(baseline) != sum(bool(r['success']) for r in reference):
                raise ValueError('Duplicate completed case identity')
            ids = sorted(actual.keys() & baseline.keys())
            for metric in METRICS:
                result.append(dict(weight_id=weight, route=route, num_agents=n,
                    reference_weight=reference_weight, reference_route=reference_route,
                    cohort='completed_in_both_conditions', paired_case_ids=ids,
                    **paired_changes([actual[i] for i in ids], [baseline[i] for i in ids], metric)))
    return result


def exposures(source, field):
    rows = []
    for (weight, route, n, value), tasks in sorted(source.items()):
        rows.append(dict(weight_id=weight, route=route, num_agents=n,
            cohort='completed_missions', **{field: value}, tasks=len(tasks),
            uav_tasks=sum(t.get('executed_x') == 1 for t in tasks),
            cloud_tasks=sum(t.get('executed_x') == 0 for t in tasks),
            on_time=sum(t['deadline_status'] == 'on_time' for t in tasks),
            late=sum(t['deadline_status'] == 'late' for t in tasks),
            undelivered=sum(t['deadline_status'] == 'undelivered' for t in tasks),
            observed_latency_p95_seconds=report.quantile((t.get('observed_latency_seconds') for t in tasks), .95),
            fixed_previous_tasks_mean=report.mean(t['fixed_previous_tasks'] for t in tasks),
            queue_mean_seconds=report.mean(t['command_path_queue_seconds'] for t in tasks),
            interruption_mean_seconds=report.mean(t['command_path_interruption_seconds'] for t in tasks),
            active_peer_aoi_seconds=report.ratio(sum(t['active_peer_aoi_sum'] for t in tasks),
                                               sum(t['active_peer_aoi_samples'] for t in tasks)),
            active_peer_aoi_samples=sum(t['active_peer_aoi_samples'] for t in tasks)))
    return rows


def number(value, multiplier=1., decimals=2):
    return '—' if value is None else f'{value*multiplier:.{decimals}f}'


def numerical_report(destination, manifest, summary, contrasts):
    table = {(r['weight_id'], r['route'], r['num_agents']): r for r in summary}
    weights = [w['id'] for w in manifest['weights']]
    lines = ['# Latency–energy weight sensitivity', '',
        'Parameter-only pilot results for completed missions. The production allocator, '
        'radio, workload, CPU capacities and frozen PANAMA policy are unchanged. '
        'The 16/32-robot cohorts remain exploratory scale tests.', '',
        'Each cell reports its included episode count. Latency and deadline rates pool '
        'tasks from those episodes; energy and waiting are episode means. CPU utilization '
        'uses the complete observed duration, including idle time. Missing cells have no '
        'completed mission and are undefined. Paired differences use only case identities '
        'completed in both conditions; their 95% bootstrap intervals are conditional on that cohort.', '',
        '## Allocated policy', '',
        '| Robots | Weight | Included n | UAV (%) | Cloud (%) | E2E p95 (ms) | Predicted mean (ms) | Energy (J/episode) | Deadline misses (%) |',
        '|---:|:---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in report.COUNTS:
        for w in weights:
            r = table[w, 'allocated', n]
            lines.append(f"| {n} | {w} | {r['included_cases']} | {number(r['uav_percent'])} | {number(r['cloud_percent'])} | "
                f"{number(r['observed_latency_p95_seconds'],1000)} | {number(r['predicted_latency_mean_seconds'],1000)} | "
                f"{number(r['mean_episode_energy_j'],decimals=3)} | {number(r['deadline_miss_percent'])} |")
    lines += ['', '## Weight effects', '']
    for n in report.COUNTS:
        base = table['W1', 'allocated', n]
        changes = []
        for w in weights[1:]:
            row = table[w, 'allocated', n]
            if row['included_cases'] and base['included_cases']:
                changes.append(f"{w}: energy {100*(row['mean_episode_energy_j']/base['mean_episode_energy_j']-1):+.1f}%, "
                    f"pooled p95 {(row['observed_latency_p95_seconds']-base['observed_latency_p95_seconds'])*1000:+.2f} ms")
        lines.append(f"- **{n} robots:** " + '; '.join(changes) + '.')
    lines += ['', 'These point estimates use each cell’s completed cohort. Differences in '
        'case composition can change the ranking; use the paired comparisons for matched effects.', '',
        '## Paired W2 minus W1', '',
        '| Robots | Common completed n | Energy difference (J), 95% CI | Mean per-case p95 difference (ms), 95% CI |',
        '|---:|---:|---:|---:|']
    effects = {(r['weight_id'], r['route'], r['num_agents'], r['reference_weight'], r['reference_route'], r['metric']): r for r in contrasts}
    for n in report.COUNTS:
        energy = effects['W2', 'allocated', n, 'W1', 'allocated', 'modeled_energy_j']
        latency = effects['W2', 'allocated', n, 'W1', 'allocated', 'case_command_p95_seconds']
        def interval(row, factor=1.):
            return '—' if not row['paired_cases'] else f"{number(row['difference'],factor,3)} [{number(row['ci95_low'],factor,3)}, {number(row['ci95_high'],factor,3)}]"
        lines.append(f"| {n} | {energy['paired_cases']} | {interval(energy)} | {interval(latency,1000)} |")
    lines += ['', 'A one-pair interval is a degenerate resampling result, not evidence of '
        'precise population effects. The pooled task p95 and mean paired per-case p95 are different estimands.', '',
        '## Interpretation', '']
    for n in report.COUNTS:
        valid = [table[w, 'allocated', n] for w in weights if table[w, 'allocated', n]['included_cases']]
        clouds = ', '.join(r['weight_id'] for r in valid if r['cloud_percent'] > 50) or 'none'
        energy = min(valid, key=lambda r: r['mean_episode_energy_j'])['weight_id']
        latency = min(valid, key=lambda r: r['observed_latency_p95_seconds'])['weight_id']
        misses = min(r['deadline_miss_percent'] for r in valid)
        best = '/'.join(r['weight_id'] for r in valid if abs(r['deadline_miss_percent']-misses) < 1e-10)
        lines.append(f'- **{n} robots:** cloud-majority settings {clouds}; lowest mean energy {energy}; '
                     f'lowest pooled p95 {latency}; lowest deadline-miss rate {best}.')
    lines += ['', 'Increasing the latency coefficient does not guarantee a lower realized tail '
        'latency. The objective optimizes priority-weighted snapshot latency, while measured '
        'E2E latency also includes subsequent transport, contention and fixed earlier assignments. '
        'First-round placement, active-count and handover tables retain those exposures. '
        'The conditional results do not estimate the effect on mission success.', '',
        '## Reproduction and provenance', '',
        f"- Frozen model SHA256: `{manifest['model_sha256']}`; policy updates during this study: 0.",
        f"- Timeline SHA256: `{manifest['timeline_sha256']}`; map SHA256: `{manifest['map_sha256']}`.",
        '- Configurations differ only in latency/energy coefficients. Hard solver deadlines remain disabled.',
        '- `cases.json` preserves starts/goals, seeds and orbital offsets. `publication.json` identifies the selected episode packets.',
        '- The full 375-episode records, prior all-mission summaries and mission-success figures remain local in the source run.',
        '- Rebuild without simulation: `python scripts/publish_omega_completed.py <source-run> <new-destination>`.',
        '- Verify: `python verify_bundle.py`. `charts.zip` contains the same eight published figures in PNG/PDF/SVG.', '']
    (destination/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    comparison = ['# Completed-mission comparison', '',
        '| Robots | Weight | Route | Included n | UAV / cloud tasks | E2E p95 (ms) | Energy (J/episode) | Misses (%) | Makespan (s) | Waiting steps |',
        '|---:|:---:|:---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        comparison.append(f"| {r['num_agents']} | {r['weight_id']} | {report.LABELS[r['route']]} | {r['included_cases']} | "
            f"{r['uav_tasks']} / {r['cloud_tasks']} | {number(r['observed_latency_p95_seconds'],1000)} | "
            f"{number(r['mean_episode_energy_j'],decimals=3)} | {number(r['deadline_miss_percent'])} | "
            f"{number(r['successful_makespan_mean_seconds'])} | {number(r['waiting_steps_mean'])} |")
    (destination/'analysis/comparison_tables.md').write_text('\n'.join(comparison)+'\n', encoding='utf-8')


def seal_bundle(destination):
    """Archive only inventoried publication files; local-only figures cannot leak in."""
    destination = Path(destination)
    files = sorted(p for p in destination.rglob('*') if p.is_file()
                   and p.name not in ('files.csv', 'charts.zip', 'results.zip')
                   and '06_mission_success' not in p.relative_to(destination).parts)
    csv_table(destination/'files.csv', [dict(path=p.relative_to(destination).as_posix(),
        sha256=sha256(p), bytes=p.stat().st_size) for p in files])
    for name, selected in (('charts.zip', [p for p in files if 'figures' in p.relative_to(destination).parts]),
                           ('results.zip', [p for p in files if 'figures' not in p.relative_to(destination).parts]+[destination/'files.csv'])):
        with zipfile.ZipFile(destination/name, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in selected:
                archive.write(path, path.relative_to(destination))


def retain_source_evidence(source, destination):
    """Preserve original source audits and frozen code independently of outcome selection."""
    folder=destination/'analysis'
    folder.mkdir(parents=True,exist_ok=True)
    for name in ('baseline_validation_v1.json','status_v1_gate_failure.json','validation_revision.json',
                 'independent_audit.json','readback_audit.json','independent_audit.stdout.log','status.json'):
        if (source/name).exists():shutil.copy2(source/name,destination/name)
    for name in ('frozen_harness','postprocessing'):
        shutil.copytree(source/'analysis'/name,folder/name)
    for name in ('parameter_validation.json','postprocessing_provenance.json'):
        shutil.copy2(source/'analysis'/name,folder/name)


def supporting_analysis(source, destination, summary, contrasts, episodes, selected_ids):
    """Retain source audit evidence and recompute derived tables on the selected cohort."""
    folder=destination/'analysis'
    selected={tuple(i) for i in selected_ids}
    completed_w1={case for weight,route,case in selected if weight=='W1' and route=='allocated'}
    for name in ('matched_snapshots.csv','matched_snapshot_objectives.csv','objective_monotonicity.csv','all_cloud_invariance.csv'):
        with (source/'analysis'/name).open(encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        if name=='all_cloud_invariance.csv':
            rows=[r for r in rows if ('W1','cloud',r['case_id']) in selected and (r['weight_id'],'cloud',r['case_id']) in selected]
            cohort='completed_in_both_cloud_conditions'
        else:
            rows=[r for r in rows if r['case_id'] in completed_w1]
            cohort='admissions_from_completed_W1_allocated_missions'
        csv_table(folder/name,[dict(r,cohort=cohort) for r in rows])
    table={(r['weight_id'],r['route'],r['num_agents']):r for r in summary}
    allocated=[r for r in summary if r['route']=='allocated']
    pareto=[]
    for row in allocated:
        others=[r for r in allocated if r['num_agents']==row['num_agents'] and r['included_cases']]
        dominated=[r['weight_id'] for r in others if r['observed_latency_p95_seconds']<=row['observed_latency_p95_seconds']
            and r['mean_episode_energy_j']<=row['mean_episode_energy_j']
            and (r['observed_latency_p95_seconds']<row['observed_latency_p95_seconds'] or r['mean_episode_energy_j']<row['mean_episode_energy_j'])]
        pareto.append(dict(weight_id=row['weight_id'],num_agents=row['num_agents'],cohort='completed_missions',
            included_cases=row['included_cases'],p95_ms=row['observed_latency_p95_seconds']*1000,
            energy_episode_j=row['mean_episode_energy_j'],energy_task_j=row['energy_per_generated_task_j'],
            cloud_percent=row['cloud_percent'],miss_percent=row['deadline_miss_percent'],dominated_by=dominated,
            empirical_pareto=not dominated))
    csv_table(folder/'allocated_tradeoffs.csv',pareto)
    with (destination/'handover.csv').open(encoding='utf-8') as stream:
        exposures={(r['weight_id'],int(r['num_agents']),r['exposure']):r for r in csv.DictReader(stream) if r['route']=='allocated'}
    availability=[]
    for row in allocated:
        w,n=row['weight_id'],row['num_agents']
        missing=exposures.get((w,n,'unavailable_backhaul'),{})
        available=exposures.get((w,n,'available_backhaul'),{})
        count,forced,free=int(missing.get('tasks',0)),int(missing.get('uav_tasks',0)),int(available.get('tasks',0))
        availability.append(dict(weight_id=w,num_agents=n,cohort='completed_missions',all_tasks=row['tasks'],
            unavailable_admissions=count,unavailable_percent=100*count/row['tasks'],unavailable_uav_tasks=forced,
            uav_tasks_with_unavailable_backhaul_percent=100*forced/row['uav_tasks'] if row['uav_tasks'] else None,
            available_admissions=free,cloud_percent_when_available=100*int(available.get('cloud_tasks',0))/free if free else None,
            fixed_previous_tasks_mean=row['fixed_previous_tasks_mean']))
    csv_table(folder/'availability_attribution.csv',availability)
    priority=defaultdict(list)
    with gzip.open(destination/'tasks.csv.gz','rt',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            location={'1':'uav','0':'cloud'}.get(row['executed_x'])
            if location:priority[row['weight_id'],row['route'],int(row['num_agents']),location].append((float(row['weight']),int(row['cycle'])))
    csv_table(folder/'priority_by_location.csv',[dict(weight_id=w,route=route,num_agents=n,location=location,
        cohort='completed_missions',tasks=len(values),priority_resource_weight_mean=report.mean(v[0] for v in values),
        first_round_tasks=sum(v[1]==0 for v in values),first_round_priority_weight_mean=report.mean(v[0] for v in values if v[1]==0))
        for (w,route,n,location),values in sorted(priority.items())])
    effects={(r['weight_id'],r['route'],r['num_agents'],r['reference_weight'],r['reference_route'],r['metric']):r for r in contrasts}
    dominance=[]
    for row in allocated:
        w,n=row['weight_id'],row['num_agents']
        pairs={route:effects[w,'allocated',n,w,route,'case_command_p95_seconds']['paired_cases'] for route in ('edge','cloud')}
        available=all(pairs.values())
        differences=[(effects[w,'allocated',n,w,route,'case_command_p95_seconds']['difference'],
                      effects[w,'allocated',n,w,route,'modeled_energy_j']['difference']) for route in ('edge','cloud')]
        dominance.append(dict(weight_id=w,num_agents=n,cohort='completed_in_both_conditions',
            estimator='mean_paired_case_p95_and_episode_energy',edge_paired_cases=pairs['edge'],cloud_paired_cases=pairs['cloud'],
            strict_latency_and_energy_dominance=all(lat < -1e-9 and energy < -1e-9 for lat,energy in differences) if available else None,
            no_worse_latency_and_energy_with_one_strict_improvement=all(lat<=1e-9 and energy<=1e-9 and (lat < -1e-9 or energy < -1e-9) for lat,energy in differences) if available else None,
            tolerance=1e-9))
    csv_table(folder/'control_dominance.csv',dominance)
    csv_table(folder/'w2_paired_effects.csv',[r for r in contrasts if r['weight_id']=='W2' and r['route']==r['reference_route']=='allocated'
        and r['reference_weight']=='W1' and r['metric'] in ('modeled_energy_j','case_command_p95_seconds')])
    report.write(folder/'findings.json',dict(cohort='completed_missions',included_episodes=len(episodes),
        included_tasks=sum(r['tasks'] for r in summary),allocated_summary=allocated,
        common_completed_pair_counts=[{k:r[k] for k in ('weight_id','num_agents','paired_cases')} for r in contrasts
            if r['route']==r['reference_route']=='allocated' and r['reference_weight']=='W1' and r['metric']=='modeled_energy_j']))
    report.write(folder/'report_review.json',dict(status='generated',cohort='completed_missions',
        published_episodes=len(episodes),task_records=sum(r['tasks'] for r in summary),simulation_changes=0,solver_changes=0))
    retain_source_evidence(source,destination)


def publish(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Use a new destination; source results are immutable')
    manifest = report.read(source/'manifest.json')
    if report.read(source/'status.json')['status'] != 'complete' or report.read(source/'independent_audit.json')['status'] != 'passed':
        raise ValueError('A complete independently audited source study is required')
    paths = sorted((source/'episodes').glob('*/*/*.result.json'))
    if len(paths) != manifest['expected_episodes']:
        raise ValueError('Incomplete source matrix')
    destination.mkdir(parents=True)
    (destination/'analysis').mkdir()
    cases = {c['case_id']: c for c in report.read(source/'cases.json')}
    expected = {(w['id'], r['route'], c) for w in manifest['weights'] for r in manifest['conditions'] for c in cases}
    groups, taskgroups, cpugroups = defaultdict(list), defaultdict(list), defaultdict(list)
    attempted, metadata = Counter(), {}
    episodes, rounds, first, audits, inventory = [], [], [], [], []
    active, handover = defaultdict(list), defaultdict(list)
    selected_ids, seen = [], set()
    streams = []
    for name in ('tasks', 'allocations'):
        with gzip.open(source/(name+'.csv.gz'), 'rt', encoding='utf-8') as stream:
            fields = csv.DictReader(stream).fieldnames
        streams.append(report.StreamCSV(destination/(name+'.csv.gz'), fields))
    try:
        for path in paths:
            packet = report.read(path)
            report.validate_packet(packet, cases[packet['row']['case_id']])
            if packet['manifest_sha256'] != sha256(source/'manifest.json'):
                raise ValueError('Source packet manifest mismatch')
            if sha256(path) != report.read(path.with_suffix('.meta.json'))['sha256']:
                raise ValueError('Source transaction changed')
            row, weight = dict(packet['row']), packet['weight']
            identity = dict(weight_id=weight['id'], latency_coefficient=weight['latency'], energy_coefficient=weight['energy'],
                case_id=row['case_id'], num_agents=row['num_agents'], route=row['route'], condition=row['condition'])
            unique = (weight['id'], row['route'], row['case_id'])
            if unique in seen:
                raise ValueError('Duplicate source episode')
            seen.add(unique)
            key = (weight['id'], row['route'], row['num_agents'])
            metadata[key] = identity
            attempted[key] += 1
            inventory.append(dict(identity=list(unique), packet_sha256=sha256(path), included=bool(row['success'])))
            if not row['success']:
                continue
            selected_ids.append(list(unique))
            row.update(identity)
            episodes.append(row)
            groups[key].append(row)
            taskgroups[key].extend(packet['tasks'])
            cpugroups[key].extend(packet['cpu'])
            cyclecounts = {r['cycle']: r['tasks'] for r in packet['rounds']}
            for task in packet['tasks']:
                streams[0].row(dict(task, **{k:v for k,v in identity.items() if k not in task}))
                active[key+(cyclecounts[task['cycle']],)].append(task)
                handover[key+('unavailable_backhaul' if task['backhaul_available_at_admission'] is False else 'available_backhaul',)].append(task)
                handover[key+('handover_lifetime' if task['handover_overlaps_lifetime'] else 'outside_handover',)].append(task)
            for allocation in packet['allocations']:
                streams[1].row(dict(allocation, **identity))
            rounds.extend(dict(r, weight_id=weight['id']) for r in packet['rounds'])
            first.append(dict(identity, tasks=sum(t['cycle']==0 for t in packet['tasks']),
                uav=sum(t['cycle']==0 and t.get('executed_x')==1 for t in packet['tasks']),
                cloud=sum(t['cycle']==0 and t.get('executed_x')==0 for t in packet['tasks'])))
            audits.append(dict(identity, status=packet['audit']['status'], packet_sha256=sha256(path),
                archive_sha256=packet['audit']['archive_sha256']))
    finally:
        for stream in streams:
            stream.close()
    if seen != expected:
        raise ValueError('Source identities differ from the complete paired matrix')
    for key in metadata:
        groups.setdefault(key, [])
    first_key = next(k for k,v in groups.items() if v)
    template = report.summarize(groups[first_key], taskgroups[first_key], cpugroups[first_key])
    summary = [completed_summary(groups[key], taskgroups[key], cpugroups[key], metadata[key], template, attempted[key]) for key in sorted(metadata)]
    contrasts = completed_comparisons(groups)
    for name, rows in (('summary',summary),('episodes',episodes),('rounds',rounds),('first_round',first),
                       ('comparisons',contrasts),('audit',audits),('active_count',exposures(active,'actual_active_robots')),
                       ('handover',exposures(handover,'exposure'))):
        csv_table(destination/(name+'.csv'), rows)
    for name in ('manifest.json', 'cases.json', 'baseline_validation.json'):
        shutil.copy2(source/name, destination/name)
    shutil.copytree(source/'configs', destination/'configs')
    publication = dict(protocol='omega_completed_missions_v1', cohort='completed_missions',
        source_run=str(source), source_manifest_sha256=sha256(source/'manifest.json'),
        source_episodes=len(paths), included_episodes=len(episodes), excluded_episodes=len(paths)-len(episodes),
        included_tasks=sum(r['tasks_generated'] for r in episodes), included_identities=selected_ids,
        source_inventory_sha256=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        source_independent_audit_sha256=sha256(source/'independent_audit.json'),
        publication_source_sha256=sha256(__file__), figure_source_sha256=sha256(Path(report.__file__)),
        postprocessing_directory='analysis/completed_postprocessing',
        paired_cohort='intersection_of_completed_case_identities', mission_success_figures='local_only',
        simulation_changes=0, solver_changes=0)
    report.write(destination/'publication.json', publication)
    report.figures(destination, summary, include_mission_success=False,
        cohort_label='Completed missions only; n varies (see tables); n=0: no data')
    numerical_report(destination, manifest, summary, contrasts)
    supporting_analysis(source,destination,summary,contrasts,episodes,selected_ids)
    (destination/'README.md').write_text('# Latency–energy sensitivity\n\n'
        'Completed-mission parameter-only results. Included counts are recorded per cell; '
        'empty cells have undefined metrics. The 16/32-robot cohorts are exploratory.\n\n'
        '- [Numerical report](report.md) and [complete comparison table](analysis/comparison_tables.md)\n'
        '- [Summary](summary.csv), [included episodes](episodes.csv), [paired differences](comparisons.csv)\n'
        '- [Publication provenance](publication.json) and [validation](validation.json)\n'
        '- [Charts ZIP](charts.zip) and [tables/configurations/report ZIP](results.zip)\n\n'
        'Mission-success plots and full-cohort outcome tables remain local. The retained source audit, status, '
        'read-back and frozen-harness files document the complete 375-episode source study; '
        '`publication.json`, `publication_audit.json` and `validation.json` describe the selected publication. '
        'Full traces and model assets are not distributed. '
        'Run `python verify_bundle.py` to check the bundle.\n', encoding='utf-8')
    report.write(destination/'validation.json', dict(status='passed', cohort='completed_missions',
        included_episodes=len(episodes), source_episodes=len(paths), task_records=publication['included_tasks'],
        source_packet_transactions_checked=len(paths), causal_replay_passes=len(audits),
        paired_case_identities=True, exclusive_deadline_partition=True, binary_placements=True,
        unsuccessful_episodes_in_published_tables=0, mission_success_figures_exported=0,
        empty_cells=[{k:r[k] for k in ('weight_id','route','num_agents')} for r in summary if not r['included_cases']]))
    shutil.copy2(Path(__file__).with_name('verify_omega_bundle.py'), destination/'verify_bundle.py')
    for name in ('publish_omega_completed.py', 'report_omega_sensitivity.py'):
        target=destination/'analysis/completed_postprocessing'/name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(Path(__file__).with_name(name), target)
    seal_bundle(destination)
    return publication


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args=parser.parse_args()
    result=publish(args.source,args.destination)
    print(json.dumps({k:v for k,v in result.items() if k != 'included_identities'},indent=2))
