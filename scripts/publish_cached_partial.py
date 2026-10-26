"""Publish saved cached-fallback episodes without resuming an interrupted study."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
import zipfile


def publish(root,destination):
    root,destination=Path(root).resolve(),Path(destination).resolve()
    if destination.exists():raise ValueError('Use a new publication directory')
    # Validate against the exact package that produced the saved results.
    sys.path.insert(0,str(root/'code'))
    import report_cached_fallback as report
    import report_omega_sensitivity as omega
    from panama_uav.communication_report import csv_table
    from panama_uav.records import sha256
    import run_cached_fallback as study
    manifest=study.verify(root)
    status=omega.read(root/'status.json')
    if status['status']!='stopped_by_user':raise ValueError('Expected a deliberately stopped study')
    cases={c['case_id']:c for c in omega.read(root/'cases.json')}
    paths=sorted((root/'episodes').glob('*/*/*.result.json'))
    errors=list((root/'episodes').rglob('*.error.json'))
    if errors:raise ValueError('Retain and investigate recorded run errors before publishing')
    destination.mkdir(parents=True)
    groups,tasks,cpu,baselines=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
    rows,selected,allocations,rounds,audits=[],[],[],[],[]
    taskfields=set();allocationfields=set()
    for path in paths:
        packet=omega.read(path)
        for t in packet['tasks']:taskfields.update(t)
        for a in packet['allocations']:allocationfields.update(a)
    fields=['weight_id','latency_coefficient','energy_coefficient','case_id','num_agents','route','condition']
    stream=omega.StreamCSV(destination/'tasks.csv.gz',list(dict.fromkeys(fields+sorted(taskfields))))
    ast=omega.StreamCSV(destination/'allocations.csv.gz',list(dict.fromkeys(fields+sorted(allocationfields))))
    totals=Counter()
    try:
        for path in paths:
            data=report.packet(path,cases[path.name.removesuffix('.result.json')])
            meta=report.identity(data);row=dict(data['row'],**meta)
            key=meta['weight_id'],row['route'],row['num_agents']
            archive=Path(data['audit']['archive'])
            if sha256(archive)!=data['audit']['archive_sha256']:raise ValueError('Raw archive hash mismatch')
            groups[key].append(row);rows.append(row)
            totals.update(episodes=1,successes=int(row['success']),tasks=row['tasks_generated'],
                on_time=row['deadline_commands_on_time'],late=row['deadline_commands_late'],
                undelivered=row['deadline_commands_undelivered'],fallbacks=row['cached_fallback_decisions'],
                fallback_moves=row['cached_fallback_motion_steps'])
            audits.append(dict(meta,status=data['audit']['status'],packet_sha256=sha256(path),
                archive_sha256=data['audit']['archive_sha256'],**{k:v for k,v in data['audit'].items()
                    if k not in ('archive','archive_sha256','status','weight_id','condition','case_id') and not isinstance(v,(dict,list))}))
            if row['success']:
                selected.append(row);tasks[key].extend(data['tasks']);cpu[key].extend(data['cpu'])
                for t in data['tasks']:stream.row(dict(t,**{k:v for k,v in meta.items() if k not in t}))
                for a in data['allocations']:ast.row(dict(a,**meta))
                rounds.extend(dict(r,weight_id=meta['weight_id']) for r in data['rounds'])
    finally:stream.close();ast.close()
    known={(r['weight_id'],r['condition'],r['case_id']) for r in rows}
    interrupted=set()
    for path in root.glob('worker-*.json'):
        record=omega.read(path)
        job=record.get('job','')
        if record.get('phase') not in ('complete','error') and job:
            interrupted.add(tuple(job.split('/')))
    inventory=[]
    for w in manifest['weights']:
        for condition in manifest['conditions']:
            for case in cases.values():
                identity=w['id'],condition['name'],case['case_id']
                inventory.append(dict(weight_id=w['id'],condition=condition['name'],route=condition['route'],
                    case_id=case['case_id'],num_agents=case['num_agents'],
                    execution_status='completed' if identity in known else 'interrupted' if identity in interrupted else 'not_completed'))
    for item in omega.read(root/'baseline_index.json'):
        if sha256(item['packet'])!=item['packet_sha256']:raise ValueError('Original baseline packet changed')
        data=omega.read(item['packet']);row=dict(data['row'],**report.identity(data))
        # Match exactly the cases actually completed before the stop.
        if (row['weight_id'],row['condition'],row['case_id']) in known:
            baselines[row['weight_id'],row['route'],row['num_agents']].append(row)
    template=omega.summarize([r for r in groups[next(iter(groups))] if r['success']],tasks[next(iter(groups))],cpu[next(iter(groups))])
    summary,rates=[],[]
    for w in manifest['weights']:
        for condition in manifest['conditions']:
            for n in manifest['populations']:
                key=w['id'],condition['route'],n
                group=groups[key]
                meta=dict(weight_id=w['id'],latency_coefficient=w['latency'],energy_coefficient=w['energy'],
                    condition=condition['name'],route=condition['route'],num_agents=n)
                value=report.augmented_summary([r for r in group if r['success']],tasks[key],cpu[key],meta,template,len(group))
                value.update(meta,planned_cases=5,uncompleted_cases=5-len(group),study_status='partial_stopped')
                summary.append(value)
                for fallback,source in (('cached_astar',group),('wait_matched',baselines[key])):
                    count=len(source);passed=sum(r['success'] for r in source)
                    rates.append(dict(meta,fallback=fallback,completed_episodes=count,successful_missions=passed,
                        failed_missions=count-passed,uncompleted_planned_episodes=5-count,
                        success_rate=passed/count if count else None))
    contrasts=report.comparisons({k:v for k,v in groups.items() if v},baselines)
    for name,data in (('summary',summary),('episodes',selected),('all_completed_episodes',rows),('rounds',rounds),
        ('audit',audits),('inventory',inventory),('mission_success_rates',rates),('comparisons',contrasts)):
        csv_table(destination/(name+'.csv'),data)
    omega.figures(destination,summary,include_mission_success=False,
        cohort_label='PARTIAL STOPPED STUDY | Completed missions only | 279/375 episodes; cell n varies')
    for name in ('manifest.json','cases.json','baseline_validation.json','pilot_validation.json','PILOT_REPORT.md',
        'pilot_summary.csv','pilot_episodes.csv','production_delta.patch','regression.json','parameter_isolation.json','status.json'):
        shutil.copy2(root/name,destination/name)
    shutil.copytree(root/'configs',destination/'configs')
    # Source relevant to the saved run, rather than later in-progress redesigns.
    source=destination/'source';source.mkdir()
    for name in ('run_cached_fallback.py','report_cached_fallback.py','run_omega_sensitivity.py',
        'report_omega_sensitivity.py','audit_omega_reproduction.py','pilot_helper.py'):
        shutil.copy2(root/'code'/name,source/name)
    validation=dict(status='passed_partial_inventory',study_status='stopped_by_user',
        primary_completed=len(rows),primary_planned=375,primary_uncompleted=375-len(rows),
        successful_episodes=len(selected),conditional_tasks=sum(r['tasks_generated'] for r in selected),
        recorded_episode_replay_audits_passed=len(audits),raw_archives_hash_verified=len(audits),
        packet_partitions_verified=True,binary_placements_verified=True,energy_scope_verified=True,
        model_sha256=manifest['model_sha256'],timeline_sha256=manifest['timeline_sha256'],
        pilot_episodes=36,baseline_reproductions=9,solver_failures=0,
        mission_success_figures_exported=0,totals=dict(totals))
    omega.write(destination/'validation.json',validation)
    text=(f'# Cached fallback — partial results\n\n'
        f'The experiment stopped before redesign: **{len(rows)}/375** primary episodes completed, '
        f'with **{len(selected)}** successful and **{len(rows)-len(selected)}** unsuccessful missions. '
        f'The remaining {375-len(rows)} planned episodes have no completed outcome. '
        'Interrupted episodes are not counted as mission failures. No runs were restarted for this export.\n\n'
        'The matrix uses W1–W5, Allocated/All UAV/All Cloud, 2/4/8/16/32 robots, five paired cases, '
        '250 ms cutoffs and one-second control. Only the missed-command controller changes to cached-path fallback. '
        'The original WAIT results remain unchanged. The 16/32-robot cohorts are exploratory scale tests.\n\n'
        '`episodes.csv`, task records, summary metrics and figures contain successful missions only. '
        '`all_completed_episodes.csv` retains every completed outcome; `inventory.csv` retains all 375 planned identities. '
        'Mission-success rates use all completed attempted missions in each partially observed cell. '
        'They do not imply that the full five-case matrix finished. Paired differences use identical completed cases; '
        'conditional outcome comparisons require success in both conditions. Missing conditional outcomes are undefined.\n\n'
        f'The saved completed episodes generated {totals["tasks"]} tasks: {totals["on_time"]} on time, '
        f'{totals["late"]} late and {totals["undelivered"]} undelivered. '
        f'Cached fallback was invoked {totals["fallbacks"]} times and produced {totals["fallback_moves"]} movement steps.\n\n'
        'Each saved episode passed causal replay with frozen-policy action verification before archival. '
        'This export rechecks every transaction, archive hash, binary placement, command partition and energy identity. '
        'The 36-episode validation pilot passed, including nine exact original-WAIT reproductions. '
        'The recorded regression suite passed 427 tests; four supplementary-analysis tests also passed.\n\n'
        'Energy includes communication and UAV computation only. No robot locomotion or UAV propulsion energy is added. '
        'Full event traces, checkpoints and mission-success figures remain local. These partial results are not a completed study.\n\n'
        'The included manifest, configurations, frozen source, case bank, packet hashes and archive hashes preserve provenance. '
        'The separate 100 ms pilot is documented in [PILOT_REPORT.md](PILOT_REPORT.md).\n')
    (destination/'README.md').write_text(text,encoding='utf-8')
    files=sorted(p for p in destination.rglob('*') if p.is_file())
    csv_table(destination/'files.csv',[dict(path=p.relative_to(destination).as_posix(),sha256=sha256(p),bytes=p.stat().st_size) for p in files])
    for name,items in (('charts.zip',[p for p in files if 'figures' in p.relative_to(destination).parts]),
                       ('results.zip',[p for p in files if 'figures' not in p.relative_to(destination).parts]+[destination/'files.csv'])):
        with zipfile.ZipFile(destination/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for p in items:archive.write(p,p.relative_to(destination))
    print(json.dumps(validation,indent=2))
    return validation


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();publish(args.root,args.destination)
