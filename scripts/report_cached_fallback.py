"""Measured pilot and primary results for cached-path missed-command fallback."""
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import report_omega_sensitivity as omega
from panama_uav.communication_report import csv_table,paired_changes
from panama_uav.records import sha256


def packet(path,case):
    data=omega.read(path)
    if sha256(path)!=omega.read(path.with_suffix('.meta.json'))['sha256']:
        raise ValueError('A committed result transaction changed')
    # Legacy validation equated misses with WAITs; cached movement is accounted separately.
    row=data['row']
    checked=dict(data,row=dict(row,command_cutoff_misses=row.get('command_cutoff_waits',row['command_cutoff_misses'])))
    omega.validate_packet(checked,case)
    if 'cached_fallback_decisions' in row:
        if row['cached_fallback_decisions']!=row['command_cutoff_misses']:
            raise ValueError('Missing cached fallback decision')
        if row['cached_fallback_move_intents']+row['cached_fallback_wait_intents']!=row['cached_fallback_decisions']:
            raise ValueError('Cached decision partition mismatch')
        if row['cached_fallback_motion_steps']+row['cached_fallback_resolver_blocks']!=row['cached_fallback_move_intents']:
            raise ValueError('Cached physical-movement partition mismatch')
    if abs(row['modeled_energy_lower_j']-row['radio_energy_lower_j']-row['drone_compute_energy_j'])>1e-7:
        raise ValueError('Energy scope changed')
    return data


def identity(data):
    row,w=data['row'],data['weight']
    return dict(weight_id=w['id'],latency_coefficient=w['latency'],energy_coefficient=w['energy'],
        case_id=row['case_id'],num_agents=row['num_agents'],route=row['route'],condition=row['condition'])


def pilot_report(root):
    root=Path(root);cases={c['case_id']:c for c in omega.read(root/'cases.json')}
    results=[];groups=defaultdict(list)
    variants=(('primary','cached_astar',.25),('wait250','wait',.25),('wait100','wait',.1),('cached100','cached_astar',.1))
    for variant,mode,cutoff in variants:
        base=root if variant=='primary' else root/'validation'/variant
        paths=sorted((base/'episodes/W1').glob('*/*.result.json'))
        paths=[p for p in paths if cases[p.name.removesuffix('.result.json')]['num_agents'] in (2,4,8)
               and p.name.endswith('-000.result.json')]
        if len(paths)!=9:raise ValueError('Each paired pilot condition requires all nine cases')
        for path in paths:
            data=packet(path,cases[path.name.removesuffix('.result.json')]);row=dict(data['row'],**identity(data))
            row.update(fallback=mode,cutoff_seconds=cutoff,variant=variant)
            results.append(row);groups[cutoff,mode,row['route'],row['num_agents']].append(row)
    csv_table(root/'pilot_episodes.csv',results)
    summary=[]
    for (cutoff,mode,route,n),rows in sorted(groups.items()):
        generated=sum(r['tasks_generated'] for r in rows)
        summary.append(dict(cutoff_seconds=cutoff,fallback=mode,route=route,num_agents=n,cases=len(rows),
            successful_missions=sum(r['success'] for r in rows),commands=generated,
            on_time=sum(r['deadline_commands_on_time'] for r in rows),late=sum(r['deadline_commands_late'] for r in rows),
            undelivered=sum(r['deadline_commands_undelivered'] for r in rows),
            cutoff_misses=sum(r['command_cutoff_misses'] for r in rows),
            cutoff_waits=sum(r['deadline_cutoff_wait_tasks'] for r in rows),
            fallback_decisions=sum(r.get('cached_fallback_decisions',0) for r in rows),
            fallback_motion_steps=sum(r.get('cached_fallback_motion_steps',0) for r in rows),
            waiting_steps=omega.mean(r['waiting_steps'] for r in rows),
            makespan_completed_seconds=omega.mean(r['makespan_seconds'] for r in rows),
            active_peer_aoi_seconds=omega.ratio(sum(r['active_peer_aoi_sum'] for r in rows),sum(r['active_peer_aoi_samples'] for r in rows))))
    csv_table(root/'pilot_summary.csv',summary)
    lines=['# Cached fallback paired validation','',
        'One existing case at each of 2/4/8 robots, three routes, W1, and a 250-step cap. '
        'Both WAIT and cached fallback run at 250 ms and 100 ms; the control interval stays one second. '
        'These diagnostic cases validate execution and exposure, rather than estimate population effects.','',
        '| Cutoff (ms) | Fallback | Robots | Route | Commands | On time | Late / undelivered | Cutoff WAITs | Fallback moves | Completed |',
        '|---:|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['cutoff_seconds']*1000:g} | {r['fallback']} | {r['num_agents']} | {omega.LABELS[r['route']]} | "
            f"{r['commands']} | {r['on_time']} | {r['late']} / {r['undelivered']} | {r['cutoff_waits']} | {r['fallback_motion_steps']} | {r['successful_missions']}/{r['cases']} |")
    (root/'PILOT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    baseline=omega.read(root/'baseline_validation.json')
    if baseline['status']!='passed' or baseline['cases']!=9:raise ValueError('Legacy reproduction gate not passed')
    result=dict(status='passed',pilot_episodes=len(results),primary_reusable_episodes=9,
        baseline_reproductions=9,causal_audits=len(results),primary_cutoff_seconds=.25,pilot_cutoffs_seconds=[.1,.25],
        fallback_decisions=sum(r.get('cached_fallback_decisions',0) for r in results),
        fallback_motion_steps=sum(r.get('cached_fallback_motion_steps',0) for r in results),
        paired_inputs=True,all_attempts_retained=True,solver_failures=0)
    omega.write(root/'pilot_validation.json',result)
    return result


def empty_summary(meta,template):
    value={k:None for k in template};value.update(meta)
    for key in ('cases','tasks','uav_tasks','cloud_tasks','commands','on_time_commands','late_commands',
                'undelivered_commands','cutoff_waits','first_round_tasks','first_round_uav','first_round_cloud',
                'active_peer_aoi_samples','observed_latency_samples'):value[key]=0
    return value


def augmented_summary(rows,tasks,cpu,meta,template,attempted):
    value=omega.summarize(rows,tasks,cpu) if rows else empty_summary(meta,template)
    for key in ('successful_missions','success_rate','unsuccessful_episode_energy_mean_j'):
        value.pop(key,None)
    value.update(cohort='completed_missions',included_cases=len(rows),attempted_cases=attempted,
        excluded_cases=attempted-len(rows),cutoff_waits=sum(r['deadline_cutoff_wait_tasks'] for r in rows))
    def total(field):return sum(r.get(field,0) or 0 for r in rows)
    pairs=total('fov_relevant_pairs');epochs=sum(r['steps'] for r in rows)
    value.update(fov_relevant_pairs=pairs,fov_errors=total('fov_errors'),
        xi_fov=omega.ratio(total('fov_errors'),pairs),
        xi_aoi_epoch_sum=sum((r['mean_xi_aoi'] or 0)*r['steps'] for r in rows),
        xi_aoi_epochs=epochs,mean_decision_xi_aoi=omega.ratio(sum((r['mean_xi_aoi'] or 0)*r['steps'] for r in rows),epochs),
        vertex_relevant_pairs=total('vertex_relevant_pairs'),vertex_e2e_errors=total('vertex_e2e_errors'),
        xi_vertex_e2e=omega.ratio(total('vertex_e2e_errors'),total('vertex_relevant_pairs')),
        xi_vertex_aoi=omega.ratio(total('vertex_aoi_error_sum'),total('vertex_relevant_pairs')),
        attempted_vertex_conflicts=total('attempted_vertex_conflicts'),attempted_edge_swaps=total('attempted_edge_swaps'),
        attempted_vertex_conflicts_mean=omega.mean(r['attempted_vertex_conflicts'] for r in rows),
        attempted_edge_swaps_mean=omega.mean(r['attempted_edge_swaps'] for r in rows),
        action_order_known_pairs=total('action_order_known_pairs'),action_dt_ordering_errors=total('action_dt_ordering_errors'),
        action_order_error_rate=omega.ratio(total('action_dt_ordering_errors'),total('action_order_known_pairs')),
        decision_peer_aoi_epoch_sum=sum((r['mean_peer_source_age_seconds'] or 0)*r['steps'] for r in rows),
        mean_decision_cached_peer_aoi_seconds=omega.ratio(sum((r['mean_peer_source_age_seconds'] or 0)*r['steps'] for r in rows),epochs),
        missed_vertex_conflicts=total('missed_vertex_conflicts'),false_positive_vertex_conflicts=total('false_positive_vertex_conflicts'),
        active_decisions=total('active_decisions'),command_cutoff_misses=total('command_cutoff_misses'),
        cached_fallback_decisions=total('cached_fallback_decisions'),
        cached_fallback_motion_steps=total('cached_fallback_motion_steps'),
        cached_fallback_move_intents=total('cached_fallback_move_intents'),
        cached_fallback_wait_intents=total('cached_fallback_wait_intents'),
        cached_fallback_resolver_blocks=total('cached_fallback_resolver_blocks'),
        fallback_movement_percent=100*total('cached_fallback_motion_steps')/total('cached_fallback_decisions') if total('cached_fallback_decisions') else None,
        oracle_action_comparisons=total('decision_action_shadow_comparisons'),
        oracle_action_disagreements=total('decision_action_shadow_disagreements'),
        oracle_action_disagreement_percent=100*total('decision_action_shadow_disagreements')/total('decision_action_shadow_comparisons') if total('decision_action_shadow_comparisons') else None)
    return value


def comparisons(groups,baseline):
    result=[]
    metrics=('makespan_seconds','waiting_steps','case_command_p95_seconds','command_on_time_rate',
             'modeled_energy_j','mean_drone_cpu_utilization','mean_cloud_cpu_utilization','cutoff_waits')
    for (w,route,n),rows in sorted(groups.items()):
        reference=baseline[w,route,n]
        common={r['case_id'] for r in rows if r['success']} & {r['case_id'] for r in reference if r['success']}
        actual=[dict(r,command_cutoff_misses=r['deadline_cutoff_wait_tasks']) for r in rows if r['case_id'] in common]
        old=[dict(r,command_cutoff_misses=r['deadline_cutoff_wait_tasks']) for r in reference if r['case_id'] in common]
        for metric in metrics:
            result.append(dict(weight_id=w,route=route,num_agents=n,reference='original_WAIT_250ms',
                cohort='completed_in_both_conditions',paired_case_ids=sorted(common),**paired_changes(actual,old,metric)))
        result.append(dict(weight_id=w,route=route,num_agents=n,reference='original_WAIT_250ms',
            cohort='all_attempted_missions',paired_case_ids=sorted(r['case_id'] for r in rows),
            **paired_changes(rows,reference,'mission_success_rate')))
    return result


def additional_figures(root,summary,success_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table={(r['weight_id'],r['route'],r['num_agents']):r for r in summary}
    x=np.arange(5);width=.23
    def panels(folder,name,field,label,scale=1.,private=False):
        fig,axes=plt.subplots(1,5,figsize=(17,3.5),squeeze=False)
        values=[r[field]*scale for r in summary if r.get(field) is not None]
        upper=max(values+[1e-6])*1.15
        for i,(weight,_,_) in enumerate(omega.WEIGHTS):
            ax=axes[0,i]
            for j,route in enumerate(omega.ROUTES):
                data=[table[weight,route,n].get(field) for n in omega.COUNTS]
                ax.bar(x+(j-1)*width,[float('nan') if v is None else v*scale for v in data],width,
                    color=omega.COLORS[route],label=omega.LABELS[route])
                for k,value in enumerate(data):
                    if value is None:ax.text(x[k]+(j-1)*width,.02,'no data',rotation=90,ha='center',fontsize=6,transform=ax.get_xaxis_transform())
            ax.set_title(weight);ax.set_xticks(x,['2','4','8','16*','32*']);ax.set_xlabel('Robots');ax.set_ylim(0,upper)
            if i==0:ax.set_ylabel(label)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
        fig.text(.5,-.01,'Cached fallback | Completed missions only; included n in tables | * Exploratory scale tests',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,0,1,.93));omega.save_figure(fig,root/'figures'/folder,name)
    panels('08_aoi','active_inference_peer_aoi','active_peer_aoi_seconds','DT inference active-peer AoI (s)')
    panels('08_aoi','decision_cached_peer_aoi','mean_decision_cached_peer_aoi_seconds','Decision cached-peer AoI (s)')
    panels('09_ordering','fov_ordering_error','xi_fov','FoV-relevant ordering errors (%)',100.)
    panels('09_ordering','aoi_weighted_inconsistency','mean_decision_xi_aoi','Mean decision AoI-weighted inconsistency')
    panels('09_ordering','vertex_only_exploratory','xi_vertex_e2e','Exploratory vertex-only inconsistency (%)',100.)
    panels('10_conflicts','vertex_conflicts','attempted_vertex_conflicts_mean','Attempted vertex conflicts / mission')
    panels('10_conflicts','edge_conflicts','attempted_edge_swaps_mean','Attempted edge swaps / mission')
    panels('11_mission','waiting','waiting_steps_mean','Waiting robot steps / mission')
    panels('11_mission','makespan','successful_makespan_mean_seconds','Completed-mission makespan (s)')
    panels('12_fallback','fallback_movement','fallback_movement_percent','Fallback decisions producing movement (%)')
    # Success uses every attempted mission; its figure remains local.
    fig,axes=plt.subplots(1,5,figsize=(17,3.5),squeeze=False)
    rates={(r['weight_id'],r['route'],r['num_agents']):r for r in success_rows if r['fallback']=='cached_astar'}
    for i,(weight,_,_) in enumerate(omega.WEIGHTS):
        ax=axes[0,i]
        for j,route in enumerate(omega.ROUTES):
            ax.bar(x+(j-1)*width,[rates[weight,route,n]['success_rate']*100 for n in omega.COUNTS],width,
                color=omega.COLORS[route],label=omega.LABELS[route])
        ax.set_title(weight);ax.set_xticks(x,['2','4','8','16*','32*']);ax.set_ylim(0,105)
        if i==0:ax.set_ylabel('Mission success (% of all five attempts)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.93));omega.save_figure(fig,root/'private_figures','mission_success')


def report_study(root):
    root=Path(root);manifest=omega.read(root/'manifest.json');cases={c['case_id']:c for c in omega.read(root/'cases.json')}
    paths=sorted((root/'episodes').glob('*/*/*.result.json'))
    if len(paths)!=375:raise ValueError('All 375 primary episodes are required')
    destination=root/'publication'
    if destination.exists():raise ValueError('Keep an existing publication; use a new report revision directory')
    destination.mkdir();(destination/'analysis').mkdir()
    groups,taskgroups,cpugroups,baseline=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
    allrows=[];selected=[];rounds=[];audits=[];seen=set();schema=set();allocation_schema=set()
    for path in paths:
        data=omega.read(path)
        for item in data['tasks']:schema.update(item)
        for item in data['allocations']:allocation_schema.update(item)
    idfields=['weight_id','latency_coefficient','energy_coefficient','case_id','num_agents','route','condition']
    taskstream=omega.StreamCSV(destination/'tasks.csv.gz',list(dict.fromkeys(idfields+sorted(schema))))
    allocationstream=omega.StreamCSV(destination/'allocations.csv.gz',list(dict.fromkeys(idfields+sorted(allocation_schema))))
    totals=Counter()
    try:
        for path in paths:
            case=cases[path.name.removesuffix('.result.json')];data=packet(path,case);meta=identity(data)
            row=dict(data['row'],**meta);key=(meta['weight_id'],row['route'],row['num_agents'])
            unique=key[:2]+(row['case_id'],)
            if unique in seen:raise ValueError('Duplicate primary episode')
            seen.add(unique);groups[key].append(row);allrows.append(row)
            totals.update(episodes=1,tasks=row['tasks_generated'],on_time=row['deadline_commands_on_time'],
                late=row['deadline_commands_late'],undelivered=row['deadline_commands_undelivered'],
                successes=int(row['success']),fallbacks=row['cached_fallback_decisions'],fallback_moves=row['cached_fallback_motion_steps'])
            audits.append(dict(meta,status=data['audit']['status'],packet_sha256=sha256(path),archive=data['audit']['archive'],archive_sha256=data['audit']['archive_sha256']))
            if not row['success']:continue
            selected.append(row);taskgroups[key].extend(data['tasks']);cpugroups[key].extend(data['cpu'])
            for task in data['tasks']:taskstream.row(dict(task,**{k:v for k,v in meta.items() if k not in task}))
            for allocation in data['allocations']:allocationstream.row(dict(allocation,**meta))
            rounds.extend(dict(r,weight_id=meta['weight_id']) for r in data['rounds'])
    finally:taskstream.close();allocationstream.close()
    expected={(w['id'],c['route'],case) for w in manifest['weights'] for c in manifest['conditions'] for case in cases}
    if seen!=expected:raise ValueError('Primary paired case matrix is incomplete')
    for item in omega.read(root/'baseline_index.json'):
        if sha256(item['packet'])!=item['packet_sha256']:raise ValueError('Historical baseline changed')
        data=omega.read(item['packet']);row=dict(data['row'],**identity(data))
        baseline[row['weight_id'],row['route'],row['num_agents']].append(row)
    first_key=next(k for k,rows in groups.items() if any(r['success'] for r in rows))
    template=omega.summarize([r for r in groups[first_key] if r['success']],taskgroups[first_key],cpugroups[first_key])
    summary=[];success_rows=[]
    for key,rows in sorted(groups.items()):
        meta={k:rows[0][k] for k in ('weight_id','latency_coefficient','energy_coefficient','condition','route','num_agents')}
        summary.append(augmented_summary([r for r in rows if r['success']],taskgroups[key],cpugroups[key],meta,template,len(rows)))
        for fallback,source in (('cached_astar',rows),('wait',baseline[key])):
            passed=sum(r['success'] for r in source);count=len(source)
            z=1.959963984540054;p=passed/count;factor=1+z*z/count
            center=(p+z*z/(2*count))/factor;half=z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/factor
            success_rows.append(dict(meta,fallback=fallback,cutoff_seconds=.25,attempted_missions=count,
                completed_missions=passed,success_rate=p,ci95_low=max(0.,center-half),ci95_high=min(1.,center+half)))
    contrasts=comparisons(groups,baseline)
    for name,rows in (('summary',summary),('episodes',selected),('rounds',rounds),('audit',audits),
                      ('mission_success_rates',success_rows),('comparisons',contrasts)):
        csv_table(destination/(name+'.csv'),rows)
    csv_table(root/'all_attempts.csv',allrows)
    omega.figures(destination,summary,include_mission_success=False,
        cohort_label='Cached fallback; completed missions only; n varies; n=0: no data')
    additional_figures(destination,summary,success_rows)
    # Private success figures remain outside the distributable publication.
    shutil.move(destination/'private_figures',root/'private_figures')
    for name in ('manifest.json','cases.json','baseline_validation.json','pilot_validation.json','PILOT_REPORT.md','pilot_summary.csv','pilot_episodes.csv','production_delta.patch'):
        shutil.copy2(root/name,destination/name)
    shutil.copytree(root/'configs',destination/'configs')
    lines=['# Cached-path fallback sensitivity','',
        'The primary matrix contains 375 new 250 ms runs: W1–W5, three routes, five populations, and five paired cases. '
        'The 1-second control interval, workload, CPU, radio, priorities and frozen MAPF policy are unchanged. '
        'Only active missed-command execution changes from WAIT to cached-path fallback. '
        'The original WAIT experiment is preserved and paired by case identity.','',
        '## Accounting','',
        f"Primary attempts: {totals['episodes']}; completed: {totals['successes']}. "
        f"Generated commands: {totals['tasks']}; on time: {totals['on_time']}; late: {totals['late']}; undelivered: {totals['undelivered']}. "
        f"Cached fallback decisions: {totals['fallbacks']}; actual fallback movement steps: {totals['fallback_moves']}.",'',
        'Outcome plots and summary metrics use completed missions. Mission-success rates use all five attempted cases in each cell; '
        'their figure remains local. Paired outcome differences use cases completed by both treatments. '
        'Empty outcome cohorts and zero exposure denominators have undefined metrics. '
        'The 16/32-robot cohorts remain exploratory scale tests.','',
        '## Mission success: cached fallback versus original WAIT','',
        '| Robots | Weight | Route | WAIT completed / attempted | Cached completed / attempted |',
        '|---:|:---:|---|---:|---:|']
    rates={(r['weight_id'],r['route'],r['num_agents'],r['fallback']):r for r in success_rows}
    for row in summary:
        key=row['weight_id'],row['route'],row['num_agents'];old=rates[key+('wait',)];new=rates[key+('cached_astar',)]
        lines.append(f"| {row['num_agents']} | {row['weight_id']} | {omega.LABELS[row['route']]} | {old['completed_missions']}/{old['attempted_missions']} | {new['completed_missions']}/{new['attempted_missions']} |")
    lines+=['','## Metrics and interpretation','',
        'A fallback move never counts as a delivered command. Command lateness remains sensing-to-robot reception; '
        'predicted allocation latency is recorded separately. Cutoff misses, requested WAITs, fallback move intents, '
        'executed fallback moves and resolver blocks have separate denominators. '
        'The fallback reads delivered cached A* intentions and priorities plus local pose, obstacles and occupancy. '
        'It neither replans nor calls a local DQN; a missing/exhausted path or a higher-priority conflict causes WAIT.','',
        'FoV ordering uses original vertex-plus-edge relevance. Vertex-only results remain a separate exploratory figure. '
        'DT inference active-peer AoI and decision cached-peer AoI are plotted separately. '
        'Counts and exposures are stored with normalized metrics. Physical conflict plots show attempted conflicts '
        'before the unchanged delivered-priority resolver. CPU utilization includes mission idle periods.','',
        'Energy retains the original communication plus UAV computation model. Robot movement energy is excluded; '
        'no extra local-fallback CPU energy model is introduced. The fixed remote task workload and CPU accounting are unchanged.','',
        'The 100 ms pilot is sensitivity evidence, not a change to the primary deadline. '
        'See [PILOT_REPORT.md](PILOT_REPORT.md). Each comparison uses matched WAIT and cached cases at the same cutoff.','',
        '## Reproduction','',
        '`python scripts/run_cached_fallback.py --root <new-root> --prepare --launch --stage pilot`','',
        '`python scripts/run_cached_fallback.py --root <root> --launch --stage full --resume`','',
        f"Frozen model SHA256: `{manifest['model_sha256']}`. Timeline SHA256: `{manifest['timeline_sha256']}`. "
        'Source hashes, unchanged paired seeds/starts/goals/orbital offsets, configuration files and production delta are included. '
        'Raw full-cohort traces and records remain local.','']
    (destination/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    publication=dict(protocol=manifest['protocol'],cohort='completed_missions',source_episodes=375,
        included_episodes=len(selected),included_tasks=sum(r['tasks'] for r in summary),
        mission_success_denominator='all_attempted_missions',model_sha256=manifest['model_sha256'],
        source_manifest_sha256=sha256(root/'manifest.json'),included_identities=[[r['weight_id'],r['route'],r['case_id']] for r in selected])
    omega.write(destination/'publication.json',publication)
    validation=dict(status='passed',primary_episodes=375,pilot_episodes=36,baseline_reproductions=9,
        totals=dict(totals),paired_identities=True,exclusive_deadline_partition=True,binary_placements=True,
        energy_excludes_robot_movement=True,solver_failures=len(list((root/'episodes').glob('*/*/*.error.json'))),
        included_episodes=len(selected),included_tasks=publication['included_tasks'])
    if validation['solver_failures']:raise ValueError('A solver failure remains recorded')
    omega.write(root/'validation.json',validation);omega.write(destination/'validation.json',validation)
    (destination/'README.md').write_text('# Cached fallback experiment\n\n'
        '[Report](report.md), [pilot](PILOT_REPORT.md), [outcomes](summary.csv), '
        '[mission success on all attempts](mission_success_rates.csv), [paired differences](comparisons.csv).\n\n'
        'Outcome tables and plots use completed missions. Mission-success figures and full raw records stay local.\n',encoding='utf-8')
    files=[p for p in destination.rglob('*') if p.is_file()]
    csv_table(destination/'files.csv',[dict(path=p.relative_to(destination).as_posix(),sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(files)])
    for name,items in (('charts.zip',[p for p in files if 'figures' in p.relative_to(destination).parts]),
                       ('results.zip',[p for p in files if 'figures' not in p.relative_to(destination).parts]+[destination/'files.csv'])):
        with zipfile.ZipFile(destination/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path in items:archive.write(path,path.relative_to(destination))
    return validation
