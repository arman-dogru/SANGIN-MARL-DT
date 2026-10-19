"""Frozen-policy 100 ms decision budget and paired satellite-handover ablations."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace, asdict
from datetime import datetime, UTC
import argparse
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_leo_inference as base

PROJECT = Path(__file__).resolve().parents[1]
FIELDS = base.FIELDS
COUNTS = base.COUNTS


def conditions(config, root):
    common = replace(config.communication, profile='physical', contention_enabled=False,
                     leo_timeline=str(root/'assets/orbit_native.npz'))
    result = [('perfect', replace(common, profile='perfect', stale_information=(), handover_enabled=False), 'perfect', 'perfect')]
    for group in ('edge_control', 'handover_off', 'handover_on', 'stress_10s', 'stress_20s'):
        for fields in FIELDS:
            field = 'all' if len(fields) == 3 else fields[0]
            timeline = root/'assets'/('orbit_stress10.npz' if group == 'stress_10s' else
                                      'orbit_stress20.npz' if group == 'stress_20s' else 'orbit_native.npz')
            comm = replace(common, mission_route='uav' if group == 'edge_control' else 'cloud',
                stale_information=fields, handover_enabled=group != 'handover_off', leo_timeline=str(timeline))
            result.append((f'{group}_{field}', comm, group, field))
    return result


def prepare(args):
    root = base.prepare(args)
    from panama_uav.handover_study import derive_timeline
    for duration, name in ((0, 'native'), (10, 'stress10'), (20, 'stress20')):
        derive_timeline(root/'assets/orbit.npz', root/f'assets/orbit_{name}.npz', duration)
    for script in ('run_deadline_inference.py', 'plot_deadline_inference.py'):
        shutil.copy2(PROJECT/'scripts'/script, root/'code'/script)
    previous = args.previous_root.resolve()
    shutil.copy2(previous/'held_out_cases.json', root/'held_out_cases.json')
    plan = json.loads((root/'plan.json').read_text())
    assert base.sha(previous/'baseline.pt') == plan['baseline_model_sha256']
    config = base.config_from(plan['configuration'])
    plan.update(inference_protocol='decision-deadline-v1', decision_deadline_seconds=.1,
        previous_root=str(previous), case_bank_sha256=base.sha(root/'held_out_cases.json'),
        timeline_sha256={p.name:base.sha(p) for p in (root/'assets').glob('orbit*.npz')},
        script_sha256={p.name:base.sha(p) for p in (root/'code').glob('*.py')},
        planned_episode_executions=4200, unique_conditions=21,
        conditions=[dict(name=n, transport=asdict(c), group=g, fields=f) for n,c,g,f in conditions(config,root)],
        timing='Reports at k seconds; arrivals through k+0.1 processed before decision; one action per existing 1-second cycle; no extra robot steps or communication-triggered stopping',
        handover_ablation='Only actual satellite-change interruptions toggled; resource reconfiguration, geometry, rates, frame guards, queues and propagation held fixed',
        stress='10/20 seconds total service interruption at actual satellite changes; artificial stress, overlapping outages unioned, no per-agent reservation windows',
        perfect_reference='Same frozen model, seeds, starts/goals as previous run; perfect trajectory outcomes must match',
        trained_policy_changes='None: architecture, parameters, rewards, local sensing and task transitions unchanged')
    for key in ('controlled_delays_steps', 'pairing', 'profile_separation'):
        plan.pop(key, None)
    base.write(root/'plan.json', plan)
    return root


def init_worker(root, device):
    global ROOT, PLAN, CONFIG, SPECS
    ROOT = Path(root)
    base.init_worker(root, device)
    PLAN = json.loads((ROOT/'plan.json').read_text())
    CONFIG = base.config_from(PLAN['configuration'])
    SPECS = {n:(c,g,f) for n,c,g,f in conditions(CONFIG,ROOT)}


def union_seconds(windows):
    merged = []
    for start, end in sorted(windows):
        if end <= start: continue
        if merged and start <= merged[-1][1]: merged[-1][1] = max(merged[-1][1],end)
        else: merged.append([start,end])
    return sum(end-start for start,end in merged)


def episode_job(task):
    import numpy as np
    from panama_uav.evaluation import run_episode
    name, case, *extra = task
    prefix = extra[0] if extra else ''
    assert prefix in ('', 'preflight')
    comm, group, field = SPECS[name]
    config = replace(CONFIG, communication=comm)
    row, trace = run_episode(config, base.MODEL, case, base.DEVICE, base.PAYLOAD['learner_updates'], decision_deadline_seconds=.1)
    assert trace['frames'][0]['positions'] == case['starts'] and trace['goals'] == case['goals']
    frames = [f for f in trace['frames'] if f['decision'] is not None]
    assert len(frames) == row['steps']
    decisions = [f['decision'] for f in frames]
    assert all(abs(d['at']-d['cycle_start']-.1)<1e-9 for d in decisions)
    selected = [d['inputs'][a][k] for d in decisions if not d['initial_briefing']
                for a in d['inputs'] for k in comm.stale_information]
    late = sum(v['response_missed_deadline'] for v in selected)
    stale = sum(v['source_from_previous_cycle'] for v in selected)
    row.update(condition=name, stage=COUNTS.index(case['num_agents'])+1, group=group, fields=field,
        mission_route=comm.mission_route, stale_information=list(comm.stale_information),
        policy_model_sha256=PLAN['baseline_model_sha256'], decision_deadline_seconds=.1,
        response_opportunities=len(selected), deadline_misses=late,
        deadline_miss_fraction=late/len(selected) if selected else 0.,
        previous_cycle_sources=stale, source_stale_fraction=stale/len(selected) if selected else 0.,
        artificial_handover_seconds=10 if group=='stress_10s' else 20 if group=='stress_20s' else 0,
        handover_enabled=comm.handover_enabled, diagnostic=bool(prefix),
        interruption_seconds_realized=union_seconds(trace['handover_windows']))
    for key in ('task','intentions','priorities'):
        ages = [v[key] for f in frames for v in f['information_age'].values()]
        row[f'decision_mean_{key}_age_seconds'] = float(np.mean(ages))
        row[f'decision_max_{key}_age_seconds'] = float(max(ages))
        if key not in comm.stale_information or comm.profile=='perfect': assert max(ages)==0
    row['selected_input_mean_age_seconds'] = float(np.mean([v['age_seconds'] for v in selected])) if selected else 0.
    delivered = [e['age_seconds'] for e in trace['transport_events'] if e['event']=='delivered' and e['kind']=='information']
    row.update(information_deliveries=len(delivered), information_delivery_p95_seconds=float(np.quantile(delivered,.95)) if delivered else None,
        information_delivery_max_seconds=max(delivered) if delivered else None,
        maximum_queue_bytes=max(sum(f['queue_bytes'].values()) for f in trace['frames']))
    handovers = [e for e in trace['transport_events'] if e['event']=='orbit_handover'
                 and e['source_window'].get('service_category')=='satellite_change']
    row['satellite_changes_observed'] = len(handovers)
    row['satellite_interruption_events_applied'] = sum(e['source_window']['interruption_applied'] for e in handovers)
    assert row['steps']<=250 and row['par2']==(row['steps'] if row['success'] else 500)
    target = ROOT/prefix/name/f"{case['case_id']}.json.gz"
    temp = target.with_suffix('.gz.tmp')
    with gzip.open(temp,'wt',encoding='utf-8',compresslevel=1) as f: json.dump(trace,f,separators=(',',':'),allow_nan=False)
    os.replace(temp,target)
    return row


def summarize(rows, reference, handover_off):
    import numpy as np
    from panama_uav.evaluation import paired_summary
    summary = paired_summary(rows, reference)
    off_summary = {r['num_agents']:r for r in paired_summary(rows, handover_off)} if handover_off else {}
    for item in summary:
        group = [r for r in rows if r['num_agents']==item['num_agents']]
        first = group[0]
        for key in ('condition','stage','group','fields','mission_route','artificial_handover_seconds'):
            item[key] = first[key]
        item['mean_episode_steps'] = float(np.mean([r['steps'] for r in group]))
        opportunities = sum(r['response_opportunities'] for r in group)
        item.update(deadline_miss_fraction=sum(r['deadline_misses'] for r in group)/opportunities if opportunities else 0.,
            source_stale_fraction=sum(r['previous_cycle_sources'] for r in group)/opportunities if opportunities else 0.,
            selected_input_mean_age_seconds=float(np.mean([r['selected_input_mean_age_seconds'] for r in group])),
            mean_queue_peak_bytes=float(np.mean([r['maximum_queue_bytes'] for r in group])),
            mean_realized_interruption_seconds=float(np.mean([r['interruption_seconds_realized'] for r in group])),
            total_satellite_changes_observed=sum(r['satellite_changes_observed'] for r in group),
            applied_satellite_interruption_events=sum(r['satellite_interruption_events_applied'] for r in group))
        for key in ('paired_par2_delta','paired_par2_ci95','paired_success_delta','paired_success_ci95'):
            item['against_handover_off_'+key] = off_summary[item['num_agents']][key] if off_summary else None
    return summary


def run(root):
    import psutil
    root = root.resolve()
    sys.path.insert(0,str(root/'code'))
    from panama_uav.records import append_jsonl, export_csv
    plan = json.loads((root/'plan.json').read_text())
    assert base.sha(root/'baseline.pt')==plan['baseline_model_sha256']
    assert base.sha(root/'held_out_cases.json')==plan['case_bank_sha256']
    assert all(base.sha(root/'assets'/n)==h for n,h in plan['timeline_sha256'].items())
    assert all(base.sha(root/'code'/n)==h for n,h in plan['script_sha256'].items())
    prior = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    if prior.get('status')=='complete':
        print('Already complete; retained measured status and runtime.',flush=True)
        return
    config = base.config_from(plan['configuration'])
    specs = conditions(config,root)
    bank = json.loads((root/'held_out_cases.json').read_text())
    base.write(root/'manifest.json',plan)
    base.write(root/'launch.json',dict(pid=os.getpid(),creation_utc=datetime.fromtimestamp(psutil.Process().create_time(),UTC).isoformat(),
        command_args=[sys.executable,*sys.argv]))
    state = dict(status='running',started_utc=prior.get('started_utc',base.utc()),completed_conditions=[],
        completed_episodes=0,planned_episodes=len(bank)*len(specs),workers=plan['workers'])
    start = time.monotonic()
    elapsed_prior = prior.get('elapsed_seconds',0)
    summaries, reference, off = [], [], {}
    pool = ProcessPoolExecutor(plan['workers'],mp_context=multiprocessing.get_context('spawn'),
        initializer=init_worker,initargs=(str(root),plan['device']))
    try:
        for name,comm,group,field in specs:
            directory = root/name
            directory.mkdir(exist_ok=True)
            path = directory/'episodes.jsonl'
            rows = [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []
            done = {r['case_id'] for r in rows}
            assert len(done)==len(rows)
            state.update(active_condition=name,condition_episodes=len(rows),elapsed_seconds=elapsed_prior+time.monotonic()-start)
            state['completed_episodes'] += len(rows)
            base.write(root/'status.json',state)
            print(f'START {name} ({len(rows)}/200 retained)',flush=True)
            pending = {pool.submit(episode_job,(name,case)) for case in bank if case['case_id'] not in done}
            progress = time.monotonic()
            while pending:
                finished,pending = wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                if not finished and time.monotonic()-progress>600: raise RuntimeError('No episode completed for ten minutes')
                for future in finished:
                    row = future.result()
                    append_jsonl(path,row)
                    rows.append(row)
                    state['completed_episodes']+=1
                    state['condition_episodes']+=1
                    state.update(latest_case=row['case_id'],latest_stage=row['stage'],elapsed_seconds=elapsed_prior+time.monotonic()-start)
                    base.write(root/'status.json',state)
                    progress = time.monotonic()
                if state['condition_episodes']%50==0: print(json.dumps(state),flush=True)
            assert len(rows)==len(bank)
            rows.sort(key=lambda r:(r['num_agents'],r['case']))
            if name=='perfect': reference=rows
            if group=='handover_off': off[field]=rows
            summary = summarize(rows,reference,off.get(field))
            summaries.extend(summary)
            base.write(directory/'summary.json',summary)
            export_csv(path,directory/'episodes.csv')
            base.write(root/'summary.json',summaries)
            state['completed_conditions'].append(name)
            base.write(root/'status.json',state)
        pool.shutdown()
        state.update(status='complete',active_condition=None,finished_utc=base.utc(),elapsed_seconds=elapsed_prior+time.monotonic()-start)
        base.write(root/'status.json',state)
        print(json.dumps(state),flush=True)
    except BaseException as error:
        state.update(status='error',error=repr(error),failed_utc=base.utc(),elapsed_seconds=elapsed_prior+time.monotonic()-start)
        base.write(root/'status.json',state)
        for p in (pool._processes or {}).values():
            if p.is_alive(): p.terminate()
        pool.shutdown(wait=False,cancel_futures=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',type=Path,default=Path('output/leo-learning/leo-learning-20261009-01-recovery-03'))
    p.add_argument('--previous-root',type=Path,default=Path('output/inference/leo-baseline-20261019-02'))
    p.add_argument('--run-id')
    p.add_argument('--root',type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--device',default='cuda')
    args = p.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['OMP_NUM_THREADS']='1'
    if not 1<=args.workers<=8: p.error('Use 1–8 workers')
    if not args.root and not args.run_id: p.error('--run-id or --root is required')
    run(args.root if args.root else prepare(args))


if __name__=='__main__': main()
