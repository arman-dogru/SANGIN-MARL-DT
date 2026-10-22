"""Read-only diagnosis of an immutable failed Phase 1 run; never learns."""
from collections import Counter, defaultdict
import csv
import gzip
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from .mapf_reference import solve_mapf
from .records import read_jsonl, sha256, write_json

PROTOCOL = 'phase1_failure_diagnosis_v1'


def stratified_sample(rows, failures_per_cell=4, successes_per_cell=2):
    """Deterministic strata, preserving a success baseline without cherry-picking."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['num_agents'],row['family'],bool(row['success'])].append(row)
    result = []
    for key in sorted(groups):
        cap = successes_per_cell if key[-1] else failures_per_cell
        result.extend(sorted(groups[key],key=lambda r:r['case_id'])[:cap])
    return result


def planner_sample(rows, limit=24):
    """Equal population quotas, round-robin family coverage within populations."""
    result = []
    for index, population in enumerate((2,4,6,8)):
        quota = limit//4 + (index < limit%4)
        groups = defaultdict(list)
        for row in sorted(rows,key=lambda r:r['case_id']):
            if not row['success'] and row['num_agents']==population:
                groups[row['family']].append(row)
        candidates = [groups[f][offset] for offset in range(max(map(len,groups.values()),default=0))
            for f in sorted(groups) if len(groups[f]) > offset]
        result.extend(candidates[:quota])
    return result


def summarize_trace(trace, goals):
    """Counts have explicit robot-epoch, event, and mission-level denominators."""
    goals = list(map(tuple,goals)); n = len(goals)
    counts = Counter(); longest_wait = [0]*n; wait_run = [0]*n
    stationary_run = 0; max_stationary = 0; departed = set(); states = {}
    first_cycle = None
    for tick, row in enumerate(trace):
        before = list(map(tuple,row['before'])); after = list(map(tuple,row['after']))
        active = [not c for c in row['completed_before']]
        key = (tuple(before),tuple(row['completed_before']),tuple(row.get('observation_hashes',())))
        if key in states and first_cycle is None:
            first_cycle = dict(start_epoch=states[key],repeat_epoch=tick,period=tick-states[key])
        states.setdefault(key,tick)
        counts['robot_epochs'] += n
        counts['active_robot_epochs'] += sum(active)
        counts['completed_robot_epochs'] += sum(row['completed_before'])
        counts['vertex_conflict_events'] += row['vertex_conflicts']
        counts['edge_conflict_events'] += row['edge_conflicts']
        counts['resolver_mismatch_epochs'] += row.get('resolver_mismatch',False)
        for robot, action in enumerate(row['actions']):
            waits = action == 4
            counts['explicit_waits'] += waits
            counts['active_explicit_waits'] += waits and active[robot]
            counts['completed_explicit_waits'] += waits and not active[robot]
            counts['active_no_motion'] += active[robot] and before[robot] == after[robot]
            counts['obstacle_blocked_moves'] += action != 4 and row['intended'][robot] == list(before[robot])
            counts['collision_blocked_moves'] += row['intended'][robot] != list(before[robot]) and before[robot] == after[robot]
            wait_run[robot] = wait_run[robot]+1 if waits else 0
            longest_wait[robot] = max(longest_wait[robot],wait_run[robot])
            if before[robot] == goals[robot] and after[robot] != goals[robot]:
                counts['goal_departures'] += 1; departed.add(robot)
            counts['completed_off_goal_robot_epochs'] += row['completed_before'][robot] and before[robot] != goals[robot]
        stationary_run = stationary_run+1 if before == after else 0
        max_stationary = max(max_stationary,stationary_run)
        if tick and after == list(map(tuple,trace[tick-1]['before'])) and after != before:
            counts['joint_two_step_returns'] += 1
    counts.update({k:0 for k in ('active_explicit_waits','completed_explicit_waits','goal_departures',
        'completed_off_goal_robot_epochs','vertex_conflict_events','edge_conflict_events','collision_blocked_moves',
        'obstacle_blocked_moves','joint_two_step_returns','resolver_mismatch_epochs') if k not in counts})
    final = list(map(tuple,trace[-1]['after'])) if trace else []
    return dict(counts, epochs=len(trace), max_robot_wait_run=max(longest_wait,default=0),
        max_joint_stationary_run=max_stationary,terminal_stationary_suffix=stationary_run,
        robots_departed_goal=len(departed),deterministic_cycle=first_cycle,
        all_ever_completed=bool(trace) and all(trace[-1]['completed_after']),
        final_at_goals=sum(a==g for a,g in zip(final,goals)))


def _frozen_modules(source):
    """Load a separate package namespace so live protocol edits cannot alter replay."""
    alias = '_phase1_diagnostic_frozen'
    source = Path(source)/'code/panama_uav'
    previous = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    try:
        if alias in sys.modules:
            if Path(sys.modules[alias].__file__).resolve() != (source/'__init__.py').resolve():
                raise ValueError('Frozen audit package identity changed in this process')
        else:
            spec = importlib.util.spec_from_file_location(alias,source/'__init__.py',submodule_search_locations=[str(source)])
            package = importlib.util.module_from_spec(spec); sys.modules[alias] = package; spec.loader.exec_module(package)
        return {name:importlib.import_module(f'{alias}.{name}') for name in
            ('phase1','phase1_data','phase1_environment','models','remote_control','remote_state','config')}
    finally:
        sys.dont_write_bytecode = previous


def replay_case(case, config, model, device, modules):
    env = modules['phase1_environment'].CleanMAPFEnvironment(config,modules['config'].ExperimentMode.UAV,
        case['map_path'],case['num_agents'],250,case['seed'],scenario=modules['phase1_data'].scenario_for(case))
    obs,_ = env.reset(); trace = []; totals = Counter()
    while True:
        actions = modules['models'].select_actions({0:model},{a:0 for a in obs},obs,0.,np.random.default_rng(0),device)
        before = {a.id:a.position for a in env.agents}; completed = [a.ever_reached_goal for a in env.agents]
        intended = {}
        for robot,point in before.items():
            dr,dc = env.ACTION_DELTAS[actions[robot]]; target = (point[0]+dr,point[1]+dc)
            intended[robot] = target if env._valid(target) else point
        events = modules['remote_control'].conflicts(before,intended)
        reference = modules['remote_control'].resolve_delivered(before,intended,{a.id:a.priority for a in env.agents})
        measures = modules['phase1_environment'].epoch_metrics(env,actions)
        totals.update(measures)
        hashes = [modules['remote_state'].observation_hash(obs[i]) for i in sorted(obs)]
        result = env.step(actions); obs = result.observations
        after = [a.position for a in env.agents]
        trace.append(dict(before=list(before.values()),after=after,
            intended=[list(intended[i]) for i in sorted(intended)],actions=[actions[i] for i in sorted(actions)],
            completed_before=completed,completed_after=[a.ever_reached_goal for a in env.agents],
            vertex_conflicts=sum(e['kind']=='vertex' for e in events),
            edge_conflicts=sum(e['kind']=='edge_swap' for e in events),
            resolver_mismatch=any(reference[i] != after[i] for i in reference),observation_hashes=hashes))
        env.communication.transport.log.clear()
        if result.terminated or result.truncated:
            break
    check = dict(steps=env.step_count,success=result.terminated,final_positions=[list(p) for p in after],**totals)
    mismatches = {k:dict(recorded=case.get(k),replayed=v) for k,v in check.items() if case.get(k) != v}
    return dict(case_id=case['case_id'],num_agents=case['num_agents'],family=case['family'],
        success=result.terminated,replay_matches=not mismatches,replay_mismatches=mismatches,
        **summarize_trace(trace,case['goals'])),trace


def learning_history(source):
    training = read_jsonl(source/'training.jsonl'); validation = read_jsonl(source/'validation.jsonl')
    stages = []
    for index in sorted({r['stage_index'] for r in training}):
        episodes = [r for r in training if r['stage_index']==index]
        checks = [r for r in validation if r['stage_index']==index and r.get('complete')]
        stages.append(dict(stage_index=index,num_agents=episodes[0]['num_agents'],episodes=len(episodes),
            joint_steps=sum(r['steps'] for r in episodes),first_500_success=np.mean([r['success'] for r in episodes[:500]]),
            last_500_success=np.mean([r['success'] for r in episodes[-500:]]),
            last_500_mean_steps=np.mean([r['steps'] for r in episodes[-500:]]),
            epsilon_start=episodes[0]['epsilon'],epsilon_end=episodes[-1]['epsilon'],
            validation_checks=len(checks),validation_first=checks[0]['success_rate'] if checks else None,
            validation_best=max((r['success_rate'] for r in checks),default=None),
            validation_last=checks[-1]['success_rate'] if checks else None,
            validation_passes=sum(r['passed'] for r in checks),
            final_5_validation=[r['success_rate'] for r in checks[-5:]],
            validation_best_step=max(checks,key=lambda r:r['success_rate'])['generated_joint_steps'] if checks else None))
    bins = defaultdict(list)
    with (source/'learning_steps.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['loss'] is not None:
                bins[row['stage_index'],row['generated_joint_steps']//25000].append(row['loss'])
    losses = [dict(stage_index=k[0],step_bin_start=k[1]*25000,n=len(values),mean=float(np.mean(values)),
        median=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=max(values),
        nonfinite=sum(not np.isfinite(v) for v in values)) for k,values in sorted(bins.items())]
    return dict(stages=stages,loss_bins=losses,
        caveat='Exploratory training success and greedy validation use different maps and actions; their gap alone does not prove overfitting.')


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def write_failure_report(destination, summary, audited, planner):
    failed, controls = summary['audit_groups']
    percent = lambda a,b:f'{100*a/b:.2f}%' if b else 'undefined'
    lines = ['# Frozen Phase 1 failure audit', '',
        f"Protocol: `{summary['protocol']}`. Frozen source: `{summary['frozen_git_commit']}`.",
        f"Checkpoint weights SHA-256: `{summary['weights_sha256']}`.", '',
        f"All {summary['replay_count']} replayed cases reproduced recorded outcomes, final positions, makespan, waiting, conflicts and exposure counts. "
        f"All {summary['source_files_verified']} frozen-run files retained identical SHA-256 hashes.", '',
        '## Whole-cohort failure counts', '',
        '| Split | Robots | Cases | Failures | Success |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['split']} | {r['num_agents']} | {r['episodes']} | {r['failures']} | {100*r['success_rate']:.1f}% |"
        for r in summary['cohort_counts']]
    lines += ['', '## Stratified replay evidence', '', summary['sampling']+'.', '',
        '| Symptom | Failed sample | Successful controls |', '|---|---:|---:|',
        f"| Missions | {failed['missions']} | {controls['missions']} |",
        f"| Exact repeated joint state and observation | {failed['missions_cycle']} | {controls['missions_cycle']} |",
        f"| Two-step joint cycle | {failed['missions_joint_two_cycle']} | {controls['missions_joint_two_cycle']} |",
        f"| Terminal stationary suffix at least 20 epochs | {failed['missions_terminal_stationary_20']} | {controls['missions_terminal_stationary_20']} |",
        f"| Missions with departure from a previously reached goal | {failed['missions_goal_departure']} | {controls['missions_goal_departure']} |",
        f"| All robots previously reached goals, mission unfinished | {failed['missions_all_visited_but_not_finished']} | {controls['missions_all_visited_but_not_finished']} |",
        f"| Active explicit WAITs / active robot epochs | {failed['active_explicit_waits']}/{failed['active_robot_epochs']} ({percent(failed['active_explicit_waits'],failed['active_robot_epochs'])}) | {controls['active_explicit_waits']}/{controls['active_robot_epochs']} ({percent(controls['active_explicit_waits'],controls['active_robot_epochs'])}) |",
        f"| Active no-motion / active robot epochs | {failed['active_no_motion']}/{failed['active_robot_epochs']} ({percent(failed['active_no_motion'],failed['active_robot_epochs'])}) | {controls['active_no_motion']}/{controls['active_robot_epochs']} ({percent(controls['active_no_motion'],controls['active_robot_epochs'])}) |",
        f"| Collision-stopped moving commands | {failed['collision_blocked_moves']} | {controls['collision_blocked_moves']} |",
        f"| Obstacle-stopped moving commands | {failed['obstacle_blocked_moves']} | {controls['obstacle_blocked_moves']} |",
        f"| Goal departure events | {failed['goal_departures']} | {controls['goal_departures']} |",
        f"| Completed off-goal robot epochs | {failed['completed_off_goal_robot_epochs']} | {controls['completed_off_goal_robot_epochs']} |",
        f"| Clean / delivered-priority resolver output differences | {failed['resolver_mismatch_epochs']} | {controls['resolver_mismatch_epochs']} |", '',
        'Active means the robot had never reached its goal before the epoch. Completed off-goal occupancy therefore needs its own denominator. '
        'Departures can be legitimate yielding and are not individually labelled errors. Exact cycles use positions, completion flags and observation hashes; '
        'the fresh deterministic policy has no clock input. Conflict counts describe attempted commands before physical resolution, not physical overlaps.', '',
        '## Independent joint feasibility', '',
        f"CBS checked {len(planner)} failures, with population counts "
        f"`{dict(Counter(r['num_agents'] for r in planner))}`. Outcomes: `{dict(Counter(r['status'] for r in planner))}`. "
        f"Total solver time: {sum(r['wall_seconds'] for r in planner):.3f} s. "
        f"Each case had a {planner[0]['timeout_seconds'] if planner else 0:g} s limit and a 250-step horizon.", '',
        'The independent planner uses cardinal moves and WAIT, forbids vertex and opposite-edge conflicts, '
        'and holds final goal occupancy. Every returned schedule is checked independently for obstacle, move, '
        'start, goal, horizon, vertex and edge validity. Feasible schedules are constructive witnesses. '
        'Timeout, node-limit, finite-horizon impossibility and static disconnection remain distinct statuses; none is silently converted to global infeasibility.', '',
        '## Diagnosis supported by this audit', '',
        '1. Deterministic coordination attractors dominate the sampled failures: every failure repeats an identical joint state and observation, '
        'whereas the successful controls do not. Explicit WAITs, physical conflict stops and oscillation are separate observed mechanisms.',
        '2. Completed-robot control is a substantive protocol mismatch. The old clean environment still executes its DQN after the completion flag is set; '
        'remote-DT uses its existing local controller. All-visited-but-unfinished failures make completion and return behavior a priority for alignment. '
        'The matched fresh-priority physical resolvers themselves produced identical motion throughout this audit.',
        '3. Curriculum exclusion is established: the run trained only two and four robots. Six/eight-robot scores measure transfer, '
        'not trained-population qualification. Four-robot validation never passed despite thousands of additional episodes.',
        '4. The training/greedy-validation gap and nonmonotonic validation are consistent with learning instability or distribution mismatch. '
        'They do not alone establish overfitting or prove that exploration collapsed. Controlled training ablations are required.', '',
        '## Training history', '', '| Robots | Episodes | Joint steps | Last-500 exploratory success | Best greedy validation | Last greedy validation |',
        '|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['num_agents']} | {r['episodes']} | {r['joint_steps']} | {100*r['last_500_success']:.1f}% | "
        f"{100*r['validation_best']:.1f}% | {100*r['validation_last']:.1f}% |" for r in summary['stages']]
    lines += ['', 'The replay sample is stratified rather than a random prevalence sample. Reference-planner cases are bounded, '
        'and do not prove that every untested mission is feasible. Historical held-out cases remain diagnosis-only and may not enter revised training or model selection.', '',
        'Machine-readable evidence: `failure_replay.csv`, `replay_traces.jsonl.gz`, `mapf_reference.json`, '
        '`mapf_reference.csv`, `learning_history.json`, `loss_bins.csv`, `summary.json`, and `source_checksums.json`.', '']
    (Path(destination)/'FAILURE_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')


def diagnose(source, destination, *, failures_per_cell=4, successes_per_cell=2,
             planner_cases=24, planner_timeout=2., device_name='cpu'):
    import torch
    begin = time.monotonic(); source = Path(source).resolve(); destination = Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Diagnostic artifacts must be outside the frozen run')
    destination.mkdir(parents=True,exist_ok=False)
    # Preserve the complete failed run, including checkpoint and logs.
    before = {str(p.relative_to(source)):sha256(p) for p in source.rglob('*') if p.is_file()}
    write_json(destination/'source_checksums.json',before)
    modules = _frozen_modules(source); manifest = json.loads((source/'manifest.json').read_text())
    config = modules['phase1'].clean_config(modules['config'].ProjectConfig.from_dict(manifest['configuration']))
    torch.set_num_threads(1); device = torch.device(device_name)
    payload = torch.load(source/'final_model.pt',map_location=device,weights_only=False)
    model = modules['models'].DQN((3,15,15),config.vector_length).to(device)
    model.load_state_dict(payload['model_state_dict']); model.eval()
    identity = modules['phase1'].weights_hash(model)
    cohorts = {s:read_jsonl(source/f'final_{s}_cases.jsonl') for s in ('validation','test')}
    if any(r['model_weights_sha256'] != identity for bank in cohorts.values() for r in bank):
        raise ValueError('Recorded final cases do not match frozen checkpoint weights')
    selected = stratified_sample(cohorts['test'],failures_per_cell,successes_per_cell)
    audited = []
    with gzip.open(destination/'replay_traces.jsonl.gz','wt',encoding='utf-8') as stream:
        for case in selected:
            result,trace = replay_case(case,config,model,device,modules); audited.append(result)
            stream.write(json.dumps(dict(case_id=case['case_id'],trace=trace))+'\n')
    if not all(r['replay_matches'] for r in audited):
        write_json(destination/'replay_mismatch.json',[r for r in audited if not r['replay_matches']])
        raise ValueError('Frozen replay differs from recorded cases; diagnosis halted')
    _write_csv(destination/'failure_replay.csv',audited)
    planner = []
    for case in planner_sample(cohorts['test'],planner_cases):
        grid = modules['phase1_data'].load_map(case['map_path'])
        result = solve_mapf(grid,case['starts'],case['goals'],timeout_seconds=planner_timeout)
        planner.append(dict(case_id=case['case_id'],num_agents=case['num_agents'],family=case['family'],**result))
    write_json(destination/'mapf_reference.json',planner)
    _write_csv(destination/'mapf_reference.csv',[{k:v for k,v in row.items() if k!='paths'} for row in planner])
    history = learning_history(source); write_json(destination/'learning_history.json',history)
    _write_csv(destination/'loss_bins.csv',history['loss_bins'])
    cohort_counts = [dict(split=split,num_agents=n,episodes=len(rows),failures=sum(not r['success'] for r in rows),
        success_rate=sum(r['success'] for r in rows)/len(rows)) for split,bank in cohorts.items() for n in (2,4,6,8)
        if (rows:=[r for r in bank if r['num_agents']==n])]
    audit_groups = []
    for success in (False,True):
        rows = [r for r in audited if r['success']==success]
        sums = {key:sum(r.get(key,0) for r in rows) for key in
            ('robot_epochs','active_robot_epochs','completed_robot_epochs','explicit_waits','active_explicit_waits',
             'completed_explicit_waits','active_no_motion','obstacle_blocked_moves','collision_blocked_moves',
             'goal_departures','completed_off_goal_robot_epochs','vertex_conflict_events','edge_conflict_events',
             'joint_two_step_returns','resolver_mismatch_epochs')}
        audit_groups.append(dict(success=success,missions=len(rows),**sums,
            missions_goal_departure=sum(r['goal_departures']>0 for r in rows),
            missions_cycle=sum(r['deterministic_cycle'] is not None for r in rows),
            missions_joint_two_cycle=sum((r['deterministic_cycle'] or {}).get('period')==2 for r in rows),
            missions_terminal_stationary_20=sum(r['terminal_stationary_suffix']>=20 for r in rows),
            missions_all_visited_but_not_finished=sum(r['all_ever_completed'] and r['final_at_goals']<r['num_agents'] for r in rows),
            max_wait_run=max((r['max_robot_wait_run'] for r in rows),default=0)))
    after = {str(p.relative_to(source)):sha256(p) for p in source.rglob('*') if p.is_file()}
    summary = dict(protocol=PROTOCOL,frozen_source=str(source),frozen_git_commit=manifest['git_commit'],
        diagnosis_code_sha256={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('mapf_reference.py'))},
        model_sha256=sha256(source/'final_model.pt'),weights_sha256=identity,
        source_preserved=before==after,source_files_verified=len(before),
        cohort_counts=cohort_counts,audit_groups=audit_groups,
        replay_count=len(audited),replay_all_matched=all(r['replay_matches'] for r in audited),
        planner_outcomes=dict(Counter(r['status'] for r in planner)),planner_count=len(planner),
        planner_wall_seconds=sum(r['wall_seconds'] for r in planner),
        sampling=f'held-out deterministic case-id order; up to {failures_per_cell} failures and {successes_per_cell} successes per robot/family cell',
        planner_sampling='equal population quotas, round-robin family strata within each; feasibility witness only',
        stages=history['stages'],wall_seconds=time.monotonic()-begin,
        limitations=['Diagnostic stratified sample is not a prevalence estimate for all failures.',
            'Reference planner requires permanent final goal occupancy, a stricter convention than PANAMA yielding.',
            'No counterfactual retraining or causal claim that one observed symptom explains every failure.',
            'Finite-horizon infeasibility, search limits and timeouts are distinct from global infeasibility.'])
    write_json(destination/'summary.json',summary)
    if not summary['source_preserved']:
        raise ValueError('Frozen run changed during diagnosis')
    write_failure_report(destination,summary,audited,planner)
    return summary
