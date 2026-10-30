"""Final held-out evaluation: curriculum vs 32-agent baseline vs model8.

Two parts, both written incrementally so a wall-clock cutoff still leaves usable output:

* MAPF: held-out random-obstacle maps (32-64 cells per side, 10-30 % obstacles) whose map and case
  seeds are disjoint from every training and validation seed of ``mapf_curriculum`` (train maps use
  seed+10M+i, validation maps seed+60M+i, episodes seed+70M.., validation cases seed+900M..; test
  maps here use seed+110M+i and cases seed+1.2B+N*1M+i). Starts and goals are placed in swapped
  pairs (PANAMA reset). Case ``i`` uses test map ``i`` for every population and every model, so the
  comparison is paired. Greedy policy, clean remote-aligned execution, 250-step cap (as validation).
* LEO: the leo_study harness (``leo_study/harness/run_job.py``) with the model swapped: treatment A4,
  physical LEO network at nominal and +150 ms downlink, cloud and allocated routes, N = 8/16/32,
  ``--leo-cases`` cases each on the study's fixed 32x32 map.

Bulky per-unit files (maps, logs, LEO result folders) go to ``$MAPF_EVAL_SCRATCH`` (default
``<out>/scratch``) and are tarred into ``<out>/scratch.tar`` at the end; the compact results stay in ``<out>``.
Each unit of work runs in its own subprocess (``task``), so an out-of-memory kill or a hang only loses
that unit: it is logged in ``tasks.jsonl`` and skipped. ``run`` stops launching work at the wall
guard, kills what is still running and rewrites the summary (``results.json``, ``*.csv``,
``summary.md``) after every finished unit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROTOCOL = 'panama_mapf_eval_v1'
MAP_SEED_OFFSET = 110_000_000      # train maps +10M+i, validation maps +60M+i (mapf_curriculum.MapBank)
CASE_SEED_OFFSET = 1_200_000_000   # training episodes +70M.., validation cases +900M..
PLACEMENT_RETRIES = 5
LEO_TREATMENT = 'A4'
LEO_ROUTES = ('cloud', 'allocated')
LEO_DOWNLINKS = (0.0, 0.15)
CAP = 250


# ----------------------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:   # a line cut by a kill
            pass
    return rows


def append_jsonl(path, row):
    with open(path, 'a') as stream:
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def write_atomic(path, text):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def scratch_dir(out):
    """Bulky per-unit files (maps, task logs, LEO result folders). Kaggle limits the number of output
    files, so on Kaggle this lives outside /kaggle/working and is tarred at the end."""
    return Path(os.environ.get('MAPF_EVAL_SCRATCH') or Path(out) / 'scratch')


# ----------------------------------------------------------------------------- model location
def locate_model(spec, out):
    """``spec`` is a file path, or ``search:<run>`` (latest_model.pt of a curriculum run found under
    the search roots as a folder or inside <run>.tar), or ``search-file:<name>``."""
    kind, _, name = spec.partition(':')
    roots = [Path(p) for p in os.environ.get('MAPF_EVAL_SEARCH', '/kaggle/input').split(os.pathsep) if p]
    if kind == 'search':
        for root in roots:
            if not root.exists():
                continue
            for q in sorted(root.rglob('latest_model.pt')):
                if q.parent.name == name:
                    return q
        for root in roots:
            if not root.exists():
                continue
            for q in sorted(root.rglob(name + '.tar')):
                dst = scratch_dir(out) / 'models' / name
                dst.mkdir(parents=True, exist_ok=True)
                with tarfile.open(q) as tar:
                    member = tar.getmember(f'{name}/latest_model.pt')
                    member.name = 'latest_model.pt'
                    tar.extract(member, dst)
                return dst / 'latest_model.pt'
        raise FileNotFoundError(f'no {name}/latest_model.pt or {name}.tar under {roots}')
    if kind == 'search-file':
        for root in roots:
            if root.exists():
                found = sorted(root.rglob(name))
                if found:
                    return found[0]
        raise FileNotFoundError(f'{name} not under {roots}')
    return Path(spec)


def budget_of(payload):
    """Training budget recorded in a model payload, by stage (episodes, joint steps, agent-steps)."""
    stages = []
    if payload.get('training_protocol') == 'panama_mapf_curriculum_v1':
        history = payload.get('stage_history', [])
        prev = 0
        for h in history:
            steps = h['generated_joint_steps'] - prev
            prev = h['generated_joint_steps']
            stages.append(dict(population=h['population'], episodes=h['stage_episodes'], joint_steps=steps,
                graduated=h['graduated'], final_validation_success=h['success_rate']))
        done = {h['population'] for h in history}
        for n, eps in payload['population_episodes'].items():
            if int(n) not in done:
                stages.append(dict(population=int(n), episodes=eps,
                    joint_steps=payload['generated_joint_steps'] - prev, graduated=None,
                    final_validation_success=None, note='stage in progress when training stopped'))
    else:
        steps = payload.get('population_steps', {})
        for n, eps in payload.get('population_episodes', {}).items():
            stages.append(dict(population=int(n), episodes=eps, joint_steps=steps.get(n)))
    for s in stages:
        s['agent_steps_upper'] = s['joint_steps'] * s['population'] if s.get('joint_steps') is not None else None
    return dict(training_protocol=payload.get('training_protocol'), run_id=payload.get('run_id'),
        stages=stages, total_episodes=sum(s['episodes'] for s in stages),
        total_joint_steps=payload.get('generated_joint_steps'),
        total_agent_steps_upper=sum(s['agent_steps_upper'] or 0 for s in stages),
        learner_updates=payload.get('learner_updates'),
        episodes_at_32=sum(s['episodes'] for s in stages if s['population'] == 32),
        joint_steps_at_32=sum(s['joint_steps'] or 0 for s in stages if s['population'] == 32),
        max_trained_population=max((s['population'] for s in stages), default=None))


# ----------------------------------------------------------------------------- MAPF part
def mapf_setup(project_root):
    from .mapf_curriculum import load_spec, project_config
    spec = load_spec(Path(project_root) / 'configs/mapf_curriculum.toml')
    return spec, project_config(spec, project_root)


def test_map(out, spec, index):
    from .mapf_curriculum import random_map, write_map
    seed = spec.seed + MAP_SEED_OFFSET + index
    path = scratch_dir(out) / 'maps' / f'test-{index:05d}.map'
    grid, size, density = random_map(seed, spec.map_size, spec.density)
    if not path.exists():
        write_map(grid, path)
    return grid, path, dict(map_id=f'test-{index:05d}', map_seed=seed, map_size=size, nominal_density=density,
        density=float(grid.mean()))


def test_case(out, spec, count, index):
    from .mapf_curriculum import paired_case
    grid, path, meta = test_map(out, spec, index)
    base = spec.seed + CASE_SEED_OFFSET + count * 1_000_000 + index
    for attempt in range(PLACEMENT_RETRIES):
        seed = base + attempt * 7919
        try:
            starts, goals = paired_case(grid, count, seed, spec.pair_distance)
        except ValueError:
            continue
        return dict(meta, case_id=f"{meta['map_id']}-n{count}", map_path=str(path), family='random_obstacles',
            split='test', num_agents=count, seed=int(seed), placement_attempt=attempt,
            starts=[list(p) for p in starts], goals=[list(p) for p in goals],
            designed_coordination=True, paired=True, evaluation_index=index)
    return None


def run_mapf_case(config, model, case, device):
    import torch
    from .config import ExperimentMode
    from .models import select_actions
    from .phase1_data import scenario_for
    from .phase1_execution import RemoteAlignedCleanMAPFEnvironment
    env = RemoteAlignedCleanMAPFEnvironment(config, ExperimentMode.UAV, case['map_path'], case['num_agents'], CAP,
        case['seed'], scenario_for(case))
    obs, _ = env.reset()
    arrival = {}
    totals = Counter(dict.fromkeys(('attempted_vertex_conflicts', 'attempted_edge_swaps',
        'agents_in_attempted_collisions', 'obstacle_bumps', 'post_resolution_overlaps'), 0))
    t0 = time.monotonic()
    with torch.no_grad():
        while True:
            active = {a.id for a in env.agents if not a.ever_reached_goal}
            proposed = select_actions({0: model}, {a: 0 for a in active}, {a: obs[a] for a in active}, 0.0,
                np.random.default_rng(0), device)
            result = env.step(proposed)
            info = result.info
            for event in info.get('physical_conflicts', []) or []:
                totals['attempted_vertex_conflicts' if event['kind'] == 'vertex' else 'attempted_edge_swaps'] += 1
            totals['agents_in_attempted_collisions'] += info.get('collisions_agent_agent_step', 0)
            totals['obstacle_bumps'] += info.get('collisions_obstacle_step', 0)
            totals['post_resolution_overlaps'] += info.get('actual_agent_overlap_after_resolution', 0)
            for a in env.agents:
                if a.ever_reached_goal and a.id not in arrival:
                    arrival[a.id] = env.step_count
            obs = result.observations
            env.communication.transport.log.clear()
            if result.terminated or result.truncated:
                break
    success = bool(result.terminated)
    n = case['num_agents']
    soc_capped = sum(arrival.get(a.id, CAP) for a in env.agents)
    return dict(steps=env.step_count, success=success, makespan=env.step_count if success else None,
        capped_makespan=env.step_count if success else CAP, sum_of_costs=soc_capped if success else None,
        capped_sum_of_costs=soc_capped, agents_completed=len(arrival), completion_fraction=len(arrival) / n,
        collisions=totals['attempted_vertex_conflicts'] + totals['attempted_edge_swaps'],
        wall_seconds=time.monotonic() - t0, **totals)


def task_mapf(out, task, project_root):
    import torch
    from .training import load_model
    torch.set_num_threads(1)
    spec, config = mapf_setup(project_root)
    device = torch.device('cpu')
    model, payload = load_model(task['model_path'], config, device, require_qualified=False)
    model.eval()
    path = Path(out) / 'mapf' / f"{task['model']}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = {(r['num_agents'], r['evaluation_index']) for r in read_jsonl(path)}
    for index in range(*task['cases']):
        if (task['num_agents'], index) in seen:
            continue
        case = test_case(out, spec, task['num_agents'], index)
        if case is None:
            append_jsonl(Path(out) / 'placement_failures.jsonl', dict(num_agents=task['num_agents'], index=index))
            continue
        metrics = run_mapf_case(config, model, case, device)
        row = {k: v for k, v in case.items() if k not in ('starts', 'goals')}
        row.update(metrics, model=task['model'], protocol=PROTOCOL)
        append_jsonl(path, row)


# ----------------------------------------------------------------------------- LEO part
def load_run_job(project_root):
    harness = Path(project_root) / 'leo_study/harness/run_job.py'
    spec = importlib.util.spec_from_file_location('leo_run_job', harness)
    module = importlib.util.module_from_spec(spec)
    sys.argv = [str(harness)]          # run_job reads --keep-trace from argv at import
    spec.loader.exec_module(module)
    return module


def leo_job(route, downlink, count, case):
    group = 'nominal' if not downlink else 'delay'
    return dict(group=group, treatment=LEO_TREATMENT, route=route, network='physical', downlink_seconds=downlink,
        num_agents=count, case=case,
        job_id=f'{group}__{LEO_TREATMENT}__{route}__physical__dl{int(round(downlink * 1000))}__n{count}__c{case:02d}')


def task_leo(out, task, project_root):
    run_job = load_run_job(project_root)
    run_job.MODEL = str(task['model_path'])
    root = scratch_dir(out) / 'leo' / task['model']
    root.mkdir(parents=True, exist_ok=True)
    run_job.HEAVY_LOCK = root / f'heavy-{os.getpid()}.lock'   # own lock file: no cross-worker serialisation
    run_job.TRIM_FILE = root / 'no-trim.json'                  # nothing trimmed; failures are caught per task
    job = task['job']
    packet = run_job.run(job, str(root))
    final = root / job['job_id'] / 'result.json'
    if packet is not None:
        packet['row']['model_sha256'] = task['model_sha256']   # run_job hard-codes model8's hash
        packet['row']['model'] = task['model']
        text = json.dumps(packet, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        write_atomic(final, text)
        compact = json.loads(text)
        append_jsonl(Path(out) / 'leo_rows.jsonl', dict(model=task['model'], job=compact['job'], row=compact['row']))
    for lock in root.glob(f'heavy-{os.getpid()}.lock'):
        lock.unlink()


# ----------------------------------------------------------------------------- summary
def safe_summarize(out):
    try:
        return summarize(out)
    except Exception as error:   # a summary bug must never stop the evaluation itself
        append_jsonl(Path(out) / 'summary_errors.jsonl', dict(error=repr(error), unix=time.time()))
        return None


def summarize(out):
    out = Path(out)
    meta = json.loads((out / 'eval_manifest.json').read_text())
    models = list(meta['models'])
    tasks = read_jsonl(out / 'tasks.jsonl')
    mapf = {m: read_jsonl(out / 'mapf' / f'{m}.jsonl') for m in models}
    cells = []
    by_case = defaultdict(dict)
    for m in models:
        groups = defaultdict(list)
        for r in mapf[m]:
            groups[r['num_agents']].append(r)
            by_case[(r['num_agents'], r['evaluation_index'])][m] = r['success']
        for n in sorted(groups):
            rows = groups[n]
            k = sum(r['success'] for r in rows)
            lo, hi = wilson(k, len(rows))
            cells.append(dict(part='mapf', model=m, num_agents=n, episodes=len(rows), successes=k,
                success_rate=k / len(rows), success_ci95_low=lo, success_ci95_high=hi,
                makespan_success_mean=mean([r['makespan'] for r in rows]),
                capped_makespan_mean=mean([r['capped_makespan'] for r in rows]),
                sum_of_costs_success_mean=mean([r['sum_of_costs'] for r in rows]),
                capped_sum_of_costs_mean=mean([r['capped_sum_of_costs'] for r in rows]),
                completion_fraction_mean=mean([r['completion_fraction'] for r in rows]),
                collisions_per_episode=mean([r['collisions'] for r in rows]),
                vertex_conflicts_per_episode=mean([r['attempted_vertex_conflicts'] for r in rows]),
                edge_swaps_per_episode=mean([r['attempted_edge_swaps'] for r in rows]),
                obstacle_bumps_per_episode=mean([r['obstacle_bumps'] for r in rows]),
                post_resolution_overlaps=sum(r['post_resolution_overlaps'] for r in rows),
                mean_case_wall_seconds=mean([r['wall_seconds'] for r in rows])))
    # paired success differences on cases every model finished (bootstrap over cases)
    paired = []
    rng = np.random.default_rng(0)
    for n in sorted({key[0] for key in by_case}):
        common = [v for key, v in by_case.items() if key[0] == n and all(m in v for m in models)]
        for a, b in (('curriculum', 'baseline32'), ('curriculum', 'model8'), ('baseline32', 'model8')):
            if a not in models or b not in models or not common:
                continue
            d = np.array([int(v[a]) - int(v[b]) for v in common], dtype=float)
            boot = rng.choice(d, size=(2000, len(d))).mean(1)
            paired.append(dict(num_agents=n, a=a, b=b, cases=len(d), success_diff=float(d.mean()),
                ci95_low=float(np.quantile(boot, 0.025)), ci95_high=float(np.quantile(boot, 0.975)),
                a_only=int((d > 0).sum()), b_only=int((d < 0).sum())))
    leo_cells = []
    for m in models:
        groups = defaultdict(list)
        unique = {p['job']['job_id']: p for p in read_jsonl(out / 'leo_rows.jsonl') if p['model'] == m}
        for p in unique.values():
            j, r = p['job'], p['row']
            groups[(j['route'], j['downlink_seconds'], j['num_agents'])].append(r)
        for (route, dl, n), rows in sorted(groups.items()):
            k = sum(bool(r['success']) for r in rows)
            lo, hi = wilson(k, len(rows))
            succ = [r for r in rows if r['success']]
            leo_cells.append(dict(part='leo', model=m, treatment=LEO_TREATMENT, route=route,
                downlink_added_ms=int(round(dl * 1000)), num_agents=n, episodes=len(rows), successes=k,
                success_rate=k / len(rows), success_ci95_low=lo, success_ci95_high=hi,
                makespan_seconds_success_mean=mean([r['makespan_seconds'] for r in succ]),
                sum_of_costs_seconds_success_mean=mean([r['mean_first_completion_time_seconds'] * n for r in succ
                    if r.get('mean_first_completion_time_seconds') is not None]),
                completion_fraction_mean=mean([r.get('ever_completion_fraction') for r in rows]),
                collisions_per_episode=mean([r.get('attempted_vertex_conflicts', 0) + r.get('attempted_edge_swaps', 0)
                    for r in rows]),
                post_resolution_conflicts=sum(r.get('post_resolution_conflicts', 0) for r in rows),
                collision_failures=sum(r.get('collision_failure', 0) for r in rows),
                cached_fallback_decisions_mean=mean([r.get('cached_fallback_decisions') for r in rows]),
                mean_sim_wall_seconds=mean([r.get('sim_wall_seconds') for r in rows]),
                peak_rss_gb_max=max(r.get('peak_rss_bytes', 0) for r in rows) / 1e9))
    failed = [t for t in tasks if t.get('status') != 'ok']
    result = dict(protocol=PROTOCOL, generated_unix=time.time(), manifest=meta, mapf_cells=cells,
        mapf_paired_success_differences=paired, leo_cells=leo_cells, failed_or_skipped_tasks=failed,
        tasks_finished=sum(t.get('status') == 'ok' for t in tasks), tasks_planned=meta.get('tasks_planned'),
        placement_failures=read_jsonl(out / 'placement_failures.jsonl'))
    write_atomic(out / 'results.json', json.dumps(result, indent=1))
    for name, rows in (('mapf_results.csv', cells), ('leo_results.csv', leo_cells), ('mapf_paired.csv', paired)):
        if rows:
            keys = list(rows[0])
            tmp = out / (name + '.tmp')
            with tmp.open('w', newline='') as stream:
                w = csv.DictWriter(stream, keys)
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, out / name)
    write_atomic(out / 'summary.md', markdown(result))
    return result


def _f(v, pct=False, digits=1):
    if v is None:
        return '-'
    return f'{100 * v:.{digits}f}%' if pct else f'{v:.{digits}f}'


def markdown(result):
    meta = result['manifest']
    budgets = meta['budgets']
    lines = ['# PANAMA MAPF final evaluation', '',
        f"Protocol `{PROTOCOL}`. Status: **{meta.get('status', 'running')}**. Tasks finished "
        f"{result['tasks_finished']}/{result['tasks_planned']}; failed or skipped {len(result['failed_or_skipped_tasks'])}.",
        '', '## Training budgets (NOT budget-matched)', '',
        '**This comparison is not budget-matched.** The curriculum model reached the 32-agent stage after '
        'its 16-agent stage hit the 3,000-episode cap without graduating (best validation 89 %, final 84 %), '
        f"and it then trained only **{budgets.get('curriculum', {}).get('episodes_at_32', '?')} episodes at 32 agents**, "
        f"against **{budgets.get('baseline32', {}).get('episodes_at_32', '?')} for the 32-agent baseline**. "
        'model8 was never trained at 32 agents (max 16) and came from a different protocol and map set. '
        'Totals differ as well, so differences below mix the effect of the curriculum with the effect of budget.', '',
        '| model | stages (N: episodes / joint steps) | total episodes | total joint steps | agent-steps (joint x N) | episodes at N=32 |',
        '|---|---|---|---|---|---|']
    for m, b in budgets.items():
        stages = ', '.join(f"{s['population']}: {s['episodes']:,} / {s['joint_steps'] or 0:,}" for s in b['stages'])
        lines.append(f"| {m} | {stages} | {b['total_episodes']:,} | {b['total_joint_steps'] or 0:,} | "
            f"{b['total_agent_steps_upper']:,} | {b['episodes_at_32']:,} |")
    lines += ['', '## Held-out paired MAPF maps', '',
        f"Random obstacle maps {meta['mapf']['map_size']} cells per side, density {meta['mapf']['density']}, "
        f"swapped start/goal pairs, map and case seeds disjoint from training and validation, case i uses map i "
        f"for every N and model. Planned {meta['mapf']['cases']} cases per (model, N). Greedy, {CAP}-step cap. "
        'Success CI: Wilson 95 %. Makespan and sum of costs (sum of first-arrival steps) are on successful '
        f'episodes; capped versions count failures as {CAP}. Collisions are attempted vertex conflicts plus edge '
        'swaps per episode (the executor resolves them; post-resolution overlaps are reported separately).', '',
        '| model | N | n | success (95% CI) | makespan | capped makespan | sum of costs | collisions/ep | overlaps |',
        '|---|---|---|---|---|---|---|---|---|']
    for c in sorted(result['mapf_cells'], key=lambda c: (c['num_agents'], c['model'])):
        lines.append(f"| {c['model']} | {c['num_agents']} | {c['episodes']} | {_f(c['success_rate'], True)} "
            f"({_f(c['success_ci95_low'], True)}-{_f(c['success_ci95_high'], True)}) | {_f(c['makespan_success_mean'])} | "
            f"{_f(c['capped_makespan_mean'])} | {_f(c['sum_of_costs_success_mean'])} | {_f(c['collisions_per_episode'], digits=2)} | "
            f"{c['post_resolution_overlaps']} |")
    if result['mapf_paired_success_differences']:
        lines += ['', 'Paired success differences (same cases; bootstrap 95 % CI over cases):', '',
            '| N | a - b | cases | diff | 95% CI | a-only / b-only successes |', '|---|---|---|---|---|---|']
        for p in result['mapf_paired_success_differences']:
            lines.append(f"| {p['num_agents']} | {p['a']} - {p['b']} | {p['cases']} | {_f(p['success_diff'], True)} | "
                f"{_f(p['ci95_low'], True)} to {_f(p['ci95_high'], True)} | {p['a_only']} / {p['b_only']} |")
    lines += ['', '## LEO subset (leo_study harness, A4 = mixed speed + cached fallback)', '',
        f"Physical LEO network, nominal and +150 ms added downlink, cloud and allocated routes, "
        f"N={'/'.join(map(str, meta['leo']['populations']))}, {meta['leo']['cases']} cases each on the study's fixed "
        '32x32 map (the same cases as leo_study, so model8 rows reproduce that study; this part is not a held-out map '
        'test). Makespan and sum of costs (N x mean first-completion time) in seconds, on successful episodes. '
        'Collisions are attempted vertex conflicts plus edge swaps per episode.', '',
        '| model | route | +DL ms | N | n | success (95% CI) | makespan s | sum of costs s | collisions/ep |',
        '|---|---|---|---|---|---|---|---|---|']
    for c in sorted(result['leo_cells'], key=lambda c: (c['route'], c['downlink_added_ms'], c['num_agents'], c['model'])):
        lines.append(f"| {c['model']} | {c['route']} | {c['downlink_added_ms']} | {c['num_agents']} | {c['episodes']} | "
            f"{_f(c['success_rate'], True)} ({_f(c['success_ci95_low'], True)}-{_f(c['success_ci95_high'], True)}) | "
            f"{_f(c['makespan_seconds_success_mean'])} | {_f(c['sum_of_costs_seconds_success_mean'])} | "
            f"{_f(c['collisions_per_episode'], digits=2)} |")
    if result['failed_or_skipped_tasks']:
        lines += ['', '## Failed or skipped units', '']
        for t in result['failed_or_skipped_tasks']:
            lines.append(f"- {t['task']['kind']} {t['task'].get('model')} {t['task'].get('job', {}).get('job_id') or t['task'].get('num_agents')} "
                f"{t['task'].get('cases', '')}: {t['status']} ({t.get('seconds', 0):.0f} s)")
    if result['placement_failures']:
        lines += ['', f"Placement failures (no paired placement after {PLACEMENT_RETRIES} seeds): "
            f"{len(result['placement_failures'])}"]
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------------------- orchestration
def plan(models, populations, cases, primary, chunk, leo_cases, leo_populations):
    """Priority order: MAPF cases [0, primary) round-robin, then LEO, then MAPF [primary, cases)."""
    def mapf_block(lo, hi):
        block = []
        for start in range(lo, hi, chunk):
            for n in sorted(populations, reverse=True):
                for m in models:
                    block.append(dict(kind='mapf', model=m, num_agents=n, cases=[start, min(hi, start + chunk)]))
        return block
    tasks = mapf_block(0, min(primary, cases))
    for case in range(leo_cases):
        for n in leo_populations:
            for route in LEO_ROUTES:
                for dl in LEO_DOWNLINKS:
                    for m in models:
                        tasks.append(dict(kind='leo', model=m, job=leo_job(route, dl, n, case)))
    tasks += mapf_block(min(primary, cases), cases)
    return tasks


def run(out, model_specs, populations, cases, primary, chunk, leo_cases, leo_populations, workers, max_wall,
        task_timeout, project_root):
    begin = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    models = {}
    for name, spec in model_specs.items():
        path = locate_model(spec, out)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        models[name] = dict(path=str(path), sha256=sha256_file(path), budget=budget_of(payload),
            weights_sha256=payload.get('model_weights_sha256'))
    spec, _ = mapf_setup(project_root)
    for i in range(cases):
        test_map(out, spec, i)
    tasks = plan(list(models), populations, cases, primary, chunk, leo_cases, leo_populations)
    meta = dict(protocol=PROTOCOL, status='running', started_unix=time.time(), max_wall_seconds=max_wall,
        workers=workers, models={m: {k: v for k, v in d.items() if k != 'budget'} for m, d in models.items()},
        budgets={m: d['budget'] for m, d in models.items()}, tasks_planned=len(tasks),
        mapf=dict(populations=list(populations), cases=cases, primary_cases=primary, chunk=chunk,
            map_size=list(spec.map_size), density=list(spec.density), pair_distance=list(spec.pair_distance),
            map_seed_offset=MAP_SEED_OFFSET, case_seed_offset=CASE_SEED_OFFSET, cap=CAP),
        leo=dict(treatment=LEO_TREATMENT, routes=list(LEO_ROUTES), downlinks=list(LEO_DOWNLINKS),
            populations=list(leo_populations), cases=leo_cases))
    write_atomic(out / 'eval_manifest.json', json.dumps(meta, indent=1))
    logs = scratch_dir(out) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    done_keys = {json.dumps(t['task'], sort_keys=True) for t in read_jsonl(out / 'tasks.jsonl') if t['status'] == 'ok'}
    queue = [t for t in tasks if json.dumps(t, sort_keys=True) not in done_keys]
    running = []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    counter = 0
    safe_summarize(out)
    status = 'complete'
    while queue or running:
        now = time.monotonic()
        if now - begin >= max_wall:
            status = 'wall_guard'
            break
        while queue and len(running) < workers:
            task = queue.pop(0)
            full = dict(task)
            full['model_path'] = models[task['model']]['path']
            full['model_sha256'] = models[task['model']]['sha256']
            counter += 1
            log = open(logs / f'task-{counter:04d}.log', 'w')
            proc = subprocess.Popen([sys.executable, '-m', 'panama_uav.mapf_eval', 'task', str(out), json.dumps(full),
                '--project-root', str(project_root)], stdout=log, stderr=subprocess.STDOUT, env=env,
                start_new_session=True)
            running.append((proc, task, time.monotonic(), log))
        time.sleep(1.0)
        still = []
        for proc, task, start, log in running:
            rc = proc.poll()
            age = time.monotonic() - start
            if rc is None and age < task_timeout:
                still.append((proc, task, start, log))
                continue
            if rc is None:
                os.killpg(proc.pid, 9)
                proc.wait()
                state = 'timeout'
            else:
                state = 'ok' if rc == 0 else ('killed_signal_9_likely_oom' if rc == -9 else f'exit:{rc}')
            log.close()
            append_jsonl(out / 'tasks.jsonl', dict(task=task, status=state, seconds=age, finished_unix=time.time()))
            print(f'{time.monotonic() - begin:8.0f}s {state} {json.dumps(task)}', flush=True)
            safe_summarize(out)
        running = still
    for proc, task, start, log in running:
        os.killpg(proc.pid, 9)
        proc.wait()
        log.close()
        append_jsonl(out / 'tasks.jsonl', dict(task=task, status='killed_at_wall_guard',
            seconds=time.monotonic() - start, finished_unix=time.time()))
    for task in queue:
        append_jsonl(out / 'tasks.jsonl', dict(task=task, status='not_started_wall_guard', seconds=0.0))
    meta['status'] = status
    meta['wall_seconds'] = time.monotonic() - begin
    write_atomic(out / 'eval_manifest.json', json.dumps(meta, indent=1))
    result = summarize(out)
    scratch = scratch_dir(out)
    if scratch.exists() and scratch.resolve().parent != out.resolve():
        subprocess.run(['tar', '-cf', str(out / 'scratch.tar.tmp'), '-C', str(scratch.parent), scratch.name])
        os.replace(out / 'scratch.tar.tmp', out / 'scratch.tar')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('out', type=Path)
    r.add_argument('--model', action='append', default=[], help='name=path | name=search:<run> | name=search-file:<file>')
    r.add_argument('--populations', type=int, nargs='+', default=[2, 4, 8, 16, 32])
    r.add_argument('--cases', type=int, default=300)
    r.add_argument('--primary-cases', type=int, default=200)
    r.add_argument('--chunk', type=int, default=25)
    r.add_argument('--leo-cases', type=int, default=10)
    r.add_argument('--leo-populations', type=int, nargs='*', default=[8, 16, 32])
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--max-wall-seconds', type=float, default=36000.0)
    r.add_argument('--task-timeout', type=float, default=5400.0)
    r.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    t = sub.add_parser('task')
    t.add_argument('out', type=Path)
    t.add_argument('task')
    t.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    s = sub.add_parser('summarize')
    s.add_argument('out', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'task':
        task = json.loads(args.task)
        (task_mapf if task['kind'] == 'mapf' else task_leo)(args.out, task, args.project_root)
        return
    if args.command == 'summarize':
        summarize(args.out)
        return
    if args.max_wall_seconds > 37800:
        parser.error('wall guard must be <= 10.5 h (Kaggle hard limit 11 h)')
    specs = dict(item.split('=', 1) for item in args.model) or {
        'curriculum': 'search:mapf-cur', 'baseline32': 'search:mapf-base', 'model8': 'search-file:model8.pt'}
    result = run(args.out, specs, args.populations, args.cases, args.primary_cases, args.chunk, args.leo_cases,
        args.leo_populations, args.workers, args.max_wall_seconds, args.task_timeout, args.project_root)
    print(json.dumps(dict(status=result['manifest'].get('status'), finished=result['tasks_finished'],
        planned=result['tasks_planned'], failed=len(result['failed_or_skipped_tasks'])), indent=1))


if __name__ == '__main__':
    main()
