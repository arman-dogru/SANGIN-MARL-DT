"""Versioned clean-only Phase 1 correction, paired ablations and strict final gate.

V1 is retained verbatim apart from an optional dataset exclusion argument. V2
uses remote execution kernels; no network/allocation services enter learning.
"""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import tomllib

import numpy as np
import torch

from .config import COMMUNICATION_SCHEMA, ExperimentMode, ProjectConfig, load_config
from .learning import _learn
from .models import choose_device, cpu_state_dict, select_actions
from .phase1 import clean_config, hardware, summarize_cases, weights_hash
from .phase1_data import FAMILIES, case_for, prepare_dataset, scenario_for, verify_dataset
from .phase1_environment import epoch_metrics
from .phase1_execution import RemoteAlignedCleanMAPFEnvironment
from .records import append_jsonl, sha256, write_json
from .scenarios import load_map
from .training import TrainingSession, atomic_save

PROTOCOL = 'clean_mapf_phase1_v2'
METRICS = 'clean_mapf_metrics_v2'
COUNTS = (2, 4, 6, 8)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    path = Path(path)
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def load_protocol(path):
    with Path(path).open('rb') as stream:
        spec = tomllib.load(stream)
    if spec['protocol'] != PROTOCOL:
        raise ValueError('Revised training requires the V2 protocol')
    if not spec['variants'] or len({v['name'] for v in spec['variants']}) != len(spec['variants']):
        raise ValueError('Unique nonempty variant names required')
    if len(spec['ablation']['seeds']) < 2:
        raise ValueError('Ablation selection requires at least two paired learning seeds')
    for variant in spec['variants']:
        if set(variant) != {'name', 'learning_rate', 'gamma', 'epsilon_decay_steps'}:
            raise ValueError('Only declared learning alternatives may differ in an ablation')
        if not 0 < variant['gamma'] < 1 or variant['learning_rate'] <= 0 or variant['epsilon_decay_steps'] <= 0:
            raise ValueError('Invalid DQN alternative')
    return spec


def variant_config(base, variant, budget):
    return clean_config(replace(base, training=replace(base.training,
        learning_rate=variant['learning_rate'], gamma=variant['gamma'],
        epsilon_decay_steps=variant['epsilon_decay_steps'], budget_seconds=budget,
        checkpoint_interval_steps=min(base.training.checkpoint_interval_steps, 1000 if budget <= 600 else 5000))))


def reconcile_logs(root, offsets):
    """Roll back worker-owned logs, including files absent at the checkpoint.

    Monitoring records are not worker logs and must never be truncated here.
    """
    root = Path(root)
    owned = {'learning_steps.jsonl', 'training.jsonl', 'validation.jsonl',
             'final_validation_cases.jsonl', 'final_test_cases.jsonl'}
    owned.update(p.name for p in root.glob('validation-*.jsonl'))
    for name in owned | set(offsets):
        if Path(name).name != name or not name.endswith('.jsonl'):
            raise ValueError('Invalid checkpoint log path')
        if name == 'monitor_history.jsonl':
            continue
        path = root/name
        size = offsets.get(name, 0)
        if size < 0 or (not path.exists() and size):
            raise ValueError('Checkpoint log is missing or invalid')
        if path.exists():
            if path.stat().st_size < size:
                raise ValueError('Checkpoint log is shorter than its committed offset')
            with path.open('r+b') as stream:
                stream.truncate(size)


def resume_elapsed(root, checkpoint_elapsed, started_unix):
    """Conservatively charge an unclosed worker lease, including downtime.

    A hard crash cannot refund work since the last model checkpoint. Charging
    until the next attempt may sacrifice idle time, but cannot extend budget.
    Graceful interruptions close the lease and do not charge subsequent idle.
    """
    root = Path(root)
    elapsed = checkpoint_elapsed
    if (root/'status.json').exists():
        elapsed = max(elapsed, read_json(root/'status.json').get('wall_seconds', 0))
    if (root/'budget_lease.json').exists():
        lease = read_json(root/'budget_lease.json')
        elapsed = max(elapsed, lease.get('charged_seconds', 0))
        if lease.get('active'):
            elapsed = max(elapsed, lease['previous_elapsed'] + max(0, started_unix-lease['started_unix']))
    return elapsed


def prepare_data(path, spec, historical):
    previous = read_json(historical)
    # The entire old bank is excluded, including historical validation/test.
    excluded = {m['canonical'] for m in previous['maps']}
    excluded.update(previous['reserved_stress_maps'])
    dataset = prepare_dataset(path, seed=spec['dataset_seed'],
        train_maps=spec['train_maps_per_family'], eval_maps=spec['evaluation_maps_per_family'],
        eval_cases=spec['cases_per_family'], excluded_canonicals=excluded)
    dataset.update(schema='clean_mapf_split_v2', protocol=PROTOCOL,
        historical_dataset_sha256=sha256(historical), excluded_historical_maps=sorted(excluded),
        final_test_policy='new split; no ablation evaluation or model selection on test')
    if {m['canonical'] for m in dataset['maps']} & excluded:
        raise ValueError('Historical map leakage')
    write_json(Path(path)/'dataset.json', dataset)
    return dataset


def validate_data(dataset):
    verify_dataset(dataset)
    if dataset['schema'] != 'clean_mapf_split_v2' or dataset['protocol'] != PROTOCOL:
        raise ValueError('V2 requires a new, untouched test split')
    if {m['canonical'] for m in dataset['maps']} & set(dataset['excluded_historical_maps']):
        raise ValueError('Historical map leakage')


def balanced_population(episode_index):
    """A qualification failure can never prevent exposure to any population."""
    return COUNTS[episode_index % len(COUNTS)]


class RevisedSession(TrainingSession):
    def __init__(self, config, dataset, seed, run_id, device):
        self.dataset = dataset
        self.population_episodes = Counter()
        self.population_steps = Counter()
        self.case_number = 0
        self.current_case = None
        super().__init__(clean_config(config), 'clean', seed, run_id, device)

    def start_episode(self):
        count = balanced_population(self.episodes)
        local = self.population_episodes[count]
        self.stage_index = COUNTS.index(count)
        self.stage_episodes = local
        family = FAMILIES[local % len(FAMILIES)]
        options = [m for m in self.dataset['maps'] if m['split'] == 'train' and m['family'] == family]
        episode_seed = self.seed + 30_000_000 + count * 1_000_000 + local
        rng = np.random.default_rng(episode_seed)
        record = options[int(rng.integers(len(options)))]
        self.current_case = case_for(load_map(record['path']), record, count, episode_seed, coordinated=local % 2 == 0)
        self.environment = RemoteAlignedCleanMAPFEnvironment(self.config, ExperimentMode.UAV,
            record['path'], count, 250, episode_seed, scenario_for(self.current_case))
        self.observations, _ = self.environment.reset()
        self.action_rng = np.random.default_rng(episode_seed)
        if not self.actor_models:
            self.actor_models[0] = self._actor(self.latest_published_weights)
        version = max(self.actor_models)
        self.deployed_versions = {a: version for a in self.observations}
        self.episode_reward = 0.0
        self.epoch_totals = Counter()

    def step(self):
        cfg = self.config.training
        epsilon = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-self.generated_steps/cfg.epsilon_decay_steps)
        active = {a.id for a in self.environment.agents if not a.ever_reached_goal}
        proposed = select_actions(self.actor_models, self.deployed_versions,
            {a: self.observations[a] for a in active}, epsilon, self.action_rng, self.device)
        actions = self.environment.prepare_actions(proposed).executed
        measures = epoch_metrics(self.environment, actions)
        before = {a.id: a.position for a in self.environment.agents}
        result = self.environment.step(proposed)
        self.generated_steps += 1
        self.population_steps[self.environment.num_agents] += 1
        done = result.terminated or result.truncated
        # A robot's learned role ends at first completion. Local-controller
        # actions after completion must not be mislabeled as DQN experience.
        for robot in sorted(active):
            terminal = done or self.environment.agents[robot].ever_reached_goal
            self.replay.add((self.observations[robot], actions[robot], result.rewards[robot],
                result.observations[robot], terminal), float(self.generated_steps))
            self.delivered_experiences += 1
        self.last_loss = _learn(self.policy, self.target, self.optimizer, self.replay,
            cfg.batch_size, cfg.gamma, cfg.tau, self.device)
        self.learning_opportunities += 1
        if self.last_loss is not None:
            self.learner_updates += 1
        if self.generated_steps % cfg.model_broadcast_steps == 0:
            self.published_version += 1
            self.model_publications += 1
            self.latest_published_weights = {k: v.numpy() for k, v in cpu_state_dict(self.policy).items()}
            self.actor_models = {self.published_version: self._actor(self.latest_published_weights)}
            self.deployed_versions = {a: self.published_version for a in self.observations}
        self.observations = result.observations
        self.episode_reward += sum(result.rewards.values())
        measures['blocked_move_steps'] = sum(actions[a.id] != 4 and a.position == before[a.id]
            for a in self.environment.agents if a.id in active)
        self.epoch_totals.update(measures)
        self.environment.communication.transport.log.clear()
        row = dict(protocol=PROTOCOL, metrics=METRICS, generated_joint_steps=self.generated_steps,
            learner_updates=self.learner_updates, num_agents=self.environment.num_agents,
            case_id=self.current_case['case_id'], family=self.current_case['family'],
            loss=self.last_loss, epsilon=epsilon, replay_size=len(self.replay),
            active_replay_transitions=len(active), communication_delay_seconds=0.0, **measures)
        if self.generated_steps % 256 == 0:
            values = {key: torch.as_tensor(np.stack([o[key] for o in self.observations.values()]),
                device=self.device) for key in ('spatial', 'vector')}
            with torch.no_grad():
                q = self.policy(values)
                row.update(q_abs_mean=float(q.abs().mean()), q_abs_max=float(q.abs().max()),
                    q_finite=bool(torch.isfinite(q).all()))
            row['gradient_l2_after_clip'] = math.sqrt(sum(float(p.grad.detach().square().sum())
                for p in self.policy.parameters() if p.grad is not None))
        episode = None
        if done:
            self.population_episodes[self.environment.num_agents] += 1
            self.stage_episodes = self.population_episodes[self.environment.num_agents]
            self.episodes += 1
            episode = dict(self.current_case, protocol=PROTOCOL, metrics=METRICS,
                steps=self.environment.step_count, success=result.terminated, episodes=self.episodes,
                generated_joint_steps=self.generated_steps, reward=self.episode_reward,
                epsilon=epsilon, **self.epoch_totals)
        return row, episode


def subset(bank, cases_per_family):
    """Matched stratified prefix: designed interactions and random endpoints."""
    size = max(c['evaluation_index'] for c in bank)+1
    if not 0 < cases_per_family <= size:
        raise ValueError('Validation subset must fit each complete family bank')
    designed = (cases_per_family+1)//2
    random_count = cases_per_family-designed
    indices = set(range(designed)) | set(range(size//2,size//2+random_count))
    return sorted([c for c in bank if c['evaluation_index'] in indices],
                  key=lambda c: (c['evaluation_index'], COUNTS.index(c['num_agents']), FAMILIES.index(c['family'])))


def evaluate(config, model, bank, device, path, tag, deadline, stop=lambda: False, *, diagnostics=False):
    identity = weights_hash(model)
    existing = read_rows(path)
    if any(r['tag'] != tag or r['model_weights_sha256'] != identity for r in existing):
        raise ValueError('Evaluation cannot mix models or metric versions')
    if len({r['case_id'] for r in existing}) != len(existing):
        raise ValueError('Duplicate evaluation cases')
    seen = {r['case_id'] for r in existing}
    for case in bank:
        if case['case_id'] in seen:
            continue
        if time.monotonic() >= deadline or stop():
            break
        env = RemoteAlignedCleanMAPFEnvironment(config, ExperimentMode.UAV,
            case['map_path'], case['num_agents'], 250, case['seed'], scenario_for(case))
        obs, _ = env.reset()
        totals = Counter()
        if diagnostics:
            from .phase1_cycles import CycleTracker
            tracker=CycleTracker()
        while time.monotonic() < deadline and not stop():
            active = {a.id for a in env.agents if not a.ever_reached_goal}
            proposed = select_actions({0: model}, {a: 0 for a in active},
                {a: obs[a] for a in active}, 0.0, np.random.default_rng(0), device)
            actions = env.prepare_actions(proposed).executed
            if diagnostics:
                tracker.observe(env,obs)
            totals.update(epoch_metrics(env, actions))
            before = {a.id: a.position for a in env.agents}
            if diagnostics:
                physical_before={a.id:(a.position,a.ever_reached_goal) for a in env.agents}
            result = env.step(proposed)
            if diagnostics:
                tracker.moved(physical_before,env,active)
            totals['blocked_move_steps'] += sum(actions[a.id] != 4 and a.position == before[a.id]
                for a in env.agents if a.id in active)
            obs = result.observations
            env.communication.transport.log.clear()
            if result.terminated or result.truncated:
                row = dict(case, tag=tag, protocol=PROTOCOL, metrics=METRICS,
                    model_weights_sha256=identity, steps=env.step_count, success=result.terminated,
                    final_positions=[list(a.position) for a in env.agents], **totals)
                if diagnostics:
                    row.update(tracker.record())
                    row.update(ever_completed_robots=sum(a.ever_reached_goal for a in env.agents),
                        all_visited_without_completion=all(a.ever_reached_goal for a in env.agents) and not result.terminated)
                append_jsonl(path, row)
                existing.append(row)
                break
        else:
            break
    return existing


def summarize(records, expected=200):
    result = summarize_cases(records, expected)
    result.update(metrics=METRICS, blocked_move_steps=sum(r.get('blocked_move_steps', 0) for r in records))
    return result


def payload(session, manifest, validation=(), test=(), compatibility=None, ablation=False):
    spec = manifest['spec']
    exposure = all(session.population_episodes[n] >= spec['full']['minimum_episodes_per_population'] for n in COUNTS)
    qualified = (not ablation and exposure and len(validation) == len(test) == 4
        and all(r['passed'] for r in [*validation, *test])
        and compatibility is not None and compatibility.get('status') == 'passed')
    return dict(communication_schema=COMMUNICATION_SCHEMA, policy_family='panama_0.1_task',
        training_protocol=PROTOCOL, metric_version=METRICS, condition='clean',
        task_reward=session.config.to_dict()['reward'], model_state_dict=cpu_state_dict(session.policy),
        spatial_shape=(3, 15, 15), vector_length=session.config.vector_length,
        dataset_sha256=manifest['dataset_sha256'], run_id=session.run_id, seed=session.seed,
        learner_updates=session.learner_updates, generated_joint_steps=session.generated_steps,
        population_episodes=dict(session.population_episodes), population_steps=dict(session.population_steps),
        minimum_episodes_per_population=spec['full']['minimum_episodes_per_population'],
        curriculum_completed=exposure, qualification=list(validation), held_out=list(test),
        phase1_qualified=qualified, smoke=ablation, ablation=ablation, compatibility=compatibility,
        model_weights_sha256=weights_hash(session.policy), initialization='new deterministic weights',
        execution_contract='fresh remote-DT physical resolver, priorities and completed local control',
        observation_contract='unchanged remote_observation with instantaneous fresh state',
        training_run_protocol=spec['full'].get('extended_protocol',PROTOCOL),
        source_commit=manifest.get('source_commit'),budget_plan_sha256=manifest.get('budget_plan_sha256'),
        configured_budget_seconds=session.config.training.budget_seconds)


def verify_payload(data):
    from .phase1 import verify_payload as strict_v1
    if data.get('training_protocol') != PROTOCOL or data.get('metric_version') != METRICS:
        return False
    if data.get('minimum_episodes_per_population') != 500 or data.get('ablation'):
        return False
    if any(int(data.get('population_episodes', {}).get(n, 0)) < 500 for n in COUNTS):
        return False
    compatibility = data.get('compatibility') or {}
    if compatibility.get('execution_contract') != 'remote-aligned-v2':
        return False
    return strict_v1(dict(data, training_protocol='clean_mapf_phase1_v1'))


def report_run(root, state):
    fields = []
    for split, key in [('validation', 'qualification'), ('test', 'held_out')]:
        for item in state.get(key, []):
            for family, group in [('all', item), *item.get('per_family', {}).items()]:
                fields.append(dict(split=split, num_agents=item['num_agents'], family=family,
                    **{k: v for k, v in group.items() if k not in ('per_family', 'num_agents')}))
    if fields:
        names = list(dict.fromkeys(k for r in fields for k in r))
        with (root/'per_population_and_scenario.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader(); writer.writerows(fields)
    lines = ['# Revised Phase 1', '', f"Status: **{state['status']}**. Qualified: **{state['qualified']}**.", '',
        f"Runtime: {state['wall_seconds']:.2f} s; learning steps: {state['generated_joint_steps']}; "
        f"episodes by population: {state['population_episodes']}.", '',
        'Ablations cannot qualify and never evaluate the untouched test bank. All claims below refer to '
        'the recorded frozen current weights, not earlier curriculum results.', '',
        '| Split | Robots | Completed | Success | Wilson 95% CI | Passed |',
        '| --- | ---: | ---: | ---: | --- | --- |']
    for split, key in [('validation', 'qualification'), ('test', 'held_out')]:
        for r in state.get(key, []):
            lo, hi = r['success_ci95']
            ci = f'{lo:.1%}–{hi:.1%}' if lo is not None else 'unavailable'
            success = f"{r['success_rate']:.1%}" if r['success_rate'] is not None else 'unavailable'
            lines.append(f"| {split} | {r['num_agents']} | {r['episodes']} | {success} | {ci} | {r['passed']} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run_one(root, manifest, device_name='cuda', resume=False):
    root = Path(root)
    start = time.monotonic()
    started_unix = time.time()
    spec = manifest['spec']
    ablation = manifest['mode'] == 'ablation'
    config = ProjectConfig.from_dict(manifest['configuration'])
    dataset = read_json(manifest['dataset_path'])
    validate_data(dataset)
    if sha256(manifest['dataset_path']) != manifest['dataset_sha256']:
        raise ValueError('Dataset identity changed')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = choose_device(device_name)
    checkpoint = root/'checkpoint.pt'
    if checkpoint.exists():
        if not resume:
            raise ValueError('Explicit resume required; never overwrite weights')
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['manifest_sha256'] != sha256(root/'manifest.json'):
            raise ValueError('Resume manifest changed')
        session = TrainingSession.restore(saved['session'], config, 'clean', manifest['seed'], root.name, device)
        state = saved['state']
        if state['mode'] == 'complete':
            return read_json(root/'status.json')
        reconcile_logs(root, saved['log_offsets'])
    else:
        session = RevisedSession(config, dataset, manifest['seed'], root.name, device)
        state = dict(mode='train', elapsed_seconds=manifest.get('setup_seconds', 0),
            qualification=[], held_out=[], last_validation_steps=0, compatibility=None)
    previous = resume_elapsed(root, state['elapsed_seconds'], started_unix)
    budget = config.training.budget_seconds
    end = start + max(0, budget-previous)
    settings = spec['ablation'] if ablation else spec['full']
    reserve = settings['evaluation_reserve_seconds']
    shutdown = 5 if ablation else settings['shutdown_reserve_seconds']
    train_end = end-reserve-shutdown
    write_json(root/'budget_lease.json',dict(active=True,previous_elapsed=previous,
        started_unix=started_unix,charged_seconds=previous))
    requested = {'stop': False}
    old = signal.signal(signal.SIGINT, lambda *_: requested.update(stop=True))
    def elapsed():
        return previous+time.monotonic()-start
    def status(reason, qualified=False):
        row = dict(protocol=PROTOCOL, metrics=METRICS, status=reason, mode=state['mode'], qualified=qualified,
            wall_seconds=elapsed(), budget_seconds=budget, generated_joint_steps=session.generated_steps,
            learner_updates=session.learner_updates, episodes=session.episodes,
            population_episodes=dict(session.population_episodes), population_steps=dict(session.population_steps),
            qualification=state['qualification'], held_out=state['held_out'], hardware=hardware(device),
            ablation=ablation, variant=manifest['variant'], seed=manifest['seed'])
        if settings.get('extended_protocol'):
            row.update(training_run_protocol=settings['extended_protocol'],
                termination_reason=state.get('termination_reason'),
                held_out_skipped_reason=state.get('held_out_skipped_reason'))
        write_json(root/'status.json', row)
        return row
    def save():
        state['elapsed_seconds'] = elapsed()
        offsets = {p.name: p.stat().st_size for p in root.glob('*.jsonl')}
        atomic_save(dict(session=session.checkpoint(), state=state, log_offsets=offsets,
            manifest_sha256=sha256(root/'manifest.json')), checkpoint)
        atomic_save(payload(session, manifest, state['qualification'], state['held_out'],
            state['compatibility'], ablation), root/'latest_model.pt')
    try:
        status('training'); save()
        target = settings['joint_steps'] if ablation else config.training.target_environment_steps
        while state['mode'] == 'train' and session.generated_steps < target and time.monotonic() < train_end and not requested['stop']:
            if session.environment.all_agents_at_goals or session.environment.step_count >= 250:
                session.start_episode()
            record, episode = session.step()
            record['wall_seconds'] = elapsed()
            append_jsonl(root/'learning_steps.jsonl', record)
            if episode:
                episode['wall_seconds'] = elapsed()
                append_jsonl(root/'training.jsonl', episode)
            if session.generated_steps % 128 == 0:
                status('training')
            if session.generated_steps % config.training.checkpoint_interval_steps == 0:
                save()
            if not ablation and session.generated_steps-state['last_validation_steps'] >= settings['validation_interval_steps']:
                tag = f'validation-{session.generated_steps}'
                records = evaluate(config, session.policy, subset(dataset['banks']['validation'],
                    settings['validation_cases_per_family']), device, root/f'{tag}.jsonl', tag,
                    train_end, lambda: requested['stop'],diagnostics=bool(settings.get('extended_protocol')))
                summaries = [dict(num_agents=n, **summarize([r for r in records if r['num_agents'] == n],
                    5*settings['validation_cases_per_family'])) for n in COUNTS]
                validation_record=dict(generated_joint_steps=session.generated_steps,
                    model_weights_sha256=weights_hash(session.policy), populations=summaries)
                append_jsonl(root/'validation.jsonl', validation_record)
                state['last_validation_steps'] = session.generated_steps
                # Validation reports learning progress; it neither gates exposure
                # nor chooses weights using the untouched test set.
                if settings.get('extended_protocol'):
                    from .phase1_stopping import training_stop
                    history=state.setdefault('validation_history',[])
                    history.append(validation_record)
                    decision=training_stop(history,session.population_episodes,
                        patience=settings['plateau_patience'],min_delta=settings['plateau_min_delta'],
                        min_steps=settings['plateau_min_steps'])
                    if decision:
                        state['termination_reason']=decision
                        save()
                        break
                save()
        if requested['stop']:
            result = status('interrupted_partial'); save(); return result
        state.setdefault('termination_reason','training_budget_or_step_cap')
        state['mode'] = 'final'; save(); status('final_evaluation')
        if not ablation and state['compatibility'] is None and time.monotonic() < end-shutdown:
            from .phase1_execution import verify_execution_compatibility
            state['compatibility'] = verify_execution_compatibility(config, session.policy,
                dataset['banks']['validation'], device, root)
            save()
        bank = subset(dataset['banks']['validation'], settings['validation_cases_per_family']) if ablation else dataset['banks']['validation']
        gate_test=bool(settings.get('gate_test_on_validation')) and not ablation
        splits = [('validation', bank)] if ablation or gate_test else [('validation', bank), ('test', dataset['banks']['test'])]
        ordered = {split: sorted(bank, key=lambda c:(c['evaluation_index'],
            COUNTS.index(c['num_agents']), FAMILIES.index(c['family']))) for split,bank in splits}
        for batch in range(0, max(map(len,ordered.values())), 20):
            for split, bank in ordered.items():
                prefix = bank[:batch+20]
                records = evaluate(config, session.policy, prefix,
                    device, root/f'final_{split}_cases.jsonl', f'final-{split}', end-shutdown,
                    lambda: requested['stop'],diagnostics=bool(settings.get('extended_protocol')))
                expected = settings['validation_cases_per_family']*5 if ablation else 200
                result = [dict(num_agents=n, **summarize([r for r in records if r['num_agents'] == n], expected)) for n in COUNTS]
                state['qualification' if split == 'validation' else 'held_out'] = result
            if time.monotonic() >= end-shutdown or requested['stop']:
                break
        if gate_test:
            from .phase1_stopping import may_open_heldout
            if may_open_heldout(state['qualification'],session.population_episodes,state['compatibility']):
                test_bank=sorted(dataset['banks']['test'],key=lambda c:(c['evaluation_index'],
                    COUNTS.index(c['num_agents']),FAMILIES.index(c['family'])))
                for batch in range(0,len(test_bank),20):
                    records=evaluate(config,session.policy,test_bank[:batch+20],device,
                        root/'final_test_cases.jsonl','final-test',end-shutdown,
                        lambda:requested['stop'],diagnostics=True)
                    state['held_out']=[dict(num_agents=n,**summarize([r for r in records if r['num_agents']==n],200)) for n in COUNTS]
                    if time.monotonic()>=end-shutdown or requested['stop']:
                        break
            else:
                state['held_out_skipped_reason']='coverage, complete strict final validation or execution compatibility did not pass; test remains untouched'
        save()
        if requested['stop']:
            result = status('interrupted_partial'); save(); return result
        final = payload(session, manifest, state['qualification'], state['held_out'], state['compatibility'], ablation)
        atomic_save(final, root/'final_model.pt')
        required = state['qualification'] if ablation else state['qualification']+state['held_out']
        complete = len(required) == (4 if ablation else 8) and all(r['complete'] for r in required)
        if gate_test and state.get('held_out_skipped_reason') and len(state['qualification'])==4 and all(r['complete'] for r in state['qualification']):
            complete=True
        if complete:
            state['mode'] = 'complete'
            reason = 'ablation_complete' if ablation else ('qualified' if final['phase1_qualified'] else 'not_qualified')
        else:
            reason = 'ablation_partial' if ablation else 'not_qualified_partial'
        result = status(reason, final['phase1_qualified'])
        save(); report_run(root, result)
        return result
    except Exception as exc:
        result = status('error')
        result['error'] = f'{type(exc).__name__}: {exc}'
        write_json(root/'status.json', result)
        save()
        raise
    finally:
        write_json(root/'budget_lease.json',dict(active=False,previous_elapsed=previous,
            started_unix=started_unix,charged_seconds=elapsed()))
        signal.signal(signal.SIGINT, old)


def paired_differences(candidate, baseline, samples=4000):
    """Hierarchical paired bootstrap: seed blocks, then shared map blocks."""
    def index(items):
        return {(r['training_seed'], r['case_id']): r for r in items}
    a, b = index(candidate), index(baseline)
    if set(a) != set(b) or len(a) != len(candidate) or len(b) != len(baseline):
        raise ValueError('Paired ablations require identical unique cases and seeds')
    for key in a:
        if any(a[key][field] != b[key][field] for field in ('map_canonical','starts','goals','seed','num_agents')):
            raise ValueError('Paired-case identity differs')
    seeds = sorted({k[0] for k in a})
    maps = sorted({r['map_id'] for r in a.values()})
    differences = np.array([[np.sum([float(a[k]['success'])-float(b[k]['success']) for k in a
        if k[0] == seed and a[k]['map_id'] == map_id]) for map_id in maps] for seed in seeds])
    sizes = np.array([[sum(k[0] == seed and a[k]['map_id'] == map_id for k in a)
        for map_id in maps] for seed in seeds])
    rng = np.random.default_rng(42)
    seed_indices = rng.integers(0, len(seeds), (samples, len(seeds)))
    map_indices = rng.integers(0, len(maps), (samples, len(maps)))
    values = np.array([differences[s[:,None], m[None,:]].sum()/sizes[s[:,None],m[None,:]].sum()
        for s,m in zip(seed_indices,map_indices)])
    return dict(mean=float(differences.sum()/sizes.sum()), ci95=np.quantile(values,[.025,.975]).tolist(),
        method='paired seed-and-map block bootstrap', seed_blocks=len(seeds), map_blocks=len(maps))


def compare_ablations(study):
    study = Path(study)
    setup = read_json(study/'study.json')
    spec = setup['spec']
    dataset = read_json(setup['dataset_path'])
    bank = subset(dataset['banks']['validation'], spec['ablation']['validation_cases_per_family'])
    expected = {r['case_id'] for r in bank}
    records = {}
    summary = []
    resources = []
    for variant in spec['variants']:
        name = variant['name']
        cohort = []
        for seed in spec['ablation']['seeds']:
            folder = study/'ablations'/f'{name}-seed{seed}'
            state = read_json(folder/'status.json')
            if state['status'] != 'ablation_complete' or state['generated_joint_steps'] != spec['ablation']['joint_steps']:
                raise ValueError('Ablation incomplete or unequal learning-step exposure; no selection allowed')
            if [r['num_agents'] for r in state['qualification']] != list(COUNTS):
                raise ValueError('All four population summaries required')
            if any(r['episodes'] != 5*spec['ablation']['validation_cases_per_family'] for r in state['qualification']):
                raise ValueError('Incomplete paired validation cohort')
            saved = read_rows(folder/'final_validation_cases.jsonl')
            if len(saved) != len(expected) or {r['case_id'] for r in saved} != expected:
                raise ValueError('Missing, duplicate or unexpected paired validation cases')
            if len({r['model_weights_sha256'] for r in saved}) != 1:
                raise ValueError('Validation weights changed')
            resources.append(dict(variant=name,seed=seed,wall_seconds=state['wall_seconds'],
                learner_updates=state['learner_updates'],generated_joint_steps=state['generated_joint_steps'],
                episodes=state['episodes'],population_episodes=json.dumps(state['population_episodes']),
                population_steps=json.dumps(state['population_steps']),gpu=state['hardware']['gpu']))
            for r in saved:
                cohort.append(dict(r, training_seed=seed, variant=name))
        records[name] = cohort
        for n in COUNTS:
            values = [r for r in cohort if r['num_agents'] == n]
            for family in ('all',*FAMILIES):
                group = values if family=='all' else [r for r in values if r['family']==family]
                summary.append(dict(variant=name,num_agents=n,family=family,episodes=len(group),
                    success_rate=float(np.mean([r['success'] for r in group])),
                    mean_capped_makespan=float(np.mean([r['steps'] for r in group])),
                    waiting_steps=float(np.mean([r['waiting_steps'] for r in group])),
                    relevant_pairs=sum(r['relevant_pairs'] for r in group),
                    directed_pairs=sum(r['directed_pairs'] for r in group),
                    vertex_conflicts=sum(r['vertex_conflicts'] for r in group),
                    edge_conflicts=sum(r['edge_conflicts'] for r in group)))
    baseline = records['original_hp']
    comparisons = {name: paired_differences(values, baseline) for name,values in records.items()}
    best = max(records, key=lambda name: np.mean([r['success'] for r in records[name]]))
    evidence = comparisons[best]
    gains = [np.mean([r['success'] for r in records[best] if r['num_agents']==n])-
             np.mean([r['success'] for r in baseline if r['num_agents']==n]) for n in COUNTS]
    # This is a launch-justification criterion, never a qualification threshold.
    comparative = best != 'original_hp' and evidence['mean'] >= spec['ablation']['minimum_success_gain'] and evidence['ci95'][0] > 0 and min(gains) >= -0.05
    rates = [np.mean([r['success'] for r in records[best] if r['num_agents']==n]) for n in COUNTS]
    strong_absolute = min(rates) >= .90
    justified = comparative or strong_absolute
    decision = dict(protocol=PROTOCOL, selected=best, full_run_justified=bool(justified),
        criterion='>=5pp paired overall gain, positive lower paired CI, no population loses >5pp; OR >=90% validation at every population',
        comparative_evidence=bool(comparative), strong_absolute_evidence=bool(strong_absolute),
        comparisons=comparisons, population_gain=dict(zip(COUNTS, map(float,gains))),
        limitation='Two learning seeds and short validation ablations do not establish qualification; no test cases evaluated.')
    write_json(study/'ablation_comparison.json', decision)
    with (study/'ablation_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    with (study/'ablation_resources.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(resources[0]));writer.writeheader();writer.writerows(resources)
    plot_comparison(study,summary,spec)
    return decision


def plot_comparison(study,summary,spec):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for variant in spec['variants']:
        name=variant['name']
        groups=[next(r for r in summary if r['variant']==name and r['num_agents']==n and r['family']=='all') for n in COUNTS]
        for axis,field in zip(axes,('success_rate','mean_capped_makespan')):
            axis.plot(COUNTS,[r[field] for r in groups],marker='o',label=name)
    axes[0].set(ylabel='Greedy validation success',ylim=(0,1.02))
    axes[1].set(ylabel='Mean episode steps (250-step cap)')
    for axis in axes:
        axis.set(xlabel='Robots',xticks=COUNTS);axis.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('V2 short paired ablations — validation only; no qualification claim')
    fig.savefig(study/'ablation_comparison.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for variant in spec['variants']:
        for seed in spec['ablation']['seeds']:
            values=read_rows(study/'ablations'/f"{variant['name']}-seed{seed}"/'training.jsonl')
            if not values:
                continue
            window=min(10,len(values))
            x=[r['generated_joint_steps'] for r in values][window-1:]
            for axis,field in zip(axes,('success','steps')):
                y=np.convolve([float(r[field]) for r in values],np.ones(window)/window,mode='valid')
                axis.plot(x,y,label=f"{variant['name']} seed{seed}",alpha=.8)
    axes[0].set(ylabel='Rolling exploratory training success',ylim=(0,1.02))
    axes[1].set(ylabel='Rolling episode length')
    for axis in axes:
        axis.set(xlabel='Joint learning steps');axis.grid(alpha=.2)
    axes[1].legend(fontsize=6)
    fig.savefig(study/'ablation_training_curves.png',dpi=160)
    plt.close(fig)


def worker(study, mode, device='cuda', resume=False):
    study = Path(study).resolve()
    setup = read_json(study/'study.json')
    spec = setup['spec']
    code = Path(__file__).parent
    if {p.name:sha256(p) for p in code.glob('*.py')} != setup['code_sha256']:
        raise ValueError('Frozen implementation changed')
    if mode == 'ablations':
        tasks = [(v,s) for s in spec['ablation']['seeds'] for v in spec['variants']]
    else:
        decision = read_json(study/'ablation_comparison.json')
        if not decision['full_run_justified']:
            raise ValueError('Validation does not justify a full run')
        tasks = [(next(v for v in spec['variants'] if v['name']==decision['selected']),spec['ablation']['seeds'][0])]
    base = ProjectConfig.from_dict(setup['base_configuration'])
    for variant,seed in tasks:
        folder = study/('ablations' if mode=='ablations' else 'full')/f"{variant['name']}-seed{seed}"
        folder.mkdir(parents=True,exist_ok=True)
        budget = spec['ablation']['budget_seconds_per_run'] if mode=='ablations' else spec['full']['budget_seconds']
        config = variant_config(base, variant, budget)
        manifest = dict(protocol=PROTOCOL, mode='ablation' if mode=='ablations' else 'full',
            variant=variant['name'], seed=seed, spec=spec, configuration=config.to_dict(),
            source_commit=setup['source_commit'], code_sha256=setup['code_sha256'],
            dataset_path=setup['dataset_path'],dataset_sha256=setup['dataset_sha256'],setup_seconds=0,
            initialization='new deterministic random weights; no historical checkpoints')
        if (folder/'manifest.json').exists():
            if read_json(folder/'manifest.json') != manifest:
                raise ValueError('Run manifest changed')
            if (folder/'status.json').exists() and read_json(folder/'status.json')['mode']=='complete':
                continue
            if not resume:
                raise ValueError('Explicit resume required')
        else:
            write_json(folder/'manifest.json',manifest)
        write_json(study/'status.json',dict(status='running',mode=mode,variant=variant['name'],seed=seed,
            run=str(folder),updated_utc=datetime.now(timezone.utc).isoformat()))
        outcome = run_one(folder,manifest,device,resume=resume)
        if outcome['status'] in {'interrupted_partial','ablation_partial','not_qualified_partial'}:
            write_json(study/'status.json',dict(status=outcome['status'],mode=mode,
                variant=variant['name'],seed=seed,run=str(folder),
                updated_utc=datetime.now(timezone.utc).isoformat()))
            return outcome
    result = compare_ablations(study) if mode=='ablations' else read_json(folder/'status.json')
    write_json(study/'status.json',dict(status='complete',mode=mode,result=result,
        updated_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',required=True);parser.add_argument('--mode',choices=['ablations','full'],required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try:
        worker(args.study,args.mode,args.device,args.resume)
    except Exception as error:
        study=Path(args.study)
        previous=read_json(study/'status.json') if (study/'status.json').exists() else {}
        previous.update(status='error',error=f'{type(error).__name__}: {error}',
            updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(study/'status.json',previous)
        raise
