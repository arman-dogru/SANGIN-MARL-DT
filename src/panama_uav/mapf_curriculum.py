"""Staged PANAMA MAPF curriculum on procedural random-obstacle maps.

Every episode uses the PANAMA reset of arXiv:2508.06767 Sec. III-A1: agents are
placed in pairs whose start and goal cells are swapped (start_a = goal_b,
start_b = goal_a), which deliberately creates head-on conflicts. Maps are
random obstacle grids with side length and obstacle density drawn per map.

Learner, observation, rewards and the remote-aligned execution are those of
``phase1_revised.RevisedSession``. Differences: one population per stage,
greedy validation on a fixed 200-case paired bank every N training episodes,
graduation at a success threshold (or at the per-stage episode cap, flagged),
replay reset on graduation (paper) and an optional epsilon restart per stage.
A checkpoint holds the whole session (weights, target, AdamW state, replay,
RNG, current episode), so a run resumes across 11-hour Kaggle sessions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import math
import signal
import time
import tomllib

import numpy as np
import torch

from .config import COMMUNICATION_SCHEMA, CurriculumStage, ExperimentMode, load_config
from .environment import astar_path
from .models import choose_device
from .phase1 import weights_hash
from .phase1_data import connected, scenario_for
from .phase1_execution import RemoteAlignedCleanMAPFEnvironment
from .phase1_revised import RevisedSession, evaluate
from .records import append_jsonl, sha256, write_json
from .scenarios import load_map
from .training import TrainingSession, atomic_save, fingerprint, new_replay

PROTOCOL = 'panama_mapf_curriculum_v1'


@dataclass(frozen=True)
class CurriculumSpec:
    populations: tuple[int, ...] = (2, 4, 8, 16, 32)
    map_size: tuple[int, int] = (32, 64)
    density: tuple[float, float] = (0.10, 0.30)
    pair_distance: tuple[int, int] = (4, 40)
    episodes_per_stage: int = 10_000
    validation_interval_episodes: int = 500
    validation_episodes: int = 200
    graduation_success_rate: float = 0.95
    advance_on_cap: bool = True
    epsilon_reset_per_stage: bool = True
    reset_replay_on_graduation: bool = True
    train_maps: int = 2000
    validation_maps: int = 100
    max_episode_steps: int = 250
    session_wall_seconds: float = 39_600.0
    shutdown_reserve_seconds: float = 300.0
    checkpoint_interval_episodes: int = 100
    seed: int = 42
    base_config: str = 'configs/phase1_clean.toml'
    learning_rate: float = 1e-3
    gamma: float = 0.90
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 20_000
    replay_capacity: int = 50_000
    batch_size: int = 64
    torch_threads: int = 1

    def __post_init__(self):
        if not self.populations or any(n < 2 or n % 2 for n in self.populations):
            raise ValueError('Paired placement needs even populations >= 2')
        if not 8 <= self.map_size[0] <= self.map_size[1] or not 0 <= self.density[0] <= self.density[1] < 0.5:
            raise ValueError('Invalid map size or density range')
        if not 1 <= self.pair_distance[0] <= self.pair_distance[1]:
            raise ValueError('Invalid pair distance band')
        if self.max_episode_steps > 250 or self.validation_episodes <= 0 or self.validation_interval_episodes <= 0:
            raise ValueError('Invalid episode or validation settings')


def load_spec(path):
    with Path(path).open('rb') as stream:
        raw = tomllib.load(stream)
    if raw.pop('protocol', PROTOCOL) != PROTOCOL:
        raise ValueError('Not a curriculum protocol file')
    for key in ('populations', 'map_size', 'density', 'pair_distance'):
        if key in raw:
            raw[key] = tuple(raw[key])
    return CurriculumSpec(**raw)


def random_map(seed, size_range, density_range):
    """Uniform random obstacles, largest 4-connected free component kept."""
    rng = np.random.default_rng(seed)
    size = int(rng.integers(size_range[0], size_range[1] + 1))
    density = float(rng.uniform(*density_range))
    grid = connected((rng.random((size, size)) < density).astype(np.uint8))
    return grid, size, density


def write_map(grid, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = grid.shape
    path.write_text(f'type octile\nheight {h}\nwidth {w}\nmap\n' +
        '\n'.join(''.join('@' if x else '.' for x in row) for row in grid) + '\n', encoding='utf-8')
    return path


def paired_case(grid, count, seed, distance=(4, 40), attempts=2000):
    """PANAMA reset: count/2 pairs, each pair's start and goal cells swapped."""
    if count % 2:
        raise ValueError('Paired placement needs an even number of agents')
    rng = np.random.default_rng(seed)
    free = np.argwhere(grid == 0)
    lo, hi = distance
    used, starts, goals = set(), [], []
    for _ in range(count // 2):
        for _ in range(attempts):
            a = tuple(int(v) for v in free[rng.integers(len(free))])
            if a in used:
                continue
            delta = np.abs(free - np.array(a)).sum(1)
            near = free[(delta >= lo) & (delta <= hi)]
            if not len(near):
                continue
            b = tuple(int(v) for v in near[rng.integers(len(near))])
            if b in used:
                continue
            path = astar_path(grid, a, b)
            if not path or not lo <= len(path) - 1 <= hi:
                continue
            used |= {a, b}
            starts += [a, b]
            goals += [b, a]
            break
        else:
            raise ValueError(f'Could not place {count} paired agents')
    return starts, goals


class MapBank:
    """Deterministic, disjoint train/validation map files under the run root."""

    def __init__(self, root, spec):
        self.root = Path(root)
        self.spec = spec
        self.records = {}
        for split, offset, number in (('train', 0, spec.train_maps), ('validation', 50_000_000, spec.validation_maps)):
            rows = []
            for index in range(number):
                map_seed = spec.seed + 10_000_000 + offset + index
                relative = f'maps/{split}-{index:05d}.map'
                grid, size, density = random_map(map_seed, spec.map_size, spec.density)
                write_map(grid, self.root / relative)
                rows.append(dict(map_id=f'{split}-{index:05d}', split=split, seed=map_seed, path=relative,
                    size=size, nominal_density=density, density=float(grid.mean())))
            self.records[split] = rows

    def case(self, split, count, seed, index):
        rows = self.records[split]
        record = rows[int(np.random.default_rng(seed).integers(len(rows)))]
        path = str(self.root / record['path'])
        starts, goals = paired_case(load_map(path), count, seed, self.spec.pair_distance)
        return dict(case_id=f"{record['map_id']}-n{count}-seed{seed}", map_id=record['map_id'], family='random_obstacles',
            split=split, map_path=path, map_size=record['size'], density=record['density'],
            num_agents=count, seed=int(seed), starts=[list(p) for p in starts], goals=[list(p) for p in goals],
            designed_coordination=True, paired=True, evaluation_index=index)

    def validation_bank(self, count, stage_index):
        return [self.case('validation', count, self.spec.seed + 900_000_000 + stage_index * 1_000_000 + i, i)
                for i in range(self.spec.validation_episodes)]


def project_config(spec, project_root):
    base = load_config(Path(project_root) / spec.base_config)
    stages = tuple(CurriculumStage(f'agents-{n}', num_agents=n, max_episode_steps=spec.max_episode_steps,
        min_episodes=0, max_episodes=spec.episodes_per_stage, graduation_success_rate=spec.graduation_success_rate)
        for n in spec.populations)
    total = len(spec.populations) * spec.episodes_per_stage * spec.max_episode_steps
    training = replace(base.training, learning_rate=spec.learning_rate, gamma=spec.gamma,
        epsilon_start=spec.epsilon_start, epsilon_end=spec.epsilon_end, epsilon_decay_steps=spec.epsilon_decay_steps,
        replay_capacity=spec.replay_capacity, batch_size=spec.batch_size, target_environment_steps=total,
        budget_seconds=spec.session_wall_seconds)
    return replace(base, curriculum=stages, training=training)


class CurriculumSession(RevisedSession):
    def __init__(self, config, spec, bank, seed, run_id, device):
        self.spec = spec
        self.bank = bank
        self.stage_start_step = 0
        self.stage_history = []
        super().__init__(config, None, seed, run_id, device)

    @property
    def population(self):
        return self.spec.populations[self.stage_index]

    def exploration_step_count(self):
        if self.spec.epsilon_reset_per_stage:
            return self.generated_steps - self.stage_start_step
        return self.generated_steps

    def start_episode(self):
        count = self.population
        local = self.population_episodes[count]
        self.stage_episodes = local
        episode_seed = self.seed + 70_000_000 + self.stage_index * 1_000_000 + local
        self.current_case = self.bank.case('train', count, episode_seed, local)
        self.environment = RemoteAlignedCleanMAPFEnvironment(self.config, ExperimentMode.UAV,
            self.current_case['map_path'], count, self.spec.max_episode_steps, episode_seed,
            scenario_for(self.current_case))
        self.observations, _ = self.environment.reset()
        self.action_rng = np.random.default_rng(episode_seed)
        if not self.actor_models:
            self.actor_models[0] = self._actor(self.latest_published_weights)
        version = max(self.actor_models)
        self.deployed_versions = {a: version for a in self.observations}
        self.episode_reward = 0.0
        from collections import Counter
        self.epoch_totals = Counter()

    def advance(self, validation, graduated):
        self.stage_history.append(dict(validation, stage_index=self.stage_index, population=self.population,
            stage_episodes=self.population_episodes[self.population], graduated=graduated,
            generated_joint_steps=self.generated_steps))
        self.stage_index += 1
        self.stage_start_step = self.generated_steps
        if self.spec.reset_replay_on_graduation:
            self.replay = new_replay(self.config, self.seed + self.stage_index)
        if self.stage_index < len(self.spec.populations):
            self.start_episode()


def model_payload(session, spec):
    manifest = Path(session.bank.root) / 'manifest.json'
    return dict(communication_schema=COMMUNICATION_SCHEMA, policy_family='panama_0.1_task',
        training_protocol=PROTOCOL, condition='clean',
        dataset_sha256=sha256(manifest) if manifest.exists() else None,
        model_state_dict={k: v.detach().cpu().clone() for k, v in session.policy.state_dict().items()},
        spatial_shape=(3, 15, 15), vector_length=session.config.vector_length,
        task_reward=session.config.to_dict()['reward'], spec=asdict(spec), run_id=session.run_id,
        seed=session.seed, stage_index=session.stage_index, stage_history=list(session.stage_history),
        population_episodes=dict(session.population_episodes), generated_joint_steps=session.generated_steps,
        learner_updates=session.learner_updates, model_weights_sha256=weights_hash(session.policy),
        curriculum_completed=session.stage_index >= len(spec.populations))


def _truncate_logs(root, offsets):
    for path in root.glob('*.jsonl'):
        size = offsets.get(path.name)
        if size is None:
            path.unlink()
        else:
            with path.open('r+b') as stream:
                stream.truncate(size)


def run(root, spec_path, device_name='cpu', project_root=None, max_wall_seconds=None):
    """Train or resume ``root``; returns the status dict. Stops cleanly at the session wall guard."""
    begin = time.monotonic()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec = load_spec(spec_path)
    project_root = Path(project_root or Path(spec_path).resolve().parent.parent)
    config = project_config(spec, project_root)
    torch.set_num_threads(spec.torch_threads)
    device = choose_device(device_name)
    wall = min(spec.session_wall_seconds, max_wall_seconds or math.inf)
    train_end = begin + wall - spec.shutdown_reserve_seconds
    checkpoint = root / 'checkpoint.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['spec'] != asdict(spec):
            raise ValueError('Resume spec changed; use a fresh run directory')
        # The project root differs between the box and Kaggle; everything else must match.
        previous = saved['session']['session'].config
        if replace(previous, project_root=config.project_root) != config:
            raise ValueError('Resume configuration changed; use a fresh run directory')
        saved['session']['config_fingerprint'] = fingerprint(config)
        session = TrainingSession.restore(saved['session'], config, 'clean', spec.seed, root.name, device)
        session.config = config
        session.bank.root = root
        state = saved['state']
        _truncate_logs(root, saved['log_offsets'])
    else:
        bank = MapBank(root, spec)
        write_json(root / 'manifest.json', dict(protocol=PROTOCOL, spec=asdict(spec), config=config.to_dict(),
            maps={k: v for k, v in bank.records.items()}))
        session = CurriculumSession(config, spec, bank, spec.seed, root.name, device)
        state = dict(sessions=0, wall_seconds=0.0, last_validation_episode=0, validation_pending=False)
    state['sessions'] += 1
    previous_wall = state['wall_seconds']
    stop = {'requested': False}
    old_handlers = {s: signal.signal(s, lambda *_: stop.update(requested=True)) for s in (signal.SIGINT, signal.SIGTERM)}

    def save():
        state['wall_seconds'] = previous_wall + time.monotonic() - begin
        offsets = {p.name: p.stat().st_size for p in root.glob('*.jsonl')}
        atomic_save(dict(protocol=PROTOCOL, spec=asdict(spec), session=session.checkpoint(), state=state,
            log_offsets=offsets), checkpoint)
        atomic_save(model_payload(session, spec), root / 'latest_model.pt')

    def validate():
        stage = session.stage_index
        count = session.population
        episodes = session.population_episodes[count]
        tag = f'validation-s{stage}-e{episodes}'
        records = evaluate(config, session.policy, session.bank.validation_bank(count, stage), device,
            root / f'{tag}.jsonl', tag, train_end, lambda: stop['requested'], protocol=PROTOCOL, metrics=PROTOCOL)
        done = len(records) == spec.validation_episodes
        result = dict(tag=tag, stage_index=stage, population=count, stage_episodes=episodes,
            episodes=len(records), complete=done, successes=sum(r['success'] for r in records),
            success_rate=sum(r['success'] for r in records) / max(1, len(records)),
            mean_steps=float(np.mean([r['steps'] for r in records])) if records else None,
            generated_joint_steps=session.generated_steps, model_weights_sha256=weights_hash(session.policy),
            wall_seconds=previous_wall + time.monotonic() - begin)
        if not done:
            state['validation_pending'] = True
            return
        state['validation_pending'] = False
        state['last_validation_episode'] = episodes
        append_jsonl(root / 'validation.jsonl', result)
        graduated = result['success_rate'] >= spec.graduation_success_rate
        if not graduated and episodes >= spec.episodes_per_stage and not spec.advance_on_cap:
            state['halted'] = True
        elif graduated or episodes >= spec.episodes_per_stage:
            atomic_save(model_payload(session, spec), root / f'stage_{stage}_n{count}_model.pt')
            session.advance(result, graduated)
            state['last_validation_episode'] = 0
        save()

    reason = 'session_wall_guard'
    try:
        while session.stage_index < len(spec.populations):
            if stop['requested'] or time.monotonic() >= train_end:
                reason = 'interrupted' if stop['requested'] else 'session_wall_guard'
                break
            if state.get('halted'):
                reason = 'stage_cap_not_graduated'
                break
            if state['validation_pending']:
                validate()
                continue
            env = session.environment
            if env.all_agents_at_goals or env.step_count >= spec.max_episode_steps:
                session.start_episode()
            record, episode = session.step()
            if episode:
                episode.pop('starts', None); episode.pop('goals', None)
                episode['wall_seconds'] = previous_wall + time.monotonic() - begin
                episode['stage_index'] = session.stage_index
                append_jsonl(root / 'training.jsonl', episode)
                local = session.population_episodes[session.population]
                if local % spec.checkpoint_interval_episodes == 0:
                    save()
                if (local - state['last_validation_episode'] >= spec.validation_interval_episodes
                        or local >= spec.episodes_per_stage):
                    state['validation_pending'] = True
        else:
            reason = 'curriculum_complete'
    finally:
        save()
        status = dict(protocol=PROTOCOL, status=reason, stage_index=session.stage_index,
            populations=list(spec.populations), population_episodes=dict(session.population_episodes),
            generated_joint_steps=session.generated_steps, learner_updates=session.learner_updates,
            stage_history=session.stage_history, sessions=state['sessions'], wall_seconds=state['wall_seconds'])
        write_json(root / 'status.json', status)
        for s, handler in old_handlers.items():
            signal.signal(s, handler)
    return status


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('root', type=Path)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-wall-seconds', type=float)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.spec, args.device, max_wall_seconds=args.max_wall_seconds), indent=1))
