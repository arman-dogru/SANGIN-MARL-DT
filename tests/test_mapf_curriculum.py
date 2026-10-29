from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from panama_uav.environment import astar_path
from panama_uav.mapf_curriculum import (CurriculumSession, CurriculumSpec, MapBank, load_spec, paired_case,
    project_config, random_map, run)
from panama_uav.scenarios import load_map

ROOT = Path(__file__).resolve().parents[1]


def test_random_map_size_and_density_ranges():
    for seed in range(20):
        grid, size, density = random_map(seed, (32, 64), (0.10, 0.30))
        assert grid.shape == (size, size) and 32 <= size <= 64
        assert 0.10 <= density <= 0.30
        assert grid.mean() >= density - 0.03  # connected() only adds obstacles


def test_paired_case_swaps_starts_and_goals():
    grid, _, _ = random_map(3, (48, 48), (0.3, 0.3))
    starts, goals = paired_case(grid, 32, seed=7, distance=(4, 40))
    assert len(set(starts)) == len(set(goals)) == 32
    for k in range(0, 32, 2):
        assert starts[k] == goals[k + 1] and starts[k + 1] == goals[k]
        assert 4 <= len(astar_path(grid, starts[k], goals[k])) - 1 <= 40
    assert all(grid[p] == 0 for p in starts)
    with pytest.raises(ValueError):
        paired_case(grid, 3, seed=1)


def test_spec_rejects_odd_population_and_config_files_load():
    with pytest.raises(ValueError):
        CurriculumSpec(populations=(2, 5))
    for name in ('mapf_curriculum.toml', 'mapf_baseline32.toml'):
        spec = load_spec(ROOT / 'configs' / name)
        assert spec.map_size == (32, 64) and spec.density == (0.10, 0.30)
    assert load_spec(ROOT / 'configs/mapf_curriculum.toml').populations == (2, 4, 8, 16, 32)


def test_epsilon_restarts_and_replay_resets_on_advance(tmp_path):
    spec = CurriculumSpec(populations=(2, 4), train_maps=2, validation_maps=1, validation_episodes=1,
        replay_capacity=512, map_size=(32, 32))
    session = CurriculumSession(project_config(spec, ROOT), spec, MapBank(tmp_path, spec), 1, 'test', torch.device('cpu'))
    for _ in range(70):
        session.step()
    assert session.exploration_step_count() == 70 and len(session.replay) > 0
    session.advance(dict(success_rate=0.0), graduated=False)
    assert session.stage_index == 1 and session.exploration_step_count() == 0 and len(session.replay) == 0
    assert session.environment.num_agents == 4 and session.stage_history[0]['graduated'] is False
    case = session.current_case
    assert case['paired'] and case['starts'][0] == case['goals'][1]


def test_run_checkpoints_and_resumes(tmp_path):
    spec_path = tmp_path / 'spec.toml'
    spec_path.write_text('protocol = "panama_mapf_curriculum_v1"\npopulations = [2, 4]\nmap_size = [32, 32]\n'
        'episodes_per_stage = 2\nvalidation_interval_episodes = 2\nvalidation_episodes = 1\ntrain_maps = 2\n'
        'validation_maps = 1\nreplay_capacity = 2000\ncheckpoint_interval_episodes = 1\n'
        f'base_config = "{ROOT / "configs/phase1_clean.toml"}"\nshutdown_reserve_seconds = 0\n')
    root = tmp_path / 'run'
    first = run(root, spec_path, project_root=ROOT, max_wall_seconds=8)
    assert first['status'] == 'session_wall_guard' and (root / 'checkpoint.pt').exists()
    saved = torch.load(root / 'checkpoint.pt', weights_only=False)
    steps = saved['session']['session'].generated_steps
    assert steps > 0 and saved['session']['session'].optimizer.state  # AdamW moments are checkpointed
    final = run(root, spec_path, project_root=ROOT, max_wall_seconds=600)
    assert final['status'] == 'curriculum_complete' and final['sessions'] == 2
    assert final['generated_joint_steps'] > steps
    assert [h['population'] for h in final['stage_history']] == [2, 4]
    assert (root / 'stage_1_n4_model.pt').exists()


def test_stage_model_loads_through_leo_load_model(tmp_path):
    from panama_uav.config import load_config
    from panama_uav.mapf_curriculum import model_payload
    from panama_uav.training import atomic_save, load_model
    spec = CurriculumSpec(populations=(2,), train_maps=1, validation_maps=1, replay_capacity=64, map_size=(32, 32))
    session = CurriculumSession(project_config(spec, ROOT), spec, MapBank(tmp_path, spec), 1, 't', torch.device('cpu'))
    atomic_save(model_payload(session, spec), tmp_path / 'm.pt')
    leo = load_config(ROOT / 'configs/paper_remote_dt_250ms_uav6_cached.toml')
    model, payload = load_model(tmp_path / 'm.pt', leo, torch.device('cpu'), require_qualified=False)
    assert payload['training_protocol'] == 'panama_mapf_curriculum_v1'
    with pytest.raises(ValueError):
        load_model(tmp_path / 'm.pt', leo, torch.device('cpu'), require_qualified=True)
