import json
from pathlib import Path

import torch

from panama_uav import mapf_eval
from panama_uav.mapf_curriculum import CurriculumSession, CurriculumSpec, MapBank, model_payload, project_config
from panama_uav.training import atomic_save

ROOT = Path(__file__).resolve().parents[1]


def _model(tmp_path):
    spec = CurriculumSpec(populations=(2,), train_maps=1, validation_maps=1, replay_capacity=64, map_size=(32, 32))
    session = CurriculumSession(project_config(spec, ROOT), spec, MapBank(tmp_path / 'bank', spec), 1, 't',
        torch.device('cpu'))
    atomic_save(model_payload(session, spec), tmp_path / 'm.pt')
    return tmp_path / 'm.pt'


def test_test_cases_are_paired_deterministic_and_disjoint_from_training(tmp_path):
    spec, _ = mapf_eval.mapf_setup(ROOT)
    a = mapf_eval.test_case(tmp_path / 'a', spec, 8, 3)
    b = mapf_eval.test_case(tmp_path / 'b', spec, 8, 3)
    assert a['starts'] == b['starts'] and a['map_seed'] == b['map_seed']
    for k in range(0, 8, 2):
        assert a['starts'][k] == a['goals'][k + 1] and a['starts'][k + 1] == a['goals'][k]
    assert 32 <= a['map_size'] <= 64
    train = range(spec.seed + 10_000_000, spec.seed + 10_000_000 + spec.train_maps)
    validation = range(spec.seed + 60_000_000, spec.seed + 60_000_000 + spec.validation_maps)
    assert a['map_seed'] not in train and a['map_seed'] not in validation
    assert a['seed'] >= spec.seed + 1_200_000_000   # training episodes +70M.., validation cases +900M..
    assert mapf_eval.test_case(tmp_path / 'a', spec, 2, 3)['map_id'] == a['map_id']   # same map across N


def test_wilson_and_plan_order():
    lo, hi = mapf_eval.wilson(95, 100)
    assert lo < 0.95 < hi and 0.88 < lo and hi < 0.98
    tasks = mapf_eval.plan(['x', 'y'], [2, 32], cases=4, primary=2, chunk=1, leo_cases=1, leo_populations=[8])
    kinds = [t['kind'] for t in tasks]
    assert kinds == ['mapf'] * 8 + ['leo'] * 8 + ['mapf'] * 8
    assert tasks[0]['num_agents'] == 32 and tasks[-1]['cases'] == [3, 4]
    assert {t['job']['route'] for t in tasks if t['kind'] == 'leo'} == {'cloud', 'allocated'}


def test_budget_records_stages(tmp_path):
    payload = dict(training_protocol='panama_mapf_curriculum_v1', generated_joint_steps=1000,
        population_episodes={2: 10, 32: 4}, stage_history=[dict(population=2, stage_episodes=10,
            generated_joint_steps=600, graduated=True, success_rate=0.96)])
    budget = mapf_eval.budget_of(payload)
    assert [(s['population'], s['episodes'], s['joint_steps']) for s in budget['stages']] == [(2, 10, 600), (32, 4, 400)]
    assert budget['episodes_at_32'] == 4 and budget['joint_steps_at_32'] == 400 and budget['total_episodes'] == 14
    model8 = mapf_eval.budget_of(dict(training_protocol='x', population_episodes={2: 5, 16: 7},
        population_steps={2: 50, 16: 70}, generated_joint_steps=120))
    assert model8['total_episodes'] == 12 and model8['total_agent_steps_upper'] == 2 * 50 + 16 * 70


def test_run_writes_incremental_results_and_respects_wall_guard(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', str(ROOT / 'src'))
    path = _model(tmp_path)
    out = tmp_path / 'eval'
    result = mapf_eval.run(out, {'curriculum': str(path), 'baseline32': str(path)}, [2], cases=2, primary=2,
        chunk=1, leo_cases=0, leo_populations=[], workers=2, max_wall=600, task_timeout=300, project_root=ROOT)
    assert result['tasks_finished'] == 4 and not result['failed_or_skipped_tasks']
    cells = {c['model']: c for c in result['mapf_cells']}
    assert cells['curriculum']['episodes'] == 2 and cells['curriculum']['success_ci95_low'] is not None
    assert result['mapf_paired_success_differences'][0]['success_diff'] == 0.0   # same weights, same cases
    summary = (out / 'summary.md').read_text()
    assert 'not budget-matched' in summary and (out / 'mapf_results.csv').exists()
    cut = mapf_eval.run(tmp_path / 'cut', {'curriculum': str(path)}, [2], cases=50, primary=50, chunk=50,
        leo_cases=0, leo_populations=[], workers=1, max_wall=0.5, task_timeout=300, project_root=ROOT)
    assert cut['manifest']['status'] == 'wall_guard'
    assert json.loads((tmp_path / 'cut' / 'eval_manifest.json').read_text())['status'] == 'wall_guard'
