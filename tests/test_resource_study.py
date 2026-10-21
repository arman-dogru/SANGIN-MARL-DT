from pathlib import Path
import json

import pytest

from panama_uav.config import ProjectConfig, load_config
from panama_uav.resource_study import conditions, optimize_snapshot, paired_metrics, run_study

ROOT = Path(__file__).resolve().parents[1]


def test_conditions_have_four_policies_and_exact_primary_grid():
    config = load_config(ROOT/"configs/resource_study.toml")
    primary = conditions(config, secondary=False)
    assert len(primary) == 37
    assert len({name for name, _, _ in primary}) == 37
    for _, cfg, tags in primary[1:]:
        assert cfg.resource_allocation.enabled
        assert cfg.resource_allocation.policy in ("equal", "static", "aoi", "priority")
        assert tags["uplink_factor"] in (.25, 1, 4)
        assert tags["cpu_factor"] in (.25, 1, 4)
    all_conditions = conditions(config)
    assert len(all_conditions) == 97
    assert len({name for name, _, _ in all_conditions}) == 97
    assert ProjectConfig.from_dict(config.to_dict()) == config
    legacy = load_config(ROOT/"configs/base.toml").to_dict()
    del legacy["resource_allocation"]
    assert not ProjectConfig.from_dict(legacy).resource_allocation.enabled


def test_snapshot_command_and_failure_artifact(tmp_path):
    result = optimize_snapshot(ROOT/"tests/fixtures/allocation_snapshot.json", tmp_path/"result.json")
    assert result["absolute_gap"] >= -1e-6
    data = result["snapshot"]
    data["cloud_cycles_per_second"] = data["drone_cycles_per_second"] = 0
    (tmp_path/"bad.json").write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        optimize_snapshot(tmp_path/"bad.json", tmp_path/"bad-result.json")
    assert (tmp_path/"bad-result.failed.json").exists()


def test_paired_interval_has_expected_direction():
    rows = [dict(case_id="a", waiting_steps=3), dict(case_id="b", waiting_steps=5)]
    reference = [dict(case_id="a", waiting_steps=1), dict(case_id="b", waiting_steps=3)]
    result = paired_metrics(rows, reference, samples=20)["waiting_steps"]
    assert result == dict(delta=2., pairs=2, ci95=[2., 2.])


def test_paired_metrics_match_case_ids_and_exclude_unmatched_cases():
    rows = [dict(case_id="a", waiting_steps=3), dict(case_id="b", waiting_steps=10),
            dict(case_id="unmatched", waiting_steps=1000)]
    reference = [dict(case_id="b", waiting_steps=6), dict(case_id="a", waiting_steps=1)]
    result = paired_metrics(rows, reference, samples=200)["waiting_steps"]
    assert result["delta"] == 3 and result["pairs"] == 2
    assert result["ci95"] == [2., 4.]


def test_paper_run_rejects_missing_qualification_and_timeline(tmp_path):
    config = load_config(ROOT/"configs/resource_study.toml")
    with pytest.raises(ValueError, match="qualified checkpoint"):
        run_study(config, tmp_path/"paper")
    assert not (tmp_path/"paper").exists()


def test_paired_diagnostic_produces_real_records_and_figures(tmp_path):
    config = load_config(ROOT/"configs/resource_study.toml")
    rows = run_study(config, tmp_path/"diagnostic", diagnostic=True, cases=1,
        counts=(2,), secondary=False, selected=["up-1-cpu-1__equal", "up-1-cpu-1__priority"], max_steps=3)
    root = tmp_path/"diagnostic"
    assert len(rows) == 3
    assert json.loads((root/"status.json").read_text())["status"] == "complete"
    assert (root/"figures/xi_fov_vs_compute.svg").exists()
    assert (root/"up-1-cpu-1__priority/allocations.jsonl").exists()
    assert (root/"up-1-cpu-1__priority/tasks.csv").exists()
    manifest = json.loads((root/"manifest.json").read_text())
    assert manifest["diagnostic"]
    assert manifest["model"]["kind"] == "untrained_frozen_diagnostic"
    assert all(r["cases"] == 1 for r in rows)
    priority = next(r for r in rows if r["policy"] == "priority")
    assert priority["paired_equal"]["waiting_steps"]["pairs"] == 1
    assert priority["paired_nominal"]["waiting_steps"] == dict(delta=0., pairs=1, ci95=[0.,0.])


def test_resume_preserves_finished_cases(tmp_path):
    config = load_config(ROOT/"configs/resource_study.toml")
    args = dict(diagnostic=True, cases=1, counts=(2,), secondary=False,
                selected=["up-1-cpu-1__priority"], max_steps=2)
    root = tmp_path/"resumable"
    run_study(config, root, **args)
    before = (root/"up-1-cpu-1__priority/episodes.jsonl").read_bytes()
    run_study(config, root, resume=True, **args)
    assert (root/"up-1-cpu-1__priority/episodes.jsonl").read_bytes() == before
    assert len(json.loads((root/"manifest.json").read_text())["resumptions"]) == 1
