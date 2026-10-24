"""Isolation and telemetry checks for the parameter-only pilot harness."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from panama_uav.communication_effects import measurements
from panama_uav.config import load_config
from panama_uav.deadline_study import deadline_config
from panama_uav.evaluation import case_bank, run_episode
from panama_uav.inference_controls import InferenceCondition, inference_class
from panama_uav.models import DQN, select_action
from panama_uav.staleness_audit import audit_stress
from panama_uav.staleness_remote import controlled_class
from test_deadline_study import inputs

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uav6_pilot", ROOT / "scripts/run_uav6_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
audit_spec = importlib.util.spec_from_file_location("uav6_audit", ROOT / "scripts/audit_uav6_pilot.py")
audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit)


def test_configuration_changes_exactly_one_parameter():
    base = ROOT / "configs/paper_remote_dt_250ms.toml"
    modified = ROOT / "configs/paper_remote_dt_250ms_uav6.toml"
    assert pilot.parameter_change(base, modified) == {"resource_allocation.drone_cycles_per_second": [1e9, 6e9]}
    original, actual = load_config(base), load_config(modified)
    assert replace(actual, resource_allocation=replace(actual.resource_allocation,
                                                     drone_cycles_per_second=1e9)) == original


@pytest.mark.parametrize("old,new", [("cloud_cycles_per_second = 1e10", "cloud_cycles_per_second = 2e10"),
    ("task_input_bits = 100000", "task_input_bits = 200000"),
    ("enforce_task_deadlines = false", "enforce_task_deadlines = true")])
def test_extra_experimental_changes_are_rejected(tmp_path, old, new):
    path = tmp_path / "extra.toml"
    text = (ROOT / "configs/paper_remote_dt_250ms_uav6.toml").read_text()
    assert old in text
    path.write_text(text.replace(old, new))
    with pytest.raises(ValueError, match="change only"):
        pilot.parameter_change(ROOT / "configs/paper_remote_dt_250ms.toml", path)


def test_population_scope_requires_explicit_qualification():
    payload = dict(qualification=[dict(num_agents=n, episodes=200, success_rate=.95) for n in (2, 4, 6, 8)])
    actual, excluded = pilot.population_scope(payload, (2, 4, 8, 16))
    assert actual == [2, 4, 8] and [e["num_agents"] for e in excluded] == [16]
    with pytest.raises(ValueError, match="No requested population"):
        pilot.population_scope(payload, (16,))


def test_execution_conditions_keep_every_other_resolved_setting(tmp_path):
    config = load_config(ROOT / "configs/paper_remote_dt_250ms_uav6.toml")
    for condition in pilot.CONDITIONS:
        actual = pilot.execution_config(config, condition, tmp_path, dict(orbital_offset_seconds=100.))
        assert actual.resource_allocation == replace(config.resource_allocation,
            drone_cycles_per_second=condition["cpu"], failure_directory=str(tmp_path / condition["name"] / "failures"))
        assert actual.communication == replace(config.communication,
            mission_route="uav" if condition["route"] == "edge" else "cloud",
            backhaul_time_offset_seconds=config.communication.backhaul_time_offset_seconds+100.)
        assert actual.map == config.map and actual.uav == config.uav and actual.reward == config.reward


def test_cpu_profile_integrates_allocated_capacity_and_concurrent_jobs():
    history = [dict(event="compute_allocation", pool="drone", time=t, utilization=u, jobs=[{}]*n)
               for t, u, n in ((0., 0., 0), (1., .8, 2), (2., .4, 1), (3., 0., 0))]
    actual = pilot.pool_profile(history, 4., "drone")
    assert actual == pytest.approx(dict(utilization_seconds=1.2, concurrent_job_seconds=3.,
                                      occupied_seconds=2., max_concurrent_jobs=2))
    assert pilot.pool_profile(history, 4., "cloud")["max_concurrent_jobs"] == 0
    with pytest.raises(ValueError, match="capacity"):
        pilot.pool_profile([dict(history[1], utilization=1.2)], 4., "drone")


def test_handover_overlap_uses_half_open_windows():
    assert not pilot.overlaps(.05, .1, [(0., .05)])
    assert not pilot.overlaps(0., .05, [(.05, .1)])
    assert pilot.overlaps(.049, .1, [(0., .05)])


def test_reference_audit_allows_only_the_expected_capacity_checksum_change():
    old = [dict(positions=[[1, 2]], actions={0: 4}, event_checksum="old")]
    changed_checksum = [dict(old[0], event_checksum="new")]
    audit.compare_frames(old, changed_checksum, capacity_changed=True)
    with pytest.raises(AssertionError):
        audit.compare_frames(old, changed_checksum)
    with pytest.raises(AssertionError):
        audit.compare_frames(old, [dict(changed_checksum[0], positions=[[1, 3]])], capacity_changed=True)
    with pytest.raises(AssertionError):
        audit.compare_frames(old, [dict(changed_checksum[0], actions={0: 0})], capacity_changed=True)


@pytest.mark.parametrize("condition", pilot.CONDITIONS)
def test_read_only_telemetry_and_replay_preserve_actual_execution(inputs, tmp_path, condition):
    original, _ = inputs
    config = deadline_config(original, 250)
    config = replace(config, resource_allocation=replace(config.resource_allocation, remote_acceptance_approved=True))
    case = case_bank(config, 1, (2,))[0]
    case["orbital_offset_seconds"] = 0.
    cfg = pilot.execution_config(config, condition, tmp_path, case)
    c = InferenceCondition(condition["name"], condition["route"])
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = DQN((3, 15, 15), config.vector_length).eval()
    row, trace = run_episode(cfg, model, case, torch.device("cpu"), max_steps=2,
                             remote_communication_class=inference_class(c))
    row, tasks, rounds = measurements(row, trace, c)
    before = json.dumps(trace, sort_keys=True)
    row, tasks, rounds = pilot.enrich(row, tasks, rounds, trace)
    assert json.dumps(trace, sort_keys=True) == before
    assert all(t["fixed_previous_tasks"] <= 1 for t in tasks)
    assert sum(r["uav_tasks"]+r["cloud_tasks"] for r in rounds) == len(tasks)
    for pool in ("drone", "cloud"):
        assert 0 <= row[pool+"_utilization_seconds"] <= row["observed_seconds"] + 1e-6
    shadow = lambda location, robot, obs: select_action(model, obs, 0., np.random.default_rng(0), torch.device("cpu"))
    assert audit_stress(trace, shadow)["status"] == "passed"
    if condition["route"] == "allocated":
        _, production = run_episode(cfg, model, case, torch.device("cpu"), max_steps=2,
                                    remote_communication_class=controlled_class(0))
        assert trace["frames"] == production["frames"]
        assert trace["remote_events"] == production["remote_events"]


def test_prepare_rejects_unqualified_checkpoint_before_creating_output(inputs, tmp_path):
    _, checkpoint = inputs
    output = tmp_path / "unqualified"
    with pytest.raises(ValueError, match="qualified"):
        pilot.prepare(output, ROOT / "configs/paper_remote_dt_250ms.toml",
            ROOT / "configs/paper_remote_dt_250ms_uav6.toml", checkpoint,
            ROOT / "results/communication/cases.json", cases=1)
    assert not output.exists()
