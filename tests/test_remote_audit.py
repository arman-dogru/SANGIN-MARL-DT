import copy
import json

import pytest
import torch

from panama_uav.evaluation import case_bank, run_episode
from panama_uav.models import DQN, select_action
from panama_uav.remote_audit import replay_trace
from test_remote_execution import remote_config


def episode(shadow=True):
    config = remote_config(drone_cycles_per_second=1e11, oracle_shadow=shadow)
    torch.manual_seed(42)
    model = DQN((3, 15, 15), config.vector_length).eval()
    row, trace = run_episode(config, model, case_bank(config, 1, (2,))[0], torch.device("cpu"), max_steps=3)
    return model, row, trace


def test_archive_replay_rebuilds_tensors_stores_actions_pairs_and_outcome():
    import numpy as np
    model, row, trace = episode()
    restored = json.loads(json.dumps(trace, allow_nan=False))
    def infer(location, agent, observation):
        return select_action(model, observation, 0, np.random.default_rng(0), torch.device("cpu"))
    audit = replay_trace(restored, infer)
    assert audit["status"] == "passed" and audit["consistency_decisions"] == row["steps"]
    assert audit["model_actions_verified"] == audit["rebuilt_inferences"] > 0
    assert audit["final_checksum"] == trace["remote_events"][-1]["checksum"]
    assert audit["allocations"] > 0
    for snapshot in audit["pair_consistency"]:
        for pair in snapshot["pairs"]:
            assert pair["e2e_ordering_error"] <= pair["uplink_ordering_error"]+pair["downlink_ordering_error"]
            assert pair["self_age_seconds"] == pytest.approx(pair["self_pre_age_seconds"]+pair["self_post_age_seconds"])


def test_archive_tampering_and_deleted_events_fail_the_audit():
    _, _, trace = episode()
    altered = copy.deepcopy(trace)
    altered["remote_events"][-1]["time"] += .01
    with pytest.raises(ValueError, match="checksum"):
        replay_trace(altered)
    altered = copy.deepcopy(trace)
    altered["remote_events"].pop(12)
    with pytest.raises(ValueError):
        replay_trace(altered)


def test_action_oracle_cannot_change_actual_trajectory_or_event_archive():
    _, actual_row, actual = episode(False)
    _, shadow_row, shadow = episode(True)
    assert actual_row["success"] == shadow_row["success"]
    assert actual["remote_events"] == shadow["remote_events"]
    assert actual["outcome"] == shadow["outcome"]
    assert any(e["event"] == "action_shadow" for e in shadow["oracle_events"])


def test_remote_metric_populations_and_end_of_episode_censoring():
    from panama_uav.resource_study import episode_metrics
    _, row, trace = episode()
    metrics = episode_metrics(row, trace)
    assert metrics["tasks_attempted"] == 2*row["steps"]
    assert metrics["commands_used"] <= metrics["commands_accepted"] <= metrics["commands_delivered"] <= metrics["tasks_attempted"]
    assert metrics["task_latency_p50_seconds"] <= metrics["task_latency_p95_seconds"] <= metrics["task_latency_p99_seconds"]
    assert metrics["action_order_known_pairs"] >= 0
    assert metrics["sum_path_costs"] == metrics["travel_steps"]+metrics["waiting_steps"]


def test_missed_physical_vertex_conflict_is_not_a_zero_error_safety_claim():
    from panama_uav.consistency import consistency_snapshot
    from panama_uav.information import InformationCache
    truth = {0: dict(position=(0, 0), path=[(0, 0), (0, 1)], priority=1, completed=False),
             1: dict(position=(0, 2), path=[(0, 2), (0, 1)], priority=2, completed=False)}
    def entry(peer, value, source=0, generation=1):
        cache = InformationCache()
        cache.accept(generation, value, 1, {peer: source})
        return cache
    stale = {**truth[1], "path": [(10, 10), (10, 11)]}
    states = {r: {0: entry(0, truth[0]), 1: entry(1, stale)} for r in truth}
    dt = {0: entry(0, {**truth[0], "priority": 3}), 1: entry(1, truth[1])}
    snapshot = consistency_snapshot({0: (0, 0), 1: (0, 2)}, truth, states, 2, radius=2, horizon=1, dt_states=dt)
    assert snapshot["zero_relevant_pair_case"] and snapshot["xi_fov"] == 0
    assert snapshot["missed_physical_conflicts"] == 2 and snapshot["physical_relevant_pairs"] == 2
    pair = snapshot["pairs"][0]
    assert pair["physical_vertex_conflict_steps"] == [1]
    assert pair["e2e_ordering_error"] == 0 and pair["uplink_ordering_error"] == pair["downlink_ordering_error"] == 1
    assert pair["self_age_seconds"] == pair["self_pre_age_seconds"]+pair["self_post_age_seconds"]


def test_censored_tasks_keep_cutoff_and_deadline_misses_and_energy_bounds():
    from panama_uav.resource_study import episode_metrics
    config = remote_config(drone_cycles_per_second=1e3, cloud_cycles_per_second=1e3,
        task_deadline_seconds=.5)
    torch.manual_seed(42)
    model = DQN((3, 15, 15), config.vector_length).eval()
    row, trace = run_episode(config, model, case_bank(config, 1, (2,))[0], torch.device("cpu"), max_steps=1)
    metrics = episode_metrics(row, trace)
    assert metrics["tasks_attempted"] == metrics["tasks_unfinished"] == metrics["command_cutoff_misses"] == 2
    assert metrics["deadline_censored_tasks"] == metrics["task_deadline_misses"] == 2
    assert metrics["commands_delivered"] == metrics["commands_used"] == 0
    assert metrics["modeled_energy_upper_j"] >= metrics["modeled_energy_lower_j"] > 0
