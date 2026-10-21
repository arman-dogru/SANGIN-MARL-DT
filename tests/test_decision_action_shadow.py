from functools import lru_cache

import pytest
import torch

from panama_uav.evaluation import case_bank, run_episode
from panama_uav.models import DQN
from panama_uav.remote_state import observation_hash
from panama_uav.resource_study import episode_metrics
from test_remote_execution import corridor_environment, environment, remote_config


@pytest.fixture(autouse=True)
def isolated_solver_templates(monkeypatch):
    # These tests change CPU scales to force missed cutoffs. Keep their numeric
    # warm starts local, without disturbing other allocation regression cases.
    import panama_uav.allocation as allocation
    monkeypatch.setattr(allocation, "_conic_template",
        lru_cache(maxsize=24)(allocation._conic_template.__wrapped__))


def test_every_active_robot_gets_an_epoch_oracle_when_no_job_has_completed():
    env = environment(remote_config(drone_cycles_per_second=1e5, cloud_cycles_per_second=1e5))
    calls = []
    def infer(location, robot, observation):
        calls.append((location, robot, env.communication.transport.now))
        return 3
    env.communication.shadow_inference = infer
    for cycle in range(3):
        env.prepare_decision(.1)
        assert env.communication.execution_actions == {0: 4, 1: 4}
        env.step(env.communication.execution_actions.copy())
    shadows = [e for e in env.communication.oracle_events if e["event"] == "decision_action_shadow"]
    assert len(shadows) == len(calls) == 6
    assert not any(e["event"] == "action_shadow" for e in env.communication.oracle_events)
    assert {(e["cycle"], e["robot_id"]) for e in shadows} == {(n, r) for n in range(3) for r in range(2)}
    for event in shadows:
        assert event["time"] == event["cutoff_time"] == pytest.approx(event["cycle"]+.1)
        assert event["source_min"] == event["cutoff_time"]
        assert set(event["sources"].values()) == {event["cutoff_time"]}
        assert event["actual_action"] == 4 and event["oracle_action"] == 3 and event["disagrees"]
        assert event["command_cutoff_missed"] and event["timeout_reason"] == "no_current_command"
        assert event["actual_command_job_id"] is None and event["actual_command_arrived_at"] is None
        assert event["observation_hash"] == observation_hash(event["observation"])
    assert [time for _, _, time in calls] == [e["cutoff_time"] for e in shadows]


def test_cutoff_oracle_is_recomputed_instead_of_reusing_completion_action():
    env = environment()
    env.communication.shadow_inference = lambda location, robot, observation: (
        0 if env.communication.transport.now < .05 else 3)
    env.prepare_decision(.1)
    completion = [e for e in env.communication.oracle_events if e["event"] == "action_shadow"]
    decisions = [e for e in env.communication.oracle_events if e["event"] == "decision_action_shadow"]
    assert len(completion) == len(decisions) == 2
    assert all(e["oracle_action"] == 0 and e["time"] < .05 for e in completion)
    assert all(e["oracle_action"] == 3 and e["time"] == .1 for e in decisions)
    assert all(not e["command_cutoff_missed"] and e["actual_command_arrived_at"] <= .1 for e in decisions)
    assert all(e["actual_action"] == env.communication.execution_actions[e["robot_id"]] for e in decisions)


def test_completed_robot_retains_local_control_and_has_no_epoch_dqn_call():
    env = corridor_environment({0: 4, 1: 4}, completed=True)
    calls = []
    def infer(location, robot, observation):
        calls.append(robot)
        return 3
    env.communication.shadow_inference = infer
    env.prepare_decision(.1)
    assert calls == [1]
    epochs = [e for e in env.communication.oracle_events if e["event"] == "decision_action_shadow"]
    local = [e for e in env.communication.oracle_events if e["event"] == "local_control_shadow"]
    assert [e["robot_id"] for e in epochs] == [1] and [e["robot_id"] for e in local] == [0]


def test_oracle_cannot_change_timeout_trajectory_actual_events_or_shared_compute():
    outputs = []
    for enabled in (False, True):
        env = environment(remote_config(drone_cycles_per_second=1e5, cloud_cycles_per_second=1e5,
            oracle_shadow=enabled))
        env.communication.shadow_inference = lambda location, robot, observation: 3
        for _ in range(3):
            env.prepare_decision(.1)
            env.step(env.communication.execution_actions.copy())
        comm = env.communication
        outputs.append((comm.remote_events, comm.execution_actions, [a.position for a in env.agents],
                        {name: pool.snapshot() for name, pool in comm.transport.compute_pools.items()}))
        assert sum(e["event"] == "decision_action_shadow" for e in comm.oracle_events) == (6 if enabled else 0)
    assert outputs[0] == outputs[1]


def test_epoch_action_metrics_include_timeout_waits_without_completion_comparisons():
    config = remote_config(drone_cycles_per_second=1e5, cloud_cycles_per_second=1e5)
    torch.manual_seed(42)
    model = DQN((3, 15, 15), config.vector_length).eval()
    row, trace = run_episode(config, model, case_bank(config, 1, (2,))[0], torch.device("cpu"), max_steps=3)
    metrics = episode_metrics(row, trace)
    epochs = [e for e in trace["oracle_events"] if e["event"] == "decision_action_shadow"]
    assert metrics["action_shadow_comparisons"] == 0
    assert metrics["decision_action_shadow_comparisons"] == metrics["decision_action_shadow_cutoff_miss_comparisons"] == 6
    assert metrics["command_cutoff_misses"] == 6
    disagreement = sum(e["actual_action"] != e["oracle_action"] for e in epochs)
    assert metrics["decision_action_shadow_disagreements"] == metrics["decision_action_shadow_cutoff_miss_disagreements"] == disagreement
    assert metrics["decision_action_shadow_unnecessary_wait_steps"] == disagreement
    assert metrics["decision_action_shadow_disagreement_rate"] == pytest.approx(disagreement/6)
