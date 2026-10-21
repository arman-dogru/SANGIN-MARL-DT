from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from panama_uav.config import ExperimentMode, load_config
from panama_uav.environment import ExperimentEnvironment
from panama_uav.evaluation import case_bank, run_episode
from panama_uav.models import DQN

ROOT = Path(__file__).resolve().parents[1]


def remote_config(**parameters):
    config = load_config(ROOT/"configs/base.toml")
    values = dict(enabled=True, execution_mode="remote_dt", cycles_per_bit=10, energy_coefficient=0)
    values.update(parameters)
    return replace(config, communication=replace(config.communication, profile="physical",
        stale_information=("task", "intentions", "priorities"), handover_enabled=False,
        contention_enabled=False), resource_allocation=replace(config.resource_allocation,
        **values))


def environment(config=None, count=2, calls=None):
    env = ExperimentEnvironment(config or remote_config(drone_cycles_per_second=1e11),
        ExperimentMode.UAV, "maps/random-32-32-10.map", count, 250, 123)
    def infer(location, agent, observation):
        if calls is not None:
            calls.append(dict(location=location, agent=agent, time=env.communication.transport.now,
                              observation=observation))
        return 4
    env.remote_inference = infer
    env.reset()
    return env


@pytest.mark.parametrize("location,capacity", [("drone", 1e11), ("cloud", 1e6)])
def test_dqn_runs_at_selected_processor_after_upload_and_compute(location, capacity):
    calls = []
    env = environment(remote_config(drone_cycles_per_second=capacity), calls=calls)
    assert not calls
    env.prepare_decision(.1)
    assert calls and {c["location"] for c in calls} == {location}
    comm = env.communication
    for job in comm.jobs.values():
        assert job["access_arrived_at"] <= job["allocation_at"] <= job["compute_arrived_at"]
        assert job["compute_arrived_at"] <= job["compute_completed_at"] <= job["delivered_at"] <= .1
        assert comm.command_contexts[job["agent"]]["command"]["location"] == location
        if location == "cloud":
            assert job["backhaul_energy_j"] > 0
    assert all(c["arrived_at"] <= c["cutoff_time"] for c in comm.command_contexts.values())


def test_one_arrival_is_allocated_without_waiting_for_seven_reports():
    env = environment(count=8)
    for agent in range(1, 8):
        env.communication.transport.agent_rate_caps[(agent, "access_up")] = 1000
    env.communication.transport.advance(.003, env.communication._receive)
    events = [e for e in env.communication.transport.log if e["event"] == "paper_allocation"]
    assert len(events) == 1 and events[0]["new_job"] == "0:0"
    assert len([j for j in env.communication.jobs.values() if "access_arrived_at" in j]) == 1


def test_transmitted_sensing_cannot_read_later_physical_positions():
    calls = []
    env = environment(calls=calls)
    original = {a: {k: v.copy() for k, v in o.items()} for a, o in env.observations().items()}
    env.agents[1].position = (0, 0)  # No new upload: changing physical truth is not a DT input.
    env.prepare_decision(.1)
    for call in calls:
        for key in ("spatial", "vector"):
            np.testing.assert_array_equal(call["observation"][key], original[call["agent"]][key])


def test_timeout_wait_and_late_command_never_replays():
    env = environment(remote_config(drone_cycles_per_second=1e5, cloud_cycles_per_second=1e5))
    env.prepare_decision(.1)
    assert env.communication.execution_actions == {0: 4, 1: 4}
    assert all(c["command_cutoff_missed"] for c in env.communication.command_contexts.values())
    env.communication.transport.advance(25, env.communication._receive)
    arrivals = [e for e in env.communication.remote_events if e["event"] == "command_arrived"]
    assert arrivals and all(not e["accepted"] and e["reason"] == "rejected_obsolete" for e in arrivals)


def test_evaluation_never_calls_local_action_selection(monkeypatch):
    import panama_uav.evaluation as evaluation
    def forbidden(*args, **kwargs):
        raise AssertionError("Local inference invoked in remote mode")
    monkeypatch.setattr(evaluation, "select_actions", forbidden)
    config = remote_config(drone_cycles_per_second=1e11)
    model = DQN((3, 15, 15), config.vector_length).eval()
    case = case_bank(config, cases=1, counts=(2,))[0]
    row, trace = run_episode(config, model, case, torch.device("cpu"), max_steps=2)
    assert row["execution_mode"] == "remote_dt"
    assert any(e["event"] == "remote_inference" for e in trace["remote_events"])
    assert all(not step["observations"] for step in trace["allocation_loop"])


def test_new_configuration_is_separate_and_full_runs_are_gated(tmp_path):
    from panama_uav.resource_study import conditions, run_study
    cfg = load_config(ROOT/"configs/paper_remote_dt.toml")
    assert cfg.resource_allocation.execution_mode == "remote_dt"
    assert cfg.resource_allocation.task_deadline_seconds is None
    assert load_config(ROOT/"configs/paper_study.toml").resource_allocation.execution_mode == "support"
    assert len(conditions(cfg)) == 6
    assert "up-1-cpu-1__equal_legacy" in {name for name, _, _ in conditions(cfg)}
    with pytest.raises(ValueError, match="acceptance approval"):
        run_study(cfg, tmp_path/"must-not-start")
    assert not (tmp_path/"must-not-start").exists()


def test_stale_dt_priority_changes_offload_but_oracle_is_only_shadow():
    config = remote_config(drone_cycles_per_second=1e9, cycles_per_bit=1000, energy_coefficient=1)
    config = replace(config, communication=replace(config.communication,
        one_way_propagation_seconds=.035, satellite_uplink_mbps=10))
    env = environment(config)
    comm = env.communication
    comm.replicas["drone"].states[0].value["priority"] = 10
    comm.replicas["drone"].states[1].value["priority"] = 1
    comm.truth[0]["priority"], comm.truth[1]["priority"] = 1, 10
    comm.transport.advance(.003, comm._receive)
    actual = next(e for e in comm.transport.log if e["event"] == "paper_allocation")
    shadow = comm.oracle_events[0]
    assert actual["resource_weights"][0] == 1 and shadow["physical_weights"][0] == 4
    assert comm.jobs["0:0"]["executed_x"] == 1, actual["snapshot"]
    assert shadow["oracle"]["tasks"][0]["x"] == 0
    assert shadow["truth_scored_regret"] == pytest.approx(.02, abs=1e-8)


def test_shadow_toggle_cannot_change_actual_events_or_commands():
    outputs = []
    for shadow in (False, True):
        env = environment(remote_config(drone_cycles_per_second=1e11, oracle_shadow=shadow))
        env.prepare_decision(.1)
        outputs.append((env.communication.remote_events, env.communication.execution_actions))
    assert outputs[0] == outputs[1]


def test_reference_scoring_uses_identical_truth_coefficients():
    from panama_uav.allocation import AllocationTask, AllocationSnapshot, score_allocation, solve_binary
    tasks = (AllocationTask("a", 1e5, 1000, 4, 50e6, 10e6, fixed_x=1),
             AllocationTask("b", 1e5, 1000, 1, 50e6, 10e6, fixed_x=1))
    actual = AllocationSnapshot(tasks, energy_coefficient=0)
    truth = replace(actual, tasks=(replace(tasks[0], weight=1), replace(tasks[1], weight=4)))
    actual_solution, oracle = solve_binary(actual), solve_binary(truth)
    assert actual_solution.objective == pytest.approx(oracle.objective)  # Native subtraction would be wrong.
    scored = score_allocation(truth, actual_solution.tasks)
    assert scored.objective-oracle.objective == pytest.approx(.45)


def corridor_environment(actions, response="terminate", completed=False):
    from panama_uav.scenarios import Scenario, load_map
    cfg = remote_config(drone_cycles_per_second=1e11, on_collision=response)
    cfg = replace(cfg, communication=replace(cfg.communication, profile="perfect"))
    grid = load_map(cfg.resolve("maps/random-32-32-10.map"))
    row, col = next((r, c) for r in range(1, 30) for c in range(1, 28)
                    if not grid[r-1:r+2, c:c+4].any())
    starts = ((row, col), (row, col+2))
    goals = (starts[0] if completed else (row, col+3), (row, col))
    scenario = Scenario(Path("fixture.scen"), cfg.resolve("maps/random-32-32-10.map"), starts, goals)
    env = ExperimentEnvironment(cfg, ExperimentMode.UAV, "maps/random-32-32-10.map", 2, 250, 123, scenario)
    env.remote_inference = lambda location, robot, observation: actions[robot]
    env.reset()
    return env


def test_physical_vertex_collision_terminates_without_an_oracle_winner():
    env = corridor_environment({0: 3, 1: 2})
    env.prepare_decision(.1)
    before = [a.position for a in env.agents]
    result = env.step(env.communication.execution_actions.copy())
    assert result.terminated and result.info["failure_reason"] == "robot_collision"
    assert [a.position for a in env.agents] == before
    events = [e for e in env.communication.remote_events if e["event"] == "physical_collision"]
    assert len(events) == 1 and events[0]["conflicts"][0]["kind"] == "vertex"


def test_optional_conflict_resolution_uses_delivered_dt_priorities():
    env = corridor_environment({0: 3, 1: 2}, "dt_resolve")
    env.prepare_decision(.1)
    before = [a.position for a in env.agents]
    # Force the delivered order to disagree with current physical priorities.
    env.communication.command_contexts[0]["resolution_priority"] = 100
    env.communication.command_contexts[1]["resolution_priority"] = 0
    env.agents[0].priority, env.agents[1].priority = 0, 100
    result = env.step(env.communication.execution_actions.copy())
    assert result.info["physical_conflicts"] and env.failure_reason is None
    assert env.agents[0].position == before[0]
    assert env.agents[1].position == (before[1][0], before[1][1]-1)


def test_edge_swap_logged_and_resolution_handles_stationary_cascades():
    from panama_uav.remote_control import conflicts, resolve_delivered
    before = {0: (1, 1), 1: (1, 2), 2: (1, 3)}
    swap = {0: (1, 2), 1: (1, 1), 2: (1, 3)}
    assert conflicts(before, swap)[0]["kind"] == "edge_swap"
    assert resolve_delivered(before, swap, {0: 1, 1: 2, 2: 3}) == before
    cascade = {0: (1, 2), 1: (1, 3), 2: (1, 3)}
    assert resolve_delivered(before, cascade, {0: 1, 1: 2, 2: 3}) == before


def test_completed_robot_yields_returns_and_never_gets_new_compute_tasks():
    env = corridor_environment({0: 4, 1: 4}, completed=True)
    original = env.agents[0].position
    env.prepare_decision(.1)
    assert env.communication.execution_actions[0] != 4
    assert env.communication.command_contexts[0]["completed_local_control"]
    env.step(env.communication.execution_actions.copy())
    assert env.agents[0].position != original and env.agents[0].ever_reached_goal
    assert not any(j["agent"] == 0 for j in env.communication.jobs.values())
    # Peer intentions no longer occupy the goal; local sensing permits return.
    cache = env.communication.states[0][1]
    cache.value["path"] = [env.agents[1].position]
    env.prepare_decision(.1)
    env.step(env.communication.execution_actions.copy())
    assert env.agents[0].position == original
    final = [e for e in env.communication.remote_events
             if e["event"] == "message_sent" and e["kind"] == "remote_completion"]
    assert len(final) == 1 and env.agents[0].first_completion_time == 0


def test_remote_motion_cannot_bypass_command_cutoff():
    env = environment()
    with pytest.raises(RuntimeError, match="Prepare"):
        env.step({0: 4, 1: 4})
    env.prepare_decision(.1)
    with pytest.raises(ValueError, match="delivered"):
        env.step({0: 3, 1: 3})


def test_task_deadline_monitoring_is_separate_from_command_cutoff():
    env = environment(remote_config(drone_cycles_per_second=1e5, cloud_cycles_per_second=1e5,
        task_deadline_seconds=.5))
    env.prepare_decision(.1)
    records = env.communication.task_records()
    assert all(r["task_deadline_enabled"] and not r["task_deadline_missed"] for r in records)
    assert all(c["command_cutoff_missed"] for c in env.communication.command_contexts.values())
    env.communication.transport.advance(1, env.communication._receive)
    assert all(r["task_deadline_missed"] and r["deadline_censored"] for r in env.communication.task_records())
