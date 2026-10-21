from functools import lru_cache

import pytest
import torch

from panama_uav.config import load_config
from panama_uav.evaluation import case_bank, run_episode
from panama_uav.models import DQN
from panama_uav.remote_state import ActionCommand
from panama_uav.remote_metrics import remote_metrics
from panama_uav.resource_study import episode_metrics
from panama_uav.task_deadlines import command_deadline_fields, command_deadline_metrics
from test_remote_execution import ROOT, environment, remote_config


@pytest.fixture(autouse=True)
def isolated_solver_templates(monkeypatch):
    import panama_uav.allocation as allocation
    monkeypatch.setattr(allocation, "_conic_template",
        lru_cache(maxsize=24)(allocation._conic_template.__wrapped__))


def deadline_environment():
    return environment(remote_config(drone_cycles_per_second=1e5, cloud_cycles_per_second=1e5,
        task_deadline_seconds=.1, enforce_task_deadlines=False))


def send_timed_command(comm, arrival):
    """Controlled response timing through the actual delivery/decision calendar."""
    comm.transport.advance(.05, comm._receive)
    comm.jobs["0:0"]["compute_completed_at"] = .05
    command = ActionCommand(0, 0, 3, .05, .1, 0, "drone", {0: 0.}, "0:0", "fixture", {}, {})
    comm._send_remote(0, "remote_action_command", dict(job_id="0:0", command=command.payload()),
        (f"process:{arrival-.05}",), version=0)


@pytest.mark.parametrize("source", [0., 1., 123.])
@pytest.mark.parametrize("latency,status", [(.099, "on_time"), (.100, "on_time"), (.101, "late")])
def test_deadline_boundaries_use_complete_arrival_since_sensing(source, latency, status):
    task = dict(job_id="a", source_time=source, deadline_at=source+.1, delivered_at=source+latency,
        compute_completed_at=source+.02, predicted_latency_seconds=.001)
    result = command_deadline_fields(task, source+.2)
    assert result["deadline_status"] == status
    assert result["deadline_met"] == (status == "on_time")
    assert result["task_deadline_missed"] == (status == "late")
    assert not result["deadline_censored"]


@pytest.mark.parametrize("arrival,on_time", [(.099, True), (.100, True), (.101, False)])
def test_actual_command_reception_and_cutoff_at_99_100_101_ms(arrival, on_time):
    env = deadline_environment()
    comm = env.communication
    send_timed_command(comm, arrival)
    env.prepare_decision(.1)
    assert comm.execution_actions[0] == (3 if on_time else 4)
    if not on_time:
        waiting = next(t for t in comm.task_records() if t["job_id"] == "0:0")
        assert waiting["deadline_status"] == "undelivered" and waiting["task_deadline_missed"]
        assert waiting["cutoff_induced_wait"]
        comm.transport.advance(.101, comm._receive)
    record = next(t for t in comm.task_records() if t["job_id"] == "0:0")
    assert record["observed_latency_seconds"] == pytest.approx(arrival)
    assert record["deadline_status"] == ("on_time" if on_time else "late")
    assert record["deadline_met"] == on_time
    assert record["cutoff_induced_wait"] == (not on_time)
    assert record["command_accepted"] == on_time
    assert record["command_cutoff_at"] == .1
    assert ("used_at" in record) == on_time


def test_unfinished_task_is_pending_before_deadline_and_missed_at_exact_cutoff():
    env = deadline_environment()
    comm = env.communication
    comm.transport.advance(.05, comm._receive)
    before = comm.task_records()
    assert all(t["deadline_pending"] and t["deadline_censored"] and t["deadline_met"] is None for t in before)
    assert all(not t["task_deadline_missed"] and t["cutoff_induced_wait"] is None for t in before)
    env.prepare_decision(.1)
    after = comm.task_records()
    assert all(t["deadline_status"] == "undelivered" and t["deadline_censored"] for t in after)
    assert all(t["task_deadline_missed"] and t["deadline_undelivered_overdue"] and not t["deadline_pending"] for t in after)
    assert all(t["cutoff_induced_wait"] and t["deadline_met"] is False for t in after)
    assert all("observed_latency_seconds" not in t for t in after)
    assert all(t["deadline_latency_lower_bound_seconds"] == .1 for t in after)
    assert sum(len(pool.jobs) for pool in comm.transport.compute_pools.values()) == 2
    metrics = command_deadline_metrics(after, .1)
    assert metrics["deadline_commands_undelivered"] == metrics["task_deadline_misses"] == metrics["deadline_censored_tasks"] == 2
    assert metrics["deadline_cutoff_wait_tasks"] == 2 and metrics["deadline_compliance_rate"] == 0


def test_late_arrival_transitions_out_of_censored_category_without_an_extra_miss():
    task = dict(job_id="a", source_time=0., deadline_at=.1, cutoff_induced_wait=True)
    censored = command_deadline_metrics([task], .1)
    late = command_deadline_metrics([{**task, "delivered_at": .101}], .2)
    assert censored["task_deadline_misses"] == late["task_deadline_misses"] == 1
    assert censored["deadline_commands_undelivered"] == censored["deadline_censored_tasks"] == 1
    assert late["deadline_commands_undelivered"] == late["deadline_censored_tasks"] == 0
    assert late["deadline_commands_late"] == late["deadline_cutoff_wait_tasks"] == 1


def test_duplicate_late_copy_cannot_overwrite_on_time_arrival_or_acceptance():
    env = deadline_environment()
    comm = env.communication
    send_timed_command(comm, .099)
    send_timed_command(comm, .110)
    env.prepare_decision(.1)
    comm.transport.advance(.110, comm._receive)
    tasks = comm.task_records()
    first = next(t for t in tasks if t["job_id"] == "0:0")
    assert first["delivered_at"] == pytest.approx(.099) and first["deadline_met"]
    assert first["command_accepted"] and first["command_arrival_count"] == 2
    assert not first["cutoff_induced_wait"]
    metrics = command_deadline_metrics(tasks, .110)
    assert metrics["deadline_commands_on_time"] == 1 and metrics["deadline_commands_late"] == 0
    assert metrics["deadline_commands_undelivered"] == 1
    assert metrics["task_deadline_misses"] == 1  # Only the other robot's undelivered task.
    aggregate = remote_metrics(dict(remote_events=comm.remote_events, resource_tasks=tasks,
        oracle_events=comm.oracle_events, transport_events=comm.transport.log,
        frames=[dict(time=.110)], remote_settings=dict(meters_per_cell=1.), outcome=dict(failure_reason=None)))
    assert aggregate["commands_delivered"] == 2
    assert aggregate["commands_delivered_unique"] == aggregate["duplicate_command_arrivals"] == 1
    assert aggregate["tasks_accepted"] == 1
    assert aggregate["deadline_commands_on_time"] == 1 and aggregate["deadline_commands_late"] == 0


def test_exclusive_categories_and_compliance_include_overdue_censored_tasks():
    tasks = [dict(job_id="on", source_time=0., deadline_at=.1, delivered_at=.099),
        dict(job_id="late", source_time=0., deadline_at=.1, delivered_at=.101, cutoff_induced_wait=True),
        dict(job_id="overdue", source_time=0., deadline_at=.1, cutoff_induced_wait=True),
        dict(job_id="pending", source_time=1., deadline_at=1.1),
        dict(job_id="legacy", source_time=0., deadline_at=None)]
    metrics = command_deadline_metrics(tasks, 1.05)
    assert metrics["deadline_tasks_monitored"] == 4
    assert (metrics["deadline_commands_on_time"], metrics["deadline_commands_late"], metrics["deadline_commands_undelivered"]) == (1, 1, 2)
    assert metrics["deadline_pending_tasks"] == metrics["deadline_undelivered_overdue_tasks"] == 1
    assert metrics["deadline_tasks_assessed"] == 3 and metrics["deadline_compliance_rate"] == pytest.approx(1/3)
    assert metrics["task_deadline_misses"] == metrics["deadline_censored_tasks"] == metrics["deadline_cutoff_wait_tasks"] == 2
    with pytest.raises(ValueError, match="one record"):
        command_deadline_metrics([tasks[0], tasks[0]], 1.05)


def test_monitor_is_opt_in_for_legacy_configurations_and_has_no_fake_compliance():
    for filename in ("base.toml", "paper_study.toml", "resource_study.toml"):
        assert load_config(ROOT/"configs"/filename).resource_allocation.task_deadline_seconds is None
    paper = load_config(ROOT/"configs/paper_remote_dt.toml").resource_allocation
    assert paper.task_deadline_seconds == paper.decision_budget_seconds == paper.command_cutoff_seconds == .1
    assert not paper.enforce_task_deadlines
    metrics = command_deadline_metrics([dict(job_id="old", source_time=0.)], 1.)
    assert metrics["deadline_tasks_monitored"] == metrics["task_deadline_misses"] == 0
    assert metrics["deadline_commands_on_time"] is metrics["deadline_commands_late"] is metrics["deadline_compliance_rate"] is None


def test_measured_deadline_preserves_real_and_oracle_actions_and_unconstrained_allocation():
    outputs = []
    for deadline in (None, .1):
        config = remote_config(drone_cycles_per_second=1e11, task_deadline_seconds=deadline,
            enforce_task_deadlines=False)
        torch.manual_seed(42)
        model = DQN((3, 15, 15), config.vector_length).eval()
        row, trace = run_episode(config, model, case_bank(config, 1, (2,))[0], torch.device("cpu"), max_steps=3)
        allocations = [e for e in trace["transport_events"] if e["event"] == "paper_allocation"]
        assert all(t["deadline_seconds"] is None for e in allocations for t in e["snapshot"]["tasks"])
        outputs.append((trace["remote_events"], trace["outcome"], [e for e in trace["oracle_events"]
            if e["event"] in ("action_shadow", "decision_action_shadow")]))
        metrics = episode_metrics(row, trace)
        if deadline is not None:
            assert metrics["deadline_tasks_monitored"] == metrics["tasks_attempted"]
            assert sum(metrics[k] for k in ("deadline_commands_on_time", "deadline_commands_late", "deadline_commands_undelivered")) == metrics["tasks_attempted"]
            assert metrics["commands_delivered_unique"] == metrics["commands_delivered"] and metrics["duplicate_command_arrivals"] == 0
    assert outputs[0] == outputs[1]
