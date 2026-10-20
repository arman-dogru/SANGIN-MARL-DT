from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from panama_uav.config import ExperimentMode, load_config
from panama_uav.environment import ExperimentEnvironment
from panama_uav.information import InformationCache
from panama_uav.consistency import consistency_snapshot
from panama_uav.paper_radio import access_budget, backhaul_budget
from panama_uav.transport import Transport

ROOT = Path(__file__).resolve().parents[1]


def config(**resource):
    c = load_config(ROOT/"configs/base.toml")
    return replace(c, communication=replace(c.communication, profile="physical",
        stale_information=("task", "intentions", "priorities"), handover_enabled=False,
        contention_enabled=False), resource_allocation=replace(c.resource_allocation,
            enabled=True, **resource))


def environment(c=None):
    c = c or config()
    return ExperimentEnvironment(c, ExperimentMode.UAV, "maps/random-32-32-10.map", 2, 250, 123)


def test_mission_executes_and_accounts_for_both_transmitters():
    env = environment()
    env.reset()
    env.step({0: 4, 1: 4})
    env.prepare_decision(.1)
    env.step({0: 4, 1: 4})
    comm = env.communication
    assert comm.allocations >= 1
    delivered = [j for j in comm.jobs.values() if "delivered_at" in j]
    assert delivered
    for j in delivered:
        assert j["executed_x"] in (0, 1)
        assert j["robot_energy_j"] > 0
        if j["executed_x"] == 0:
            assert j["backhaul_energy_j"] > 0
            assert j["uplink_propagation_seconds"] == pytest.approx(.025)
            assert j["response_seconds"] >= .025
        assert j["observed_latency_seconds"] >= j["compute_elapsed_seconds"]
    for receiver, caches in comm.states.items():
        for peer, cached in caches.items():
            assert comm.caches[receiver]["priorities"].value[peer] == cached.value["priority"]
            assert comm.caches[receiver]["intentions"].value[peer]["path"] == cached.value["path"]
    assert env.communication.snapshot()["consistency"]["directed_pairs"] == 2


def test_out_of_order_state_cannot_roll_back_source():
    env = environment()
    env.reset()
    comm = env.communication
    state = dict(comm.states[0][1].value)
    def send(source, generation, priority):
        state["priority"] = priority
        comm.transport.send(0, "paper_state", dict(job_id="test", peer=1,
            source_time=source, state=dict(state)), generated_at=generation, version=int(source))
        comm.transport.advance(comm.transport.now, comm._receive)
    send(2, 3, 4)
    send(1, 4, 9)  # later computation of older source data must also be rejected
    assert comm.states[0][1].value["priority"] == 4
    assert comm.states[0][1].source_timestamps == {1: 2}


def test_completed_priority_below_unreachable_and_completed_weight_floor():
    env = environment()
    env.reset()
    env.agents[0].ever_reached_goal = True
    env.agents[1].distance_to_goal = float("inf")
    env._refresh_priorities()
    assert env.agents[0].priority > env.agents[1].priority
    reports = {a.id: env.communication._report(a) for a in env.agents}
    assert env.communication._weights(reports, {0: 99, 1: 0})[0] == 1


def test_legacy_default_and_atomic_selection_validation():
    assert not load_config(ROOT/"configs/base.toml").resource_allocation.enabled
    c = config()
    c = replace(c, communication=replace(c.communication, stale_information=("priorities",)))
    with pytest.raises(ValueError, match="atomically"):
        environment(c).reset()
    c = config()
    c = replace(c, communication=replace(c.communication, profile="controlled"))
    with pytest.raises(ValueError, match="physical radio"):
        environment(c).reset()


def test_budget_and_rf_units():
    c = config()
    radio = access_budget(80, c.resource_allocation, 50e6)
    assert 0 < radio["applied_bps"] <= 50e6
    lower = access_budget(90, c.resource_allocation, 50e6)
    assert lower["rate_bound_bps"] < radio["rate_bound_bps"]
    b = backhaul_budget(Transport(c.communication), c.resource_allocation)
    assert b["round_trip_seconds"] == pytest.approx(.05)
    assert b["air_to_space_one_way_seconds"] == pytest.approx(550/299792.458)
    assert b["terminal_eirp_dbw"] == pytest.approx(
        10*np.log10(2)+b["implied_antenna_gain_dbi"])
    assert b["one_way_seconds"] == pytest.approx(b["air_to_space_one_way_seconds"]
        +b["satellite_gateway_one_way_seconds"]+b["ground_one_way_seconds"])


def cache(path, priority, source=0):
    c = InformationCache()
    c.accept(source, dict(position=path[0], path=path, priority=priority, completed=False),
             0, {0: source})
    return c


def test_directed_fov_vertex_edge_stale_self_and_zero_denominator():
    truth = {0: dict(priority=1), 1: dict(priority=2)}
    states = {0: {0: cache([(0, 0), (0, 1)], 3), 1: cache([(0, 1), (0, 0)], 2)},
              1: {0: cache([(0, 0), (0, 1)], 1), 1: cache([(0, 1), (0, 0)], 2)}}
    snapshot = consistency_snapshot({0: (0, 0), 1: (0, 1)}, truth, states, 2, radius=1, horizon=1)
    assert snapshot["relevant_pairs"] == 2
    assert snapshot["fov_errors"] == 1 and snapshot["xi_fov"] == .5
    assert snapshot["pairs"][0]["edge_conflict_steps"] == [0]
    assert snapshot["xi_aoi"] == 1
    far = consistency_snapshot({0: (20, 20), 1: (20, 21)}, truth, states, 2, radius=1)
    assert far["relevant_pairs"] == far["xi_fov"] == 0
    states[0][1] = cache([(0, 1), (0, 1)], 2, source=1)
    boundary = consistency_snapshot({0: (0, 0), 1: (20, 21)}, truth, states, 2, radius=1, horizon=1)
    assert boundary["pairs"][0]["vertex_conflict_steps"] == [1]


def test_perfect_paper_reference_is_fresh():
    c = config()
    c = replace(c, communication=replace(c.communication, profile="perfect"))
    env = environment(c)
    env.reset()
    env.step({0: 4, 1: 4})
    env.prepare_decision(.1)
    snapshot = env.communication.snapshot()
    assert snapshot["consistency"]["fov_errors"] == 0
    assert not env.communication.jobs
    assert all(age == 0 for fields in env.communication.ages().values() for age in fields.values())


def test_overloaded_pools_account_for_existing_work_without_oversubscription():
    env = environment(config(drone_cycles_per_second=3e7, cloud_cycles_per_second=3e7))
    env.reset()
    for _ in range(4):
        env.step({0: 4, 1: 4})
        env.prepare_decision(.1)
    comm = env.communication
    events = comm.transport.log
    assert any(len(pool.jobs) > 1 for pool in comm.transport.compute_pools.values())
    for event in events:
        if event["event"] == "compute_allocation":
            assert sum(j["cycles_per_second"] for j in event["jobs"]) <= event["capacity_cycles_per_second"]*(1+1e-10)
    allocations = [e for e in events if e["event"] == "paper_allocation"]
    assert any(any(t["fixed_x"] is not None for t in a["snapshot"]["tasks"]) for a in allocations)


def test_self_completion_uses_the_cached_state_in_paper_mode():
    env = environment()
    env.reset()
    env.agents[0].ever_reached_goal = True
    assert env.observation(env.agents[0])["vector"][1] == 0


def test_fragmented_backhaul_propagation_is_not_added_for_each_chunk():
    c = config(task_input_bits=1000000, energy_coefficient=0,
        drone_cycles_per_second=1e7, cloud_cycles_per_second=1e10)
    c = replace(c, communication=replace(c.communication, packet_payload_bytes=10000))
    env = environment(c)
    env.reset()
    env.step({0: 4, 1: 4})
    env.communication.transport.advance(3, env.communication._receive)
    for job in env.communication.jobs.values():
        assert job["executed_x"] == 0
        assert job["uplink_propagation_seconds"] == pytest.approx(.025)
        assert job["backhaul_energy_j"] > .2  # one million logical bits plus wire overhead
