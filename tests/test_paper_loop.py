from dataclasses import replace

import numpy as np
import pytest

from panama_uav.allocation import AllocationTask, AllocationSnapshot, compare_allocations
from panama_uav.consistency import consistency_snapshot
from panama_uav.information import InformationCache
from panama_uav.validation import independent_primal


def state(priority, position):
    c = InformationCache()
    c.accept(0, dict(priority=priority, position=position, path=[position, (0, 0)], completed=False), 0, {0: 0})
    return c


def test_uplink_and_downlink_errors_can_cancel():
    truth = {0: dict(priority=1), 1: dict(priority=2)}
    dt = {0: state(3, (0, 0)), 1: state(2, (0, 1))}
    states = {0: {0: state(1, (0, 0)), 1: state(2, (0, 1))},
              1: {0: state(1, (0, 0)), 1: state(2, (0, 1))}}
    result = consistency_snapshot({0: (0, 0), 1: (0, 1)}, truth, states, 1, dt_states=dt)
    assert result["fov_errors"] == 0
    assert result["fov_uplink_errors"] == result["fov_downlink_errors"] == 2
    for p in result["pairs"]:
        assert p["e2e_ordering_error"] <= p["uplink_ordering_error"]+p["downlink_ordering_error"]


@pytest.mark.parametrize("energy", [.1, 1., 10.])
def test_vectorized_dvfs_matches_independent_enumeration(energy):
    rng = np.random.default_rng(22)
    tasks = tuple(AllocationTask(str(i), float(rng.uniform(5e4, 2e5)), 1000,
        float(rng.uniform(1, 4)), 50e6, 10e6) for i in range(4))
    snapshot = AllocationSnapshot(tasks, drone_kappa=1e-28,
        energy_coefficient=energy, include_robot_upload_in_objective=False)
    comparison = compare_allocations(snapshot)
    direct = independent_primal(snapshot)
    assert comparison["binary"]["objective"] == pytest.approx(direct["objective"], rel=1e-6)
    assert comparison["binary"]["assignments_checked"] == 16
