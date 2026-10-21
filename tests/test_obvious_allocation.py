"""Hand-calculated allocation outcomes, independent of optimization helpers."""
from dataclasses import replace

import numpy as np
import pytest

from panama_uav.allocation import AllocationTask, AllocationSnapshot, compare_allocations


def work(task_id="a", **kwargs):
    return replace(AllocationTask(task_id, 1e5, 1000, 1, 50e6, 10e6), **kwargs)


@pytest.mark.parametrize("round_trip,expected_x", [(0, 0), (.079, 0), (.081, 1), (5, 1)])
def test_single_task_switches_at_hand_calculated_80ms(round_trip, expected_x):
    # C=1e8 cycles: drone 0.1 s, cloud 0.01 s plus 0.01 s backhaul.
    # The common 0.002 s robot upload cancels: cloud wins below 0.08 s RTT.
    snapshot = AllocationSnapshot((work(propagation_round_trip_seconds=round_trip),),
        energy_coefficient=0)
    result = compare_allocations(snapshot)
    expected_objective = min(.102, .022+round_trip)
    for mode in ("binary", "relaxed"):
        solved = result[mode]
        assert solved["tasks"][0]["x"] == pytest.approx(expected_x, abs=1e-6)
        assert solved["objective"] == pytest.approx(expected_objective, abs=1e-7)
    selected = result["binary"]["tasks"][0]
    assert selected["drone_share" if expected_x else "cloud_share"] == pytest.approx(1)


def test_identical_processors_balance_four_equal_jobs():
    tasks = tuple(work(str(i), backhaul_bps=1e12, propagation_round_trip_seconds=0)
                  for i in range(4))
    snapshot = AllocationSnapshot(tasks, drone_cycles_per_second=1e9,
        cloud_cycles_per_second=1e9, energy_coefficient=0)
    result = compare_allocations(snapshot)
    binary = result["binary"]
    # If k jobs use the cloud, compute cost is 0.1*(k^2+(4-k)^2), minimized at k=2.
    assert sum(t["x"] == 0 for t in binary["tasks"]) == 2
    for t in binary["tasks"]:
        assert t["drone_share" if t["x"] else "cloud_share"] == pytest.approx(.5)
    assert binary["objective"] == pytest.approx(.008+.8+2e-7, abs=1e-10)
    assert result["relaxed"]["objective"] == pytest.approx(binary["objective"], abs=1e-7)


def test_fourfold_priority_gets_twice_the_cpu_for_equal_work():
    tasks = (replace(work("high", fixed_x=1), weight=4), work("low", fixed_x=1))
    result = compare_allocations(AllocationSnapshot(tasks, energy_coefficient=0))
    # sqrt(4):sqrt(1) = 2:1; weighted latency is .6+.3+.01 = .91 s.
    for mode in ("binary", "relaxed"):
        np.testing.assert_allclose([t["drone_share"] for t in result[mode]["tasks"]],
            [2/3, 1/3], atol=2e-5)
        assert result[mode]["objective"] == pytest.approx(.91, abs=1e-7)
