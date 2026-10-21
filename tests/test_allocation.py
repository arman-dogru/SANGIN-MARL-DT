from dataclasses import replace

import numpy as np
import pytest

from panama_uav.allocation import (AllocationTask, AllocationSnapshot, AllocationFailure,
    compare_allocations, solve_binary, solve_relaxed, priority_weights)


def task(name="a", **kwargs):
    return AllocationTask(**dict(dict(task_id=name, input_bits=1e5, cycles_per_bit=1000,
        weight=1.0, access_bps=50e6, backhaul_bps=10e6), **kwargs))


@pytest.mark.parametrize("n", [1, 2, 4, 6, 8])
def test_relaxation_is_a_bound_and_capacity_feasible(n):
    snapshot = AllocationSnapshot(tuple(task(str(i), weight=1+i/3) for i in range(n)))
    result = compare_allocations(snapshot)
    assert result["absolute_gap"] >= -1e-6
    relaxed = result["relaxed"]
    assert relaxed["is_dcp"] and relaxed["residual"] < 1e-6
    rows = relaxed["tasks"]
    assert sum(r["drone_share"] for r in rows) <= 1+1e-7
    assert sum(r["cloud_share"] for r in rows) <= 1+1e-7
    for r in rows:
        assert 0 <= r["drone_normalized_rate"] <= 1+1e-7
        assert 0 <= r["cloud_normalized_rate"] <= 1+1e-7


@pytest.mark.parametrize("x", [0, 1])
def test_fixed_assignment_matches_closed_form(x):
    tasks = tuple(task(str(i), input_bits=1e5*(i+1), weight=i+1, fixed_x=x) for i in range(3))
    snapshot = AllocationSnapshot(tasks)
    exact, relaxed = solve_binary(snapshot), solve_relaxed(snapshot)
    field = "drone_share" if x else "cloud_share"
    expected = np.sqrt([t.cycles*t.weight for t in tasks])
    expected /= expected.sum()
    np.testing.assert_allclose([r[field] for r in exact.tasks], expected)
    np.testing.assert_allclose([r[field] for r in relaxed.tasks], expected, atol=2e-5)
    assert relaxed.objective == pytest.approx(exact.objective, rel=1e-6)


def test_units_and_backhaul_energy():
    s = AllocationSnapshot((task(fixed_x=0),))
    row = solve_binary(s).tasks[0]
    assert row["access_seconds"] == pytest.approx(.002)
    assert row["backhaul_seconds"] == pytest.approx(.01)
    assert row["cloud_compute_seconds"] == pytest.approx(.01)
    assert row["propagation_seconds"] == pytest.approx(.05)
    assert row["latency_seconds"] == pytest.approx(.072)
    assert row["robot_energy_j"] == pytest.approx(.0004)
    assert row["backhaul_energy_j"] == pytest.approx(.02)


def test_optional_deadlines_change_cpu_rates_in_both_references():
    tasks = (task("a", fixed_x=1, deadline_seconds=.122),
             task("b", weight=16, fixed_x=1))
    snapshot = AllocationSnapshot(tasks, energy_coefficient=0)
    for solution in (solve_binary(snapshot), solve_relaxed(snapshot)):
        assert solution.tasks[0]["latency_seconds"] <= .122+1e-6
        assert solution.tasks[0]["drone_share"] == pytest.approx(5/6, abs=2e-5)
    unconstrained = solve_binary(replace(snapshot, tasks=(replace(tasks[0], deadline_seconds=None), tasks[1])))
    assert unconstrained.tasks[0]["latency_seconds"] > .122


def test_deadline_binary_enumeration_and_fractional_bound():
    snapshot = AllocationSnapshot((task("a", deadline_seconds=.11), task("b", deadline_seconds=.11)))
    result = compare_allocations(snapshot)
    assert result["absolute_gap"] >= 0 and result["binary"]["assignments_checked"] == 4
    for solve in (solve_binary, solve_relaxed):
        with pytest.raises(AllocationFailure):
            solve(replace(snapshot, tasks=(task(deadline_seconds=.001),)))


@pytest.mark.parametrize("mode", [solve_binary, solve_relaxed])
def test_unavailable_and_zero_work_routes(mode):
    edge = mode(AllocationSnapshot((task(backhaul_bps=0),), cloud_cycles_per_second=0))
    assert edge.tasks[0]["x"] == 1
    cloud = mode(AllocationSnapshot((task(),), drone_cycles_per_second=0))
    assert cloud.tasks[0]["x"] == 0
    empty = mode(AllocationSnapshot((task(input_bits=0, cycles_per_bit=0),)))
    assert np.isfinite(empty.objective)
    assert mode(AllocationSnapshot(())).objective == 0
    with pytest.raises(AllocationFailure):
        mode(AllocationSnapshot((task(),), drone_cycles_per_second=0, cloud_cycles_per_second=0))


def test_bounded_weights_and_energy_tradeoff():
    np.testing.assert_allclose(priority_weights([3, 3, 9], [False, False, True]), [4, 4, 1])
    np.testing.assert_allclose(priority_weights([1, 3, 9], [False, False, True]), [4, 1, 1])
    s = AllocationSnapshot((task(),), energy_coefficient=0)
    assert solve_binary(s).tasks[0]["x"] == 0
    assert solve_binary(replace(s, energy_coefficient=100)).tasks[0]["x"] == 1


def test_random_heterogeneous_instances():
    rng = np.random.default_rng(12)
    for n in (2, 4, 8):
        tasks = tuple(task(str(i), input_bits=float(rng.uniform(1e4, 2e5)),
            weight=float(rng.uniform(1, 4)), backhaul_bps=float(rng.uniform(1e6, 50e6))) for i in range(n))
        compare_allocations(AllocationSnapshot(tasks))


@pytest.mark.parametrize("x", [0, 1])
def test_zero_work_does_not_reserve_cpu_from_other_jobs(x):
    snapshot = AllocationSnapshot((task("empty", input_bits=0, fixed_x=x),
                                   task("working", fixed_x=x)))
    result = compare_allocations(snapshot)
    zero, working = result["relaxed"]["tasks"]
    assert zero["drone_share"] == pytest.approx(0, abs=1e-8)
    assert zero["cloud_share"] == pytest.approx(0, abs=1e-8)
    assert working["drone_share" if x else "cloud_share"] == pytest.approx(1, abs=1e-6)
