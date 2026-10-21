"""Predictable allocation cases with independent primal CVXPY enumeration."""
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

from .allocation import AllocationTask, AllocationSnapshot, compare_allocations
from .records import write_json


def sanity_scenarios():
    tasks = tuple(AllocationTask(str(i), 1e5, 1000, 1, 50e6, 10e6) for i in range(4))
    base = AllocationSnapshot(tasks, energy_coefficient=0)
    return [
        ("high_satellite_delay", replace(base, tasks=tuple(replace(t,
            propagation_round_trip_seconds=5) for t in tasks)), [1, 1, 1, 1]),
        ("fast_cloud_slow_drone", replace(base, drone_cycles_per_second=1e6,
            tasks=tuple(replace(t, propagation_round_trip_seconds=0) for t in tasks)), [0, 0, 0, 0]),
        ("slow_backhaul", replace(base, tasks=tuple(replace(t,
            backhaul_bps=1e3) for t in tasks)), [1, 1, 1, 1]),
        ("slow_cloud_cpu", replace(base, cloud_cycles_per_second=1e6), [1, 1, 1, 1]),
        ("unavailable_backhaul", replace(base, tasks=tuple(replace(t, backhaul_bps=0) for t in tasks)), [1, 1, 1, 1]),
        ("unavailable_drone", replace(base, drone_cycles_per_second=0), [0, 0, 0, 0]),
        ("single_task_low_propagation", replace(base, tasks=(replace(tasks[0],
            propagation_round_trip_seconds=.01),)), [0]),
        ("single_task_high_propagation", replace(base, tasks=(replace(tasks[0],
            propagation_round_trip_seconds=1),)), [1]),
        ("bounded_priority_fixed_drone", replace(base, tasks=(replace(tasks[0], weight=4, fixed_x=1),
            replace(tasks[1], weight=1, fixed_x=1))), [1, 1]),
        ("dvfs_idle_capacity", AllocationSnapshot((replace(tasks[0], fixed_x=1),),
            drone_cycles_per_second=30e9, drone_kappa=1e-28, energy_coefficient=1), [1]),
    ]


def independent_primal(snapshot):
    """Direct C/f + kappa*C*f^2 per fixed route; no substitution or KKT helper."""
    import cvxpy as cp
    allowed = []
    for t in snapshot.tasks:
        routes = [x for x, available in ((0, snapshot.cloud_cycles_per_second > 0 and t.backhaul_bps > 0),
                                         (1, snapshot.drone_cycles_per_second > 0))
                  if available and t.fixed_x in (None, x)]
        allowed.append(routes)
    best = None
    for assignment in product(*allowed):
        objective, constraints = 0, []
        for route, capacity in ((1, snapshot.drone_cycles_per_second), (0, snapshot.cloud_cycles_per_second)):
            selected = [t for t, x in zip(snapshot.tasks, assignment) if x == route]
            if not selected:
                continue
            u = cp.Variable(len(selected), pos=True)
            scale = min(capacity, 1e9)
            constraints.append(cp.sum(u) <= capacity/scale)
            for i, t in enumerate(selected):
                delay = t.input_bits/t.access_bps
                energy = t.robot_power_w*delay if snapshot.include_robot_upload_in_objective else 0
                if route == 0:
                    delay += t.input_bits/t.backhaul_bps+t.propagation_round_trip_seconds
                    energy += t.drone_power_w*t.input_bits/t.backhaul_bps
                else:
                    energy += snapshot.drone_kappa*t.cycles*scale**2*cp.square(u[i])
                objective += snapshot.latency_coefficient*t.weight*(delay+t.cycles/scale*cp.inv_pos(u[i]))
                objective += snapshot.energy_coefficient*energy
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
        if problem.status != "optimal":
            raise RuntimeError(f"Independent primal solve failed: {problem.status}")
        if best is None or problem.value < best["objective"]:
            best = dict(objective=float(problem.value), x=list(assignment))
    return best


def run_sanity(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, snapshot, expected in sanity_scenarios():
        result = compare_allocations(snapshot)
        independent = independent_primal(snapshot)
        actual = [int(t["x"]) for t in result["binary"]["tasks"]]
        difference = abs(result["binary"]["objective"]-independent["objective"])
        if actual != expected or difference > 1e-6*max(1, independent["objective"]):
            write_json(output/f"{name}.failed.json", dict(result=result, independent=independent, expected=expected))
            raise AssertionError(f"Sanity scenario {name} failed")
        if name == "bounded_priority_fixed_drone":
            np.testing.assert_allclose([t["drone_share"] for t in result["binary"]["tasks"]], [2/3, 1/3])
        if name == "dvfs_idle_capacity":
            assert result["binary"]["tasks"][0]["drone_share"] < .1
        write_json(output/f"{name}.json", dict(result=result, independent=independent, expected=expected))
        rows.append(dict(scenario=name, expected=expected, actual=actual, status="passed",
            relaxed_x=[t["x"] for t in result["relaxed"]["tasks"]],
            drone_shares=[t["drone_share"] for t in result["binary"]["tasks"]],
            cloud_shares=[t["cloud_share"] for t in result["binary"]["tasks"]],
            is_dcp=result["relaxed"]["is_dcp"], residual=result["relaxed"]["residual"],
            objective=result["binary"]["objective"], independent_objective=independent["objective"],
            difference=difference, relaxation_gap=result["absolute_gap"]))
    write_json(output/"summary.json", rows)
    return rows
