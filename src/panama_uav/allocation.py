"""Equation 16 with fixed radio/powers, no deadlines, and explicit units.

The fractional result is a scheduling lower bound, not a split-job runtime.
CVXPY is deliberately imported lazily so legacy studies need no new dependency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class AllocationTask:
    task_id: str
    input_bits: float
    cycles_per_bit: float
    weight: float
    access_bps: float
    backhaul_bps: float
    propagation_round_trip_seconds: float = 0.05
    robot_power_w: float = 0.2
    drone_power_w: float = 2.0
    fixed_x: int | None = None
    required_cycles: float | None = None  # remaining work for already assigned jobs

    @property
    def cycles(self):
        return self.input_bits * self.cycles_per_bit if self.required_cycles is None else self.required_cycles

    def __post_init__(self):
        values = (self.input_bits, self.cycles_per_bit, self.weight, self.access_bps,
                  self.backhaul_bps, self.propagation_round_trip_seconds,
                  self.robot_power_w, self.drone_power_w)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Task values must be finite and nonnegative")
        if self.weight <= 0 or (self.input_bits > 0 and self.access_bps <= 0):
            raise ValueError("Positive weight and an available access route are required")
        if self.fixed_x not in (None, 0, 1):
            raise ValueError("fixed_x must be None, 0 (cloud), or 1 (drone)")
        if self.required_cycles is not None and (not math.isfinite(self.required_cycles) or self.required_cycles < 0):
            raise ValueError("Required cycles must be finite and nonnegative")


@dataclass(frozen=True)
class AllocationSnapshot:
    tasks: tuple[AllocationTask, ...]
    drone_cycles_per_second: float = 1e9
    cloud_cycles_per_second: float = 1e10
    energy_coefficient: float = 1.0  # per joule; reference energy = 1 J
    latency_coefficient: float = 1.0  # per second; reference time = 1 s

    def __post_init__(self):
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("Duplicate task IDs")
        for v in (self.drone_cycles_per_second, self.cloud_cycles_per_second,
                  self.energy_coefficient, self.latency_coefficient):
            if not math.isfinite(v) or v < 0:
                raise ValueError("Capacities and coefficients must be finite and nonnegative")
        if self.latency_coefficient <= 0:
            raise ValueError("Latency coefficient must be positive")

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "tasks": tuple(AllocationTask(**t) for t in data["tasks"])})


@dataclass
class AllocationResult:
    mode: str
    status: str
    objective: float
    energy_objective: float
    latency_objective: float
    tasks: list[dict]
    solve_seconds: float
    residual: float = 0.0
    is_dcp: bool = True

    def to_dict(self):
        return asdict(self)


class AllocationFailure(RuntimeError):
    def __init__(self, message, snapshot, details=None):
        super().__init__(message)
        self.snapshot = asdict(snapshot)
        self.details = details or {}


def priority_weights(priorities, completed, minimum=1.0, maximum=4.0):
    """Bounded inverse priority map, with an explicit equal-priority case."""
    p = np.asarray(priorities, dtype=float)
    done = np.asarray(completed, dtype=bool)
    if not 0 < minimum <= maximum or len(p) != len(done) or not np.all(np.isfinite(p)):
        raise ValueError("Invalid priorities or weight bounds")
    result = np.full(len(p), minimum)
    active = p[~done]
    if len(active):
        span = float(np.max(active) - np.min(active))
        result[~done] = maximum if span == 0 else (
            minimum + (maximum-minimum)*(np.max(active)-active)/span)
    return result


def cpu_shares(cycles, weights):
    roots = np.sqrt(np.asarray(cycles, float)*np.asarray(weights, float))
    return roots/roots.sum() if roots.sum() else np.zeros_like(roots)


def _evaluate(snapshot, x, a, b, mode, elapsed, residual=0.0):
    rows = []
    for t, xi, ai, bi in zip(snapshot.tasks, x, a, b):
        access = t.input_bits/t.access_bps if t.input_bits else 0.0
        backhaul = t.input_bits/t.backhaul_bps if t.backhaul_bps else 0.0
        def compute(fraction, share, capacity):
            if not t.cycles or fraction == 0:
                return 0.0
            if share <= 0 or capacity <= 0:
                raise AllocationFailure("Positive compute work has no allocated CPU", snapshot,
                    dict(task_id=t.task_id, fraction=float(fraction), share=float(share)))
            return t.cycles/capacity*fraction*fraction/share
        drone_time = compute(xi, ai, snapshot.drone_cycles_per_second)
        cloud_time = compute(1-xi, bi, snapshot.cloud_cycles_per_second)
        latency = access+(1-xi)*(backhaul+t.propagation_round_trip_seconds)+drone_time+cloud_time
        robot_energy = t.robot_power_w*access
        backhaul_energy = t.drone_power_w*(1-xi)*backhaul
        rows.append(dict(task_id=t.task_id, x=float(xi), drone_share=float(ai),
            cloud_share=float(bi),
            drone_normalized_rate=float(ai/xi) if xi else 0.0,
            cloud_normalized_rate=float(bi/(1-xi)) if xi != 1 else 0.0,
            drone_cycles_per_second=float(snapshot.drone_cycles_per_second*ai/xi) if xi else 0.0,
            cloud_cycles_per_second=float(snapshot.cloud_cycles_per_second*bi/(1-xi)) if xi != 1 else 0.0,
            access_seconds=access, backhaul_seconds=(1-xi)*backhaul,
            propagation_seconds=(1-xi)*t.propagation_round_trip_seconds,
            drone_compute_seconds=drone_time, cloud_compute_seconds=cloud_time,
            latency_seconds=latency, robot_energy_j=robot_energy,
            backhaul_energy_j=backhaul_energy, energy_j=robot_energy+backhaul_energy,
            weighted_latency_seconds=t.weight*latency))
    energy = snapshot.energy_coefficient*sum(r["energy_j"] for r in rows)
    latency = snapshot.latency_coefficient*sum(r["weighted_latency_seconds"] for r in rows)
    return AllocationResult(mode, "optimal", energy+latency, energy, latency, rows, elapsed, residual)


def _allowed(snapshot, task):
    edge = snapshot.drone_cycles_per_second > 0 or task.cycles == 0
    cloud = ((snapshot.cloud_cycles_per_second > 0 or task.cycles == 0)
             and (task.backhaul_bps > 0 or task.input_bits == 0))
    return [x for x, ok in ((0, cloud), (1, edge))
            if ok and task.fixed_x in (None, x)]


def solve_binary(snapshot):
    start = perf_counter()
    if sum(t.fixed_x is None for t in snapshot.tasks) > 8:
        raise ValueError("Exact reference supports at most eight undecided tasks")
    choices = [_allowed(snapshot, t) for t in snapshot.tasks]
    if any(not c for c in choices):
        raise AllocationFailure("No available execution route", snapshot)
    best = None
    cycles = np.array([t.cycles for t in snapshot.tasks])
    weights = np.array([t.weight for t in snapshot.tasks])
    for assignment in product(*choices):
        x = np.array(assignment, dtype=float)
        a = cpu_shares(cycles*x, weights)
        b = cpu_shares(cycles*(1-x), weights)
        result = _evaluate(snapshot, x, a, b, "binary_exact", 0.0)
        if best is None or result.objective < best.objective-1e-12:
            best = result
    best.solve_seconds = perf_counter()-start
    return best


def solve_relaxed(snapshot):
    start = perf_counter()
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise AllocationFailure("Install the optimization extra (CVXPY/Clarabel)", snapshot) from exc
    n = len(snapshot.tasks)
    if not n:
        return _evaluate(snapshot, [], [], [], "relaxed", perf_counter()-start)
    x, a, b = (cp.Variable(n) for _ in range(3))
    qd, qc = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    constraints = [x >= 0, x <= 1, a >= 0, b >= 0, a <= x, b <= 1-x,
                   cp.sum(a) <= 1, cp.sum(b) <= 1]
    objective = 0
    for i, t in enumerate(snapshot.tasks):
        allowed = _allowed(snapshot, t)
        if not allowed:
            raise AllocationFailure("No available execution route", snapshot)
        if len(allowed) == 1:
            constraints.append(x[i] == allowed[0])
        # ||(2*z, q-u)|| <= q+u represents z^2 <= q*u,
        # including the closed endpoint z=u=q=0.
        if t.cycles:
            constraints.extend((cp.SOC(qd[i]+a[i], cp.hstack([2*x[i], qd[i]-a[i]])),
                                cp.SOC(qc[i]+b[i], cp.hstack([2*(1-x[i]), qc[i]-b[i]]))))
        else:
            # A zero-work task needs no processor even when its route is fixed.
            # Keeping x^2 <= q*a here would force a spurious positive share.
            constraints.extend((a[i] == 0, b[i] == 0, qd[i] == 0, qc[i] == 0))
        d_g = t.input_bits/t.access_bps if t.input_bits else 0.0
        d_a = t.input_bits/t.backhaul_bps if t.backhaul_bps else 0.0
        td = t.cycles/snapshot.drone_cycles_per_second if snapshot.drone_cycles_per_second else 0.0
        tc = t.cycles/snapshot.cloud_cycles_per_second if snapshot.cloud_cycles_per_second else 0.0
        latency = d_g+(1-x[i])*(d_a+t.propagation_round_trip_seconds)+td*qd[i]+tc*qc[i]
        energy = t.robot_power_w*d_g+t.drone_power_w*(1-x[i])*d_a
        objective += snapshot.energy_coefficient*energy+snapshot.latency_coefficient*t.weight*latency
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
    except Exception as exc:
        raise AllocationFailure(str(exc), snapshot, {"is_dcp": problem.is_dcp()}) from exc
    if problem.status != "optimal":
        raise AllocationFailure("Relaxation did not solve optimally", snapshot, {"status": problem.status})
    residual = max(float(np.max(c.violation())) for c in constraints)
    if not math.isfinite(residual) or residual > 1e-6:
        raise AllocationFailure("Allocation constraint residual exceeded tolerance", snapshot, {"residual": residual})
    xv = np.clip(x.value, 0, 1)
    # Snap only solver endpoints, then recompute feasible resource shares at x.
    xv[xv < 1e-8] = 0
    xv[xv > 1-1e-8] = 1
    for i, t in enumerate(snapshot.tasks):
        allowed = _allowed(snapshot, t)
        if len(allowed) == 1:
            xv[i] = allowed[0]
    av = np.minimum(np.maximum(a.value, 0), xv)
    bv = np.minimum(np.maximum(b.value, 0), 1-xv)
    av /= max(1.0, float(av.sum()))
    bv /= max(1.0, float(bv.sum()))
    result = _evaluate(snapshot, xv, av, bv, "relaxed", perf_counter()-start, residual)
    if not math.isfinite(result.objective) or abs(result.objective-problem.value) > 1e-6*max(1, abs(problem.value)):
        raise AllocationFailure("Recovered allocation disagrees with conic objective", snapshot)
    return result


def solve_allocation(snapshot, mode="relaxed"):
    if mode == "relaxed":
        return solve_relaxed(snapshot)
    if mode == "binary_exact":
        return solve_binary(snapshot)
    raise ValueError("Unknown allocation mode")


def compare_allocations(snapshot):
    relaxed, binary = solve_relaxed(snapshot), solve_binary(snapshot)
    gap = binary.objective-relaxed.objective
    if gap < -1e-6*max(1, abs(binary.objective)):
        raise AllocationFailure("Relaxation exceeds binary reference", snapshot)
    return {"snapshot": asdict(snapshot), "relaxed": relaxed.to_dict(),
            "binary": binary.to_dict(), "absolute_gap": max(0.0, gap),
            "raw_objective_difference": gap,
            "relative_gap": max(0.0, gap)/max(abs(binary.objective), 1e-12)}
