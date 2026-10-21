"""Equation 16 with fixed radio/powers and explicit units.

The fractional result is a scheduling lower bound, not a split-job runtime.
CVXPY is deliberately imported lazily so legacy studies need no new dependency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from functools import lru_cache
import math
import warnings
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
    deadline_seconds: float | None = None  # optional analytical snapshot latency bound

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
        if self.deadline_seconds is not None and not math.isfinite(self.deadline_seconds):
            raise ValueError("A deadline must be finite; omit it for no hard deadline")


@dataclass(frozen=True)
class AllocationSnapshot:
    tasks: tuple[AllocationTask, ...]
    drone_cycles_per_second: float = 1e9
    cloud_cycles_per_second: float = 1e10
    energy_coefficient: float = 1.0  # per joule; reference energy = 1 J
    latency_coefficient: float = 1.0  # per second; reference time = 1 s
    drone_kappa: float = 0.0  # legacy default; paper DVFS coefficient [J/(cycle*(cycle/s)^2)]
    include_robot_upload_in_objective: bool = True

    def __post_init__(self):
        if len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("Duplicate task IDs")
        for v in (self.drone_cycles_per_second, self.cloud_cycles_per_second,
                  self.energy_coefficient, self.latency_coefficient, self.drone_kappa):
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
    assignments_checked: int = 1
    solver_attempts: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class AllocationFailure(RuntimeError):
    def __init__(self, message, snapshot, details=None):
        super().__init__(message)
        self.snapshot = asdict(snapshot)
        self.details = details or {}

    def __reduce__(self):
        # Preserve the snapshot when an exception crosses a process boundary.
        return (type(self), (str(self), AllocationSnapshot.from_dict(self.snapshot), self.details))


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


def processor_shares(cycles, weights, capacity, kappa=0.0, energy_coefficient=1.0, latency_coefficient=1.0):
    """KKT solution including DVFS; returns actual capacity fractions, possibly idle.

    In normalized rates u=f/F, minimize sum(A/u+B*u^2). The latency-only
    case is Equations 18--19. The general case solves -A/u^2+2*B*u+lambda=0.
    """
    cycles, weights = np.asarray(cycles, float), np.asarray(weights, float)
    if not cycles.size or not np.any(cycles):
        return np.zeros_like(cycles)
    if kappa == 0 or energy_coefficient == 0:
        return cpu_shares(cycles, weights)
    return _processor_batch(cycles[None, :], weights, capacity, kappa,
                            energy_coefficient, latency_coefficient)[0]


def _processor_batch(cycles, weights, capacity, kappa=0.0, energy_coefficient=1.0, latency_coefficient=1.0):
    roots = np.sqrt(cycles*weights)
    totals = roots.sum(axis=1, keepdims=True)
    if not kappa or not energy_coefficient:
        return np.divide(roots, totals, out=np.zeros_like(roots), where=totals > 0)
    A = latency_coefficient*weights*cycles/capacity
    B = energy_coefficient*kappa*cycles*capacity**2
    free = np.cbrt(np.divide(A, 2*B, out=np.zeros_like(A), where=B > 0))
    binding = free.sum(axis=1) > 1
    result = free.copy()
    if not binding.any():
        return result
    A, B = A[binding], B[binding]
    upper = np.sum(np.sqrt(A), axis=1)**2
    lower = np.zeros_like(upper)
    multiplier = upper/2
    active = A > 0
    for _ in range(60):
        lam = multiplier[:, None]
        u = np.minimum(np.cbrt(np.divide(A, 2*B, out=np.zeros_like(A), where=B > 0)),
                       np.sqrt(A/lam))
        # Newton on the monotone cubic from an upper bound stays above its root.
        for _ in range(18):
            numerator = 2*B*u**3+lam*u**2-A
            denominator = 6*B*u**2+2*lam*u
            change = np.divide(numerator, denominator, out=np.zeros_like(u), where=active)
            u -= change
            if np.max(np.abs(change)) < 1e-14:
                break
        error = u.sum(axis=1)-1
        if np.max(np.abs(error)) < 1e-12:
            break
        lower = np.where(error > 0, multiplier, lower)
        upper = np.where(error > 0, upper, multiplier)
        derivative = -np.sum(np.divide(u, 6*B*u+2*lam, out=np.zeros_like(u), where=active), axis=1)
        proposal = multiplier-error/derivative
        multiplier = np.where((proposal > lower) & (proposal < upper), proposal, (lower+upper)/2)
    else:
        raise ArithmeticError("CPU KKT allocation did not converge")
    result[binding] = u/np.maximum(1, u.sum(axis=1, keepdims=True))
    return result


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
        if t.deadline_seconds is not None and latency > t.deadline_seconds+1e-6*max(1, abs(t.deadline_seconds)):
            raise AllocationFailure("Recovered allocation violates snapshot deadline", snapshot,
                dict(task_id=t.task_id, latency=latency, deadline=t.deadline_seconds))
        robot_energy = t.robot_power_w*access
        backhaul_energy = t.drone_power_w*(1-xi)*backhaul
        drone_energy = (snapshot.drone_kappa*t.cycles*snapshot.drone_cycles_per_second**2*ai**2/xi
                        if xi else 0.0)
        objective_energy = backhaul_energy+drone_energy+(
            robot_energy if snapshot.include_robot_upload_in_objective else 0.0)
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
            backhaul_energy_j=backhaul_energy, drone_compute_energy_j=drone_energy,
            energy_j=robot_energy+backhaul_energy+drone_energy, objective_energy_j=objective_energy,
            weighted_latency_seconds=t.weight*latency))
    energy = snapshot.energy_coefficient*sum(r["objective_energy_j"] for r in rows)
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
    if any(t.deadline_seconds is not None for t in snapshot.tasks):
        # Optional hard bounds invalidate the unconstrained closed-form CPU KKT
        # rates. Enumerate the same assignments, solving each convex CPU problem.
        best, checked = None, 0
        for assignment in product(*choices):
            from dataclasses import replace
            candidate = replace(snapshot, tasks=tuple(replace(t, fixed_x=x)
                for t, x in zip(snapshot.tasks, assignment)))
            checked += 1
            try:
                result = solve_relaxed(candidate)
            except AllocationFailure as exc:
                if exc.details.get("status") in ("infeasible", "infeasible_inaccurate"):
                    continue
                raise
            if best is None or result.objective < best.objective:
                best = result
        if best is None:
            raise AllocationFailure("No assignment satisfies snapshot deadlines", snapshot, {"status": "infeasible"})
        best.mode, best.solve_seconds, best.assignments_checked = "binary_exact", perf_counter()-start, checked
        return best
    cycles = np.array([t.cycles for t in snapshot.tasks])
    weights = np.array([t.weight for t in snapshot.tasks])
    x = np.array(list(product(*choices)), dtype=float).reshape(-1, len(cycles)) if len(cycles) else np.empty((1, 0))
    a = _processor_batch(cycles*x, weights, snapshot.drone_cycles_per_second or 1,
        snapshot.drone_kappa, snapshot.energy_coefficient, snapshot.latency_coefficient)
    b = _processor_batch(cycles*(1-x), weights, snapshot.cloud_cycles_per_second or 1)
    access = np.array([t.input_bits/t.access_bps if t.input_bits else 0 for t in snapshot.tasks])
    backhaul = np.array([t.input_bits/t.backhaul_bps if t.backhaul_bps else 0 for t in snapshot.tasks])
    propagation = np.array([t.propagation_round_trip_seconds for t in snapshot.tasks])
    drone_time = np.divide(cycles*x, a*(snapshot.drone_cycles_per_second or 1), out=np.zeros_like(a), where=a > 0)
    cloud_time = np.divide(cycles*(1-x), b*(snapshot.cloud_cycles_per_second or 1), out=np.zeros_like(b), where=b > 0)
    latency = access+(1-x)*(backhaul+propagation)+drone_time+cloud_time
    energy = (1-x)*backhaul*np.array([t.drone_power_w for t in snapshot.tasks])
    energy += snapshot.drone_kappa*cycles*(a*snapshot.drone_cycles_per_second)**2
    if snapshot.include_robot_upload_in_objective:
        energy += access*np.array([t.robot_power_w for t in snapshot.tasks])
    objective = snapshot.latency_coefficient*np.sum(weights*latency, axis=1)+snapshot.energy_coefficient*energy.sum(axis=1)
    selected = int(np.argmin(objective))
    best = _evaluate(snapshot, x[selected], a[selected], b[selected], "binary_exact", perf_counter()-start)
    best.assignments_checked = len(x)
    return best


@lru_cache(maxsize=24)
def _conic_template(n, zero_work, dvfs, allowed_routes, deadline_indices=()):
    """Reusable DPP graph; every snapshot still performs a fresh numerical solve."""
    import cvxpy as cp
    x, a, b = (cp.Variable(n) for _ in range(3))
    qd, qc = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    ed = cp.Variable(n, nonneg=True) if dvfs else None
    coefficients = cp.Parameter((4, n), nonneg=True)
    lower, upper = cp.Parameter(n, nonneg=True), cp.Parameter(n, nonneg=True)
    deadline_bounds, deadline_backhaul = cp.Parameter(n), cp.Parameter(n, nonneg=True)
    constraints = [x >= lower, x <= upper, a >= 0, b >= 0, a <= x, b <= 1-x,
                   cp.sum(a) <= 1, cp.sum(b) <= 1]
    for i in range(n):
        if i not in zero_work:
            if 1 in allowed_routes[i]:
                constraints.append(cp.SOC(qd[i]+a[i], cp.hstack([2*coefficients[1,i]*x[i], qd[i]-a[i]])))
            else:
                constraints.extend((a[i] == 0, qd[i] == 0))
            if 0 in allowed_routes[i]:
                constraints.append(cp.SOC(qc[i]+b[i], cp.hstack([2*coefficients[2,i]*(1-x[i]), qc[i]-b[i]])))
            else:
                constraints.extend((b[i] == 0, qc[i] == 0))
        else:
            constraints.extend((a[i] == 0, b[i] == 0, qd[i] == 0, qc[i] == 0))
        if ed is not None:
            if 1 in allowed_routes[i]:
                constraints.append(cp.SOC(ed[i]+x[i], cp.hstack([2*coefficients[3,i]*a[i], ed[i]-x[i]])))
            else:
                constraints.append(ed[i] == 0)
        if i in deadline_indices:
            constraints.append(qd[i]+qc[i]+deadline_backhaul[i]*(1-x[i]) <= deadline_bounds[i])
    objective = coefficients[0] @ (1-x)+cp.sum(qd+qc)
    if ed is not None:
        objective += cp.sum(ed)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    assert problem.is_dcp(dpp=True)
    return problem, constraints, x, a, b, coefficients, lower, upper, deadline_bounds, deadline_backhaul


def _refine_cpu(snapshot, x):
    """Exact feasible CPU minimizer at the conic x, without minimum allocations."""
    from scipy.optimize import brentq
    C = np.array([t.cycles for t in snapshot.tasks])
    w = snapshot.latency_coefficient*np.array([t.weight for t in snapshot.tasks])
    def pool(A, B, limits):
        active = (A > 0) & (limits > 0)
        if not active.any():
            return np.zeros_like(A)
        free = np.where(B > 0, np.cbrt(np.divide(A, 2*B, out=np.zeros_like(A), where=B > 0)), limits)
        def allocate(multiplier):
            value = np.minimum(limits, free)
            if multiplier:
                value = np.minimum(value, np.sqrt(A/multiplier))
            for _ in range(24):
                derivative = 6*B*value**2+2*multiplier*value
                change = np.divide(2*B*value**3+multiplier*value**2-A, derivative,
                                   out=np.zeros_like(value), where=(derivative > 0) & active)
                updated = np.minimum(limits, np.maximum(0, value-change))
                if np.max(np.abs(updated-value)) < 1e-13:
                    break
                value = updated
            return np.where(active, value, 0)
        value = allocate(0)
        if value.sum() > 1:
            scale = np.sum(np.sqrt(A))**2
            edge = allocate(scale)
            if edge.sum() >= 1-1e-12:
                value = edge
            else:
                mu = brentq(lambda m: allocate(m*scale).sum()-1, 0, 1, xtol=1e-12)
                value = allocate(mu*scale)
        return value/max(1, float(value.sum()))
    A = w*C*x**2/(snapshot.drone_cycles_per_second or 1)
    B = np.divide(snapshot.energy_coefficient*snapshot.drone_kappa*C*snapshot.drone_cycles_per_second**2,
                  x, out=np.zeros_like(x), where=x > 0)
    a = pool(A, B, x)
    cloud = 1-x
    b = pool(w*C*cloud**2/(snapshot.cloud_cycles_per_second or 1), np.zeros_like(x), cloud)
    return a, b


def solve_relaxed(snapshot):
    start = perf_counter()
    n = len(snapshot.tasks)
    if not n:
        return _evaluate(snapshot, [], [], [], "relaxed", perf_counter()-start)
    allowed_routes = tuple(tuple(_allowed(snapshot, t)) for t in snapshot.tasks)
    if any(not routes for routes in allowed_routes):
        raise AllocationFailure("No available execution route", snapshot)
    try:
        problem, constraints, x, a, b, coefficients, lower, upper, bounds, backhaul = _conic_template(
            n, tuple(i for i, t in enumerate(snapshot.tasks) if not t.cycles), bool(snapshot.drone_kappa), allowed_routes,
            tuple(i for i, t in enumerate(snapshot.tasks) if t.deadline_seconds is not None))
    except ImportError as exc:
        raise AllocationFailure("Install the optimization extra (CVXPY/Clarabel)", snapshot) from exc
    values, lows, highs, constant = [], [], [], 0.0
    for t in snapshot.tasks:
        allowed = _allowed(snapshot, t)
        if not allowed:
            raise AllocationFailure("No available execution route", snapshot)
        lows.append(min(allowed))
        highs.append(max(allowed))
        d_g = t.input_bits/t.access_bps if t.input_bits else 0.0
        d_a = t.input_bits/t.backhaul_bps if t.backhaul_bps else 0.0
        w = snapshot.latency_coefficient*t.weight
        constant += w*d_g+(snapshot.energy_coefficient*t.robot_power_w*d_g
                            if snapshot.include_robot_upload_in_objective else 0)
        values.append([w*(d_a+t.propagation_round_trip_seconds)+snapshot.energy_coefficient*t.drone_power_w*d_a,
            w*t.cycles/snapshot.drone_cycles_per_second if snapshot.drone_cycles_per_second else 0,
            w*t.cycles/snapshot.cloud_cycles_per_second if snapshot.cloud_cycles_per_second else 0,
            snapshot.energy_coefficient*snapshot.drone_kappa*t.cycles*snapshot.drone_cycles_per_second**2])
    scaled = np.asarray(values).T
    scaled[1:] = np.sqrt(scaled[1:])
    coefficients.value = scaled
    lower.value, upper.value = np.asarray(lows), np.asarray(highs)
    bounds.value = [snapshot.latency_coefficient*t.weight*(t.deadline_seconds-(t.input_bits/t.access_bps if t.input_bits else 0))
                    if t.deadline_seconds is not None else 0 for t in snapshot.tasks]
    backhaul.value = [snapshot.latency_coefficient*t.weight*((t.input_bits/t.backhaul_bps if t.backhaul_bps else 0)
                     +t.propagation_round_trip_seconds) for t in snapshot.tasks]
    attempts = []
    for tolerance in (1e-8, 1e-7):
        try:
            with warnings.catch_warnings(record=True) as captured:
                problem.solve(solver="CLARABEL", enforce_dpp=True, warm_start=not attempts,
                    max_iter=300, tol_gap_abs=tolerance, tol_gap_rel=tolerance, tol_feas=tolerance)
            attempts.append(dict(tolerance=tolerance, status=problem.status,
                                 warnings=[str(w.message) for w in captured]))
        except Exception as exc:
            attempts.append(dict(tolerance=tolerance, error=str(exc)))
            if tolerance == 1e-7:
                raise AllocationFailure(str(exc), snapshot, {"is_dcp": problem.is_dcp(), "attempts": attempts}) from exc
            continue
        if problem.status != "optimal_inaccurate":
            break
    if problem.status != "optimal":
        raise AllocationFailure("Relaxation did not solve optimally", snapshot, {"status": problem.status, "attempts": attempts})
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
    if any(t.deadline_seconds is not None for t in snapshot.tasks):
        av, bv = np.maximum(0, a.value), np.maximum(0, b.value)
    else:
        av, bv = _refine_cpu(snapshot, xv)
    result = _evaluate(snapshot, xv, av, bv, "relaxed", perf_counter()-start, residual)
    result.solver_attempts = attempts
    if not math.isfinite(result.objective) or abs(result.objective-problem.value-constant) > 1e-6*max(1, abs(problem.value+constant)):
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


def score_allocation(snapshot, allocation):
    """Score unchanged feasible rates/routes under a common reference objective.

    This does not optimize or mutate either snapshot. In particular, native
    objectives with different priority weights must not be subtracted as regret.
    """
    rows = {r["task_id"]: r for r in allocation}
    if set(rows) != {t.task_id for t in snapshot.tasks}:
        raise ValueError("Reference score requires the identical task set")
    x = np.asarray([rows[t.task_id]["x"] for t in snapshot.tasks])
    a = np.asarray([rows[t.task_id]["drone_share"] for t in snapshot.tasks])
    b = np.asarray([rows[t.task_id]["cloud_share"] for t in snapshot.tasks])
    if (not np.all(np.isfinite(np.concatenate((x, a, b)))) or np.any(x < -1e-8)
        or np.any(x > 1+1e-8) or np.any(a < -1e-8) or np.any(b < -1e-8)
        or np.any(a > x+1e-8) or np.any(b > 1-x+1e-8)
        or a.sum() > 1+1e-8 or b.sum() > 1+1e-8):
        raise ValueError("Reference allocation violates capacity or conditional bounds")
    for task, value in zip(snapshot.tasks, x):
        if task.fixed_x is not None and value != task.fixed_x:
            raise ValueError("Reference allocation changes a committed route")
    return _evaluate(snapshot, x, a, b, "reference_score", 0)
