"""Numerical equivalent of the uploaded MATLAB/CVX model; not MATLAB execution.

Enumerate every binary route and solve each convex continuous problem with
CVXPY/Clarabel. Verify the winning objective independently using scalar KKT solves.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import cvxpy as cp
import numpy as np
from scipy.optimize import brentq
from scipy.io import savemat


ROOT = Path(__file__).resolve().parent
ORIGINAL_SHA256 = "31a53647c1d737d9a08c6f6e3647f80ed7b3a45870d9f3b3a39e54ea59e7f87b"


def vector(value, n):
    result = np.full(n, value, dtype=float) if np.isscalar(value) else np.asarray(value, float).reshape(-1)
    if len(result) != n:
        raise ValueError("Vector length does not match workload")
    return result


def make_input(settings, n, kappa):
    size = settings["input_bits"]
    scale = settings["cpuScale"]
    return dict(C=[size*settings["cycles_per_bit"]]*n,
        pi=np.linspace(settings["priority_weight_max"], settings["priority_weight_min"], n).tolist(),
        dD=[size/settings["access_bps"]]*n,
        dC=[size/settings["access_bps"]+size/settings["backhaul_bps"]
            +settings["propagation_round_trip_seconds"]]*n,
        Ebackhaul=[settings["drone_power_w"]*size/settings["backhaul_bps"]]*n,
        FDmax=settings["FDmax"], FCmax=settings["FCmax"], omega2=settings["omega2"],
        omega3=settings["omega3"], kappaD=kappa, Tref=settings["Tref"], Eref=settings["Eref"],
        cpuScale=scale, epsCPU=settings["epsCPU"], tau=[settings.get("tau_seconds", 1e6)]*n)


def prepare(p):
    C = np.asarray(p["C"], float).reshape(-1)
    n = len(C)
    if n < 1 or n > 8 or np.any(C <= 0) or not np.isfinite(C).all():
        raise ValueError("This exhaustive runner supports 1--8 positive workloads")
    values = dict(C=C, n=n)
    for name in ("pi", "dD", "dC", "Ebackhaul", "kappaD"):
        values[name] = vector(p[name], n)
        if not np.isfinite(values[name]).all() or np.any(values[name] < 0):
            raise ValueError(f"Invalid {name}")
    if np.any(values["pi"] <= 0):
        raise ValueError("Weights must be positive")
    for name, default in (("cpuScale", 1e9), ("epsCPU", 1e-6), ("Tref", 1), ("Eref", 1)):
        values[name] = float(p.get(name, default))
        if not math.isfinite(values[name]) or values[name] <= 0:
            raise ValueError(f"Invalid {name}")
    for name in ("FDmax", "FCmax"):
        if not math.isfinite(p[name]) or p[name] <= 0:
            raise ValueError(f"Invalid {name}")
    for name in ("omega2", "omega3"):
        if not math.isfinite(p[name]) or p[name] < 0:
            raise ValueError(f"Invalid {name}")
    scale = values["cpuScale"]
    values.update(Cbar=C/scale, FDbar=p["FDmax"]/scale, FCbar=p["FCmax"]/scale,
                  kappaBar=values["kappaD"]*scale**3)
    for name, capacity in (("fDmin", values["FDbar"]), ("fCmin", values["FCbar"])):
        minimum = vector(p[name], n)/scale if name in p else np.full(n, max(1e-6*capacity, 1e-8))
        if np.any(minimum <= 0) or np.any(minimum > capacity) or not np.isfinite(minimum).all():
            raise ValueError(f"Invalid {name}")
        values[name] = minimum
    values["tau"] = vector(p.get("tau", math.inf), n)
    if np.any(np.isnan(values["tau"])) or np.any(values["tau"] <= 0):
        raise ValueError("Invalid deadline")
    if "fixedX" in p:
        fixed = vector(p["fixedX"], n)
        if not np.isin(fixed, [0, 1]).all():
            raise ValueError("fixedX must be binary")
    return values


def numerical_result(p, v, x, fd, fc):
    fd, fc = np.asarray(fd).copy(), np.asarray(fc).copy()
    fd[x == 0] = 0
    fc[x == 1] = 0
    if np.any(fd[x == 1] <= 0) or np.any(fc[x == 0] <= 0):
        raise RuntimeError("Active CPU rate is not positive")
    computation = np.zeros(v["n"])
    computation[x == 1] = v["Cbar"][x == 1]/fd[x == 1]
    computation[x == 0] = v["Cbar"][x == 0]/fc[x == 0]
    latency = x*v["dD"]+(1-x)*v["dC"]+computation
    compute_energy = v["kappaBar"]*v["Cbar"]*fd**2
    forwarding_energy = (1-x)*v["Ebackhaul"]
    energy = compute_energy+forwarding_energy
    weighted_latency = float(np.dot(v["pi"], latency))
    energy_sum = float(energy.sum())
    objective = p["omega2"]*weighted_latency/v["Tref"]+p["omega3"]*energy_sum/v["Eref"]
    td = np.where(x == 1, latency, np.nan)
    tc = np.where(x == 0, latency, np.nan)
    return dict(status="optimal", optval=objective, x=x.astype(int).tolist(),
        fD=(fd*v["cpuScale"]).tolist(), fC=(fc*v["cpuScale"]).tolist(),
        fD_GHz=fd.tolist(), fC_GHz=fc.tolist(), T=latency.tolist(), ED=energy.tolist(),
        drone_compute_energy_j=compute_energy.tolist(), backhaul_energy_j=forwarding_energy.tolist(),
        latencyObjective=weighted_latency, energyObjective=energy_sum,
        normalizedLatencyObjective=weighted_latency/v["Tref"],
        normalizedEnergyObjective=energy_sum/v["Eref"],
        TD=[float(a) if math.isfinite(a) else None for a in td],
        TC=[float(a) if math.isfinite(a) else None for a in tc],
        floorD=(v["dD"]+v["Cbar"]/v["FDbar"]).tolist(),
        floorC=(v["dC"]+v["Cbar"]/v["FCbar"]).tolist(),
        individuallyImpossible=(np.flatnonzero(v["tau"] < np.minimum(
            v["dD"]+v["Cbar"]/v["FDbar"], v["dC"]+v["Cbar"]/v["FCbar"]))+1).tolist(),
        scale=dict(cpuScale=v["cpuScale"], Cbar=v["Cbar"].tolist(), FDbar=v["FDbar"],
                   FCbar=v["FCbar"], kappaBar=v["kappaBar"].tolist()))


def solve_equivalent(p):
    v = prepare(p)
    n, eps = v["n"], v["epsCPU"]
    x = cp.Parameter(n, nonneg=True)
    fd, fc = cp.Variable(n), cp.Variable(n)
    qd, qc = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
    latency = cp.multiply(x, v["dD"])+cp.multiply(1-x, v["dC"])+cp.multiply(v["Cbar"], qd+qc)
    energy = cp.multiply(v["kappaBar"]*v["Cbar"], cp.square(fd))+cp.multiply(1-x, v["Ebackhaul"])
    constraints = [cp.sum(fd) <= v["FDbar"], cp.sum(fc) <= v["FCbar"],
        fd >= cp.multiply(v["fDmin"], x), fd <= v["FDbar"]*x,
        fc >= cp.multiply(v["fCmin"], 1-x), fc <= v["FCbar"]*(1-x)]
    for i in range(n):
        denom_d = fd[i]+eps*(1-x[i])
        denom_c = fc[i]+eps*x[i]
        constraints.extend([cp.SOC(qd[i]+denom_d, cp.hstack([2*x[i], qd[i]-denom_d])),
                            cp.SOC(qc[i]+denom_c, cp.hstack([2*(1-x[i]), qc[i]-denom_c]))])
        if math.isfinite(v["tau"][i]):
            constraints.append(latency[i] <= v["tau"][i])
    problem = cp.Problem(cp.Minimize(p["omega2"]*(v["pi"] @ latency)/v["Tref"]
                                   +p["omega3"]*cp.sum(energy)/v["Eref"]), constraints)
    assert problem.is_dcp() and problem.is_dpp()
    best, branch_rows = None, []
    assignments = [p["fixedX"]] if "fixedX" in p else product((0, 1), repeat=n)
    start = perf_counter()
    for assignment in assignments:
        xv = np.array(assignment, float)
        x.value = xv
        problem.solve(solver="CLARABEL", warm_start=True, tol_gap_abs=1e-9,
                      tol_gap_rel=1e-9, tol_feas=1e-9)
        branch = dict(x=xv.astype(int).tolist(), status=problem.status)
        if problem.status == "infeasible":
            branch_rows.append(branch)
            continue
        if problem.status != "optimal":
            raise RuntimeError(f"Unresolved assignment {assignment}: {problem.status}")
        residual = max(float(np.max(c.violation())) for c in constraints)
        if not math.isfinite(residual) or residual > 1e-6:
            raise RuntimeError(f"Residual too high: {residual}")
        result = numerical_result(p, v, xv, fd.value, fc.value)
        if abs(result["optval"]-problem.value) > 1e-6*max(1, abs(problem.value)):
            raise RuntimeError("Recomputed objective disagrees with the conic solve")
        branch.update(optval=result["optval"], residual=residual)
        branch_rows.append(branch)
        if best is None or result["optval"] < best["optval"]-1e-8:
            best = result
    elapsed = perf_counter()-start
    if best is None:
        return dict(status="infeasible", assignments_checked=len(branch_rows), branches=branch_rows,
                    elapsed_seconds=elapsed, matlab_executed=False)
    best.update(assignments_checked=len(branch_rows), branches=branch_rows, elapsed_seconds=elapsed,
                matlab_executed=False, backend="CVXPY/CLARABEL with exhaustive binary enumeration",
                maximum_constraint_residual=max(r.get("residual", 0) for r in branch_rows), is_dcp=True, is_dpp=True)
    return best


def kkt_pool(A, B, minimum, capacity):
    """Independent scalar derivative/root calculation; no CVXPY expressions."""
    if not len(A):
        return np.array([])
    if minimum.sum() > capacity+1e-12:
        return None
    def allocation(multiplier):
        rates = []
        for a, b, lower in zip(A, B, minimum):
            derivative = lambda f: -a/f**2+2*b*f+multiplier
            if derivative(lower) >= 0:
                rates.append(lower)
            elif derivative(capacity) <= 0:
                rates.append(capacity)
            else:
                rates.append(brentq(derivative, lower, capacity, xtol=1e-13))
        return np.array(rates)
    free = allocation(0)
    if free.sum() <= capacity+1e-12:
        return free
    upper = 1.0
    while allocation(upper).sum() > capacity:
        upper *= 2
    multiplier = brentq(lambda lam: allocation(lam).sum()-capacity, 0, upper, xtol=1e-12)
    return allocation(multiplier)


def independent_reference(p):
    v = prepare(p)
    best = None
    assignments = [p["fixedX"]] if "fixedX" in p else product((0, 1), repeat=v["n"])
    A = p["omega2"]*v["pi"]*v["Cbar"]/v["Tref"]
    B = p["omega3"]*v["kappaBar"]*v["Cbar"]/v["Eref"]
    for assignment in assignments:
        x = np.array(assignment, float)
        fd, fc = np.zeros(v["n"]), np.zeros(v["n"])
        edge, cloud = x == 1, x == 0
        slack = v["tau"] - (x*v["dD"]+(1-x)*v["dC"])
        if np.any(slack <= 0):
            continue
        deadline_minimum = v["Cbar"]/slack
        rd = kkt_pool(A[edge], B[edge], np.maximum(v["fDmin"][edge], deadline_minimum[edge]), v["FDbar"])
        rc = kkt_pool(A[cloud], np.zeros(sum(cloud)), np.maximum(v["fCmin"][cloud], deadline_minimum[cloud]), v["FCbar"])
        if rd is None or rc is None:
            continue
        fd[edge], fc[cloud] = rd, rc
        result = numerical_result(p, v, x, fd, fc)
        if best is None or result["optval"] < best["optval"]-1e-8:
            best = result
    return best


def scenario_inputs(settings, tau_mode="large", solved=None):
    """Yield cases in script order so E can use the just-computed C latency."""
    cases = {}
    for case in settings["cases"]:
        p = dict(cases[case["inherits"]] if "inherits" in case else settings["base"])
        p.update(case["overrides"])
        description = case["description"]
        if tau_mode == "original":
            p["tau"] = [math.inf]*len(p["C"])
            if "original_tau" in case:
                rule = case["original_tau"]
                source = solved[rule["reference_case"]]
                p["tau"] = (rule["latency_multiplier"]*np.asarray(source["T"])).tolist()
                description = f"Original deadline test: tau={rule['latency_multiplier']:g}*{rule['reference_case']}.T"
        else:
            p["tau"] = [settings["tau_seconds"]]*len(p["C"])
        cases[case["id"]] = p
        yield case["id"], description, p


def write_json(path, value):
    """Keep JSON standards-compliant while retaining infinite tau explicitly."""
    def safe(item):
        if isinstance(item, dict):
            return {key: safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(val) for val in item]
        if isinstance(item, (float, np.floating)) and math.isinf(item):
            return "Infinity" if item > 0 else "-Infinity"
        return item
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding="utf-8")


def write_reports(output, inputs, solutions, summaries, tau_mode="large"):
    task_rows = []
    tau_note = ("Original script deadlines: tau = Infinity for A-D and F; "
                "Case E uses tau = 1.10 * the latency computed in Case C."
                if tau_mode == "original" else
                "All tasks use tau = 1,000,000 seconds. Case E repeats Case C because the original "
                "tight deadline experiment is disabled.")
    lines = ["# User-supplied CVX scenario results", "",
        "Backend: CVXPY/Clarabel with exhaustive binary enumeration; MATLAB/CVX was not executed.", "",
        tau_note, "", "x=1 means Drone-AP; x=0 means cloud.", "",
        "Objective = omega2 * sum(pi*T)/Tref + omega3 * sum(ED)/Eref.", "",
        "| Case | x | Objective | Weighted latency (s) | Energy (J) |",
        "| --- | --- | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['case']} | {row['x']} | {row['objective']:.9f} | "
                     f"{row['weighted_latency_s']:.9f} | {row['energy_j']:.9f} |")
    lines.extend(["", f"Checked {sum(row['assignments'] for row in summaries)} binary assignments. "
        f"Maximum independent KKT objective difference: {max(row['kkt_objective_difference'] for row in summaries):.3g}.",
        f"Maximum conic constraint residual across all branches: "
        f"{max(sol['maximum_constraint_residual'] for sol in solutions):.3g} (scaled model units).", "",
        "Case A's label describes favorable cloud delay/capacity, but the high forwarding",
        "energies for robots 1 and 2 make a mixed assignment optimal with omega3=0.9.",
        "The drone CPU energy term can make leaving CPU capacity unused optimal.", ""])
    for (label, description, p), sol in zip(inputs, solutions):
        lines.extend([f"## {label}: {description}", "",
            "| Robot | x | Drone CPU (GHz) | Cloud CPU (GHz) | Latency (s) | Energy (J) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |"])
        for i in range(len(p["C"])):
            lines.append(f"| {i+1} | {sol['x'][i]} | {sol['fD_GHz'][i]:.9f} | "
                         f"{sol['fC_GHz'][i]:.9f} | {sol['T'][i]:.9f} | {sol['ED'][i]:.9f} |")
            task_rows.append(dict(case=label, robot=i+1, x=sol["x"][i], cycles=p["C"][i],
                priority=p["pi"][i], drone_cpu_ghz=sol["fD_GHz"][i], cloud_cpu_ghz=sol["fC_GHz"][i],
                latency_s=sol["T"][i], energy_j=sol["ED"][i],
                compute_energy_j=sol["drone_compute_energy_j"][i],
                backhaul_energy_j=sol["backhaul_energy_j"][i], tau_seconds=p["tau"][i]))
        lines.extend(["", f"Drone CPU: {sum(sol['fD_GHz']):.9f}/{p['FDmax']/1e9:g} GHz. "
            f"Cloud CPU: {sum(sol['fC_GHz']):.9f}/{p['FCmax']/1e9:g} GHz.", "",
            "Tau (s): " + ", ".join(f"{value:.9g}" for value in p["tau"]) + ".", ""])
    with (output/"per_task.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    (output/"summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=ROOT/"parameters.json")
    parser.add_argument("--scenarios", type=Path, default=ROOT/"scenarios"/"user_validation.json")
    parser.add_argument("--benchmarks", action="store_true", help="Run the earlier synthetic fixtures instead of the supplied cases")
    parser.add_argument("--tau-mode", choices=("large", "original"), default="large",
                        help="large: tau=1000000; original: infinite tau except E=1.10*Case C latency")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.parameters if args.benchmarks else args.scenarios
    settings = json.loads(source.read_text(encoding="utf-8"))
    if args.benchmarks and args.tau_mode != "large":
        parser.error("--tau-mode original applies to the supplied Cases A-F, not --benchmarks")
    if args.output is None:
        directory = "user_validation_original_tau" if args.tau_mode == "original" else "user_validation"
        args.output = ROOT/"results"/("cvxpy" if args.benchmarks else directory)
    solved = {}
    if args.benchmarks:
        inputs = [(f"n{n}_kappa{kappa:g}", "Earlier synthetic fixture", make_input(settings, n, kappa))
                  for kappa in settings["kappaD_cases"] for n in settings["populations"]]
    else:
        inputs = scenario_inputs(settings, args.tau_mode, solved)
    args.output.mkdir(parents=True, exist_ok=True)
    originals = sorted((ROOT/"originals").glob("*/*.m"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    if len(originals) != 2 or set(hashes.values()) != {ORIGINAL_SHA256}:
        raise RuntimeError("Expected the two unchanged, identical source attachments")
    manifest = dict(matlab_executed=False, backend="CVXPY/CLARABEL with exhaustive binary enumeration",
        uploads_identical=True, unique_solver_count=1, source_sha256=hashes, parameters=settings,
        scenario_file=str(source.resolve()), scenario_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        tau_mode=args.tau_mode,
        tau_seconds=None if args.tau_mode == "original" else settings.get("tau_seconds", 1e6),
        applied_deadlines=("Infinity for A-D and F; E=1.10*Case C latency" if args.tau_mode == "original"
                           else "1000000 seconds for every task"),
        json_infinity_encoding="Positive infinity is represented by the string Infinity; MAT files retain numeric infinity.",
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(), cvxpy=cp.__version__,
        caveat="These are numerical-equivalent results, not an execution of the MATLAB/CVX files.")
    write_json(args.output/"manifest.json", manifest)
    summaries, solutions, completed_inputs = [], [], []
    for label, description, p in inputs:
        n = len(p["C"])
        write_json(args.output/f"{label}.input.json", p)
        try:
            solution = solve_equivalent(p)
        except Exception as failure:
            write_json(args.output/f"{label}.failed.json",
                       dict(parameters=p, error=f"{type(failure).__name__}: {failure}"))
            raise
        result = dict(description=description, parameters=p, solution=solution)
        write_json(args.output/f"{label}.json", result)
        if solution["status"] != "optimal":
            raise RuntimeError(f"Scenario {label} unexpectedly infeasible; input and result preserved")
        reference = independent_reference(p)
        if reference is None:
            raise RuntimeError(f"Independent KKT reference found no feasible assignment for {label}")
        difference = abs(solution["optval"]-reference["optval"])
        if difference > 1e-6*max(1, abs(reference["optval"])):
            raise RuntimeError(f"Independent KKT check failed: {difference}")
        solution["independent_kkt_objective"] = reference["optval"]
        solution["independent_objective_difference"] = difference
        solution["minimum_deadline_slack_s"] = float(min(np.asarray(p["tau"])-solution["T"]))
        if args.benchmarks:
            result["robot_upload_constant_j"] = n*settings["robot_power_w"]*settings["input_bits"]/settings["access_bps"]
        result["note"] = "The supplied ED objective includes drone CPU and forwarding energy; it omits robot-upload energy."
        write_json(args.output/f"{label}.json", result)
        mat_solution = {key: value for key, value in solution.items() if key not in ("branches", "TD", "TC")}
        for key in ("TD", "TC"):
            mat_solution[key] = np.array([math.nan if val is None else val for val in solution[key]])
        savemat(args.output/f"{label}.mat", dict(parameters=p, solution=mat_solution), long_field_names=True)
        summaries.append(dict(case=label, description=description, agents=n, kappaD=p["kappaD"], status=solution["status"],
            objective=solution["optval"], x=" ".join(map(str, solution["x"])),
            weighted_latency_s=solution["latencyObjective"], energy_j=solution["energyObjective"],
            max_task_latency_s=max(solution["T"]),
            drone_cpu_ghz=sum(solution["fD_GHz"]), cloud_cpu_ghz=sum(solution["fC_GHz"]),
            assignments=solution["assignments_checked"], solve_seconds=solution["elapsed_seconds"],
            tau_seconds=min(p["tau"]), minimum_deadline_slack_s=solution["minimum_deadline_slack_s"],
            kkt_objective_difference=difference))
        solutions.append(solution)
        completed_inputs.append((label, description, p))
        solved[label] = solution
        print(f"{label}: optimal objective={solution['optval']:.9f} x={solution['x']} KKT difference={difference:.3g}", flush=True)
    with (args.output/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(args.output/"summary.json", summaries)
    if not args.benchmarks:
        write_reports(args.output, completed_inputs, solutions, summaries, args.tau_mode)
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
