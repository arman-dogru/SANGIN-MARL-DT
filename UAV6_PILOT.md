# Parameter-only UAV CPU pilot

This diagnostic tests whether the existing priority-aware allocator shifts work
toward the cloud as robot count increases when UAV capacity is raised to 6 GHz.
It uses a frozen PANAMA policy; there are no learning updates or injected
communication impairments. Results are parameter-only pilot evidence, not
validated paper findings.

## Configuration and execution conditions

[`configs/paper_remote_dt_250ms_uav6.toml`](configs/paper_remote_dt_250ms_uav6.toml)
copies [`paper_remote_dt_250ms.toml`](configs/paper_remote_dt_250ms.toml) with one
parsed change:

```toml
drone_cycles_per_second = 6e9
```

Cloud capacity remains 1e10 cycles/s, input remains 100,000 bits at 1,000
cycles/bit, LEO capacity caps remain 10/50 Mbps, UAV dynamic-compute coefficient
remains 1e-28, and latency/energy coefficients remain 1/1. Decision, command and
task cutoffs remain 250 ms with hard solver deadlines disabled. The control
interval remains one second. Antenna/EIRP, conducted power, orbital inputs,
handover, priorities, radio scheduling and robot dynamics remain unchanged.

| Condition | UAV capacity | Placement |
| --- | ---: | --- |
| `allocated-1ghz` | 1e9 cycles/s | Existing production allocator |
| `allocated-6ghz` | 6e9 cycles/s | Existing production allocator |
| `uav-6ghz` | 6e9 cycles/s | Existing all-UAV binary reference |
| `cloud-6ghz` | 6e9 cycles/s | Existing all-cloud binary reference; cloud stays 1e10 |

The isolated [`scripts/run_uav6_pilot.py`](scripts/run_uav6_pilot.py) reuses
`communication_effects.condition_config`, `inference_controls.inference_class`,
`evaluation.run_episode` and `staleness_audit.audit_stress`. It does not change
allocation, admission, CVXPY, transport, compute pools, movement or production
instrumentation. Fixed routes are experimental controls; the allocated arms
delegate admission to the existing production implementation.

## Cohort and qualification

The diagnostic requests 2/4/8/16 robots and uses five paired cases per supported
population with a 250-step cap. The archived checkpoint is qualified at
2/4/6/8 robots, so 16 is excluded explicitly. No checkpoint qualification is
bypassed. The resulting cohort contains 15 paired cases and 60 simulations.

The same nominal map, seeds, starts/goals and orbital offsets are retained from
the first five cases in each matching population of
[`results/communication/cases.json`](results/communication/cases.json).
Preparation checks these identities before execution. Treatment trajectories
can diverge after commands affect movement.

Frozen checkpoint SHA-256:
`be085bec93c4b8603629b0e964b10f7b5a99220386e9c968ac17f960759ffc0f`.
Archived timeline SHA-256:
`c24a66247cbf13e0b08a285e918b4b4e6115fe29d268e862cd5d5bd71efe187f`.
The original checkpoint is retained in
`output/leo-learning/leo-learning-20261009-01-recovery-03/clean/final_model.pt`
under the PANAMA+UAV project. A verified local copy is
`artifacts/inputs/final_model.pt`. The timeline is `data/leo/tremblant_full.npz`.
These external assets are not supplied by a fresh clone.

## Reproduction

From the repository root, with matching assets and dependencies installed:

```powershell
.venv\Scripts\python.exe scripts/run_uav6_pilot.py --root output/uav6_pilot/smoke-20261024-01 --prepare --launch --counts 2 --cases 1 --cap 3 --workers 2 --budget-seconds 300
.venv\Scripts\python.exe scripts/run_uav6_pilot.py --root output/uav6_pilot/uav6-20261024-01 --prepare --launch --counts 2 4 8 16 --cases 5 --cap 250 --workers 4 --budget-seconds 3600
```

Use a new output folder for a new run. Preparation freezes source, the isolated
harness, both configs, model, map, orbital timeline and case bank. Workers use
that frozen code, one CPU thread each, greedy actions and deterministic seeds.
`status.json`, per-worker JSON and stdout/stderr logs show progress. No recurring
monitoring task is created. A recorded launch cannot be duplicated, and previous
episodes cannot be overwritten or restarted by this harness. Solver failures
stop execution and retain the failing case/snapshot.

Rebuild reports without rerunning episodes:

```powershell
.venv\Scripts\python.exe output/uav6_pilot/uav6-20261024-01/code/run_uav6_pilot.py --root output/uav6_pilot/uav6-20261024-01 --report
```

Set `PYTHONPATH` to the run's absolute `code/` directory when rebuilding from an
environment where another package version is installed. All source hashes are
recorded in `manifest.json`.

Tests:

```powershell
.venv\Scripts\python.exe -m pytest -q --tb=short --basetemp tmp/pytest-uav6-review
```

Use a fresh temporary-directory name for each invocation.

## Measurements and interpretation

`summary.csv` reports UAV/cloud task counts and fractions, first-round placement,
observed command mean/p50/p95, deadline partitions, solver-predicted snapshot
latency, CPU utilization/concurrency, energy, mission outcomes and active-peer
AoI. The first round retains the full initial population; mission-wide counts
include later rounds with fewer active robots. The example placement hypotheses
are not supplied to the optimizer.

Observed latency runs from sensing to command arrival at the robot. Predicted
latency describes a fixed-input solver snapshot. Subsequent admissions, evolving
CPU shares, existing fixed assignments, complete wire payloads and response
queuing can separate the two. The energy-aware compute pools can use less than
the 6 GHz limit; capacity is an upper bound rather than a forced clock rate.

`handover.csv` retains exposure counts, timing, command misses and active-peer
AoI for tasks sensed inside/outside native handover/service windows, admissions
with available/unavailable backhaul, and admissions with/without fixed previous
jobs. Resource-reconfiguration windows and actual satellite attachment changes
are counted separately. These descriptive associations do not estimate an
isolated causal handover penalty. Task records retain command-path radio queue,
guard and interrupted-service times. Processor sharing has no separate FIFO
compute queue; concurrent service and elapsed compute time capture contention.

CPU statistics integrate existing allocation events over the full simulated
mission horizon, including idle intervals. Active-peer age excludes peers marked
completed by the fresh physical reference at inference and retains its sample
denominator. Energy covers existing robot uplink, UAV backhaul uplink and UAV
dynamic-compute terms; other platform costs remain unparameterized.

Every episode passes causal replay and frozen-policy action checks. Source,
base config and previously published results are hash-checked for preservation.
`comparisons.csv` gives paired bootstrap intervals against the 1 GHz allocator;
five cases and one map support diagnostic conclusions only. All raw archives,
checkpoints and the immutable run snapshot remain local.

## Completed diagnostic

The [published parameter-only pilot](results/uav6/report.md) contains 60 episodes,
8,746 tasks and 2,883 inference rounds. The 6 GHz allocator meets every observed
cutoff at 2/4/8 robots, but its cloud shares of 47.4%/44.6%/45.4% do not establish
an increasing cloud-placement transition. Energy rises and mission-success
counts remain unchanged. Sixteen robots remain outside checkpoint qualification.
The full suite passes 374 tests; the four-condition smoke and all 60 causal
replay/action audits pass.

The independent postprocessing harness adds load/availability strata and checks
the 1 GHz baseline and cloud-only control against preserved historical archives:

```powershell
.venv\Scripts\python.exe scripts/audit_uav6_pilot.py --root output/uav6_pilot/uav6-20261024-01 --previous output/communication_effects/inference-20261023-01
```

It requires the original local per-case transactions and historical archives.
Baseline frames match exactly. Cloud frames differ only in the event checksum
because allocation snapshots record the new unused UAV capacity; physical state,
commands, cache/timing fields and task arrivals match. The complete causal chain
is independently replayed for each run. `placement_by_active_count.csv` separates
mission population from actual active load, while `handover_details.csv` includes
cutoff-WAIT counts and compute elapsed time. The supplemental CPU plot is
reproducible with the published `analysis/plot_cpu.py` against the local run.
