# Equation 16 resource allocation and paper-mode experiments

## Running the implementation

Use Python 3.11+ and install the declared extras:

```powershell
python -m pip install -e ".[dev,optimization,leo]"
python -m panama_uav allocation-snapshot --input tests/fixtures/allocation_snapshot.json --output output/resources/snapshot.json
python -m panama_uav --config configs/resource_study.toml resource-conditions --primary-only
```

The snapshot command writes the input, fractional optimum, exact binary optimum,
absolute and relative relaxation gaps, component latencies and energies, normalized
allocations, conditional CPU rates, solver residual, and solver wall time. A solver
failure writes a `.failed.json` snapshot and raises an error. There is no fallback
to a heuristic allocation.

A short, explicitly untrained diagnostic requires no checkpoint or orbit download:

```powershell
python -m panama_uav --config configs/resource_study.toml resource-study --run-id resource-diagnostic-01 --diagnostic --cases 1 --agents 2 4 6 8 --max-steps 6 --primary-only
```

For an orbital resource study, prepare the qualified clean checkpoint and archived
full-route timeline with `scripts/prepare_paper_inputs.py`. Pass the checkpoint
and timeline paths using `--checkpoint` and `--timeline`. These large inputs and
their generated provenance records remain local. Choose a new run ID; existing
output directories are rejected unless explicitly resuming the same case bank.

```powershell
python -m panama_uav --config configs/paper_study.toml resource-study --run-id resource-paper-01 --model artifacts/inputs/final_model.pt --workers 8
python -m panama_uav resource-report --path output/resources/resource-paper-01
```

Non-diagnostic runs enforce the existing qualification gate, all populations
2/4/6/8, 50 held-out cases per population, the 250-step cap, and an orbital timeline.
The same frozen model and case seeds are used in every condition. This command does
not train a new policy. Diagnostic records and manifests explicitly identify an
untrained model when no checkpoint was supplied; their results are not paper evidence.

## Mathematical model

The reference is the supplied four-page draft, *Joint SAGIN Resource Allocation and
Multi-Agent Path Finding for Digital-Twin-Supported Remote Robotic Applications*.
In that draft, the optimization is Equation 16, the deadline is 16b, the power bound
is 16c, and Equation 18 is the closed-form drone allocation. The implementation
removes the task-deadline constraint and fixes transmit powers and radio rates for
each scheduling snapshot. Cloud computing is on the ground, reached through LEO.

For task i, let C_i=s_i*c_i, d_G,i=s_i/R_G,i, and d_A,i=s_i/R_A,i. The original
transmit-energy objective is

    sum_i [omega_E * (P_i*d_G,i + P_D*(1-x_i)*d_A,i) + omega_T*pi_i*T_i].

The first energy term is robot-upload energy and is constant in a fixed snapshot.
The second is drone backhaul transmit energy. Powers are conducted watts, not EIRP.
Propagation and idle time do not consume transmit energy. The editable manuscript
also includes drone dynamic CPU energy, so `configs/paper_study.toml` adds

    E_D,i = x_i*kappa_D*C_i*f_D,i^2,

with kappa_D=1e-28. In the relaxation its closed perspective is
kappa_D*C_i*F_D^2*a_i^2/x_i. The default kappa=0 retains the original
transmit-only model. Robot-upload energy is recorded in every mode; paper mode
omits this constant from the objective through `include_robot_upload_in_objective=false`.
Propulsion, receive electronics, static CPU power and cloud electricity remain
outside the modeled energy. These are simulation assumptions, not hardware data.

For the scheduling relaxation, define a_i=x_i*f_D,i/F_D and
b_i=(1-x_i)*f_C,i/F_C. Then

    T_i = d_G,i + (1-x_i)*(d_A,i + T_prop)
          + (C_i/F_D)*x_i^2/a_i + (C_i/F_C)*(1-x_i)^2/b_i.

The constraints are sum(a)<=1, sum(b)<=1, 0<=a_i<=x_i<=1, and
0<=b_i<=1-x_i. The individual upper bounds are a valid tightening of the binary
model: conditional task CPU rates cannot exceed the entire processor. The recovered
normalized conditional rates are a_i/x_i and b_i/(1-x_i), with unused branches zero.

Each quadratic-over-linear epigraph uses the exact cone

    norm([2*z, q-u]) <= q+u, q>=0, u>=0,

which represents z^2<=q*u and includes z=u=q=0. No artificial positive allocation
floor is needed. CVXPY verifies DCP and Clarabel solves the SOCP. Physical values are
in bits, bits/s, cycles, cycles/s, seconds, watts, and joules. Objective reference
units are 1 J and 1 s; both coefficients default to 1 and latency weight stays positive.

The exact binary reference enumerates at most eight undecided tasks (256 routes).
For each route, latency-only pools use the Equation 18/19 share proportional to
sqrt(pi_i*C_i). With drone dynamic energy, the unique positive rates instead obey

    -omega_T*pi_i*C_i/f_D,i^2 + 2*omega_E*kappa_D*C_i*f_D,i + lambda_D = 0,

with nonnegative capacity multiplier and complementary slackness. A scalar
multiplier search solves the bounded CPU subproblem. Capacity can remain idle:
forcing full utilization would increase the DVFS objective in that case. Cloud
energy is unmodeled, so its square-root allocation still applies. Existing
unfinished tasks may also be present; their routes remain fixed. The continuous
solution is a lower bound for this same frozen
snapshot, including the fixed previous assignments. It is not a guarantee about
the optimality of an entire dynamic mission or its realized latency.

## Executable service and assumptions

`resource_allocation.enabled=false` is the legacy default. Paper mode requires
physical radio service or a perfect reference. It generates one logical task per
robot per control cycle. Its input is the source report plus
synthetic padding. A batch is allocated only after all its input reports have
arrived at the drone, so the optimizer never reads newer robot truth. Batch waiting
is measured. Allocation uses report priorities and the self-coordination source
time piggybacked by each reporting robot.

The binary reference determines execution. Drone jobs enter the drone pool; cloud
jobs first traverse satellite uplink. Each finite pool recalculates shares from
remaining work on arrival/completion. Earlier routes are fixed and their remaining
cycles stay in allocation snapshots. Outstanding work cannot repeatedly receive a
fresh full CPU budget. Pool completions use cancellation tokens to invalidate
events superseded by new sharing rates.

The qualified implementation offloads guidance/coordination support. The frozen
action network consumes local sensing and received caches at each robot decision
epoch. This is the complete PANAMA--priority--allocation--delivery--PANAMA feedback
loop. Fully remote action inference needs a separate observation/action transport
experiment; the measured results do not establish its timing or performance.

Each completed task generates guidance G_i and a bundled S_i=(I_i,rho_i,g_i).
Guidance returns to its robot; S_i is sent to all receivers, including itself.
Cloud responses traverse satellite downlink and robot access. Results are actually
serialized even though the analytical result-download term is neglected. Incoming
generation time/version ordering and an additional source-time guard prevent
rollback, including older input jobs that finish after newer ones.

The initial mission briefing is fresh before the clock starts, as in the retained
simulator. A robot's local pose and obstacle/robot sensing remain current. Paper
mode's completion feature comes from cached self-state; intentions and priorities
share one source state. Available intentions are clipped cell by cell to the square
FoV, without the legacy Manhattan peer-distance gate. Observation tensor shapes
and neural weights remain unchanged. This is a frozen-policy sensitivity study.

Initial assumptions in `configs/paper_study.toml` (the older
`configs/resource_study.toml` retains transmit-only energy):

| Quantity | Value |
| --- | --- |
| Input size | 100,000 logical bits |
| Work per input bit | 1,000 cycles/bit |
| Drone / cloud processors | 1e9 / 1e10 cycles/s |
| Robot / drone transmit power | 0.2 / 2 W |
| Drone dynamic CPU coefficient | 1e-28 |
| Access uplink / downlink ceilings | 50 / 50 Mbit/s |
| Satellite uplink / downlink ceilings | 10 / 50 Mbit/s |
| Bounded resource weights | 1 to 4 |
| Control cycle / observation budget | 1 s / 100 ms |

The 100 ms budget determines when the policy consumes caches. It is not a task
deadline, feasibility constraint, or reason to discard an unfinished job.

G2A uses Equation 5 with the existing probabilistic-LoS path loss, an assumed
-174 dBm/Hz thermal noise density, 7 dB noise figure, 20 MHz bandwidth, and the
configured robot power. Its Shannon rate is capped by the access allocation.

For fixed-geometry diagnostics, A2S uses an assumed 550 km drone-satellite range,
3 dB atmospheric loss, 14.25 GHz, 20 MHz bandwidth, -5 dB/K satellite G/T, 3 dB
implementation loss, and efficiency 0.5. These follow the existing link-budget
parameterization. A 35 dBW terminal EIRP and 2 W conducted power imply approximately
32 dBi antenna gain. This is an engineering assumption, not a UAV hardware claim.
The configured one-way full-route delay is 25 ms: space legs are reported separately
using assumed 550 km drone and gateway ranges and the remainder is ground delay.

Orbital paper runs use an association-average derivative of the archived
ITU-R/geometry timeline. Range and dB attenuation are averaged per serving
association/scheduling interval and path loss/SNR/rates are recomputed; rates
themselves are not averaged. Original interruption windows, gateway bottlenecks,
guard intervals and the raw archive remain available. The archived attenuation
uses ITU-Rpy 0.4.0 and its P.618-13 default, with a fixed long-term exceedance
quantile rather than sampled instantaneous weather or small-scale fading.
Changing EIRP requires rebuilding that timeline. Effective rates are capped by the
configured allocation. Full-route one-way propagation includes both space legs and
ground delay. It is applied once outbound and once inbound, not doubled again on
each hop. Fragmentation does not sum one-way propagation once per chunk.

Application priorities retain the smaller-is-higher ordering. Completed agents
remain below every active agent, including an unreachable active path. Resources
use an inverse min/max map over active priorities; equal active priorities receive
the maximum weight, and completed agents always receive the minimum positive
weight. The policy comparisons are equal weights, episode-initial static weights,
AoI-only weights based on the reported self-state source time, and current priority
weights. AoI weights are clipped to the same configured bounds. Completed-task
weights retain the floor under every policy.

The equal-policy control uses the minimum weight (1), whereas tied active agents
under the priority map use the maximum weight (4 nominally). The sum of weights
is not normalized between policies. Comparisons therefore include changes in the
latency/energy emphasis as well as relative priority; they do not isolate a fixed
total priority budget. At weight ratio one, all four policies receive identical
weights and must reproduce equal-policy execution for the paired case bank.

## Measurements and experiments

Equations 28--36 are evaluated per directed receiver-peer pair. Vertex/edge
conflicts use cached self and peer intention segments and the receiver's current
square FoV. The horizon includes the current cell plus five future cells; short
paths repeat their final cell. Ordering errors separately compare generated DT
priorities with physical priorities (UL), receiver priorities with generated DT
priorities (DL), and receiver priorities with physical priorities (E2E). Both
operands use the same directed relation. UL and DL errors may cancel, and E2E
never exceeds their sum. Physical truth is reserved for evaluation.
AoI weighting uses the larger cached operand age
and a 1 s reference. Store raw ordering errors, relevant-pair counts, relevant
errors, and pair-level source ages; a zero relevant-pair denominator gives zero.
This metric is measured and does not enter the allocation objective.
The plotted coordination AoI is the receiver's oldest accepted state-source age,
including self, averaged over receivers and recorded epochs. It is distinct from
the pair-level maximum age used in the AoI-weighted inconsistency. Recorded epochs
include the initial decision snapshot and terminal sample, with actual timestamps.

The primary sweep contains the 3x3 uplink/common-CPU multiplier grid {0.25,1,4}
under all four policies, plus a fresh-coordination reference: 37 conditions.
Secondary sweeps independently vary drone CPU, cloud CPU, downlink, input bits,
and cycles/bit by 0.25 and 4, weight ratios by 1 and 16 (baseline 4), and energy
coefficients by 0, 0.1 and 10 (baseline 1): 97 total conditions. Fresh and equal-policy
comparisons are paired by case ID, with seeded 2,000-sample bootstrap intervals.
Each policy also has paired comparisons against its own nominal resource
condition, isolating resource changes while keeping the weight policy fixed.

Outputs include a manifest with full configurations, code/model/map/timeline/case
hashes and package versions; compressed episode traces; allocation and task JSONL;
CSV tables; summaries; and PNG/SVG plots. Task records distinguish logical input
size from wire overhead, predicted snapshot latency from observed task latency,
and original work from outstanding remaining cycles. Source/generation/delivery
times and per-chunk radio service permit stage-by-stage audit.
Task latency and delivered-task counts use guidance arrival at the requesting
robot. Broadcast coordination states have separate per-receiver deliveries,
including self, which are measured through cache AoI and directed inconsistency.

Transmit energy is credited only when a service chunk completes. Delivered-task
energy is exact within the simulation. At episode termination an unfinished chunk
can have consumed some energy; mission totals therefore expose lower/upper bounds
using its reserved airtime rather than presenting an invented exact total. Pending
and delivered task counts accompany latency/energy averages to expose censoring.
Solver wall time is logged separately and is not added to simulated task CPU time.
Drone CPU energy integrates kappa_D*f_i(t)^3, including elapsed unfinished service;
CPU utilization includes idle intervals.

The acceptance suite covers conic feasibility and binary bounds, closed-form CPU
shares, unavailable routes, energy units, fragmentation, asymmetric/full-route
propagation, overlapping and overloaded processors, atomic and out-of-order cache
delivery, stale self state, completed priority, FoV boundaries and paired reporting.
Neither beneficial policy differences nor monotonic mission improvement are forced.

## Validation and output locations

Run the predictable scenarios and regression suite from the project root:

```powershell
python -m panama_uav allocation-sanity --output output/sanity-new
python -m pytest -q --basetemp tmp/pytest-new
```

Use a new output and temporary directory for each invocation. Predictable cases
cover high propagation delay, slow finite backhaul, slow finite cloud CPU,
unavailable routes, priority-weighted CPU sharing and energy-aware CPU idling.
The independent primal CVXPY reference enumerates binary assignments without
using the production conic substitution or KKT helpers. Hand-calculated tests
cover the single-task 80 ms routing threshold, equal-processor load balancing
and the 2:1 CPU ratio for weights 4:1 with equal work. These are controlled
test inputs, not hardware or mission-performance measurements.

The repository retains source code, configurations, supplied input scenarios and
curated regression fixtures. Run logs, allocation snapshots, solver reports,
figures and manuscript drafts remain local under ignored output directories.
The existing `paper/`, `research/` and `cvx_solvers/results/` directories are not
versioned. No experimental results or custom manuscript are required to run the
implementation or its tests.
