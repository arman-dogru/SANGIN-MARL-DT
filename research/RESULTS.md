# Completed PANAMA allocation study

The full qualified-policy study `paper-allocation-20261021` completed all **19,400
episodes across 97 conditions**. Each condition uses the same 50 held-out cases
at each of 2, 4, 6 and 8 robots, with a 250-step cap. The case bank, complete
configuration, package versions and source/model/channel hashes accompany the
[published results](results/verification.json). No condition is omitted.

## Records and scope

| Record | Count |
| --- | ---: |
| Episodes | 19,400 |
| Generated coordination-support tasks | 10,177,608 |
| Tasks whose guidance reached the requesting robot | 10,077,968 |
| Unfinished tasks at episode termination | 99,640 |
| Allocation snapshots | 1,714,431 |

Every episode's metrics are in [CSV](results/episodes.csv) and
[compressed JSONL](results/episodes.jsonl.gz). The report also includes
trace-backed allocation, task/transmission and coordination/decision samples,
with source-trace hashes and export hashes. Complete raw task/allocation logs and
compressed traces remain locally in
`output/resources/paper-allocation-20261021/` (approximately 136 GB).

The frozen action network executes at robot decision epochs using local sensing
and delivered caches. Offloaded finite-CPU jobs provide guidance and coordination
support. The results validate that feedback loop; they do not measure serialized
remote neural-network inference. Workload sizes and processor energy coefficients
are experimental assumptions rather than hardware measurements.

## Measured comparisons

The following eight-robot results use current PANAMA-priority weights. Differences
are paired against the same policy at nominal resources. Each interval uses 50
matched cases and 2,000 bootstrap resamples; intervals are pointwise, without a
family-wise multiple-comparison guarantee.

| Resources | Mission completion | Mean delivered-task latency | Mean episode FoV inconsistency |
| --- | ---: | ---: | ---: |
| Nominal | 34% | 143.63 ms | 1.322% |
| One-quarter common CPU | 88% | 495.91 ms | 1.194% |
| Fourfold common CPU | 86% | 89.75 ms | 2.879% |
| Fourfold uplink and common CPU | 98% | 67.03 ms | 0.333% |

For jointly fourfold uplink and CPU capacity, the paired differences from nominal
are **+64 percentage points** in completion (95% CI [50, 78]), **-76.61 ms** in
task latency ([-77.85, -75.28]), and **-0.989 percentage points** in FoV
inconsistency ([-2.444, -0.067]).

The CPU comparisons demonstrate a non-monotonic mission response: both increasing
and decreasing common CPU capacity improve completion over nominal, with opposite
latency effects. Discrete guidance/cache versions, delayed self and peer state,
whole-task routing and the frozen policy can all affect the loop. These results
do not identify a unique causal mechanism or establish that lower task latency
always improves coordination.

At nominal resources, priority versus equal weights reduces mean task latency
by 3.63 ms (95% paired CI approximately [-4.5, -2.8] ms). The completion difference
is +4 percentage points ([-6, 14]); the FoV inconsistency difference is -1.678
percentage points ([-5.90, 0.77]). Those latter intervals include zero. Full
results for every population, policy and sensitivity are in
[summary.json](results/summary.json), including paired comparisons with equal
weights, fresh coordination and the same policy's nominal resources.

Equal weights are all 1; the other policies use the bounded configured range.
The total weight is not normalized across policies, so policy comparisons include
both relative priority and a change in aggregate latency-versus-energy emphasis.
The ratio-one identity controls independently confirm equal execution inputs.

## Independent validation

- The regression suite passes **121 tests**. Ten predictable
  [sanity scenarios](solver-sanity-20261021/summary.json) agree with an independent
  primal CVXPY formulation and exhaustive binary enumeration, including high
  satellite delay, slow links/processors, unavailable routes and deliberate DVFS
  idling. [SOLVER_VALIDATION.md](SOLVER_VALIDATION.md) records the expanded checks
  and hand-calculated routing/CPU-sharing cases.
- The [physical-model audit](physics_audit.json) verifies RF units, access and
  backhaul budgets, conducted power versus EIRP, complete round-trip propagation
  and the small UAV-displacement approximation.
- [Full-grid replay](full_loop_audit.json) covers 388 traces: the first case at
  every condition and population. It reproduces **38,688 decisions**, checks
  **241,336 tasks** and **38,160 allocation snapshots**, including capacity,
  binary execution, source-time causality and UL/DL/E2E identities. Its largest
  solver residual is 5.315e-7, below the 1e-6 acceptance threshold. The largest
  relative relaxation gap in this sample is 4.761%; this is not an exhaustive
  maximum over the full study.
- [Exhaustive nominal replay](baseline_full_loop_audit.json) reproduces all
  81,914 decisions and checks all 487,862 tasks in the separate 1,000-episode
  nominal run. Its physical outcomes also match the full-grid nominal repeat;
  see [repeat audit](nominal_repeat_audit.json).
- All **1,000 weight-identity comparisons** pass independently;
  see [identity audit](weight_identity_audit.json).
- [Record counts](full_record_audit.json) and the
  [publication verification](results/verification.json) cover every completed
  case. `verify_published_records.py` additionally checks export hashes and case
  identities against the actual committed Git blobs.

## Interpretation and reproduction

Delivered-task latency ends at guidance arrival; coordination-state delivery is
measured separately through source-time AoI and inconsistency. Tasks are not
drained after termination, so conditional delivery means must be read alongside
unfinished counts and mission-energy bounds. A zero FoV relevance denominator
gives zero error. Conflicts absent from cached intentions are outside this metric;
low FoV inconsistency does not establish collision freedom or mission success.

Orbital offsets are paired within each population. Different populations sample
different archive windows, so population curves do not isolate team size from
orbital scenario effects. The source-time AoI plotted in the paper averages each
receiver's oldest accepted state age, including self; the AoI-weighted ordering
metric instead uses each pair's maximum operand age.

[REPRODUCE.md](REPRODUCE.md) contains commands for inputs, runs, audits, exports
and the manuscript. [RESOURCE_ALLOCATION.md](../RESOURCE_ALLOCATION.md) documents
the equations and assumptions. Figures can be regenerated from saved measurements.
The [comparative plots](results/figures/README.md) are included with the data in
PNG, SVG and PDF, with hashes in the publication verification record.
Further manuscript work is deferred; this delivery covers implementation, tests
and completed simulation data.
