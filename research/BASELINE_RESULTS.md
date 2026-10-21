# Completed nominal-resource comparison

The nominal comparison completed **1,000 qualified-policy episodes**: equal,
static-priority, AoI-only, current PANAMA-priority, and fresh-reference conditions,
with 50 held-out cases at each of 2/4/6/8 robots and a 250-step cap.

The raw records contain 487,862 generated support tasks, 483,862 delivered tasks,
and 74,092 allocation snapshots. All case, task and snapshot counts passed the
publisher's checks. Independent replay reproduced 1,760 recorded PANAMA decisions
across 20 sampled traces and verified source-time causality, binary execution and
the UL/DL/E2E ordering-error inequality. It also independently checked 10,944 saved
tasks for bit/cycle identities, predicted computation and transmission times,
transmitter energy, full-route round-trip propagation and observed stage sums.
See `baseline_loop_audit.json`.

The subsequent exhaustive audit reproduced **all 81,914 decisions in all 1,000
nominal episodes**. It also checked all **487,862 task records**, covering 74,092
allocation snapshots, with the same causal, physical and ordering identities.
The complete evidence is in `baseline_full_loop_audit.json`; the earlier sample
audit remains available separately.

At eight robots, current-priority versus equal weights gives:

| Outcome | Equal | PANAMA priority | Paired difference, 95% CI |
| --- | ---: | ---: | --- |
| Mean delivered-task latency | 147.27 ms | 143.63 ms | -3.63 ms [-4.5, -2.8] |
| Mission success | 30% | 34% | +4 percentage points [-6, +14] |
| Mean episode FoV inconsistency | 3.00% | 1.32% | -1.68 percentage points [-5.90, +0.77] |

The nominal latency difference is directional in its pointwise interval. The
completion and inconsistency intervals include zero; these results do not
establish a universal mission-success advantage for current-priority allocation.
Other populations and policies remain visible in the full tables and plots.

Equal weights are all 1; priority/AoI policies use their configured 1--4 range.
Weights are not normalized to the same sum, so these comparisons change aggregate
latency-versus-energy emphasis alongside relative prioritization. The full grid's
weight-ratio-one identity control will independently check the common-weight limit.

The archived model was qualified under fresh information. These experiments use
its frozen weights under delayed guidance and coordination; they do not retrain
it for the resource conditions. Low FoV inconsistency alone does not guarantee
mission success. Delivered-task means are conditional on delivery, and final-cycle
tasks remain undrained at episode termination.

The implemented action policy consumes local sensing and received caches at robot
decision epochs; the offloaded jobs provide guidance/coordination support. A fully
serialized remote action network would require a separate timing experiment.
The revised manuscript describes this implemented architecture explicitly.

An independent repeat inside the full grid reproduced all 1,000 nominal episodes'
physical outcomes and task metrics. Host-dependent solver wall time is excluded
from that equality check and never advances simulated time. See
`nominal_repeat_audit.json` for all checked quantities.

The full **97-condition / 19,400-episode sensitivity grid remains in progress**.
Its unfinished conditions are not presented as completed results. Raw nominal
traces remain at `output/resources/paper-baseline-20261021/`; tracked summaries,
case seeds, parameters and verification are in `baseline_results/`. The manuscript
already includes the completed nominal tables and scientific figures.
