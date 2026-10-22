> Completed 1,200-episode 250 ms study. This publication changes filenames and delivery notes only; the source results are unchanged.

# Full 250 ms evaluation: findings for review

Current-priority allocation reduces latency and preferentially serves high-priority robots, but it does not improve mission success or consistently improve mission duration. Modeled energy rises. These are mixed scheduling results, not evidence of an overall mission improvement.

All 1,200 episodes completed: six treatments, 50 paired cases for each of 2/4/6/8 robots, and a 250-step cap. Decision, command and task cutoffs are 250 ms; the control interval remains 1 s and hard solver deadlines remain disabled. Frozen checkpoint, maps, seeds, starts/goals and orbital offsets are identical within paired cases. Workload, CPU capacities, radio scheduling, PANAMA and movement rules were unchanged. The previous 100/300/500 ms pilots remain separate sensitivity evidence; none was rerun for this evaluation.

Primary comparison: current PANAMA priority minus matched equal-weight allocation. Intervals use 2,000 paired-case bootstrap samples, seed 42, retaining all tasks within cases. They are pointwise 95% intervals without multiplicity correction.

| Robots | Success, both treatments | Capped duration difference (s), 95% CI | Mean episode p95 difference (ms), 95% CI | Modeled energy difference (J), 95% CI |
| ---: | ---: | --- | --- | --- |
| 2 | 100% | +0.000 [+0.000, +0.000] | -35.436 [-40.020, -30.704] | +0.008 [+0.007, +0.009] |
| 4 | 98% | +0.000 [-0.120, +0.120] | -47.407 [-56.483, -37.696] | +0.082 [+0.056, +0.110] |
| 6 | 96% | +0.300 [+0.040, +0.680] | -49.961 [-62.951, -37.618] | +0.044 [+0.025, +0.066] |
| 8 | 98% | -0.120 [-0.280, +0.000] | -60.235 [-76.132, -45.866] | +0.026 [+0.010, +0.044] |

Success is identical in all six treatments: 50/50, 49/50, 48/50 and 49/50 cases. The four unsuccessful cases are shared with the fresh-coordination reference: agents-4-case-035, agents-6-case-023, agents-6-case-047 and agents-8-case-002. Inspection of the fresh traces shows sustained WAITs, oscillation, or a repeated blocked edge swap. They reach the cap without completing; priority allocation does not resolve them. A [0,0] bootstrap interval for success reflects no observed discordant pairs, not proof of equivalence.

The p95 result above averages each episode's p95 before taking paired differences. It differs from the pooled task p95: at four robots, the pooled p95 is approximately 202.456 ms in both treatments despite the lower mean episode p95. Both estimands and full latency distributions are saved.

| Robots | Overall on-time %, equal / current | UAV task share %, equal / current | UAV on-time %, equal / current | UAV compliance difference (percentage points), 95% CI |
| ---: | --- | --- | --- | --- |
| 2 | 100.000 / 100.000 | 7.29 / 7.29 | 100.00 / 100.00 | +0.000 [+0.000, +0.000] |
| 4 | 95.300 / 95.343 | 22.05 / 13.93 | 78.69 / 66.56 | -12.124 [-19.340, -6.555] |
| 6 | 94.273 / 94.291 | 21.34 / 19.30 | 73.16 / 70.43 | -2.733 [-4.778, -1.094] |
| 8 | 93.791 / 93.823 | 19.26 / 18.19 | 67.76 / 66.04 | -1.714 [-3.062, -0.595] |

All cloud commands arrive on time in the primary treatments. At eight robots, overall timeliness improves by only 0.032 percentage points (95% CI 0.001 to 0.069), while UAV-conditional compliance falls. The largest conditional decline is at four robots, approximately 12.1 percentage points. Routing changes the tasks left on the UAV; these site cohorts are not identical-task counterfactuals. The data do not support a universal service improvement. Across all six treatments there are 142,020 on-time and 6,516 late commands, zero undelivered commands, and zero censored or pending tasks at final observation.

| Robots | Additional high-versus-low latency preference (ms), 95% CI |
| ---: | --- |
| 2 | -3.652 [-4.132, -3.246] |
| 4 | -10.879 [-12.331, -9.383] |
| 6 | -8.737 [-10.209, -7.291] |
| 8 | -10.220 [-11.939, -8.606] |

This is the high-minus-low latency gap under current priority minus the same gap under equal allocation. Negative values indicate stronger preference for high-priority tasks. Independent physical-state strata corroborate this pattern. Deadline preference is less consistent: its high-minus-low compliance contrast excludes zero only at six robots. The strata use pre-policy communicated PANAMA weights [1,2), [2,3), [3,4], with physical-reference strata also saved. Trajectories, active populations and routes change with treatment, so this is descriptive service evidence, not a causal same-task effect. The existing robot-ID tie-break can spread near-equal priorities across the resource-weight range; a lone active robot receives weight 4.

Measured zero-rate CPU waiting is zero throughout these primary treatments. This does not mean unconstrained compute: jobs share processors and can use lower conditional CPU rates. Mean UAV/cloud utilization at eight robots is 10.43%/4.23% under equal allocation and 9.86%/4.30% under current priority. Task service above the C/F capacity floor, allocation intervals and radio queueing are recorded separately. Total modeled energy increases at every population despite lower UAV computation energy at 4/6/8 robots; additional cloud routing/backhaul contributes to this trade-off. Energy omits propulsion, static electronics, receiver electronics and cloud electricity.

FoV ordering errors are rare: both primary treatments have two supported errors at six robots and zero at the other populations. Their relevant-pair denominators differ. No paired FoV-inconsistency or actual-versus-cutoff-oracle disagreement improvement is established. At eight robots, current/equal have 827/819 action disagreements; missed vertex conflicts are 10/12 and false positives 30/30. Most physical conflicts at eight robots belong to the shared capped edge-swap case. Zero relevant-pair epochs remain explicit. The vertex-only metric remains exploratory; shadow oracle actions are diagnostics, not independently rolled-out oracle missions.

All six treatments are retained in summary.csv and comparisons.csv: fresh coordination, matched equal, current priority, episode-initial static priority, AoI-only, and legacy equal. Negative and mixed comparisons are retained. priority.csv includes latency and compliance by communicated and physical priority level; location_comparisons.csv contains the site-conditional intervals.

Validation: the full suite passes 235 tests with one existing low-elevation ITU-R approximation warning. All 1,200 replay audits pass, verifying 148,536 actual model actions, 148,536 per-cutoff oracle actions and 1,222,946 directed ordering records. Independent checks confirm paired-case identity, unchanged execution modules, deadline/priority/site denominators, empty groups and event integrity. The local checks.json records these checks. Inference remains limited to the fixed map, qualified checkpoint, archived orbital timeline and synthetic workload; the pointwise intervals do not establish generalization or family-wise significance.

The complete original run and manuscript material remain local; selected results are published here. See details.md for detailed estimator tables, plots/overview.png for the main contrasts, plots/*.png for individual contrasts, plots/latency_*.png for distributions, and method.json / location_method.json for provenance.
