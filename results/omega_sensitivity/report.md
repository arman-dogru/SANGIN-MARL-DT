# Latency–energy weight sensitivity

Parameter-only sensitivity results. The policy, workload, radio model, admission, binary allocator and compute scheduler are unchanged. The 16- and 32-robot cohorts are exploratory scale tests; checkpoint qualification metadata is unchanged.

The matrix contains 375 simulations: 5 weights × three routes × five populations × 5 paired cases, capped at 250 steps. No control results were reused.

## Findings

The tested weights change CPU rates, task identities and placement, but the cohort shows a modest transition in mission-wide routing. All settings use mixed UAV/cloud execution with two robots. None becomes cloud-majority at 16 robots. At 32 robots, cloud shares range from 48.83% to 57.79%; W2 has the largest cloud share. The hypothesized sharp low-population UAV to high-population cloud transition is not established by these cases.

W2 (latency 1, energy 2) provides a descriptive compromise: mean episode energy is 11.0–24.8% below W1, with the same success counts at every population. Smaller-population latency increases remain within the 250 ms cutoff. At 32 robots, the energy and per-case p95 confidence intervals include zero; the two successful missions also have different case identities, with only one common completed pair. The paired makespan result there rests on that single pair. These observations support follow-up testing of W2, rather than a general performance guarantee.

W3 gives the lowest mean modeled energy at every population, with higher observed p95 than W1. Its 32-robot success count is 1/5 versus W1’s 2/5. Higher latency coefficients give uneven tail-latency changes: W4 increases four-robot p95, and W5 increases eight- and 32-robot p95. W4 has the lowest allocated miss rate at 16 and 32 robots, while success falls from 5/5 to 4/5 and from 2/5 to 1/5, respectively. W5 reaches 3/5 successes at 32 robots while its miss rate rises to 5.40%. Deadline compliance and mission completion therefore do not improve together consistently in this cohort.

### Paired W2 minus W1 effects

Energy uses all five paired episodes. Latency is the mean difference in per-case p95, which differs from subtracting the pooled p95 values in the comparison table. Intervals are paired case-bootstrap 95% intervals.

| Robots | Energy difference (J/episode), 95% CI | Per-case p95 difference (ms), 95% CI | W1 / W2 successes |
|---:|---:|---:|:---:|
| 2 | -0.387 [-0.416, -0.357] | +15.20 [+15.20, +15.20] | 5/5 / 5/5 |
| 4 | -0.932 [-1.201, -0.712] | +12.37 [+8.89, +15.20] | 5/5 / 5/5 |
| 8 | -1.365 [-2.484, -0.587] | +4.00 [+1.28, +7.83] | 4/5 / 4/5 |
| 16 | -0.848 [-1.092, -0.638] | +0.67 [-6.82, +7.58] | 5/5 / 5/5 |
| 32 | -1.984 [-5.711, +1.330] | +4.83 [-3.23, +14.61] | 2/5 / 2/5 |

## Comparative results

The full 75-cell table is in [comparison_tables.md](analysis/comparison_tables.md). [summary.csv](summary.csv) includes all metrics; [episodes.csv](episodes.csv) retains all 375 case results.

## Research questions


1. **Higher latency/energy ratio:** 2 robots: W4/W5 p95 changes -16.7% / -16.7%; 4 robots: W4/W5 p95 changes +6.9% / -1.2%; 8 robots: W4/W5 p95 changes -2.7% / +3.5%; 16 robots: W4/W5 p95 changes -4.2% / -4.2%; 32 robots: W4/W5 p95 changes -0.6% / +3.8%.
2. **Higher energy/latency ratio:** 2 robots: W2/W3 energy changes -22.8% / -37.1%; 4 robots: W2/W3 energy changes -24.8% / -48.9%; 8 robots: W2/W3 energy changes -18.0% / -35.0%; 16 robots: W2/W3 energy changes -11.0% / -24.3%; 32 robots: W2/W3 energy changes -11.5% / -20.4%. Observed latency costs are given above.
3. **UAV preference at 2–4 robots:** 2 robots: W1 52.6%, W2 52.6%, W3 52.6%, W4 53.9%, W5 53.9%; 4 robots: W1 55.4%, W2 52.1%, W3 53.1%, W4 65.1%, W5 53.5%.
4. **Mixed allocation at eight robots:** W1: 54.6% UAV, W2: 56.0% UAV, W3: 60.2% UAV, W4: 53.6% UAV, W5: 52.6% UAV.
5. **Cloud dominance at scale:** 16 robots: none above 50%; 32 robots: W1, W2, W3, W5.
6. **Lowest miss rate:** 2 robots: W1, W2, W3, W4, W5; 4 robots: W1, W2, W3, W4, W5; 8 robots: W1, W2, W3, W4, W5; 16 robots: W4; 32 robots: W4.
7. **Single balance across populations:** 2 robots minimum energy W3, minimum p95 W4, 4 robots minimum energy W3, minimum p95 W5, 8 robots minimum energy W3, minimum p95 W4, 16 robots minimum energy W3, minimum p95 W4, 32 robots minimum energy W3, minimum p95 W2. These empirical minima do not establish a universal setting from five cases.
8. **Objective versus route/assignment constraints:** matched_snapshots.csv re-solves identical first-round admissions using each weight. W1: 0 placement changes in 310 matched admissions; W2: 51 placement changes in 310 matched admissions; W3: 142 placement changes in 310 matched admissions; W4: 45 placement changes in 310 matched admissions; W5: 58 placement changes in 310 matched admissions. These comparisons hold radio inputs, priorities and previous assignments fixed; mission-wide placement also reflects admission availability and changed trajectories. active_count.csv and handover.csv retain those exposure counts.
9. **Strict observed latency-and-energy dominance over both fixed routes:** 2 robots: none; 4 robots: none; 8 robots: none; 16 robots: W2; 32 robots: W1, W2, W3. Paired uncertainty is in comparisons.csv. Latency ties within 1 ns also give W3 a lower-energy, no-worse-p95 comparison against both controls at two and four robots. These are observed point-estimate comparisons; control_dominance.csv records the tie rule.

10. **Deadline compliance and mission completion:** 2 robots: success spans 5–5/5 and miss rates 0.00–0.00%; 4 robots: success spans 5–5/5 and miss rates 0.00–0.00%; 8 robots: success spans 4–4/5 and miss rates 0.00–0.00%; 16 robots: success spans 4–5/5 and miss rates 1.42–1.89%; 32 robots: success spans 1–3/5 and miss rates 4.61–5.40%. The paired comparisons describe association; finite cases and changing trajectories limit attribution.

## Definitions and validation

Observed E2E latency is robot command arrival minus sensing time. Predicted latency is the solver admission snapshot estimate and excludes future scheduling changes. Quantiles pool delivered tasks within each cell; undelivered tasks remain in deadline and task denominators. A miss is late plus undelivered; pending/censored counts are reported separately. The runtime cutoff is 250 ms and hard solver deadlines remain disabled.

Energy is the existing modeled lower/upper mission tally, with transmitter and UAV-computation components. Cloud-computation and platform propulsion energy are not added. Per-task energy divides total mission energy by generated tasks, including unsuccessful missions; successful-only energy and failed-episode energy are separate fields. Longer failed episodes can increase total energy without a change in per-task efficiency.

CPU utilization divides allocated-capacity time by the complete observed mission duration, including idle periods. Mean concurrent jobs while active divides job-seconds by occupied pool-seconds. Existing processor sharing has no separate waiting queue: service contention appears in compute elapsed time and concurrent-job telemetry.

First-round counts average the five cycle-zero admissions per case; mission-wide percentages use all generated tasks. Active-count and handover tables provide subgroup denominators. Completed-mission makespan and all-episode waiting counts are retained. Paired percentile-bootstrap 95% intervals use 2,000 case resamples; five-case intervals are descriptive and small-sample uncertainty remains substantial.

Every episode passes the existing causal/frozen-action replay before it is committed. Baseline reproduction compares physical frames/event chains, binary allocations, CPU service, task timing and mission metrics. Only real host timings are excluded from exact comparison; fractional solver diagnostics have a recorded numerical tolerance. Protected production and historical result hashes are checked before and after execution.

## Provenance and reproduction

- Model SHA256: `be085bec93c4b8603629b0e964b10f7b5a99220386e9c968ac17f960759ffc0f`; learner updates: 502905; policy updates during study: 0.
- Timeline SHA256: `c24a66247cbf13e0b08a285e918b4b4e6115fe29d268e862cd5d5bd71efe187f`; map SHA256: `4240fddfa77d88b72ce779e02acf46a5ff056a3b04af5a4e35f7bc86cdfba3ec`.
- Cases, seeds, starts/goals, orbital offsets and configuration/source hashes are in cases.json and manifest.json.
- Run: `.\.venv\Scripts\python.exe scripts/run_omega_sensitivity.py --root output/omega_sensitivity/<new-id> --prepare --launch`.
- Resume only a stopped incomplete run: append `--resume` to `--launch` with the same root. Existing case transactions cannot be overwritten.
- Rebuild reports: `.\.venv\Scripts\python.exe scripts/run_omega_sensitivity.py --root <run-root> --report`.
- Full causal trace archives and checkpoints remain local under the run directory. Exported CSVs, configurations, report and figures are self-contained summaries.

## Interpretation and robustness

For W1, unavailable backhaul accounts for 6.00–7.23% of admissions and 11.81–13.58% of UAV assignments across populations. Most UAV assignments occur while backhaul is available. The 310 matched admissions produce 51/142/45/58 placement changes under W2/W3/W4/W5, respectively, with radio inputs, priorities and previous assignments fixed. This demonstrates a direct objective effect. A unique decomposition of whole-mission effects remains limited by state feedback and fixed previous assignments.

W1’s mean radio queue contribution rises from 1.19 ms at two robots to 46.23 ms at 32 robots, alongside a mean observed-minus-predicted latency gap rising from 1.63 to 41.15 ms. The mean number of fixed previous jobs rises from 0.49 to 9.48. Cloud compute peaks at one simultaneous job in these W1 runs, while UAV peaks reach 24 at 32 robots. Radio queueing and the difference between admission CPU predictions and later processor sharing explain why snapshot latency and delivered-command latency diverge.

The all-UAV 32-robot control misses every deadline despite approximately 53.3% mission-time CPU utilization: the one-second control interval includes idle time after the 250 ms cutoff. Its 32 tasks require 3.2 billion cycles per round, or at least 533 ms of aggregate service at 6 GHz, even before radio overhead.


All latency values use the simulator clock. Host inference/solver wall times are diagnostic and do not drive modeled CPU service. Cloud-computation energy is absent from the preserved objective and mission energy model. Weight effects on modeled energy therefore do not establish whole-system hardware energy savings.

The paired cases and orbital offsets match within each population. Maps/initial conditions and orbital offsets differ between populations, so cross-population trends cannot be attributed to robot count alone. First-round admissions, active-count strata and route-availability strata help interpret that limitation.

Stable placement percentages can hide changes in which priority levels run at each location. analysis/priority_by_location.csv reports admission resource weights by execution location; the priority policy and range remain unchanged. two robots, W1: mean UAV resource weight 1.20, active-job UAV frequency 1.75 GHz; two robots, W3: mean UAV resource weight 1.20, active-job UAV frequency 1.10 GHz; two robots, W4: mean UAV resource weight 3.80, active-job UAV frequency 3.29 GHz; two robots, W5: mean UAV resource weight 3.80, active-job UAV frequency 4.06 GHz.

The confidence intervals in comparisons.csv resample paired cases. The latency contrast is the mean difference in per-case p95, whereas summary.csv pools delivered tasks to compute each cell p95. These are distinct estimands. Task counts vary with mission duration and completion, and generated-task denominators remain explicit.

analysis/availability_attribution.csv separates unavailable-backhaul admissions from available-route choices, including the fraction of UAV assignments exposed to unavailability. Fixed previous jobs remain in each snapshot. The matched-snapshot solves identify direct objective effects under those fixed inputs; a unique mission-wide decomposition of objective, availability and trajectory feedback is not identified by this parameter-only design.

Matched admissions: 310 identical W1 snapshots, each solved at all five weight ratios. Of 1240 adjacent-ratio checks, 0 violate the expected weighted-latency/energy monotonicity beyond tolerance. The objective uses a priority-weighted sum of snapshot latencies. The reported tail percentile and command-deadline rate describe other aspects of the realized schedule, including future radio service and admission feedback.

All-cloud invariance: 100/100 paired weight-versus-W1 cases match the reported execution metrics within 1e-9. All simulations were executed; no controls were reused. Detailed differences, if present, are retained in analysis/all_cloud_invariance.csv.

| Robots | Weight | Cloud (%) | E2E p95 (ms) | Energy/episode (J) | Energy/generated task (J) | Misses (%) | Success | Empirical Pareto |
|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 2 | W1 | 47.40 | 62.87 | 1.700 | 0.02759 | 0.00 | 5/5 | yes |
| 2 | W2 | 47.40 | 78.07 | 1.313 | 0.02131 | 0.00 | 5/5 | yes |
| 2 | W3 | 47.40 | 97.22 | 1.069 | 0.01736 | 0.00 | 5/5 | yes |
| 2 | W4 | 46.08 | 52.37 | 4.338 | 0.07089 | 0.00 | 5/5 | yes |
| 2 | W5 | 46.08 | 52.37 | 6.397 | 0.10453 | 0.00 | 5/5 | no |
| 4 | W1 | 44.63 | 63.02 | 3.756 | 0.03881 | 0.00 | 5/5 | yes |
| 4 | W2 | 47.93 | 78.22 | 2.825 | 0.02918 | 0.00 | 5/5 | yes |
| 4 | W3 | 46.90 | 97.37 | 1.919 | 0.01982 | 0.00 | 5/5 | yes |
| 4 | W4 | 34.92 | 67.39 | 5.204 | 0.05376 | 0.00 | 5/5 | no |
| 4 | W5 | 46.49 | 62.29 | 7.315 | 0.07557 | 0.00 | 5/5 | yes |
| 8 | W1 | 45.43 | 92.39 | 7.577 | 0.02767 | 0.00 | 4/5 | yes |
| 8 | W2 | 44.04 | 101.46 | 6.212 | 0.02272 | 0.00 | 4/5 | yes |
| 8 | W3 | 39.82 | 120.44 | 4.926 | 0.01803 | 0.00 | 4/5 | yes |
| 8 | W4 | 46.42 | 89.91 | 13.152 | 0.04800 | 0.00 | 4/5 | yes |
| 8 | W5 | 47.37 | 95.65 | 17.423 | 0.06368 | 0.00 | 4/5 | no |
| 16 | W1 | 47.08 | 149.68 | 7.736 | 0.02218 | 1.83 | 5/5 | yes |
| 16 | W2 | 46.56 | 155.88 | 6.888 | 0.01956 | 1.82 | 5/5 | yes |
| 16 | W3 | 47.31 | 172.31 | 5.855 | 0.01677 | 1.89 | 5/5 | yes |
| 16 | W4 | 48.04 | 143.43 | 15.600 | 0.03562 | 1.42 | 4/5 | yes |
| 16 | W5 | 48.62 | 143.43 | 19.419 | 0.04454 | 1.42 | 4/5 | no |
| 32 | W1 | 53.13 | 245.54 | 17.318 | 0.01909 | 4.76 | 2/5 | no |
| 32 | W2 | 57.79 | 243.31 | 15.334 | 0.01721 | 4.74 | 2/5 | yes |
| 32 | W3 | 50.98 | 248.10 | 13.793 | 0.01530 | 4.73 | 1/5 | yes |
| 32 | W4 | 48.83 | 244.06 | 31.990 | 0.03246 | 4.61 | 1/5 | no |
| 32 | W5 | 51.24 | 254.98 | 25.834 | 0.02982 | 5.40 | 3/5 | no |

Pareto membership compares only the tested allocated settings using pooled p95 latency and mean episode energy. It is a descriptive pilot result and does not account for statistical uncertainty or impose a preferred allocation split. Success Wilson intervals and the full paired contrasts are retained in the CSVs.

### Reproduction audit correction

The initial harness rejected an unused fractional minimizer distance; the maximum recorded fractional-x difference is 0.000137363401. The original rejection and status are preserved in baseline_validation_v1.json and status_v1_gate_failure.json. The revised audit retains exact executed-output matching and checks fractional objectives and residuals against the unchanged solver tolerances. Production, simulator, policy and solver changes: zero. Completed episodes were not rerun. validation_revision.json records the isolated audit/controller hashes.

Primary simulation/reproduction/report workflow wall time: 7.367 hours, including baseline reproduction, audit recovery and report generation. The fixed 24-hour lease was preserved during recovery. No learning updates occurred.
