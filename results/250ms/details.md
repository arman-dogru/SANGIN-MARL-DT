> Detailed estimators and confidence intervals from the completed study; original run provenance is retained.

# Full 250 ms priority-aware evaluation

Run commit: `9ea9d9a075acccac03b4c2c13df2c915df06e246`. Cases per population: 50. Diagnostic: False. Episodes: 1200.

Primary contrasts are current-priority minus matched equal-weight allocation. 95% percentile intervals resample matched cases, preserving all robots/tasks within a case. Successful makespan is conditional on both missions succeeding; capped duration includes failures. Pooled rates preserve task/pair numerator and denominator counts; unsupported rates remain missing.

## Primary mission comparison

| Robots | Metric | Difference | 95% CI | Paired cases |
| ---: | --- | ---: | --- | ---: |
| 2 | success | 0 | [0, 0] | 50 |
| 2 | restricted_makespan_seconds | 0 | [0, 0] | 50 |
| 2 | waiting_steps | 0 | [0, 0] | 50 |
| 2 | command_cutoff_misses | 0 | [0, 0] | 50 |
| 2 | task_latency_p95_seconds | -0.0354365 | [-0.0400201, -0.0307038] | 50 |
| 2 | modeled_energy_lower_j | 0.007808 | [0.0067344, 0.008784] | 50 |
| 2 | on_time_rate | 0 | [0, 0] | 50 |
| 2 | xi_fov_supported | 0 | [0, 0] | 50 |
| 2 | cutoff_action_disagreement_rate | 0 | [0, 0] | 50 |
| 4 | success | 0 | [0, 0] | 50 |
| 4 | restricted_makespan_seconds | 0 | [-0.12, 0.12] | 50 |
| 4 | waiting_steps | -0.02 | [-0.06, 0] | 50 |
| 4 | command_cutoff_misses | -0.04 | [-0.1, 0] | 50 |
| 4 | task_latency_p95_seconds | -0.047407 | [-0.0564834, -0.0376965] | 50 |
| 4 | modeled_energy_lower_j | 0.0821881 | [0.0556619, 0.110227] | 50 |
| 4 | on_time_rate | 0.000429042 | [-2.43579e-05, 0.00104415] | 50 |
| 4 | xi_fov_supported | 0 | [0, 0] | 50 |
| 4 | cutoff_action_disagreement_rate | 0.000451526 | [-0.000876424, 0.00173253] | 50 |
| 6 | success | 0 | [0, 0] | 50 |
| 6 | restricted_makespan_seconds | 0.3 | [0.04, 0.68] | 50 |
| 6 | waiting_steps | 0 | [-0.18, 0.2] | 50 |
| 6 | command_cutoff_misses | -0.02 | [-0.12, 0.1] | 50 |
| 6 | task_latency_p95_seconds | -0.0499612 | [-0.0629513, -0.0376175] | 50 |
| 6 | modeled_energy_lower_j | 0.0441399 | [0.0252087, 0.0657522] | 50 |
| 6 | on_time_rate | 0.00018458 | [-0.000451233, 0.00077271] | 50 |
| 6 | xi_fov_supported | -0.000267611 | [-0.00128707, 0.000171601] | 50 |
| 6 | cutoff_action_disagreement_rate | 0.00144598 | [-0.00113333, 0.00416719] | 50 |
| 8 | success | 0 | [0, 0] | 50 |
| 8 | restricted_makespan_seconds | -0.12 | [-0.28, 0] | 50 |
| 8 | waiting_steps | -0.1 | [-0.38, 0.1] | 50 |
| 8 | command_cutoff_misses | -0.08 | [-0.16, -0.02] | 50 |
| 8 | task_latency_p95_seconds | -0.0602346 | [-0.076132, -0.0458659] | 50 |
| 8 | modeled_energy_lower_j | 0.0263463 | [0.0104908, 0.0437682] | 50 |
| 8 | on_time_rate | 0.000322136 | [1.14268e-05, 0.000690263] | 50 |
| 8 | xi_fov_supported | 0 | [0, 0] | 50 |
| 8 | cutoff_action_disagreement_rate | 0.000861803 | [-0.00134501, 0.00307068] | 50 |

## Preferential service and limits

Priority-level tables use fixed reference PANAMA weights: low [1,2), middle [2,3), high [3,4]. The communicated reference is reconstructed before the treatment replaces weights; physical oracle levels are separate. Equal allocations therefore retain meaningful PANAMA strata. Missing/unadmitted tasks remain unknown. All cases, including zero-exposure cases, enter pooled-rate bootstrap samples.

Preferential-service differences compare the high-minus-low service gap under current priority with the same gap under matched equal allocation. Negative latency differences and positive compliance differences indicate stronger preference for high-priority tasks. These contrasts are separate from mission success/makespan effects. Different trajectories produce different post-treatment task strata, so this is descriptive service evidence rather than a causal same-task comparison. Reference weights also change with the active population: a lone active robot receives weight 4. Active-team sizes are retained in task records; high/low contrasts alone do not isolate causality.

| Robots | Priority source | Difference in high-minus-low gap | Estimate | 95% CI | Cases |
| ---: | --- | --- | ---: | --- | ---: |
| 2 | communicated | high_minus_low_latency | -0.00365217 | [-0.0041319, -0.00324595] | 50 |
| 2 | communicated | high_minus_low_compliance | 0 | [0, 0] | 50 |
| 2 | physical | high_minus_low_latency | -0.00365217 | [-0.0041319, -0.00324595] | 50 |
| 2 | physical | high_minus_low_compliance | 0 | [0, 0] | 50 |
| 4 | communicated | high_minus_low_latency | -0.0108791 | [-0.0123306, -0.00938302] | 50 |
| 4 | communicated | high_minus_low_compliance | 0.00052498 | [-0.000102173, 0.00161024] | 50 |
| 4 | physical | high_minus_low_latency | -0.0106051 | [-0.0120462, -0.00911401] | 50 |
| 4 | physical | high_minus_low_compliance | 0.000522689 | [-7.52478e-05, 0.00158124] | 50 |
| 6 | communicated | high_minus_low_latency | -0.00873735 | [-0.0102088, -0.00729077] | 50 |
| 6 | communicated | high_minus_low_compliance | 0.00236177 | [0.000814856, 0.00430795] | 50 |
| 6 | physical | high_minus_low_latency | -0.00854648 | [-0.010088, -0.00703251] | 50 |
| 6 | physical | high_minus_low_compliance | 0.00235317 | [0.000678366, 0.0044901] | 50 |
| 8 | communicated | high_minus_low_latency | -0.0102199 | [-0.0119388, -0.0086055] | 50 |
| 8 | communicated | high_minus_low_compliance | 0.00100444 | [-0.000421241, 0.00288322] | 50 |
| 8 | physical | high_minus_low_latency | -0.0103185 | [-0.0120003, -0.00871006] | 50 |
| 8 | physical | high_minus_low_compliance | 0.00132814 | [-0.000431582, 0.00357588] | 50 |

CPU service uses processor sharing. Zero-rate waiting is measured from allocation intervals; it is not a FIFO queue. Elapsed service beyond C/F includes sharing and DVFS, and is reported separately. Delivered-task latency excludes right-censored tasks; undelivered counts remain explicit. FoV/vertex values with no relevant directed pairs do not establish consistency. Vertex-only results remain exploratory. Energy excludes propulsion, static electronics and cloud electricity.

PANAMA priorities include the existing robot-ID tie-break. When path distances are equal, very small numerical priority differences can span the reference weight range. Service bins describe that implemented mapping; raw priorities and weights remain available for interpretation. Intervals are pointwise without multiplicity correction and are conditional on this fixed map, checkpoint, synthetic workload and orbital archive.

All six treatments and every measured mission contrast are retained in CSVs, including null, negative and mixed findings. Sensitivity pilots remain separate evidence and are not pooled into full-study inference. Actual-model inference, cutoff oracle coverage/actions, event checksums, case identity, CPU utilization and deadline denominators are replay-audited. See validation.json.
