# Extended UAV CPU experiment set

Parameter-only pilot results.

| Robots | Condition | Missions | UAV/cloud tasks | Cloud % | E2E mean/p50/p95 ms | On-time % | Makespan | Waiting | Energy J/episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | allocated-1ghz | 5/5 | 22/286 | 92.86 | 63.93/60.02/152 | 100 | 31.4 | 0.8 | 1.242 |
| 2 | allocated-6ghz | 5/5 | 162/146 | 47.4 | 55.73/50.99/62.87 | 100 | 31.4 | 0.8 | 1.7 |
| 2 | uav-6ghz | 5/5 | 308/0 | 0 | 50.77/39.14/62.87 | 100 | 31.4 | 0.8 | 3.261 |
| 2 | cloud-6ghz | 5/5 | 0/308 | 100 | 58.68/60.02/98.46 | 100 | 31.4 | 0.8 | 1.309 |
| 4 | allocated-1ghz | 5/5 | 34/470 | 93.25 | 77.7/62.04/202.5 | 96.03 | 34.2 | 4 | 2.025 |
| 4 | allocated-6ghz | 5/5 | 268/216 | 44.63 | 53.74/51.06/63.02 | 100 | 33.2 | 0 | 3.756 |
| 4 | uav-6ghz | 5/5 | 484/0 | 0 | 62.86/61.95/81.76 | 100 | 33.2 | 0 | 3.727 |
| 4 | cloud-6ghz | 5/5 | 0/484 | 100 | 65.92/61.99/99.86 | 100 | 33.2 | 0 | 2.058 |
| 8 | allocated-1ghz | 4/5 | 212/1228 | 85.28 | 102.5/63.94/202.8 | 95.35 | 38.5 | 16 | 5.531 |
| 8 | allocated-6ghz | 4/5 | 747/622 | 45.43 | 64.83/63.32/92.39 | 100 | 36.25 | 0.75 | 7.577 |
| 8 | uav-6ghz | 4/5 | 1372/0 | 0 | 90.14/94.23/144.3 | 100 | 35.5 | 0.5 | 8.224 |
| 8 | cloud-6ghz | 4/5 | 0/1377 | 100 | 77.14/63.91/127.6 | 100 | 36.25 | 0.75 | 5.856 |
| 16 | allocated-1ghz | 5/5 | 319/1498 | 82.44 | 173.9/107.7/591.6 | 94.11 | 41.2 | 25.4 | 6.747 |
| 16 | allocated-6ghz | 5/5 | 923/821 | 47.08 | 96.4/95.1/149.7 | 98.17 | 41.4 | 11.2 | 7.736 |
| 16 | uav-6ghz | 3/5 | 3095/0 | 0 | 152.9/164.2/268.1 | 84.91 | 55 | 93.67 | 13.55 |
| 16 | cloud-6ghz | 5/5 | 0/1723 | 100 | 115.2/106.3/204.9 | 99.65 | 41.2 | 5 | 7.332 |
| 32 | allocated-1ghz | 2/5 | 1060/5226 | 83.14 | 247.2/150.9/402.1 | 75.93 | 69.5 | 325.5 | 23.28 |
| 32 | allocated-6ghz | 2/5 | 2126/2410 | 53.13 | 138.6/130.3/245.5 | 95.24 | 57 | 64 | 17.32 |
| 32 | uav-6ghz | 0/5 | 40000/0 | 0 | 520/531.5/568 | 0 |  -  |  -  | 20.64 |
| 32 | cloud-6ghz | 3/5 | 0/5141 | 100 | 176.1/162.9/340.5 | 76.48 | 68 | 275.3 | 21.87 |

Parameters: UAV 6e9 cycles/s (1e9 for original allocator), cloud 1e10 cycles/s; 100000 bits at 1000 cycles/bit; 10/50 Mbps LEO; kappa 1e-28; energy/latency coefficients 1/1; 250 ms command cutoffs, one-second control, 250-step cap. Original frozen model, timeline and production kernels are retained.

Existing 2/4/8-agent records are reused unchanged. New populations use five paired cases and the same seed/geometry generator. Predicted solver-snapshot latency is separate from observed sensing-to-command delivery. CPU utilization/concurrent load, active-peer AoI, handover exposure, fixed prior assignments and energy are retained in the tables.

Files: combined_summary.csv/json, combined_episodes.csv, combined_comparisons.csv; summary.csv, handover.csv, placement_by_active_count.csv, episodes.csv, tasks.csv, rounds.csv and validation.json for the added populations. Raw audited per-case archives remain under episodes/. No learning updates occur.

## Combined analysis

The full set contains 100 primary simulations: five paired cases at each of 2/4/8/16/32 robots, with four conditions. The instantaneous MAPF check is a separate reference on the ten added cases.

| Robots | Instantaneous MAPF success |
| --- | --- |
| 16 | 5/5 |
| 32 | 2/5 |

Command timing and delayed state availability can change joint trajectories in both directions with a frozen policy. The reference and networked mission outcomes are reported separately.

The matching instantaneous reference uses per-agent inference; 6,682 comparisons with batched inference produced identical actions.

6 GHz allocator cloud shares: 2 robots 47.40%, 4 robots 44.63%, 8 robots 45.43%, 16 robots 47.08%, 32 robots 53.13%.

| Robots | First-round UAV tasks (mean) | First-round cloud tasks (mean) | Cloud tasks over mission (%) |
| --- | --- | --- | --- |
| 2 | 1.40 | 0.60 | 47.40 |
| 4 | 3.00 | 1.00 | 44.63 |
| 8 | 5.80 | 2.20 | 45.43 |
| 16 | 7.20 | 8.80 | 47.08 |
| 32 | 12.60 | 19.40 | 53.13 |

Initial-round allocations favor UAV execution at 2/4/8 robots and cloud execution at 16/32 robots. Mission-wide percentages also include later rounds with fewer active robots. `figures/first_round_placement.png` shows the initial task counts; `placement_by_active_count.csv` retains the active-population strata.

`combined_pairwise.csv` provides every paired treatment contrast with 95% intervals. Summary latency quantiles pool delivered tasks; paired p95 contrasts compare per-case p95 values. `placement_intervals.csv` gives case-clustered placement intervals. `combined_coordination.csv` retains ordering/action-disagreement numerators and denominators. `accounting_audit.json` verifies record identity and preservation. These are small parameter-only pilot cohorts on one map.

`case_outcomes.csv` lists paired mission outcomes, including the supplementary instantaneous reference for the added populations.

### Findings at 16 and 32 robots

At 16 robots, the 1 GHz and 6 GHz allocators completed 5/5 and 5/5 missions. On-time delivery changed from 94.11% to 98.17%; observed command p95 changed from 591.64 to 149.68 ms. Modeled energy per episode changed from 6.747 to 7.736 J.

At 32 robots, the 1 GHz and 6 GHz allocators completed 2/5 and 2/5 missions. On-time delivery changed from 75.93% to 95.24%; observed command p95 changed from 402.06 to 245.54 ms. Modeled energy per episode changed from 23.282 to 17.318 J.

All-cloud execution at 32 robots completed 3/5 missions, with 76.48% on-time commands. The paired outcomes include mixed changes in mission completion; command timeliness alone does not explain every outcome.

All-UAV execution at 32 robots completed 0/5 missions, with 0.00% on-time commands, 53.33% UAV CPU utilization and 32 concurrent jobs at peak. Round-level contention and the 250 ms cutoff remain important even with unused capacity between control rounds.

Cloud placement is not monotonic across the full population range. The 6 GHz setting improves command timeliness in the added cohorts. The paired mission counts and first-round versus mission-wide placement table show the limits of interpreting capacity changes as a fixed placement transition.
