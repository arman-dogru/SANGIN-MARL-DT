# Frozen PANAMA communication inference: findings

The nominal cloud, allocated and instant-reference arms each completed 39/40 missions. All-UAV completed 10/40. Command timing and waiting differ even where completion counts match. Deliberate peer staleness changed DQN decisions even with every cloud command on time. The strongest injected LEO delay prevented completion in both cloud-capable arms.

## Configuration and evidence

Forty paired cases cover 2/4/6/8 robots, ten per population, on the archived 32×32 nominal map. Each case retains identical starts/goals, seed, policy and orbital offset across 41 conditions. There are 160 primary simulations, 120 matched ideal-network simulations and 1,360 single-factor sensitivity simulations. All episodes retain the 250-step cap, one-second control and 250 ms command/task cutoffs; hard optimizer deadlines remain disabled. The instant reference uses the existing fresh-state exception and removes modeled network/CPU service. Matched ideal-network controls retain finite CPU service and causal DT updates.

Frozen policy: `be085bec93c4b8603629b0e964b10f7b5a99220386e9c968ac17f960759ffc0f`. Package versions and frozen source hashes are in `manifest.json`; source base is `cbbbc04c9780e4a7dddaaed2949fde57d5a79187` with the recorded working-tree snapshot. Study wall time was 7697.0 seconds (2.138 hours), with eight single-thread CPU inference workers. No learning updates occurred.

The workload remains 100,000 logical bits and 1,000 cycles/bit; UAV/cloud capacities remain 1e9/1e10 cycles/s. Robot/UAV conducted uplink power remains 0.2/2 W. Native access and LEO capacity caps, link budgets, queues and archived propagation are retained. Additional delays/outages are controlled assumptions, not measured network conditions.

## Four primary arms

| Robots | Arm | Missions | On-time commands | Command p95 (ms) | Makespan (steps) | Waiting (robot steps) | Modeled energy (J) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | Instant baseline | 10/10 | 100.00% | 0.0 | 26.20 | 0.40 | 0.000 |
| 2 | Allocated | 10/10 | 100.00% | 152.0 | 26.20 | 0.40 | 1.042 |
| 2 | All UAV | 10/10 | 100.00% | 202.3 | 26.20 | 0.40 | 0.319 |
| 2 | All cloud | 10/10 | 100.00% | 98.3 | 26.20 | 0.40 | 1.096 |
| 4 | Instant baseline | 10/10 | 100.00% | 0.0 | 33.60 | 0.00 | 0.000 |
| 4 | Allocated | 10/10 | 95.14% | 202.5 | 35.00 | 5.40 | 2.181 |
| 4 | All UAV | 0/10 | 0.00% | 402.5 | — | — | 2.358 |
| 4 | All cloud | 10/10 | 100.00% | 100.0 | 33.60 | 0.00 | 2.245 |
| 6 | Instant baseline | 10/10 | 100.00% | 0.0 | 31.30 | 0.50 | 0.000 |
| 6 | Allocated | 10/10 | 93.97% | 340.3 | 33.20 | 8.20 | 2.396 |
| 6 | All UAV | 0/10 | 0.00% | 603.1 | — | — | 2.526 |
| 6 | All cloud | 10/10 | 100.00% | 109.1 | 31.50 | 0.40 | 2.586 |
| 8 | Instant baseline | 9/10 | 100.00% | 0.0 | 33.56 | 1.33 | 0.000 |
| 8 | Allocated | 9/10 | 94.65% | 349.5 | 35.89 | 13.78 | 4.445 |
| 8 | All UAV | 0/10 | 0.00% | 803.3 | — | — | 2.896 |
| 8 | All cloud | 9/10 | 100.00% | 127.7 | 34.22 | 0.56 | 4.715 |

Command p95 in this table pools observed commands. The figures also report the separately named mean case p95 and its case-bootstrap interval. Full per-population CPU utilization, offloading, source-age, energy and outcome counts are in `primary.csv`.

The same case, `agents-8-case-002`, reached the 250-step cap in the instant, allocated and cloud arms. The nominal experiment therefore provides no observed additional success loss for either cloud-capable arm. This small cohort does not establish population-level equivalence.

All-UAV command p95 is approximately 402/603/803 ms at 4/6/8 robots. The corresponding finite-CPU ideal-network controls also complete no missions and meet no command deadlines. Under the retained processor sharing and workload, this is a compute-service limit at the selected cutoff. Its flat additional-delay response has a zero-compliance floor.

## Paired primary differences

Cloud minus allocated, eight robots; intervals use 2,000 paired case-cluster bootstrap resamples, seed 20261023. Absolute mission-success intervals use Wilson bounds.

| Quantity | Paired cases | Difference | 95% CI |
| --- | --- | --- | --- |
| Mission success (pp) | 10 | 0.000 | [0.000, 0.000] |
| On-time commands (pp) | 10 | 5.351 | [3.236, 7.284] |
| Makespan (steps) | 9 | -1.667 | [-2.556, -0.556] |
| Waiting (robot steps) | 9 | -13.222 | [-17.000, -9.664] |
| Modeled energy (J) | 10 | 0.269 | [0.191, 0.386] |

Cloud serves commands more reliably and reduces waiting/makespan in this cohort, at higher modeled energy. Allocation optimizes its fixed-input weighted-latency/energy snapshot, including transmission and UAV dynamic-compute energy, with no hard command-deadline constraint. It does not establish an end-to-end or mission optimum.

## Controlled communication effects

| Eight-robot condition | Missions | On-time | Command p95 (ms) | Makespan | Waiting |
| --- | --- | --- | --- | --- | --- |
| cloud__nominal | 9/10 | 100.00% | 127.7 | 34.22 | 0.56 |
| cloud__uplink-0.15 | 9/10 | 79.14% | 277.8 | 46.78 | 69.56 |
| cloud__downlink-0.15 | 9/10 | 74.84% | 277.6 | 50.00 | 83.44 |
| cloud__compute-0.15 | 9/10 | 74.84% | 277.6 | 50.00 | 83.44 |
| cloud__interruption-0.25 | 9/10 | 90.32% | 314.8 | 37.67 | 19.44 |
| cloud__interruption-1 | 9/10 | 90.32% | 1063.8 | 37.67 | 19.44 |
| cloud__backhaul-0.15 | 0/10 | 0.00% | 429.0 | — | — |
| allocated__nominal | 9/10 | 94.65% | 349.5 | 35.89 | 13.78 |
| allocated__uplink-0.05 | 9/10 | 100.00% | 166.6 | 33.89 | 0.56 |
| allocated__backhaul-0.15 | 0/10 | 11.84% | 628.9 | — | — |

At eight robots, adding 150 ms downlink delay reduces cloud command timeliness by 25.16 pp (95% CI −30.42 to −18.53) and adds 15.78 makespan steps (9.22 to 23.11). One-second interruptions every ten seconds reduce timeliness by 9.68 pp (−9.93 to −9.34) and add 3.44 makespan steps (2.33 to 4.33). Adding 150 ms to each LEO direction produces zero cloud completions and zero on-time cloud commands.

The 250 ms and one-second interruptions have the same observed cloud mission outcomes here, despite different command-tail latency. Both cross the cutoff at the affected ticks. Commands received within the cutoff share the same one-second control schedule. Mission effects therefore need not increase smoothly with sub-cutoff delay. All 41 conditions and all populations remain in `summary.csv` and `comparisons.csv`.

### Admission timing explains the non-monotone allocated response

The nominal allocated arm has 163 eight-robot jobs admitted while the native backhaul budget reports unavailable service. All 163 use the UAV; 125 are late. All 2,173 jobs admitted with available backhaul are on time. In the cloud-only arm, 161 jobs see unavailable backhaul at admission and all arrive on time after queuing. The additional 50 ms access-uplink hold shifts admission past those unavailable instants: no jobs see unavailable backhaul, all 2,215 commands arrive on time, and mean makespan falls by 2.00 steps (95% CI −2.56 to −1.44).

These results expose sensitivity to admission phase and instantaneous availability. The retained allocator does not optimize waiting for imminent route recovery; the fixed-cloud route continues through the transport queue. `mechanism_audit.json` contains the admission counts from every eight-robot case in the listed conditions. No allocation algorithm was changed.

## Peer staleness and coordination

| Robots | Condition | Missions | On-time | Inference FoV errors/exposures | Stale/fresh action disagreement |
| --- | --- | --- | --- | --- | --- |
| 2 | cloud__nominal | 10/10 | 100.00% | 0/24 | 0.00% |
| 2 | cloud__aoi-3 | 10/10 | 100.00% | 15/34 | 3.97% |
| 4 | cloud__nominal | 10/10 | 100.00% | 0/30 | 1.23% |
| 4 | cloud__aoi-3 | 10/10 | 100.00% | 3/42 | 4.50% |
| 6 | cloud__nominal | 10/10 | 100.00% | 0/42 | 2.14% |
| 6 | cloud__aoi-3 | 10/10 | 100.00% | 25/74 | 8.14% |
| 8 | cloud__nominal | 9/10 | 100.00% | 1/522 | 2.30% |
| 8 | cloud__aoi-3 | 9/10 | 100.00% | 11/515 | 6.88% |

For eight cloud robots, the three-step peer hold increases stale/fresh DQN disagreement from 2.30% to 6.88%, a paired increase of 4.58 pp (95% CI 2.89 to 7.09), while command timeliness remains 100%. The inference FoV error count rises from 1/522 to 11/515. Mean makespan increases by 1.67 steps, with a 95% interval of 0.00 to 3.56; mission completion remains 9/10. The experiment shows changed decisions and ordering, with a modest and uncertain mission-timing effect in this small bank.

Inference physical-reference FoV errors and the original robot-cached FoV metric remain separate fields. Original cached FoV error is zero in these nominal runs. Empty relevant-pair denominators remain unavailable, including eight-robot UAV-only runs. Exploratory vertex-only records remain separate in the raw telemetry. Sparse and repeated exposures produce wide ordering-error intervals.

### Absolute source age includes completed robots

| Eight-robot condition | Active-peer age (s) | Active peer samples | Completed-peer age (s) | Completed peer samples |
| --- | --- | --- | --- | --- |
| cloud__nominal | 0.546 | 10314 | 66.021 | 5205 |
| cloud__ideal_network | 0.537 | 10322 | 66.416 | 5162 |
| cloud__aoi-3 | 3.143 | 10244 | 65.230 | 5317 |

Completed robots retain their last delivered state after reporting stops. The resulting overall nominal cloud peer age of 22.51 seconds is dominated by these retained records; the finite-CPU ideal-network control has a similar overall age. Active-peer age is about 0.55 seconds nominally and 3.14 seconds with the three-step hold. The supplementary `aoi_hold_sensitivity` figure uses imposed hold on the horizontal axis; the original all-peer-age figure and metrics are retained.

## Validation, scope and reproducibility

All 360 regression tests passed, with one existing ITU-R low-elevation warning. All 1,640 archives passed causal replay and frozen-policy action checks. Independent transaction checks cover 819,045 task records and 169,876 inference rounds: 152,595 on-time, 666,430 late, and 20 undelivered commands, plus 666,450 cutoff-WAIT tags. The tags are separate from the exclusive arrival partition. CPU bounds, forced/binary assignments, per-round denominators and aggregate/per-case record identities pass.

The study uses one nominal map and ten paired cases per population. Delays/outages are controlled assumptions. CPU times derive from logical cycles and finite pools; added compute sensitivity is post-CPU inference overhead. The allocator retains native rate/propagation inputs and does not receive injected transit/overhead as an extra optimization input. Energy covers robot access uplink, UAV backhaul uplink and UAV dynamic compute; cloud compute, downlink transmitters, receivers, static platform and propulsion power are unparameterized. Reported energy is the observed modeled lower-bound component.

Reproduction metadata is in `manifest.json` and `cases.json`. Tables, compressed task/round telemetry and vector/600 dpi figures are published in this folder. Original assets, frozen source snapshot and raw event/observation archives remain in the local run directory. `validation.json`, `denominator_audit.json`, `mechanism_audit.json` and `preservation.json` hold the checks. The earlier 1,200-episode study, 216-episode staleness pilot and original checkpoint remain unchanged. `analysis/` contains the supplemental calculations and plotting code; its scripts expect the original run layout. See [README.md](README.md) for file mappings and reproduction inputs.
