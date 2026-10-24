# Parameter-only pilot results

5 paired cases per supported population; 60 simulations. The only experimental parameter change is UAV capacity from 1 to 6 GHz. All other configured parameters remain unchanged.

16 robots were excluded: the frozen checkpoint is qualified for 2/4/6/8 robots, with no 16-robot qualification. No unqualified model was substituted.

| Robots | Condition | UAV/cloud tasks | Cloud % | First-round UAV/cloud | E2E mean / p50 / p95 (ms) | On time % | Missions | Makespan | Waiting | Energy/episode J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | allocated-1ghz | 22/286 | 92.86 | 0.8/1.2 | 63.93/60.02/152 | 100 | 5/5 | 31.4 | 0.8 | 1.242 |
| 2 | allocated-6ghz | 162/146 | 47.4 | 1.4/0.6 | 55.73/50.99/62.87 | 100 | 5/5 | 31.4 | 0.8 | 1.7 |
| 2 | uav-6ghz | 308/0 | 0 | 2/0 | 50.77/39.14/62.87 | 100 | 5/5 | 31.4 | 0.8 | 3.261 |
| 2 | cloud-6ghz | 0/308 | 100 | 0/2 | 58.68/60.02/98.46 | 100 | 5/5 | 31.4 | 0.8 | 1.309 |
| 4 | allocated-1ghz | 34/470 | 93.25 | 1.6/2.4 | 77.7/62.04/202.5 | 96.03 | 5/5 | 34.2 | 4 | 2.025 |
| 4 | allocated-6ghz | 268/216 | 44.63 | 3/1 | 53.74/51.06/63.02 | 100 | 5/5 | 33.2 | 0 | 3.756 |
| 4 | uav-6ghz | 484/0 | 0 | 4/0 | 62.86/61.95/81.76 | 100 | 5/5 | 33.2 | 0 | 3.727 |
| 4 | cloud-6ghz | 0/484 | 100 | 0/4 | 65.92/61.99/99.86 | 100 | 5/5 | 33.2 | 0 | 2.058 |
| 8 | allocated-1ghz | 212/1228 | 85.28 | 3.8/4.2 | 102.5/63.94/202.8 | 95.35 | 4/5 | 38.5 | 16 | 5.531 |
| 8 | allocated-6ghz | 747/622 | 45.43 | 5.8/2.2 | 64.83/63.32/92.39 | 100 | 4/5 | 36.25 | 0.75 | 7.577 |
| 8 | uav-6ghz | 1372/0 | 0 | 8/0 | 90.14/94.23/144.3 | 100 | 4/5 | 35.5 | 0.5 | 8.224 |
| 8 | cloud-6ghz | 0/1377 | 100 | 0/8 | 77.14/63.91/127.6 | 100 | 4/5 | 36.25 | 0.75 | 5.856 |

## Placement evidence

Measured cloud fractions under the 6 GHz allocator: 2 robots: 47.40%, 4 robots: 44.63%, 8 robots: 45.43%. Cloud share does not increase strictly across all tested populations. The 16-robot cloud-dominance hypothesis remains untested. First-round counts are reported separately from mission-wide task percentages; admissions are sequential and existing assignments remain fixed.

## Timing, contention and handover diagnostics

`summary.csv` separates predicted solver-snapshot latency from observed sensing-to-robot command latency, including means and p50/p95. The snapshot does not include future arrivals, changing CPU shares, complete wire serialization or all response queuing. `handover.csv` compares tasks sensed within native handover/service windows, outside those windows, admitted with unavailable/available backhaul, and with/without prior fixed jobs. These are exposure associations; no handover model was disabled.

CPU utilization integrates actual allocated rates over simulated time; concurrent-job means and maxima come from existing compute-allocation events. All-UAV/cloud controls reuse the existing forced binary route harness. The cloud route can queue through temporary backhaul unavailability. Active-peer AoI excludes peers marked completed in the fresh physical reference at inference; its sample denominator is saved.

Energy includes robot uplink, UAV backhaul uplink and UAV dynamic compute. Cloud electricity, downlink power, receivers, static power and propulsion remain unmodeled. Increasing the UAV capacity does not force it to run at the limit: the existing energy-aware processor-sharing rule can leave capacity idle.

## Provenance and limitations

Frozen checkpoint SHA-256: `be085bec93c4b8603629b0e964b10f7b5a99220386e9c968ac17f960759ffc0f`. Archived timeline SHA-256: `c24a66247cbf13e0b08a285e918b4b4e6115fe29d268e862cd5d5bd71efe187f`. Source base: `cb8ccee82c4df26741db0caefda930235a23e65f` plus the hash-recorded isolated pilot harness. `manifest.json` stores complete conditions, case identities, config/source/input hashes and package versions. No policy updates occur.

This small, single-map diagnostic is parameter-only pilot evidence, not validated paper findings. Existing solver, production source, configurations and historical result bundles are hash-checked without edits. All episodes require causal replay and frozen-model action verification. `comparisons.csv` contains case-paired 95% bootstrap intervals against the 1 GHz allocator. Sixteen-robot suitability cannot be established with this checkpoint.

Exact preparation/launch commands and external-input requirements are documented in [UAV6_PILOT.md](../../UAV6_PILOT.md) in the repository. Original per-case archives and the frozen source/input snapshot remain local.
