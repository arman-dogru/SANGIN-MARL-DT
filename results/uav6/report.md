# Parameter-only pilot results

**The intended population-driven UAV-to-cloud transition did not appear.**
The 6 GHz allocator used cloud execution for 47.4%, 44.6% and 45.4% of tasks at
2/4/8 robots. It maintained a modest UAV majority at each tested population.
It did remove the observed command-cutoff failures, with higher modeled energy.
Sixteen robots were not run: this frozen checkpoint has qualification only at
2/4/6/8 robots. No unqualified checkpoint or qualification override was used.

Five paired cases per supported population, four conditions and a 250-step cap
produced 60 simulations in 139.3 seconds using four single-thread CPU workers.
All settings except UAV capacity remain those of the 250 ms baseline. The
separate all-UAV/cloud conditions reuse existing placement controls.

## Placement and command performance

| Robots | 6 GHz UAV/cloud tasks | Cloud share | First-round UAV/cloud mean | E2E mean / p50 / p95 (ms) | On time: 1 GHz → 6 GHz | Missions: both allocators |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 162/146 | 47.4% | 1.4/0.6 | 55.7/51.0/62.9 | 100.00% → 100% | 5/5 |
| 4 | 268/216 | 44.6% | 3.0/1.0 | 53.7/51.1/63.0 | 96.03% → 100% | 5/5 |
| 8 | 747/622 | 45.4% | 5.8/2.2 | 64.8/63.3/92.4 | 95.35% → 100% | 4/5 |

[overview.md](overview.md) gives all four conditions' task counts, observed
latency, mission outcomes, makespan, waiting and energy. `summary.csv` contains
all numerical parameters and metrics; `first_round.csv` retains each case.
The measured first round at four robots averages 3 UAV/1 cloud. At eight it
averages 5.8 UAV/2.2 cloud, rather than the hypothesized 4/4. Neither value was
supplied as an assignment target.

The absence of a rising cloud-share trend also holds after stratifying by
actual active population: while all initial robots remain active and backhaul
is available at admission, cloud shares are 50.0%, 45.7% and 49.5% at 2/4/8
robots. `placement_by_active_count.csv` preserves all other load/availability
strata. Thus the mission-wide result is not explained solely by robots reaching
goals or the long failed eight-robot case.

## CPU capacity, predicted latency and freshness

| Robots | UAV/cloud utilization | Peak concurrent UAV/cloud jobs | UAV jobs while busy (mean) | Solver predicted mean / observed mean (ms) | Active-peer AoI: 1 GHz → 6 GHz (s) |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.72%/0.93% | 2/1 | 1.04 | 54.1/55.7 | 0.535 → 0.533 |
| 4 | 2.69%/1.30% | 4/1 | 1.49 | 49.8/53.7 | 0.542 → 0.736 |
| 8 | 3.15%/1.57% | 8/1 | 1.88 | 57.9/64.8 | 0.602 → 0.636 |

Utilization includes idle time between one-second control rounds. UAV peak load
rises, but its mean occupied concurrency remains 1.04/1.49/1.88 jobs under the
6 GHz allocator. Cloud peak concurrency is one; serial radio arrivals and its
short compute stage do not create overlapping cloud jobs in this cohort.
Increasing a capacity limit does not force that clock rate: the unchanged
energy-aware processor sharing can leave capacity idle.

All-UAV at 6 GHz also meets every observed cutoff, including eight robots
(command p95 144.3 ms). This cohort therefore does not establish strong UAV
capacity pressure requiring progressively more cloud execution. The original
1 GHz allocator is cloud-heavy already (92.9%, 93.3%, 85.3% cloud).

The solver sees sequential admissions and remaining work with prior placements
fixed; it does not solve a fresh simultaneous N-robot placement on every round.
Snapshots contain up to seven previously fixed jobs here. Predicted latency
excludes future admissions and full response/wire queuing. Recorded command-path
radio queue means for the 6 GHz allocator rise from 1.2 to 3.7 to 6.8 ms.
The saved predicted p50/p95 and observed p50/p95 remain separate columns.

Active-peer AoI is 0.533/0.736/0.636 seconds with sample counts 302/1,090/5,676.
It does not uniformly improve with faster CPU service; the four-robot result
increases from 0.542 to 0.736 seconds. Peers marked completed in the fresh
physical reference at inference are excluded from this age measure.

## Handover-associated misses and fixed placements

At eight robots, all 101 baseline admissions with unavailable backhaul go to
UAV; 67 are late, with mean E2E latency 489.1 ms and p95 803.1 ms. The 6 GHz
allocator sends all 99 unavailable-admission jobs to UAV, with
zero misses, mean 92.1 ms and p95 144.8 ms. Every available-admission task in
both allocated arms arrives on time. These observed misses are concentrated in
native handover/service-window admissions, rather than general radio queuing.

The all-cloud control queues those 99 unavailable-admission tasks through the
native transport and still meets every cutoff: mean 122.9 ms, p95 173.7 ms.
Their mean command-path queue/interrupted-service times are 59.5/13.1 ms.
This illustrates why instantaneous unavailability and a queued cloud route can
produce different placement/timing behavior without changing either model.
`handover_details.csv` includes cutoff-WAIT counts and active-peer age by
exposure; `handover.csv` additionally separates prior-fixed-job presence.
Resource reconfiguration windows and actual satellite attachment events are
counted separately. These exposure contrasts are descriptive.

## Mission and energy trade-off

Relative to the 1 GHz allocator, the 6 GHz allocator changes makespan by 0,
−1.0 and −2.25 steps at 2/4/8 robots. The paired 95% intervals are [0, 0],
[−1.6, −0.4] and [−3.0, −1.5]. Waiting falls from 4.0 to 0 steps at four robots
and 16.0 to 0.75 at eight. Observed mission counts remain 5/5, 5/5 and 4/5.
The common failure, `agents-8-case-002`, reaches the cap in all four conditions.

Modeled energy per episode rises from 1.242 to 1.700 J, 2.025 to 3.756 J,
and 5.531 to 7.577 J: increases of 36.8%, 85.5% and 37.0%. Paired energy
increases have positive intervals in this cohort. At eight robots, all-cloud
uses 5.856 J/episode versus the 6 GHz allocator's 7.577 J; its command p95 is
127.6 ms versus 92.4 ms. Energy covers robot uplink, UAV backhaul uplink and
UAV dynamic compute, with other platform costs unparameterized.

**Suitability:** 6 GHz is a useful diagnostic setting for removing the observed
UAV timing floor through eight robots. It does not demonstrate the proposed
robot-count-driven cloud transition, uniformly improve freshness, or improve
mission success in this sample. It should remain an experimental parameter;
16-robot behavior and broader scenario reliability are unestablished.

## Validation and reproduction

All 374 regression tests passed, including 14 pilot tests; one existing ITU-R
low-elevation warning remains. The four-condition, three-step smoke passed.
All 60 mission archives passed causal replay and frozen-model action checks.
Independent checks match 8,746 tasks and 2,883 inference rounds to their atomic
case transactions. All 15 baseline trajectories and task timings reproduce the
historical run. All 15 all-cloud controls reproduce every frame field except
its event-chain checksum, which changes when allocation snapshots record the
unused UAV capacity as 6 GHz; physical execution and task timings match.

The pilot verifies 211 protected source/base-config/historical-result files.
`validation.json`, `independent_audit.json`, `baseline_reproduction.json` and
`tests.xml` retain evidence. The frozen model SHA-256 is
`be085bec93c4b8603629b0e964b10f7b5a99220386e9c968ac17f960759ffc0f`;
timeline SHA-256 is
`c24a66247cbf13e0b08a285e918b4b4e6115fe29d268e862cd5d5bd71efe187f`.
The source base is `cb8ccee82c4df26741db0caefda930235a23e65f` plus the
hash-recorded isolated harness; no production implementation changes occur.

The cohort uses one map and five cases per supported population; intervals are
small-sample diagnostics, not evidence of general equivalence. These are
**parameter-only pilot results, not validated paper findings**. Exact commands,
external inputs and measurement definitions are in
[UAV6_PILOT.md](../../UAV6_PILOT.md). The checkpoint, orbital archive, frozen
source snapshot and raw mission archives remain local.
