> Completed 216-episode controlled stress pilot, not nominal SAGIN measurements. No full stress sweep was run.

# Controlled staleness pilot: findings for review

Injected peer staleness changes the actions actually delivered to robots and can degrade mission outcomes. This effect remains visible with ideal transport, zero command lateness and no cutoff WAITs. The pilot does not show a consistent mission-level mitigation benefit from priority scheduling.

All 216 bounded episodes completed: three 32x32 layouts, 2/4 robots, two rotated cases per cell, 0/1/2/3/4/5-step peer holds, and fresh-transport/matched-equal/current-priority treatments. The episode cap is 60 steps, the control interval is 1 s, and all three cutoffs remain 250 ms with hard solver deadlines disabled. These are controlled stress conditions, not nominal SAGIN delay measurements. The qualified original checkpoint is frozen; the new layouts are outside its qualification scope. No PANAMA, solver, CPU, workload, radio or collision-rule change was made.

Peer state is held at the actual UAV/cloud inference read after accepted network arrival. Its original source/generation times and atomic intention/priority/completion state are preserved. Own sensing is unchanged. The zero-delay causal event chain, actions and arrival times match the original controller exactly in regression tests. The t=0 briefing is retained during warm-up. At positive delay, fresh_transport means ideal delivery plus the intentional peer hold; it does not mean fresh peer information.

## Conflict exposure at zero-delay fresh transport

| Scenario | Robots | Relevant / all directed peer observations | Vertex / edge | Exposed cycles / cycles with movement | Fresh success | Screening |
| --- | ---: | --- | --- | --- | --- | --- |
| intersection | 2 | 16 / 36 | 16 / 0 | 8 / 8 | 100% | below exposure threshold |
| intersection | 4 | 80 / 258 | 76 / 4 | 13 / 13 | 100% | pass |
| narrow_corridor | 2 | 30 / 53 | 8 / 22 | 15 / 15 | 100% | pass |
| narrow_corridor | 4 | 266 / 627 | 204 / 62 | 75 / 75 | 50% | pass |
| shared_bottleneck | 2 | 4 / 32 | 4 / 0 | 2 / 2 | 100% | below exposure threshold |
| shared_bottleneck | 4 | 87 / 291 | 39 / 48 | 19 / 19 | 100% | pass |

The pre-inspection screen requires at least 20 physical-reference relevant directed-pair observations, exposure in both cases and some exposed cycles with movement. All dense cells and the sparse corridor pass; the sparse intersection and bottleneck fall below the count threshold. This screens exposure, not statistical power. Directed pairs and repeated epochs are not independent conflicts. The archived progressing_exposure_cycles field counts any movement, including oscillation, rather than proven progress toward goals. The rotated four-robot corridor already has a capped fresh-baseline failure.

## All treatment and delay outcomes

The following totals pool the 12 fixed cases descriptively. Longer episodes create more task opportunities; raw command counts therefore have different exposure denominators. Formal paired intervals remain within scenario/robot-count cells.

| Delay (steps) | Treatment | Successful cases / 12 | Mean capped duration (s) | Used commands differing from fresh peers | Cutoff WAITs |
| ---: | --- | --- | ---: | --- | ---: |
| 0 | fresh_transport | 11 / 12 | 18.25 | 0 / 513 (0.0%) | 0 |
| 0 | equal | 10 / 12 | 21.67 | 11 / 547 (2.0%) | 24 |
| 0 | priority | 10 / 12 | 22.50 | 19 / 558 (3.4%) | 24 |
| 1 | fresh_transport | 10 / 12 | 24.58 | 118 / 883 (13.4%) | 0 |
| 1 | equal | 10 / 12 | 23.42 | 104 / 797 (13.0%) | 51 |
| 1 | priority | 10 / 12 | 23.67 | 87 / 800 (10.9%) | 50 |
| 2 | fresh_transport | 9 / 12 | 26.83 | 96 / 918 (10.5%) | 0 |
| 2 | equal | 9 / 12 | 27.25 | 102 / 853 (12.0%) | 51 |
| 2 | priority | 9 / 12 | 27.25 | 129 / 856 (15.1%) | 50 |
| 3 | fresh_transport | 11 / 12 | 23.08 | 148 / 698 (21.2%) | 0 |
| 3 | equal | 11 / 12 | 23.33 | 132 / 671 (19.7%) | 36 |
| 3 | priority | 11 / 12 | 23.33 | 115 / 671 (17.1%) | 36 |
| 4 | fresh_transport | 10 / 12 | 26.42 | 124 / 793 (15.6%) | 0 |
| 4 | equal | 10 / 12 | 26.42 | 124 / 765 (16.2%) | 36 |
| 4 | priority | 10 / 12 | 26.42 | 131 / 765 (17.1%) | 36 |
| 5 | fresh_transport | 8 / 12 | 33.58 | 165 / 1029 (16.0%) | 0 |
| 5 | equal | 8 / 12 | 33.83 | 152 / 1003 (15.2%) | 48 |
| 5 | priority | 8 / 12 | 33.83 | 145 / 1006 (14.4%) | 44 |

At zero ideal delay, all 513 used commands match the same-self fresh-peer oracle. At five steps, 165/1,029 differ (16.0%) with zero cutoff WAITs. Mission success falls from 11/12 to 8/12 and mean capped duration rises from 18.25 s to 33.58 s. The trajectory response is not monotonic: delays of 1/2/3/4/5 steps produce ideal-transport successes 10/9/11/10/8 out of 12. Stale intentions can change symmetry breaking, routes, waiting and subsequent state exposure; increasing lag does not guarantee a monotonically worse individual rollout.

In the four-robot bottleneck, ideal-transport mean duration rises from 18 to 31 s at five steps, with 35.5% used-command disagreement and both missions still completing. The sparse opposing corridor falls from 2/2 successes to 0/2 at five steps. The four-robot intersection falls from 2/2 to 1/2. These cases show coordination sensitivity, but do not establish nominal network degradation or broad checkpoint generalization.

## Paired uncertainty and priority mitigation

Intervals below are percentile paired-case bootstrap intervals (2,000 draws, seed 42), with only two cases per cell. They describe these two rotations; they are unstable, pointwise, uncorrected for multiple comparisons, and are not reliable population-level confidence statements. Rotations and robot counts also share designed geometry. A zero-width interval does not establish equivalence. All effects, including negative/null effects, are retained in comparisons.csv.

| Four-robot scenario | Five-step fresh-transport duration change (s), 95% CI | Priority-mitigation duration interaction (s), 95% CI |
| --- | --- | --- |
| intersection | +25.0 [+3.0, +47.0] | -1.0 [-2.0, +0.0] |
| narrow_corridor | +4.5 [+0.0, +9.0] | +0.0 [+0.0, +0.0] |
| shared_bottleneck | +13.0 [+12.0, +14.0] | -4.0 [-8.0, +0.0] |

The mitigation interaction is [priority(d)-priority(0)] minus [equal(d)-equal(0)]. At five steps all three dense-scenario mission-success interactions are zero; duration intervals include zero. Some used-command disagreement interactions favor priority, but this does not translate into a demonstrated mission benefit. Across the 12 fixed cases, priority and equal have identical success counts at every delay; priority mean duration is higher at zero and one step and identical at steps 2-5. Priority also has fewer cutoff WAITs in some conditions. It is not a general remedy for delayed observation semantics.

Inference-view FoV ordering errors rise above zero under controlled holds, while raw physical-reference relevance also changes. Both the cached-view and physical-reference rates are retained to avoid hiding missed conflicts. Original robot-cache metrics are unchanged; inference metrics have separate prefixes. Vertex-only results remain exploratory. The same-self shadow compares only peer replacement at the same compute instant; the existing cutoff oracle additionally includes command lateness and other sensing-age differences. Shadow actions are diagnostics, not independently rolled-out fresh missions.

Across the pilot, 486 commands are late, 0 are undelivered, and 0 are censored. The active-decision denominators partition into used commands and cutoff WAITs without double counting. Changes in trajectories can change later traffic, allocation inputs and task counts; allocation and radio kernels themselves remain unchanged.

## Validation and delivery

The full suite passes 256 tests with one existing low-elevation ITU-R approximation warning. All 216 archives pass model/event replay: 14,612 actual actions, 29,224 paired peer-shadow actions, and 14,612 cutoff-oracle actions verified. Tests cover 0-5-step selection, arrival boundaries, rollback, source provenance, inference-channel isolation, exact zero equivalence, diagnostic checksum tampering, case reproducibility, paired contrasts and the full-sweep gate. Source/model/timeline/configuration identities and complete archive hashes are retained.

The prior 1,200-episode directory retains all 1,308 files with identical paths, byte sizes and modification timestamps; selected manifest/status/validation hashes also match the pre-pilot snapshot. See preservation.json. The complete original run remains local; selected results are published here. No full stress sweep has been launched. Before a full study, review the sparse low-exposure controls, fresh-baseline corridor stall, 60-step censoring/caps, small designed case bank and out-of-distribution layout assumptions.
