# Latency–energy weight sensitivity

Parameter-only pilot results for completed missions. The production allocator, radio, workload, CPU capacities and frozen PANAMA policy are unchanged. The 16/32-robot cohorts remain exploratory scale tests.

Each cell reports its included episode count. Latency and deadline rates pool tasks from those episodes; energy and waiting are episode means. CPU utilization uses the complete observed duration, including idle time. Missing cells have no completed mission and are undefined. Paired differences use only case identities completed in both conditions; their 95% bootstrap intervals are conditional on that cohort.

## Allocated policy

| Robots | Weight | Included n | UAV (%) | Cloud (%) | E2E p95 (ms) | Predicted mean (ms) | Energy (J/episode) | Deadline misses (%) |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | W1 | 5 | 52.60 | 47.40 | 62.87 | 54.10 | 1.700 | 0.00 |
| 2 | W2 | 5 | 52.60 | 47.40 | 78.07 | 61.90 | 1.313 | 0.00 |
| 2 | W3 | 5 | 52.60 | 47.40 | 97.22 | 71.72 | 1.069 | 0.00 |
| 2 | W4 | 5 | 53.92 | 46.08 | 52.37 | 39.91 | 4.338 | 0.00 |
| 2 | W5 | 5 | 53.92 | 46.08 | 52.37 | 36.81 | 6.397 | 0.00 |
| 4 | W1 | 5 | 55.37 | 44.63 | 63.02 | 49.82 | 3.756 | 0.00 |
| 4 | W2 | 5 | 52.07 | 47.93 | 78.22 | 55.65 | 2.825 | 0.00 |
| 4 | W3 | 5 | 53.10 | 46.90 | 97.37 | 67.98 | 1.919 | 0.00 |
| 4 | W4 | 5 | 65.08 | 34.92 | 67.39 | 42.58 | 5.204 | 0.00 |
| 4 | W5 | 5 | 53.51 | 46.49 | 62.29 | 40.55 | 7.315 | 0.00 |
| 8 | W1 | 4 | 55.64 | 44.36 | 116.99 | 59.52 | 5.317 | 0.00 |
| 8 | W2 | 4 | 57.33 | 42.67 | 117.85 | 63.11 | 4.489 | 0.00 |
| 8 | W3 | 4 | 62.60 | 37.40 | 126.48 | 71.74 | 3.582 | 0.00 |
| 8 | W4 | 4 | 54.02 | 45.98 | 116.07 | 54.71 | 7.183 | 0.00 |
| 8 | W5 | 4 | 52.20 | 47.80 | 116.17 | 53.30 | 8.533 | 0.00 |
| 16 | W1 | 5 | 52.92 | 47.08 | 149.68 | 75.44 | 7.736 | 1.83 |
| 16 | W2 | 5 | 53.44 | 46.56 | 155.88 | 78.35 | 6.888 | 1.82 |
| 16 | W3 | 5 | 52.69 | 47.31 | 172.31 | 83.96 | 5.855 | 1.89 |
| 16 | W4 | 4 | 50.78 | 49.22 | 150.71 | 73.81 | 9.713 | 2.20 |
| 16 | W5 | 4 | 50.21 | 49.79 | 159.62 | 72.81 | 10.717 | 2.21 |
| 32 | W1 | 2 | 45.90 | 54.10 | 242.25 | 100.94 | 14.850 | 4.16 |
| 32 | W2 | 2 | 45.33 | 54.67 | 251.09 | 101.55 | 11.925 | 5.45 |
| 32 | W3 | 1 | 48.82 | 51.18 | 307.96 | 104.34 | 11.999 | 6.96 |
| 32 | W4 | 1 | 48.27 | 51.73 | 303.77 | 96.28 | 16.824 | 6.19 |
| 32 | W5 | 3 | 46.35 | 53.65 | 254.76 | 98.36 | 17.801 | 5.71 |

## Weight effects

- **2 robots:** W2: energy -22.8%, pooled p95 +15.20 ms; W3: energy -37.1%, pooled p95 +34.35 ms; W4: energy +155.3%, pooled p95 -10.50 ms; W5: energy +276.4%, pooled p95 -10.50 ms.
- **4 robots:** W2: energy -24.8%, pooled p95 +15.20 ms; W3: energy -48.9%, pooled p95 +34.35 ms; W4: energy +38.5%, pooled p95 +4.37 ms; W5: energy +94.7%, pooled p95 -0.73 ms.
- **8 robots:** W2: energy -15.6%, pooled p95 +0.86 ms; W3: energy -32.6%, pooled p95 +9.49 ms; W4: energy +35.1%, pooled p95 -0.93 ms; W5: energy +60.5%, pooled p95 -0.82 ms.
- **16 robots:** W2: energy -11.0%, pooled p95 +6.20 ms; W3: energy -24.3%, pooled p95 +22.63 ms; W4: energy +25.5%, pooled p95 +1.03 ms; W5: energy +38.5%, pooled p95 +9.95 ms.
- **32 robots:** W2: energy -19.7%, pooled p95 +8.83 ms; W3: energy -19.2%, pooled p95 +65.71 ms; W4: energy +13.3%, pooled p95 +61.52 ms; W5: energy +19.9%, pooled p95 +12.51 ms.

These point estimates use each cell’s completed cohort. Differences in case composition can change the ranking; use the paired comparisons for matched effects.

## Paired W2 minus W1

| Robots | Common completed n | Energy difference (J), 95% CI | Mean per-case p95 difference (ms), 95% CI |
|---:|---:|---:|---:|
| 2 | 5 | -0.387 [-0.416, -0.357] | 15.200 [15.200, 15.200] |
| 4 | 5 | -0.932 [-1.201, -0.712] | 12.371 [8.887, 15.200] |
| 8 | 4 | -0.829 [-1.211, -0.521] | 2.203 [1.096, 3.959] |
| 16 | 5 | -0.848 [-1.092, -0.638] | 0.667 [-6.820, 7.584] |
| 32 | 1 | -1.775 [-1.775, -1.775] | -2.370 [-2.370, -2.370] |

A one-pair interval is a degenerate resampling result, not evidence of precise population effects. The pooled task p95 and mean paired per-case p95 are different estimands.

## Interpretation

- **2 robots:** cloud-majority settings none; lowest mean energy W3; lowest pooled p95 W4; lowest deadline-miss rate W1/W2/W3/W4/W5.
- **4 robots:** cloud-majority settings none; lowest mean energy W3; lowest pooled p95 W5; lowest deadline-miss rate W1/W2/W3/W4/W5.
- **8 robots:** cloud-majority settings none; lowest mean energy W3; lowest pooled p95 W4; lowest deadline-miss rate W1/W2/W3/W4/W5.
- **16 robots:** cloud-majority settings none; lowest mean energy W3; lowest pooled p95 W1; lowest deadline-miss rate W2.
- **32 robots:** cloud-majority settings W1, W2, W3, W4, W5; lowest mean energy W2; lowest pooled p95 W1; lowest deadline-miss rate W1.

Increasing the latency coefficient does not guarantee a lower realized tail latency. The objective optimizes priority-weighted snapshot latency, while measured E2E latency also includes subsequent transport, contention and fixed earlier assignments. First-round placement, active-count and handover tables retain those exposures. The conditional results do not estimate the effect on mission success.

## Reproduction and provenance

- Frozen model SHA256: `be085bec93c4b8603629b0e964b10f7b5a99220386e9c968ac17f960759ffc0f`; policy updates during this study: 0.
- Timeline SHA256: `c24a66247cbf13e0b08a285e918b4b4e6115fe29d268e862cd5d5bd71efe187f`; map SHA256: `4240fddfa77d88b72ce779e02acf46a5ff056a3b04af5a4e35f7bc86cdfba3ec`.
- Configurations differ only in latency/energy coefficients. Hard solver deadlines remain disabled.
- `cases.json` preserves starts/goals, seeds and orbital offsets. `publication.json` identifies the selected episode packets.
- The full 375-episode records, prior all-mission summaries and mission-success figures remain local in the source run.
- Rebuild without simulation: `python scripts/publish_omega_completed.py <source-run> <new-destination>`.
- Verify: `python verify_bundle.py`. `charts.zip` contains the same eight published figures in PNG/PDF/SVG.
