# W1 command-latency distributions

Read-only CDF and log-tail CCDF derived from `results/omega_sensitivity/tasks.csv.gz`. W1 uses latency/energy coefficients 1/1. All curves pool delivered commands from published successful missions. The late/undelivered fractions divide by all task observations in the same successful-mission cohort. Undelivered commands remain right-censored and receive no fabricated latency.

The step-CDF implementation is shared with the existing priority-evaluation plotting code. Every task identity, successful-episode count, command partition, mean, p50, pooled p95, and deadline percentage is checked against the existing publication. No simulations are executed; source result hashes are unchanged.

The CCDF is P(T > t | delivered). Its terminal zero is stored in the numerical distribution table; the logarithmic plot ends at the positive left limit at the maximum observation. The 16/32-robot cohorts remain exploratory scale tests.

| Robots | Policy | Successful missions | Delivered n | p95 (ms) | Late (%) | Undelivered (%) |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Allocator | 5 | 308 | 62.87 | 0.00 | 0.00 |
| 2 | All UAV | 5 | 308 | 62.87 | 0.00 | 0.00 |
| 2 | All cloud | 5 | 308 | 98.46 | 0.00 | 0.00 |
| 4 | Allocator | 5 | 484 | 63.02 | 0.00 | 0.00 |
| 4 | All UAV | 5 | 484 | 81.76 | 0.00 | 0.00 |
| 4 | All cloud | 5 | 484 | 99.86 | 0.00 | 0.00 |
| 8 | Allocator | 4 | 771 | 116.99 | 0.00 | 0.00 |
| 8 | All UAV | 4 | 774 | 147.55 | 0.00 | 0.00 |
| 8 | All cloud | 4 | 779 | 129.51 | 0.00 | 0.00 |
| 16 | Allocator | 5 | 1744 | 149.68 | 1.83 | 0.00 |
| 16 | All UAV | 3 | 1297 | 267.92 | 19.81 | 0.00 |
| 16 | All cloud | 5 | 1723 | 204.87 | 0.35 | 0.00 |
| 32 | Allocator | 2 | 1636 | 242.25 | 4.16 | 0.00 |
| 32 | All UAV | 0 | 0 | — | — | — |
| 32 | All cloud | 3 | 3002 | 341.12 | 22.15 | 0.00 |

Reproduce in a new directory:

```powershell
.\.venv\Scripts\python.exe scripts/plot_omega_latency.py --output results/omega_latency_w1_v2
```
