# Obvious allocation cases

The approved Python equivalent uses **CVXPY and Clarabel**. The expanded rerun
passes all ten predictable cases, with both the fractional scheduling solution
and exact binary assignment saved. `x=1` means drone and `x=0` means ground cloud
through LEO. No task deadline is imposed.

| Scenario | Expected and observed executable assignment |
| --- | --- |
| 5 s satellite round-trip propagation | `[1,1,1,1]`: all drone |
| Fast cloud, slow drone | `[0,0,0,0]`: all cloud |
| Finite backhaul reduced to 1 kbit/s | `[1,1,1,1]`: all drone |
| Finite cloud CPU reduced to 1 Mcycle/s | `[1,1,1,1]`: all drone |
| Unavailable backhaul | `[1,1,1,1]`: all drone |
| Unavailable drone CPU | `[0,0,0,0]`: all cloud |
| One task, 10 ms propagation | `[0]`: cloud |
| One task, 1 s propagation | `[1]`: drone |
| Equal work, forced drone, priority weights 4:1 | CPU shares `2/3, 1/3` |
| Energy-aware DVFS with abundant drone capacity | Uses 5.70% of CPU; deliberate idling |

The first eight cases isolate latency using energy coefficient zero. The priority
case also uses latency alone. The DVFS case uses energy and latency coefficients
one and a nonzero drone energy coefficient. Fixed radio powers and rates are
retained; no power optimization or deadline penalty creates these outcomes.

For the four-robot high-propagation case, each task has 100,000 input bits and
100 million cycles. With a 1 GHz drone, each receives 250 MHz and takes 0.4 s
of computation plus 0.002 s upload. Thus all-drone weighted latency is 1.608 s.
The solver returns exactly those shares and objective. If all four instead used
the 10 GHz cloud, each task would take 5.052 s including the 5 s round trip.

## Independent checks

The direct reference enumerates every binary assignment and solves the original
`C/f + kappa*C*f^2` CPU allocation. It uses neither the conic variable substitution
nor the production KKT/CPU-share helpers. All ten objective differences are below
7.6e-10. Every relaxed problem is DCP-certified, passes capacity checks and has
objective no greater than its binary reference within tolerance.

Additional regression cases use hand arithmetic rather than another optimizer:

- **Single-task propagation threshold:** drone delay is `0.102 s`; cloud delay
  is `0.022 s + RTT`. Cloud must win below 80 ms and drone above it. Checks at
  0, 79, 81 and 5,000 ms pass for both binary and relaxed results.
- **Identical processors:** four equal tasks and two equal 1 GHz CPUs minimize
  computation cost `0.1*(k^2+(4-k)^2)` at two jobs per CPU. The solver assigns two
  to each processor with half the selected pool per job.
- **Known priority ratio:** equal work and weights 4:1 require square-root CPU
  shares 2:1, giving the hand-calculated weighted latency 0.91 s.

The complete code suite passes **121 tests**, including overlapping compute,
physical radio/energy accounting, atomic coordination caches and saved numerical
failure fixtures. The only warning is the existing ITU-R gaseous-attenuation
approximation warning. The final test run has no failures, errors or skipped
tests; the earlier Windows temp-folder permission issue is avoided by the explicit
workspace-local base directory.

## Reproduce and inspect

```powershell
.\.venv\Scripts\python.exe -m panama_uav allocation-sanity --output output\sanity-new
.\.venv\Scripts\python.exe -m pytest -q --basetemp tmp\pytest-new
```

Use a new output/base directory for each run. The committed
[summary](solver-sanity-20261021/summary.json) includes expected and actual `x`,
relaxed `x`, CPU shares, objective agreement, DCP status and numerical residuals.
Each scenario JSON preserves its input snapshot, full allocations and independent
solution. [solver_test_run.json](solver_test_run.json) records counts, package
versions and source hashes. The [complete simulation report](RESULTS.md) covers
all 19,400 finished mission episodes; manuscript work is deferred.
