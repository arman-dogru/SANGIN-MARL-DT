# Latency–energy sensitivity study

This parameter-only study varies the existing allocator's latency and energy
coefficients. PANAMA weights, priorities, physical movement, task admission,
binary placement, CPU sharing and transport remain unchanged. The reference is
the completed 6 GHz UAV pilot and its 16/32-robot extension.

| Configuration | Latency coefficient | Energy coefficient |
|---|---:|---:|
| W1 | 1 | 1 |
| W2 | 1 | 2 |
| W3 | 1 | 4 |
| W4 | 2 | 1 |
| W5 | 4 | 1 |

Each configuration runs Allocated, All UAV and All Cloud on the same five cases
at 2/4/8/16/32 robots: 375 episodes, each capped at 250 steps. Fixed routes are
executed at every weight because the coefficients also enter CPU allocation.
The 16/32 cohorts are exploratory scale tests; checkpoint qualification metadata
is preserved.

## Inputs and execution

The five generated TOMLs differ only in the two objective coefficients. Each
retains the original 6 GHz UAV, 10 GHz cloud, 100,000-bit tasks, 1,000 cycles/bit,
UAV kappa of 1e-28, priorities from 1 to 4, 250 ms cutoff/deadline and one-second
control interval. Hard optimizer deadlines stay disabled.

The harness copies the original frozen production package, model, orbital
timeline and maps into a new run directory. It copies the paired case identities
without regenerating starts, goals, seeds or orbital offsets. Source, input and
historical-result hashes are verified before and after execution.

```powershell
.\.venv\Scripts\python.exe scripts/run_omega_sensitivity.py `
  --root output/omega_sensitivity/<new-run-id> --prepare --launch
```

The first full-length W1 case must reproduce before the remaining W1 cases run.
All 75 W1 episodes must then reproduce before W2–W5 begin. Checks cover physical
frames and event chains, binary allocations, CPU service, task/command timing and
mission metrics. Real host timings are excluded from exact equality. Fractional
diagnostics require optimal/DCP status, the production feasibility limit of
1e-6, and objective agreement within twice the recorded conic tolerance times
the larger of one and the objective magnitude. Fractional minimizers need not
be identical at finite solver tolerance. No diagnostic certificate can excuse a
changed executed assignment or virtual timestamp.

Four single-thread workers run by default, with at most two 16/32-robot cases at
once. The fixed wall cap is 24 hours. Atomic case transactions support resuming a
confirmed stopped run with `--launch --resume`; completed cases cannot be
overwritten. Errors retain their traceback and any solver snapshot and block
continuation pending diagnosis. No failed simulation is silently dropped.

The first study launch used a 1e-4 distance test on unused fractional minimizers.
A baseline case differed by 0.000137 in fractional x while its objective differed
by only 1.94e-10; all executed records were exact. The corrected audit compares
objective accuracy and feasibility using the unchanged solver's own tolerances.
The initial audit failure and validation revision are retained. The recovery
controller changes only this audit and resumes completed case transactions
without rerunning them or changing the fixed study budget.

## Records and audits

Every episode runs the existing causal replay and frozen-policy action audit.
Full trace dictionaries are archived locally with gzip and pickle protocol 5;
this changes storage serialization only. Checkpoints and full traces are omitted
from distributable result bundles. Only verified, locally generated pickle
archives should be opened.

The local run's `episodes.csv` records every mission; its `tasks.csv.gz` records
every generated task. The repository publication selects completed missions
before aggregating any telemetry. Published episode, task, allocation, first-round,
active-count and handover tables all use that same selection. `summary.csv`
records included and excluded episode counts for each weight/route/population.
`comparisons.csv` uses the intersection of completed case identities in both
conditions, with 2,000-resample percentile-bootstrap 95% intervals. Cohorts of
one paired case give degenerate intervals and cannot establish precise effects.

Observed E2E latency uses sensing-to-command-arrival time and is separate from
the optimizer's predicted snapshot latency. On-time, late and undelivered task
counts are exclusive and complete; censored and pending counts are also saved.
CPU utilization averages over the full mission duration, including idle periods.
Energy uses the existing mission tally and component definitions, with generated
tasks in included episodes as the per-task denominator. Empty completed cohorts
have undefined outcome metrics, rather than zero energy, latency or miss rates.
Full-cohort accounting and unsuccessful-episode records remain local.
Original source audit, read-back, status and frozen-harness evidence remains in
the repository for the complete 375-episode execution. Publication metadata and
validation identify the completed-mission subset used in all outcome tables.

Matched-snapshot controls rerun the unchanged binary solver on saved W1
first-round admissions. Only objective weights change; radio inputs, priorities
and fixed previous assignments remain constant. These offline solves help
separate direct objective effects from mission trajectory and admission effects.

## Outputs

The local run retains the complete report, configurations, manifests, CSVs and
all figures. The repository contains eight completed-mission figures in six
groups, each exported as PNG, PDF and SVG. Mission-success plots remain local
and are excluded from both published ZIPs. `charts.zip` contains the published
figures; `results.zip` contains the tables, configurations and report.

```powershell
.\.venv\Scripts\python.exe scripts/run_omega_sensitivity.py `
  --root output/omega_sensitivity/<run-id> --report
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_omega_sensitivity.py tests/test_uav6_pilot.py `
  --basetemp tmp/pytest-omega
```

Results are added to `results/omega_sensitivity/` after execution and validation.
Previous configurations, pilot results, baselines and manuscript files remain
unchanged.

```powershell
.\.venv\Scripts\python.exe scripts/analyse_omega_sensitivity.py `
  output/omega_sensitivity/<run-id>
.\.venv\Scripts\python.exe scripts/finalize_omega_report.py `
  output/omega_sensitivity/<run-id>
.\.venv\Scripts\python.exe scripts/audit_omega_sensitivity.py `
  output/omega_sensitivity/<run-id> --read-back
.\.venv\Scripts\python.exe scripts/publish_omega_completed.py `
  output/omega_sensitivity/<run-id> <new-publication-directory>
```

The standalone bundle verifier checks file hashes, case pairing, task placement,
deadline partitions and ZIP integrity without loading the checkpoint. Result
files retain their recorded bytes through Git using `.gitattributes`.

Read-back evidence is retained in `readback_audit.json`, tied to the manifest and
archive inventory. A later accounting-only refresh preserves that evidence.
Exports require a new directory when an earlier snapshot already exists.

The source study contains 375 episodes and 314,007 generated tasks. All 75 W1
execution reproductions pass, all 375 episode audits pass, and 30 complete trace
archives pass read-back checks. The primary workflow takes 7.37 hours. The
published [numerical report](results/omega_sensitivity/report.md) and CSVs use
completed missions only. `publication.json` records the source manifest, selected
episode identities and reporting-source hashes. The original all-mission
publication is preserved locally under
`output/omega_sensitivity/omega-20261025-01/published_all_missions_01`.
