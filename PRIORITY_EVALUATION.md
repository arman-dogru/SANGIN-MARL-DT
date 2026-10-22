# Accepted 250 ms evaluation protocol

The user explicitly approved accepting the remote-DT implementation and opening
the full-run gate for this evaluation on 2026-10-21. Only
`configs/paper_remote_dt_250ms.toml` records that approval. The default 100 ms
configuration remains gated. This is user acceptance, not a claim of author
approval of the exploratory vertex-only metric.

The evaluation uses all six existing nominal treatments, 50 shared cases per
2/4/6/8-robot population, a 250-step cap, the qualified frozen checkpoint and the
archived orbital timeline. All three cutoffs are 250 ms; the control interval is
1 second and hard solver deadlines remain disabled. Workload, CPU capacities,
PANAMA, binary allocation, radio scheduling and physical movement are unchanged.

```powershell
.venv\Scripts\python.exe -m panama_uav --config configs/paper_remote_dt_250ms.toml resource-study --run-id full-250ms-priority-new-id --model artifacts/inputs/final_model.pt --cases 50 --agents 2 4 6 8 --max-steps 250 --workers 2
.venv\Scripts\python.exe -m panama_uav priority-report --path output/resources/full-250ms-priority-new-id --model artifacts/inputs/final_model.pt --workers 2 --sensitivity-path output/deadlines/deadline-sensitivity-pilot-20261021-01
```

`resource-study` retains its existing acceptance gate and qualified-model checks.
Do not use `--diagnostic` to run around either check. Existing per-case archives
are audited before saving; `priority-report` independently replays event chains,
rebuilds actual model actions, verifies cutoff oracle actions/coverage, checks
case/asset identity and recomputes deadline and CPU-utilization accounting.
All results stay under ignored `output/`. The sensitivity directory is verified
as compatible evidence, referenced separately, and never pooled into inference
or rerun. Large archives are processed with bounded worker concurrency.

## Primary comparison and inference

The primary contrast is **current priority minus matched equal weights**, within
each robot population. Equal weights use the same active-team mean reference
weight, rather than changing the overall scale of the energy/latency objective.
All six treatments and all prespecified metric contrasts are retained, including
null, unsupported, negative and mixed outcomes.

The inference unit is the **paired case**, not individual tasks or robots.
Percentile 95% bootstrap intervals use 2,000 draws with seed 42. Mean case
differences preserve matched starts/goals, seeds and orbital offsets. Pooled rate
differences resample whole cases and aggregate numerator/denominator counts;
zero-exposure cases remain in sampling. Unsupported rates and zero-denominator
replicates are explicit. Successful-makespan contrasts require success in both
runs and report excluded pairs. Success and capped duration retain failures.
Case p95 differences concern mean paired case p95s; exact pooled task quantiles
and delivered-command CDFs are reported separately. Censored tasks are counted,
not imputed into latency distributions.

Intervals are pointwise and have no multiple-comparison adjustment. Fifty cases,
one fixed map, one frozen checkpoint and one orbital archive do not establish
performance on other environments or hardware. A cutoff-oracle action comparison
evaluates the physical state on the actual trajectory, not a counterfactual
oracle mission rollout.

## Priority-level service

Stratifying by the policy's applied resource weights would make equal-weight
robots indistinguishable. Instead reconstruct the **pre-policy PANAMA reference
weights** at each admission, using the existing inverse-distance mapping to
`[1,4]`, then use fixed bins:

| Reference weight | Service stratum |
| --- | --- |
| `[1,2)` | Low PANAMA priority |
| `[2,3)` | Middle PANAMA priority |
| `[3,4]` | High PANAMA priority |
| No allocation available | Unknown; retained separately |

The primary source is communicated DT priorities available to infrastructure.
The separately logged physical oracle priorities provide a second evaluation
basis. The reference mapping is applied to all treatments; it never changes an
allocation or actual task weight. Raw priorities, reference/applied weights,
active-team size and both stratum labels remain in task CSVs.

Preferential service compares the **high-minus-low service gap under priority**
with that under matched equal allocation. A negative difference in latency gaps
or positive difference in compliance gaps supports greater preference for high
priorities. These service contrasts are separate from mission-level effects.
Different trajectories create different task strata; lone active robots receive
reference weight 4, and resource contention varies with active-team size. Thus
these are descriptive service comparisons, not causal effects on identical tasks.

The existing numeric PANAMA priority includes a robot-ID tie-break. Very small
priority differences between equal-distance robots can therefore span the
reference weight range. The analysis preserves this implemented mapping and
retains raw priorities/weights; it does not reinterpret the bins as different
physical path distances or change the allocator.

## Accounting and outputs

Processor sharing starts jobs immediately. CPU queueing is measured as time with
zero assigned service rate; it must not be confused with the additional elapsed
service caused by sharing or DVFS. Both zero-rate waiting and elapsed time beyond
the full-capacity floor `C/F` are retained. Utilization integrates allocated CPU
fractions over each full observation horizon. Unfinished CPU elapsed times are
censored; the mean completed-job CPU elapsed time is labelled separately.

Existing FoV and vertex metrics remain intact. A derived supported ratio is null
when its relevant-pair denominator is zero. Potential vertex conflicts, missed
forecasts and physically attempted vertex conflicts are separate observations.
The vertex-only metric remains exploratory. Modeled energy retains the existing
limits: no propulsion, static electronics, receiver or cloud-electricity model.

`priority_analysis/` contains:

- `episodes.csv`: original metrics plus audited CPU and deadline accounting.
- `tasks_by_priority.csv`: timing, routes, priorities, reference/applied weights,
  service strata and censor flags for each logical task.
- `priority_case_metrics.csv`: service counts and latency summaries by case and
  stratum for both communicated and physical priority sources.
- `priority_summary.csv`: pooled service-level rates, exact latency quantiles and
  explicit task/CPU censoring denominators. `execution_location_summary.csv`
  reports the same timing quantities by UAV/cloud site with CPU utilization and
  resident-job means. `overall_summary.csv` adds all-treatment mission/energy means.
- `paired_treatment_differences.csv`: all six treatment contrasts against matched
  equal, with estimators, paired counts, exclusions and confidence intervals.
- `preferential_service_*.csv`: within-policy service gaps and paired differences
  between those gaps.
- `latency_distributions.csv` and CDF plots: pooled latency distributions overall,
  by service priority and by execution location, with censoring denominators.
- `replay_audits.csv`, `validation.json`, `methodology.json`: identities, audit
  totals, source commits, statistical definitions and bootstrap settings.
- `RESEARCH_REPORT.md`: the main mission contrasts, preferential-service findings
  and limitations; all remaining metrics remain available in CSVs.
