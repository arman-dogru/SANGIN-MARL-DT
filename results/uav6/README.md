# Parameter-only UAV CPU experiment set

The combined set contains **100 simulations**: five paired cases at each of
2/4/8/16/32 robots, four execution conditions and a 250-step episode cap. UAV
capacity changes from 1 to 6 GHz; the frozen policy, cloud capacity, workload,
radio model, priorities and physical controllers retain their existing settings.
These are parameter-only pilot results on one map.

| Condition | 16-robot mission success | 32-robot mission success |
| --- | ---: | ---: |
| Allocator, UAV 1 GHz | 5/5 | 2/5 |
| Allocator, UAV 6 GHz | 5/5 | 2/5 |
| All-UAV, 6 GHz | 3/5 | 0/5 |
| All-cloud | 5/5 | 3/5 |

At 32 robots, the 6 GHz allocator raises on-time command delivery from 75.93% to
95.24%; mission success remains 2/5. Its first-round placements average 12.6 UAV
and 19.4 cloud tasks. At 16 robots those averages are 7.2 and 8.8. Mission-wide
cloud shares are 47.08% and 53.13% at 16 and 32 robots.

- [Combined report](agents16-32/report.md): all populations, findings and parameters.
- [Combined summary](agents16-32/combined_summary.csv): placement, observed and
  predicted latency, deadline compliance, CPU utilization/concurrency, energy,
  mission outcomes and active-peer AoI.
- [Paired differences](agents16-32/combined_pairwise.csv): treatment comparisons
  with 95% case-bootstrap intervals.
- [Paired mission outcomes](agents16-32/case_outcomes.csv) and
  [coordination metrics](agents16-32/combined_coordination.csv): individual
  outcomes and ordering/action-disagreement exposure denominators.
- [Combined diagnostic figure](agents16-32/figures/combined_diagnostics.png) and
  [first-round placement](agents16-32/figures/first_round_placement.png), also
  supplied as PDF and SVG.
- [Added cohort](agents16-32/): 40 simulations, 64,342 tasks and 4,050 rounds,
  with compressed task records, handover strata, paired inputs and provenance.
- [Instantaneous MAPF reference](agents16-32/mapf_reference_serial.json): the
  ten added cases evaluated separately with fresh information. Mission success
  is 5/5 at 16 and 2/5 at 32 robots; 6,682 serial/batched action comparisons agree.
- [Accounting audit](agents16-32/accounting_audit.json) and
  [bundle validation](agents16-32/bundle_validation.json): all 40 added replay
  audits pass, record partitions match and 211 existing source/history files
  retain their hashes.

The original 60-episode snapshot remains in this directory, including its
[report](report.md), [summary](summary.csv), [cases](cases.json), raw metric
tables and [plots](plots/). Its numerical records remain unchanged. The added
cohort and combined tables are stored under `agents16-32/`.

Verify both bundles without model or orbital assets:

```powershell
python results/uav6/analysis/verify_bundle.py
python results/uav6/agents16-32/analysis/verify_bundle.py
```

[../files.csv](../files.csv) records source and published hashes. Gzip files
decompress to the original CSV/JSONL bytes. Run manifests retain original local
paths. Checkpoints, orbital assets and large replay archives stay local.
[UAV6_PILOT.md](../../UAV6_PILOT.md) describes the configuration and commands;
the extension's [analysis directory](agents16-32/analysis/) contains the exact
driver, postprocessing and reference-replay sources.
