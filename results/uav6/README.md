# Parameter-only UAV CPU pilot results

All outputs in this folder are **parameter-only pilot results**, not validated
paper findings. The sole parameter change is UAV CPU capacity from 1 to 6 GHz;
the four conditions are original allocation, 6 GHz allocation, all-UAV at 6 GHz
and all-cloud. The 60 episodes use five paired cases each at 2/4/8 robots.
Sixteen robots are excluded because the frozen model lacks that qualification.

- [report.md](report.md): findings, suitability and limitations.
- [overview.md](overview.md): all four conditions' numerical comparison.
- [summary.csv](summary.csv), [comparisons.csv](comparisons.csv): full metrics and
  paired differences with 95% intervals.
- [handover_details.csv](handover_details.csv), [handover.csv](handover.csv):
  exposure counts, latency, cutoff WAITs, active-peer AoI and prior-fixed-job strata.
- [first_round.csv](first_round.csv), [placement_by_active_count.csv](placement_by_active_count.csv):
  initial placements and active-load/availability strata.
- [episodes.csv](episodes.csv), [rounds.csv](rounds.csv), `tasks.csv.gz`:
  all 60 missions, 2,883 rounds and 8,746 tasks.
- [plots/](plots/): placement, E2E latency, CPU utilization/concurrency and
  deadline compliance in PDF/SVG/PNG.
- [base.toml](base.toml), [pilot.toml](pilot.toml), [manifest.json](manifest.json),
  [cases.json](cases.json): exact configurations, paired inputs, hashes and provenance.
- [validation.json](validation.json), [independent_audit.json](independent_audit.json),
  [baseline_reproduction.json](baseline_reproduction.json), `audits.jsonl.gz`,
  [tests.xml](tests.xml), [focused_tests.xml](focused_tests.xml): validation evidence.
- [smoke_validation.json](smoke_validation.json), [smoke_status.json](smoke_status.json):
  the separate four-condition three-step smoke; not included in the 60-episode cohort.

Gzip files decompress to the exact original CSV/JSONL bytes.
[../files.csv](../files.csv) maps copied artifacts to original paths and SHA-256.
Manifests retain original local paths as provenance. No numerical values were
changed for publication. The immutable source snapshot, checkpoint, timeline and
raw replay archives remain local under `output/uav6_pilot/uav6-20261024-01`.
The existing result bundles remain unchanged.

[../../UAV6_PILOT.md](../../UAV6_PILOT.md) gives exact commands and input identities.
[../../scripts/audit_uav6_pilot.py](../../scripts/audit_uav6_pilot.py) reproduces
independent audits using local transactions/historical archives. The supplemental
[analysis/plot_cpu.py](analysis/plot_cpu.py) expects the original run layout.

Verify this published bundle without external model/orbital files:

```powershell
python results/uav6/analysis/verify_bundle.py
```
