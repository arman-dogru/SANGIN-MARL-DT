# 16/32-robot population extension

This folder adds 40 paired parameter-only pilot simulations to the original
60-episode UAV CPU experiment. Five cases per population run four conditions:
the 1 GHz allocator, the 6 GHz allocator, all-UAV at 6 GHz, and all-cloud.
The cap is 250 control steps with 250 ms command cutoffs and a one-second control
interval. No policy updates occur.

[`report.md`](report.md) presents the combined 100-run results. Files prefixed
`combined_` include 2/4/8/16/32 robots; unprefixed episode/task/round/summary
tables contain the added 16/32 cohort. `case_outcomes.csv` lists paired missions.
`placement_intervals.csv` and `combined_pairwise.csv` retain case-bootstrap
intervals. `combined_coordination.csv` retains exposure numerators/denominators.

The ten-case instantaneous reference is separate from the four primary
conditions. `mapf_reference_serial.json` records its outcomes and
`analysis/reference_reproduction.json` records its independent reproduction.

`manifest.json`, `cases.json`, the TOML files and `environment.json` preserve
input identities and versions. `accounting_audit.json` verifies the original
atomic transactions against the aggregate records. `audits.jsonl.gz` records
all 40 passing causal replay/action audits; large trace archives remain local.

Verify the published records with Python 3.11 or later:

```powershell
python results/uav6/agents16-32/analysis/verify_bundle.py
```

The verifier checks hashes, decompressed bytes, paired identities, exclusive
deadline partitions, binary placements, task/round denominators, pooled latency
quantiles, combined summaries and frozen source-script identities. It requires
the adjacent original result bundle, but no model, timeline or simulator setup.

`analysis/extend.py` is the exact local population driver used for the run.
It delegates simulation to the original frozen pilot harness; the only harness
adapter buffers equivalent JSON encoding for large archives. It preserves the
production solver, transport and movement implementations.
`analysis/base_pilot.py` preserves that adapted harness. `analysis/finalize.py`
reproduces combined comparisons/figures from the complete local run, and
`analysis/replay_instantaneous.py` verifies the saved reference.

The original local layout is
`output/uav6_pilot/uav6-20261024-01/extensions/agents16-32-01`.
See [the experiment guide](../../../UAV6_PILOT.md) and
[local reproduction commands](analysis/REPRODUCE.md). The fixed model, timeline
and map hashes match the original cohort. Raw numerical files are copied
unchanged; `tasks.csv.gz` and `audits.jsonl.gz` use lossless gzip compression.
