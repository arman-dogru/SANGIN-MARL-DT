# Communication results

Frozen PANAMA inference across allocated, all-cloud, all-UAV and instant-baseline
execution. Forty paired cases cover 2/4/6/8 robots. Four primary arms, three
finite-CPU ideal-network controls and 34 single-factor sensitivities produce
1,640 simulations. Each episode uses a one-second control interval, a 250 ms
command cutoff and a 250-step cap.

- [report.md](report.md): measured findings, paired confidence intervals and limitations.
- [primary.csv](primary.csv) and [primary_comparisons.csv](primary_comparisons.csv):
  the four primary arms and six pairwise contrasts by population.
- [summary.csv](summary.csv) and [comparisons.csv](comparisons.csv): all 41 conditions.
- [episodes.csv](episodes.csv): all 1,640 mission records.
- `tasks.csv.gz` and `rounds.csv.gz`: all 819,045 tasks and 169,876 inference rounds.
- [plots/](plots/): seven figure sets in vector PDF/SVG and 600 dpi PNG.
- [manifest.json](manifest.json), [cases.json](cases.json) and
  [status.json](status.json): exact configuration, input/source hashes, paired cases
  and completed runtime.
- [validation.json](validation.json), [denominator_audit.json](denominator_audit.json),
  [mechanism_audit.json](mechanism_audit.json), [preservation.json](preservation.json),
  `audits.jsonl.gz` and [tests.xml](tests.xml): replay, denominator, mechanism,
  preservation and regression evidence.
- [bundle_validation.json](bundle_validation.json): published-file hash, compression
  and record-count checks.

The gzip files contain the original CSV/JSONL bytes. For example:

```python
import csv
import gzip

with gzip.open('results/communication/tasks.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
    for task in csv.DictReader(stream):
        pass  # Process one task at a time.
```

[../files.csv](../files.csv) maps published names to original local filenames,
source SHA-256 and published SHA-256. No numerical values were recalculated for
publication. `source_review_manifest.json` retains the original run's artifact
hashes and filenames. The files in [analysis/](analysis/) are the original
supplemental analysis scripts; they expect the original run layout, including
raw archives for telemetry and mechanism audits.

The original run is `output/communication_effects/inference-20261023-01`.
Checkpoint, archived orbital timeline, source snapshot and multi-gigabyte raw
replays remain local. Their identities are recorded in the manifest; local paths
are provenance, not assets supplied by a fresh clone. The frozen policy SHA-256
is `be085bec93c4b8603629b0e964b10f7b5a99220386e9c968ac17f960759ffc0f`.

[../../COMMUNICATION_EFFECTS.md](../../COMMUNICATION_EFFECTS.md) documents the
workflow for reproducing the experiment with the matching external inputs.
The original results and all historical experiments are preserved.

Verify the published hashes, decompressed bytes and paired record identities
without the external assets:

```powershell
python results/communication/analysis/verify_bundle.py
```
