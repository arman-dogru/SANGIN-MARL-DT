# Results

Completed results, copied without moving or changing the original runs.

| Folder | Study | Episodes |
| --- | --- | ---: |
| [250ms](250ms/report.md) | Full priority-allocation evaluation at 250 ms | 1,200 |
| [staleness](staleness/report.md) | Controlled peer-staleness pilot | 216 |
| [deadlines](deadlines/report.md) | 100/250/300/500 ms sensitivity pilot | 96 |

Each folder contains `report.md`, `summary.csv`, `episodes.csv`,
`comparisons.csv`, provenance/audit JSON files, and `plots/`.
Original CSV columns and numerical values are preserved. `250ms/tasks.csv.gz`
contains the full 148,536-row task table; decompress it to read the CSV.

The pilots are diagnostic evidence, not full studies. Staleness is an artificial
stress condition, and vertex-only metrics remain exploratory. Reports preserve
negative findings, uncertainty and exposure denominators. Historical approval
statements refer to the original run, not the current training status.

`files.csv` maps each published artifact to its original local file and SHA-256.
The original filename/path references in reports have been shortened where
possible; their findings were not recalculated. Manifests retain original paths
as provenance, so some refer to local assets absent from this repository.

Multi-gigabyte raw replay archives, checkpoints, orbital input and manuscripts
remain local. The active Phase 1 training run is not included or modified.
