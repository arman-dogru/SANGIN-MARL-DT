# Results

Completed results, copied without moving or changing the original runs.

| Folder | Study | Episodes |
| --- | --- | ---: |
| [250ms](250ms/report.md) | Full priority-allocation evaluation at 250 ms | 1,200 |
| [staleness](staleness/report.md) | Controlled peer-staleness pilot | 216 |
| [deadlines](deadlines/report.md) | 100/250/300/500 ms sensitivity pilot | 96 |
| [communication](communication/report.md) | Four execution placements and controlled communication sensitivities | 1,640 |
| [uav6](uav6/report.md) | Parameter-only 1/6 GHz UAV CPU pilot | 60 |

Each folder contains `report.md`, `summary.csv`, `episodes.csv`,
`comparisons.csv`, provenance/audit JSON files, and `plots/`.
Original CSV columns and numerical values are preserved. `250ms/tasks.csv.gz`
contains the full 148,536-row task table; decompress it to read the CSV.
`communication/tasks.csv.gz` and `communication/rounds.csv.gz` contain all
819,045 tasks and 169,876 inference rounds from the frozen-policy study.
The communication folder also includes `primary.csv`, `primary_comparisons.csv`
and a README describing its validation and reproduction inputs.
`uav6/` compares four placements on five cases per supported population; 16 robots
are explicitly excluded by model qualification. All UAV CPU pilot tables and
figures are diagnostic parameter-only evidence.

The pilots are diagnostic evidence, not full studies. Staleness is an artificial
stress condition, and vertex-only metrics remain exploratory. Reports preserve
negative findings, uncertainty and exposure denominators. Configuration and
qualification statements describe the checkpoint and protocol used in each study.

`files.csv` maps each published artifact to its original local file and SHA-256.
The original filename/path references in reports have been shortened where
possible; their findings were not recalculated. Manifests retain original paths
as provenance, so some refer to local assets absent from this repository.

Multi-gigabyte raw replay archives, checkpoints, orbital input and manuscripts
remain local. Phase 1 training artifacts are excluded from these published summaries.
