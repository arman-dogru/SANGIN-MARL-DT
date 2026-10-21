# Reproduce the paper study

All work is on `main`. Install the project's optimization, development, and LEO
extras. The immutable input hashes and qualification evidence are recorded in
`assets_manifest.json`; the orbital source/protocol/RF metadata is included in
`physics_audit.json`. The large model and timelines remain ignored local inputs.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_paper_inputs.py --checkpoint "D:\Projects\PANAMA+UAV\panama_uav\output\leo-learning\leo-learning-20261009-01-recovery-03\clean\final_model.pt" --timeline "D:\Projects\PANAMA+UAV\panama_uav\data\leo\tremblant_full.npz"
.\.venv\Scripts\python.exe -m panama_uav allocation-sanity --output output\sanity-new
.\.venv\Scripts\python.exe -m panama_uav --config configs\paper_study.toml resource-study --run-id paper-allocation-20261021 --model artifacts\inputs\final_model.pt --workers 8
```

A run ID must be new. To continue its saved case bank after an interruption, add
`--resume`. Configuration, qualified checkpoint, channel, seeds, map and execution
kernel identities must match. Resume events preserve code hashes and numerical
revisions; existing episode rows are not duplicated. Raw outputs are retained at
`output/resources/<run-id>/`: per-condition episode tables, task/allocation JSONL,
and complete compressed traces with policy observations/actions, priorities,
source/generation times, resource decisions, service events and metrics.

The complete protocol is 97 conditions x 4 populations x 50 held-out cases =
19,400 episodes, each capped at 250 steps. The same qualified frozen policy is
used throughout. Case-specific archived orbital offsets span the six-hour window
and remain paired across conditions. Time is simulated; no wall-clock solver time
is added to logical task CPU demand.

After completion:

```powershell
.\.venv\Scripts\python.exe scripts\audit_resource_loop.py --study output\resources\paper-allocation-20261021 --output research\full_loop_audit.json
.\.venv\Scripts\python.exe scripts\verify_resource_records.py --study output\resources\paper-allocation-20261021 --output research\full_record_audit.json
.\.venv\Scripts\python.exe scripts\report_paper_results.py --study output\resources\paper-allocation-20261021
.\.venv\Scripts\python.exe scripts\paper_diagrams.py
```

The report publisher rejects partial or diagnostic runs and checks all case and
intermediate-record counts before writing the paper's numerical results. Summary
data, case seeds, full configuration/provenance and verification are copied into
`research/results/`; compact scientific plots are saved in PNG, SVG and PDF under
`paper/figures/`. Full raw traces remain local because of their size.

Open `paper/` in the LaTeX editor, or compile `paper/main.tex` with a standard LaTeX
distribution. This workstation also uses the official portable Tectonic 0.17.0
binary under ignored `tmp/tools/tectonic/`. Bibliographic citations link to the
original author preprints and to the implemented ITU-R P.618-13 revision.

The completed nominal comparison is reproducible with the same resource-study
command and a new run ID, adding `--conditions up-1-cpu-1__equal
up-1-cpu-1__static up-1-cpu-1__aoi up-1-cpu-1__priority`. The fresh reference is
included automatically. This gives 1,000 episodes. Publish it with
`--study output/resources/<nominal-run-id> --output research/baseline_results`.
`scripts/compare_resource_repeats.py` compares physical outcomes between repeated
case banks, excluding host-dependent solver wall time. Current validation matches
all 1,000 nominal episodes against their independent full-grid repeats.

Worker count changes throughput only; this workstation uses 24 of 36 logical
cores for the full study. Keep the configuration and frozen model fixed. The
action policy executes at robot decision epochs using local sensing and delivered
caches; finite-CPU offloaded jobs generate guidance/coordination support.
