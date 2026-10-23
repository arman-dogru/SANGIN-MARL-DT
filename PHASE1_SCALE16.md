# Clean PANAMA MAPF through 16 robots

`clean_mapf_phase1_v3_scale16` evaluates whether additional episode coverage can
produce one reliable clean policy for 2, 4, 6, 8, 12 and 16 robots. Configuration:
[`configs/phase1_scale.toml`](configs/phase1_scale.toml). Operational status and
the later research phases are recorded in [`plan.md`](plan.md).

## Execution and learning

The learner retains the strongest V2 validation settings: learning rate 0.001,
discount 0.90 and a global epsilon decay of 20,000 joint steps. The observation
remains three 15×15 spatial channels and a 12-element vector; the action space
remains four cardinal moves plus WAIT. Information is fresh and instantaneous.
The SAGIN allocator, radio model, workload and CPU assumptions are unchanged.

Training uses `RemoteAlignedCleanMAPFEnvironment`: active robots use the DQN;
completed robots use the existing local yielding/return controller. Completed
local actions never enter DQN replay. Conflicts are resolved with the same
priority and movement kernels as remote-DT execution. Explicit WAITs, blocked
movement and completed local motion are measured separately.

The episode cap is **2,500 completed episodes per population**, **15,000 total**,
with a 250-step cap per episode. The maximum training workload is therefore
3,750,000 joint steps. A fixed measured wall-time cap bounds the complete worker,
including setup, validation, checkpointing and final evaluation. The runtime
estimator rejects projections above the separate 72-hour protocol ceiling.
Neither limit is extended during execution.

Population introduction occurs every 40 completed episodes. Selection then
rehearses the introduced population with the fewest completed episodes, with a
deterministic population tie-break and family rotation. An unsuccessful stage
cannot indefinitely prevent introduction of larger populations. A greedy
validation check covers all six populations after every 1,000 training episodes.
Introduction and rehearsal do not depend on validation success.

This protocol tests episode coverage with the original learner. Population-local
exploration or other substantial learner changes require a separately versioned,
paired controlled comparison. The failed V2 comparison remains unchanged; its
original full-run launch criterion is not reclassified as passed.

## Data and reference planning

The five families are random obstacles, intersections, corridors, bottlenecks
and warehouses. Each family has 20 training maps, five validation maps and five
independent test maps. Both evaluation splits contain 200 cases per population,
40 per family: **1,200 validation and 1,200 held-out cases**. Half the cases in
each family deliberately create timed FoV-relevant interactions.

Canonical map hashes exclude historical V1/V2 banks and controlled-stress maps
under cropping, translation, rotation and reflection. Training uses training
maps only. The test bank is not evaluated during calibration or training.
The initial map size remains 32×32. Larger geometry requires a separate dataset
version if measured density or feasibility makes the existing geometry unsuitable.

The bounded reference audit samples designed/random validation cases for every
family at 12 and 16 robots. Independent CBS supplies constructive MAPF witnesses
with vertex and edge-swap checks. Timeouts and node limits are unresolved outcomes;
they are not infeasibility certificates. Reference feasibility concerns geometric
MAPF solutions, not a guarantee that the learned policy/local controller follows them.

## Qualification and stopping

`PANAMA-CLEAN` names a qualified frozen checkpoint only. The same current model
must meet all requirements simultaneously:

- At least 500 completed training episodes in every population.
- Complete 200-case validation and held-out cohorts in every population.
- At least 95% mission success in each population and every 40-case family.
- Original coordination-exposure thresholds: at least 20 exposed cases/family
  and 200 relevant directed pairs/population.
- Passing physical correctness, deterministic replay and actual zero-delay
  remote-DT observation/action/priority/reward/movement compatibility at all six populations.

The held-out bank opens only after coverage, complete final validation and
compatibility pass. Ablations, smoke runs, partial cohorts and historical stage
passes cannot confer qualification. Qualified-model loading checks the six
population contract and current-weight compatibility identity.

Learning stops at its episode/step/wall cap, complete-cohort readiness after
minimum coverage, or a declared plateau: at least 90,000 joint steps, seven
complete validation checks and no improvement greater than one percentage point
across six checks, without recovery in the latter half. Partial checks cannot
trigger a plateau or graduation. All previous checkpoints and runs remain intact.

## Calibration and execution

Prepare a new disjoint dataset and inspect larger-population feasibility:

```powershell
.venv\Scripts\python.exe scripts/run_phase1_scale.py --run-id scale16-data-01 --prepare-data --reference-audit
```

A bounded calibration trains a disposable model for two complete episodes per
population, measures conditional training throughput and checkpoint cost, and
times validation inference. Its weights never initialize the full run:

```powershell
.venv\Scripts\python.exe scripts/run_phase1_scale.py --run-id scale16-calibration-01 --benchmark --device cuda
```

The runtime plan charges every planned failure and evaluation case at 250 steps,
all periodic full validation checks, both final cohorts, compatibility,
checkpointing and shutdown. A 20% margin is applied, then the wall cap is rounded
up to 30-minute units. A runtime forecast cannot establish eventual success.

Commit source before freezing a new full run. The benchmark must match the
dataset, code hashes, protocol and recorded budget:

```powershell
.venv\Scripts\python.exe scripts/run_phase1_scale.py --run-id scale16-clean-01 --prepare --benchmark-json output/phase1_scale/scale16-calibration-01/benchmark.json
# Register the hourly scheduler first. The flag attests successful registration.
.venv\Scripts\python.exe scripts/run_phase1_scale.py --run-id scale16-clean-01 --launch --device cuda --monitor-registered
```

Run IDs are unique. `--resume --launch` requires a saved checkpoint and remaining
budget; a missing checkpoint cannot trigger fresh initialization. Launch guards
reject overlapping PANAMA workers. Worker source is loaded from the run's frozen
`code/` directory, so later repository edits do not change an active experiment.

## Monitoring and artifacts

Hourly monitoring verifies full process command and creation time, including
Windows venv launcher/child processes. Advancing log/checkpoint values establish
progress; PID existence alone does not. `research_monitor.py` records CPU/GPU
snapshots, memory, remaining wall budget, steps/episodes by population, recent
loss/Q/replay/epsilon, WAITs, blocked motion and greedy validation. Device GPU
utilization is labelled as a device-wide snapshot. The monitor never launches work.

Each check appends compact local JSONL evidence and a timestamped operational
entry to `plan.md`. A paused or completed worker has no implied active monitoring.
Source, dataset and conservative budget-lease checks precede any continuation.

Local artifacts include `manifest.json`, `protocol.toml`, `budget_plan.json`,
resumable `checkpoint.pt`, inference checkpoints, learning/episode/validation logs,
`compatibility.json`, per-case and failure CSVs, population/family summaries and
`SCALE_REPORT.md`. Models, maps and large results stay outside version control.

## Current evidence

The completed V2 comparison reached 90.0/93.75/93.75/80.0% validation success for
2/4/6/8 robots under its original settings. It did not qualify. All 32 sampled
failed replays repeated full states; the sample was outcome-stratified, not an
unbiased failure-frequency estimate. Both gamma-0.99 alternatives failed every
terminal greedy validation case within their short budgets.

Scaling regression tests exercise actual clean/remote execution at 12/16 robots,
population/family exposure, deterministic resume, capped budgets and test gating.
They identified an audit-only integer-versus-JSON-string key ordering discrepancy
for two-digit robot IDs. Provenance hashing now normalizes keys before sorting,
matching recorded event canonicalization. Physical decisions and transport are unchanged.

No scale-16 policy is qualified at protocol introduction. Communication-degradation
and communication-aware studies remain gated on the required clean baseline.
