# Clean PANAMA MAPF through 16 robots

`clean_mapf_phase1_v3_scale16` evaluates whether additional episode coverage can
produce one reliable clean policy for 2, 4, 6, 8, 12 and 16 robots. Configuration:
[`configs/phase1_scale.toml`](configs/phase1_scale.toml). Run manifests and the
completion review below record provenance, outcomes and unmet qualification gates.

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

Each check appends compact local JSONL evidence. A paused or completed worker
has no implied active monitoring.
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

## Completed coverage experiment

`output/phase1_scale/clean-16-01` stopped on the predeclared validation-plateau
rule on 2026-10-23. Frozen source: `f84cfe5`; final weight hash:
`64b38b2d87ab85be38411c099e10dcdd8cfbb41da488e70bf8fe14f17c543c58`.
The checkpoint is **unqualified**. No communication experiments follow this run.

The worker completed 394,285 joint steps, 394,254 learner updates and 9,000 training
episodes: 1,500 per population and 300 per population/family. Charged wall time,
including validation, compatibility, checkpointing and shutdown, was **14,222.391 s
(3 h 57 min 02 s)**. The recorded conservative cap was 37 hours; the 2,500-episode
population cap was a ceiling alongside the stopping rule. Remaining wall time does
not extend a completed run or authorize fresh initialization.

Final greedy validation uses the same current weights on 200 independent
validation cases per population. The intervals below are Wilson case intervals;
local reports also include intervals resampling 25 map clusters. Makespan includes
250-step failures. WAITs count active-robot WAIT actions per mission.

| Robots | Success | 95% CI | Mean capped makespan | Mean WAITs |
| --- | --- | --- | --- | --- |
| 2 | 194/200 (97.0%) | 93.6–98.6% | 23.11 | 3.855 |
| 4 | 191/200 (95.5%) | 91.7–97.6% | 33.03 | 3.620 |
| 6 | 192/200 (96.0%) | 92.3–98.0% | 35.36 | 7.235 |
| 8 | 188/200 (94.0%) | 89.8–96.5% | 42.49 | 9.170 |
| 12 | 177/200 (88.5%) | 83.3–92.2% | 58.84 | 22.320 |
| 16 | 164/200 (82.0%) | 76.1–86.7% | 77.43 | 28.125 |

Family success, with 40 cases per cell:

| Robots | Random obstacles | Intersection | Corridor | Bottleneck | Warehouse |
| --- | --- | --- | --- | --- | --- |
| 2 | 90.0% | 100.0% | 97.5% | 97.5% | 100.0% |
| 4 | 92.5% | 97.5% | 90.0% | 100.0% | 97.5% |
| 6 | 85.0% | 100.0% | 97.5% | 100.0% | 97.5% |
| 8 | 87.5% | 95.0% | 92.5% | 97.5% | 97.5% |
| 12 | 70.0% | 95.0% | 87.5% | 97.5% | 92.5% |
| 16 | 67.5% | 92.5% | 67.5% | 95.0% | 87.5% |

All 94 failures repeat full states: 43 stationary cycles, 47 period-two cycles and
four period-four cycles. Random-obstacle/corridor cases account for 70/94 failures;
12/16-robot cases account for 59/94. No failed mission had every robot visit its
goal. Loss/Q probes remain finite, and the full learning log has
zero physical collision errors or communication-impaired steps. These findings
point to persistent greedy-policy cycles and limited generalization; numerical
stability and low training loss do not establish MAPF competence.

The plateau decision was independently reproduced at the ninth complete check:
the best aggregate of the last six checks was 92.167%, versus 91.333% before that
window, below the required improvement greater than one percentage point. Recovery
within the window also stayed below that threshold. No partial cohort triggered
the stop. Training coverage and all original exposure gates passed; success failed
at 8/12/16 robots and in 13/30 population/family cells. **Zero held-out cases were
evaluated.** Both strict payload verification and qualified-model loading reject
the checkpoint.

The completion audit verified 12,000 validation records, every learning row, CSV
denominators and 1,200/1,200 agreement between final and last-periodic evaluation.
Bounded multistep clean/remote-DT execution and event replay passed at all six
populations, including completed-robot yielding/return. Twelve sampled GPU case
replays matched exactly. CPU replay matched 11/12: one 12-robot corridor case
changed action at epoch 9 because CPU/GPU Q-value differences of about 1.13e-4
crossed a top-two margin of about 4.5e-5. Both trajectories failed the mission.
Cross-device bitwise trajectory reproducibility is therefore not established.

Detailed artifacts remain local: `SCALE_REPORT.md`, validation CSVs/JSONLs,
`validation_curves.png`, `compatibility.json`, and
`final_review_20261023_b/FINAL_REVIEW.md` / `final_audit.json` /
`cross_device_action_probe.json`. Preservation checks passed for all 260 failed-V1
file hashes, all eight V2 primary model weights, the original checkpoint and
historical SAGIN/staleness study inventories.

Additional episode coverage alone does not support another long run. A separately
versioned, bounded paired comparison could test population-local exploration
against the global clock while retaining the learner and execution interfaces.
The current run does not establish that exploration caused the generalization
deficit; any subsequent intervention needs validation evidence and a recorded cost
plan. No new trial is launched. Monitoring is paused pending review; held-out tests,
communication-degradation studies and communication-aware training remain closed.
