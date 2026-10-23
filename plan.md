# SANGIN research execution plan

## Current status

- Phase: **1 clean PANAMA training through 16 robots**.
- Clean curriculum: stage **6**, introduced populations **[2, 4, 6, 8, 12, 16]**; no population qualified.
- Active research workers: **`output/phase1_scale/clean-16-01`**, frozen source `f84cfe5`. Hourly monitor: **active**, registered before GPU calibration and training.
- Episode budget: **2,500 completed episodes per population; 15,000 total per clean run**.
  The later communication-aware comparison uses the same episode cap. Measured full-run wall cap: **37 hours (133,200 s)**; final-evaluation reserve: **90 minutes**.
- Qualified policies: **none**. `PANAMA-CLEAN` and `PANAMA-COMM` do not exist as qualified artifacts.
- Starting source: `9d52899`. Work stays on `main`; source/configuration/data freeze before each launch.
- Current execution: fresh clean training under the frozen episode/wall limits; complete greedy validation every 1,000 episodes. Later communication experiments remain gated on clean qualification.
- Latest audited snapshot (**2026-10-23 01:32 EDT**): 204,318 joint steps through completed episodes;
  4,386 completed episodes, 731 per population. Fourth complete greedy validation:
  **96.5/96.0/95.5/91.0/90.0/81.5%** success at 2/4/6/8/12/16 robots. Minimum training-episode
  coverage is met; population/family success gates remain unmet. Test bank closed; no policy qualified.
- Scientific stop: the old primary comparison failed its launch rule. No old extended run was launched.
  The scale-16 coverage experiment uses a separate versioned protocol; that result and gate remain unchanged.

## Research objectives and phase gates

Separate clean MAPF competence, network impairments, allocation effects and adaptation through learning.
Preserve every previous source revision, checkpoint, failed experiment, dataset and study result.
Large results/checkpoints remain local. Commit code, tests, configuration, methodology and this plan only.

| Phase | Deliverable | Prerequisites / current disposition |
| --- | --- | --- |
| 0 | This plan, provenance, explicit budgets, functioning hourly monitoring | Complete: 342 regression tests; measured 37-hour cap; hourly monitoring active |
| 1 | Frozen `PANAMA-CLEAN`, qualified simultaneously at 2/4/6/8/12/16 robots | Current phase; independent full validation, test, exposure, compatibility and replay required |
| 2 | Versioned SAGIN benchmark, complete return path and RF/capacity provenance | Independent audits may proceed; scheduler/radio/workload unchanged initially |
| 3 | Paired frozen-policy communication-degradation curves | Blocked until Phase 1 qualification and Phase 2 validation |
| 4 | `PANAMA-COMM` and equal-budget clean-continued control | Blocked until measured Phase 3 degradation and a declared training/acceptance protocol |
| 5 | Independent recovery/generalization evaluation | Frozen qualified clean/communication-aware/control models, independent cases |
| 6 | Joint causal analysis, figures, negative findings and methodological limitations | Evidence from preceding phases; no paper performance claims before evidence |
| 7 | Final preservation, qualification, replay, provenance and reporting audit | All required artifacts actually verified |

Network benchmark: preserve robot→UAV→robot and robot→UAV→satellite/gateway→cloud→return routes.
Separate G2A/A2G/A2S/S2A serialization, propagation, queues, frame scheduling, interruptions and deadlines.
Do not count analytical round-trip propagation twice. G2A is an average probabilistic-LoS model;
A2G capacity and satellite RF assumptions require provenance audit. Attenuation statistics are not packet-loss
probabilities. Synthetic staleness/erasure remains labelled injected stress, not measured network reliability.

Communication experiments will pair maps, starts/goals, seeds, frozen weights, CPU/workload, motion rules
and horizons. Pilot manageable treatment levels before freezing a primary design. Separate late commands
from stale observations and priority errors. Always report mission outcomes as well as network outcomes.
Communication-aware training initially retains five movement actions and an independently controlled scheduler.
Compare frozen clean, equal-budget ideal continuation and impaired-communication fine-tuning. Additional
observations require causal availability, a new schema and a separate controlled comparison.
Recovery reports absolute loss/recovery and paired intervals; the recovery ratio is unavailable for a zero
or predeclared negligible degradation denominator. Exploratory vertex-only metrics remain separate.

## Existing evidence and preserved artifacts

- Failed V1: `output/phase1_clean/phase1-clean-20261021-01`, source `13f47a0`.
  Held-out success: 97.0/85.0/74.5/61.0% (2/4/6/8). Curriculum stopped at four robots.
- V2 corrections: `b6264fa`; primary frozen source `8954c49`; measured-budget tooling `9d52899`.
- Primary: `output/phase1_revised/phase1-comparison-20261022-01`.
  Eight 30,000-step runs, seeds 42/73; 240,000 joint steps and 1,280 validation evaluations on 160 unique cases.
  Best settings: learning rate .001, gamma .90, epsilon decay 20,000 steps.
  Population success: 90.0/93.75/93.75/80.0%; unqualified. Lower-rate alternative: 65.94% overall;
  both gamma .99 alternatives: 0%. Full launch criterion failed. New held-out test was not evaluated.
- Bounded current-model diagnosis: 32/32 sampled failures repeated full states, 0/16 successful controls did.
  The sample was outcome-stratified. All 48 diagnostic replays matched. Loss/Q probes were finite.
  Longer-horizon alternatives had rising loss/Q magnitudes. No validation plateau can be inferred from
  one terminal greedy check per ablation.
- Historical 1,200-episode SAGIN study, 216-episode stress pilot, deadline pilots and their inputs remain immutable.
  The primary audit verified 260 V1 file hashes plus old-study inventories and the original checkpoint.
- Scale-16 release regression: **342 passed**; actual bounded clean/remote-DT execution compatibility
  passed at all six populations. Qualification still requires compatibility for the final current weights.

## Phase 1 protocol and next controlled comparison

Protocol: `clean_mapf_phase1_v3_scale16` (new; historical V1/V2 retain their contracts).
Keep fresh instantaneous observations, PANAMA's 3×15×15 spatial / 12-vector observation, five actions,
priorities, rewards, collision resolver, completed local control, active-only replay and physical time.
No SAGIN impairments, resource-allocation/radio/CPU/workload changes or retraining on evaluation maps.

New deterministic train/validation/test maps exclude historical V1/V2 maps and the controlled stress maps,
including canonical rotations/reflections. Five families: random obstacles, intersections, corridors,
bottlenecks, warehouses. Full validation and untouched test each contain 200 cases/population (40/family),
1,200 cases/split. Case identities and map hashes freeze before training. Reference planning uses bounded
independent CBS; feasible paths are checked; timeouts/node limits remain unresolved, never infeasible.
Keep 32×32 geometry unless measured feasibility/exposure demonstrates a need for separately versioned maps.

Candidate hypothesis for a subsequent bounded comparison: progressive population introduction plus rehearsal needs
population-specific exploration exposure for newly introduced 12/16-robot tasks. Compare global-clock
epsilon with population-clock epsilon under otherwise identical progressive schedules and original gamma
.90/lr .001. This hypothesis follows residual repeated-state failures and short population exposure;
it does not establish exploration collapse in the old comparison. Gamma .99 alternatives are not repeated
without new supporting evidence. The initial scale-16 coverage run retains the original global-clock
exploration and learner. It tests whether increased episode coverage resolves the remaining MAPF failures.
Population-clock exploration remains a candidate intervention requiring its own controlled comparison.

Introduce populations every 40 completed training episodes, independent of success gates. Rehearse introduced
populations with least-completed-episode selection and rotate all families within each population.
An unsuccessful early stage cannot exclude later populations permanently. Track all introduced populations
on greedy validation to expose forgetting. A graduation record belongs to the current frozen weights only.

Substantial exploration changes require paired learning seeds and validation-only selection. The comparison
selection criterion remains a >=5 pp paired improvement with a positive lower 95% interval and no >5 pp population
loss, or >=90% in every target population. The new six-population comparison is separate from the old four-population gate.
The initial scale-16 run is an episode-coverage experiment with the strongest existing validation settings,
not an algorithm-selection claim. The historical V2 full-run gate remains failed. Record throughput and a
complete coverage/evaluation budget before launch. Runtime estimates do not establish qualification.

## Qualification and stopping

One frozen policy must simultaneously pass every population at >=95% on complete independent validation
and untouched test: 200 cases/population, 40/family, >=95% in every family, original coordination-exposure
requirements, >=500 completed training episodes/population, movement correctness, deterministic replay and
actual zero-delay remote-DT execution compatibility at all six populations. Historical stage passes,
partial cohorts, pooled training counts from separate models and ablations cannot qualify a model.
The test bank stays closed until coverage, full final validation and compatibility pass.

Each run has a fixed source/config/data/seed identity, cumulative budget reservation and conservative
resume lease. No duplicate worker or fresh restart under a resume identity. Stop on numerical errors,
exhausted budget, infrastructure failure, unsupported selection or a declared validation plateau.
Plateau checks require complete cohorts and sufficient coverage; partial checks cannot trigger graduation.
Retain the best validation-verified unqualified checkpoint with explicit unmet gates; never publish it as
the qualified communication baseline. Later iterations require a diagnosed mechanism, new protocol/run
and sufficient remaining authorized budget. Opened test outcomes never select interventions.

## Resource and launch register

The clean-run training cap is 2,500 completed episodes per population (15,000 total), equivalent to at most
3,750,000 joint steps under the 250-step episode cap. A measured wall-time cap also bounds execution.
Charge GPU worker wall time including initialization, validation/checkpoints and recovery.
Training/test preparation and bounded CPU regression/feasibility diagnostics are implementation validation.
Consumed V1/V2 budgets remain separate from the new run's accounting.

Before a launch record: source commit + hashes, configuration hash, disjoint dataset hash, seeds, run/output ID,
comparison hypothesis, individual cap, remaining cumulative budget, checkpoint rule, validation schedule,
final evaluation reserve, stop rule and successful hourly automation registration.

## Hourly monitoring contract

Use the existing thread automation, update its cadence to hourly, and activate it before any long worker starts.
Do not claim active monitoring without a successful scheduler response. Resume monitoring for successive runs;
pause it and record the reason if there is no worker or a scientific/resource gate blocks further execution.

At each check verify PID creation time and full command including launcher/child, and advancing steps,
episodes or case timestamps. Record CPU/GPU/memory, elapsed/remaining budget, checkpoint age/integrity,
population steps/episodes, loss/Q/gradients/replay/epsilon, latest greedy validation, WAITs/blocked motion,
conflicts/cycles/deadlocks/makespan, forgetting/readiness, evidence and continue/diagnose/resume/stop decision.
Append an America/Toronto timestamp and next scheduled check here. Large raw monitor records stay local.
Only resume a confirmed stopped authorized worker from its matching checkpoint and remaining budget.

## Progress and decision history (America/Toronto)

- **2026-10-22 22:49 EDT** — Phase 0 initialized at `9d52899`; working tree synchronized with `main`.
  No active run. The primary V2 comparison remains an unsupported-selection stop. Scale-16 workflow,
  regression and feasibility validation started. Next automated check: paused pending a validated launch.
- **2026-10-22** — Training budget set to 2,500 completed episodes per population, 15,000 total per run.
  The initial scale-16 experiment retains original DQN settings and tests additional episode coverage.
  Communication-cutoff training retains the same future cap and remains gated on clean qualification.
  Documentation convention: describe implementation, configuration, evidence and limitations; keep
  conversational approval narratives and assistant progress prose out of repository documentation.

- **2026-10-22 23:29 EDT** — New disjoint dataset `output/phase1_scale/data-16-01`: 150 maps, 1,200 validation cases and 1,200 untouched test cases. Independent bounded CBS verified 16/20 sampled large-population missions: 9/10 at 12 robots and 7/10 at 16 robots. Four 2-second timeouts remain unresolved; no sampled case was proven infeasible. Feasible witness makespan was at most 37 steps. The full regression suite passed 340 tests; 22 targeted checks passed after the final budget/monitor guards. The hourly scheduler is configured and paused; activation precedes GPU calibration/training. Next step: final regression, frozen code commit and measured runtime calibration.

- **2026-10-22 23:31 EDT** — Release regression: 342 passed in 124.53 s, with one ITU-R low-elevation advisory. Preservation audit passed: all 260 failed-V1 file hashes, 1,308 historical 250 ms study files, 243 staleness-pilot files, the original input checkpoint and all eight primary-comparison model weights. Scale-16 source is ready to freeze. Next step: bounded disposable GPU cost calibration, then a recorded wall-time plan.

- **2026-10-22 23:36 EDT** — Calibration `calibrate-16-01` (source `947e991`): 3,000 joint learning steps, 12 completed episodes (two/population), 78.657 s total. No held-out cases evaluated; calibration weights are disposable. Conditional training rates: 263,004 / 264,201 / 233,191 / 209,084 / 159,787 / 107,162 joint steps/hour for 2/4/6/8/12/16 robots. The all-250-step training projection is 20.16 h; 15 complete validation checks, checkpoint overhead and 90-minute final evaluation reserve produce 30.81 h before the 20% margin. The fixed full-run cap is 37 h, rounded to 30-minute units, alongside the 15,000-episode / 3,750,000-step caps. Hourly automation registration succeeded. Next step: freeze and launch `clean-16-01`; next automated check follows the hourly cadence. No qualification or clean-performance claim follows from this cost calibration.

- **2026-10-22 23:39 EDT** — `clean-16-01`: continue; 3,072 joint steps, 12 completed episodes; by population {'2': 12}. Runtime 0.01 h; remaining wall budget 36.99 h. Recent loss 0.0023310681651764753; WAITs/blocked moves 64/23 over 256 logged joint steps. Checkpoint valid: True. Next scheduled check: 2026-10-23T00:39:32.804719-04:00. Raw evidence: `D:\Projects\SANGIN-dev\output\phase1_scale\clean-16-01/monitor_history.jsonl`.

- **2026-10-22 23:41 EDT** — `clean-16-01`: continue; 11,392 joint steps, 76 completed episodes; by population {'2': 40, '4': 36}. Runtime 0.05 h; remaining wall budget 36.95 h. Recent loss 0.0036247653993086715; WAITs/blocked moves 81/59 over 256 logged joint steps. Checkpoint valid: True. Next scheduled check: 2026-10-23T00:41:39.539190-04:00. Raw evidence: `D:\Projects\SANGIN-dev\output\phase1_scale\clean-16-01/monitor_history.jsonl`.

- **2026-10-22 23:41 EDT** — `clean-16-01` launched at 23:38 EDT from `f84cfe5`; worker/checkpoint identity and advancing logs verified. Latest status: 11,648 joint steps, 79 completed episodes; by population {'2': 40, '4': 39}. The run has a 15,000-episode ceiling and fixed 37-hour wall cap. First complete greedy validation is pending; early exploratory training outcomes are not qualification evidence. Hourly monitoring is active; next check follows the hourly cadence. Source, data, workload and motion rules remain frozen.

- **2026-10-23 00:36 EDT** — `clean-16-01`: continue; 106,496 joint steps, 1,998 completed episodes; by population {'2': 333, '4': 333, '6': 333, '8': 333, '12': 333, '16': 333}. Runtime 0.84 h; remaining wall budget 36.16 h. Recent loss 0.004275670290553535; WAITs/blocked moves 76/35 over 256 logged joint steps. Checkpoint valid: True. Next scheduled check: 2026-10-23T01:36:43.662488-04:00. Raw evidence: `D:\Projects\SANGIN-dev\output\phase1_scale\clean-16-01/monitor_history.jsonl`.

- **2026-10-23 00:38 EDT** — Both scheduled 1,200-case greedy validation cohorts are complete.
  Learning resumed after the second cohort: 106,880 joint steps and 2,009 completed episodes;
  335 episodes each at 2/4/6/8/12 robots and 334 at 16. Every population has 66–67 episodes per family.
  Worker PID 78484 matches the launch command and creation time; CPU 71.7%, RSS 2.17 GiB.
  Checkpoint manifest identity passed (105,000-step checkpoint at the 00:36 check).
  All 59 frozen Python hashes, budget hash, dataset/map validation, evaluation case identities,
  model identities and metric denominator checks passed. Stderr is empty.
  Active lease elapsed **1.00 h**, remaining **36.00 h**; this includes evaluation time that was
  absent from the earlier 00:36 training-status timestamp. Recent probes remain finite:
  Q absolute maximum 2.087, clipped gradient norm 0.03065, replay 50,000, epsilon 0.1043;
  no physical collision errors in the inspected 256 learning rows. Device-wide GPU snapshot at 00:36:
  13% / 0% utilization on the two RTX 3070 devices during validation; this is not worker-exclusive utilization.
  Local detailed evidence: `output/phase1_scale/clean-16-01/monitor_audit_20261023_00.json`.

  Greedy validation uses identical 200-case population cohorts at both checkpoints. The intervals below
  describe the second checkpoint (106,556 joint steps, weight hash `94c9dc4c1c04…`). Wilson intervals
  treat cases as independent; the additional bootstrap intervals resample the 25 map clusters.
  Makespan includes 250-step failures; WAIT counts are summed active-robot WAIT actions per mission.

  | Robots | First success | Second success | Wilson 95% CI | Map-cluster 95% CI | Mean capped makespan | Mean WAITs | Relevant / directed pairs |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 2 | 88.0% | 196/200 (98.0%) | 95.0–99.2% | 95.5–100.0% | 20.72 | 1.355 | 1,742 / 8,288 |
  | 4 | 84.5% | 190/200 (95.0%) | 91.0–97.3% | 91.5–98.0% | 34.36 | 4.195 | 6,018 / 82,452 |
  | 6 | 79.5% | 185/200 (92.5%) | 88.0–95.4% | 88.0–96.0% | 43.88 | 11.245 | 10,730 / 263,310 |
  | 8 | 78.0% | 178/200 (89.0%) | 83.9–92.6% | 81.0–95.5% | 54.47 | 22.465 | 27,624 / 610,008 |
  | 12 | 77.0% | 175/200 (87.5%) | 82.2–91.4% | 79.5–94.0% | 63.61 | 20.025 | 37,146 / 1,679,304 |
  | 16 | 67.0% | 160/200 (80.0%) | 73.9–85.0% | 72.0–87.5% | 83.87 | 34.085 | 79,064 / 4,025,520 |

  Second-cohort family success (40 cases per cell):

  | Robots | Random obstacles | Intersection | Corridor | Bottleneck | Warehouse |
  | --- | --- | --- | --- | --- | --- |
  | 2 | 90.0% | 100.0% | 100.0% | 100.0% | 100.0% |
  | 4 | 85.0% | 100.0% | 95.0% | 97.5% | 97.5% |
  | 6 | 77.5% | 100.0% | 95.0% | 95.0% | 95.0% |
  | 8 | 72.5% | 87.5% | 85.0% | 100.0% | 100.0% |
  | 12 | 65.0% | 95.0% | 90.0% | 92.5% | 95.0% |
  | 16 | 57.5% | 87.5% | 82.5% | 85.0% | 87.5% |

  All six aggregate rates improved between these two checks; this is early learning evidence, not
  qualification or a plateau analysis. All 116 second-cohort failures contain repeated full states;
  one 16-robot failure visited every goal without simultaneous mission completion. Random-obstacle
  generalization is the largest current family deficit. Attempted vertex/edge conflicts total
  6,566/1,090 before physical resolution. Blocked moves total 14,649, including obstacle/resolution blocking;
  attempted conflicts are distinct from physical collisions.
  Original coordination-exposure gates pass in each population, but no population has the required
  500 completed training episodes, and population/family success gates remain unmet.
  The untouched test bank remains closed. Continue the unchanged frozen run within its existing caps.
  Next hourly check: **2026-10-23 01:38 EDT**.

- **2026-10-23 01:31 EDT** — `clean-16-01`: continue; 202,624 joint steps, 4,353 completed episodes; by population {'2': 726, '4': 726, '6': 726, '8': 725, '12': 725, '16': 725}. Runtime 1.88 h; remaining wall budget 35.12 h. Recent loss 0.003072034031902149; WAITs/blocked moves 41/8 over 256 logged joint steps. Checkpoint valid: True. Next scheduled check: 2026-10-23T02:31:23.039588-04:00. Raw evidence: `D:\Projects\SANGIN-dev\output\phase1_scale\clean-16-01/monitor_history.jsonl`.

- **2026-10-23 01:32 EDT** — `clean-16-01`: continue. Minimum training coverage is now met:
  4,386 completed episodes, **731 per population**, and 204,318 joint steps through those episodes.
  Population training steps (2/4/6/8/12/16): 21,188 / 24,705 / 28,355 / 32,943 / 44,372 / 52,755;
  each population has 146–147 episodes per family. Advancing learning/episode logs and four complete
  1,200-case validation cohorts establish progress. Worker PID 78484 and its creation time/full command
  match the recorded Windows launcher/child identity. CPU 86%, RSS 2.18 GiB; device-wide RTX 3070
  utilization 46% / 0% (01:31 snapshot). Lease elapsed **1.89 h**, remaining **35.11 h** of the fixed 37 h.
  The 200,000-step checkpoint passed manifest identity and was 57 s old at 01:31. All 59 frozen source
  hashes, budget/dataset/map hashes, evaluation case/model identities and denominator checks passed;
  stderr remains empty. Last 256 learning rows: loss mean 0.002879, 43 active WAITs, 36 blocked moves,
  16/1 attempted vertex/edge conflicts and zero physical collision errors. Latest finite probe:
  Q absolute maximum 2.082, clipped gradient norm 0.04175, replay 50,000, epsilon 0.10003.

  Fourth complete greedy validation (188,572-step weights `fa498a73e762…`), 200 cases/population.
  Success intervals are Wilson; map-cluster intervals resample the 25 validation maps.
  Makespan includes 250-step failures; WAITs count active-robot WAIT actions per mission.

  | Robots | Success | Wilson 95% CI | Map-cluster 95% CI | Mean capped makespan | Mean WAITs | Relevant / directed pairs |
  | --- | --- | --- | --- | --- | --- | --- |
  | 2 | 193/200 (96.5%) | 93.0–98.3% | 92.0–100.0% | 24.11 | 6.200 | 1,242 / 9,644 |
  | 4 | 192/200 (96.0%) | 92.3–98.0% | 92.0–99.5% | 32.12 | 8.025 | 3,666 / 77,076 |
  | 6 | 191/200 (95.5%) | 91.7–97.6% | 91.5–98.5% | 36.95 | 9.700 | 7,454 / 221,700 |
  | 8 | 182/200 (91.0%) | 86.2–94.2% | 86.0–95.0% | 49.52 | 21.880 | 17,616 / 554,624 |
  | 12 | 180/200 (90.0%) | 85.1–93.4% | 85.5–94.0% | 56.81 | 17.210 | 32,098 / 1,499,784 |
  | 16 | 163/200 (81.5%) | 75.5–86.3% | 75.0–88.0% | 80.20 | 39.825 | 76,746 / 3,849,360 |

  Third-to-fourth validation rates: 99.0/95.5/94.5/88.0/90.0/81.0% to
  96.5/96.0/95.5/91.0/90.0/81.5%. Gains are mixed: two-robot performance regressed 2.5 pp,
  while larger populations improved or stayed level. All 99 fourth-cohort failures contain repeated
  full states; none visited every goal without simultaneous completion. Random-obstacle family success
  remains below 95% everywhere: 82.5/87.5/85.0/82.5/85.0/65.0%. The 16-robot corridor family is 75.0%.
  Latest attempted vertex/edge conflicts total 6,572/342 before physical resolution; blocked moves total
  11,127. These are distinct from physical collisions. Original exposure gates pass in every population.
  No policy qualifies: 8/12/16 aggregate success and multiple family gates remain unmet, and final
  current-model compatibility/test evaluation has not begun. The test bank remains closed.
  Four complete checks cannot trigger the frozen seven-check minimum for a plateau stop.
  Continue the unchanged run within its existing caps. Raw evidence remains local in
  `output/phase1_scale/clean-16-01/monitor_audit_20261023_013202.json` and `monitor_history.jsonl`.
  Next hourly check: **2026-10-23 02:32 EDT**.
