# SANGIN research execution plan

## Current status

- Phase: **1 clean PANAMA training through 16 robots**.
- Clean curriculum: stage **2**, introduced populations **[2, 4]**; no population graduated.
- Active research workers: **`output/phase1_scale/clean-16-01`**, frozen source `f84cfe5`. Hourly monitor: **active**, registered before GPU calibration and training.
- Episode budget: **2,500 completed episodes per population; 15,000 total per clean run**.
  The later communication-aware comparison uses the same episode cap. Measured full-run wall cap: **37 hours (133,200 s)**; final-evaluation reserve: **90 minutes**.
- Qualified policies: **none**. `PANAMA-CLEAN` and `PANAMA-COMM` do not exist as qualified artifacts.
- Starting source: `9d52899`. Work stays on `main`; source/configuration/data freeze before each launch.
- Current execution: fresh clean training under the frozen episode/wall limits; complete greedy validation every 1,000 episodes. Later communication experiments remain gated on clean qualification.
- Scientific stop: the old primary comparison failed its launch rule. No old extended run was launched.
  The scale-16 coverage experiment uses a separate versioned protocol; that result and gate remain unchanged.

## Research objectives and phase gates

Separate clean MAPF competence, network impairments, allocation effects and adaptation through learning.
Preserve every previous source revision, checkpoint, failed experiment, dataset and study result.
Large results/checkpoints remain local. Commit code, tests, configuration, methodology and this plan only.

| Phase | Deliverable | Prerequisites / current disposition |
| --- | --- | --- |
| 0 | This plan, provenance, explicit budgets, functioning hourly monitoring | Episode cap fixed; runtime estimation and workflow validation in progress |
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
- Latest prior regression result: **325 passed**; prior bounded remote-DT execution compatibility passed.

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

Introduce populations by a declared joint-step schedule independent of success gates. Rehearse introduced
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
