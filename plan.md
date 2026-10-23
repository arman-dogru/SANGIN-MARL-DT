# SANGIN research execution plan

## Current status

- Phase: **0 implementation and Phase 1 six-population protocol validation**.
- Clean curriculum: preparing **2 / 4 / 6 / 8 / 12 / 16**; no population graduated.
- Active research workers: **none**. Hourly monitor: **paused** until an authorized worker launches.
- GPU authorization for the new program: **pending**. The previous run's budget is not reusable authorization.
- Qualified policies: **none**. `PANAMA-CLEAN` and `PANAMA-COMM` do not exist as qualified artifacts.
- Starting source: `9d52899`. Work stays on `main`; source/configuration/data freeze before each launch.
- Immediate work: scalable clean training, leakage/feasibility checks, strict six-population qualification,
  checkpoint/replay compatibility, cumulative budget enforcement and hourly monitoring.
- Scientific stop: the old primary comparison failed its launch rule. No old extended run was launched.
  This request authorizes a new versioned protocol, not a change to that result or its gate.

## Research objectives and phase gates

Separate clean MAPF competence, network impairments, allocation effects and adaptation through learning.
Preserve every previous source revision, checkpoint, failed experiment, dataset and study result.
Large results/checkpoints remain local. Commit code, tests, configuration, methodology and this plan only.

| Phase | Deliverable | Prerequisites / current disposition |
| --- | --- | --- |
| 0 | This plan, provenance, explicit cumulative budgets, functioning hourly monitoring | In implementation; GPU budget pending |
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

Hypothesis for the next bounded comparison: progressive population introduction plus rehearsal needs
population-specific exploration exposure for newly introduced 12/16-robot tasks. Compare global-clock
epsilon with population-clock epsilon under otherwise identical progressive schedules and original gamma
.90/lr .001. This hypothesis follows residual repeated-state failures and short population exposure;
it does not claim the old comparison proved exploration collapse. Do not repeat failed gamma .99 runs blindly.

Introduce populations by a declared joint-step schedule independent of success gates. Rehearse introduced
populations with least-completed-episode selection and rotate all families within each population.
An unsuccessful early stage cannot exclude later populations permanently. Track all introduced populations
on greedy validation to expose forgetting. A graduation record belongs to the current frozen weights only.

Substantial exploration changes require paired learning seeds and validation-only selection. The launch
criterion remains a >=5 pp paired improvement with a positive lower 95% interval and no >5 pp population
loss, or >=90% in every target population. The new six-population comparison is separate from the old four-population gate.
Do not launch full training after unsupported selection. Record throughput and a complete coverage/evaluation
budget before a supported full run. Runtime is evidence about cost, not a promise of qualification.

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

No new GPU run is authorized yet. Asked the user for an explicit cumulative budget (6/12/24 GPU-hours).
Conservatively charge GPU worker wall time including initialization, validation/checkpoints and recovery.
Training/test preparation and bounded CPU regression/feasibility diagnostics are implementation validation.
Do not infer fresh authorization from the already consumed V1/V2 budgets.

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

- **2026-10-22 22:49 EDT** — Read the new research-program request; pulled `main` (up to date at `9d52899`).
  No active run. Prior primary comparison remains an unsupported-selection stop. Started Phase 0 and new
  six-population Phase 1 implementation; requested the missing cumulative GPU budget. Next automated check:
  paused until an authorized launch; next immediate action is implementation/regression and feasibility validation.
