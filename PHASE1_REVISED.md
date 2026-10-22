# Phase 1 correction: clean PANAMA V2

The failed `clean_mapf_phase1_v1` run is preserved. Its final greedy held-out
success was 97%, 85%, 74.5%, and 61% for 2/4/6/8 robots. V2 changes only the
opt-in clean learning workflow and execution wrapper. It does not change SAGIN
allocation, workload, CPU capacities, radio scheduling, remote physical rules,
observations, five actions, rewards, or the 95% qualification threshold.

## Ranked diagnosis before retraining

1. **Greedy deterministic traps and completed-robot control mismatch.** A
   stratified replay sample contains 65 failed and 40 successful held-out
   missions. Every sampled failure enters an exact repeated joint position and
   observation cycle; none of the controls does. Thirty-seven failures have a
   terminal stationary run of at least 20 steps; 27 have period-two cycles.
   Forty failures contain completed-goal departures, and in 31 every robot has
   visited its goal without joint mission completion. Active WAITs comprise
   5,123/16,065 active decision epochs in failures versus 47/2,315 in controls.
   The old model controlled completed robots while remote-DT uses a local
   controller. Replaying 18 failures and 20 successes with aligned execution
   recovered five failures and regressed none of the successes. These selected
   samples diagnose mechanisms; they do not estimate population-wide gains.
2. **Curriculum starvation and forgetting.** Two robots received 3,250 episodes
   and 116,989 steps; four received 8,750 episodes and 355,203 steps. Six/eight
   received no training. Four-robot complete greedy validation fluctuated from
   81.5% to 93.5% and never qualified. Two-robot validation on identical cases
   fell from 198/200 to 191/200 after four-robot training (eight new failures,
   one recovery). Final qualification must use current weights, not stage
   history. The final partial 92-case stage evaluation is not a full result.
3. **Training success conflates exploration with greedy performance.** The last
   1,000 training episodes had 99.4% success at epsilon 0.1. Fifteen selected
   greedy validation failures all failed at 250 steps; epsilon 0.1 escaped in
   all 45 diagnostic replays (three RNG replicates per case). Fresh endpoints
   on training maps achieved only 38/40 and 33/40 greedy successes at 2/4
   robots. Map overfitting alone therefore does not explain the gap. The
   diagnostic replicates are not independent mission cases.
4. **Replay role contamination, aggressive updates and short return horizon.**
   Completed robots occupy 26,938/50,000 entries in the final replay buffer;
   6,264 of those entries label a moving DQN action. Four-robot training began
   with epsilon already about 0.1026. Gamma 0.9 discounts by half after 6.58
   steps. Learning rate 0.001 and gamma 0.9 warrant controlled alternatives;
   all 472,146 recorded losses were finite, so numerical divergence is not
   established. V1 did not record Q/TD/gradient statistics.
5. **Joint infeasibility is not the explanation for the bounded reference
   sample.** Independent conflict-based search found verified collision-free
   plans for 24 failed cases, six per population; the longest plan takes 38
   steps. No solver timed out. The planner uses cardinal moves/WAIT, no vertex
   or opposite-edge conflicts, and holds goals after final arrival. A feasible
   plan is a constructive witness; timeouts, node limits and finite-horizon
   infeasibility are distinct outcomes. This does not prove every dataset case
   feasible or reference-optimality.

Failure replays reproduced all saved metrics exactly. The V1 physical resolver
and delivered-priority resolver produced identical positions in every audited
epoch; their collision-penalty bookkeeping can differ. V2 explicitly uses the
remote resolver and penalty convention rather than attributing all failures to
collision resolution. Full local diagnostics and compressed traces reside in
`output/phase1_diagnosis/phase1-diagnosis-20261022-01/`; no historical run was
modified or trained on during diagnosis.

## V2 execution and learning contract

`RemoteAlignedCleanMAPFEnvironment` reuses `remote_observation`,
`completed_action` and `resolve_delivered`. The completed local controller may
yield away from a goal when required and return; it is not replaced by a new
permanent goal-lock rule. Only active robots generate DQN replay transitions.
First goal completion ends that robot's learned return. Episode success still
requires all robots physically at their goals together. A 250-step truncation
remains terminal under this finite-episode objective.

Fresh priorities, robot-ID ties, reward accounting, completed control and full
multi-step observations/actions are checked against the actual perfect remote
environment. The perfect-reference cache has horizon-limited bootstrap paths
at epoch zero and full fresh paths subsequently; V2 mirrors this behavior.
Clean timestamps use integer control ticks, so these checks do not claim
nominal network timing or command-deadline compliance.

Training rotates **2, 4, 6, 8 robots every episode**, cycling all five map
families within each population. Failed validation never blocks later
populations. Replay is retained across populations, avoiding a stage reset
that removes earlier experience. This is a coverage curriculum, not a lowered
qualification gate. Full qualification still requires at least 500 completed
training episodes for every population, all four 200-case final validation and
test cohorts, at least 95% success overall and in every 40-case family, the
original exposure thresholds, and passing extended execution compatibility.

## Data and controlled ablations

`configs/phase1_revised.toml` declares all alternatives and budgets. A new seed
generates 100 training, 25 validation and 25 untouched test maps. Canonical
geometry checks exclude **all** historical Phase 1 maps and controlled-stress
maps, as well as translations, rotations and reflections. Half of each family
contains designed coordination and half diverse random endpoints. Ablation
validation samples both halves equally. No historical held-out or stress case
enters learning, and the new test bank is never evaluated for ablation
selection. Final test performance cannot trigger further training in a run.

| Alternative | Learning rate | Gamma | Epsilon decay steps |
| --- | ---: | ---: | ---: |
| original_hp | 0.001 | 0.90 | 20,000 |
| lower_lr | 0.0003 | 0.90 | 20,000 |
| long_horizon | 0.0003 | 0.99 | 20,000 |
| long_exploration | 0.0003 | 0.99 | 100,000 |

Gamma 0.99 increases the return horizon from about 10 to 100 discounted steps;
the physical episode cap remains 250. Each alternative uses seeds 42/73,
6,000 joint learning steps and the same 40 validation cases per population.
Each run has a 360-second cap, including initialization and final validation;
120 seconds plus a five-second shutdown margin are reserved for evaluation.
Shared dataset/source preparation is measured separately. Unequal or incomplete
step/case counts prohibit selection. Maps, initial weights per seed, scenario
seed streams and the learning-step budget are matched. Different policies can
finish episodes at different times, so executed training cases and population
step counts need not be identical; both are logged explicitly.

The 6,000-step workflow check produced only 26 training episodes in its first
run. It is retained as preliminary evidence, not used alone to choose full
training. `configs/phase1_comparison.toml` specifies the primary bounded
comparison: the same four alternatives, two seeds, map/case bank and validation
cohort, with **30,000 joint steps per run**, a **900-second cap per run**,
180 seconds reserved for evaluation and a five-second shutdown margin.
The eight-run maximum is two hours; no existing run's budget or weights are
extended. Each starts from the same deterministic seed initialization in a new
study folder. Only the primary comparison can justify launching the one full
run. The final test bank remains untouched across both stages.

Selection reports paired seed-and-map bootstrap 95% intervals. Two learning
seeds and short-run validation cannot establish qualification. A full run is
justified only by at least a five-percentage-point gain over matched original
hyperparameters with positive lower paired interval and no population losing
more than five points, or by at least 90% validation success in every
population. This is a launch criterion only. Final qualification remains 95%.
Intervals are exploratory and not adjusted for choosing among alternatives.
If evidence is weak or a run is incomplete, the workflow records that outcome
and does not launch full training automatically.

## Reproducibility and execution

```powershell
.venv\Scripts\python.exe scripts/diagnose_phase1.py --help
.venv\Scripts\python.exe -m pytest -q --basetemp=tmp/pytest-revised-unique
.venv\Scripts\python.exe scripts/run_phase1_revised.py --study-id revised-01 --prepare --launch --mode ablations
# Current extended workflow: inspect primary selection, then record its budget.
.venv\Scripts\python.exe scripts/run_phase1_extended.py --analyse --estimate
```

Every study freezes source hashes, commit, protocol/configuration, map/data
hashes, hardware, seeds and source provenance. Each alternative and full run
gets its own manifest, checkpoint, optimizer/replay/RNG state and model files.
Explicit resume restores the same budget, roles, RNG streams and log offsets.
Checkpoints occur at most 1,000 ablation steps or 5,000 full-run steps apart. A crash rolls back all
worker-owned logs to those offsets, including logs created after the initial
checkpoint. An unclosed runtime lease conservatively charges elapsed wall
time through the next attempt, including downtime; a graceful stop closes the
lease. Lost work cannot refund the training budget. Interrupting one ablation
stops the batch instead of launching the next alternative.
Duplicate launcher/child processes are rejected. The original full worker starts fresh
weights under the selected protocol and has the same 10,800-second cap with
1,800 seconds plus 60 seconds reserved for final evaluation/shutdown.
The current request supersedes this automatic three-hour choice: use
`PHASE1_EXTENDED.md` to record measured throughput, budget sufficiency and a new
explicit full-run cap before execution. The active ablation snapshots are unchanged.

V1 checkpoints remain loadable under their original rules. V2 checkpoints
declare `clean_mapf_phase1_v2` and `clean_mapf_metrics_v2`; ablations, incomplete
curricula and failed final gates are rejected by qualified mission loading.
Outputs and models remain local under ignored `output/phase1_revised/`.
Per-case records retain explicit active WAITs, blocked movement separately,
attempted vertex/edge conflicts, repeated relevant/direct pair-epoch exposure,
latency-free clean makespan, source weights and split identities. Q/gradient
probes supplement scalar loss without changing DQN updates. No communication
impairments or communication-aware training are introduced.
