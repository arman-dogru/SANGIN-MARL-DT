# Phase 1 V2: measured-budget extended training

This workflow supersedes automatic three-hour full launches. Existing V1/V2
runs and the active 30,000-step primary comparison remain frozen. No training
is performed with communication impairments and no SAGIN model is modified.

## Execution order and selection

1. Finish the existing four configurations and two seeds without changing their
   30,000-step/900-second limits, maps, cases or checkpoint state.
2. Recompute the original selection rule from every paired validation cohort.
   A candidate must gain at least five percentage points over original settings,
   have a positive lower paired 95% interval and lose no more than five points
   in any population, **or** reach at least 90% validation success in every
   population. These are unchanged launch criteria, not qualification gates.
3. If neither criterion passes, stop and report the evidence. Do not use a
   larger budget to bypass unsupported configuration selection.
4. If supported, benchmark the selected frozen model on validation cases only,
   write `budget_plan.json` and `BUDGET.md`, then prepare a **new** extended run.
   The estimate and explicit budget exist before its manifest and launch.

`--analyse` exports per-population/per-family success and seed/map bootstrap
intervals, capped makespan, active WAITs, blocked movement, loss/Q/gradient
stability and measured training throughput. `--diagnose` replays at most 64
validation missions: one failure and one success per population/model/seed,
when available. These outcome-stratified cycle checks are mechanism diagnostics,
not an independent performance estimate. No test result informs selection.

## Budget measurement

Training logs assign elapsed wall time and joint steps to the population
currently executing. Conditional steps/hour and completed episodes/hour are
reported for 2/4/6/8 robots. They are **not** four simultaneous throughputs.
Checkpoint and startup gaps remain charged. An unfinished last episode
contributes cost and steps, but is never counted as a completed episode.

The estimator uses both learning seeds and the slower complete-window or tail
step rate. It records an empirical projection and a conservative projection in
which all 2,000 required training episodes consume the full 250 steps (500,000
joint steps). It budgets periodic greedy validation, checkpoints already
present in measured training cost, compatibility, complete final validation
and held-out testing, initialization and shutdown. A bounded 32-case frozen
validation replay supplies per-population evaluation cost; the 90th percentile
seconds/step is charged at 250 steps for every planned evaluation case.

A 20% safety factor is recorded. The resulting full cap is rounded up to
30-minute units and is at least three hours. Three-hour sufficiency is reported
separately for the empirical and conservative estimates. Estimates exceeding
the supported one-day limit are rejected rather than silently capped. A
runtime estimate cannot guarantee 95% success. No existing budget is extended.

## Extended run and bounded stopping

The version is `clean_mapf_phase1_v2_extended_v1`, while the model retains the
V2 PANAMA observation/action schema and base metric version. Fresh deterministic
weights use the first predeclared learning seed. Ablation weights are not used
to initialize full training. The selected learning rate, gamma and epsilon
schedule are preserved. All four populations and five families retain the V2
episode rotation and active-only replay. No qualification gate blocks coverage.

Greedy validation occurs every 30,000 joint learning steps. After at least 500
completed training episodes **per population**, learning can stop early when
the small progress cohort passes the success targets; full independent final
validation must still pass. Learning also stops after a plateau: at least
90,000 steps, seven complete progress checks, and six consecutive checks with
no gain beyond one percentage point over the earlier best, provided the recent
half-window is not still recovering by more than one point. A fixed cap always
applies. Partial validations never trigger a plateau or readiness decision.

The final model is the current frozen policy, not a historical stage gate.
The untouched test remains closed until all training coverage, complete
800-case strict final validation and extended execution compatibility pass.
Failure of these gates is reported with test results explicitly unavailable.
Once final evaluation begins, resume cannot return to learning. A completed
test is never used to choose another configuration or resume this run.

All original qualification requirements remain: at least 95% success for each
population and every 40-case family in independent validation **and** held-out
test; original coordination exposure; at least 500 training episodes per
population; and passing remote-aligned execution and deterministic replay
checks. A plateau, partial evaluation or ablation is never a qualified model.

## Commands and provenance

```powershell
# After the primary comparison and its worker finish:
.venv\Scripts\python.exe scripts/run_phase1_extended.py --analyse --diagnose
# Only after the predeclared selection rule passes:
.venv\Scripts\python.exe scripts/run_phase1_extended.py --estimate
# Review the saved budget before preparing the new run:
.venv\Scripts\python.exe scripts/run_phase1_extended.py --run-id phase1-extended-01 --prepare --launch
# Continue only a confirmed stopped worker, retaining its runtime lease:
.venv\Scripts\python.exe scripts/run_phase1_extended.py --run-id phase1-extended-01 --resume --launch
```

The new folder is `output/phase1_extended/<run-id>/`. It freezes source,
selected settings, comparison/input hashes, copied budget plan, original
data identity, seed and initialization. Checkpoints retain optimizer, replay,
RNG and remaining runtime; duplicate/overlapping PANAMA workers are rejected.
No prior folder or model is overwritten. All large results/models remain local.

Cycle metrics are separately versioned `clean_mapf_cycle_metrics_v1`; repeated
joint states include actual observations, completion flags and the distinct
epoch-zero cache bootstrap. Active no-motion and local-controller goal yielding
are recorded separately. Local yielding is not a completed DQN action.
`EXTENDED_REPORT.md`, per-case/failure CSVs, population/family summaries and
`validation_curves.png` report estimated versus actual cost and final outcomes.
Two learning seeds and four progress maps per family limit interval precision;
full qualification uses all five validation/test maps per family. Bootstrap
intervals are exploratory and have no multiplicity correction.

Two-hour checks inspect worker creation/command identity and actual progress,
not just a live PID. Normal progress stays quiet. Errors, stalls, plateau stops,
qualification and completion warrant an update. Unsupported selection ends
training; further bounded iteration requires new validation evidence rather
than indefinite extension or lowered thresholds.
