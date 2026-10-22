# Phase 1: clean PANAMA MAPF training

This workflow trains a new policy from deterministic random weights. It never
retrains over, edits or replaces the previous checkpoints, the 1,200-episode
remote-DT study, or the 216-episode controlled-staleness pilot. No communication
degradation or communication-aware training/evaluation is launched.

## Environment and observation contract

The DQN architecture, three 15x15 channels, 12-element vector, five actions,
PANAMA distance/completion priorities and robot-ID tie-break, default rewards,
250-step cap and local physical conflict resolver are retained. Fresh state is
available instantaneously. No allocation, compute service, orbital data, radio
queues, communication deadlines or information holds enter training. Actor
weights are copied instantaneously every 16 steps, retaining the existing clean
learner cadence without serializing imaginary radio packets. Double DQN, AdamW,
prioritized replay, soft target updates and epsilon exploration reuse the
existing learning implementation and configuration.

The historical local guidance consumer removes the current path cell before
constructing waypoints, whereas remote-DT expects the full path. Local intentions
also have a peer-distance eligibility check that remote's square-FoV cell
clipping does not use. Phase 1 resolves these interface differences explicitly:
`CleanMAPFEnvironment` feeds fresh states through the **unchanged remote_observation**
constructor. This preserves the intended observation structure and remote input
semantics, rather than claiming the old local arrays were identical. No legacy
observation or movement code is changed. The new payload declares this contract.

The remote post-completion local controller remains distinct from local MAPF DQN
control of completed robots. Compatibility proves fresh active observations and
actions match and audits short ideal remote rollouts; it does not claim complete
mission equivalence across those existing completion behaviors.

## Data separation and coordination coverage

Five procedural families cover random obstacles, intersections, corridors with
passing spaces, shared bottlenecks and warehouse aisles. Every free-cell map is
connected. Training has 20 maps per family; validation and test each have five
independently generated maps per family. No map is shared across splits. A
canonical hash rejects equivalent maps under crop/translation, rotation and
reflection, including all original controlled-stress pilot geometries. The
pilot's maps, endpoints and results remain reserved for evaluation.

Training cases are generated only on training maps using deterministic seeds,
balanced family selection and a 50/50 mixture of deliberately interacting and
diverse endpoint cases. Interacting cases must have a timed vertex/edge path
conflict inside the square FoV. Endpoints are unique and individually reachable;
single-agent reachability is not a certificate of joint MAPF solvability.

Validation and test each contain 200 cases per population (40 per family) for
2/4/6/8 robots. Half of each family is explicitly coordination-exposed. Both
cohorts are frozen before learning. Validation guides curriculum progression;
test cases cannot be used for training, model selection or curriculum decisions.
Once final evaluation begins, resume never returns to learning. A later run
using test findings must use a new untouched test split to support an independent
held-out claim.

## Curriculum, budget and qualification

`configs/phase1_clean.toml` uses the previously measured fast-pilot DQN settings
(learning rate .001, gamma .90, epsilon decay 20,000, replay capacity 50,000).
These learning hyperparameters do not change task rewards or physical dynamics.
The run budget is **10,800 seconds (three hours)**, including data preparation,
initialization, training, validation, checkpointing and final evaluation. The
last 1,800 seconds are reserved for current-model qualification and held-out
testing, plus a 60-second shutdown reserve. Incomplete evaluation stays partial;
the clock is not extended to manufacture qualification.
Final evaluation interleaves both splits, all populations and families in small
batches, so a time-limited partial result is not restricted to easy populations.

Curriculum phases remain 2/4/6/8 robots, at least 500 episodes per phase, with
200-case greedy validation every 250 episodes. The success threshold remains at
least 95%. This protocol additionally requires at least 95% within every family,
at least 20 exposed cases per 40-case family, and at least 200 relevant directed
pair observations per population. Exposure counts include repeated decision
epochs, not independent physical collisions.

Earlier curriculum qualifications do not certify the final weights. The frozen
final checkpoint must pass all four populations on validation **and** held-out
test cohorts, as well as the remote compatibility checks. A failed/partial or
smoke checkpoint cannot pass `load_model(..., require_qualified=True)` under the
new multi-map protocol. The old single-map checkpoint loader retains its checks.
The new loader does not pretend a multi-map model was qualified on the historical
single map. Qualification is a measured success-rate target, not a claim that
the lower 95% confidence bound also exceeds 95%.

## Execution and monitoring

```powershell
# Prepare and launch the short structural/learning/replay smoke test.
.venv\Scripts\python.exe scripts/run_phase1_clean.py --run-id phase1-smoke-01 --smoke --launch --device cuda
# After tests and smoke validation pass, launch the reviewed three-hour run.
.venv\Scripts\python.exe scripts/run_phase1_clean.py --run-id phase1-clean-01 --launch --device cuda
# Explicit continuation, preserving optimizer, replay, all RNG streams and budget.
.venv\Scripts\python.exe scripts/run_phase1_clean.py --run-id phase1-clean-01 --resume --launch --device cuda
```

Each run gets an immutable source snapshot, disjoint dataset and manifest under
ignored `output/phase1_clean/`. Background workers are hidden and identified by
PID, creation time and full command. Duplicate live launches are rejected.
Resume verifies dataset/configuration/source identities, restores full learning
state and truncates logs to checkpoint-consistent offsets. Partial evaluation
cohorts are reused only with identical weights; an unfinished greedy case may
be restarted deterministically and is not counted until complete.

Two-hour checks should inspect status, log/checkpoint progress, actual process
identity, errors and GPU/RAM use. They must not duplicate workers, change running
code/data/budgets, restart from fresh weights or begin another study. On completion
they should review local final qualification, held-out failures, compatibility,
curves and runtime and report negative/partial outcomes faithfully.

## Outputs and limits

All models and results stay local. `checkpoint.pt` holds resumable state;
`latest_model.pt`, per-stage models and `final_model.pt` are new artifacts.
`manifest.json` records source/configuration/dataset hashes and initialization;
`status.json` reports runtime, hardware, progress and qualification.
Training and validation JSONL logs retain case IDs and seeds; final per-case
CSVs include failed endpoints/final positions. Per-family/population tables
include success, capped makespan, waiting, attempted vertex/edge conflicts,
coordination exposure and case-level Wilson intervals. Map-cluster bootstrap
intervals for success, makespan and waiting account for shared map cases;
five maps per family still give limited precision. Pointwise intervals have no
multiplicity correction. `training_curves.png`, `compatibility.json` and
`PHASE1_REPORT.md` support review. The smoke uses reduced maps/cases/episode caps
and can never be presented as a qualified model or held-out benchmark.
